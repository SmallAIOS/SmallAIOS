// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS minimal whole-segment garbage collection (Phase 7 v1).
//!
//! Phase 7's write path produces free-space fragmentation as files are
//! created, written, truncated, and deleted: blocks freed inside an
//! otherwise-occupied segment cannot be reused without first relocating
//! the segment's surviving live blocks. The full segment-level GC
//! sophistication (multi-victim selection, age-based heuristics, hot/
//! cold separation) is Phase 9. This module implements the minimal
//! "whole-segment" variant the Phase 7 spec calls for:
//!
//! - **Trigger**: when free segments fall below 5 % of total segments.
//! - **Victim selection**: pick the most fragmented data segment
//!   (highest dead-block ratio).
//! - **Relocation**: copy each live 4 KiB block into a fresh segment,
//!   update the owning inode's `i_addr` (or indirect block), update
//!   NAT (block address rewrite is implicit because we rewrite the
//!   inode), update SIT (mark the new block valid; clear the old),
//!   update SSA (record reverse-mapping for the new block).
//! - **Cooperative yields**: between block relocations, return a
//!   [`GcYield`] hint. The kernel's task scheduler honors it by
//!   running queued foreground writes before resuming GC. (The yield
//!   is advisory — embedded targets without a preemptive scheduler
//!   loop on every relocation, but a foreground write request bumps
//!   `should_yield` so GC backs off mid-pass.)
//!
//! Public surface: [`run_gc_pass`] is invoked from
//! [`crate::f2fs::F2fs::write_file`] when free-segment count drops
//! below threshold, and exposed publicly so tests and a future
//! background-GC task can drive it directly.

extern crate alloc;

use alloc::vec::Vec;

/// Cooperative yield decision after each block relocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcYield {
    /// GC may continue with the next block.
    Continue,
    /// GC SHOULD pause — a foreground request is pending.
    Yield,
}

/// Statistics returned by a GC pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GcStats {
    /// Number of victim segments freed during the pass.
    pub segments_freed: u32,
    /// Number of live blocks relocated.
    pub blocks_relocated: u32,
    /// Number of cooperative yields surfaced.
    pub yields: u32,
}

/// Decide whether GC should run given the current free-segment count.
///
/// Returns `true` if `free_segments < ceil(total_segments * 5 / 100)`,
/// matching the Phase 7 trigger threshold.
#[inline]
pub fn should_run_gc(free_segments: u32, total_segments: u32) -> bool {
    if total_segments == 0 {
        return false;
    }
    let threshold = total_segments.saturating_mul(5).div_ceil(100).max(1);
    free_segments < threshold
}

/// Score one segment for victim selection. Higher score = more
/// fragmented = better victim candidate.
///
/// The Phase 7 heuristic is "dead block ratio": `(total - valid) /
/// total`. Segments with `valid_blocks == 0` are already free and skip
/// scoring.
#[inline]
pub fn fragmentation_score(valid_blocks: u16, blocks_per_segment: u32) -> u32 {
    let total = blocks_per_segment;
    if total == 0 {
        return 0;
    }
    let dead = total.saturating_sub(valid_blocks as u32);
    // Scale to ppm so we get integer ordering without floats.
    dead.saturating_mul(1_000_000)
        .checked_div(total)
        .unwrap_or(0)
}

/// Pick the most fragmented victim segment from a list of (segno,
/// valid_blocks) candidates. Returns `None` if every candidate is
/// either already free (valid==0) or fully valid (valid==total).
pub fn pick_victim(candidates: &[(u32, u16)], blocks_per_segment: u32) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for &(segno, vb) in candidates {
        if vb == 0 || vb as u32 >= blocks_per_segment {
            continue;
        }
        let score = fragmentation_score(vb, blocks_per_segment);
        if score == 0 {
            continue;
        }
        match best {
            None => best = Some((segno, score)),
            Some((_, s)) if score > s => best = Some((segno, score)),
            _ => {}
        }
    }
    best.map(|(s, _)| s)
}

/// Result of a single GC iteration step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcRelocation {
    /// Block index within the victim segment that was relocated (0-511).
    pub src_block_idx: u16,
    /// New physical block address for the relocated block.
    pub dst_block_addr: u32,
    /// The owning inode-id for the moved block (from SSA).
    pub owning_nid: u32,
    /// Yield hint to the caller.
    pub yield_hint: GcYield,
}

/// Drive a single GC pass over `victim_blocks`, relocating each live
/// block via `relocate_one`. The caller provides a yield-poll callback
/// (`should_yield`) that returns `true` if a foreground request is
/// pending; when that happens the GC pass stops mid-stream and returns
/// the partial stats.
///
/// `relocate_one(src_block_idx, owning_nid)` is responsible for:
/// 1. Reading the live block from the victim segment.
/// 2. Allocating a fresh block in the current data segment.
/// 3. Writing the bytes there.
/// 4. Updating NAT/SIT/SSA + the owning inode's pointer.
///
/// On success it returns the new physical block address. On error it
/// returns `Err`; the GC pass aborts and returns the error to the
/// caller without marking the victim segment free.
pub fn run_gc_pass<F, Y, E>(
    victim_blocks: &[(u16, u32)],
    mut relocate_one: F,
    mut should_yield: Y,
) -> Result<(GcStats, Vec<GcRelocation>), E>
where
    F: FnMut(u16, u32) -> Result<u32, E>,
    Y: FnMut() -> bool,
{
    let mut stats = GcStats::default();
    let mut log = Vec::with_capacity(victim_blocks.len());
    for &(idx, nid) in victim_blocks {
        if should_yield() {
            stats.yields += 1;
            log.push(GcRelocation {
                src_block_idx: idx,
                dst_block_addr: 0,
                owning_nid: nid,
                yield_hint: GcYield::Yield,
            });
            break;
        }
        let dst = relocate_one(idx, nid)?;
        stats.blocks_relocated += 1;
        log.push(GcRelocation {
            src_block_idx: idx,
            dst_block_addr: dst,
            owning_nid: nid,
            yield_hint: GcYield::Continue,
        });
    }
    // The caller flips `segments_freed` if the victim segment is fully
    // drained (i.e. the relocation set covered every live block).
    Ok((stats, log))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_trigger_below_5_percent() {
        // 100 segments → threshold = ceil(100 * 5 / 100) = 5.
        assert!(should_run_gc(4, 100));
        assert!(!should_run_gc(5, 100));
        assert!(!should_run_gc(50, 100));
    }

    #[test]
    fn threshold_handles_tiny_total() {
        // Tiny pools: threshold floors to 1.
        assert!(should_run_gc(0, 4));
        assert!(!should_run_gc(1, 4));
    }

    #[test]
    fn threshold_zero_total_no_gc() {
        assert!(!should_run_gc(0, 0));
    }

    #[test]
    fn fragmentation_score_monotone() {
        // Fewer valid blocks ⇒ higher score.
        let s_full = fragmentation_score(512, 512);
        let s_half = fragmentation_score(256, 512);
        let s_low = fragmentation_score(64, 512);
        assert_eq!(s_full, 0);
        assert!(s_low > s_half);
        assert!(s_half > s_full);
    }

    #[test]
    fn fragmentation_score_zero_total_safe() {
        assert_eq!(fragmentation_score(100, 0), 0);
    }

    #[test]
    fn pick_victim_picks_highest_score() {
        let cands = [(10u32, 500u16), (11, 200), (12, 511)];
        let pick = pick_victim(&cands, 512).unwrap();
        assert_eq!(pick, 11); // 200/512 most fragmented
    }

    #[test]
    fn pick_victim_skips_full_and_empty() {
        let cands = [(10u32, 0u16), (11, 512)];
        assert!(pick_victim(&cands, 512).is_none());
    }

    #[test]
    fn pick_victim_empty_list() {
        assert!(pick_victim(&[], 512).is_none());
    }

    #[test]
    fn run_gc_pass_relocates_all() {
        let victim = [(0u16, 7u32), (1, 7), (2, 8)];
        let mut next_addr = 100u32;
        let (stats, log) = run_gc_pass::<_, _, ()>(
            &victim,
            |_idx, _nid| {
                let a = next_addr;
                next_addr += 1;
                Ok(a)
            },
            || false,
        )
        .unwrap();
        assert_eq!(stats.blocks_relocated, 3);
        assert_eq!(stats.yields, 0);
        assert_eq!(log.len(), 3);
        assert_eq!(log[0].dst_block_addr, 100);
        assert_eq!(log[2].dst_block_addr, 102);
    }

    #[test]
    fn run_gc_pass_yields_mid_stream() {
        use core::cell::Cell;
        let victim = [(0u16, 7u32), (1, 7), (2, 8), (3, 9)];
        let count = Cell::new(0u32);
        let (stats, log) = run_gc_pass::<_, _, ()>(
            &victim,
            |_idx, _nid| {
                let a = 100 + count.get();
                count.set(count.get() + 1);
                Ok(a)
            },
            || count.get() >= 2,
        )
        .unwrap();
        // 2 relocated, then yield bails before block #2.
        assert_eq!(stats.blocks_relocated, 2);
        assert_eq!(stats.yields, 1);
        assert_eq!(log.len(), 3);
        assert_eq!(log[2].yield_hint, GcYield::Yield);
    }

    #[test]
    fn run_gc_pass_propagates_relocate_error() {
        let victim = [(0u16, 7u32)];
        let r: Result<_, &'static str> = run_gc_pass(&victim, |_, _| Err("io fail"), || false);
        assert_eq!(r, Err("io fail"));
    }

    #[test]
    fn yield_enum_inequality() {
        assert_ne!(GcYield::Continue, GcYield::Yield);
    }
}
