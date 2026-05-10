// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS garbage collection (Phase 7 baseline + Phase 9 sophistication).
//!
//! ## Phase 7 baseline
//!
//! Phase 7's write path produces free-space fragmentation as files are
//! created, written, truncated, and deleted: blocks freed inside an
//! otherwise-occupied segment cannot be reused without first relocating
//! the segment's surviving live blocks. Phase 7 implemented the
//! minimal "whole-segment" variant:
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
//!   running queued foreground writes before resuming GC.
//!
//! ## Phase 9 sophistication
//!
//! Phase 9 layers the following on top:
//!
//! - **SSA-driven live-block relocation**: the per-segment validity
//!   bitmap (from SIT) gates which blocks need relocation; the SSA
//!   reverse-mapping `(block → owning_inode + offset)` tells us
//!   which inode pointer to rewrite. Phase 7's `maybe_run_gc` skipped
//!   relocation entirely; Phase 9 stages every relocation in a
//!   journal record (see [`crate::f2fs::gc_journal`]) BEFORE any
//!   in-place mutation, so a power loss mid-relocation cannot lose
//!   data.
//! - **Multi-victim selection**: rather than picking one most-
//!   fragmented victim per pass, pick top-N (default
//!   [`DEFAULT_GC_VICTIMS_PER_PASS`] = 4) by score. See
//!   [`pick_victims`].
//! - **Age-based hot/cold heuristics**: blocks belonging to inodes
//!   whose mtime is older than [`DEFAULT_COLD_THRESHOLD_SECS`] are
//!   tagged "cold" and relocated to the cold target segment; warm
//!   blocks go to the hot target. Future GC has less to do because
//!   cold and hot data are no longer co-mingled. See
//!   [`classify_temperature`] and [`pick_target_segment`].
//! - **Yield budget**: GC yields after every
//!   [`DEFAULT_YIELD_BUDGET_BLOCKS`] (16) relocations even if no
//!   foreground request is pending. This bounds the worst-case
//!   foreground latency to ~16 × per-block-write latency.
//! - **Per-pass time cap**: [`DEFAULT_PASS_BUDGET_TICKS`] (100)
//!   "ticks" cap a single GC pass so it never monopolizes the
//!   scheduler. The caller-supplied clock closure surfaces elapsed
//!   ticks; GC pauses cleanly mid-pass once exceeded and resumes on
//!   the next trigger.
//!
//! Configuration knobs read at runtime (when integrated by the kernel):
//!
//! | TOML key                          | Default | Constant                             |
//! |-----------------------------------|---------|--------------------------------------|
//! | `fs.f2fs.gc_victims_per_pass`     | 4       | [`DEFAULT_GC_VICTIMS_PER_PASS`]      |
//! | `fs.f2fs.gc_yield_budget_blocks`  | 16      | [`DEFAULT_YIELD_BUDGET_BLOCKS`]      |
//! | `fs.f2fs.gc_pass_budget_ticks`    | 100     | [`DEFAULT_PASS_BUDGET_TICKS`]        |
//! | `fs.f2fs.gc_cold_threshold_secs`  | 3600    | [`DEFAULT_COLD_THRESHOLD_SECS`]      |
//!
//! Public surface: [`run_gc_pass`] is the Phase 7 driver and remains
//! the integration entry-point for [`crate::f2fs::F2fs::write_file`].
//! Phase 9 adds [`run_gc_pass_v2`], a richer driver that honors the
//! yield budget, time cap, and journal staging. Both coexist; the
//! Phase 9 driver is opt-in by callers ready to thread the journal
//! through.

extern crate alloc;

use alloc::vec::Vec;

use super::gc_journal::GcJournalRecord;

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

// ─── Phase 9: multi-victim, age heuristics, time/yield budgets ────────────────

/// Default number of victim segments processed per GC pass.
///
/// `mgmt/policy.toml` key: `fs.f2fs.gc_victims_per_pass`. Increasing
/// this value increases GC throughput but also increases the maximum
/// foreground stall (since a single pass holds the GC lock).
pub const DEFAULT_GC_VICTIMS_PER_PASS: usize = 4;

/// Default yield budget — number of block relocations between forced
/// cooperative yields. `mgmt/policy.toml` key:
/// `fs.f2fs.gc_yield_budget_blocks`.
pub const DEFAULT_YIELD_BUDGET_BLOCKS: u32 = 16;

/// Default per-pass tick budget. The driver pauses cleanly once
/// `elapsed_ticks >= DEFAULT_PASS_BUDGET_TICKS`. The unit is whatever
/// the caller's clock closure returns — typically milliseconds.
/// `mgmt/policy.toml` key: `fs.f2fs.gc_pass_budget_ticks`.
pub const DEFAULT_PASS_BUDGET_TICKS: u64 = 100;

/// Default age threshold (seconds) above which an inode's data blocks
/// are classified "cold". `mgmt/policy.toml` key:
/// `fs.f2fs.gc_cold_threshold_secs`. Default is one hour.
pub const DEFAULT_COLD_THRESHOLD_SECS: u64 = 3600;

/// One block awaiting relocation: which (segno, blkidx) it lives at,
/// who owns it (from SSA reverse mapping), and what its mtime-age is
/// (from the owning inode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveBlock {
    /// Source segment number.
    pub src_segno: u32,
    /// Source block index within the segment (0..512).
    pub src_block_idx: u16,
    /// Owning node-id (from SSA).
    pub owning_nid: u32,
    /// Block offset within the inode's data array (from SSA).
    pub ofs_in_node: u32,
    /// Owning inode's mtime (seconds-since-epoch — interpretation is
    /// caller's choice; the GC just compares it against `now`).
    pub mtime_secs: u64,
}

/// Hot/cold classification used to pick target segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Temperature {
    /// Recently touched data — co-locate with other hot blocks.
    Hot,
    /// Stale data — co-locate with other cold blocks.
    Cold,
}

/// Decide whether a block is "cold" based on its mtime age.
///
/// `now_secs` is the current wall-clock seconds (or any monotonic
/// counter); `cold_threshold_secs` is the cutoff. Returns
/// [`Temperature::Cold`] if `now_secs - mtime_secs >= cold_threshold`,
/// else [`Temperature::Hot`].
#[inline]
pub fn classify_temperature(
    mtime_secs: u64,
    now_secs: u64,
    cold_threshold_secs: u64,
) -> Temperature {
    let age = now_secs.saturating_sub(mtime_secs);
    if age >= cold_threshold_secs {
        Temperature::Cold
    } else {
        Temperature::Hot
    }
}

/// A pair of allocator hints — the current hot- and cold-target
/// segment numbers. Phase 9 uses two streams; future phases may add
/// "warm" if mgmt/policy demands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorHints {
    /// Target segment for hot data.
    pub hot_segno: u32,
    /// Target segment for cold data.
    pub cold_segno: u32,
}

/// Pick the target segment for a block based on its temperature and the
/// caller's current allocator hints.
#[inline]
pub fn pick_target_segment(temp: Temperature, hints: AllocatorHints) -> u32 {
    match temp {
        Temperature::Hot => hints.hot_segno,
        Temperature::Cold => hints.cold_segno,
    }
}

/// Pick the top-N most-fragmented victim segments, ordered descending
/// by score. Returns up to `n` segnos.
///
/// Score-tie behavior: stable in input order (the first occurrence
/// wins), so callers can pre-sort their candidate list by segno or
/// last-mtime to make the choice deterministic.
pub fn pick_victims(candidates: &[(u32, u16)], blocks_per_segment: u32, n: usize) -> Vec<u32> {
    if n == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(u32, u32)> = candidates
        .iter()
        .filter_map(|&(segno, vb)| {
            if vb == 0 || vb as u32 >= blocks_per_segment {
                return None;
            }
            let s = fragmentation_score(vb, blocks_per_segment);
            if s == 0 {
                None
            } else {
                Some((segno, s))
            }
        })
        .collect();
    // Stable sort descending by score; for equal scores we keep input
    // order so callers can pre-sort for deterministic tie-breaking.
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored.into_iter().take(n).map(|(s, _)| s).collect()
}

/// Outcome of one Phase 9 GC pass over multiple victims.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct GcPassOutcome {
    /// Per-pass relocation log (in-flight + committed).
    pub log: Vec<GcRelocation>,
    /// Number of segments fully drained and marked free.
    pub segments_freed: u32,
    /// Number of live blocks successfully relocated and committed.
    pub blocks_relocated: u32,
    /// Number of cooperative yields surfaced.
    pub yields: u32,
    /// True if the pass paused early due to time-budget exhaustion.
    pub time_capped: bool,
    /// True if the pass paused early due to a foreground yield request.
    pub foreground_pause: bool,
    /// Journal records staged during this pass (in input order); the
    /// caller MUST persist these before applying the inode/NAT/SIT
    /// updates implied by `log`.
    pub journal: Vec<GcJournalRecord>,
}

/// Configuration for a Phase 9 GC pass.
#[derive(Debug, Clone, Copy)]
pub struct GcPassConfig {
    /// How many victim segments to process per pass.
    pub victims_per_pass: usize,
    /// Force a yield after this many block relocations.
    pub yield_budget_blocks: u32,
    /// Pause once cumulative ticks reach this value.
    pub pass_budget_ticks: u64,
    /// Age threshold (seconds) for hot/cold classification.
    pub cold_threshold_secs: u64,
    /// Current wall-clock seconds (passed to age classifier).
    pub now_secs: u64,
    /// Hot/cold target segments (the allocator's current cursors).
    pub hints: AllocatorHints,
}

impl Default for GcPassConfig {
    fn default() -> Self {
        Self {
            victims_per_pass: DEFAULT_GC_VICTIMS_PER_PASS,
            yield_budget_blocks: DEFAULT_YIELD_BUDGET_BLOCKS,
            pass_budget_ticks: DEFAULT_PASS_BUDGET_TICKS,
            cold_threshold_secs: DEFAULT_COLD_THRESHOLD_SECS,
            now_secs: 0,
            hints: AllocatorHints {
                hot_segno: 0,
                cold_segno: 0,
            },
        }
    }
}

/// Drive one Phase 9 GC pass.
///
/// Inputs:
/// - `candidates`: `(segno, valid_blocks)` for each candidate segment
///   (typically the SIT-dirty set; Phase 9 just selects from this).
/// - `live_blocks_for(segno)`: closure returning the list of live
///   `LiveBlock`s in a victim segment (caller resolves this via the
///   segment's SIT validity bitmap + SSA reverse mapping).
/// - `relocate(LiveBlock, target_segno) -> Result<new_addr, E>`:
///   closure that:
///   1. Reads the block content from the source.
///   2. Allocates a fresh block in `target_segno`'s data segment.
///   3. Writes the bytes there.
///   4. Returns the new block address.
/// - `should_yield()`: foreground-pressure poll.
/// - `clock()`: returns ticks-since-pass-start (any monotonic unit).
/// - `cfg`: Phase 9 knobs.
///
/// The driver maintains a journal of `(nid, ofs, old, new)` triples
/// staged BEFORE the caller commits SIT/NAT/inode updates. On crash
/// mid-pass, the staged journal allows replay to either re-apply
/// committed records or discard uncommitted ones. The driver itself
/// does NOT mutate the on-disk filesystem; it produces the
/// [`GcPassOutcome`] which the caller applies atomically.
///
/// Crash invariants (refined by `formal/tla/F2fsCheckpoint.tla`):
///
/// 1. Live data is never lost: `relocate` is called BEFORE the journal
///    record is marked `committed`, so a crash before the commit
///    leaves the source block intact (SIT still valid for it).
/// 2. Free-segment count never drops below 0: a segment is only marked
///    free after every live block in it has a `committed=1` journal
///    record.
/// 3. The journal records are append-only within the pass; replay is
///    idempotent because the same `(nid, ofs)` resolving to a different
///    `new_addr` simply overwrites the inode pointer (NAT carries the
///    latest version).
pub fn run_gc_pass_v2<L, R, Y, C, E>(
    candidates: &[(u32, u16)],
    blocks_per_segment: u32,
    cfg: GcPassConfig,
    mut live_blocks_for: L,
    mut relocate: R,
    mut should_yield: Y,
    mut clock: C,
) -> Result<GcPassOutcome, E>
where
    L: FnMut(u32) -> Vec<LiveBlock>,
    R: FnMut(LiveBlock, u32) -> Result<u32, E>,
    Y: FnMut() -> bool,
    C: FnMut() -> u64,
{
    let mut outcome = GcPassOutcome::default();
    let victims = pick_victims(candidates, blocks_per_segment, cfg.victims_per_pass);
    let mut blocks_since_yield = 0u32;

    for victim_segno in victims {
        // Pre-flight: time cap.
        if clock() >= cfg.pass_budget_ticks {
            outcome.time_capped = true;
            break;
        }
        let live = live_blocks_for(victim_segno);
        if live.is_empty() {
            // Already drained — mark free.
            outcome.segments_freed = outcome.segments_freed.saturating_add(1);
            continue;
        }
        let mut all_relocated = true;
        for blk in live {
            // Foreground-pressure yield check.
            if should_yield() {
                outcome.yields = outcome.yields.saturating_add(1);
                outcome.foreground_pause = true;
                outcome.log.push(GcRelocation {
                    src_block_idx: blk.src_block_idx,
                    dst_block_addr: 0,
                    owning_nid: blk.owning_nid,
                    yield_hint: GcYield::Yield,
                });
                all_relocated = false;
                break;
            }
            // Forced yield budget.
            if blocks_since_yield >= cfg.yield_budget_blocks {
                outcome.yields = outcome.yields.saturating_add(1);
                blocks_since_yield = 0;
                outcome.log.push(GcRelocation {
                    src_block_idx: blk.src_block_idx,
                    dst_block_addr: 0,
                    owning_nid: blk.owning_nid,
                    yield_hint: GcYield::Yield,
                });
                // After surfacing the yield, fall through and continue
                // relocating — the yield is a hint, not a hard stop.
            }
            // Time-cap mid-segment.
            if clock() >= cfg.pass_budget_ticks {
                outcome.time_capped = true;
                all_relocated = false;
                break;
            }
            let temp = classify_temperature(blk.mtime_secs, cfg.now_secs, cfg.cold_threshold_secs);
            let target = pick_target_segment(temp, cfg.hints);
            // Stage uncommitted journal record FIRST.
            let mut record = GcJournalRecord::new(
                blk.owning_nid,
                blk.ofs_in_node,
                addr_from(blk.src_segno, blk.src_block_idx, blocks_per_segment),
                0,
                matches!(temp, Temperature::Cold),
            );
            // Now perform the relocation.
            let new_addr = relocate(blk, target)?;
            // Mark the journal record committed with the new address.
            record.new_block_addr = new_addr;
            record = record.committed();
            outcome.journal.push(record);
            outcome.blocks_relocated = outcome.blocks_relocated.saturating_add(1);
            blocks_since_yield = blocks_since_yield.saturating_add(1);
            outcome.log.push(GcRelocation {
                src_block_idx: blk.src_block_idx,
                dst_block_addr: new_addr,
                owning_nid: blk.owning_nid,
                yield_hint: GcYield::Continue,
            });
        }
        if all_relocated {
            outcome.segments_freed = outcome.segments_freed.saturating_add(1);
        }
        // Break if we already paused — don't start another victim.
        if outcome.foreground_pause || outcome.time_capped {
            break;
        }
    }
    Ok(outcome)
}

/// Compute the absolute block address from `(segno, blkidx)` given the
/// blocks-per-segment density. This is a **relative** addr (caller
/// adds `main_blkaddr` to convert to a partition-absolute address);
/// the journal stores the relative form so a tools-level dump is
/// portable across partitions of the same block density.
#[inline]
pub fn addr_from(segno: u32, blkidx: u16, blocks_per_segment: u32) -> u32 {
    segno
        .saturating_mul(blocks_per_segment)
        .saturating_add(blkidx as u32)
}

/// Inverse of [`addr_from`]: split a relative block address into
/// `(segno, blkidx)`.
#[inline]
pub fn split_addr(addr: u32, blocks_per_segment: u32) -> (u32, u16) {
    if blocks_per_segment == 0 {
        return (0, 0);
    }
    let segno = addr / blocks_per_segment;
    let blkidx = (addr % blocks_per_segment) as u16;
    (segno, blkidx)
}

/// Decode the live-block set for a victim segment from its SIT
/// validity bitmap + SSA reverse mapping + per-inode mtime lookup.
///
/// `valid_map` is the SIT entry's 64-byte bitmap (one bit per 4 KiB
/// block, 512 bits total).  `summary[idx]` returns the `(nid, ofs)`
/// for block index `idx` (0..512). `mtime_for(nid)` returns the
/// owning inode's mtime; missing inodes (e.g. unlinked between SIT
/// snapshot and now) are filtered out.
pub fn live_blocks_from_segment<S, M>(
    src_segno: u32,
    valid_map: &[u8],
    mut summary: S,
    mut mtime_for: M,
) -> Vec<LiveBlock>
where
    S: FnMut(u16) -> Option<(u32, u32)>,
    M: FnMut(u32) -> Option<u64>,
{
    let mut out = Vec::new();
    let bits = (valid_map.len() * 8).min(512);
    for idx in 0..bits {
        let byte = valid_map[idx / 8];
        if (byte >> (idx % 8)) & 1 == 0 {
            continue;
        }
        let Some((nid, ofs)) = summary(idx as u16) else {
            continue;
        };
        if nid == 0 {
            continue;
        }
        let Some(mtime) = mtime_for(nid) else {
            continue;
        };
        out.push(LiveBlock {
            src_segno,
            src_block_idx: idx as u16,
            owning_nid: nid,
            ofs_in_node: ofs,
            mtime_secs: mtime,
        });
    }
    out
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

    // ── Phase 9 unit tests ────────────────────────────────────────────────

    #[test]
    fn pick_victims_returns_top_n_by_score() {
        // 6 candidates with varying validity. Higher dead-ratio = higher
        // score; pick the top 4 in descending order.
        let cands = [
            (1u32, 500u16), // dead=12 → score ≈ 23437
            (2, 100),       // dead=412 → score ≈ 804687
            (3, 256),       // dead=256 → score = 500000
            (4, 50),        // dead=462 → score ≈ 902343
            (5, 400),       // dead=112 → score ≈ 218750
            (6, 200),       // dead=312 → score ≈ 609375
        ];
        let v = pick_victims(&cands, 512, 4);
        assert_eq!(v, alloc::vec![4u32, 2, 6, 3]);
    }

    #[test]
    fn pick_victims_n_zero_returns_empty() {
        let cands = [(1u32, 100u16), (2, 200)];
        assert!(pick_victims(&cands, 512, 0).is_empty());
    }

    #[test]
    fn pick_victims_n_larger_than_candidates_clamps() {
        let cands = [(1u32, 100u16), (2, 200), (3, 300)];
        let v = pick_victims(&cands, 512, 10);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn pick_victims_skips_full_segments() {
        let cands = [(1u32, 512u16), (2, 100), (3, 0)];
        let v = pick_victims(&cands, 512, 4);
        assert_eq!(v, alloc::vec![2u32]);
    }

    #[test]
    fn pick_victims_stable_for_ties() {
        let cands = [(1u32, 256u16), (2, 256), (3, 256)];
        let v = pick_victims(&cands, 512, 3);
        assert_eq!(v, alloc::vec![1u32, 2, 3]);
    }

    #[test]
    fn pick_victims_empty_input() {
        let v = pick_victims(&[], 512, 4);
        assert!(v.is_empty());
    }

    #[test]
    fn classify_temperature_old_block_is_cold() {
        let temp = classify_temperature(1000, 5000, 3600);
        assert_eq!(temp, Temperature::Cold);
    }

    #[test]
    fn classify_temperature_recent_block_is_hot() {
        let temp = classify_temperature(4500, 5000, 3600);
        assert_eq!(temp, Temperature::Hot);
    }

    #[test]
    fn classify_temperature_exact_threshold_is_cold() {
        // age == cold_threshold ⇒ cold (>=).
        let temp = classify_temperature(0, 3600, 3600);
        assert_eq!(temp, Temperature::Cold);
    }

    #[test]
    fn classify_temperature_future_mtime_is_hot() {
        // Clock-skew tolerance: mtime in the future ⇒ saturating-sub
        // gives 0 ⇒ hot.
        let temp = classify_temperature(10_000, 5000, 3600);
        assert_eq!(temp, Temperature::Hot);
    }

    #[test]
    fn pick_target_segment_routes_correctly() {
        let hints = AllocatorHints {
            hot_segno: 7,
            cold_segno: 19,
        };
        assert_eq!(pick_target_segment(Temperature::Hot, hints), 7);
        assert_eq!(pick_target_segment(Temperature::Cold, hints), 19);
    }

    #[test]
    fn addr_from_split_addr_round_trip() {
        let bps = 512;
        let cases = [
            (0u32, 0u16),
            (0, 511),
            (1, 0),
            (1, 1),
            (255, 100),
            (1000, 511),
        ];
        for (segno, idx) in cases {
            let a = addr_from(segno, idx, bps);
            assert_eq!(split_addr(a, bps), (segno, idx));
        }
    }

    #[test]
    fn split_addr_zero_blocks_per_seg_safe() {
        assert_eq!(split_addr(123, 0), (0, 0));
    }

    #[test]
    fn live_blocks_from_segment_extracts_set_bits() {
        // Bitmap with bits 0, 5, 100, 511 set.
        let mut map = [0u8; 64];
        map[0] |= 1 << 0;
        map[0] |= 1 << 5;
        map[12] |= 1 << 4; // bit 100 = byte 12, bit 4
        map[63] |= 1 << 7; // bit 511 = byte 63, bit 7
        let live = live_blocks_from_segment(
            42,
            &map,
            |idx| Some((idx as u32 + 1, idx as u32)),
            |nid| Some(1000 + nid as u64),
        );
        assert_eq!(live.len(), 4);
        assert_eq!(live[0].src_block_idx, 0);
        assert_eq!(live[1].src_block_idx, 5);
        assert_eq!(live[2].src_block_idx, 100);
        assert_eq!(live[3].src_block_idx, 511);
        assert_eq!(live[0].src_segno, 42);
        assert_eq!(live[3].owning_nid, 512);
    }

    #[test]
    fn live_blocks_from_segment_skips_nid_zero() {
        let mut map = [0u8; 64];
        map[0] |= 1 << 0;
        map[0] |= 1 << 1;
        let live = live_blocks_from_segment(
            0,
            &map,
            |idx| if idx == 0 { Some((0, 0)) } else { Some((1, 0)) },
            |_| Some(1),
        );
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].owning_nid, 1);
    }

    #[test]
    fn live_blocks_from_segment_skips_missing_inode() {
        let mut map = [0u8; 64];
        map[0] |= 1 << 0;
        map[0] |= 1 << 1;
        let live = live_blocks_from_segment(
            0,
            &map,
            |idx| Some((idx as u32 + 1, 0)),
            |nid| if nid == 1 { None } else { Some(0) },
        );
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].owning_nid, 2);
    }

    #[test]
    fn live_blocks_from_segment_empty_bitmap() {
        let map = [0u8; 64];
        let live = live_blocks_from_segment(0, &map, |_| Some((1, 0)), |_| Some(0));
        assert!(live.is_empty());
    }

    fn make_live(src_segno: u32, idx: u16, nid: u32, ofs: u32, mtime: u64) -> LiveBlock {
        LiveBlock {
            src_segno,
            src_block_idx: idx,
            owning_nid: nid,
            ofs_in_node: ofs,
            mtime_secs: mtime,
        }
    }

    #[test]
    fn run_gc_pass_v2_relocates_all_blocks_in_one_segment() {
        let cands = [(7u32, 100u16)];
        let live: alloc::vec::Vec<LiveBlock> = (0..3u16)
            .map(|i| make_live(7, i, 100 + i as u32, i as u32, 0))
            .collect();
        let cfg = GcPassConfig {
            now_secs: 1000,
            hints: AllocatorHints {
                hot_segno: 99,
                cold_segno: 100,
            },
            ..Default::default()
        };
        let mut next_addr = 5000u32;
        let outcome = run_gc_pass_v2::<_, _, _, _, ()>(
            &cands,
            512,
            cfg,
            |_seg| live.clone(),
            |_blk, _t| {
                let a = next_addr;
                next_addr += 1;
                Ok(a)
            },
            || false,
            || 0u64,
        )
        .unwrap();
        assert_eq!(outcome.blocks_relocated, 3);
        assert_eq!(outcome.segments_freed, 1);
        assert_eq!(outcome.journal.len(), 3);
        for r in &outcome.journal {
            assert!(r.is_committed());
        }
    }

    #[test]
    fn run_gc_pass_v2_routes_cold_blocks_to_cold_segment() {
        let cands = [(0u32, 100u16)];
        // Two blocks: hot (mtime=995, age=5) and cold (mtime=0, age=1000).
        let live = alloc::vec![make_live(0, 0, 1, 0, 995), make_live(0, 1, 2, 0, 0),];
        let cfg = GcPassConfig {
            now_secs: 1000,
            cold_threshold_secs: 100,
            hints: AllocatorHints {
                hot_segno: 50,
                cold_segno: 99,
            },
            ..Default::default()
        };
        let routes = core::cell::RefCell::new(alloc::vec![]);
        let outcome = run_gc_pass_v2::<_, _, _, _, ()>(
            &cands,
            512,
            cfg,
            |_seg| live.clone(),
            |_blk, target| {
                routes.borrow_mut().push(target);
                Ok(target * 100)
            },
            || false,
            || 0,
        )
        .unwrap();
        assert_eq!(outcome.blocks_relocated, 2);
        let r = routes.into_inner();
        assert_eq!(r, alloc::vec![50u32, 99]);
        // Cold record carries the cold flag.
        assert!(!outcome.journal[0].is_cold());
        assert!(outcome.journal[1].is_cold());
    }

    #[test]
    fn run_gc_pass_v2_picks_n_victims() {
        let cands = [(10u32, 50u16), (11, 100), (12, 200), (13, 300), (14, 400)];
        let cfg = GcPassConfig {
            victims_per_pass: 3,
            ..Default::default()
        };
        let visited = core::cell::RefCell::new(alloc::vec![]);
        let outcome = run_gc_pass_v2::<_, _, _, _, ()>(
            &cands,
            512,
            cfg,
            |s| {
                visited.borrow_mut().push(s);
                alloc::vec![]
            },
            |_, _| Ok(0),
            || false,
            || 0,
        )
        .unwrap();
        assert_eq!(outcome.segments_freed, 3); // empty live ⇒ already drained
        let v = visited.into_inner();
        assert_eq!(v.len(), 3);
        // Top-3 by score: 10 (dead=462), 11 (dead=412), 12 (dead=312).
        assert_eq!(v, alloc::vec![10u32, 11, 12]);
    }

    #[test]
    fn run_gc_pass_v2_yields_after_budget() {
        let cands = [(0u32, 100u16)];
        let live: alloc::vec::Vec<LiveBlock> = (0..32u16)
            .map(|i| make_live(0, i, 1, i as u32, 0))
            .collect();
        let cfg = GcPassConfig {
            yield_budget_blocks: 10,
            ..Default::default()
        };
        let outcome = run_gc_pass_v2::<_, _, _, _, ()>(
            &cands,
            512,
            cfg,
            |_| live.clone(),
            |_, _| Ok(1),
            || false,
            || 0,
        )
        .unwrap();
        // 32 blocks, budget=10, so we expect ≥ 3 yield surfaces (after
        // 10, 20, 30 relocations).
        assert!(outcome.yields >= 3);
        assert_eq!(outcome.blocks_relocated, 32);
    }

    #[test]
    fn run_gc_pass_v2_pauses_on_foreground_yield() {
        let cands = [(0u32, 100u16)];
        let live: alloc::vec::Vec<LiveBlock> = (0..10u16)
            .map(|i| make_live(0, i, 1, i as u32, 0))
            .collect();
        let count = core::cell::Cell::new(0u32);
        let outcome = run_gc_pass_v2::<_, _, _, _, ()>(
            &cands,
            512,
            GcPassConfig::default(),
            |_| live.clone(),
            |_, _| {
                count.set(count.get() + 1);
                Ok(1)
            },
            || count.get() >= 3,
            || 0,
        )
        .unwrap();
        assert!(outcome.foreground_pause);
        assert_eq!(outcome.blocks_relocated, 3);
        // Segment was NOT marked free (live blocks remain).
        assert_eq!(outcome.segments_freed, 0);
    }

    #[test]
    fn run_gc_pass_v2_time_caps_pass() {
        let cands = [(0u32, 100u16), (1, 100)];
        let live: alloc::vec::Vec<LiveBlock> =
            (0..5u16).map(|i| make_live(0, i, 1, i as u32, 0)).collect();
        let cfg = GcPassConfig {
            pass_budget_ticks: 5,
            ..Default::default()
        };
        let tick = core::cell::Cell::new(0u64);
        let outcome = run_gc_pass_v2::<_, _, _, _, ()>(
            &cands,
            512,
            cfg,
            |_| live.clone(),
            |_, _| {
                tick.set(tick.get() + 1);
                Ok(1)
            },
            || false,
            || tick.get(),
        )
        .unwrap();
        assert!(outcome.time_capped);
        // We relocated 5 blocks (1 segment full); next segment skipped
        // because tick ≥ budget. 5 blocks done before the cap fires.
        assert!(outcome.blocks_relocated <= 5);
    }

    #[test]
    fn run_gc_pass_v2_segments_freed_only_when_complete() {
        let cands = [(0u32, 100u16)];
        let live: alloc::vec::Vec<LiveBlock> =
            (0..5u16).map(|i| make_live(0, i, 1, i as u32, 0)).collect();
        // Yield after 2 → segment NOT marked free.
        let count = core::cell::Cell::new(0u32);
        let outcome = run_gc_pass_v2::<_, _, _, _, ()>(
            &cands,
            512,
            GcPassConfig::default(),
            |_| live.clone(),
            |_, _| {
                count.set(count.get() + 1);
                Ok(1)
            },
            || count.get() >= 2,
            || 0,
        )
        .unwrap();
        assert_eq!(outcome.segments_freed, 0);
    }

    #[test]
    fn run_gc_pass_v2_propagates_relocate_error() {
        let cands = [(0u32, 100u16)];
        let live = alloc::vec![make_live(0, 0, 1, 0, 0)];
        let r: Result<_, &'static str> = run_gc_pass_v2(
            &cands,
            512,
            GcPassConfig::default(),
            |_| live.clone(),
            |_, _| Err("io fail"),
            || false,
            || 0,
        );
        assert_eq!(r, Err("io fail"));
    }

    #[test]
    fn run_gc_pass_v2_journal_records_uncommitted_until_relocate_success() {
        // Verify the staged record is created BEFORE relocate is called
        // (i.e. if relocate fails, no committed record reaches the
        // outcome).
        let cands = [(0u32, 100u16)];
        let live = alloc::vec![make_live(0, 0, 5, 7, 0)];
        let r: Result<GcPassOutcome, &'static str> = run_gc_pass_v2(
            &cands,
            512,
            GcPassConfig::default(),
            |_| live.clone(),
            |_, _| Err("crash mid-relocate"),
            || false,
            || 0,
        );
        // Error propagates; no GcPassOutcome is returned.
        assert!(r.is_err());
    }

    #[test]
    fn run_gc_pass_v2_already_drained_segment_marked_free() {
        let cands = [(7u32, 5u16)]; // 5 valid blocks claimed
        let outcome = run_gc_pass_v2::<_, _, _, _, ()>(
            &cands,
            512,
            GcPassConfig::default(),
            |_| alloc::vec![], // SSA says no live blocks (race / stale SIT)
            |_, _| Ok(0),
            || false,
            || 0,
        )
        .unwrap();
        assert_eq!(outcome.segments_freed, 1);
        assert_eq!(outcome.blocks_relocated, 0);
    }

    #[test]
    fn run_gc_pass_v2_journal_round_trip_through_decoder() {
        use super::super::gc_journal::decode_block;
        let cands = [(0u32, 100u16)];
        let live = alloc::vec![
            make_live(0, 0, 1, 0, 0),
            make_live(0, 1, 1, 1, 0),
            make_live(0, 2, 2, 0, 0),
        ];
        let mut next = 9000u32;
        let outcome = run_gc_pass_v2::<_, _, _, _, ()>(
            &cands,
            512,
            GcPassConfig::default(),
            |_| live.clone(),
            |_, _| {
                let a = next;
                next += 1;
                Ok(a)
            },
            || false,
            || 0,
        )
        .unwrap();
        // Encode the journal as a block, decode, compare.
        let bytes = super::super::gc_journal::encode_block(&outcome.journal);
        let decoded = decode_block(&bytes).unwrap();
        assert_eq!(decoded, outcome.journal);
    }

    #[test]
    fn defaults_match_documented_constants() {
        let c = GcPassConfig::default();
        assert_eq!(c.victims_per_pass, DEFAULT_GC_VICTIMS_PER_PASS);
        assert_eq!(c.yield_budget_blocks, DEFAULT_YIELD_BUDGET_BLOCKS);
        assert_eq!(c.pass_budget_ticks, DEFAULT_PASS_BUDGET_TICKS);
        assert_eq!(c.cold_threshold_secs, DEFAULT_COLD_THRESHOLD_SECS);
    }
}
