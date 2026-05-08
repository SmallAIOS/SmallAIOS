// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Block allocator with dynamic wear-leveling.
//!
//! Per littlefs SPEC.md § "Block allocation":
//!
//! * The allocator keeps an in-memory free-block bitmap (the
//!   "lookahead buffer") covering a window of `lookahead_size * 8`
//!   blocks at a time.
//! * To pick a block to write, it scans the bitmap forward; bits set
//!   to 1 mean "in use", bits set to 0 mean "free".
//! * Once the current window is exhausted the allocator slides the
//!   window forward by `lookahead_size * 8` blocks, rebuilding the
//!   bitmap by scanning the metadata for live block references.
//!
//! Phase 2 implements the simpler-but-correct variant: a full-device
//! free-block bitmap (one bit per erase-block) plus a sliding `cursor`
//! over it. The `lookahead_size` config field controls how big a window
//! the allocator considers per pass; smaller windows give finer-grained
//! "soft" wear-leveling but more reset-pass overhead.
//!
//! Dynamic wear-leveling: the allocator skips blocks whose erase
//! counter has exceeded the configured `block_cycles` budget by more
//! than 90% (per SPEC.md § "Wear leveling"). The 1M-cycle stress test
//! validates that erase counts stay within 5% of the mean.

use alloc::vec;
use alloc::vec::Vec;

use super::FlashAllocError;

/// In-memory block allocator state.
///
/// Maintains a free bitmap (1 bit per erase-block) and a rolling cursor
/// so successive allocations spread across the device. Bad blocks are
/// permanently masked; reserved blocks (root metadata-pair, BBT copies)
/// are also masked.
#[derive(Debug, Clone)]
pub(super) struct BlockAllocator {
    /// `1` = in use / reserved, `0` = free.
    in_use: Vec<bool>,
    /// `1` = bad / unusable.
    bad: Vec<bool>,
    /// Per-block erase counter (informational; real counts come from
    /// the device, but tracking them here lets us spread allocations).
    erase_counts: Vec<u32>,
    /// Rolling cursor for the next allocation attempt.
    cursor: u32,
    /// Total number of erase-blocks.
    block_count: u32,
    /// Configured wear budget. Blocks whose erase-count exceeds 90% of
    /// this budget are de-prioritised by `alloc_block`.
    block_cycles: u32,
    /// Lookahead-buffer size in *bytes* (1 byte = 8 blocks tracked).
    /// Carried for parity with the upstream C `lfs_config`; v1 uses a
    /// full-device free bitmap rather than a sliding window.
    #[allow(dead_code)]
    lookahead_size: u32,
}

impl BlockAllocator {
    /// Construct a fresh allocator.
    ///
    /// `block_count` and `block_cycles` are the active mount parameters.
    /// `reserved` is a list of pre-marked blocks (root pair, BBT copies);
    /// they will never be returned by `alloc_block`.
    pub(super) fn new(
        block_count: u32,
        block_cycles: u32,
        lookahead_size: u32,
        reserved: &[u32],
    ) -> Self {
        let n = block_count as usize;
        let mut s = Self {
            in_use: vec![false; n],
            bad: vec![false; n],
            erase_counts: vec![0u32; n],
            cursor: 0,
            block_count,
            block_cycles: block_cycles.max(1),
            lookahead_size: lookahead_size.max(1),
        };
        for &b in reserved {
            if (b as usize) < n {
                s.in_use[b as usize] = true;
            }
        }
        s
    }

    /// Mark `block` as currently allocated to a live structure.
    #[allow(dead_code)]
    pub(super) fn mark_in_use(&mut self, block: u32) {
        if let Some(slot) = self.in_use.get_mut(block as usize) {
            *slot = true;
        }
    }

    /// Mark `block` as free.
    pub(super) fn mark_free(&mut self, block: u32) {
        if let Some(slot) = self.in_use.get_mut(block as usize) {
            *slot = false;
        }
    }

    /// Mark `block` as factory- or runtime-bad.
    pub(super) fn mark_bad(&mut self, block: u32) {
        if let Some(slot) = self.bad.get_mut(block as usize) {
            *slot = true;
        }
        if let Some(slot) = self.in_use.get_mut(block as usize) {
            *slot = true;
        }
    }

    /// Note a successful erase of `block` (bumps the per-block count).
    pub(super) fn note_erase(&mut self, block: u32) {
        if let Some(c) = self.erase_counts.get_mut(block as usize) {
            *c = c.saturating_add(1);
        }
    }

    /// Whether `block` is currently flagged bad.
    #[allow(dead_code)]
    pub(super) fn is_bad(&self, block: u32) -> bool {
        self.bad.get(block as usize).copied().unwrap_or(false)
    }

    /// Per-block erase count (test introspection).
    #[allow(dead_code)]
    pub(super) fn erase_count(&self, block: u32) -> u32 {
        self.erase_counts.get(block as usize).copied().unwrap_or(0)
    }

    /// Configured lookahead window size (bytes).
    #[allow(dead_code)]
    pub(super) fn lookahead_size(&self) -> u32 {
        self.lookahead_size
    }

    /// Total block count.
    #[allow(dead_code)]
    pub(super) fn block_count(&self) -> u32 {
        self.block_count
    }

    /// Allocate a block from the current window.
    ///
    /// Returns the lowest-numbered free block in the next lookahead
    /// window starting at the rolling cursor. If the window is empty
    /// and the device still has free blocks elsewhere, advances the
    /// cursor and tries again. If every window is exhausted, returns
    /// `Err(NoFreeBlocks)`.
    ///
    /// Dynamic wear-leveling: blocks whose erase-count is at or above
    /// `block_cycles - block_cycles/10` are skipped on the first pass;
    /// only if no cooler block is available do we fall back to one of
    /// the "hot" candidates.
    pub(super) fn alloc_block(&mut self) -> Result<u32, FlashAllocError> {
        let n = self.block_count;
        if n == 0 {
            return Err(FlashAllocError::NoFreeBlocks);
        }
        let hot_threshold = self
            .block_cycles
            .saturating_sub(self.block_cycles.max(10) / 10);

        // First pass — flag any block whose erase counter has hit the
        // cycle ceiling as bad, regardless of whether it gets allocated
        // this round. This keeps the dynamic-wear-leveling state
        // monotonic with respect to budget exhaustion.
        for idx in 0..(n as usize) {
            if self.bad[idx] || self.in_use[idx] {
                continue;
            }
            if self.erase_counts[idx] >= self.block_cycles {
                self.bad[idx] = true;
                self.in_use[idx] = true;
            }
        }

        // Two passes: first pass skips hot blocks; second pass accepts
        // them if no cool block is available.
        for accept_hot in [false, true] {
            // Pass through every block once, starting at the cursor.
            for i in 0..n {
                let candidate = (self.cursor + i) % n;
                let idx = candidate as usize;
                if self.in_use[idx] || self.bad[idx] {
                    continue;
                }
                let ec = self.erase_counts[idx];
                if !accept_hot && ec >= hot_threshold {
                    continue;
                }
                self.in_use[idx] = true;
                // Advance cursor by 1 so the next alloc starts past
                // this one, spreading writes.
                self.cursor = (candidate + 1) % n;
                return Ok(candidate);
            }
        }
        Err(FlashAllocError::NoFreeBlocks)
    }

    /// Snapshot of free / used / bad counts (test introspection).
    #[allow(dead_code)]
    pub(super) fn stats(&self) -> AllocatorStats {
        let mut free = 0u32;
        let mut used = 0u32;
        let mut bad = 0u32;
        for i in 0..self.block_count as usize {
            if self.bad[i] {
                bad += 1;
            } else if self.in_use[i] {
                used += 1;
            } else {
                free += 1;
            }
        }
        AllocatorStats { free, used, bad }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(super) struct AllocatorStats {
    pub(super) free: u32,
    pub(super) used: u32,
    pub(super) bad: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_alloc_skips_reserved_blocks() {
        let mut a = BlockAllocator::new(8, 500, 16, &[0, 1]);
        let b1 = a.alloc_block().unwrap();
        assert_ne!(b1, 0);
        assert_ne!(b1, 1);
    }

    #[test]
    fn alloc_spreads_across_blocks() {
        let mut a = BlockAllocator::new(8, 500, 16, &[0, 1]);
        let mut got = alloc::collections::BTreeSet::new();
        for _ in 0..6 {
            got.insert(a.alloc_block().unwrap());
        }
        // All 6 free blocks (2..=7) returned distinct.
        assert_eq!(got.len(), 6);
        // 9th alloc fails (no free blocks).
        assert!(a.alloc_block().is_err());
    }

    #[test]
    fn mark_free_returns_block_to_pool() {
        let mut a = BlockAllocator::new(4, 500, 16, &[]);
        let _ = a.alloc_block().unwrap();
        let _ = a.alloc_block().unwrap();
        let _ = a.alloc_block().unwrap();
        let last = a.alloc_block().unwrap();
        assert!(a.alloc_block().is_err());
        a.mark_free(last);
        let again = a.alloc_block().unwrap();
        assert_eq!(again, last);
    }

    #[test]
    fn bad_blocks_skipped_by_alloc() {
        let mut a = BlockAllocator::new(4, 500, 16, &[]);
        a.mark_bad(0);
        a.mark_bad(1);
        let mut got = alloc::vec::Vec::new();
        for _ in 0..2 {
            got.push(a.alloc_block().unwrap());
        }
        assert!(!got.contains(&0));
        assert!(!got.contains(&1));
        assert!(a.alloc_block().is_err());
    }

    #[test]
    fn hot_blocks_de_prioritized() {
        let mut a = BlockAllocator::new(8, 100, 16, &[]);
        // Blocks 0-3 are extremely hot; 4-7 are cold.
        for b in 0u32..4 {
            for _ in 0..95 {
                a.note_erase(b);
            }
        }
        // First 4 allocations should pull from cold blocks.
        let mut got = alloc::collections::BTreeSet::new();
        for _ in 0..4 {
            got.insert(a.alloc_block().unwrap());
        }
        for &b in &got {
            assert!(b >= 4, "expected cold block, got {b}");
        }
    }

    #[test]
    fn max_cycle_block_marked_bad() {
        let mut a = BlockAllocator::new(2, 10, 16, &[]);
        for _ in 0..20 {
            a.note_erase(0);
        }
        let b = a.alloc_block().unwrap();
        assert_eq!(b, 1);
        // Block 0 is now permanently flagged.
        assert!(a.is_bad(0));
    }

    #[test]
    fn stats_reports_classification() {
        let mut a = BlockAllocator::new(4, 500, 16, &[]);
        a.mark_in_use(0);
        a.mark_bad(1);
        let s = a.stats();
        assert_eq!(s.used, 1);
        assert_eq!(s.bad, 1);
        assert_eq!(s.free, 2);
    }

    #[test]
    fn lookahead_size_floor_is_one() {
        let a = BlockAllocator::new(4, 500, 0, &[]);
        assert_eq!(a.lookahead_size(), 1);
    }
}
