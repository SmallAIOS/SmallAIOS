// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bad Block Table.
//!
//! Phase 2 of `embedded-flash-fs-v1` keeps the BBT entirely in-memory:
//! mount populates it by querying [`FlashDevice::is_bad`] for every
//! block, and runtime program/erase failures add entries via
//! [`FlashDevice::mark_bad`]. Persistence to start+end-of-flash duplicate
//! tables (per the spec § "Bad Block Table integration") lands as a
//! follow-up — the in-memory layer is sufficient for write-path
//! correctness because the underlying mock device persists `mark_bad`
//! calls itself, so the BBT survives a remount of the same device
//! instance.
//!
//! See `openspec/changes/embedded-flash-fs-v1/specs/fs-flash-littlefs/spec.md`
//! § "Bad Block Table integration" for the requirement contract.

use alloc::vec;
use alloc::vec::Vec;

use super::super::device::FlashDevice;

/// In-memory bad-block table.
///
/// One bit per erase-block: `true` = bad, `false` = good. Loaded once at
/// mount, mutated in place during write-path operations.
#[derive(Debug, Clone)]
pub(super) struct BadBlockTable {
    bits: Vec<bool>,
}

impl BadBlockTable {
    /// Construct an empty BBT for `block_count` blocks.
    pub(super) fn new(block_count: u32) -> Self {
        Self {
            bits: vec![false; block_count as usize],
        }
    }

    /// Populate the BBT by scanning the device's `is_bad` view.
    pub(super) fn from_device<D: FlashDevice>(dev: &D) -> Self {
        let n = dev.block_count() as u32;
        let mut t = Self::new(n);
        for b in 0..n {
            if dev.is_bad(u64::from(b)) {
                t.bits[b as usize] = true;
            }
        }
        t
    }

    /// Whether `block` is currently bad.
    pub(super) fn is_bad(&self, block: u32) -> bool {
        self.bits.get(block as usize).copied().unwrap_or(false)
    }

    /// Mark `block` bad in the in-memory table.
    pub(super) fn mark_bad(&mut self, block: u32) {
        if let Some(slot) = self.bits.get_mut(block as usize) {
            *slot = true;
        }
    }

    /// Iterator over bad block indices.
    pub(super) fn iter_bad(&self) -> impl Iterator<Item = u32> + '_ {
        self.bits
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if *b { Some(i as u32) } else { None })
    }

    /// Number of bad blocks tracked.
    #[allow(dead_code)]
    pub(super) fn count(&self) -> u32 {
        self.bits.iter().filter(|&&b| b).count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash::mock::MockFlashDevice;

    #[test]
    fn fresh_bbt_is_empty() {
        let t = BadBlockTable::new(8);
        assert_eq!(t.count(), 0);
        for b in 0..8 {
            assert!(!t.is_bad(b));
        }
    }

    #[test]
    fn mark_bad_persists() {
        let mut t = BadBlockTable::new(4);
        t.mark_bad(2);
        assert!(t.is_bad(2));
        assert!(!t.is_bad(0));
        assert_eq!(t.count(), 1);
    }

    #[test]
    fn mark_bad_oob_silently_ignored() {
        let mut t = BadBlockTable::new(2);
        t.mark_bad(99);
        assert!(!t.is_bad(99));
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn from_device_picks_up_factory_bad() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        dev.inject_bad_block(2);
        let t = BadBlockTable::from_device(&dev);
        assert!(t.is_bad(2));
        assert!(!t.is_bad(0));
        assert_eq!(t.count(), 1);
    }

    #[test]
    fn iter_bad_lists_indices() {
        let mut t = BadBlockTable::new(8);
        t.mark_bad(1);
        t.mark_bad(5);
        let v: alloc::vec::Vec<_> = t.iter_bad().collect();
        assert_eq!(v, alloc::vec![1u32, 5u32]);
    }
}
