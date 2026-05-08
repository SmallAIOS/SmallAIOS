// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! In-memory mock [`FlashDevice`] for tests, fixtures, and CI.
//!
//! Mirrors NOR semantics: every byte starts at `0xFF` after `new()`, and
//! `program` may only flip bits 1→0 within a page that is currently in
//! its erased state. To flip a 0→1 the page MUST first be erased
//! (which clears the entire enclosing erase-block).
//!
//! Failure-injection knobs let the conformance suite exercise the
//! wear-leveling / BBT recovery paths without hardware:
//!
//! * [`MockFlashDevice::inject_bad_block_on_erase`] — the next erase of
//!   a configured block returns [`FlashError::EraseFailure`].
//! * [`MockFlashDevice::inject_bit_flip`] — toggle a single bit in
//!   storage; the next read sees the flipped bit (used to drive
//!   [`FlashError::EccUncorrectable`] paths once an ECC layer wraps the
//!   mock; on the bare mock the flip is observable to the caller).
//! * [`MockFlashDevice::inject_bad_block`] — pre-mark a block bad
//!   (factory-style).
//!
//! The mock is `#![no_std]`-friendly: it uses `alloc::vec::Vec`.

use alloc::vec;
use alloc::vec::Vec;

use super::device::{check_program_alignment, check_range, FlashDevice, FlashError};

/// In-memory mock [`FlashDevice`].
///
/// Used by:
///
/// * Phase 1 conformance tests (read / program / erase / alignment /
///   bad-block paths).
/// * Phase 3 littlefs read-path tests (mount a synthetic littlefs
///   image laid out in the mock's storage).
/// * Future phase 6 1M-cycle wear-leveling stress.
#[derive(Debug)]
pub struct MockFlashDevice {
    storage: Vec<u8>,
    block_size: u32,
    page_size: u32,
    block_count: u64,
    erase_counts: Vec<u32>,
    bad_blocks: Vec<bool>,
    inject_eperm_on_erase: Option<u64>,
    /// Counters: total successful reads / programs / erases.
    read_count: u32,
    program_count: u32,
    erase_count_total: u32,
}

impl MockFlashDevice {
    /// Construct a fresh mock with `blocks` erase-blocks, each of
    /// `block_size` bytes split into pages of `page_size`.
    ///
    /// # Panics
    ///
    /// * If `block_size == 0`, `page_size == 0`, or `block_size %
    ///   page_size != 0`.
    /// * If the total capacity exceeds `usize::MAX`.
    pub fn new(blocks: u64, block_size: u32, page_size: u32) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        assert!(page_size > 0, "page_size must be > 0");
        assert!(
            block_size.is_multiple_of(page_size),
            "block_size ({block_size}) must be a multiple of page_size ({page_size})"
        );
        let total = (block_size as usize)
            .checked_mul(blocks as usize)
            .expect("mock device too large for usize");
        Self {
            storage: vec![0xFFu8; total],
            block_size,
            page_size,
            block_count: blocks,
            erase_counts: vec![0u32; blocks as usize],
            bad_blocks: vec![false; blocks as usize],
            inject_eperm_on_erase: None,
            read_count: 0,
            program_count: 0,
            erase_count_total: 0,
        }
    }

    /// Pre-mark `block` as bad (factory-style).
    pub fn inject_bad_block(&mut self, block: u64) {
        if let Some(slot) = self.bad_blocks.get_mut(block as usize) {
            *slot = true;
        }
    }

    /// Make the next [`erase`](FlashDevice::erase) of `block` return
    /// [`FlashError::EraseFailure`]. Used to drive the wear-leveling /
    /// BBT runtime-bad path.
    pub fn inject_bad_block_on_erase(&mut self, block: u64) {
        self.inject_eperm_on_erase = Some(block);
    }

    /// Toggle a single bit in storage at byte `offset`, bit position
    /// `bit` (0..8). Useful for ECC-injection tests once an ECC layer
    /// wraps the mock.
    pub fn inject_bit_flip(&mut self, offset: u64, bit: u8) {
        assert!(bit < 8, "bit must be 0..8");
        if let Some(byte) = self.storage.get_mut(offset as usize) {
            *byte ^= 1 << bit;
        }
    }

    /// Per-block erase count (test introspection).
    pub fn erase_count(&self, block: u64) -> u32 {
        self.erase_counts.get(block as usize).copied().unwrap_or(0)
    }

    /// Total successful read calls.
    pub fn read_count(&self) -> u32 {
        self.read_count
    }

    /// Total successful program calls.
    pub fn program_count(&self) -> u32 {
        self.program_count
    }

    /// Total successful erase calls.
    pub fn erase_total(&self) -> u32 {
        self.erase_count_total
    }

    /// Direct byte-level snapshot of storage (`offset`, `len`). Bypasses
    /// the public `read` path so ECC injection isn't observable.
    pub fn snapshot(&self, offset: u64, len: usize) -> Vec<u8> {
        self.storage[offset as usize..offset as usize + len].to_vec()
    }

    /// Direct byte-level write into storage, bypassing program
    /// semantics. Used by fixture generators to lay down a known-good
    /// image (e.g. a littlefs golden fixture) before mounting.
    pub fn force_write(&mut self, offset: u64, data: &[u8]) {
        let start = offset as usize;
        let end = start + data.len();
        self.storage[start..end].copy_from_slice(data);
    }

    /// Direct byte-level fill of storage with `byte`, bypassing program
    /// semantics. Useful in tests that want to leave the mock in an
    /// erased state without iterating per block.
    pub fn force_fill(&mut self, byte: u8) {
        for cell in &mut self.storage {
            *cell = byte;
        }
    }
}

impl FlashDevice for MockFlashDevice {
    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), FlashError> {
        check_range(offset, buf.len() as u64, self.block_size, self.block_count)?;
        let start = offset as usize;
        let end = start + buf.len();
        buf.copy_from_slice(&self.storage[start..end]);
        // SAFETY-NOTE: the read_count update would normally need
        // interior mutability, but tests inspect it via &mut access only;
        // for simplicity the mock keeps it on &mut self in `program` /
        // `erase`. We omit the `read` counter to keep the trait shape
        // honest (read takes &self).
        Ok(())
    }

    fn program(&mut self, offset: u64, buf: &[u8]) -> Result<(), FlashError> {
        check_range(offset, buf.len() as u64, self.block_size, self.block_count)?;
        check_program_alignment(offset, buf.len() as u64, self.page_size)?;

        // Reject if any byte in the target window is in a bad block.
        let first_block = offset / u64::from(self.block_size);
        let last_byte = offset + buf.len() as u64;
        let last_block = (last_byte - 1) / u64::from(self.block_size);
        for blk in first_block..=last_block {
            if self.bad_blocks[blk as usize] {
                return Err(FlashError::BadBlock);
            }
        }

        // NOR semantics: bits may only flip 1 → 0. Equivalently:
        // (current & buf) must equal buf — every bit set in buf must
        // currently be 1 in storage. If any bit set in buf is 0 in
        // storage, that's a 0→1 flip request → ProgramOnDirty.
        let start = offset as usize;
        for (i, byte) in buf.iter().enumerate() {
            let cur = self.storage[start + i];
            // bits we're trying to set to 1 that are currently 0:
            let flip_to_one = byte & !cur;
            if flip_to_one != 0 {
                return Err(FlashError::ProgramOnDirty);
            }
        }

        // Apply the program (current AND buf).
        for (i, byte) in buf.iter().enumerate() {
            self.storage[start + i] &= *byte;
        }
        self.program_count += 1;
        Ok(())
    }

    fn erase(&mut self, block: u64) -> Result<(), FlashError> {
        if block >= self.block_count {
            return Err(FlashError::OutOfRange);
        }
        if self.bad_blocks[block as usize] {
            return Err(FlashError::BadBlock);
        }

        if let Some(target) = self.inject_eperm_on_erase {
            if target == block {
                self.inject_eperm_on_erase = None;
                return Err(FlashError::EraseFailure);
            }
        }

        let start = (block as usize) * (self.block_size as usize);
        let end = start + self.block_size as usize;
        for cell in &mut self.storage[start..end] {
            *cell = 0xFF;
        }
        self.erase_counts[block as usize] = self.erase_counts[block as usize].saturating_add(1);
        self.erase_count_total = self.erase_count_total.saturating_add(1);
        Ok(())
    }

    fn block_size_bytes(&self) -> u32 {
        self.block_size
    }

    fn page_size_bytes(&self) -> u32 {
        self.page_size
    }

    fn block_count(&self) -> u64 {
        self.block_count
    }

    fn is_bad(&self, block: u64) -> bool {
        self.bad_blocks
            .get(block as usize)
            .copied()
            .unwrap_or(false)
    }

    fn mark_bad(&mut self, block: u64) -> Result<(), FlashError> {
        if block >= self.block_count {
            return Err(FlashError::OutOfRange);
        }
        self.bad_blocks[block as usize] = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_device_reads_all_ones() {
        let dev = MockFlashDevice::new(4, 4096, 256);
        let mut buf = vec![0u8; 4096];
        dev.read(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn program_then_read_round_trip() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        let data = [0xAAu8; 256];
        dev.program(0, &data).unwrap();
        let mut got = [0u8; 256];
        dev.read(0, &mut got).unwrap();
        assert_eq!(got, data);
    }

    #[test]
    fn program_on_dirty_rejected() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        // Program bits to 0 (e.g. 0x00 at offset 0).
        dev.program(0, &[0u8; 256]).unwrap();
        // Now request 0xFF (i.e. all bits 1) — every bit is a 0→1 flip.
        let res = dev.program(0, &[0xFFu8; 256]);
        assert_eq!(res, Err(FlashError::ProgramOnDirty));
        // Storage unchanged.
        let mut got = [0u8; 256];
        dev.read(0, &mut got).unwrap();
        assert_eq!(got, [0u8; 256]);
    }

    #[test]
    fn erase_resets_to_all_ones() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        dev.program(0, &[0u8; 256]).unwrap();
        dev.erase(0).unwrap();
        let mut got = [0u8; 4096];
        dev.read(0, &mut got).unwrap();
        assert!(got.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn erase_increments_counter() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        assert_eq!(dev.erase_count(0), 0);
        dev.erase(0).unwrap();
        dev.erase(0).unwrap();
        assert_eq!(dev.erase_count(0), 2);
        assert_eq!(dev.erase_total(), 2);
    }

    #[test]
    fn bad_block_rejects_program() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        dev.inject_bad_block(1);
        assert!(dev.is_bad(1));
        let res = dev.program(4096, &[0u8; 256]);
        assert_eq!(res, Err(FlashError::BadBlock));
    }

    #[test]
    fn bad_block_rejects_erase() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        dev.inject_bad_block(2);
        assert_eq!(dev.erase(2), Err(FlashError::BadBlock));
    }

    #[test]
    fn mark_bad_persists() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        assert!(!dev.is_bad(3));
        dev.mark_bad(3).unwrap();
        assert!(dev.is_bad(3));
    }

    #[test]
    fn inject_eperm_on_erase_fires_once() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        dev.inject_bad_block_on_erase(1);
        assert_eq!(dev.erase(1), Err(FlashError::EraseFailure));
        // Second call passes (one-shot injection).
        assert!(dev.erase(1).is_ok());
    }

    #[test]
    fn unaligned_program_rejected() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        // Offset 1 is not page-aligned (page = 256).
        let res = dev.program(1, &[0u8; 256]);
        assert_eq!(res, Err(FlashError::Unaligned));
        // Length 255 is not page-aligned.
        let res = dev.program(0, &[0u8; 255]);
        assert_eq!(res, Err(FlashError::Unaligned));
    }

    #[test]
    fn out_of_range_read_rejected() {
        let dev = MockFlashDevice::new(4, 4096, 256);
        let mut buf = vec![0u8; 100];
        // Capacity = 4 * 4096 = 16384.
        assert_eq!(dev.read(16384, &mut buf), Err(FlashError::OutOfRange));
    }

    #[test]
    fn out_of_range_erase_rejected() {
        let mut dev = MockFlashDevice::new(4, 4096, 256);
        assert_eq!(dev.erase(4), Err(FlashError::OutOfRange));
    }

    #[test]
    fn capacity_bytes_default() {
        let dev = MockFlashDevice::new(8, 4096, 256);
        assert_eq!(dev.capacity_bytes(), 8 * 4096);
    }

    #[test]
    fn force_write_bypasses_semantics() {
        let mut dev = MockFlashDevice::new(2, 4096, 256);
        // Lay down a 0x55 pattern without going through `program`
        // (which would require alignment, etc.).
        dev.force_write(0, &[0x55u8; 4096]);
        let mut got = [0u8; 4096];
        dev.read(0, &mut got).unwrap();
        assert!(got.iter().all(|&b| b == 0x55));
    }

    #[test]
    fn force_fill_resets_storage() {
        let mut dev = MockFlashDevice::new(2, 4096, 256);
        dev.force_fill(0xAA);
        let mut got = [0u8; 4096];
        dev.read(0, &mut got).unwrap();
        assert!(got.iter().all(|&b| b == 0xAA));
    }
}
