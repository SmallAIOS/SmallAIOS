// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! In-memory mock [`BlockDevice`] for tests and CI.
//!
//! `MockBlockDevice` keeps a single contiguous backing buffer and
//! services reads/writes by `memcpy`. It also has injectable failure
//! modes so the conformance suite can drive the retry path:
//!
//! - [`MockBlockDevice::set_transient_failures`] makes the next *N*
//!   read/write calls return [`BlockError::DeviceBusy`] (transient).
//! - [`MockBlockDevice::set_permanent_failure`] makes every subsequent
//!   call return the chosen error.
//! - [`MockBlockDevice::set_bad_crc_block`] flags one specific LBA as
//!   producing [`BlockError::BadCrc`] on read.
//!
//! The mock is `#![no_std]`-friendly: it uses `alloc::vec::Vec` and
//! `core::cell::RefCell` for interior mutability so the trait methods
//! can keep `&self` semantics where the spec requires it.

use core::cell::RefCell;

use super::{check_buf_alignment, check_lba_in_range, BlockDevice, BlockError};

/// Internal mutable state of [`MockBlockDevice`].
///
/// Wrapped in `RefCell` so reads (which take `&self`) can still
/// decrement counters and read-out injected errors.
#[derive(Debug)]
struct State {
    storage: alloc::vec::Vec<u8>,
    block_size_bytes: u32,
    block_count: u64,
    transient_failures_remaining: u32,
    permanent_failure: Option<BlockError>,
    bad_crc_blocks: alloc::vec::Vec<u64>,
    flush_count: u32,
    read_count: u32,
    write_count: u32,
}

/// In-memory mock [`BlockDevice`] for tests.
///
/// Construct with [`new`](Self::new) (zero-initialized backing) or
/// [`with_pattern`](Self::with_pattern) (fill with `lba_byte * 251` so
/// each block has a distinct fingerprint).
#[derive(Debug)]
pub struct MockBlockDevice {
    state: RefCell<State>,
}

impl MockBlockDevice {
    /// Construct a mock with `block_count` zero-initialized blocks of
    /// `block_size_bytes`.
    pub fn new(block_size_bytes: u32, block_count: u64) -> Self {
        let total = (block_size_bytes as usize)
            .checked_mul(block_count as usize)
            .expect("mock device too large for usize");
        Self {
            state: RefCell::new(State {
                storage: alloc::vec![0u8; total],
                block_size_bytes,
                block_count,
                transient_failures_remaining: 0,
                permanent_failure: None,
                bad_crc_blocks: alloc::vec::Vec::new(),
                flush_count: 0,
                read_count: 0,
                write_count: 0,
            }),
        }
    }

    /// Construct a mock pre-filled with a deterministic pattern so
    /// each block has unique contents (`byte_i = (block_index +
    /// offset_in_block) % 251`).
    pub fn with_pattern(block_size_bytes: u32, block_count: u64) -> Self {
        let dev = Self::new(block_size_bytes, block_count);
        {
            let mut state = dev.state.borrow_mut();
            for blk in 0..block_count {
                for off in 0..block_size_bytes as usize {
                    let byte = ((blk as usize).wrapping_add(off) % 251) as u8;
                    let idx = blk as usize * block_size_bytes as usize + off;
                    state.storage[idx] = byte;
                }
            }
        }
        dev
    }

    /// Inject `n` transient failures into the next `n` read/write
    /// calls. Each surfaces as [`BlockError::DeviceBusy`] (which is
    /// classified as retryable by [`crate::block::retry`]).
    pub fn set_transient_failures(&self, n: u32) {
        self.state.borrow_mut().transient_failures_remaining = n;
    }

    /// Inject a permanent failure: every subsequent read/write returns
    /// this error until cleared.
    pub fn set_permanent_failure(&self, err: BlockError) {
        self.state.borrow_mut().permanent_failure = Some(err);
    }

    /// Clear any injected permanent failure.
    pub fn clear_permanent_failure(&self) {
        self.state.borrow_mut().permanent_failure = None;
    }

    /// Mark `lba` as producing [`BlockError::BadCrc`] on read until
    /// cleared with [`clear_bad_crc_blocks`](Self::clear_bad_crc_blocks).
    pub fn set_bad_crc_block(&self, lba: u64) {
        self.state.borrow_mut().bad_crc_blocks.push(lba);
    }

    /// Clear all CRC-bad markers.
    pub fn clear_bad_crc_blocks(&self) {
        self.state.borrow_mut().bad_crc_blocks.clear();
    }

    /// Flush call count (test introspection).
    pub fn flush_count(&self) -> u32 {
        self.state.borrow().flush_count
    }

    /// Successful-read call count (test introspection).
    pub fn read_count(&self) -> u32 {
        self.state.borrow().read_count
    }

    /// Successful-write call count (test introspection).
    pub fn write_count(&self) -> u32 {
        self.state.borrow().write_count
    }

    /// Pre-load specific bytes at an LBA for deterministic-content
    /// tests.
    pub fn preload(&self, lba: u64, data: &[u8]) {
        let mut state = self.state.borrow_mut();
        let bs = state.block_size_bytes as usize;
        assert!(
            data.len().is_multiple_of(bs),
            "preload buf must be block-aligned"
        );
        let start = lba as usize * bs;
        let end = start + data.len();
        state.storage[start..end].copy_from_slice(data);
    }

    /// Read raw bytes for inspection (does NOT count toward
    /// [`read_count`](Self::read_count)).
    pub fn snapshot(&self, lba: u64, len: usize) -> alloc::vec::Vec<u8> {
        let state = self.state.borrow();
        let bs = state.block_size_bytes as usize;
        let start = lba as usize * bs;
        state.storage[start..start + len].to_vec()
    }
}

impl BlockDevice for MockBlockDevice {
    fn read_block(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let mut state = self.state.borrow_mut();
        check_buf_alignment(buf.len(), state.block_size_bytes)?;
        check_lba_in_range(lba, buf.len(), state.block_size_bytes, state.block_count)?;

        if let Some(err) = state.permanent_failure {
            return Err(err);
        }
        if state.transient_failures_remaining > 0 {
            state.transient_failures_remaining -= 1;
            return Err(BlockError::DeviceBusy);
        }
        // Bad-CRC injection — applies to any block in the I/O range.
        let blocks = (buf.len() / state.block_size_bytes as usize) as u64;
        for cur in lba..lba + blocks {
            if state.bad_crc_blocks.contains(&cur) {
                return Err(BlockError::BadCrc);
            }
        }

        let bs = state.block_size_bytes as usize;
        let start = lba as usize * bs;
        let end = start + buf.len();
        buf.copy_from_slice(&state.storage[start..end]);
        state.read_count += 1;
        Ok(())
    }

    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        let mut state = self.state.borrow_mut();
        check_buf_alignment(buf.len(), state.block_size_bytes)?;
        check_lba_in_range(lba, buf.len(), state.block_size_bytes, state.block_count)?;

        if let Some(err) = state.permanent_failure {
            return Err(err);
        }
        if state.transient_failures_remaining > 0 {
            state.transient_failures_remaining -= 1;
            return Err(BlockError::DeviceBusy);
        }

        let bs = state.block_size_bytes as usize;
        let start = lba as usize * bs;
        let end = start + buf.len();
        state.storage[start..end].copy_from_slice(buf);
        state.write_count += 1;
        Ok(())
    }

    fn block_size_bytes(&self) -> u32 {
        self.state.borrow().block_size_bytes
    }

    fn block_count(&self) -> u64 {
        self.state.borrow().block_count
    }

    fn flush(&mut self) -> Result<(), BlockError> {
        let mut state = self.state.borrow_mut();
        if let Some(err) = state.permanent_failure {
            return Err(err);
        }
        state.flush_count += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_fill_distinct_blocks() {
        let dev = MockBlockDevice::with_pattern(512, 4);
        let b0 = dev.snapshot(0, 512);
        let b1 = dev.snapshot(1, 512);
        let b2 = dev.snapshot(2, 512);
        let b3 = dev.snapshot(3, 512);
        assert_ne!(b0, b1);
        assert_ne!(b1, b2);
        assert_ne!(b2, b3);
    }

    #[test]
    fn read_returns_stored_bytes() {
        let dev = MockBlockDevice::with_pattern(512, 4);
        let expected = dev.snapshot(2, 512);
        let mut got = alloc::vec![0u8; 512];
        dev.read_block(2, &mut got).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn transient_failure_decrements() {
        let mut dev = MockBlockDevice::new(512, 4);
        dev.set_transient_failures(2);
        let mut buf = alloc::vec![0u8; 512];
        assert_eq!(dev.read_block(0, &mut buf), Err(BlockError::DeviceBusy));
        assert_eq!(dev.read_block(0, &mut buf), Err(BlockError::DeviceBusy));
        assert!(dev.read_block(0, &mut buf).is_ok());
        // Same for writes.
        dev.set_transient_failures(1);
        assert_eq!(
            dev.write_block(0, &alloc::vec![0u8; 512]),
            Err(BlockError::DeviceBusy)
        );
        assert!(dev.write_block(0, &alloc::vec![0u8; 512]).is_ok());
    }

    #[test]
    fn permanent_failure_persists() {
        let dev = MockBlockDevice::new(512, 4);
        dev.set_permanent_failure(BlockError::MediaError);
        let mut buf = alloc::vec![0u8; 512];
        assert_eq!(dev.read_block(0, &mut buf), Err(BlockError::MediaError));
        assert_eq!(dev.read_block(0, &mut buf), Err(BlockError::MediaError));
        dev.clear_permanent_failure();
        assert!(dev.read_block(0, &mut buf).is_ok());
    }

    #[test]
    fn bad_crc_injection() {
        let dev = MockBlockDevice::new(512, 4);
        dev.set_bad_crc_block(2);
        let mut buf = alloc::vec![0u8; 512];
        assert!(dev.read_block(0, &mut buf).is_ok());
        assert!(dev.read_block(1, &mut buf).is_ok());
        assert_eq!(dev.read_block(2, &mut buf), Err(BlockError::BadCrc));
        assert!(dev.read_block(3, &mut buf).is_ok());
    }

    #[test]
    fn flush_increments_counter() {
        let mut dev = MockBlockDevice::new(512, 4);
        assert_eq!(dev.flush_count(), 0);
        dev.flush().unwrap();
        dev.flush().unwrap();
        assert_eq!(dev.flush_count(), 2);
    }

    #[test]
    fn unaligned_read_rejected() {
        let dev = MockBlockDevice::new(512, 4);
        let mut buf = alloc::vec![0u8; 513];
        assert_eq!(dev.read_block(0, &mut buf), Err(BlockError::Unaligned));
    }

    #[test]
    fn out_of_range_read_rejected() {
        let dev = MockBlockDevice::new(512, 4);
        let mut buf = alloc::vec![0u8; 512];
        assert_eq!(dev.read_block(4, &mut buf), Err(BlockError::OutOfRange));
        assert_eq!(dev.read_block(5, &mut buf), Err(BlockError::OutOfRange));
    }

    #[test]
    fn capacity_bytes_works() {
        let dev = MockBlockDevice::new(4096, 16);
        assert_eq!(dev.capacity_bytes(), 65536);
    }
}
