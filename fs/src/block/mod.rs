// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Block-device abstraction for SmallAIOS on-disk filesystems.
//!
//! This module defines the single `BlockDevice` trait used by every
//! consumer of block I/O in the `fs` crate (squashfs, F2FS, GPT parser,
//! A/B boot pointer, delta apply pipeline). The trait is intentionally
//! minimal:
//!
//! ```ignore
//! pub trait BlockDevice {
//!     fn read_block(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError>;
//!     fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError>;
//!     fn block_size_bytes(&self) -> u32;
//!     fn block_count(&self) -> u64;
//!     fn flush(&mut self) -> Result<(), BlockError>;
//! }
//! ```
//!
//! The crate operates **natively at 4 KiB block size**. Devices that
//! report a different logical/physical sector size are wrapped by an
//! emulation layer in [`sector`] so consumers always see 4 KiB blocks.
//!
//! Errors are typed (`BlockError`) and converted to POSIX errno values
//! at the syscall boundary by [`errno::block_error_to_errno`]. The
//! retry policy (per-op timeout + 3-retry exponential backoff) is in
//! [`retry`]; fail-after-retries is non-configurable so an unattended
//! appliance never wedges on a single bad sector.
//!
//! Owned by `embedded-filesystem-v1` Phase 1.

pub mod errno;
pub mod retry;
pub mod sector;

pub mod mock;

// Convenience re-exports so `use smallaios_fs::block::block_error_to_errno`
// works without an extra `errno::` segment.
pub use errno::{block_error_to_errno, block_error_to_negative_errno};

use core::fmt;

/// Native filesystem block size used by the SmallAIOS `fs` crate.
///
/// The squashfs reader, F2FS reader/writer, GPT parser, A/B boot
/// pointer, and delta apply pipeline all express their I/O in terms
/// of this value. Underlying block devices that report a different
/// logical/physical sector size are wrapped by [`sector`] so callers
/// always see 4 KiB blocks.
pub const FS_BLOCK_SIZE_BYTES: u32 = 4096;

/// Block-device trait implemented by every per-arch driver.
///
/// `BlockDevice` is the only abstraction layer between the on-disk
/// filesystem code and the underlying transport (virtio-blk, NVMe,
/// SDHCI/eMMC, AHCI). Each implementation documents its own native
/// sector size; consumers that always want 4 KiB blocks should wrap
/// the device with [`sector::SectorAdapter`].
///
/// # Alignment
///
/// `buf` length on `read_block`/`write_block` MUST be a multiple of
/// [`block_size_bytes`](Self::block_size_bytes). Otherwise the call
/// returns [`BlockError::Unaligned`] without issuing any I/O.
///
/// # Bounds
///
/// `lba >= block_count()` returns [`BlockError::OutOfRange`].
pub trait BlockDevice {
    /// Read one or more contiguous logical blocks starting at `lba`.
    ///
    /// `buf.len()` MUST be a non-zero multiple of
    /// [`block_size_bytes`](Self::block_size_bytes).
    fn read_block(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError>;

    /// Write one or more contiguous logical blocks starting at `lba`.
    ///
    /// `buf.len()` MUST be a non-zero multiple of
    /// [`block_size_bytes`](Self::block_size_bytes).
    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError>;

    /// Logical block size of this device in bytes.
    ///
    /// For most modern devices this is 4096. Devices that expose 512 B
    /// logical sectors (Advanced Format with 512e, or pre-AF legacy
    /// drives) report 512 here and rely on [`sector::SectorAdapter`]
    /// to translate FS-level 4 KiB I/O.
    fn block_size_bytes(&self) -> u32;

    /// Number of logical blocks on the device.
    fn block_count(&self) -> u64;

    /// Force any in-flight writes to non-volatile media.
    ///
    /// On virtio-blk this issues `VIRTIO_BLK_T_FLUSH`. On NVMe it
    /// issues a Flush command. On SDHCI/eMMC it issues
    /// `CMD_FLUSH_CACHE`.
    fn flush(&mut self) -> Result<(), BlockError>;

    /// Total logical capacity in bytes.
    ///
    /// Default implementation: `block_size_bytes * block_count`.
    fn capacity_bytes(&self) -> u64 {
        self.block_count()
            .saturating_mul(self.block_size_bytes() as u64)
    }
}

/// Typed block-I/O errors.
///
/// Mapped to POSIX errno at the syscall boundary by
/// [`errno::block_error_to_errno`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockError {
    /// Hardware reported a media-level read/write failure (CRC bit
    /// already accounted as `BadCrc`, this is the catch-all for ECC
    /// uncorrectable, sector relocation failures, etc.).
    MediaError,
    /// The device is not present (hot-removed, never enumerated, or
    /// transport went down).
    NotPresent,
    /// Per-op timeout fired and all retries are exhausted.
    Timeout,
    /// Read completed but the device-reported integrity check
    /// (CRC32C, T10-DIF, NVMe metadata, etc.) failed.
    BadCrc,
    /// `buf.len()` is not a multiple of `block_size_bytes()`, OR
    /// the LBA is misaligned for this transport.
    Unaligned,
    /// `lba >= block_count()` or the I/O extends past end-of-device.
    OutOfRange,
    /// The transport returned a "device busy" / queue-full code that
    /// is not transient enough to retry inside this op.
    DeviceBusy,
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BlockError::MediaError => "media error",
            BlockError::NotPresent => "device not present",
            BlockError::Timeout => "timeout after retries",
            BlockError::BadCrc => "bad CRC",
            BlockError::Unaligned => "unaligned buffer or LBA",
            BlockError::OutOfRange => "LBA out of range",
            BlockError::DeviceBusy => "device busy",
        };
        f.write_str(s)
    }
}

/// Validate that a buffer length is a non-zero multiple of the
/// device's logical block size.
///
/// Used by every `BlockDevice` implementation in this crate as the
/// first check in `read_block` / `write_block`. Returns
/// `Err(BlockError::Unaligned)` for `buf.len() == 0`,
/// `buf.len() % block_size_bytes != 0`.
#[inline]
pub fn check_buf_alignment(buf_len: usize, block_size_bytes: u32) -> Result<(), BlockError> {
    if buf_len == 0 {
        return Err(BlockError::Unaligned);
    }
    let bs = block_size_bytes as usize;
    if bs == 0 || !buf_len.is_multiple_of(bs) {
        return Err(BlockError::Unaligned);
    }
    Ok(())
}

/// Validate that an I/O of `(buf.len() / block_size_bytes)` blocks
/// starting at `lba` fits within the device.
#[inline]
pub fn check_lba_in_range(
    lba: u64,
    buf_len: usize,
    block_size_bytes: u32,
    block_count: u64,
) -> Result<(), BlockError> {
    if block_count == 0 {
        return Err(BlockError::OutOfRange);
    }
    if lba >= block_count {
        return Err(BlockError::OutOfRange);
    }
    let blocks = (buf_len / block_size_bytes as usize) as u64;
    let last = lba.checked_add(blocks).ok_or(BlockError::OutOfRange)?;
    if last > block_count {
        return Err(BlockError::OutOfRange);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buf_alignment_zero_rejected() {
        assert_eq!(check_buf_alignment(0, 4096), Err(BlockError::Unaligned));
    }

    #[test]
    fn buf_alignment_misaligned_rejected() {
        assert_eq!(check_buf_alignment(4097, 4096), Err(BlockError::Unaligned));
        assert_eq!(check_buf_alignment(512, 4096), Err(BlockError::Unaligned));
        assert_eq!(check_buf_alignment(8193, 4096), Err(BlockError::Unaligned));
    }

    #[test]
    fn buf_alignment_aligned_accepted() {
        assert!(check_buf_alignment(4096, 4096).is_ok());
        assert!(check_buf_alignment(8192, 4096).is_ok());
        assert!(check_buf_alignment(512, 512).is_ok());
        assert!(check_buf_alignment(4096, 512).is_ok());
    }

    #[test]
    fn lba_at_or_past_end_rejected() {
        assert_eq!(
            check_lba_in_range(100, 4096, 4096, 100),
            Err(BlockError::OutOfRange)
        );
        assert_eq!(
            check_lba_in_range(101, 4096, 4096, 100),
            Err(BlockError::OutOfRange)
        );
    }

    #[test]
    fn lba_in_range_ok() {
        assert!(check_lba_in_range(0, 4096, 4096, 100).is_ok());
        assert!(check_lba_in_range(99, 4096, 4096, 100).is_ok());
        // 8 KiB starting at lba 98 covers lba 98 and 99 — fits
        assert!(check_lba_in_range(98, 8192, 4096, 100).is_ok());
    }

    #[test]
    fn lba_extends_past_end_rejected() {
        // 8 KiB starting at lba 99 needs lba 99 and 100, only 99 exists
        assert_eq!(
            check_lba_in_range(99, 8192, 4096, 100),
            Err(BlockError::OutOfRange)
        );
    }

    #[test]
    fn empty_device_rejects_any_lba() {
        assert_eq!(
            check_lba_in_range(0, 4096, 4096, 0),
            Err(BlockError::OutOfRange)
        );
    }

    #[test]
    fn block_error_display() {
        use core::fmt::Write;
        let mut buf = alloc::string::String::new();
        write!(buf, "{}", BlockError::Timeout).unwrap();
        assert!(buf.contains("timeout"));
    }
}
