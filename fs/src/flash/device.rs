// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Raw-flash device abstraction (`FlashDevice` trait + `FlashError`).
//!
//! `FlashDevice` is the single abstraction layer between the
//! littlefs reader/writer (and any future raw-flash filesystem) and the
//! underlying transport — QSPI NOR, ONFI NAND, or the in-memory mock.
//!
//! Distinct from [`crate::block::BlockDevice`] because raw flash needs:
//!
//! * Explicit erase-block accounting — pages can only be programmed
//!   *after* the containing erase-block has been erased. A
//!   `BlockDevice` masks that detail.
//! * Per-block bad-block reporting — NAND parts ship from the factory
//!   with bad blocks marked, and additional blocks may go bad during
//!   operation.
//! * Asymmetric program/erase semantics — bits inside an erased page
//!   may flip 1→0 by `program`, but only `erase` can flip them back to
//!   1. The trait surface enforces that semantics.
//!
//! See `openspec/changes/embedded-flash-fs-v1/specs/fs-flash-device/spec.md`
//! for the authoritative contract.

use core::fmt;

/// Trait implemented by every raw-flash backend (QSPI NOR, ONFI NAND,
/// in-memory mock).
///
/// # Semantics
///
/// * `read(offset, buf)` — read `buf.len()` bytes starting at byte
///   `offset`. The flash medium is byte-addressable for reads. Returns
///   [`FlashError::OutOfRange`] if the region does not fit, and
///   [`FlashError::EccUncorrectable`] when ECC machinery detects a
///   bit error it cannot correct.
///
/// * `program(offset, buf)` — write `buf.len()` bytes starting at byte
///   `offset`. Bits may only flip 1→0; any attempt to flip a 0→1 (i.e.
///   the page is not in its erased state) returns
///   [`FlashError::ProgramOnDirty`] and the page MUST be unchanged.
///   `offset` and `buf.len()` SHOULD be multiples of
///   [`page_size_bytes`](Self::page_size_bytes); a non-aligned program
///   returns [`FlashError::Unaligned`].
///
/// * `erase(block)` — erase the entire erase-block whose index is
///   `block` (block 0 covers bytes `0 .. block_size_bytes()`). After a
///   successful erase, every byte in the block reads as `0xFF`. Partial
///   erases are not permitted. If the medium reports an erase-failure
///   for the block, returns [`FlashError::EraseFailure`]; callers
///   typically respond by marking the block bad.
///
/// * `is_bad(block)` / `mark_bad(block)` — bad-block table interface.
///   `is_bad` reads the runtime BBT; `mark_bad` adds a runtime entry.
///   Allocators MUST consult `is_bad` before any program / erase.
pub trait FlashDevice {
    /// Read `buf.len()` bytes starting at byte `offset` from the flash
    /// medium.
    ///
    /// `offset + buf.len()` must be ≤ `block_size_bytes() * block_count()`.
    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), FlashError>;

    /// Program `buf.len()` bytes starting at byte `offset`.
    ///
    /// Programs MUST hit a region that has been previously erased and
    /// not yet programmed; otherwise [`FlashError::ProgramOnDirty`] is
    /// returned. A program that crosses an erase-block boundary is
    /// allowed iff *both* erase-blocks are in their erased state.
    fn program(&mut self, offset: u64, buf: &[u8]) -> Result<(), FlashError>;

    /// Erase the entire erase-block at index `block`.
    ///
    /// On success, every byte in `block * block_size_bytes() ..
    /// (block + 1) * block_size_bytes()` reads as `0xFF`.
    fn erase(&mut self, block: u64) -> Result<(), FlashError>;

    /// Erase-block size in bytes (the smallest unit of `erase`).
    ///
    /// QSPI NOR defaults to 4096; ONFI NAND defaults to 131072.
    fn block_size_bytes(&self) -> u32;

    /// Page size in bytes (the smallest unit of `program`).
    ///
    /// QSPI NOR defaults to 256; ONFI NAND defaults to 4096.
    fn page_size_bytes(&self) -> u32;

    /// Total number of erase-blocks on the device.
    fn block_count(&self) -> u64;

    /// Whether `block` is currently marked bad (factory or runtime).
    ///
    /// Allocators MUST consult this before any program / erase.
    fn is_bad(&self, block: u64) -> bool;

    /// Mark `block` as bad (runtime).
    ///
    /// Persistent BBT updates (writing the redundant copies at start +
    /// end of flash) are the wear-leveling layer's responsibility; this
    /// trait just records the in-memory bit so subsequent allocations
    /// skip the block.
    fn mark_bad(&mut self, block: u64) -> Result<(), FlashError>;

    /// Total medium capacity in bytes.
    ///
    /// Default impl: `block_size_bytes() * block_count()`.
    fn capacity_bytes(&self) -> u64 {
        self.block_count()
            .saturating_mul(u64::from(self.block_size_bytes()))
    }
}

/// Typed flash-I/O errors.
///
/// Mapped to POSIX errno at the syscall boundary (see
/// [`flash_error_to_errno`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlashError {
    /// Catch-all media-level read/write failure (controller surfaced an
    /// unrecoverable medium error).
    MediaError,
    /// Read returned data, but the medium-level ECC machinery flagged
    /// an uncorrectable bit error.
    EccUncorrectable,
    /// The device is not present (transport down, never enumerated).
    NotPresent,
    /// `program` was attempted against a page that is not in its
    /// erased state (a 1→0 bit-flip would be required to reach the
    /// requested pattern).
    ProgramOnDirty,
    /// The medium reported an erase failure for the requested block.
    /// Wear-leveling layer typically responds by adding the block to
    /// the BBT.
    EraseFailure,
    /// Program failure for a non-erase reason (controller surfaced a
    /// program-failed condition for an erased page).
    ProgramFailed,
    /// The requested block is in the BBT.
    BadBlock,
    /// The transport returned a busy / queue-full condition.
    DeviceBusy,
    /// Operation timed out.
    Timeout,
    /// `offset + len` exceeds device capacity, or `block` ≥
    /// `block_count()`.
    OutOfRange,
    /// `offset` or `buf.len()` is not aligned to `page_size_bytes()`
    /// (for program) or otherwise violates the medium's alignment.
    Unaligned,
}

impl fmt::Display for FlashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FlashError::MediaError => "media error",
            FlashError::EccUncorrectable => "ECC uncorrectable",
            FlashError::NotPresent => "device not present",
            FlashError::ProgramOnDirty => "program on dirty page",
            FlashError::EraseFailure => "erase failure",
            FlashError::ProgramFailed => "program failure",
            FlashError::BadBlock => "block is bad",
            FlashError::DeviceBusy => "device busy",
            FlashError::Timeout => "timeout",
            FlashError::OutOfRange => "out of range",
            FlashError::Unaligned => "unaligned",
        };
        f.write_str(s)
    }
}

// ─── POSIX errno conversion ─────────────────────────────────────────────────

/// `EIO` — I/O error.
pub const EIO: i32 = 5;
/// `ENXIO` — no such device or address.
pub const ENXIO: i32 = 6;
/// `EBUSY` — device or resource busy.
pub const EBUSY: i32 = 16;
/// `EINVAL` — invalid argument.
pub const EINVAL: i32 = 22;
/// `EROFS` — read-only file system.
pub const EROFS: i32 = 30;
/// `ETIMEDOUT` — operation timed out.
pub const ETIMEDOUT: i32 = 110;
/// `ENOMEDIUM` — no medium found.
pub const ENOMEDIUM: i32 = 123;

/// Map a [`FlashError`] to a positive POSIX errno.
///
/// Per `fs-flash-device` spec:
///
/// | `FlashError`                                 | POSIX errno  |
/// |----------------------------------------------|--------------|
/// | `MediaError`, `EccUncorrectable`, `BadBlock` | `EIO`        |
/// | `ProgramFailed`                              | `EIO`        |
/// | `NotPresent`                                 | `ENOMEDIUM`  |
/// | `ProgramOnDirty`, `EraseFailure`             | `EROFS`      |
/// | `Timeout`                                    | `ETIMEDOUT`  |
/// | `Unaligned`, `OutOfRange`                    | `EINVAL`     |
/// | `DeviceBusy`                                 | `EBUSY`      |
#[inline]
pub const fn flash_error_to_errno(err: FlashError) -> i32 {
    match err {
        FlashError::MediaError
        | FlashError::EccUncorrectable
        | FlashError::BadBlock
        | FlashError::ProgramFailed => EIO,
        FlashError::NotPresent => ENOMEDIUM,
        FlashError::ProgramOnDirty | FlashError::EraseFailure => EROFS,
        FlashError::Timeout => ETIMEDOUT,
        FlashError::Unaligned | FlashError::OutOfRange => EINVAL,
        FlashError::DeviceBusy => EBUSY,
    }
}

/// Map a [`FlashError`] to a Linux-style negative errno.
#[inline]
pub const fn flash_error_to_negative_errno(err: FlashError) -> i32 {
    -flash_error_to_errno(err)
}

// ─── Range / alignment helpers ──────────────────────────────────────────────

/// Validate that a byte range `[offset, offset + len)` fits within the
/// flash medium.
#[inline]
pub fn check_range(
    offset: u64,
    len: u64,
    block_size_bytes: u32,
    block_count: u64,
) -> Result<(), FlashError> {
    let cap = block_count.saturating_mul(u64::from(block_size_bytes));
    let end = offset.checked_add(len).ok_or(FlashError::OutOfRange)?;
    if end > cap {
        return Err(FlashError::OutOfRange);
    }
    Ok(())
}

/// Validate that a program operation is page-aligned in both `offset`
/// and `buf.len()`.
#[inline]
pub fn check_program_alignment(
    offset: u64,
    len: u64,
    page_size_bytes: u32,
) -> Result<(), FlashError> {
    let p = u64::from(page_size_bytes);
    if p == 0 {
        return Err(FlashError::Unaligned);
    }
    if !offset.is_multiple_of(p) || !len.is_multiple_of(p) {
        return Err(FlashError::Unaligned);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errno_constants_match_linux() {
        assert_eq!(EIO, 5);
        assert_eq!(ENXIO, 6);
        assert_eq!(EBUSY, 16);
        assert_eq!(EINVAL, 22);
        assert_eq!(EROFS, 30);
        assert_eq!(ETIMEDOUT, 110);
        assert_eq!(ENOMEDIUM, 123);
    }

    #[test]
    fn errno_mapping_per_spec() {
        assert_eq!(flash_error_to_errno(FlashError::MediaError), EIO);
        assert_eq!(flash_error_to_errno(FlashError::EccUncorrectable), EIO);
        assert_eq!(flash_error_to_errno(FlashError::BadBlock), EIO);
        assert_eq!(flash_error_to_errno(FlashError::ProgramFailed), EIO);
        assert_eq!(flash_error_to_errno(FlashError::NotPresent), ENOMEDIUM);
        assert_eq!(flash_error_to_errno(FlashError::ProgramOnDirty), EROFS);
        assert_eq!(flash_error_to_errno(FlashError::EraseFailure), EROFS);
        assert_eq!(flash_error_to_errno(FlashError::Timeout), ETIMEDOUT);
        assert_eq!(flash_error_to_errno(FlashError::Unaligned), EINVAL);
        assert_eq!(flash_error_to_errno(FlashError::OutOfRange), EINVAL);
        assert_eq!(flash_error_to_errno(FlashError::DeviceBusy), EBUSY);
    }

    #[test]
    fn negative_errno_negates() {
        assert_eq!(flash_error_to_negative_errno(FlashError::MediaError), -EIO);
        assert_eq!(
            flash_error_to_negative_errno(FlashError::ProgramOnDirty),
            -EROFS
        );
    }

    #[test]
    fn display_strings_present() {
        use core::fmt::Write;
        for err in [
            FlashError::MediaError,
            FlashError::EccUncorrectable,
            FlashError::NotPresent,
            FlashError::ProgramOnDirty,
            FlashError::EraseFailure,
            FlashError::ProgramFailed,
            FlashError::BadBlock,
            FlashError::DeviceBusy,
            FlashError::Timeout,
            FlashError::OutOfRange,
            FlashError::Unaligned,
        ] {
            let mut buf = alloc::string::String::new();
            write!(buf, "{}", err).unwrap();
            assert!(!buf.is_empty(), "empty Display for {:?}", err);
        }
    }

    #[test]
    fn check_range_rejects_overflow() {
        // u64::MAX-overflowing add
        assert_eq!(
            check_range(u64::MAX - 5, 100, 4096, 16),
            Err(FlashError::OutOfRange)
        );
    }

    #[test]
    fn check_range_rejects_past_end() {
        // 16 blocks * 4096 = 65536. Asking for 100 bytes at 65500 fits;
        // 100 bytes at 65500 gives end = 65600 > 65536.
        assert_eq!(
            check_range(65500, 100, 4096, 16),
            Err(FlashError::OutOfRange)
        );
    }

    #[test]
    fn check_range_accepts_in_range() {
        assert!(check_range(0, 100, 4096, 16).is_ok());
        assert!(check_range(65535, 1, 4096, 16).is_ok());
        assert!(check_range(0, 65536, 4096, 16).is_ok());
    }

    #[test]
    fn program_alignment_checks() {
        assert!(check_program_alignment(0, 256, 256).is_ok());
        assert!(check_program_alignment(512, 768, 256).is_ok());
        assert_eq!(
            check_program_alignment(1, 256, 256),
            Err(FlashError::Unaligned)
        );
        assert_eq!(
            check_program_alignment(0, 255, 256),
            Err(FlashError::Unaligned)
        );
        assert_eq!(
            check_program_alignment(0, 256, 0),
            Err(FlashError::Unaligned)
        );
    }
}
