// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! POSIX errno conversion for [`BlockError`](super::BlockError).
//!
//! Per `embedded-filesystem-v1` spec `fs-block-device`:
//!
//! | `BlockError`            | POSIX errno |
//! |-------------------------|-------------|
//! | `MediaError`, `BadCrc`  | `-EIO`      |
//! | `NotPresent`            | `-ENXIO`    |
//! | `Timeout`               | `-ETIMEDOUT`|
//! | `Unaligned`, `OutOfRange` | `-EINVAL` |
//! | `DeviceBusy`            | `-EBUSY`    |
//!
//! These errno values are surfaced to userspace by `posix-vfs` at the
//! syscall boundary in Phase 6. They are negated (Linux-style negative
//! return) by syscall dispatchers; this module returns the *positive*
//! integer for callers that want a raw errno.

use super::BlockError;

/// `EIO` — I/O error.
pub const EIO: i32 = 5;
/// `ENXIO` — no such device or address.
pub const ENXIO: i32 = 6;
/// `EBUSY` — device or resource busy.
pub const EBUSY: i32 = 16;
/// `EINVAL` — invalid argument.
pub const EINVAL: i32 = 22;
/// `ENOSPC` — no space left on device.
pub const ENOSPC: i32 = 28;
/// `ETIMEDOUT` — connection / operation timed out.
pub const ETIMEDOUT: i32 = 110;

/// Map a [`BlockError`] to a positive POSIX errno.
///
/// Callers that want the Linux negative-errno convention should
/// negate the result.
#[inline]
pub const fn block_error_to_errno(err: BlockError) -> i32 {
    match err {
        BlockError::MediaError | BlockError::BadCrc => EIO,
        BlockError::NotPresent => ENXIO,
        BlockError::Timeout => ETIMEDOUT,
        BlockError::Unaligned | BlockError::OutOfRange => EINVAL,
        BlockError::DeviceBusy => EBUSY,
    }
}

/// Map a [`BlockError`] to a Linux-style negative errno
/// (`-EIO`, `-ETIMEDOUT`, ...).
///
/// This is the form `posix-vfs` returns from syscalls.
#[inline]
pub const fn block_error_to_negative_errno(err: BlockError) -> i32 {
    -block_error_to_errno(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_error_maps_to_eio() {
        assert_eq!(block_error_to_errno(BlockError::MediaError), EIO);
        assert_eq!(block_error_to_negative_errno(BlockError::MediaError), -EIO);
    }

    #[test]
    fn bad_crc_maps_to_eio() {
        assert_eq!(block_error_to_errno(BlockError::BadCrc), EIO);
    }

    #[test]
    fn not_present_maps_to_enxio() {
        assert_eq!(block_error_to_errno(BlockError::NotPresent), ENXIO);
    }

    #[test]
    fn timeout_maps_to_etimedout() {
        assert_eq!(block_error_to_errno(BlockError::Timeout), ETIMEDOUT);
        assert_eq!(
            block_error_to_negative_errno(BlockError::Timeout),
            -ETIMEDOUT
        );
    }

    #[test]
    fn unaligned_maps_to_einval() {
        assert_eq!(block_error_to_errno(BlockError::Unaligned), EINVAL);
    }

    #[test]
    fn out_of_range_maps_to_einval() {
        assert_eq!(block_error_to_errno(BlockError::OutOfRange), EINVAL);
    }

    #[test]
    fn device_busy_maps_to_ebusy() {
        assert_eq!(block_error_to_errno(BlockError::DeviceBusy), EBUSY);
    }

    #[test]
    fn errno_constants_match_linux() {
        // Cross-check against the canonical Linux x86-64 / aarch64
        // errno table. These constants must not drift.
        assert_eq!(EIO, 5);
        assert_eq!(ENXIO, 6);
        assert_eq!(EBUSY, 16);
        assert_eq!(EINVAL, 22);
        assert_eq!(ENOSPC, 28);
        assert_eq!(ETIMEDOUT, 110);
    }
}
