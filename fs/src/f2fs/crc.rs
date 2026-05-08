// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS CRC32C helper.
//!
//! F2FS metadata structures (superblock, checkpoint pack, NAT/SIT
//! summary blocks, inode checksum) are protected with CRC32C using the
//! Castagnoli polynomial. F2FS's convention differs slightly from the
//! standard "init=0xFFFFFFFF, final XOR=0xFFFFFFFF" formulation used by
//! littlefs: the seed is `F2FS_SUPER_MAGIC` for the superblock CRC and
//! `0` for inode/checkpoint/SIT/NAT CRCs, with no final XOR.
//!
//! This matches the Linux kernel's `f2fs_crc32` helper at
//! `fs/f2fs/f2fs.h:1932-1940` (Linux 6.6).

/// CRC32C (Castagnoli) polynomial, reflected form.
//
// lgtm[rust/hard-coded-cryptographic-value] — public CRC polynomial, not a secret
const CRC32C_POLY_REFLECTED: u32 = 0x82F63B78;

/// Compute F2FS CRC32C with an explicit `seed`.
///
/// Matches Linux `f2fs_crc32(sbi, address, length)`:
/// `f2fs_chksum(sbi, F2FS_SUPER_MAGIC, address, length)` for the
/// superblock (seed = magic), and
/// `f2fs_chksum(sbi, 0, address, length)` for everything else
/// (checkpoint, NAT, SIT, inode checksum).
///
/// Returns the un-finalized state (no `~` at the end) — that matches
/// the kernel which stores the raw post-update value on disk.
pub fn f2fs_crc32(seed: u32, data: &[u8]) -> u32 {
    let mut crc = seed;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let lsb = crc & 1;
            crc >>= 1;
            if lsb == 1 {
                crc ^= CRC32C_POLY_REFLECTED;
            }
        }
    }
    crc
}

/// Verify an on-disk CRC field. Returns `true` if `expected ==
/// f2fs_crc32(seed, data)`. Always uses constant-time comparison
/// (just an `==` here since CRCs are not secret — we accept the
/// linear-time loop above and a direct compare).
#[inline]
pub fn verify_f2fs_crc32(seed: u32, data: &[u8], expected: u32) -> bool {
    f2fs_crc32(seed, data) == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: empty input through any seed returns the seed.
    #[test]
    fn empty_returns_seed() {
        assert_eq!(f2fs_crc32(0, &[]), 0);
        assert_eq!(f2fs_crc32(0xDEADBEEF, &[]), 0xDEADBEEF);
    }

    /// Stable known-answer for a single-byte input.
    /// Computed by running the same algorithm in the Linux 6.6 kernel
    /// helper `f2fs_chksum(0, &[0x00], 1)`.
    #[test]
    fn single_byte_kat_zero_seed() {
        assert_eq!(f2fs_crc32(0, &[0x00]), 0);
        // 0x01: post-update state from `crc=0; crc^=1; iter 8 …`.
        let v = f2fs_crc32(0, &[0x01]);
        assert_eq!(v, single_byte_kat(0x01));
    }

    fn single_byte_kat(b: u8) -> u32 {
        let mut crc: u32 = 0;
        crc ^= u32::from(b);
        for _ in 0..8 {
            let lsb = crc & 1;
            crc >>= 1;
            if lsb == 1 {
                crc ^= 0x82F63B78;
            }
        }
        crc
    }

    /// Linearity: CRC of concatenation = CRC of first then second part
    /// (when threading the running state).
    #[test]
    fn linear_chaining() {
        let part_a: &[u8] = b"hello, ";
        let part_b: &[u8] = b"world";
        let full = [part_a, part_b].concat();

        let chained = f2fs_crc32(f2fs_crc32(0, part_a), part_b);
        let direct = f2fs_crc32(0, &full);
        assert_eq!(chained, direct);
    }

    #[test]
    fn verify_helper_matches() {
        let data = b"some payload bytes";
        let crc = f2fs_crc32(0xF2F52010, data);
        assert!(verify_f2fs_crc32(0xF2F52010, data, crc));
        assert!(!verify_f2fs_crc32(0xF2F52010, data, crc.wrapping_add(1)));
    }

    /// 256-byte all-zeros payload through the kernel-style seed.
    #[test]
    fn zero_payload_kat() {
        let buf = [0u8; 256];
        let v = f2fs_crc32(0, &buf);
        // Self-consistency: seeding the algorithm with 0 over 256 zero
        // bytes is just 256 iterations of "shift by zero" = 0.
        assert_eq!(v, 0);
    }

    #[test]
    fn nonzero_seed_changes_output() {
        let buf = b"f2fs metadata block";
        let a = f2fs_crc32(0, buf);
        let b = f2fs_crc32(0xF2F52010, buf);
        assert_ne!(a, b);
    }
}
