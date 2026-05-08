// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS superblock parser (Linux 6.6 LTS feature set).
//!
//! The F2FS partition keeps two copies of the superblock at fixed byte
//! offsets within the partition's first segment:
//!
//! - **Primary**: byte offset `1024` (sector 2 if the device exposes
//!   512 B sectors).
//! - **Secondary**: byte offset `9216` (= 1024 + 8192).
//!
//! Each copy is a 4 KiB block whose first 1024 bytes are the
//! superblock fields proper (a `struct f2fs_super_block` in the Linux
//! kernel). The trailing 3 KiB of the block is reserved/zero.
//!
//! ## Layout (offsets within the 4 KiB block, little-endian)
//!
//! Pinned to `fs/f2fs/f2fs.h:struct f2fs_super_block` in Linux 6.6.
//! Field offsets that are load-bearing for the reader are tabulated;
//! the rest is parsed defensively but ignored.
//!
//! | Off  | Size | Field                               |
//! |-----:|-----:|-------------------------------------|
//! |  `0` |   4  | `magic` (= `F2FS_SUPER_MAGIC`)      |
//! |  `4` |   2  | `major_ver`                         |
//! |  `6` |   2  | `minor_ver`                         |
//! |  `8` |   4  | `log_sectorsize` (log2 sector size) |
//! | `12` |   4  | `log_sectors_per_block`             |
//! | `16` |   4  | `log_blocksize` (log2 → 12 = 4 KiB) |
//! | `20` |   4  | `log_blocks_per_seg` (= 9 → 512)    |
//! | `24` |   4  | `segs_per_sec`                      |
//! | `28` |   4  | `secs_per_zone`                     |
//! | `32` |   4  | `checksum_offset` (CRC field offset)|
//! | `36` |   8  | `block_count`                       |
//! | `44` |   4  | `section_count`                     |
//! | `48` |   4  | `segment_count`                     |
//! | `52` |   4  | `segment_count_ckpt`                |
//! | `56` |   4  | `segment_count_sit`                 |
//! | `60` |   4  | `segment_count_nat`                 |
//! | `64` |   4  | `segment_count_ssa`                 |
//! | `68` |   4  | `segment_count_main`                |
//! | `72` |   4  | `segment0_blkaddr`                  |
//! | `76` |   4  | `cp_blkaddr`                        |
//! | `80` |   4  | `sit_blkaddr`                       |
//! | `84` |   4  | `nat_blkaddr`                       |
//! | `88` |   4  | `ssa_blkaddr`                       |
//! | `92` |   4  | `main_blkaddr`                      |
//! | `96` |   4  | `root_ino`                          |
//! |`100` |   4  | `node_ino`                          |
//! |`104` |   4  | `meta_ino`                          |
//! |`108` |  16  | `uuid[16]`                          |
//! |`124` | 1024 | `volume_name[512]` (UTF-16LE)       |
//! |`...` |   …  | ... (inheritance, devices, etc.)    |
//! |`...` |   4  | `feature` (u32 mandatory + compat)  |
//!
//! For the v1 reader we treat `feature` as a single `u64` (low 32 bits
//! from the `feature` field at offset 136 in Linux 6.6) so the
//! mandatory-bit set covers everything F2FS-tools emits.

extern crate alloc;

use alloc::string::String;

use super::crc::verify_f2fs_crc32;
use super::error::F2fsError;
use super::{u32_le, u64_le, FeatureBit, F2FS_BLOCK_SIZE};

/// F2FS superblock magic number (`0xF2F52010`).
//
// lgtm[rust/hard-coded-cryptographic-value] — public on-disk format magic constant
pub const F2FS_SUPER_MAGIC: u32 = 0xF2F5_2010;

/// Byte offset of the primary superblock from the partition start.
pub const PRIMARY_SUPERBLOCK_OFFSET: u64 = 1024;

/// Byte offset of the secondary superblock from the partition start.
pub const SECONDARY_SUPERBLOCK_OFFSET: u64 = 9216;

/// Length of the on-disk `struct f2fs_super_block` (1 KiB).
///
/// The Linux 6.6 layout is exactly 1024 bytes; the parent 4 KiB block
/// just zero-pads the rest.
pub const F2FS_SUPER_BLOCK_BYTES: usize = 1024;

/// CRC seed for the superblock (the seed the kernel passes is the
/// magic itself; see `f2fs_crc32`).
const SUPERBLOCK_CRC_SEED: u32 = F2FS_SUPER_MAGIC;

/// Major version this reader supports (Linux 6.6 mainline F2FS).
pub const SUPPORTED_MAJOR_VERSION: u16 = 1;

// ─── Mandatory feature bit set (Linux 6.6) ───────────────────────────────────
//
// The `feature` field is a 32-bit bitfield in the on-disk superblock.
// Each bit represents an F2FS feature; mandatory bits MUST be
// understood by the reader or mount fails.
//
// References: `fs/f2fs/f2fs.h:F2FS_FEATURE_*`.

/// `F2FS_FEATURE_ENCRYPT` — per-file encryption.
pub const FEATURE_ENCRYPT: u64 = 0x0001;
/// `F2FS_FEATURE_BLKZONED` — host-managed zoned block devices.
pub const FEATURE_BLKZONED: u64 = 0x0002;
/// `F2FS_FEATURE_ATOMIC_WRITE` — atomic-write support.
pub const FEATURE_ATOMIC_WRITE: u64 = 0x0004;
/// `F2FS_FEATURE_EXTRA_ATTR` — extended inode attributes.
pub const FEATURE_EXTRA_ATTR: u64 = 0x0008;
/// `F2FS_FEATURE_PRJQUOTA` — project quota.
pub const FEATURE_PRJQUOTA: u64 = 0x0010;
/// `F2FS_FEATURE_INODE_CHKSUM` — per-inode checksum field.
pub const FEATURE_INODE_CHKSUM: u64 = 0x0020;
/// `F2FS_FEATURE_FLEXIBLE_INLINE_XATTR`.
pub const FEATURE_FLEX_INLINE_XATTR: u64 = 0x0040;
/// `F2FS_FEATURE_QUOTA_INO` — quota inodes.
pub const FEATURE_QUOTA_INO: u64 = 0x0080;
/// `F2FS_FEATURE_INODE_CRTIME` — birth time field.
pub const FEATURE_INODE_CRTIME: u64 = 0x0100;
/// `F2FS_FEATURE_LOST_FOUND` — `lost+found` reservation.
pub const FEATURE_LOST_FOUND: u64 = 0x0200;
/// `F2FS_FEATURE_VERITY` — fs-verity.
pub const FEATURE_VERITY: u64 = 0x0400;
/// `F2FS_FEATURE_SB_CHKSUM` — superblock checksum field.
pub const FEATURE_SB_CHKSUM: u64 = 0x0800;
/// `F2FS_FEATURE_CASEFOLD` — directory casefold.
pub const FEATURE_CASEFOLD: u64 = 0x1000;
/// `F2FS_FEATURE_COMPRESSION` — file compression.
pub const FEATURE_COMPRESSION: u64 = 0x2000;
/// `F2FS_FEATURE_RO` — partition is read-only.
pub const FEATURE_RO: u64 = 0x4000;

/// Mask of feature bits this reader understands. Any bit outside this
/// mask in the on-disk `feature` field causes mount to fail with
/// [`F2fsError::UnsupportedFeature`].
///
/// **Design choice (v1 read path):** we accept all the bits that
/// `mkfs.f2fs` emits by default for Linux 6.6 — `EXTRA_ATTR`,
/// `INODE_CHKSUM`, `FLEX_INLINE_XATTR`, `QUOTA_INO`, `INODE_CRTIME`,
/// `SB_CHKSUM` — because they affect parsing but not data semantics.
/// Bits like `ENCRYPT`, `COMPRESSION`, `VERITY`, `CASEFOLD`,
/// `BLKZONED`, `ATOMIC_WRITE` change the on-disk layout in ways the
/// v1 reader cannot interpret correctly, so they are rejected even
/// though they're "compatible" in Linux's sense — better to fail
/// closed than serve wrong bytes.
pub const SUPPORTED_FEATURES: u64 = FEATURE_EXTRA_ATTR
    | FEATURE_INODE_CHKSUM
    | FEATURE_FLEX_INLINE_XATTR
    | FEATURE_QUOTA_INO
    | FEATURE_INODE_CRTIME
    | FEATURE_SB_CHKSUM
    | FEATURE_LOST_FOUND
    | FEATURE_PRJQUOTA
    | FEATURE_RO;

/// Parsed F2FS superblock (Linux 6.6 layout, little-endian fields).
#[derive(Debug, Clone)]
pub struct Superblock {
    pub magic: u32,
    pub major_ver: u16,
    pub minor_ver: u16,
    pub log_sectorsize: u32,
    pub log_sectors_per_block: u32,
    pub log_blocksize: u32,
    pub log_blocks_per_seg: u32,
    pub segs_per_sec: u32,
    pub secs_per_zone: u32,
    pub checksum_offset: u32,
    pub block_count: u64,
    pub section_count: u32,
    pub segment_count: u32,
    pub segment_count_ckpt: u32,
    pub segment_count_sit: u32,
    pub segment_count_nat: u32,
    pub segment_count_ssa: u32,
    pub segment_count_main: u32,
    pub segment0_blkaddr: u32,
    pub cp_blkaddr: u32,
    pub sit_blkaddr: u32,
    pub nat_blkaddr: u32,
    pub ssa_blkaddr: u32,
    pub main_blkaddr: u32,
    pub root_ino: u32,
    pub node_ino: u32,
    pub meta_ino: u32,
    pub uuid: [u8; 16],
    pub volume_name: String,
    pub feature: u64,
    /// On-disk CRC32C value (offset = `checksum_offset` if non-zero,
    /// else not present). Stored here for diagnostics.
    pub stored_crc: u32,
}

impl Superblock {
    /// Parse a 4 KiB block containing an F2FS superblock at the
    /// partition-relative offset `block_offset` (1024 or 9216).
    ///
    /// `block_offset` is the byte offset of the superblock within
    /// `block` itself, not within the partition. For both primary and
    /// secondary copies the block is the 4 KiB the surrounding helper
    /// just read; the on-disk fields begin at offset 0 within that
    /// 1 KiB-prefix slice.
    pub fn parse(buf: &[u8]) -> Result<Self, F2fsError> {
        if buf.len() < F2FS_SUPER_BLOCK_BYTES {
            return Err(F2fsError::InodeCorrupt);
        }
        let magic = u32_le(&buf[0..4]);
        if magic != F2FS_SUPER_MAGIC {
            return Err(F2fsError::BadMagic);
        }
        let major_ver = u16::from_le_bytes([buf[4], buf[5]]);
        let minor_ver = u16::from_le_bytes([buf[6], buf[7]]);
        if major_ver != SUPPORTED_MAJOR_VERSION {
            return Err(F2fsError::UnsupportedVersion(major_ver, minor_ver));
        }

        let log_sectorsize = u32_le(&buf[8..12]);
        let log_sectors_per_block = u32_le(&buf[12..16]);
        let log_blocksize = u32_le(&buf[16..20]);
        let log_blocks_per_seg = u32_le(&buf[20..24]);

        // Sanity-check the geometry: F2FS hard-codes 4 KiB blocks.
        if log_blocksize != 12 {
            return Err(F2fsError::UnsupportedVersion(major_ver, minor_ver));
        }

        let segs_per_sec = u32_le(&buf[24..28]);
        let secs_per_zone = u32_le(&buf[28..32]);
        let checksum_offset = u32_le(&buf[32..36]);

        let block_count = u64_le(&buf[36..44]);
        let section_count = u32_le(&buf[44..48]);
        let segment_count = u32_le(&buf[48..52]);
        let segment_count_ckpt = u32_le(&buf[52..56]);
        let segment_count_sit = u32_le(&buf[56..60]);
        let segment_count_nat = u32_le(&buf[60..64]);
        let segment_count_ssa = u32_le(&buf[64..68]);
        let segment_count_main = u32_le(&buf[68..72]);

        let segment0_blkaddr = u32_le(&buf[72..76]);
        let cp_blkaddr = u32_le(&buf[76..80]);
        let sit_blkaddr = u32_le(&buf[80..84]);
        let nat_blkaddr = u32_le(&buf[84..88]);
        let ssa_blkaddr = u32_le(&buf[88..92]);
        let main_blkaddr = u32_le(&buf[92..96]);

        let root_ino = u32_le(&buf[96..100]);
        let node_ino = u32_le(&buf[100..104]);
        let meta_ino = u32_le(&buf[104..108]);

        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&buf[108..124]);

        // `volume_name[512]` is UTF-16LE, NUL-padded. Decode lossily.
        let vol_bytes = &buf[124..124 + 512];
        let volume_name = decode_utf16le_lossy(vol_bytes);

        // `feature` field is at offset 124 + 512 = 636 in 6.6.
        let feature = u32_le(&buf[636..640]) as u64;

        // Stored CRC at the offset declared by `checksum_offset`. If
        // zero, the superblock has no CRC and we accept it (older
        // mkfs.f2fs builds did not write one).
        let stored_crc =
            if checksum_offset != 0 && (checksum_offset as usize) + 4 <= F2FS_SUPER_BLOCK_BYTES {
                u32_le(&buf[checksum_offset as usize..checksum_offset as usize + 4])
            } else {
                0
            };

        Ok(Self {
            magic,
            major_ver,
            minor_ver,
            log_sectorsize,
            log_sectors_per_block,
            log_blocksize,
            log_blocks_per_seg,
            segs_per_sec,
            secs_per_zone,
            checksum_offset,
            block_count,
            section_count,
            segment_count,
            segment_count_ckpt,
            segment_count_sit,
            segment_count_nat,
            segment_count_ssa,
            segment_count_main,
            segment0_blkaddr,
            cp_blkaddr,
            sit_blkaddr,
            nat_blkaddr,
            ssa_blkaddr,
            main_blkaddr,
            root_ino,
            node_ino,
            meta_ino,
            uuid,
            volume_name,
            feature,
            stored_crc,
        })
    }

    /// Verify the superblock CRC. If the superblock declared no CRC
    /// (the legacy case where `checksum_offset == 0`) this returns
    /// `Ok(())` unconditionally.
    pub fn verify_crc(&self, raw_block: &[u8]) -> Result<(), F2fsError> {
        if self.checksum_offset == 0 {
            return Ok(());
        }
        let off = self.checksum_offset as usize;
        if off + 4 > F2FS_SUPER_BLOCK_BYTES || raw_block.len() < F2FS_SUPER_BLOCK_BYTES {
            return Err(F2fsError::SuperblockCrcMismatch);
        }
        // Kernel computes the CRC over the bytes [0..checksum_offset].
        let chk = &raw_block[..off];
        if !verify_f2fs_crc32(SUPERBLOCK_CRC_SEED, chk, self.stored_crc) {
            return Err(F2fsError::SuperblockCrcMismatch);
        }
        Ok(())
    }

    /// Reject any mandatory feature bits this reader does not handle.
    pub fn check_features(&self) -> Result<(), F2fsError> {
        let unsupported = self.feature & !SUPPORTED_FEATURES;
        if unsupported != 0 {
            // Surface only the lowest set bit so the error message is
            // unambiguous.
            let lowest = unsupported & unsupported.wrapping_neg();
            return Err(F2fsError::UnsupportedFeature(lowest));
        }
        Ok(())
    }

    /// True if the named feature bit is set.
    pub fn has_feature(&self, bit: u64) -> bool {
        (self.feature & bit) != 0
    }

    /// Total partition size in bytes implied by the superblock.
    pub fn capacity_bytes(&self) -> u64 {
        self.block_count.saturating_mul(F2FS_BLOCK_SIZE as u64)
    }

    /// Format the feature-bit mask for log/audit use.
    pub fn feature_summary(&self) -> alloc::vec::Vec<FeatureBit> {
        let mut out = alloc::vec::Vec::new();
        let mut bits = self.feature;
        while bits != 0 {
            let lowest = bits & bits.wrapping_neg();
            out.push(FeatureBit(lowest));
            bits &= bits - 1;
        }
        out
    }
}

/// Decode a UTF-16LE NUL-terminated buffer to a String, lossily.
fn decode_utf16le_lossy(bytes: &[u8]) -> String {
    let mut units: alloc::vec::Vec<u16> = alloc::vec::Vec::new();
    let mut i = 0;
    while i + 1 < bytes.len() {
        let u = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        if u == 0 {
            break;
        }
        units.push(u);
        i += 2;
    }
    String::from_utf16_lossy(&units)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f2fs::crc::f2fs_crc32;

    fn make_minimal_sb() -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec![0u8; F2FS_SUPER_BLOCK_BYTES];
        buf[0..4].copy_from_slice(&F2FS_SUPER_MAGIC.to_le_bytes());
        buf[4..6].copy_from_slice(&1u16.to_le_bytes());
        buf[6..8].copy_from_slice(&0u16.to_le_bytes());
        buf[8..12].copy_from_slice(&9u32.to_le_bytes()); // 512-byte sector
        buf[12..16].copy_from_slice(&3u32.to_le_bytes()); // 8 sectors/blk
        buf[16..20].copy_from_slice(&12u32.to_le_bytes()); // 4 KiB blocks
        buf[20..24].copy_from_slice(&9u32.to_le_bytes()); // 512 blk/seg
        buf[24..28].copy_from_slice(&1u32.to_le_bytes());
        buf[28..32].copy_from_slice(&1u32.to_le_bytes());
        buf[32..36].copy_from_slice(&0u32.to_le_bytes()); // no CRC
        buf[36..44].copy_from_slice(&16384u64.to_le_bytes());
        buf[44..48].copy_from_slice(&32u32.to_le_bytes());
        buf[48..52].copy_from_slice(&32u32.to_le_bytes());
        buf[52..56].copy_from_slice(&1u32.to_le_bytes()); // ckpt
        buf[56..60].copy_from_slice(&1u32.to_le_bytes()); // sit
        buf[60..64].copy_from_slice(&1u32.to_le_bytes()); // nat
        buf[64..68].copy_from_slice(&1u32.to_le_bytes()); // ssa
        buf[68..72].copy_from_slice(&28u32.to_le_bytes()); // main
        buf[72..76].copy_from_slice(&512u32.to_le_bytes()); // segment0_blkaddr
        buf[76..80].copy_from_slice(&512u32.to_le_bytes());
        buf[80..84].copy_from_slice(&1024u32.to_le_bytes());
        buf[84..88].copy_from_slice(&1536u32.to_le_bytes());
        buf[88..92].copy_from_slice(&2048u32.to_le_bytes());
        buf[92..96].copy_from_slice(&2560u32.to_le_bytes());
        buf[96..100].copy_from_slice(&3u32.to_le_bytes());
        buf[100..104].copy_from_slice(&1u32.to_le_bytes());
        buf[104..108].copy_from_slice(&2u32.to_le_bytes());
        buf
    }

    #[test]
    fn parse_minimal_round_trip() {
        let buf = make_minimal_sb();
        let sb = Superblock::parse(&buf).unwrap();
        assert_eq!(sb.magic, F2FS_SUPER_MAGIC);
        assert_eq!(sb.major_ver, 1);
        assert_eq!(sb.log_blocksize, 12);
        assert_eq!(sb.log_blocks_per_seg, 9);
        assert_eq!(sb.block_count, 16384);
        assert_eq!(sb.root_ino, 3);
        assert_eq!(sb.feature, 0);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = make_minimal_sb();
        buf[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(Superblock::parse(&buf), Err(F2fsError::BadMagic)));
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut buf = make_minimal_sb();
        buf[4..6].copy_from_slice(&2u16.to_le_bytes());
        let err = Superblock::parse(&buf).unwrap_err();
        match err {
            F2fsError::UnsupportedVersion(2, _) => {}
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn truncated_buffer_rejected() {
        let buf = alloc::vec![0u8; 32];
        assert!(matches!(
            Superblock::parse(&buf),
            Err(F2fsError::InodeCorrupt)
        ));
    }

    #[test]
    fn unknown_mandatory_feature_rejected() {
        let mut buf = make_minimal_sb();
        // Set ENCRYPT bit (0x0001), which v1 read path does not handle.
        buf[636..640].copy_from_slice(&(FEATURE_ENCRYPT as u32).to_le_bytes());
        let sb = Superblock::parse(&buf).unwrap();
        assert!(matches!(
            sb.check_features(),
            Err(F2fsError::UnsupportedFeature(FEATURE_ENCRYPT))
        ));
    }

    #[test]
    fn supported_feature_set_ok() {
        let mut buf = make_minimal_sb();
        let supported = FEATURE_EXTRA_ATTR | FEATURE_INODE_CHKSUM | FEATURE_QUOTA_INO;
        buf[636..640].copy_from_slice(&(supported as u32).to_le_bytes());
        let sb = Superblock::parse(&buf).unwrap();
        assert!(sb.check_features().is_ok());
        assert!(sb.has_feature(FEATURE_INODE_CHKSUM));
    }

    #[test]
    fn crc_zero_skips_check() {
        let buf = make_minimal_sb();
        let sb = Superblock::parse(&buf).unwrap();
        assert!(sb.verify_crc(&buf).is_ok());
    }

    #[test]
    fn crc_set_round_trip() {
        let mut buf = make_minimal_sb();
        // Place the CRC at offset 1020 (last 4 bytes).
        let crc_off = 1020u32;
        buf[32..36].copy_from_slice(&crc_off.to_le_bytes());
        // Compute CRC over [0..crc_off] and write at crc_off.
        let crc = f2fs_crc32(F2FS_SUPER_MAGIC, &buf[..crc_off as usize]);
        buf[crc_off as usize..crc_off as usize + 4].copy_from_slice(&crc.to_le_bytes());
        let sb = Superblock::parse(&buf).unwrap();
        assert_eq!(sb.checksum_offset, crc_off);
        assert_eq!(sb.stored_crc, crc);
        assert_eq!(sb.verify_crc(&buf), Ok(()));
    }

    #[test]
    fn crc_mismatch_detected() {
        let mut buf = make_minimal_sb();
        let crc_off = 1020u32;
        buf[32..36].copy_from_slice(&crc_off.to_le_bytes());
        let crc = f2fs_crc32(F2FS_SUPER_MAGIC, &buf[..crc_off as usize]);
        buf[crc_off as usize..crc_off as usize + 4]
            .copy_from_slice(&crc.wrapping_add(1).to_le_bytes());
        let sb = Superblock::parse(&buf).unwrap();
        assert!(matches!(
            sb.verify_crc(&buf),
            Err(F2fsError::SuperblockCrcMismatch)
        ));
    }

    #[test]
    fn capacity_bytes_overflow_safe() {
        let mut buf = make_minimal_sb();
        buf[36..44].copy_from_slice(&u64::MAX.to_le_bytes());
        let sb = Superblock::parse(&buf).unwrap();
        assert_eq!(sb.capacity_bytes(), u64::MAX);
    }

    #[test]
    fn volume_name_lossy_decode() {
        let mut buf = make_minimal_sb();
        // Encode "Data" as UTF-16LE.
        let utf16: alloc::vec::Vec<u8> = "Data"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        buf[124..124 + utf16.len()].copy_from_slice(&utf16);
        let sb = Superblock::parse(&buf).unwrap();
        assert_eq!(sb.volume_name, "Data");
    }

    #[test]
    fn feature_summary_lists_set_bits() {
        let mut buf = make_minimal_sb();
        let supported = FEATURE_EXTRA_ATTR | FEATURE_INODE_CHKSUM;
        buf[636..640].copy_from_slice(&(supported as u32).to_le_bytes());
        let sb = Superblock::parse(&buf).unwrap();
        let summary = sb.feature_summary();
        assert_eq!(summary.len(), 2);
    }
}
