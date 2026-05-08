// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Squashfs 4.0 superblock parser.
//!
//! The squashfs 4.0 superblock is exactly 96 bytes, little-endian, at
//! offset `0` of the partition. Layout (from `mksquashfs(1)` and the
//! Linux 6.6 kernel `fs/squashfs/squashfs_fs.h`):
//!
//! | Offset | Size | Name                  |
//! |-------:|-----:|-----------------------|
//! |    `0` | `4`  | `magic`               |
//! |    `4` | `4`  | `inode_count`         |
//! |    `8` | `4`  | `mod_time`            |
//! |   `12` | `4`  | `block_size`          |
//! |   `16` | `4`  | `fragment_count`      |
//! |   `20` | `2`  | `compression_id`      |
//! |   `22` | `2`  | `block_log`           |
//! |   `24` | `2`  | `flags`               |
//! |   `26` | `2`  | `id_count`            |
//! |   `28` | `2`  | `version_major`       |
//! |   `30` | `2`  | `version_minor`       |
//! |   `32` | `8`  | `root_inode_ref`      |
//! |   `40` | `8`  | `bytes_used`          |
//! |   `48` | `8`  | `id_table_start`      |
//! |   `56` | `8`  | `xattr_table_start`   |
//! |   `64` | `8`  | `inode_table_start`   |
//! |   `72` | `8`  | `directory_table_start` |
//! |   `80` | `8`  | `fragment_table_start`  |
//! |   `88` | `8`  | `lookup_table_start`    |
//!
//! `block_log` is `log2(block_size)` and is checked for consistency with
//! `block_size`. Valid block sizes are 4 KiB, 8 KiB, 16 KiB, 32 KiB,
//! 64 KiB, 128 KiB, 256 KiB, 512 KiB, and 1 MiB.

use super::{SquashfsError, SQUASHFS_MAGIC, SQUASHFS_MAJOR_VERSION};

/// Squashfs 4.0 superblock fixed length on disk.
pub const SUPERBLOCK_SIZE: usize = 96;

/// Squashfs `flags` bitfield: inodes are stored uncompressed.
pub const FLAG_NOI: u16 = 0x0001;
/// Squashfs `flags`: data blocks stored uncompressed.
pub const FLAG_NOD: u16 = 0x0002;
/// Squashfs `flags`: fragments stored uncompressed.
pub const FLAG_NOF: u16 = 0x0008;
/// Squashfs `flags`: image has no fragments.
pub const FLAG_NO_FRAGMENTS: u16 = 0x0010;
/// Squashfs `flags`: always-fragment mode.
pub const FLAG_ALWAYS_FRAGMENTS: u16 = 0x0020;
/// Squashfs `flags`: duplicate detection.
pub const FLAG_DUPLICATES: u16 = 0x0040;
/// Squashfs `flags`: exportable through NFS.
pub const FLAG_EXPORTABLE: u16 = 0x0080;
/// Squashfs `flags`: extended-attribute table is uncompressed.
pub const FLAG_NOX: u16 = 0x0100;
/// Squashfs `flags`: no xattr table at all.
pub const FLAG_NO_XATTR: u16 = 0x0200;
/// Squashfs `flags`: compressor options block follows the superblock.
pub const FLAG_COMP_OPT: u16 = 0x0400;
/// Squashfs `flags`: ID table uncompressed.
pub const FLAG_NOID: u16 = 0x0800;

/// Compression algorithm IDs from the squashfs 4.0 superblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Compression {
    /// `1` — gzip (DEFLATE / RFC 1951).
    Gzip = 1,
    /// `4` — xz (LZMA2 inside xz container).
    Xz = 4,
    /// `5` — lz4 (block format).
    Lz4 = 5,
    /// `6` — zstd (RFC 8478).
    Zstd = 6,
}

impl Compression {
    /// Map a raw `compression_id` field to the supported set.
    ///
    /// IDs `2` (lzma) and `3` (lzo) are explicitly rejected — neither
    /// is a SmallAIOS-supported algorithm and the legacy LZMA1 format
    /// is no longer produced by `mksquashfs(1)`. Unknown IDs (anything
    /// outside `1, 4, 5, 6`) are also rejected.
    pub fn from_id(id: u16) -> Result<Self, SquashfsError> {
        match id {
            1 => Ok(Self::Gzip),
            4 => Ok(Self::Xz),
            5 => Ok(Self::Lz4),
            6 => Ok(Self::Zstd),
            _ => Err(SquashfsError::UnsupportedCompression(id)),
        }
    }

    /// Short static name suitable for error messages and audit-ring
    /// stubs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gzip => "gzip",
            Self::Xz => "xz",
            Self::Lz4 => "lz4",
            Self::Zstd => "zstd",
        }
    }
}

/// Parsed squashfs 4.0 superblock.
#[derive(Debug, Clone)]
pub struct Superblock {
    pub inode_count: u32,
    pub mod_time: u32,
    pub block_size: u32,
    pub fragment_count: u32,
    pub compression: Compression,
    pub block_log: u16,
    pub flags: u16,
    pub id_count: u16,
    pub version_major: u16,
    pub version_minor: u16,
    pub root_inode_ref: u64,
    pub bytes_used: u64,
    pub id_table_start: u64,
    pub xattr_table_start: u64,
    pub inode_table_start: u64,
    pub directory_table_start: u64,
    pub fragment_table_start: u64,
    pub lookup_table_start: u64,
}

impl Superblock {
    /// Parse a 96-byte superblock from the start of a squashfs image.
    pub fn parse(buf: &[u8]) -> Result<Self, SquashfsError> {
        if buf.len() < SUPERBLOCK_SIZE {
            return Err(SquashfsError::CorruptMetadata);
        }
        let magic = u32_le(&buf[0..4]);
        if magic != SQUASHFS_MAGIC {
            return Err(SquashfsError::BadMagic);
        }
        let version_major = u16_le(&buf[28..30]);
        let version_minor = u16_le(&buf[30..32]);
        if version_major != SQUASHFS_MAJOR_VERSION {
            return Err(SquashfsError::UnsupportedMajorVersion(version_major));
        }

        let compression_id = u16_le(&buf[20..22]);
        let compression = Compression::from_id(compression_id)?;

        let block_size = u32_le(&buf[12..16]);
        let block_log = u16_le(&buf[22..24]);

        // `block_size` MUST be a power of two between 4 KiB and 1 MiB.
        if !block_size.is_power_of_two()
            || !(4096..=1_048_576).contains(&block_size)
            || (block_size as u64).trailing_zeros() != block_log as u32
        {
            return Err(SquashfsError::CorruptMetadata);
        }

        Ok(Self {
            inode_count: u32_le(&buf[4..8]),
            mod_time: u32_le(&buf[8..12]),
            block_size,
            fragment_count: u32_le(&buf[16..20]),
            compression,
            block_log,
            flags: u16_le(&buf[24..26]),
            id_count: u16_le(&buf[26..28]),
            version_major,
            version_minor,
            root_inode_ref: u64_le(&buf[32..40]),
            bytes_used: u64_le(&buf[40..48]),
            id_table_start: u64_le(&buf[48..56]),
            xattr_table_start: u64_le(&buf[56..64]),
            inode_table_start: u64_le(&buf[64..72]),
            directory_table_start: u64_le(&buf[72..80]),
            fragment_table_start: u64_le(&buf[80..88]),
            lookup_table_start: u64_le(&buf[88..96]),
        })
    }

    /// True iff the `xattr_table_start` is the sentinel
    /// `0xFFFF_FFFF_FFFF_FFFF` (no xattr table).
    pub fn has_xattr_table(&self) -> bool {
        self.xattr_table_start != u64::MAX && self.flags & FLAG_NO_XATTR == 0
    }

    /// True iff the image has a fragment table.
    pub fn has_fragments(&self) -> bool {
        self.fragment_count > 0
            && self.flags & FLAG_NO_FRAGMENTS == 0
            && self.fragment_table_start != u64::MAX
    }

    /// True iff the image has an export table.
    pub fn has_exports(&self) -> bool {
        self.flags & FLAG_EXPORTABLE != 0 && self.lookup_table_start != u64::MAX
    }
}

#[inline]
fn u16_le(buf: &[u8]) -> u16 {
    u16::from_le_bytes([buf[0], buf[1]])
}
#[inline]
fn u32_le(buf: &[u8]) -> u32 {
    u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
}
#[inline]
fn u64_le(buf: &[u8]) -> u64 {
    u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_superblock(major: u16, comp_id: u16, block_size: u32, block_log: u16) -> [u8; 96] {
        let mut buf = [0u8; 96];
        buf[0..4].copy_from_slice(&SQUASHFS_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // inode_count
        buf[12..16].copy_from_slice(&block_size.to_le_bytes());
        buf[20..22].copy_from_slice(&comp_id.to_le_bytes());
        buf[22..24].copy_from_slice(&block_log.to_le_bytes());
        buf[28..30].copy_from_slice(&major.to_le_bytes());
        buf[30..32].copy_from_slice(&0u16.to_le_bytes());
        buf
    }

    #[test]
    fn parse_minimal_4_0() {
        let sb = make_superblock(4, 6, 131_072, 17);
        let parsed = Superblock::parse(&sb).unwrap();
        assert_eq!(parsed.compression, Compression::Zstd);
        assert_eq!(parsed.block_size, 131_072);
        assert_eq!(parsed.version_major, 4);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut sb = make_superblock(4, 6, 131_072, 17);
        sb[0] ^= 0x55;
        assert!(matches!(
            Superblock::parse(&sb),
            Err(SquashfsError::BadMagic)
        ));
    }

    #[test]
    fn rejects_3x_major() {
        let sb = make_superblock(3, 1, 131_072, 17);
        assert!(matches!(
            Superblock::parse(&sb),
            Err(SquashfsError::UnsupportedMajorVersion(3))
        ));
    }

    #[test]
    fn rejects_5x_major() {
        let sb = make_superblock(5, 1, 131_072, 17);
        assert!(matches!(
            Superblock::parse(&sb),
            Err(SquashfsError::UnsupportedMajorVersion(5))
        ));
    }

    #[test]
    fn rejects_lzma_id_2() {
        let sb = make_superblock(4, 2, 131_072, 17);
        assert!(matches!(
            Superblock::parse(&sb),
            Err(SquashfsError::UnsupportedCompression(2))
        ));
    }

    #[test]
    fn rejects_lzo_id_3() {
        let sb = make_superblock(4, 3, 131_072, 17);
        assert!(matches!(
            Superblock::parse(&sb),
            Err(SquashfsError::UnsupportedCompression(3))
        ));
    }

    #[test]
    fn accepts_id_1_4_5_6() {
        for (id, want) in [
            (1, Compression::Gzip),
            (4, Compression::Xz),
            (5, Compression::Lz4),
            (6, Compression::Zstd),
        ] {
            let sb = make_superblock(4, id, 131_072, 17);
            let p = Superblock::parse(&sb).unwrap();
            assert_eq!(p.compression, want);
        }
    }

    #[test]
    fn rejects_inconsistent_block_log() {
        let sb = make_superblock(4, 6, 131_072, 16); // log mismatch (should be 17)
        assert!(matches!(
            Superblock::parse(&sb),
            Err(SquashfsError::CorruptMetadata)
        ));
    }

    #[test]
    fn rejects_block_size_below_min() {
        let sb = make_superblock(4, 6, 2048, 11);
        assert!(matches!(
            Superblock::parse(&sb),
            Err(SquashfsError::CorruptMetadata)
        ));
    }

    #[test]
    fn rejects_block_size_above_max() {
        let sb = make_superblock(4, 6, 2_097_152, 21);
        assert!(matches!(
            Superblock::parse(&sb),
            Err(SquashfsError::CorruptMetadata)
        ));
    }

    #[test]
    fn short_buffer_rejected() {
        let buf = [0u8; 95];
        assert!(matches!(
            Superblock::parse(&buf),
            Err(SquashfsError::CorruptMetadata)
        ));
    }

    #[test]
    fn flags_xattr_no_xattr() {
        let mut sb = make_superblock(4, 6, 4096, 12);
        sb[24..26].copy_from_slice(&FLAG_NO_XATTR.to_le_bytes());
        // xattr_table_start is irrelevant when FLAG_NO_XATTR is set.
        let parsed = Superblock::parse(&sb).unwrap();
        assert!(!parsed.has_xattr_table());
    }
}
