// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS Segment Information Table (SIT) reader.
//!
//! The SIT is laid out across `segment_count_sit` segments, each
//! holding (block-size / SIT_ENTRY_SIZE) entries. Each entry tracks
//! one main-area segment's:
//!
//! - `vblocks` — count of valid blocks in the segment, plus segment
//!   type encoded in the high bits.
//! - `valid_map[64]` — 512-bit bitmap (one bit per 4 KiB block).
//! - `mtime` — last modification time (used by the GC heuristic; for
//!   the read path we just expose it).
//!
//! On-disk per Linux 6.6 `struct f2fs_sit_entry`: `__le16 vblocks` +
//! `__u8 valid_map[64]` + `__le64 mtime` = 74 bytes per entry. With
//! `block_size = 4096`, that yields 55 entries per SIT block (4096 /
//! 74 = 55, with 36 bytes of trailing padding ignored).
//!
//! The SIT blocks are also written in the kernel as
//! `struct f2fs_sit_block { struct f2fs_sit_entry entries[SIT_ENTRY_PER_BLOCK]; }`
//! where `SIT_ENTRY_PER_BLOCK = 55` is fixed by the format.

extern crate alloc;

use super::error::F2fsError;
use super::{u16_le, u64_le, F2FS_BLOCK_SIZE};

/// Bytes per SIT entry on disk.
pub const SIT_ENTRY_SIZE: usize = 74;

/// SIT entries per 4 KiB block.
pub const SIT_ENTRY_PER_BLOCK: usize = (F2FS_BLOCK_SIZE as usize) / SIT_ENTRY_SIZE;

/// Bytes of valid bitmap per SIT entry (one bit per 4 KiB block in a
/// 2 MiB segment = 512 bits = 64 bytes).
pub const SIT_VBLOCK_MAP_BYTES: usize = 64;

/// Mask for the segment-type bits in the high 2 bits of `vblocks`.
pub const SIT_VBLOCKS_TYPE_MASK: u16 = 0xC000;
/// Mask for the count of valid blocks in the low 14 bits of `vblocks`.
pub const SIT_VBLOCKS_COUNT_MASK: u16 = 0x3FFF;

/// Decoded F2FS segment type stored in the high bits of `vblocks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    /// Hot/warm/cold node (3 levels).
    Node(u8),
    /// Hot/warm/cold data (3 levels).
    Data(u8),
}

/// Parsed SIT entry.
#[derive(Debug, Clone)]
pub struct SitEntry {
    /// Number of valid 4 KiB blocks in this segment (0..=512).
    pub valid_blocks: u16,
    /// Bitmap of which blocks are valid (64 bytes = 512 bits).
    pub valid_map: [u8; SIT_VBLOCK_MAP_BYTES],
    /// Modification timestamp (Linux nsec since boot or wall clock —
    /// not authoritative; used only by GC).
    pub mtime: u64,
    /// Encoded type (top 2 bits of the on-disk `vblocks`).
    pub raw_type_bits: u16,
}

impl SitEntry {
    /// Parse one SIT entry from a byte slice (`buf.len() ==
    /// [`SIT_ENTRY_SIZE`]).
    pub fn parse(buf: &[u8]) -> Result<Self, F2fsError> {
        if buf.len() < SIT_ENTRY_SIZE {
            return Err(F2fsError::SitBlockCrcMismatch);
        }
        let raw = u16_le(&buf[0..2]);
        let valid_blocks = raw & SIT_VBLOCKS_COUNT_MASK;
        let raw_type_bits = raw & SIT_VBLOCKS_TYPE_MASK;
        let mut valid_map = [0u8; SIT_VBLOCK_MAP_BYTES];
        valid_map.copy_from_slice(&buf[2..2 + SIT_VBLOCK_MAP_BYTES]);
        let mtime = u64_le(&buf[2 + SIT_VBLOCK_MAP_BYTES..2 + SIT_VBLOCK_MAP_BYTES + 8]);
        Ok(Self {
            valid_blocks,
            valid_map,
            mtime,
            raw_type_bits,
        })
    }

    /// True if the `block_idx`-th block of the segment is valid.
    pub fn block_is_valid(&self, block_idx: usize) -> bool {
        if block_idx >= 512 {
            return false;
        }
        let byte = block_idx / 8;
        let bit = block_idx % 8;
        (self.valid_map[byte] >> bit) & 1 == 1
    }

    /// Compute the segment type from the raw bits.
    ///
    /// The kernel encodes `CURSEG_HOT_DATA = 0`, `CURSEG_WARM_DATA = 1`,
    /// `CURSEG_COLD_DATA = 2`, `CURSEG_HOT_NODE = 3`,
    /// `CURSEG_WARM_NODE = 4`, `CURSEG_COLD_NODE = 5` — but those are
    /// stored in the SSA, not SIT. The SIT keeps only a 2-bit
    /// "data vs node" flag for fast scanning. We surface the raw
    /// bits so callers that need the SSA-level type can do their own
    /// decoding.
    pub fn segment_type(&self) -> SegmentType {
        match self.raw_type_bits {
            0x0000 | 0x4000 => SegmentType::Data((self.raw_type_bits >> 14) as u8),
            0x8000 | 0xC000 => SegmentType::Node(((self.raw_type_bits >> 14) & 0x1) as u8),
            _ => SegmentType::Data(0),
        }
    }
}

/// Parse all SIT entries from a 4 KiB SIT block.
pub fn parse_sit_block(block: &[u8]) -> Result<alloc::vec::Vec<SitEntry>, F2fsError> {
    if block.len() < F2FS_BLOCK_SIZE as usize {
        return Err(F2fsError::SitBlockCrcMismatch);
    }
    let mut entries = alloc::vec::Vec::with_capacity(SIT_ENTRY_PER_BLOCK);
    for i in 0..SIT_ENTRY_PER_BLOCK {
        let off = i * SIT_ENTRY_SIZE;
        entries.push(SitEntry::parse(&block[off..off + SIT_ENTRY_SIZE])?);
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(vblocks: u16, valid_bits: &[usize], mtime: u64) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec![0u8; SIT_ENTRY_SIZE];
        buf[0..2].copy_from_slice(&vblocks.to_le_bytes());
        for &bit in valid_bits {
            let byte = bit / 8;
            let off = bit % 8;
            buf[2 + byte] |= 1u8 << off;
        }
        buf[2 + SIT_VBLOCK_MAP_BYTES..2 + SIT_VBLOCK_MAP_BYTES + 8]
            .copy_from_slice(&mtime.to_le_bytes());
        buf
    }

    #[test]
    fn parse_entry_round_trip() {
        let buf = make_entry(5, &[0, 1, 2, 5, 7], 1234567);
        let entry = SitEntry::parse(&buf).unwrap();
        assert_eq!(entry.valid_blocks, 5);
        assert_eq!(entry.mtime, 1234567);
        assert!(entry.block_is_valid(0));
        assert!(entry.block_is_valid(7));
        assert!(!entry.block_is_valid(3));
        assert!(!entry.block_is_valid(511));
    }

    #[test]
    fn block_idx_bounds_safe() {
        let buf = make_entry(0, &[], 0);
        let entry = SitEntry::parse(&buf).unwrap();
        assert!(!entry.block_is_valid(512));
        assert!(!entry.block_is_valid(usize::MAX));
    }

    #[test]
    fn type_bits_decoded() {
        let buf = make_entry(0xC000 | 5, &[], 0);
        let entry = SitEntry::parse(&buf).unwrap();
        assert_eq!(entry.valid_blocks, 5);
        assert_eq!(entry.raw_type_bits, 0xC000);
        assert!(matches!(entry.segment_type(), SegmentType::Node(_)));
    }

    #[test]
    fn parse_block_round_trip() {
        let mut block = alloc::vec![0u8; F2FS_BLOCK_SIZE as usize];
        for i in 0..SIT_ENTRY_PER_BLOCK {
            let off = i * SIT_ENTRY_SIZE;
            let raw = i as u16 + 1;
            block[off..off + 2].copy_from_slice(&raw.to_le_bytes());
        }
        let entries = parse_sit_block(&block).unwrap();
        assert_eq!(entries.len(), SIT_ENTRY_PER_BLOCK);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.valid_blocks, i as u16 + 1);
        }
    }

    #[test]
    fn truncated_block_rejected() {
        let block = alloc::vec![0u8; 100];
        assert!(matches!(
            parse_sit_block(&block),
            Err(F2fsError::SitBlockCrcMismatch)
        ));
    }

    #[test]
    fn truncated_entry_rejected() {
        let buf = alloc::vec![0u8; SIT_ENTRY_SIZE - 1];
        assert!(SitEntry::parse(&buf).is_err());
    }

    #[test]
    fn full_512_bits_valid() {
        let bits: alloc::vec::Vec<usize> = (0..512).collect();
        let buf = make_entry(512, &bits, 0);
        let entry = SitEntry::parse(&buf).unwrap();
        assert_eq!(entry.valid_blocks, 512);
        for i in 0..512 {
            assert!(entry.block_is_valid(i));
        }
    }
}
