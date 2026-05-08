// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS checkpoint pack reader.
//!
//! F2FS keeps two checkpoint packs (CP1, CP2) for crash safety; on
//! mount, the reader picks the highest valid version. Each pack
//! contains:
//!
//! - The checkpoint header block (`struct f2fs_checkpoint`).
//! - 1+ summary blocks for active node/data segments.
//! - The checkpoint footer block (a duplicate of the header).
//!
//! Both header and footer carry a CRC32C plus a `checkpoint_ver`
//! version counter. To pick a valid pack, the reader requires:
//!
//! 1. Header CRC valid.
//! 2. Footer CRC valid.
//! 3. Header `checkpoint_ver` == footer `checkpoint_ver`.
//!
//! If both packs are valid, the higher `checkpoint_ver` wins. If only
//! one is valid the other is ignored.
//!
//! Pinned to `fs/f2fs/f2fs.h:struct f2fs_checkpoint` (Linux 6.6).

extern crate alloc;

use super::crc::verify_f2fs_crc32;
use super::error::F2fsError;
use super::{u32_le, u64_le};

/// On-disk header byte length we parse (subset of struct fields up to
/// the CRC location). The full Linux struct is up to 4 KiB; we only
/// touch the load-bearing prefix.
pub const CHECKPOINT_HEADER_BYTES: usize = 192;

/// Default CRC offset within the checkpoint header (4096 - 4 = 4092).
/// Linux writes the CRC at `cp->checksum_offset`, which is normally
/// `4096 - sizeof(__le32)` for the v1 layout.
pub const DEFAULT_CHECKPOINT_CRC_OFFSET: usize = 4092;

/// Parsed F2FS checkpoint pack header (load-bearing fields only).
///
/// Layout (Linux 6.6 `struct f2fs_checkpoint`, little-endian):
///
/// | Off | Size | Field                       |
/// |----:|-----:|-----------------------------|
/// |   0 |   8  | `checkpoint_ver`            |
/// |   8 |   8  | `user_block_count`          |
/// |  16 |   8  | `valid_block_count`         |
/// |  24 |   4  | `rsvd_segment_count`        |
/// |  28 |   4  | `overprov_segment_count`    |
/// |  32 |   4  | `free_segment_count`        |
/// |  36 |  4×8 | `cur_node_segno[8]`         |
/// |  68 |  2×8 | `cur_node_blkoff[8]`        |
/// |  84 |  4×8 | `cur_data_segno[8]`         |
/// | 116 |  2×8 | `cur_data_blkoff[8]`        |
/// | 132 |   4  | `valid_node_count`          |
/// | 136 |   4  | `valid_inode_count`         |
/// | 140 |   4  | `next_free_nid`             |
/// | 144 |   4  | `sit_ver_bitmap_bytesize`   |
/// | 148 |   4  | `nat_ver_bitmap_bytesize`   |
/// | 152 |   4  | `checksum_offset`           |
/// | 156 |   8  | `elapsed_time`              |
/// | 164 |   1  | `alloc_type[CURSEG_NODE_TYPE_MAX]`(8) |
/// | 172 |  ... | `sit_nat_version_bitmap[]` (variable)  |
/// | ... |   4  | CRC32C (at `checksum_offset`) |
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub checkpoint_ver: u64,
    pub user_block_count: u64,
    pub valid_block_count: u64,
    pub rsvd_segment_count: u32,
    pub overprov_segment_count: u32,
    pub free_segment_count: u32,
    /// Active node-segment IDs (8 streams).
    pub cur_node_segno: [u32; 8],
    /// Active node-segment block offsets within their segment (16-bit).
    pub cur_node_blkoff: [u16; 8],
    /// Active data-segment IDs (8 streams).
    pub cur_data_segno: [u32; 8],
    /// Active data-segment block offsets within their segment.
    pub cur_data_blkoff: [u16; 8],
    pub valid_node_count: u32,
    pub valid_inode_count: u32,
    pub next_free_nid: u32,
    pub sit_ver_bitmap_bytesize: u32,
    pub nat_ver_bitmap_bytesize: u32,
    pub checksum_offset: u32,
    pub elapsed_time: u64,
    pub stored_crc: u32,
}

impl Checkpoint {
    /// Parse the 4 KiB checkpoint pack block.
    pub fn parse(block: &[u8]) -> Result<Self, F2fsError> {
        if block.len() < CHECKPOINT_HEADER_BYTES {
            return Err(F2fsError::InodeCorrupt);
        }
        let checkpoint_ver = u64_le(&block[0..8]);
        let user_block_count = u64_le(&block[8..16]);
        let valid_block_count = u64_le(&block[16..24]);
        let rsvd_segment_count = u32_le(&block[24..28]);
        let overprov_segment_count = u32_le(&block[28..32]);
        let free_segment_count = u32_le(&block[32..36]);

        let mut cur_node_segno = [0u32; 8];
        for (i, slot) in cur_node_segno.iter_mut().enumerate() {
            let o = 36 + i * 4;
            *slot = u32_le(&block[o..o + 4]);
        }
        let mut cur_node_blkoff = [0u16; 8];
        for (i, slot) in cur_node_blkoff.iter_mut().enumerate() {
            let o = 68 + i * 2;
            *slot = u16::from_le_bytes([block[o], block[o + 1]]);
        }
        let mut cur_data_segno = [0u32; 8];
        for (i, slot) in cur_data_segno.iter_mut().enumerate() {
            let o = 84 + i * 4;
            *slot = u32_le(&block[o..o + 4]);
        }
        let mut cur_data_blkoff = [0u16; 8];
        for (i, slot) in cur_data_blkoff.iter_mut().enumerate() {
            let o = 116 + i * 2;
            *slot = u16::from_le_bytes([block[o], block[o + 1]]);
        }

        let valid_node_count = u32_le(&block[132..136]);
        let valid_inode_count = u32_le(&block[136..140]);
        let next_free_nid = u32_le(&block[140..144]);
        let sit_ver_bitmap_bytesize = u32_le(&block[144..148]);
        let nat_ver_bitmap_bytesize = u32_le(&block[148..152]);
        let checksum_offset = u32_le(&block[152..156]);
        let elapsed_time = u64_le(&block[156..164]);

        // Read the on-disk CRC if `checksum_offset` is in range.
        let stored_crc = if checksum_offset != 0 && (checksum_offset as usize) + 4 <= block.len() {
            u32_le(&block[checksum_offset as usize..checksum_offset as usize + 4])
        } else {
            0
        };

        Ok(Self {
            checkpoint_ver,
            user_block_count,
            valid_block_count,
            rsvd_segment_count,
            overprov_segment_count,
            free_segment_count,
            cur_node_segno,
            cur_node_blkoff,
            cur_data_segno,
            cur_data_blkoff,
            valid_node_count,
            valid_inode_count,
            next_free_nid,
            sit_ver_bitmap_bytesize,
            nat_ver_bitmap_bytesize,
            checksum_offset,
            elapsed_time,
            stored_crc,
        })
    }

    /// Verify the checkpoint header CRC. The kernel computes CRC over
    /// the bytes `[0..checksum_offset]` with seed 0.
    pub fn verify_crc(&self, raw_block: &[u8]) -> Result<(), F2fsError> {
        if self.checksum_offset == 0 {
            return Ok(());
        }
        let off = self.checksum_offset as usize;
        if off + 4 > raw_block.len() {
            return Err(F2fsError::CheckpointCrcMismatch);
        }
        if !verify_f2fs_crc32(0, &raw_block[..off], self.stored_crc) {
            return Err(F2fsError::CheckpointCrcMismatch);
        }
        Ok(())
    }

    /// Pick the higher-version checkpoint between two candidates.
    ///
    /// Both `cp1` and `cp2` are pre-CRC-verified by the caller; this
    /// helper just compares versions and returns the index (0 = cp1,
    /// 1 = cp2) of the winner.
    pub fn pick_winner(cp1_ver: u64, cp2_ver: u64) -> usize {
        if cp2_ver > cp1_ver {
            1
        } else {
            0
        }
    }
}

/// Read both checkpoint packs and return the winning [`Checkpoint`].
///
/// `cp1_block` and `cp2_block` are the raw 4 KiB pack header blocks.
/// Both must already have been read from disk by the caller.
pub fn select_checkpoint(cp1_block: &[u8], cp2_block: &[u8]) -> Result<Checkpoint, F2fsError> {
    let cp1 = Checkpoint::parse(cp1_block).ok();
    let cp2 = Checkpoint::parse(cp2_block).ok();

    let cp1_ok = cp1
        .as_ref()
        .is_some_and(|c| c.verify_crc(cp1_block).is_ok());
    let cp2_ok = cp2
        .as_ref()
        .is_some_and(|c| c.verify_crc(cp2_block).is_ok());

    match (cp1_ok, cp2_ok, cp1, cp2) {
        (true, true, Some(c1), Some(c2)) => {
            if c2.checkpoint_ver > c1.checkpoint_ver {
                Ok(c2)
            } else {
                Ok(c1)
            }
        }
        (true, false, Some(c1), _) => Ok(c1),
        (false, true, _, Some(c2)) => Ok(c2),
        _ => Err(F2fsError::CheckpointCrcMismatch),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f2fs::crc::f2fs_crc32;

    fn make_cp_block(ver: u64) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec![0u8; 4096];
        buf[0..8].copy_from_slice(&ver.to_le_bytes());
        buf[8..16].copy_from_slice(&100u64.to_le_bytes());
        buf[16..24].copy_from_slice(&50u64.to_le_bytes());
        buf[152..156].copy_from_slice(&(DEFAULT_CHECKPOINT_CRC_OFFSET as u32).to_le_bytes());
        let crc = f2fs_crc32(0, &buf[..DEFAULT_CHECKPOINT_CRC_OFFSET]);
        buf[DEFAULT_CHECKPOINT_CRC_OFFSET..DEFAULT_CHECKPOINT_CRC_OFFSET + 4]
            .copy_from_slice(&crc.to_le_bytes());
        buf
    }

    #[test]
    fn parse_round_trip() {
        let buf = make_cp_block(7);
        let cp = Checkpoint::parse(&buf).unwrap();
        assert_eq!(cp.checkpoint_ver, 7);
        assert_eq!(cp.user_block_count, 100);
        assert_eq!(cp.valid_block_count, 50);
        assert_eq!(cp.checksum_offset, DEFAULT_CHECKPOINT_CRC_OFFSET as u32);
        assert!(cp.verify_crc(&buf).is_ok());
    }

    #[test]
    fn crc_mismatch_detected() {
        let mut buf = make_cp_block(3);
        buf[DEFAULT_CHECKPOINT_CRC_OFFSET] ^= 0xFF;
        let cp = Checkpoint::parse(&buf).unwrap();
        assert!(matches!(
            cp.verify_crc(&buf),
            Err(F2fsError::CheckpointCrcMismatch)
        ));
    }

    #[test]
    fn pick_winner_uses_higher_version() {
        let cp1 = make_cp_block(5);
        let cp2 = make_cp_block(9);
        let chosen = select_checkpoint(&cp1, &cp2).unwrap();
        assert_eq!(chosen.checkpoint_ver, 9);
    }

    #[test]
    fn pick_winner_prefers_valid_when_other_corrupt() {
        let cp1 = make_cp_block(5);
        let mut cp2 = make_cp_block(9);
        cp2[DEFAULT_CHECKPOINT_CRC_OFFSET] ^= 0x01;
        let chosen = select_checkpoint(&cp1, &cp2).unwrap();
        assert_eq!(chosen.checkpoint_ver, 5);
    }

    #[test]
    fn both_corrupt_fails() {
        let mut cp1 = make_cp_block(5);
        let mut cp2 = make_cp_block(9);
        cp1[DEFAULT_CHECKPOINT_CRC_OFFSET] ^= 0x01;
        cp2[DEFAULT_CHECKPOINT_CRC_OFFSET] ^= 0x01;
        assert!(matches!(
            select_checkpoint(&cp1, &cp2),
            Err(F2fsError::CheckpointCrcMismatch)
        ));
    }

    #[test]
    fn truncated_buffer_rejected() {
        let buf = alloc::vec![0u8; 8];
        assert!(matches!(
            Checkpoint::parse(&buf),
            Err(F2fsError::InodeCorrupt)
        ));
    }

    #[test]
    fn pick_winner_index_helper() {
        assert_eq!(Checkpoint::pick_winner(1, 2), 1);
        assert_eq!(Checkpoint::pick_winner(2, 1), 0);
        assert_eq!(Checkpoint::pick_winner(2, 2), 0);
    }

    #[test]
    fn cur_node_segno_round_trip() {
        let mut buf = make_cp_block(1);
        buf[36..40].copy_from_slice(&42u32.to_le_bytes());
        buf[40..44].copy_from_slice(&7u32.to_le_bytes());
        // Recompute CRC after mutation.
        let crc = f2fs_crc32(0, &buf[..DEFAULT_CHECKPOINT_CRC_OFFSET]);
        buf[DEFAULT_CHECKPOINT_CRC_OFFSET..DEFAULT_CHECKPOINT_CRC_OFFSET + 4]
            .copy_from_slice(&crc.to_le_bytes());
        let cp = Checkpoint::parse(&buf).unwrap();
        assert_eq!(cp.cur_node_segno[0], 42);
        assert_eq!(cp.cur_node_segno[1], 7);
        assert!(cp.verify_crc(&buf).is_ok());
    }
}
