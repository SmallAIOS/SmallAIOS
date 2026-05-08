// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS Segment Summary Area (SSA) reader.
//!
//! The SSA stores per-segment reverse-mapping summary blocks. For each
//! main-area segment there is one 4 KiB summary block laid out as:
//!
//! - `n_nats` (u32) — number of NAT-journal entries (legacy in-block).
//! - `n_sits` (u32) — number of SIT-journal entries.
//! - `entries[512]` — one [`SummaryEntry`] per 4 KiB block in the
//!   segment.
//! - `footer` — `entry_type` (1 byte), padding, CRC32C.
//!
//! Each [`SummaryEntry`] carries a 7-byte payload mapping a physical
//! block back to its owning node-id + offset:
//!
//! - `nid` (u32) — owning node-id.
//! - `version` (u8)
//! - `ofs_in_node` (u16) — block index within the inode's data array.
//!
//! For a read-only mount we only need to parse the summary entries
//! when doing reverse-mapping (e.g. to confirm that a block actually
//! belongs to the inode the NAT lookup said it should). The v1 read
//! path uses NAT directly and treats the SSA as advisory.

extern crate alloc;

use super::error::F2fsError;
use super::{u16_le, u32_le, F2FS_BLOCK_SIZE};

/// Bytes per SSA summary entry.
pub const SUMMARY_ENTRY_SIZE: usize = 7;

/// Summary entries per 4 KiB block (one per 4 KiB block in a segment).
pub const SUMMARY_ENTRY_PER_BLOCK: usize = 512;

/// Decoded SSA summary entry (per-block reverse mapping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SummaryEntry {
    pub nid: u32,
    pub version: u8,
    pub ofs_in_node: u16,
}

impl SummaryEntry {
    /// Parse one summary entry. `buf.len() >= 7` required.
    pub fn parse(buf: &[u8]) -> Result<Self, F2fsError> {
        if buf.len() < SUMMARY_ENTRY_SIZE {
            return Err(F2fsError::InodeCorrupt);
        }
        let nid = u32_le(&buf[0..4]);
        let version = buf[4];
        let ofs_in_node = u16_le(&buf[5..7]);
        Ok(Self {
            nid,
            version,
            ofs_in_node,
        })
    }
}

/// Parsed SSA summary block (header + 512 entries).
#[derive(Debug, Clone)]
pub struct SummaryBlock {
    pub n_nats: u32,
    pub n_sits: u32,
    pub entries: alloc::vec::Vec<SummaryEntry>,
}

impl SummaryBlock {
    /// Parse a 4 KiB SSA summary block.
    pub fn parse(block: &[u8]) -> Result<Self, F2fsError> {
        if block.len() < F2FS_BLOCK_SIZE as usize {
            return Err(F2fsError::InodeCorrupt);
        }
        let n_nats = u32_le(&block[0..4]);
        let n_sits = u32_le(&block[4..8]);
        let mut entries = alloc::vec::Vec::with_capacity(SUMMARY_ENTRY_PER_BLOCK);
        // Entries start after the 8-byte header (Linux 6.6 layout puts
        // them at offset 8; we skip the journals which would precede
        // them on a written-back checkpoint, but for the conformance
        // path entries[] starts at offset 8).
        for i in 0..SUMMARY_ENTRY_PER_BLOCK {
            let off = 8 + i * SUMMARY_ENTRY_SIZE;
            if off + SUMMARY_ENTRY_SIZE > block.len() {
                return Err(F2fsError::InodeCorrupt);
            }
            entries.push(SummaryEntry::parse(&block[off..off + SUMMARY_ENTRY_SIZE])?);
        }
        Ok(Self {
            n_nats,
            n_sits,
            entries,
        })
    }

    /// Look up the reverse-mapping for the `block_idx`-th block of the
    /// segment.
    pub fn entry(&self, block_idx: usize) -> Option<&SummaryEntry> {
        self.entries.get(block_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_summary_block(entries: &[(u32, u8, u16)]) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec![0u8; F2FS_BLOCK_SIZE as usize];
        buf[0..4].copy_from_slice(&0u32.to_le_bytes());
        buf[4..8].copy_from_slice(&0u32.to_le_bytes());
        for (i, &(nid, ver, ofs)) in entries.iter().enumerate() {
            let off = 8 + i * SUMMARY_ENTRY_SIZE;
            buf[off..off + 4].copy_from_slice(&nid.to_le_bytes());
            buf[off + 4] = ver;
            buf[off + 5..off + 7].copy_from_slice(&ofs.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parse_round_trip() {
        let block = make_summary_block(&[(7, 1, 0), (7, 1, 1), (8, 2, 0)]);
        let sb = SummaryBlock::parse(&block).unwrap();
        assert_eq!(sb.entries.len(), 512);
        assert_eq!(sb.entries[0].nid, 7);
        assert_eq!(sb.entries[0].ofs_in_node, 0);
        assert_eq!(sb.entries[1].nid, 7);
        assert_eq!(sb.entries[2].nid, 8);
        assert_eq!(sb.entries[2].version, 2);
        // Entry 3 is zeroed by default.
        assert_eq!(
            sb.entries[3],
            SummaryEntry {
                nid: 0,
                version: 0,
                ofs_in_node: 0
            }
        );
    }

    #[test]
    fn entry_lookup() {
        let block = make_summary_block(&[(7, 1, 0)]);
        let sb = SummaryBlock::parse(&block).unwrap();
        assert!(sb.entry(0).is_some());
        assert!(sb.entry(1024).is_none());
    }

    #[test]
    fn truncated_block_rejected() {
        let block = alloc::vec![0u8; 100];
        assert!(matches!(
            SummaryBlock::parse(&block),
            Err(F2fsError::InodeCorrupt)
        ));
    }

    #[test]
    fn truncated_entry_rejected() {
        let buf = alloc::vec![0u8; 4];
        assert!(SummaryEntry::parse(&buf).is_err());
    }

    #[test]
    fn entry_round_trip_direct() {
        let mut buf = alloc::vec![0u8; SUMMARY_ENTRY_SIZE];
        buf[0..4].copy_from_slice(&42u32.to_le_bytes());
        buf[4] = 5;
        buf[5..7].copy_from_slice(&7u16.to_le_bytes());
        let e = SummaryEntry::parse(&buf).unwrap();
        assert_eq!(e.nid, 42);
        assert_eq!(e.version, 5);
        assert_eq!(e.ofs_in_node, 7);
    }
}
