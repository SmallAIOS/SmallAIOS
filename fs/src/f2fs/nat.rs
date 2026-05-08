// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS Node Address Table (NAT) reader.
//!
//! The NAT maps a 32-bit `nid` (node-id) to a physical block address
//! in the main area. Each NAT entry is 9 bytes on disk:
//!
//! - `version` (1 byte) — bumps each time the entry is rewritten.
//! - `ino` (4 bytes) — owning inode number (0 = unused entry).
//! - `block_addr` (4 bytes) — physical block address (`NULL_ADDR = 0`,
//!   `NEW_ADDR = -1` are sentinel values).
//!
//! NAT blocks are 4 KiB and contain `4096 / 9 = 455` entries with 1
//! byte of trailing padding (Linux 6.6 `NAT_ENTRY_PER_BLOCK = 455`).
//!
//! NAT segments are organized in pairs (each NAT block has a primary
//! and a secondary copy chosen by a per-block bit in the NAT version
//! bitmap from the checkpoint).

extern crate alloc;

use super::error::F2fsError;
use super::{u32_le, F2FS_BLOCK_SIZE};

/// Bytes per NAT entry on disk.
pub const NAT_ENTRY_SIZE: usize = 9;

/// NAT entries per 4 KiB block.
pub const NAT_ENTRY_PER_BLOCK: usize = (F2FS_BLOCK_SIZE as usize) / NAT_ENTRY_SIZE;

/// Sentinel block address: "no allocation".
pub const NULL_ADDR: u32 = 0;
/// Sentinel block address: "newly allocated, not yet flushed".
pub const NEW_ADDR: u32 = u32::MAX;

/// Decoded NAT entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NatEntry {
    pub version: u8,
    pub ino: u32,
    pub block_addr: u32,
}

impl NatEntry {
    /// Parse one NAT entry. `buf.len() >= [`NAT_ENTRY_SIZE`]` required.
    pub fn parse(buf: &[u8]) -> Result<Self, F2fsError> {
        if buf.len() < NAT_ENTRY_SIZE {
            return Err(F2fsError::NatBlockCrcMismatch);
        }
        let version = buf[0];
        let ino = u32_le(&buf[1..5]);
        let block_addr = u32_le(&buf[5..9]);
        Ok(Self {
            version,
            ino,
            block_addr,
        })
    }

    /// True if this entry is a sentinel "unallocated" slot.
    pub fn is_unallocated(&self) -> bool {
        self.block_addr == NULL_ADDR
    }

    /// True if this entry was allocated but not yet flushed.
    pub fn is_new(&self) -> bool {
        self.block_addr == NEW_ADDR
    }

    /// True if the entry holds a valid physical block address.
    pub fn is_valid_addr(&self) -> bool {
        self.block_addr != NULL_ADDR && self.block_addr != NEW_ADDR
    }
}

/// Parse a 4 KiB NAT block into the 455 entries it holds.
pub fn parse_nat_block(block: &[u8]) -> Result<alloc::vec::Vec<NatEntry>, F2fsError> {
    if block.len() < F2FS_BLOCK_SIZE as usize {
        return Err(F2fsError::NatBlockCrcMismatch);
    }
    let mut out = alloc::vec::Vec::with_capacity(NAT_ENTRY_PER_BLOCK);
    for i in 0..NAT_ENTRY_PER_BLOCK {
        let off = i * NAT_ENTRY_SIZE;
        out.push(NatEntry::parse(&block[off..off + NAT_ENTRY_SIZE])?);
    }
    Ok(out)
}

/// Compute the NAT block index that contains a given `nid`.
#[inline]
pub fn nid_to_nat_block_idx(nid: u32) -> u32 {
    nid / NAT_ENTRY_PER_BLOCK as u32
}

/// Compute the entry-within-block offset for a given `nid`.
#[inline]
pub fn nid_to_entry_off(nid: u32) -> usize {
    (nid as usize) % NAT_ENTRY_PER_BLOCK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_parse_round_trip() {
        let buf = [0x05u8, 0x07, 0x00, 0x00, 0x00, 0x40, 0x10, 0x00, 0x00];
        let e = NatEntry::parse(&buf).unwrap();
        assert_eq!(e.version, 5);
        assert_eq!(e.ino, 7);
        assert_eq!(e.block_addr, 0x1040);
        assert!(e.is_valid_addr());
    }

    #[test]
    fn unallocated_detected() {
        let buf = [0u8; NAT_ENTRY_SIZE];
        let e = NatEntry::parse(&buf).unwrap();
        assert!(e.is_unallocated());
        assert!(!e.is_new());
        assert!(!e.is_valid_addr());
    }

    #[test]
    fn new_addr_detected() {
        let mut buf = [0u8; NAT_ENTRY_SIZE];
        buf[5..9].copy_from_slice(&NEW_ADDR.to_le_bytes());
        let e = NatEntry::parse(&buf).unwrap();
        assert!(e.is_new());
        assert!(!e.is_valid_addr());
    }

    #[test]
    fn truncated_buffer_rejected() {
        let buf = [0u8; 4];
        assert!(matches!(
            NatEntry::parse(&buf),
            Err(F2fsError::NatBlockCrcMismatch)
        ));
    }

    #[test]
    fn parse_nat_block_round_trip() {
        let mut block = alloc::vec![0u8; F2FS_BLOCK_SIZE as usize];
        for i in 0..NAT_ENTRY_PER_BLOCK {
            let off = i * NAT_ENTRY_SIZE;
            block[off] = (i & 0xFF) as u8;
            block[off + 1..off + 5].copy_from_slice(&(i as u32 + 1).to_le_bytes());
            block[off + 5..off + 9].copy_from_slice(&(0x10000 + i as u32).to_le_bytes());
        }
        let entries = parse_nat_block(&block).unwrap();
        assert_eq!(entries.len(), NAT_ENTRY_PER_BLOCK);
        assert_eq!(entries[0].ino, 1);
        assert_eq!(entries[0].block_addr, 0x10000);
        assert_eq!(entries[100].ino, 101);
    }

    #[test]
    fn nid_routing() {
        assert_eq!(nid_to_nat_block_idx(0), 0);
        assert_eq!(nid_to_nat_block_idx(454), 0);
        assert_eq!(nid_to_nat_block_idx(455), 1);
        assert_eq!(nid_to_entry_off(0), 0);
        assert_eq!(nid_to_entry_off(454), 454);
        assert_eq!(nid_to_entry_off(455), 0);
    }

    #[test]
    fn truncated_block_rejected() {
        let block = alloc::vec![0u8; 100];
        assert!(matches!(
            parse_nat_block(&block),
            Err(F2fsError::NatBlockCrcMismatch)
        ));
    }
}
