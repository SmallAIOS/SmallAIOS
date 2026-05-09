// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS journal — in-memory record of in-flight rename operations.
//!
//! Phase 7 introduces a tiny journal to make `rename(src, dst)` atomic
//! across power loss. The flow is:
//!
//! 1. The caller invokes [`crate::f2fs::F2fs::rename`].
//! 2. We **stage** a [`JournalEntry::Rename`] record into the
//!    in-memory write-state. The entry holds the src/dst paths, the
//!    moved inode-id, and (for clobber-renames) the displaced
//!    `dst_old_ino`.
//! 3. We update the parent directories' dentry blocks (we already do
//!    this in the write path).
//! 4. On [`crate::f2fs::F2fs::fsync`] / checkpoint commit we **persist**
//!    the journal alongside the dirty NAT/SIT/inode blocks.
//! 5. On commit success the entry is **retired** (cleared from memory).
//!
//! Crash recovery: on the next mount, any persisted-but-unretired
//! journal entry is *replayed*. Since both pre- and post-rename states
//! are atomic snapshots of the directory tree, the system always
//! reaches a consistent state — either both src and dst dentries reflect
//! the move, or neither does.
//!
//! ## On-disk layout (Phase 7 v1)
//!
//! Journal entries are written into the second 4 KiB block of the
//! checkpoint pack (the "journal block"), following a tiny header:
//!
//! ```text
//! +------------+--------+-------+-----+--------+
//! | magic (u32)| ver(u8)| n (u8)| pad | entries|
//! +------------+--------+-------+-----+--------+
//! ```
//!
//! Each entry is a fixed 16-byte record:
//!
//! ```text
//! | kind | rsvd | src_nid | dst_nid | dst_old_nid | flags |
//! ```
//!
//! Path strings are *not* persisted in v1; the in-memory rename uses
//! parent-inode + name lookups. Recovery scans both parents to detect
//! whether the rename committed (dst already points at the moved nid)
//! or not (src still points at it).

extern crate alloc;

use alloc::vec::Vec;

use super::error::F2fsError;
use super::F2FS_BLOCK_SIZE;

/// Magic number stamped at the top of the journal block (`'F2JL'`).
pub const F2FS_JOURNAL_MAGIC: u32 = 0x4C_4A_32_46;
/// Journal-format version (Phase 7 = 1).
pub const F2FS_JOURNAL_VERSION: u8 = 1;
/// Bytes per journal entry.
pub const JOURNAL_ENTRY_BYTES: usize = 24;
/// Maximum entries per journal block (4096 - 8 header) / 24.
pub const MAX_JOURNAL_ENTRIES: usize = (F2FS_BLOCK_SIZE as usize - 8) / JOURNAL_ENTRY_BYTES;

/// Discriminator for the journal record kind.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalKind {
    /// Empty/sentinel.
    None = 0,
    /// In-flight rename(src, dst).
    Rename = 1,
}

impl JournalKind {
    fn from_raw(b: u8) -> Self {
        match b {
            1 => Self::Rename,
            _ => Self::None,
        }
    }
}

/// One in-flight rename record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenameRecord {
    /// Inode of the parent directory holding `src`.
    pub src_parent_nid: u32,
    /// Inode of the parent directory holding `dst`.
    pub dst_parent_nid: u32,
    /// The moved inode (the file/dir being renamed).
    pub moved_nid: u32,
    /// If a `dst` entry already existed, the inode it pointed to (so
    /// recovery can decide whether to reclaim it). `0` if `dst` did not
    /// exist before the rename.
    pub dst_old_nid: u32,
    /// Bit 0: 1 = rename committed (dst now points at moved_nid),
    /// 0 = rename pending (src still owns moved_nid).
    pub flags: u8,
}

/// One journal entry (currently only `Rename`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalEntry {
    /// Empty slot.
    None,
    /// In-flight rename.
    Rename(RenameRecord),
}

impl JournalEntry {
    /// Encode the entry into [`JOURNAL_ENTRY_BYTES`] bytes.
    pub fn encode(&self, out: &mut [u8; JOURNAL_ENTRY_BYTES]) {
        out.fill(0);
        match self {
            Self::None => out[0] = JournalKind::None as u8,
            Self::Rename(r) => {
                out[0] = JournalKind::Rename as u8;
                out[1] = r.flags;
                // 2 bytes reserved
                out[4..8].copy_from_slice(&r.src_parent_nid.to_le_bytes());
                out[8..12].copy_from_slice(&r.dst_parent_nid.to_le_bytes());
                out[12..16].copy_from_slice(&r.moved_nid.to_le_bytes());
                out[16..20].copy_from_slice(&r.dst_old_nid.to_le_bytes());
                // 4 trailing bytes reserved
            }
        }
    }

    /// Decode an entry from [`JOURNAL_ENTRY_BYTES`] bytes.
    pub fn decode(buf: &[u8; JOURNAL_ENTRY_BYTES]) -> Self {
        match JournalKind::from_raw(buf[0]) {
            JournalKind::None => Self::None,
            JournalKind::Rename => {
                let flags = buf[1];
                let src_parent_nid = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
                let dst_parent_nid = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
                let moved_nid = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
                let dst_old_nid = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
                Self::Rename(RenameRecord {
                    src_parent_nid,
                    dst_parent_nid,
                    moved_nid,
                    dst_old_nid,
                    flags,
                })
            }
        }
    }
}

/// Encode a slice of journal entries into a 4 KiB block.
///
/// Layout: `[magic(4)|ver(1)|count(1)|reserved(2)| entries...]`.
pub fn encode_journal_block(entries: &[JournalEntry]) -> [u8; F2FS_BLOCK_SIZE as usize] {
    let mut block = [0u8; F2FS_BLOCK_SIZE as usize];
    block[0..4].copy_from_slice(&F2FS_JOURNAL_MAGIC.to_le_bytes());
    block[4] = F2FS_JOURNAL_VERSION;
    let n = core::cmp::min(entries.len(), MAX_JOURNAL_ENTRIES);
    block[5] = n as u8;
    for (i, e) in entries.iter().take(n).enumerate() {
        let off = 8 + i * JOURNAL_ENTRY_BYTES;
        let mut buf = [0u8; JOURNAL_ENTRY_BYTES];
        e.encode(&mut buf);
        block[off..off + JOURNAL_ENTRY_BYTES].copy_from_slice(&buf);
    }
    block
}

/// Decode a 4 KiB journal block produced by [`encode_journal_block`].
///
/// Returns an empty list if the magic is wrong or the version is
/// unrecognized — the caller treats this as "no in-flight ops".
pub fn decode_journal_block(block: &[u8]) -> Result<Vec<JournalEntry>, F2fsError> {
    if block.len() < F2FS_BLOCK_SIZE as usize {
        return Err(F2fsError::CheckpointCrcMismatch);
    }
    let magic = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    if magic != F2FS_JOURNAL_MAGIC {
        return Ok(Vec::new());
    }
    if block[4] != F2FS_JOURNAL_VERSION {
        return Ok(Vec::new());
    }
    let n = block[5] as usize;
    if n > MAX_JOURNAL_ENTRIES {
        return Err(F2fsError::CheckpointCrcMismatch);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = 8 + i * JOURNAL_ENTRY_BYTES;
        let mut buf = [0u8; JOURNAL_ENTRY_BYTES];
        buf.copy_from_slice(&block[off..off + JOURNAL_ENTRY_BYTES]);
        out.push(JournalEntry::decode(&buf));
    }
    Ok(out)
}

/// Bit 0 of [`RenameRecord::flags`]: 1 = committed, 0 = pending.
pub const FLAG_RENAME_COMMITTED: u8 = 0x01;

#[cfg(test)]
mod tests {
    use super::*;

    fn rename_rec() -> RenameRecord {
        RenameRecord {
            src_parent_nid: 3,
            dst_parent_nid: 3,
            moved_nid: 42,
            dst_old_nid: 0,
            flags: 0,
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        let entries = alloc::vec![
            JournalEntry::Rename(rename_rec()),
            JournalEntry::None,
            JournalEntry::Rename(RenameRecord {
                moved_nid: 99,
                ..rename_rec()
            }),
        ];
        let block = encode_journal_block(&entries);
        let decoded = decode_journal_block(&block).unwrap();
        assert_eq!(decoded.len(), entries.len());
        for (a, b) in decoded.iter().zip(entries.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn empty_block_yields_empty_list() {
        let block = [0u8; F2FS_BLOCK_SIZE as usize];
        let decoded = decode_journal_block(&block).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn bad_magic_is_no_ops() {
        let mut block = [0u8; F2FS_BLOCK_SIZE as usize];
        block[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let decoded = decode_journal_block(&block).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn version_mismatch_treated_as_no_ops() {
        let mut block = encode_journal_block(&[JournalEntry::Rename(rename_rec())]);
        block[4] = 99;
        let decoded = decode_journal_block(&block).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn truncated_block_rejected() {
        let block = alloc::vec![0u8; 10];
        assert!(decode_journal_block(&block).is_err());
    }

    #[test]
    fn count_overflow_rejected() {
        let mut block = encode_journal_block(&[]);
        block[5] = 255;
        assert!(decode_journal_block(&block).is_err());
    }

    #[test]
    fn entry_kind_decode() {
        assert_eq!(JournalKind::from_raw(0), JournalKind::None);
        assert_eq!(JournalKind::from_raw(1), JournalKind::Rename);
        assert_eq!(JournalKind::from_raw(99), JournalKind::None);
    }

    #[test]
    fn flag_committed_round_trip() {
        let rec = RenameRecord {
            flags: FLAG_RENAME_COMMITTED,
            ..rename_rec()
        };
        let entries = [JournalEntry::Rename(rec)];
        let block = encode_journal_block(&entries);
        let decoded = decode_journal_block(&block).unwrap();
        match decoded[0] {
            JournalEntry::Rename(r) => assert_eq!(r.flags, FLAG_RENAME_COMMITTED),
            _ => panic!(),
        }
    }
}
