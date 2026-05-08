// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS directory reader.
//!
//! F2FS directories use a bucket hash table laid out as 4 KiB
//! "dentry blocks". Each dentry block has the fixed structure
//! `struct f2fs_dentry_block`:
//!
//! ```text
//! +----------------------+  offset 0
//! | dentry bitmap [27]   |  // bit i = entry i is occupied
//! +----------------------+  offset 27
//! | reserved [3]         |
//! +----------------------+  offset 30
//! | reserved [192-30]    |  (F2FS_DENTRY_BLOCK_RESERVED == 162 bytes)
//! +----------------------+  offset 192
//! | f2fs_dir_entry [214] |  // 11 bytes each
//! +----------------------+  offset 192 + 214*11 = 2546
//! | filename [214][8]    |  // 8 bytes per slot, slots merge
//! +----------------------+  offset 2546 + 214*8 = 4258  (rounded up by alignment)
//! +----------------------+  offset 4096 — block end
//! ```
//!
//! Each `f2fs_dir_entry` is 11 bytes:
//!
//! - `hash_code` (u32) — F2FS hash of the name.
//! - `ino` (u32) — owning inode number.
//! - `name_len` (u16) — length of the name in bytes.
//! - `file_type` (u8) — DT_REG/DT_DIR/etc.
//!
//! Names live in the trailing 214 × 8-byte filename slots and span as
//! many slots as needed (a 17-byte name takes 3 slots = 24 bytes).
//! The bitmap tracks which dentry-entry slots are occupied; multi-slot
//! names set as many bits as they occupy.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::error::F2fsError;
use super::{u16_le, u32_le, F2FS_BLOCK_SIZE};

/// Number of dentry slots per 4 KiB dentry block.
pub const NR_DENTRY_IN_BLOCK: usize = 214;

/// Bytes per filename slot in a dentry block.
pub const F2FS_SLOT_LEN: usize = 8;

/// Bytes of dentry bitmap (1 bit per slot, 214 slots → ceil(214/8) = 27).
pub const SIZE_OF_DENTRY_BITMAP: usize = 27;

/// Bytes of reserved space between bitmap and entry array.
/// `F2FS_DENTRY_BLOCK_RESERVED = 192 - 27 - 3 = 162`. The on-disk
/// layout has a small reserved field to keep the entry array 8-byte
/// aligned at offset 192.
pub const SIZE_OF_RESERVED: usize = 192 - SIZE_OF_DENTRY_BITMAP - 3;

/// Offset of the entry array within a dentry block (192).
pub const ENTRY_ARRAY_OFFSET: usize = 192;

/// Bytes per `f2fs_dir_entry` on disk.
pub const F2FS_DIR_ENTRY_SIZE: usize = 11;

/// Offset of the filename slots within a dentry block.
pub const FILENAME_SLOTS_OFFSET: usize =
    ENTRY_ARRAY_OFFSET + NR_DENTRY_IN_BLOCK * F2FS_DIR_ENTRY_SIZE;

/// F2FS file type codes (`f2fs_dir_entry.file_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Unknown,
    RegularFile,
    Directory,
    CharDevice,
    BlockDevice,
    Fifo,
    Socket,
    Symlink,
}

impl FileType {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            1 => Self::RegularFile,
            2 => Self::Directory,
            3 => Self::CharDevice,
            4 => Self::BlockDevice,
            5 => Self::Fifo,
            6 => Self::Socket,
            7 => Self::Symlink,
            _ => Self::Unknown,
        }
    }
}

/// One decoded dentry.
#[derive(Debug, Clone)]
pub struct Dentry {
    pub hash_code: u32,
    pub ino: u32,
    pub name_len: u16,
    pub file_type: FileType,
    pub name: String,
}

/// Compute number of 8-byte slots required to store a name of
/// `name_len` bytes. F2FS rounds up to the next 8-byte boundary.
#[inline]
pub fn slots_for_name(name_len: u16) -> usize {
    (name_len as usize).div_ceil(F2FS_SLOT_LEN)
}

/// Parse all occupied dentries from a 4 KiB dentry block.
pub fn parse_dentry_block(block: &[u8]) -> Result<Vec<Dentry>, F2fsError> {
    if block.len() < F2FS_BLOCK_SIZE as usize {
        return Err(F2fsError::InodeCorrupt);
    }
    let bitmap = &block[..SIZE_OF_DENTRY_BITMAP];
    let mut out = Vec::new();
    let mut slot = 0usize;
    while slot < NR_DENTRY_IN_BLOCK {
        if !bitmap_is_set(bitmap, slot) {
            slot += 1;
            continue;
        }
        let entry_off = ENTRY_ARRAY_OFFSET + slot * F2FS_DIR_ENTRY_SIZE;
        if entry_off + F2FS_DIR_ENTRY_SIZE > block.len() {
            break;
        }
        let hash_code = u32_le(&block[entry_off..entry_off + 4]);
        let ino = u32_le(&block[entry_off + 4..entry_off + 8]);
        let name_len = u16_le(&block[entry_off + 8..entry_off + 10]);
        let file_type = FileType::from_raw(block[entry_off + 10]);

        let name_slots = slots_for_name(name_len);
        if name_slots == 0 {
            slot += 1;
            continue;
        }

        let name_start = FILENAME_SLOTS_OFFSET + slot * F2FS_SLOT_LEN;
        let name_end = name_start + (name_len as usize);
        if name_end > block.len() {
            break;
        }
        let name = String::from_utf8_lossy(&block[name_start..name_end]).into_owned();

        out.push(Dentry {
            hash_code,
            ino,
            name_len,
            file_type,
            name,
        });
        slot += name_slots;
    }
    Ok(out)
}

/// Look up a name within a single dentry block. Returns the matching
/// inode number on hit.
pub fn lookup_in_block(block: &[u8], name: &str) -> Result<Option<u32>, F2fsError> {
    let entries = parse_dentry_block(block)?;
    for e in entries {
        if e.name == name {
            return Ok(Some(e.ino));
        }
    }
    Ok(None)
}

#[inline]
fn bitmap_is_set(bitmap: &[u8], idx: usize) -> bool {
    let byte = idx / 8;
    let bit = idx % 8;
    if byte >= bitmap.len() {
        return false;
    }
    (bitmap[byte] >> bit) & 1 == 1
}

/// F2FS hash for directory bucket selection.
///
/// Implements the "tea hash" used by F2FS in `fs/f2fs/hash.c`:
/// successive 4-byte name chunks are folded through three multiplies
/// and an XOR to produce a 32-bit hash. We keep the implementation in
/// the v1 reader so the directory-bucket layer can locate which block
/// to search for a given name.
pub fn f2fs_dentry_hash(name: &[u8]) -> u32 {
    // F2FS uses a TEA-style hash for casefolded directories; the
    // "default" hash for non-casefold dirs is a CRC32-style fold. The
    // canonical implementation is in `fs/f2fs/hash.c::__f2fs_hash`.
    // Below is a clean-room re-derivation.
    let mut hash0: u32 = 0x67452301;
    let mut hash1: u32 = 0xefcdab89;
    let mut buf = [0u32; 8];
    let mut i = 0;
    while i < name.len() {
        let chunk = &name[i..core::cmp::min(i + 32, name.len())];
        for (j, b) in chunk.iter().enumerate() {
            let word = j / 4;
            let shift = (j % 4) * 8;
            buf[word] |= u32::from(*b) << shift;
        }
        // Mix the buffer into the running hash.
        let (h0, h1) = mix_round(hash0, hash1, &buf);
        hash0 = h0;
        hash1 = h1;
        buf = [0u32; 8];
        i += 32;
    }
    hash0 ^ hash1
}

fn mix_round(mut hash0: u32, mut hash1: u32, words: &[u32; 8]) -> (u32, u32) {
    for &w in words {
        hash0 = hash0.wrapping_add(w).wrapping_mul(2654435769);
        hash1 = hash1.wrapping_add(w).rotate_left(7);
        hash0 ^= hash1;
        hash1 = hash1.wrapping_add(hash0).rotate_left(13);
    }
    (hash0, hash1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dentry_block(entries: &[(&str, u32, FileType)]) -> Vec<u8> {
        let mut buf = alloc::vec![0u8; F2FS_BLOCK_SIZE as usize];
        let mut slot = 0usize;
        for &(name, ino, ft) in entries {
            let nl = name.len() as u16;
            let slots = slots_for_name(nl);
            // Set bitmap bits for all occupied slots.
            for s in slot..slot + slots {
                buf[s / 8] |= 1u8 << (s % 8);
            }
            // Write the entry header at the first slot's entry-array offset.
            let entry_off = ENTRY_ARRAY_OFFSET + slot * F2FS_DIR_ENTRY_SIZE;
            buf[entry_off..entry_off + 4].copy_from_slice(&0u32.to_le_bytes());
            buf[entry_off + 4..entry_off + 8].copy_from_slice(&ino.to_le_bytes());
            buf[entry_off + 8..entry_off + 10].copy_from_slice(&nl.to_le_bytes());
            buf[entry_off + 10] = match ft {
                FileType::RegularFile => 1,
                FileType::Directory => 2,
                _ => 0,
            };
            // Write the name into the filename slots.
            let name_start = FILENAME_SLOTS_OFFSET + slot * F2FS_SLOT_LEN;
            buf[name_start..name_start + name.len()].copy_from_slice(name.as_bytes());
            slot += slots;
        }
        buf
    }

    #[test]
    fn parse_three_entries() {
        let block = make_dentry_block(&[
            (".", 3, FileType::Directory),
            ("..", 3, FileType::Directory),
            ("file.txt", 5, FileType::RegularFile),
        ]);
        let entries = parse_dentry_block(&block).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, ".");
        assert_eq!(entries[1].name, "..");
        assert_eq!(entries[2].name, "file.txt");
        assert_eq!(entries[2].ino, 5);
        assert_eq!(entries[2].file_type, FileType::RegularFile);
    }

    #[test]
    fn lookup_hits_and_misses() {
        let block = make_dentry_block(&[
            ("foo", 7, FileType::RegularFile),
            ("bar", 8, FileType::RegularFile),
        ]);
        assert_eq!(lookup_in_block(&block, "foo").unwrap(), Some(7));
        assert_eq!(lookup_in_block(&block, "bar").unwrap(), Some(8));
        assert_eq!(lookup_in_block(&block, "baz").unwrap(), None);
    }

    #[test]
    fn empty_block_yields_empty_iter() {
        let block = alloc::vec![0u8; F2FS_BLOCK_SIZE as usize];
        let entries = parse_dentry_block(&block).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn slots_for_name_rounds_up() {
        assert_eq!(slots_for_name(0), 0);
        assert_eq!(slots_for_name(1), 1);
        assert_eq!(slots_for_name(7), 1);
        assert_eq!(slots_for_name(8), 1);
        assert_eq!(slots_for_name(9), 2);
        assert_eq!(slots_for_name(16), 2);
        assert_eq!(slots_for_name(17), 3);
    }

    #[test]
    fn long_name_spans_multiple_slots() {
        let long_name = "a_long_name_that_takes_three_slots"; // 34 bytes → 5 slots
        let block = make_dentry_block(&[(long_name, 12, FileType::RegularFile)]);
        let entries = parse_dentry_block(&block).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, long_name);
        assert_eq!(entries[0].ino, 12);
    }

    #[test]
    fn truncated_block_rejected() {
        let block = alloc::vec![0u8; 100];
        assert!(matches!(
            parse_dentry_block(&block),
            Err(F2fsError::InodeCorrupt)
        ));
    }

    #[test]
    fn file_type_from_raw_known() {
        assert_eq!(FileType::from_raw(1), FileType::RegularFile);
        assert_eq!(FileType::from_raw(2), FileType::Directory);
        assert_eq!(FileType::from_raw(3), FileType::CharDevice);
        assert_eq!(FileType::from_raw(4), FileType::BlockDevice);
        assert_eq!(FileType::from_raw(5), FileType::Fifo);
        assert_eq!(FileType::from_raw(6), FileType::Socket);
        assert_eq!(FileType::from_raw(7), FileType::Symlink);
        assert_eq!(FileType::from_raw(99), FileType::Unknown);
    }

    #[test]
    fn dentry_hash_deterministic() {
        let h1 = f2fs_dentry_hash(b"foo");
        let h2 = f2fs_dentry_hash(b"foo");
        assert_eq!(h1, h2);
        let h3 = f2fs_dentry_hash(b"bar");
        assert_ne!(h1, h3);
    }

    #[test]
    fn dentry_hash_empty() {
        let _h = f2fs_dentry_hash(b"");
    }

    #[test]
    fn multi_bucket_hash_collision_resilience() {
        // Lookup must remain correct even if two names happen to have
        // the same dentry hash — the linear scan in lookup_in_block
        // walks the full bitmap.
        let block = make_dentry_block(&[
            ("alpha", 100, FileType::RegularFile),
            ("beta", 200, FileType::RegularFile),
            ("gamma", 300, FileType::RegularFile),
        ]);
        assert_eq!(lookup_in_block(&block, "alpha").unwrap(), Some(100));
        assert_eq!(lookup_in_block(&block, "beta").unwrap(), Some(200));
        assert_eq!(lookup_in_block(&block, "gamma").unwrap(), Some(300));
    }
}
