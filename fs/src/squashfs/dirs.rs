// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Squashfs 4.0 directory-table reader.
//!
//! The directory table stores the body of every directory inode as a
//! sequence of *headers* and per-entry records:
//!
//! ```text
//!   Header (12 bytes):
//!     u32 count              // entries that follow share start_block
//!     u32 start_block        // metadata-block byte offset within inode table
//!     u32 inode_number_base
//!   Entry:
//!     u16 offset_in_block    // relative to start_block
//!     i16 inode_number_delta // signed; add to inode_number_base
//!     u16 inode_kind         // see inodes::INODE_KIND_*
//!     u16 name_size          // bytes - 1
//!     [u8; name_size+1] name
//! ```
//!
//! `count` here is "entries - 1" — i.e. there is always at least one
//! entry per header. We adjust for that on read.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::metadata::Cursor;
use super::SquashfsError;

/// One parsed directory entry.
#[derive(Debug, Clone)]
pub struct ParsedDirEntry {
    /// Inode reference high 48 bits = `start_block`, low 16 bits =
    /// `offset_in_block`. Use [`super::metadata::Cursor::seek_ref`] to
    /// land on the inode.
    pub inode_ref: u64,
    pub inode_number: u32,
    pub kind: u16,
    pub name: String,
}

/// Parse all directory entries up to `total_bytes` from the directory
/// table. The cursor MUST be positioned at the start of the
/// directory's body (per `BasicDirInode::start_block` /
/// `BasicDirInode::offset`).
pub fn parse_directory(
    cursor: &mut Cursor<'_>,
    total_bytes: u32,
) -> Result<Vec<ParsedDirEntry>, SquashfsError> {
    let mut out: Vec<ParsedDirEntry> = Vec::new();
    let mut consumed: u32 = 0;
    while consumed < total_bytes {
        let count = cursor.read_u32()?;
        let start_block = cursor.read_u32()?;
        let inode_number_base = cursor.read_u32()?;
        consumed = consumed
            .checked_add(12)
            .ok_or(SquashfsError::SizeOverflow)?;
        // Per spec: `count` is "entries - 1".
        let real_count = (count as u64) + 1;
        for _ in 0..real_count {
            let offset_in_block = cursor.read_u16()?;
            let inode_number_delta = cursor.read_i16()?;
            let kind = cursor.read_u16()?;
            let name_size = cursor.read_u16()? as usize;
            let name_len = name_size + 1;
            consumed = consumed
                .checked_add(8 + name_len as u32)
                .ok_or(SquashfsError::SizeOverflow)?;
            let mut name_bytes = alloc::vec![0u8; name_len];
            cursor.read_exact(&mut name_bytes)?;
            let inode_ref = ((start_block as u64) << 16) | (offset_in_block as u64);
            let inode_number = (inode_number_base as i64 + inode_number_delta as i64) as u32;
            out.push(ParsedDirEntry {
                inode_ref,
                inode_number,
                kind,
                name: super::lossy_string(&name_bytes),
            });
            if consumed >= total_bytes {
                return Ok(out);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::squashfs::metadata::Cursor;
    use crate::squashfs::superblock::{Compression, Superblock};
    use alloc::vec::Vec;

    fn dummy_superblock() -> Superblock {
        Superblock {
            inode_count: 1,
            mod_time: 0,
            block_size: 4096,
            fragment_count: 0,
            compression: Compression::Gzip,
            block_log: 12,
            flags: 0,
            id_count: 0,
            version_major: 4,
            version_minor: 0,
            root_inode_ref: 0,
            bytes_used: 0,
            id_table_start: 0,
            xattr_table_start: u64::MAX,
            inode_table_start: 0,
            directory_table_start: 0,
            fragment_table_start: u64::MAX,
            lookup_table_start: u64::MAX,
        }
    }

    fn write_uncompressed_block(image: &mut Vec<u8>, payload: &[u8]) -> usize {
        let off = image.len();
        let hdr = (payload.len() as u16) | 0x8000;
        image.extend_from_slice(&hdr.to_le_bytes());
        image.extend_from_slice(payload);
        off
    }

    fn build_dir_blob(entries: &[(u32, u16, u16, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        // One header for all entries.
        let header_count: u32 = entries.len() as u32 - 1;
        let start_block: u32 = 0;
        let inode_number_base: u32 = 1;
        buf.extend_from_slice(&header_count.to_le_bytes());
        buf.extend_from_slice(&start_block.to_le_bytes());
        buf.extend_from_slice(&inode_number_base.to_le_bytes());
        for &(offset_in_block, kind, _which, name) in entries {
            buf.extend_from_slice(&(offset_in_block as u16).to_le_bytes());
            buf.extend_from_slice(&0i16.to_le_bytes());
            buf.extend_from_slice(&kind.to_le_bytes());
            buf.extend_from_slice(&((name.len() - 1) as u16).to_le_bytes());
            buf.extend_from_slice(name);
        }
        buf
    }

    #[test]
    fn parses_two_entries() {
        let entries = [
            (10u32, 2u16, 0u16, &b"file.bin"[..]),
            (200u32, 1u16, 0u16, &b"subdir"[..]),
        ];
        let blob = build_dir_blob(&entries);
        let mut img = Vec::new();
        write_uncompressed_block(&mut img, &blob);
        let sb = dummy_superblock();
        let mut c = Cursor::new(&img, &sb, 0).unwrap();
        let parsed = parse_directory(&mut c, blob.len() as u32).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "file.bin");
        assert_eq!(parsed[0].kind, 2);
        assert_eq!(parsed[1].name, "subdir");
        assert_eq!(parsed[1].kind, 1);
    }
}
