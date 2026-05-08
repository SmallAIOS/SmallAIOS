// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Squashfs 4.0 fragment-table reader.
//!
//! The fragment table records the location and size of every "tail
//! fragment" — the last <block_size partial block of every regular
//! file is packed into a shared fragment block to save space. Each
//! fragment entry is 16 bytes:
//!
//! ```text
//!   u64 start          // absolute byte offset in the squashfs image
//!   u32 size           // high bit = "stored uncompressed"
//!   u32 unused
//! ```
//!
//! The fragment table is a two-level structure:
//!
//! - The "top" table at `superblock.fragment_table_start` is an array
//!   of `u64` entries. Each entry is the absolute byte offset of one
//!   metadata block whose decompressed payload contains the actual
//!   fragment entries. The top table count is
//!   `ceil(fragment_count / fragments_per_metadata_block)` where
//!   `fragments_per_metadata_block = 8192 / 16 = 512`.
//!
//! `read_fragment_table` resolves the chain and returns a flat
//! `Vec<FragmentEntry>` of length `fragment_count`.

extern crate alloc;

use alloc::vec::Vec;

use super::metadata::read_metadata_block;
use super::superblock::Superblock;
use super::SquashfsError;

/// Number of fragment entries that fit in one 8 KiB metadata block.
pub const FRAGMENTS_PER_BLOCK: usize = 512; // 8192 / 16

/// On-disk fragment-entry size.
pub const FRAGMENT_ENTRY_BYTES: usize = 16;

/// One parsed fragment entry.
#[derive(Debug, Clone)]
pub struct FragmentEntry {
    /// Absolute byte offset of the fragment block in the image.
    pub start: u64,
    /// Compressed/raw size with high bit indicating "stored
    /// uncompressed".
    pub size: u32,
}

impl FragmentEntry {
    /// Number of bytes the on-disk fragment block occupies (low 24
    /// bits of `size`).
    pub fn on_disk_size(&self) -> u32 {
        self.size & 0x00FF_FFFF
    }

    /// True iff the fragment block was stored uncompressed.
    pub fn is_uncompressed(&self) -> bool {
        self.size & 0x0100_0000 != 0
    }
}

/// Read the entire fragment table.
pub fn read_fragment_table(
    image: &[u8],
    sb: &Superblock,
) -> Result<Vec<FragmentEntry>, SquashfsError> {
    if !sb.has_fragments() {
        return Ok(Vec::new());
    }
    let count = sb.fragment_count as usize;
    let top_entries = count.div_ceil(FRAGMENTS_PER_BLOCK);
    let top_table_start =
        usize::try_from(sb.fragment_table_start).map_err(|_| SquashfsError::SizeOverflow)?;
    if top_table_start + top_entries * 8 > image.len() {
        return Err(SquashfsError::OutOfRange);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..top_entries {
        let off_pos = top_table_start + i * 8;
        let block_offset = u64::from_le_bytes(image[off_pos..off_pos + 8].try_into().unwrap());
        let block_offset =
            usize::try_from(block_offset).map_err(|_| SquashfsError::SizeOverflow)?;
        let (payload, _consumed) = read_metadata_block(image, block_offset, sb)?;
        let mut p = 0usize;
        while p + FRAGMENT_ENTRY_BYTES <= payload.len() && out.len() < count {
            let start = u64::from_le_bytes(payload[p..p + 8].try_into().unwrap());
            let size = u32::from_le_bytes(payload[p + 8..p + 12].try_into().unwrap());
            // Last 4 bytes are unused.
            p += FRAGMENT_ENTRY_BYTES;
            out.push(FragmentEntry { start, size });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::squashfs::superblock::{Compression, Superblock};

    fn dummy_superblock(fragment_count: u32, fragment_table_start: u64) -> Superblock {
        Superblock {
            inode_count: 1,
            mod_time: 0,
            block_size: 4096,
            fragment_count,
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
            fragment_table_start,
            lookup_table_start: u64::MAX,
        }
    }

    #[test]
    fn empty_when_no_fragments() {
        let sb = dummy_superblock(0, u64::MAX);
        let v = read_fragment_table(&[], &sb).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn reads_one_fragment() {
        // Fragment metadata-block payload: 16 bytes.
        let mut frag_payload = Vec::new();
        frag_payload.extend_from_slice(&0x1000u64.to_le_bytes()); // start
        frag_payload.extend_from_slice(&0x0080u32.to_le_bytes()); // size
        frag_payload.extend_from_slice(&[0u8; 4]); // unused
                                                   // Wrap in a metadata block with the "uncompressed" flag.
        let mut meta_block = Vec::new();
        let hdr = (frag_payload.len() as u16) | 0x8000;
        meta_block.extend_from_slice(&hdr.to_le_bytes());
        meta_block.extend_from_slice(&frag_payload);
        // Image: [meta_block][top_table (one u64 = 0)].
        let mut image = Vec::new();
        image.extend_from_slice(&meta_block);
        let top_start = image.len() as u64;
        // Top table entry: byte offset of the metadata block.
        image.extend_from_slice(&0u64.to_le_bytes());
        let sb = dummy_superblock(1, top_start);
        let v = read_fragment_table(&image, &sb).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].start, 0x1000);
        assert_eq!(v[0].size, 0x0080);
    }
}
