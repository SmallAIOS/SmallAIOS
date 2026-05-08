// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Squashfs 4.0 UID/GID lookup-table reader.
//!
//! Inodes store UID/GID as 16-bit indices into a single shared table
//! of `u32` values. Each table entry is 4 bytes; up to
//! `8192 / 4 = 2048` entries fit in one metadata block.
//!
//! Two-level layout: top table at `id_table_start` is an array of
//! `u64` byte offsets to the metadata blocks holding the actual IDs.

extern crate alloc;

use alloc::vec::Vec;

use super::metadata::read_metadata_block;
use super::superblock::Superblock;
use super::SquashfsError;

/// Number of `u32` ID entries that fit in one 8 KiB metadata block.
pub const IDS_PER_BLOCK: usize = 2048;

/// Read the entire UID/GID table.
pub fn read_id_table(image: &[u8], sb: &Superblock) -> Result<Vec<u32>, SquashfsError> {
    let count = sb.id_count as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    let top_entries = count.div_ceil(IDS_PER_BLOCK);
    let top_start = usize::try_from(sb.id_table_start).map_err(|_| SquashfsError::SizeOverflow)?;
    if top_start + top_entries * 8 > image.len() {
        return Err(SquashfsError::OutOfRange);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..top_entries {
        let off_pos = top_start + i * 8;
        let block_offset = u64::from_le_bytes(image[off_pos..off_pos + 8].try_into().unwrap());
        let block_offset =
            usize::try_from(block_offset).map_err(|_| SquashfsError::SizeOverflow)?;
        let (payload, _consumed) = read_metadata_block(image, block_offset, sb)?;
        let mut p = 0usize;
        while p + 4 <= payload.len() && out.len() < count {
            let id = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap());
            out.push(id);
            p += 4;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::squashfs::superblock::{Compression, Superblock};

    fn dummy_superblock(id_count: u16, id_table_start: u64) -> Superblock {
        Superblock {
            inode_count: 1,
            mod_time: 0,
            block_size: 4096,
            fragment_count: 0,
            compression: Compression::Gzip,
            block_log: 12,
            flags: 0,
            id_count,
            version_major: 4,
            version_minor: 0,
            root_inode_ref: 0,
            bytes_used: 0,
            id_table_start,
            xattr_table_start: u64::MAX,
            inode_table_start: 0,
            directory_table_start: 0,
            fragment_table_start: u64::MAX,
            lookup_table_start: u64::MAX,
        }
    }

    #[test]
    fn empty_when_no_ids() {
        let sb = dummy_superblock(0, 0);
        assert!(read_id_table(&[], &sb).unwrap().is_empty());
    }

    #[test]
    fn reads_three_ids() {
        let mut payload = Vec::new();
        for &id in &[1000u32, 2000, 3000] {
            payload.extend_from_slice(&id.to_le_bytes());
        }
        let mut image = Vec::new();
        let hdr = (payload.len() as u16) | 0x8000;
        image.extend_from_slice(&hdr.to_le_bytes());
        image.extend_from_slice(&payload);
        let top_start = image.len() as u64;
        image.extend_from_slice(&0u64.to_le_bytes());
        let sb = dummy_superblock(3, top_start);
        let v = read_id_table(&image, &sb).unwrap();
        assert_eq!(v, alloc::vec![1000u32, 2000, 3000]);
    }
}
