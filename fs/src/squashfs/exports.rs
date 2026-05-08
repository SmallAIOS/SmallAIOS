// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Squashfs 4.0 export-table reader (NFS handles).
//!
//! Optional table referenced by `superblock.lookup_table_start`. When
//! the `EXPORTABLE` flag is set, this table is an array of u64 inode
//! references indexed by `inode_number - 1`. SmallAIOS does not
//! re-export squashfs over NFS today — the reader supports the table
//! purely for forward compatibility and conformance.

extern crate alloc;

use alloc::vec::Vec;

use super::metadata::read_metadata_block;
use super::superblock::Superblock;
use super::SquashfsError;

/// Number of u64 export entries that fit in one 8 KiB metadata block.
pub const EXPORTS_PER_BLOCK: usize = 1024;

/// Read the entire export table.
pub fn read_export_table(image: &[u8], sb: &Superblock) -> Result<Vec<u64>, SquashfsError> {
    if !sb.has_exports() {
        return Ok(Vec::new());
    }
    let count = sb.inode_count as usize;
    let top_entries = count.div_ceil(EXPORTS_PER_BLOCK);
    let top_start =
        usize::try_from(sb.lookup_table_start).map_err(|_| SquashfsError::SizeOverflow)?;
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
        while p + 8 <= payload.len() && out.len() < count {
            let r = u64::from_le_bytes(payload[p..p + 8].try_into().unwrap());
            out.push(r);
            p += 8;
        }
    }
    Ok(out)
}
