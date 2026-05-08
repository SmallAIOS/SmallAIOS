// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Squashfs 4.0 inode-table reader.
//!
//! The inode table is a sequence of metadata blocks that store all
//! file/directory/symlink/device-file entries. Each inode starts with
//! a 16-byte common header (`InodeHeader`) followed by an inode-kind-
//! specific body. We currently implement the Basic Directory, Basic
//! File, Extended Directory, Extended File, and Basic Symlink kinds —
//! the five kinds that mksquashfs(1) actually emits for typical model
//! payloads. Block / character / FIFO / socket inodes are parsed only
//! to the extent needed to walk past them; SmallAIOS does not surface
//! them as files.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::metadata::Cursor;
use super::superblock::Superblock;
use super::SquashfsError;

/// Inode kinds defined by squashfs 4.0 (`fs/squashfs/squashfs_fs.h`).
pub const INODE_KIND_BASIC_DIR: u16 = 1;
pub const INODE_KIND_BASIC_FILE: u16 = 2;
pub const INODE_KIND_BASIC_SYMLINK: u16 = 3;
pub const INODE_KIND_BASIC_BLOCK: u16 = 4;
pub const INODE_KIND_BASIC_CHAR: u16 = 5;
pub const INODE_KIND_BASIC_FIFO: u16 = 6;
pub const INODE_KIND_BASIC_SOCKET: u16 = 7;
pub const INODE_KIND_EXT_DIR: u16 = 8;
pub const INODE_KIND_EXT_FILE: u16 = 9;
pub const INODE_KIND_EXT_SYMLINK: u16 = 10;
pub const INODE_KIND_EXT_BLOCK: u16 = 11;
pub const INODE_KIND_EXT_CHAR: u16 = 12;
pub const INODE_KIND_EXT_FIFO: u16 = 13;
pub const INODE_KIND_EXT_SOCKET: u16 = 14;

/// Common 16-byte header at the start of every inode.
#[derive(Debug, Clone)]
pub struct InodeHeader {
    pub kind: u16,
    pub permissions: u16,
    pub uid_idx: u16,
    pub gid_idx: u16,
    pub mtime: u32,
    pub inode_number: u32,
}

impl InodeHeader {
    pub fn read(cursor: &mut Cursor<'_>) -> Result<Self, SquashfsError> {
        let kind = cursor.read_u16()?;
        let permissions = cursor.read_u16()?;
        let uid_idx = cursor.read_u16()?;
        let gid_idx = cursor.read_u16()?;
        let mtime = cursor.read_u32()?;
        let inode_number = cursor.read_u32()?;
        Ok(Self {
            kind,
            permissions,
            uid_idx,
            gid_idx,
            mtime,
            inode_number,
        })
    }
}

/// Squashfs Basic Directory inode body (after the 16-byte header).
#[derive(Debug, Clone)]
pub struct BasicDirInode {
    pub start_block: u32,
    pub nlink: u32,
    /// Size of the directory entries in the directory table (3
    /// fewer than the on-disk value per squashfs 4.0 quirk;
    /// already adjusted here).
    pub size: u32,
    pub offset: u16,
    pub parent_inode: u32,
}

impl BasicDirInode {
    pub fn read(cursor: &mut Cursor<'_>) -> Result<Self, SquashfsError> {
        let start_block = cursor.read_u32()?;
        let nlink = cursor.read_u32()?;
        let size_raw = cursor.read_u16()? as u32;
        let offset = cursor.read_u16()?;
        let parent_inode = cursor.read_u32()?;
        // Per the squashfs spec, `size_raw` is "real size + 3"; subtract
        // here so callers get the actual byte length.
        let size = size_raw.saturating_sub(3);
        Ok(Self {
            start_block,
            nlink,
            size,
            offset,
            parent_inode,
        })
    }
}

/// Squashfs Extended Directory inode body.
#[derive(Debug, Clone)]
pub struct ExtDirInode {
    pub nlink: u32,
    pub size: u32,
    pub start_block: u32,
    pub parent_inode: u32,
    pub i_count: u16,
    pub offset: u16,
    pub xattr_idx: u32,
    /// Index entries: each is `index, start_block, name_size, name_bytes`.
    /// We don't need them for opening files; provide raw bytes for
    /// callers that want to do range reads on big directories.
    pub index_blob: Vec<u8>,
}

impl ExtDirInode {
    pub fn read(cursor: &mut Cursor<'_>) -> Result<Self, SquashfsError> {
        let nlink = cursor.read_u32()?;
        let size = cursor.read_u32()?;
        let start_block = cursor.read_u32()?;
        let parent_inode = cursor.read_u32()?;
        let i_count = cursor.read_u16()?;
        let offset = cursor.read_u16()?;
        let xattr_idx = cursor.read_u32()?;
        // Index entries follow. Each is 12 bytes plus a name_size byte
        // count of name bytes. We read but discard.
        let mut blob: Vec<u8> = Vec::new();
        for _ in 0..i_count {
            let mut hdr = [0u8; 12];
            cursor.read_exact(&mut hdr)?;
            let name_size = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize + 1;
            blob.extend_from_slice(&hdr);
            let mut name = alloc::vec![0u8; name_size];
            cursor.read_exact(&mut name)?;
            blob.extend_from_slice(&name);
        }
        Ok(Self {
            nlink,
            size,
            start_block,
            parent_inode,
            i_count,
            offset,
            xattr_idx,
            index_blob: blob,
        })
    }
}

/// Squashfs Basic File inode body.
#[derive(Debug, Clone)]
pub struct BasicFileInode {
    pub blocks_start: u32,
    pub frag_idx: u32,
    pub frag_offset: u32,
    pub file_size: u32,
    /// Compressed sizes of each full block. The high bit of each entry
    /// is the "stored uncompressed" flag (squashfs's standard
    /// convention).
    pub block_sizes: Vec<u32>,
}

impl BasicFileInode {
    pub fn read(cursor: &mut Cursor<'_>, sb: &Superblock) -> Result<Self, SquashfsError> {
        let blocks_start = cursor.read_u32()?;
        let frag_idx = cursor.read_u32()?;
        let frag_offset = cursor.read_u32()?;
        let file_size = cursor.read_u32()?;
        let nblocks = file_block_count(file_size as u64, sb.block_size as u64, frag_idx);
        let mut block_sizes = Vec::with_capacity(nblocks);
        for _ in 0..nblocks {
            block_sizes.push(cursor.read_u32()?);
        }
        Ok(Self {
            blocks_start,
            frag_idx,
            frag_offset,
            file_size,
            block_sizes,
        })
    }
}

/// Squashfs Extended File inode body.
#[derive(Debug, Clone)]
pub struct ExtFileInode {
    pub blocks_start: u64,
    pub file_size: u64,
    pub sparse: u64,
    pub nlink: u32,
    pub frag_idx: u32,
    pub frag_offset: u32,
    pub xattr_idx: u32,
    pub block_sizes: Vec<u32>,
}

impl ExtFileInode {
    pub fn read(cursor: &mut Cursor<'_>, sb: &Superblock) -> Result<Self, SquashfsError> {
        let blocks_start = cursor.read_u64()?;
        let file_size = cursor.read_u64()?;
        let sparse = cursor.read_u64()?;
        let nlink = cursor.read_u32()?;
        let frag_idx = cursor.read_u32()?;
        let frag_offset = cursor.read_u32()?;
        let xattr_idx = cursor.read_u32()?;
        let nblocks = file_block_count(file_size, sb.block_size as u64, frag_idx);
        let mut block_sizes = Vec::with_capacity(nblocks);
        for _ in 0..nblocks {
            block_sizes.push(cursor.read_u32()?);
        }
        Ok(Self {
            blocks_start,
            file_size,
            sparse,
            nlink,
            frag_idx,
            frag_offset,
            xattr_idx,
            block_sizes,
        })
    }
}

/// Squashfs Basic Symlink inode body.
#[derive(Debug, Clone)]
pub struct BasicSymlinkInode {
    pub nlink: u32,
    pub target: String,
}

impl BasicSymlinkInode {
    pub fn read(cursor: &mut Cursor<'_>) -> Result<Self, SquashfsError> {
        let nlink = cursor.read_u32()?;
        let target_size = cursor.read_u32()? as usize;
        let mut target_bytes = alloc::vec![0u8; target_size];
        cursor.read_exact(&mut target_bytes)?;
        Ok(Self {
            nlink,
            target: super::lossy_string(&target_bytes),
        })
    }
}

/// Number of full data blocks in a file with `file_size` bytes.
///
/// If `frag_idx == 0xFFFFFFFF` the file has no fragment, so the count
/// is `ceil(file_size / block_size)`. Otherwise the trailing partial
/// block is stored as a fragment, so we use `file_size / block_size`
/// (floor).
fn file_block_count(file_size: u64, block_size: u64, frag_idx: u32) -> usize {
    if file_size == 0 {
        return 0;
    }
    let q = file_size / block_size;
    let r = file_size % block_size;
    if frag_idx == 0xFFFF_FFFF {
        if r == 0 {
            q as usize
        } else {
            (q + 1) as usize
        }
    } else {
        q as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_count_no_fragment() {
        assert_eq!(file_block_count(0, 4096, 0xFFFF_FFFF), 0);
        assert_eq!(file_block_count(1, 4096, 0xFFFF_FFFF), 1);
        assert_eq!(file_block_count(4096, 4096, 0xFFFF_FFFF), 1);
        assert_eq!(file_block_count(4097, 4096, 0xFFFF_FFFF), 2);
    }

    #[test]
    fn block_count_with_fragment() {
        assert_eq!(file_block_count(0, 4096, 0), 0);
        assert_eq!(file_block_count(1, 4096, 0), 0);
        assert_eq!(file_block_count(4095, 4096, 0), 0);
        assert_eq!(file_block_count(4096, 4096, 0), 1);
        assert_eq!(file_block_count(4097, 4096, 0), 1);
    }
}
