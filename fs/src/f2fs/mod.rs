// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS read-only reader (Linux 6.6 LTS feature set).
//!
//! Implements a clean-room, `#![no_std]` Rust reader for the F2FS
//! on-disk format produced by `mkfs.f2fs` from the f2fs-tools release
//! that matches Linux 6.6 LTS. The Linux kernel sources at
//! `fs/f2fs/f2fs.h` and the F2FS technical paper
//! (<https://www.usenix.org/conference/fast15/technical-sessions/presentation/lee>)
//! are the authoritative references; every constant and layout in this
//! module is pinned to those public format-spec documents.
//!
//! ## Scope (Phase 5 — read path only)
//!
//! - Superblock parsing with primary/secondary fallback.
//! - Checkpoint pack reader picking the highest valid version.
//! - SIT (Segment Information Table) reader with on-demand block load.
//! - NAT (Node Address Table) reader, two-pair format.
//! - SSA (Segment Summary Area) reader.
//! - Inode parser: size, mode, mtime/ctime/atime, inline data,
//!   extents, single-indirect block pointers (double/triple deferred).
//! - Directory bucket-hash lookup + iterator.
//! - Data-block read path: inode extent map + NAT lookup → physical
//!   LBA → `BlockDevice::read_block`.
//! - Top-level [`mount::F2fs`] API: `mount`, `open_path`, `read_file`,
//!   `read_dir`, `stat`.
//!
//! Write path, fsync/checkpoint commit, garbage collection, and the
//! `/data/` VFS mount point are deferred to Phases 6-9 of the
//! `embedded-filesystem-v1` change.
//!
//! ## On-disk geometry
//!
//! F2FS partitions are laid out as:
//!
//! ```text
//! +----------+-------+--------+------+------+--------+
//! | superblk | CP    | SIT    | NAT  | SSA  | main   |
//! | (2 SBs)  | pack  | area   | area | area | area   |
//! +----------+-------+--------+------+------+--------+
//! ```
//!
//! - Two superblock copies live in the first segment (offset 1024 and
//!   9216 within the partition's first 4 KiB-block-aligned region).
//! - The checkpoint pack is two copies (CP1, CP2) for crash safety;
//!   the reader picks the higher version with a valid CRC.
//! - The main area is divided into segments of 2 MiB (512 × 4 KiB).
//!
//! Owned by `embedded-filesystem-v1` Phase 5.

extern crate alloc;

use core::fmt;

use crate::block::BlockError;

pub mod checkpoint;
pub mod crc;
pub mod data;
pub mod dir;
pub mod error;
pub mod gc;
pub mod inode;
pub mod journal;
pub mod mount;
pub mod nat;
pub mod sit;
pub mod ssa;
pub mod superblock;
pub mod write;
#[cfg(test)]
pub(crate) mod write_test_vectors;

pub use error::F2fsError;
pub use mount::{DirIter, F2fs, Inode, InodeKind, Stat};
pub use superblock::{Superblock, F2FS_SUPER_MAGIC};
pub use write::{GcStats, WriteStats};

/// F2FS native block size (4 KiB).
///
/// F2FS blocks are always 4 KiB; the on-disk format hard-codes this.
/// All offsets in the on-disk structures are block addresses (multiply
/// by [`F2FS_BLOCK_SIZE`] to get a byte offset within the partition).
pub const F2FS_BLOCK_SIZE: u32 = 4096;

/// Number of 4 KiB blocks per F2FS segment (default 512 → 2 MiB).
pub const F2FS_BLOCKS_PER_SEG: u32 = 512;

/// F2FS segment size in bytes (2 MiB default).
pub const F2FS_SEGMENT_SIZE: u32 = F2FS_BLOCK_SIZE * F2FS_BLOCKS_PER_SEG;

/// Reserved root inode number. Linux F2FS hard-codes this value.
pub const F2FS_ROOT_INO: u32 = 3;

/// Reserved node-id-zero sentinel (means "no node").
pub const F2FS_NULL_NID: u32 = 0;

/// Read raw bytes from a partition by absolute byte offset (mirrors
/// the helper in `squashfs::read_partition_bytes`).
///
/// `partition_lba` and `partition_size_bytes` describe the partition's
/// position on the underlying device. Reads scratch one underlying
/// block at a time; the device's `block_size_bytes()` is used for I/O.
pub(crate) fn read_partition_bytes<D: crate::block::BlockDevice + ?Sized>(
    device: &D,
    partition_lba: u64,
    partition_size_bytes: u64,
    offset: u64,
    out: &mut [u8],
) -> Result<(), F2fsError> {
    let bs = device.block_size_bytes() as u64;
    if bs == 0 {
        return Err(F2fsError::IoError(BlockError::Unaligned));
    }
    let len = out.len() as u64;
    let end = offset.checked_add(len).ok_or(F2fsError::SizeOverflow)?;
    if end > partition_size_bytes {
        return Err(F2fsError::ReadPastEof);
    }

    let mut written: usize = 0;
    let mut cursor = offset;
    let mut remaining = len;
    let mut scratch: alloc::vec::Vec<u8> = alloc::vec![0u8; bs as usize];
    while remaining > 0 {
        let abs_byte = partition_lba.saturating_mul(bs) + cursor;
        let block = abs_byte / bs;
        let in_block_off = (abs_byte % bs) as usize;
        device
            .read_block(block, &mut scratch)
            .map_err(F2fsError::IoError)?;
        let take = core::cmp::min(remaining as usize, bs as usize - in_block_off);
        out[written..written + take].copy_from_slice(&scratch[in_block_off..in_block_off + take]);
        written += take;
        cursor += take as u64;
        remaining -= take as u64;
    }
    Ok(())
}

/// Read one F2FS 4 KiB block by its block-address (relative to the
/// partition start). Wraps [`read_partition_bytes`] for callers that
/// always want a full block.
pub(crate) fn read_f2fs_block<D: crate::block::BlockDevice + ?Sized>(
    device: &D,
    partition_lba: u64,
    partition_size_bytes: u64,
    block_addr: u32,
    out: &mut [u8; F2FS_BLOCK_SIZE as usize],
) -> Result<(), F2fsError> {
    let offset = (block_addr as u64) * F2FS_BLOCK_SIZE as u64;
    read_partition_bytes(device, partition_lba, partition_size_bytes, offset, out)
}

/// Little-endian u16 from a slice (`buf.len() >= 2` required).
#[inline]
pub(crate) fn u16_le(buf: &[u8]) -> u16 {
    u16::from_le_bytes([buf[0], buf[1]])
}

/// Little-endian u32 from a slice (`buf.len() >= 4` required).
#[inline]
pub(crate) fn u32_le(buf: &[u8]) -> u32 {
    u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
}

/// Little-endian u64 from a slice (`buf.len() >= 8` required).
#[inline]
pub(crate) fn u64_le(buf: &[u8]) -> u64 {
    u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ])
}

/// Display helper: hex-format a feature-bit mask for error messages.
pub(crate) fn feature_bit_name(bit: u64) -> &'static str {
    match bit {
        0x0001 => "encrypt",
        0x0002 => "blkzoned",
        0x0004 => "atomic_write",
        0x0008 => "extra_attr",
        0x0010 => "project_quota",
        0x0020 => "inode_checksum",
        0x0040 => "flexible_inline_xattr",
        0x0080 => "quota_ino",
        0x0100 => "inode_crtime",
        0x0200 => "lost_found",
        0x0400 => "verity",
        0x0800 => "sb_checksum",
        0x1000 => "casefold",
        0x2000 => "compression",
        0x4000 => "ro",
        _ => "unknown",
    }
}

/// Format a feature bit + name for log/audit messages.
pub struct FeatureBit(pub u64);

impl fmt::Display for FeatureBit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:04x} ({})", self.0, feature_bit_name(self.0))
    }
}
