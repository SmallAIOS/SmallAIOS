// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS data-block read path.
//!
//! Combines:
//!
//! 1. The inode's direct block-address array (`i_addr[]`) for file
//!    offsets `< ADDRS_PER_INODE * 4 KiB` (≈ 3.6 MiB).
//! 2. The first single-indirect node (`i_nid[2]` in Linux 6.6 — but
//!    direct-pointer slots `i_nid[0]` and `i_nid[1]` precede it; the
//!    v1 reader handles only direct + one level of single-indirect).
//! 3. The NAT lookup that translates a node-id to its physical block
//!    address.
//!
//! Files larger than the direct + single-indirect range surface
//! [`F2fsError::DeepIndirectionUnsupported`].
//!
//! The v1 read path consults the cache module if it exists; if the
//! `crate::cache` module is empty (Phase 6 not landed) we fall through
//! to direct `BlockDevice::read_block` calls.

extern crate alloc;

use super::error::F2fsError;
use super::inode::{Inode, ADDRS_PER_INODE, NIDS_PER_INODE};
use super::nat::{nid_to_entry_off, nid_to_nat_block_idx, parse_nat_block, NatEntry};
use super::superblock::Superblock;
use super::{read_f2fs_block, F2FS_BLOCK_SIZE, F2FS_NULL_NID};
use crate::block::BlockDevice;

/// Bytes of file data that fit through direct pointers in the inode.
pub const MAX_DIRECT_FILE_BYTES: u64 = (ADDRS_PER_INODE as u64) * (F2FS_BLOCK_SIZE as u64);

/// Number of pointers per single-indirect node block.
/// A 4 KiB indirect node block holds (4096 - 16 footer) / 4 = 1020
/// 32-bit block addresses, but the kernel reserves 4 bytes at the
/// start so we use 1018 to be safe; the canonical Linux 6.6 value is
/// `ADDRS_PER_BLOCK = 1018`.
pub const ADDRS_PER_BLOCK: usize = 1018;

/// Maximum file offset reachable through direct + single-indirect.
pub const MAX_INDIRECT_FILE_BYTES: u64 =
    MAX_DIRECT_FILE_BYTES + (ADDRS_PER_BLOCK as u64) * (F2FS_BLOCK_SIZE as u64);

/// Look up a NAT entry by `nid` given the on-disk NAT layout.
///
/// `nat_blkaddr` is the partition-relative block address of the NAT
/// area's first block (from the superblock). For the v1 reader we use
/// the primary NAT pair (the secondary copy + version-bitmap selection
/// is a Phase 6+ optimization).
pub fn lookup_nat<D: BlockDevice + ?Sized>(
    device: &D,
    partition_lba: u64,
    partition_size_bytes: u64,
    sb: &Superblock,
    nid: u32,
) -> Result<NatEntry, F2fsError> {
    if nid == F2FS_NULL_NID {
        return Err(F2fsError::NodeIdOutOfRange);
    }
    let block_idx = nid_to_nat_block_idx(nid);
    let entry_off = nid_to_entry_off(nid);

    // Bound by total NAT capacity = segments × blocks-per-segment.
    let nat_capacity_blocks = sb
        .segment_count_nat
        .saturating_mul(super::F2FS_BLOCKS_PER_SEG);
    if block_idx >= nat_capacity_blocks {
        return Err(F2fsError::NodeIdOutOfRange);
    }

    let phys = sb.nat_blkaddr.saturating_add(block_idx);
    let mut block = [0u8; F2FS_BLOCK_SIZE as usize];
    read_f2fs_block(
        device,
        partition_lba,
        partition_size_bytes,
        phys,
        &mut block,
    )?;
    let entries = parse_nat_block(&block)?;
    entries
        .get(entry_off)
        .copied()
        .ok_or(F2fsError::NodeIdOutOfRange)
}

/// Compute the physical block address for the `block_idx`-th 4 KiB
/// block of `inode`. Returns `None` for sparse holes (no NAT mapping).
pub fn resolve_block_addr<D: BlockDevice + ?Sized>(
    device: &D,
    partition_lba: u64,
    partition_size_bytes: u64,
    sb: &Superblock,
    inode: &Inode,
    block_idx: usize,
) -> Result<Option<u32>, F2fsError> {
    if block_idx < ADDRS_PER_INODE {
        let addr = inode.direct_block_addr(block_idx).unwrap_or(0);
        if addr == 0 || addr == u32::MAX {
            return Ok(None);
        }
        return Ok(Some(addr));
    }

    // Single-indirect path: walk i_nid[0] (the first direct-node
    // pointer) which references a 4 KiB node block holding 1018 block
    // addresses. v1 supports only this one level.
    let indirect_idx = block_idx - ADDRS_PER_INODE;
    if indirect_idx >= ADDRS_PER_BLOCK {
        return Err(F2fsError::DeepIndirectionUnsupported);
    }

    if NIDS_PER_INODE == 0 {
        return Err(F2fsError::DeepIndirectionUnsupported);
    }
    let nid = inode.i_nid[0];
    if nid == F2FS_NULL_NID {
        return Ok(None);
    }
    let nat = lookup_nat(device, partition_lba, partition_size_bytes, sb, nid)?;
    if !nat.is_valid_addr() {
        return Ok(None);
    }
    let mut node_block = [0u8; F2FS_BLOCK_SIZE as usize];
    read_f2fs_block(
        device,
        partition_lba,
        partition_size_bytes,
        nat.block_addr,
        &mut node_block,
    )?;
    let off = indirect_idx * 4;
    if off + 4 > node_block.len() {
        return Err(F2fsError::BadBlockAddress);
    }
    let addr = u32::from_le_bytes([
        node_block[off],
        node_block[off + 1],
        node_block[off + 2],
        node_block[off + 3],
    ]);
    if addr == 0 || addr == u32::MAX {
        return Ok(None);
    }
    Ok(Some(addr))
}

/// Read up to `out.len()` bytes of file data into `out` starting at
/// byte offset `offset` within the file.
///
/// Handles inline data, direct pointers, and single-indirect.
/// Returns the number of bytes actually read (which may be shorter
/// than `out.len()` if the read hits EOF).
#[allow(clippy::too_many_arguments)]
pub fn read_file_bytes<D: BlockDevice + ?Sized>(
    device: &D,
    partition_lba: u64,
    partition_size_bytes: u64,
    sb: &Superblock,
    inode: &Inode,
    inode_block: &[u8; F2FS_BLOCK_SIZE as usize],
    offset: u64,
    out: &mut [u8],
) -> Result<usize, F2fsError> {
    if offset >= inode.size {
        return Ok(0);
    }
    let max = core::cmp::min(out.len() as u64, inode.size - offset);
    if max == 0 {
        return Ok(0);
    }
    let mut written: usize = 0;
    let mut cursor = offset;
    let mut remaining = max;

    while remaining > 0 {
        let block_idx = (cursor / F2FS_BLOCK_SIZE as u64) as usize;
        let in_block_off = (cursor % F2FS_BLOCK_SIZE as u64) as usize;
        let take = core::cmp::min(remaining as usize, F2FS_BLOCK_SIZE as usize - in_block_off);

        // Special case: inline data lives in the inode block itself,
        // starting at the inline data area (offset 360 in v1 layout).
        // For inline files the kernel ignores i_addr[]; size <= 3692
        // bytes (4096 - 360 - 16 footer ≈ 3720, with safety margin).
        if inode.has_inline_data {
            // Inline data area starts at byte 360 within the inode block.
            const INLINE_DATA_OFFSET: usize = 360;
            const INLINE_DATA_END: usize = 4080; // before footer
            let max_inline = INLINE_DATA_END - INLINE_DATA_OFFSET;
            if (cursor as usize) >= max_inline {
                return Err(F2fsError::InlineDataCorrupt);
            }
            let inline_off = INLINE_DATA_OFFSET + cursor as usize;
            let inline_take = core::cmp::min(take, INLINE_DATA_END - inline_off);
            out[written..written + inline_take]
                .copy_from_slice(&inode_block[inline_off..inline_off + inline_take]);
            written += inline_take;
            cursor += inline_take as u64;
            remaining -= inline_take as u64;
            continue;
        }

        let block_addr = resolve_block_addr(
            device,
            partition_lba,
            partition_size_bytes,
            sb,
            inode,
            block_idx,
        )?;
        match block_addr {
            None => {
                // Sparse hole — return zeros.
                for byte in out[written..written + take].iter_mut() {
                    *byte = 0;
                }
            }
            Some(addr) => {
                let mut block = [0u8; F2FS_BLOCK_SIZE as usize];
                read_f2fs_block(
                    device,
                    partition_lba,
                    partition_size_bytes,
                    addr,
                    &mut block,
                )?;
                out[written..written + take]
                    .copy_from_slice(&block[in_block_off..in_block_off + take]);
            }
        }
        written += take;
        cursor += take as u64;
        remaining -= take as u64;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f2fs::inode::{Inode, InodeKind, ADDRS_PER_INODE, NIDS_PER_INODE, S_IFREG};

    fn empty_inode() -> Inode {
        Inode {
            mode: S_IFREG,
            kind: InodeKind::Regular,
            uid: 0,
            gid: 0,
            links: 1,
            size: 0,
            blocks: 0,
            atime: 0,
            ctime: 0,
            mtime: 0,
            atime_nsec: 0,
            ctime_nsec: 0,
            mtime_nsec: 0,
            generation: 0,
            xattr_nid: 0,
            flags: 0,
            pino: 0,
            name: alloc::string::String::new(),
            dir_level: 0,
            inline_flags: 0,
            i_addr: alloc::vec![0u32; ADDRS_PER_INODE],
            i_nid: [0u32; NIDS_PER_INODE],
            has_inline_data: false,
            has_inline_dentry: false,
        }
    }

    #[test]
    fn max_direct_constant_correct() {
        // 923 × 4 KiB ≈ 3.6 MiB.
        assert_eq!(MAX_DIRECT_FILE_BYTES, 923 * 4096);
    }

    #[test]
    fn max_indirect_constant_correct() {
        assert_eq!(MAX_INDIRECT_FILE_BYTES, MAX_DIRECT_FILE_BYTES + 1018 * 4096);
    }

    #[test]
    fn resolve_direct_returns_addr() {
        // No device call needed — direct path.
        let mut inode = empty_inode();
        inode.i_addr[5] = 0x100;
        // Build a tiny mock that should never be hit.
        let mock = crate::block::mock::MockBlockDevice::new(4096, 1);
        let sb = make_minimal_sb();
        let r = resolve_block_addr(&mock, 0, 4096, &sb, &inode, 5).unwrap();
        assert_eq!(r, Some(0x100));
    }

    #[test]
    fn resolve_direct_zero_is_hole() {
        let inode = empty_inode();
        let mock = crate::block::mock::MockBlockDevice::new(4096, 1);
        let sb = make_minimal_sb();
        let r = resolve_block_addr(&mock, 0, 4096, &sb, &inode, 5).unwrap();
        assert_eq!(r, None);
    }

    #[test]
    fn beyond_single_indirect_unsupported() {
        let inode = empty_inode();
        let mock = crate::block::mock::MockBlockDevice::new(4096, 1);
        let sb = make_minimal_sb();
        let big_idx = ADDRS_PER_INODE + ADDRS_PER_BLOCK + 1;
        let r = resolve_block_addr(&mock, 0, 4096, &sb, &inode, big_idx);
        assert!(matches!(r, Err(F2fsError::DeepIndirectionUnsupported)));
    }

    fn make_minimal_sb() -> super::Superblock {
        super::Superblock {
            magic: super::super::F2FS_SUPER_MAGIC,
            major_ver: 1,
            minor_ver: 0,
            log_sectorsize: 9,
            log_sectors_per_block: 3,
            log_blocksize: 12,
            log_blocks_per_seg: 9,
            segs_per_sec: 1,
            secs_per_zone: 1,
            checksum_offset: 0,
            block_count: 1024,
            section_count: 4,
            segment_count: 4,
            segment_count_ckpt: 1,
            segment_count_sit: 1,
            segment_count_nat: 1,
            segment_count_ssa: 1,
            segment_count_main: 1,
            segment0_blkaddr: 512,
            cp_blkaddr: 512,
            sit_blkaddr: 1024,
            nat_blkaddr: 1536,
            ssa_blkaddr: 2048,
            main_blkaddr: 2560,
            root_ino: 3,
            node_ino: 1,
            meta_ino: 2,
            uuid: [0u8; 16],
            volume_name: alloc::string::String::new(),
            feature: 0,
            stored_crc: 0,
        }
    }

    #[test]
    fn read_inline_data_round_trip() {
        let mut inode = empty_inode();
        inode.has_inline_data = true;
        inode.size = 16;
        let mut inode_block = [0u8; F2FS_BLOCK_SIZE as usize];
        let payload: &[u8; 16] = b"hello inline f2!";
        inode_block[360..360 + 16].copy_from_slice(payload);

        let mock = crate::block::mock::MockBlockDevice::new(4096, 1);
        let sb = make_minimal_sb();
        let mut out = [0u8; 16];
        let n = read_file_bytes(&mock, 0, 4096, &sb, &inode, &inode_block, 0, &mut out).unwrap();
        assert_eq!(n, 16);
        assert_eq!(&out, payload);
    }

    #[test]
    fn read_past_eof_returns_zero() {
        let mut inode = empty_inode();
        inode.size = 100;
        let mock = crate::block::mock::MockBlockDevice::new(4096, 1);
        let sb = make_minimal_sb();
        let inode_block = [0u8; F2FS_BLOCK_SIZE as usize];
        let mut out = [0u8; 16];
        let n = read_file_bytes(&mock, 0, 4096, &sb, &inode, &inode_block, 200, &mut out).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn read_partial_truncated_to_eof() {
        let mut inode = empty_inode();
        inode.has_inline_data = true;
        inode.size = 8;
        let mut inode_block = [0u8; F2FS_BLOCK_SIZE as usize];
        inode_block[360..360 + 8].copy_from_slice(b"abcdefgh");

        let mock = crate::block::mock::MockBlockDevice::new(4096, 1);
        let sb = make_minimal_sb();
        let mut out = [0u8; 32];
        let n = read_file_bytes(&mock, 0, 4096, &sb, &inode, &inode_block, 0, &mut out).unwrap();
        assert_eq!(n, 8);
        assert_eq!(&out[..8], b"abcdefgh");
    }
}
