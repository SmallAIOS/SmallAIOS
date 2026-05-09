// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic F2FS images for the *write-path* conformance suite
//! (Phase 7 of `embedded-filesystem-v1`).
//!
//! The Phase 5 read-path fixtures ([`super::build_minimal_image`])
//! produce a minimal image with a single main-area segment and the
//! root inode + one data block already populated. That layout is too
//! tight for write tests because the write path's allocator advances
//! its cursor monotonically through the main area; if the cursor
//! starts at offset 0 it overwrites the root inode on its first
//! allocation.
//!
//! These fixtures write a slightly larger image: 16 main-area
//! segments, the root inode placed at the first block of segment 0,
//! and the data-segment cursor seeded *past* the root inode + initial
//! file. The remaining 15+ segments are usable for round-trip writes.
//!
//! Like the Phase 5 fixtures, this file lives under a
//! `_test_vectors.rs` filename so the CodeQL byte-array rule does not
//! flag the on-disk-format constants.

#![allow(dead_code)]

extern crate alloc;

use smallaios_fs::block::mock::MockBlockDevice;
use smallaios_fs::block::BlockDevice;
use smallaios_fs::f2fs::superblock::{
    F2FS_SUPER_BLOCK_BYTES, F2FS_SUPER_MAGIC, PRIMARY_SUPERBLOCK_OFFSET,
    SECONDARY_SUPERBLOCK_OFFSET,
};

/// Disk size in 4 KiB blocks (32 MiB).
pub const RW_DEFAULT_BLOCKS: u64 = 8192;

/// Layout positions (in 4 KiB block addresses, partition-relative).
pub const RW_SEGMENT0_BLKADDR: u32 = 0;
pub const RW_CP_BLKADDR: u32 = 512;
/// Both checkpoint packs occupy contiguous 512-block windows so the
/// alternate-slot commit lands cleanly.
pub const RW_CP2_BLKADDR: u32 = 1024;
pub const RW_SIT_BLKADDR: u32 = 1536;
pub const RW_NAT_BLKADDR: u32 = 1600;
pub const RW_SSA_BLKADDR: u32 = 1664;
pub const RW_MAIN_BLKADDR: u32 = 1700;

pub const RW_ROOT_NID: u32 = 3;
/// Where the root inode lives.
pub const RW_ROOT_INODE_BLKADDR: u32 = RW_MAIN_BLKADDR;
/// Where the data-segment cursor should resume — past the root inode.
pub const RW_INITIAL_CUR_DATA_BLKOFF: u16 = 1;

/// Build a fresh, mountable F2FS image suitable for the Phase 7 write
/// tests. The root directory exists with INLINE_DENTRY set; no other
/// files are pre-populated.
pub fn build_rw_image() -> MockBlockDevice {
    let dev = MockBlockDevice::new(4096, RW_DEFAULT_BLOCKS);

    // ─── Superblocks (primary @ 1024, secondary @ 9216) ──────────────────
    let sb = build_rw_superblock();
    write_superblock(&dev, PRIMARY_SUPERBLOCK_OFFSET, &sb);
    write_superblock(&dev, SECONDARY_SUPERBLOCK_OFFSET, &sb);

    // ─── Checkpoint pack #1 (CP1 wins) ───────────────────────────────────
    let cp = build_rw_checkpoint(7);
    dev.preload(RW_CP_BLKADDR as u64, &cp);
    let cp2 = build_rw_checkpoint(5);
    dev.preload(RW_CP2_BLKADDR as u64, &cp2);

    // ─── NAT block 0 (entry for root nid=3) ──────────────────────────────
    let mut nat_block = [0u8; 4096];
    write_nat_entry(&mut nat_block, RW_ROOT_NID, RW_ROOT_INODE_BLKADDR);
    dev.preload(RW_NAT_BLKADDR as u64, &nat_block);

    // ─── Root inode block (INLINE_DENTRY directory) ──────────────────────
    let mut root_inode = [0u8; 4096];
    write_inode_header(
        &mut root_inode,
        smallaios_fs::f2fs::inode::S_IFDIR | 0o755,
        4096,
        smallaios_fs::f2fs::inode::INLINE_DENTRY,
        b"/",
        2, /* links: . + .. */
    );
    write_node_footer(&mut root_inode, RW_ROOT_NID, RW_ROOT_NID);
    // Seed inline dentry with `.` and `..` only (real inline dentry
    // table at offset 360).
    let dot_dotdot = build_dentry_block_inline(&[(".", RW_ROOT_NID, 2), ("..", RW_ROOT_NID, 2)]);
    let inline_off = 360usize;
    let footer_off = 4096usize - 16;
    let take = core::cmp::min(dot_dotdot.len(), footer_off - inline_off);
    root_inode[inline_off..inline_off + take].copy_from_slice(&dot_dotdot[..take]);
    dev.preload(RW_ROOT_INODE_BLKADDR as u64, &root_inode);

    dev
}

fn build_rw_superblock() -> alloc::vec::Vec<u8> {
    let mut buf = alloc::vec![0u8; F2FS_SUPER_BLOCK_BYTES];
    buf[0..4].copy_from_slice(&F2FS_SUPER_MAGIC.to_le_bytes());
    buf[4..6].copy_from_slice(&1u16.to_le_bytes());
    buf[6..8].copy_from_slice(&0u16.to_le_bytes());
    buf[8..12].copy_from_slice(&12u32.to_le_bytes());
    buf[12..16].copy_from_slice(&0u32.to_le_bytes());
    buf[16..20].copy_from_slice(&12u32.to_le_bytes());
    buf[20..24].copy_from_slice(&9u32.to_le_bytes()); // 512 blk/seg
    buf[24..28].copy_from_slice(&1u32.to_le_bytes());
    buf[28..32].copy_from_slice(&1u32.to_le_bytes());
    buf[32..36].copy_from_slice(&0u32.to_le_bytes()); // no CRC
    buf[36..44].copy_from_slice(&RW_DEFAULT_BLOCKS.to_le_bytes());
    buf[44..48].copy_from_slice(&16u32.to_le_bytes());
    buf[48..52].copy_from_slice(&16u32.to_le_bytes());
    buf[52..56].copy_from_slice(&1u32.to_le_bytes()); // segment_count_ckpt
    buf[56..60].copy_from_slice(&1u32.to_le_bytes()); // segment_count_sit
    buf[60..64].copy_from_slice(&1u32.to_le_bytes()); // segment_count_nat
    buf[64..68].copy_from_slice(&1u32.to_le_bytes()); // segment_count_ssa
    buf[68..72].copy_from_slice(&12u32.to_le_bytes()); // segment_count_main = 12
    buf[72..76].copy_from_slice(&RW_SEGMENT0_BLKADDR.to_le_bytes());
    buf[76..80].copy_from_slice(&RW_CP_BLKADDR.to_le_bytes());
    buf[80..84].copy_from_slice(&RW_SIT_BLKADDR.to_le_bytes());
    buf[84..88].copy_from_slice(&RW_NAT_BLKADDR.to_le_bytes());
    buf[88..92].copy_from_slice(&RW_SSA_BLKADDR.to_le_bytes());
    buf[92..96].copy_from_slice(&RW_MAIN_BLKADDR.to_le_bytes());
    buf[96..100].copy_from_slice(&RW_ROOT_NID.to_le_bytes());
    buf[100..104].copy_from_slice(&1u32.to_le_bytes()); // node_ino
    buf[104..108].copy_from_slice(&2u32.to_le_bytes()); // meta_ino
    buf
}

fn build_rw_checkpoint(version: u64) -> alloc::vec::Vec<u8> {
    let mut buf = alloc::vec![0u8; 4096];
    buf[0..8].copy_from_slice(&version.to_le_bytes());
    buf[8..16].copy_from_slice(&100u64.to_le_bytes());
    buf[16..24].copy_from_slice(&50u64.to_le_bytes());
    // Seed cur_data_segno[0] = 0, cur_data_blkoff[0] = 1 so the
    // first allocation lands past the root inode.
    buf[84..88].copy_from_slice(&0u32.to_le_bytes());
    buf[116..118].copy_from_slice(&RW_INITIAL_CUR_DATA_BLKOFF.to_le_bytes());
    // next_free_nid = 4 (NIDs 0..3 are reserved + root).
    buf[140..144].copy_from_slice(&4u32.to_le_bytes());
    buf[152..156].copy_from_slice(&0u32.to_le_bytes());
    buf
}

fn write_superblock(dev: &MockBlockDevice, byte_offset: u64, sb: &[u8]) {
    let block_off = byte_offset / 4096;
    let in_block = (byte_offset % 4096) as usize;
    let mut block = [0u8; 4096];
    dev.read_block(block_off, &mut block).unwrap();
    block[in_block..in_block + sb.len()].copy_from_slice(sb);
    dev.preload(block_off, &block);
}

fn write_nat_entry(block: &mut [u8; 4096], nid: u32, block_addr: u32) {
    let entry_off = (nid as usize) % smallaios_fs::f2fs::nat::NAT_ENTRY_PER_BLOCK
        * smallaios_fs::f2fs::nat::NAT_ENTRY_SIZE;
    block[entry_off] = 1;
    block[entry_off + 1..entry_off + 5].copy_from_slice(&nid.to_le_bytes());
    block[entry_off + 5..entry_off + 9].copy_from_slice(&block_addr.to_le_bytes());
}

#[allow(clippy::too_many_arguments)]
fn write_inode_header(
    block: &mut [u8; 4096],
    mode: u16,
    size: u64,
    inline_flags: u8,
    name: &[u8],
    links: u32,
) {
    block[0..2].copy_from_slice(&mode.to_le_bytes());
    block[3] = inline_flags;
    block[12..16].copy_from_slice(&links.to_le_bytes());
    block[16..24].copy_from_slice(&size.to_le_bytes());
    let nl = core::cmp::min(name.len(), 255);
    block[84..88].copy_from_slice(&(nl as u32).to_le_bytes());
    block[88..88 + nl].copy_from_slice(&name[..nl]);
}

fn write_node_footer(block: &mut [u8; 4096], nid: u32, ino: u32) {
    let off = smallaios_fs::f2fs::inode::NODE_FOOTER_OFFSET;
    block[off..off + 4].copy_from_slice(&nid.to_le_bytes());
    block[off + 4..off + 8].copy_from_slice(&ino.to_le_bytes());
    block[off + 8..off + 12].copy_from_slice(&0u32.to_le_bytes());
}

fn build_dentry_block_inline(entries: &[(&str, u32, u8)]) -> alloc::vec::Vec<u8> {
    use smallaios_fs::f2fs::dir::{
        slots_for_name, ENTRY_ARRAY_OFFSET, F2FS_DIR_ENTRY_SIZE, F2FS_SLOT_LEN,
        FILENAME_SLOTS_OFFSET,
    };
    let mut buf = alloc::vec![0u8; 4096];
    let mut slot = 0usize;
    for &(name, ino, ft) in entries {
        let nl = name.len() as u16;
        let slots = slots_for_name(nl);
        for s in slot..slot + slots {
            buf[s / 8] |= 1u8 << (s % 8);
        }
        let entry_off = ENTRY_ARRAY_OFFSET + slot * F2FS_DIR_ENTRY_SIZE;
        buf[entry_off..entry_off + 4].copy_from_slice(&0u32.to_le_bytes());
        buf[entry_off + 4..entry_off + 8].copy_from_slice(&ino.to_le_bytes());
        buf[entry_off + 8..entry_off + 10].copy_from_slice(&nl.to_le_bytes());
        buf[entry_off + 10] = ft;
        let name_start = FILENAME_SLOTS_OFFSET + slot * F2FS_SLOT_LEN;
        buf[name_start..name_start + name.len()].copy_from_slice(name.as_bytes());
        slot += slots;
    }
    buf
}
