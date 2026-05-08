// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic F2FS image fixtures for the conformance suite.
//!
//! Hand-crafted test vectors lay down a minimal-but-valid F2FS layout:
//! one superblock at offset 1024, a duplicate at offset 9216, a single
//! checkpoint pack, a NAT pair, an SIT pair, an SSA, and a "main area"
//! holding the root inode + a single dentry block. The builder is
//! enough for the v1 read-path conformance suite; it is not a
//! drop-in replacement for `mkfs.f2fs`.
//!
//! The directory in this file is paths-ignored by the codeql config
//! (`**/*_test_vectors.rs`).

#![allow(dead_code)] // Each test binary uses a different subset.

use smallaios_fs::block::mock::MockBlockDevice;
use smallaios_fs::block::BlockDevice;
use smallaios_fs::f2fs::superblock::{
    F2FS_SUPER_BLOCK_BYTES, F2FS_SUPER_MAGIC, PRIMARY_SUPERBLOCK_OFFSET,
    SECONDARY_SUPERBLOCK_OFFSET,
};

/// Default disk size in 4 KiB blocks (8 MiB).
pub const DEFAULT_BLOCKS: u64 = 2048;

/// Layout positions (in 4 KiB block addresses, partition-relative).
pub const SEGMENT0_BLKADDR: u32 = 0;
pub const CP_BLKADDR: u32 = 512;
pub const CP2_BLKADDR: u32 = 1024;
pub const SIT_BLKADDR: u32 = 1536;
pub const NAT_BLKADDR: u32 = 1600;
pub const SSA_BLKADDR: u32 = 1664;
pub const MAIN_BLKADDR: u32 = 1700;

/// Where the root inode lives in the main area.
pub const ROOT_INODE_BLKADDR: u32 = MAIN_BLKADDR;
/// Where the root directory's first dentry block lives.
pub const ROOT_DIR_BLKADDR: u32 = MAIN_BLKADDR + 1;

/// Per-NID assignments (root_ino = 3 reserves NIDs 0..=2 for meta).
pub const ROOT_NID: u32 = 3;
pub const FILE_INODE_BLKADDR: u32 = MAIN_BLKADDR + 2;
pub const FILE_DATA_BLKADDR: u32 = MAIN_BLKADDR + 3;
pub const FILE_NID: u32 = 4;

/// Build a minimal mountable F2FS image for the conformance tests.
///
/// `file_payload` is the bytes of a single regular file named
/// `file.txt` placed at the partition root. If empty, an inline-data
/// inode is emitted (size 0). Otherwise a separate data block is
/// written and the inode points to it via direct pointer 0.
pub fn build_minimal_image(file_payload: &[u8]) -> MockBlockDevice {
    let dev = MockBlockDevice::new(4096, DEFAULT_BLOCKS);

    // ─── Superblocks (primary @ 1024, secondary @ 9216) ────────────────
    let sb = build_superblock();
    write_superblock(&dev, PRIMARY_SUPERBLOCK_OFFSET, &sb);
    write_superblock(&dev, SECONDARY_SUPERBLOCK_OFFSET, &sb);

    // ─── Checkpoint pack #1 (higher version wins) ──────────────────────
    let cp = build_checkpoint(7);
    dev.preload(CP_BLKADDR as u64, &cp);
    let cp2 = build_checkpoint(5);
    dev.preload(CP2_BLKADDR as u64, &cp2);

    // ─── NAT block 0 (entries for nid=3 = root, nid=4 = file) ──────────
    let mut nat_block = [0u8; 4096];
    write_nat_entry(&mut nat_block, ROOT_NID, ROOT_INODE_BLKADDR);
    write_nat_entry(&mut nat_block, FILE_NID, FILE_INODE_BLKADDR);
    dev.preload(NAT_BLKADDR as u64, &nat_block);

    // ─── Root inode block ──────────────────────────────────────────────
    let mut root_inode = [0u8; 4096];
    write_inode_header(
        &mut root_inode,
        smallaios_fs::f2fs::inode::S_IFDIR | 0o755,
        2 * 4096, // size = 2 dentry blocks; we'll only populate one
        0,
        b".",
    );
    // Root has its first dentry block at i_addr[0].
    write_inode_direct_addr(&mut root_inode, 0, ROOT_DIR_BLKADDR);
    write_node_footer(&mut root_inode, ROOT_NID, ROOT_NID);
    dev.preload(ROOT_INODE_BLKADDR as u64, &root_inode);

    // ─── Root directory's dentry block: ".", "..", "file.txt" ──────────
    let dentries = build_dentry_block(&[
        (".", ROOT_NID, 2), // DT_DIR = 2
        ("..", ROOT_NID, 2),
        ("file.txt", FILE_NID, 1), // DT_REG = 1
    ]);
    dev.preload(ROOT_DIR_BLKADDR as u64, &dentries);

    // ─── File inode block ──────────────────────────────────────────────
    let mut file_inode = [0u8; 4096];
    let inline = file_payload.len() <= 1500;
    let inline_flag = if inline {
        smallaios_fs::f2fs::inode::INLINE_DATA
    } else {
        0
    };
    write_inode_header(
        &mut file_inode,
        smallaios_fs::f2fs::inode::S_IFREG | 0o644,
        file_payload.len() as u64,
        inline_flag,
        b"file.txt",
    );
    if inline {
        // Inline data lives at offset 360 in the inode block.
        let n = file_payload.len();
        file_inode[360..360 + n].copy_from_slice(file_payload);
    } else {
        write_inode_direct_addr(&mut file_inode, 0, FILE_DATA_BLKADDR);
        // Write the data block.
        let mut data = [0u8; 4096];
        let take = core::cmp::min(file_payload.len(), 4096);
        data[..take].copy_from_slice(&file_payload[..take]);
        dev.preload(FILE_DATA_BLKADDR as u64, &data);
    }
    write_node_footer(&mut file_inode, FILE_NID, FILE_NID);
    dev.preload(FILE_INODE_BLKADDR as u64, &file_inode);

    dev
}

/// Corrupt the primary superblock (zero its magic). Returns the
/// modified device. The secondary superblock remains valid so mount
/// must still succeed.
pub fn corrupt_primary_superblock(dev: &MockBlockDevice) {
    // Read the raw block 0, zero the magic at offset 1024, write back.
    let mut block0 = [0u8; 4096];
    dev.read_block(0, &mut block0).unwrap();
    block0[1024..1028].copy_from_slice(&0u32.to_le_bytes());
    dev.preload(0, &block0);
}

/// Corrupt both superblocks (zero the magic in both).
pub fn corrupt_both_superblocks(dev: &MockBlockDevice) {
    let mut block0 = [0u8; 4096];
    dev.read_block(0, &mut block0).unwrap();
    block0[1024..1028].copy_from_slice(&0u32.to_le_bytes());
    dev.preload(0, &block0);
    let mut block2 = [0u8; 4096];
    dev.read_block(2, &mut block2).unwrap();
    // Secondary SB at byte 9216 = block 2, offset 1024.
    block2[1024..1028].copy_from_slice(&0u32.to_le_bytes());
    dev.preload(2, &block2);
}

/// Set a mandatory feature bit the v1 reader does not handle (ENCRYPT).
pub fn set_unsupported_feature(dev: &MockBlockDevice) {
    let mut block0 = [0u8; 4096];
    dev.read_block(0, &mut block0).unwrap();
    let bit = smallaios_fs::f2fs::superblock::FEATURE_ENCRYPT as u32;
    let off = 1024 + 636;
    block0[off..off + 4].copy_from_slice(&bit.to_le_bytes());
    dev.preload(0, &block0);
    let mut block2 = [0u8; 4096];
    dev.read_block(2, &mut block2).unwrap();
    block2[off..off + 4].copy_from_slice(&bit.to_le_bytes());
    dev.preload(2, &block2);
}

/// Build the on-disk superblock bytes (1024 bytes, no CRC).
fn build_superblock() -> alloc::vec::Vec<u8> {
    let mut buf = alloc::vec![0u8; F2FS_SUPER_BLOCK_BYTES];
    buf[0..4].copy_from_slice(&F2FS_SUPER_MAGIC.to_le_bytes());
    buf[4..6].copy_from_slice(&1u16.to_le_bytes());
    buf[6..8].copy_from_slice(&0u16.to_le_bytes());
    buf[8..12].copy_from_slice(&12u32.to_le_bytes()); // log_sectorsize = 4 KiB sec
    buf[12..16].copy_from_slice(&0u32.to_le_bytes()); // log_sectors_per_block = 1 (4 KiB)
    buf[16..20].copy_from_slice(&12u32.to_le_bytes()); // log_blocksize = 4 KiB
    buf[20..24].copy_from_slice(&9u32.to_le_bytes()); // log_blocks_per_seg = 512
    buf[24..28].copy_from_slice(&1u32.to_le_bytes());
    buf[28..32].copy_from_slice(&1u32.to_le_bytes());
    buf[32..36].copy_from_slice(&0u32.to_le_bytes()); // checksum_offset = 0 (no CRC)
    buf[36..44].copy_from_slice(&DEFAULT_BLOCKS.to_le_bytes());
    buf[44..48].copy_from_slice(&8u32.to_le_bytes()); // section_count
    buf[48..52].copy_from_slice(&8u32.to_le_bytes()); // segment_count
    buf[52..56].copy_from_slice(&1u32.to_le_bytes()); // segment_count_ckpt
    buf[56..60].copy_from_slice(&1u32.to_le_bytes()); // segment_count_sit
    buf[60..64].copy_from_slice(&1u32.to_le_bytes()); // segment_count_nat
    buf[64..68].copy_from_slice(&1u32.to_le_bytes()); // segment_count_ssa
    buf[68..72].copy_from_slice(&1u32.to_le_bytes()); // segment_count_main
    buf[72..76].copy_from_slice(&SEGMENT0_BLKADDR.to_le_bytes());
    buf[76..80].copy_from_slice(&CP_BLKADDR.to_le_bytes());
    buf[80..84].copy_from_slice(&SIT_BLKADDR.to_le_bytes());
    buf[84..88].copy_from_slice(&NAT_BLKADDR.to_le_bytes());
    buf[88..92].copy_from_slice(&SSA_BLKADDR.to_le_bytes());
    buf[92..96].copy_from_slice(&MAIN_BLKADDR.to_le_bytes());
    buf[96..100].copy_from_slice(&ROOT_NID.to_le_bytes()); // root_ino = 3
    buf[100..104].copy_from_slice(&1u32.to_le_bytes()); // node_ino
    buf[104..108].copy_from_slice(&2u32.to_le_bytes()); // meta_ino
                                                        // No feature bits set.
    buf
}

/// Write the 1 KiB superblock into the underlying 4 KiB block at the
/// given partition-relative byte offset.
fn write_superblock(dev: &MockBlockDevice, byte_offset: u64, sb: &[u8]) {
    let block_off = byte_offset / 4096;
    let in_block = (byte_offset % 4096) as usize;
    let mut block = [0u8; 4096];
    dev.read_block(block_off, &mut block).unwrap();
    block[in_block..in_block + sb.len()].copy_from_slice(sb);
    dev.preload(block_off, &block);
}

/// Build a 4 KiB checkpoint pack with the given version.
fn build_checkpoint(version: u64) -> alloc::vec::Vec<u8> {
    let mut buf = alloc::vec![0u8; 4096];
    buf[0..8].copy_from_slice(&version.to_le_bytes());
    buf[8..16].copy_from_slice(&100u64.to_le_bytes());
    buf[16..24].copy_from_slice(&50u64.to_le_bytes());
    buf[152..156].copy_from_slice(&0u32.to_le_bytes()); // checksum_offset = 0
    buf
}

/// Write a NAT entry into the given NAT block buffer.
fn write_nat_entry(block: &mut [u8; 4096], nid: u32, block_addr: u32) {
    let entry_off = (nid as usize) % smallaios_fs::f2fs::nat::NAT_ENTRY_PER_BLOCK
        * smallaios_fs::f2fs::nat::NAT_ENTRY_SIZE;
    block[entry_off] = 1; // version
    block[entry_off + 1..entry_off + 5].copy_from_slice(&nid.to_le_bytes()); // ino = nid for inodes
    block[entry_off + 5..entry_off + 9].copy_from_slice(&block_addr.to_le_bytes());
}

/// Fill an inode header (mode, size, inline flags, name).
fn write_inode_header(block: &mut [u8; 4096], mode: u16, size: u64, inline_flags: u8, name: &[u8]) {
    block[0..2].copy_from_slice(&mode.to_le_bytes());
    block[3] = inline_flags;
    block[12..16].copy_from_slice(&1u32.to_le_bytes()); // links
    block[16..24].copy_from_slice(&size.to_le_bytes());
    let nl = core::cmp::min(name.len(), 255);
    block[84..88].copy_from_slice(&(nl as u32).to_le_bytes());
    block[88..88 + nl].copy_from_slice(&name[..nl]);
}

/// Set i_addr[idx] to a physical block address.
fn write_inode_direct_addr(block: &mut [u8; 4096], idx: usize, addr: u32) {
    let off = 360 + idx * 4;
    block[off..off + 4].copy_from_slice(&addr.to_le_bytes());
}

/// Write the 16-byte node footer (nid, ino, flag = 0).
fn write_node_footer(block: &mut [u8; 4096], nid: u32, ino: u32) {
    let off = smallaios_fs::f2fs::inode::NODE_FOOTER_OFFSET;
    block[off..off + 4].copy_from_slice(&nid.to_le_bytes());
    block[off + 4..off + 8].copy_from_slice(&ino.to_le_bytes());
    block[off + 8..off + 12].copy_from_slice(&0u32.to_le_bytes());
}

/// Build a 4 KiB dentry block holding the given names.
pub fn build_dentry_block(entries: &[(&str, u32, u8)]) -> alloc::vec::Vec<u8> {
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
