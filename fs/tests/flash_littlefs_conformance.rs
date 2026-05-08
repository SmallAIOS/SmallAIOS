// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Conformance tests for `fs::flash::littlefs` (Phase 1 read path).
//!
//! These tests build small synthetic littlefs v2.x metadata-pair
//! images directly inside a [`MockFlashDevice`], then mount them with
//! the production reader and assert that:
//!
//! * The superblock magic + version checks behave per spec.
//! * Files stored inline read back byte-exact.
//! * Files stored as a single CTZ block read back byte-exact.
//! * Multi-block CTZ files (chain length > 1) read back byte-exact.
//! * Directory iteration enumerates the right entries.
//! * Corrupt CRC, wrong magic, and wrong major version are all
//!   rejected with the spec-mandated error.
//!
//! ## Why a hand-rolled fixture (no `littlefs2` dev-dep yet)
//!
//! The Phase-1 deliverable explicitly allows the fixture generator to
//! be pure-Rust and skips the optional `littlefs2` Rust-port oracle to
//! keep the dependency graph audit-free. The fixture format here uses
//! the *same* on-disk layout the reader parses (revision counter →
//! XOR'd tag stream → CRC tag), so a passing test confirms the
//! parser end-to-end. A subsequent phase will add a weekly cron job
//! that round-trips against `mklittlefs`.

#![cfg(all(feature = "fs-flash", feature = "fs-flash-mock"))]

use smallaios_fs::flash::littlefs::{
    crc32c, DirEntryKind, LittleFs, LittleFsConfig, LittleFsError,
};
use smallaios_fs::flash::mock::MockFlashDevice;
use smallaios_fs::flash::FlashDevice;

// ─── Fixture builder ─────────────────────────────────────────────────────────
//
// Builds one littlefs v2 metadata-pair half. Tag encoding per
// SPEC.md is big-endian 32-bit:
//
//   bit  31      = valid flag (0 = present, 1 = past commit boundary)
//   bits 30..20  = 11-bit type
//   bits 19..10  = 10-bit id
//   bits  9..0   = 10-bit length
//
// Tags are stored XOR'd against the previous tag word; the running CRC
// covers everything (as on disk) and the CRC tag's payload is the
// expected post-update inverted CRC.

const TAG_NAME_REG: u16 = 0x001;
const TAG_NAME_DIR: u16 = 0x002;
const TAG_NAME_SUPERBLOCK: u16 = 0x0FF;
const TAG_STRUCT_DIR: u16 = 0x200;
const TAG_STRUCT_INLINE: u16 = 0x201;
const TAG_STRUCT_CTZ: u16 = 0x202;
const TAG_CRC_NORMAL: u16 = 0x500;

const CRC32C_POLY_REFLECTED: u32 = 0x82F63B78;

fn crc32c_update(mut crc: u32, data: &[u8]) -> u32 {
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let lsb = crc & 1;
            crc >>= 1;
            if lsb == 1 {
                crc ^= CRC32C_POLY_REFLECTED;
            }
        }
    }
    crc
}

#[derive(Clone)]
struct EntrySpec {
    typ_name: u16,
    id: u16,
    name: alloc::vec::Vec<u8>,
    typ_struct: u16,
    struct_payload: alloc::vec::Vec<u8>,
}

extern crate alloc;

fn pack_tag_word(typ: u16, id: u16, len: u32) -> u32 {
    // valid=0 in bit 31; type in bits 30..20; id in bits 19..10; len in bits 9..0
    ((u32::from(typ) & 0x7FF) << 20) | ((u32::from(id) & 0x3FF) << 10) | (len & 0x3FF)
}

fn build_metadata_half(revision: u32, entries: &[EntrySpec]) -> alloc::vec::Vec<u8> {
    let mut buf = alloc::vec::Vec::new();
    buf.extend_from_slice(&revision.to_le_bytes());

    let mut prev_tag_word: u32 = 0;
    let mut running_crc: u32 = 0xFFFF_FFFF;
    running_crc = crc32c_update(running_crc, &revision.to_le_bytes());

    for e in entries {
        // name tag
        let name_tag_word = pack_tag_word(e.typ_name, e.id, e.name.len() as u32);
        let on_disk = name_tag_word ^ prev_tag_word;
        buf.extend_from_slice(&on_disk.to_be_bytes());
        running_crc = crc32c_update(running_crc, &on_disk.to_be_bytes());
        buf.extend_from_slice(&e.name);
        running_crc = crc32c_update(running_crc, &e.name);
        prev_tag_word = name_tag_word;

        // struct tag
        let struct_tag_word = pack_tag_word(e.typ_struct, e.id, e.struct_payload.len() as u32);
        let on_disk2 = struct_tag_word ^ prev_tag_word;
        buf.extend_from_slice(&on_disk2.to_be_bytes());
        running_crc = crc32c_update(running_crc, &on_disk2.to_be_bytes());
        buf.extend_from_slice(&e.struct_payload);
        running_crc = crc32c_update(running_crc, &e.struct_payload);
        prev_tag_word = struct_tag_word;
    }

    // Final CRC tag.
    let crc_tag_word = pack_tag_word(TAG_CRC_NORMAL, 0, 4);
    let on_disk_crc = crc_tag_word ^ prev_tag_word;
    buf.extend_from_slice(&on_disk_crc.to_be_bytes());
    running_crc = crc32c_update(running_crc, &on_disk_crc.to_be_bytes());
    let final_crc = !running_crc;
    buf.extend_from_slice(&final_crc.to_le_bytes());

    buf
}

fn write_pair(
    dev: &mut MockFlashDevice,
    pair: [u32; 2],
    rev_a: u32,
    rev_b: u32,
    entries: &[EntrySpec],
) {
    let block_size = dev.block_size_bytes() as usize;
    let mut half_a = build_metadata_half(rev_a, entries);
    let mut half_b = build_metadata_half(rev_b, entries);
    half_a.resize(block_size, 0xFF);
    half_b.resize(block_size, 0xFF);
    dev.force_write(u64::from(pair[0]) * block_size as u64, &half_a);
    dev.force_write(u64::from(pair[1]) * block_size as u64, &half_b);
}

/// Build the standard superblock entry.
fn superblock_entry(
    version_major: u16,
    version_minor: u16,
    block_size: u32,
    block_count: u32,
) -> EntrySpec {
    let version = (u32::from(version_major) << 16) | u32::from(version_minor);
    let mut payload = alloc::vec::Vec::new();
    payload.extend_from_slice(&version.to_le_bytes());
    payload.extend_from_slice(&block_size.to_le_bytes());
    payload.extend_from_slice(&block_count.to_le_bytes());
    payload.extend_from_slice(&255u32.to_le_bytes()); // name_max
    payload.extend_from_slice(&0x7FFF_FFFFu32.to_le_bytes()); // file_max
    payload.extend_from_slice(&1022u32.to_le_bytes()); // attr_max
    EntrySpec {
        typ_name: TAG_NAME_SUPERBLOCK,
        id: 0,
        name: b"littlefs".to_vec(),
        typ_struct: TAG_STRUCT_INLINE,
        struct_payload: payload,
    }
}

fn inline_file_entry(id: u16, name: &str, contents: &[u8]) -> EntrySpec {
    EntrySpec {
        typ_name: TAG_NAME_REG,
        id,
        name: name.as_bytes().to_vec(),
        typ_struct: TAG_STRUCT_INLINE,
        struct_payload: contents.to_vec(),
    }
}

fn ctz_file_entry(id: u16, name: &str, head: u32, size: u32) -> EntrySpec {
    let mut payload = alloc::vec::Vec::new();
    payload.extend_from_slice(&head.to_le_bytes());
    payload.extend_from_slice(&size.to_le_bytes());
    EntrySpec {
        typ_name: TAG_NAME_REG,
        id,
        name: name.as_bytes().to_vec(),
        typ_struct: TAG_STRUCT_CTZ,
        struct_payload: payload,
    }
}

fn dir_entry(id: u16, name: &str, pair: [u32; 2]) -> EntrySpec {
    let mut payload = alloc::vec::Vec::new();
    payload.extend_from_slice(&pair[0].to_le_bytes());
    payload.extend_from_slice(&pair[1].to_le_bytes());
    EntrySpec {
        typ_name: TAG_NAME_DIR,
        id,
        name: name.as_bytes().to_vec(),
        typ_struct: TAG_STRUCT_DIR,
        struct_payload: payload,
    }
}

fn fresh_device(blocks: u64, block_size: u32) -> MockFlashDevice {
    MockFlashDevice::new(blocks, block_size, 256)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn t01_mount_with_only_superblock() {
    let mut dev = fresh_device(8, 4096);
    let entries = [superblock_entry(2, 0, 4096, 8)];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let fs = LittleFs::mount(&mut dev, cfg).expect("mount succeeds");
    assert_eq!(fs.superblock().major(), 2);
    assert_eq!(fs.superblock().minor(), 0);
    assert_eq!(fs.superblock().block_size, 4096);
    assert_eq!(fs.superblock().block_count, 8);
    assert!(fs.superblock().is_v2());
}

#[test]
fn t02_mount_rejects_v1_image() {
    let mut dev = fresh_device(8, 4096);
    let entries = [superblock_entry(1, 0, 4096, 8)];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let res = LittleFs::mount(&mut dev, cfg);
    match res {
        Err(LittleFsError::UnsupportedMajorVersion(1)) => {}
        Err(other) => panic!("expected v1 rejection, got {other:?}"),
        Ok(_) => panic!("expected v1 rejection, got Ok"),
    }
}

#[test]
fn t03_mount_v3_image_rejected() {
    let mut dev = fresh_device(8, 4096);
    let entries = [superblock_entry(3, 0, 4096, 8)];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    match LittleFs::mount(&mut dev, cfg) {
        Err(LittleFsError::UnsupportedMajorVersion(3)) => {}
        Err(other) => panic!("expected v3 rejection, got {other:?}"),
        Ok(_) => panic!("expected v3 rejection, got Ok"),
    }
}

#[test]
fn t04_mount_with_no_image_errors_corrupted() {
    // Erased flash: both halves are full of 0xFF, so the first tag
    // word's valid bit is set immediately and neither half is valid.
    let mut dev = fresh_device(8, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let res = LittleFs::mount(&mut dev, cfg);
    assert!(matches!(res, Err(LittleFsError::Corrupted)));
}

#[test]
fn t05_mount_rejects_wrong_magic() {
    let mut dev = fresh_device(8, 4096);
    let mut entries = [superblock_entry(2, 0, 4096, 8)];
    entries[0].name = b"otherfs1".to_vec();
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let res = LittleFs::mount(&mut dev, cfg);
    assert!(matches!(res, Err(LittleFsError::BadMagic)));
}

#[test]
fn t06_mount_higher_revision_wins_on_pair() {
    let mut dev = fresh_device(8, 4096);
    let entries_a = [
        superblock_entry(2, 0, 4096, 8),
        inline_file_entry(1, "old", b"old-payload"),
    ];
    let entries_b = [
        superblock_entry(2, 0, 4096, 8),
        inline_file_entry(1, "new", b"new-payload"),
    ];
    // Half A: revision 1. Half B: revision 2 (newer wins).
    let block_size = dev.block_size_bytes() as usize;
    let mut half_a = build_metadata_half(1, &entries_a);
    let mut half_b = build_metadata_half(2, &entries_b);
    half_a.resize(block_size, 0xFF);
    half_b.resize(block_size, 0xFF);
    dev.force_write(0, &half_a);
    dev.force_write(block_size as u64, &half_b);

    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let entries = fs.read_dir("/").unwrap();
    let names: alloc::vec::Vec<_> = entries.iter().map(|e| e.name.clone()).collect();
    assert!(
        names.contains(&"new".to_string()),
        "new file must be visible: {names:?}"
    );
    assert!(
        !names.contains(&"old".to_string()),
        "old file should NOT be visible"
    );
}

#[test]
fn t07_inline_file_round_trip() {
    let mut dev = fresh_device(8, 4096);
    let payload = b"hello, smallaios";
    let entries = [
        superblock_entry(2, 0, 4096, 8),
        inline_file_entry(1, "greet.txt", payload),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut file = fs.open_read("/greet.txt").unwrap();
    assert_eq!(file.size(), payload.len() as u32);
    assert!(file.is_inline());
    let mut buf = alloc::vec![0u8; payload.len()];
    let n = fs.read_file(&mut file, &mut buf).unwrap();
    assert_eq!(n, payload.len());
    assert_eq!(buf, payload);
}

#[test]
fn t08_inline_file_partial_read() {
    let mut dev = fresh_device(8, 4096);
    let payload = b"abcdefghij";
    let entries = [
        superblock_entry(2, 0, 4096, 8),
        inline_file_entry(1, "x", payload),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut file = fs.open_read("/x").unwrap();
    let mut buf = [0u8; 4];
    let n1 = fs.read_file(&mut file, &mut buf).unwrap();
    assert_eq!(n1, 4);
    assert_eq!(&buf, b"abcd");
    let n2 = fs.read_file(&mut file, &mut buf).unwrap();
    assert_eq!(n2, 4);
    assert_eq!(&buf, b"efgh");
    let mut tail = [0u8; 4];
    let n3 = fs.read_file(&mut file, &mut tail).unwrap();
    assert_eq!(n3, 2);
    assert_eq!(&tail[..2], b"ij");
    // EOF.
    let n4 = fs.read_file(&mut file, &mut tail).unwrap();
    assert_eq!(n4, 0);
}

#[test]
fn t09_open_read_missing_file_not_found() {
    let mut dev = fresh_device(8, 4096);
    let entries = [superblock_entry(2, 0, 4096, 8)];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let res = fs.open_read("/missing");
    assert!(matches!(res, Err(LittleFsError::NotFound)));
}

#[test]
fn t10_bad_path_no_leading_slash() {
    let mut dev = fresh_device(8, 4096);
    let entries = [superblock_entry(2, 0, 4096, 8)];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let res = fs.open_read("foo");
    assert!(matches!(res, Err(LittleFsError::BadPath)));
}

#[test]
fn t11_root_dir_lists_files() {
    let mut dev = fresh_device(8, 4096);
    let entries = [
        superblock_entry(2, 0, 4096, 8),
        inline_file_entry(1, "alpha", b"a"),
        inline_file_entry(2, "beta", b"b"),
        inline_file_entry(3, "gamma", b"g"),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listed = fs.read_dir("/").unwrap();
    let names: alloc::vec::Vec<_> = listed.iter().map(|e| e.name.clone()).collect();
    assert!(names.contains(&"alpha".to_string()));
    assert!(names.contains(&"beta".to_string()));
    assert!(names.contains(&"gamma".to_string()));
    // Superblock entry must not appear.
    assert!(!names.contains(&"littlefs".to_string()));
    for e in &listed {
        assert_eq!(e.kind, DirEntryKind::File);
    }
}

#[test]
fn t12_subdirectory_lookup() {
    let mut dev = fresh_device(16, 4096);
    // Sub-dir at pair (4, 5) holds one file.
    let sub_entries = [inline_file_entry(0, "leaf", b"leaf-bytes")];
    write_pair(&mut dev, [4, 5], 1, 1, &sub_entries);
    // Root pair (0, 1) holds a dir entry pointing at (4, 5).
    let root_entries = [
        superblock_entry(2, 0, 4096, 16),
        dir_entry(1, "sub", [4, 5]),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &root_entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let sub = fs.read_dir("/sub").unwrap();
    let names: alloc::vec::Vec<_> = sub.iter().map(|e| e.name.clone()).collect();
    assert_eq!(names, alloc::vec!["leaf".to_string()]);
}

#[test]
fn t13_subdirectory_file_read() {
    let mut dev = fresh_device(16, 4096);
    let sub_entries = [inline_file_entry(0, "leaf", b"deep-leaf")];
    write_pair(&mut dev, [4, 5], 1, 1, &sub_entries);
    let root_entries = [
        superblock_entry(2, 0, 4096, 16),
        dir_entry(1, "sub", [4, 5]),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &root_entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/sub/leaf").unwrap();
    let mut buf = alloc::vec![0u8; 9];
    let n = fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(n, 9);
    assert_eq!(&buf, b"deep-leaf");
}

#[test]
fn t14_dir_entries_show_kind() {
    let mut dev = fresh_device(16, 4096);
    let sub_entries: [EntrySpec; 0] = [];
    write_pair(&mut dev, [4, 5], 1, 1, &sub_entries);
    let root_entries = [
        superblock_entry(2, 0, 4096, 16),
        inline_file_entry(1, "afile", b"x"),
        dir_entry(2, "adir", [4, 5]),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &root_entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listed = fs.read_dir("/").unwrap();
    let by_name: alloc::collections::BTreeMap<_, _> =
        listed.iter().map(|e| (e.name.clone(), e.kind)).collect();
    assert_eq!(by_name.get("afile"), Some(&DirEntryKind::File));
    assert_eq!(by_name.get("adir"), Some(&DirEntryKind::Dir));
}

#[test]
fn t15_corrupt_crc_is_rejected() {
    let mut dev = fresh_device(8, 4096);
    let entries = [
        superblock_entry(2, 0, 4096, 8),
        inline_file_entry(1, "hello", b"bytes"),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    // Corrupt one byte in the middle of half A. We pick byte 50 (well
    // within the tag stream and CRC-protected). Half B remains valid
    // so the mount should still succeed using B.
    {
        let block_size = dev.block_size_bytes() as usize;
        let mut snap = dev.snapshot(0, block_size);
        snap[50] ^= 0xFF;
        dev.force_write(0, &snap);
    }
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).expect("half-B is still valid");
    let mut f = fs.open_read("/hello").unwrap();
    let mut buf = [0u8; 5];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(&buf, b"bytes");
}

#[test]
fn t16_both_halves_corrupt_errors() {
    let mut dev = fresh_device(8, 4096);
    let entries = [superblock_entry(2, 0, 4096, 8)];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    // Corrupt both halves.
    let block_size = dev.block_size_bytes() as usize;
    let mut a = dev.snapshot(0, block_size);
    a[20] ^= 0xFF;
    dev.force_write(0, &a);
    let mut b = dev.snapshot(block_size as u64, block_size);
    b[20] ^= 0xFF;
    dev.force_write(block_size as u64, &b);
    let cfg = LittleFsConfig::for_device(&dev);
    let res = LittleFs::mount(&mut dev, cfg);
    assert!(matches!(res, Err(LittleFsError::Corrupted)));
}

#[test]
fn t17_ctz_single_block_round_trip() {
    let mut dev = fresh_device(16, 4096);
    // CTZ block 0 holds the file payload (logical index 0 has full
    // payload = 4096 bytes available; size = 256 fits comfortably).
    let payload: alloc::vec::Vec<u8> = (0..256u32).map(|i| (i % 251) as u8).collect();
    let head_block: u32 = 4;
    dev.force_write(u64::from(head_block) * 4096, &payload);
    let entries = [
        superblock_entry(2, 0, 4096, 16),
        ctz_file_entry(1, "ctz1", head_block, payload.len() as u32),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/ctz1").unwrap();
    assert!(!f.is_inline());
    assert_eq!(f.size(), payload.len() as u32);
    let mut buf = alloc::vec![0u8; payload.len()];
    let n = fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(n, payload.len());
    assert_eq!(buf, payload);
}

#[test]
fn t18_ctz_multi_block_round_trip() {
    // Two-block chain: logical block 0 holds 4096 bytes, logical
    // block 1 holds (4096 - 4) = 4092 bytes; "by-1" pointer at the
    // last 4 bytes of block 1 points at block 0.
    let mut dev = fresh_device(32, 4096);
    let p0: alloc::vec::Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    let p1_payload: alloc::vec::Vec<u8> = (0..4092u32).map(|i| ((i + 7) % 251) as u8).collect();
    let block0_phys: u32 = 4;
    let block1_phys: u32 = 5; // chain head
    dev.force_write(u64::from(block0_phys) * 4096, &p0);
    let mut p1 = p1_payload.clone();
    p1.resize(4096, 0xFF);
    p1[4092..4096].copy_from_slice(&block0_phys.to_le_bytes());
    dev.force_write(u64::from(block1_phys) * 4096, &p1);

    let total_size = (p0.len() + p1_payload.len()) as u32;
    let entries = [
        superblock_entry(2, 0, 4096, 32),
        ctz_file_entry(1, "twob", block1_phys, total_size),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/twob").unwrap();
    assert_eq!(f.size(), total_size);
    let mut buf = alloc::vec![0u8; total_size as usize];
    let n = fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(n, total_size as usize);
    assert_eq!(&buf[..p0.len()], &p0[..]);
    assert_eq!(&buf[p0.len()..], &p1_payload[..]);
}

#[test]
fn t19_ctz_partial_read() {
    // Two-block chain like t18; verify reading just the second
    // logical block returns the expected slice.
    let mut dev = fresh_device(32, 4096);
    let p0: alloc::vec::Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    let p1_payload: alloc::vec::Vec<u8> = (0..4092u32).map(|i| ((i + 7) % 251) as u8).collect();
    let block0_phys: u32 = 4;
    let block1_phys: u32 = 5;
    dev.force_write(u64::from(block0_phys) * 4096, &p0);
    let mut p1 = p1_payload.clone();
    p1.resize(4096, 0xFF);
    p1[4092..4096].copy_from_slice(&block0_phys.to_le_bytes());
    dev.force_write(u64::from(block1_phys) * 4096, &p1);
    let total_size = (p0.len() + p1_payload.len()) as u32;
    let entries = [
        superblock_entry(2, 0, 4096, 32),
        ctz_file_entry(1, "twob", block1_phys, total_size),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/twob").unwrap();
    // Burn the first p0.len() bytes.
    let mut sink = alloc::vec![0u8; p0.len()];
    fs.read_file(&mut f, &mut sink).unwrap();
    // Read the trailing chunk.
    let mut tail = alloc::vec![0u8; p1_payload.len()];
    let n = fs.read_file(&mut f, &mut tail).unwrap();
    assert_eq!(n, p1_payload.len());
    assert_eq!(tail, p1_payload);
}

#[test]
fn t20_open_read_on_directory_errors() {
    let mut dev = fresh_device(16, 4096);
    let sub_entries: [EntrySpec; 0] = [];
    write_pair(&mut dev, [4, 5], 1, 1, &sub_entries);
    let entries = [
        superblock_entry(2, 0, 4096, 16),
        dir_entry(1, "subdir", [4, 5]),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let res = fs.open_read("/subdir");
    // We registered "subdir" as a dir-name, not a file-name; lookup
    // returns NotFound (because find_named_id requires the file-type
    // marker).
    assert!(matches!(res, Err(LittleFsError::NotFound)));
}

#[test]
fn t21_mount_then_remount_idempotent() {
    let mut dev = fresh_device(8, 4096);
    let entries = [
        superblock_entry(2, 0, 4096, 8),
        inline_file_entry(1, "stable", b"v"),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let _ = LittleFs::mount(&mut dev, cfg).unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/stable").unwrap();
    let mut buf = [0u8; 1];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(buf, *b"v");
}

#[test]
fn t22_block_size_mismatch_in_superblock() {
    // We can write a superblock that says block_size=8192 but the
    // device has 4096-byte blocks. The reader trusts the config it
    // was passed (which is the device's actual block_size). The
    // superblock's block_size field is informational here.
    let mut dev = fresh_device(8, 4096);
    let entries = [superblock_entry(2, 0, 8192, 4)];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let fs = LittleFs::mount(&mut dev, cfg).unwrap();
    assert_eq!(fs.superblock().block_size, 8192);
    assert_eq!(fs.config().block_size, 4096);
}

#[test]
fn t23_crc32c_helper_matches_classic_check() {
    // The CRC32C "check" value (string "123456789") is 0xe3069283.
    assert_eq!(crc32c(b"123456789"), 0xe306_9283);
}

#[test]
fn t24_zero_length_inline_file() {
    let mut dev = fresh_device(8, 4096);
    let entries = [
        superblock_entry(2, 0, 4096, 8),
        inline_file_entry(1, "empty", b""),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/empty").unwrap();
    assert_eq!(f.size(), 0);
    let mut buf = [0u8; 16];
    let n = fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(n, 0);
}

#[test]
fn t25_many_files_iteration_consistent() {
    let mut dev = fresh_device(16, 4096);
    let mut entries = alloc::vec![superblock_entry(2, 0, 4096, 16)];
    for i in 0..16u16 {
        let name = alloc::format!("file{i}");
        let body = alloc::format!("body-{i}");
        entries.push(inline_file_entry(i + 1, &name, body.as_bytes()));
    }
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listed = fs.read_dir("/").unwrap();
    assert_eq!(listed.len(), 16);
    for i in 0..16u16 {
        let name = alloc::format!("file{i}");
        assert!(listed.iter().any(|e| e.name == name), "missing {name}");
    }
    // Spot-check round-trip on one of them.
    let mut f = fs.open_read("/file7").unwrap();
    let mut buf = alloc::vec![0u8; 6];
    let n = fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(n, 6);
    assert_eq!(buf, b"body-7");
}

#[test]
fn t26_path_with_trailing_slash_resolves_to_dir() {
    let mut dev = fresh_device(16, 4096);
    let sub_entries = [inline_file_entry(0, "x", b"y")];
    write_pair(&mut dev, [4, 5], 1, 1, &sub_entries);
    let root_entries = [superblock_entry(2, 0, 4096, 16), dir_entry(1, "d", [4, 5])];
    write_pair(&mut dev, [0, 1], 1, 1, &root_entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listed = fs.read_dir("/d/").unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].name, "x");
}
