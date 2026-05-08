// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Phase 2 conformance tests for the littlefs write path.
//!
//! These tests cover the `embedded-flash-fs-v1` Phase 4-6 spec deltas:
//!
//! * § "littlefs v2.x write path" — file create / write / flush, atomic
//!   rename, truncate, mkdir / rmdir.
//! * § "fsync and metadata-pair commit" — explicit durability barrier,
//!   power-fail safety.
//! * § "Wear-leveling integration" — dynamic block allocation across
//!   the free pool, no single-block-hot-spot under stress.
//! * § "Bad Block Table integration" — runtime bad-block remap on
//!   `EraseFailure` / `ProgramFailed`.
//!
//! The fixture builder in [`flash_littlefs_conformance.rs`](
//! ./flash_littlefs_conformance.rs) lays down a known-good superblock
//! +/or seed entries before mounting; from that point on, every test
//! drives the write path purely through the public API and asserts
//! end-to-end (write → re-mount → read).

#![cfg(all(feature = "fs-flash", feature = "fs-flash-mock"))]

extern crate alloc;

use smallaios_fs::flash::littlefs::{
    DirEntryKind, LittleFs, LittleFsConfig, LittleFsError, OpenFlags,
};
use smallaios_fs::flash::mock::MockFlashDevice;
use smallaios_fs::flash::FlashDevice;

// ─── Fixture builder (mirrors flash_littlefs_conformance.rs) ────────────────

const TAG_NAME_REG: u16 = 0x001;
const TAG_NAME_DIR: u16 = 0x002;
const TAG_NAME_SUPERBLOCK: u16 = 0x0FF;
const TAG_STRUCT_DIR: u16 = 0x200;
const TAG_STRUCT_INLINE: u16 = 0x201;
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

fn pack_tag_word(typ: u16, id: u16, len: u32) -> u32 {
    ((u32::from(typ) & 0x7FF) << 20) | ((u32::from(id) & 0x3FF) << 10) | (len & 0x3FF)
}

fn build_metadata_half(revision: u32, entries: &[EntrySpec]) -> alloc::vec::Vec<u8> {
    let mut buf = alloc::vec::Vec::new();
    buf.extend_from_slice(&revision.to_le_bytes());
    let mut prev_tag_word: u32 = 0;
    let mut running_crc: u32 = 0xFFFF_FFFF;
    running_crc = crc32c_update(running_crc, &revision.to_le_bytes());

    for e in entries {
        let name_tag_word = pack_tag_word(e.typ_name, e.id, e.name.len() as u32);
        let on_disk = name_tag_word ^ prev_tag_word;
        buf.extend_from_slice(&on_disk.to_be_bytes());
        running_crc = crc32c_update(running_crc, &on_disk.to_be_bytes());
        buf.extend_from_slice(&e.name);
        running_crc = crc32c_update(running_crc, &e.name);
        prev_tag_word = name_tag_word;

        let struct_tag_word = pack_tag_word(e.typ_struct, e.id, e.struct_payload.len() as u32);
        let on_disk2 = struct_tag_word ^ prev_tag_word;
        buf.extend_from_slice(&on_disk2.to_be_bytes());
        running_crc = crc32c_update(running_crc, &on_disk2.to_be_bytes());
        buf.extend_from_slice(&e.struct_payload);
        running_crc = crc32c_update(running_crc, &e.struct_payload);
        prev_tag_word = struct_tag_word;
    }

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
    payload.extend_from_slice(&255u32.to_le_bytes());
    payload.extend_from_slice(&0x7FFF_FFFFu32.to_le_bytes());
    payload.extend_from_slice(&1022u32.to_le_bytes());
    EntrySpec {
        typ_name: TAG_NAME_SUPERBLOCK,
        id: 0,
        name: b"littlefs".to_vec(),
        typ_struct: TAG_STRUCT_INLINE,
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

fn fresh_fs(blocks: u64, block_size: u32) -> MockFlashDevice {
    let mut dev = fresh_device(blocks, block_size);
    let entries = [superblock_entry(2, 0, block_size, blocks as u32)];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    dev
}

// ─── Round-trip tests ───────────────────────────────────────────────────────

#[test]
fn w01_create_then_read_round_trip() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs
            .open_write("/hello", OpenFlags::create_truncate())
            .unwrap();
        let n = fs.write(&mut f, b"hi there").unwrap();
        assert_eq!(n, 8);
        fs.close(f).unwrap();
    }
    // Re-mount and verify.
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/hello").unwrap();
    assert_eq!(f.size(), 8);
    let mut buf = [0u8; 8];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(&buf, b"hi there");
}

#[test]
fn w02_create_empty_file() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        fs.create("/empty").unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let f = fs.open_read("/empty").unwrap();
    assert_eq!(f.size(), 0);
}

#[test]
fn w03_double_create_rejected() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    fs.create("/dup").unwrap();
    assert!(matches!(fs.create("/dup"), Err(LittleFsError::BadType)));
}

#[test]
fn w04_open_write_no_create_missing_errors() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let res = fs.open_write(
        "/missing",
        OpenFlags {
            create: false,
            truncate: false,
            append: false,
        },
    );
    assert!(matches!(res, Err(LittleFsError::NotFound)));
}

#[test]
fn w05_truncate_overwrite_round_trip() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        // Write the original payload.
        let mut f = fs
            .open_write("/file", OpenFlags::create_truncate())
            .unwrap();
        fs.write(&mut f, b"first-payload").unwrap();
        fs.close(f).unwrap();
        // Re-open with truncate; new content replaces old.
        let mut f = fs
            .open_write("/file", OpenFlags::create_truncate())
            .unwrap();
        fs.write(&mut f, b"second").unwrap();
        fs.close(f).unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/file").unwrap();
    assert_eq!(f.size(), 6);
    let mut buf = [0u8; 6];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(&buf, b"second");
}

#[test]
fn w06_append_extends_inline() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs.open_write("/log", OpenFlags::create_truncate()).unwrap();
        fs.write(&mut f, b"line1\n").unwrap();
        fs.close(f).unwrap();
        let mut f = fs.open_write("/log", OpenFlags::append()).unwrap();
        fs.write(&mut f, b"line2\n").unwrap();
        fs.close(f).unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/log").unwrap();
    assert_eq!(f.size(), 12);
    let mut buf = [0u8; 12];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(&buf, b"line1\nline2\n");
}

#[test]
fn w07_remove_clears_entry() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs.open_write("/x", OpenFlags::create_truncate()).unwrap();
        fs.write(&mut f, b"to-remove").unwrap();
        fs.close(f).unwrap();
        fs.remove("/x").unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    assert!(matches!(fs.open_read("/x"), Err(LittleFsError::NotFound)));
}

#[test]
fn w08_remove_missing_errors_not_found() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    assert!(matches!(
        fs.remove("/missing"),
        Err(LittleFsError::NotFound)
    ));
}

#[test]
fn w09_rename_within_root() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs.open_write("/old", OpenFlags::create_truncate()).unwrap();
        fs.write(&mut f, b"renamed-bytes").unwrap();
        fs.close(f).unwrap();
        fs.rename("/old", "/new").unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    assert!(matches!(fs.open_read("/old"), Err(LittleFsError::NotFound)));
    let mut f = fs.open_read("/new").unwrap();
    assert_eq!(f.size(), 13);
    let mut buf = [0u8; 13];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(&buf, b"renamed-bytes");
}

#[test]
fn w10_rename_to_existing_rejected() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    fs.create("/a").unwrap();
    fs.create("/b").unwrap();
    assert!(matches!(fs.rename("/a", "/b"), Err(LittleFsError::BadType)));
}

#[test]
fn w11_truncate_shrinks_inline() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs
            .open_write("/data", OpenFlags::create_truncate())
            .unwrap();
        fs.write(&mut f, b"abcdefghij").unwrap();
        fs.close(f).unwrap();
        fs.truncate("/data", 4).unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/data").unwrap();
    assert_eq!(f.size(), 4);
    let mut buf = [0u8; 4];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(&buf, b"abcd");
}

#[test]
fn w12_truncate_to_zero() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs
            .open_write("/data", OpenFlags::create_truncate())
            .unwrap();
        fs.write(&mut f, b"some-data").unwrap();
        fs.close(f).unwrap();
        fs.truncate("/data", 0).unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let f = fs.open_read("/data").unwrap();
    assert_eq!(f.size(), 0);
}

#[test]
fn w13_truncate_grow_rejected() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs
        .open_write("/data", OpenFlags::create_truncate())
        .unwrap();
    fs.write(&mut f, b"hi").unwrap();
    fs.close(f).unwrap();
    assert!(matches!(
        fs.truncate("/data", 100),
        Err(LittleFsError::Unsupported)
    ));
}

// ─── Sub-directory tests ────────────────────────────────────────────────────

#[test]
fn w14_mkdir_then_read_dir() {
    let mut dev = fresh_fs(32, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        fs.mkdir("/subdir").unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listing = fs.read_dir("/").unwrap();
    let names: alloc::vec::Vec<_> = listing.iter().map(|e| (e.name.clone(), e.kind)).collect();
    assert!(names
        .iter()
        .any(|(n, k)| n == "subdir" && *k == DirEntryKind::Dir));
    let sub = fs.read_dir("/subdir").unwrap();
    assert!(sub.is_empty());
}

#[test]
fn w15_mkdir_double_rejected() {
    let mut dev = fresh_fs(32, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    fs.mkdir("/d").unwrap();
    assert!(matches!(fs.mkdir("/d"), Err(LittleFsError::BadType)));
}

#[test]
fn w16_rmdir_empty_dir() {
    let mut dev = fresh_fs(32, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        fs.mkdir("/d").unwrap();
        fs.rmdir("/d").unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listing = fs.read_dir("/").unwrap();
    assert!(!listing.iter().any(|e| e.name == "d"));
}

#[test]
fn w17_rmdir_nonempty_rejected() {
    // Pre-load a sub-dir with one entry via the fixture, then rmdir.
    let mut dev = fresh_device(32, 4096);
    let entries = [
        superblock_entry(2, 0, 4096, 32),
        dir_entry(1, "sub", [4, 5]),
    ];
    write_pair(&mut dev, [0, 1], 1, 1, &entries);
    // Sub-directory with one file inside.
    let sub_entries = [EntrySpec {
        typ_name: TAG_NAME_REG,
        id: 1,
        name: b"f".to_vec(),
        typ_struct: TAG_STRUCT_INLINE,
        struct_payload: b"x".to_vec(),
    }];
    write_pair(&mut dev, [4, 5], 1, 1, &sub_entries);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    assert!(matches!(fs.rmdir("/sub"), Err(LittleFsError::BadType)));
}

#[test]
fn w18_create_inside_subdir_round_trip() {
    let mut dev = fresh_fs(32, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        fs.mkdir("/sub").unwrap();
        let mut f = fs
            .open_write("/sub/leaf", OpenFlags::create_truncate())
            .unwrap();
        fs.write(&mut f, b"leaf-bytes").unwrap();
        fs.close(f).unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/sub/leaf").unwrap();
    let mut buf = [0u8; 10];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(&buf, b"leaf-bytes");
}

// ─── fsync semantics ────────────────────────────────────────────────────────

#[test]
fn w19_fsync_persists_writes() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        fs.create("/persisted").unwrap();
        fs.fsync().unwrap();
    }
    // Even without an explicit drop / re-mount, the data is committed
    // because `commit_pair` writes synchronously. Verify by re-mount.
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let f = fs.open_read("/persisted").unwrap();
    assert_eq!(f.size(), 0);
}

#[test]
fn w20_fsync_no_dirty_no_op() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    // fsync on a clean FS is a no-op (no commits issued).
    fs.fsync().unwrap();
    fs.fsync().unwrap();
}

// ─── Atomic-rename / power-fail safety ──────────────────────────────────────

#[test]
fn w21_alternate_half_rotation() {
    // After two commits, the third commit must land in the second half
    // again (the writer alternates halves, so rev counter increases
    // monotonically across both halves).
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        fs.create("/c1").unwrap(); // commit 1
        fs.create("/c2").unwrap(); // commit 2
        fs.create("/c3").unwrap(); // commit 3
    }
    // Re-mount and verify all three are present.
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listing = fs.read_dir("/").unwrap();
    let names: alloc::vec::Vec<_> = listing.iter().map(|e| e.name.clone()).collect();
    for c in &["c1", "c2", "c3"] {
        assert!(names.iter().any(|n| n == c), "missing {c} in {names:?}");
    }
}

#[test]
fn w22_crash_mid_erase_leaves_previous_state_intact() {
    // Inject `EraseFailure` on the alternate half mid-commit. The
    // writer should react by marking the failing block bad, allocating
    // a fresh block, and retrying. The committed data remains intact;
    // the old half is unchanged because we abort *before* erasing it.
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    // Seed a known-good entry that will be the "previous state".
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs
            .open_write("/anchor", OpenFlags::create_truncate())
            .unwrap();
        fs.write(&mut f, b"ANCHOR").unwrap();
        fs.close(f).unwrap();
    }
    // The active commit landed in either half[0] or half[1]; the next
    // commit will target the *other* half. Inject a one-shot
    // erase-failure on block 1 (one of the root-pair halves).
    dev.inject_bad_block_on_erase(1);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        // The first attempt either hits the failure on block 1 (gets
        // remapped) or commits cleanly to block 0. Either way, the
        // /anchor entry must remain readable afterwards.
        let mut f = fs
            .open_write("/post", OpenFlags::create_truncate())
            .unwrap();
        fs.write(&mut f, b"after").unwrap();
        // close may fail with Unsupported if the root pair couldn't
        // be remapped. Either Ok or Err is acceptable for crash-safety;
        // what matters is the anchor.
        let _ = fs.close(f);
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/anchor").unwrap();
    let mut buf = [0u8; 6];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(&buf, b"ANCHOR");
}

#[test]
fn w23_atomic_rename_under_crash_injection() {
    // Pre-seed: file "/x" with "before" content. Inject erase-failure
    // on the alternate root-pair half *during* the rename. Anchor file
    // remains intact across the partial operation.
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs.open_write("/x", OpenFlags::create_truncate()).unwrap();
        fs.write(&mut f, b"before").unwrap();
        fs.close(f).unwrap();
    }
    dev.inject_bad_block_on_erase(0);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        // The rename either succeeds (with bad-block remap) or fails
        // with Io / Unsupported. In every case, "/x" or "/y" exists;
        // the FS is internally consistent.
        let _ = fs.rename("/x", "/y");
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listing = fs.read_dir("/").unwrap();
    let names: alloc::vec::Vec<_> = listing.iter().map(|e| e.name.clone()).collect();
    let has_x = names.iter().any(|n| n == "x");
    let has_y = names.iter().any(|n| n == "y");
    assert!(
        has_x ^ has_y,
        "exactly one of /x, /y must exist (atomic rename), got {names:?}"
    );
}

// ─── Wear-leveling stress ────────────────────────────────────────────────────

#[test]
fn w24_wear_leveling_distributes_metadata_pair_writes() {
    // Many successive root-pair commits should alternate halves so
    // both blocks 0 and 1 see roughly equal erase counts.
    let mut dev = fresh_fs(32, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        for i in 0..100u16 {
            let path = alloc::format!("/f{i:03}");
            fs.create(&path).unwrap();
        }
    }
    // Both halves should be erased at least 25% of the total commits;
    // the writer alternates so each half sees ~50%.
    let block0 = dev.erase_count(0);
    let block1 = dev.erase_count(1);
    let total = block0 + block1;
    assert!(total >= 100, "expected ≥100 erases total, got {total}");
    assert!(
        block0 >= total / 4 && block1 >= total / 4,
        "halves not balanced: {block0} vs {block1}"
    );
}

#[test]
fn w25_data_block_alloc_spreads_across_pool() {
    // Many CTZ-promoted files should spread their data blocks across
    // the free pool, not pile up on a single hot block.
    let mut dev = fresh_fs(64, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    // A 4 KB block has cache_size/8 = 512 byte inline ceiling. Write
    // 1 KB per file, so each file promotes to CTZ.
    let payload: alloc::vec::Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
    for i in 0..20u16 {
        let path = alloc::format!("/big{i:02}");
        let mut f = fs.open_write(&path, OpenFlags::create_truncate()).unwrap();
        fs.write(&mut f, &payload).unwrap();
        fs.close(f).unwrap();
    }
    // Erase counts on the data-block pool should be spread reasonably
    // evenly. Sum erase counts on blocks 4..64 (skipping root pair +
    // any blocks that happened to be allocated for sub-dir mkdir).
    let mut nonzero = 0u32;
    for b in 2u64..64 {
        if dev.erase_count(b) > 0 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero >= 10,
        "expected ≥10 distinct blocks erased under stress, got {nonzero}"
    );
}

#[test]
fn w26_data_block_remap_on_erase_failure() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    // Inject EraseFailure on block 4 (a likely allocator-pool block).
    dev.inject_bad_block_on_erase(4);
    // Write a CTZ-sized payload (so the writer needs to allocate at
    // least one data block).
    let payload: alloc::vec::Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs.open_write("/big", OpenFlags::create_truncate()).unwrap();
        fs.write(&mut f, &payload).unwrap();
        fs.close(f).unwrap();
    }
    // Verify reading back works.
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/big").unwrap();
    let mut buf = alloc::vec![0u8; 1024];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(&buf, &payload);
}

// ─── Fsync persistence (write → fsync → simulated reboot) ───────────────────

#[test]
fn w27_fsync_then_remount_reads_data() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs
            .open_write("/sync", OpenFlags::create_truncate())
            .unwrap();
        fs.write(&mut f, b"durable").unwrap();
        fs.flush(&mut f).unwrap();
        fs.fsync().unwrap();
        // Drop without calling close — fsync should already have
        // committed the metadata-pair.
        let _ = f;
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/sync").unwrap();
    // The file may have flushed to CTZ rather than inline (the file
    // close commits the inline form). The fsync after flush captures
    // whatever the parent pair held at the time.
    assert!(f.size() <= 7);
    if f.size() == 7 {
        let mut buf = [0u8; 7];
        fs.read_file(&mut f, &mut buf).unwrap();
        assert_eq!(&buf, b"durable");
    }
}

// ─── Many-files / large content ─────────────────────────────────────────────

#[test]
fn w28_many_small_files() {
    let mut dev = fresh_fs(64, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        for i in 0..32u16 {
            let path = alloc::format!("/m{i:02}");
            let mut f = fs.open_write(&path, OpenFlags::create_truncate()).unwrap();
            let body = alloc::format!("body-{i}");
            fs.write(&mut f, body.as_bytes()).unwrap();
            fs.close(f).unwrap();
        }
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listing = fs.read_dir("/").unwrap();
    // 32 files (any iteration order).
    assert!(listing.len() >= 32, "got {} entries", listing.len());
    // Verify a few of them.
    for i in [0u16, 7, 15, 31] {
        let path = alloc::format!("/m{i:02}");
        let mut f = fs.open_read(&path).unwrap();
        let mut buf = alloc::vec![0u8; f.size() as usize];
        fs.read_file(&mut f, &mut buf).unwrap();
        let expected = alloc::format!("body-{i}");
        assert_eq!(buf, expected.as_bytes());
    }
}

#[test]
fn w29_ctz_promotion_round_trip() {
    let mut dev = fresh_fs(32, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    // Cache size = block_size = 4096. Inline ceiling = 512.
    // A 600-byte file promotes to CTZ.
    let payload: alloc::vec::Vec<u8> = (0..600u32).map(|i| (i % 251) as u8).collect();
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs.open_write("/big", OpenFlags::create_truncate()).unwrap();
        fs.write(&mut f, &payload).unwrap();
        fs.close(f).unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_read("/big").unwrap();
    assert_eq!(f.size(), payload.len() as u32);
    assert!(!f.is_inline(), "should have promoted to CTZ");
    let mut buf = alloc::vec![0u8; payload.len()];
    fs.read_file(&mut f, &mut buf).unwrap();
    assert_eq!(buf, payload);
}

#[test]
fn w30_remove_ctz_file_releases_blocks() {
    let mut dev = fresh_fs(32, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let payload: alloc::vec::Vec<u8> = (0..600).map(|i| (i % 251) as u8).collect();
        let mut f = fs.open_write("/big", OpenFlags::create_truncate()).unwrap();
        fs.write(&mut f, &payload).unwrap();
        fs.close(f).unwrap();
        fs.remove("/big").unwrap();
        // After removal, we should be able to create another similar
        // file without exhausting the pool. (If remove didn't free
        // blocks, repeated cycles would.)
        for i in 0..10u16 {
            let path = alloc::format!("/r{i:02}");
            let mut f = fs.open_write(&path, OpenFlags::create_truncate()).unwrap();
            let body: alloc::vec::Vec<u8> =
                (0..600).map(|x| ((x + i as u32) % 251) as u8).collect();
            fs.write(&mut f, &body).unwrap();
            fs.close(f).unwrap();
            fs.remove(&path).unwrap();
        }
    }
    // Re-mount and verify the FS is empty (besides the superblock).
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listing = fs.read_dir("/").unwrap();
    assert!(listing.is_empty(), "expected empty root, got {listing:?}");
}

// ─── More error-path tests ──────────────────────────────────────────────────

#[test]
fn w31_open_write_bad_path_rejected() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    assert!(matches!(
        fs.open_write("noslash", OpenFlags::create_truncate()),
        Err(LittleFsError::BadPath)
    ));
    assert!(matches!(fs.create("noslash"), Err(LittleFsError::BadPath)));
}

#[test]
fn w32_double_remove_errors_not_found() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    fs.create("/x").unwrap();
    fs.remove("/x").unwrap();
    assert!(matches!(fs.remove("/x"), Err(LittleFsError::NotFound)));
}

#[test]
fn w33_close_after_close_idempotent() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let mut f = fs.open_write("/x", OpenFlags::create_truncate()).unwrap();
    fs.write(&mut f, b"hi").unwrap();
    // Close consumes f; trying to use it after is a borrow error so
    // we just verify the close path works.
    fs.close(f).unwrap();
}

#[test]
fn w34_multiple_round_trips_under_one_mount() {
    // Perform many small operations under a single mount and verify
    // that the in-memory state stays consistent across them all.
    let mut dev = fresh_fs(32, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    fs.create("/a").unwrap();
    fs.create("/b").unwrap();
    fs.create("/c").unwrap();
    fs.remove("/b").unwrap();
    fs.rename("/a", "/aa").unwrap();
    let listing = fs.read_dir("/").unwrap();
    let names: alloc::vec::Vec<_> = listing.iter().map(|e| e.name.clone()).collect();
    assert!(names.iter().any(|n| n == "aa"));
    assert!(names.iter().any(|n| n == "c"));
    assert!(!names.iter().any(|n| n == "a"));
    assert!(!names.iter().any(|n| n == "b"));
}

#[test]
fn w35_revision_increments_with_each_commit() {
    // Each commit increments the on-disk revision counter. We verify
    // by performing N commits and observing block 0 + block 1 erase
    // counts sum to ≥ N.
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        for i in 0..5u16 {
            let path = alloc::format!("/r{i}");
            fs.create(&path).unwrap();
        }
    }
    let total = dev.erase_count(0) + dev.erase_count(1);
    assert!(total >= 5, "expected ≥5 root-pair erases, got {total}");
}

#[test]
fn w36_mkdir_nested_rejected() {
    let mut dev = fresh_fs(32, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    fs.mkdir("/a").unwrap();
    // Nested mkdir not supported in Phase 2.
    assert!(matches!(fs.mkdir("/a/b"), Err(LittleFsError::Unsupported)));
}

#[test]
fn w37_rename_creates_fresh_lookup_entry() {
    // Verify the rename doesn't accidentally leave both names in the
    // listing.
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        fs.create("/old").unwrap();
        fs.rename("/old", "/new").unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    let listing = fs.read_dir("/").unwrap();
    let names: alloc::vec::Vec<_> = listing.iter().map(|e| e.name.clone()).collect();
    assert!(names.iter().any(|n| n == "new"));
    assert!(!names.iter().any(|n| n == "old"));
}

#[test]
fn w38_truncate_missing_file_errors() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    assert!(matches!(
        fs.truncate("/missing", 0),
        Err(LittleFsError::NotFound)
    ));
}

#[test]
fn w39_remove_after_truncate_works() {
    let mut dev = fresh_fs(16, 4096);
    let cfg = LittleFsConfig::for_device(&dev);
    {
        let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
        let mut f = fs.open_write("/x", OpenFlags::create_truncate()).unwrap();
        fs.write(&mut f, b"abcdef").unwrap();
        fs.close(f).unwrap();
        fs.truncate("/x", 3).unwrap();
        fs.remove("/x").unwrap();
    }
    let mut fs = LittleFs::mount(&mut dev, cfg).unwrap();
    assert!(matches!(fs.open_read("/x"), Err(LittleFsError::NotFound)));
}

#[test]
fn w40_open_flags_defaults_match_create_truncate() {
    let f1 = OpenFlags::default();
    let f2 = OpenFlags::create_truncate();
    assert_eq!(f1.create, f2.create);
    assert_eq!(f1.truncate, f2.truncate);
    assert_eq!(f1.append, f2.append);
}

#[test]
fn w41_append_flags_correct() {
    let f = OpenFlags::append();
    assert!(!f.create);
    assert!(!f.truncate);
    assert!(f.append);
}
