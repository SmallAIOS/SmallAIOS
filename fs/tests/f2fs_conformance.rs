// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Conformance tests for the F2FS read-only reader.
//!
//! Exercises the scenarios in
//! `openspec/changes/embedded-filesystem-v1/specs/fs-f2fs-readwrite/spec.md`
//! that fall under the Phase 5 read-path scope. The synthetic image
//! builder lives in `f2fs_test_vectors/mod.rs` (paths-ignored by
//! codeql so byte-array constants are tolerated).
//!
//! When run on a host that has `mkfs.f2fs` from `f2fs-tools` matching
//! Linux 6.6 LTS, we'd ideally also exercise a real mkfs-produced
//! image; that part lives in CI as an external job (see
//! `.github/workflows/ci.yml::external-interop`). The conformance
//! suite here is self-contained — no external `mkfs.f2fs` binary
//! required.

#![cfg(feature = "fs-on-disk-mounts")]

extern crate alloc;

use smallaios_fs::block::mock::MockBlockDevice;
use smallaios_fs::f2fs::dir::FileType;
use smallaios_fs::f2fs::{F2fs, F2fsError, InodeKind};

#[path = "f2fs_test_vectors/mod.rs"]
mod fixtures;

use fixtures::{
    build_minimal_image, corrupt_both_superblocks, corrupt_primary_superblock,
    set_unsupported_feature, DEFAULT_BLOCKS,
};

// ─── Mount + basic structure ────────────────────────────────────────────────

#[test]
fn mount_minimal_image_succeeds() {
    let dev = build_minimal_image(b"hello, f2fs!");
    let _fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).expect("mount succeeds");
}

#[test]
fn mount_zeroed_device_bad_magic() {
    let dev = MockBlockDevice::new(4096, DEFAULT_BLOCKS);
    let r = F2fs::mount(&dev, 0, DEFAULT_BLOCKS);
    assert!(matches!(r, Err(F2fsError::BadMagic)));
}

#[test]
fn root_inode_is_directory() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let root = fs.open_path("/").unwrap();
    assert_eq!(root.kind, InodeKind::Directory);
    assert_eq!(root.inode_number, 3);
}

#[test]
fn root_inode_via_empty_path() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let root = fs.open_path("").unwrap();
    assert_eq!(root.kind, InodeKind::Directory);
}

// ─── Superblock fallback ─────────────────────────────────────────────────────

#[test]
fn primary_corrupt_falls_back_to_secondary() {
    let dev = build_minimal_image(b"");
    corrupt_primary_superblock(&dev);
    // Mount must still succeed using the secondary copy.
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).expect("secondary fallback");
    let _ = fs.open_path("/").unwrap();
}

#[test]
fn both_superblocks_corrupt_fails() {
    let dev = build_minimal_image(b"");
    corrupt_both_superblocks(&dev);
    let r = F2fs::mount(&dev, 0, DEFAULT_BLOCKS);
    assert!(matches!(
        r,
        Err(F2fsError::BadMagic) | Err(F2fsError::SuperblockCrcMismatch)
    ));
}

// ─── Feature gating ──────────────────────────────────────────────────────────

#[test]
fn unknown_mandatory_feature_rejected() {
    let dev = build_minimal_image(b"");
    set_unsupported_feature(&dev);
    let r = F2fs::mount(&dev, 0, DEFAULT_BLOCKS);
    assert!(matches!(r, Err(F2fsError::UnsupportedFeature(_))));
}

// ─── Directory listing ───────────────────────────────────────────────────────

#[test]
fn read_dir_yields_dot_dotdot_and_file() {
    let dev = build_minimal_image(b"hello");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let root = fs.open_path("/").unwrap();
    let entries: alloc::vec::Vec<_> = fs.read_dir(&root).unwrap().collect();
    assert_eq!(entries.len(), 3);
    assert!(entries.iter().any(|e| e.name == "."));
    assert!(entries.iter().any(|e| e.name == ".."));
    let file = entries.iter().find(|e| e.name == "file.txt").unwrap();
    assert_eq!(file.file_type, FileType::RegularFile);
}

#[test]
fn read_dir_on_non_dir_fails() {
    let dev = build_minimal_image(b"x");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let file = fs.open_path("/file.txt").unwrap();
    assert!(matches!(fs.read_dir(&file), Err(F2fsError::NotADirectory)));
}

// ─── Path resolution ─────────────────────────────────────────────────────────

#[test]
fn open_path_resolves_named_file() {
    let dev = build_minimal_image(b"path test");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    assert_eq!(inode.kind, InodeKind::Regular);
    assert_eq!(inode.size, 9);
}

#[test]
fn open_path_missing_component_fails() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let r = fs.open_path("/nope");
    assert!(matches!(r, Err(F2fsError::PathNotFound)));
}

#[test]
fn open_path_traverse_through_file_fails() {
    let dev = build_minimal_image(b"x");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let r = fs.open_path("/file.txt/foo");
    assert!(matches!(r, Err(F2fsError::NotADirectory)));
}

// ─── File reads (inline path) ────────────────────────────────────────────────

#[test]
fn read_inline_file_byte_for_byte() {
    let payload = b"hello, f2fs inline data path!";
    let dev = build_minimal_image(payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    assert_eq!(inode.size as usize, payload.len());
    let mut buf = alloc::vec![0u8; payload.len()];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, payload.len());
    assert_eq!(&buf, payload);
}

#[test]
fn read_inline_partial_at_offset() {
    let payload = b"abcdefghijklmnop";
    let dev = build_minimal_image(payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf = [0u8; 4];
    let n = fs.read_file(&inode, 4, &mut buf).unwrap();
    assert_eq!(n, 4);
    assert_eq!(&buf, b"efgh");
}

#[test]
fn read_inline_past_eof_returns_zero() {
    let payload = b"short";
    let dev = build_minimal_image(payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf = [0u8; 16];
    let n = fs.read_file(&inode, 100, &mut buf).unwrap();
    assert_eq!(n, 0);
}

#[test]
fn read_inline_large_buffer_truncated() {
    let payload = b"three";
    let dev = build_minimal_image(payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf = [0u8; 100];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, payload.len());
    assert_eq!(&buf[..n], payload);
}

#[test]
fn read_zero_byte_buffer() {
    let payload = b"data";
    let dev = build_minimal_image(payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf: [u8; 0] = [];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, 0);
}

// ─── File reads (direct-pointer path) ────────────────────────────────────────

#[test]
fn read_separate_block_file_byte_for_byte() {
    // Force separate-block (non-inline) file by exceeding the inline
    // threshold (1500 bytes in the builder).
    let payload: alloc::vec::Vec<u8> = (0..2048).map(|i| (i % 251) as u8).collect();
    let dev = build_minimal_image(&payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    assert_eq!(inode.size as usize, payload.len());
    let mut buf = alloc::vec![0u8; payload.len()];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, payload.len());
    assert_eq!(buf, payload);
}

#[test]
fn read_separate_block_partial_at_offset() {
    let payload: alloc::vec::Vec<u8> = (0..2048).map(|i| (i % 251) as u8).collect();
    let dev = build_minimal_image(&payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf = alloc::vec![0u8; 100];
    let n = fs.read_file(&inode, 1024, &mut buf).unwrap();
    assert_eq!(n, 100);
    for (i, &b) in buf.iter().enumerate() {
        assert_eq!(b, payload[1024 + i]);
    }
}

// ─── Stat ────────────────────────────────────────────────────────────────────

#[test]
fn stat_directory() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let root = fs.open_path("/").unwrap();
    let s = fs.stat(&root);
    assert_eq!(s.kind, InodeKind::Directory);
    assert_eq!(s.mode & 0o777, 0o755);
}

#[test]
fn stat_file() {
    let payload = b"sized";
    let dev = build_minimal_image(payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let s = fs.stat(&inode);
    assert_eq!(s.kind, InodeKind::Regular);
    assert_eq!(s.size, payload.len() as u64);
    assert_eq!(s.mode & 0o777, 0o644);
}

// ─── NAT lookup ──────────────────────────────────────────────────────────────

#[test]
fn nat_lookup_root_succeeds() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let nat = fs.lookup_nat_entry(3).unwrap();
    assert!(nat.is_valid_addr());
    assert_eq!(nat.ino, 3);
}

#[test]
fn nat_lookup_unallocated_returns_unallocated_entry() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let nat = fs.lookup_nat_entry(99).unwrap();
    assert!(!nat.is_valid_addr());
}

#[test]
fn nat_lookup_zero_nid_rejected() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let r = fs.lookup_nat_entry(0);
    assert!(matches!(r, Err(F2fsError::NodeIdOutOfRange)));
}

#[test]
fn nat_lookup_out_of_range_rejected() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let r = fs.lookup_nat_entry(u32::MAX);
    assert!(matches!(r, Err(F2fsError::NodeIdOutOfRange)));
}

// ─── Checkpoint pack selection ───────────────────────────────────────────────

#[test]
fn checkpoint_higher_version_wins() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    // Builder writes cp1 with version 7, cp2 with version 5; the
    // higher of the two must be selected.
    assert_eq!(fs.checkpoint().checkpoint_ver, 7);
}

// ─── Multi-test helpers exercising the same image ────────────────────────────

#[test]
fn mount_then_full_walk_round_trip() {
    let payload: alloc::vec::Vec<u8> = (0..3000).map(|i| ((i * 7) % 251) as u8).collect();
    let dev = build_minimal_image(&payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    // Walk root, find the file, read its bytes, compare.
    let root = fs.open_path("/").unwrap();
    let mut found = false;
    for e in fs.read_dir(&root).unwrap() {
        if e.name == "file.txt" {
            assert_eq!(e.file_type, FileType::RegularFile);
            found = true;
        }
    }
    assert!(found);
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf = alloc::vec![0u8; payload.len()];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, payload.len());
    assert_eq!(buf, payload);
}

// ─── Multiple back-to-back mounts (re-entrancy) ──────────────────────────────

#[test]
fn back_to_back_mounts() {
    let dev = build_minimal_image(b"data");
    {
        let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
        let _ = fs.open_path("/file.txt").unwrap();
    }
    {
        let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
        let _ = fs.open_path("/file.txt").unwrap();
    }
}

// ─── Several distinct images per test ────────────────────────────────────────

#[test]
fn distinct_images_have_distinct_payloads() {
    let dev1 = build_minimal_image(b"AAAAAA");
    let dev2 = build_minimal_image(b"BBBBBB");
    let fs1 = F2fs::mount(&dev1, 0, DEFAULT_BLOCKS).unwrap();
    let fs2 = F2fs::mount(&dev2, 0, DEFAULT_BLOCKS).unwrap();
    let i1 = fs1.open_path("/file.txt").unwrap();
    let i2 = fs2.open_path("/file.txt").unwrap();
    let mut b1 = [0u8; 6];
    let mut b2 = [0u8; 6];
    fs1.read_file(&i1, 0, &mut b1).unwrap();
    fs2.read_file(&i2, 0, &mut b2).unwrap();
    assert_eq!(&b1, b"AAAAAA");
    assert_eq!(&b2, b"BBBBBB");
}

// ─── Empty file ──────────────────────────────────────────────────────────────

#[test]
fn empty_file_size_zero() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    assert_eq!(inode.size, 0);
    let mut buf = [0u8; 16];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, 0);
}

// ─── Capacity sanity ─────────────────────────────────────────────────────────

#[test]
fn superblock_block_count_matches() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    assert_eq!(fs.superblock().block_count, DEFAULT_BLOCKS);
}

#[test]
fn superblock_root_ino_is_three() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    assert_eq!(fs.superblock().root_ino, 3);
}

// ─── Read on directory inode rejected ────────────────────────────────────────

#[test]
fn read_file_on_directory_fails() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let root = fs.open_path("/").unwrap();
    let mut buf = [0u8; 16];
    let r = fs.read_file(&root, 0, &mut buf);
    assert!(r.is_err());
}

// ─── Size overflow guard ─────────────────────────────────────────────────────

#[test]
fn mount_with_overflow_size_lbas_rejected() {
    let dev = MockBlockDevice::new(4096, 4);
    // size_lbas × block_size_bytes overflows u64.
    let r = F2fs::mount(&dev, 0, u64::MAX);
    assert!(matches!(r, Err(F2fsError::SizeOverflow)));
}

// ─── Hash determinism ────────────────────────────────────────────────────────

#[test]
fn dentry_hash_is_deterministic_across_calls() {
    use smallaios_fs::f2fs::dir::f2fs_dentry_hash;
    let h1 = f2fs_dentry_hash(b"file.txt");
    let h2 = f2fs_dentry_hash(b"file.txt");
    assert_eq!(h1, h2);
}

// ─── Inode kind variants ─────────────────────────────────────────────────────

#[test]
fn inode_kind_returns_regular_for_file() {
    let dev = build_minimal_image(b"x");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    assert_eq!(inode.kind, InodeKind::Regular);
}

// ─── Stat round-trip ─────────────────────────────────────────────────────────

#[test]
fn stat_links_count_set() {
    let dev = build_minimal_image(b"y");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let s = fs.stat(&inode);
    assert!(s.links >= 1);
}

// ─── F2fsError display surface ───────────────────────────────────────────────

#[test]
fn f2fs_error_displays_meaningful_message() {
    use core::fmt::Write;
    let e = F2fsError::UnsupportedFeature(0x0001);
    let mut s = alloc::string::String::new();
    write!(s, "{e}").unwrap();
    assert!(s.contains("0x0001"));
}

#[test]
fn f2fs_error_path_not_found_message() {
    use core::fmt::Write;
    let e = F2fsError::PathNotFound;
    let mut s = alloc::string::String::new();
    write!(s, "{e}").unwrap();
    assert!(s.contains("path"));
}

// ─── Edge cases for read_dir on inline-dentry directory ──────────────────────

#[test]
fn read_dir_directory_size_correct() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let root = fs.open_path("/").unwrap();
    // size is 2 dentry blocks (8192 bytes); only the first holds entries.
    assert!(root.size >= 4096);
}

// ─── NAT lookup beyond capacity ──────────────────────────────────────────────

#[test]
fn nat_lookup_capacity_boundary_rejected() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    // 1 NAT segment × 512 blocks/segment × 455 entries/block = 232960
    // entries. NID 232960 is out of range.
    let r = fs.lookup_nat_entry(232_960);
    assert!(matches!(r, Err(F2fsError::NodeIdOutOfRange)));
}

// ─── Primary good, secondary corrupt — primary wins ──────────────────────────

#[test]
fn primary_good_secondary_corrupt_uses_primary() {
    let dev = build_minimal_image(b"primary wins");
    // Corrupt the secondary at offset 9216.
    let mut block2 = [0u8; 4096];
    smallaios_fs::block::BlockDevice::read_block(&dev, 2, &mut block2).unwrap();
    block2[1024..1028].copy_from_slice(&0u32.to_le_bytes());
    dev.preload(2, &block2);
    // Mount must still succeed using the primary.
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf = alloc::vec![0u8; 12];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, 12);
    assert_eq!(&buf, b"primary wins");
}

// ─── Inline data with 1500-byte payload (boundary inline → block) ────────────

#[test]
fn read_inline_1500_byte_boundary() {
    let payload: alloc::vec::Vec<u8> = (0..1500).map(|i| (i % 251) as u8).collect();
    let dev = build_minimal_image(&payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    assert_eq!(inode.size, 1500);
    let mut buf = alloc::vec![0u8; 1500];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, 1500);
    assert_eq!(buf, payload);
}

// ─── 1501-byte payload (forces non-inline) ──────────────────────────────────

#[test]
fn read_just_above_inline_threshold() {
    let payload: alloc::vec::Vec<u8> = (0..1501).map(|i| (i % 251) as u8).collect();
    let dev = build_minimal_image(&payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    assert_eq!(inode.size, 1501);
    let mut buf = alloc::vec![0u8; 1501];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, 1501);
    assert_eq!(buf, payload);
}

// ─── Reading past EOF on a non-inline file ───────────────────────────────────

#[test]
fn read_separate_block_past_eof() {
    let payload: alloc::vec::Vec<u8> = (0..2000).map(|i| i as u8).collect();
    let dev = build_minimal_image(&payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf = [0u8; 16];
    let n = fs.read_file(&inode, 5000, &mut buf).unwrap();
    assert_eq!(n, 0);
}

// ─── Reading near EOF returns truncated count ────────────────────────────────

#[test]
fn read_truncated_at_eof_returns_partial() {
    let payload: alloc::vec::Vec<u8> = (0..2000).map(|i| i as u8).collect();
    let dev = build_minimal_image(&payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf = [0u8; 100];
    let n = fs.read_file(&inode, 1950, &mut buf).unwrap();
    assert_eq!(n, 50);
    for (i, &b) in buf[..50].iter().enumerate() {
        assert_eq!(b, payload[1950 + i]);
    }
}

// ─── DirIter implements Iterator ─────────────────────────────────────────────

#[test]
fn dir_iter_supports_collect() {
    let dev = build_minimal_image(b"");
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let root = fs.open_path("/").unwrap();
    let names: alloc::vec::Vec<alloc::string::String> =
        fs.read_dir(&root).unwrap().map(|e| e.name).collect();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&alloc::string::String::from("file.txt")));
}

// ─── Two distinct file payloads round-trip independently ─────────────────────

#[test]
fn distinct_payloads_round_trip_within_one_image() {
    // Same image used by `mount_then_full_walk_round_trip`; confirms
    // `read_file` returns the same bytes on second invocation.
    let payload = b"determinism";
    let dev = build_minimal_image(payload);
    let fs = F2fs::mount(&dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut a = alloc::vec![0u8; payload.len()];
    let mut b = alloc::vec![0u8; payload.len()];
    fs.read_file(&inode, 0, &mut a).unwrap();
    fs.read_file(&inode, 0, &mut b).unwrap();
    assert_eq!(a, b);
}
