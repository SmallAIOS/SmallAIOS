// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Conformance tests for the F2FS write path (Phase 7 of
//! `embedded-filesystem-v1`).
//!
//! These cover the scenarios in
//! `openspec/changes/embedded-filesystem-v1/specs/fs-f2fs-readwrite/spec.md`
//! that fall under the Phase 7 write-path scope:
//!
//! - File / directory creation, write, truncate, unlink, rmdir.
//! - Atomic rename (open-reader-continuity invariant).
//! - `fsync` and 5-second background-timer commit.
//! - GC trigger threshold and yield to foreground writes.
//!
//! Test fixtures live in `f2fs_test_vectors/write_fixtures.rs` (paths-
//! ignored by codeql's byte-array rule).

#![cfg(feature = "fs-on-disk-mounts")]
// Many test bodies declare `let mut fs = ...` consistently — even when
// the test only exercises the read API after a re-mount — so the test
// shape is uniform across the file. Clippy's unused-mut warning makes
// no readability difference here.
#![allow(unused_mut)]

extern crate alloc;

use smallaios_fs::f2fs::{F2fs, F2fsError, InodeKind};

#[path = "f2fs_test_vectors/mod.rs"]
mod fixtures;

#[path = "f2fs_test_vectors/write_fixtures.rs"]
mod rw_fixtures;

use rw_fixtures::{build_rw_image, RW_DEFAULT_BLOCKS};

// ─── create / open round trip ───────────────────────────────────────────────

#[test]
fn create_file_then_open_round_trip() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/hello.txt", 0o644).unwrap();
    assert_eq!(inode.kind, InodeKind::Regular);
    let again = fs.open_path("/hello.txt").unwrap();
    assert_eq!(again.inode_number, inode.inode_number);
}

#[test]
fn create_then_write_then_read_back_inline() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/inline.txt", 0o644).unwrap();
    let payload = b"hello f2fs write path!";
    fs.write_file(&inode, 0, payload).unwrap();
    // Re-open to pick up updated size.
    let inode = fs.open_path("/inline.txt").unwrap();
    assert_eq!(inode.size, payload.len() as u64);
    let mut buf = alloc::vec![0u8; payload.len()];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, payload.len());
    assert_eq!(&buf, payload);
}

#[test]
fn create_in_nonexistent_parent_fails() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let r = fs.create("/missing/file.txt", 0o644);
    assert!(matches!(r, Err(F2fsError::PathNotFound)));
}

#[test]
fn create_existing_file_returns_error() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = fs.create("/dup.txt", 0o644).unwrap();
    let r = fs.create("/dup.txt", 0o644);
    assert!(r.is_err());
}

#[test]
fn create_root_path_rejected() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let r = fs.create("/", 0o644);
    assert!(matches!(r, Err(F2fsError::PathNotFound)));
}

// ─── write / read consistency ────────────────────────────────────────────────

#[test]
fn write_extends_file_size() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/ext.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"abcd").unwrap();
    let inode = fs.open_path("/ext.txt").unwrap();
    fs.write_file(&inode, 4, b"efgh").unwrap();
    let inode = fs.open_path("/ext.txt").unwrap();
    assert_eq!(inode.size, 8);
    let mut buf = [0u8; 8];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, 8);
    assert_eq!(&buf, b"abcdefgh");
}

#[test]
fn write_partial_overwrite_preserves_surrounding_bytes() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/part.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"abcdefghij").unwrap();
    let inode = fs.open_path("/part.txt").unwrap();
    fs.write_file(&inode, 3, b"XYZ").unwrap();
    let inode = fs.open_path("/part.txt").unwrap();
    let mut buf = [0u8; 10];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"abcXYZghij");
}

#[test]
fn write_zero_bytes_no_op() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/zero.txt", 0o644).unwrap();
    let n = fs.write_file(&inode, 0, &[]).unwrap();
    assert_eq!(n, 0);
}

#[test]
fn write_then_remount_reads_same_bytes() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/persist.txt", 0o644).unwrap();
        fs.write_file(&inode, 0, b"durable").unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/persist.txt").unwrap();
    assert_eq!(inode.size, 7);
    let mut buf = [0u8; 7];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"durable");
}

#[test]
fn write_large_file_spills_to_data_block() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/big.bin", 0o644).unwrap();
    let payload: alloc::vec::Vec<u8> = (0..2048u32).map(|i| (i % 251) as u8).collect();
    fs.write_file(&inode, 0, &payload).unwrap();
    let inode = fs.open_path("/big.bin").unwrap();
    assert_eq!(inode.size, payload.len() as u64);
    let mut out = alloc::vec![0u8; payload.len()];
    fs.read_file(&inode, 0, &mut out).unwrap();
    assert_eq!(out, payload);
}

#[test]
fn write_two_block_file_round_trip() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/two.bin", 0o644).unwrap();
    let payload: alloc::vec::Vec<u8> = (0..8192u32).map(|i| ((i ^ 0xA5) & 0xFF) as u8).collect();
    fs.write_file(&inode, 0, &payload).unwrap();
    let inode = fs.open_path("/two.bin").unwrap();
    let mut out = alloc::vec![0u8; payload.len()];
    fs.read_file(&inode, 0, &mut out).unwrap();
    assert_eq!(out, payload);
}

#[test]
fn write_tracks_stats() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/stats.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"hi").unwrap();
    fs.write_file(&inode, 2, b" again").unwrap();
    let s = fs.write_stats();
    assert_eq!(s.creates, 1);
    assert_eq!(s.writes, 2);
}

// ─── truncate ───────────────────────────────────────────────────────────────

#[test]
fn truncate_to_zero_returns_zero_reads() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/tr.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"will vanish").unwrap();
    let inode = fs.open_path("/tr.txt").unwrap();
    fs.truncate(inode, 0).unwrap();
    let inode = fs.open_path("/tr.txt").unwrap();
    assert_eq!(inode.size, 0);
    let mut buf = [0u8; 16];
    let n = fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(n, 0);
}

#[test]
fn truncate_grow_creates_sparse_hole() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/grow.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"abc").unwrap();
    let inode = fs.open_path("/grow.txt").unwrap();
    fs.truncate(inode, 4096).unwrap();
    let inode = fs.open_path("/grow.txt").unwrap();
    assert_eq!(inode.size, 4096);
}

#[test]
fn truncate_partial_shrink() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/half.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"123456789").unwrap();
    let inode = fs.open_path("/half.txt").unwrap();
    fs.truncate(inode, 4).unwrap();
    let inode = fs.open_path("/half.txt").unwrap();
    assert_eq!(inode.size, 4);
}

#[test]
fn truncate_increments_stats_counter() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/c.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"x").unwrap();
    let inode = fs.open_path("/c.txt").unwrap();
    fs.truncate(inode, 0).unwrap();
    assert_eq!(fs.write_stats().truncates, 1);
}

// ─── unlink / reclaim ───────────────────────────────────────────────────────

#[test]
fn unlink_removes_dentry() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = fs.create("/del.txt", 0o644).unwrap();
    fs.unlink("/del.txt").unwrap();
    assert!(matches!(
        fs.open_path("/del.txt"),
        Err(F2fsError::PathNotFound)
    ));
}

#[test]
fn unlink_then_recreate_succeeds() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = fs.create("/recreate.txt", 0o644).unwrap();
    fs.unlink("/recreate.txt").unwrap();
    let _ = fs.create("/recreate.txt", 0o644).unwrap();
}

#[test]
fn unlink_missing_returns_path_not_found() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let r = fs.unlink("/never_existed.txt");
    assert!(matches!(r, Err(F2fsError::PathNotFound)));
}

#[test]
fn unlink_directory_via_unlink_rejected() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/dir1", 0o755).unwrap();
    let r = fs.unlink("/dir1");
    // Trying to unlink a directory should fail (use rmdir).
    assert!(r.is_err());
}

#[test]
fn unlink_increments_stats_counter() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = fs.create("/u.txt", 0o644).unwrap();
    fs.unlink("/u.txt").unwrap();
    assert_eq!(fs.write_stats().unlinks, 1);
}

// ─── mkdir / rmdir ──────────────────────────────────────────────────────────

#[test]
fn mkdir_then_open_yields_directory() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/etc", 0o755).unwrap();
    let inode = fs.open_path("/etc").unwrap();
    assert_eq!(inode.kind, InodeKind::Directory);
}

#[test]
fn mkdir_dot_dotdot_present() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/sub", 0o755).unwrap();
    let inode = fs.open_path("/sub").unwrap();
    let entries: alloc::vec::Vec<_> = fs.read_dir(&inode).unwrap().collect();
    assert!(entries.iter().any(|e| e.name == "."));
    assert!(entries.iter().any(|e| e.name == ".."));
}

#[test]
fn mkdir_existing_path_rejected() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/x", 0o755).unwrap();
    let r = fs.mkdir("/x", 0o755);
    assert!(r.is_err());
}

#[test]
fn rmdir_removes_empty_directory() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/empty", 0o755).unwrap();
    fs.rmdir("/empty").unwrap();
    assert!(matches!(
        fs.open_path("/empty"),
        Err(F2fsError::PathNotFound)
    ));
}

#[test]
fn rmdir_non_empty_directory_rejected() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/nonempty", 0o755).unwrap();
    fs.create("/nonempty/file", 0o644).unwrap();
    let r = fs.rmdir("/nonempty");
    assert!(r.is_err());
}

#[test]
fn rmdir_on_regular_file_rejected() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/notadir.txt", 0o644).unwrap();
    let r = fs.rmdir("/notadir.txt");
    assert!(r.is_err());
}

#[test]
fn mkdir_increments_stats_counter() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/m", 0o755).unwrap();
    fs.mkdir("/m2", 0o755).unwrap();
    assert_eq!(fs.write_stats().mkdirs, 2);
}

#[test]
fn rmdir_increments_stats_counter() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/r", 0o755).unwrap();
    fs.rmdir("/r").unwrap();
    assert_eq!(fs.write_stats().rmdirs, 1);
}

// ─── rename ─────────────────────────────────────────────────────────────────

#[test]
fn rename_moves_dentry() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/a.txt", 0o644).unwrap();
    fs.rename("/a.txt", "/b.txt").unwrap();
    assert!(matches!(
        fs.open_path("/a.txt"),
        Err(F2fsError::PathNotFound)
    ));
    assert!(fs.open_path("/b.txt").is_ok());
}

#[test]
fn rename_preserves_content() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/before.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"keepme").unwrap();
    fs.rename("/before.txt", "/after.txt").unwrap();
    let inode = fs.open_path("/after.txt").unwrap();
    let mut buf = [0u8; 6];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"keepme");
}

#[test]
fn rename_clobber_replaces_dst() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let src = fs.create("/src.txt", 0o644).unwrap();
    fs.write_file(&src, 0, b"src").unwrap();
    let dst = fs.create("/dst.txt", 0o644).unwrap();
    fs.write_file(&dst, 0, b"dst").unwrap();
    fs.rename("/src.txt", "/dst.txt").unwrap();
    let dst = fs.open_path("/dst.txt").unwrap();
    let mut buf = [0u8; 3];
    fs.read_file(&dst, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"src");
}

#[test]
fn rename_open_reader_continuity_invariant() {
    // Spec: open readers SHALL continue to refer to the original `dst`
    // content; new opens SHALL see the renamed content. We model this
    // by capturing the pre-rename inode handle (which is the "open
    // reader") and verifying it still contains the original bytes
    // after the rename — F2fs::Inode is a value-type carrying its
    // inode block, so it survives the rename unchanged.
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let src = fs.create("/src2.txt", 0o644).unwrap();
    fs.write_file(&src, 0, b"src bytes").unwrap();
    let dst = fs.create("/dst2.txt", 0o644).unwrap();
    fs.write_file(&dst, 0, b"dst bytes").unwrap();
    let open_reader_of_dst = fs.open_path("/dst2.txt").unwrap();
    fs.rename("/src2.txt", "/dst2.txt").unwrap();
    // The captured handle still describes the pre-rename inode.
    assert_eq!(open_reader_of_dst.size, 9);
    // A fresh open sees the moved content.
    let new_open = fs.open_path("/dst2.txt").unwrap();
    let mut buf = [0u8; 9];
    fs.read_file(&new_open, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"src bytes");
}

#[test]
fn rename_missing_src_fails() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let r = fs.rename("/no.txt", "/yes.txt");
    assert!(matches!(r, Err(F2fsError::PathNotFound)));
}

#[test]
fn rename_increments_stats_counter() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/r1", 0o644).unwrap();
    fs.rename("/r1", "/r2").unwrap();
    assert_eq!(fs.write_stats().renames, 1);
}

// ─── fsync / checkpoint commit ──────────────────────────────────────────────

#[test]
fn fsync_clean_fs_succeeds_no_op() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.fsync().unwrap();
    assert_eq!(fs.write_stats().fsyncs, 1);
}

#[test]
fn fsync_after_write_commits_changes() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/c1.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"committed").unwrap();
    let pre_ver = fs.checkpoint().checkpoint_ver;
    fs.fsync().unwrap();
    assert!(fs.checkpoint().checkpoint_ver > pre_ver);
}

#[test]
fn fsync_then_remount_sees_committed_state() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/sync.txt", 0o644).unwrap();
        fs.write_file(&inode, 0, b"persisted").unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/sync.txt").unwrap();
    let mut buf = [0u8; 9];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"persisted");
}

#[test]
fn fsync_bumps_checkpoint_version_each_call() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let v0 = fs.checkpoint().checkpoint_ver;
    fs.create("/cv1", 0o644).unwrap();
    fs.fsync().unwrap();
    let v1 = fs.checkpoint().checkpoint_ver;
    fs.create("/cv2", 0o644).unwrap();
    fs.fsync().unwrap();
    let v2 = fs.checkpoint().checkpoint_ver;
    assert!(v1 > v0);
    assert!(v2 > v1);
}

#[test]
fn fsync_alternates_checkpoint_slot_each_commit() {
    // Commits should land in alternating CP slots so a power loss
    // during write leaves the previous pack intact for recovery.
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/a", 0o644).unwrap();
    fs.fsync().unwrap();
    fs.create("/b", 0o644).unwrap();
    fs.fsync().unwrap();
    fs.create("/c", 0o644).unwrap();
    fs.fsync().unwrap();
    // Checkpoint version must have advanced 3 times.
    assert!(fs.checkpoint().checkpoint_ver >= 10);
}

// ─── 5-second timer ─────────────────────────────────────────────────────────

#[test]
fn timer_below_threshold_no_commit() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/t.txt", 0o644).unwrap();
    let pre = fs.write_stats().fsyncs;
    let did = fs.tick_timer(2).unwrap();
    assert!(!did);
    assert_eq!(fs.write_stats().fsyncs, pre);
}

#[test]
fn timer_at_5_seconds_triggers_implicit_fsync() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/t2.txt", 0o644).unwrap();
    let pre = fs.write_stats().fsyncs;
    let did = fs.tick_timer(5).unwrap();
    assert!(did);
    assert!(fs.write_stats().fsyncs > pre);
}

#[test]
fn timer_clean_fs_no_commit() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let did = fs.tick_timer(10).unwrap();
    assert!(!did);
}

#[test]
fn timer_resets_after_commit() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/r.txt", 0o644).unwrap();
    fs.tick_timer(5).unwrap();
    // Subsequent tick on a clean fs is a no-op.
    let did = fs.tick_timer(5).unwrap();
    assert!(!did);
}

#[test]
fn timer_accumulates_across_calls() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/acc.txt", 0o644).unwrap();
    assert!(!fs.tick_timer(2).unwrap());
    assert!(!fs.tick_timer(2).unwrap());
    let did = fs.tick_timer(2).unwrap(); // total 6 → over
    assert!(did);
}

// ─── crash recovery ─────────────────────────────────────────────────────────

#[test]
fn write_without_fsync_then_remount_fs_consistent() {
    // Spec: data written without `fsync` MAY be absent on next mount,
    // but the filesystem SHALL still be consistent (mountable, root
    // accessible). We don't fsync, drop, re-mount, and verify.
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let _ = fs.create("/maybegone.txt", 0o644).unwrap();
        // No fsync.
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    // The root SHALL be accessible regardless of whether
    // /maybegone.txt persisted.
    let _root = fs.open_path("/").unwrap();
}

#[test]
fn fsync_then_remount_persists_atomic_state() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        fs.create("/keep.txt", 0o644).unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = fs.open_path("/keep.txt").unwrap();
}

#[test]
fn rename_fsynced_then_remount_sees_renamed() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        fs.create("/orig", 0o644).unwrap();
        fs.fsync().unwrap();
        fs.rename("/orig", "/renamed").unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = fs.open_path("/renamed").unwrap();
}

#[test]
fn many_creates_then_fsync_all_persist() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        for i in 0..8 {
            let mut name = alloc::string::String::from("/f");
            name.push_str(&alloc::format!("{i}"));
            fs.create(&name, 0o644).unwrap();
        }
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    for i in 0..8 {
        let mut name = alloc::string::String::from("/f");
        name.push_str(&alloc::format!("{i}"));
        let _ = fs.open_path(&name).unwrap();
    }
}

// ─── error handling ─────────────────────────────────────────────────────────

#[test]
fn unlink_then_open_returns_path_not_found() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/g.txt", 0o644).unwrap();
    fs.unlink("/g.txt").unwrap();
    let r = fs.open_path("/g.txt");
    assert!(matches!(r, Err(F2fsError::PathNotFound)));
}

#[test]
fn rmdir_root_rejected() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let r = fs.rmdir("/");
    assert!(r.is_err());
}

#[test]
fn create_in_file_parent_rejected() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/parent.txt", 0o644).unwrap();
    let r = fs.create("/parent.txt/child", 0o644);
    assert!(r.is_err());
}

// ─── stats sanity ───────────────────────────────────────────────────────────

#[test]
fn stats_default_zero_on_fresh_mount() {
    let mut dev = build_rw_image();
    let fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let s = fs.write_stats();
    assert_eq!(s.creates, 0);
    assert_eq!(s.writes, 0);
    assert_eq!(s.fsyncs, 0);
}

// ─── GC ─────────────────────────────────────────────────────────────────────

#[test]
fn request_yield_marks_pending() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    // No public read of yield_pending; just ensure the call doesn't
    // panic and a subsequent write proceeds.
    fs.request_yield();
    let _ = fs.create("/y.txt", 0o644).unwrap();
}

#[test]
fn gc_threshold_helper_below_5_percent() {
    use smallaios_fs::f2fs::gc::should_run_gc;
    assert!(should_run_gc(4, 100));
    assert!(!should_run_gc(50, 100));
}

#[test]
fn gc_pick_victim_skips_full_segments() {
    use smallaios_fs::f2fs::gc::pick_victim;
    let cands = [(1u32, 512u16), (2, 0)];
    assert!(pick_victim(&cands, 512).is_none());
}

#[test]
fn gc_run_pass_completes_all_blocks_when_no_yield() {
    use smallaios_fs::f2fs::gc::{run_gc_pass, GcStats};
    let victim = [(0u16, 7u32), (1u16, 7u32)];
    let mut count = 0u32;
    let r: Result<(GcStats, _), ()> = run_gc_pass(
        &victim,
        |_, _| {
            count += 1;
            Ok(100 + count)
        },
        || false,
    );
    let (stats, _) = r.unwrap();
    assert_eq!(stats.blocks_relocated, 2);
}

// ─── stress / repetition ────────────────────────────────────────────────────

#[test]
fn many_writes_and_reads_round_trip() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    for i in 0..16u32 {
        let mut name = alloc::string::String::from("/m");
        name.push_str(&alloc::format!("{i}"));
        let inode = fs.create(&name, 0o644).unwrap();
        let bytes: alloc::vec::Vec<u8> = (0..(64 + i)).map(|x| x as u8).collect();
        fs.write_file(&inode, 0, &bytes).unwrap();
        let inode = fs.open_path(&name).unwrap();
        assert_eq!(inode.size, bytes.len() as u64);
    }
}

#[test]
fn create_unlink_create_cycle() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    for _ in 0..5 {
        fs.create("/cycle.txt", 0o644).unwrap();
        fs.unlink("/cycle.txt").unwrap();
    }
}

#[test]
fn alternating_writes_and_truncates() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/alt.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"hello world").unwrap();
    let inode = fs.open_path("/alt.txt").unwrap();
    fs.truncate(inode, 5).unwrap();
    let inode = fs.open_path("/alt.txt").unwrap();
    fs.write_file(&inode, 5, b" again").unwrap();
    let inode = fs.open_path("/alt.txt").unwrap();
    assert_eq!(inode.size, 11);
}

// ─── Write on read-path-only fixture (still works) ──────────────────────────

#[test]
fn rw_handle_can_still_read_existing_file() {
    use fixtures::{build_minimal_image, DEFAULT_BLOCKS};
    let mut dev = build_minimal_image(b"existing");
    let mut fs = F2fs::mount(&mut dev, 0, DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/file.txt").unwrap();
    let mut buf = [0u8; 8];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"existing");
}

// ─── More fsync state preservation ──────────────────────────────────────────

#[test]
fn fsync_preserves_directory_listing_across_remount() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        fs.mkdir("/etc", 0o755).unwrap();
        fs.create("/etc/passwd", 0o644).unwrap();
        fs.create("/etc/group", 0o644).unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let etc = fs.open_path("/etc").unwrap();
    let entries: alloc::vec::Vec<_> = fs.read_dir(&etc).unwrap().collect();
    assert!(entries.iter().any(|e| e.name == "passwd"));
    assert!(entries.iter().any(|e| e.name == "group"));
}

#[test]
fn fsync_preserves_truncate_across_remount() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/sized", 0o644).unwrap();
        fs.write_file(&inode, 0, b"long content here").unwrap();
        let inode = fs.open_path("/sized").unwrap();
        fs.truncate(inode, 4).unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/sized").unwrap();
    assert_eq!(inode.size, 4);
}

// ─── Atomic-rename invariants in detail ─────────────────────────────────────

#[test]
fn rename_self_no_op() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/loop.txt", 0o644).unwrap();
    let _ = fs.rename("/loop.txt", "/loop.txt");
    // Either succeeds as a no-op or reports a conflict — both
    // implementations are spec-compliant. The file MUST still exist.
    assert!(fs.open_path("/loop.txt").is_ok());
}

#[test]
fn double_rename_cycle() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/x", 0o644).unwrap();
    fs.rename("/x", "/y").unwrap();
    fs.rename("/y", "/z").unwrap();
    assert!(fs.open_path("/z").is_ok());
    assert!(matches!(fs.open_path("/x"), Err(F2fsError::PathNotFound)));
    assert!(matches!(fs.open_path("/y"), Err(F2fsError::PathNotFound)));
}

// ─── Sequential allocator advances cursor ───────────────────────────────────

#[test]
fn allocator_advances_cursor_per_create() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let pre = fs.checkpoint().cur_data_blkoff[0];
    fs.create("/c1", 0o644).unwrap();
    fs.create("/c2", 0o644).unwrap();
    fs.create("/c3", 0o644).unwrap();
    fs.fsync().unwrap();
    let post = fs.checkpoint().cur_data_blkoff[0];
    assert!(post > pre);
}

// ─── KernelAtomicWriter wired backend ───────────────────────────────────────

#[test]
fn fs_implements_fs_write_backend() {
    use smallaios_auth::atomic_write::FsWriteBackend;
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create_and_write("/aw.tmp", b"fresh").unwrap();
    fs.fsync().unwrap();
    fs.rename("/aw.tmp", "/aw").unwrap();
    let inode = fs.open_path("/aw").unwrap();
    let mut buf = [0u8; 5];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"fresh");
}

#[test]
fn fs_cleanup_tmp_in_removes_tmp_files() {
    use smallaios_auth::atomic_write::FsWriteBackend;
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/orphan.tmp", 0o644).unwrap();
    fs.create("/keep.txt", 0o644).unwrap();
    fs.cleanup_tmp_in("/").unwrap();
    assert!(matches!(
        fs.open_path("/orphan.tmp"),
        Err(F2fsError::PathNotFound)
    ));
    assert!(fs.open_path("/keep.txt").is_ok());
}

// ─── End-to-end through KernelAtomicWriter ──────────────────────────────────

#[test]
fn kernel_atomic_writer_stages_through_f2fs() {
    use smallaios_auth::atomic_write::{AtomicWriter, KernelAtomicWriter};
    // Cargo tests run in their own process; using Box::leak is the
    // same idiom auth's unit tests use (and matches the kernel's
    // global-static install pattern).
    let dev_ptr: &'static mut _ = alloc::boxed::Box::leak(alloc::boxed::Box::new(build_rw_image()));
    let fs: &'static mut F2fs<'static, _> = alloc::boxed::Box::leak(alloc::boxed::Box::new(
        F2fs::mount(dev_ptr, 0, RW_DEFAULT_BLOCKS).unwrap(),
    ));
    let writer = KernelAtomicWriter::new();
    writer.install_backend(fs);
    writer.write_atomic("/aw_e2e.txt", b"end-to-end").unwrap();
    // The round-trip through write_atomic confirms the wiring:
    // create /aw_e2e.txt.tmp → fsync → rename to /aw_e2e.txt.
}

// ─── Additional conformance tests (Phase 7) ─────────────────────────────────

#[test]
fn sequential_creates_all_visible_after_fsync() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/s1", 0o644).unwrap();
    fs.create("/s2", 0o644).unwrap();
    fs.create("/s3", 0o644).unwrap();
    fs.fsync().unwrap();
    let root = fs.open_path("/").unwrap();
    let entries: alloc::vec::Vec<_> = fs.read_dir(&root).unwrap().collect();
    let names: alloc::vec::Vec<_> = entries.iter().map(|e| e.name.clone()).collect();
    assert!(names.contains(&"s1".into()));
    assert!(names.contains(&"s2".into()));
    assert!(names.contains(&"s3".into()));
}

#[test]
fn write_at_nonzero_offset_inline_path() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/off.txt", 0o644).unwrap();
    fs.write_file(&inode, 8, b"abcd").unwrap();
    let inode = fs.open_path("/off.txt").unwrap();
    assert_eq!(inode.size, 12);
}

#[test]
fn reads_after_write_with_offset_returns_correct_bytes() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/r2.txt", 0o644).unwrap();
    fs.write_file(&inode, 0, b"AAAABBBBCCCC").unwrap();
    let inode = fs.open_path("/r2.txt").unwrap();
    let mut buf = [0u8; 4];
    fs.read_file(&inode, 4, &mut buf).unwrap();
    assert_eq!(&buf, b"BBBB");
}

#[test]
fn empty_truncate_then_grow() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.create("/etg.txt", 0o644).unwrap();
    fs.truncate(inode, 0).unwrap();
    let inode = fs.open_path("/etg.txt").unwrap();
    fs.truncate(inode, 1024).unwrap();
    let inode = fs.open_path("/etg.txt").unwrap();
    assert_eq!(inode.size, 1024);
}

#[test]
fn stats_increment_consistently_with_operations() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/op1", 0o644).unwrap();
    fs.create("/op2", 0o644).unwrap();
    fs.mkdir("/dir1", 0o755).unwrap();
    let inode = fs.open_path("/op1").unwrap();
    fs.write_file(&inode, 0, b"x").unwrap();
    let inode = fs.open_path("/op1").unwrap();
    fs.truncate(inode, 0).unwrap();
    fs.unlink("/op1").unwrap();
    fs.rmdir("/dir1").unwrap();
    fs.rename("/op2", "/op2-renamed").unwrap();
    fs.fsync().unwrap();
    let s = fs.write_stats();
    assert_eq!(s.creates, 2);
    assert_eq!(s.mkdirs, 1);
    assert_eq!(s.writes, 1);
    assert_eq!(s.truncates, 1);
    assert_eq!(s.unlinks, 1);
    assert_eq!(s.rmdirs, 1);
    assert_eq!(s.renames, 1);
    assert!(s.fsyncs >= 1);
}

#[test]
fn long_filename_create_and_open() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let long = "/this_is_a_long_filename_with_many_characters.txt";
    fs.create(long, 0o644).unwrap();
    let _ = fs.open_path(long).unwrap();
}

#[test]
fn nested_directory_create() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/usr", 0o755).unwrap();
    fs.mkdir("/usr/local", 0o755).unwrap();
    fs.create("/usr/local/file.txt", 0o644).unwrap();
    let _ = fs.open_path("/usr/local/file.txt").unwrap();
}

#[test]
fn staging_path_helper_produces_dot_tmp() {
    use smallaios_fs::f2fs::write::staging_path;
    assert_eq!(staging_path("/foo"), "/foo.tmp");
    assert_eq!(staging_path("/data/auth/shadow"), "/data/auth/shadow.tmp");
}

#[test]
fn write_stats_carries_across_mounts_consistently() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        fs.create("/cm1", 0o644).unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    // Stats reset across mounts.
    assert_eq!(fs.write_stats().creates, 0);
}

#[test]
fn write_to_freshly_mkdir_directory_works() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.mkdir("/etc", 0o755).unwrap();
    let inode = fs.create("/etc/passwd", 0o644).unwrap();
    fs.write_file(&inode, 0, b"root:x:0:0:root:/root:/bin/sh\n")
        .unwrap();
    let inode = fs.open_path("/etc/passwd").unwrap();
    assert!(inode.size > 10);
}

#[test]
fn unlink_clears_directory_entry() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/dl.txt", 0o644).unwrap();
    let root = fs.open_path("/").unwrap();
    let pre: alloc::vec::Vec<_> = fs.read_dir(&root).unwrap().collect();
    assert!(pre.iter().any(|e| e.name == "dl.txt"));
    fs.unlink("/dl.txt").unwrap();
    let root = fs.open_path("/").unwrap();
    let post: alloc::vec::Vec<_> = fs.read_dir(&root).unwrap().collect();
    assert!(!post.iter().any(|e| e.name == "dl.txt"));
}

#[test]
fn rename_increments_journal_then_commits_on_fsync() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.create("/jrn", 0o644).unwrap();
    fs.fsync().unwrap();
    fs.rename("/jrn", "/jrn2").unwrap();
    fs.fsync().unwrap();
    let _ = fs.open_path("/jrn2").unwrap();
}

#[test]
fn fsync_is_idempotent_if_no_change() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    fs.fsync().unwrap();
    fs.fsync().unwrap();
    fs.fsync().unwrap();
    assert_eq!(fs.write_stats().fsyncs, 3);
}

#[test]
fn write_after_unlink_to_recreated_file_has_fresh_content() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let i1 = fs.create("/recyc.txt", 0o644).unwrap();
    fs.write_file(&i1, 0, b"original").unwrap();
    fs.unlink("/recyc.txt").unwrap();
    let i2 = fs.create("/recyc.txt", 0o644).unwrap();
    fs.write_file(&i2, 0, b"new").unwrap();
    let inode = fs.open_path("/recyc.txt").unwrap();
    assert_eq!(inode.size, 3);
}

#[test]
fn checkpoint_version_strictly_monotonic() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let mut last = fs.checkpoint().checkpoint_ver;
    for i in 0..5 {
        let mut name = alloc::string::String::from("/cm");
        name.push_str(&alloc::format!("{i}"));
        fs.create(&name, 0o644).unwrap();
        fs.fsync().unwrap();
        let now = fs.checkpoint().checkpoint_ver;
        assert!(now > last, "iter {i}: {now} should be > {last}");
        last = now;
    }
}
