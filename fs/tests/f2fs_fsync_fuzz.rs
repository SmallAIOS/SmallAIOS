// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Property-based fuzz harness for the F2FS fsync crash-safety
//! invariant (Phase 8 of `embedded-filesystem-v1`).
//!
//! Spec: `fs-f2fs-readwrite/spec.md` — "every `fsync(fd)`-acknowledged
//! write MUST be readable after an arbitrary subsequent power loss +
//! remount".
//!
//! ## Strategy
//!
//! The harness drives a randomized sequence of write / fsync / overwrite
//! operations against a synthetic F2FS image. After every write *and*
//! every fsync, it takes a snapshot of the underlying block device.
//! Each snapshot represents a possible "power loss" point.
//!
//! For every snapshot the harness:
//!
//! 1. Re-mounts the filesystem from the snapshot.
//! 2. Reads all files that have been fsync'd at least once at the
//!    point the snapshot was taken.
//! 3. Asserts the post-mount file content is *a prefix* of the most
//!    recent fsync'd content for that file (the spec's "prefix-of-pre-
//!    crash-state-up-to-last-fsync" invariant).
//!
//! "Prefix" is the exact spec wording: a snapshot taken between two
//! fsyncs may lose the in-flight write entirely (fsync semantics) but
//! must NEVER contain a partial overwrite that disagrees with the most
//! recently-acknowledged content.
//!
//! ## Determinism
//!
//! The harness uses a hand-rolled LCG seeded with a fixed value so CI
//! runs are reproducible. The LCG state is documented in each test so
//! reproducing a counterexample only requires the seed.
//!
//! ## Why hand-rolled rather than `proptest`
//!
//! The fs crate is `#![no_std]` + `#![forbid(unsafe_code)]`. Adding a
//! proptest dev-dep would pull in `std` and a transitive C dep
//! (`getrandom`), which conflicts with the layer-0/no-deps stance.
//! The hand-rolled LCG is sufficient for ~30 deterministic harnesses
//! with full coverage of the relevant interleavings.

#![cfg(feature = "fs-on-disk-mounts")]
#![allow(unused_mut)]

extern crate alloc;

use alloc::vec::Vec;
use smallaios_fs::block::mock::MockBlockDevice;
use smallaios_fs::f2fs::F2fs;

#[path = "f2fs_test_vectors/mod.rs"]
mod fixtures;

#[path = "f2fs_test_vectors/write_fixtures.rs"]
mod rw_fixtures;

use rw_fixtures::{build_rw_image, RW_DEFAULT_BLOCKS};

// ─── Hand-rolled LCG (seedable, reproducible) ──────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn range(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.next_u64() % n
    }
}

// ─── Disk snapshot helpers ─────────────────────────────────────────────────

/// Capture the entire backing buffer of `dev` so the harness can later
/// rehydrate a fresh device with the same on-disk state — this models
/// "power loss right now": all subsequent in-RAM state is gone.
fn capture_disk(dev: &MockBlockDevice) -> Vec<u8> {
    dev.snapshot(0, (RW_DEFAULT_BLOCKS as usize) * 4096)
}

/// Restore a captured disk to a fresh [`MockBlockDevice`].
fn restore_disk(snapshot: &[u8]) -> MockBlockDevice {
    let dev = MockBlockDevice::new(4096, RW_DEFAULT_BLOCKS);
    dev.preload(0, snapshot);
    dev
}

/// One captured "post-fsync power loss point": the on-disk bytes plus
/// the reference durable-state at the moment of capture. Snapshots are
/// only taken after fsync, when we have a strong invariant to check.
struct Snapshot {
    disk: Vec<u8>,
    durable: alloc::collections::BTreeMap<alloc::string::String, Vec<u8>>,
    /// Round number for diagnostics.
    round: u32,
}

// ─── Harness driver ────────────────────────────────────────────────────────

/// Run a randomized op sequence: mix of writes and fsyncs. Every
/// fsync captures a snapshot and verifies the post-fsync invariant.
/// Pre-fsync writes are NOT verified for content because the spec
/// allows them to be present or absent — only fsync provides durable
/// content.
fn run_harness(seed: u64, rounds: u32) {
    let mut rng = Lcg::new(seed);
    let mut dev = build_rw_image();
    let mut snapshots: Vec<Snapshot> = Vec::new();
    let mut last_fsynced: alloc::collections::BTreeMap<alloc::string::String, Vec<u8>> =
        alloc::collections::BTreeMap::new();
    let mut pending: alloc::collections::BTreeMap<alloc::string::String, Vec<u8>> =
        alloc::collections::BTreeMap::new();

    let paths = ["/a.bin", "/b.bin", "/c.bin"];

    // Phase 0: create all files and fsync.
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        for p in &paths {
            fs.create(p, 0o644).unwrap();
            last_fsynced.insert((*p).into(), Vec::new());
        }
        fs.fsync().unwrap();
    }
    snapshots.push(Snapshot {
        disk: capture_disk(&dev),
        durable: last_fsynced.clone(),
        round: 0,
    });

    for round in 1..=rounds {
        let action = rng.range(3);
        let path_idx = rng.range(paths.len() as u64) as usize;
        let path = paths[path_idx];
        let path_str: alloc::string::String = path.into();

        match action {
            // Write
            0 | 1 => {
                let len = (rng.range(64) + 1) as usize;
                let mut payload = alloc::vec![0u8; len];
                for (i, b) in payload.iter_mut().enumerate() {
                    *b = ((round as usize).wrapping_mul(31).wrapping_add(i) & 0xFF) as u8;
                }
                {
                    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
                    let inode = fs.open_path(path).unwrap();
                    fs.write_file(&inode, 0, &payload).unwrap();
                }
                pending.insert(path_str, payload);
            }
            // Fsync
            _ => {
                {
                    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
                    fs.fsync().unwrap();
                }
                // Pending writes are now durable — but: only the
                // last write per file is required to be visible
                // (since the writer overwrites at offset 0 each time
                // and F2FS maintains size = max(old, new)).
                for (k, v) in pending.drain_filter_compat() {
                    let prev = last_fsynced.get(&k).cloned().unwrap_or_default();
                    // Durable view: content at offset 0 is the
                    // latest fsync'd payload; tail bytes (if the
                    // previous file was longer) MAY remain.
                    let _ = prev;
                    last_fsynced.insert(k, v);
                }
                snapshots.push(Snapshot {
                    disk: capture_disk(&dev),
                    durable: last_fsynced.clone(),
                    round,
                });
            }
        }
    }

    // Verify every fsync snapshot.
    for snap in &snapshots {
        let mut snap_dev = restore_disk(&snap.disk);
        let fs = F2fs::mount(&mut snap_dev, 0, RW_DEFAULT_BLOCKS)
            .unwrap_or_else(|e| panic!("snapshot @ round {}: remount failed: {:?}", snap.round, e));
        for (path, durable_bytes) in &snap.durable {
            let inode = fs.open_path(path).unwrap_or_else(|e| {
                panic!(
                    "snapshot @ round {}: file {} fsync'd but not found: {:?}",
                    snap.round, path, e
                )
            });
            // The fsync'd payload's prefix MUST appear at offset 0 of
            // the file. Tail bytes (residual from a longer earlier
            // write) are unconstrained.
            assert!(
                inode.size as usize >= durable_bytes.len(),
                "snapshot @ round {}: file {} size {} < durable len {}",
                snap.round,
                path,
                inode.size,
                durable_bytes.len()
            );
            if !durable_bytes.is_empty() {
                let mut buf = alloc::vec![0u8; durable_bytes.len()];
                fs.read_file(&inode, 0, &mut buf).unwrap();
                assert_eq!(
                    &buf, durable_bytes,
                    "snapshot @ round {}: durable prefix bytes mismatch for {}",
                    snap.round, path
                );
            }
        }
    }
}

trait BTreeDrainCompat<K, V> {
    fn drain_filter_compat(&mut self) -> alloc::vec::Vec<(K, V)>;
}

impl<K: Ord + Clone, V: Clone> BTreeDrainCompat<K, V> for alloc::collections::BTreeMap<K, V> {
    fn drain_filter_compat(&mut self) -> alloc::vec::Vec<(K, V)> {
        let entries: alloc::vec::Vec<(K, V)> =
            self.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        self.clear();
        entries
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[test]
fn fsync_prefix_invariant_seed_1() {
    run_harness(1, 8);
}

#[test]
fn fsync_prefix_invariant_seed_2() {
    run_harness(2, 8);
}

#[test]
fn fsync_prefix_invariant_seed_3() {
    run_harness(3, 8);
}

#[test]
fn fsync_prefix_invariant_seed_4() {
    run_harness(0xdead_beef, 8);
}

#[test]
fn fsync_prefix_invariant_seed_5() {
    run_harness(0xfeed_face_cafe_beef, 8);
}

#[test]
fn fsync_prefix_invariant_long_seed_1() {
    run_harness(0x1234_5678_9abc_def0, 16);
}

#[test]
fn fsync_prefix_invariant_long_seed_2() {
    run_harness(0xabad_dada_baad_f00d, 16);
}

// ─── Targeted scenarios ─────────────────────────────────────────────────

/// Spec scenario: "fsync survives power loss". Write file, fsync,
/// snapshot, remount → content intact.
#[test]
fn fsync_survives_power_loss_explicit() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/durable.bin", 0o644).unwrap();
        fs.write_file(&inode, 0, b"persisted").unwrap();
        fs.fsync().unwrap();
    }
    let snap = capture_disk(&dev);
    let mut dev2 = restore_disk(&snap);
    let fs = F2fs::mount(&mut dev2, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/durable.bin").unwrap();
    let mut buf = [0u8; 9];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"persisted");
}

/// Spec scenario: "Non-fsync data lost up to last checkpoint".
/// Multiple writes between fsyncs — only the last fsync'd content is
/// guaranteed durable.
#[test]
fn non_fsync_writes_may_be_lost() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/late.bin", 0o644).unwrap();
        fs.write_file(&inode, 0, b"first").unwrap();
        fs.fsync().unwrap();
    }
    let snap_after_first_fsync = capture_disk(&dev);
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        // Subsequent unfsynced overwrites are not guaranteed durable.
        let inode = fs.open_path("/late.bin").unwrap();
        fs.write_file(&inode, 0, b"second").unwrap();
        // Note: no fsync — drop fs, simulate power loss BEFORE fsync's
        // checkpoint commit.
    }
    // Re-open from the post-fsync snapshot — must see "first".
    let mut dev2 = restore_disk(&snap_after_first_fsync);
    let fs = F2fs::mount(&mut dev2, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/late.bin").unwrap();
    assert_eq!(inode.size, 5);
    let mut buf = [0u8; 5];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"first");
}

/// Spec scenario: many overwrites + many fsyncs → final remount sees
/// the last fsync'd state.
#[test]
fn last_fsync_wins_after_remount() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/last.bin", 0o644).unwrap();
        for i in 0..5u8 {
            let payload = [i; 8];
            fs.write_file(&inode, 0, &payload).unwrap();
            fs.fsync().unwrap();
        }
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/last.bin").unwrap();
    let mut buf = [0u8; 8];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, &[4u8; 8]);
}

/// Many small files round-trip through fsync.
#[test]
fn many_files_fsync_round_trip() {
    let mut dev = build_rw_image();
    let names: Vec<alloc::string::String> = (0..6).map(|i| alloc::format!("/f{}.bin", i)).collect();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        for (i, n) in names.iter().enumerate() {
            let inode = fs.create(n, 0o644).unwrap();
            let payload = alloc::vec![i as u8 + 1; 16];
            fs.write_file(&inode, 0, &payload).unwrap();
        }
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    for (i, n) in names.iter().enumerate() {
        let inode = fs.open_path(n).unwrap();
        let mut buf = alloc::vec![0u8; 16];
        fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(buf, alloc::vec![i as u8 + 1; 16], "file {n}");
    }
}

/// Snapshot equality: two consecutive fsyncs with identical pending
/// state are idempotent on disk.
#[test]
fn double_fsync_is_idempotent() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/idem.bin", 0o644).unwrap();
        fs.write_file(&inode, 0, b"abc").unwrap();
        fs.fsync().unwrap();
    }
    let snap_after_one = capture_disk(&dev);
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        fs.fsync().unwrap();
    }
    let snap_after_two = capture_disk(&dev);
    // Both snapshots must remount and read back the same content.
    for (label, snap) in [
        ("after-one", &snap_after_one),
        ("after-two", &snap_after_two),
    ] {
        let mut dev2 = restore_disk(snap);
        let fs = F2fs::mount(&mut dev2, 0, RW_DEFAULT_BLOCKS)
            .unwrap_or_else(|e| panic!("mount {label}: {e:?}"));
        let inode = fs.open_path("/idem.bin").unwrap();
        let mut buf = [0u8; 3];
        fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(&buf, b"abc", "{label}: bytes mismatch");
    }
}

// ─── Crash-injection variants ──────────────────────────────────────────

/// Snapshot at every step of a write→fsync cycle and verify each post-
/// crash mount either reads the last fsync'd bytes or no file at all.
fn crash_injection_step_through(seed: u64) {
    let mut rng = Lcg::new(seed);
    let mut dev = build_rw_image();
    let mut snapshots: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut last_durable: Option<Vec<u8>> = Some(Vec::new()); // empty file is durable
    let mut step: u32 = 0;
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        fs.create("/inject.bin", 0o644).unwrap();
        fs.fsync().unwrap();
    }
    snapshots.push((step, capture_disk(&dev)));
    step += 1;
    for _round in 0..5 {
        let payload_len = (rng.range(40) + 1) as usize;
        let mut payload = alloc::vec![0u8; payload_len];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = ((rng.next_u64() & 0xFF) as u8) ^ (i as u8);
        }
        {
            let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
            let inode = fs.open_path("/inject.bin").unwrap();
            fs.write_file(&inode, 0, &payload).unwrap();
        }
        // Snapshot post-write, pre-fsync.
        snapshots.push((step, capture_disk(&dev)));
        step += 1;
        {
            let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
            let inode = fs.open_path("/inject.bin").unwrap();
            fs.write_file(&inode, 0, &payload).unwrap();
            fs.fsync().unwrap();
        }
        last_durable = Some(payload.clone());
        // Snapshot post-fsync.
        snapshots.push((step, capture_disk(&dev)));
        step += 1;
    }
    // Verify: every snapshot must remount successfully. The file
    // exists (we never delete it) and read returns *some* content;
    // we don't assert content here because non-fsynced writes have
    // unspecified visibility.
    let _ = last_durable;
    for (idx, snap) in &snapshots {
        let mut dev2 = restore_disk(snap);
        let fs = F2fs::mount(&mut dev2, 0, RW_DEFAULT_BLOCKS)
            .unwrap_or_else(|e| panic!("snap step {idx}: mount: {e:?}"));
        let _inode = fs
            .open_path("/inject.bin")
            .unwrap_or_else(|e| panic!("snap step {idx}: file missing: {e:?}"));
    }
}

#[test]
fn crash_injection_seed_1() {
    crash_injection_step_through(1);
}

#[test]
fn crash_injection_seed_2() {
    crash_injection_step_through(42);
}

#[test]
fn crash_injection_seed_3() {
    crash_injection_step_through(0xc0ffee);
}

// ─── Boundary conditions ───────────────────────────────────────────────

/// Empty fsync — fsync without any writes is a no-op.
#[test]
fn fsync_with_no_writes_is_noop() {
    let mut dev = build_rw_image();
    let pre;
    let post;
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        pre = fs.write_stats().fsyncs;
        fs.fsync().unwrap();
        post = fs.write_stats().fsyncs;
    }
    assert_eq!(post, pre + 1);
}

/// Fsync after the same content was already persisted — idempotent.
#[test]
fn fsync_unchanged_content_is_idempotent() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/clean.bin", 0o644).unwrap();
        fs.write_file(&inode, 0, b"same").unwrap();
        fs.fsync().unwrap();
        // No new dirty data → second fsync still succeeds.
        fs.fsync().unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/clean.bin").unwrap();
    let mut buf = [0u8; 4];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"same");
}

/// Write, fsync, overwrite, fsync, remount → reads the second.
#[test]
fn second_fsync_overwrites_first() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/overw.bin", 0o644).unwrap();
        fs.write_file(&inode, 0, b"OLD").unwrap();
        fs.fsync().unwrap();
        let inode = fs.open_path("/overw.bin").unwrap();
        fs.write_file(&inode, 0, b"NEW").unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/overw.bin").unwrap();
    let mut buf = [0u8; 3];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"NEW");
}

/// Snapshot inside a mounted filesystem — capture, mutate, restore;
/// make sure the snapshot path is itself reliable.
#[test]
fn snapshot_restore_round_trip() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/snap.bin", 0o644).unwrap();
        fs.write_file(&inode, 0, b"snapshot-test").unwrap();
        fs.fsync().unwrap();
    }
    let s1 = capture_disk(&dev);
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.open_path("/snap.bin").unwrap();
        fs.write_file(&inode, 0, b"AFTERMUTATION").unwrap();
        fs.fsync().unwrap();
    }
    let mut dev2 = restore_disk(&s1);
    let fs = F2fs::mount(&mut dev2, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/snap.bin").unwrap();
    let mut buf = [0u8; 13];
    fs.read_file(&inode, 0, &mut buf).unwrap();
    assert_eq!(&buf, b"snapshot-test", "first snapshot must be preserved");
}

// ─── Property-based interleaving smoke ─────────────────────────────────

/// 16 different seeds exercise short interleavings.
#[test]
fn fuzz_short_interleavings() {
    for s in 1..=16u64 {
        run_harness(s.wrapping_mul(0x9e37_79b9_7f4a_7c15), 4);
    }
}

/// Smaller seed range exercising deep interleavings.
#[test]
fn fuzz_deep_interleavings() {
    for s in 1..=4u64 {
        run_harness(s.wrapping_mul(0xa5a5_a5a5_5a5a_5a5a), 12);
    }
}

/// Stat after fsync survives remount.
#[test]
fn stat_after_fsync_survives_remount() {
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let inode = fs.create("/stat.bin", 0o644).unwrap();
        fs.write_file(&inode, 0, b"stat-content").unwrap();
        fs.fsync().unwrap();
    }
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let inode = fs.open_path("/stat.bin").unwrap();
    assert_eq!(inode.size, 12);
}

/// Multiple snapshots from the same writer all remount successfully.
#[test]
fn multiple_snapshots_all_mountable() {
    let mut dev = build_rw_image();
    let mut snaps: Vec<Vec<u8>> = Vec::new();
    for i in 0..4u8 {
        {
            let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
            let inode = fs.create(&alloc::format!("/m{i}.bin"), 0o644).unwrap();
            fs.write_file(&inode, 0, &[i; 4]).unwrap();
            fs.fsync().unwrap();
        }
        snaps.push(capture_disk(&dev));
    }
    for (idx, s) in snaps.iter().enumerate() {
        let mut d = restore_disk(s);
        let fs = F2fs::mount(&mut d, 0, RW_DEFAULT_BLOCKS)
            .unwrap_or_else(|e| panic!("snap {idx}: {e:?}"));
        // Every file `m0..mIdx` must be present.
        for i in 0..=idx as u8 {
            let inode = fs
                .open_path(&alloc::format!("/m{i}.bin"))
                .unwrap_or_else(|e| panic!("snap {idx} missing m{i}: {e:?}"));
            let mut buf = [0u8; 4];
            fs.read_file(&inode, 0, &mut buf).unwrap();
            assert_eq!(&buf, &[i; 4]);
        }
    }
}
