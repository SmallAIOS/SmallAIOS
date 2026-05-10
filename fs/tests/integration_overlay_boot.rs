// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the `embedded-overlay-v1` boot handlers
//! (`run_boot_cleanup`, `run_boot_conflict_scan`, first-boot
//! `models-upper/` directory creation) wired into the F2FS mount path.
//!
//! These exercise the surface introduced in the
//! `embedded-filesystem-v1-impl-integration` change:
//!
//! - `ensure_models_upper_dir` creates the upper-layer root with
//!   mode 0700 on first boot AND is a no-op on subsequent boots.
//! - `sweep_overlay_upper` removes every `<name>.tmp` orphan.
//! - `scan_overlay_conflicts` emits one audit event per new conflict.
//! - `run_overlay_boot_handlers` composes all three in canonical order.
//! - The audit-event projection on [`OverlayBootReport`] surfaces
//!   exactly one tag per observable change.
//!
//! Backed by [`MockWritableUpperLayer`] for the cleanup tests and the
//! capturing audit sink from [`crate::overlay`] for the conflict tests.
//! End-to-end tests with a real F2FS handle live in
//! `integration_overlay_boot_f2fs.rs` (this file uses the in-memory
//! mock to keep test latency low).

#![cfg(all(feature = "fs-on-disk-mounts", feature = "fs-overlay-mounts"))]
#![allow(unused_mut)]

extern crate alloc;

use alloc::string::ToString;
use alloc::vec::Vec;

use smallaios_fs::f2fs::F2fs;
use smallaios_fs::mount::{
    ensure_models_upper_dir, run_overlay_boot_handlers, scan_overlay_conflicts,
    sweep_overlay_upper, AUDIT_TAG_MODELS_UPPER_INITIALIZED, AUDIT_TAG_MODEL_ADD_ABORTED_RECOVERED,
};
use smallaios_fs::overlay::{
    hash_bytes, sha3_sidecar_path, CapturingConflictAuditSink, ConflictMemo, ConflictScanEvent,
    MapLowerManifest, MockWritableUpperLayer, OverlayConflict,
};

#[path = "f2fs_test_vectors/mod.rs"]
mod fixtures;

#[path = "f2fs_test_vectors/write_fixtures.rs"]
mod rw_fixtures;

use rw_fixtures::{build_rw_image, RW_DEFAULT_BLOCKS};

// ─── Helpers ────────────────────────────────────────────────────────────────

fn upper_with_files(entries: &[(&str, &[u8])]) -> MockWritableUpperLayer {
    let mut u = MockWritableUpperLayer::new();
    for (name, bytes) in entries {
        u.add_file(name, bytes);
    }
    u
}

fn upper_with_signed_files(entries: &[(&str, &[u8])]) -> MockWritableUpperLayer {
    let mut u = MockWritableUpperLayer::new();
    for (name, bytes) in entries {
        u.add_file(name, bytes);
        let digest = hash_bytes(bytes);
        let sc = smallaios_fs::overlay::encode_sidecar_bytes(&digest);
        u.add_file(&sha3_sidecar_path(name), &sc);
    }
    u
}

// ─── Boot cleanup: orphan `.tmp` removal ────────────────────────────────────

#[test]
fn sweep_empty_upper_returns_nothing_removed() {
    let mut u = MockWritableUpperLayer::new();
    let report = sweep_overlay_upper(&mut u);
    assert!(!report.any_removed());
    assert!(report.errors.is_empty());
}

#[test]
fn sweep_single_orphan_tmp_removed() {
    let mut u = upper_with_files(&[("foo.tmp", b"junk"), ("real.onnx", b"keep")]);
    let report = sweep_overlay_upper(&mut u);
    assert_eq!(report.count(), 1);
    assert_eq!(report.removed[0], "foo.tmp");
    assert!(!u.exists("foo.tmp"));
    assert!(u.exists("real.onnx"));
}

#[test]
fn sweep_multiple_orphans_removed() {
    let mut u = upper_with_files(&[("a.tmp", b"x"), ("b.tmp", b"y"), ("c.onnx", b"z")]);
    let report = sweep_overlay_upper(&mut u);
    assert_eq!(report.count(), 2);
    assert!(!u.exists("a.tmp"));
    assert!(!u.exists("b.tmp"));
    assert!(u.exists("c.onnx"));
}

#[test]
fn sweep_does_not_remove_sidecar_or_signature_files() {
    let mut u = upper_with_files(&[
        ("foo.onnx", b"keep"),
        ("foo.onnx.sha3", b"sidecar"),
        ("foo.onnx.sig", b"signature"),
        ("bar.whiteout", b""),
    ]);
    let report = sweep_overlay_upper(&mut u);
    assert!(!report.any_removed());
    assert!(u.exists("foo.onnx"));
    assert!(u.exists("foo.onnx.sha3"));
    assert!(u.exists("foo.onnx.sig"));
}

#[test]
fn sweep_idempotent_on_second_run() {
    let mut u = upper_with_files(&[("a.tmp", b"x")]);
    let r1 = sweep_overlay_upper(&mut u);
    assert_eq!(r1.count(), 1);
    let r2 = sweep_overlay_upper(&mut u);
    assert_eq!(r2.count(), 0, "second run is idempotent");
}

// ─── Boot conflict scan: audit emission ────────────────────────────────────

#[test]
fn conflict_scan_no_conflicts_emits_nothing() {
    let upper = upper_with_signed_files(&[("alpha.onnx", b"hello")]);
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let report = scan_overlay_conflicts(&upper, &lower, &memo, &mut audit).unwrap();
    assert_eq!(report.new_conflicts.len(), 0);
    assert!(audit.is_empty());
}

#[test]
fn conflict_scan_new_conflict_emits_one_audit_event() {
    let upper_bytes = b"upper-bytes";
    let upper = upper_with_signed_files(&[("conflict.onnx", upper_bytes)]);
    let mut lower = MapLowerManifest::new();
    let lower_digest = hash_bytes(b"lower-bytes");
    lower.insert("conflict.onnx", lower_digest);

    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let report = scan_overlay_conflicts(&upper, &lower, &memo, &mut audit).unwrap();

    assert_eq!(report.new_conflicts.len(), 1);
    assert_eq!(audit.conflict_count(), 1);
    match &audit.events[0] {
        ConflictScanEvent::Conflict {
            name,
            lower_sha3,
            upper_sha3,
        } => {
            assert_eq!(name, "conflict.onnx");
            assert_eq!(*lower_sha3, lower_digest);
            assert_eq!(*upper_sha3, hash_bytes(upper_bytes));
        }
        other => panic!("expected Conflict event, got {other:?}"),
    }
}

#[test]
fn conflict_scan_emits_warning_for_missing_sidecar() {
    let mut u = MockWritableUpperLayer::new();
    u.add_file("noside.onnx", b"upper");
    // No sidecar → MissingFingerprint warning.
    let mut lower = MapLowerManifest::new();
    lower.insert("noside.onnx", hash_bytes(b"lower"));
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let _ = scan_overlay_conflicts(&u, &lower, &memo, &mut audit).unwrap();
    let warns: Vec<_> = audit
        .events
        .iter()
        .filter(|e| matches!(e, ConflictScanEvent::Warning { .. }))
        .collect();
    assert!(!warns.is_empty(), "expected at least one warning event");
}

#[test]
fn conflict_scan_pre_existing_conflict_suppressed_by_memo() {
    let upper_bytes = b"upper-bytes";
    let upper = upper_with_signed_files(&[("c.onnx", upper_bytes)]);
    let mut lower = MapLowerManifest::new();
    let lower_digest = hash_bytes(b"lower-bytes");
    lower.insert("c.onnx", lower_digest);
    let upper_digest = hash_bytes(upper_bytes);

    let mut memo = ConflictMemo::new();
    memo.record(&OverlayConflict {
        name: "c.onnx".to_string(),
        lower_sha3: lower_digest,
        upper_sha3: upper_digest,
    });

    let mut audit = CapturingConflictAuditSink::new();
    let _ = scan_overlay_conflicts(&upper, &lower, &memo, &mut audit).unwrap();
    assert_eq!(audit.conflict_count(), 0);
}

#[test]
fn conflict_scan_multiple_conflicts_emit_one_event_each() {
    let upper = upper_with_signed_files(&[
        ("alpha.onnx", b"u_alpha"),
        ("beta.onnx", b"u_beta"),
        ("gamma.onnx", b"u_gamma"),
    ]);
    let mut lower = MapLowerManifest::new();
    lower.insert("alpha.onnx", hash_bytes(b"l_alpha"));
    lower.insert("beta.onnx", hash_bytes(b"l_beta"));
    lower.insert("gamma.onnx", hash_bytes(b"l_gamma"));
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let report = scan_overlay_conflicts(&upper, &lower, &memo, &mut audit).unwrap();
    assert_eq!(report.new_conflicts.len(), 3);
    assert_eq!(audit.conflict_count(), 3);
}

#[test]
fn conflict_scan_emits_no_event_when_lower_lacks_name() {
    let upper = upper_with_signed_files(&[("only_in_upper.onnx", b"data")]);
    let lower = MapLowerManifest::new(); // empty
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let report = scan_overlay_conflicts(&upper, &lower, &memo, &mut audit).unwrap();
    assert_eq!(report.new_conflicts.len(), 0);
}

// ─── First-boot models-upper directory creation ────────────────────────────

#[test]
fn ensure_models_upper_creates_dir_on_first_boot() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let initialized = ensure_models_upper_dir(&mut fs).unwrap();
    assert!(initialized);
    // Subsequent open succeeds.
    let inode = fs.open_path("/data/models-upper").unwrap();
    assert!(matches!(
        inode.kind,
        smallaios_fs::f2fs::InodeKind::Directory
    ));
}

#[test]
fn ensure_models_upper_uses_mode_0700() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    let inode = fs.open_path("/data/models-upper").unwrap();
    let stat = fs.stat(&inode);
    // Lower 9 bits hold the rwx triplets — assert top-of-7-bit pattern
    // matches 0o700 (no group/other bits set).
    assert_eq!(
        stat.mode & 0o077,
        0,
        "models-upper SHALL NOT be group/other-readable, got mode = {:o}",
        stat.mode
    );
    assert_eq!(stat.mode & 0o700, 0o700, "owner-rwx must be set");
}

#[test]
fn ensure_models_upper_idempotent_on_second_boot() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let init1 = ensure_models_upper_dir(&mut fs).unwrap();
    assert!(init1, "first call creates");
    let init2 = ensure_models_upper_dir(&mut fs).unwrap();
    assert!(!init2, "second call is no-op");
}

#[test]
fn ensure_models_upper_creates_data_parent_if_missing() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    // Pristine fixture has no /data dir; ensure_models_upper must
    // bootstrap it as part of the upper-layer init.
    assert!(fs.open_path("/data").is_err());
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    assert!(fs.open_path("/data").is_ok());
    assert!(fs.open_path("/data/models-upper").is_ok());
}

// ─── End-to-end overlay boot handlers ──────────────────────────────────────

#[test]
fn run_overlay_boot_handlers_first_boot_initializes_upper() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (boot_report, _scan) =
        run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert!(boot_report.upper_initialized);
    assert!(boot_report.cleanup.removed.is_empty());
    assert!(boot_report.aborted_recovered.is_empty());
}

#[test]
fn run_overlay_boot_handlers_second_boot_no_init() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (r1, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert!(r1.upper_initialized);
    let mut audit2 = CapturingConflictAuditSink::new();
    let (r2, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit2).unwrap();
    assert!(!r2.upper_initialized, "second boot is idempotent");
    assert!(audit2.is_empty(), "no new audit events on stable boot");
}

#[test]
fn run_overlay_boot_handlers_sweeps_orphan_tmp_files() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    // Bootstrap the upper directory + drop a synthetic orphan into it.
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    let inode = fs.create("/data/models-upper/foo.tmp", 0o644).unwrap();
    fs.write_file(&inode, 0, b"orphan").unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (boot_report, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert_eq!(boot_report.cleanup.count(), 1);
    assert_eq!(boot_report.cleanup.removed[0], "foo.tmp");
    assert!(fs.open_path("/data/models-upper/foo.tmp").is_err());
}

#[test]
fn run_overlay_boot_handlers_emits_audit_for_recovered_orphan() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    let inode = fs.create("/data/models-upper/x.tmp", 0o644).unwrap();
    fs.write_file(&inode, 0, b"x").unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (boot_report, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    let events = boot_report.pending_audit_events();
    let recovered: Vec<_> = events
        .iter()
        .filter(|(t, _)| *t == AUDIT_TAG_MODEL_ADD_ABORTED_RECOVERED)
        .collect();
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].1, "x.tmp");
}

#[test]
fn run_overlay_boot_handlers_emits_audit_for_first_boot_init() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (boot_report, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    let events = boot_report.pending_audit_events();
    assert!(
        events
            .iter()
            .any(|(t, _)| *t == AUDIT_TAG_MODELS_UPPER_INITIALIZED),
        "first boot SHALL emit data_models_upper_initialized"
    );
}

#[test]
fn run_overlay_boot_handlers_clean_state_is_clean() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (boot_report, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert!(boot_report.is_clean(), "no orphans, no init, no errors");
    assert!(boot_report.pending_audit_events().is_empty());
}

#[test]
fn run_overlay_boot_handlers_does_not_emit_for_non_conflict_lower() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    let mut lower = MapLowerManifest::new();
    lower.insert("not-shadowed.onnx", hash_bytes(b"lower-only"));
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (_, scan_report) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert_eq!(scan_report.new_conflicts.len(), 0);
}

// ─── Audit-event projection on OverlayBootReport ───────────────────────────

#[test]
fn pending_audit_events_empty_for_clean_run() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (boot_report, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert!(boot_report.pending_audit_events().is_empty());
}

#[test]
fn pending_audit_events_includes_init_and_recoveries() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    // First-boot upper-init is what we want to verify, so this test
    // does NOT pre-create the dir. We do still need to drop an orphan
    // there ahead of the first boot pass — sneak it in by creating
    // the dir, dropping the orphan, then unlinking the dir is not
    // possible (rmdir blocks on non-empty). Instead, we run the boot
    // handler twice: the first creates the dir, the second observes
    // an inserted orphan we drop after.
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit1 = CapturingConflictAuditSink::new();
    let (r1, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit1).unwrap();
    assert!(r1.upper_initialized);
    // Drop a synthetic orphan now.
    let inode = fs.create("/data/models-upper/orphan.tmp", 0o644).unwrap();
    fs.write_file(&inode, 0, b"x").unwrap();
    let mut audit2 = CapturingConflictAuditSink::new();
    let (r2, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit2).unwrap();
    assert!(!r2.upper_initialized, "second boot does not re-init");
    let events = r2.pending_audit_events();
    assert!(events
        .iter()
        .any(|(t, n)| *t == AUDIT_TAG_MODEL_ADD_ABORTED_RECOVERED && n == "orphan.tmp"));
}

// ─── Multiple orphans in one boot ──────────────────────────────────────────

#[test]
fn boot_handlers_recover_multiple_orphans_in_single_pass() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    for i in 0..5 {
        let path = alloc::format!("/data/models-upper/leak{i}.tmp");
        let inode = fs.create(&path, 0o644).unwrap();
        fs.write_file(&inode, 0, &[i as u8; 16]).unwrap();
    }
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (r, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert_eq!(r.cleanup.count(), 5);
    let events = r.pending_audit_events();
    let recovered: Vec<_> = events
        .iter()
        .filter(|(t, _)| *t == AUDIT_TAG_MODEL_ADD_ABORTED_RECOVERED)
        .collect();
    assert_eq!(recovered.len(), 5);
}

#[test]
fn boot_handlers_real_sweep_observed_via_open_path() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    let inode = fs.create("/data/models-upper/zap.tmp", 0o644).unwrap();
    fs.write_file(&inode, 0, b"zap").unwrap();
    assert!(fs.open_path("/data/models-upper/zap.tmp").is_ok());
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let _ = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert!(fs.open_path("/data/models-upper/zap.tmp").is_err());
}

// ─── Conflict scan on real F2FS-backed upper ───────────────────────────────

#[test]
fn boot_handlers_emit_conflict_for_real_f2fs_upper_shadowing_lower() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    // Drop a real upper entry + its sidecar.
    let upper_bytes: Vec<u8> = b"upper-payload".to_vec();
    let upper_inode = fs
        .create("/data/models-upper/shadowed.onnx", 0o644)
        .unwrap();
    fs.write_file(&upper_inode, 0, &upper_bytes).unwrap();
    let upper_digest = hash_bytes(&upper_bytes);
    let sidecar = smallaios_fs::overlay::encode_sidecar_bytes(&upper_digest);
    let sidecar_inode = fs
        .create("/data/models-upper/shadowed.onnx.sha3", 0o644)
        .unwrap();
    fs.write_file(&sidecar_inode, 0, &sidecar).unwrap();

    let mut lower = MapLowerManifest::new();
    lower.insert("shadowed.onnx", hash_bytes(b"lower-payload"));
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (_, scan) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert_eq!(scan.new_conflicts.len(), 1);
    assert_eq!(scan.new_conflicts[0].name, "shadowed.onnx");
    assert_eq!(audit.conflict_count(), 1);
}

#[test]
fn boot_handlers_audit_tag_constants_round_trip() {
    assert_eq!(
        AUDIT_TAG_MODELS_UPPER_INITIALIZED,
        "data_models_upper_initialized"
    );
    assert_eq!(
        AUDIT_TAG_MODEL_ADD_ABORTED_RECOVERED,
        "model_add_aborted_recovered"
    );
}

// ─── Explicit suppressing memo: no new audit events ────────────────────────

#[test]
fn second_boot_with_memo_filled_suppresses_existing_conflicts() {
    let upper_bytes = b"upper";
    let upper = upper_with_signed_files(&[("dup.onnx", upper_bytes)]);
    let mut lower = MapLowerManifest::new();
    let lower_digest = hash_bytes(b"lower");
    lower.insert("dup.onnx", lower_digest);
    let upper_digest = hash_bytes(upper_bytes);
    let mut memo = ConflictMemo::new();
    memo.record(&OverlayConflict {
        name: "dup.onnx".to_string(),
        lower_sha3: lower_digest,
        upper_sha3: upper_digest,
    });
    let mut audit = CapturingConflictAuditSink::new();
    let _ = scan_overlay_conflicts(&upper, &lower, &memo, &mut audit).unwrap();
    assert!(audit.is_empty());
}

// ─── Boot report struct shape ──────────────────────────────────────────────

#[test]
fn boot_report_is_clean_returns_true_for_quiet_boot() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (r, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert!(r.is_clean());
}

#[test]
fn boot_report_is_clean_returns_false_after_orphan_recovery() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let _ = ensure_models_upper_dir(&mut fs).unwrap();
    let inode = fs.create("/data/models-upper/leftover.tmp", 0o644).unwrap();
    fs.write_file(&inode, 0, b"x").unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (r, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert!(!r.is_clean());
}

#[test]
fn boot_report_is_clean_returns_false_on_first_boot() {
    let mut dev = build_rw_image();
    let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
    let lower = MapLowerManifest::new();
    let memo = ConflictMemo::new();
    let mut audit = CapturingConflictAuditSink::new();
    let (r, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
    assert!(!r.is_clean(), "first-boot upper init counts as a change");
}

// ─── Combined round-trip: orphan + first-boot + clean reboot ───────────────

#[test]
fn three_phase_workflow_first_boot_orphan_clean_boot() {
    // Phase 1: first boot — upper created.
    let mut dev = build_rw_image();
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let lower = MapLowerManifest::new();
        let memo = ConflictMemo::new();
        let mut audit = CapturingConflictAuditSink::new();
        let (r, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
        assert!(r.upper_initialized);
        // Drop an orphan for phase 2.
        let inode = fs.create("/data/models-upper/p2.tmp", 0o644).unwrap();
        fs.write_file(&inode, 0, b"x").unwrap();
        fs.fsync().unwrap();
    }
    // Phase 2: orphan-recovery boot.
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let lower = MapLowerManifest::new();
        let memo = ConflictMemo::new();
        let mut audit = CapturingConflictAuditSink::new();
        let (r, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
        assert!(!r.upper_initialized);
        assert_eq!(r.cleanup.count(), 1);
        fs.fsync().unwrap();
    }
    // Phase 3: clean boot — nothing to do.
    {
        let mut fs = F2fs::mount(&mut dev, 0, RW_DEFAULT_BLOCKS).unwrap();
        let lower = MapLowerManifest::new();
        let memo = ConflictMemo::new();
        let mut audit = CapturingConflictAuditSink::new();
        let (r, _) = run_overlay_boot_handlers(&mut fs, &lower, &memo, &mut audit).unwrap();
        assert!(r.is_clean());
        assert!(r.pending_audit_events().is_empty());
        assert!(audit.is_empty());
    }
}
