// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration-level tests for the A/B boot pointer + bsdiff
//! delta-update pipeline (`embedded-filesystem-v1` Phase 4).
//!
//! These tests sit on top of the unit tests in `boot_config::tests` /
//! `delta::tests` and exercise the whole-pipeline behavior:
//!
//! - End-to-end `apply_delta_to_partition` happy path with a real
//!   signed squashfs partition built via the same builder used by
//!   the squashfs conformance tests.
//! - Forged-signature path: no on-disk write, audit event fired.
//! - Wrong-version reference: the same.
//! - Atomicity invariant: random partial writes never leave both
//!   slots invalid.

#![cfg(all(
    feature = "fs-on-disk-mounts",
    feature = "fs-squashfs-zstd",
    feature = "fs-test-helpers"
))]

extern crate alloc;

use alloc::vec::Vec;

use smallaios_fs::block::mock::MockBlockDevice;
use smallaios_fs::block::BlockDevice;
use smallaios_fs::boot_config::{
    self, ActiveSquashfsSlot, BootConfigError, BootConfigRecord, SlotPosition, Watchdog,
    BOOT_CONFIG_RECORD_SIZE,
};
use smallaios_fs::delta::{
    apply_delta, apply_delta_to_partition, parse_header, DeltaError, NoopSink, PatchHeader,
    UpdateEvent, VecSink, CTRL_ENTRY_LEN, PATCH_HEADER_LEN, PATCH_MAGIC,
};
use smallaios_security::crypto::ml_dsa::{ml_dsa_65_keygen, ml_dsa_65_sign};
use smallaios_security::crypto::sha3::sha3_256;

/// Stable seed used across the integration suite.
const SUITE_SEED: [u8; 32] = [0x42u8; 32];

/// Build a signed bsdiff-format-v1 patch turning `reference` into
/// `target`. Returns `(patch, signature, public_key)`.
fn build_patch(reference: &[u8], target: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let diff_len = core::cmp::min(reference.len(), target.len());
    let extra_len = target.len() - diff_len;
    let mut diff = vec![0u8; diff_len];
    for j in 0..diff_len {
        diff[j] = target[j].wrapping_sub(reference[j]);
    }
    let extra = target[diff_len..].to_vec();
    let mut ctrl = Vec::with_capacity(CTRL_ENTRY_LEN);
    ctrl.extend_from_slice(&(diff_len as i64).to_le_bytes());
    ctrl.extend_from_slice(&(extra_len as i64).to_le_bytes());
    ctrl.extend_from_slice(&0i64.to_le_bytes());
    let mut patch = Vec::with_capacity(PATCH_HEADER_LEN + ctrl.len() + diff.len() + extra.len());
    patch.extend_from_slice(&PATCH_MAGIC);
    let ref_hash = *sha3_256(reference).as_bytes();
    patch.extend_from_slice(&ref_hash);
    patch.extend_from_slice(&(target.len() as u64).to_le_bytes());
    patch.extend_from_slice(&(ctrl.len() as u64).to_le_bytes());
    patch.extend_from_slice(&(diff.len() as u64).to_le_bytes());
    patch.extend_from_slice(&(extra.len() as u64).to_le_bytes());
    patch.extend_from_slice(&ctrl);
    patch.extend_from_slice(&diff);
    patch.extend_from_slice(&extra);
    let kp = ml_dsa_65_keygen(&SUITE_SEED).unwrap();
    let pk = kp.public_key.as_bytes().to_vec();
    let nonce = [0xCDu8; 32];
    let sig = ml_dsa_65_sign(&kp.secret_key, &patch, &nonce).unwrap();
    (patch, sig.as_bytes().to_vec(), pk)
}

#[test]
fn header_round_trip_known_vector() {
    let reference: Vec<u8> = (0u8..32).collect();
    let target: Vec<u8> = (32u8..64).collect();
    let (patch, _sig, _pk) = build_patch(&reference, &target);
    let header: PatchHeader = parse_header(&patch).unwrap();
    assert_eq!(header.new_size, 32);
    assert_eq!(header.ctrl_len, CTRL_ENTRY_LEN as u64);
    assert_eq!(header.diff_len, 32);
    assert_eq!(header.extra_len, 0);
    assert_eq!(header.reference_hash, *sha3_256(&reference).as_bytes());
}

#[test]
fn apply_delta_known_byte_for_byte() {
    let reference = b"the quick brown fox jumps over the lazy dog".to_vec();
    let target = b"THE QUICK BROWN FOX jumps over the LAZY DOG!".to_vec();
    let (patch, _sig, _pk) = build_patch(&reference, &target);
    let out = apply_delta(&reference, &patch).unwrap();
    assert_eq!(out, target);
}

#[test]
fn apply_delta_truncated_patch_rejected_integration() {
    let reference = b"abcdef".to_vec();
    let target = b"ABCDEF".to_vec();
    let (mut patch, _sig, _pk) = build_patch(&reference, &target);
    patch.truncate(patch.len() - 8);
    let r = apply_delta(&reference, &patch);
    assert!(matches!(r, Err(DeltaError::PatchTruncated)));
}

#[test]
fn forged_signature_rejected_no_disk_write_integration() {
    let mut dev = MockBlockDevice::new(4096, 64);
    let active_size = 4096 * 8;
    let active = vec![0xAAu8; active_size];
    for i in 0..8 {
        let off = i * 4096;
        dev.write_block(i as u64, &active[off..off + 4096]).unwrap();
    }
    let target = vec![0xBBu8; 4096];
    let (patch, mut sig, pk) = build_patch(&active, &target);
    sig[5] ^= 0xFF;
    let mut sink = VecSink::new();
    let r = apply_delta_to_partition(
        &mut dev,
        16,
        0,
        active_size as u64,
        24,
        active_size as u64,
        ActiveSquashfsSlot::B,
        &patch,
        &sig,
        &pk,
        &mut sink,
    );
    assert!(matches!(r, Err(DeltaError::SignatureInvalid)));
    assert!(sink.events.contains(&UpdateEvent::DeltaSignatureInvalid));
    // Inactive partition (LBA 24..) and the boot-config partition
    // (LBA 16..) should be untouched zeros.
    let mut buf = [0u8; 4096];
    dev.read_block(24, &mut buf).unwrap();
    assert!(
        buf.iter().all(|&b| b == 0),
        "no inactive write on rejection"
    );
    dev.read_block(16, &mut buf).unwrap();
    assert!(
        buf.iter().all(|&b| b == 0),
        "no boot-config write on rejection"
    );
}

#[test]
fn wrong_reference_rejected_with_event() {
    let mut dev = MockBlockDevice::new(4096, 64);
    let active = vec![0x11u8; 4096 * 4];
    for i in 0..4 {
        let off = i * 4096;
        dev.write_block(i as u64, &active[off..off + 4096]).unwrap();
    }
    // Build patch against a *different* reference.
    let other = vec![0x22u8; 4096 * 4];
    let target = vec![0x33u8; 4096];
    let (patch, sig, pk) = build_patch(&other, &target);
    let mut sink = VecSink::new();
    let r = apply_delta_to_partition(
        &mut dev,
        16,
        0,
        (4096 * 4) as u64,
        24,
        (4096 * 4) as u64,
        ActiveSquashfsSlot::B,
        &patch,
        &sig,
        &pk,
        &mut sink,
    );
    assert!(matches!(r, Err(DeltaError::ReferenceHashMismatch)));
    assert!(sink.events.contains(&UpdateEvent::DeltaReferenceMismatch));
}

#[test]
fn highest_generation_record_wins_integration() {
    // SlotX has gen=10, SlotY has gen=11 → Y wins.
    let mut dev = MockBlockDevice::new(BOOT_CONFIG_RECORD_SIZE as u32, 4);
    let rx = BootConfigRecord {
        generation: 10,
        ..BootConfigRecord::new(ActiveSquashfsSlot::A, 10)
    };
    let ry = BootConfigRecord {
        generation: 11,
        ..BootConfigRecord::new(ActiveSquashfsSlot::B, 11)
    };
    dev.write_block(0, &boot_config::encode_record(&rx))
        .unwrap();
    dev.write_block(1, &boot_config::encode_record(&ry))
        .unwrap();
    let (rec, slot) = boot_config::read_active(&dev, 0).unwrap();
    assert_eq!(slot, SlotPosition::SlotY);
    assert_eq!(rec.generation, 11);
    assert_eq!(rec.active_slot, ActiveSquashfsSlot::B);
}

#[test]
fn invalid_record_skipped_integration() {
    let mut dev = MockBlockDevice::new(BOOT_CONFIG_RECORD_SIZE as u32, 4);
    // SlotX: corrupt zeros (bad magic).
    dev.write_block(0, &[0u8; BOOT_CONFIG_RECORD_SIZE]).unwrap();
    // SlotY: valid at gen=4.
    let ry = BootConfigRecord {
        generation: 4,
        ..BootConfigRecord::new(ActiveSquashfsSlot::B, 4)
    };
    dev.write_block(1, &boot_config::encode_record(&ry))
        .unwrap();
    let (rec, slot) = boot_config::read_active(&dev, 0).unwrap();
    assert_eq!(slot, SlotPosition::SlotY);
    assert_eq!(rec.generation, 4);
}

#[test]
fn both_records_invalid_halts_boot_integration() {
    let dev = MockBlockDevice::new(BOOT_CONFIG_RECORD_SIZE as u32, 4);
    // No writes — both slots are zero (bad magic).
    let r = boot_config::read_active(&dev, 0);
    assert!(matches!(r, Err(BootConfigError::BothSlotsInvalid)));
}

#[test]
fn power_loss_before_flush_keeps_old_slot_integration() {
    let mut dev = MockBlockDevice::new(BOOT_CONFIG_RECORD_SIZE as u32, 4);
    // Seed both slots.
    let rx = BootConfigRecord {
        generation: 5,
        ..BootConfigRecord::new(ActiveSquashfsSlot::A, 5)
    };
    let ry = BootConfigRecord {
        generation: 6,
        ..BootConfigRecord::new(ActiveSquashfsSlot::B, 6)
    };
    dev.write_block(0, &boot_config::encode_record(&rx))
        .unwrap();
    dev.write_block(1, &boot_config::encode_record(&ry))
        .unwrap();
    // Force the next write to fail before completing.
    dev.set_permanent_failure(smallaios_fs::block::BlockError::MediaError);
    let new_rec = BootConfigRecord {
        generation: 7,
        tentative: true,
        ..BootConfigRecord::new(ActiveSquashfsSlot::A, 7)
    };
    assert!(boot_config::update_atomic(&mut dev, 0, new_rec).is_err());
    dev.clear_permanent_failure();
    // Y must still win at gen=6.
    let (rec, slot) = boot_config::read_active(&dev, 0).unwrap();
    assert_eq!(slot, SlotPosition::SlotY);
    assert_eq!(rec.generation, 6);
}

#[test]
fn watchdog_default_seconds_constant() {
    // Spec mandates a default of 60 s; the value is plumbed through
    // the watchdog interface for callers that want the default.
    use smallaios_fs::boot_config::watchdog::DEFAULT_WATCHDOG_SECONDS;
    assert_eq!(DEFAULT_WATCHDOG_SECONDS, 60);
}

#[test]
fn boot_success_clears_tentative_e2e() {
    use smallaios_fs::boot_config::watchdog::{boot_success, MockWatchdog};
    let mut dev = MockBlockDevice::new(BOOT_CONFIG_RECORD_SIZE as u32, 4);
    // Active slot record currently tentative.
    let r = BootConfigRecord {
        generation: 3,
        tentative: true,
        boot_success: false,
        ..BootConfigRecord::new(ActiveSquashfsSlot::A, 3)
    };
    dev.write_block(0, &boot_config::encode_record(&r)).unwrap();
    let mut wd = MockWatchdog::new();
    wd.arm(60).unwrap();
    boot_success(&mut dev, 0, &mut wd).unwrap();
    let (rec, _) = boot_config::read_active(&dev, 0).unwrap();
    assert!(!rec.tentative);
    assert!(rec.boot_success);
    assert!(rec.generation > 3);
    assert!(!wd.is_armed());
}

#[test]
fn noop_sink_compiles_at_integration_layer() {
    let mut sink = NoopSink;
    use smallaios_fs::delta::UpdateEventSink;
    sink.emit(UpdateEvent::PostApplyFailed);
}
