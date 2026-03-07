// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the verified-boot feature.
//!
//! These tests exercise the full boot integrity pipeline:
//! kernel measurement, model measurement, log sealing, and policy enforcement.

#![cfg(feature = "verified-boot")]

use smallaios_kernel::boot_integrity::*;
use smallaios_security::crypto::ed25519;
use smallaios_security::crypto::sha3::{sha3_256, Sha3_256Digest, SHA3_256_DIGEST_LEN};
use smallaios_security::crypto::verify::VerificationPolicy;

/// Simulate a full boot sequence with verified-boot enabled.
#[test]
fn full_boot_measurement_sequence() {
    let mut log = BootMeasurementLog::new();

    // 1. Measure kernel text+rodata (with valid signature)
    let seed = [0x42u8; 32];
    let kp = ed25519::ed25519_keygen(&seed);
    let kernel_data = b"simulated kernel .text and .rodata sections";
    let hash = sha3_256(kernel_data);
    let sig = ed25519::ed25519_sign(&kp.secret_key, hash.as_bytes());
    let boot_sig = BootSignature {
        signature: *sig.as_bytes(),
        expected_hash: {
            let mut h = [0u8; SHA3_256_DIGEST_LEN];
            h.copy_from_slice(hash.as_bytes());
            h
        },
    };
    let pk = *kp.public_key.as_bytes();
    let status = measure_kernel(kernel_data, Some(&boot_sig), Some(&pk), 1, &mut log).unwrap();
    assert_eq!(status, VerifyStatus::Verified);

    // 2. Measure DTB
    measure_dtb(b"fake device tree blob", 2, &mut log).unwrap();

    // 3. Measure an ONNX model (verified)
    let model_hash = sha3_256(b"model data");
    measure_model(
        b"resnet50.onnx",
        model_hash,
        VerifyStatus::Verified,
        3,
        &mut log,
    )
    .unwrap();

    // 4. Seal the log
    log.seal();
    assert!(log.is_sealed());

    // 5. Verify log contents
    assert_eq!(log.len(), 3);
    let entries = log.entries();

    let e0 = entries[0].as_ref().unwrap();
    assert_eq!(e0.component_id(), b"kernel:text+rodata");
    assert_eq!(e0.status(), VerifyStatus::Verified);
    assert_eq!(e0.timestamp(), 1);

    let e1 = entries[1].as_ref().unwrap();
    assert_eq!(e1.component_id(), b"firmware:dtb");
    assert_eq!(e1.status(), VerifyStatus::Unverified);

    let e2 = entries[2].as_ref().unwrap();
    assert_eq!(e2.component_id(), b"model:resnet50.onnx");
    assert_eq!(e2.status(), VerifyStatus::Verified);

    // 6. Verify log is immutable
    let extra = BootMeasurement::new(
        b"extra",
        Sha3_256Digest::from_bytes([0; SHA3_256_DIGEST_LEN]),
        999,
        VerifyStatus::Failed,
    );
    assert_eq!(log.add_measurement(extra), Err(MeasurementError::LogSealed));
}

/// Enforce mode rejects unsigned kernel (no boot signature).
#[test]
fn enforce_mode_kernel_no_signature() {
    let mut log = BootMeasurementLog::new();
    let kernel_data = b"kernel without embedded signature";
    let config = BootVerifyConfig {
        kernel_policy: VerificationPolicy::Enforce,
        model_policy: VerificationPolicy::Enforce,
    };

    let status = measure_kernel(kernel_data, None, None, 0, &mut log).unwrap();
    assert_eq!(status, VerifyStatus::Unverified);

    // In Enforce mode, the caller (boot code) would halt on Unverified
    assert!(config.kernel_policy.should_reject());
}

/// WarnOnly mode allows unsigned kernel.
#[test]
fn warn_mode_kernel_no_signature() {
    let config = BootVerifyConfig::default();
    assert_eq!(config.kernel_policy, VerificationPolicy::WarnOnly);
    assert!(!config.kernel_policy.should_reject());
    assert!(config.kernel_policy.should_verify());
}

/// Tampered kernel detected (hash mismatch).
#[test]
fn tampered_kernel_detected() {
    let seed = [0x42u8; 32];
    let kp = ed25519::ed25519_keygen(&seed);

    // Sign for original data
    let original = b"original kernel data";
    let hash = sha3_256(original);
    let sig = ed25519::ed25519_sign(&kp.secret_key, hash.as_bytes());
    let boot_sig = BootSignature {
        signature: *sig.as_bytes(),
        expected_hash: {
            let mut h = [0u8; SHA3_256_DIGEST_LEN];
            h.copy_from_slice(hash.as_bytes());
            h
        },
    };
    let pk = *kp.public_key.as_bytes();

    // Measure with different data (tampered)
    let tampered = b"tampered kernel data!!!!";
    let mut log = BootMeasurementLog::new();
    let status = measure_kernel(tampered, Some(&boot_sig), Some(&pk), 0, &mut log).unwrap();
    assert_eq!(status, VerifyStatus::Failed);

    let entry = log.entries()[0].as_ref().unwrap();
    assert_eq!(entry.status(), VerifyStatus::Failed);
}

/// Multiple models can be measured in sequence.
#[test]
fn multiple_model_measurements() {
    let mut log = BootMeasurementLog::new();

    for i in 0..5u8 {
        let hash = Sha3_256Digest::from_bytes([i; SHA3_256_DIGEST_LEN]);
        let name = [b'a' + i; 8];
        measure_model(&name, hash, VerifyStatus::Verified, i as u64, &mut log).unwrap();
    }

    assert_eq!(log.len(), 5);
    for (i, entry) in log.entries().iter().enumerate() {
        let e = entry.as_ref().unwrap();
        assert_eq!(e.status(), VerifyStatus::Verified);
        assert_eq!(e.timestamp(), i as u64);
    }
}
