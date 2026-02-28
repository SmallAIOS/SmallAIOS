// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Boot integrity verification and measurement log.
//!
//! When the `verified-boot` feature is enabled, this module provides:
//! - A boot measurement log recording SHA-3-256 hashes of loaded components
//! - Kernel self-integrity verification via embedded Ed25519-signed hash
//! - Integration with the ONNX model verification pipeline

use smallaios_security::crypto::sha3::{Sha3_256Digest, SHA3_256_DIGEST_LEN};
use smallaios_security::crypto::verify::VerificationPolicy;

// ─── Verify Status ───────────────────────────────────────────────────────────

/// Verification status for a boot measurement entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyStatus {
    /// Component hash matched a valid signature.
    Verified,
    /// Component was measured but no signature was available or checked.
    Unverified,
    /// Component hash or signature verification failed.
    Failed,
}

// ─── Boot Measurement ────────────────────────────────────────────────────────

/// Maximum length of a component identifier in bytes.
pub const MAX_COMPONENT_ID_LEN: usize = 64;

/// A single boot measurement entry.
#[derive(Clone)]
pub struct BootMeasurement {
    /// Component identifier (e.g., "kernel:text+rodata", "firmware:dtb").
    component_id: [u8; MAX_COMPONENT_ID_LEN],
    /// Actual length of the component ID.
    component_id_len: usize,
    /// SHA-3-256 hash of the component data.
    hash: Sha3_256Digest,
    /// Monotonic timestamp (ticks since boot).
    timestamp: u64,
    /// Verification status.
    status: VerifyStatus,
}

impl BootMeasurement {
    /// Create a new boot measurement.
    pub fn new(
        component_id: &[u8],
        hash: Sha3_256Digest,
        timestamp: u64,
        status: VerifyStatus,
    ) -> Self {
        let len = component_id.len().min(MAX_COMPONENT_ID_LEN);
        let mut id_buf = [0u8; MAX_COMPONENT_ID_LEN];
        id_buf[..len].copy_from_slice(&component_id[..len]);
        Self {
            component_id: id_buf,
            component_id_len: len,
            hash,
            timestamp,
            status,
        }
    }

    /// Return the component identifier as a byte slice.
    pub fn component_id(&self) -> &[u8] {
        &self.component_id[..self.component_id_len]
    }

    /// Return the SHA-3-256 hash.
    pub fn hash(&self) -> &Sha3_256Digest {
        &self.hash
    }

    /// Return the timestamp.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Return the verification status.
    pub fn status(&self) -> VerifyStatus {
        self.status
    }
}

// Provide a Debug impl that doesn't print the full 64-byte array
impl core::fmt::Debug for BootMeasurement {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BootMeasurement")
            .field("component_id_len", &self.component_id_len)
            .field("timestamp", &self.timestamp)
            .field("status", &self.status)
            .finish()
    }
}

// ─── Boot Measurement Log ────────────────────────────────────────────────────

/// Maximum number of entries in the boot measurement log.
pub const MAX_BOOT_MEASUREMENTS: usize = 32;

/// Errors from the boot measurement log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementError {
    /// The log has been sealed and no more entries can be added.
    LogSealed,
    /// The log is full (capacity reached).
    LogFull,
}

/// Boot measurement log — records SHA-3-256 hashes of components loaded at boot.
///
/// Fixed-capacity, no-alloc, immutable after sealing.
pub struct BootMeasurementLog {
    entries: [Option<BootMeasurement>; MAX_BOOT_MEASUREMENTS],
    count: usize,
    sealed: bool,
}

impl BootMeasurementLog {
    /// Create a new empty measurement log.
    pub const fn new() -> Self {
        // const-compatible initialization without Option array macro
        const NONE: Option<BootMeasurement> = None;
        Self {
            entries: [NONE; MAX_BOOT_MEASUREMENTS],
            count: 0,
            sealed: false,
        }
    }

    /// Add a measurement to the log.
    ///
    /// Returns `Err(MeasurementError::LogSealed)` if the log has been sealed,
    /// or `Err(MeasurementError::LogFull)` if capacity is reached.
    pub fn add_measurement(&mut self, measurement: BootMeasurement) -> Result<(), MeasurementError> {
        if self.sealed {
            return Err(MeasurementError::LogSealed);
        }
        if self.count >= MAX_BOOT_MEASUREMENTS {
            return Err(MeasurementError::LogFull);
        }
        self.entries[self.count] = Some(measurement);
        self.count += 1;
        Ok(())
    }

    /// Seal the log, making it immutable.
    pub fn seal(&mut self) {
        self.sealed = true;
    }

    /// Check if the log has been sealed.
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Return the number of entries recorded.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Return all recorded measurement entries.
    pub fn entries(&self) -> &[Option<BootMeasurement>] {
        &self.entries[..self.count]
    }
}

impl Default for BootMeasurementLog {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Boot Signature Section ──────────────────────────────────────────────────

/// Magic bytes identifying a SmallAIOS boot signature block.
pub const BOOT_SIG_MAGIC: [u8; 8] = *b"SAIOBOOT";

/// Size of the boot signature section: magic(8) + Ed25519 sig(64) + SHA-3 hash(32) = 104 bytes.
pub const BOOT_SIG_SIZE: usize = 8 + 64 + SHA3_256_DIGEST_LEN;

/// Parsed boot signature from the `.boot_sig` section.
#[derive(Debug, Clone)]
pub struct BootSignature {
    /// Ed25519 signature over the expected hash.
    pub signature: [u8; 64],
    /// Expected SHA-3-256 hash of kernel .text + .rodata.
    pub expected_hash: [u8; SHA3_256_DIGEST_LEN],
}

impl BootSignature {
    /// Parse a boot signature from raw bytes.
    ///
    /// Returns `None` if the magic bytes don't match or the data is too short.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < BOOT_SIG_SIZE {
            return None;
        }
        if data[..8] != BOOT_SIG_MAGIC {
            return None;
        }
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&data[8..72]);
        let mut expected_hash = [0u8; SHA3_256_DIGEST_LEN];
        expected_hash.copy_from_slice(&data[72..72 + SHA3_256_DIGEST_LEN]);
        Some(Self {
            signature,
            expected_hash,
        })
    }
}

/// Compute the SHA-3-256 hash of a memory region (kernel .text + .rodata).
///
/// # Safety
/// The caller must ensure `start` and `len` describe a valid readable memory region.
pub unsafe fn compute_section_hash(start: *const u8, len: usize) -> Sha3_256Digest {
    // SAFETY: Caller guarantees the memory region is valid and readable.
    let data = unsafe { core::slice::from_raw_parts(start, len) };
    smallaios_security::crypto::sha3::sha3_256(data)
}

/// Verify kernel integrity using an embedded boot signature.
///
/// Returns the verification status and optionally the computed hash.
pub fn verify_kernel_signature(
    computed_hash: &Sha3_256Digest,
    boot_sig: &BootSignature,
    ed25519_public_key: &[u8; 32],
) -> VerifyStatus {
    // Check hash matches first
    if computed_hash.as_bytes() != &boot_sig.expected_hash {
        return VerifyStatus::Failed;
    }

    // Verify Ed25519 signature over the expected hash
    let pk = smallaios_security::crypto::ed25519::Ed25519PublicKey::from_bytes(*ed25519_public_key);
    let sig =
        smallaios_security::crypto::ed25519::Ed25519Signature::from_bytes(boot_sig.signature);

    match smallaios_security::crypto::ed25519::ed25519_verify(&pk, &boot_sig.expected_hash, &sig) {
        Ok(()) => VerifyStatus::Verified,
        Err(_) => VerifyStatus::Failed,
    }
}

// ─── Boot Verification Configuration ─────────────────────────────────────────

/// Boot verification configuration.
#[derive(Debug, Clone)]
pub struct BootVerifyConfig {
    /// Policy for kernel self-integrity verification.
    pub kernel_policy: VerificationPolicy,
    /// Policy for ONNX model signature verification.
    pub model_policy: VerificationPolicy,
}

impl Default for BootVerifyConfig {
    fn default() -> Self {
        Self {
            kernel_policy: VerificationPolicy::WarnOnly,
            model_policy: VerificationPolicy::WarnOnly,
        }
    }
}

// ─── Per-Architecture Boot Measurement Hooks ─────────────────────────────────

/// Measure a DTB (Device Tree Blob) and record in the measurement log.
///
/// Called from AArch64 and RISC-V boot paths when a DTB pointer is provided
/// by firmware. The DTB is hashed with SHA-3-256 and recorded as `firmware:dtb`.
///
/// # Arguments
/// * `dtb_data` — The raw DTB bytes
/// * `timestamp` — Monotonic boot timestamp
/// * `log` — The boot measurement log to record into
pub fn measure_dtb(
    dtb_data: &[u8],
    timestamp: u64,
    log: &mut BootMeasurementLog,
) -> Result<(), MeasurementError> {
    let hash = smallaios_security::crypto::sha3::sha3_256(dtb_data);
    let measurement = BootMeasurement::new(b"firmware:dtb", hash, timestamp, VerifyStatus::Unverified);
    log.add_measurement(measurement)
}

/// Measure a Multiboot2 info structure and record in the measurement log.
///
/// Called from x86-64 boot path. The Multiboot2 info structure is hashed
/// with SHA-3-256 and recorded as `firmware:multiboot2`.
///
/// # Arguments
/// * `mb2_data` — The raw Multiboot2 info structure bytes
/// * `timestamp` — Monotonic boot timestamp
/// * `log` — The boot measurement log to record into
pub fn measure_multiboot2(
    mb2_data: &[u8],
    timestamp: u64,
    log: &mut BootMeasurementLog,
) -> Result<(), MeasurementError> {
    let hash = smallaios_security::crypto::sha3::sha3_256(mb2_data);
    let measurement = BootMeasurement::new(
        b"firmware:multiboot2",
        hash,
        timestamp,
        VerifyStatus::Unverified,
    );
    log.add_measurement(measurement)
}

/// Measure the kernel .text + .rodata sections and record in the measurement log.
///
/// Called at early boot on all architectures. Optionally verifies the hash
/// against an embedded boot signature.
///
/// # Arguments
/// * `text_rodata` — Combined .text + .rodata section bytes
/// * `boot_sig` — Optional parsed boot signature for verification
/// * `ed25519_pk` — Optional Ed25519 public key for signature verification
/// * `timestamp` — Monotonic boot timestamp
/// * `log` — The boot measurement log to record into
pub fn measure_kernel(
    text_rodata: &[u8],
    boot_sig: Option<&BootSignature>,
    ed25519_pk: Option<&[u8; 32]>,
    timestamp: u64,
    log: &mut BootMeasurementLog,
) -> Result<VerifyStatus, MeasurementError> {
    let hash = smallaios_security::crypto::sha3::sha3_256(text_rodata);

    let status = match (boot_sig, ed25519_pk) {
        (Some(sig), Some(pk)) => verify_kernel_signature(&hash, sig, pk),
        (Some(_), None) | (None, Some(_)) => VerifyStatus::Unverified,
        (None, None) => VerifyStatus::Unverified,
    };

    let measurement = BootMeasurement::new(b"kernel:text+rodata", hash, timestamp, status);
    log.add_measurement(measurement)?;
    Ok(status)
}

/// Record an ONNX model measurement in the boot measurement log.
///
/// Called from the ONNX runtime model load path when verified-boot is enabled.
///
/// # Arguments
/// * `model_name` — Name/identifier for the model
/// * `model_hash` — SHA-3-256 hash of the model data
/// * `status` — Verification result
/// * `timestamp` — Monotonic timestamp
/// * `log` — The boot measurement log to record into
pub fn measure_model(
    model_name: &[u8],
    model_hash: Sha3_256Digest,
    status: VerifyStatus,
    timestamp: u64,
    log: &mut BootMeasurementLog,
) -> Result<(), MeasurementError> {
    // Prefix model names with "model:" for the component ID
    let mut component_id = [0u8; MAX_COMPONENT_ID_LEN];
    let prefix = b"model:";
    let prefix_len = prefix.len();
    component_id[..prefix_len].copy_from_slice(prefix);
    let name_len = model_name.len().min(MAX_COMPONENT_ID_LEN - prefix_len);
    component_id[prefix_len..prefix_len + name_len].copy_from_slice(&model_name[..name_len]);

    let measurement = BootMeasurement::new(
        &component_id[..prefix_len + name_len],
        model_hash,
        timestamp,
        status,
    );
    log.add_measurement(measurement)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_status_variants() {
        assert_ne!(VerifyStatus::Verified, VerifyStatus::Failed);
        assert_ne!(VerifyStatus::Verified, VerifyStatus::Unverified);
        assert_ne!(VerifyStatus::Unverified, VerifyStatus::Failed);
    }

    #[test]
    fn boot_measurement_creation() {
        let hash = Sha3_256Digest::from_bytes([0xAA; SHA3_256_DIGEST_LEN]);
        let m = BootMeasurement::new(b"kernel:text+rodata", hash, 42, VerifyStatus::Verified);

        assert_eq!(m.component_id(), b"kernel:text+rodata");
        assert_eq!(m.hash(), &hash);
        assert_eq!(m.timestamp(), 42);
        assert_eq!(m.status(), VerifyStatus::Verified);
    }

    #[test]
    fn boot_measurement_truncates_long_id() {
        let long_id = [b'X'; 128];
        let hash = Sha3_256Digest::from_bytes([0; SHA3_256_DIGEST_LEN]);
        let m = BootMeasurement::new(&long_id, hash, 0, VerifyStatus::Unverified);
        assert_eq!(m.component_id().len(), MAX_COMPONENT_ID_LEN);
    }

    #[test]
    fn measurement_log_add_and_query() {
        let mut log = BootMeasurementLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);

        let hash = Sha3_256Digest::from_bytes([0x11; SHA3_256_DIGEST_LEN]);
        let m = BootMeasurement::new(b"kernel:text+rodata", hash, 1, VerifyStatus::Verified);
        log.add_measurement(m).unwrap();

        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
        let entry = log.entries()[0].as_ref().unwrap();
        assert_eq!(entry.component_id(), b"kernel:text+rodata");
        assert_eq!(entry.status(), VerifyStatus::Verified);
    }

    #[test]
    fn measurement_log_seal_prevents_adds() {
        let mut log = BootMeasurementLog::new();
        let hash = Sha3_256Digest::from_bytes([0; SHA3_256_DIGEST_LEN]);
        let m = BootMeasurement::new(b"test", hash, 0, VerifyStatus::Unverified);
        log.add_measurement(m).unwrap();

        log.seal();
        assert!(log.is_sealed());

        let m2 = BootMeasurement::new(b"after-seal", hash, 1, VerifyStatus::Verified);
        assert_eq!(
            log.add_measurement(m2),
            Err(MeasurementError::LogSealed)
        );
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn measurement_log_capacity_limit() {
        let mut log = BootMeasurementLog::new();
        let hash = Sha3_256Digest::from_bytes([0; SHA3_256_DIGEST_LEN]);

        for i in 0..MAX_BOOT_MEASUREMENTS {
            let m = BootMeasurement::new(
                &[i as u8],
                hash,
                i as u64,
                VerifyStatus::Unverified,
            );
            log.add_measurement(m).unwrap();
        }
        assert_eq!(log.len(), MAX_BOOT_MEASUREMENTS);

        let m = BootMeasurement::new(b"overflow", hash, 999, VerifyStatus::Failed);
        assert_eq!(log.add_measurement(m), Err(MeasurementError::LogFull));
    }

    #[test]
    fn measurement_log_entries_returns_correct_slice() {
        let mut log = BootMeasurementLog::new();
        let hash = Sha3_256Digest::from_bytes([0; SHA3_256_DIGEST_LEN]);

        for i in 0..5 {
            let m = BootMeasurement::new(
                &[b'a' + i as u8],
                hash,
                i as u64,
                VerifyStatus::Verified,
            );
            log.add_measurement(m).unwrap();
        }

        let entries = log.entries();
        assert_eq!(entries.len(), 5);
        for (i, entry) in entries.iter().enumerate() {
            let e = entry.as_ref().unwrap();
            assert_eq!(e.timestamp(), i as u64);
        }
    }

    #[test]
    fn boot_signature_parse_valid() {
        let mut data = [0u8; BOOT_SIG_SIZE];
        data[..8].copy_from_slice(&BOOT_SIG_MAGIC);
        // Fill signature with 0xAA
        for b in &mut data[8..72] {
            *b = 0xAA;
        }
        // Fill hash with 0xBB
        for b in &mut data[72..104] {
            *b = 0xBB;
        }

        let sig = BootSignature::from_bytes(&data).unwrap();
        assert_eq!(sig.signature, [0xAA; 64]);
        assert_eq!(sig.expected_hash, [0xBB; SHA3_256_DIGEST_LEN]);
    }

    #[test]
    fn boot_signature_parse_bad_magic() {
        let mut data = [0u8; BOOT_SIG_SIZE];
        data[..8].copy_from_slice(b"BADMAGIC");
        assert!(BootSignature::from_bytes(&data).is_none());
    }

    #[test]
    fn boot_signature_parse_too_short() {
        let data = [0u8; BOOT_SIG_SIZE - 1];
        assert!(BootSignature::from_bytes(&data).is_none());
    }

    #[test]
    fn boot_verify_config_default() {
        let config = BootVerifyConfig::default();
        assert_eq!(config.kernel_policy, VerificationPolicy::WarnOnly);
        assert_eq!(config.model_policy, VerificationPolicy::WarnOnly);
    }

    #[test]
    fn verify_kernel_signature_with_valid_keypair() {
        // Generate an Ed25519 key pair
        let seed = [0x42u8; 32];
        let kp = smallaios_security::crypto::ed25519::ed25519_keygen(&seed);

        // Create a "kernel hash"
        let kernel_data = b"fake kernel text and rodata sections";
        let hash = smallaios_security::crypto::sha3::sha3_256(kernel_data);

        // Sign the hash
        let sig = smallaios_security::crypto::ed25519::ed25519_sign(
            &kp.secret_key,
            hash.as_bytes(),
        );

        let boot_sig = BootSignature {
            signature: *sig.as_bytes(),
            expected_hash: {
                let mut h = [0u8; SHA3_256_DIGEST_LEN];
                h.copy_from_slice(hash.as_bytes());
                h
            },
        };

        let status = verify_kernel_signature(&hash, &boot_sig, &*kp.public_key.as_bytes());
        assert_eq!(status, VerifyStatus::Verified);
    }

    #[test]
    fn verify_kernel_signature_hash_mismatch() {
        let seed = [0x42u8; 32];
        let kp = smallaios_security::crypto::ed25519::ed25519_keygen(&seed);

        let hash = smallaios_security::crypto::sha3::sha3_256(b"original data");
        let wrong_hash = smallaios_security::crypto::sha3::sha3_256(b"different data");

        let sig = smallaios_security::crypto::ed25519::ed25519_sign(
            &kp.secret_key,
            hash.as_bytes(),
        );

        let boot_sig = BootSignature {
            signature: *sig.as_bytes(),
            expected_hash: {
                let mut h = [0u8; SHA3_256_DIGEST_LEN];
                h.copy_from_slice(hash.as_bytes());
                h
            },
        };

        // Pass wrong hash — should fail
        let status = verify_kernel_signature(&wrong_hash, &boot_sig, &*kp.public_key.as_bytes());
        assert_eq!(status, VerifyStatus::Failed);
    }

    #[test]
    fn verify_kernel_signature_bad_signature() {
        let seed = [0x42u8; 32];
        let kp = smallaios_security::crypto::ed25519::ed25519_keygen(&seed);

        let hash = smallaios_security::crypto::sha3::sha3_256(b"kernel data");

        let boot_sig = BootSignature {
            signature: [0xFF; 64], // invalid signature
            expected_hash: {
                let mut h = [0u8; SHA3_256_DIGEST_LEN];
                h.copy_from_slice(hash.as_bytes());
                h
            },
        };

        let status = verify_kernel_signature(&hash, &boot_sig, &*kp.public_key.as_bytes());
        assert_eq!(status, VerifyStatus::Failed);
    }

    #[test]
    fn verification_policy_methods() {
        assert!(VerificationPolicy::Enforce.should_verify());
        assert!(VerificationPolicy::Enforce.should_reject());

        assert!(VerificationPolicy::WarnOnly.should_verify());
        assert!(!VerificationPolicy::WarnOnly.should_reject());

        assert!(!VerificationPolicy::Disabled.should_verify());
        assert!(!VerificationPolicy::Disabled.should_reject());

        assert_eq!(VerificationPolicy::default(), VerificationPolicy::WarnOnly);
    }

    // ── Per-architecture measurement hook tests ──────────────────────────

    #[test]
    fn measure_dtb_records_entry() {
        let mut log = BootMeasurementLog::new();
        let dtb_data = b"fake device tree blob";
        measure_dtb(dtb_data, 100, &mut log).unwrap();

        assert_eq!(log.len(), 1);
        let entry = log.entries()[0].as_ref().unwrap();
        assert_eq!(entry.component_id(), b"firmware:dtb");
        assert_eq!(entry.timestamp(), 100);
        assert_eq!(entry.status(), VerifyStatus::Unverified);
    }

    #[test]
    fn measure_multiboot2_records_entry() {
        let mut log = BootMeasurementLog::new();
        let mb2_data = b"fake multiboot2 info structure";
        measure_multiboot2(mb2_data, 200, &mut log).unwrap();

        assert_eq!(log.len(), 1);
        let entry = log.entries()[0].as_ref().unwrap();
        assert_eq!(entry.component_id(), b"firmware:multiboot2");
        assert_eq!(entry.timestamp(), 200);
    }

    #[test]
    fn measure_kernel_without_signature() {
        let mut log = BootMeasurementLog::new();
        let kernel_data = b"fake kernel text and rodata";
        let status = measure_kernel(kernel_data, None, None, 50, &mut log).unwrap();

        assert_eq!(status, VerifyStatus::Unverified);
        assert_eq!(log.len(), 1);
        let entry = log.entries()[0].as_ref().unwrap();
        assert_eq!(entry.component_id(), b"kernel:text+rodata");
    }

    #[test]
    fn measure_kernel_with_valid_signature() {
        let seed = [0x42u8; 32];
        let kp = smallaios_security::crypto::ed25519::ed25519_keygen(&seed);
        let kernel_data = b"fake kernel text and rodata";
        let hash = smallaios_security::crypto::sha3::sha3_256(kernel_data);
        let sig = smallaios_security::crypto::ed25519::ed25519_sign(
            &kp.secret_key,
            hash.as_bytes(),
        );
        let boot_sig = BootSignature {
            signature: *sig.as_bytes(),
            expected_hash: {
                let mut h = [0u8; SHA3_256_DIGEST_LEN];
                h.copy_from_slice(hash.as_bytes());
                h
            },
        };
        let pk = *kp.public_key.as_bytes();

        let mut log = BootMeasurementLog::new();
        let status = measure_kernel(kernel_data, Some(&boot_sig), Some(&pk), 10, &mut log).unwrap();

        assert_eq!(status, VerifyStatus::Verified);
        let entry = log.entries()[0].as_ref().unwrap();
        assert_eq!(entry.status(), VerifyStatus::Verified);
    }

    #[test]
    fn measure_model_records_entry() {
        let mut log = BootMeasurementLog::new();
        let hash = Sha3_256Digest::from_bytes([0xCC; SHA3_256_DIGEST_LEN]);
        measure_model(b"resnet50", hash, VerifyStatus::Verified, 300, &mut log).unwrap();

        assert_eq!(log.len(), 1);
        let entry = log.entries()[0].as_ref().unwrap();
        assert_eq!(entry.component_id(), b"model:resnet50");
        assert_eq!(entry.status(), VerifyStatus::Verified);
        assert_eq!(entry.timestamp(), 300);
    }

    #[test]
    fn measure_model_truncates_long_name() {
        let mut log = BootMeasurementLog::new();
        let long_name = [b'M'; 128]; // longer than MAX_COMPONENT_ID_LEN - "model:" prefix
        let hash = Sha3_256Digest::from_bytes([0; SHA3_256_DIGEST_LEN]);
        measure_model(&long_name, hash, VerifyStatus::Unverified, 0, &mut log).unwrap();

        let entry = log.entries()[0].as_ref().unwrap();
        assert!(entry.component_id().starts_with(b"model:"));
        assert!(entry.component_id().len() <= MAX_COMPONENT_ID_LEN);
    }

    #[test]
    fn measure_dtb_fails_on_sealed_log() {
        let mut log = BootMeasurementLog::new();
        log.seal();
        let result = measure_dtb(b"dtb data", 0, &mut log);
        assert_eq!(result, Err(MeasurementError::LogSealed));
    }
}
