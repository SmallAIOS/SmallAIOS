// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Build attestation signing per NIST SP 800-53 SA-10.
//!
//! Provides structs for creating and verifying build attestation records.
//! Each attestation signs: (binary_hash, sbom_hash, toolchain_version, build_timestamp)
//! using ML-DSA-65.
//!
//! Also provides reproducible build verification (compare SHA-256 hashes of two builds).

use alloc::string::String;

/// ML-DSA-65 signature size in bytes.
const ML_DSA_65_SIGNATURE_SIZE: usize = 3309;

/// SHA-256 hash size in bytes.
const SHA256_HASH_SIZE: usize = 32;

/// Build attestation record.
///
/// Contains all fields that are signed together to create a verifiable
/// record of a build's provenance.
#[derive(Debug, Clone)]
pub struct BuildAttestation {
    /// SHA-256 hash of the output binary.
    pub binary_hash: [u8; SHA256_HASH_SIZE],
    /// SHA-256 hash of the SBOM document.
    pub sbom_hash: [u8; SHA256_HASH_SIZE],
    /// Rust toolchain version string (e.g., "nightly-2026-02-01").
    pub toolchain_version: String,
    /// Build timestamp (nanoseconds since epoch).
    pub build_timestamp_ns: u64,
    /// Source commit hash (20 bytes, Git SHA-1).
    pub source_commit: [u8; 20],
    /// Builder identity (e.g., CI job ID or developer key fingerprint).
    pub builder_identity: String,
    /// ML-DSA-65 signature over the attestation fields.
    /// Empty until `sign()` is called.
    pub signature: [u8; ML_DSA_65_SIGNATURE_SIZE],
    /// Whether the signature field has been populated.
    pub is_signed: bool,
}

impl BuildAttestation {
    /// Create a new unsigned attestation.
    pub fn new(
        binary_hash: [u8; SHA256_HASH_SIZE],
        sbom_hash: [u8; SHA256_HASH_SIZE],
        toolchain_version: String,
        build_timestamp_ns: u64,
    ) -> Self {
        Self {
            binary_hash,
            sbom_hash,
            toolchain_version,
            build_timestamp_ns,
            source_commit: [0u8; 20],
            builder_identity: String::new(),
            signature: [0u8; ML_DSA_65_SIGNATURE_SIZE],
            is_signed: false,
        }
    }

    /// Set the source commit hash.
    pub fn with_source_commit(mut self, commit: [u8; 20]) -> Self {
        self.source_commit = commit;
        self
    }

    /// Set the builder identity.
    pub fn with_builder_identity(mut self, identity: String) -> Self {
        self.builder_identity = identity;
        self
    }

    /// Serialize the attestation fields for signing.
    /// Returns the bytes that should be signed with ML-DSA-65.
    ///
    /// Format: binary_hash || sbom_hash || toolchain_bytes || timestamp_be || source_commit
    pub fn signing_payload(&self) -> SigningPayload {
        let toolchain_bytes = self.toolchain_version.as_bytes();
        let total_len = SHA256_HASH_SIZE * 2
            + 4  // toolchain length prefix (u32 BE)
            + toolchain_bytes.len()
            + 8  // timestamp u64 BE
            + 20; // source commit

        let mut payload = SigningPayload::new(total_len);
        payload.append(&self.binary_hash);
        payload.append(&self.sbom_hash);
        payload.append(&(toolchain_bytes.len() as u32).to_be_bytes());
        payload.append(toolchain_bytes);
        payload.append(&self.build_timestamp_ns.to_be_bytes());
        payload.append(&self.source_commit);
        payload
    }

    /// Mark the attestation as signed with the given signature.
    pub fn set_signature(&mut self, sig: &[u8]) -> bool {
        if sig.len() != ML_DSA_65_SIGNATURE_SIZE {
            return false;
        }
        self.signature[..ML_DSA_65_SIGNATURE_SIZE].copy_from_slice(sig);
        self.is_signed = true;
        true
    }

    /// Validate that all required fields are populated.
    pub fn validate(&self) -> Result<(), AttestationError> {
        if self.binary_hash == [0u8; SHA256_HASH_SIZE] {
            return Err(AttestationError::MissingBinaryHash);
        }
        if self.sbom_hash == [0u8; SHA256_HASH_SIZE] {
            return Err(AttestationError::MissingSbomHash);
        }
        if self.toolchain_version.is_empty() {
            return Err(AttestationError::MissingToolchainVersion);
        }
        if self.build_timestamp_ns == 0 {
            return Err(AttestationError::MissingTimestamp);
        }
        Ok(())
    }
}

/// Errors in attestation validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestationError {
    MissingBinaryHash,
    MissingSbomHash,
    MissingToolchainVersion,
    MissingTimestamp,
}

/// A buffer for holding the signing payload.
///
/// Uses a fixed-size stack buffer to avoid heap allocation for the
/// common case of small payloads.
pub struct SigningPayload {
    data: [u8; 256],
    len: usize,
    capacity: usize,
}

impl SigningPayload {
    fn new(capacity: usize) -> Self {
        Self {
            data: [0u8; 256],
            len: 0,
            capacity: capacity.min(256),
        }
    }

    fn append(&mut self, bytes: &[u8]) {
        let end = (self.len + bytes.len()).min(self.capacity);
        let copy_len = end - self.len;
        self.data[self.len..end].copy_from_slice(&bytes[..copy_len]);
        self.len = end;
    }

    /// Get the payload bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len]
    }

    /// Length of the payload.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the payload is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Reproducible build verification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReproducibleBuildResult {
    /// Both builds produced identical output.
    Match,
    /// Builds produced different output.
    Mismatch,
}

/// Compare two build output hashes for reproducibility verification.
///
/// Both hashes should be SHA-256 of the final binary. Returns `Match`
/// only if every byte is identical.
pub fn verify_reproducible_build(
    build1_hash: &[u8; SHA256_HASH_SIZE],
    build2_hash: &[u8; SHA256_HASH_SIZE],
) -> ReproducibleBuildResult {
    // Constant-time comparison to avoid timing side channels
    let mut diff = 0u8;
    for i in 0..SHA256_HASH_SIZE {
        diff |= build1_hash[i] ^ build2_hash[i];
    }
    if diff == 0 {
        ReproducibleBuildResult::Match
    } else {
        ReproducibleBuildResult::Mismatch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hash(seed: u8) -> [u8; SHA256_HASH_SIZE] {
        let mut h = [0u8; SHA256_HASH_SIZE];
        for (i, byte) in h.iter_mut().enumerate() {
            *byte = seed.wrapping_add(i as u8);
        }
        h
    }

    #[test]
    fn create_attestation() {
        let att = BuildAttestation::new(
            sample_hash(1),
            sample_hash(2),
            String::from("nightly-2026-02-01"),
            1_000_000_000,
        );
        assert!(!att.is_signed);
        assert_eq!(att.toolchain_version, "nightly-2026-02-01");
        assert_eq!(att.build_timestamp_ns, 1_000_000_000);
    }

    #[test]
    fn attestation_builder_pattern() {
        let mut commit = [0u8; 20];
        commit[0] = 0xab;
        commit[1] = 0xcd;

        let att = BuildAttestation::new(
            sample_hash(1),
            sample_hash(2),
            String::from("nightly-2026-02-01"),
            1_000_000_000,
        )
        .with_source_commit(commit)
        .with_builder_identity(String::from("ci-job-123"));

        assert_eq!(att.source_commit[0], 0xab);
        assert_eq!(att.builder_identity, "ci-job-123");
    }

    #[test]
    fn signing_payload_not_empty() {
        let att = BuildAttestation::new(
            sample_hash(1),
            sample_hash(2),
            String::from("nightly"),
            1_000,
        );
        let payload = att.signing_payload();
        assert!(!payload.is_empty());
    }

    #[test]
    fn signing_payload_deterministic() {
        let att = BuildAttestation::new(
            sample_hash(1),
            sample_hash(2),
            String::from("nightly-2026-02-01"),
            1_000_000,
        );
        let p1 = att.signing_payload();
        let p2 = att.signing_payload();
        assert_eq!(p1.as_bytes(), p2.as_bytes());
    }

    #[test]
    fn signing_payload_differs_with_different_inputs() {
        let att1 = BuildAttestation::new(
            sample_hash(1),
            sample_hash(2),
            String::from("nightly"),
            1_000,
        );
        let att2 = BuildAttestation::new(
            sample_hash(3),
            sample_hash(2),
            String::from("nightly"),
            1_000,
        );
        assert_ne!(
            att1.signing_payload().as_bytes(),
            att2.signing_payload().as_bytes()
        );
    }

    #[test]
    fn set_signature_correct_size() {
        let mut att = BuildAttestation::new(
            sample_hash(1),
            sample_hash(2),
            String::from("nightly"),
            1_000,
        );
        let sig = [0x42u8; ML_DSA_65_SIGNATURE_SIZE];
        assert!(att.set_signature(&sig));
        assert!(att.is_signed);
        assert_eq!(att.signature[0], 0x42);
    }

    #[test]
    fn set_signature_wrong_size_rejected() {
        let mut att = BuildAttestation::new(
            sample_hash(1),
            sample_hash(2),
            String::from("nightly"),
            1_000,
        );
        let sig = [0x42u8; 100];
        assert!(!att.set_signature(&sig));
        assert!(!att.is_signed);
    }

    #[test]
    fn validate_complete_attestation() {
        let att = BuildAttestation::new(
            sample_hash(1),
            sample_hash(2),
            String::from("nightly-2026-02-01"),
            1_000_000,
        );
        assert!(att.validate().is_ok());
    }

    #[test]
    fn validate_missing_binary_hash() {
        let att = BuildAttestation::new(
            [0u8; SHA256_HASH_SIZE],
            sample_hash(2),
            String::from("nightly"),
            1_000,
        );
        assert_eq!(att.validate(), Err(AttestationError::MissingBinaryHash));
    }

    #[test]
    fn validate_missing_sbom_hash() {
        let att = BuildAttestation::new(
            sample_hash(1),
            [0u8; SHA256_HASH_SIZE],
            String::from("nightly"),
            1_000,
        );
        assert_eq!(att.validate(), Err(AttestationError::MissingSbomHash));
    }

    #[test]
    fn validate_missing_toolchain() {
        let att = BuildAttestation::new(sample_hash(1), sample_hash(2), String::new(), 1_000);
        assert_eq!(
            att.validate(),
            Err(AttestationError::MissingToolchainVersion)
        );
    }

    #[test]
    fn validate_missing_timestamp() {
        let att = BuildAttestation::new(sample_hash(1), sample_hash(2), String::from("nightly"), 0);
        assert_eq!(att.validate(), Err(AttestationError::MissingTimestamp));
    }

    #[test]
    fn reproducible_build_match() {
        let h1 = sample_hash(42);
        let h2 = sample_hash(42);
        assert_eq!(
            verify_reproducible_build(&h1, &h2),
            ReproducibleBuildResult::Match
        );
    }

    #[test]
    fn reproducible_build_mismatch() {
        let h1 = sample_hash(42);
        let h2 = sample_hash(43);
        assert_eq!(
            verify_reproducible_build(&h1, &h2),
            ReproducibleBuildResult::Mismatch
        );
    }

    #[test]
    fn reproducible_build_single_bit_diff() {
        let h1 = sample_hash(42);
        let mut h2 = sample_hash(42);
        h2[31] ^= 0x01; // flip one bit
        assert_eq!(
            verify_reproducible_build(&h1, &h2),
            ReproducibleBuildResult::Mismatch
        );
    }
}
