// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ONNX model signature verification.
//!
//! This module provides the infrastructure for verifying the authenticity
//! and integrity of ONNX models before they are loaded for inference.
//! Every model loaded by SmallAIOS MUST have a valid signature.
//!
//! # Signature Format
//!
//! A model signature consists of:
//!
//! 1. **Model hash**: SHA-3-256 digest of the ONNX model binary
//! 2. **Signature**: ML-DSA-65 (or hybrid Ed25519+ML-DSA-65) signature
//!    over the model hash concatenated with metadata
//! 3. **Metadata**: Model name, version, timestamp, signer identity
//!
//! # Verification Flow
//!
//! 1. Compute SHA-3-256(model_bytes)
//! 2. Reconstruct the signed message: hash || metadata
//! 3. Verify the ML-DSA-65 (or hybrid) signature against the trusted public key
//! 4. Check that the metadata (version, timestamp) is within acceptable bounds
//!
//! # Trust Model
//!
//! Public keys are provisioned at build time or via secure boot.
//! The set of trusted signing keys is immutable at runtime.
//!
//! # Implementation
//!
//! The verification logic uses SHA-3-256 for model hashing, ML-DSA-65
//! for post-quantum signatures, and optionally hybrid Ed25519+ML-DSA-65.

#![allow(unused)]

use core::fmt;

use super::hybrid::{
    HybridSigPublicKey, HybridSignature, ED25519_PK_LEN, HYBRID_SIG_LEN, HYBRID_SIG_PK_LEN,
};
use super::ml_dsa::{MlDsaPublicKey, MlDsaSignature, ML_DSA_65_PK_LEN, ML_DSA_65_SIG_LEN};
use super::sha3::{Sha3_256Digest, SHA3_256_DIGEST_LEN};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum model name length in bytes.
pub const MAX_MODEL_NAME_LEN: usize = 256;

/// Maximum model binary size (2 GiB).
pub const MAX_MODEL_SIZE: usize = 2 * 1024 * 1024 * 1024;

/// Signature metadata version (current).
pub const SIGNATURE_FORMAT_VERSION: u32 = 1;

/// Magic bytes identifying a SmallAIOS model signature block.
pub const SIGNATURE_MAGIC: [u8; 8] = *b"SAIOS\x00\x01\x00";

// ─── Verification Policy ─────────────────────────────────────────────────────

/// Policy controlling how verification failures are handled.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum VerificationPolicy {
    /// Reject on verification failure (halt boot or reject model load).
    Enforce,
    /// Log a warning on failure but continue operation.
    #[default]
    WarnOnly,
    /// Skip verification entirely (no hash computation or signature checks).
    Disabled,
}

impl VerificationPolicy {
    /// Returns `true` if verification should be performed.
    pub fn should_verify(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// Returns `true` if failures should halt/reject the operation.
    pub fn should_reject(&self) -> bool {
        matches!(self, Self::Enforce)
    }
}

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from model verification operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyError {
    /// Model exceeds maximum allowed size.
    ModelTooLarge,
    /// Model name exceeds maximum length.
    NameTooLong,
    /// Invalid signature format or magic bytes.
    InvalidSignatureFormat,
    /// Unsupported signature format version.
    UnsupportedVersion,
    /// SHA-3-256 hash mismatch (model was tampered with).
    HashMismatch,
    /// ML-DSA-65 signature verification failed.
    SignatureInvalid,
    /// Hybrid signature verification failed.
    HybridSignatureInvalid,
    /// Signing key not in trusted key set.
    UntrustedKey,
    /// Signature timestamp outside acceptable window.
    TimestampOutOfRange,
    /// No signature found for the model.
    NoSignature,
    /// Operation not yet implemented.
    NotImplemented,
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelTooLarge => {
                write!(f, "model exceeds maximum size ({} bytes)", MAX_MODEL_SIZE)
            }
            Self::NameTooLong => write!(f, "model name exceeds {} bytes", MAX_MODEL_NAME_LEN),
            Self::InvalidSignatureFormat => write!(f, "invalid model signature format"),
            Self::UnsupportedVersion => write!(f, "unsupported signature format version"),
            Self::HashMismatch => write!(f, "model hash does not match signature"),
            Self::SignatureInvalid => write!(f, "ML-DSA-65 signature verification failed"),
            Self::HybridSignatureInvalid => write!(f, "hybrid signature verification failed"),
            Self::UntrustedKey => write!(f, "signing key not in trusted key set"),
            Self::TimestampOutOfRange => write!(f, "signature timestamp out of acceptable range"),
            Self::NoSignature => write!(f, "no signature found for model"),
            Self::NotImplemented => write!(f, "model verification not yet implemented"),
        }
    }
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// Signature scheme used for the model signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SignatureScheme {
    /// ML-DSA-65 only (post-quantum).
    MlDsa65 = 1,
    /// Hybrid Ed25519 + ML-DSA-65 (default).
    HybridEdMlDsa = 2,
}

impl SignatureScheme {
    /// Create from a raw byte value.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            1 => Some(Self::MlDsa65),
            2 => Some(Self::HybridEdMlDsa),
            _ => None,
        }
    }
}

/// Metadata about a signed ONNX model.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name (UTF-8, max 256 bytes).
    name: [u8; MAX_MODEL_NAME_LEN],
    /// Actual length of the name.
    name_len: usize,
    /// Model version (semantic, encoded as u32: major.minor.patch).
    version: u32,
    /// Signing timestamp (Unix epoch seconds).
    timestamp: u64,
    /// SHA-3-256 hash of the model binary.
    model_hash: Sha3_256Digest,
    /// Signature scheme used.
    scheme: SignatureScheme,
}

impl ModelMetadata {
    /// Create new metadata.
    pub fn new(
        name: &[u8],
        version: u32,
        timestamp: u64,
        model_hash: Sha3_256Digest,
        scheme: SignatureScheme,
    ) -> Result<Self, VerifyError> {
        if name.len() > MAX_MODEL_NAME_LEN {
            return Err(VerifyError::NameTooLong);
        }
        let mut name_buf = [0u8; MAX_MODEL_NAME_LEN];
        name_buf[..name.len()].copy_from_slice(name);
        Ok(Self {
            name: name_buf,
            name_len: name.len(),
            version,
            timestamp,
            model_hash,
            scheme,
        })
    }

    /// Return the model name as a byte slice.
    pub fn name(&self) -> &[u8] {
        &self.name[..self.name_len]
    }

    /// Return the model version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Return the signing timestamp.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Return the model hash.
    pub fn model_hash(&self) -> &Sha3_256Digest {
        &self.model_hash
    }

    /// Return the signature scheme.
    pub fn scheme(&self) -> SignatureScheme {
        self.scheme
    }

    /// Serialize the metadata into a byte buffer for signature verification.
    ///
    /// Format: model_hash (32) || version (4, LE) || timestamp (8, LE) || name_len (2, LE) || name
    /// Returns the number of bytes written.
    pub fn serialize(&self, out: &mut [u8]) -> Result<usize, VerifyError> {
        let needed = SHA3_256_DIGEST_LEN + 4 + 8 + 2 + self.name_len;
        if out.len() < needed {
            return Err(VerifyError::InvalidSignatureFormat);
        }
        let mut offset = 0;

        // Model hash (32 bytes)
        out[offset..offset + SHA3_256_DIGEST_LEN].copy_from_slice(self.model_hash.as_bytes());
        offset += SHA3_256_DIGEST_LEN;

        // Version (4 bytes, little-endian)
        out[offset..offset + 4].copy_from_slice(&self.version.to_le_bytes());
        offset += 4;

        // Timestamp (8 bytes, little-endian)
        out[offset..offset + 8].copy_from_slice(&self.timestamp.to_le_bytes());
        offset += 8;

        // Name length (2 bytes, little-endian)
        out[offset..offset + 2].copy_from_slice(&(self.name_len as u16).to_le_bytes());
        offset += 2;

        // Name bytes
        out[offset..offset + self.name_len].copy_from_slice(&self.name[..self.name_len]);
        offset += self.name_len;

        Ok(offset)
    }
}

/// A model signature block containing the signature and metadata.
#[derive(Debug)]
pub struct ModelSignature {
    /// The metadata that was signed.
    pub metadata: ModelMetadata,
    /// Raw signature bytes (ML-DSA-65 or hybrid).
    /// Using a fixed buffer large enough for the largest signature (hybrid: 3373 bytes).
    signature_bytes: [u8; super::hybrid::HYBRID_SIG_LEN],
    /// Actual signature length.
    signature_len: usize,
}

impl ModelSignature {
    /// Create a model signature with ML-DSA-65.
    pub fn new_ml_dsa(metadata: ModelMetadata, signature: &MlDsaSignature) -> Self {
        let mut sig_bytes = [0u8; super::hybrid::HYBRID_SIG_LEN];
        sig_bytes[..ML_DSA_65_SIG_LEN].copy_from_slice(signature.as_bytes());
        Self {
            metadata,
            signature_bytes: sig_bytes,
            signature_len: ML_DSA_65_SIG_LEN,
        }
    }

    /// Create a model signature with hybrid Ed25519 + ML-DSA-65.
    pub fn new_hybrid(metadata: ModelMetadata, signature: &HybridSignature) -> Self {
        let mut sig_bytes = [0u8; super::hybrid::HYBRID_SIG_LEN];
        let ed_sig = signature.ed25519_sig();
        let ml_sig = signature.ml_dsa_sig();
        sig_bytes[..super::hybrid::ED25519_SIG_LEN].copy_from_slice(ed_sig);
        sig_bytes[super::hybrid::ED25519_SIG_LEN..super::hybrid::ED25519_SIG_LEN + ml_sig.len()]
            .copy_from_slice(ml_sig);
        Self {
            metadata,
            signature_bytes: sig_bytes,
            signature_len: HYBRID_SIG_LEN,
        }
    }

    /// Return the raw signature bytes.
    pub fn signature_bytes(&self) -> &[u8] {
        &self.signature_bytes[..self.signature_len]
    }
}

// ─── Trusted Key Store ───────────────────────────────────────────────────────

/// Maximum number of trusted signing keys.
pub const MAX_TRUSTED_KEYS: usize = 16;

/// A store of trusted signing public keys.
///
/// In production, this is populated at build time and is immutable at runtime.
pub struct TrustedKeyStore {
    /// ML-DSA-65 trusted public key hashes (SHA-3-256 of each key).
    ml_dsa_key_hashes: [[u8; SHA3_256_DIGEST_LEN]; MAX_TRUSTED_KEYS],
    /// Number of ML-DSA-65 trusted keys.
    ml_dsa_count: usize,
    /// Hybrid trusted public key hashes (SHA-3-256 of each combined key).
    hybrid_key_hashes: [[u8; SHA3_256_DIGEST_LEN]; MAX_TRUSTED_KEYS],
    /// Number of hybrid trusted keys.
    hybrid_count: usize,
}

impl TrustedKeyStore {
    /// Create an empty key store.
    pub const fn new() -> Self {
        Self {
            ml_dsa_key_hashes: [[0u8; SHA3_256_DIGEST_LEN]; MAX_TRUSTED_KEYS],
            ml_dsa_count: 0,
            hybrid_key_hashes: [[0u8; SHA3_256_DIGEST_LEN]; MAX_TRUSTED_KEYS],
            hybrid_count: 0,
        }
    }

    /// Add a trusted ML-DSA-65 public key (by its SHA-3-256 hash).
    pub fn add_ml_dsa_key(
        &mut self,
        key_hash: [u8; SHA3_256_DIGEST_LEN],
    ) -> Result<(), VerifyError> {
        if self.ml_dsa_count >= MAX_TRUSTED_KEYS {
            return Err(VerifyError::UntrustedKey);
        }
        self.ml_dsa_key_hashes[self.ml_dsa_count] = key_hash;
        self.ml_dsa_count += 1;
        Ok(())
    }

    /// Add a trusted hybrid public key (by its SHA-3-256 hash).
    pub fn add_hybrid_key(
        &mut self,
        key_hash: [u8; SHA3_256_DIGEST_LEN],
    ) -> Result<(), VerifyError> {
        if self.hybrid_count >= MAX_TRUSTED_KEYS {
            return Err(VerifyError::UntrustedKey);
        }
        self.hybrid_key_hashes[self.hybrid_count] = key_hash;
        self.hybrid_count += 1;
        Ok(())
    }

    /// Check if an ML-DSA-65 key hash is trusted (constant-time lookup).
    pub fn is_ml_dsa_trusted(&self, key_hash: &[u8; SHA3_256_DIGEST_LEN]) -> bool {
        let mut found = false;
        // Always iterate all entries to avoid timing leaks
        for i in 0..MAX_TRUSTED_KEYS {
            if i < self.ml_dsa_count {
                let eq = super::constant_time::ct_eq(key_hash, &self.ml_dsa_key_hashes[i]);
                found |= eq.to_bool();
            }
        }
        found
    }

    /// Check if a hybrid key hash is trusted (constant-time lookup).
    pub fn is_hybrid_trusted(&self, key_hash: &[u8; SHA3_256_DIGEST_LEN]) -> bool {
        let mut found = false;
        for i in 0..MAX_TRUSTED_KEYS {
            if i < self.hybrid_count {
                let eq = super::constant_time::ct_eq(key_hash, &self.hybrid_key_hashes[i]);
                found |= eq.to_bool();
            }
        }
        found
    }

    /// Return the number of trusted ML-DSA keys.
    pub fn ml_dsa_key_count(&self) -> usize {
        self.ml_dsa_count
    }

    /// Return the number of trusted hybrid keys.
    pub fn hybrid_key_count(&self) -> usize {
        self.hybrid_count
    }
}

impl Default for TrustedKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TrustedKeyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrustedKeyStore")
            .field("ml_dsa_keys", &self.ml_dsa_count)
            .field("hybrid_keys", &self.hybrid_count)
            .finish()
    }
}

// ─── Verification Functions ──────────────────────────────────────────────────

/// Verify an ONNX model's signature using ML-DSA-65.
///
/// # Arguments
///
/// * `model_bytes` — The raw ONNX model binary
/// * `signature` — The model signature block
/// * `public_key` — The ML-DSA-65 public key to verify against
/// * `trusted_keys` — The trusted key store
///
/// # Returns
///
/// `Ok(())` if the model is authentic and the key is trusted.
pub fn verify_model_ml_dsa(
    model_bytes: &[u8],
    signature: &ModelSignature,
    public_key: &MlDsaPublicKey,
    trusted_keys: &TrustedKeyStore,
) -> Result<(), VerifyError> {
    if model_bytes.len() > MAX_MODEL_SIZE {
        return Err(VerifyError::ModelTooLarge);
    }

    // 1. Compute SHA-3-256(model_bytes) and compare with metadata hash
    let computed_hash = super::sha3::sha3_256(model_bytes);
    if computed_hash != *signature.metadata.model_hash() {
        return Err(VerifyError::HashMismatch);
    }

    // 2. Check key is in trusted key store
    let pk_hash = super::sha3::sha3_256(public_key.as_bytes());
    let mut pk_hash_bytes = [0u8; SHA3_256_DIGEST_LEN];
    pk_hash_bytes.copy_from_slice(pk_hash.as_bytes());
    if !trusted_keys.is_ml_dsa_trusted(&pk_hash_bytes) {
        return Err(VerifyError::UntrustedKey);
    }

    // 3. Serialize metadata and verify ML-DSA-65 signature
    let mut msg_buf = [0u8; SHA3_256_DIGEST_LEN + 4 + 8 + 2 + MAX_MODEL_NAME_LEN];
    let msg_len = signature
        .metadata
        .serialize(&mut msg_buf)
        .map_err(|_| VerifyError::InvalidSignatureFormat)?;

    let sig_bytes = signature.signature_bytes();
    let ml_dsa_sig =
        MlDsaSignature::from_slice(sig_bytes).map_err(|_| VerifyError::InvalidSignatureFormat)?;

    super::ml_dsa::ml_dsa_65_verify(public_key, &msg_buf[..msg_len], &ml_dsa_sig)
        .map_err(|_| VerifyError::SignatureInvalid)?;

    Ok(())
}

/// Verify an ONNX model's signature using hybrid Ed25519 + ML-DSA-65.
///
/// This is the default verification mode for SmallAIOS.
pub fn verify_model_hybrid(
    model_bytes: &[u8],
    signature: &ModelSignature,
    public_key: &HybridSigPublicKey,
    trusted_keys: &TrustedKeyStore,
) -> Result<(), VerifyError> {
    if model_bytes.len() > MAX_MODEL_SIZE {
        return Err(VerifyError::ModelTooLarge);
    }

    // 1. Compute SHA-3-256(model_bytes) and compare with metadata hash
    let computed_hash = super::sha3::sha3_256(model_bytes);
    if computed_hash != *signature.metadata.model_hash() {
        return Err(VerifyError::HashMismatch);
    }

    // 2. Check key is in trusted key store (hash the combined key)
    let mut combined_key = [0u8; HYBRID_SIG_PK_LEN];
    combined_key[..ED25519_PK_LEN].copy_from_slice(public_key.ed25519_pk());
    combined_key[ED25519_PK_LEN..].copy_from_slice(public_key.ml_dsa_pk());
    let pk_hash = super::sha3::sha3_256(&combined_key);
    let mut pk_hash_bytes = [0u8; SHA3_256_DIGEST_LEN];
    pk_hash_bytes.copy_from_slice(pk_hash.as_bytes());
    if !trusted_keys.is_hybrid_trusted(&pk_hash_bytes) {
        return Err(VerifyError::UntrustedKey);
    }

    // 3. Serialize metadata for verification
    let mut msg_buf = [0u8; SHA3_256_DIGEST_LEN + 4 + 8 + 2 + MAX_MODEL_NAME_LEN];
    let msg_len = signature
        .metadata
        .serialize(&mut msg_buf)
        .map_err(|_| VerifyError::InvalidSignatureFormat)?;

    // 4. Unpack hybrid signature and verify both components
    let sig_bytes = signature.signature_bytes();
    if sig_bytes.len() != HYBRID_SIG_LEN {
        return Err(VerifyError::InvalidSignatureFormat);
    }

    let mut ed_sig_bytes = [0u8; super::hybrid::ED25519_SIG_LEN];
    ed_sig_bytes.copy_from_slice(&sig_bytes[..super::hybrid::ED25519_SIG_LEN]);
    let mut ml_dsa_sig_bytes = [0u8; super::ml_dsa::ML_DSA_65_SIG_LEN];
    ml_dsa_sig_bytes.copy_from_slice(&sig_bytes[super::hybrid::ED25519_SIG_LEN..]);

    let hybrid_sig = HybridSignature::from_components(ed_sig_bytes, ml_dsa_sig_bytes);
    super::hybrid::hybrid_verify(public_key, &msg_buf[..msg_len], &hybrid_sig)
        .map_err(|_| VerifyError::HybridSignatureInvalid)?;

    Ok(())
}

/// Compute the SHA-3-256 hash of a model binary.
///
/// This is the first step in both signing and verification.
pub fn hash_model(model_bytes: &[u8]) -> Result<Sha3_256Digest, VerifyError> {
    if model_bytes.len() > MAX_MODEL_SIZE {
        return Err(VerifyError::ModelTooLarge);
    }
    Ok(super::sha3::sha3_256(model_bytes))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn verify_constants() {
        assert_eq!(MAX_MODEL_NAME_LEN, 256);
        assert_eq!(MAX_MODEL_SIZE, 2 * 1024 * 1024 * 1024);
        assert_eq!(SIGNATURE_FORMAT_VERSION, 1);
        assert_eq!(&SIGNATURE_MAGIC, b"SAIOS\x00\x01\x00");
        assert_eq!(MAX_TRUSTED_KEYS, 16);
    }

    #[test]
    fn signature_scheme_from_byte() {
        assert_eq!(
            SignatureScheme::from_byte(1),
            Some(SignatureScheme::MlDsa65)
        );
        assert_eq!(
            SignatureScheme::from_byte(2),
            Some(SignatureScheme::HybridEdMlDsa)
        );
        assert_eq!(SignatureScheme::from_byte(0), None);
        assert_eq!(SignatureScheme::from_byte(3), None);
    }

    #[test]
    fn model_metadata_creation() {
        let hash = Sha3_256Digest::from_bytes([0xAA; SHA3_256_DIGEST_LEN]);
        let meta = ModelMetadata::new(
            b"test-model",
            0x00_01_00_00, // v1.0.0
            1_700_000_000,
            hash,
            SignatureScheme::HybridEdMlDsa,
        )
        .unwrap();

        assert_eq!(meta.name(), b"test-model");
        assert_eq!(meta.version(), 0x00_01_00_00);
        assert_eq!(meta.timestamp(), 1_700_000_000);
        assert_eq!(meta.scheme(), SignatureScheme::HybridEdMlDsa);
    }

    #[test]
    fn model_metadata_name_too_long() {
        let hash = Sha3_256Digest::from_bytes([0; SHA3_256_DIGEST_LEN]);
        let long_name = [b'A'; MAX_MODEL_NAME_LEN + 1];
        let result = ModelMetadata::new(&long_name, 1, 0, hash, SignatureScheme::MlDsa65);
        assert_eq!(result.err(), Some(VerifyError::NameTooLong));
    }

    #[test]
    fn model_metadata_serialize() {
        let hash = Sha3_256Digest::from_bytes([0x42; SHA3_256_DIGEST_LEN]);
        let meta =
            ModelMetadata::new(b"mymodel", 1, 12345, hash, SignatureScheme::MlDsa65).unwrap();

        let mut buf = [0u8; 512];
        let len = meta.serialize(&mut buf).unwrap();

        // Expected: 32 (hash) + 4 (version) + 8 (timestamp) + 2 (name_len) + 7 (name) = 53
        assert_eq!(len, 53);

        // Verify hash is at the start
        assert_eq!(&buf[..32], &[0x42; 32]);

        // Verify version
        assert_eq!(&buf[32..36], &1u32.to_le_bytes());

        // Verify timestamp
        assert_eq!(&buf[36..44], &12345u64.to_le_bytes());

        // Verify name length
        assert_eq!(&buf[44..46], &7u16.to_le_bytes());

        // Verify name
        assert_eq!(&buf[46..53], b"mymodel");
    }

    #[test]
    fn trusted_key_store_empty() {
        let store = TrustedKeyStore::new();
        assert_eq!(store.ml_dsa_key_count(), 0);
        assert_eq!(store.hybrid_key_count(), 0);
        assert!(!store.is_ml_dsa_trusted(&[0; SHA3_256_DIGEST_LEN]));
    }

    #[test]
    fn trusted_key_store_add_and_find() {
        let mut store = TrustedKeyStore::new();
        let key_hash = [0xAB; SHA3_256_DIGEST_LEN];
        store.add_ml_dsa_key(key_hash).unwrap();

        assert_eq!(store.ml_dsa_key_count(), 1);
        assert!(store.is_ml_dsa_trusted(&key_hash));
        assert!(!store.is_ml_dsa_trusted(&[0; SHA3_256_DIGEST_LEN]));
    }

    #[test]
    fn trusted_key_store_add_hybrid() {
        let mut store = TrustedKeyStore::new();
        let key_hash = [0xCD; SHA3_256_DIGEST_LEN];
        store.add_hybrid_key(key_hash).unwrap();

        assert_eq!(store.hybrid_key_count(), 1);
        assert!(store.is_hybrid_trusted(&key_hash));
        assert!(!store.is_hybrid_trusted(&[0; SHA3_256_DIGEST_LEN]));
    }

    #[test]
    fn trusted_key_store_full() {
        let mut store = TrustedKeyStore::new();
        for i in 0..MAX_TRUSTED_KEYS {
            let mut hash = [0u8; SHA3_256_DIGEST_LEN];
            hash[0] = i as u8;
            store.add_ml_dsa_key(hash).unwrap();
        }
        assert_eq!(store.ml_dsa_key_count(), MAX_TRUSTED_KEYS);
        // Next add should fail
        assert_eq!(
            store.add_ml_dsa_key([0xFF; SHA3_256_DIGEST_LEN]),
            Err(VerifyError::UntrustedKey)
        );
    }

    #[test]
    fn hash_model_small() {
        let model_bytes = b"fake onnx model data";
        let hash = hash_model(model_bytes).unwrap();
        assert_eq!(hash.as_bytes().len(), SHA3_256_DIGEST_LEN);
        // The hash should match sha3_256 of the same data
        let expected = super::super::sha3::sha3_256(model_bytes);
        assert_eq!(hash, expected);
    }

    #[test]
    fn verify_model_ml_dsa_end_to_end() {
        let model = b"fake ONNX model binary data";
        let model_hash = hash_model(model).unwrap();

        // Generate ML-DSA-65 key pair
        let seed = [0x42u8; 32];
        let kp = super::super::ml_dsa::ml_dsa_65_keygen(&seed).unwrap();

        // Create metadata and serialize the message to sign
        let meta = ModelMetadata::new(
            b"test-model",
            1,
            1700000000,
            model_hash,
            SignatureScheme::MlDsa65,
        )
        .unwrap();
        let mut msg_buf = [0u8; 512];
        let msg_len = meta.serialize(&mut msg_buf).unwrap();

        // Sign the serialized metadata
        let rng_seed = [0x37u8; 32];
        let sig =
            super::super::ml_dsa::ml_dsa_65_sign(&kp.secret_key, &msg_buf[..msg_len], &rng_seed)
                .unwrap();

        let model_sig = ModelSignature::new_ml_dsa(meta, &sig);

        // Add key to trusted store
        let pk_hash = super::super::sha3::sha3_256(kp.public_key.as_bytes());
        let mut pk_hash_bytes = [0u8; SHA3_256_DIGEST_LEN];
        pk_hash_bytes.copy_from_slice(pk_hash.as_bytes());
        let mut store = TrustedKeyStore::new();
        store.add_ml_dsa_key(pk_hash_bytes).unwrap();

        // Verify
        let result = verify_model_ml_dsa(model, &model_sig, &kp.public_key, &store);
        assert!(result.is_ok(), "ML-DSA model verification should succeed");
    }

    #[test]
    fn verify_model_ml_dsa_hash_mismatch() {
        let model = b"real model";
        let wrong_hash = super::super::sha3::sha3_256(b"different model");
        let meta = ModelMetadata::new(b"test", 1, 0, wrong_hash, SignatureScheme::MlDsa65).unwrap();
        let sig_bytes = MlDsaSignature::from_bytes([0u8; ML_DSA_65_SIG_LEN]);
        let model_sig = ModelSignature::new_ml_dsa(meta, &sig_bytes);
        let pk = MlDsaPublicKey::from_bytes([0u8; ML_DSA_65_PK_LEN]);
        let store = TrustedKeyStore::new();

        assert_eq!(
            verify_model_ml_dsa(model, &model_sig, &pk, &store),
            Err(VerifyError::HashMismatch)
        );
    }

    #[test]
    fn verify_model_ml_dsa_untrusted_key() {
        let model = b"model data";
        let hash = hash_model(model).unwrap();
        let meta = ModelMetadata::new(b"test", 1, 0, hash, SignatureScheme::MlDsa65).unwrap();
        let sig_bytes = MlDsaSignature::from_bytes([0u8; ML_DSA_65_SIG_LEN]);
        let model_sig = ModelSignature::new_ml_dsa(meta, &sig_bytes);
        let pk = MlDsaPublicKey::from_bytes([0u8; ML_DSA_65_PK_LEN]);
        let store = TrustedKeyStore::new(); // empty store

        assert_eq!(
            verify_model_ml_dsa(model, &model_sig, &pk, &store),
            Err(VerifyError::UntrustedKey)
        );
    }

    #[test]
    fn verify_model_hybrid_end_to_end() {
        let model = b"fake ONNX model binary data for hybrid";
        let model_hash = hash_model(model).unwrap();

        // Generate hybrid key pair
        let seed = [0x42u8; 64];
        let kp = super::super::hybrid::hybrid_sig_keygen(&seed).unwrap();

        // Create metadata and serialize the message to sign
        let meta = ModelMetadata::new(
            b"hybrid-model",
            1,
            1700000000,
            model_hash,
            SignatureScheme::HybridEdMlDsa,
        )
        .unwrap();
        let mut msg_buf = [0u8; 512];
        let msg_len = meta.serialize(&mut msg_buf).unwrap();

        // Sign with hybrid
        let rng_seed = [0x37u8; 32];
        let sig = super::super::hybrid::hybrid_sign(&kp.secret_key, &msg_buf[..msg_len], &rng_seed)
            .unwrap();
        let model_sig = ModelSignature::new_hybrid(meta, &sig);

        // Add key to trusted store (hash the combined pk)
        let mut combined_key = [0u8; HYBRID_SIG_PK_LEN];
        combined_key[..ED25519_PK_LEN].copy_from_slice(kp.public_key.ed25519_pk());
        combined_key[ED25519_PK_LEN..].copy_from_slice(kp.public_key.ml_dsa_pk());
        let pk_hash = super::super::sha3::sha3_256(&combined_key);
        let mut pk_hash_bytes = [0u8; SHA3_256_DIGEST_LEN];
        pk_hash_bytes.copy_from_slice(pk_hash.as_bytes());
        let mut store = TrustedKeyStore::new();
        store.add_hybrid_key(pk_hash_bytes).unwrap();

        let result = verify_model_hybrid(model, &model_sig, &kp.public_key, &store);
        assert!(result.is_ok(), "hybrid model verification should succeed");
    }

    #[test]
    fn trusted_key_store_debug() {
        let store = TrustedKeyStore::new();
        let debug = format!("{:?}", store);
        assert!(debug.contains("TrustedKeyStore"));
        assert!(debug.contains("ml_dsa_keys"));
        assert!(debug.contains("hybrid_keys"));
    }

    // ---- Coverage for VerificationPolicy methods ----

    #[test]
    fn verification_policy_should_verify() {
        assert!(VerificationPolicy::Enforce.should_verify());
        assert!(VerificationPolicy::WarnOnly.should_verify());
        assert!(!VerificationPolicy::Disabled.should_verify());
    }

    #[test]
    fn verification_policy_should_reject() {
        assert!(VerificationPolicy::Enforce.should_reject());
        assert!(!VerificationPolicy::WarnOnly.should_reject());
        assert!(!VerificationPolicy::Disabled.should_reject());
    }

    // ---- Coverage for VerifyError Display ----

    #[test]
    fn verify_error_display() {
        assert_eq!(
            format!("{}", VerifyError::ModelTooLarge),
            format!("model exceeds maximum size ({} bytes)", MAX_MODEL_SIZE)
        );
        assert_eq!(
            format!("{}", VerifyError::NameTooLong),
            format!("model name exceeds {} bytes", MAX_MODEL_NAME_LEN)
        );
        assert_eq!(
            format!("{}", VerifyError::InvalidSignatureFormat),
            "invalid model signature format"
        );
        assert_eq!(
            format!("{}", VerifyError::UnsupportedVersion),
            "unsupported signature format version"
        );
        assert_eq!(
            format!("{}", VerifyError::HashMismatch),
            "model hash does not match signature"
        );
        assert_eq!(
            format!("{}", VerifyError::SignatureInvalid),
            "ML-DSA-65 signature verification failed"
        );
        assert_eq!(
            format!("{}", VerifyError::HybridSignatureInvalid),
            "hybrid signature verification failed"
        );
        assert_eq!(
            format!("{}", VerifyError::UntrustedKey),
            "signing key not in trusted key set"
        );
        assert_eq!(
            format!("{}", VerifyError::TimestampOutOfRange),
            "signature timestamp out of acceptable range"
        );
        assert_eq!(
            format!("{}", VerifyError::NoSignature),
            "no signature found for model"
        );
        assert_eq!(
            format!("{}", VerifyError::NotImplemented),
            "model verification not yet implemented"
        );
    }

    // ---- Coverage for ModelMetadata::serialize error path ----

    #[test]
    fn model_metadata_serialize_buffer_too_small() {
        let hash = Sha3_256Digest::from_bytes([0x42; SHA3_256_DIGEST_LEN]);
        let meta =
            ModelMetadata::new(b"mymodel", 1, 12345, hash, SignatureScheme::MlDsa65).unwrap();
        let mut tiny_buf = [0u8; 4]; // way too small
        assert_eq!(
            meta.serialize(&mut tiny_buf),
            Err(VerifyError::InvalidSignatureFormat)
        );
    }
}
