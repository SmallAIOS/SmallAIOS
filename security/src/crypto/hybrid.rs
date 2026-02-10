// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hybrid post-quantum + classical cryptographic schemes.
//!
//! SmallAIOS defaults to hybrid mode, combining classical and post-quantum
//! algorithms for defense-in-depth:
//!
//! - **Hybrid KEM**: X25519 + ML-KEM-768
//!   - Classical ECDH (X25519) for immediate security
//!   - ML-KEM-768 for post-quantum resistance
//!   - Combined shared secret = SHA-3-256(X25519_ss || ML-KEM_ss)
//!
//! - **Hybrid Signature**: Ed25519 + ML-DSA-65
//!   - Classical EdDSA (Ed25519) for compatibility
//!   - ML-DSA-65 for post-quantum resistance
//!   - Both signatures must verify for acceptance
//!
//! # Rationale
//!
//! Hybrid mode ensures that if either the classical or post-quantum algorithm
//! is broken, the combined scheme remains secure. This follows NIST guidance
//! and CNSA 2.0 transition recommendations.
//!
//! # Status
//!
//! This module defines the API surface. The classical primitives (X25519,
//! Ed25519) and the composition logic are pending implementation.

#![allow(unused)]

use core::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

/// X25519 public key length in bytes.
pub const X25519_PK_LEN: usize = 32;

/// X25519 secret key length in bytes.
pub const X25519_SK_LEN: usize = 32;

/// X25519 shared secret length in bytes.
pub const X25519_SS_LEN: usize = 32;

/// Ed25519 public key length in bytes.
pub const ED25519_PK_LEN: usize = 32;

/// Ed25519 secret key length in bytes (seed + public key).
pub const ED25519_SK_LEN: usize = 64;

/// Ed25519 signature length in bytes.
pub const ED25519_SIG_LEN: usize = 64;

/// Hybrid KEM combined public key length: X25519 (32) + ML-KEM-768 (1184).
pub const HYBRID_KEM_PK_LEN: usize = X25519_PK_LEN + super::ml_kem::ML_KEM_768_PK_LEN;

/// Hybrid KEM combined secret key length: X25519 (32) + ML-KEM-768 (2400).
pub const HYBRID_KEM_SK_LEN: usize = X25519_SK_LEN + super::ml_kem::ML_KEM_768_SK_LEN;

/// Hybrid KEM combined ciphertext length: X25519 ephemeral (32) + ML-KEM-768 (1088).
pub const HYBRID_KEM_CT_LEN: usize = X25519_PK_LEN + super::ml_kem::ML_KEM_768_CT_LEN;

/// Hybrid KEM shared secret length: SHA-3-256 output (32 bytes).
pub const HYBRID_KEM_SS_LEN: usize = 32;

/// Hybrid signature combined public key length: Ed25519 (32) + ML-DSA-65 (1952).
pub const HYBRID_SIG_PK_LEN: usize = ED25519_PK_LEN + super::ml_dsa::ML_DSA_65_PK_LEN;

/// Hybrid signature combined secret key length: Ed25519 (64) + ML-DSA-65 (4032).
pub const HYBRID_SIG_SK_LEN: usize = ED25519_SK_LEN + super::ml_dsa::ML_DSA_65_SK_LEN;

/// Hybrid signature combined signature length: Ed25519 (64) + ML-DSA-65 (3309).
pub const HYBRID_SIG_LEN: usize = ED25519_SIG_LEN + super::ml_dsa::ML_DSA_65_SIG_LEN;

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from hybrid cryptographic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridError {
    /// Invalid public key length.
    InvalidPublicKeyLength,
    /// Invalid secret key length.
    InvalidSecretKeyLength,
    /// Invalid ciphertext length.
    InvalidCiphertextLength,
    /// Invalid signature length.
    InvalidSignatureLength,
    /// Classical (X25519/Ed25519) component failed.
    ClassicalFailure,
    /// Post-quantum (ML-KEM/ML-DSA) component failed.
    PostQuantumFailure,
    /// Signature verification failed (either component).
    VerificationFailed,
    /// RNG failure.
    RngFailure,
    /// Operation not yet implemented.
    NotImplemented,
}

impl fmt::Display for HybridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPublicKeyLength => write!(f, "invalid hybrid public key length"),
            Self::InvalidSecretKeyLength => write!(f, "invalid hybrid secret key length"),
            Self::InvalidCiphertextLength => write!(f, "invalid hybrid ciphertext length"),
            Self::InvalidSignatureLength => write!(f, "invalid hybrid signature length"),
            Self::ClassicalFailure => write!(f, "classical cryptographic component failure"),
            Self::PostQuantumFailure => write!(f, "post-quantum cryptographic component failure"),
            Self::VerificationFailed => write!(f, "hybrid signature verification failed"),
            Self::RngFailure => write!(f, "random number generation failure"),
            Self::NotImplemented => write!(f, "hybrid scheme not yet implemented"),
        }
    }
}

// ─── Hybrid KEM Types ────────────────────────────────────────────────────────

/// Hybrid KEM public key: X25519 public key || ML-KEM-768 public key.
#[derive(PartialEq, Eq)]
pub struct HybridKemPublicKey {
    /// X25519 component (32 bytes).
    x25519: [u8; X25519_PK_LEN],
    /// ML-KEM-768 component (1184 bytes).
    ml_kem: [u8; super::ml_kem::ML_KEM_768_PK_LEN],
}

impl HybridKemPublicKey {
    /// Create from individual components.
    pub fn from_components(
        x25519: [u8; X25519_PK_LEN],
        ml_kem: [u8; super::ml_kem::ML_KEM_768_PK_LEN],
    ) -> Self {
        Self { x25519, ml_kem }
    }

    /// Return the X25519 component.
    pub fn x25519_pk(&self) -> &[u8; X25519_PK_LEN] {
        &self.x25519
    }

    /// Return the ML-KEM-768 component.
    pub fn ml_kem_pk(&self) -> &[u8] {
        &self.ml_kem
    }

    /// Total combined key length.
    pub fn len(&self) -> usize {
        HYBRID_KEM_PK_LEN
    }
}

impl fmt::Debug for HybridKemPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridKemPublicKey({} bytes)", HYBRID_KEM_PK_LEN)
    }
}

/// Hybrid KEM secret key: X25519 secret key || ML-KEM-768 secret key.
#[derive(PartialEq, Eq)]
pub struct HybridKemSecretKey {
    x25519: [u8; X25519_SK_LEN],
    ml_kem: [u8; super::ml_kem::ML_KEM_768_SK_LEN],
}

impl HybridKemSecretKey {
    /// Create from individual components.
    pub fn from_components(
        x25519: [u8; X25519_SK_LEN],
        ml_kem: [u8; super::ml_kem::ML_KEM_768_SK_LEN],
    ) -> Self {
        Self { x25519, ml_kem }
    }

    /// Return the X25519 component.
    pub fn x25519_sk(&self) -> &[u8; X25519_SK_LEN] {
        &self.x25519
    }

    /// Return the ML-KEM-768 component.
    pub fn ml_kem_sk(&self) -> &[u8] {
        &self.ml_kem
    }

    /// Total combined key length.
    pub fn len(&self) -> usize {
        HYBRID_KEM_SK_LEN
    }
}

impl fmt::Debug for HybridKemSecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridKemSecretKey([REDACTED])")
    }
}

/// Hybrid KEM ciphertext: X25519 ephemeral public key || ML-KEM-768 ciphertext.
#[derive(PartialEq, Eq)]
pub struct HybridKemCiphertext {
    x25519_ephemeral: [u8; X25519_PK_LEN],
    ml_kem_ct: [u8; super::ml_kem::ML_KEM_768_CT_LEN],
}

impl HybridKemCiphertext {
    /// Create from individual components.
    pub fn from_components(
        x25519_ephemeral: [u8; X25519_PK_LEN],
        ml_kem_ct: [u8; super::ml_kem::ML_KEM_768_CT_LEN],
    ) -> Self {
        Self {
            x25519_ephemeral,
            ml_kem_ct,
        }
    }

    /// Return the X25519 ephemeral public key.
    pub fn x25519_ephemeral(&self) -> &[u8; X25519_PK_LEN] {
        &self.x25519_ephemeral
    }

    /// Return the ML-KEM-768 ciphertext.
    pub fn ml_kem_ct(&self) -> &[u8] {
        &self.ml_kem_ct
    }

    /// Total combined ciphertext length.
    pub fn len(&self) -> usize {
        HYBRID_KEM_CT_LEN
    }
}

impl fmt::Debug for HybridKemCiphertext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridKemCiphertext({} bytes)", HYBRID_KEM_CT_LEN)
    }
}

/// Hybrid KEM shared secret (32 bytes, derived from both components).
#[derive(Clone, PartialEq, Eq)]
pub struct HybridKemSharedSecret {
    bytes: [u8; HYBRID_KEM_SS_LEN],
}

impl HybridKemSharedSecret {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: [u8; HYBRID_KEM_SS_LEN]) -> Self {
        Self { bytes }
    }

    /// Return the shared secret bytes.
    pub fn as_bytes(&self) -> &[u8; HYBRID_KEM_SS_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for HybridKemSharedSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridKemSharedSecret([REDACTED])")
    }
}

/// Hybrid KEM key pair.
#[derive(PartialEq, Eq)]
pub struct HybridKemKeyPair {
    pub public_key: HybridKemPublicKey,
    pub secret_key: HybridKemSecretKey,
}

impl fmt::Debug for HybridKemKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HybridKemKeyPair")
            .field("public_key", &self.public_key)
            .field("secret_key", &self.secret_key)
            .finish()
    }
}

// ─── Hybrid Signature Types ─────────────────────────────────────────────────

/// Hybrid signature public key: Ed25519 public key || ML-DSA-65 public key.
#[derive(PartialEq, Eq)]
pub struct HybridSigPublicKey {
    ed25519: [u8; ED25519_PK_LEN],
    ml_dsa: [u8; super::ml_dsa::ML_DSA_65_PK_LEN],
}

impl HybridSigPublicKey {
    /// Create from individual components.
    pub fn from_components(
        ed25519: [u8; ED25519_PK_LEN],
        ml_dsa: [u8; super::ml_dsa::ML_DSA_65_PK_LEN],
    ) -> Self {
        Self { ed25519, ml_dsa }
    }

    /// Return the Ed25519 component.
    pub fn ed25519_pk(&self) -> &[u8; ED25519_PK_LEN] {
        &self.ed25519
    }

    /// Return the ML-DSA-65 component.
    pub fn ml_dsa_pk(&self) -> &[u8] {
        &self.ml_dsa
    }

    /// Total combined key length.
    pub fn len(&self) -> usize {
        HYBRID_SIG_PK_LEN
    }
}

impl fmt::Debug for HybridSigPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridSigPublicKey({} bytes)", HYBRID_SIG_PK_LEN)
    }
}

/// Hybrid signature secret key: Ed25519 secret key || ML-DSA-65 secret key.
#[derive(PartialEq, Eq)]
pub struct HybridSigSecretKey {
    ed25519: [u8; ED25519_SK_LEN],
    ml_dsa: [u8; super::ml_dsa::ML_DSA_65_SK_LEN],
}

impl HybridSigSecretKey {
    /// Create from individual components.
    pub fn from_components(
        ed25519: [u8; ED25519_SK_LEN],
        ml_dsa: [u8; super::ml_dsa::ML_DSA_65_SK_LEN],
    ) -> Self {
        Self { ed25519, ml_dsa }
    }

    /// Return the Ed25519 component.
    pub fn ed25519_sk(&self) -> &[u8; ED25519_SK_LEN] {
        &self.ed25519
    }

    /// Return the ML-DSA-65 component.
    pub fn ml_dsa_sk(&self) -> &[u8] {
        &self.ml_dsa
    }

    /// Total combined key length.
    pub fn len(&self) -> usize {
        HYBRID_SIG_SK_LEN
    }
}

impl fmt::Debug for HybridSigSecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridSigSecretKey([REDACTED])")
    }
}

/// Hybrid signature: Ed25519 signature || ML-DSA-65 signature.
///
/// Both components must verify independently for the hybrid
/// signature to be considered valid.
#[derive(PartialEq, Eq)]
pub struct HybridSignature {
    ed25519_sig: [u8; ED25519_SIG_LEN],
    ml_dsa_sig: [u8; super::ml_dsa::ML_DSA_65_SIG_LEN],
}

impl HybridSignature {
    /// Create from individual components.
    pub fn from_components(
        ed25519_sig: [u8; ED25519_SIG_LEN],
        ml_dsa_sig: [u8; super::ml_dsa::ML_DSA_65_SIG_LEN],
    ) -> Self {
        Self {
            ed25519_sig,
            ml_dsa_sig,
        }
    }

    /// Return the Ed25519 signature component.
    pub fn ed25519_sig(&self) -> &[u8; ED25519_SIG_LEN] {
        &self.ed25519_sig
    }

    /// Return the ML-DSA-65 signature component.
    pub fn ml_dsa_sig(&self) -> &[u8] {
        &self.ml_dsa_sig
    }

    /// Total combined signature length.
    pub fn len(&self) -> usize {
        HYBRID_SIG_LEN
    }
}

impl fmt::Debug for HybridSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridSignature({} bytes)", HYBRID_SIG_LEN)
    }
}

/// Hybrid signature key pair.
#[derive(PartialEq, Eq)]
pub struct HybridSigKeyPair {
    pub public_key: HybridSigPublicKey,
    pub secret_key: HybridSigSecretKey,
}

impl fmt::Debug for HybridSigKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HybridSigKeyPair")
            .field("public_key", &self.public_key)
            .field("secret_key", &self.secret_key)
            .finish()
    }
}

// ─── Hybrid KEM Operations ──────────────────────────────────────────────────

/// Generate a hybrid KEM key pair (X25519 + ML-KEM-768).
///
/// Requires 96 bytes of randomness: 32 for X25519 + 64 for ML-KEM-768.
pub fn hybrid_kem_keygen(seed: &[u8; 96]) -> Result<HybridKemKeyPair, HybridError> {
    // Stub: generate X25519 keypair from seed[0..32],
    //       generate ML-KEM-768 keypair from seed[32..96]
    Err(HybridError::NotImplemented)
}

/// Hybrid encapsulation: produce a ciphertext and combined shared secret.
///
/// The shared secret is SHA-3-256(x25519_ss || ml_kem_ss).
pub fn hybrid_kem_encaps(
    pk: &HybridKemPublicKey,
    seed: &[u8; 64],
) -> Result<(HybridKemCiphertext, HybridKemSharedSecret), HybridError> {
    // Stub: X25519 ECDH + ML-KEM-768 encaps, combine with SHA-3-256
    Err(HybridError::NotImplemented)
}

/// Hybrid decapsulation: recover the combined shared secret.
pub fn hybrid_kem_decaps(
    sk: &HybridKemSecretKey,
    ct: &HybridKemCiphertext,
) -> Result<HybridKemSharedSecret, HybridError> {
    // Stub: X25519 ECDH + ML-KEM-768 decaps, combine with SHA-3-256
    Err(HybridError::NotImplemented)
}

// ─── Hybrid Signature Operations ────────────────────────────────────────────

/// Generate a hybrid signature key pair (Ed25519 + ML-DSA-65).
///
/// Requires 64 bytes of randomness: 32 for Ed25519 + 32 for ML-DSA-65.
pub fn hybrid_sig_keygen(seed: &[u8; 64]) -> Result<HybridSigKeyPair, HybridError> {
    // Stub: generate Ed25519 keypair from seed[0..32],
    //       generate ML-DSA-65 keypair from seed[32..64]
    Err(HybridError::NotImplemented)
}

/// Hybrid signing: produce Ed25519 + ML-DSA-65 signatures.
///
/// Both algorithms sign the same message independently.
pub fn hybrid_sign(
    sk: &HybridSigSecretKey,
    message: &[u8],
    random_seed: &[u8; 32],
) -> Result<HybridSignature, HybridError> {
    // Stub: Ed25519 sign + ML-DSA-65 sign
    Err(HybridError::NotImplemented)
}

/// Hybrid verification: both Ed25519 and ML-DSA-65 signatures must verify.
///
/// This is the strictest mode — if either component fails, the entire
/// verification fails. This ensures security even if one algorithm is broken.
pub fn hybrid_verify(
    pk: &HybridSigPublicKey,
    message: &[u8],
    signature: &HybridSignature,
) -> Result<(), HybridError> {
    // Stub: verify Ed25519 AND ML-DSA-65
    Err(HybridError::NotImplemented)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use alloc::format;
    use super::*;

    #[test]
    fn hybrid_kem_constant_sizes() {
        assert_eq!(HYBRID_KEM_PK_LEN, 32 + 1184); // 1216
        assert_eq!(HYBRID_KEM_SK_LEN, 32 + 2400); // 2432
        assert_eq!(HYBRID_KEM_CT_LEN, 32 + 1088); // 1120
        assert_eq!(HYBRID_KEM_SS_LEN, 32);
    }

    #[test]
    fn hybrid_sig_constant_sizes() {
        assert_eq!(HYBRID_SIG_PK_LEN, 32 + 1952); // 1984
        assert_eq!(HYBRID_SIG_SK_LEN, 64 + 4032); // 4096
        assert_eq!(HYBRID_SIG_LEN, 64 + 3309);    // 3373
    }

    #[test]
    fn hybrid_kem_public_key_construction() {
        let pk = HybridKemPublicKey::from_components(
            [0x01; X25519_PK_LEN],
            [0x02; super::super::ml_kem::ML_KEM_768_PK_LEN],
        );
        assert_eq!(pk.x25519_pk()[0], 0x01);
        assert_eq!(pk.ml_kem_pk()[0], 0x02);
        assert_eq!(pk.len(), HYBRID_KEM_PK_LEN);
    }

    #[test]
    fn hybrid_kem_secret_key_debug_redacted() {
        let sk = HybridKemSecretKey::from_components(
            [0; X25519_SK_LEN],
            [0; super::super::ml_kem::ML_KEM_768_SK_LEN],
        );
        let debug = format!("{:?}", sk);
        assert_eq!(debug, "HybridKemSecretKey([REDACTED])");
    }

    #[test]
    fn hybrid_kem_ciphertext_construction() {
        let ct = HybridKemCiphertext::from_components(
            [0xAA; X25519_PK_LEN],
            [0xBB; super::super::ml_kem::ML_KEM_768_CT_LEN],
        );
        assert_eq!(ct.x25519_ephemeral()[0], 0xAA);
        assert_eq!(ct.ml_kem_ct()[0], 0xBB);
        assert_eq!(ct.len(), HYBRID_KEM_CT_LEN);
    }

    #[test]
    fn hybrid_kem_shared_secret_redacted() {
        let ss = HybridKemSharedSecret::from_bytes([0xFF; HYBRID_KEM_SS_LEN]);
        let debug = format!("{:?}", ss);
        assert_eq!(debug, "HybridKemSharedSecret([REDACTED])");
    }

    #[test]
    fn hybrid_sig_public_key_construction() {
        let pk = HybridSigPublicKey::from_components(
            [0x01; ED25519_PK_LEN],
            [0x02; super::super::ml_dsa::ML_DSA_65_PK_LEN],
        );
        assert_eq!(pk.ed25519_pk()[0], 0x01);
        assert_eq!(pk.ml_dsa_pk()[0], 0x02);
        assert_eq!(pk.len(), HYBRID_SIG_PK_LEN);
    }

    #[test]
    fn hybrid_sig_secret_key_debug_redacted() {
        let sk = HybridSigSecretKey::from_components(
            [0; ED25519_SK_LEN],
            [0; super::super::ml_dsa::ML_DSA_65_SK_LEN],
        );
        let debug = format!("{:?}", sk);
        assert_eq!(debug, "HybridSigSecretKey([REDACTED])");
    }

    #[test]
    fn hybrid_signature_construction() {
        let sig = HybridSignature::from_components(
            [0xCC; ED25519_SIG_LEN],
            [0xDD; super::super::ml_dsa::ML_DSA_65_SIG_LEN],
        );
        assert_eq!(sig.ed25519_sig()[0], 0xCC);
        assert_eq!(sig.ml_dsa_sig()[0], 0xDD);
        assert_eq!(sig.len(), HYBRID_SIG_LEN);
    }

    #[test]
    fn hybrid_kem_keygen_stub() {
        let seed = [0u8; 96];
        assert_eq!(hybrid_kem_keygen(&seed), Err(HybridError::NotImplemented));
    }

    #[test]
    fn hybrid_kem_encaps_stub() {
        let pk = HybridKemPublicKey::from_components(
            [0; X25519_PK_LEN],
            [0; super::super::ml_kem::ML_KEM_768_PK_LEN],
        );
        let seed = [0u8; 64];
        assert_eq!(hybrid_kem_encaps(&pk, &seed), Err(HybridError::NotImplemented));
    }

    #[test]
    fn hybrid_kem_decaps_stub() {
        let sk = HybridKemSecretKey::from_components(
            [0; X25519_SK_LEN],
            [0; super::super::ml_kem::ML_KEM_768_SK_LEN],
        );
        let ct = HybridKemCiphertext::from_components(
            [0; X25519_PK_LEN],
            [0; super::super::ml_kem::ML_KEM_768_CT_LEN],
        );
        assert_eq!(hybrid_kem_decaps(&sk, &ct), Err(HybridError::NotImplemented));
    }

    #[test]
    fn hybrid_sig_keygen_stub() {
        let seed = [0u8; 64];
        assert_eq!(hybrid_sig_keygen(&seed), Err(HybridError::NotImplemented));
    }

    #[test]
    fn hybrid_sign_stub() {
        let sk = HybridSigSecretKey::from_components(
            [0; ED25519_SK_LEN],
            [0; super::super::ml_dsa::ML_DSA_65_SK_LEN],
        );
        let seed = [0u8; 32];
        assert_eq!(hybrid_sign(&sk, b"msg", &seed), Err(HybridError::NotImplemented));
    }

    #[test]
    fn hybrid_verify_stub() {
        let pk = HybridSigPublicKey::from_components(
            [0; ED25519_PK_LEN],
            [0; super::super::ml_dsa::ML_DSA_65_PK_LEN],
        );
        let sig = HybridSignature::from_components(
            [0; ED25519_SIG_LEN],
            [0; super::super::ml_dsa::ML_DSA_65_SIG_LEN],
        );
        assert_eq!(hybrid_verify(&pk, b"msg", &sig), Err(HybridError::NotImplemented));
    }
}
