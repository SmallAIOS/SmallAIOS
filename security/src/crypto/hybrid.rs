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
//! # Implementation
//!
//! Hybrid KEM uses X25519 for the classical component and ML-KEM-768 for PQC.
//! Hybrid signatures use Ed25519 for the classical component and ML-DSA-65 for PQC.

#![allow(unused)]

use super::ed25519::{ed25519_keygen, ed25519_sign, ed25519_verify};
use super::ml_dsa::{
    ml_dsa_65_keygen, ml_dsa_65_sign, ml_dsa_65_verify, MlDsaPublicKey, MlDsaSecretKey,
    MlDsaSignature,
};
use super::ml_kem::{
    ml_kem_768_decaps, ml_kem_768_encaps, ml_kem_768_keygen, MlKemPublicKey, MlKemSecretKey,
};
use super::sha3::sha3_256;
use super::x25519::{x25519_dh, x25519_keygen, X25519PublicKey, X25519SecretKey};
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

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Returns whether the ciphertext is empty (always false for fixed-size types).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Returns whether the signature is empty (always false for fixed-size types).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    // Generate X25519 key pair from seed[0..32]
    let x_seed: &[u8; 32] = seed[..32].try_into().unwrap();
    let x_kp = x25519_keygen(x_seed);

    // Generate ML-KEM-768 key pair from seed[32..96]
    let kem_seed: &[u8; 64] = seed[32..96].try_into().unwrap();
    let kem_kp = ml_kem_768_keygen(kem_seed).map_err(|_| HybridError::PostQuantumFailure)?;

    let mut ml_kem_pk = [0u8; super::ml_kem::ML_KEM_768_PK_LEN];
    ml_kem_pk.copy_from_slice(kem_kp.public_key.as_bytes());
    let mut ml_kem_sk = [0u8; super::ml_kem::ML_KEM_768_SK_LEN];
    ml_kem_sk.copy_from_slice(kem_kp.secret_key.as_bytes());

    Ok(HybridKemKeyPair {
        public_key: HybridKemPublicKey::from_components(*x_kp.public_key.as_bytes(), ml_kem_pk),
        secret_key: HybridKemSecretKey::from_components(*x_kp.secret_key.as_bytes(), ml_kem_sk),
    })
}

/// Hybrid encapsulation: produce a ciphertext and combined shared secret.
///
/// The shared secret is SHA-3-256(x25519_ss || ml_kem_ss).
/// seed: 32 bytes for X25519 ephemeral + 32 bytes for ML-KEM encaps randomness.
pub fn hybrid_kem_encaps(
    pk: &HybridKemPublicKey,
    seed: &[u8; 64],
) -> Result<(HybridKemCiphertext, HybridKemSharedSecret), HybridError> {
    // X25519: generate ephemeral key pair and compute DH shared secret
    let x_eph_seed: &[u8; 32] = seed[..32].try_into().unwrap();
    let x_eph_kp = x25519_keygen(x_eph_seed);
    let x_peer_pk = X25519PublicKey::from_bytes(*pk.x25519_pk());
    let x_ss =
        x25519_dh(&x_eph_kp.secret_key, &x_peer_pk).map_err(|_| HybridError::ClassicalFailure)?;

    // ML-KEM-768: encapsulate
    let kem_seed: &[u8; 32] = seed[32..64].try_into().unwrap();
    let ml_kem_pk =
        MlKemPublicKey::from_slice(pk.ml_kem_pk()).map_err(|_| HybridError::PostQuantumFailure)?;
    let (kem_ct, kem_ss) =
        ml_kem_768_encaps(&ml_kem_pk, kem_seed).map_err(|_| HybridError::PostQuantumFailure)?;

    // Combine shared secrets: SHA-3-256(x25519_ss || ml_kem_ss)
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&x_ss);
    combined[32..].copy_from_slice(kem_ss.as_bytes());
    let digest = sha3_256(&combined);
    let mut ss_bytes = [0u8; HYBRID_KEM_SS_LEN];
    ss_bytes.copy_from_slice(digest.as_bytes());

    // Build ciphertext: X25519 ephemeral public key || ML-KEM ciphertext
    let mut ml_kem_ct_bytes = [0u8; super::ml_kem::ML_KEM_768_CT_LEN];
    ml_kem_ct_bytes.copy_from_slice(kem_ct.as_bytes());

    Ok((
        HybridKemCiphertext::from_components(*x_eph_kp.public_key.as_bytes(), ml_kem_ct_bytes),
        HybridKemSharedSecret::from_bytes(ss_bytes),
    ))
}

/// Hybrid decapsulation: recover the combined shared secret.
pub fn hybrid_kem_decaps(
    sk: &HybridKemSecretKey,
    ct: &HybridKemCiphertext,
) -> Result<HybridKemSharedSecret, HybridError> {
    // X25519: DH with ephemeral public key from ciphertext
    let x_sk = X25519SecretKey::from_bytes(*sk.x25519_sk());
    let x_eph_pk = X25519PublicKey::from_bytes(*ct.x25519_ephemeral());
    let x_ss = x25519_dh(&x_sk, &x_eph_pk).map_err(|_| HybridError::ClassicalFailure)?;

    // ML-KEM-768: decapsulate
    let ml_kem_sk =
        MlKemSecretKey::from_slice(sk.ml_kem_sk()).map_err(|_| HybridError::PostQuantumFailure)?;
    let ml_kem_ct = super::ml_kem::MlKemCiphertext::from_slice(ct.ml_kem_ct())
        .map_err(|_| HybridError::PostQuantumFailure)?;
    let kem_ss =
        ml_kem_768_decaps(&ml_kem_sk, &ml_kem_ct).map_err(|_| HybridError::PostQuantumFailure)?;

    // Combine shared secrets: SHA-3-256(x25519_ss || ml_kem_ss)
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&x_ss);
    combined[32..].copy_from_slice(kem_ss.as_bytes());
    let digest = sha3_256(&combined);
    let mut ss_bytes = [0u8; HYBRID_KEM_SS_LEN];
    ss_bytes.copy_from_slice(digest.as_bytes());

    Ok(HybridKemSharedSecret::from_bytes(ss_bytes))
}

// ─── Hybrid Signature Operations ────────────────────────────────────────────

/// Generate a hybrid signature key pair (Ed25519 + ML-DSA-65).
///
/// Requires 64 bytes of randomness: 32 for Ed25519 + 32 for ML-DSA-65.
pub fn hybrid_sig_keygen(seed: &[u8; 64]) -> Result<HybridSigKeyPair, HybridError> {
    // Generate Ed25519 key pair from seed[0..32]
    let ed_seed: &[u8; 32] = seed[..32].try_into().unwrap();
    let ed_kp = ed25519_keygen(ed_seed);

    // Generate ML-DSA-65 key pair from seed[32..64]
    let dsa_seed: &[u8; 32] = seed[32..64].try_into().unwrap();
    let dsa_kp = ml_dsa_65_keygen(dsa_seed).map_err(|_| HybridError::PostQuantumFailure)?;

    let mut ml_dsa_pk = [0u8; super::ml_dsa::ML_DSA_65_PK_LEN];
    ml_dsa_pk.copy_from_slice(dsa_kp.public_key.as_bytes());
    let mut ml_dsa_sk = [0u8; super::ml_dsa::ML_DSA_65_SK_LEN];
    ml_dsa_sk.copy_from_slice(dsa_kp.secret_key.as_bytes());

    Ok(HybridSigKeyPair {
        public_key: HybridSigPublicKey::from_components(*ed_kp.public_key.as_bytes(), ml_dsa_pk),
        secret_key: HybridSigSecretKey::from_components(*ed_kp.secret_key.as_bytes(), ml_dsa_sk),
    })
}

/// Hybrid signing: produce Ed25519 + ML-DSA-65 signatures.
///
/// Both algorithms sign the same message independently.
pub fn hybrid_sign(
    sk: &HybridSigSecretKey,
    message: &[u8],
    random_seed: &[u8; 32],
) -> Result<HybridSignature, HybridError> {
    // Ed25519 sign (deterministic, doesn't use random_seed)
    let ed_sk = super::ed25519::Ed25519SecretKey::from_bytes(*sk.ed25519_sk());
    let ed_sig = ed25519_sign(&ed_sk, message);

    // ML-DSA-65 sign (hedged with random_seed)
    let dsa_sk =
        MlDsaSecretKey::from_slice(sk.ml_dsa_sk()).map_err(|_| HybridError::PostQuantumFailure)?;
    let dsa_sig = ml_dsa_65_sign(&dsa_sk, message, random_seed)
        .map_err(|_| HybridError::PostQuantumFailure)?;

    let mut ml_dsa_sig_bytes = [0u8; super::ml_dsa::ML_DSA_65_SIG_LEN];
    ml_dsa_sig_bytes.copy_from_slice(dsa_sig.as_bytes());

    Ok(HybridSignature::from_components(
        *ed_sig.as_bytes(),
        ml_dsa_sig_bytes,
    ))
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
    // Verify Ed25519 component
    let ed_pk = super::ed25519::Ed25519PublicKey::from_bytes(*pk.ed25519_pk());
    let ed_sig = super::ed25519::Ed25519Signature::from_bytes(*signature.ed25519_sig());
    ed25519_verify(&ed_pk, message, &ed_sig).map_err(|_| HybridError::VerificationFailed)?;

    // Verify ML-DSA-65 component
    let dsa_pk =
        MlDsaPublicKey::from_slice(pk.ml_dsa_pk()).map_err(|_| HybridError::VerificationFailed)?;
    let dsa_sig = MlDsaSignature::from_slice(signature.ml_dsa_sig())
        .map_err(|_| HybridError::VerificationFailed)?;
    ml_dsa_65_verify(&dsa_pk, message, &dsa_sig).map_err(|_| HybridError::VerificationFailed)?;

    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

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
        assert_eq!(HYBRID_SIG_LEN, 64 + 3309); // 3373
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
    fn hybrid_kem_keygen_encaps_decaps_roundtrip() {
        let mut seed = [0u8; 96];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = i as u8;
        }
        let kp = hybrid_kem_keygen(&seed).expect("keygen should succeed");
        assert_eq!(kp.public_key.len(), HYBRID_KEM_PK_LEN);
        assert_eq!(kp.secret_key.len(), HYBRID_KEM_SK_LEN);

        let encaps_seed = [0x42u8; 64];
        let (ct, ss_encaps) =
            hybrid_kem_encaps(&kp.public_key, &encaps_seed).expect("encaps should succeed");
        assert_eq!(ct.len(), HYBRID_KEM_CT_LEN);

        let ss_decaps = hybrid_kem_decaps(&kp.secret_key, &ct).expect("decaps should succeed");

        assert_eq!(
            ss_encaps.as_bytes(),
            ss_decaps.as_bytes(),
            "hybrid KEM encaps/decaps shared secrets must match"
        );
    }

    #[test]
    fn hybrid_kem_deterministic() {
        let seed = [0xBBu8; 96];
        let kp1 = hybrid_kem_keygen(&seed).unwrap();
        let kp2 = hybrid_kem_keygen(&seed).unwrap();
        assert_eq!(kp1.public_key.x25519_pk(), kp2.public_key.x25519_pk());
        assert_eq!(kp1.public_key.ml_kem_pk(), kp2.public_key.ml_kem_pk());
    }

    #[test]
    fn hybrid_kem_ss_not_all_zeros() {
        let seed = [0x42u8; 96];
        let kp = hybrid_kem_keygen(&seed).unwrap();
        let encaps_seed = [0x01u8; 64];
        let (_ct, ss) = hybrid_kem_encaps(&kp.public_key, &encaps_seed).unwrap();
        assert!(
            ss.as_bytes().iter().any(|&b| b != 0),
            "shared secret should not be all zeros"
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "hybrid (Ed25519+ML-DSA-65) keygen+sign+verify takes many minutes under Miri's interpreter; UB coverage comes from the hybrid KEM tests / native runs"
    )]
    fn hybrid_sig_keygen_sign_verify_roundtrip() {
        let seed = [0x42u8; 64];
        let kp = hybrid_sig_keygen(&seed).expect("keygen should succeed");
        assert_eq!(kp.public_key.len(), HYBRID_SIG_PK_LEN);
        assert_eq!(kp.secret_key.len(), HYBRID_SIG_SK_LEN);

        let message = b"test message for hybrid signature";
        let rng_seed = [0x37u8; 32];
        let sig = hybrid_sign(&kp.secret_key, message, &rng_seed).expect("signing should succeed");
        assert_eq!(sig.len(), HYBRID_SIG_LEN);

        let result = hybrid_verify(&kp.public_key, message, &sig);
        assert!(result.is_ok(), "verification should succeed");
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "hybrid (Ed25519+ML-DSA-65) keygen+sign+verify takes many minutes under Miri's interpreter; UB coverage comes from the hybrid KEM tests / native runs"
    )]
    fn hybrid_sig_wrong_message_fails() {
        let seed = [0x42u8; 64];
        let kp = hybrid_sig_keygen(&seed).unwrap();
        let rng_seed = [0x37u8; 32];
        let sig = hybrid_sign(&kp.secret_key, b"correct", &rng_seed).unwrap();
        let result = hybrid_verify(&kp.public_key, b"wrong", &sig);
        assert_eq!(result, Err(HybridError::VerificationFailed));
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "two hybrid (Ed25519+ML-DSA-65) keygens take ~3.5 min under Miri's interpreter; UB coverage comes from the hybrid KEM tests / native runs"
    )]
    fn hybrid_sig_deterministic_keygen() {
        let seed = [0xAAu8; 64];
        let kp1 = hybrid_sig_keygen(&seed).unwrap();
        let kp2 = hybrid_sig_keygen(&seed).unwrap();
        assert_eq!(kp1.public_key.ed25519_pk(), kp2.public_key.ed25519_pk());
        assert_eq!(kp1.public_key.ml_dsa_pk(), kp2.public_key.ml_dsa_pk());
    }

    #[test]
    fn hybrid_error_display_all() {
        assert_eq!(
            format!("{}", HybridError::InvalidPublicKeyLength),
            "invalid hybrid public key length"
        );
        assert_eq!(
            format!("{}", HybridError::InvalidSecretKeyLength),
            "invalid hybrid secret key length"
        );
        assert_eq!(
            format!("{}", HybridError::InvalidCiphertextLength),
            "invalid hybrid ciphertext length"
        );
        assert_eq!(
            format!("{}", HybridError::InvalidSignatureLength),
            "invalid hybrid signature length"
        );
        assert_eq!(
            format!("{}", HybridError::ClassicalFailure),
            "classical cryptographic component failure"
        );
        assert_eq!(
            format!("{}", HybridError::PostQuantumFailure),
            "post-quantum cryptographic component failure"
        );
        assert_eq!(
            format!("{}", HybridError::VerificationFailed),
            "hybrid signature verification failed"
        );
        assert_eq!(
            format!("{}", HybridError::RngFailure),
            "random number generation failure"
        );
        assert_eq!(
            format!("{}", HybridError::NotImplemented),
            "hybrid scheme not yet implemented"
        );
    }

    #[test]
    fn hybrid_kem_public_key_is_empty() {
        let pk = HybridKemPublicKey::from_components(
            [0; X25519_PK_LEN],
            [0; super::super::ml_kem::ML_KEM_768_PK_LEN],
        );
        assert!(!pk.is_empty());
    }

    #[test]
    fn hybrid_kem_secret_key_len_and_is_empty() {
        let sk = HybridKemSecretKey::from_components(
            [0; X25519_SK_LEN],
            [0; super::super::ml_kem::ML_KEM_768_SK_LEN],
        );
        assert_eq!(sk.len(), HYBRID_KEM_SK_LEN);
        assert!(!sk.is_empty());
    }

    #[test]
    fn hybrid_kem_ciphertext_is_empty() {
        let ct = HybridKemCiphertext::from_components(
            [0; X25519_PK_LEN],
            [0; super::super::ml_kem::ML_KEM_768_CT_LEN],
        );
        assert!(!ct.is_empty());
    }

    #[test]
    fn hybrid_sig_public_key_is_empty() {
        let pk = HybridSigPublicKey::from_components(
            [0; ED25519_PK_LEN],
            [0; super::super::ml_dsa::ML_DSA_65_PK_LEN],
        );
        assert!(!pk.is_empty());
    }

    #[test]
    fn hybrid_sig_secret_key_len_and_is_empty() {
        let sk = HybridSigSecretKey::from_components(
            [0; ED25519_SK_LEN],
            [0; super::super::ml_dsa::ML_DSA_65_SK_LEN],
        );
        assert_eq!(sk.len(), HYBRID_SIG_SK_LEN);
        assert!(!sk.is_empty());
    }

    #[test]
    fn hybrid_signature_is_empty() {
        let sig = HybridSignature::from_components(
            [0; ED25519_SIG_LEN],
            [0; super::super::ml_dsa::ML_DSA_65_SIG_LEN],
        );
        assert!(!sig.is_empty());
    }

    #[test]
    fn hybrid_kem_public_key_debug() {
        let pk = HybridKemPublicKey::from_components(
            [0; X25519_PK_LEN],
            [0; super::super::ml_kem::ML_KEM_768_PK_LEN],
        );
        let debug = format!("{:?}", pk);
        assert!(debug.contains("HybridKemPublicKey"));
    }

    #[test]
    fn hybrid_kem_ciphertext_debug() {
        let ct = HybridKemCiphertext::from_components(
            [0; X25519_PK_LEN],
            [0; super::super::ml_kem::ML_KEM_768_CT_LEN],
        );
        let debug = format!("{:?}", ct);
        assert!(debug.contains("HybridKemCiphertext"));
    }

    #[test]
    fn hybrid_kem_keypair_debug() {
        let seed = [0x42u8; 96];
        let kp = hybrid_kem_keygen(&seed).unwrap();
        let debug = format!("{:?}", kp);
        assert!(debug.contains("HybridKemKeyPair"));
    }

    #[test]
    fn hybrid_sig_public_key_debug() {
        let pk = HybridSigPublicKey::from_components(
            [0; ED25519_PK_LEN],
            [0; super::super::ml_dsa::ML_DSA_65_PK_LEN],
        );
        let debug = format!("{:?}", pk);
        assert!(debug.contains("HybridSigPublicKey"));
    }

    #[test]
    fn hybrid_signature_debug() {
        let sig = HybridSignature::from_components(
            [0; ED25519_SIG_LEN],
            [0; super::super::ml_dsa::ML_DSA_65_SIG_LEN],
        );
        let debug = format!("{:?}", sig);
        assert!(debug.contains("HybridSignature"));
    }

    #[test]
    fn hybrid_sig_keypair_debug() {
        let seed = [0x42u8; 64];
        let kp = hybrid_sig_keygen(&seed).unwrap();
        let debug = format!("{:?}", kp);
        assert!(debug.contains("HybridSigKeyPair"));
    }

    #[test]
    fn hybrid_kem_shared_secret_as_bytes() {
        let ss = HybridKemSharedSecret::from_bytes([0xAB; HYBRID_KEM_SS_LEN]);
        assert_eq!(ss.as_bytes()[0], 0xAB);
        assert_eq!(ss.as_bytes().len(), HYBRID_KEM_SS_LEN);
    }

    #[test]
    fn hybrid_kem_secret_key_components() {
        let x_sk = [0x11u8; X25519_SK_LEN];
        let ml_sk = [0x22u8; super::super::ml_kem::ML_KEM_768_SK_LEN];
        let sk = HybridKemSecretKey::from_components(x_sk, ml_sk);
        assert_eq!(sk.x25519_sk()[0], 0x11);
        assert_eq!(sk.ml_kem_sk()[0], 0x22);
    }

    #[test]
    fn hybrid_sig_secret_key_components() {
        let ed_sk = [0x33u8; ED25519_SK_LEN];
        let ml_sk = [0x44u8; super::super::ml_dsa::ML_DSA_65_SK_LEN];
        let sk = HybridSigSecretKey::from_components(ed_sk, ml_sk);
        assert_eq!(sk.ed25519_sk()[0], 0x33);
        assert_eq!(sk.ml_dsa_sk()[0], 0x44);
    }
}
