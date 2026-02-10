// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ML-DSA-65 digital signature algorithm (FIPS 204).
//!
//! ML-DSA (Module-Lattice-based Digital Signature Algorithm) is a
//! post-quantum digital signature standard, formerly known as CRYSTALS-Dilithium.
//! The 65 parameter set provides NIST Security Level 3 (~AES-192 equivalent).
//!
//! # Parameters (ML-DSA-65)
//!
//! | Parameter        | Size (bytes) |
//! |-----------------|-------------|
//! | Public key      | 1952        |
//! | Secret key      | 4032        |
//! | Signature       | 3309        |
//!
//! # Operations
//!
//! 1. **KeyGen**: Generate (public_key, secret_key) pair
//! 2. **Sign**: Using secret_key + message, produce signature
//! 3. **Verify**: Using public_key + message + signature, verify authenticity
//!
//! # Security
//!
//! ML-DSA-65 provides EUF-CMA (Existential Unforgeability under Chosen Message
//! Attack) security at NIST Level 3. It is used in SmallAIOS for:
//!
//! - ONNX model signature verification
//! - Secure boot chain validation
//! - IPC message authentication
//!
//! # Status
//!
//! This module defines the API surface with correct type sizes.
//! Full lattice arithmetic implementation is pending.

#![allow(unused)]

use core::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

/// ML-DSA-65 public key length in bytes.
pub const ML_DSA_65_PK_LEN: usize = 1952;

/// ML-DSA-65 secret key length in bytes.
pub const ML_DSA_65_SK_LEN: usize = 4032;

/// ML-DSA-65 signature length in bytes.
pub const ML_DSA_65_SIG_LEN: usize = 3309;

/// ML-DSA-65 module dimensions: (k, l) = (6, 5).
pub const ML_DSA_65_K: usize = 6;
pub const ML_DSA_65_L: usize = 5;

/// Polynomial ring dimension (n = 256).
pub const ML_DSA_N: usize = 256;

/// Modulus q = 8380417.
pub const ML_DSA_Q: u32 = 8_380_417;

/// Number of dropped bits from t (d = 13).
pub const ML_DSA_65_D: usize = 13;

/// Challenge weight (tau = 49).
pub const ML_DSA_65_TAU: usize = 49;

/// Bound on secret key coefficients (eta = 4).
pub const ML_DSA_65_ETA: usize = 4;

/// Bound on signature z coefficients (gamma1 = 2^19).
pub const ML_DSA_65_GAMMA1: u32 = 1 << 19;

/// Low-order rounding range (gamma2 = (q-1)/32).
pub const ML_DSA_65_GAMMA2: u32 = (ML_DSA_Q - 1) / 32;

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from ML-DSA-65 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlDsaError {
    /// Invalid public key length.
    InvalidPublicKeyLength,
    /// Invalid secret key length.
    InvalidSecretKeyLength,
    /// Invalid signature length.
    InvalidSignatureLength,
    /// Signature verification failed.
    VerificationFailed,
    /// Signing failed (rejection sampling exceeded limit).
    SigningFailed,
    /// RNG failure during key generation or signing.
    RngFailure,
    /// Message too large.
    MessageTooLarge,
    /// Operation not yet implemented.
    NotImplemented,
}

impl fmt::Display for MlDsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPublicKeyLength => {
                write!(
                    f,
                    "invalid public key length (expected {} bytes)",
                    ML_DSA_65_PK_LEN
                )
            }
            Self::InvalidSecretKeyLength => {
                write!(
                    f,
                    "invalid secret key length (expected {} bytes)",
                    ML_DSA_65_SK_LEN
                )
            }
            Self::InvalidSignatureLength => {
                write!(
                    f,
                    "invalid signature length (expected {} bytes)",
                    ML_DSA_65_SIG_LEN
                )
            }
            Self::VerificationFailed => write!(f, "ML-DSA signature verification failed"),
            Self::SigningFailed => {
                write!(f, "ML-DSA signing failed (rejection sampling exhausted)")
            }
            Self::RngFailure => write!(f, "random number generation failure"),
            Self::MessageTooLarge => write!(f, "message too large for signing"),
            Self::NotImplemented => write!(f, "ML-DSA-65 not yet implemented"),
        }
    }
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// ML-DSA-65 public key (verification key).
#[derive(PartialEq, Eq)]
pub struct MlDsaPublicKey {
    bytes: [u8; ML_DSA_65_PK_LEN],
}

impl MlDsaPublicKey {
    /// Create a public key from a byte array.
    pub fn from_bytes(bytes: [u8; ML_DSA_65_PK_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a public key from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlDsaError> {
        if slice.len() != ML_DSA_65_PK_LEN {
            return Err(MlDsaError::InvalidPublicKeyLength);
        }
        let mut bytes = [0u8; ML_DSA_65_PK_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the public key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the public key length.
    pub fn len(&self) -> usize {
        ML_DSA_65_PK_LEN
    }

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlDsaPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlDsaPublicKey({} bytes)", ML_DSA_65_PK_LEN)
    }
}

/// ML-DSA-65 secret key (signing key).
#[derive(PartialEq, Eq)]
pub struct MlDsaSecretKey {
    bytes: [u8; ML_DSA_65_SK_LEN],
}

impl MlDsaSecretKey {
    /// Create a secret key from a byte array.
    pub fn from_bytes(bytes: [u8; ML_DSA_65_SK_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a secret key from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlDsaError> {
        if slice.len() != ML_DSA_65_SK_LEN {
            return Err(MlDsaError::InvalidSecretKeyLength);
        }
        let mut bytes = [0u8; ML_DSA_65_SK_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the secret key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the secret key length.
    pub fn len(&self) -> usize {
        ML_DSA_65_SK_LEN
    }

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlDsaSecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlDsaSecretKey([REDACTED])")
    }
}

/// ML-DSA-65 signature.
#[derive(PartialEq, Eq)]
pub struct MlDsaSignature {
    bytes: [u8; ML_DSA_65_SIG_LEN],
}

impl MlDsaSignature {
    /// Create a signature from a byte array.
    pub fn from_bytes(bytes: [u8; ML_DSA_65_SIG_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a signature from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlDsaError> {
        if slice.len() != ML_DSA_65_SIG_LEN {
            return Err(MlDsaError::InvalidSignatureLength);
        }
        let mut bytes = [0u8; ML_DSA_65_SIG_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the signature bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the signature length.
    pub fn len(&self) -> usize {
        ML_DSA_65_SIG_LEN
    }

    /// Returns whether the signature is empty (always false for fixed-size types).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlDsaSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlDsaSignature({} bytes)", ML_DSA_65_SIG_LEN)
    }
}

/// ML-DSA-65 key pair (public + secret key).
#[derive(PartialEq, Eq)]
pub struct MlDsaKeyPair {
    pub public_key: MlDsaPublicKey,
    pub secret_key: MlDsaSecretKey,
}

impl fmt::Debug for MlDsaKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlDsaKeyPair")
            .field("public_key", &self.public_key)
            .field("secret_key", &self.secret_key)
            .finish()
    }
}

// ─── Operations ──────────────────────────────────────────────────────────────

/// Generate an ML-DSA-65 key pair.
///
/// Requires a 32-byte random seed from the CSPRNG.
pub fn ml_dsa_65_keygen(seed: &[u8; 32]) -> Result<MlDsaKeyPair, MlDsaError> {
    // Stub: will implement FIPS 204 Algorithm 1 (ML-DSA.KeyGen)
    Err(MlDsaError::NotImplemented)
}

/// Sign a message using ML-DSA-65.
///
/// # Arguments
///
/// * `sk` — The signing (secret) key
/// * `message` — The message to sign
/// * `random_seed` — 32 bytes of randomness for hedged signing
///
/// # Returns
///
/// The signature on success. Signing uses rejection sampling and may
/// theoretically fail if the sampling limit is exceeded (astronomically unlikely).
pub fn ml_dsa_65_sign(
    sk: &MlDsaSecretKey,
    message: &[u8],
    random_seed: &[u8; 32],
) -> Result<MlDsaSignature, MlDsaError> {
    // Stub: will implement FIPS 204 Algorithm 2 (ML-DSA.Sign)
    Err(MlDsaError::NotImplemented)
}

/// Verify an ML-DSA-65 signature.
///
/// # Arguments
///
/// * `pk` — The verification (public) key
/// * `message` — The message that was signed
/// * `signature` — The signature to verify
///
/// # Returns
///
/// `Ok(())` if the signature is valid, `Err(VerificationFailed)` otherwise.
pub fn ml_dsa_65_verify(
    pk: &MlDsaPublicKey,
    message: &[u8],
    signature: &MlDsaSignature,
) -> Result<(), MlDsaError> {
    // Stub: will implement FIPS 204 Algorithm 3 (ML-DSA.Verify)
    Err(MlDsaError::NotImplemented)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn ml_dsa_constant_sizes() {
        assert_eq!(ML_DSA_65_PK_LEN, 1952);
        assert_eq!(ML_DSA_65_SK_LEN, 4032);
        assert_eq!(ML_DSA_65_SIG_LEN, 3309);
        assert_eq!(ML_DSA_65_K, 6);
        assert_eq!(ML_DSA_65_L, 5);
        assert_eq!(ML_DSA_N, 256);
        assert_eq!(ML_DSA_Q, 8_380_417);
    }

    #[test]
    fn ml_dsa_derived_constants() {
        assert_eq!(ML_DSA_65_D, 13);
        assert_eq!(ML_DSA_65_TAU, 49);
        assert_eq!(ML_DSA_65_ETA, 4);
        assert_eq!(ML_DSA_65_GAMMA1, 524_288); // 2^19
        assert_eq!(ML_DSA_65_GAMMA2, 261_888); // (q-1)/32
    }

    #[test]
    fn ml_dsa_public_key_from_slice_valid() {
        let bytes = [0x42u8; ML_DSA_65_PK_LEN];
        let pk = MlDsaPublicKey::from_slice(&bytes).unwrap();
        assert_eq!(pk.len(), ML_DSA_65_PK_LEN);
        assert_eq!(pk.as_bytes()[0], 0x42);
    }

    #[test]
    fn ml_dsa_public_key_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlDsaPublicKey::from_slice(&bytes),
            Err(MlDsaError::InvalidPublicKeyLength)
        );
    }

    #[test]
    fn ml_dsa_secret_key_from_slice_valid() {
        let bytes = [0xAB; ML_DSA_65_SK_LEN];
        let sk = MlDsaSecretKey::from_slice(&bytes).unwrap();
        assert_eq!(sk.len(), ML_DSA_65_SK_LEN);
    }

    #[test]
    fn ml_dsa_secret_key_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlDsaSecretKey::from_slice(&bytes),
            Err(MlDsaError::InvalidSecretKeyLength)
        );
    }

    #[test]
    fn ml_dsa_signature_from_slice_valid() {
        let bytes = [0xCD; ML_DSA_65_SIG_LEN];
        let sig = MlDsaSignature::from_slice(&bytes).unwrap();
        assert_eq!(sig.len(), ML_DSA_65_SIG_LEN);
    }

    #[test]
    fn ml_dsa_signature_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlDsaSignature::from_slice(&bytes),
            Err(MlDsaError::InvalidSignatureLength)
        );
    }

    #[test]
    fn ml_dsa_keygen_stub() {
        let seed = [0u8; 32];
        assert_eq!(ml_dsa_65_keygen(&seed), Err(MlDsaError::NotImplemented));
    }

    #[test]
    fn ml_dsa_sign_stub() {
        let sk = MlDsaSecretKey::from_bytes([0u8; ML_DSA_65_SK_LEN]);
        let seed = [0u8; 32];
        assert_eq!(
            ml_dsa_65_sign(&sk, b"message", &seed),
            Err(MlDsaError::NotImplemented)
        );
    }

    #[test]
    fn ml_dsa_verify_stub() {
        let pk = MlDsaPublicKey::from_bytes([0u8; ML_DSA_65_PK_LEN]);
        let sig = MlDsaSignature::from_bytes([0u8; ML_DSA_65_SIG_LEN]);
        assert_eq!(
            ml_dsa_65_verify(&pk, b"message", &sig),
            Err(MlDsaError::NotImplemented)
        );
    }

    #[test]
    fn ml_dsa_secret_key_debug_redacted() {
        let sk = MlDsaSecretKey::from_bytes([0xFF; ML_DSA_65_SK_LEN]);
        let debug = format!("{:?}", sk);
        assert_eq!(debug, "MlDsaSecretKey([REDACTED])");
    }

    #[test]
    fn ml_dsa_public_key_debug() {
        let pk = MlDsaPublicKey::from_bytes([0; ML_DSA_65_PK_LEN]);
        let debug = format!("{:?}", pk);
        assert!(debug.contains("1952"));
    }

    #[test]
    fn ml_dsa_signature_debug() {
        let sig = MlDsaSignature::from_bytes([0; ML_DSA_65_SIG_LEN]);
        let debug = format!("{:?}", sig);
        assert!(debug.contains("3309"));
    }
}
