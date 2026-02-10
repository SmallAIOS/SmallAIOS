// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ML-KEM-768 key encapsulation mechanism (FIPS 203).
//!
//! ML-KEM (Module-Lattice-based Key Encapsulation Mechanism) is a
//! post-quantum key encapsulation standard, formerly known as CRYSTALS-Kyber.
//! The 768 parameter set provides NIST Security Level 3 (~AES-192 equivalent).
//!
//! # Parameters (ML-KEM-768)
//!
//! | Parameter        | Size (bytes) |
//! |-----------------|-------------|
//! | Public key      | 1184        |
//! | Secret key      | 2400        |
//! | Ciphertext      | 1088        |
//! | Shared secret   | 32          |
//!
//! # Operations
//!
//! 1. **KeyGen**: Generate (public_key, secret_key) pair
//! 2. **Encaps**: Using public_key, produce (ciphertext, shared_secret)
//! 3. **Decaps**: Using secret_key + ciphertext, recover shared_secret
//!
//! # Status
//!
//! This module defines the API surface with correct type sizes.
//! Full lattice arithmetic implementation is pending.

#![allow(unused)]

use core::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

/// ML-KEM-768 public key length in bytes.
pub const ML_KEM_768_PK_LEN: usize = 1184;

/// ML-KEM-768 secret key length in bytes.
pub const ML_KEM_768_SK_LEN: usize = 2400;

/// ML-KEM-768 ciphertext length in bytes.
pub const ML_KEM_768_CT_LEN: usize = 1088;

/// ML-KEM-768 shared secret length in bytes.
pub const ML_KEM_768_SS_LEN: usize = 32;

/// ML-KEM-768 module rank (k = 3).
pub const ML_KEM_768_K: usize = 3;

/// Polynomial ring dimension (n = 256).
pub const ML_KEM_N: usize = 256;

/// Modulus q = 3329.
pub const ML_KEM_Q: u16 = 3329;

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from ML-KEM-768 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlKemError {
    /// Invalid public key length.
    InvalidPublicKeyLength,
    /// Invalid secret key length.
    InvalidSecretKeyLength,
    /// Invalid ciphertext length.
    InvalidCiphertextLength,
    /// Decapsulation failure (implicit rejection).
    DecapsulationFailure,
    /// RNG failure during key generation.
    RngFailure,
    /// Operation not yet implemented.
    NotImplemented,
}

impl fmt::Display for MlKemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPublicKeyLength => {
                write!(
                    f,
                    "invalid public key length (expected {} bytes)",
                    ML_KEM_768_PK_LEN
                )
            }
            Self::InvalidSecretKeyLength => {
                write!(
                    f,
                    "invalid secret key length (expected {} bytes)",
                    ML_KEM_768_SK_LEN
                )
            }
            Self::InvalidCiphertextLength => {
                write!(
                    f,
                    "invalid ciphertext length (expected {} bytes)",
                    ML_KEM_768_CT_LEN
                )
            }
            Self::DecapsulationFailure => write!(f, "ML-KEM decapsulation failure"),
            Self::RngFailure => write!(f, "random number generation failure"),
            Self::NotImplemented => write!(f, "ML-KEM-768 not yet implemented"),
        }
    }
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// ML-KEM-768 public key (encapsulation key).
#[derive(PartialEq, Eq)]
pub struct MlKemPublicKey {
    bytes: [u8; ML_KEM_768_PK_LEN],
}

impl MlKemPublicKey {
    /// Create a public key from a byte array.
    pub fn from_bytes(bytes: [u8; ML_KEM_768_PK_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a public key from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlKemError> {
        if slice.len() != ML_KEM_768_PK_LEN {
            return Err(MlKemError::InvalidPublicKeyLength);
        }
        let mut bytes = [0u8; ML_KEM_768_PK_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the public key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the public key length.
    pub fn len(&self) -> usize {
        ML_KEM_768_PK_LEN
    }

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlKemPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKemPublicKey({} bytes)", ML_KEM_768_PK_LEN)
    }
}

/// ML-KEM-768 secret key (decapsulation key).
#[derive(PartialEq, Eq)]
pub struct MlKemSecretKey {
    bytes: [u8; ML_KEM_768_SK_LEN],
}

impl MlKemSecretKey {
    /// Create a secret key from a byte array.
    pub fn from_bytes(bytes: [u8; ML_KEM_768_SK_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a secret key from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlKemError> {
        if slice.len() != ML_KEM_768_SK_LEN {
            return Err(MlKemError::InvalidSecretKeyLength);
        }
        let mut bytes = [0u8; ML_KEM_768_SK_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the secret key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the secret key length.
    pub fn len(&self) -> usize {
        ML_KEM_768_SK_LEN
    }

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlKemSecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKemSecretKey([REDACTED])")
    }
}

/// ML-KEM-768 ciphertext.
#[derive(PartialEq, Eq)]
pub struct MlKemCiphertext {
    bytes: [u8; ML_KEM_768_CT_LEN],
}

impl MlKemCiphertext {
    /// Create a ciphertext from a byte array.
    pub fn from_bytes(bytes: [u8; ML_KEM_768_CT_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a ciphertext from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlKemError> {
        if slice.len() != ML_KEM_768_CT_LEN {
            return Err(MlKemError::InvalidCiphertextLength);
        }
        let mut bytes = [0u8; ML_KEM_768_CT_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the ciphertext bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the ciphertext length.
    pub fn len(&self) -> usize {
        ML_KEM_768_CT_LEN
    }

    /// Returns whether the ciphertext is empty (always false for fixed-size types).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlKemCiphertext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKemCiphertext({} bytes)", ML_KEM_768_CT_LEN)
    }
}

/// ML-KEM-768 shared secret (32 bytes).
#[derive(Clone, PartialEq, Eq)]
pub struct MlKemSharedSecret {
    bytes: [u8; ML_KEM_768_SS_LEN],
}

impl MlKemSharedSecret {
    /// Create a shared secret from a byte array.
    pub fn from_bytes(bytes: [u8; ML_KEM_768_SS_LEN]) -> Self {
        Self { bytes }
    }

    /// Return the shared secret bytes.
    pub fn as_bytes(&self) -> &[u8; ML_KEM_768_SS_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for MlKemSharedSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKemSharedSecret([REDACTED])")
    }
}

/// ML-KEM-768 key pair (public + secret key).
#[derive(PartialEq, Eq)]
pub struct MlKemKeyPair {
    pub public_key: MlKemPublicKey,
    pub secret_key: MlKemSecretKey,
}

impl fmt::Debug for MlKemKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlKemKeyPair")
            .field("public_key", &self.public_key)
            .field("secret_key", &self.secret_key)
            .finish()
    }
}

// ─── Operations ──────────────────────────────────────────────────────────────

/// Generate an ML-KEM-768 key pair.
///
/// Requires a 64-byte random seed (d || z) from the CSPRNG.
/// - d (32 bytes): seed for matrix/vector generation
/// - z (32 bytes): implicit rejection seed
pub fn ml_kem_768_keygen(seed: &[u8; 64]) -> Result<MlKemKeyPair, MlKemError> {
    // Stub: will implement FIPS 203 Algorithm 15 (ML-KEM.KeyGen)
    Err(MlKemError::NotImplemented)
}

/// Encapsulate: generate a shared secret and ciphertext using a public key.
///
/// Requires a 32-byte random seed from the CSPRNG.
pub fn ml_kem_768_encaps(
    pk: &MlKemPublicKey,
    random_seed: &[u8; 32],
) -> Result<(MlKemCiphertext, MlKemSharedSecret), MlKemError> {
    // Stub: will implement FIPS 203 Algorithm 16 (ML-KEM.Encaps)
    Err(MlKemError::NotImplemented)
}

/// Decapsulate: recover the shared secret from ciphertext using the secret key.
///
/// Uses implicit rejection: if decapsulation fails, a pseudorandom
/// shared secret is returned (derived from sk and ct) to prevent
/// chosen-ciphertext attacks.
pub fn ml_kem_768_decaps(
    sk: &MlKemSecretKey,
    ct: &MlKemCiphertext,
) -> Result<MlKemSharedSecret, MlKemError> {
    // Stub: will implement FIPS 203 Algorithm 17 (ML-KEM.Decaps)
    Err(MlKemError::NotImplemented)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn ml_kem_constant_sizes() {
        assert_eq!(ML_KEM_768_PK_LEN, 1184);
        assert_eq!(ML_KEM_768_SK_LEN, 2400);
        assert_eq!(ML_KEM_768_CT_LEN, 1088);
        assert_eq!(ML_KEM_768_SS_LEN, 32);
        assert_eq!(ML_KEM_768_K, 3);
        assert_eq!(ML_KEM_N, 256);
        assert_eq!(ML_KEM_Q, 3329);
    }

    #[test]
    fn ml_kem_public_key_from_slice_valid() {
        let bytes = [0x42u8; ML_KEM_768_PK_LEN];
        let pk = MlKemPublicKey::from_slice(&bytes).unwrap();
        assert_eq!(pk.len(), ML_KEM_768_PK_LEN);
        assert_eq!(pk.as_bytes()[0], 0x42);
    }

    #[test]
    fn ml_kem_public_key_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlKemPublicKey::from_slice(&bytes),
            Err(MlKemError::InvalidPublicKeyLength)
        );
    }

    #[test]
    fn ml_kem_secret_key_from_slice_valid() {
        let bytes = [0xAB; ML_KEM_768_SK_LEN];
        let sk = MlKemSecretKey::from_slice(&bytes).unwrap();
        assert_eq!(sk.len(), ML_KEM_768_SK_LEN);
    }

    #[test]
    fn ml_kem_secret_key_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlKemSecretKey::from_slice(&bytes),
            Err(MlKemError::InvalidSecretKeyLength)
        );
    }

    #[test]
    fn ml_kem_ciphertext_from_slice_valid() {
        let bytes = [0xCD; ML_KEM_768_CT_LEN];
        let ct = MlKemCiphertext::from_slice(&bytes).unwrap();
        assert_eq!(ct.len(), ML_KEM_768_CT_LEN);
    }

    #[test]
    fn ml_kem_ciphertext_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlKemCiphertext::from_slice(&bytes),
            Err(MlKemError::InvalidCiphertextLength)
        );
    }

    #[test]
    fn ml_kem_shared_secret_construction() {
        let ss = MlKemSharedSecret::from_bytes([0xEF; ML_KEM_768_SS_LEN]);
        assert_eq!(ss.as_bytes().len(), ML_KEM_768_SS_LEN);
        assert_eq!(ss.as_bytes()[0], 0xEF);
    }

    #[test]
    fn ml_kem_keygen_stub() {
        let seed = [0u8; 64];
        assert_eq!(ml_kem_768_keygen(&seed), Err(MlKemError::NotImplemented));
    }

    #[test]
    fn ml_kem_encaps_stub() {
        let pk = MlKemPublicKey::from_bytes([0u8; ML_KEM_768_PK_LEN]);
        let seed = [0u8; 32];
        assert_eq!(
            ml_kem_768_encaps(&pk, &seed),
            Err(MlKemError::NotImplemented)
        );
    }

    #[test]
    fn ml_kem_decaps_stub() {
        let sk = MlKemSecretKey::from_bytes([0u8; ML_KEM_768_SK_LEN]);
        let ct = MlKemCiphertext::from_bytes([0u8; ML_KEM_768_CT_LEN]);
        assert_eq!(ml_kem_768_decaps(&sk, &ct), Err(MlKemError::NotImplemented));
    }

    #[test]
    fn ml_kem_secret_key_debug_redacted() {
        let sk = MlKemSecretKey::from_bytes([0xFF; ML_KEM_768_SK_LEN]);
        let debug = format!("{:?}", sk);
        assert_eq!(debug, "MlKemSecretKey([REDACTED])");
    }

    #[test]
    fn ml_kem_shared_secret_debug_redacted() {
        let ss = MlKemSharedSecret::from_bytes([0xFF; ML_KEM_768_SS_LEN]);
        let debug = format!("{:?}", ss);
        assert_eq!(debug, "MlKemSharedSecret([REDACTED])");
    }

    #[test]
    fn ml_kem_public_key_debug() {
        let pk = MlKemPublicKey::from_bytes([0; ML_KEM_768_PK_LEN]);
        let debug = format!("{:?}", pk);
        assert!(debug.contains("1184"));
    }
}
