// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AES-256-GCM authenticated encryption (FIPS 197 + SP 800-38D).
//!
//! Provides authenticated encryption with associated data (AEAD) using
//! AES-256 in Galois/Counter Mode. This is the symmetric cipher used for:
//!
//! - Encrypting IPC messages between tasks
//! - Protecting data at rest (tensor storage)
//! - TLS 1.3 record-layer encryption
//!
//! # Parameters
//!
//! - **Key**: 256 bits (32 bytes)
//! - **Nonce**: 96 bits (12 bytes) — MUST be unique per key
//! - **Tag**: 128 bits (16 bytes) — authentication tag
//!
//! # Status
//!
//! This module defines the API surface. The actual AES-256-GCM implementation
//! will use hardware acceleration (AES-NI on x86-64, ARMv8-CE on AArch64)
//! when available, with a constant-time software fallback.

#![allow(unused)]

use core::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

/// AES-256 key length in bytes.
pub const AES256_KEY_LEN: usize = 32;

/// GCM nonce (IV) length in bytes.
pub const GCM_NONCE_LEN: usize = 12;

/// GCM authentication tag length in bytes.
pub const GCM_TAG_LEN: usize = 16;

/// AES block size in bytes.
pub const AES_BLOCK_SIZE: usize = 16;

/// Maximum plaintext length per encryption operation (2^36 - 32 bytes per SP 800-38D).
pub const GCM_MAX_PLAINTEXT_LEN: usize = (1 << 36) - 32;

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from AES-256-GCM operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AesGcmError {
    /// Invalid key length (must be 32 bytes).
    InvalidKeyLength,
    /// Invalid nonce length (must be 12 bytes).
    InvalidNonceLength,
    /// Plaintext exceeds maximum allowed length.
    PlaintextTooLong,
    /// Output buffer too small.
    BufferTooSmall,
    /// Authentication tag verification failed (decryption).
    AuthenticationFailed,
    /// Operation not yet implemented.
    NotImplemented,
}

impl fmt::Display for AesGcmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKeyLength => {
                write!(f, "invalid key length (expected {} bytes)", AES256_KEY_LEN)
            }
            Self::InvalidNonceLength => {
                write!(f, "invalid nonce length (expected {} bytes)", GCM_NONCE_LEN)
            }
            Self::PlaintextTooLong => write!(f, "plaintext exceeds maximum length"),
            Self::BufferTooSmall => write!(f, "output buffer too small"),
            Self::AuthenticationFailed => write!(f, "GCM authentication tag mismatch"),
            Self::NotImplemented => write!(f, "AES-256-GCM not yet implemented"),
        }
    }
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// AES-256 encryption key (zeroized on drop in production).
#[derive(Clone, PartialEq, Eq)]
pub struct AesKey {
    bytes: [u8; AES256_KEY_LEN],
}

impl AesKey {
    /// Create a key from a 32-byte array.
    pub fn from_bytes(bytes: [u8; AES256_KEY_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a key from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, AesGcmError> {
        if slice.len() != AES256_KEY_LEN {
            return Err(AesGcmError::InvalidKeyLength);
        }
        let mut bytes = [0u8; AES256_KEY_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the key material as a byte slice.
    pub fn as_bytes(&self) -> &[u8; AES256_KEY_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for AesKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AesKey([REDACTED])")
    }
}

/// GCM nonce (96-bit initialization vector).
///
/// CRITICAL: Each nonce MUST be unique for a given key. Nonce reuse
/// completely breaks GCM's security guarantees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GcmNonce {
    bytes: [u8; GCM_NONCE_LEN],
}

impl GcmNonce {
    /// Create a nonce from a 12-byte array.
    pub fn from_bytes(bytes: [u8; GCM_NONCE_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a nonce from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, AesGcmError> {
        if slice.len() != GCM_NONCE_LEN {
            return Err(AesGcmError::InvalidNonceLength);
        }
        let mut bytes = [0u8; GCM_NONCE_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the nonce bytes.
    pub fn as_bytes(&self) -> &[u8; GCM_NONCE_LEN] {
        &self.bytes
    }
}

/// GCM authentication tag (128 bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GcmTag {
    bytes: [u8; GCM_TAG_LEN],
}

impl GcmTag {
    /// Create a tag from a 16-byte array.
    pub fn from_bytes(bytes: [u8; GCM_TAG_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a tag from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, AesGcmError> {
        if slice.len() != GCM_TAG_LEN {
            return Err(AesGcmError::BufferTooSmall);
        }
        let mut bytes = [0u8; GCM_TAG_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the tag bytes.
    pub fn as_bytes(&self) -> &[u8; GCM_TAG_LEN] {
        &self.bytes
    }
}

// ─── AES-256-GCM Cipher ─────────────────────────────────────────────────────

/// AES-256-GCM cipher instance.
///
/// Holds the expanded key schedule for AES-256 and provides
/// authenticated encryption and decryption operations.
pub struct Aes256Gcm {
    /// The raw key (key schedule expansion is part of the implementation).
    key: AesKey,
}

impl Aes256Gcm {
    /// Create a new AES-256-GCM cipher with the given key.
    pub fn new(key: AesKey) -> Self {
        Self { key }
    }

    /// Encrypt plaintext with associated data.
    ///
    /// # Arguments
    ///
    /// * `nonce` — 12-byte unique nonce
    /// * `aad` — Associated data (authenticated but not encrypted)
    /// * `plaintext` — Data to encrypt
    /// * `ciphertext` — Output buffer (must be >= plaintext.len())
    ///
    /// # Returns
    ///
    /// The authentication tag on success.
    pub fn encrypt(
        &self,
        nonce: &GcmNonce,
        aad: &[u8],
        plaintext: &[u8],
        ciphertext: &mut [u8],
    ) -> Result<GcmTag, AesGcmError> {
        if plaintext.len() > GCM_MAX_PLAINTEXT_LEN {
            return Err(AesGcmError::PlaintextTooLong);
        }
        if ciphertext.len() < plaintext.len() {
            return Err(AesGcmError::BufferTooSmall);
        }
        // Stub: actual AES-256-GCM implementation pending
        Err(AesGcmError::NotImplemented)
    }

    /// Decrypt ciphertext with associated data, verifying the authentication tag.
    ///
    /// # Arguments
    ///
    /// * `nonce` — 12-byte nonce used during encryption
    /// * `aad` — Associated data used during encryption
    /// * `ciphertext` — Encrypted data
    /// * `tag` — Authentication tag from encryption
    /// * `plaintext` — Output buffer (must be >= ciphertext.len())
    ///
    /// # Returns
    ///
    /// `Ok(())` if authentication succeeds and plaintext is written.
    pub fn decrypt(
        &self,
        nonce: &GcmNonce,
        aad: &[u8],
        ciphertext: &[u8],
        tag: &GcmTag,
        plaintext: &mut [u8],
    ) -> Result<(), AesGcmError> {
        if plaintext.len() < ciphertext.len() {
            return Err(AesGcmError::BufferTooSmall);
        }
        // Stub: actual AES-256-GCM implementation pending
        Err(AesGcmError::NotImplemented)
    }
}

impl fmt::Debug for Aes256Gcm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Aes256Gcm {{ key: [REDACTED] }}")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn aes_key_from_bytes() {
        let key = AesKey::from_bytes([0x42; AES256_KEY_LEN]);
        assert_eq!(key.as_bytes()[0], 0x42);
        assert_eq!(key.as_bytes().len(), AES256_KEY_LEN);
    }

    #[test]
    fn aes_key_from_slice_valid() {
        let bytes = [0xAB; AES256_KEY_LEN];
        let key = AesKey::from_slice(&bytes).unwrap();
        assert_eq!(key.as_bytes(), &bytes);
    }

    #[test]
    fn aes_key_from_slice_invalid_length() {
        let bytes = [0u8; 16]; // Too short
        assert_eq!(
            AesKey::from_slice(&bytes),
            Err(AesGcmError::InvalidKeyLength)
        );
    }

    #[test]
    fn aes_key_debug_redacted() {
        let key = AesKey::from_bytes([0xFF; AES256_KEY_LEN]);
        let debug = format!("{:?}", key);
        assert_eq!(debug, "AesKey([REDACTED])");
        assert!(
            !debug.contains("ff"),
            "key material must not appear in debug output"
        );
    }

    #[test]
    fn gcm_nonce_from_bytes() {
        let nonce = GcmNonce::from_bytes([0x01; GCM_NONCE_LEN]);
        assert_eq!(nonce.as_bytes().len(), GCM_NONCE_LEN);
    }

    #[test]
    fn gcm_nonce_from_slice_invalid() {
        assert_eq!(
            GcmNonce::from_slice(&[0u8; 8]),
            Err(AesGcmError::InvalidNonceLength)
        );
    }

    #[test]
    fn gcm_tag_from_bytes() {
        let tag = GcmTag::from_bytes([0xDE; GCM_TAG_LEN]);
        assert_eq!(tag.as_bytes()[0], 0xDE);
        assert_eq!(tag.as_bytes().len(), GCM_TAG_LEN);
    }

    #[test]
    fn aes256_gcm_encrypt_stub_returns_not_implemented() {
        let cipher = Aes256Gcm::new(AesKey::from_bytes([0; AES256_KEY_LEN]));
        let nonce = GcmNonce::from_bytes([0; GCM_NONCE_LEN]);
        let mut ct = [0u8; 16];
        let result = cipher.encrypt(&nonce, b"", &[0u8; 16], &mut ct);
        assert_eq!(result, Err(AesGcmError::NotImplemented));
    }

    #[test]
    fn aes256_gcm_decrypt_stub_returns_not_implemented() {
        let cipher = Aes256Gcm::new(AesKey::from_bytes([0; AES256_KEY_LEN]));
        let nonce = GcmNonce::from_bytes([0; GCM_NONCE_LEN]);
        let tag = GcmTag::from_bytes([0; GCM_TAG_LEN]);
        let mut pt = [0u8; 16];
        let result = cipher.decrypt(&nonce, b"", &[0u8; 16], &tag, &mut pt);
        assert_eq!(result, Err(AesGcmError::NotImplemented));
    }

    #[test]
    fn aes256_gcm_encrypt_buffer_too_small() {
        let cipher = Aes256Gcm::new(AesKey::from_bytes([0; AES256_KEY_LEN]));
        let nonce = GcmNonce::from_bytes([0; GCM_NONCE_LEN]);
        let mut ct = [0u8; 4]; // Too small for 16-byte plaintext
        let result = cipher.encrypt(&nonce, b"", &[0u8; 16], &mut ct);
        assert_eq!(result, Err(AesGcmError::BufferTooSmall));
    }

    #[test]
    fn aes256_gcm_decrypt_buffer_too_small() {
        let cipher = Aes256Gcm::new(AesKey::from_bytes([0; AES256_KEY_LEN]));
        let nonce = GcmNonce::from_bytes([0; GCM_NONCE_LEN]);
        let tag = GcmTag::from_bytes([0; GCM_TAG_LEN]);
        let mut pt = [0u8; 4]; // Too small
        let result = cipher.decrypt(&nonce, b"", &[0u8; 16], &tag, &mut pt);
        assert_eq!(result, Err(AesGcmError::BufferTooSmall));
    }

    #[test]
    fn constant_sizes() {
        assert_eq!(AES256_KEY_LEN, 32);
        assert_eq!(GCM_NONCE_LEN, 12);
        assert_eq!(GCM_TAG_LEN, 16);
        assert_eq!(AES_BLOCK_SIZE, 16);
    }
}
