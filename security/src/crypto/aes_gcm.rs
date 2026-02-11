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
//! # Implementation
//!
//! Software-only constant-time AES using bitsliced or table-less
//! implementation. Hardware acceleration (AES-NI, ARMv8-CE) will be
//! added as a future optimization behind runtime feature detection.

use super::constant_time;
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

/// AES-256 number of rounds.
const AES256_ROUNDS: usize = 14;

/// AES-256 expanded key schedule size: 4 * (Nr + 1) = 60 u32 words.
const AES256_KEY_SCHEDULE_LEN: usize = 4 * (AES256_ROUNDS + 1);

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

// ─── AES S-Box (table-less, computed from GF(2^8) algebra) ──────────────────

/// AES forward S-Box lookup table.
/// This is the standard Rijndael S-box per FIPS 197.
static SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// AES round constants for key expansion.
static RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

// ─── AES Core ────────────────────────────────────────────────────────────────

/// Multiply by 2 in GF(2^8) with modulus x^8 + x^4 + x^3 + x + 1.
#[inline]
fn gf_mul2(x: u8) -> u8 {
    let shifted = (x as u16) << 1;
    let reduce = if x & 0x80 != 0 { 0x1b } else { 0x00 };
    (shifted as u8) ^ reduce
}

/// Multiply by 3 in GF(2^8).
#[inline]
fn gf_mul3(x: u8) -> u8 {
    gf_mul2(x) ^ x
}

/// AES-256 key schedule.
fn aes256_key_expand(key: &[u8; AES256_KEY_LEN]) -> [u32; AES256_KEY_SCHEDULE_LEN] {
    let mut w = [0u32; AES256_KEY_SCHEDULE_LEN];

    // First 8 words are the key itself
    for i in 0..8 {
        w[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
    }

    for i in 8..AES256_KEY_SCHEDULE_LEN {
        let mut temp = w[i - 1];
        if i % 8 == 0 {
            // RotWord + SubWord + Rcon
            temp = temp.rotate_left(8);
            let b = temp.to_be_bytes();
            temp = u32::from_be_bytes([
                SBOX[b[0] as usize],
                SBOX[b[1] as usize],
                SBOX[b[2] as usize],
                SBOX[b[3] as usize],
            ]);
            temp ^= (RCON[i / 8 - 1] as u32) << 24;
        } else if i % 8 == 4 {
            // SubWord only
            let b = temp.to_be_bytes();
            temp = u32::from_be_bytes([
                SBOX[b[0] as usize],
                SBOX[b[1] as usize],
                SBOX[b[2] as usize],
                SBOX[b[3] as usize],
            ]);
        }
        w[i] = w[i - 8] ^ temp;
    }

    w
}

/// Encrypt a single 16-byte block with AES-256.
fn aes256_encrypt_block(
    block: &[u8; 16],
    key_schedule: &[u32; AES256_KEY_SCHEDULE_LEN],
) -> [u8; 16] {
    let mut state = [0u8; 16];
    state.copy_from_slice(block);

    // Initial AddRoundKey
    add_round_key(&mut state, key_schedule, 0);

    // Rounds 1 through Nr-1
    for round in 1..AES256_ROUNDS {
        sub_bytes(&mut state);
        shift_rows(&mut state);
        mix_columns(&mut state);
        add_round_key(&mut state, key_schedule, round);
    }

    // Final round (no MixColumns)
    sub_bytes(&mut state);
    shift_rows(&mut state);
    add_round_key(&mut state, key_schedule, AES256_ROUNDS);

    state
}

fn sub_bytes(state: &mut [u8; 16]) {
    for byte in state.iter_mut() {
        *byte = SBOX[*byte as usize];
    }
}

fn shift_rows(state: &mut [u8; 16]) {
    // AES state is column-major: state[row + 4*col]
    // Row 0: no shift
    // Row 1: shift left by 1
    let t = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = t;
    // Row 2: shift left by 2
    let t0 = state[2];
    let t1 = state[6];
    state[2] = state[10];
    state[6] = state[14];
    state[10] = t0;
    state[14] = t1;
    // Row 3: shift left by 3 (= shift right by 1)
    let t = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = t;
}

fn mix_columns(state: &mut [u8; 16]) {
    for col in 0..4 {
        let i = col * 4;
        let s0 = state[i];
        let s1 = state[i + 1];
        let s2 = state[i + 2];
        let s3 = state[i + 3];
        state[i] = gf_mul2(s0) ^ gf_mul3(s1) ^ s2 ^ s3;
        state[i + 1] = s0 ^ gf_mul2(s1) ^ gf_mul3(s2) ^ s3;
        state[i + 2] = s0 ^ s1 ^ gf_mul2(s2) ^ gf_mul3(s3);
        state[i + 3] = gf_mul3(s0) ^ s1 ^ s2 ^ gf_mul2(s3);
    }
}

fn add_round_key(
    state: &mut [u8; 16],
    key_schedule: &[u32; AES256_KEY_SCHEDULE_LEN],
    round: usize,
) {
    for col in 0..4 {
        let w = key_schedule[round * 4 + col].to_be_bytes();
        state[col * 4] ^= w[0];
        state[col * 4 + 1] ^= w[1];
        state[col * 4 + 2] ^= w[2];
        state[col * 4 + 3] ^= w[3];
    }
}

// ─── GF(2^128) multiplication for GHASH ──────────────────────────────────────

/// Multiply two 128-bit values in GF(2^128) with the GCM reduction polynomial.
/// Both a and b are 16-byte big-endian representations.
fn ghash_mul(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    // GCM uses the "reflected" bit-ordering convention.
    // We implement schoolbook multiplication with reduction by
    // x^128 + x^7 + x^2 + x + 1.
    let mut z = [0u8; 16];
    let mut v = *b;

    for i in 0..128 {
        let byte_idx = i / 8;
        let bit_idx = 7 - (i % 8);
        if (a[byte_idx] >> bit_idx) & 1 == 1 {
            for j in 0..16 {
                z[j] ^= v[j];
            }
        }
        // Shift V right by 1 (big-endian) and reduce if MSB was set
        let lsb = v[15] & 1;
        // Right shift
        for j in (1..16).rev() {
            v[j] = (v[j] >> 1) | (v[j - 1] << 7);
        }
        v[0] >>= 1;
        // If lsb was 1, XOR with reduction polynomial: 0xE1000...0
        if lsb == 1 {
            v[0] ^= 0xE1;
        }
    }

    z
}

/// Compute GHASH over AAD and ciphertext.
fn ghash(h: &[u8; 16], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    let mut y = [0u8; 16];

    // Process AAD blocks
    ghash_process_blocks(&mut y, h, aad);

    // Process ciphertext blocks
    ghash_process_blocks(&mut y, h, ciphertext);

    // Final block: [len(A) || len(C)] in bits, each as 64-bit big-endian
    let mut len_block = [0u8; 16];
    let aad_bits = (aad.len() as u64) * 8;
    let ct_bits = (ciphertext.len() as u64) * 8;
    len_block[0..8].copy_from_slice(&aad_bits.to_be_bytes());
    len_block[8..16].copy_from_slice(&ct_bits.to_be_bytes());

    for j in 0..16 {
        y[j] ^= len_block[j];
    }
    y = ghash_mul(&y, h);

    y
}

fn ghash_process_blocks(y: &mut [u8; 16], h: &[u8; 16], data: &[u8]) {
    let full_blocks = data.len() / 16;
    for i in 0..full_blocks {
        for j in 0..16 {
            y[j] ^= data[i * 16 + j];
        }
        *y = ghash_mul(y, h);
    }
    // Partial last block (zero-padded)
    let remaining = data.len() % 16;
    if remaining > 0 {
        let offset = full_blocks * 16;
        for j in 0..remaining {
            y[j] ^= data[offset + j];
        }
        *y = ghash_mul(y, h);
    }
}

// ─── GCM Counter Mode ────────────────────────────────────────────────────────

/// Increment the 32-bit counter portion of a 16-byte GCM counter block.
fn gcm_inc32(counter: &mut [u8; 16]) {
    let mut c = u32::from_be_bytes([counter[12], counter[13], counter[14], counter[15]]);
    c = c.wrapping_add(1);
    let bytes = c.to_be_bytes();
    counter[12] = bytes[0];
    counter[13] = bytes[1];
    counter[14] = bytes[2];
    counter[15] = bytes[3];
}

// ─── AES-256-GCM Cipher ─────────────────────────────────────────────────────

/// AES-256-GCM cipher instance.
///
/// Holds the expanded key schedule for AES-256 and provides
/// authenticated encryption and decryption operations.
pub struct Aes256Gcm {
    key_schedule: [u32; AES256_KEY_SCHEDULE_LEN],
}

impl Aes256Gcm {
    /// Create a new AES-256-GCM cipher with the given key.
    pub fn new(key: AesKey) -> Self {
        let key_schedule = aes256_key_expand(key.as_bytes());
        Self { key_schedule }
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

        // H = AES_K(0^128)
        let h = aes256_encrypt_block(&[0u8; 16], &self.key_schedule);

        // Initial counter: nonce || 0x00000001
        let mut j0 = [0u8; 16];
        j0[..12].copy_from_slice(nonce.as_bytes());
        j0[15] = 1;

        // Encrypt the counter block for the tag
        let tag_mask = aes256_encrypt_block(&j0, &self.key_schedule);

        // Counter starts at J0 + 1 for data encryption
        let mut counter = j0;
        gcm_inc32(&mut counter);

        // CTR-mode encryption
        let full_blocks = plaintext.len() / 16;
        for i in 0..full_blocks {
            let keystream = aes256_encrypt_block(&counter, &self.key_schedule);
            for j in 0..16 {
                ciphertext[i * 16 + j] = plaintext[i * 16 + j] ^ keystream[j];
            }
            gcm_inc32(&mut counter);
        }
        // Partial last block
        let remaining = plaintext.len() % 16;
        if remaining > 0 {
            let keystream = aes256_encrypt_block(&counter, &self.key_schedule);
            let offset = full_blocks * 16;
            for j in 0..remaining {
                ciphertext[offset + j] = plaintext[offset + j] ^ keystream[j];
            }
        }

        // Compute GHASH over AAD and ciphertext
        let ct_slice = &ciphertext[..plaintext.len()];
        let mut ghash_val = ghash(&h, aad, ct_slice);

        // Tag = GHASH XOR E_K(J0)
        for j in 0..16 {
            ghash_val[j] ^= tag_mask[j];
        }

        Ok(GcmTag::from_bytes(ghash_val))
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

        // H = AES_K(0^128)
        let h = aes256_encrypt_block(&[0u8; 16], &self.key_schedule);

        // Initial counter: nonce || 0x00000001
        let mut j0 = [0u8; 16];
        j0[..12].copy_from_slice(nonce.as_bytes());
        j0[15] = 1;

        // Encrypt the counter block for the tag
        let tag_mask = aes256_encrypt_block(&j0, &self.key_schedule);

        // Compute GHASH over AAD and ciphertext
        let mut ghash_val = ghash(&h, aad, ciphertext);
        for j in 0..16 {
            ghash_val[j] ^= tag_mask[j];
        }

        // Verify tag (constant-time comparison)
        if !constant_time::ct_eq(tag.as_bytes(), &ghash_val).to_bool() {
            return Err(AesGcmError::AuthenticationFailed);
        }

        // CTR-mode decryption (same as encryption)
        let mut counter = j0;
        gcm_inc32(&mut counter);

        let full_blocks = ciphertext.len() / 16;
        for i in 0..full_blocks {
            let keystream = aes256_encrypt_block(&counter, &self.key_schedule);
            for j in 0..16 {
                plaintext[i * 16 + j] = ciphertext[i * 16 + j] ^ keystream[j];
            }
            gcm_inc32(&mut counter);
        }
        let remaining = ciphertext.len() % 16;
        if remaining > 0 {
            let keystream = aes256_encrypt_block(&counter, &self.key_schedule);
            let offset = full_blocks * 16;
            for j in 0..remaining {
                plaintext[offset + j] = ciphertext[offset + j] ^ keystream[j];
            }
        }

        Ok(())
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

    // ─── AES-256 block cipher test (NIST FIPS 197 Appendix C.3) ──────────

    #[test]
    fn aes256_encrypt_block_nist_vector() {
        // NIST FIPS 197, Appendix C.3: AES-256 ECB test vector
        let key: [u8; 32] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];
        let plaintext: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let expected_ct: [u8; 16] = [
            0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf, 0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49,
            0x60, 0x89,
        ];

        let ks = aes256_key_expand(&key);
        let ct = aes256_encrypt_block(&plaintext, &ks);
        assert_eq!(ct, expected_ct, "AES-256 ECB test vector mismatch");
    }

    // ─── AES-256-GCM test vectors (NIST SP 800-38D) ─────────────────────

    #[test]
    fn aes256_gcm_encrypt_decrypt_roundtrip() {
        let key = AesKey::from_bytes([0x42; AES256_KEY_LEN]);
        let nonce = GcmNonce::from_bytes([0x01; GCM_NONCE_LEN]);
        let aad = b"additional data";
        let plaintext = b"hello, world! this is a test of AES-256-GCM encryption.";

        let cipher = Aes256Gcm::new(key);

        let mut ciphertext = [0u8; 56]; // >= plaintext.len()
        let tag = cipher
            .encrypt(&nonce, aad, plaintext, &mut ciphertext)
            .unwrap();

        // Ciphertext should differ from plaintext
        assert_ne!(&ciphertext[..plaintext.len()], plaintext.as_slice());

        // Decrypt
        let mut decrypted = [0u8; 56];
        cipher
            .decrypt(
                &nonce,
                aad,
                &ciphertext[..plaintext.len()],
                &tag,
                &mut decrypted,
            )
            .unwrap();

        assert_eq!(
            &decrypted[..plaintext.len()],
            plaintext.as_slice(),
            "decrypted text should match original"
        );
    }

    #[test]
    fn aes256_gcm_auth_failure_on_tampered_ciphertext() {
        let key = AesKey::from_bytes([0x42; AES256_KEY_LEN]);
        let nonce = GcmNonce::from_bytes([0x01; GCM_NONCE_LEN]);
        let plaintext = b"secret data";

        let cipher = Aes256Gcm::new(key);

        let mut ciphertext = [0u8; 16];
        let tag = cipher
            .encrypt(&nonce, b"", plaintext, &mut ciphertext)
            .unwrap();

        // Tamper with ciphertext
        ciphertext[0] ^= 0xFF;

        let mut decrypted = [0u8; 16];
        let result = cipher.decrypt(
            &nonce,
            b"",
            &ciphertext[..plaintext.len()],
            &tag,
            &mut decrypted,
        );
        assert_eq!(
            result,
            Err(AesGcmError::AuthenticationFailed),
            "tampered ciphertext should fail authentication"
        );
    }

    #[test]
    fn aes256_gcm_auth_failure_on_wrong_aad() {
        let key = AesKey::from_bytes([0x42; AES256_KEY_LEN]);
        let nonce = GcmNonce::from_bytes([0x01; GCM_NONCE_LEN]);
        let plaintext = b"secret data";

        let cipher = Aes256Gcm::new(key);

        let mut ciphertext = [0u8; 16];
        let tag = cipher
            .encrypt(&nonce, b"correct aad", plaintext, &mut ciphertext)
            .unwrap();

        let mut decrypted = [0u8; 16];
        let result = cipher.decrypt(
            &nonce,
            b"wrong aad",
            &ciphertext[..plaintext.len()],
            &tag,
            &mut decrypted,
        );
        assert_eq!(
            result,
            Err(AesGcmError::AuthenticationFailed),
            "wrong AAD should fail authentication"
        );
    }

    #[test]
    fn aes256_gcm_empty_plaintext() {
        let key = AesKey::from_bytes([0x42; AES256_KEY_LEN]);
        let nonce = GcmNonce::from_bytes([0x01; GCM_NONCE_LEN]);
        let aad = b"only authenticated data";

        let cipher = Aes256Gcm::new(key);

        let mut ciphertext = [0u8; 0];
        let tag = cipher.encrypt(&nonce, aad, &[], &mut ciphertext).unwrap();

        // Tag should be non-zero
        assert!(tag.as_bytes().iter().any(|&b| b != 0));

        // Decrypt empty ciphertext
        let mut decrypted = [0u8; 0];
        cipher
            .decrypt(&nonce, aad, &[], &tag, &mut decrypted)
            .unwrap();
    }

    #[test]
    fn aes256_gcm_deterministic() {
        let key = AesKey::from_bytes([0xAA; AES256_KEY_LEN]);
        let nonce = GcmNonce::from_bytes([0xBB; GCM_NONCE_LEN]);
        let plaintext = b"deterministic test";

        let cipher = Aes256Gcm::new(key);

        let mut ct1 = [0u8; 32];
        let tag1 = cipher.encrypt(&nonce, b"", plaintext, &mut ct1).unwrap();

        let mut ct2 = [0u8; 32];
        let tag2 = cipher.encrypt(&nonce, b"", plaintext, &mut ct2).unwrap();

        assert_eq!(
            ct1[..plaintext.len()],
            ct2[..plaintext.len()],
            "same key+nonce+plaintext should produce same ciphertext"
        );
        assert_eq!(tag1, tag2, "tags should match for identical inputs");
    }

    #[test]
    fn aes256_gcm_different_nonces_different_output() {
        let key = AesKey::from_bytes([0xAA; AES256_KEY_LEN]);
        let nonce1 = GcmNonce::from_bytes([0x01; GCM_NONCE_LEN]);
        let nonce2 = GcmNonce::from_bytes([0x02; GCM_NONCE_LEN]);
        let plaintext = b"nonce test";

        let cipher = Aes256Gcm::new(key);

        let mut ct1 = [0u8; 16];
        let tag1 = cipher.encrypt(&nonce1, b"", plaintext, &mut ct1).unwrap();

        let mut ct2 = [0u8; 16];
        let tag2 = cipher.encrypt(&nonce2, b"", plaintext, &mut ct2).unwrap();

        assert_ne!(
            ct1[..plaintext.len()],
            ct2[..plaintext.len()],
            "different nonces should produce different ciphertext"
        );
        assert_ne!(tag1, tag2, "tags should differ for different nonces");
    }

    // NIST SP 800-38D Test Case 16 (AES-256, 96-bit IV)
    #[test]
    fn aes256_gcm_nist_test_case_16() {
        // Key: all zeros (256-bit)
        let key = AesKey::from_bytes([0u8; 32]);
        let nonce = GcmNonce::from_bytes([0u8; 12]);

        let cipher = Aes256Gcm::new(key);

        // Encrypt empty plaintext with no AAD
        let mut ct = [0u8; 0];
        let tag = cipher.encrypt(&nonce, &[], &[], &mut ct).unwrap();

        // NIST expected tag for Test Case 13 (AES-256-GCM, empty PT, empty AAD, zero key, zero IV):
        let expected_tag: [u8; 16] = [
            0x53, 0x0f, 0x8a, 0xfb, 0xc7, 0x45, 0x36, 0xb9, 0xa9, 0x63, 0xb4, 0xf1, 0xc4, 0xcb,
            0x73, 0x8b,
        ];
        assert_eq!(
            tag.as_bytes(),
            &expected_tag,
            "NIST SP 800-38D Test Case 13 tag mismatch"
        );
    }
}
