// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC packet protection (RFC 9001 Section 5).
//!
//! Provides AEAD encrypt/decrypt interface for QUIC packet payloads
//! and header protection for packet number confidentiality.
//!
//! This is a stub that defines the interface. Full AES-256-GCM and
//! ChaCha20-Poly1305 implementations will come from the security crate.

use crate::NetError;
use alloc::vec::Vec;

/// AEAD tag length (16 bytes for both AES-GCM and ChaCha20-Poly1305).
pub const AEAD_TAG_LEN: usize = 16;

/// AES-GCM nonce length (12 bytes).
pub const AEAD_NONCE_LEN: usize = 12;

/// Header protection sample length (16 bytes).
pub const HP_SAMPLE_LEN: usize = 16;

/// QUIC cipher suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipherSuite {
    /// TLS_AES_128_GCM_SHA256.
    Aes128Gcm,
    /// TLS_AES_256_GCM_SHA384.
    Aes256Gcm,
    /// TLS_CHACHA20_POLY1305_SHA256.
    ChaCha20Poly1305,
}

/// QUIC encryption level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionLevel {
    /// Initial (derived from connection ID + salt).
    Initial,
    /// Handshake (TLS handshake keys).
    Handshake,
    /// 0-RTT (early data keys).
    ZeroRtt,
    /// 1-RTT (application data keys).
    OneRtt,
}

/// QUIC packet protection keys for one direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketProtectionKeys {
    /// Traffic key (for AEAD).
    pub key: Vec<u8>,
    /// Traffic IV (combined with packet number to form nonce).
    pub iv: Vec<u8>,
    /// Header protection key.
    pub hp_key: Vec<u8>,
    /// Cipher suite.
    pub cipher: CipherSuite,
    /// Key phase (for 1-RTT key rotation).
    pub key_phase: bool,
}

impl PacketProtectionKeys {
    /// Create new packet protection keys.
    pub fn new(key: &[u8], iv: &[u8], hp_key: &[u8], cipher: CipherSuite) -> Self {
        Self {
            key: Vec::from(key),
            iv: Vec::from(iv),
            hp_key: Vec::from(hp_key),
            cipher,
            key_phase: false,
        }
    }

    /// Compute the AEAD nonce from the IV and packet number.
    ///
    /// Per RFC 9001 Section 5.3: nonce = iv XOR packet_number (left-padded).
    pub fn compute_nonce(&self, packet_number: u64) -> [u8; AEAD_NONCE_LEN] {
        let mut nonce = [0u8; AEAD_NONCE_LEN];
        nonce.copy_from_slice(&self.iv[..AEAD_NONCE_LEN.min(self.iv.len())]);
        let pn_bytes = packet_number.to_be_bytes();
        // XOR the last 8 bytes of the nonce with the packet number
        for i in 0..8 {
            nonce[AEAD_NONCE_LEN - 8 + i] ^= pn_bytes[i];
        }
        nonce
    }
}

/// AEAD encryption (stub — returns plaintext with a dummy tag appended).
///
/// Real implementation would use AES-256-GCM or ChaCha20-Poly1305 from
/// the security crate.
pub fn aead_encrypt(
    keys: &PacketProtectionKeys,
    packet_number: u64,
    aad: &[u8],
    plaintext: &[u8],
    output: &mut [u8],
) -> Result<usize, NetError> {
    let total = plaintext.len() + AEAD_TAG_LEN;
    if output.len() < total {
        return Err(NetError::BufferTooSmall);
    }
    let nonce = keys.compute_nonce(packet_number);

    // Stub: XOR with first key byte for minimal obfuscation
    let key_byte = keys.key.first().copied().unwrap_or(0);
    for (i, byte) in plaintext.iter().enumerate() {
        output[i] = byte ^ key_byte ^ nonce[i % AEAD_NONCE_LEN];
    }
    // Stub tag: keyed hash of nonce + AAD + ciphertext
    let mut tag = [0u8; AEAD_TAG_LEN];
    // Seed tag from key (use rotating XOR with position to avoid cancellation)
    for (i, &byte) in keys.key.iter().enumerate() {
        tag[i % AEAD_TAG_LEN] =
            tag[i % AEAD_TAG_LEN].wrapping_add(byte.wrapping_mul((i as u8).wrapping_add(1)));
    }
    // Mix in nonce, AAD, and ciphertext
    for (i, &byte) in nonce
        .iter()
        .chain(aad.iter())
        .chain(output[..plaintext.len()].iter())
        .enumerate()
    {
        tag[i % AEAD_TAG_LEN] ^= byte;
    }
    output[plaintext.len()..total].copy_from_slice(&tag);
    Ok(total)
}

/// AEAD decryption (stub — reverses the stub encryption).
pub fn aead_decrypt(
    keys: &PacketProtectionKeys,
    packet_number: u64,
    aad: &[u8],
    ciphertext: &[u8],
    output: &mut [u8],
) -> Result<usize, NetError> {
    if ciphertext.len() < AEAD_TAG_LEN {
        return Err(NetError::PacketTooShort);
    }
    let plaintext_len = ciphertext.len() - AEAD_TAG_LEN;
    if output.len() < plaintext_len {
        return Err(NetError::BufferTooSmall);
    }
    let nonce = keys.compute_nonce(packet_number);

    // Verify tag
    let encrypted = &ciphertext[..plaintext_len];
    let received_tag = &ciphertext[plaintext_len..];
    let mut expected_tag = [0u8; AEAD_TAG_LEN];
    // Seed from key (same as encrypt)
    for (i, &byte) in keys.key.iter().enumerate() {
        expected_tag[i % AEAD_TAG_LEN] = expected_tag[i % AEAD_TAG_LEN]
            .wrapping_add(byte.wrapping_mul((i as u8).wrapping_add(1)));
    }
    // Mix in nonce, AAD, and ciphertext
    for (i, &byte) in nonce
        .iter()
        .chain(aad.iter())
        .chain(encrypted.iter())
        .enumerate()
    {
        expected_tag[i % AEAD_TAG_LEN] ^= byte;
    }
    if received_tag != expected_tag {
        return Err(NetError::ChecksumMismatch);
    }

    // Decrypt
    let key_byte = keys.key.first().copied().unwrap_or(0);
    for (i, byte) in encrypted.iter().enumerate() {
        output[i] = byte ^ key_byte ^ nonce[i % AEAD_NONCE_LEN];
    }
    Ok(plaintext_len)
}

/// Apply header protection (stub — XORs first byte and PN bytes with HP mask).
pub fn apply_header_protection(
    hp_key: &[u8],
    sample: &[u8; HP_SAMPLE_LEN],
    first_byte: &mut u8,
    pn_bytes: &mut [u8],
) {
    // Stub: derive mask from hp_key and sample via XOR
    let mut mask = [0u8; 5];
    for (i, m) in mask.iter_mut().enumerate() {
        *m = hp_key.get(i).copied().unwrap_or(0) ^ sample[i];
    }

    // Protect first byte (lower 4 bits for long header, lower 5 for short)
    let is_long = *first_byte & 0x80 != 0;
    if is_long {
        *first_byte ^= mask[0] & 0x0F;
    } else {
        *first_byte ^= mask[0] & 0x1F;
    }

    // Protect packet number bytes
    for (i, byte) in pn_bytes.iter_mut().enumerate() {
        if i + 1 < mask.len() {
            *byte ^= mask[i + 1];
        }
    }
}

/// Remove header protection (same operation as apply — XOR is its own inverse).
pub fn remove_header_protection(
    hp_key: &[u8],
    sample: &[u8; HP_SAMPLE_LEN],
    first_byte: &mut u8,
    pn_bytes: &mut [u8],
) {
    // XOR is self-inverse — same function
    apply_header_protection(hp_key, sample, first_byte, pn_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keys() -> PacketProtectionKeys {
        PacketProtectionKeys::new(
            &[0xAA; 32],
            &[0xBB; 12],
            &[0xCC; 16],
            CipherSuite::Aes256Gcm,
        )
    }

    #[test]
    fn test_compute_nonce() {
        let keys = test_keys();
        let nonce = keys.compute_nonce(1);
        assert_eq!(nonce.len(), 12);
        // Last byte should be 0xBB ^ 0x01
        assert_eq!(nonce[11], 0xBB ^ 0x01);
        // Bytes before the PN area should remain as IV
        assert_eq!(nonce[0], 0xBB);
    }

    #[test]
    fn test_compute_nonce_zero() {
        let keys = test_keys();
        let nonce = keys.compute_nonce(0);
        // Should be the same as IV when PN = 0
        for (i, &b) in nonce.iter().enumerate() {
            assert_eq!(b, keys.iv[i]);
        }
    }

    #[test]
    fn test_aead_encrypt_decrypt_roundtrip() {
        let keys = test_keys();
        let plaintext = b"Hello, QUIC!";
        let aad = b"additional data";
        let mut ciphertext = [0u8; 128];
        let ct_len = aead_encrypt(&keys, 1, aad, plaintext, &mut ciphertext).unwrap();
        assert_eq!(ct_len, plaintext.len() + AEAD_TAG_LEN);

        let mut decrypted = [0u8; 128];
        let pt_len = aead_decrypt(&keys, 1, aad, &ciphertext[..ct_len], &mut decrypted).unwrap();
        assert_eq!(pt_len, plaintext.len());
        assert_eq!(&decrypted[..pt_len], plaintext);
    }

    #[test]
    fn test_aead_decrypt_wrong_key() {
        let keys = test_keys();
        let plaintext = b"secret data";
        let aad = b"aad";
        let mut ciphertext = [0u8; 128];
        let ct_len = aead_encrypt(&keys, 1, aad, plaintext, &mut ciphertext).unwrap();

        // Decrypt with different keys
        let wrong_keys = PacketProtectionKeys::new(
            &[0x01; 32],
            &[0xBB; 12],
            &[0xCC; 16],
            CipherSuite::Aes256Gcm,
        );
        let mut decrypted = [0u8; 128];
        assert_eq!(
            aead_decrypt(&wrong_keys, 1, aad, &ciphertext[..ct_len], &mut decrypted).err(),
            Some(NetError::ChecksumMismatch)
        );
    }

    #[test]
    fn test_aead_decrypt_wrong_pn() {
        let keys = test_keys();
        let plaintext = b"data";
        let aad = b"aad";
        let mut ciphertext = [0u8; 64];
        let ct_len = aead_encrypt(&keys, 1, aad, plaintext, &mut ciphertext).unwrap();

        let mut decrypted = [0u8; 64];
        assert_eq!(
            aead_decrypt(&keys, 2, aad, &ciphertext[..ct_len], &mut decrypted).err(),
            Some(NetError::ChecksumMismatch)
        );
    }

    #[test]
    fn test_aead_decrypt_wrong_aad() {
        let keys = test_keys();
        let plaintext = b"data";
        let aad = b"correct aad";
        let mut ciphertext = [0u8; 64];
        let ct_len = aead_encrypt(&keys, 1, aad, plaintext, &mut ciphertext).unwrap();

        let mut decrypted = [0u8; 64];
        assert_eq!(
            aead_decrypt(
                &keys,
                1,
                b"wrong aad",
                &ciphertext[..ct_len],
                &mut decrypted
            )
            .err(),
            Some(NetError::ChecksumMismatch)
        );
    }

    #[test]
    fn test_aead_encrypt_buffer_too_small() {
        let keys = test_keys();
        let mut buf = [0u8; 5];
        assert_eq!(
            aead_encrypt(&keys, 1, b"", b"data longer than buf", &mut buf).err(),
            Some(NetError::BufferTooSmall)
        );
    }

    #[test]
    fn test_aead_decrypt_too_short() {
        let keys = test_keys();
        let mut output = [0u8; 32];
        assert_eq!(
            aead_decrypt(&keys, 1, b"", &[0u8; 10], &mut output).err(),
            Some(NetError::PacketTooShort)
        );
    }

    #[test]
    fn test_header_protection_roundtrip() {
        let hp_key = [0xDD; 16];
        let sample = [0xEE; HP_SAMPLE_LEN];
        let mut first_byte = 0xC3u8; // Long header
        let original_first = first_byte;
        let mut pn_bytes = [0x00, 0x01, 0x02, 0x03];
        let original_pn = pn_bytes;

        apply_header_protection(&hp_key, &sample, &mut first_byte, &mut pn_bytes);
        // Should be different now
        assert_ne!(first_byte, original_first);

        remove_header_protection(&hp_key, &sample, &mut first_byte, &mut pn_bytes);
        assert_eq!(first_byte, original_first);
        assert_eq!(pn_bytes, original_pn);
    }

    #[test]
    fn test_header_protection_short_header() {
        let hp_key = [0xDD; 16];
        let sample = [0xEE; HP_SAMPLE_LEN];
        let mut first_byte = 0x43u8; // Short header (bit 7 = 0)
        let original = first_byte;
        let mut pn = [0x00];

        apply_header_protection(&hp_key, &sample, &mut first_byte, &mut pn);
        assert_ne!(first_byte, original);

        remove_header_protection(&hp_key, &sample, &mut first_byte, &mut pn);
        assert_eq!(first_byte, original);
    }

    #[test]
    fn test_cipher_suite_variants() {
        assert_ne!(CipherSuite::Aes128Gcm, CipherSuite::Aes256Gcm);
        assert_ne!(CipherSuite::Aes256Gcm, CipherSuite::ChaCha20Poly1305);
    }

    #[test]
    fn test_encryption_level_variants() {
        assert_ne!(EncryptionLevel::Initial, EncryptionLevel::Handshake);
        assert_ne!(EncryptionLevel::ZeroRtt, EncryptionLevel::OneRtt);
    }

    #[test]
    fn test_aead_empty_plaintext() {
        let keys = test_keys();
        let mut ciphertext = [0u8; 32];
        let ct_len = aead_encrypt(&keys, 0, b"", b"", &mut ciphertext).unwrap();
        assert_eq!(ct_len, AEAD_TAG_LEN);

        let mut decrypted = [0u8; 32];
        let pt_len = aead_decrypt(&keys, 0, b"", &ciphertext[..ct_len], &mut decrypted).unwrap();
        assert_eq!(pt_len, 0);
    }
}
