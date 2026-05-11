// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ChaCha20-Poly1305 AEAD (RFC 8439).
//!
//! TLS 1.3 `TLS_CHACHA20_POLY1305_SHA256` cipher suite. Used
//! by `tls-client` for negotiated wire encryption when AES
//! hardware is unavailable or the operator prefers ChaCha20.
//!
//! ## API
//!
//! ```ignore
//! use smallaios_security::crypto::chacha20_poly1305 as aead;
//! let mut buf = b"plaintext".to_vec();
//! let tag = aead::seal(&key, &nonce, &aad, &mut buf);
//! // buf is now ciphertext; tag goes on the wire alongside it.
//! aead::open(&key, &nonce, &aad, &mut buf, &tag)?;
//! // buf is back to plaintext, or open() returned Err on tamper.
//! ```

use super::chacha20::{self, KEY_LEN as CC_KEY_LEN, NONCE_LEN as CC_NONCE_LEN};
use super::poly1305;

/// 256-bit AEAD key (same as ChaCha20).
pub const KEY_LEN: usize = CC_KEY_LEN;

/// 96-bit IETF nonce (same as ChaCha20).
pub const NONCE_LEN: usize = CC_NONCE_LEN;

/// 16-byte authentication tag.
pub const TAG_LEN: usize = poly1305::TAG_LEN;

/// AEAD authentication failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthFail;

/// Derive the Poly1305 one-time key from `(key, nonce)` per
/// RFC 8439 § 2.6: run ChaCha20 with counter = 0 against a
/// 64-byte all-zero buffer, take the first 32 bytes.
fn derive_poly_key(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    chacha20::apply_keystream(key, nonce, 0, &mut buf);
    let mut out = [0u8; 32];
    out.copy_from_slice(&buf[..32]);
    out
}

/// Build the Poly1305 input per RFC 8439 § 2.8:
/// `aad || pad16(aad) || ciphertext || pad16(ciphertext) ||
///  aad_len_le8 || ct_len_le8`.
fn build_poly_input(aad: &[u8], ciphertext: &[u8]) -> alloc::vec::Vec<u8> {
    let aad_pad = (16 - (aad.len() % 16)) % 16;
    let ct_pad = (16 - (ciphertext.len() % 16)) % 16;
    let mut input =
        alloc::vec::Vec::with_capacity(aad.len() + aad_pad + ciphertext.len() + ct_pad + 16);
    input.extend_from_slice(aad);
    input.resize(input.len() + aad_pad, 0);
    input.extend_from_slice(ciphertext);
    input.resize(input.len() + ct_pad, 0);
    input.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    input.extend_from_slice(&(ciphertext.len() as u64).to_le_bytes());
    input
}

/// Seal `buf` in place. `buf` enters as plaintext, exits as
/// ciphertext. Returns the 16-byte authentication tag.
pub fn seal(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buf: &mut [u8],
) -> [u8; TAG_LEN] {
    let poly_key = derive_poly_key(key, nonce);
    // ChaCha20 stream starts at counter 1 for the encryption
    // step — counter 0 was consumed by the Poly1305 key
    // derivation.
    chacha20::apply_keystream(key, nonce, 1, buf);
    let poly_input = build_poly_input(aad, buf);
    poly1305::poly1305(&poly_key, &poly_input)
}

/// Open `buf` in place. `buf` enters as ciphertext, exits as
/// plaintext on success. Returns `Err(AuthFail)` on
/// authentication mismatch — `buf`'s contents on failure are
/// unspecified and MUST NOT be revealed to the caller.
pub fn open(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buf: &mut [u8],
    tag: &[u8; TAG_LEN],
) -> Result<(), AuthFail> {
    let poly_key = derive_poly_key(key, nonce);
    let poly_input = build_poly_input(aad, buf);
    let computed = poly1305::poly1305(&poly_key, &poly_input);
    if !constant_time_eq(&computed, tag) {
        return Err(AuthFail);
    }
    chacha20::apply_keystream(key, nonce, 1, buf);
    Ok(())
}

fn constant_time_eq(a: &[u8; TAG_LEN], b: &[u8; TAG_LEN]) -> bool {
    let mut acc = 0u8;
    for i in 0..TAG_LEN {
        acc |= a[i] ^ b[i];
    }
    acc == 0
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;

    // RFC 8439 § 2.8.2 test vector.
    #[test]
    fn rfc_8439_2_8_2() {
        let plaintext = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let aad: [u8; 12] = [
            0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
        ];
        let key: [u8; 32] = [
            0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d,
            0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b,
            0x9c, 0x9d, 0x9e, 0x9f,
        ];
        let nonce: [u8; 12] = [
            0x07, 0, 0, 0, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
        ];
        let mut buf = plaintext.to_vec();
        let tag = seal(&key, &nonce, &aad, &mut buf);
        let expected_ct_first = [
            0xd3, 0x1a, 0x8d, 0x34, 0x64, 0x8e, 0x60, 0xdb, 0x7b, 0x86, 0xaf, 0xbc, 0x53, 0xef,
            0x7e, 0xc2,
        ];
        let expected_tag = [
            0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60,
            0x06, 0x91,
        ];
        assert_eq!(&buf[..16], &expected_ct_first);
        assert_eq!(tag, expected_tag);

        // Round-trip.
        open(&key, &nonce, &aad, &mut buf, &tag).unwrap();
        assert_eq!(&buf, plaintext);
    }

    #[test]
    fn tampered_ciphertext_rejected() {
        let key = [0x42u8; KEY_LEN];
        let nonce = [0x9eu8; NONCE_LEN];
        let aad = b"context";
        let mut buf = b"hello world".to_vec();
        let tag = seal(&key, &nonce, aad, &mut buf);
        buf[0] ^= 0x01;
        assert_eq!(open(&key, &nonce, aad, &mut buf, &tag), Err(AuthFail));
    }

    #[test]
    fn tampered_tag_rejected() {
        let key = [0u8; KEY_LEN];
        let nonce = [0u8; NONCE_LEN];
        let mut buf = b"data".to_vec();
        let mut tag = seal(&key, &nonce, b"", &mut buf);
        tag[0] ^= 0x01;
        assert_eq!(open(&key, &nonce, b"", &mut buf, &tag), Err(AuthFail));
    }

    #[test]
    fn tampered_aad_rejected() {
        let key = [0u8; KEY_LEN];
        let nonce = [0u8; NONCE_LEN];
        let mut buf = b"data".to_vec();
        let tag = seal(&key, &nonce, b"good", &mut buf);
        assert_eq!(open(&key, &nonce, b"bad", &mut buf, &tag), Err(AuthFail));
    }

    #[test]
    fn empty_plaintext_round_trip() {
        let key = [0x42u8; KEY_LEN];
        let nonce = [0x9eu8; NONCE_LEN];
        let aad = b"associated";
        let mut buf: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
        let tag = seal(&key, &nonce, aad, &mut buf);
        open(&key, &nonce, aad, &mut buf, &tag).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn empty_aad_round_trip() {
        let key = [0x42u8; KEY_LEN];
        let nonce = [0x9eu8; NONCE_LEN];
        let original = b"plaintext bytes";
        let mut buf = original.to_vec();
        let tag = seal(&key, &nonce, &[], &mut buf);
        open(&key, &nonce, &[], &mut buf, &tag).unwrap();
        assert_eq!(&buf, original);
    }
}
