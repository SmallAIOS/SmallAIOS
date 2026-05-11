// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ChaCha20 stream cipher (IETF RFC 8439).
//!
//! Used as the keystream half of ChaCha20-Poly1305 AEAD, which
//! TLS 1.3 mandates for `TLS_CHACHA20_POLY1305_SHA256` — the
//! cipher suite preferred on hardware without AES acceleration
//! (low-power ARM cores without the Cryptography Extension,
//! RISC-V boards before the Zk* extensions).
//!
//! ## API
//!
//! ```ignore
//! use smallaios_security::crypto::chacha20::ChaCha20;
//! let mut c = ChaCha20::new(&key, &nonce);
//! c.apply_keystream(buf);  // in-place XOR
//! ```

#![allow(clippy::many_single_char_names)]

/// 256-bit key.
pub const KEY_LEN: usize = 32;

/// 96-bit IETF nonce.
pub const NONCE_LEN: usize = 12;

/// 64-byte block size.
pub const BLOCK_LEN: usize = 64;

/// RFC 8439 § 2.3 "expand 32-byte k" constants.
// lgtm[rust/hard-coded-cryptographic-value] — RFC 8439 sigma, public constant.
const SIGMA: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

#[inline(always)]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(7);
}

fn block(key: &[u8; KEY_LEN], counter: u32, nonce: &[u8; NONCE_LEN], out: &mut [u8; BLOCK_LEN]) {
    let mut state = [0u32; 16];
    state[0..4].copy_from_slice(&SIGMA);
    for i in 0..8 {
        state[4 + i] =
            u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
    }
    state[12] = counter;
    for i in 0..3 {
        state[13 + i] = u32::from_le_bytes([
            nonce[i * 4],
            nonce[i * 4 + 1],
            nonce[i * 4 + 2],
            nonce[i * 4 + 3],
        ]);
    }
    let initial = state;

    for _ in 0..10 {
        // Column rounds.
        quarter_round(&mut state, 0, 4, 8, 12);
        quarter_round(&mut state, 1, 5, 9, 13);
        quarter_round(&mut state, 2, 6, 10, 14);
        quarter_round(&mut state, 3, 7, 11, 15);
        // Diagonal rounds.
        quarter_round(&mut state, 0, 5, 10, 15);
        quarter_round(&mut state, 1, 6, 11, 12);
        quarter_round(&mut state, 2, 7, 8, 13);
        quarter_round(&mut state, 3, 4, 9, 14);
    }
    for i in 0..16 {
        state[i] = state[i].wrapping_add(initial[i]);
    }
    for i in 0..16 {
        out[i * 4..i * 4 + 4].copy_from_slice(&state[i].to_le_bytes());
    }
}

/// Apply the ChaCha20 keystream to `buf` in place, starting at
/// the given block counter. Caller is responsible for using a
/// fresh `(key, nonce)` pair per message.
pub fn apply_keystream(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], counter: u32, buf: &mut [u8]) {
    let mut block_buf = [0u8; BLOCK_LEN];
    let mut ctr = counter;
    for chunk in buf.chunks_mut(BLOCK_LEN) {
        block(key, ctr, nonce, &mut block_buf);
        for (i, b) in chunk.iter_mut().enumerate() {
            *b ^= block_buf[i];
        }
        ctr = ctr.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // RFC 8439 § 2.3.2 test vector.
    #[test]
    fn block_vector_8439_2_3_2() {
        let key: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let nonce: [u8; 12] = [0, 0, 0, 9, 0, 0, 0, 0x4a, 0, 0, 0, 0];
        let mut out = [0u8; 64];
        block(&key, 1, &nonce, &mut out);
        let expected: [u8; 64] = [
            0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20,
            0x71, 0xc4, 0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a,
            0xc3, 0xd4, 0x6c, 0x4e, 0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09, 0x14, 0xc2,
            0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2, 0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9,
            0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
        ];
        assert_eq!(out, expected);
    }

    // RFC 8439 § 2.4.2 encrypt vector: "Ladies and Gentlemen of the class of '99…"
    #[test]
    fn keystream_vector_8439_2_4_2() {
        let key: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ];
        let nonce: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0x4a, 0, 0, 0, 0];
        let plaintext = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let mut buf = plaintext.to_vec();
        apply_keystream(&key, &nonce, 1, &mut buf);
        let expected_first = [
            0x6e, 0x2e, 0x35, 0x9a, 0x25, 0x68, 0xf9, 0x80, 0x41, 0xba, 0x07, 0x28, 0xdd, 0x0d,
            0x69, 0x81,
        ];
        assert_eq!(&buf[..16], &expected_first);
    }

    #[test]
    fn keystream_xor_is_involutive() {
        let key = [0x42u8; KEY_LEN];
        let nonce = [0x9eu8; NONCE_LEN];
        let original = b"the quick brown fox jumps over the lazy dog";
        let mut buf = original.to_vec();
        apply_keystream(&key, &nonce, 0, &mut buf);
        assert_ne!(&buf, original);
        apply_keystream(&key, &nonce, 0, &mut buf);
        assert_eq!(&buf, original);
    }
}
