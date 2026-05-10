// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HMAC-SHA1 (RFC 2104) — clean-room `#![no_std]` implementation.
//!
//! This module is the keyed-MAC primitive RFC 6238 mandates for TOTP.
//! The construction is exactly as RFC 2104 §2:
//!
//! ```text
//! HMAC(K, m) = H((K' XOR opad) || H((K' XOR ipad) || m))
//! ```
//!
//! where
//! - `K'` is the working key: `H(K)` if `len(K) > B`, else `K` zero-padded
//!   to `B` bytes;
//! - `B` is the SHA-1 block length (64 bytes);
//! - `ipad = 0x36 * B` and `opad = 0x5C * B`.
//!
//! ## Why HMAC-SHA1 specifically?
//!
//! See [`crate::sha1`] — same justification: RFC 6238 mandates SHA-1 for
//! authenticator-app interop. Despite SHA-1 being collision-broken,
//! HMAC-SHA1's PRF security (the property HMAC actually relies on) holds
//! per RFC 6151 §2.3. Future SHA-3 TOTP variants will plug in alongside
//! this module without breaking it.

use crate::sha1::{Sha1, BLOCK_LEN, DIGEST_LEN};

/// HMAC-SHA1 inner-padding byte (RFC 2104 §2).
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 2104 §2 ipad, public constant
const IPAD: u8 = 0x36;

/// HMAC-SHA1 outer-padding byte (RFC 2104 §2).
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 2104 §2 opad, public constant
const OPAD: u8 = 0x5C;

/// Streaming HMAC-SHA1 context.
///
/// Construct with [`HmacSha1::new`]; absorb message bytes via
/// [`HmacSha1::update`]; finalise with [`HmacSha1::finalize`].
///
/// The context holds the key-derived working state (the inner SHA-1
/// initialised with `K' XOR ipad`) plus the precomputed `K' XOR opad`
/// so finalise only needs one extra SHA-1 pass.
#[derive(Clone)]
pub struct HmacSha1 {
    /// Inner SHA-1, primed with `K' XOR ipad`. `update` feeds straight
    /// in.
    inner: Sha1,
    /// Outer key (`K' XOR opad`), kept for finalise.
    outer_key: [u8; BLOCK_LEN],
}

impl HmacSha1 {
    /// Construct a fresh HMAC-SHA1 instance for the given key.
    ///
    /// Per RFC 2104 §2:
    /// - If `key.len() > BLOCK_LEN`, the working key is `SHA1(key)`
    ///   right-padded with zeros.
    /// - Otherwise the working key is `key` right-padded with zeros to
    ///   `BLOCK_LEN`.
    pub fn new(key: &[u8]) -> Self {
        let mut working_key = [0u8; BLOCK_LEN];
        if key.len() > BLOCK_LEN {
            let mut h = Sha1::new();
            h.update(key);
            let d = h.finalize();
            working_key[..DIGEST_LEN].copy_from_slice(&d);
        } else {
            working_key[..key.len()].copy_from_slice(key);
        }

        let mut inner_key = [0u8; BLOCK_LEN];
        let mut outer_key = [0u8; BLOCK_LEN];
        for i in 0..BLOCK_LEN {
            inner_key[i] = working_key[i] ^ IPAD;
            outer_key[i] = working_key[i] ^ OPAD;
        }

        let mut inner = Sha1::new();
        inner.update(&inner_key);

        Self { inner, outer_key }
    }

    /// Absorb message bytes.
    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    /// Finalise and return the 20-byte HMAC tag.
    ///
    /// Consumes the context — clone before finalize for incremental
    /// MAC checks.
    pub fn finalize(self) -> [u8; DIGEST_LEN] {
        let inner_digest = self.inner.finalize();
        let mut outer = Sha1::new();
        outer.update(&self.outer_key);
        outer.update(&inner_digest);
        outer.finalize()
    }
}

/// One-shot HMAC-SHA1: `tag = HMAC-SHA1(key, message)`.
pub fn hmac_sha1(key: &[u8], message: &[u8]) -> [u8; DIGEST_LEN] {
    let mut h = HmacSha1::new(key);
    h.update(message);
    h.finalize()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "hmac_sha1_test_vectors.rs"]
mod test_vectors;

#[cfg(test)]
mod tests {
    use super::test_vectors::*;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn hex(digest: &[u8; DIGEST_LEN]) -> alloc::string::String {
        let mut s = alloc::string::String::with_capacity(DIGEST_LEN * 2);
        for b in digest {
            s.push(nibble_hex(b >> 4));
            s.push(nibble_hex(b & 0x0F));
        }
        s
    }

    fn nibble_hex(n: u8) -> char {
        match n {
            0..=9 => (b'0' + n) as char,
            _ => (b'a' + n - 10) as char,
        }
    }

    #[test]
    fn rfc2202_vector_1() {
        // RFC 2202 §3 Test Case 1.
        let tag = hmac_sha1(RFC2202_KEY_1, RFC2202_DATA_1);
        assert_eq!(hex(&tag), RFC2202_TAG_1_HEX);
    }

    #[test]
    fn rfc2202_vector_2() {
        // RFC 2202 §3 Test Case 2.
        let tag = hmac_sha1(RFC2202_KEY_2, RFC2202_DATA_2);
        assert_eq!(hex(&tag), RFC2202_TAG_2_HEX);
    }

    #[test]
    fn rfc2202_vector_3() {
        // RFC 2202 §3 Test Case 3 — repeated 0xDD * 50.
        let key: Vec<u8> = vec![0xAA; 20];
        let data: Vec<u8> = vec![0xDD; 50];
        let tag = hmac_sha1(&key, &data);
        assert_eq!(hex(&tag), RFC2202_TAG_3_HEX);
    }

    #[test]
    fn rfc2202_vector_4() {
        // RFC 2202 §3 Test Case 4 — non-trivial 25-byte key, repeated 0xCD * 50.
        let data: Vec<u8> = vec![0xCD; 50];
        let tag = hmac_sha1(RFC2202_KEY_4, &data);
        assert_eq!(hex(&tag), RFC2202_TAG_4_HEX);
    }

    #[test]
    fn rfc2202_vector_5() {
        // RFC 2202 §3 Test Case 5 — 20-byte 0x0C key.
        let key: Vec<u8> = vec![0x0C; 20];
        let tag = hmac_sha1(&key, RFC2202_DATA_5);
        assert_eq!(hex(&tag), RFC2202_TAG_5_HEX);
    }

    #[test]
    fn rfc2202_vector_6() {
        // RFC 2202 §3 Test Case 6 — long key (80 bytes, hashed first).
        let key: Vec<u8> = vec![0xAA; 80];
        let tag = hmac_sha1(&key, RFC2202_DATA_6);
        assert_eq!(hex(&tag), RFC2202_TAG_6_HEX);
    }

    #[test]
    fn rfc2202_vector_7() {
        // RFC 2202 §3 Test Case 7 — long key + long message.
        let key: Vec<u8> = vec![0xAA; 80];
        let tag = hmac_sha1(&key, RFC2202_DATA_7);
        assert_eq!(hex(&tag), RFC2202_TAG_7_HEX);
    }

    #[test]
    fn streaming_matches_one_shot() {
        let key = b"streaming-key";
        let message = b"abcdefghijklmnopqrstuvwxyz0123456789";
        let one_shot = hmac_sha1(key, message);
        let mut h = HmacSha1::new(key);
        for byte in message {
            h.update(&[*byte]);
        }
        let streamed = h.finalize();
        assert_eq!(one_shot, streamed);
    }

    #[test]
    fn key_at_block_boundary() {
        // Key exactly BLOCK_LEN bytes — should NOT be hashed first.
        let key: Vec<u8> = (0..64u8).collect();
        let message = b"check-block-boundary";
        let tag = hmac_sha1(&key, message);
        // No KAT here — just smoke-test that we get a deterministic
        // 20-byte result with no panics on the boundary.
        assert_eq!(tag.len(), 20);
        // Re-run independent context — must be identical.
        let tag2 = hmac_sha1(&key, message);
        assert_eq!(tag, tag2);
    }

    #[test]
    fn long_key_is_hashed_first() {
        let key: Vec<u8> = vec![0x01; 200];
        let mut prehash = Sha1::new();
        prehash.update(&key);
        let prehashed = prehash.finalize();
        let mut padded = [0u8; 64];
        padded[..20].copy_from_slice(&prehashed);
        // HMAC of long key SHALL equal HMAC of its SHA1-then-padded form.
        let m = b"verify-long-key-handling";
        let with_long_key = hmac_sha1(&key, m);
        let with_prehashed = hmac_sha1(&padded, m);
        assert_eq!(with_long_key, with_prehashed);
    }

    #[test]
    fn empty_key_does_not_panic() {
        let tag = hmac_sha1(b"", b"any message");
        assert_eq!(tag.len(), 20);
    }

    #[test]
    fn empty_message_is_well_defined() {
        let tag = hmac_sha1(b"some key", b"");
        let tag2 = hmac_sha1(b"some key", b"");
        assert_eq!(tag, tag2);
    }
}
