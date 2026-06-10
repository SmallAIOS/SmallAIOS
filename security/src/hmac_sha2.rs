// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HMAC-SHA-256 and HMAC-SHA-384 (RFC 2104 / FIPS 198-1) —
//! clean-room `#![no_std]` implementations.
//!
//! The construction is exactly as RFC 2104 §2:
//!
//! ```text
//! HMAC(K, m) = H((K' XOR opad) || H((K' XOR ipad) || m))
//! ```
//!
//! where
//! - `K'` is the working key: `H(K)` if `len(K) > B`, else `K` zero-padded
//!   to `B` bytes;
//! - `B` is the hash block length (64 bytes for SHA-256, 128 for SHA-384);
//! - `ipad = 0x36 * B` and `opad = 0x5C * B`.
//!
//! ## Why HMAC-SHA-2?
//!
//! TLS 1.3 (RFC 8446) keys its entire key schedule with HKDF
//! (RFC 5869), which is built on HMAC over the cipher suite's hash:
//! HMAC-SHA-256 for `TLS_CHACHA20_POLY1305_SHA256`, HMAC-SHA-384 for
//! `TLS_AES_256_GCM_SHA384`. The Finished message verify_data is a
//! bare HMAC over the transcript hash. See
//! [tls-tcp-client-v1](../../openspec/changes/tls-tcp-client-v1) for
//! the consuming change. Like [`crate::sha2`], this is an interop
//! primitive, not a first-party SmallAIOS construction.

use crate::sha2::{Sha256, Sha384, BLOCK_LEN, DIGEST_LEN, SHA384_BLOCK_LEN, SHA384_DIGEST_LEN};

/// HMAC inner-padding byte (RFC 2104 §2).
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 2104 §2 ipad, public constant
const IPAD: u8 = 0x36;

/// HMAC outer-padding byte (RFC 2104 §2).
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 2104 §2 opad, public constant
const OPAD: u8 = 0x5C;

/// Streaming HMAC-SHA-256 context.
///
/// Construct with [`HmacSha256::new`]; absorb message bytes via
/// [`HmacSha256::update`]; finalise with [`HmacSha256::finalize`].
#[derive(Clone)]
pub struct HmacSha256 {
    /// Inner SHA-256, primed with `K' XOR ipad`.
    inner: Sha256,
    /// Outer key (`K' XOR opad`), kept for finalise.
    outer_key: [u8; BLOCK_LEN],
}

impl HmacSha256 {
    /// Construct a fresh HMAC-SHA-256 instance for the given key.
    pub fn new(key: &[u8]) -> Self {
        let mut working = [0u8; BLOCK_LEN];
        if key.len() > BLOCK_LEN {
            let mut h = Sha256::new();
            h.update(key);
            working[..DIGEST_LEN].copy_from_slice(&h.finalize());
        } else {
            working[..key.len()].copy_from_slice(key);
        }
        let mut inner = Sha256::new();
        let mut ipad_block = [0u8; BLOCK_LEN];
        let mut outer_key = [0u8; BLOCK_LEN];
        for i in 0..BLOCK_LEN {
            ipad_block[i] = working[i] ^ IPAD;
            outer_key[i] = working[i] ^ OPAD;
        }
        inner.update(&ipad_block);
        Self { inner, outer_key }
    }

    /// Absorb message bytes.
    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    /// Finalise and return the 32-byte MAC.
    pub fn finalize(self) -> [u8; DIGEST_LEN] {
        let inner_digest = self.inner.finalize();
        let mut outer = Sha256::new();
        outer.update(&self.outer_key);
        outer.update(&inner_digest);
        outer.finalize()
    }
}

/// One-shot HMAC-SHA-256.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; DIGEST_LEN] {
    let mut mac = HmacSha256::new(key);
    mac.update(message);
    mac.finalize()
}

/// Streaming HMAC-SHA-384 context.
///
/// Construct with [`HmacSha384::new`]; absorb message bytes via
/// [`HmacSha384::update`]; finalise with [`HmacSha384::finalize`].
#[derive(Clone)]
pub struct HmacSha384 {
    /// Inner SHA-384, primed with `K' XOR ipad`.
    inner: Sha384,
    /// Outer key (`K' XOR opad`), kept for finalise.
    outer_key: [u8; SHA384_BLOCK_LEN],
}

impl HmacSha384 {
    /// Construct a fresh HMAC-SHA-384 instance for the given key.
    pub fn new(key: &[u8]) -> Self {
        let mut working = [0u8; SHA384_BLOCK_LEN];
        if key.len() > SHA384_BLOCK_LEN {
            let mut h = Sha384::new();
            h.update(key);
            working[..SHA384_DIGEST_LEN].copy_from_slice(&h.finalize());
        } else {
            working[..key.len()].copy_from_slice(key);
        }
        let mut inner = Sha384::new();
        let mut ipad_block = [0u8; SHA384_BLOCK_LEN];
        let mut outer_key = [0u8; SHA384_BLOCK_LEN];
        for i in 0..SHA384_BLOCK_LEN {
            ipad_block[i] = working[i] ^ IPAD;
            outer_key[i] = working[i] ^ OPAD;
        }
        inner.update(&ipad_block);
        Self { inner, outer_key }
    }

    /// Absorb message bytes.
    pub fn update(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    /// Finalise and return the 48-byte MAC.
    pub fn finalize(self) -> [u8; SHA384_DIGEST_LEN] {
        let inner_digest = self.inner.finalize();
        let mut outer = Sha384::new();
        outer.update(&self.outer_key);
        outer.update(&inner_digest);
        outer.finalize()
    }
}

/// One-shot HMAC-SHA-384.
pub fn hmac_sha384(key: &[u8], message: &[u8]) -> [u8; SHA384_DIGEST_LEN] {
    let mut mac = HmacSha384::new(key);
    mac.update(message);
    mac.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> alloc::string::String {
        use alloc::string::String;
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
            s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
        }
        s
    }

    // RFC 4231 test-case inputs. Expected outputs cross-validated
    // against two independent oracles (Python `hmac` stdlib and
    // OpenSSL 3.0) and the RFC 4231 published values.

    #[test]
    fn rfc4231_tc1() {
        let key = [0x0bu8; 20];
        assert_eq!(
            hex(&hmac_sha256(&key, b"Hi There")),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(
            hex(&hmac_sha384(&key, b"Hi There")),
            "afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59c\
             faea9ea9076ede7f4af152e8b2fa9cb6"
        );
    }

    #[test]
    fn rfc4231_tc2_short_key() {
        let key = b"Jefe";
        let msg = b"what do ya want for nothing?";
        assert_eq!(
            hex(&hmac_sha256(key, msg)),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hex(&hmac_sha384(key, msg)),
            "af45d2e376484031617f78d2b58a6b1b9c7ef464f5a01b47e42ec3736322445e\
             8e2240ca5e69e2c78b3239ecfab21649"
        );
    }

    #[test]
    fn rfc4231_tc3_binary() {
        let key = [0xaau8; 20];
        let msg = [0xddu8; 50];
        assert_eq!(
            hex(&hmac_sha256(&key, &msg)),
            "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"
        );
        assert_eq!(
            hex(&hmac_sha384(&key, &msg)),
            "88062608d3e6ad8a0aa2ace014c8a86f0aa635d947ac9febe83ef4e55966144b\
             2a5ab39dc13814b94e3ab6e101a34f27"
        );
    }

    #[test]
    fn rfc4231_tc6_key_longer_than_block() {
        // 131-byte key exceeds both the SHA-256 (64) and SHA-384
        // (128) block lengths, exercising the H(K) reduction path.
        let key = [0xaau8; 131];
        let msg = b"Test Using Larger Than Block-Size Key - Hash Key First";
        assert_eq!(
            hex(&hmac_sha256(&key, msg)),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
        assert_eq!(
            hex(&hmac_sha384(&key, msg)),
            "4ece084485813e9088d2c63a041bc5b44f9ef1012a2b588f3cd11f05033ac4c6\
             0c2ef6ab4030fe8296248df163f44952"
        );
    }

    #[test]
    fn rfc4231_tc7_key_and_data_longer_than_block() {
        let key = [0xaau8; 131];
        let msg: &[u8] = b"This is a test using a larger than block-size key and a \
larger than block-size data. The key needs to be hashed before being used by \
the HMAC algorithm.";
        assert_eq!(
            hex(&hmac_sha256(&key, msg)),
            "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"
        );
        assert_eq!(
            hex(&hmac_sha384(&key, msg)),
            "6617178e941f020d351e2f254e8fd32c602420feb0b8fb9adccebb82461e99c5\
             a678cc31e799176d3860e6110c46523e"
        );
    }

    #[test]
    fn streaming_equals_oneshot() {
        let key = b"a moderately sized hmac key";
        let msg = b"The quick brown fox jumps over the lazy dog, repeatedly.";
        let oneshot256 = hmac_sha256(key, msg);
        let mut mac = HmacSha256::new(key);
        for chunk in msg.chunks(5) {
            mac.update(chunk);
        }
        assert_eq!(oneshot256, mac.finalize());

        let oneshot384 = hmac_sha384(key, msg);
        let mut mac = HmacSha384::new(key);
        for chunk in msg.chunks(5) {
            mac.update(chunk);
        }
        assert_eq!(oneshot384, mac.finalize());
    }
}
