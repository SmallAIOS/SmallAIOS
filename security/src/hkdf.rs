// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HKDF (RFC 5869) over SHA-256 and SHA-384 — clean-room
//! `#![no_std]` implementation.
//!
//! HKDF is the extract-then-expand key-derivation function TLS 1.3
//! (RFC 8446 §7.1) builds its entire key schedule on. The hash is
//! the negotiated cipher suite's: SHA-256 for
//! `TLS_CHACHA20_POLY1305_SHA256`, SHA-384 for
//! `TLS_AES_256_GCM_SHA384`.
//!
//! ```text
//! HKDF-Extract(salt, IKM)       -> PRK   (one HMAC)
//! HKDF-Expand(PRK, info, L)     -> OKM   (ceil(L/HashLen) HMACs)
//! ```
//!
//! The API is alloc-free: `expand` writes into a caller-supplied
//! buffer. Per RFC 5869 §2.3, `L ≤ 255 * HashLen`; longer requests
//! are refused.

use crate::hmac_sha2::{HmacSha256, HmacSha384};
use crate::sha2::{DIGEST_LEN, SHA384_DIGEST_LEN};

/// Error returned when an `expand` request exceeds the RFC 5869
/// §2.3 output cap of `255 * HashLen` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HkdfLengthError;

/// HKDF-Extract with HMAC-SHA-256. An empty `salt` is treated as
/// `HashLen` zero bytes per RFC 5869 §2.2.
pub fn hkdf_extract_sha256(salt: &[u8], ikm: &[u8]) -> [u8; DIGEST_LEN] {
    let zero = [0u8; DIGEST_LEN];
    let salt = if salt.is_empty() { &zero[..] } else { salt };
    let mut mac = HmacSha256::new(salt);
    mac.update(ikm);
    mac.finalize()
}

/// HKDF-Expand with HMAC-SHA-256, filling all of `okm`.
pub fn hkdf_expand_sha256(
    prk: &[u8; DIGEST_LEN],
    info: &[u8],
    okm: &mut [u8],
) -> Result<(), HkdfLengthError> {
    if okm.len() > 255 * DIGEST_LEN {
        return Err(HkdfLengthError);
    }
    let mut t = [0u8; DIGEST_LEN];
    let mut t_len = 0usize;
    let mut counter = 1u8;
    let mut written = 0usize;
    while written < okm.len() {
        let mut mac = HmacSha256::new(prk);
        mac.update(&t[..t_len]);
        mac.update(info);
        mac.update(&[counter]);
        t = mac.finalize();
        t_len = DIGEST_LEN;
        let take = (okm.len() - written).min(DIGEST_LEN);
        okm[written..written + take].copy_from_slice(&t[..take]);
        written += take;
        counter = counter.wrapping_add(1);
    }
    Ok(())
}

/// HKDF-Extract with HMAC-SHA-384. An empty `salt` is treated as
/// `HashLen` zero bytes per RFC 5869 §2.2.
pub fn hkdf_extract_sha384(salt: &[u8], ikm: &[u8]) -> [u8; SHA384_DIGEST_LEN] {
    let zero = [0u8; SHA384_DIGEST_LEN];
    let salt = if salt.is_empty() { &zero[..] } else { salt };
    let mut mac = HmacSha384::new(salt);
    mac.update(ikm);
    mac.finalize()
}

/// HKDF-Expand with HMAC-SHA-384, filling all of `okm`.
pub fn hkdf_expand_sha384(
    prk: &[u8; SHA384_DIGEST_LEN],
    info: &[u8],
    okm: &mut [u8],
) -> Result<(), HkdfLengthError> {
    if okm.len() > 255 * SHA384_DIGEST_LEN {
        return Err(HkdfLengthError);
    }
    let mut t = [0u8; SHA384_DIGEST_LEN];
    let mut t_len = 0usize;
    let mut counter = 1u8;
    let mut written = 0usize;
    while written < okm.len() {
        let mut mac = HmacSha384::new(prk);
        mac.update(&t[..t_len]);
        mac.update(info);
        mac.update(&[counter]);
        t = mac.finalize();
        t_len = SHA384_DIGEST_LEN;
        let take = (okm.len() - written).min(SHA384_DIGEST_LEN);
        okm[written..written + take].copy_from_slice(&t[..take]);
        written += take;
        counter = counter.wrapping_add(1);
    }
    Ok(())
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

    // RFC 5869 A.1 (test case 1) inputs. SHA-256 expected values are
    // the RFC's published vectors; SHA-384 values were generated from
    // the same inputs and cross-validated against two independent
    // oracles (Python stdlib HKDF construction and OpenSSL 3.0
    // `openssl kdf ... HKDF`).
    const IKM: [u8; 22] = [0x0b; 22];
    const SALT: [u8; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x0a, 0x0b, 0x0c];
    const INFO: [u8; 10] = [0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9];

    #[test]
    fn rfc5869_tc1_sha256() {
        let prk = hkdf_extract_sha256(&SALT, &IKM);
        assert_eq!(
            hex(&prk),
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
        );
        let mut okm = [0u8; 42];
        hkdf_expand_sha256(&prk, &INFO, &mut okm).unwrap();
        assert_eq!(
            hex(&okm),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf\
             34007208d5b887185865"
        );
    }

    #[test]
    fn rfc5869_tc1_inputs_sha384() {
        let prk = hkdf_extract_sha384(&SALT, &IKM);
        assert_eq!(
            hex(&prk),
            "704b39990779ce1dc548052c7dc39f303570dd13fb39f7acc564680bef80e8de\
             c70ee9a7e1f3e293ef68eceb072a5ade"
        );
        let mut okm = [0u8; 42];
        hkdf_expand_sha384(&prk, &INFO, &mut okm).unwrap();
        assert_eq!(
            hex(&okm),
            "9b5097a86038b805309076a44b3a9f38063e25b516dcbf369f394cfab43685f7\
             48b6457763e4f0204fc5"
        );
    }

    #[test]
    fn empty_salt_means_zero_block() {
        // RFC 5869 §2.2: salt defaults to HashLen zero bytes.
        let explicit256 = hkdf_extract_sha256(&[0u8; DIGEST_LEN], b"ikm");
        assert_eq!(hkdf_extract_sha256(&[], b"ikm"), explicit256);
        let explicit384 = hkdf_extract_sha384(&[0u8; SHA384_DIGEST_LEN], b"ikm");
        assert_eq!(hkdf_extract_sha384(&[], b"ikm"), explicit384);
    }

    #[test]
    fn expand_multi_block_and_caps() {
        // 100-byte OKM spans 4 SHA-256 blocks; check the chained-T
        // path by comparing a prefix-read with a full read.
        let prk = hkdf_extract_sha256(&SALT, &IKM);
        let mut long = [0u8; 100];
        hkdf_expand_sha256(&prk, &INFO, &mut long).unwrap();
        let mut short = [0u8; 42];
        hkdf_expand_sha256(&prk, &INFO, &mut short).unwrap();
        assert_eq!(&long[..42], &short[..]);

        // RFC 5869 §2.3 cap: 255 * HashLen.
        let mut too_big = alloc::vec![0u8; 255 * DIGEST_LEN + 1];
        assert_eq!(
            hkdf_expand_sha256(&prk, &INFO, &mut too_big),
            Err(HkdfLengthError)
        );
        let mut max_ok = alloc::vec![0u8; 255 * DIGEST_LEN];
        hkdf_expand_sha256(&prk, &INFO, &mut max_ok).unwrap();
    }

    #[test]
    fn zero_length_okm_is_noop() {
        let prk = hkdf_extract_sha256(&SALT, &IKM);
        let mut empty: [u8; 0] = [];
        hkdf_expand_sha256(&prk, &INFO, &mut empty).unwrap();
        let prk384 = hkdf_extract_sha384(&SALT, &IKM);
        hkdf_expand_sha384(&prk384, &INFO, &mut empty).unwrap();
    }
}
