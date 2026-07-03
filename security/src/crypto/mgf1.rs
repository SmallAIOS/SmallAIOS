// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MGF1 mask generation (RFC 8017 Appendix B.2.1) and the hash selector
//! shared by RSA-PSS verification.
//!
//! One MGF1 body serves SHA-256/384/512 via [`PssHash`]; within a single
//! PSS verification the mask hash always matches the message-digest hash.

extern crate alloc;

use alloc::vec::Vec;

use crate::sha2::{sha256, sha384, sha512};

/// The hash function backing an RSA-PSS scheme. The three variants map to
/// the TLS 1.3 `rsa_pss_rsae_sha256/384/512` signature schemes; SHA-1 is
/// deliberately absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PssHash {
    Sha256,
    Sha384,
    Sha512,
}

impl PssHash {
    /// Digest length in bytes (`hLen`).
    pub fn hlen(self) -> usize {
        match self {
            PssHash::Sha256 => 32,
            PssHash::Sha384 => 48,
            PssHash::Sha512 => 64,
        }
    }

    /// One-shot digest of `data`, returned as a byte vector so a single
    /// call site handles all three output widths.
    pub fn digest(self, data: &[u8]) -> Vec<u8> {
        match self {
            PssHash::Sha256 => sha256(data).to_vec(),
            PssHash::Sha384 => sha384(data).to_vec(),
            PssHash::Sha512 => sha512(data).to_vec(),
        }
    }

    /// Digest of the concatenation `a || b` without allocating the joined
    /// input twice (used for `H'` over the padded PSS message).
    pub fn digest_concat(self, parts: &[&[u8]]) -> Vec<u8> {
        let mut buf = Vec::new();
        for p in parts {
            buf.extend_from_slice(p);
        }
        self.digest(&buf)
    }
}

/// MGF1: generate a `mask_len`-byte mask from `seed` using `hash`.
///
/// Per RFC 8017 B.2.1, the mask is the concatenation of
/// `hash(seed || C)` for a 4-byte big-endian counter `C = 0, 1, 2, …`,
/// truncated to exactly `mask_len` bytes.
pub fn mgf1(seed: &[u8], mask_len: usize, hash: PssHash) -> Vec<u8> {
    let hlen = hash.hlen();
    let mut mask = Vec::with_capacity(mask_len);
    let mut counter: u32 = 0;
    while mask.len() < mask_len {
        let block = hash.digest_concat(&[seed, &counter.to_be_bytes()]);
        let take = (mask_len - mask.len()).min(hlen);
        mask.extend_from_slice(&block[..take]);
        counter += 1;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hlen_matches_digest_width() {
        for h in [PssHash::Sha256, PssHash::Sha384, PssHash::Sha512] {
            assert_eq!(h.hlen(), h.digest(b"x").len());
        }
    }

    #[test]
    fn mgf1_follows_counter_construction() {
        // Block 0 of the mask must equal hash(seed || 0x00000000), block 1
        // hash(seed || 0x00000001), etc. — RFC 8017 B.2.1.
        let seed = b"seed material";
        for h in [PssHash::Sha256, PssHash::Sha384, PssHash::Sha512] {
            let hlen = h.hlen();
            let mask = mgf1(seed, hlen * 2 + 5, h);
            let block0 = h.digest_concat(&[seed, &0u32.to_be_bytes()]);
            let block1 = h.digest_concat(&[seed, &1u32.to_be_bytes()]);
            let block2 = h.digest_concat(&[seed, &2u32.to_be_bytes()]);
            assert_eq!(&mask[..hlen], &block0[..]);
            assert_eq!(&mask[hlen..2 * hlen], &block1[..]);
            // Final partial block is truncated to exactly the request.
            assert_eq!(&mask[2 * hlen..], &block2[..5]);
            assert_eq!(mask.len(), hlen * 2 + 5);
        }
    }

    #[test]
    fn mgf1_single_block_is_one_hash() {
        let seed = b"abc";
        let h = PssHash::Sha256;
        let mask = mgf1(seed, 32, h);
        assert_eq!(mask, h.digest_concat(&[seed, &0u32.to_be_bytes()]));
    }

    #[test]
    fn mgf1_shorter_than_hlen_truncates() {
        let mask = mgf1(b"seed", 5, PssHash::Sha384);
        assert_eq!(mask.len(), 5);
        let full = PssHash::Sha384.digest_concat(&[b"seed", &0u32.to_be_bytes()]);
        assert_eq!(mask, &full[..5]);
    }
}
