// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SHA-256 (FIPS 180-4) — clean-room `#![no_std]` implementation.
//!
//! ## Why ship SHA-256?
//!
//! SmallAIOS's native crypto stack is PQ-aligned (SHA-3-256, ML-DSA-65,
//! ML-KEM-768). SHA-256 is *not* used for any first-party SmallAIOS
//! primitive. It is required strictly for **interop** with external
//! systems whose wire format the SmallAIOS image must speak verbatim:
//!
//! - immudb's Merkle-tree commit log uses SHA-256 over its leaf and
//!   internal nodes ([verifiable-audit-log-v1](../../../openspec/changes/verifiable-audit-log-v1)).
//! - Standard TLS 1.3 certificate fingerprints and SubjectPublicKeyInfo
//!   hashes use SHA-256.
//! - Transparency-log ecosystems (Sigstore Rekor, RFC 6962 CT) all hash
//!   with SHA-256.
//!
//! ## Threat model
//!
//! SHA-256 is collision-resistant and preimage-resistant under current
//! cryptanalysis. It is **not** quantum-resistant for preimage
//! attacks (Grover's algorithm halves the security level), so we do
//! not use it where post-quantum strength matters. For interop
//! purposes it is the right primitive.
//!
//! ## API
//!
//! ```ignore
//! use smallaios_security::sha2::Sha256;
//! let mut h = Sha256::new();
//! h.update(b"abc");
//! let digest: [u8; 32] = h.finalize();
//! // or one-shot:
//! let digest = smallaios_security::sha2::sha256(b"abc");
//! ```

#![allow(clippy::many_single_char_names)]

/// SHA-256 digest length in bytes.
pub const DIGEST_LEN: usize = 32;

/// SHA-256 block length in bytes.
pub const BLOCK_LEN: usize = 64;

// FIPS 180-4 §5.3.3 initial hash value H(0).
// lgtm[rust/hard-coded-cryptographic-value] — FIPS 180-4 IV, public constant.
const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

// FIPS 180-4 §4.2.2 round constants K[0..63].
// lgtm[rust/hard-coded-cryptographic-value] — FIPS 180-4 constants, public.
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Incremental SHA-256 hasher.
#[derive(Clone, Debug)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; BLOCK_LEN],
    buffered: usize,
    total_bits: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Create a fresh hasher.
    pub const fn new() -> Self {
        Self {
            state: H0,
            buffer: [0u8; BLOCK_LEN],
            buffered: 0,
            total_bits: 0,
        }
    }

    /// Absorb bytes into the hasher.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bits = self
            .total_bits
            .wrapping_add((data.len() as u64).wrapping_mul(8));
        if self.buffered > 0 {
            let take = (BLOCK_LEN - self.buffered).min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == BLOCK_LEN {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while data.len() >= BLOCK_LEN {
            let mut block = [0u8; BLOCK_LEN];
            block.copy_from_slice(&data[..BLOCK_LEN]);
            self.compress(&block);
            data = &data[BLOCK_LEN..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    /// Finalize the hash and return the 32-byte digest.
    pub fn finalize(mut self) -> [u8; DIGEST_LEN] {
        // Pad: 0x80 byte, zeros to ≡ 56 (mod 64), then 8-byte BE length.
        let bits = self.total_bits;
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > BLOCK_LEN - 8 {
            for b in &mut self.buffer[self.buffered..] {
                *b = 0;
            }
            let block = self.buffer;
            self.compress(&block);
            self.buffered = 0;
        }
        for b in &mut self.buffer[self.buffered..BLOCK_LEN - 8] {
            *b = 0;
        }
        self.buffer[BLOCK_LEN - 8..].copy_from_slice(&bits.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);

        let mut out = [0u8; DIGEST_LEN];
        for (i, w) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; BLOCK_LEN]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

/// One-shot SHA-256 over `data`.
pub fn sha256(data: &[u8]) -> [u8; DIGEST_LEN] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
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

    // NIST FIPS 180-4 published test vectors.
    #[test]
    fn nist_kat_empty() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn nist_kat_abc() {
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn nist_kat_two_block_message() {
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            hex(&sha256(msg)),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "hashing 1 MB takes >5 min under Miri's interpreter; UB coverage comes from the small-input KATs / native runs"
    )]
    fn nist_kat_one_million_a() {
        let mut h = Sha256::new();
        let chunk = alloc::vec![b'a'; 1024];
        for _ in 0..1000 {
            h.update(&chunk[..1000]);
        }
        // 1000 iterations × 1000 bytes = 1,000,000.
        assert_eq!(
            hex(&h.finalize()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn incremental_equals_oneshot() {
        let msg = b"The quick brown fox jumps over the lazy dog";
        let oneshot = sha256(msg);
        let mut h = Sha256::new();
        for chunk in msg.chunks(7) {
            h.update(chunk);
        }
        assert_eq!(oneshot, h.finalize());
    }

    #[test]
    fn empty_chunks_are_noops() {
        let mut h = Sha256::new();
        h.update(b"");
        h.update(b"abc");
        h.update(b"");
        assert_eq!(
            hex(&h.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
