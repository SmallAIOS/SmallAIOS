// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SHA-256 and SHA-384 (FIPS 180-4) — clean-room `#![no_std]`
//! implementations.
//!
//! ## Why ship SHA-2?
//!
//! SmallAIOS's native crypto stack is PQ-aligned (SHA-3-256, ML-DSA-65,
//! ML-KEM-768). SHA-2 is *not* used for any first-party SmallAIOS
//! primitive. It is required strictly for **interop** with external
//! systems whose wire format the SmallAIOS image must speak verbatim:
//!
//! - immudb's Merkle-tree commit log uses SHA-256 over its leaf and
//!   internal nodes ([verifiable-audit-log-v1](../../../openspec/changes/verifiable-audit-log-v1)).
//! - Standard TLS 1.3 certificate fingerprints and SubjectPublicKeyInfo
//!   hashes use SHA-256.
//! - Transparency-log ecosystems (Sigstore Rekor, RFC 6962 CT) all hash
//!   with SHA-256.
//! - The TLS 1.3 cipher suite `TLS_AES_256_GCM_SHA384`
//!   ([tls-tcp-client-v1](../../../openspec/changes/tls-tcp-client-v1))
//!   keys its HKDF transcript schedule with SHA-384.
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

// ─── SHA-384 ─────────────────────────────────────────────────────────────────

/// SHA-384 digest length in bytes.
pub const SHA384_DIGEST_LEN: usize = 48;

/// SHA-384 block length in bytes (the SHA-512 core block).
pub const SHA384_BLOCK_LEN: usize = 128;

// FIPS 180-4 §5.3.4 initial hash value H(0) for SHA-384.
// lgtm[rust/hard-coded-cryptographic-value] — FIPS 180-4 IV, public constant.
const H0_384: [u64; 8] = [
    0xcbbb_9d5d_c105_9ed8,
    0x629a_292a_367c_d507,
    0x9159_015a_3070_dd17,
    0x152f_ecd8_f70e_5939,
    0x6733_2667_ffc0_0b31,
    0x8eb4_4a87_6858_1511,
    0xdb0c_2e0d_64f9_8fa7,
    0x47b5_481d_befa_4fa4,
];

// FIPS 180-4 §4.2.3 round constants K[0..79] (shared SHA-512 core).
// lgtm[rust/hard-coded-cryptographic-value] — FIPS 180-4 constants, public.
const K512: [u64; 80] = [
    0x428a_2f98_d728_ae22,
    0x7137_4491_23ef_65cd,
    0xb5c0_fbcf_ec4d_3b2f,
    0xe9b5_dba5_8189_dbbc,
    0x3956_c25b_f348_b538,
    0x59f1_11f1_b605_d019,
    0x923f_82a4_af19_4f9b,
    0xab1c_5ed5_da6d_8118,
    0xd807_aa98_a303_0242,
    0x1283_5b01_4570_6fbe,
    0x2431_85be_4ee4_b28c,
    0x550c_7dc3_d5ff_b4e2,
    0x72be_5d74_f27b_896f,
    0x80de_b1fe_3b16_96b1,
    0x9bdc_06a7_25c7_1235,
    0xc19b_f174_cf69_2694,
    0xe49b_69c1_9ef1_4ad2,
    0xefbe_4786_384f_25e3,
    0x0fc1_9dc6_8b8c_d5b5,
    0x240c_a1cc_77ac_9c65,
    0x2de9_2c6f_592b_0275,
    0x4a74_84aa_6ea6_e483,
    0x5cb0_a9dc_bd41_fbd4,
    0x76f9_88da_8311_53b5,
    0x983e_5152_ee66_dfab,
    0xa831_c66d_2db4_3210,
    0xb003_27c8_98fb_213f,
    0xbf59_7fc7_beef_0ee4,
    0xc6e0_0bf3_3da8_8fc2,
    0xd5a7_9147_930a_a725,
    0x06ca_6351_e003_826f,
    0x1429_2967_0a0e_6e70,
    0x27b7_0a85_46d2_2ffc,
    0x2e1b_2138_5c26_c926,
    0x4d2c_6dfc_5ac4_2aed,
    0x5338_0d13_9d95_b3df,
    0x650a_7354_8baf_63de,
    0x766a_0abb_3c77_b2a8,
    0x81c2_c92e_47ed_aee6,
    0x9272_2c85_1482_353b,
    0xa2bf_e8a1_4cf1_0364,
    0xa81a_664b_bc42_3001,
    0xc24b_8b70_d0f8_9791,
    0xc76c_51a3_0654_be30,
    0xd192_e819_d6ef_5218,
    0xd699_0624_5565_a910,
    0xf40e_3585_5771_202a,
    0x106a_a070_32bb_d1b8,
    0x19a4_c116_b8d2_d0c8,
    0x1e37_6c08_5141_ab53,
    0x2748_774c_df8e_eb99,
    0x34b0_bcb5_e19b_48a8,
    0x391c_0cb3_c5c9_5a63,
    0x4ed8_aa4a_e341_8acb,
    0x5b9c_ca4f_7763_e373,
    0x682e_6ff3_d6b2_b8a3,
    0x748f_82ee_5def_b2fc,
    0x78a5_636f_4317_2f60,
    0x84c8_7814_a1f0_ab72,
    0x8cc7_0208_1a64_39ec,
    0x90be_fffa_2363_1e28,
    0xa450_6ceb_de82_bde9,
    0xbef9_a3f7_b2c6_7915,
    0xc671_78f2_e372_532b,
    0xca27_3ece_ea26_619c,
    0xd186_b8c7_21c0_c207,
    0xeada_7dd6_cde0_eb1e,
    0xf57d_4f7f_ee6e_d178,
    0x06f0_67aa_7217_6fba,
    0x0a63_7dc5_a2c8_98a6,
    0x113f_9804_bef9_0dae,
    0x1b71_0b35_131c_471b,
    0x28db_77f5_2304_7d84,
    0x32ca_ab7b_40c7_2493,
    0x3c9e_be0a_15c9_bebc,
    0x431d_67c4_9c10_0d4c,
    0x4cc5_d4be_cb3e_42b6,
    0x597f_299c_fc65_7e2a,
    0x5fcb_6fab_3ad6_faec,
    0x6c44_198c_4a47_5817,
];

/// Incremental SHA-384 hasher (SHA-512 core, truncated output).
#[derive(Clone, Debug)]
pub struct Sha384 {
    state: [u64; 8],
    buffer: [u8; SHA384_BLOCK_LEN],
    buffered: usize,
    total_bits: u128,
}

impl Default for Sha384 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha384 {
    /// Create a fresh hasher.
    pub const fn new() -> Self {
        Self {
            state: H0_384,
            buffer: [0u8; SHA384_BLOCK_LEN],
            buffered: 0,
            total_bits: 0,
        }
    }

    /// Absorb bytes into the hasher.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bits = self
            .total_bits
            .wrapping_add((data.len() as u128).wrapping_mul(8));
        if self.buffered > 0 {
            let take = (SHA384_BLOCK_LEN - self.buffered).min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == SHA384_BLOCK_LEN {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while data.len() >= SHA384_BLOCK_LEN {
            let mut block = [0u8; SHA384_BLOCK_LEN];
            block.copy_from_slice(&data[..SHA384_BLOCK_LEN]);
            self.compress(&block);
            data = &data[SHA384_BLOCK_LEN..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    /// Finalize the hash and return the 48-byte digest.
    pub fn finalize(mut self) -> [u8; SHA384_DIGEST_LEN] {
        // Pad: 0x80 byte, zeros to ≡ 112 (mod 128), then 16-byte BE length.
        let bits = self.total_bits;
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > SHA384_BLOCK_LEN - 16 {
            for b in &mut self.buffer[self.buffered..] {
                *b = 0;
            }
            let block = self.buffer;
            self.compress(&block);
            self.buffered = 0;
        }
        for b in &mut self.buffer[self.buffered..SHA384_BLOCK_LEN - 16] {
            *b = 0;
        }
        self.buffer[SHA384_BLOCK_LEN - 16..].copy_from_slice(&bits.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);

        // SHA-384 truncates the 512-bit state to the first 6 words.
        let mut out = [0u8; SHA384_DIGEST_LEN];
        for (i, w) in self.state.iter().take(6).enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; SHA384_BLOCK_LEN]) {
        let mut w = [0u64; 80];
        for i in 0..16 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&block[i * 8..i * 8 + 8]);
            w[i] = u64::from_be_bytes(bytes);
        }
        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..80 {
            let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K512[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
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

/// One-shot SHA-384 over `data`.
pub fn sha384(data: &[u8]) -> [u8; SHA384_DIGEST_LEN] {
    let mut h = Sha384::new();
    h.update(data);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_util::hex;

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
    fn nist_kat_one_million_a() {
        let mut h = Sha256::new();
        let chunk = alloc::vec![b'a'; 1000];
        for _ in 0..1000 {
            h.update(&chunk);
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

    // NIST FIPS 180-4 published SHA-384 test vectors.
    #[test]
    fn sha384_kat_empty() {
        assert_eq!(
            hex(&sha384(b"")),
            "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da\
             274edebfe76f65fbd51ad2f14898b95b"
        );
    }

    #[test]
    fn sha384_kat_abc() {
        assert_eq!(
            hex(&sha384(b"abc")),
            "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed\
             8086072ba1e7cc2358baeca134c825a7"
        );
    }

    #[test]
    fn sha384_kat_two_block_message() {
        let msg = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn\
hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
        assert_eq!(
            hex(&sha384(msg)),
            "09330c33f71147e83d192fc782cd1b4753111b173b3b05d22fa08086e3b0f712\
             fcc7c71a557e2db966c3e9fa91746039"
        );
    }

    #[test]
    fn sha384_kat_one_million_a() {
        let mut h = Sha384::new();
        let chunk = alloc::vec![b'a'; 1000];
        for _ in 0..1000 {
            h.update(&chunk);
        }
        assert_eq!(
            hex(&h.finalize()),
            "9d0e1809716474cb086e834e310a4a1ced149e9c00f248527972cec5704c2a5b\
             07b8b3dc38ecc4ebae97ddd87f3d8985"
        );
    }

    #[test]
    fn sha384_incremental_equals_oneshot() {
        let msg = b"The quick brown fox jumps over the lazy dog";
        let oneshot = sha384(msg);
        assert_eq!(
            hex(&oneshot),
            "ca737f1014a48f4c0b6dd43cb177b0afd9e5169367544c494011e3317dbf9a50\
             9cb1e5dc1e85a941bbee3d7f2afbc9b1"
        );
        let mut h = Sha384::new();
        for chunk in msg.chunks(13) {
            h.update(chunk);
        }
        assert_eq!(oneshot, h.finalize());
    }

    #[test]
    fn sha384_default_equals_new() {
        let mut a = Sha384::default();
        let mut b = Sha384::new();
        a.update(b"abc");
        b.update(b"abc");
        assert_eq!(a.finalize(), b.finalize());
        let mut c = Sha256::default();
        c.update(b"abc");
        assert_eq!(c.finalize(), sha256(b"abc"));
    }

    #[test]
    fn sha384_block_boundary_lengths() {
        // Exercise the padding paths around the 128-byte block
        // boundary (111/112/127/128/129 bytes) against the
        // incremental path.
        for len in [111usize, 112, 127, 128, 129, 240, 256] {
            let msg = alloc::vec![0x5au8; len];
            let oneshot = sha384(&msg);
            let mut h = Sha384::new();
            for chunk in msg.chunks(7) {
                h.update(chunk);
            }
            assert_eq!(oneshot, h.finalize(), "len={len}");
        }
    }
}
