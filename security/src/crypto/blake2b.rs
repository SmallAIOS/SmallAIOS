// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Blake2b — cryptographic hash function (RFC 7693).
//!
//! Clean-room `#![no_std]` implementation of Blake2b. Variable digest length
//! (1..=64 bytes), optional keyed mode, and full-parameter-block construction
//! used by Argon2id (RFC 9106 §3.3, §3.4).
//!
//! The implementation follows RFC 7693 §3 closely. Each call to [`Blake2b`]
//! processes input in 128-byte blocks; the final block is padded with zeros
//! and finalized with the `f` flag set per the spec.
//!
//! # Examples
//!
//! Plain hash:
//!
//! ```ignore
//! let mut h = Blake2b::new(64);
//! h.update(b"abc");
//! let digest = h.finalize();
//! assert_eq!(digest.len(), 64);
//! ```
//!
//! Keyed hash:
//!
//! ```ignore
//! let mut h = Blake2b::new_keyed(b"my-key", 32);
//! h.update(b"message");
//! let mac = h.finalize();
//! ```
//!
//! Full parameter block (used by Argon2id's variable-length hash `H'`):
//!
//! ```ignore
//! let mut h = Blake2b::new_with_params(&Blake2bParams {
//!     digest_length: 64,
//!     key_length: 0,
//!     fanout: 1,
//!     depth: 1,
//!     ..Default::default()
//! });
//! ```
//!
//! # References
//!
//! - RFC 7693: The BLAKE2 Cryptographic Hash and Message Authentication Code
//! - <https://www.blake2.net/>

#![allow(unused)]

use core::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum Blake2b digest length in bytes.
pub const BLAKE2B_OUT_MAX: usize = 64;

/// Blake2b block size in bytes.
pub const BLAKE2B_BLOCK_BYTES: usize = 128;

/// Maximum keyed-mode key length in bytes.
pub const BLAKE2B_KEY_MAX: usize = 64;

/// Initial value vector — first 64 bits of the fractional parts of the
/// square roots of the first eight primes (2, 3, 5, 7, 11, 13, 17, 19).
const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

/// SIGMA permutations for the message-word selection in `G`. RFC 7693 §2.7.
const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/// Number of compression rounds for Blake2b.
const ROUNDS: usize = 12;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors raised by Blake2b configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blake2bError {
    /// Output length out of the supported range (1..=64).
    InvalidOutputLength,
    /// Key length out of the supported range (0..=64).
    InvalidKeyLength,
}

impl fmt::Display for Blake2bError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutputLength => write!(f, "blake2b: output length must be 1..=64"),
            Self::InvalidKeyLength => write!(f, "blake2b: key length must be 0..=64"),
        }
    }
}

// ─── Parameter block ─────────────────────────────────────────────────────────

/// Blake2b parameter block (RFC 7693 §2.5).
///
/// 64 bytes total. Plain mode uses `digest_length`/`key_length` only;
/// Argon2id's `H'` uses the full structure to bind tree-mode parameters.
#[derive(Debug, Clone, Copy)]
pub struct Blake2bParams {
    pub digest_length: u8,  // 0
    pub key_length: u8,     // 1
    pub fanout: u8,         // 2
    pub depth: u8,          // 3
    pub leaf_length: u32,   // 4..8
    pub node_offset: u64,   // 8..16
    pub node_depth: u8,     // 16
    pub inner_length: u8,   // 17
    pub reserved: [u8; 14], // 18..32
    pub salt: [u8; 16],     // 32..48
    pub personal: [u8; 16], // 48..64
}

impl Default for Blake2bParams {
    fn default() -> Self {
        Self {
            digest_length: 64,
            key_length: 0,
            fanout: 1,
            depth: 1,
            leaf_length: 0,
            node_offset: 0,
            node_depth: 0,
            inner_length: 0,
            reserved: [0u8; 14],
            salt: [0u8; 16],
            personal: [0u8; 16],
        }
    }
}

impl Blake2bParams {
    /// Serialize the parameter block to its 64-byte little-endian form so the
    /// state can be initialized as `IV ^ params` per RFC 7693 §3.2.
    fn as_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0] = self.digest_length;
        out[1] = self.key_length;
        out[2] = self.fanout;
        out[3] = self.depth;
        out[4..8].copy_from_slice(&self.leaf_length.to_le_bytes());
        out[8..16].copy_from_slice(&self.node_offset.to_le_bytes());
        out[16] = self.node_depth;
        out[17] = self.inner_length;
        out[18..32].copy_from_slice(&self.reserved);
        out[32..48].copy_from_slice(&self.salt);
        out[48..64].copy_from_slice(&self.personal);
        out
    }
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Blake2b incremental hasher.
#[derive(Clone)]
pub struct Blake2b {
    /// Chaining value `h[0..8]` per RFC 7693.
    h: [u64; 8],
    /// 128-bit message-byte counter `t`.
    t: [u64; 2],
    /// Current message buffer (one block at most).
    buf: [u8; BLAKE2B_BLOCK_BYTES],
    /// Bytes currently in `buf`.
    buflen: usize,
    /// Configured output length in bytes.
    out_len: usize,
}

impl Blake2b {
    /// Create a Blake2b hasher with `out_len` output bytes (1..=64).
    ///
    /// Equivalent to [`Blake2b::new_with_params`] with the default parameter
    /// block and `digest_length = out_len`.
    ///
    /// # Panics
    ///
    /// Panics if `out_len` is 0 or > 64. Use [`Blake2b::try_new`] for a
    /// non-panicking variant.
    pub fn new(out_len: usize) -> Self {
        Self::try_new(out_len).expect("blake2b: out_len must be 1..=64")
    }

    /// Fallible variant of [`Blake2b::new`].
    pub fn try_new(out_len: usize) -> Result<Self, Blake2bError> {
        if out_len == 0 || out_len > BLAKE2B_OUT_MAX {
            return Err(Blake2bError::InvalidOutputLength);
        }
        let params = Blake2bParams {
            digest_length: out_len as u8,
            ..Blake2bParams::default()
        };
        Ok(Self::with_params(&params, &[]))
    }

    /// Keyed Blake2b (MAC mode) with a 0..=64-byte key.
    ///
    /// # Panics
    ///
    /// Panics if `out_len` or `key.len()` are out of range. Use
    /// [`Blake2b::try_new_keyed`] for a non-panicking variant.
    pub fn new_keyed(key: &[u8], out_len: usize) -> Self {
        Self::try_new_keyed(key, out_len).expect("blake2b: invalid out_len/key length")
    }

    /// Fallible variant of [`Blake2b::new_keyed`].
    pub fn try_new_keyed(key: &[u8], out_len: usize) -> Result<Self, Blake2bError> {
        if out_len == 0 || out_len > BLAKE2B_OUT_MAX {
            return Err(Blake2bError::InvalidOutputLength);
        }
        if key.len() > BLAKE2B_KEY_MAX {
            return Err(Blake2bError::InvalidKeyLength);
        }
        let params = Blake2bParams {
            digest_length: out_len as u8,
            key_length: key.len() as u8,
            ..Blake2bParams::default()
        };
        Ok(Self::with_params(&params, key))
    }

    /// Construct a Blake2b instance with a fully specified parameter block.
    ///
    /// Caller must pass a key matching `params.key_length` (or `&[]` if
    /// `key_length == 0`). The key is padded to a full block and consumed
    /// before any user input. The non-keyed variants exposed via
    /// [`Blake2b::new_with_params`] forward an empty key.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `params.digest_length` is out of range or
    /// the key length does not match the parameter block.
    pub fn new_with_params(params: &Blake2bParams) -> Self {
        Self::with_params(params, &[])
    }

    /// Construct a Blake2b instance with a parameter block and an explicit
    /// key matching `params.key_length`.
    pub fn new_with_params_and_key(params: &Blake2bParams, key: &[u8]) -> Self {
        Self::with_params(params, key)
    }

    /// Internal constructor used by all public entry points.
    fn with_params(params: &Blake2bParams, key: &[u8]) -> Self {
        debug_assert!(params.digest_length >= 1 && params.digest_length <= BLAKE2B_OUT_MAX as u8);
        debug_assert_eq!(params.key_length as usize, key.len());
        debug_assert!(key.len() <= BLAKE2B_KEY_MAX);

        let pb = params.as_bytes();
        let mut h = IV;
        for i in 0..8 {
            let word = u64::from_le_bytes(pb[i * 8..i * 8 + 8].try_into().unwrap());
            h[i] ^= word;
        }

        let mut state = Self {
            h,
            t: [0u64; 2],
            buf: [0u8; BLAKE2B_BLOCK_BYTES],
            buflen: 0,
            out_len: params.digest_length as usize,
        };

        // Keyed mode: prepend the zero-padded key as the first block.
        if !key.is_empty() {
            let mut block = [0u8; BLAKE2B_BLOCK_BYTES];
            block[..key.len()].copy_from_slice(key);
            state.update(&block);
        }

        state
    }

    /// Absorb `input` into the hash state.
    pub fn update(&mut self, mut input: &[u8]) {
        if input.is_empty() {
            return;
        }

        // Fill the existing buffer first if it has bytes pending; we only
        // compress when the *next* byte arrives, so we leave a full block
        // unprocessed in the buffer.
        if self.buflen > 0 {
            let take = (BLAKE2B_BLOCK_BYTES - self.buflen).min(input.len());
            self.buf[self.buflen..self.buflen + take].copy_from_slice(&input[..take]);
            self.buflen += take;
            input = &input[take..];

            if input.is_empty() {
                return;
            }

            // Buffer must be full at this point. Compress the buffered block
            // and reset.
            debug_assert_eq!(self.buflen, BLAKE2B_BLOCK_BYTES);
            self.t_increment(BLAKE2B_BLOCK_BYTES as u64);
            let block = self.buf;
            self.compress(&block, false);
            self.buflen = 0;
        }

        // Process whole blocks directly out of the input slice. Leave at
        // least one byte over so finalize() always sees the last block.
        while input.len() > BLAKE2B_BLOCK_BYTES {
            self.t_increment(BLAKE2B_BLOCK_BYTES as u64);
            let mut block = [0u8; BLAKE2B_BLOCK_BYTES];
            block.copy_from_slice(&input[..BLAKE2B_BLOCK_BYTES]);
            self.compress(&block, false);
            input = &input[BLAKE2B_BLOCK_BYTES..];
        }

        // Stash the remainder.
        self.buf[..input.len()].copy_from_slice(input);
        self.buflen = input.len();
    }

    /// Finalize and write up to `out_len` bytes (the configured output
    /// length) into `out`. Returns the number of bytes written.
    pub fn finalize_into(mut self, out: &mut [u8]) -> usize {
        let n = self.out_len.min(out.len());
        // Pad the final block with zeros.
        for byte in &mut self.buf[self.buflen..] {
            *byte = 0;
        }
        self.t_increment(self.buflen as u64);
        let block = self.buf;
        self.compress(&block, true);
        // Emit `out_len` bytes of state in little-endian.
        let mut digest = [0u8; BLAKE2B_OUT_MAX];
        for (i, lane) in self.h.iter().enumerate() {
            digest[i * 8..i * 8 + 8].copy_from_slice(&lane.to_le_bytes());
        }
        out[..n].copy_from_slice(&digest[..n]);
        n
    }

    /// Finalize and return the digest as a fixed-size array of length 64.
    /// The valid bytes are the first `self.out_len`; the remaining bytes
    /// are zero-padded.
    pub fn finalize(self) -> [u8; BLAKE2B_OUT_MAX] {
        let out_len = self.out_len;
        let mut out = [0u8; BLAKE2B_OUT_MAX];
        let _ = self.finalize_into(&mut out[..out_len]);
        out
    }

    /// Return only the configured number of digest bytes as
    /// `alloc::vec::Vec<u8>`. Callers that want zero-allocation should use
    /// [`Blake2b::finalize_into`].
    pub fn finalize_vec(self) -> alloc::vec::Vec<u8> {
        let out_len = self.out_len;
        let mut out = alloc::vec![0u8; out_len];
        let _ = self.finalize_into(&mut out);
        out
    }

    /// Return the configured output length (in bytes).
    pub const fn output_length(&self) -> usize {
        self.out_len
    }

    // ─── internals ──────────────────────────────────────────────────────────

    /// Increment the 128-bit message-byte counter `t` by `n`.
    fn t_increment(&mut self, n: u64) {
        let (low, carry) = self.t[0].overflowing_add(n);
        self.t[0] = low;
        if carry {
            self.t[1] = self.t[1].wrapping_add(1);
        }
    }

    /// Compression function `F` (RFC 7693 §3.2). `final_block` sets the
    /// `f0` finalization flag.
    fn compress(&mut self, block: &[u8; BLAKE2B_BLOCK_BYTES], final_block: bool) {
        // Decode 16 little-endian 64-bit message words.
        let mut m = [0u64; 16];
        for i in 0..16 {
            m[i] = u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
        }

        // Initialize the working state v[0..16] = h[0..8] || IV[0..8].
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(&self.h);
        v[8..].copy_from_slice(&IV);

        // Mix in the byte counter.
        v[12] ^= self.t[0];
        v[13] ^= self.t[1];

        // Mix in the finalization flag for the last block.
        if final_block {
            v[14] = !v[14];
        }

        // 12 mixing rounds.
        for s in SIGMA.iter().take(ROUNDS) {
            mix(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            mix(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            mix(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            mix(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);

            mix(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            mix(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            mix(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            mix(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }

        // Update chaining value: h[i] = h[i] ^ v[i] ^ v[i+8].
        for i in 0..8 {
            self.h[i] ^= v[i] ^ v[i + 8];
        }
    }
}

/// Blake2b mixing function `G` (RFC 7693 §3.1).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn mix(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);

    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

// ─── Convenience top-level API ──────────────────────────────────────────────

/// One-shot Blake2b: `out_len` digest bytes of `input`.
pub fn blake2b(input: &[u8], out_len: usize) -> [u8; BLAKE2B_OUT_MAX] {
    let mut h = Blake2b::new(out_len);
    h.update(input);
    h.finalize()
}

/// One-shot keyed Blake2b: `out_len` digest bytes of `input` with `key`.
pub fn blake2b_keyed(key: &[u8], input: &[u8], out_len: usize) -> [u8; BLAKE2B_OUT_MAX] {
    let mut h = Blake2b::new_keyed(key, out_len);
    h.update(input);
    h.finalize()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// RFC 7693 Appendix A — Blake2b-512 of "abc".
    #[test]
    fn rfc7693_appendix_a_abc() {
        let expected: [u8; 64] = [
            0xBA, 0x80, 0xA5, 0x3F, 0x98, 0x1C, 0x4D, 0x0D, 0x6A, 0x27, 0x97, 0xB6, 0x9F, 0x12,
            0xF6, 0xE9, 0x4C, 0x21, 0x2F, 0x14, 0x68, 0x5A, 0xC4, 0xB7, 0x4B, 0x12, 0xBB, 0x6F,
            0xDB, 0xFF, 0xA2, 0xD1, 0x7D, 0x87, 0xC5, 0x39, 0x2A, 0xAB, 0x79, 0x2D, 0xC2, 0x52,
            0xD5, 0xDE, 0x45, 0x33, 0xCC, 0x95, 0x18, 0xD3, 0x8A, 0xA8, 0xDB, 0xF1, 0x92, 0x5A,
            0xB9, 0x23, 0x86, 0xED, 0xD4, 0x00, 0x99, 0x23,
        ];
        let actual = blake2b(b"abc", 64);
        assert_eq!(&actual[..64], &expected[..]);
    }

    /// Empty-string vector — Blake2b-512 of "".
    #[test]
    fn empty_input_512() {
        let expected: [u8; 64] = [
            0x78, 0x6A, 0x02, 0xF7, 0x42, 0x01, 0x59, 0x03, 0xC6, 0xC6, 0xFD, 0x85, 0x25, 0x52,
            0xD2, 0x72, 0x91, 0x2F, 0x47, 0x40, 0xE1, 0x58, 0x47, 0x61, 0x8A, 0x86, 0xE2, 0x17,
            0xF7, 0x1F, 0x54, 0x19, 0xD2, 0x5E, 0x10, 0x31, 0xAF, 0xEE, 0x58, 0x53, 0x13, 0x89,
            0x64, 0x44, 0x93, 0x4E, 0xB0, 0x4B, 0x90, 0x3A, 0x68, 0x5B, 0x14, 0x48, 0xB7, 0x55,
            0xD5, 0x6F, 0x70, 0x1A, 0xFE, 0x9B, 0xE2, 0xCE,
        ];
        let actual = blake2b(b"", 64);
        assert_eq!(&actual[..64], &expected[..]);
    }

    /// Variable output length — request 32 bytes vs 64 bytes.
    /// The parameter block changes (digest_length differs), so the digests
    /// are not simple truncations of one another.
    #[test]
    fn variable_output_length_changes_digest() {
        let h32 = blake2b(b"abc", 32);
        let h64 = blake2b(b"abc", 64);
        assert_ne!(&h32[..32], &h64[..32]);
    }

    /// Incremental update should match one-shot for a non-trivial input.
    #[test]
    fn incremental_matches_one_shot() {
        let big: Vec<u8> = (0..500u32).map(|i| (i & 0xFF) as u8).collect();
        let one_shot = blake2b(&big, 64);
        let mut h = Blake2b::new(64);
        for chunk in big.chunks(37) {
            h.update(chunk);
        }
        let inc = h.finalize();
        assert_eq!(&one_shot[..], &inc[..]);
    }

    /// Block-boundary stress: input lengths around block multiples must match.
    #[test]
    fn block_boundary_stress() {
        let buf: Vec<u8> = (0..512u32).map(|i| (i & 0xFF) as u8).collect();
        for n in [0usize, 1, 64, 127, 128, 129, 200, 256, 257, 384, 511, 512] {
            let one_shot = blake2b(&buf[..n], 64);
            let mut h = Blake2b::new(64);
            // Update in 1-byte increments to exercise the buffer code.
            for b in &buf[..n] {
                h.update(core::slice::from_ref(b));
            }
            let inc = h.finalize();
            assert_eq!(&one_shot[..], &inc[..], "mismatch at n={}", n);
        }
    }

    /// Out-of-range output length is rejected.
    #[test]
    fn out_of_range_output_length_rejected() {
        assert!(Blake2b::try_new(0).is_err());
        assert!(Blake2b::try_new(65).is_err());
    }

    /// Out-of-range key length is rejected.
    #[test]
    fn out_of_range_key_length_rejected() {
        let big_key = [0u8; 65];
        assert!(Blake2b::try_new_keyed(&big_key, 32).is_err());
    }

    /// Validation oracle: when the dev-only `blake2` crate is available,
    /// compare against it for a sweep of message lengths and output sizes.
    #[test]
    fn blake2_oracle_matches_unkeyed() {
        use blake2::digest::{Update, VariableOutput};
        use blake2::Blake2bVar;

        let buf: Vec<u8> = (0..1024u32).map(|i| (i & 0xFF) as u8).collect();
        for n in [0usize, 1, 17, 64, 127, 128, 129, 256, 511, 512, 700, 1024] {
            for out_len in [16usize, 32, 48, 64] {
                let mut oracle = Blake2bVar::new(out_len).unwrap();
                Update::update(&mut oracle, &buf[..n]);
                let mut oracle_out = vec![0u8; out_len];
                oracle.finalize_variable(&mut oracle_out).unwrap();

                let mine = blake2b(&buf[..n], out_len);
                assert_eq!(
                    &mine[..out_len],
                    oracle_out.as_slice(),
                    "mismatch n={} out_len={}",
                    n,
                    out_len
                );
            }
        }
    }
}
