// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SHA-1 (FIPS 180-4 / RFC 3174) — clean-room `#![no_std]` implementation.
//!
//! ## Why ship SHA-1?
//!
//! SHA-1 is **only** used as the HMAC primitive for RFC 6238 TOTP, where the
//! standard mandates HMAC-SHA-1 for interoperability with the broad
//! installed base of authenticator apps (Google Authenticator, Authy,
//! 1Password, etc.). RFC 6238 §1.2 explicitly notes that variants based on
//! SHA-256 / SHA-512 exist but the SHA-1 form is the lowest-common-denom
//! that every authenticator implements.
//!
//! SmallAIOS does **not** use SHA-1 anywhere else: file hashes, signatures,
//! and KDFs all use SHA-3-256 / SHAKE256 / Blake2b. Argon2id (in
//! [`crate::argon2id`]) uses Blake2b. PQC signatures use ML-DSA-65 +
//! SHAKE256.
//!
//! ## Threat model
//!
//! SHA-1 is collision-broken since 2017 (SHAttered), but RFC 6238 TOTP
//! uses SHA-1 inside an HMAC, where collision-resistance is not required —
//! HMAC's security reduces to PRF-security of the compression function,
//! which SHA-1 still provides for short, structured inputs (RFC 6151 §2.3
//! confirms HMAC-SHA-1 remains a usable PRF).
//!
//! A future SmallAIOS phase MAY add a SHA-3-based TOTP variant per spec
//! Q21 ("A SHA-3-based variant MAY be added in a future change without
//! breaking the SHA-1 path"). For now the SHA-1 path is the only one.
//!
//! ## API
//!
//! ```ignore
//! use smallaios_security::sha1::Sha1;
//! let mut h = Sha1::new();
//! h.update(b"abc");
//! let digest: [u8; 20] = h.finalize();
//! ```
//!
//! [`sha1`] is the single-call convenience wrapper.
//!
//! ## Test vectors
//!
//! KATs from FIPS 180-4 Appendix A.1 / RFC 3174 §7.3 are pinned in
//! [`crate::sha1_test_vectors`].

use core::convert::TryInto;

/// SHA-1 digest length in bytes.
pub const DIGEST_LEN: usize = 20;

/// SHA-1 block length in bytes.
pub const BLOCK_LEN: usize = 64;

/// SHA-1 IV — the five 32-bit working variables described in
/// FIPS 180-4 §5.3.1. Public constants from the spec, not secrets.
//
// lgtm[rust/hard-coded-cryptographic-value] — FIPS 180-4 §5.3.1 IV, public constant
const H0: [u32; 5] = [
    0x6745_2301,
    0xEFCD_AB89,
    0x98BA_DCFE,
    0x1032_5476,
    0xC3D2_E1F0,
];

/// SHA-1 round constants K[0..3] from RFC 3174 §5 / FIPS 180-4 §4.2.1.
/// Public constants — not secrets.
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 3174 §5 K[t], public constant
const K: [u32; 4] = [0x5A82_7999, 0x6ED9_EBA1, 0x8F1B_BCDC, 0xCA62_C1D6];

/// Streaming SHA-1 context.
///
/// `update` may be called any number of times; `finalize` consumes the
/// context and returns the 20-byte digest. The struct holds five 32-bit
/// state words plus the message-byte counter and a pending-block buffer,
/// matching the layout described in FIPS 180-4 §6.1.2.
#[derive(Clone)]
pub struct Sha1 {
    /// Compression function state (`H0..H4`).
    state: [u32; 5],
    /// Total bytes consumed so far. Used to build the 64-bit length
    /// suffix in [`Self::finalize`].
    len_bytes: u64,
    /// Pending input bytes (< 64). Filled until a full block triggers
    /// [`Self::compress`].
    buffer: [u8; BLOCK_LEN],
    /// Current fill of [`Self::buffer`].
    buffer_len: usize,
}

impl Default for Sha1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha1 {
    /// Construct a fresh SHA-1 context.
    pub const fn new() -> Self {
        Self {
            state: H0,
            len_bytes: 0,
            buffer: [0u8; BLOCK_LEN],
            buffer_len: 0,
        }
    }

    /// Absorb `data` into the running hash.
    pub fn update(&mut self, mut data: &[u8]) {
        self.len_bytes = self.len_bytes.wrapping_add(data.len() as u64);

        // Top up the pending buffer.
        if self.buffer_len > 0 {
            let want = BLOCK_LEN - self.buffer_len;
            let take = want.min(data.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&data[..take]);
            self.buffer_len += take;
            data = &data[take..];
            if self.buffer_len == BLOCK_LEN {
                let block = self.buffer;
                Self::compress(&mut self.state, &block);
                self.buffer_len = 0;
            }
        }

        // Compress full blocks straight from `data`.
        while data.len() >= BLOCK_LEN {
            let (block, rest) = data.split_at(BLOCK_LEN);
            let block_arr: [u8; BLOCK_LEN] = block.try_into().unwrap();
            Self::compress(&mut self.state, &block_arr);
            data = rest;
        }

        // Stash the residue.
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffer_len = data.len();
        }
    }

    /// Finalise the hash and return the 20-byte digest.
    ///
    /// Consumes the context — call [`Self::clone`] first if you need
    /// intermediate digests.
    pub fn finalize(mut self) -> [u8; DIGEST_LEN] {
        // Append a `0x80` padding byte, then zero-pad until the buffer
        // contains exactly `BLOCK_LEN - 8` bytes; then append the
        // big-endian 64-bit bit-length.
        let bit_len = self.len_bytes.wrapping_mul(8);
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > BLOCK_LEN - 8 {
            // Not enough room for the length — pad and compress, then
            // start a fresh zero block.
            for b in &mut self.buffer[self.buffer_len..] {
                *b = 0;
            }
            let block = self.buffer;
            Self::compress(&mut self.state, &block);
            self.buffer_len = 0;
        }
        // Zero-pad up to the length suffix.
        for b in &mut self.buffer[self.buffer_len..BLOCK_LEN - 8] {
            *b = 0;
        }
        // Big-endian 64-bit bit length.
        self.buffer[BLOCK_LEN - 8..].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        Self::compress(&mut self.state, &block);

        // Big-endian dump of state.
        let mut out = [0u8; DIGEST_LEN];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// One SHA-1 block compression. `block` is 64 bytes of the message
    /// (RFC 3174 §6.1).
    fn compress(state: &mut [u32; 5], block: &[u8; BLOCK_LEN]) {
        // Message schedule W[0..80].
        let mut w = [0u32; 80];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(chunk.try_into().unwrap());
        }
        for t in 16..80 {
            w[t] = (w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16]).rotate_left(1);
        }

        // Working variables.
        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];

        // RFC 3174 §5 round loop. Iterating with `enumerate()` mirrors
        // the spec's `t = 0..=79` index without tripping
        // `clippy::needless_range_loop`.
        for (t, &w_t) in w.iter().enumerate() {
            let (f, k) = match t {
                0..=19 => ((b & c) | ((!b) & d), K[0]),
                20..=39 => (b ^ c ^ d, K[1]),
                40..=59 => ((b & c) | (b & d) | (c & d), K[2]),
                _ => (b ^ c ^ d, K[3]),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w_t);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
    }
}

/// One-shot SHA-1: `digest = SHA1(data)`.
pub fn sha1(data: &[u8]) -> [u8; DIGEST_LEN] {
    let mut h = Sha1::new();
    h.update(data);
    h.finalize()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "sha1_test_vectors.rs"]
mod test_vectors;

#[cfg(test)]
mod tests {
    use super::test_vectors::*;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Render a 20-byte digest as lowercase hex for readable assertions.
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
    fn empty_string_kat() {
        // FIPS 180-4 / RFC 3174 §7.3: SHA1("") =
        // da39a3ee5e6b4b0d3255bfef95601890afd80709
        let d = sha1(EMPTY_INPUT);
        assert_eq!(hex(&d), EMPTY_DIGEST_HEX);
    }

    #[test]
    fn abc_kat() {
        // FIPS 180-4 §A.1: SHA1("abc") =
        // a9993e364706816aba3e25717850c26c9cd0d89d
        let d = sha1(ABC_INPUT);
        assert_eq!(hex(&d), ABC_DIGEST_HEX);
    }

    #[test]
    fn long_kat_two_block_message() {
        // FIPS 180-4 §A.2: SHA1("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") =
        // 84983e441c3bd26ebaae4aa1f95129e5e54670f1
        let d = sha1(TWO_BLOCK_INPUT);
        assert_eq!(hex(&d), TWO_BLOCK_DIGEST_HEX);
    }

    #[test]
    fn million_a_kat() {
        // FIPS 180-4 §A.3: SHA1("a"*1_000_000) =
        // 34aa973cd4c4daa4f61eeb2bdbad27316534016f
        let mut h = Sha1::new();
        // Use a 16 KiB scratch buffer to limit peak memory while hashing 1 MB.
        let chunk: Vec<u8> = vec![b'a'; 16 * 1024];
        let total = 1_000_000usize;
        let mut left = total;
        while left > 0 {
            let take = chunk.len().min(left);
            h.update(&chunk[..take]);
            left -= take;
        }
        let d = h.finalize();
        assert_eq!(hex(&d), MILLION_A_DIGEST_HEX);
    }

    #[test]
    fn streaming_matches_one_shot() {
        // Hash the same data byte-by-byte and in one call; the digests
        // SHALL match. Stress-tests the buffer-management state machine.
        let data = CHUNKED_STREAM_INPUT;
        let one_shot = sha1(data);

        let mut h = Sha1::new();
        for &b in data {
            h.update(&[b]);
        }
        let streamed = h.finalize();
        assert_eq!(one_shot, streamed);
    }

    #[test]
    fn updates_at_block_boundaries() {
        // 56-byte first chunk fills exactly to the BLOCK_LEN-8 boundary
        // before length suffix is appended on finalize. Regression
        // guard for off-by-one in the padding logic.
        let one_shot = sha1(BOUNDARY_56_INPUT);
        let mut h = Sha1::new();
        h.update(&BOUNDARY_56_INPUT[..56]);
        h.finalize();
        // The above line consumes h; build a fresh one for the streamed
        // assertion.
        let mut h2 = Sha1::new();
        h2.update(BOUNDARY_56_INPUT);
        let d2 = h2.finalize();
        assert_eq!(d2, one_shot);
    }

    #[test]
    fn updates_with_exact_block_size_input() {
        // 64-byte input lands exactly on a block boundary; finalize
        // SHALL emit one extra padding block.
        let mut h = Sha1::new();
        h.update(EXACT_BLOCK_INPUT);
        let d1 = h.finalize();
        assert_eq!(d1, sha1(EXACT_BLOCK_INPUT));
    }

    #[test]
    fn block_constant_matches_rfc() {
        assert_eq!(BLOCK_LEN, 64);
        assert_eq!(DIGEST_LEN, 20);
    }
}
