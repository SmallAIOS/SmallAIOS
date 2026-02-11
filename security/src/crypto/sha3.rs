// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SHA-3-256 and SHAKE256 implementation (FIPS 202).
//!
//! This module provides the foundational hash primitives for SmallAIOS:
//!
//! - **SHA-3-256**: A 256-bit cryptographic hash function producing a 32-byte digest.
//! - **SHAKE256**: An extendable-output function (XOF) that can produce arbitrary-length output.
//!
//! Both are built on the Keccak-f\[1600\] permutation with 24 rounds. The Keccak
//! state is a 5x5 array of 64-bit lanes (1600 bits total). SHA-3-256 uses a rate
//! of 1088 bits (136 bytes) and capacity of 512 bits. SHAKE256 also uses rate=1088
//! and capacity=512 but with a different domain separator byte.
//!
//! # Security Properties
//!
//! - SHA-3-256: 256-bit preimage resistance, 128-bit collision resistance
//! - SHAKE256: Arbitrary output length with 256-bit security level
//!
//! # References
//!
//! - FIPS 202: SHA-3 Standard
//! - <https://keccak.team/keccak_specs_summary.html>

#![allow(unused)]

use core::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

/// SHA-3-256 output length in bytes.
pub const SHA3_256_DIGEST_LEN: usize = 32;

/// SHA-3-256 rate in bytes: (1600 - 2*256) / 8 = 136.
pub const SHA3_256_RATE: usize = 136;

/// SHAKE256 rate in bytes: (1600 - 2*256) / 8 = 136.
pub const SHAKE256_RATE: usize = 136;

/// Domain separation byte for SHA-3.
const SHA3_DOMAIN: u8 = 0x06;

/// Domain separation byte for SHAKE.
const SHAKE_DOMAIN: u8 = 0x1F;

/// Number of Keccak-f[1600] rounds.
const KECCAK_ROUNDS: usize = 24;

/// Keccak-f[1600] round constants (iota step).
const RC: [u64; KECCAK_ROUNDS] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808A,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808B,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008A,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000A,
    0x0000_0000_8000_808B,
    0x8000_0000_0000_008B,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800A,
    0x8000_0000_8000_000A,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Rotation offsets for the rho step, indexed as ROTATIONS[x][y].
const ROTATIONS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from SHA-3 / SHAKE operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sha3Error {
    /// Output buffer too small for the requested digest.
    BufferTooSmall,
    /// Attempted to finalize an already-finalized state.
    AlreadyFinalized,
    /// Attempted to absorb after squeezing (SHAKE).
    InvalidState,
}

impl fmt::Display for Sha3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall => write!(f, "output buffer too small"),
            Self::AlreadyFinalized => write!(f, "state already finalized"),
            Self::InvalidState => write!(f, "invalid state transition"),
        }
    }
}

// ─── Keccak State ────────────────────────────────────────────────────────────

/// The Keccak-f[1600] state: 25 lanes of 64 bits each (5x5 grid).
///
/// Lane indexing: `state[x + 5*y]` where x, y in 0..5.
#[derive(Clone)]
struct KeccakState {
    lanes: [u64; 25],
}

impl KeccakState {
    /// Create a zeroed Keccak state.
    const fn new() -> Self {
        Self { lanes: [0u64; 25] }
    }

    /// Apply the Keccak-f[1600] permutation (24 rounds).
    fn permute(&mut self) {
        let a = &mut self.lanes;

        for rc in &RC {
            // θ (theta) step
            let mut c = [0u64; 5];
            for x in 0..5 {
                c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
            }
            let mut d = [0u64; 5];
            for x in 0..5 {
                d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            }
            for x in 0..5 {
                for y in 0..5 {
                    a[x + 5 * y] ^= d[x];
                }
            }

            // ρ (rho) and π (pi) steps combined
            let mut b = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let lane = a[x + 5 * y].rotate_left(ROTATIONS[x][y]);
                    b[y + 5 * ((2 * x + 3 * y) % 5)] = lane;
                }
            }

            // χ (chi) step
            for x in 0..5 {
                for y in 0..5 {
                    a[x + 5 * y] =
                        b[x + 5 * y] ^ ((!b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
                }
            }

            // ι (iota) step
            a[0] ^= *rc;
        }
    }

    /// XOR a block of bytes into the state (little-endian lane encoding).
    ///
    /// `data` must be at most `rate` bytes long.
    fn xor_block(&mut self, data: &[u8], rate: usize) {
        debug_assert!(data.len() <= rate);
        let lanes_to_xor = data.len() / 8;
        for i in 0..lanes_to_xor {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&data[i * 8..(i + 1) * 8]);
            self.lanes[i] ^= u64::from_le_bytes(bytes);
        }
        // Handle remaining bytes (partial lane)
        let remaining = data.len() % 8;
        if remaining > 0 {
            let mut bytes = [0u8; 8];
            let offset = lanes_to_xor * 8;
            bytes[..remaining].copy_from_slice(&data[offset..offset + remaining]);
            self.lanes[lanes_to_xor] ^= u64::from_le_bytes(bytes);
        }
    }

    /// Extract bytes from the state (little-endian lane encoding).
    fn extract(&self, out: &mut [u8], rate: usize) {
        let len = out.len().min(rate);
        let full_lanes = len / 8;
        for i in 0..full_lanes {
            let bytes = self.lanes[i].to_le_bytes();
            out[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        }
        let remaining = len % 8;
        if remaining > 0 {
            let bytes = self.lanes[full_lanes].to_le_bytes();
            let offset = full_lanes * 8;
            out[offset..offset + remaining].copy_from_slice(&bytes[..remaining]);
        }
    }
}

// ─── SHA-3-256 ───────────────────────────────────────────────────────────────

/// SHA-3-256 digest: a 32-byte hash value.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Sha3_256Digest {
    bytes: [u8; SHA3_256_DIGEST_LEN],
}

impl Sha3_256Digest {
    /// Create a digest from a 32-byte array.
    pub const fn from_bytes(bytes: [u8; SHA3_256_DIGEST_LEN]) -> Self {
        Self { bytes }
    }

    /// Return the digest as a byte slice.
    pub fn as_bytes(&self) -> &[u8; SHA3_256_DIGEST_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for Sha3_256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sha3_256Digest(")?;
        for b in &self.bytes {
            write!(f, "{:02x}", b)?;
        }
        write!(f, ")")
    }
}

/// Incremental SHA-3-256 hasher.
///
/// # Example (conceptual)
///
/// ```ignore
/// let mut hasher = Sha3_256::new();
/// hasher.update(b"hello ");
/// hasher.update(b"world");
/// let digest = hasher.finalize().unwrap();
/// ```
pub struct Sha3_256 {
    state: KeccakState,
    /// Buffer for partial blocks.
    buffer: [u8; SHA3_256_RATE],
    /// Number of bytes in the buffer.
    buf_len: usize,
    /// Total bytes absorbed (for debugging/validation).
    absorbed: u64,
    /// Whether finalize() has been called.
    finalized: bool,
}

impl Sha3_256 {
    /// Create a new SHA-3-256 hasher.
    pub fn new() -> Self {
        Self {
            state: KeccakState::new(),
            buffer: [0u8; SHA3_256_RATE],
            buf_len: 0,
            absorbed: 0,
            finalized: false,
        }
    }

    /// Absorb input data into the hash state.
    pub fn update(&mut self, data: &[u8]) -> Result<(), Sha3Error> {
        if self.finalized {
            return Err(Sha3Error::AlreadyFinalized);
        }

        let mut offset = 0;
        let mut remaining = data.len();

        // If buffer has partial data, fill it first
        if self.buf_len > 0 {
            let space = SHA3_256_RATE - self.buf_len;
            if remaining < space {
                self.buffer[self.buf_len..self.buf_len + remaining]
                    .copy_from_slice(&data[offset..offset + remaining]);
                self.buf_len += remaining;
                self.absorbed += remaining as u64;
                return Ok(());
            }
            // Fill the buffer and process it
            self.buffer[self.buf_len..SHA3_256_RATE].copy_from_slice(&data[offset..offset + space]);
            self.state.xor_block(&self.buffer, SHA3_256_RATE);
            self.state.permute();
            offset += space;
            remaining -= space;
            self.buf_len = 0;
        }

        // Process full blocks directly from input
        while remaining >= SHA3_256_RATE {
            self.state
                .xor_block(&data[offset..offset + SHA3_256_RATE], SHA3_256_RATE);
            self.state.permute();
            offset += SHA3_256_RATE;
            remaining -= SHA3_256_RATE;
        }

        // Buffer remaining bytes
        if remaining > 0 {
            self.buffer[..remaining].copy_from_slice(&data[offset..offset + remaining]);
            self.buf_len = remaining;
        }

        self.absorbed += data.len() as u64;
        Ok(())
    }

    /// Finalize the hash and return the 32-byte digest.
    ///
    /// After calling this, the hasher is consumed (further updates will error).
    pub fn finalize(&mut self) -> Result<Sha3_256Digest, Sha3Error> {
        if self.finalized {
            return Err(Sha3Error::AlreadyFinalized);
        }
        self.finalized = true;

        // Apply SHA-3 padding: domain byte || 10*1 padding
        // Pad the buffer: set domain separator, then pad10*1
        self.buffer[self.buf_len] = SHA3_DOMAIN;
        // Zero the rest of the buffer between domain byte and last byte
        for i in (self.buf_len + 1)..SHA3_256_RATE {
            self.buffer[i] = 0;
        }
        // Set the high bit of the last byte (pad10*1 terminator)
        self.buffer[SHA3_256_RATE - 1] |= 0x80;

        // Absorb the final padded block
        self.state
            .xor_block(&self.buffer[..SHA3_256_RATE], SHA3_256_RATE);
        self.state.permute();

        // Squeeze out 32 bytes
        let mut digest = [0u8; SHA3_256_DIGEST_LEN];
        self.state.extract(&mut digest, SHA3_256_RATE);

        Ok(Sha3_256Digest::from_bytes(digest))
    }
}

impl Default for Sha3_256 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot SHA-3-256 hash of the input data.
pub fn sha3_256(data: &[u8]) -> Sha3_256Digest {
    let mut hasher = Sha3_256::new();
    // update and finalize cannot fail on a fresh hasher
    hasher
        .update(data)
        .expect("fresh hasher update cannot fail");
    hasher
        .finalize()
        .expect("fresh hasher finalize cannot fail")
}

// ─── SHAKE256 ────────────────────────────────────────────────────────────────

/// SHAKE256 state machine phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShakePhase {
    /// Absorbing input data.
    Absorbing,
    /// Squeezing output data.
    Squeezing,
}

/// SHAKE256 extendable-output function (XOF).
///
/// Can absorb arbitrary input and then squeeze out an arbitrary number
/// of output bytes. Once squeezing begins, no more data can be absorbed.
pub struct Shake256 {
    state: KeccakState,
    /// Buffer for partial blocks.
    buffer: [u8; SHAKE256_RATE],
    /// Number of bytes in the buffer.
    buf_len: usize,
    /// Current phase.
    phase: ShakePhase,
    /// Squeeze offset within the current rate block.
    squeeze_offset: usize,
}

impl Shake256 {
    /// Create a new SHAKE256 XOF instance.
    pub const fn new() -> Self {
        Self {
            state: KeccakState::new(),
            buffer: [0u8; SHAKE256_RATE],
            buf_len: 0,
            phase: ShakePhase::Absorbing,
            squeeze_offset: 0,
        }
    }

    /// Absorb input data.
    pub fn absorb(&mut self, data: &[u8]) -> Result<(), Sha3Error> {
        if self.phase != ShakePhase::Absorbing {
            return Err(Sha3Error::InvalidState);
        }

        let mut offset = 0;
        let mut remaining = data.len();

        // Fill partial buffer
        if self.buf_len > 0 {
            let space = SHAKE256_RATE - self.buf_len;
            if remaining < space {
                self.buffer[self.buf_len..self.buf_len + remaining]
                    .copy_from_slice(&data[offset..offset + remaining]);
                self.buf_len += remaining;
                return Ok(());
            }
            self.buffer[self.buf_len..SHAKE256_RATE].copy_from_slice(&data[offset..offset + space]);
            self.state.xor_block(&self.buffer, SHAKE256_RATE);
            self.state.permute();
            offset += space;
            remaining -= space;
            self.buf_len = 0;
        }

        // Full blocks
        while remaining >= SHAKE256_RATE {
            self.state
                .xor_block(&data[offset..offset + SHAKE256_RATE], SHAKE256_RATE);
            self.state.permute();
            offset += SHAKE256_RATE;
            remaining -= SHAKE256_RATE;
        }

        // Buffer leftover
        if remaining > 0 {
            self.buffer[..remaining].copy_from_slice(&data[offset..offset + remaining]);
            self.buf_len = remaining;
        }

        Ok(())
    }

    /// Transition from absorbing to squeezing by applying SHAKE padding.
    fn finalize_absorb(&mut self) {
        // SHAKE256 padding: 0x1F || 10*1
        self.buffer[self.buf_len] = SHAKE_DOMAIN;
        for i in (self.buf_len + 1)..SHAKE256_RATE {
            self.buffer[i] = 0;
        }
        self.buffer[SHAKE256_RATE - 1] |= 0x80;

        self.state
            .xor_block(&self.buffer[..SHAKE256_RATE], SHAKE256_RATE);
        self.state.permute();

        self.phase = ShakePhase::Squeezing;
        self.squeeze_offset = 0;

        // Extract first block into buffer for squeezing
        self.state.extract(&mut self.buffer, SHAKE256_RATE);
    }

    /// Squeeze output bytes from the XOF.
    ///
    /// Can be called multiple times to produce an arbitrary amount of output.
    pub fn squeeze(&mut self, out: &mut [u8]) -> Result<(), Sha3Error> {
        if self.phase == ShakePhase::Absorbing {
            self.finalize_absorb();
        }

        let mut offset = 0;
        let mut remaining = out.len();

        while remaining > 0 {
            let available = SHAKE256_RATE - self.squeeze_offset;
            if remaining <= available {
                out[offset..offset + remaining].copy_from_slice(
                    &self.buffer[self.squeeze_offset..self.squeeze_offset + remaining],
                );
                self.squeeze_offset += remaining;
                return Ok(());
            }
            // Copy what's available from current block
            out[offset..offset + available]
                .copy_from_slice(&self.buffer[self.squeeze_offset..SHAKE256_RATE]);
            offset += available;
            remaining -= available;

            // Permute and extract next block
            self.state.permute();
            self.state.extract(&mut self.buffer, SHAKE256_RATE);
            self.squeeze_offset = 0;
        }

        Ok(())
    }
}

impl Default for Shake256 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot SHAKE256: absorb `data` and squeeze `output_len` bytes.
pub fn shake256(data: &[u8], output: &mut [u8]) {
    let mut xof = Shake256::new();
    xof.absorb(data).expect("fresh SHAKE256 absorb cannot fail");
    xof.squeeze(output).expect("SHAKE256 squeeze cannot fail");
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    /// NIST test vector: SHA-3-256 of empty string.
    /// Expected: a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a
    #[test]
    fn sha3_256_empty() {
        let digest = sha3_256(b"");
        let expected: [u8; 32] = [
            0xa7, 0xff, 0xc6, 0xf8, 0xbf, 0x1e, 0xd7, 0x66, 0x51, 0xc1, 0x47, 0x56, 0xa0, 0x61,
            0xd6, 0x62, 0xf5, 0x80, 0xff, 0x4d, 0xe4, 0x3b, 0x49, 0xfa, 0x82, 0xd8, 0x0a, 0x4b,
            0x80, 0xf8, 0x43, 0x4a,
        ];
        assert_eq!(
            digest.as_bytes(),
            &expected,
            "SHA-3-256 of empty string mismatch"
        );
    }

    /// NIST test vector: SHA-3-256 of "abc".
    /// Expected: 3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532
    #[test]
    fn sha3_256_abc() {
        let digest = sha3_256(b"abc");
        let expected: [u8; 32] = [
            0x3a, 0x98, 0x5d, 0xa7, 0x4f, 0xe2, 0x25, 0xb2, 0x04, 0x5c, 0x17, 0x2d, 0x6b, 0xd3,
            0x90, 0xbd, 0x85, 0x5f, 0x08, 0x6e, 0x3e, 0x9d, 0x52, 0x5b, 0x46, 0xbf, 0xe2, 0x45,
            0x11, 0x43, 0x15, 0x32,
        ];
        assert_eq!(digest.as_bytes(), &expected, "SHA-3-256 of 'abc' mismatch");
    }

    /// Test incremental hashing matches one-shot.
    #[test]
    fn sha3_256_incremental() {
        let one_shot = sha3_256(b"hello world");

        let mut hasher = Sha3_256::new();
        hasher.update(b"hello ").unwrap();
        hasher.update(b"world").unwrap();
        let incremental = hasher.finalize().unwrap();

        assert_eq!(
            one_shot, incremental,
            "incremental hash should match one-shot"
        );
    }

    /// Test that finalizing twice returns an error.
    #[test]
    fn sha3_256_double_finalize() {
        let mut hasher = Sha3_256::new();
        hasher.update(b"data").unwrap();
        let _ = hasher.finalize().unwrap();
        assert_eq!(hasher.finalize(), Err(Sha3Error::AlreadyFinalized));
    }

    /// Test that updating after finalize returns an error.
    #[test]
    fn sha3_256_update_after_finalize() {
        let mut hasher = Sha3_256::new();
        let _ = hasher.finalize().unwrap();
        assert_eq!(hasher.update(b"data"), Err(Sha3Error::AlreadyFinalized));
    }

    /// Test multi-block input (larger than rate).
    #[test]
    fn sha3_256_multi_block() {
        // 200 bytes > 136 byte rate, exercises multi-block path
        let data = [0x42u8; 200];
        let digest = sha3_256(&data);
        // Just verify it doesn't panic and returns a 32-byte digest
        assert_eq!(digest.as_bytes().len(), 32);
    }

    /// Test exact block-size input.
    #[test]
    fn sha3_256_exact_rate() {
        let data = [0xAB; SHA3_256_RATE];
        let digest = sha3_256(&data);
        assert_eq!(digest.as_bytes().len(), 32);
    }

    /// Test SHAKE256 produces deterministic output.
    #[test]
    fn shake256_deterministic() {
        let mut out1 = [0u8; 64];
        let mut out2 = [0u8; 64];
        shake256(b"test seed", &mut out1);
        shake256(b"test seed", &mut out2);
        assert_eq!(out1, out2, "SHAKE256 should be deterministic");
    }

    /// Test SHAKE256 produces different output for different inputs.
    #[test]
    fn shake256_different_inputs() {
        let mut out1 = [0u8; 32];
        let mut out2 = [0u8; 32];
        shake256(b"input a", &mut out1);
        shake256(b"input b", &mut out2);
        assert_ne!(
            out1, out2,
            "different inputs should yield different outputs"
        );
    }

    /// Test SHAKE256 incremental squeeze.
    #[test]
    fn shake256_incremental_squeeze() {
        let mut xof = Shake256::new();
        xof.absorb(b"data").unwrap();

        let mut out_a = [0u8; 16];
        let mut out_b = [0u8; 16];
        xof.squeeze(&mut out_a).unwrap();
        xof.squeeze(&mut out_b).unwrap();

        // Concatenation should match a single 32-byte squeeze
        let mut combined = [0u8; 32];
        shake256(b"data", &mut combined);
        assert_eq!(&combined[..16], &out_a);
        assert_eq!(&combined[16..], &out_b);
    }

    /// Test SHAKE256 absorb-after-squeeze fails.
    #[test]
    fn shake256_absorb_after_squeeze() {
        let mut xof = Shake256::new();
        xof.absorb(b"data").unwrap();
        let mut out = [0u8; 8];
        xof.squeeze(&mut out).unwrap();
        assert_eq!(xof.absorb(b"more"), Err(Sha3Error::InvalidState));
    }

    /// Test SHAKE256 large output (multiple rate blocks).
    #[test]
    fn shake256_large_output() {
        let mut out = [0u8; 512]; // > 136 * 3 = 408
        shake256(b"seed", &mut out);
        // Verify not all zeros
        assert!(
            out.iter().any(|&b| b != 0),
            "SHAKE256 output should not be all zeros"
        );
    }

    /// Verify Keccak round constants count.
    #[test]
    fn keccak_round_constant_count() {
        assert_eq!(RC.len(), 24, "Keccak-f[1600] has exactly 24 rounds");
    }

    /// Verify first and last round constants.
    #[test]
    fn keccak_round_constants_spot_check() {
        assert_eq!(RC[0], 0x0000_0000_0000_0001);
        assert_eq!(RC[23], 0x8000_0000_8000_8008);
    }

    /// Test digest Debug formatting.
    #[test]
    fn sha3_256_digest_debug() {
        let digest = Sha3_256Digest::from_bytes([0u8; 32]);
        let debug = format!("{:?}", digest);
        assert!(debug.starts_with("Sha3_256Digest("));
        assert!(debug.ends_with(")"));
    }
}
