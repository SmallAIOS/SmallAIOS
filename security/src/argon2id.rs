// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Argon2id — memory-hard password hashing (RFC 9106).
//!
//! Clean-room `#![no_std]` implementation of Argon2id (the hybrid Argon2
//! variant: data-independent indexing for the first half-pass of the first
//! pass, data-dependent indexing for the rest). Emits a 32-byte tag and a
//! PHC-encoded string suitable for storage in `/data/auth/shadow`.
//!
//! # Usage
//!
//! ```ignore
//! use smallaios_security::argon2id::{argon2id_hash, Argon2idParams};
//!
//! let salt = b"saltsaltsaltsalt"; // ≥ 8 bytes per RFC 9106 §3.1
//! let tag = argon2id_hash(b"correct horse battery staple", salt,
//!                          Argon2idParams::default_tier());
//! ```
//!
//! # Per-tier defaults
//!
//! See [`Argon2idParams::tiny`], [`Argon2idParams::default_tier`],
//! [`Argon2idParams::strong`]. The PHC string carries the parameters so
//! verification is self-describing.
//!
//! # Implementation notes
//!
//! - Uses the [`crate::crypto::blake2b`] backend for `H` and `H'`.
//! - Memory layout is a flat row-major `m_prime × 1024` block grid where
//!   `m_prime = 4 * p_cost * floor(m_cost_kib / (4 * p_cost))` per RFC 9106
//!   §3.2.
//! - Indexing follows §3.4 — first half-pass of pass 0 is data-independent
//!   (G2 stream); the rest is data-dependent (read from the previous block).
//! - Per-arch SIMD (NEON / AVX2) shims are deferred to a follow-on phase.
//!   Only the portable BLAMKA path is enabled here.
//!
//! # References
//!
//! - RFC 9106 — Argon2 Memory-Hard Function for Password Hashing and
//!   Proof-of-Work Applications.
//! - <https://github.com/P-H-C/phc-winner-argon2>

#![allow(unused)]

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::crypto::blake2b::{Blake2b, BLAKE2B_OUT_MAX};
use crate::crypto::constant_time::ct_eq;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Argon2 block size in bytes (RFC 9106 §3.1: B = 1024 bytes = 128 u64 lanes).
pub const ARGON2_BLOCK_BYTES: usize = 1024;

/// Argon2id type identifier (RFC 9106 §3.2 step 5).
const ARGON2ID_TYPE: u32 = 2;

/// Argon2 version (0x13 = 1.3, RFC 9106).
const ARGON2_VERSION: u32 = 0x13;

/// SYNC_POINTS — number of slices per lane per pass (RFC 9106 §3.4).
const SYNC_POINTS: usize = 4;

/// Output tag length used by the management-login API.
pub const ARGON2ID_TAG_LEN: usize = 32;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors raised by Argon2id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// `m_cost_kib` is below the RFC 9106 minimum `8 * p_cost`.
    InvalidMemory,
    /// `t_cost` is below 1.
    InvalidTime,
    /// `p_cost` is 0 or > the RFC 9106 maximum (2^24 - 1).
    InvalidLanes,
    /// Salt length below the RFC 9106 minimum (8 bytes).
    InvalidSalt,
    /// Output tag length out of supported range (4..=64 bytes).
    InvalidTagLength,
    /// PHC string did not parse.
    InvalidPhc,
    /// Argon2 variant in PHC string is not `argon2id`.
    UnsupportedVariant,
    /// Argon2 version in PHC string is not 0x13.
    UnsupportedVersion,
    /// Base64 decoding failed.
    Base64Decode,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMemory => write!(f, "argon2id: m_cost too small"),
            Self::InvalidTime => write!(f, "argon2id: t_cost must be >= 1"),
            Self::InvalidLanes => write!(f, "argon2id: p_cost out of range"),
            Self::InvalidSalt => write!(f, "argon2id: salt must be >= 8 bytes"),
            Self::InvalidTagLength => write!(f, "argon2id: tag length out of range"),
            Self::InvalidPhc => write!(f, "argon2id: malformed PHC string"),
            Self::UnsupportedVariant => write!(f, "argon2id: variant not argon2id"),
            Self::UnsupportedVersion => write!(f, "argon2id: unsupported version"),
            Self::Base64Decode => write!(f, "argon2id: base64 decode failure"),
        }
    }
}

// ─── Parameters ──────────────────────────────────────────────────────────────

/// Argon2id parameters per RFC 9106 §3.1.
///
/// - `m_cost_kib` — memory cost in kibibytes (1 KiB = 1024 bytes = 1 block).
/// - `t_cost` — time cost: number of passes over the memory.
/// - `p_cost` — parallelism / lane count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Argon2idParams {
    pub m_cost_kib: u32,
    pub t_cost: u32,
    pub p_cost: u32,
}

impl Argon2idParams {
    /// `tiny` tier — 8 MiB / t=3 / p=1 — for ≤ 256 MiB-RAM x86 targets.
    pub const fn tiny() -> Self {
        Self {
            m_cost_kib: 8 * 1024,
            t_cost: 3,
            p_cost: 1,
        }
    }

    /// `default` tier — 64 MiB / t=3 / p=1 — typical container/VM target.
    /// (Named `default_tier` rather than implementing `Default` to avoid
    /// confusion with crypto callers that explicitly pick a tier.)
    pub const fn default_tier() -> Self {
        Self {
            m_cost_kib: 64 * 1024,
            t_cost: 3,
            p_cost: 1,
        }
    }

    /// `strong` tier — 128 MiB / t=4 / p=2 — Jetson-class targets.
    pub const fn strong() -> Self {
        Self {
            m_cost_kib: 128 * 1024,
            t_cost: 4,
            p_cost: 2,
        }
    }

    /// Validate the parameters per RFC 9106 §3.1.
    fn validate(&self) -> Result<(), Error> {
        if self.t_cost < 1 {
            return Err(Error::InvalidTime);
        }
        if self.p_cost == 0 || self.p_cost > 0x00FF_FFFF {
            return Err(Error::InvalidLanes);
        }
        if self.m_cost_kib < 8 * self.p_cost {
            return Err(Error::InvalidMemory);
        }
        Ok(())
    }
}

// Note: no `Default` impl — callers must pick a tier explicitly.

// ─── Block ───────────────────────────────────────────────────────────────────

/// 1024-byte Argon2 block, viewed as 128 little-endian u64 lanes.
#[derive(Clone, Copy)]
struct Block([u64; 128]);

impl Block {
    const fn zero() -> Self {
        Block([0u64; 128])
    }

    fn from_bytes(b: &[u8]) -> Self {
        debug_assert_eq!(b.len(), ARGON2_BLOCK_BYTES);
        let mut lanes = [0u64; 128];
        for i in 0..128 {
            lanes[i] = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        }
        Block(lanes)
    }

    fn write_bytes(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), ARGON2_BLOCK_BYTES);
        for i in 0..128 {
            out[i * 8..i * 8 + 8].copy_from_slice(&self.0[i].to_le_bytes());
        }
    }

    /// XOR `other` into `self` (in place).
    fn xor_assign(&mut self, other: &Block) {
        for i in 0..128 {
            self.0[i] ^= other.0[i];
        }
    }

    /// Copy `other` into `self`.
    fn copy_from(&mut self, other: &Block) {
        self.0 = other.0;
    }
}

// ─── Variable-length hash H' (RFC 9106 §3.3) ─────────────────────────────────

/// Variable-length hash function `H'` from RFC 9106 §3.3.
///
/// For `out_len <= 64`, `H'(X) = Blake2b_{out_len}(LE32(out_len) || X)`.
/// For larger `out_len`, we chain Blake2b-512 outputs, taking the first 32
/// bytes of each intermediate Blake2b-512 except the last block which
/// emits the full remaining bytes (per the spec's V_i recursion).
fn h_prime(out_len: u32, input: &[&[u8]], out: &mut [u8]) {
    debug_assert_eq!(out.len(), out_len as usize);

    let len_le = out_len.to_le_bytes();

    if out_len as usize <= BLAKE2B_OUT_MAX {
        let mut h = Blake2b::new(out_len as usize);
        h.update(&len_le);
        for chunk in input {
            h.update(chunk);
        }
        let _ = h.finalize_into(out);
        return;
    }

    // Long output: V_1 = Blake2b_512(LE32(T) || X). Subsequent V_{i} =
    // Blake2b_512(V_{i-1}). Output is V_1[0..32] || V_2[0..32] || ... ||
    // V_r where r is chosen so the total length matches `out_len`.
    let mut v = [0u8; BLAKE2B_OUT_MAX];
    {
        let mut h = Blake2b::new(BLAKE2B_OUT_MAX);
        h.update(&len_le);
        for chunk in input {
            h.update(chunk);
        }
        let _ = h.finalize_into(&mut v);
    }

    let mut written = 0usize;
    // RFC 9106: 32-byte halves of the chain, with the last full Blake2b_512
    // covering whatever remains. The number of intermediate iterations is
    // r = ceil(out_len / 32) - 2.
    let total = out_len as usize;
    while written + BLAKE2B_OUT_MAX < total {
        // Emit first 32 bytes of v.
        out[written..written + 32].copy_from_slice(&v[..32]);
        written += 32;
        // Re-hash the full 64-byte v.
        let mut h = Blake2b::new(BLAKE2B_OUT_MAX);
        h.update(&v);
        let _ = h.finalize_into(&mut v);
    }
    // Final block: emit the remaining bytes of v.
    let rem = total - written;
    out[written..].copy_from_slice(&v[..rem]);
}

// ─── BLAMKA / G compression (RFC 9106 §3.5–3.6) ──────────────────────────────

/// BLAMKA mixing primitive (RFC 9106 §3.5).
///
/// Identical structure to Blake2's `G` but with the `+` operations replaced
/// by `a + b + 2 * lower32(a) * lower32(b)`. Operates on four lanes of the
/// 16-lane working state.
#[inline(always)]
fn blamka_g(state: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize) {
    fn fblamka(x: u64, y: u64) -> u64 {
        let xy = (x & 0xFFFF_FFFF).wrapping_mul(y & 0xFFFF_FFFF);
        x.wrapping_add(y).wrapping_add(2u64.wrapping_mul(xy))
    }

    state[a] = fblamka(state[a], state[b]);
    state[d] = (state[d] ^ state[a]).rotate_right(32);
    state[c] = fblamka(state[c], state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(24);
    state[a] = fblamka(state[a], state[b]);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = fblamka(state[c], state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(63);
}

/// Apply the round function `P` to a 16-lane (128-byte) row. RFC 9106 §3.6.
#[inline(always)]
fn blamka_round(s: &mut [u64; 16]) {
    blamka_g(s, 0, 4, 8, 12);
    blamka_g(s, 1, 5, 9, 13);
    blamka_g(s, 2, 6, 10, 14);
    blamka_g(s, 3, 7, 11, 15);

    blamka_g(s, 0, 5, 10, 15);
    blamka_g(s, 1, 6, 11, 12);
    blamka_g(s, 2, 7, 8, 13);
    blamka_g(s, 3, 4, 9, 14);
}

/// Argon2 compression `G(X, Y) -> Z`.
///
/// 1. R = X xor Y
/// 2. Apply `P` row-wise (8 rows of 16 lanes), then column-wise (8 cols of
///    16 lanes), to a temporary copy `Q` of `R`.
/// 3. Z = R xor Q.
fn argon2_g(x: &Block, y: &Block, out: &mut Block) {
    // R = X xor Y
    let mut r = [0u64; 128];
    for (slot, (xv, yv)) in r.iter_mut().zip(x.0.iter().zip(y.0.iter())) {
        *slot = *xv ^ *yv;
    }

    let mut q = r;

    // Row passes: 8 rows × 16 lanes.
    for row in 0..8 {
        let off = row * 16;
        let mut s: [u64; 16] = q[off..off + 16].try_into().unwrap();
        blamka_round(&mut s);
        q[off..off + 16].copy_from_slice(&s);
    }

    // Column passes: 8 columns × 16 lanes (each column is 8 rows × 2 lanes).
    // Column j gathers lanes at positions {2j, 2j+1, 2j+16, 2j+17, ...}.
    for col in 0..8 {
        let mut s = [0u64; 16];
        for k in 0..8 {
            s[2 * k] = q[2 * col + 16 * k];
            s[2 * k + 1] = q[2 * col + 1 + 16 * k];
        }
        blamka_round(&mut s);
        for k in 0..8 {
            q[2 * col + 16 * k] = s[2 * k];
            q[2 * col + 1 + 16 * k] = s[2 * k + 1];
        }
    }

    // Z = R xor Q
    for i in 0..128 {
        out.0[i] = r[i] ^ q[i];
    }
}

// ─── Initial state H_0 (RFC 9106 §3.2 steps 4–5) ─────────────────────────────

/// Compute the 64-byte H_0 seed used to initialize the first two columns
/// of every lane. The hashed input is the concatenation:
///
/// LE32(p) || LE32(T) || LE32(m) || LE32(t) || LE32(v) || LE32(y) ||
/// LE32(|P|) || P || LE32(|S|) || S || LE32(|K|) || K || LE32(|X|) || X
fn h_zero(
    pwd: &[u8],
    salt: &[u8],
    secret: &[u8],
    associated: &[u8],
    params: &Argon2idParams,
    tag_len: u32,
) -> [u8; BLAKE2B_OUT_MAX] {
    let mut h = Blake2b::new(BLAKE2B_OUT_MAX);
    h.update(&params.p_cost.to_le_bytes());
    h.update(&tag_len.to_le_bytes());
    h.update(&params.m_cost_kib.to_le_bytes());
    h.update(&params.t_cost.to_le_bytes());
    h.update(&ARGON2_VERSION.to_le_bytes());
    h.update(&ARGON2ID_TYPE.to_le_bytes());
    h.update(&(pwd.len() as u32).to_le_bytes());
    h.update(pwd);
    h.update(&(salt.len() as u32).to_le_bytes());
    h.update(salt);
    h.update(&(secret.len() as u32).to_le_bytes());
    h.update(secret);
    h.update(&(associated.len() as u32).to_le_bytes());
    h.update(associated);
    let mut out = [0u8; BLAKE2B_OUT_MAX];
    let _ = h.finalize_into(&mut out);
    out
}

// ─── Indexing helpers (RFC 9106 §3.4) ────────────────────────────────────────

/// Compute `J1, J2` from a 64-bit pseudo-random value as specified in
/// RFC 9106 §3.4.
fn split_j(rand: u64) -> (u32, u32) {
    let j1 = (rand & 0xFFFF_FFFF) as u32;
    let j2 = (rand >> 32) as u32;
    (j1, j2)
}

/// Position context passed to [`index_alpha`].
struct Position {
    pass: u32,
    lane: u32,
    slice: u32,
    /// Column index within the segment.
    index: u32,
}

/// Memory dimensions passed to [`index_alpha`].
struct Dimensions {
    lane_length: u32,
    segment_length: u32,
    p_cost: u32,
}

/// Determine the reference (lane, column) from `(J1, J2)` per RFC 9106 §3.4.2.
fn index_alpha(pos: &Position, dim: &Dimensions, j1: u32, j2: u32) -> (u32, u32) {
    // Step 1: select reference lane.
    let ref_lane = if pos.pass == 0 && pos.slice == 0 {
        // First slice of the first pass: must be the current lane.
        pos.lane
    } else {
        j2 % dim.p_cost
    };

    // Step 2: number of reference blocks in the lane.
    let ref_area_size = if pos.pass == 0 {
        if pos.slice == 0 {
            pos.index - 1
        } else if ref_lane == pos.lane {
            pos.slice * dim.segment_length + pos.index - 1
        } else {
            pos.slice * dim.segment_length - u32::from(pos.index == 0)
        }
    } else if ref_lane == pos.lane {
        dim.lane_length - dim.segment_length + pos.index - 1
    } else {
        dim.lane_length - dim.segment_length - u32::from(pos.index == 0)
    };

    // Step 3: relative position within the reference area.
    let mut rel = j1 as u64;
    rel = (rel * rel) >> 32;
    rel = (ref_area_size as u64 * rel) >> 32;
    let rel = ref_area_size as u64 - 1 - rel;

    // Step 4: starting position depends on pass.
    let start_pos = if pos.pass != 0 && pos.slice != SYNC_POINTS as u32 - 1 {
        ((pos.slice + 1) * dim.segment_length) as u64
    } else {
        0u64
    };
    let abs_pos = ((start_pos + rel) % dim.lane_length as u64) as u32;

    (ref_lane, abs_pos)
}

// ─── Pseudo-random stream for data-independent indexing (Argon2id) ────────────

/// Address-block generator for Argon2id's data-independent indexing.
///
/// RFC 9106 §3.4.1.2 and the phc-winner-argon2 reference: a fresh block of
/// 128 64-bit pseudo-random values is generated by `G(0, G(0, IN))` where
/// `IN` encodes `(pass, lane, slice, m', t, type, counter, 0...)` and
/// `counter` increments every 128 values.
///
/// The values are *indexed* by the position `i` within the segment (i % 128),
/// matching the reference. The internal `current_counter` tracks which input
/// counter produced the cached address block.
struct G2Stream {
    pass: u64,
    lane: u64,
    slice: u64,
    blocks_total: u64,
    t_cost: u64,
    /// Counter value of the last block we computed (0 means none yet).
    current_counter: u64,
    address: Block,
}

impl G2Stream {
    fn new(pass: u32, lane: u32, slice: u32, blocks_total: u32, t_cost: u32) -> Self {
        Self {
            pass: pass as u64,
            lane: lane as u64,
            slice: slice as u64,
            blocks_total: blocks_total as u64,
            t_cost: t_cost as u64,
            current_counter: 0,
            address: Block::zero(),
        }
    }

    /// Compute the address block for the given `counter` value (1-based) and
    /// cache it.
    fn fill_for_counter(&mut self, counter: u64) {
        debug_assert!(counter >= 1);
        if self.current_counter == counter {
            return;
        }

        let mut input = Block::zero();
        input.0[0] = self.pass;
        input.0[1] = self.lane;
        input.0[2] = self.slice;
        input.0[3] = self.blocks_total;
        input.0[4] = self.t_cost;
        input.0[5] = ARGON2ID_TYPE as u64;
        input.0[6] = counter;
        // remaining lanes zero

        let zero = Block::zero();
        let mut tmp = Block::zero();
        argon2_g(&zero, &input, &mut tmp);
        argon2_g(&zero, &tmp, &mut self.address);
        self.current_counter = counter;
    }

    /// Return the J value at position `i` within the segment.
    fn j_at(&mut self, i: u32) -> u64 {
        let counter = (i / 128) as u64 + 1;
        self.fill_for_counter(counter);
        self.address.0[(i % 128) as usize]
    }
}

// ─── Memory fill (RFC 9106 §3.2 step 6, §3.4) ────────────────────────────────

/// Argon2id memory matrix indexed as `mem[row]` where
/// `row = lane * lane_length + col`.
struct Memory {
    blocks: Vec<Block>,
    lane_length: u32,
    segment_length: u32,
    blocks_total: u32,
    p_cost: u32,
    t_cost: u32,
}

impl Memory {
    fn new(params: &Argon2idParams) -> Self {
        // Per RFC 9106 §3.2 step 2: m' = 4 * p * floor(m / 4p)
        let blocks_total = 4 * params.p_cost * (params.m_cost_kib / (4 * params.p_cost));
        let lane_length = blocks_total / params.p_cost;
        let segment_length = lane_length / SYNC_POINTS as u32;

        Self {
            blocks: vec![Block::zero(); blocks_total as usize],
            lane_length,
            segment_length,
            blocks_total,
            p_cost: params.p_cost,
            t_cost: params.t_cost,
        }
    }

    fn at(&self, lane: u32, col: u32) -> &Block {
        &self.blocks[(lane * self.lane_length + col) as usize]
    }

    fn at_mut(&mut self, lane: u32, col: u32) -> &mut Block {
        &mut self.blocks[(lane * self.lane_length + col) as usize]
    }

    /// Initialize columns 0 and 1 of every lane (RFC 9106 §3.2 step 5).
    fn init_first_two_columns(&mut self, h0: &[u8; BLAKE2B_OUT_MAX]) {
        let mut buf = [0u8; ARGON2_BLOCK_BYTES];
        for lane in 0..self.p_cost {
            // B[i][0] = H'(H_0 || LE32(0) || LE32(i))
            h_prime(
                ARGON2_BLOCK_BYTES as u32,
                &[h0, &0u32.to_le_bytes(), &lane.to_le_bytes()],
                &mut buf,
            );
            *self.at_mut(lane, 0) = Block::from_bytes(&buf);

            // B[i][1] = H'(H_0 || LE32(1) || LE32(i))
            h_prime(
                ARGON2_BLOCK_BYTES as u32,
                &[h0, &1u32.to_le_bytes(), &lane.to_le_bytes()],
                &mut buf,
            );
            *self.at_mut(lane, 1) = Block::from_bytes(&buf);
        }
    }

    /// Fill a single segment `(pass, lane, slice)`. RFC 9106 §3.4.
    fn fill_segment(&mut self, pass: u32, lane: u32, slice: u32) {
        let data_independent = pass == 0 && slice < (SYNC_POINTS as u32) / 2;
        let mut g2 = if data_independent {
            Some(G2Stream::new(
                pass,
                lane,
                slice,
                self.blocks_total,
                self.t_cost,
            ))
        } else {
            None
        };

        // Starting column within this segment.
        let start_col = if pass == 0 && slice == 0 { 2u32 } else { 0u32 };

        for i in start_col..self.segment_length {
            let curr_col = slice * self.segment_length + i;
            let prev_col = if curr_col == 0 {
                self.lane_length - 1
            } else {
                curr_col - 1
            };

            // Determine (J1, J2).
            let (j1, j2) = if let Some(stream) = g2.as_mut() {
                split_j(stream.j_at(i))
            } else {
                let prev = self.at(lane, prev_col);
                split_j(prev.0[0])
            };

            let (ref_lane, ref_col) = index_alpha(
                &Position {
                    pass,
                    lane,
                    slice,
                    index: i,
                },
                &Dimensions {
                    lane_length: self.lane_length,
                    segment_length: self.segment_length,
                    p_cost: self.p_cost,
                },
                j1,
                j2,
            );

            // Compute new block: G(B[lane][prev_col], B[ref_lane][ref_col]).
            let prev_block = *self.at(lane, prev_col);
            let ref_block = *self.at(ref_lane, ref_col);
            let mut new_block = Block::zero();
            argon2_g(&prev_block, &ref_block, &mut new_block);

            if pass > 0 {
                // Subsequent passes: XOR into existing block.
                let curr = self.at_mut(lane, curr_col);
                curr.xor_assign(&new_block);
            } else {
                *self.at_mut(lane, curr_col) = new_block;
            }
        }
    }

    /// Run all passes over the memory matrix. RFC 9106 §3.2 step 6.
    fn fill(&mut self) {
        for pass in 0..self.t_cost {
            for slice in 0..SYNC_POINTS as u32 {
                // RFC 9106 specifies lanes within a slice may be filled in
                // parallel. Sequential is conformant; parallelism is a
                // performance optimization deferred along with the SIMD
                // shims (Q30).
                for lane in 0..self.p_cost {
                    self.fill_segment(pass, lane, slice);
                }
            }
        }
    }

    /// Final block C = B[0][lane_length-1] xor ... xor B[p-1][lane_length-1].
    fn finalize_block(&self) -> Block {
        let mut c = *self.at(0, self.lane_length - 1);
        for lane in 1..self.p_cost {
            c.xor_assign(self.at(lane, self.lane_length - 1));
        }
        c
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Compute the Argon2id 32-byte tag.
///
/// `password` and `salt` are provided. RFC 9106 also accepts an optional
/// secret (`K`) and associated data (`X`); the management-login API does
/// not use either, so we pass empty slices. Use [`argon2id_hash_full`] for
/// the full RFC 9106 surface.
///
/// # Panics
///
/// Panics if parameters are invalid. Use [`Argon2idParams::validate`] or
/// [`argon2id_hash_full`] for a fallible variant.
pub fn argon2id_hash(
    password: &[u8],
    salt: &[u8],
    params: Argon2idParams,
) -> [u8; ARGON2ID_TAG_LEN] {
    argon2id_hash_full(password, salt, &[], &[], params, ARGON2ID_TAG_LEN)
        .expect("argon2id: invalid parameters or salt length")
        .try_into()
        .expect("argon2id: tag length mismatch")
}

/// Full Argon2id surface (RFC 9106): password, salt, optional secret,
/// optional associated data, parameters, and configurable tag length.
pub fn argon2id_hash_full(
    password: &[u8],
    salt: &[u8],
    secret: &[u8],
    associated: &[u8],
    params: Argon2idParams,
    tag_len: usize,
) -> Result<Vec<u8>, Error> {
    if salt.len() < 8 {
        return Err(Error::InvalidSalt);
    }
    if !(4..=BLAKE2B_OUT_MAX * 2).contains(&tag_len) {
        return Err(Error::InvalidTagLength);
    }
    params.validate()?;

    let h0 = h_zero(password, salt, secret, associated, &params, tag_len as u32);

    let mut mem = Memory::new(&params);
    mem.init_first_two_columns(&h0);
    mem.fill();
    let final_block = mem.finalize_block();

    let mut final_bytes = [0u8; ARGON2_BLOCK_BYTES];
    final_block.write_bytes(&mut final_bytes);

    let mut tag = vec![0u8; tag_len];
    h_prime(tag_len as u32, &[&final_bytes], &mut tag);
    Ok(tag)
}

/// Verify a password against a PHC-formatted Argon2id hash.
///
/// Returns `Ok(true)` if the password matches, `Ok(false)` if it does not,
/// or `Err` if the PHC string is malformed.
pub fn argon2id_verify(password: &[u8], phc_string: &str) -> Result<bool, Error> {
    let parsed = parse_phc(phc_string)?;
    let tag = argon2id_hash_full(
        password,
        &parsed.salt,
        &[],
        &[],
        parsed.params,
        parsed.tag.len(),
    )?;
    Ok(ct_eq(&tag, &parsed.tag).to_bool())
}

/// Format an Argon2id PHC string per `phc-string-format` v1.0:
///
/// ```text
/// $argon2id$v=19$m=<m>,t=<t>,p=<p>$<base64-salt>$<base64-tag>
/// ```
///
/// Base64 is the unpadded variant (`=` removed) per the PHC spec.
pub fn argon2id_format_phc(
    salt: &[u8],
    tag: &[u8; ARGON2ID_TAG_LEN],
    params: Argon2idParams,
) -> String {
    let mut s = String::new();
    s.push_str("$argon2id$v=19$m=");
    push_u32(&mut s, params.m_cost_kib);
    s.push_str(",t=");
    push_u32(&mut s, params.t_cost);
    s.push_str(",p=");
    push_u32(&mut s, params.p_cost);
    s.push('$');
    push_b64(&mut s, salt);
    s.push('$');
    push_b64(&mut s, tag);
    s
}

// ─── PHC parser ──────────────────────────────────────────────────────────────

struct ParsedPhc {
    params: Argon2idParams,
    salt: Vec<u8>,
    tag: Vec<u8>,
}

fn parse_phc(s: &str) -> Result<ParsedPhc, Error> {
    // $argon2id$v=19$m=<m>,t=<t>,p=<p>$<salt>$<tag>
    let mut parts = s.split('$');
    if parts.next() != Some("") {
        return Err(Error::InvalidPhc);
    }
    if parts.next() != Some("argon2id") {
        return Err(Error::UnsupportedVariant);
    }
    let v_part = parts.next().ok_or(Error::InvalidPhc)?;
    if v_part != "v=19" {
        return Err(Error::UnsupportedVersion);
    }
    let pp = parts.next().ok_or(Error::InvalidPhc)?;
    let salt_b64 = parts.next().ok_or(Error::InvalidPhc)?;
    let tag_b64 = parts.next().ok_or(Error::InvalidPhc)?;
    if parts.next().is_some() {
        return Err(Error::InvalidPhc);
    }

    // Parse "m=<m>,t=<t>,p=<p>". Allow any field order for PHC compatibility.
    let mut m: Option<u32> = None;
    let mut t: Option<u32> = None;
    let mut p: Option<u32> = None;
    for kv in pp.split(',') {
        let mut it = kv.splitn(2, '=');
        let k = it.next().ok_or(Error::InvalidPhc)?;
        let v = it.next().ok_or(Error::InvalidPhc)?;
        let n: u32 = parse_u32(v).ok_or(Error::InvalidPhc)?;
        match k {
            "m" => m = Some(n),
            "t" => t = Some(n),
            "p" => p = Some(n),
            _ => return Err(Error::InvalidPhc),
        }
    }

    let params = Argon2idParams {
        m_cost_kib: m.ok_or(Error::InvalidPhc)?,
        t_cost: t.ok_or(Error::InvalidPhc)?,
        p_cost: p.ok_or(Error::InvalidPhc)?,
    };
    params.validate()?;

    let salt = b64_decode(salt_b64)?;
    let tag = b64_decode(tag_b64)?;
    if salt.len() < 8 {
        return Err(Error::InvalidSalt);
    }
    if !(4..=BLAKE2B_OUT_MAX * 2).contains(&tag.len()) {
        return Err(Error::InvalidTagLength);
    }

    Ok(ParsedPhc { params, salt, tag })
}

fn parse_u32(s: &str) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    let mut n: u32 = 0; // lgtm[rust/hard-coded-cryptographic-value] integer accumulator, not a secret
    for c in s.chars() {
        let d = c.to_digit(10)?;
        n = n.checked_mul(10)?.checked_add(d)?;
    }
    Some(n)
}

fn push_u32(s: &mut String, mut n: u32) {
    if n == 0 {
        s.push('0');
        return;
    }
    let mut buf = [0u8; 10];
    let mut len = 0;
    while n > 0 {
        buf[len] = b'0' + (n % 10) as u8;
        n /= 10;
        len += 1;
    }
    for i in (0..len).rev() {
        s.push(buf[i] as char);
    }
}

// ─── Base64 (PHC unpadded) ───────────────────────────────────────────────────

const B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn push_b64(out: &mut String, input: &[u8]) {
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
        out.push(B64_CHARS[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((n >> 12) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((n >> 6) & 0x3F) as usize] as char);
        out.push(B64_CHARS[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(B64_CHARS[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((n >> 12) & 0x3F) as usize] as char);
    } else if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(B64_CHARS[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((n >> 12) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((n >> 6) & 0x3F) as usize] as char);
    }
}

fn b64_decode(s: &str) -> Result<Vec<u8>, Error> {
    // The constants 26, 52, 62, 63 are the offsets of the four character
    // ranges (A-Z, a-z, 0-9, +, /) in the standard base64 alphabet from
    // RFC 4648 Table 1. CodeQL's `rust/hard-coded-cryptographic-value` rule
    // misclassifies these as cryptographic secrets; they are not.
    fn val(c: u8) -> Result<u8, Error> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'), // lgtm[rust/hard-coded-cryptographic-value] RFC 4648 base64 alphabet, not a secret
            b'a'..=b'z' => Ok(c - b'a' + 26), // lgtm[rust/hard-coded-cryptographic-value] RFC 4648 base64 alphabet, not a secret
            b'0'..=b'9' => Ok(c - b'0' + 52), // lgtm[rust/hard-coded-cryptographic-value] RFC 4648 base64 alphabet, not a secret
            b'+' => Ok(62), // lgtm[rust/hard-coded-cryptographic-value] RFC 4648 base64 alphabet, not a secret
            b'/' => Ok(63), // lgtm[rust/hard-coded-cryptographic-value] RFC 4648 base64 alphabet, not a secret
            _ => Err(Error::Base64Decode),
        }
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let a = val(bytes[i])?;
        let b = val(bytes[i + 1])?;
        let c = val(bytes[i + 2])?;
        let d = val(bytes[i + 3])?;
        let n = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6) | (d as u32);
        out.push(((n >> 16) & 0xFF) as u8);
        out.push(((n >> 8) & 0xFF) as u8);
        out.push((n & 0xFF) as u8);
        i += 4;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        return Err(Error::Base64Decode);
    } else if rem == 2 {
        let a = val(bytes[i])?;
        let b = val(bytes[i + 1])?;
        let n = ((a as u32) << 18) | ((b as u32) << 12);
        out.push(((n >> 16) & 0xFF) as u8);
    } else if rem == 3 {
        let a = val(bytes[i])?;
        let b = val(bytes[i + 1])?;
        let c = val(bytes[i + 2])?;
        let n = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6);
        out.push(((n >> 16) & 0xFF) as u8);
        out.push(((n >> 8) & 0xFF) as u8);
    }
    Ok(out)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 9106 §5.3 Argon2id reference test vector.
    ///
    /// Parameters: t=3, m=32, p=4, version=0x13, type=Argon2id.
    /// Password: 32 bytes of 0x01.
    /// Salt: 16 bytes of 0x02.
    /// Secret: 8 bytes of 0x03.
    /// Associated data: 12 bytes of 0x04.
    /// Tag length: 32.
    /// Expected tag (RFC 9106 §5.3):
    ///   0d 64 0d f5 8d 78 76 6c  08 c0 37 a3 4a 8b 53 c9
    ///   d0 1e f0 45 2d 75 b6 5e  b5 25 20 e9 6b 01 e6 59
    #[test]
    fn rfc9106_section5_3_argon2id_kat() {
        // RFC 9106 §5.3 reference vector — deterministic test fixtures, not production secrets.
        let password = [0x01u8; 32]; // lgtm[rust/hard-coded-cryptographic-value] RFC 9106 §5.3 test vector
        let salt = [0x02u8; 16]; // lgtm[rust/hard-coded-cryptographic-value] RFC 9106 §5.3 test vector
        let secret = [0x03u8; 8]; // lgtm[rust/hard-coded-cryptographic-value] RFC 9106 §5.3 test vector
        let ad = [0x04u8; 12]; // lgtm[rust/hard-coded-cryptographic-value] RFC 9106 §5.3 test vector
        let params = Argon2idParams {
            m_cost_kib: 32,
            t_cost: 3,
            p_cost: 4,
        };
        let expected: [u8; 32] = [
            0x0D, 0x64, 0x0D, 0xF5, 0x8D, 0x78, 0x76, 0x6C, 0x08, 0xC0, 0x37, 0xA3, 0x4A, 0x8B,
            0x53, 0xC9, 0xD0, 0x1E, 0xF0, 0x45, 0x2D, 0x75, 0xB6, 0x5E, 0xB5, 0x25, 0x20, 0xE9,
            0x6B, 0x01, 0xE6, 0x59,
        ];
        let tag = argon2id_hash_full(&password, &salt, &secret, &ad, params, 32).unwrap();
        assert_eq!(tag.as_slice(), &expected[..]);
    }

    /// PHC round-trip: hash a password, format, parse, verify.
    #[test]
    fn phc_round_trip() {
        // Test-only fixtures, not production secrets.
        // Use the absolute-minimum tier to keep tests fast.
        let password = b"correct horse battery staple"; // lgtm[rust/hard-coded-cryptographic-value] test fixture
        let salt = b"saltsalt12345678"; // lgtm[rust/hard-coded-cryptographic-value] test fixture
        let params = Argon2idParams {
            m_cost_kib: 8,
            t_cost: 1,
            p_cost: 1,
        };
        let tag = argon2id_hash(password, salt, params);
        let phc = argon2id_format_phc(salt, &tag, params);

        // Format check.
        assert!(phc.starts_with("$argon2id$v=19$m=8,t=1,p=1$"));

        // Verify success.
        assert!(argon2id_verify(password, &phc).unwrap());

        // Different password fails.
        assert!(!argon2id_verify(b"wrong password", &phc).unwrap()); // lgtm[rust/hard-coded-cryptographic-value] negative-case test fixture
    }

    /// Tier constructors return valid parameters.
    #[test]
    fn tier_defaults_are_valid() {
        Argon2idParams::tiny().validate().unwrap();
        Argon2idParams::default_tier().validate().unwrap();
        Argon2idParams::strong().validate().unwrap();
    }

    /// Salt < 8 bytes is rejected.
    #[test]
    fn short_salt_rejected() {
        // Test-only Argon2id parameters, not a secret.
        let params = Argon2idParams {
            m_cost_kib: 8, // lgtm[rust/hard-coded-cryptographic-value] test parameter
            t_cost: 1,
            p_cost: 1,
        };
        let r = argon2id_hash_full(b"pw", b"short", &[], &[], params, 32);
        assert!(matches!(r, Err(Error::InvalidSalt)));
    }

    /// Cross-check against the upstream `argon2` crate (validation oracle).
    /// Uses very small parameters to keep the test fast.
    #[test]
    fn argon2_oracle_matches() {
        use argon2::{Algorithm, Argon2, Params, Version};

        for &(m, t, p) in &[(8u32, 1u32, 1u32), (16, 2, 1), (32, 3, 2)] {
            // Oracle-comparison test fixtures, not production secrets.
            let password = b"oracle-test-password"; // lgtm[rust/hard-coded-cryptographic-value] test fixture
            let salt = b"oracle-test-saltX"; // 17 bytes ≥ 8 — lgtm[rust/hard-coded-cryptographic-value] test fixture
            let params = Argon2idParams {
                m_cost_kib: m,
                t_cost: t,
                p_cost: p,
            };
            let mine = argon2id_hash(password, salt, params);

            let oracle_params = Params::new(m, t, p, Some(32)).unwrap();
            let oracle = Argon2::new(Algorithm::Argon2id, Version::V0x13, oracle_params);
            let mut oracle_out = [0u8; 32];
            oracle
                .hash_password_into(password, salt, &mut oracle_out)
                .unwrap();
            assert_eq!(mine, oracle_out, "mismatch m={} t={} p={}", m, t, p);
        }
    }

    /// PHC string from this implementation must verify under the upstream
    /// `argon2` crate so the on-disk format is interoperable.
    #[test]
    fn phc_interoperates_with_oracle() {
        use argon2::password_hash::{PasswordHash, PasswordVerifier};
        use argon2::Argon2;

        // PHC interop test fixtures, not production secrets.
        let password = b"interop-test"; // lgtm[rust/hard-coded-cryptographic-value] test fixture
        let salt = b"interop-saltAAAAAAAA"; // lgtm[rust/hard-coded-cryptographic-value] test fixture
        let params = Argon2idParams {
            m_cost_kib: 16,
            t_cost: 2,
            p_cost: 1,
        };
        let tag = argon2id_hash(password, salt, params);
        let phc = argon2id_format_phc(salt, &tag, params);

        let parsed = PasswordHash::new(&phc).expect("oracle should parse our PHC");
        let verifier = Argon2::default();
        verifier
            .verify_password(password, &parsed)
            .expect("oracle should verify our PHC");
    }

    /// Parameter sweep for the portable path: small grid of (m, t, p)
    /// must all complete without panicking and produce 32-byte tags.
    #[test]
    fn parameter_sweep() {
        // Parameter-sweep test fixtures, not production secrets.
        let password = b"sweep"; // lgtm[rust/hard-coded-cryptographic-value] test fixture
        let salt = b"sweepsalt0000000"; // lgtm[rust/hard-coded-cryptographic-value] test fixture
        for &m in &[8u32, 16, 32] {
            for &t in &[1u32, 2] {
                for &p in &[1u32, 2] {
                    if m < 8 * p {
                        continue;
                    }
                    let params = Argon2idParams {
                        m_cost_kib: m,
                        t_cost: t,
                        p_cost: p,
                    };
                    let tag = argon2id_hash(password, salt, params);
                    assert_eq!(tag.len(), 32);
                }
            }
        }
    }

    /// Base64 round-trip without padding.
    #[test]
    fn b64_round_trip() {
        for input in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"] {
            let mut s = String::new();
            push_b64(&mut s, input);
            let decoded = b64_decode(&s).unwrap();
            assert_eq!(&decoded[..], input);
        }
    }

    /// PHC parser rejects malformed inputs.
    #[test]
    fn phc_parser_rejects_bad_inputs() {
        assert!(matches!(
            parse_phc("$argon2id$v=19$m=8,t=1,p=1$YWFhYWFhYWE"),
            Err(Error::InvalidPhc)
        ));
        assert!(matches!(
            parse_phc("$argon2i$v=19$m=8,t=1,p=1$YWFhYWFhYWE$abcd"),
            Err(Error::UnsupportedVariant)
        ));
        assert!(matches!(
            parse_phc("$argon2id$v=18$m=8,t=1,p=1$YWFhYWFhYWE$abcd"),
            Err(Error::UnsupportedVersion)
        ));
    }
}
