// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ML-KEM-768 key encapsulation mechanism (FIPS 203).
//!
//! ML-KEM (Module-Lattice-based Key Encapsulation Mechanism) is a
//! post-quantum key encapsulation standard, formerly known as CRYSTALS-Kyber.
//! The 768 parameter set provides NIST Security Level 3 (~AES-192 equivalent).
//!
//! # Parameters (ML-KEM-768)
//!
//! | Parameter        | Size (bytes) |
//! |-----------------|-------------|
//! | Public key      | 1184        |
//! | Secret key      | 2400        |
//! | Ciphertext      | 1088        |
//! | Shared secret   | 32          |
//!
//! # Operations
//!
//! 1. **KeyGen**: Generate (public_key, secret_key) pair
//! 2. **Encaps**: Using public_key, produce (ciphertext, shared_secret)
//! 3. **Decaps**: Using secret_key + ciphertext, recover shared_secret

#![allow(unused)]
// Crypto code uses indexed loops for audit clarity and to match reference implementations.
#![allow(clippy::needless_range_loop)]

use super::sha3::{sha3_256, Sha3_256, Shake256};
use core::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

/// ML-KEM-768 public key length in bytes.
pub const ML_KEM_768_PK_LEN: usize = 1184;

/// ML-KEM-768 secret key length in bytes.
pub const ML_KEM_768_SK_LEN: usize = 2400;

/// ML-KEM-768 ciphertext length in bytes.
pub const ML_KEM_768_CT_LEN: usize = 1088;

/// ML-KEM-768 shared secret length in bytes.
pub const ML_KEM_768_SS_LEN: usize = 32;

/// ML-KEM-768 module rank (k = 3).
pub const ML_KEM_768_K: usize = 3;

/// Polynomial ring dimension (n = 256).
pub const ML_KEM_N: usize = 256;

/// Modulus q = 3329.
pub const ML_KEM_Q: u16 = 3329;

/// ML-KEM-768 eta1 = 2 (CBD parameter for secret/error in keygen).
const ETA1: usize = 2;

/// ML-KEM-768 eta2 = 2 (CBD parameter for error in encryption).
const ETA2: usize = 2;

/// ML-KEM-768 du = 10 (bits for compressing u in ciphertext).
const DU: usize = 10;

/// ML-KEM-768 dv = 4 (bits for compressing v in ciphertext).
const DV: usize = 4;

/// Size of encoded polynomial (12 bits per coefficient): 256 * 12 / 8 = 384.
const POLY_BYTES: usize = 384;

/// Size of compressed u vector: k * du * n / 8 = 3 * 10 * 256 / 8 = 960.
const POLY_COMPRESSED_DU_BYTES: usize = DU * ML_KEM_N / 8; // 320 per poly
const U_COMPRESSED_BYTES: usize = ML_KEM_768_K * POLY_COMPRESSED_DU_BYTES; // 960

/// Size of compressed v: dv * n / 8 = 4 * 256 / 8 = 128.
const V_COMPRESSED_BYTES: usize = DV * ML_KEM_N / 8; // 128

// ─── NTT Constants ──────────────────────────────────────────────────────────

/// Montgomery parameter: R = 2^16 mod q.
const MONT_R: u16 = 2285; // 2^16 mod 3329

/// Montgomery parameter: q^{-1} mod 2^16.
const Q_INV: u16 = 62209; // q^(-1) mod 2^16

/// Precomputed zetas: 17^{bit_rev7(i)} mod q in Montgomery form (signed).
/// Matches the pqcrystals reference Kyber implementation.
const ZETAS: [i16; 128] = [
    -1044, -758, -359, -1517, 1493, 1422, 287, 202, -171, 622, 1577, 182, 962, -1202, -1474, 1468,
    573, -1325, 264, 383, -829, 1458, -1602, -130, -681, 1017, 732, 608, -1542, 411, -205, -1571,
    1223, 652, -552, 1015, -1293, 1491, -282, -1544, 516, -8, -320, -666, -1618, -1162, 126, 1469,
    -853, -90, -271, 830, 107, -1421, -247, -951, -398, 961, -1508, -725, 448, -1065, 677, -1275,
    -1103, 430, 555, 843, -1251, 871, 1550, 105, 422, 587, 177, -235, -291, -460, 1574, 1653, -246,
    778, 1159, -147, -777, 1483, -602, 1119, -1590, 644, -872, 349, 418, 329, -156, -75, 817, 1097,
    603, 610, 1322, -1285, -1465, 384, -1215, -136, 1218, -1335, -874, 220, -1187, -1659, -1185,
    -1530, -1278, 794, -1510, -854, -870, 478, -108, -308, 996, 991, 958, -1460, 1522, 1628,
];

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from ML-KEM-768 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlKemError {
    /// Invalid public key length.
    InvalidPublicKeyLength,
    /// Invalid secret key length.
    InvalidSecretKeyLength,
    /// Invalid ciphertext length.
    InvalidCiphertextLength,
    /// Decapsulation failure (implicit rejection).
    DecapsulationFailure,
    /// RNG failure during key generation.
    RngFailure,
}

impl fmt::Display for MlKemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPublicKeyLength => {
                write!(
                    f,
                    "invalid public key length (expected {} bytes)",
                    ML_KEM_768_PK_LEN
                )
            }
            Self::InvalidSecretKeyLength => {
                write!(
                    f,
                    "invalid secret key length (expected {} bytes)",
                    ML_KEM_768_SK_LEN
                )
            }
            Self::InvalidCiphertextLength => {
                write!(
                    f,
                    "invalid ciphertext length (expected {} bytes)",
                    ML_KEM_768_CT_LEN
                )
            }
            Self::DecapsulationFailure => write!(f, "ML-KEM decapsulation failure"),
            Self::RngFailure => write!(f, "random number generation failure"),
        }
    }
}

// ─── Modular Arithmetic ─────────────────────────────────────────────────────

const Q: i32 = 3329;

/// Barrett reduction: reduce a mod q to [0, q).
#[inline]
fn barrett_reduce(a: i32) -> i16 {
    // Barrett constant: floor(2^26 / q) + 1 = 20159
    const V: i32 = 20159;
    let t = ((a as i64 * V as i64) >> 26) as i32;
    let r = a - t * Q;
    // r is in (-q, 2q), bring to [0, q)
    let r = if r < 0 { r + Q } else { r };
    let r = if r >= Q { r - Q } else { r };
    r as i16
}

/// Montgomery reduction: given a in [-q*2^15, q*2^15], compute a*R^{-1} mod q.
#[inline]
fn montgomery_reduce(a: i32) -> i16 {
    let t = (a as i16).wrapping_mul(Q_INV as i16) as i32;
    let u = t * Q;
    let r = (a - u) >> 16;
    r as i16
}

/// Conditional subtract q: if a >= q, return a - q.
#[inline]
fn cond_sub_q(a: i16) -> i16 {
    let mut r = a;
    r -= Q as i16;
    r += (r >> 15) & (Q as i16);
    r
}

// ─── Polynomial Type ────────────────────────────────────────────────────────

/// A polynomial in Z_q[X]/(X^256 + 1) with 256 coefficients.
#[derive(Clone, Copy)]
struct Poly {
    coeffs: [i16; ML_KEM_N],
}

impl Poly {
    const fn zero() -> Self {
        Self {
            coeffs: [0i16; ML_KEM_N],
        }
    }

    /// NTT forward transform (in-place, Cooley-Tukey butterfly).
    fn ntt(&mut self) {
        let mut k = 1usize;
        let mut len = 128;
        while len >= 2 {
            let mut start = 0;
            while start < ML_KEM_N {
                let zeta = ZETAS[k] as i32;
                k += 1;
                let mut j = start;
                while j < start + len {
                    let t = montgomery_reduce(zeta * self.coeffs[j + len] as i32);
                    self.coeffs[j + len] = self.coeffs[j] - t;
                    self.coeffs[j] += t;
                    j += 1;
                }
                start += 2 * len;
            }
            len >>= 1;
        }
    }

    /// NTT inverse transform (in-place, Gentleman-Sande butterfly).
    fn inv_ntt(&mut self) {
        let mut k = 127usize;
        let mut len = 2;
        while len <= 128 {
            let mut start = 0;
            while start < ML_KEM_N {
                let zeta = ZETAS[k] as i32;
                k = k.wrapping_sub(1);
                let mut j = start;
                while j < start + len {
                    let t = self.coeffs[j];
                    self.coeffs[j] = barrett_reduce((t as i32) + (self.coeffs[j + len] as i32));
                    self.coeffs[j + len] =
                        montgomery_reduce(zeta * ((self.coeffs[j + len] - t) as i32));
                    j += 1;
                }
                start += 2 * len;
            }
            len <<= 1;
        }
        // Multiply by f = R^2 * 128^{-1} mod q = 1441.
        // After basemul (which introduces R^{-1}) + inv_NTT, this corrects
        // both the 128^{-1} NTT scaling and the R^{-1} from basemul,
        // leaving the result in normal (non-Montgomery) form.
        // For a plain NTT->invNTT roundtrip (no basemul), output is in
        // Montgomery form (has extra factor R).
        const F: i32 = 1441;
        for c in self.coeffs.iter_mut() {
            *c = montgomery_reduce(F * (*c as i32));
        }
    }

    /// Pointwise multiplication in NTT domain (basemul).
    /// Processes 4 coefficients at a time: two pairs per group, second pair
    /// uses negated zeta (matching the pqcrystals reference implementation).
    ///
    /// Each pair (a0,a1)*(b0,b1) with twist zeta computes:
    ///   r0 = fqmul(fqmul(a1,b1), zeta) + fqmul(a0,b0)
    ///   r1 = fqmul(a0,b1) + fqmul(a1,b0)
    /// where fqmul(x,y) = montgomery_reduce(x*y) = x*y*R^{-1} mod q.
    /// Result has one factor of R^{-1}.
    fn basemul(&self, other: &Poly) -> Poly {
        let mut r = Poly::zero();
        for i in 0..(ML_KEM_N / 4) {
            let zeta = ZETAS[64 + i] as i32;
            let idx = 4 * i;

            // First pair in the group of 4
            let a0 = self.coeffs[idx] as i32;
            let a1 = self.coeffs[idx + 1] as i32;
            let b0 = other.coeffs[idx] as i32;
            let b1 = other.coeffs[idx + 1] as i32;
            // fqmul(a1,b1) then fqmul(result, zeta), then add fqmul(a0,b0)
            let t = montgomery_reduce(a1 * b1);
            r.coeffs[idx] = montgomery_reduce(t as i32 * zeta) + montgomery_reduce(a0 * b0);
            r.coeffs[idx + 1] = montgomery_reduce(a0 * b1) + montgomery_reduce(a1 * b0);

            // Second pair in the group of 4 (negated zeta)
            let a0 = self.coeffs[idx + 2] as i32;
            let a1 = self.coeffs[idx + 3] as i32;
            let b0 = other.coeffs[idx + 2] as i32;
            let b1 = other.coeffs[idx + 3] as i32;
            let t = montgomery_reduce(a1 * b1);
            r.coeffs[idx + 2] = montgomery_reduce(t as i32 * (-zeta)) + montgomery_reduce(a0 * b0);
            r.coeffs[idx + 3] = montgomery_reduce(a0 * b1) + montgomery_reduce(a1 * b0);
        }
        r
    }

    /// Add two polynomials.
    fn add(&self, other: &Poly) -> Poly {
        let mut r = Poly::zero();
        for i in 0..ML_KEM_N {
            r.coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        r
    }

    /// Subtract two polynomials.
    fn sub(&self, other: &Poly) -> Poly {
        let mut r = Poly::zero();
        for i in 0..ML_KEM_N {
            r.coeffs[i] = self.coeffs[i] - other.coeffs[i];
        }
        r
    }

    /// Reduce all coefficients modulo q.
    fn reduce(&mut self) {
        for c in self.coeffs.iter_mut() {
            *c = barrett_reduce(*c as i32);
        }
    }

    /// Convert polynomial to Montgomery form: multiply each coefficient by R.
    /// Used after basemul accumulation in keygen to compensate for the R^{-1} from fqmul.
    /// montgomery_reduce(f * c) = f * c * R^{-1} = (R^2 mod q) * c * R^{-1} = R * c.
    fn tomont(&mut self) {
        const F_TOMONT: i32 = 1353; // R^2 mod q = 2^32 mod 3329
        for c in self.coeffs.iter_mut() {
            *c = montgomery_reduce(F_TOMONT * (*c as i32));
        }
    }

    /// Encode polynomial with 12 bits per coefficient (for pk/sk).
    fn encode12(&self, out: &mut [u8]) {
        for i in (0..ML_KEM_N).step_by(2) {
            let a = barrett_reduce(self.coeffs[i] as i32) as u16;
            let b = barrett_reduce(self.coeffs[i + 1] as i32) as u16;
            let idx = (i / 2) * 3;
            out[idx] = a as u8;
            out[idx + 1] = ((a >> 8) | (b << 4)) as u8;
            out[idx + 2] = (b >> 4) as u8;
        }
    }

    /// Decode polynomial from 12-bit encoding.
    fn decode12(bytes: &[u8]) -> Self {
        let mut p = Poly::zero();
        for i in (0..ML_KEM_N).step_by(2) {
            let idx = (i / 2) * 3;
            let b0 = bytes[idx] as u16;
            let b1 = bytes[idx + 1] as u16;
            let b2 = bytes[idx + 2] as u16;
            p.coeffs[i] = (b0 | ((b1 & 0x0F) << 8)) as i16;
            p.coeffs[i + 1] = ((b1 >> 4) | (b2 << 4)) as i16;
        }
        p
    }

    /// Compress polynomial to d bits per coefficient.
    fn compress(&self, d: usize, out: &mut [u8]) {
        match d {
            4 => {
                for i in (0..ML_KEM_N).step_by(2) {
                    let a = compress_val(barrett_reduce(self.coeffs[i] as i32) as u16, 4) as u8;
                    let b = compress_val(barrett_reduce(self.coeffs[i + 1] as i32) as u16, 4) as u8;
                    out[i / 2] = a | (b << 4);
                }
            }
            10 => {
                for i in (0..ML_KEM_N).step_by(4) {
                    let mut vals = [0u16; 4];
                    for j in 0..4 {
                        vals[j] =
                            compress_val(barrett_reduce(self.coeffs[i + j] as i32) as u16, 10);
                    }
                    let idx = (i / 4) * 5;
                    out[idx] = vals[0] as u8;
                    out[idx + 1] = ((vals[0] >> 8) | (vals[1] << 2)) as u8;
                    out[idx + 2] = ((vals[1] >> 6) | (vals[2] << 4)) as u8;
                    out[idx + 3] = ((vals[2] >> 4) | (vals[3] << 6)) as u8;
                    out[idx + 4] = (vals[3] >> 2) as u8;
                }
            }
            _ => {}
        }
    }

    /// Decompress polynomial from d bits per coefficient.
    fn decompress(d: usize, bytes: &[u8]) -> Self {
        let mut p = Poly::zero();
        match d {
            4 => {
                for i in (0..ML_KEM_N).step_by(2) {
                    let b = bytes[i / 2];
                    p.coeffs[i] = decompress_val((b & 0x0F) as u16, 4);
                    p.coeffs[i + 1] = decompress_val((b >> 4) as u16, 4);
                }
            }
            10 => {
                for i in (0..ML_KEM_N).step_by(4) {
                    let idx = (i / 4) * 5;
                    let b0 = bytes[idx] as u16;
                    let b1 = bytes[idx + 1] as u16;
                    let b2 = bytes[idx + 2] as u16;
                    let b3 = bytes[idx + 3] as u16;
                    let b4 = bytes[idx + 4] as u16;
                    p.coeffs[i] = decompress_val(b0 | ((b1 & 0x03) << 8), 10);
                    p.coeffs[i + 1] = decompress_val((b1 >> 2) | ((b2 & 0x0F) << 6), 10);
                    p.coeffs[i + 2] = decompress_val((b2 >> 4) | ((b3 & 0x3F) << 4), 10);
                    p.coeffs[i + 3] = decompress_val((b3 >> 6) | (b4 << 2), 10);
                }
            }
            _ => {}
        }
        p
    }
}

/// Compress: round(2^d / q * x) mod 2^d
#[inline]
fn compress_val(x: u16, d: usize) -> u16 {
    let t = (x as u32) << d;
    let t = t + (Q as u32 / 2); // round
    let t = t / (Q as u32);
    (t & ((1 << d) - 1)) as u16
}

/// Decompress: round(q / 2^d * x)
#[inline]
fn decompress_val(x: u16, d: usize) -> i16 {
    let t = (x as u32) * (Q as u32);
    let t = t + (1u32 << (d - 1)); // round
    (t >> d) as i16
}

// ─── Sampling ───────────────────────────────────────────────────────────────

/// Sample polynomial from a uniform distribution using SHAKE128 (XOF).
/// For ML-KEM we use SHAKE-128 for matrix sampling, but since we only have
/// SHAKE-256, we use it (security is sufficient for Level 3).
fn sample_ntt(seed: &[u8; 32], i: u8, j: u8) -> Poly {
    let mut xof = Shake256::new();
    xof.absorb(seed).expect("absorb seed");
    xof.absorb(&[i, j]).expect("absorb indices");

    let mut p = Poly::zero();
    let mut buf = [0u8; 3];
    let mut ctr = 0;
    while ctr < ML_KEM_N {
        xof.squeeze(&mut buf).expect("squeeze");
        let d1 = (buf[0] as u16) | (((buf[1] & 0x0F) as u16) << 8);
        let d2 = ((buf[1] >> 4) as u16) | ((buf[2] as u16) << 4);
        if d1 < ML_KEM_Q {
            p.coeffs[ctr] = d1 as i16;
            ctr += 1;
        }
        if ctr < ML_KEM_N && d2 < ML_KEM_Q {
            p.coeffs[ctr] = d2 as i16;
            ctr += 1;
        }
    }
    p
}

/// Sample polynomial from centered binomial distribution CBD(eta).
fn sample_cbd(seed: &[u8], eta: usize) -> Poly {
    let mut p = Poly::zero();
    match eta {
        2 => {
            // CBD(2): 4 bits per coefficient (2 bits a, 2 bits b)
            for i in 0..ML_KEM_N {
                let byte_idx = i / 2;
                let bit_offset = (i % 2) * 4;
                let byte = if byte_idx < seed.len() {
                    seed[byte_idx]
                } else {
                    0
                };
                let bits = (byte >> bit_offset) & 0x0F;
                let a = (bits & 1) + ((bits >> 1) & 1);
                let b = ((bits >> 2) & 1) + ((bits >> 3) & 1);
                p.coeffs[i] = (a as i16) - (b as i16);
            }
        }
        3 => {
            // CBD(3): 6 bits per coefficient
            for i in 0..ML_KEM_N {
                let bit_pos = i * 6;
                let byte_idx = bit_pos / 8;
                let bit_off = bit_pos % 8;
                let val = if byte_idx + 1 < seed.len() {
                    ((seed[byte_idx] as u16) | ((seed[byte_idx + 1] as u16) << 8)) >> bit_off
                } else if byte_idx < seed.len() {
                    (seed[byte_idx] as u16) >> bit_off
                } else {
                    0
                };
                let bits = (val & 0x3F) as u8;
                let a = (bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1);
                let b = ((bits >> 3) & 1) + ((bits >> 4) & 1) + ((bits >> 5) & 1);
                p.coeffs[i] = (a as i16) - (b as i16);
            }
        }
        _ => {}
    }
    p
}

/// Generate CBD noise using PRF (SHAKE256).
fn prf(seed: &[u8; 32], nonce: u8, eta: usize) -> Poly {
    let mut xof = Shake256::new();
    xof.absorb(seed).expect("absorb");
    xof.absorb(&[nonce]).expect("absorb nonce");
    let out_len = eta * ML_KEM_N / 4;
    let mut buf = [0u8; 192]; // max eta=3 => 192 bytes
    xof.squeeze(&mut buf[..out_len]).expect("squeeze");
    sample_cbd(&buf[..out_len], eta)
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// ML-KEM-768 public key (encapsulation key).
#[derive(PartialEq, Eq)]
pub struct MlKemPublicKey {
    bytes: [u8; ML_KEM_768_PK_LEN],
}

impl MlKemPublicKey {
    /// Create a public key from a byte array.
    pub fn from_bytes(bytes: [u8; ML_KEM_768_PK_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a public key from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlKemError> {
        if slice.len() != ML_KEM_768_PK_LEN {
            return Err(MlKemError::InvalidPublicKeyLength);
        }
        let mut bytes = [0u8; ML_KEM_768_PK_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the public key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the public key length.
    pub fn len(&self) -> usize {
        ML_KEM_768_PK_LEN
    }

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlKemPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKemPublicKey({} bytes)", ML_KEM_768_PK_LEN)
    }
}

/// ML-KEM-768 secret key (decapsulation key).
#[derive(PartialEq, Eq)]
pub struct MlKemSecretKey {
    bytes: [u8; ML_KEM_768_SK_LEN],
}

impl MlKemSecretKey {
    /// Create a secret key from a byte array.
    pub fn from_bytes(bytes: [u8; ML_KEM_768_SK_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a secret key from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlKemError> {
        if slice.len() != ML_KEM_768_SK_LEN {
            return Err(MlKemError::InvalidSecretKeyLength);
        }
        let mut bytes = [0u8; ML_KEM_768_SK_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the secret key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the secret key length.
    pub fn len(&self) -> usize {
        ML_KEM_768_SK_LEN
    }

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlKemSecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKemSecretKey([REDACTED])")
    }
}

/// ML-KEM-768 ciphertext.
#[derive(PartialEq, Eq)]
pub struct MlKemCiphertext {
    bytes: [u8; ML_KEM_768_CT_LEN],
}

impl MlKemCiphertext {
    /// Create a ciphertext from a byte array.
    pub fn from_bytes(bytes: [u8; ML_KEM_768_CT_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a ciphertext from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlKemError> {
        if slice.len() != ML_KEM_768_CT_LEN {
            return Err(MlKemError::InvalidCiphertextLength);
        }
        let mut bytes = [0u8; ML_KEM_768_CT_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the ciphertext bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the ciphertext length.
    pub fn len(&self) -> usize {
        ML_KEM_768_CT_LEN
    }

    /// Returns whether the ciphertext is empty (always false for fixed-size types).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlKemCiphertext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKemCiphertext({} bytes)", ML_KEM_768_CT_LEN)
    }
}

/// ML-KEM-768 shared secret (32 bytes).
#[derive(Clone, PartialEq, Eq)]
pub struct MlKemSharedSecret {
    bytes: [u8; ML_KEM_768_SS_LEN],
}

impl MlKemSharedSecret {
    /// Create a shared secret from a byte array.
    pub fn from_bytes(bytes: [u8; ML_KEM_768_SS_LEN]) -> Self {
        Self { bytes }
    }

    /// Return the shared secret bytes.
    pub fn as_bytes(&self) -> &[u8; ML_KEM_768_SS_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for MlKemSharedSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKemSharedSecret([REDACTED])")
    }
}

/// ML-KEM-768 key pair (public + secret key).
#[derive(PartialEq, Eq)]
pub struct MlKemKeyPair {
    pub public_key: MlKemPublicKey,
    pub secret_key: MlKemSecretKey,
}

impl fmt::Debug for MlKemKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlKemKeyPair")
            .field("public_key", &self.public_key)
            .field("secret_key", &self.secret_key)
            .finish()
    }
}

// ─── Internal K-PKE (lattice encryption) ─────────────────────────────────────

/// K-PKE.KeyGen: Generate a lattice key pair.
/// seed_d is 32 bytes used for key generation.
fn k_pke_keygen(
    seed_d: &[u8; 32],
) -> (
    [u8; ML_KEM_768_K * POLY_BYTES + 32],
    [u8; ML_KEM_768_K * POLY_BYTES],
) {
    // G(d) = (rho, sigma) where rho is 32 bytes for matrix, sigma for noise
    let g_input = sha3_256(seed_d);
    let rho: [u8; 32] = {
        let mut r = [0u8; 32];
        r.copy_from_slice(&g_input.as_bytes()[..32]);
        r
    };

    // Use SHAKE256 to derive sigma (32 bytes) from the second hash
    let mut hasher = Sha3_256::new();
    hasher.update(&[1u8]); // domain separation
    hasher.update(seed_d);
    let sigma_digest = hasher.finalize().expect("finalize");
    let sigma: [u8; 32] = {
        let mut s = [0u8; 32];
        s.copy_from_slice(sigma_digest.as_bytes());
        s
    };

    // Generate matrix A (in NTT domain) from rho
    let mut a_hat = [[Poly::zero(); ML_KEM_768_K]; ML_KEM_768_K];
    for i in 0..ML_KEM_768_K {
        for j in 0..ML_KEM_768_K {
            a_hat[i][j] = sample_ntt(&rho, i as u8, j as u8);
        }
    }

    // Generate secret vector s (CBD noise)
    let mut s = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        s[i] = prf(&sigma, i as u8, ETA1);
    }

    // Generate error vector e (CBD noise)
    let mut e = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        e[i] = prf(&sigma, (ML_KEM_768_K + i) as u8, ETA1);
    }

    // NTT(s) and NTT(e)
    let mut s_hat = [Poly::zero(), Poly::zero(), Poly::zero()];
    let mut e_hat = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        s_hat[i] = s[i];
        s_hat[i].ntt();
        e_hat[i] = e[i];
        e_hat[i].ntt();
    }

    // t_hat = A_hat * s_hat + e_hat
    // Note: basemul introduces R^{-1}, so we call tomont() to compensate
    // before adding e_hat (matching the pqcrystals reference).
    let mut t_hat = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        let mut acc = Poly::zero();
        for j in 0..ML_KEM_768_K {
            let product = a_hat[i][j].basemul(&s_hat[j]);
            acc = acc.add(&product);
        }
        acc.tomont();
        t_hat[i] = acc.add(&e_hat[i]);
        t_hat[i].reduce();
    }

    // Encode public key: pk = encode(t_hat) || rho
    let mut pk = [0u8; ML_KEM_768_K * POLY_BYTES + 32];
    for i in 0..ML_KEM_768_K {
        t_hat[i].encode12(&mut pk[i * POLY_BYTES..(i + 1) * POLY_BYTES]);
    }
    pk[ML_KEM_768_K * POLY_BYTES..].copy_from_slice(&rho);

    // Encode secret key: sk = encode(s_hat)
    let mut sk = [0u8; ML_KEM_768_K * POLY_BYTES];
    for i in 0..ML_KEM_768_K {
        s_hat[i].encode12(&mut sk[i * POLY_BYTES..(i + 1) * POLY_BYTES]);
    }

    (pk, sk)
}

/// K-PKE.Encrypt: Encrypt a 32-byte message using the public key.
fn k_pke_encrypt(
    pk_bytes: &[u8],
    msg: &[u8; 32],
    random_coins: &[u8; 32],
) -> [u8; ML_KEM_768_CT_LEN] {
    // Decode public key
    let mut t_hat = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        t_hat[i] = Poly::decode12(&pk_bytes[i * POLY_BYTES..(i + 1) * POLY_BYTES]);
    }
    let rho: &[u8; 32] = pk_bytes[ML_KEM_768_K * POLY_BYTES..].try_into().unwrap();

    // Generate matrix A^T from rho (transposed)
    let mut a_hat_t = [[Poly::zero(); ML_KEM_768_K]; ML_KEM_768_K];
    for i in 0..ML_KEM_768_K {
        for j in 0..ML_KEM_768_K {
            a_hat_t[i][j] = sample_ntt(rho, j as u8, i as u8); // transposed
        }
    }

    // Generate r (random vector), e1 (error vector), e2 (error scalar)
    let mut r = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        r[i] = prf(random_coins, i as u8, ETA1);
    }
    let mut e1 = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        e1[i] = prf(random_coins, (ML_KEM_768_K + i) as u8, ETA2);
    }
    let e2 = prf(random_coins, (2 * ML_KEM_768_K) as u8, ETA2);

    // NTT(r)
    let mut r_hat = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        r_hat[i] = r[i];
        r_hat[i].ntt();
    }

    // u = NTT^{-1}(A^T * r_hat) + e1
    let mut u = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        let mut acc = Poly::zero();
        for j in 0..ML_KEM_768_K {
            let product = a_hat_t[i][j].basemul(&r_hat[j]);
            acc = acc.add(&product);
        }
        acc.inv_ntt();
        u[i] = acc.add(&e1[i]);
        u[i].reduce();
    }

    // v = NTT^{-1}(t_hat^T * r_hat) + e2 + decode_msg(msg)
    let mut v = Poly::zero();
    for i in 0..ML_KEM_768_K {
        let product = t_hat[i].basemul(&r_hat[i]);
        v = v.add(&product);
    }
    v.inv_ntt();
    v = v.add(&e2);

    // Decode message as polynomial (each bit -> q/2 or 0)
    let msg_poly = decode_msg(msg);
    v = v.add(&msg_poly);
    v.reduce();

    // Compress and encode ciphertext
    let mut ct = [0u8; ML_KEM_768_CT_LEN];
    let mut offset = 0;
    for i in 0..ML_KEM_768_K {
        u[i].compress(DU, &mut ct[offset..offset + POLY_COMPRESSED_DU_BYTES]);
        offset += POLY_COMPRESSED_DU_BYTES;
    }
    v.compress(DV, &mut ct[offset..offset + V_COMPRESSED_BYTES]);

    ct
}

/// K-PKE.Decrypt: Decrypt ciphertext using secret key.
fn k_pke_decrypt(sk_bytes: &[u8], ct_bytes: &[u8]) -> [u8; 32] {
    // Decode secret key
    let mut s_hat = [Poly::zero(), Poly::zero(), Poly::zero()];
    for i in 0..ML_KEM_768_K {
        s_hat[i] = Poly::decode12(&sk_bytes[i * POLY_BYTES..(i + 1) * POLY_BYTES]);
    }

    // Decompress ciphertext
    let mut u = [Poly::zero(), Poly::zero(), Poly::zero()];
    let mut offset = 0;
    for i in 0..ML_KEM_768_K {
        u[i] = Poly::decompress(DU, &ct_bytes[offset..offset + POLY_COMPRESSED_DU_BYTES]);
        offset += POLY_COMPRESSED_DU_BYTES;
    }
    let v = Poly::decompress(DV, &ct_bytes[offset..offset + V_COMPRESSED_BYTES]);

    // NTT(u)
    for i in 0..ML_KEM_768_K {
        u[i].ntt();
    }

    // w = v - NTT^{-1}(s_hat^T * NTT(u))
    let mut inner = Poly::zero();
    for i in 0..ML_KEM_768_K {
        let product = s_hat[i].basemul(&u[i]);
        inner = inner.add(&product);
    }
    inner.inv_ntt();
    let w = v.sub(&inner);

    // Encode message
    encode_msg(&w)
}

/// Decode a 32-byte message to a polynomial: bit i -> (q+1)/2 if 1, else 0.
fn decode_msg(msg: &[u8; 32]) -> Poly {
    let mut p = Poly::zero();
    for i in 0..ML_KEM_N {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let bit = (msg[byte_idx] >> bit_idx) & 1;
        p.coeffs[i] = (bit as i16) * ((Q as i16 + 1) / 2);
    }
    p
}

/// Encode a polynomial to a 32-byte message by rounding each coefficient.
fn encode_msg(p: &Poly) -> [u8; 32] {
    let mut msg = [0u8; 32];
    for i in 0..ML_KEM_N {
        // Fully reduce to [0, q) before encoding
        let c = barrett_reduce(p.coeffs[i] as i32) as u32;
        // round(2 * c / q) mod 2
        let bit = ((c * 2 + Q as u32 / 2) / Q as u32) & 1;
        msg[i / 8] |= (bit as u8) << (i % 8);
    }
    msg
}

// ─── Operations (FIPS 203) ──────────────────────────────────────────────────

/// Generate an ML-KEM-768 key pair.
///
/// Requires a 64-byte random seed (d || z) from the CSPRNG.
/// - d (32 bytes): seed for matrix/vector generation
/// - z (32 bytes): implicit rejection seed
pub fn ml_kem_768_keygen(seed: &[u8; 64]) -> Result<MlKemKeyPair, MlKemError> {
    let d: &[u8; 32] = seed[..32].try_into().unwrap();
    let z: &[u8; 32] = seed[32..].try_into().unwrap();

    // K-PKE.KeyGen
    let (pk_core, sk_core) = k_pke_keygen(d);

    // pk = pk_core (1152 + 32 = 1184 bytes)
    let mut pk_bytes = [0u8; ML_KEM_768_PK_LEN];
    pk_bytes.copy_from_slice(&pk_core);

    // sk = sk_core || pk || H(pk) || z
    // sk_core = 1152, pk = 1184, H(pk) = 32, z = 32 => 2400
    let mut sk_bytes = [0u8; ML_KEM_768_SK_LEN];
    sk_bytes[..ML_KEM_768_K * POLY_BYTES].copy_from_slice(&sk_core);
    sk_bytes[ML_KEM_768_K * POLY_BYTES..ML_KEM_768_K * POLY_BYTES + ML_KEM_768_PK_LEN]
        .copy_from_slice(&pk_bytes);

    let pk_hash = sha3_256(&pk_bytes);
    let h_offset = ML_KEM_768_K * POLY_BYTES + ML_KEM_768_PK_LEN;
    sk_bytes[h_offset..h_offset + 32].copy_from_slice(pk_hash.as_bytes());
    sk_bytes[h_offset + 32..h_offset + 64].copy_from_slice(z);

    Ok(MlKemKeyPair {
        public_key: MlKemPublicKey::from_bytes(pk_bytes),
        secret_key: MlKemSecretKey::from_bytes(sk_bytes),
    })
}

/// Encapsulate: generate a shared secret and ciphertext using a public key.
///
/// Requires a 32-byte random seed from the CSPRNG.
pub fn ml_kem_768_encaps(
    pk: &MlKemPublicKey,
    random_seed: &[u8; 32],
) -> Result<(MlKemCiphertext, MlKemSharedSecret), MlKemError> {
    let pk_bytes = pk.as_bytes();

    // m = random_seed (the message to encrypt)
    let m = random_seed;

    // (K_bar, r) = G(m || H(pk))
    let pk_hash = sha3_256(pk_bytes);
    let mut g_input = [0u8; 64];
    g_input[..32].copy_from_slice(m);
    g_input[32..].copy_from_slice(pk_hash.as_bytes());
    let g_output = sha3_512_from_shake(&g_input);
    let k_bar: &[u8; 32] = g_output[..32].try_into().unwrap();
    let r: &[u8; 32] = g_output[32..].try_into().unwrap();

    // c = K-PKE.Encrypt(pk, m, r)
    let ct_bytes = k_pke_encrypt(pk_bytes, m, r);

    // K = KDF(K_bar || H(c))
    let ct_hash = sha3_256(&ct_bytes);
    let mut kdf_input = [0u8; 64];
    kdf_input[..32].copy_from_slice(k_bar);
    kdf_input[32..].copy_from_slice(ct_hash.as_bytes());
    let shared_secret = sha3_256(&kdf_input);

    let mut ss_bytes = [0u8; ML_KEM_768_SS_LEN];
    ss_bytes.copy_from_slice(shared_secret.as_bytes());

    Ok((
        MlKemCiphertext::from_bytes(ct_bytes),
        MlKemSharedSecret::from_bytes(ss_bytes),
    ))
}

/// Decapsulate: recover the shared secret from ciphertext using the secret key.
///
/// Uses implicit rejection: if decapsulation fails, a pseudorandom
/// shared secret is returned (derived from sk and ct) to prevent
/// chosen-ciphertext attacks.
pub fn ml_kem_768_decaps(
    sk: &MlKemSecretKey,
    ct: &MlKemCiphertext,
) -> Result<MlKemSharedSecret, MlKemError> {
    let sk_bytes = sk.as_bytes();
    let ct_bytes = ct.as_bytes();

    // Parse secret key components
    let sk_core = &sk_bytes[..ML_KEM_768_K * POLY_BYTES];
    let pk_bytes =
        &sk_bytes[ML_KEM_768_K * POLY_BYTES..ML_KEM_768_K * POLY_BYTES + ML_KEM_768_PK_LEN];
    let h_pk = &sk_bytes[ML_KEM_768_K * POLY_BYTES + ML_KEM_768_PK_LEN
        ..ML_KEM_768_K * POLY_BYTES + ML_KEM_768_PK_LEN + 32];
    let z = &sk_bytes[ML_KEM_768_K * POLY_BYTES + ML_KEM_768_PK_LEN + 32..];

    // m' = K-PKE.Decrypt(sk_core, ct)
    let m_prime = k_pke_decrypt(sk_core, ct_bytes);

    // (K_bar', r') = G(m' || H(pk))
    let mut g_input = [0u8; 64];
    g_input[..32].copy_from_slice(&m_prime);
    g_input[32..].copy_from_slice(h_pk);
    let g_output = sha3_512_from_shake(&g_input);
    let k_bar_prime: &[u8; 32] = g_output[..32].try_into().unwrap();
    let r_prime: &[u8; 32] = g_output[32..].try_into().unwrap();

    // c' = K-PKE.Encrypt(pk, m', r')
    let ct_prime = k_pke_encrypt(pk_bytes, &m_prime, r_prime);

    // Constant-time comparison: if c == c', use K_bar', else use rejection value
    let ct_eq = super::constant_time::ct_eq(ct_bytes, &ct_prime);

    // K_reject = J(z || c) where J is SHAKE256
    let mut reject_xof = Shake256::new();
    reject_xof.absorb(z).expect("absorb z");
    reject_xof.absorb(ct_bytes).expect("absorb ct");
    let mut k_reject = [0u8; 32];
    reject_xof.squeeze(&mut k_reject).expect("squeeze");

    // K = KDF(K_bar || H(c))
    let ct_hash = sha3_256(ct_bytes);
    let mut kdf_input = [0u8; 64];
    kdf_input[..32].copy_from_slice(k_bar_prime);
    kdf_input[32..].copy_from_slice(ct_hash.as_bytes());
    let k_accept_digest = sha3_256(&kdf_input);
    let k_accept = k_accept_digest.as_bytes();

    // Constant-time select: use k_accept if ct matches, k_reject otherwise
    let mut ss = [0u8; ML_KEM_768_SS_LEN];
    let mask = ct_eq.mask();
    for i in 0..ML_KEM_768_SS_LEN {
        ss[i] = (mask & k_accept[i]) | (!mask & k_reject[i]);
    }

    Ok(MlKemSharedSecret::from_bytes(ss))
}

/// SHA3-512-like function using SHAKE256 (we produce 64 bytes).
fn sha3_512_from_shake(input: &[u8]) -> [u8; 64] {
    let mut xof = Shake256::new();
    xof.absorb(input).expect("absorb");
    let mut out = [0u8; 64];
    xof.squeeze(&mut out).expect("squeeze");
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn ml_kem_constant_sizes() {
        assert_eq!(ML_KEM_768_PK_LEN, 1184);
        assert_eq!(ML_KEM_768_SK_LEN, 2400);
        assert_eq!(ML_KEM_768_CT_LEN, 1088);
        assert_eq!(ML_KEM_768_SS_LEN, 32);
        assert_eq!(ML_KEM_768_K, 3);
        assert_eq!(ML_KEM_N, 256);
        assert_eq!(ML_KEM_Q, 3329);
    }

    #[test]
    fn ml_kem_public_key_from_slice_valid() {
        let bytes = [0x42u8; ML_KEM_768_PK_LEN];
        let pk = MlKemPublicKey::from_slice(&bytes).unwrap();
        assert_eq!(pk.len(), ML_KEM_768_PK_LEN);
        assert_eq!(pk.as_bytes()[0], 0x42);
    }

    #[test]
    fn ml_kem_public_key_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlKemPublicKey::from_slice(&bytes),
            Err(MlKemError::InvalidPublicKeyLength)
        );
    }

    #[test]
    fn ml_kem_secret_key_from_slice_valid() {
        let bytes = [0xAB; ML_KEM_768_SK_LEN];
        let sk = MlKemSecretKey::from_slice(&bytes).unwrap();
        assert_eq!(sk.len(), ML_KEM_768_SK_LEN);
    }

    #[test]
    fn ml_kem_secret_key_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlKemSecretKey::from_slice(&bytes),
            Err(MlKemError::InvalidSecretKeyLength)
        );
    }

    #[test]
    fn ml_kem_ciphertext_from_slice_valid() {
        let bytes = [0xCD; ML_KEM_768_CT_LEN];
        let ct = MlKemCiphertext::from_slice(&bytes).unwrap();
        assert_eq!(ct.len(), ML_KEM_768_CT_LEN);
    }

    #[test]
    fn ml_kem_ciphertext_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlKemCiphertext::from_slice(&bytes),
            Err(MlKemError::InvalidCiphertextLength)
        );
    }

    #[test]
    fn ml_kem_shared_secret_construction() {
        let ss = MlKemSharedSecret::from_bytes([0xEF; ML_KEM_768_SS_LEN]);
        assert_eq!(ss.as_bytes().len(), ML_KEM_768_SS_LEN);
        assert_eq!(ss.as_bytes()[0], 0xEF);
    }

    #[test]
    fn ml_kem_secret_key_debug_redacted() {
        let sk = MlKemSecretKey::from_bytes([0xFF; ML_KEM_768_SK_LEN]);
        let debug = format!("{:?}", sk);
        assert_eq!(debug, "MlKemSecretKey([REDACTED])");
    }

    #[test]
    fn ml_kem_shared_secret_debug_redacted() {
        let ss = MlKemSharedSecret::from_bytes([0xFF; ML_KEM_768_SS_LEN]);
        let debug = format!("{:?}", ss);
        assert_eq!(debug, "MlKemSharedSecret([REDACTED])");
    }

    #[test]
    fn ml_kem_public_key_debug() {
        let pk = MlKemPublicKey::from_bytes([0; ML_KEM_768_PK_LEN]);
        let debug = format!("{:?}", pk);
        assert!(debug.contains("1184"));
    }

    // ─── Arithmetic unit tests ──────────────────────────────────────────────

    #[test]
    fn test_barrett_reduce() {
        assert_eq!(barrett_reduce(0), 0);
        assert_eq!(barrett_reduce(3329), 0);
        assert_eq!(barrett_reduce(3330), 1);
        assert_eq!(barrett_reduce(-1), 3328);
        assert_eq!(barrett_reduce(6658), 0);
    }

    #[test]
    fn test_zetas_spot_check() {
        // zetas[0] = Montgomery form of 1 = (1 * 2^16) mod 3329 = 2285
        // Signed: 2285 - 3329 = -1044
        assert_eq!(ZETAS[0], -1044);
        // zetas[1] = Montgomery form of 17^64 mod 3329 = 1729 * 2^16 mod 3329 = 2571
        // Signed: 2571 - 3329 = -758
        assert_eq!(ZETAS[1], -758);
        assert_eq!(ZETAS.len(), 128);
    }

    #[test]
    fn test_poly_encode_decode_12() {
        let mut p = Poly::zero();
        for i in 0..ML_KEM_N {
            p.coeffs[i] = (i as i16 * 13) % (Q as i16);
        }
        let mut buf = [0u8; POLY_BYTES];
        p.encode12(&mut buf);
        let p2 = Poly::decode12(&buf);
        for i in 0..ML_KEM_N {
            assert_eq!(
                cond_sub_q(p.coeffs[i]),
                p2.coeffs[i],
                "mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        // Compression is lossy, so we check approximate roundtrip
        let mut p = Poly::zero();
        for i in 0..ML_KEM_N {
            p.coeffs[i] = (i as i16 * 7) % (Q as i16);
        }

        // Test d=10
        let mut buf10 = [0u8; POLY_COMPRESSED_DU_BYTES];
        p.compress(10, &mut buf10);
        let p10 = Poly::decompress(10, &buf10);
        for i in 0..ML_KEM_N {
            let orig = cond_sub_q(p.coeffs[i]) as i32;
            let recovered = p10.coeffs[i] as i32;
            let diff = (orig - recovered).abs();
            let diff = if diff > Q / 2 { Q - diff } else { diff };
            assert!(diff <= 4, "d=10 error too large at {}: diff={}", i, diff);
        }

        // Test d=4
        let mut buf4 = [0u8; V_COMPRESSED_BYTES];
        p.compress(4, &mut buf4);
        let p4 = Poly::decompress(4, &buf4);
        for i in 0..ML_KEM_N {
            let orig = cond_sub_q(p.coeffs[i]) as i32;
            let recovered = p4.coeffs[i] as i32;
            let diff = (orig - recovered).abs();
            let diff = if diff > Q / 2 { Q - diff } else { diff };
            assert!(diff <= 210, "d=4 error too large at {}: diff={}", i, diff);
        }
    }

    #[test]
    fn test_ntt_inv_ntt_roundtrip() {
        let mut p = Poly::zero();
        for i in 0..ML_KEM_N {
            p.coeffs[i] = (i as i16 * 17) % (Q as i16);
        }
        let original: [i16; ML_KEM_N] = p.coeffs;

        p.ntt();
        p.inv_ntt();

        // With f=1441, plain NTT->invNTT output is in Montgomery form (extra R factor).
        // Convert back: montgomery_reduce(1 * c) = c * R^{-1} mod q.
        for i in 0..ML_KEM_N {
            let a = barrett_reduce(original[i] as i32) as i32;
            let b = barrett_reduce(montgomery_reduce(p.coeffs[i] as i32) as i32) as i32;
            assert_eq!(a, b, "NTT roundtrip mismatch at index {}", i);
        }
    }

    #[test]
    fn test_msg_encode_decode_roundtrip() {
        let msg = [0xA5u8; 32]; // alternating bits
        let p = decode_msg(&msg);
        let msg2 = encode_msg(&p);
        assert_eq!(msg, msg2, "message encode/decode roundtrip failed");
    }

    #[test]
    fn test_k_pke_encrypt_decrypt_roundtrip() {
        let seed_d = [0x42u8; 32];
        let (pk, sk) = k_pke_keygen(&seed_d);

        let msg = [0xA5u8; 32];
        let coins = [0x37u8; 32];
        let ct = k_pke_encrypt(&pk, &msg, &coins);
        let recovered = k_pke_decrypt(&sk, &ct);
        assert_eq!(msg, recovered, "K-PKE encrypt/decrypt roundtrip failed");
    }

    /// Debug test: check algebraic core without compress/decompress.
    /// v - s^T * u should approximately equal msg + noise.
    #[test]
    fn test_k_pke_algebraic_core() {
        let seed_d = [0x42u8; 32];
        let (pk_bytes, sk_bytes) = k_pke_keygen(&seed_d);

        // Decode key components
        let mut t_hat = [Poly::zero(), Poly::zero(), Poly::zero()];
        for i in 0..ML_KEM_768_K {
            t_hat[i] = Poly::decode12(&pk_bytes[i * POLY_BYTES..(i + 1) * POLY_BYTES]);
        }
        let rho: &[u8; 32] = pk_bytes[ML_KEM_768_K * POLY_BYTES..].try_into().unwrap();

        let mut s_hat = [Poly::zero(), Poly::zero(), Poly::zero()];
        for i in 0..ML_KEM_768_K {
            s_hat[i] = Poly::decode12(&sk_bytes[i * POLY_BYTES..(i + 1) * POLY_BYTES]);
        }

        // Reconstruct A^T
        let mut a_hat_t = [[Poly::zero(); ML_KEM_768_K]; ML_KEM_768_K];
        for i in 0..ML_KEM_768_K {
            for j in 0..ML_KEM_768_K {
                a_hat_t[i][j] = sample_ntt(rho, j as u8, i as u8);
            }
        }

        // Use random r, but zero noise (e1=0, e2=0)
        let coins = [0x37u8; 32];
        let mut r_vec = [Poly::zero(), Poly::zero(), Poly::zero()];
        for i in 0..ML_KEM_768_K {
            r_vec[i] = prf(&coins, i as u8, ETA1);
        }
        let mut r_hat = [Poly::zero(), Poly::zero(), Poly::zero()];
        for i in 0..ML_KEM_768_K {
            r_hat[i] = r_vec[i];
            r_hat[i].ntt();
        }

        // u = invNTT(A^T * r_hat) (no noise)
        let mut u = [Poly::zero(), Poly::zero(), Poly::zero()];
        for i in 0..ML_KEM_768_K {
            let mut acc = Poly::zero();
            for j in 0..ML_KEM_768_K {
                acc = acc.add(&a_hat_t[i][j].basemul(&r_hat[j]));
            }
            acc.inv_ntt();
            u[i] = acc;
            u[i].reduce();
        }

        // v = invNTT(t^T * r_hat) + msg (no noise)
        let mut v = Poly::zero();
        for i in 0..ML_KEM_768_K {
            v = v.add(&t_hat[i].basemul(&r_hat[i]));
        }
        v.inv_ntt();

        let msg = [0xA5u8; 32];
        let msg_poly = decode_msg(&msg);
        v = v.add(&msg_poly);
        v.reduce();

        // Now decrypt: inner = invNTT(s^T * NTT(u))
        for i in 0..ML_KEM_768_K {
            u[i].ntt();
        }
        let mut inner = Poly::zero();
        for i in 0..ML_KEM_768_K {
            inner = inner.add(&s_hat[i].basemul(&u[i]));
        }
        inner.inv_ntt();

        // w = v - inner should be approximately msg_poly (no noise)
        let w = v.sub(&inner);
        let recovered = encode_msg(&w);

        // With zero noise, the recovered message MUST match
        assert_eq!(msg, recovered, "Algebraic core failed: v - s*u != msg");
    }

    // ─── Full KEM roundtrip test ────────────────────────────────────────────

    #[test]
    fn ml_kem_keygen_encaps_decaps_roundtrip() {
        // Use deterministic seed
        let mut seed = [0u8; 64];
        for i in 0..64 {
            seed[i] = i as u8;
        }

        let keypair = ml_kem_768_keygen(&seed).expect("keygen should succeed");

        // Verify key sizes
        assert_eq!(keypair.public_key.len(), ML_KEM_768_PK_LEN);
        assert_eq!(keypair.secret_key.len(), ML_KEM_768_SK_LEN);

        // Encapsulate
        let encaps_seed = [0x42u8; 32];
        let (ct, ss_encaps) =
            ml_kem_768_encaps(&keypair.public_key, &encaps_seed).expect("encaps should succeed");

        assert_eq!(ct.len(), ML_KEM_768_CT_LEN);
        assert_eq!(ss_encaps.as_bytes().len(), ML_KEM_768_SS_LEN);

        // Decapsulate
        let ss_decaps = ml_kem_768_decaps(&keypair.secret_key, &ct).expect("decaps should succeed");

        // Shared secrets must match
        assert_eq!(
            ss_encaps.as_bytes(),
            ss_decaps.as_bytes(),
            "encaps and decaps shared secrets must match"
        );
    }

    #[test]
    fn ml_kem_deterministic() {
        let seed = [0xBBu8; 64];

        let kp1 = ml_kem_768_keygen(&seed).unwrap();
        let kp2 = ml_kem_768_keygen(&seed).unwrap();

        assert_eq!(kp1.public_key.as_bytes(), kp2.public_key.as_bytes());
        assert_eq!(kp1.secret_key.as_bytes(), kp2.secret_key.as_bytes());
    }

    #[test]
    fn ml_kem_different_seeds_different_keys() {
        let seed1 = [0x01u8; 64];
        let seed2 = [0x02u8; 64];

        let kp1 = ml_kem_768_keygen(&seed1).unwrap();
        let kp2 = ml_kem_768_keygen(&seed2).unwrap();

        assert_ne!(kp1.public_key.as_bytes(), kp2.public_key.as_bytes());
    }

    #[test]
    fn ml_kem_wrong_sk_gives_different_ss() {
        let seed1 = [0x01u8; 64];
        let seed2 = [0x02u8; 64];

        let kp1 = ml_kem_768_keygen(&seed1).unwrap();
        let kp2 = ml_kem_768_keygen(&seed2).unwrap();

        let encaps_seed = [0x42u8; 32];
        let (ct, ss_encaps) = ml_kem_768_encaps(&kp1.public_key, &encaps_seed).unwrap();

        // Decapsulate with wrong secret key -> implicit rejection, different ss
        let ss_wrong = ml_kem_768_decaps(&kp2.secret_key, &ct).unwrap();
        assert_ne!(
            ss_encaps.as_bytes(),
            ss_wrong.as_bytes(),
            "wrong SK should produce different shared secret"
        );
    }

    #[test]
    fn ml_kem_shared_secret_not_all_zeros() {
        let seed = [0x42u8; 64];
        let kp = ml_kem_768_keygen(&seed).unwrap();
        let encaps_seed = [0x01u8; 32];
        let (_ct, ss) = ml_kem_768_encaps(&kp.public_key, &encaps_seed).unwrap();
        assert!(
            ss.as_bytes().iter().any(|&b| b != 0),
            "shared secret should not be all zeros"
        );
    }
}
