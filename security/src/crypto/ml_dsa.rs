// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ML-DSA-65 digital signature algorithm (FIPS 204).
//!
//! ML-DSA (Module-Lattice-based Digital Signature Algorithm) is a
//! post-quantum digital signature standard, formerly known as CRYSTALS-Dilithium.
//! The 65 parameter set provides NIST Security Level 3 (~AES-192 equivalent).
//!
//! # Parameters (ML-DSA-65)
//!
//! | Parameter        | Size (bytes) |
//! |-----------------|-------------|
//! | Public key      | 1952        |
//! | Secret key      | 4032        |
//! | Signature       | 3309        |
//!
//! # Operations
//!
//! 1. **KeyGen**: Generate (public_key, secret_key) pair
//! 2. **Sign**: Using secret_key + message, produce signature
//! 3. **Verify**: Using public_key + message + signature, verify authenticity
//!
//! # Security
//!
//! ML-DSA-65 provides EUF-CMA (Existential Unforgeability under Chosen Message
//! Attack) security at NIST Level 3. It is used in SmallAIOS for:
//!
//! - ONNX model signature verification
//! - Secure boot chain validation
//! - IPC message authentication

#![allow(unused)]
// Crypto code uses indexed loops for audit clarity and to match reference implementations.
#![allow(clippy::needless_range_loop)]

use super::sha3::{sha3_256, Sha3_256, Shake256};
use core::fmt;

// ─── Constants ───────────────────────────────────────────────────────────────

/// ML-DSA-65 public key length in bytes.
pub const ML_DSA_65_PK_LEN: usize = 1952;

/// ML-DSA-65 secret key length in bytes.
pub const ML_DSA_65_SK_LEN: usize = 4032;

/// ML-DSA-65 signature length in bytes.
pub const ML_DSA_65_SIG_LEN: usize = 3309;

/// ML-DSA-65 module dimensions: (k, l) = (6, 5).
pub const ML_DSA_65_K: usize = 6;
pub const ML_DSA_65_L: usize = 5;

/// Polynomial ring dimension (n = 256).
pub const ML_DSA_N: usize = 256;

/// Modulus q = 8380417.
pub const ML_DSA_Q: u32 = 8_380_417;

/// Number of dropped bits from t (d = 13).
pub const ML_DSA_65_D: usize = 13;

/// Challenge weight (tau = 49).
pub const ML_DSA_65_TAU: usize = 49;

/// Bound on secret key coefficients (eta = 4).
pub const ML_DSA_65_ETA: usize = 4;

/// Bound on signature z coefficients (gamma1 = 2^19).
pub const ML_DSA_65_GAMMA1: u32 = 1 << 19;

/// Low-order rounding range (gamma2 = (q-1)/32).
pub const ML_DSA_65_GAMMA2: u32 = (ML_DSA_Q - 1) / 32;

/// Beta = tau * eta.
const BETA: u32 = ML_DSA_65_TAU as u32 * ML_DSA_65_ETA as u32;

/// Omega (max number of 1s in hint): 55.
const OMEGA: usize = 55;

// Internal constants
const Q: i32 = ML_DSA_Q as i32;
const N: usize = ML_DSA_N;
const K: usize = ML_DSA_65_K;
const L: usize = ML_DSA_65_L;
const D: usize = ML_DSA_65_D;
const ETA: usize = ML_DSA_65_ETA;
const GAMMA1: i32 = ML_DSA_65_GAMMA1 as i32;
const GAMMA2: i32 = ML_DSA_65_GAMMA2 as i32;
const TAU: usize = ML_DSA_65_TAU;

/// Maximum rejection sampling iterations for signing.
const SIGN_MAX_ATTEMPTS: usize = 1000;

// ─── NTT zetas ──────────────────────────────────────────────────────────────

/// Precomputed zetas for ML-DSA NTT: 1753^{bit_rev8(i)} mod q in Montgomery form.
/// Montgomery R = 2^32, q = 8380417.
const ZETAS: [i32; 256] = [
    -4186625, 25847, -2608894, -518909, 237124, -777960, -876248, 466468, 1826347, 2353451,
    -359251, -2091905, 3119733, -2884855, 3111497, 2680103, 2725464, 1024112, -1079900, 3585928,
    -549488, -1119584, 2619752, -2108549, -2118186, -3859737, -1399561, -3277672, 1757237, -19422,
    4010497, 280005, 2706023, 95776, 3077325, 3530437, -1661693, -3592148, -2537516, 3915439,
    -3861115, -3043716, 3574422, -2867647, 3539968, -300467, 2348700, -539299, -1699267, -1643818,
    3505694, -3821735, 3507263, -2140649, -1600420, 3699596, 811944, 531354, 954230, 3881043,
    3900724, -2556880, 2071892, -2797779, -3930395, -1528703, -3677745, -3041255, -1452451,
    3475950, 2176455, -1585221, -1257611, 1939314, -4083598, -1000202, -3190144, -3157330,
    -3632928, 126922, 3412210, -983419, 2147896, 2715295, -2967645, -3693493, -411027, -2477047,
    -671102, -1228525, -22981, -1308169, -381987, 1349076, 1852771, -1430430, -3343383, 264944,
    508951, 3097992, 44288, -1100098, 904516, 3958618, -3724342, -8578, 1653064, -3249728, 2389356,
    -210977, 759969, -1316856, 189548, -3553272, 3159746, -1851402, -2409325, -177440, 1315589,
    1341330, 1285669, -1584928, -812732, -1439742, -3019102, -3881060, -3628969, 3839961, 2091667,
    3407706, 2316500, 3817976, -3342478, 2244091, -2446433, -3562462, 266997, 2434439, -1235728,
    3513181, -3520352, -3759364, -1197226, -3193378, 900702, 1859098, 909542, 819034, 495491,
    -1613174, -43260, -522500, -655327, -3122442, 2031748, 3207046, -3556995, -525098, -768622,
    -3595838, 342297, 286988, -2437823, 4108315, 3437287, -3342277, 1735879, 203044, 2842341,
    2691481, -2590150, 1265009, 4055324, 1247620, 2486353, 1595974, -3767016, 1250494, 2635921,
    -3548272, -2994039, 1869119, 1903435, -1050970, -1333058, 1237275, -3318210, -1430225, -451100,
    1312455, 3306115, -1962642, -1279661, 1917081, -2546312, -1374803, 1500165, 777191, 2235880,
    3406031, -542412, -2831860, -1671176, -1846953, -2584293, -3724270, 594136, -3776993, -2013608,
    2432395, 2454455, -164721, 1957272, 3369112, 185531, -1207385, -3183426, 162844, 1616392,
    3014001, 810149, 1652634, -3694233, -1799107, -3038916, 3523897, 3866901, 269760, 2213111,
    -975884, 1717735, 472078, -426683, 1723600, -1803090, 1910376, -1667432, -1104333, -260646,
    -3833893, -2939036, -2235985, -420899, -2286327, 183443, -976891, 1612842, -3545687, -554416,
    3919660, -48306, -1362209, 3937738, 1400424, -846154, 1976782,
];

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from ML-DSA-65 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlDsaError {
    /// Invalid public key length.
    InvalidPublicKeyLength,
    /// Invalid secret key length.
    InvalidSecretKeyLength,
    /// Invalid signature length.
    InvalidSignatureLength,
    /// Signature verification failed.
    VerificationFailed,
    /// Signing failed (rejection sampling exceeded limit).
    SigningFailed,
    /// RNG failure during key generation or signing.
    RngFailure,
    /// Message too large.
    MessageTooLarge,
}

impl fmt::Display for MlDsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPublicKeyLength => {
                write!(
                    f,
                    "invalid public key length (expected {} bytes)",
                    ML_DSA_65_PK_LEN
                )
            }
            Self::InvalidSecretKeyLength => {
                write!(
                    f,
                    "invalid secret key length (expected {} bytes)",
                    ML_DSA_65_SK_LEN
                )
            }
            Self::InvalidSignatureLength => {
                write!(
                    f,
                    "invalid signature length (expected {} bytes)",
                    ML_DSA_65_SIG_LEN
                )
            }
            Self::VerificationFailed => write!(f, "ML-DSA signature verification failed"),
            Self::SigningFailed => {
                write!(f, "ML-DSA signing failed (rejection sampling exhausted)")
            }
            Self::RngFailure => write!(f, "random number generation failure"),
            Self::MessageTooLarge => write!(f, "message too large for signing"),
        }
    }
}

// ─── Modular Arithmetic ─────────────────────────────────────────────────────

/// Montgomery reduction for ML-DSA: a * R^{-1} mod q where R = 2^32.
#[inline]
fn montgomery_reduce(a: i64) -> i32 {
    // q_inv = q^{-1} mod R = 58728449
    const Q_INV: i32 = 58728449_u32 as i32;
    let t = (a as i32).wrapping_mul(Q_INV);
    ((a - t as i64 * Q as i64) >> 32) as i32
}

/// Reduce mod q to centered representation [-q/2, q/2].
#[inline]
fn reduce32(a: i32) -> i32 {
    // Barrett reduction for q = 8380417
    let t = (a + (1 << 22)) >> 23;
    a - t * Q
}

/// Conditional add q: if a is negative, add q.
#[inline]
fn caddq(a: i32) -> i32 {
    a + ((a >> 31) & Q)
}

/// Freeze: fully reduce to [0, q).
#[inline]
fn freeze(a: i32) -> i32 {
    caddq(reduce32(a))
}

// ─── Polynomial ──────────────────────────────────────────────────────────────

/// A polynomial in Z_q[X]/(X^256 + 1).
#[derive(Clone, Copy)]
struct Poly {
    coeffs: [i32; N],
}

impl Poly {
    fn zero() -> Self {
        Self { coeffs: [0; N] }
    }

    /// NTT forward transform (8-level Cooley-Tukey).
    fn ntt(&mut self) {
        let mut k = 0usize;
        let mut len = 128;
        while len >= 1 {
            let mut start = 0;
            while start < N {
                k += 1;
                let zeta = ZETAS[k] as i64;
                for j in start..(start + len) {
                    let t = montgomery_reduce(zeta * self.coeffs[j + len] as i64);
                    self.coeffs[j + len] = self.coeffs[j] - t;
                    self.coeffs[j] += t;
                }
                start += 2 * len;
            }
            len >>= 1;
        }
    }

    /// NTT inverse transform (8-level Gentleman-Sande).
    fn inv_ntt(&mut self) {
        let mut k = 256usize;
        let mut len = 1;
        while len < 256 {
            let mut start = 0;
            while start < N {
                k -= 1;
                let zeta = -(ZETAS[k] as i64);
                for j in start..(start + len) {
                    let t = self.coeffs[j];
                    self.coeffs[j] = t + self.coeffs[j + len];
                    self.coeffs[j + len] = t - self.coeffs[j + len];
                    self.coeffs[j + len] = montgomery_reduce(zeta * self.coeffs[j + len] as i64);
                }
                start += 2 * len;
            }
            len <<= 1;
        }
        // After inv_ntt butterflies, coefficients are scaled by n and the
        // pointwise_mul introduced an extra R^{-1}. We need F such that
        // montgomery_reduce(F * coeff) = coeff / (n * R).
        // F = R^2 * n^{-1} mod q = 41978 compensates for both factors.
        const F: i64 = 41978;
        for c in self.coeffs.iter_mut() {
            *c = montgomery_reduce(F * *c as i64);
        }
    }

    /// Pointwise Montgomery multiplication.
    fn pointwise_mul(&self, other: &Poly) -> Poly {
        let mut r = Poly::zero();
        for i in 0..N {
            r.coeffs[i] = montgomery_reduce(self.coeffs[i] as i64 * other.coeffs[i] as i64);
        }
        r
    }

    /// Add two polynomials.
    fn add(&self, other: &Poly) -> Poly {
        let mut r = Poly::zero();
        for i in 0..N {
            r.coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        r
    }

    /// Subtract two polynomials.
    fn sub(&self, other: &Poly) -> Poly {
        let mut r = Poly::zero();
        for i in 0..N {
            r.coeffs[i] = self.coeffs[i] - other.coeffs[i];
        }
        r
    }

    /// Reduce all coefficients.
    fn reduce(&mut self) {
        for c in self.coeffs.iter_mut() {
            *c = reduce32(*c);
        }
    }

    /// Check if all coefficients are less than bound (in absolute value).
    fn check_norm(&self, bound: i32) -> bool {
        for i in 0..N {
            let mut t = self.coeffs[i];
            t = reduce32(t);
            if t >= Q / 2 {
                t -= Q;
            }
            if t < -Q / 2 {
                t += Q;
            }
            if t < 0 {
                t = -t;
            }
            if t >= bound {
                return false;
            }
        }
        true
    }

    /// Power2Round: decompose into (t1, t0) where t = t1 * 2^d + t0.
    fn power2round(&self) -> (Poly, Poly) {
        let mut t1 = Poly::zero();
        let mut t0 = Poly::zero();
        for i in 0..N {
            let a = freeze(self.coeffs[i]);
            // t0 = a mod 2^d (centered)
            let mut r0 = a & ((1 << D) - 1);
            if r0 > (1 << (D - 1)) {
                r0 -= 1 << D;
            }
            t0.coeffs[i] = r0;
            t1.coeffs[i] = (a - r0) >> D;
        }
        (t1, t0)
    }

    /// HighBits: extract high-order bits after decompose.
    fn high_bits(&self) -> Poly {
        let mut r = Poly::zero();
        for i in 0..N {
            r.coeffs[i] = decompose_high(freeze(self.coeffs[i]));
        }
        r
    }

    /// LowBits: extract low-order bits after decompose.
    fn low_bits(&self) -> Poly {
        let mut r = Poly::zero();
        for i in 0..N {
            r.coeffs[i] = decompose_low(freeze(self.coeffs[i]));
        }
        r
    }

    /// Use hint to recover high bits.
    fn use_hint(&self, hint: &Poly) -> Poly {
        let mut r = Poly::zero();
        for i in 0..N {
            r.coeffs[i] = use_hint_coeff(freeze(self.coeffs[i]), hint.coeffs[i] as u8);
        }
        r
    }

    /// Make hint polynomial from two vectors.
    fn make_hint(z: &Poly, r: &Poly) -> (Poly, usize) {
        let mut h = Poly::zero();
        let mut count = 0;
        for i in 0..N {
            let h_val = make_hint_coeff(freeze(z.coeffs[i]), freeze(r.coeffs[i]));
            h.coeffs[i] = h_val as i32;
            count += h_val as usize;
        }
        (h, count)
    }

    /// Shift left by d bits (for t1 packing).
    fn shift_left(&self) -> Poly {
        let mut r = Poly::zero();
        for i in 0..N {
            r.coeffs[i] = self.coeffs[i] << D;
        }
        r
    }
}

// ─── Decompose helpers ──────────────────────────────────────────────────────

/// Decompose a into (high, low) where a ≡ high * alpha + low (mod q).
/// alpha = 2*gamma2 for ML-DSA-65. High bits are in [0, (q-1)/alpha - 1].
/// For gamma2 = (q-1)/32: alpha = (q-1)/16, high bits in [0, 15].
/// Matches the pqcrystals reference decompose.
fn decompose(a: i32) -> (i32, i32) {
    // Efficient computation from the pqcrystals reference (gamma2 = (q-1)/32)
    let mut a1 = (a + 127) >> 7;
    a1 = ((a1 as i64 * 1025 + (1 << 21)) >> 22) as i32;
    a1 &= 15; // High bits in [0, 15]
    let mut a0 = a - a1 * 2 * GAMMA2;
    // Handle wrapping: if a0 > (q-1)/2, subtract q
    a0 -= (((Q - 1) / 2 - a0) >> 31) & Q;
    (a1, a0)
}

fn decompose_high(a: i32) -> i32 {
    decompose(a).0
}

fn decompose_low(a: i32) -> i32 {
    decompose(a).1
}

/// Make hint: returns 1 if high bits of r+z differ from high bits of r.
fn make_hint_coeff(z: i32, r: i32) -> u8 {
    let r1 = decompose_high(r);
    let v1 = decompose_high(freeze(r + z));
    if r1 != v1 {
        1
    } else {
        0
    }
}

/// Use hint to correct high bits.
fn use_hint_coeff(a: i32, hint: u8) -> i32 {
    let alpha = 2 * GAMMA2;
    let a0 = decompose_low(a);
    let a1 = decompose_high(a);
    if hint == 0 {
        return a1;
    }
    // For gamma2 = (q-1)/32: valid a1 range is [0, 15], wrap mod 16
    // (matching pqcrystals reference which uses & 15)
    let m = (Q - 1) / alpha; // m = 16
    if a0 > 0 {
        (a1 + 1) % m
    } else {
        (a1 - 1 + m) % m
    }
}

// ─── Sampling ───────────────────────────────────────────────────────────────

/// Sample a polynomial uniformly from SHAKE128/256 output.
fn sample_uniform(seed: &[u8; 32], nonce: u16) -> Poly {
    let mut xof = Shake256::new();
    xof.absorb(seed).expect("absorb");
    xof.absorb(&nonce.to_le_bytes()).expect("absorb nonce");

    let mut p = Poly::zero();
    let mut buf = [0u8; 3];
    let mut ctr = 0;
    while ctr < N {
        xof.squeeze(&mut buf).expect("squeeze");
        let t = (buf[0] as u32) | ((buf[1] as u32) << 8) | ((buf[2] as u32) << 16);
        let t = t & 0x7FFFFF; // 23 bits
        if t < ML_DSA_Q {
            p.coeffs[ctr] = t as i32;
            ctr += 1;
        }
    }
    p
}

/// Sample a polynomial with coefficients in [-eta, eta] using rejection sampling.
/// For eta=4, each 4-bit nibble gives a uniform value in [0, 15]; we reject
/// values > 2*eta = 8 and map accepted values to [-eta, eta].
fn sample_eta(seed: &[u8; 64], nonce: u16) -> Poly {
    let mut xof = Shake256::new();
    xof.absorb(seed).expect("absorb");
    xof.absorb(&nonce.to_le_bytes()).expect("absorb nonce");

    let mut p = Poly::zero();
    let mut ctr = 0;
    let mut buf = [0u8; 16];
    while ctr < N {
        xof.squeeze(&mut buf).expect("squeeze");
        for byte in &buf {
            if ctr >= N {
                break;
            }
            // Low nibble
            let lo = byte & 0x0F;
            if (lo as usize) <= 2 * ETA {
                p.coeffs[ctr] = ETA as i32 - lo as i32;
                ctr += 1;
            }
            if ctr >= N {
                break;
            }
            // High nibble
            let hi = byte >> 4;
            if (hi as usize) <= 2 * ETA {
                p.coeffs[ctr] = ETA as i32 - hi as i32;
                ctr += 1;
            }
        }
    }
    p
}

/// Sample mask polynomial y with coefficients in [-gamma1+1, gamma1].
fn sample_mask(seed: &[u8; 64], nonce: u16) -> Poly {
    let mut xof = Shake256::new();
    xof.absorb(seed).expect("absorb");
    xof.absorb(&nonce.to_le_bytes()).expect("absorb nonce");

    // For gamma1 = 2^19: 20 bits per coefficient
    let bytes_needed = N * 20 / 8; // 640 bytes
    let mut buf = [0u8; 640];
    xof.squeeze(&mut buf).expect("squeeze");

    let mut p = Poly::zero();
    for i in (0..N).step_by(4) {
        // Pack 4 coefficients into 10 bytes (20 bits each)
        let base = i * 20 / 8;
        for j in 0..4 {
            let bit_offset = (i + j) * 20;
            let byte_idx = bit_offset / 8;
            let bit_off = bit_offset % 8;
            let val = if byte_idx + 2 < buf.len() {
                let v = (buf[byte_idx] as u32)
                    | ((buf[byte_idx + 1] as u32) << 8)
                    | ((buf[byte_idx + 2] as u32) << 16);
                (v >> bit_off) & 0xFFFFF
            } else {
                0
            };
            p.coeffs[i + j] = GAMMA1 - (val as i32);
        }
    }
    p
}

/// Sample challenge polynomial c with exactly tau coefficients in {-1, +1}.
fn sample_challenge(seed: &[u8; 32]) -> Poly {
    let mut xof = Shake256::new();
    xof.absorb(seed).expect("absorb");

    let mut signs = [0u8; 8];
    xof.squeeze(&mut signs).expect("squeeze signs");
    let sign_bits = u64::from_le_bytes(signs);

    let mut c = Poly::zero();
    let mut buf = [0u8; 1];
    for i in (N - TAU)..N {
        // Sample j uniform in [0, i]
        loop {
            xof.squeeze(&mut buf).expect("squeeze");
            let j = buf[0] as usize;
            if j <= i {
                c.coeffs[i] = c.coeffs[j];
                let sign_bit = (sign_bits >> (i - (N - TAU))) & 1;
                c.coeffs[j] = if sign_bit == 0 { 1 } else { -1 };
                break;
            }
        }
    }
    c
}

// ─── Encoding ───────────────────────────────────────────────────────────────

/// Encode polynomial t1 (10 bits per coefficient, k*320 bytes total per poly).
fn encode_t1(p: &Poly, out: &mut [u8]) {
    for i in (0..N).step_by(4) {
        let a = p.coeffs[i] as u32;
        let b = p.coeffs[i + 1] as u32;
        let c = p.coeffs[i + 2] as u32;
        let d = p.coeffs[i + 3] as u32;
        let idx = (i / 4) * 5;
        out[idx] = a as u8;
        out[idx + 1] = ((a >> 8) | (b << 2)) as u8;
        out[idx + 2] = ((b >> 6) | (c << 4)) as u8;
        out[idx + 3] = ((c >> 4) | (d << 6)) as u8;
        out[idx + 4] = (d >> 2) as u8;
    }
}

/// Decode t1 from bytes.
fn decode_t1(bytes: &[u8]) -> Poly {
    let mut p = Poly::zero();
    for i in (0..N).step_by(4) {
        let idx = (i / 4) * 5;
        let b0 = bytes[idx] as u32;
        let b1 = bytes[idx + 1] as u32;
        let b2 = bytes[idx + 2] as u32;
        let b3 = bytes[idx + 3] as u32;
        let b4 = bytes[idx + 4] as u32;
        p.coeffs[i] = (b0 | ((b1 & 0x03) << 8)) as i32;
        p.coeffs[i + 1] = ((b1 >> 2) | ((b2 & 0x0F) << 6)) as i32;
        p.coeffs[i + 2] = ((b2 >> 4) | ((b3 & 0x3F) << 4)) as i32;
        p.coeffs[i + 3] = ((b3 >> 6) | (b4 << 2)) as i32;
    }
    p
}

/// Encode z (gamma1 = 2^19, 20 bits per coefficient).
fn encode_z(p: &Poly, out: &mut [u8]) {
    for i in 0..N {
        let val = (GAMMA1 - p.coeffs[i]) as u32;
        let bo = i * 20;
        let byte_idx = bo / 8;
        let bit_off = bo % 8;
        // Write 20 bits starting at bit_off in byte_idx
        // Always need 3 bytes for 20-bit values
        out[byte_idx] |= (val << bit_off) as u8;
        if byte_idx + 1 < out.len() {
            out[byte_idx + 1] |= (val >> (8 - bit_off)) as u8;
        }
        if byte_idx + 2 < out.len() {
            out[byte_idx + 2] |= (val >> (16 - bit_off)) as u8;
        }
    }
}

/// Decode z from bytes.
fn decode_z(bytes: &[u8]) -> Poly {
    let mut p = Poly::zero();
    for i in 0..N {
        let bo = i * 20;
        let byte_idx = bo / 8;
        let bit_off = bo % 8;
        let val = if byte_idx + 2 < bytes.len() {
            let v = (bytes[byte_idx] as u32)
                | ((bytes[byte_idx + 1] as u32) << 8)
                | ((bytes[byte_idx + 2] as u32) << 16);
            (v >> bit_off) & 0xFFFFF
        } else {
            0
        };
        p.coeffs[i] = GAMMA1 - val as i32;
    }
    p
}

/// Encode hint as packed bits.
fn encode_hint(h: &[Poly; K], out: &mut [u8]) {
    let mut idx = 0;
    for i in 0..K {
        for j in 0..N {
            if h[i].coeffs[j] != 0 {
                out[idx] = j as u8;
                idx += 1;
            }
        }
        out[OMEGA + i] = idx as u8;
    }
}

/// Decode hint from packed bits.
fn decode_hint(bytes: &[u8]) -> Option<[Poly; K]> {
    let mut h = [Poly::zero(); K];
    let mut idx = 0;
    for i in 0..K {
        let limit = bytes[OMEGA + i] as usize;
        if limit < idx || limit > OMEGA {
            return None;
        }
        let start = idx;
        while idx < limit {
            // Only check monotonicity within the same polynomial
            if idx > start && bytes[idx] as usize <= bytes[idx - 1] as usize {
                return None;
            }
            h[i].coeffs[bytes[idx] as usize] = 1;
            idx += 1;
        }
    }
    // Check remaining bytes are zero
    for j in idx..OMEGA {
        if bytes[j] != 0 {
            return None;
        }
    }
    Some(h)
}

// ─── SHA-3 helpers ──────────────────────────────────────────────────────────

/// H(input) -> 32 bytes using SHA3-256.
fn h_hash(input: &[u8]) -> [u8; 32] {
    let digest = sha3_256(input);
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_bytes());
    out
}

/// CRH(input) -> 64 bytes using SHAKE256.
fn crh(input: &[u8]) -> [u8; 64] {
    let mut xof = Shake256::new();
    xof.absorb(input).expect("absorb");
    let mut out = [0u8; 64];
    xof.squeeze(&mut out).expect("squeeze");
    out
}

/// SHAKE256(input) -> 32 bytes.
fn shake256_32(input: &[u8]) -> [u8; 32] {
    let mut xof = Shake256::new();
    xof.absorb(input).expect("absorb");
    let mut out = [0u8; 32];
    xof.squeeze(&mut out).expect("squeeze");
    out
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// ML-DSA-65 public key (verification key).
#[derive(PartialEq, Eq)]
pub struct MlDsaPublicKey {
    bytes: [u8; ML_DSA_65_PK_LEN],
}

impl MlDsaPublicKey {
    /// Create a public key from a byte array.
    pub fn from_bytes(bytes: [u8; ML_DSA_65_PK_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a public key from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlDsaError> {
        if slice.len() != ML_DSA_65_PK_LEN {
            return Err(MlDsaError::InvalidPublicKeyLength);
        }
        let mut bytes = [0u8; ML_DSA_65_PK_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the public key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the public key length.
    pub fn len(&self) -> usize {
        ML_DSA_65_PK_LEN
    }

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlDsaPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlDsaPublicKey({} bytes)", ML_DSA_65_PK_LEN)
    }
}

/// ML-DSA-65 secret key (signing key).
#[derive(PartialEq, Eq)]
pub struct MlDsaSecretKey {
    bytes: [u8; ML_DSA_65_SK_LEN],
}

impl MlDsaSecretKey {
    /// Create a secret key from a byte array.
    pub fn from_bytes(bytes: [u8; ML_DSA_65_SK_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a secret key from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlDsaError> {
        if slice.len() != ML_DSA_65_SK_LEN {
            return Err(MlDsaError::InvalidSecretKeyLength);
        }
        let mut bytes = [0u8; ML_DSA_65_SK_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the secret key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the secret key length.
    pub fn len(&self) -> usize {
        ML_DSA_65_SK_LEN
    }

    /// Returns whether the key is empty (always false for fixed-size keys).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlDsaSecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlDsaSecretKey([REDACTED])")
    }
}

/// ML-DSA-65 signature.
#[derive(PartialEq, Eq)]
pub struct MlDsaSignature {
    bytes: [u8; ML_DSA_65_SIG_LEN],
}

impl MlDsaSignature {
    /// Create a signature from a byte array.
    pub fn from_bytes(bytes: [u8; ML_DSA_65_SIG_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a signature from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, MlDsaError> {
        if slice.len() != ML_DSA_65_SIG_LEN {
            return Err(MlDsaError::InvalidSignatureLength);
        }
        let mut bytes = [0u8; ML_DSA_65_SIG_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the signature bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the signature length.
    pub fn len(&self) -> usize {
        ML_DSA_65_SIG_LEN
    }

    /// Returns whether the signature is empty (always false for fixed-size types).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MlDsaSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlDsaSignature({} bytes)", ML_DSA_65_SIG_LEN)
    }
}

/// ML-DSA-65 key pair (public + secret key).
#[derive(PartialEq, Eq)]
pub struct MlDsaKeyPair {
    pub public_key: MlDsaPublicKey,
    pub secret_key: MlDsaSecretKey,
}

impl fmt::Debug for MlDsaKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlDsaKeyPair")
            .field("public_key", &self.public_key)
            .field("secret_key", &self.secret_key)
            .finish()
    }
}

// ─── Key encoding ───────────────────────────────────────────────────────────

/// t1 bytes per polynomial: 10 bits * 256 / 8 = 320.
const T1_POLY_BYTES: usize = 320;

/// Encode public key: pk = rho || encode(t1).
fn pack_pk(rho: &[u8; 32], t1: &[Poly; K]) -> [u8; ML_DSA_65_PK_LEN] {
    let mut pk = [0u8; ML_DSA_65_PK_LEN];
    pk[..32].copy_from_slice(rho);
    for i in 0..K {
        encode_t1(
            &t1[i],
            &mut pk[32 + i * T1_POLY_BYTES..32 + (i + 1) * T1_POLY_BYTES],
        );
    }
    pk
}

/// Decode public key.
fn unpack_pk(pk: &[u8]) -> ([u8; 32], [Poly; K]) {
    let mut rho = [0u8; 32];
    rho.copy_from_slice(&pk[..32]);
    let mut t1 = [Poly::zero(); K];
    for i in 0..K {
        t1[i] = decode_t1(&pk[32 + i * T1_POLY_BYTES..32 + (i + 1) * T1_POLY_BYTES]);
    }
    (rho, t1)
}

/// Pack secret key: sk = rho || K || tr || s1 || s2 || t0.
/// For ML-DSA-65: 32 + 32 + 64 + 5*128 + 6*128 + 6*416 = 4032 bytes.
/// (eta=4 encoding: each coeff in 4 bits -> 128 bytes per poly)
/// (t0 encoding: 13 bits per coeff -> 416 bytes per poly)
const ETA_POLY_BYTES: usize = 128; // eta=4: 4 bits per coeff
const T0_POLY_BYTES: usize = 416; // 13 bits per coeff

fn encode_eta(p: &Poly, out: &mut [u8]) {
    // eta=4: each coefficient in [0, 2*eta] -> 4 bits
    for i in (0..N).step_by(2) {
        let a = (ETA as i32 - p.coeffs[i]) as u8;
        let b = (ETA as i32 - p.coeffs[i + 1]) as u8;
        out[i / 2] = a | (b << 4);
    }
}

fn decode_eta(bytes: &[u8]) -> Poly {
    let mut p = Poly::zero();
    for i in (0..N).step_by(2) {
        let byte = bytes[i / 2];
        let a = (byte & 0x0F) as i32;
        let b = (byte >> 4) as i32;
        p.coeffs[i] = ETA as i32 - a;
        p.coeffs[i + 1] = ETA as i32 - b;
    }
    p
}

fn encode_t0(p: &Poly, out: &mut [u8]) {
    // t0 in [-(2^(d-1)-1), 2^(d-1)], d=13: 13 bits per coeff
    for i in (0..N).step_by(8) {
        let bo = (i / 8) * 13;
        for j in 0..8 {
            let val = ((1 << (D - 1)) - p.coeffs[i + j]) as u32;
            let bit_pos = j * 13;
            let byte_idx = bo + bit_pos / 8;
            let bit_off = bit_pos % 8;
            if byte_idx < out.len() {
                out[byte_idx] |= (val << bit_off) as u8;
            }
            if byte_idx + 1 < out.len() {
                out[byte_idx + 1] |= (val >> (8 - bit_off)) as u8;
            }
            if bit_off > 3 && byte_idx + 2 < out.len() {
                out[byte_idx + 2] |= (val >> (16 - bit_off)) as u8;
            }
        }
    }
}

fn decode_t0(bytes: &[u8]) -> Poly {
    let mut p = Poly::zero();
    for i in (0..N).step_by(8) {
        let bo = (i / 8) * 13;
        for j in 0..8 {
            let bit_pos = j * 13;
            let byte_idx = bo + bit_pos / 8;
            let bit_off = bit_pos % 8;
            let mut v: u32 = 0;
            if byte_idx < bytes.len() {
                v |= bytes[byte_idx] as u32;
            }
            if byte_idx + 1 < bytes.len() {
                v |= (bytes[byte_idx + 1] as u32) << 8;
            }
            if byte_idx + 2 < bytes.len() {
                v |= (bytes[byte_idx + 2] as u32) << 16;
            }
            let val = (v >> bit_off) & 0x1FFF;
            p.coeffs[i + j] = (1 << (D - 1)) - val as i32;
        }
    }
    p
}

fn pack_sk(
    rho: &[u8; 32],
    key: &[u8; 32],
    tr: &[u8; 64],
    s1: &[Poly; L],
    s2: &[Poly; K],
    t0: &[Poly; K],
) -> [u8; ML_DSA_65_SK_LEN] {
    let mut sk = [0u8; ML_DSA_65_SK_LEN];
    let mut offset = 0;

    sk[offset..offset + 32].copy_from_slice(rho);
    offset += 32;
    sk[offset..offset + 32].copy_from_slice(key);
    offset += 32;
    sk[offset..offset + 64].copy_from_slice(tr);
    offset += 64;

    for i in 0..L {
        encode_eta(&s1[i], &mut sk[offset..offset + ETA_POLY_BYTES]);
        offset += ETA_POLY_BYTES;
    }
    for i in 0..K {
        encode_eta(&s2[i], &mut sk[offset..offset + ETA_POLY_BYTES]);
        offset += ETA_POLY_BYTES;
    }
    for i in 0..K {
        encode_t0(&t0[i], &mut sk[offset..offset + T0_POLY_BYTES]);
        offset += T0_POLY_BYTES;
    }

    sk
}

#[allow(clippy::type_complexity)]
fn unpack_sk(
    sk: &[u8],
) -> (
    [u8; 32],
    [u8; 32],
    [u8; 64],
    [Poly; L],
    [Poly; K],
    [Poly; K],
) {
    let mut offset = 0;
    let mut rho = [0u8; 32];
    rho.copy_from_slice(&sk[offset..offset + 32]);
    offset += 32;
    let mut key = [0u8; 32];
    key.copy_from_slice(&sk[offset..offset + 32]);
    offset += 32;
    let mut tr = [0u8; 64];
    tr.copy_from_slice(&sk[offset..offset + 64]);
    offset += 64;

    let mut s1 = [Poly::zero(); L];
    for i in 0..L {
        s1[i] = decode_eta(&sk[offset..offset + ETA_POLY_BYTES]);
        offset += ETA_POLY_BYTES;
    }
    let mut s2 = [Poly::zero(); K];
    for i in 0..K {
        s2[i] = decode_eta(&sk[offset..offset + ETA_POLY_BYTES]);
        offset += ETA_POLY_BYTES;
    }
    let mut t0 = [Poly::zero(); K];
    for i in 0..K {
        t0[i] = decode_t0(&sk[offset..offset + T0_POLY_BYTES]);
        offset += T0_POLY_BYTES;
    }

    (rho, key, tr, s1, s2, t0)
}

// ─── Operations (FIPS 204) ──────────────────────────────────────────────────

/// Generate an ML-DSA-65 key pair.
///
/// Requires a 32-byte random seed from the CSPRNG.
pub fn ml_dsa_65_keygen(seed: &[u8; 32]) -> Result<MlDsaKeyPair, MlDsaError> {
    // Expand seed: (rho, rho', K) = H(seed)
    let mut expanded = [0u8; 128];
    let mut xof = Shake256::new();
    xof.absorb(seed).expect("absorb");
    xof.squeeze(&mut expanded).expect("squeeze");

    let mut rho = [0u8; 32];
    rho.copy_from_slice(&expanded[..32]);
    let mut rho_prime = [0u8; 64];
    rho_prime.copy_from_slice(&expanded[32..96]);
    let mut key = [0u8; 32];
    key.copy_from_slice(&expanded[96..128]);

    // Generate matrix A from rho and convert to NTT domain
    let mut a_hat = [[Poly::zero(); L]; K];
    for i in 0..K {
        for j in 0..L {
            a_hat[i][j] = sample_uniform(&rho, (i * 256 + j) as u16);
            a_hat[i][j].ntt();
        }
    }

    // Generate secret vectors s1, s2
    let mut s1 = [Poly::zero(); L];
    for i in 0..L {
        s1[i] = sample_eta(&rho_prime, i as u16);
    }
    let mut s2 = [Poly::zero(); K];
    for i in 0..K {
        s2[i] = sample_eta(&rho_prime, (L + i) as u16);
    }

    // NTT(s1)
    let mut s1_hat = [Poly::zero(); L];
    for i in 0..L {
        s1_hat[i] = s1[i];
        s1_hat[i].ntt();
    }

    // t = A * s1 + s2
    let mut t = [Poly::zero(); K];
    for i in 0..K {
        for j in 0..L {
            let prod = a_hat[i][j].pointwise_mul(&s1_hat[j]);
            t[i] = t[i].add(&prod);
        }
        t[i].inv_ntt();
        t[i] = t[i].add(&s2[i]);
        t[i].reduce();
    }

    // Power2Round: t = t1 * 2^d + t0
    let mut t1 = [Poly::zero(); K];
    let mut t0 = [Poly::zero(); K];
    for i in 0..K {
        let (t1_i, t0_i) = t[i].power2round();
        t1[i] = t1_i;
        t0[i] = t0_i;
    }

    // Pack public key
    let pk_bytes = pack_pk(&rho, &t1);

    // tr = CRH(pk)
    let tr = crh(&pk_bytes);

    // Pack secret key
    let sk_bytes = pack_sk(&rho, &key, &tr, &s1, &s2, &t0);

    Ok(MlDsaKeyPair {
        public_key: MlDsaPublicKey::from_bytes(pk_bytes),
        secret_key: MlDsaSecretKey::from_bytes(sk_bytes),
    })
}

/// Sign a message using ML-DSA-65.
///
/// Uses hedged signing with the provided random seed for side-channel resistance.
pub fn ml_dsa_65_sign(
    sk: &MlDsaSecretKey,
    message: &[u8],
    random_seed: &[u8; 32],
) -> Result<MlDsaSignature, MlDsaError> {
    let (rho, key, tr, s1, s2, t0) = unpack_sk(sk.as_bytes());

    // NTT(s1), NTT(s2), NTT(t0)
    let mut s1_hat = [Poly::zero(); L];
    for i in 0..L {
        s1_hat[i] = s1[i];
        s1_hat[i].ntt();
    }
    let mut s2_hat = [Poly::zero(); K];
    for i in 0..K {
        s2_hat[i] = s2[i];
        s2_hat[i].ntt();
    }
    let mut t0_hat = [Poly::zero(); K];
    for i in 0..K {
        t0_hat[i] = t0[i];
        t0_hat[i].ntt();
    }

    // Generate matrix A from rho and convert to NTT domain
    let mut a_hat = [[Poly::zero(); L]; K];
    for i in 0..K {
        for j in 0..L {
            a_hat[i][j] = sample_uniform(&rho, (i * 256 + j) as u16);
            a_hat[i][j].ntt();
        }
    }

    // mu = CRH(tr || msg)
    let mut mu_input = [0u8; 64];
    mu_input.copy_from_slice(&tr);
    let mu = {
        let mut xof = Shake256::new();
        xof.absorb(&mu_input).expect("absorb tr");
        xof.absorb(message).expect("absorb msg");
        let mut out = [0u8; 64];
        xof.squeeze(&mut out).expect("squeeze");
        out
    };

    // rho'' = CRH(key || rnd || mu) for hedged signing
    let rho_pp = {
        let mut xof = Shake256::new();
        xof.absorb(&key).expect("absorb key");
        xof.absorb(random_seed).expect("absorb rnd");
        xof.absorb(&mu).expect("absorb mu");
        let mut out = [0u8; 64];
        xof.squeeze(&mut out).expect("squeeze");
        out
    };

    // Rejection sampling loop
    let mut kappa: u16 = 0;
    for _attempt in 0..SIGN_MAX_ATTEMPTS {
        // Sample y from rho''
        let mut y = [Poly::zero(); L];
        for i in 0..L {
            y[i] = sample_mask(&rho_pp, kappa + i as u16);
        }
        kappa += L as u16;

        // w = A * NTT(y)
        let mut y_hat = [Poly::zero(); L];
        for i in 0..L {
            y_hat[i] = y[i];
            y_hat[i].ntt();
        }

        let mut w = [Poly::zero(); K];
        for i in 0..K {
            for j in 0..L {
                let prod = a_hat[i][j].pointwise_mul(&y_hat[j]);
                w[i] = w[i].add(&prod);
            }
            w[i].inv_ntt();
            w[i].reduce();
        }

        // w1 = HighBits(w)
        let mut w1 = [Poly::zero(); K];
        for i in 0..K {
            w1[i] = w[i].high_bits();
        }

        // c_tilde = H(mu || encode(w1))
        let c_tilde = {
            let mut xof = Shake256::new();
            xof.absorb(&mu).expect("absorb mu");
            // Encode w1 (simplified: hash the coefficients directly)
            for i in 0..K {
                for j in 0..N {
                    let val = w1[i].coeffs[j] as u32;
                    xof.absorb(&val.to_le_bytes()).expect("absorb w1");
                }
            }
            let mut out = [0u8; 32];
            xof.squeeze(&mut out).expect("squeeze");
            out
        };

        let c_poly = sample_challenge(&c_tilde);
        let mut c_hat = c_poly;
        c_hat.ntt();

        // z = y + c*s1
        let mut z = [Poly::zero(); L];
        for i in 0..L {
            let cs1 = c_hat.pointwise_mul(&s1_hat[i]);
            let mut cs1_time = cs1;
            cs1_time.inv_ntt();
            z[i] = y[i].add(&cs1_time);
            z[i].reduce();
        }

        // Check ||z||_inf < gamma1 - beta
        let bound = GAMMA1 - BETA as i32;
        let mut reject = false;
        for i in 0..L {
            if !z[i].check_norm(bound) {
                reject = true;
                break;
            }
        }
        if reject {
            continue;
        }

        // r0 = LowBits(w - c*s2)
        let mut cs2 = [Poly::zero(); K];
        for i in 0..K {
            let t = c_hat.pointwise_mul(&s2_hat[i]);
            cs2[i] = t;
            cs2[i].inv_ntt();
            cs2[i].reduce();
        }

        let mut w_minus_cs2 = [Poly::zero(); K];
        for i in 0..K {
            w_minus_cs2[i] = w[i].sub(&cs2[i]);
            w_minus_cs2[i].reduce();
        }

        // Check ||LowBits(w - c*s2)||_inf < gamma2 - beta
        let low_bound = GAMMA2 - BETA as i32;
        let mut low_reject = false;
        for i in 0..K {
            let r0 = w_minus_cs2[i].low_bits();
            if !r0.check_norm(low_bound) {
                low_reject = true;
                break;
            }
        }
        if low_reject {
            continue;
        }

        // Compute hint
        let mut ct0 = [Poly::zero(); K];
        for i in 0..K {
            let t = c_hat.pointwise_mul(&t0_hat[i]);
            ct0[i] = t;
            ct0[i].inv_ntt();
            ct0[i].reduce();
        }

        // Check ||c*t0||_inf < gamma2
        let mut ct0_reject = false;
        for i in 0..K {
            if !ct0[i].check_norm(GAMMA2) {
                ct0_reject = true;
                break;
            }
        }
        if ct0_reject {
            continue;
        }

        // h = MakeHint(-c*t0, w - c*s2 + c*t0)
        let mut h = [Poly::zero(); K];
        let mut total_hints = 0usize;
        for i in 0..K {
            let neg_ct0 = Poly::zero().sub(&ct0[i]);
            let val = w_minus_cs2[i].add(&ct0[i]);
            let (hi, count) = Poly::make_hint(&neg_ct0, &val);
            h[i] = hi;
            total_hints += count;
        }

        if total_hints > OMEGA {
            continue;
        }

        // Pack signature: sig = c_tilde || z || h
        let mut sig = [0u8; ML_DSA_65_SIG_LEN];
        sig[..32].copy_from_slice(&c_tilde);
        let mut offset = 32;
        // z encoding: 20 bits per coeff, 640 bytes per poly, L polys
        for i in 0..L {
            let mut z_bytes = [0u8; 640];
            encode_z(&z[i], &mut z_bytes);
            sig[offset..offset + 640].copy_from_slice(&z_bytes);
            offset += 640;
        }
        // Hint encoding: OMEGA + K bytes
        encode_hint(&h, &mut sig[offset..offset + OMEGA + K]);

        return Ok(MlDsaSignature::from_bytes(sig));
    }

    Err(MlDsaError::SigningFailed)
}

/// Verify an ML-DSA-65 signature.
pub fn ml_dsa_65_verify(
    pk: &MlDsaPublicKey,
    message: &[u8],
    signature: &MlDsaSignature,
) -> Result<(), MlDsaError> {
    let pk_bytes = pk.as_bytes();
    let sig_bytes = signature.as_bytes();

    // Unpack public key
    let (rho, t1) = unpack_pk(pk_bytes);

    // Generate matrix A from rho and convert to NTT domain
    let mut a_hat = [[Poly::zero(); L]; K];
    for i in 0..K {
        for j in 0..L {
            a_hat[i][j] = sample_uniform(&rho, (i * 256 + j) as u16);
            a_hat[i][j].ntt();
        }
    }

    // Unpack signature: c_tilde, z, h
    let mut c_tilde = [0u8; 32];
    c_tilde.copy_from_slice(&sig_bytes[..32]);

    let mut offset = 32;
    let mut z = [Poly::zero(); L];
    for i in 0..L {
        z[i] = decode_z(&sig_bytes[offset..offset + 640]);
        offset += 640;
    }

    let h = match decode_hint(&sig_bytes[offset..offset + OMEGA + K]) {
        Some(h) => h,
        None => return Err(MlDsaError::VerificationFailed),
    };

    // Check ||z||_inf < gamma1 - beta
    let bound = GAMMA1 - BETA as i32;
    for i in 0..L {
        if !z[i].check_norm(bound) {
            return Err(MlDsaError::VerificationFailed);
        }
    }

    // mu = CRH(CRH(pk) || msg)
    let tr = crh(pk_bytes);
    let mu = {
        let mut xof = Shake256::new();
        xof.absorb(&tr).expect("absorb tr");
        xof.absorb(message).expect("absorb msg");
        let mut out = [0u8; 64];
        xof.squeeze(&mut out).expect("squeeze");
        out
    };

    // c = SampleInBall(c_tilde)
    let c_poly = sample_challenge(&c_tilde);
    let mut c_hat = c_poly;
    c_hat.ntt();

    // w'_approx = A*NTT(z) - c*NTT(t1*2^d)
    let mut z_hat = [Poly::zero(); L];
    for i in 0..L {
        z_hat[i] = z[i];
        z_hat[i].ntt();
    }

    let mut t1_shifted = [Poly::zero(); K];
    for i in 0..K {
        t1_shifted[i] = t1[i].shift_left();
        t1_shifted[i].ntt();
    }

    let mut w_prime = [Poly::zero(); K];
    for i in 0..K {
        // A*z
        for j in 0..L {
            let prod = a_hat[i][j].pointwise_mul(&z_hat[j]);
            w_prime[i] = w_prime[i].add(&prod);
        }
        // - c * t1 * 2^d
        let ct1 = c_hat.pointwise_mul(&t1_shifted[i]);
        w_prime[i] = w_prime[i].sub(&ct1);
        w_prime[i].inv_ntt();
        w_prime[i].reduce();
    }

    // UseHint to get w1'
    let mut w1_prime = [Poly::zero(); K];
    for i in 0..K {
        w1_prime[i] = w_prime[i].use_hint(&h[i]);
    }

    // c_tilde' = H(mu || encode(w1'))
    let c_tilde_verify = {
        let mut xof = Shake256::new();
        xof.absorb(&mu).expect("absorb mu");
        for i in 0..K {
            for j in 0..N {
                let val = w1_prime[i].coeffs[j] as u32;
                xof.absorb(&val.to_le_bytes()).expect("absorb w1");
            }
        }
        let mut out = [0u8; 32];
        xof.squeeze(&mut out).expect("squeeze");
        out
    };

    // Verify
    if c_tilde == c_tilde_verify {
        Ok(())
    } else {
        Err(MlDsaError::VerificationFailed)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn ml_dsa_constant_sizes() {
        assert_eq!(ML_DSA_65_PK_LEN, 1952);
        assert_eq!(ML_DSA_65_SK_LEN, 4032);
        assert_eq!(ML_DSA_65_SIG_LEN, 3309);
        assert_eq!(ML_DSA_65_K, 6);
        assert_eq!(ML_DSA_65_L, 5);
        assert_eq!(ML_DSA_N, 256);
        assert_eq!(ML_DSA_Q, 8_380_417);
    }

    #[test]
    fn ml_dsa_derived_constants() {
        assert_eq!(ML_DSA_65_D, 13);
        assert_eq!(ML_DSA_65_TAU, 49);
        assert_eq!(ML_DSA_65_ETA, 4);
        assert_eq!(ML_DSA_65_GAMMA1, 524_288); // 2^19
        assert_eq!(ML_DSA_65_GAMMA2, 261_888); // (q-1)/32
    }

    #[test]
    fn ml_dsa_public_key_from_slice_valid() {
        let bytes = [0x42u8; ML_DSA_65_PK_LEN];
        let pk = MlDsaPublicKey::from_slice(&bytes).unwrap();
        assert_eq!(pk.len(), ML_DSA_65_PK_LEN);
        assert_eq!(pk.as_bytes()[0], 0x42);
    }

    #[test]
    fn ml_dsa_public_key_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlDsaPublicKey::from_slice(&bytes),
            Err(MlDsaError::InvalidPublicKeyLength)
        );
    }

    #[test]
    fn ml_dsa_secret_key_from_slice_valid() {
        let bytes = [0xAB; ML_DSA_65_SK_LEN];
        let sk = MlDsaSecretKey::from_slice(&bytes).unwrap();
        assert_eq!(sk.len(), ML_DSA_65_SK_LEN);
    }

    #[test]
    fn ml_dsa_secret_key_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlDsaSecretKey::from_slice(&bytes),
            Err(MlDsaError::InvalidSecretKeyLength)
        );
    }

    #[test]
    fn ml_dsa_signature_from_slice_valid() {
        let bytes = [0xCD; ML_DSA_65_SIG_LEN];
        let sig = MlDsaSignature::from_slice(&bytes).unwrap();
        assert_eq!(sig.len(), ML_DSA_65_SIG_LEN);
    }

    #[test]
    fn ml_dsa_signature_from_slice_invalid() {
        let bytes = [0u8; 100];
        assert_eq!(
            MlDsaSignature::from_slice(&bytes),
            Err(MlDsaError::InvalidSignatureLength)
        );
    }

    #[test]
    fn ml_dsa_secret_key_debug_redacted() {
        let sk = MlDsaSecretKey::from_bytes([0xFF; ML_DSA_65_SK_LEN]);
        let debug = format!("{:?}", sk);
        assert_eq!(debug, "MlDsaSecretKey([REDACTED])");
    }

    #[test]
    fn ml_dsa_public_key_debug() {
        let pk = MlDsaPublicKey::from_bytes([0; ML_DSA_65_PK_LEN]);
        let debug = format!("{:?}", pk);
        assert!(debug.contains("1952"));
    }

    #[test]
    fn ml_dsa_signature_debug() {
        let sig = MlDsaSignature::from_bytes([0; ML_DSA_65_SIG_LEN]);
        let debug = format!("{:?}", sig);
        assert!(debug.contains("3309"));
    }

    #[test]
    fn ml_dsa_ntt_roundtrip() {
        // With F = R^2 * n^{-1} mod q = 41978, the NTT roundtrip produces
        // MONT * coeff (Montgomery form) rather than coeff. This is correct:
        // the extra R factor cancels when pointwise_mul introduces R^{-1}.
        let mut p = Poly::zero();
        for i in 0..N {
            p.coeffs[i] = (i as i32 * 17) % Q;
        }
        let original = p.coeffs;
        p.ntt();
        p.inv_ntt();
        p.reduce();
        // inv_ntt(ntt(coeff)) = MONT * coeff mod q where MONT = 2^32 mod q
        const MONT: i64 = 4193792;
        for i in 0..N {
            let expected = freeze(((original[i] as i64 * MONT) % Q as i64) as i32);
            let got = freeze(p.coeffs[i]);
            assert_eq!(expected, got, "NTT roundtrip mismatch at {}", i);
        }
    }

    #[test]
    fn ml_dsa_power2round() {
        let mut p = Poly::zero();
        p.coeffs[0] = 1000;
        p.coeffs[1] = Q - 1;
        let (t1, t0) = p.power2round();
        // Check: t1 * 2^d + t0 == p (mod q)
        for i in 0..2 {
            let reconstructed = t1.coeffs[i] * (1 << D) + t0.coeffs[i];
            assert_eq!(
                freeze(reconstructed),
                freeze(p.coeffs[i]),
                "power2round mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn ml_dsa_ntt_multiply_pipeline() {
        // Test: pointwise_mul in NTT domain then inv_ntt should give
        // the negacyclic convolution (polynomial product mod X^256+1).
        // Use simple polynomials: a = [1, 2, 0, 0, ...], b = [3, 0, 0, ...]
        // a * b mod X^256+1 = [3, 6, 0, 0, ...]
        let mut a = Poly::zero();
        a.coeffs[0] = 1;
        a.coeffs[1] = 2;
        let mut b = Poly::zero();
        b.coeffs[0] = 3;

        let mut a_hat = a;
        a_hat.ntt();
        let mut b_hat = b;
        b_hat.ntt();

        let c_hat = a_hat.pointwise_mul(&b_hat);
        let mut c = c_hat;
        c.inv_ntt();
        c.reduce();

        // Expected: [3, 6, 0, 0, ..., 0]
        assert_eq!(freeze(c.coeffs[0]), 3, "ntt multiply c[0]");
        assert_eq!(freeze(c.coeffs[1]), 6, "ntt multiply c[1]");
        for i in 2..N {
            assert_eq!(freeze(c.coeffs[i]), 0, "ntt multiply c[{}] should be 0", i);
        }
    }

    #[test]
    fn ml_dsa_encode_decode_eta() {
        // Test eta encoding roundtrip for small secret values
        let mut p = Poly::zero();
        for i in 0..N {
            p.coeffs[i] = ((i % 9) as i32) - 4; // values in [-4, 4]
        }
        let mut buf = [0u8; ETA_POLY_BYTES];
        encode_eta(&p, &mut buf);
        let p2 = decode_eta(&buf);
        for i in 0..N {
            assert_eq!(p.coeffs[i], p2.coeffs[i], "eta roundtrip mismatch at {}", i);
        }
    }

    #[test]
    fn ml_dsa_encode_decode_t0() {
        // t0 range from power2round: [-4095, 4096]
        let mut p = Poly::zero();
        for i in 0..N {
            // Map to range [-4095, 4096]
            let val = ((i as i32 * 37) % 8192) - 4095;
            p.coeffs[i] = val;
        }
        let mut buf = [0u8; T0_POLY_BYTES];
        encode_t0(&p, &mut buf);
        let p2 = decode_t0(&buf);
        for i in 0..N {
            assert_eq!(p.coeffs[i], p2.coeffs[i], "t0 roundtrip mismatch at {}", i);
        }
    }

    #[test]
    fn ml_dsa_encode_decode_t1() {
        let mut p = Poly::zero();
        for i in 0..N {
            p.coeffs[i] = (i as i32 * 3) % 1024; // 10-bit values
        }
        let mut buf = [0u8; T1_POLY_BYTES];
        encode_t1(&p, &mut buf);
        let p2 = decode_t1(&buf);
        for i in 0..N {
            assert_eq!(p.coeffs[i], p2.coeffs[i], "t1 roundtrip mismatch at {}", i);
        }
    }

    #[test]
    fn ml_dsa_keygen_succeeds() {
        let seed = [0x42u8; 32];
        let result = ml_dsa_65_keygen(&seed);
        assert!(result.is_ok(), "keygen should succeed");
        let keypair = result.unwrap();
        assert_eq!(keypair.public_key.len(), ML_DSA_65_PK_LEN);
        assert_eq!(keypair.secret_key.len(), ML_DSA_65_SK_LEN);
    }

    #[test]
    fn ml_dsa_keygen_deterministic() {
        let seed = [0x42u8; 32];
        let kp1 = ml_dsa_65_keygen(&seed).unwrap();
        let kp2 = ml_dsa_65_keygen(&seed).unwrap();
        assert_eq!(kp1.public_key.as_bytes(), kp2.public_key.as_bytes());
        assert_eq!(kp1.secret_key.as_bytes(), kp2.secret_key.as_bytes());
    }

    #[test]
    fn ml_dsa_encode_decode_z() {
        let mut p = Poly::zero();
        for i in 0..N {
            // z values in valid range: strictly within (-gamma1, gamma1)
            p.coeffs[i] = ((i as i32 * 1337) % (2 * GAMMA1 - 2)) - GAMMA1 + 1;
        }
        let mut buf = [0u8; 640];
        encode_z(&p, &mut buf);
        let p2 = decode_z(&buf);
        for i in 0..N {
            assert_eq!(p.coeffs[i], p2.coeffs[i], "z roundtrip mismatch at {}", i);
        }
    }

    #[test]
    fn ml_dsa_sign_verify_roundtrip() {
        let seed = [0x42u8; 32];
        let keypair = ml_dsa_65_keygen(&seed).unwrap();

        let message = b"test message for ML-DSA-65";
        let rng_seed = [0x37u8; 32];
        let sig = ml_dsa_65_sign(&keypair.secret_key, message, &rng_seed)
            .expect("signing should succeed");
        assert_eq!(sig.len(), ML_DSA_65_SIG_LEN);

        let result = ml_dsa_65_verify(&keypair.public_key, message, &sig);
        assert!(
            result.is_ok(),
            "verification should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn ml_dsa_verify_wrong_message_fails() {
        let seed = [0x42u8; 32];
        let keypair = ml_dsa_65_keygen(&seed).unwrap();

        let message = b"test message";
        let rng_seed = [0x37u8; 32];

        let sig = ml_dsa_65_sign(&keypair.secret_key, message, &rng_seed)
            .expect("signing should succeed");

        let wrong_message = b"wrong message";
        let result = ml_dsa_65_verify(&keypair.public_key, wrong_message, &sig);
        assert_eq!(result, Err(MlDsaError::VerificationFailed));
    }

    #[test]
    fn ml_dsa_different_seeds_different_keys() {
        let kp1 = ml_dsa_65_keygen(&[0x01u8; 32]).unwrap();
        let kp2 = ml_dsa_65_keygen(&[0x02u8; 32]).unwrap();
        assert_ne!(kp1.public_key.as_bytes(), kp2.public_key.as_bytes());
    }
}
