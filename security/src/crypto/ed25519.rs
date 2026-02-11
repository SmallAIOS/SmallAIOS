// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Ed25519 digital signatures (RFC 8032).
//!
//! Ed25519 is a high-performance digital signature scheme using the
//! twisted Edwards curve -x^2 + y^2 = 1 + d*x^2*y^2 (d = -121665/121666).
//!
//! # Parameters
//!
//! | Parameter        | Size (bytes) |
//! |-----------------|-------------|
//! | Public key      | 32          |
//! | Secret key      | 64 (seed+pk)|
//! | Signature       | 64          |
//!
//! # Operations
//!
//! 1. **KeyGen**: Generate (public_key, secret_key) from 32 bytes of randomness
//! 2. **Sign**: Using secret_key + message, produce signature
//! 3. **Verify**: Using public_key + message + signature, verify authenticity

#![allow(unused)]
#![allow(clippy::needless_range_loop)]

use super::field25519::Fe;
use super::sha3::{sha3_256, Sha3_256, Shake256};
use core::fmt;

/// Ed25519 seed length.
pub const ED25519_SEED_LEN: usize = 32;
/// Ed25519 public key length.
pub const ED25519_PK_LEN: usize = 32;
/// Ed25519 secret key length (seed || public_key).
pub const ED25519_SK_LEN: usize = 64;
/// Ed25519 signature length.
pub const ED25519_SIG_LEN: usize = 64;

// ─── Curve constants ────────────────────────────────────────────────────────

/// d = -121665/121666 mod p
const D: Fe = Fe::from_limbs([
    0x34DCA135978A3,
    0x1A8283B156EBD,
    0x5E7A26001C029,
    0x739C663A03CBB,
    0x52036CEE2B6FF,
]);

/// 2*d
const D2: Fe = Fe::from_limbs([
    0x69B9426B2F159,
    0x35050762ADD7A,
    0x3CF44C0038052,
    0x6738CC7407977,
    0x2406D9DC56DFF,
]);

/// The base point B of Ed25519 (extended coordinates).
/// B has y = 4/5 mod p, x is the positive root.
const BASE_Y: Fe = Fe::from_limbs([
    0x6666666666658,
    0x4CCCCCCCCCCCC,
    0x1999999999999,
    0x3333333333333,
    0x6666666666666,
]);

/// The order of the base point: l = 2^252 + 27742317777372353535851937790883648493
const L: [u8; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
];

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from Ed25519 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ed25519Error {
    /// Signature verification failed.
    VerificationFailed,
    /// Invalid public key (not on curve).
    InvalidPublicKey,
    /// Invalid signature format.
    InvalidSignature,
}

impl fmt::Display for Ed25519Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VerificationFailed => write!(f, "Ed25519 signature verification failed"),
            Self::InvalidPublicKey => write!(f, "invalid Ed25519 public key"),
            Self::InvalidSignature => write!(f, "invalid Ed25519 signature format"),
        }
    }
}

// ─── Extended Point ──────────────────────────────────────────────────────────

/// A point on Ed25519 in extended coordinates (X, Y, Z, T) where
/// x = X/Z, y = Y/Z, T = X*Y/Z.
#[derive(Clone, Copy)]
struct ExtPoint {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

impl ExtPoint {
    /// The identity point (neutral element).
    fn identity() -> Self {
        ExtPoint {
            x: Fe::ZERO,
            y: Fe::ONE,
            z: Fe::ONE,
            t: Fe::ZERO,
        }
    }

    /// Double a point.
    fn double(&self) -> Self {
        let a = self.x.sq();
        let b = self.y.sq();
        let c = self.z.sq().mul_small(2);
        let d = a.neg(); // -a because a = -1 in twisted Edwards
        let e = self.x.add(&self.y).sq().sub(&a).sub(&b);
        let g = d.add(&b);
        let f = g.sub(&c);
        let h = d.sub(&b);
        let x3 = e.mul(&f);
        let y3 = g.mul(&h);
        let t3 = e.mul(&h);
        let z3 = f.mul(&g);
        ExtPoint {
            x: x3,
            y: y3,
            z: z3,
            t: t3,
        }
    }

    /// Add two points.
    fn add(&self, other: &ExtPoint) -> Self {
        let a = self.y.sub(&self.x).mul(&other.y.sub(&other.x));
        let b = self.y.add(&self.x).mul(&other.y.add(&other.x));
        let c = self.t.mul(&other.t).mul(&D2);
        let d = self.z.mul(&other.z).mul_small(2);
        let e = b.sub(&a);
        let f = d.sub(&c);
        let g = d.add(&c);
        let h = b.add(&a);
        let x3 = e.mul(&f);
        let y3 = g.mul(&h);
        let t3 = e.mul(&h);
        let z3 = f.mul(&g);
        ExtPoint {
            x: x3,
            y: y3,
            z: z3,
            t: t3,
        }
    }

    /// Negate a point.
    fn neg(&self) -> Self {
        ExtPoint {
            x: self.x.neg(),
            y: self.y,
            z: self.z,
            t: self.t.neg(),
        }
    }

    /// Scalar multiplication using double-and-add (left-to-right).
    fn scalar_mul(&self, scalar: &[u8; 32]) -> Self {
        let mut result = ExtPoint::identity();
        let mut found_one = false;

        for byte_idx in (0..32).rev() {
            for bit_idx in (0..8).rev() {
                if found_one {
                    result = result.double();
                }
                let bit = (scalar[byte_idx] >> bit_idx) & 1;
                if bit == 1 {
                    if found_one {
                        result = result.add(self);
                    } else {
                        result = *self;
                        found_one = true;
                    }
                }
            }
        }
        result
    }

    /// Encode point to 32 bytes (compressed form: y with sign of x in MSB).
    fn encode(&self) -> [u8; 32] {
        let z_inv = self.z.invert();
        let x = self.x.mul(&z_inv);
        let y = self.y.mul(&z_inv);
        let mut s = y.to_bytes();
        s[31] ^= (x.is_negative() as u8) << 7;
        s
    }

    /// Decode a point from 32 bytes.
    fn decode(s: &[u8; 32]) -> Option<Self> {
        let mut y_bytes = *s;
        let x_sign = (y_bytes[31] >> 7) & 1;
        y_bytes[31] &= 0x7F;

        let y = Fe::from_bytes(&y_bytes);
        let y2 = y.sq();

        // x^2 = (y^2 - 1) / (d*y^2 + 1)
        let num = y2.sub(&Fe::ONE);
        let den = D.mul(&y2).add(&Fe::ONE);
        let den_inv = den.invert();
        let x2 = num.mul(&den_inv);

        if x2.is_zero() {
            if x_sign == 1 {
                return None; // x should be 0 but sign says negative
            }
            return Some(ExtPoint {
                x: Fe::ZERO,
                y,
                z: Fe::ONE,
                t: Fe::ZERO,
            });
        }

        let mut x = x2.sqrt()?;

        if x.is_negative() as u8 != x_sign {
            x = x.neg();
        }

        let t = x.mul(&y);
        Some(ExtPoint {
            x,
            y,
            z: Fe::ONE,
            t,
        })
    }

    /// Check if this is the identity point.
    fn is_identity(&self) -> bool {
        self.x.is_zero() && self.y.mul(&self.z.invert()) == Fe::ONE
    }
}

// ─── Base point ──────────────────────────────────────────────────────────────

/// Get the Ed25519 base point.
fn basepoint() -> ExtPoint {
    // Compute x from y = 4/5 mod p
    let y = Fe::from_limbs([
        0x6666666666658,
        0x4CCCCCCCCCCCC,
        0x1999999999999,
        0x3333333333333,
        0x6666666666666,
    ]);
    let y2 = y.sq();
    let num = y2.sub(&Fe::ONE);
    let den = D.mul(&y2).add(&Fe::ONE);
    let x2 = num.mul(&den.invert());
    let mut x = x2.sqrt().expect("base point x must have sqrt");
    // x should be positive (even)
    if x.is_negative() {
        x = x.neg();
    }
    let t = x.mul(&y);
    ExtPoint {
        x,
        y,
        z: Fe::ONE,
        t,
    }
}

// ─── SHA-512 using SHAKE256 ─────────────────────────────────────────────────

/// Produce 64 bytes of hash output (used as SHA-512 substitute).
fn hash_512(data: &[u8]) -> [u8; 64] {
    let mut xof = Shake256::new();
    xof.absorb(b"Ed25519-SHA512").expect("absorb");
    xof.absorb(data).expect("absorb");
    let mut out = [0u8; 64];
    xof.squeeze(&mut out).expect("squeeze");
    out
}

/// Produce 64 bytes of hash from multiple inputs.
fn hash_512_multi(parts: &[&[u8]]) -> [u8; 64] {
    let mut xof = Shake256::new();
    xof.absorb(b"Ed25519-SHA512").expect("absorb");
    for part in parts {
        xof.absorb(part).expect("absorb");
    }
    let mut out = [0u8; 64];
    xof.squeeze(&mut out).expect("squeeze");
    out
}

// ─── Scalar reduction ───────────────────────────────────────────────────────

/// Load a 512-bit number from 64 bytes into 24 limbs of 21 bits each.
/// This uses a simple, verifiable method: load into u64 limbs first, then extract 21-bit windows.
fn load_512_to_24_limbs(s: &[u8; 64]) -> [i64; 24] {
    // First load into 8 u64 limbs
    let mut w = [0u64; 8];
    for i in 0..8 {
        let base = i * 8;
        for j in 0..8 {
            w[i] |= (s[base + j] as u64) << (j * 8);
        }
    }

    // Now extract 24 windows of 21 bits each from the 512-bit value.
    // Bit position of limb i starts at i*21.
    let mut a = [0i64; 24];
    for i in 0..24 {
        let bit_pos = i * 21;
        let word_idx = bit_pos / 64;
        let bit_idx = bit_pos % 64;

        if word_idx >= 8 {
            break;
        }

        let mut val = w[word_idx] >> bit_idx;
        // If the 21-bit window spans two u64 words, grab bits from the next word
        if bit_idx + 21 > 64 && word_idx + 1 < 8 {
            val |= w[word_idx + 1] << (64 - bit_idx);
        }
        a[i] = (val & 0x1FFFFF) as i64;
    }

    // The last limb (a[23]) gets all remaining bits without masking
    // Bit position 23*21 = 483, word 483/64 = 7, bit 483%64 = 35
    // Remaining bits: 512 - 483 = 29 bits
    let bit_pos = 23 * 21;
    let word_idx = bit_pos / 64;
    let bit_idx = bit_pos % 64;
    if word_idx < 8 {
        a[23] = (w[word_idx] >> bit_idx) as i64;
    }

    a
}

/// Load a 256-bit number from 32 bytes into 12 limbs of 21 bits each.
fn load_256_to_12_limbs(s: &[u8; 32]) -> [i64; 12] {
    let mut w = [0u64; 4];
    for i in 0..4 {
        let base = i * 8;
        for j in 0..8 {
            w[i] |= (s[base + j] as u64) << (j * 8);
        }
    }

    let mut a = [0i64; 12];
    for i in 0..12 {
        let bit_pos = i * 21;
        let word_idx = bit_pos / 64;
        let bit_idx = bit_pos % 64;

        if word_idx >= 4 {
            break;
        }

        let mut val = w[word_idx] >> bit_idx;
        if bit_idx + 21 > 64 && word_idx + 1 < 4 {
            val |= w[word_idx + 1] << (64 - bit_idx);
        }
        a[i] = (val & 0x1FFFFF) as i64;
    }

    // Last limb (a[11]) at bit 231, word 3, bit 39: remaining 256-231=25 bits
    let bit_pos = 11 * 21;
    let word_idx = bit_pos / 64;
    let bit_idx = bit_pos % 64;
    if word_idx < 4 {
        a[11] = (w[word_idx] >> bit_idx) as i64;
    }

    a
}

/// Pack 12 limbs (21 bits each) into 32 bytes, little-endian.
/// Assumes limbs are in [0, 2^21) after carry propagation.
fn pack_limbs_to_bytes(a: &[i64; 24]) -> [u8; 32] {
    // Reconstruct into 4 u64 words
    let mut w = [0u64; 4];
    for i in 0..12 {
        let bit_pos = i * 21;
        let word_idx = bit_pos / 64;
        let bit_idx = bit_pos % 64;

        let val = a[i] as u64;
        w[word_idx] |= val << bit_idx;
        if bit_idx + 21 > 64 && word_idx + 1 < 4 {
            w[word_idx + 1] |= val >> (64 - bit_idx);
        }
    }

    let mut out = [0u8; 32];
    for i in 0..4 {
        let base = i * 8;
        for j in 0..8 {
            out[base + j] = (w[i] >> (j * 8)) as u8;
        }
    }
    out
}

/// Reduce 24 limbs (21 bits each) modulo l using the donna/ref10 algorithm.
/// l decomposed: c_0=666643, c_1=470296, c_2=654183, c_3=-997805, c_4=136657, c_5=-683901
fn reduce_limbs_mod_l(a: &mut [i64; 24]) {
    // First pass: fold limbs 18..23 down by 12 positions
    a[11] += a[23] * 666643;
    a[12] += a[23] * 470296;
    a[13] += a[23] * 654183;
    a[14] -= a[23] * 997805;
    a[15] += a[23] * 136657;
    a[16] -= a[23] * 683901;
    a[23] = 0;

    a[10] += a[22] * 666643;
    a[11] += a[22] * 470296;
    a[12] += a[22] * 654183;
    a[13] -= a[22] * 997805;
    a[14] += a[22] * 136657;
    a[15] -= a[22] * 683901;
    a[22] = 0;

    a[9] += a[21] * 666643;
    a[10] += a[21] * 470296;
    a[11] += a[21] * 654183;
    a[12] -= a[21] * 997805;
    a[13] += a[21] * 136657;
    a[14] -= a[21] * 683901;
    a[21] = 0;

    a[8] += a[20] * 666643;
    a[9] += a[20] * 470296;
    a[10] += a[20] * 654183;
    a[11] -= a[20] * 997805;
    a[12] += a[20] * 136657;
    a[13] -= a[20] * 683901;
    a[20] = 0;

    a[7] += a[19] * 666643;
    a[8] += a[19] * 470296;
    a[9] += a[19] * 654183;
    a[10] -= a[19] * 997805;
    a[11] += a[19] * 136657;
    a[12] -= a[19] * 683901;
    a[19] = 0;

    a[6] += a[18] * 666643;
    a[7] += a[18] * 470296;
    a[8] += a[18] * 654183;
    a[9] -= a[18] * 997805;
    a[10] += a[18] * 136657;
    a[11] -= a[18] * 683901;
    a[18] = 0;

    // Carry propagation with rounding (signed → centered representation)
    for i in 0..17 {
        let carry = (a[i] + (1 << 20)) >> 21;
        a[i] -= carry << 21;
        a[i + 1] += carry;
    }

    // Second pass: fold limbs 12..17 down
    a[0] += a[12] * 666643;
    a[1] += a[12] * 470296;
    a[2] += a[12] * 654183;
    a[3] -= a[12] * 997805;
    a[4] += a[12] * 136657;
    a[5] -= a[12] * 683901;
    a[12] = 0;

    a[1] += a[13] * 666643;
    a[2] += a[13] * 470296;
    a[3] += a[13] * 654183;
    a[4] -= a[13] * 997805;
    a[5] += a[13] * 136657;
    a[6] -= a[13] * 683901;
    a[13] = 0;

    a[2] += a[14] * 666643;
    a[3] += a[14] * 470296;
    a[4] += a[14] * 654183;
    a[5] -= a[14] * 997805;
    a[6] += a[14] * 136657;
    a[7] -= a[14] * 683901;
    a[14] = 0;

    a[3] += a[15] * 666643;
    a[4] += a[15] * 470296;
    a[5] += a[15] * 654183;
    a[6] -= a[15] * 997805;
    a[7] += a[15] * 136657;
    a[8] -= a[15] * 683901;
    a[15] = 0;

    a[4] += a[16] * 666643;
    a[5] += a[16] * 470296;
    a[6] += a[16] * 654183;
    a[7] -= a[16] * 997805;
    a[8] += a[16] * 136657;
    a[9] -= a[16] * 683901;
    a[16] = 0;

    a[5] += a[17] * 666643;
    a[6] += a[17] * 470296;
    a[7] += a[17] * 654183;
    a[8] -= a[17] * 997805;
    a[9] += a[17] * 136657;
    a[10] -= a[17] * 683901;
    a[17] = 0;

    // Unsigned carry propagation through a[0..12]
    for i in 0..12 {
        let carry = a[i] >> 21;
        a[i] -= carry << 21;
        a[i + 1] += carry;
    }

    // Fold a[12] one more time
    a[0] += a[12] * 666643;
    a[1] += a[12] * 470296;
    a[2] += a[12] * 654183;
    a[3] -= a[12] * 997805;
    a[4] += a[12] * 136657;
    a[5] -= a[12] * 683901;
    a[12] = 0;

    // Final unsigned carry propagation
    for i in 0..11 {
        let carry = a[i] >> 21;
        a[i] -= carry << 21;
        a[i + 1] += carry;
    }
}

/// Reduce a 64-byte (512-bit) scalar modulo l.
fn sc_reduce(s: &[u8; 64]) -> [u8; 32] {
    let mut a = load_512_to_24_limbs(s);
    reduce_limbs_mod_l(&mut a);
    pack_limbs_to_bytes(&a)
}

/// Scalar multiply-add: r = (a*b + c) mod l.
fn sc_muladd(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> [u8; 32] {
    let al = load_256_to_12_limbs(a);
    let bl = load_256_to_12_limbs(b);
    let cl = load_256_to_12_limbs(c);

    // Schoolbook multiply a*b -> 24 limbs (21-bit each)
    let mut product = [0i64; 24];
    for i in 0..12 {
        for j in 0..12 {
            product[i + j] += al[i] * bl[j];
        }
    }

    // Add c to the lower limbs
    for i in 0..12 {
        product[i] += cl[i];
    }

    // Carry propagation to keep limbs bounded before reduction
    for i in 0..23 {
        let carry = product[i] >> 21;
        product[i] &= 0x1FFFFF;
        product[i + 1] += carry;
    }

    reduce_limbs_mod_l(&mut product);
    pack_limbs_to_bytes(&product)
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// Ed25519 public key (32 bytes).
#[derive(Clone, PartialEq, Eq)]
pub struct Ed25519PublicKey {
    bytes: [u8; ED25519_PK_LEN],
}

impl Ed25519PublicKey {
    pub fn from_bytes(bytes: [u8; ED25519_PK_LEN]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; ED25519_PK_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for Ed25519PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ed25519PublicKey({} bytes)", ED25519_PK_LEN)
    }
}

/// Ed25519 secret key (64 bytes: seed || public_key).
#[derive(Clone, PartialEq, Eq)]
pub struct Ed25519SecretKey {
    bytes: [u8; ED25519_SK_LEN],
}

impl Ed25519SecretKey {
    pub fn from_bytes(bytes: [u8; ED25519_SK_LEN]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; ED25519_SK_LEN] {
        &self.bytes
    }

    pub fn seed(&self) -> &[u8; 32] {
        self.bytes[..32].try_into().unwrap()
    }

    pub fn public_key_bytes(&self) -> &[u8; 32] {
        self.bytes[32..].try_into().unwrap()
    }
}

impl fmt::Debug for Ed25519SecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ed25519SecretKey([REDACTED])")
    }
}

/// Ed25519 signature (64 bytes: R || S).
#[derive(Clone, PartialEq, Eq)]
pub struct Ed25519Signature {
    bytes: [u8; ED25519_SIG_LEN],
}

impl Ed25519Signature {
    pub fn from_bytes(bytes: [u8; ED25519_SIG_LEN]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; ED25519_SIG_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for Ed25519Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ed25519Signature({} bytes)", ED25519_SIG_LEN)
    }
}

/// Ed25519 key pair.
pub struct Ed25519KeyPair {
    pub public_key: Ed25519PublicKey,
    pub secret_key: Ed25519SecretKey,
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Generate an Ed25519 key pair from a 32-byte seed.
pub fn ed25519_keygen(seed: &[u8; 32]) -> Ed25519KeyPair {
    // Hash the seed to get the expanded secret key
    let h = hash_512(seed);

    let mut a = [0u8; 32];
    a.copy_from_slice(&h[..32]);
    // Clamp
    a[0] &= 248;
    a[31] &= 63;
    a[31] |= 64;

    // Compute public key = a * B
    let bp = basepoint();
    let pk_point = bp.scalar_mul(&a);
    let pk_bytes = pk_point.encode();

    // Secret key = seed || public_key
    let mut sk_bytes = [0u8; ED25519_SK_LEN];
    sk_bytes[..32].copy_from_slice(seed);
    sk_bytes[32..].copy_from_slice(&pk_bytes);

    Ed25519KeyPair {
        public_key: Ed25519PublicKey::from_bytes(pk_bytes),
        secret_key: Ed25519SecretKey::from_bytes(sk_bytes),
    }
}

/// Sign a message using Ed25519.
pub fn ed25519_sign(sk: &Ed25519SecretKey, message: &[u8]) -> Ed25519Signature {
    let seed = sk.seed();
    let pk_bytes = sk.public_key_bytes();

    // Expand secret key
    let h = hash_512(seed);
    let mut a = [0u8; 32];
    a.copy_from_slice(&h[..32]);
    a[0] &= 248;
    a[31] &= 63;
    a[31] |= 64;

    // r = H(h[32..64] || message) mod l
    let r_hash = hash_512_multi(&[&h[32..], message]);
    let r = sc_reduce(&r_hash);

    // R = r * B
    let bp = basepoint();
    let r_point = bp.scalar_mul(&r);
    let r_bytes = r_point.encode();

    // S = (r + H(R || pk || message) * a) mod l
    let k_hash = hash_512_multi(&[&r_bytes, pk_bytes, message]);
    let k = sc_reduce(&k_hash);

    let s = sc_muladd(&k, &a, &r);

    let mut sig = [0u8; ED25519_SIG_LEN];
    sig[..32].copy_from_slice(&r_bytes);
    sig[32..].copy_from_slice(&s);

    Ed25519Signature::from_bytes(sig)
}

/// Verify an Ed25519 signature.
pub fn ed25519_verify(
    pk: &Ed25519PublicKey,
    message: &[u8],
    signature: &Ed25519Signature,
) -> Result<(), Ed25519Error> {
    let sig = signature.as_bytes();

    // Decode R
    let r_bytes: &[u8; 32] = sig[..32].try_into().unwrap();
    let r_point = ExtPoint::decode(r_bytes).ok_or(Ed25519Error::InvalidSignature)?;

    // Decode S (must be < l)
    let s_bytes: &[u8; 32] = sig[32..].try_into().unwrap();

    // Check S < l
    if !scalar_is_canonical(s_bytes) {
        return Err(Ed25519Error::InvalidSignature);
    }

    // Decode public key
    let pk_point = ExtPoint::decode(pk.as_bytes()).ok_or(Ed25519Error::InvalidPublicKey)?;

    // k = H(R || pk || message) mod l
    let k_hash = hash_512_multi(&[r_bytes, pk.as_bytes(), message]);
    let k = sc_reduce(&k_hash);

    // Check: S*B = R + k*A
    // Equivalently: S*B - k*A = R
    let bp = basepoint();
    let sb = bp.scalar_mul(s_bytes);
    let ka = pk_point.scalar_mul(&k);
    let rhs = sb.add(&ka.neg());

    let rhs_bytes = rhs.encode();
    if rhs_bytes == *r_bytes {
        Ok(())
    } else {
        Err(Ed25519Error::VerificationFailed)
    }
}

/// Check if a scalar is canonical (< l).
fn scalar_is_canonical(s: &[u8; 32]) -> bool {
    // Compare s < L byte by byte from MSB
    for i in (0..32).rev() {
        if s[i] < L[i] {
            return true;
        }
        if s[i] > L[i] {
            return false;
        }
    }
    false // s == L is not canonical
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ed25519_keygen_produces_valid_pk() {
        let seed = [0x42u8; 32];
        let kp = ed25519_keygen(&seed);
        // Public key should be decodable
        let pk_point = ExtPoint::decode(kp.public_key.as_bytes());
        assert!(pk_point.is_some(), "public key should decode");
    }

    #[test]
    fn ed25519_deterministic_keygen() {
        let seed = [0xAAu8; 32];
        let kp1 = ed25519_keygen(&seed);
        let kp2 = ed25519_keygen(&seed);
        assert_eq!(kp1.public_key.as_bytes(), kp2.public_key.as_bytes());
    }

    #[test]
    fn sc_reduce_known_value() {
        use core::fmt::Write;
        // Test: reduce bytes(0..64) mod l
        let mut input = [0u8; 64];
        for i in 0..64 {
            input[i] = i as u8;
        }

        // Debug: check loading
        let a = load_512_to_24_limbs(&input);
        // Python-verified expected limbs from load:
        let py_limbs: [i64; 24] = [
            131328, 532504, 115073, 1315344, 1097904, 493318, 541760, 164450, 1512981, 1099968,
            460486, 1981498, 135681, 1184145, 1874068, 337156, 797482, 1667433, 805899, 550500,
            1270611, 1874971, 1109224, 132630439,
        ];

        // Check if any limbs differ from what Python computed
        let mut bad = false;
        let mut msg = alloc::string::String::new();
        for i in 0..24 {
            if a[i] != py_limbs[i] {
                write!(&mut msg, "a[{}]: rust={} py={}\n", i, a[i], py_limbs[i]).unwrap();
                bad = true;
            }
        }
        if bad {
            panic!("Limb loading mismatch:\n{}", msg);
        }

        // Full test
        let result = sc_reduce(&input);
        let expected: [u8; 32] = [
            0x7a, 0x3c, 0x62, 0x82, 0xf0, 0x2d, 0x37, 0xa0, 0x50, 0x23, 0xb6, 0x0d, 0x54, 0x28,
            0xe6, 0xcc, 0x59, 0x61, 0xd4, 0xc3, 0x12, 0x21, 0x93, 0x7a, 0xda, 0xe0, 0xb5, 0x74,
            0xe4, 0xd0, 0x72, 0x05,
        ];
        if result != expected {
            let mut msg2 = alloc::string::String::new();
            write!(&mut msg2, "got:      ").unwrap();
            for b in &result {
                write!(&mut msg2, "{:02x}", b).unwrap();
            }
            write!(&mut msg2, "\nexpected: ").unwrap();
            for b in &expected {
                write!(&mut msg2, "{:02x}", b).unwrap();
            }
            panic!("sc_reduce mismatch\n{}", msg2);
        }
    }

    #[test]
    fn ed25519_sign_verify_roundtrip() {
        let seed = [0x42u8; 32];
        let kp = ed25519_keygen(&seed);
        let message = b"Hello, Ed25519!";

        let sig = ed25519_sign(&kp.secret_key, message);
        let result = ed25519_verify(&kp.public_key, message, &sig);
        assert!(result.is_ok(), "signature should verify");
    }

    #[test]
    fn ed25519_wrong_message_fails() {
        let seed = [0x42u8; 32];
        let kp = ed25519_keygen(&seed);

        let sig = ed25519_sign(&kp.secret_key, b"correct message");
        let result = ed25519_verify(&kp.public_key, b"wrong message", &sig);
        assert_eq!(result, Err(Ed25519Error::VerificationFailed));
    }

    #[test]
    fn ed25519_wrong_key_fails() {
        let kp1 = ed25519_keygen(&[0x01u8; 32]);
        let kp2 = ed25519_keygen(&[0x02u8; 32]);
        let message = b"test message";

        let sig = ed25519_sign(&kp1.secret_key, message);
        let result = ed25519_verify(&kp2.public_key, message, &sig);
        assert_eq!(result, Err(Ed25519Error::VerificationFailed));
    }

    #[test]
    fn ed25519_different_seeds_different_keys() {
        let kp1 = ed25519_keygen(&[0x01u8; 32]);
        let kp2 = ed25519_keygen(&[0x02u8; 32]);
        assert_ne!(kp1.public_key.as_bytes(), kp2.public_key.as_bytes());
    }

    #[test]
    fn ed25519_empty_message() {
        let seed = [0x42u8; 32];
        let kp = ed25519_keygen(&seed);
        let sig = ed25519_sign(&kp.secret_key, b"");
        assert!(ed25519_verify(&kp.public_key, b"", &sig).is_ok());
    }

    #[test]
    fn ed25519_long_message() {
        let seed = [0x42u8; 32];
        let kp = ed25519_keygen(&seed);
        let message = [0xABu8; 1024];
        let sig = ed25519_sign(&kp.secret_key, &message);
        assert!(ed25519_verify(&kp.public_key, &message, &sig).is_ok());
    }

    #[test]
    fn basepoint_is_on_curve() {
        let bp = basepoint();
        // Check: -x^2 + y^2 = 1 + d*x^2*y^2
        let x = bp.x.mul(&bp.z.invert());
        let y = bp.y.mul(&bp.z.invert());
        let x2 = x.sq();
        let y2 = y.sq();
        let lhs = y2.sub(&x2);
        let rhs = Fe::ONE.add(&D.mul(&x2).mul(&y2));
        assert_eq!(lhs, rhs, "base point should satisfy curve equation");
    }

    #[test]
    fn scalar_reduce_identity() {
        // Reducing a small number should be identity
        let mut input = [0u8; 64];
        input[0] = 42;
        let result = sc_reduce(&input);
        assert_eq!(result[0], 42);
        assert!(result[1..].iter().all(|&b| b == 0));
    }

    #[test]
    fn scalar_canonical_check() {
        let zero = [0u8; 32];
        assert!(scalar_is_canonical(&zero));

        let mut almost_l = L;
        almost_l[0] -= 1;
        assert!(scalar_is_canonical(&almost_l));

        assert!(!scalar_is_canonical(&L));
    }
}

#[cfg(test)]
mod sign_debug_tests {
    use super::*;

    #[test]
    fn debug_sign_verify_step_by_step() {
        let seed = [0x42u8; 32];
        let kp = ed25519_keygen(&seed);
        let message = b"Hello, Ed25519!";

        // Replicate signing internals
        let pk_bytes = kp.secret_key.public_key_bytes();
        let h = hash_512(&seed);
        let mut a = [0u8; 32];
        a.copy_from_slice(&h[..32]);
        a[0] &= 248;
        a[31] &= 63;
        a[31] |= 64;

        // r = H(h[32..64] || message) mod l
        let r_hash = hash_512_multi(&[&h[32..], message.as_slice()]);
        let r = sc_reduce(&r_hash);

        // R = r * B
        let bp = basepoint();
        let r_point = bp.scalar_mul(&r);
        let r_bytes = r_point.encode();

        // k = H(R || pk || message) mod l
        let k_hash_sign = hash_512_multi(&[&r_bytes, pk_bytes, message.as_slice()]);
        let k_sign = sc_reduce(&k_hash_sign);

        // S = (k*a + r) mod l
        let s = sc_muladd(&k_sign, &a, &r);

        // Now replicate verification
        let sig_r_bytes: &[u8; 32] = &r_bytes;
        let k_hash_verify = hash_512_multi(&[sig_r_bytes.as_slice(), pk_bytes, message.as_slice()]);
        let k_verify = sc_reduce(&k_hash_verify);

        // k should be the same in sign and verify
        assert_eq!(
            k_sign, k_verify,
            "challenge hash k must match between sign and verify"
        );

        // Check equation: S*B = R + k*A
        let sb = bp.scalar_mul(&s);
        let a_point = bp.scalar_mul(&a);
        let ka = a_point.scalar_mul(&k_sign);
        let rhs = r_point.add(&ka);

        assert_eq!(sb.encode(), rhs.encode(), "S*B must equal R + k*A");

        // Now check what verify actually computes
        let pk_point = ExtPoint::decode(pk_bytes).unwrap();
        let ka_verify = pk_point.scalar_mul(&k_verify);
        let rhs_verify = sb.add(&ka_verify.neg());

        assert_eq!(rhs_verify.encode(), r_bytes, "S*B - k*A must equal R");

        // Check pk_point matches a_point
        let a_encoded = a_point.encode();
        let pk_encoded = pk_point.encode();
        assert_eq!(a_encoded, *pk_bytes, "a*B must equal pk");
    }
}
