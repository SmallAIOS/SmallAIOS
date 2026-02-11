// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! X25519 Diffie-Hellman key exchange (RFC 7748).
//!
//! X25519 is an elliptic curve Diffie-Hellman function using Curve25519.
//! It provides ~128-bit security with 32-byte keys and shared secrets.
//!
//! # Operations
//!
//! 1. **KeyGen**: Generate (public_key, secret_key) from 32 bytes of randomness
//! 2. **DH**: Compute shared secret from (my_secret, their_public)

#![allow(unused)]
#![allow(clippy::needless_range_loop)]

use super::field25519::Fe;
use core::fmt;

/// X25519 key length in bytes.
pub const X25519_KEY_LEN: usize = 32;

/// The basepoint for Curve25519: u = 9.
const BASEPOINT: [u8; 32] = {
    let mut b = [0u8; 32];
    b[0] = 9;
    b
};

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from X25519 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X25519Error {
    /// The computed shared secret is all zeros (low-order point).
    LowOrderPoint,
}

impl fmt::Display for X25519Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LowOrderPoint => write!(f, "X25519 shared secret is zero (low-order point)"),
        }
    }
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// An X25519 public key (32 bytes, a u-coordinate on Curve25519).
#[derive(Clone, PartialEq, Eq)]
pub struct X25519PublicKey {
    bytes: [u8; X25519_KEY_LEN],
}

impl X25519PublicKey {
    pub fn from_bytes(bytes: [u8; X25519_KEY_LEN]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; X25519_KEY_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for X25519PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "X25519PublicKey({} bytes)", X25519_KEY_LEN)
    }
}

/// An X25519 secret key (32 bytes, clamped scalar).
#[derive(Clone, PartialEq, Eq)]
pub struct X25519SecretKey {
    bytes: [u8; X25519_KEY_LEN],
}

impl X25519SecretKey {
    pub fn from_bytes(bytes: [u8; X25519_KEY_LEN]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; X25519_KEY_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for X25519SecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "X25519SecretKey([REDACTED])")
    }
}

/// An X25519 key pair.
pub struct X25519KeyPair {
    pub public_key: X25519PublicKey,
    pub secret_key: X25519SecretKey,
}

// ─── Scalar clamping ────────────────────────────────────────────────────────

/// Clamp a 32-byte scalar per RFC 7748.
fn clamp(k: &mut [u8; 32]) {
    k[0] &= 248; // Clear bottom 3 bits
    k[31] &= 127; // Clear top bit
    k[31] |= 64; // Set second-to-top bit
}

// ─── Montgomery ladder ──────────────────────────────────────────────────────

/// X25519 scalar multiplication: compute k * u on Curve25519.
/// Uses the Montgomery ladder (constant-time).
fn x25519_scalar_mult(k: &[u8; 32], u: &[u8; 32]) -> [u8; 32] {
    let mut scalar = *k;
    clamp(&mut scalar);

    // Per RFC 7748, mask the top bit of the u-coordinate
    let mut u_masked = *u;
    u_masked[31] &= 0x7F;
    let u_fe = Fe::from_bytes(&u_masked);

    // Montgomery ladder state: (x2, z2) = (1, 0), (x3, z3) = (u, 1)
    let mut x2 = Fe::ONE;
    let mut z2 = Fe::ZERO;
    let mut x3 = u_fe;
    let mut z3 = Fe::ONE;

    let mut swap: u64 = 0;

    // Process bits from 254 down to 0
    for pos in (0..=254).rev() {
        let byte = scalar[pos / 8];
        let bit = ((byte >> (pos & 7)) & 1) as u64;
        swap ^= bit;
        x2.cswap(&mut x3, swap);
        z2.cswap(&mut z3, swap);
        swap = bit;

        let a = x2.add(&z2);
        let aa = a.sq();
        let b = x2.sub(&z2);
        let bb = b.sq();
        let e = aa.sub(&bb);
        let c = x3.add(&z3);
        let d = x3.sub(&z3);
        let da = d.mul(&a);
        let cb = c.mul(&b);
        x3 = da.add(&cb);
        x3 = x3.sq();
        z3 = da.sub(&cb);
        z3 = z3.sq();
        z3 = z3.mul(&u_fe);
        x2 = aa.mul(&bb);
        // a24 = (A-2)/4 = 121665 for Curve25519 (A = 486662)
        z2 = e.mul_small(121665).add(&aa);
        z2 = z2.mul(&e);
    }

    x2.cswap(&mut x3, swap);
    z2.cswap(&mut z3, swap);

    let result = x2.mul(&z2.invert());
    result.to_bytes()
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Generate an X25519 key pair from 32 bytes of randomness.
pub fn x25519_keygen(seed: &[u8; 32]) -> X25519KeyPair {
    let secret_key = X25519SecretKey::from_bytes(*seed);
    let public_bytes = x25519_scalar_mult(seed, &BASEPOINT);
    let public_key = X25519PublicKey::from_bytes(public_bytes);
    X25519KeyPair {
        public_key,
        secret_key,
    }
}

/// Compute the X25519 shared secret: secret * public.
pub fn x25519_dh(
    secret: &X25519SecretKey,
    public: &X25519PublicKey,
) -> Result<[u8; 32], X25519Error> {
    let result = x25519_scalar_mult(secret.as_bytes(), public.as_bytes());
    // Check for all-zero output (low-order point)
    if result.iter().all(|&b| b == 0) {
        return Err(X25519Error::LowOrderPoint);
    }
    Ok(result)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 7748 test vector 1.
    #[test]
    fn rfc7748_test_vector_1() {
        let scalar = [
            0xa5, 0x46, 0xe3, 0x6b, 0xf0, 0x52, 0x7c, 0x9d, 0x3b, 0x16, 0x15, 0x4b, 0x82, 0x46,
            0x5e, 0xdd, 0x62, 0x14, 0x4c, 0x0a, 0xc1, 0xfc, 0x5a, 0x18, 0x50, 0x6a, 0x22, 0x44,
            0xba, 0x44, 0x9a, 0xc4,
        ];
        let u_coord = [
            0xe6, 0xdb, 0x68, 0x67, 0x58, 0x30, 0x30, 0xdb, 0x35, 0x94, 0xc1, 0xa4, 0x24, 0xb1,
            0x5f, 0x7c, 0x72, 0x66, 0x24, 0xec, 0x26, 0xb3, 0x35, 0x3b, 0x10, 0xa9, 0x03, 0xa6,
            0xd0, 0xab, 0x1c, 0x4c,
        ];
        let expected = [
            0xc3, 0xda, 0x55, 0x37, 0x9d, 0xe9, 0xc6, 0x90, 0x8e, 0x94, 0xea, 0x4d, 0xf2, 0x8d,
            0x08, 0x4f, 0x32, 0xec, 0xcf, 0x03, 0x49, 0x1c, 0x71, 0xf7, 0x54, 0xb4, 0x07, 0x55,
            0x77, 0xa2, 0x85, 0x52,
        ];
        let result = x25519_scalar_mult(&scalar, &u_coord);
        assert_eq!(result, expected, "RFC 7748 test vector 1 failed");
    }

    /// RFC 7748 test vector 2.
    #[test]
    fn rfc7748_test_vector_2() {
        let scalar = [
            0x4b, 0x66, 0xe9, 0xd4, 0xd1, 0xb4, 0x67, 0x3c, 0x5a, 0xd2, 0x26, 0x91, 0x95, 0x7d,
            0x6a, 0xf5, 0xc1, 0x1b, 0x64, 0x21, 0xe0, 0xea, 0x01, 0xd4, 0x2c, 0xa4, 0x16, 0x9e,
            0x79, 0x18, 0xba, 0x0d,
        ];
        let u_coord = [
            0xe5, 0x21, 0x0f, 0x12, 0x78, 0x68, 0x11, 0xd3, 0xf4, 0xb7, 0x95, 0x9d, 0x05, 0x38,
            0xae, 0x2c, 0x31, 0xdb, 0xe7, 0x10, 0x6f, 0xc0, 0x3c, 0x3e, 0xfc, 0x4c, 0xd5, 0x49,
            0xc7, 0x15, 0xa4, 0x93,
        ];
        let expected = [
            0x95, 0xcb, 0xde, 0x94, 0x76, 0xe8, 0x90, 0x7d, 0x7a, 0xad, 0xe4, 0x5c, 0xb4, 0xb8,
            0x73, 0xf8, 0x8b, 0x59, 0x5a, 0x68, 0x79, 0x9f, 0xa1, 0x52, 0xe6, 0xf8, 0xf7, 0x64,
            0x7a, 0xac, 0x79, 0x57,
        ];
        let result = x25519_scalar_mult(&scalar, &u_coord);
        assert_eq!(result, expected, "RFC 7748 test vector 2 failed");
    }

    /// Test base point multiplication (keygen).
    #[test]
    fn x25519_basepoint_mult() {
        // Known: 9 * basepoint should give a specific result
        let mut scalar = [0u8; 32];
        scalar[0] = 9; // Just use 9 as the scalar
        let result = x25519_scalar_mult(&scalar, &BASEPOINT);
        // The result should not be zero
        assert!(result.iter().any(|&b| b != 0));
    }

    #[test]
    fn x25519_keygen_roundtrip() {
        let seed_a = [0x42u8; 32];
        let seed_b = [0x37u8; 32];

        let kp_a = x25519_keygen(&seed_a);
        let kp_b = x25519_keygen(&seed_b);

        let ss_a = x25519_dh(&kp_a.secret_key, &kp_b.public_key).unwrap();
        let ss_b = x25519_dh(&kp_b.secret_key, &kp_a.public_key).unwrap();

        assert_eq!(ss_a, ss_b, "DH shared secrets should match");
    }

    #[test]
    fn x25519_different_seeds_different_keys() {
        let kp1 = x25519_keygen(&[0x01u8; 32]);
        let kp2 = x25519_keygen(&[0x02u8; 32]);
        assert_ne!(kp1.public_key.as_bytes(), kp2.public_key.as_bytes());
    }

    #[test]
    fn x25519_deterministic() {
        let seed = [0xAAu8; 32];
        let kp1 = x25519_keygen(&seed);
        let kp2 = x25519_keygen(&seed);
        assert_eq!(kp1.public_key.as_bytes(), kp2.public_key.as_bytes());
    }
}
