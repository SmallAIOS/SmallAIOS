// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Field arithmetic for GF(2^255 - 19).
//!
//! Implements modular arithmetic over the prime field used by Curve25519
//! and Ed25519. Uses a radix-2^51 representation with 5 limbs.
//!
//! All operations are designed to be constant-time (no secret-dependent branches).

#![allow(unused)]
#![allow(clippy::needless_range_loop)]

/// A field element in GF(2^255 - 19), represented as 5 limbs in radix 2^51.
#[derive(Clone, Copy)]
pub struct Fe([u64; 5]);

/// The prime p = 2^255 - 19.
const P: [u64; 5] = [
    0x7FFFFFFFFFFED, // 2^51 - 19
    0x7FFFFFFFFFFFF,
    0x7FFFFFFFFFFFF,
    0x7FFFFFFFFFFFF,
    0x7FFFFFFFFFFFF,
];

impl Fe {
    pub const ZERO: Fe = Fe([0, 0, 0, 0, 0]);
    pub const ONE: Fe = Fe([1, 0, 0, 0, 0]);

    /// Create from 5 limbs.
    pub const fn from_limbs(limbs: [u64; 5]) -> Self {
        Fe(limbs)
    }

    /// Load a field element from 32 bytes (little-endian).
    pub fn from_bytes(s: &[u8; 32]) -> Self {
        let mut h = [0u64; 5];
        // Load 32 bytes into 5 51-bit limbs using 8-byte overlapping reads
        let load8 = |b: &[u8], start: usize| -> u64 {
            let mut r = 0u64;
            for i in 0..8 {
                if start + i < b.len() {
                    r |= (b[start + i] as u64) << (8 * i);
                }
            }
            r
        };
        // limb 0: bits 0..50   (start byte 0, no shift)
        h[0] = load8(s, 0) & 0x7FFFFFFFFFFFF;
        // limb 1: bits 51..101 (start byte 6, shift right 3)
        h[1] = (load8(s, 6) >> 3) & 0x7FFFFFFFFFFFF;
        // limb 2: bits 102..152 (start byte 12, shift right 6)
        h[2] = (load8(s, 12) >> 6) & 0x7FFFFFFFFFFFF;
        // limb 3: bits 153..203 (start byte 19, shift right 1)
        h[3] = (load8(s, 19) >> 1) & 0x7FFFFFFFFFFFF;
        // limb 4: bits 204..254 (start byte 25, shift right 4)
        h[4] = (load8(s, 25) >> 4) & 0x7FFFFFFFFFFFF;
        Fe(h)
    }

    /// Store a field element to 32 bytes (little-endian), fully reduced.
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut h = self.reduce();
        // Fully reduce: ensure h < p
        // Add 19 and check if >= 2^255
        let mut q = (h.0[0] + 19) >> 51;
        q = (h.0[1] + q) >> 51;
        q = (h.0[2] + q) >> 51;
        q = (h.0[3] + q) >> 51;
        q = (h.0[4] + q) >> 51;

        // q is 0 or 1. If 1, subtract p (equivalently, add 19 and reduce)
        h.0[0] += 19 * q;
        let mut carry = h.0[0] >> 51;
        h.0[0] &= 0x7FFFFFFFFFFFF;
        for i in 1..5 {
            h.0[i] += carry;
            carry = h.0[i] >> 51;
            h.0[i] &= 0x7FFFFFFFFFFFF;
        }

        // Pack 5 limbs of 51 bits into 32 bytes (little-endian).
        // Use a bit accumulator to shift out bytes.
        let mut s = [0u8; 32];
        let mut acc: u64 = 0;
        let mut acc_bits: u32 = 0;
        let mut byte_pos = 0;

        for &limb in &h.0 {
            // Add this limb to the accumulator.
            // acc_bits is always < 8 when we enter here, so acc_bits + 51 <= 58,
            // which fits in u64.
            acc |= limb << acc_bits;
            acc_bits += 51;

            // Extract complete bytes
            while acc_bits >= 8 && byte_pos < 32 {
                s[byte_pos] = acc as u8;
                acc >>= 8;
                acc_bits -= 8;
                byte_pos += 1;
            }
        }
        // Flush remaining bits
        if byte_pos < 32 && acc_bits > 0 {
            s[byte_pos] = acc as u8;
        }
        s
    }

    /// Carry-propagate to keep limbs < 2^52.
    fn reduce(self) -> Self {
        let mut h = self.0;
        let mut carry;
        for _ in 0..2 {
            carry = h[0] >> 51;
            h[0] &= 0x7FFFFFFFFFFFF;
            h[1] += carry;
            carry = h[1] >> 51;
            h[1] &= 0x7FFFFFFFFFFFF;
            h[2] += carry;
            carry = h[2] >> 51;
            h[2] &= 0x7FFFFFFFFFFFF;
            h[3] += carry;
            carry = h[3] >> 51;
            h[3] &= 0x7FFFFFFFFFFFF;
            h[4] += carry;
            carry = h[4] >> 51;
            h[4] &= 0x7FFFFFFFFFFFF;
            h[0] += carry * 19;
        }
        Fe(h)
    }

    /// Addition: a + b.
    pub fn add(&self, b: &Fe) -> Fe {
        Fe([
            self.0[0] + b.0[0],
            self.0[1] + b.0[1],
            self.0[2] + b.0[2],
            self.0[3] + b.0[3],
            self.0[4] + b.0[4],
        ])
        .reduce()
    }

    /// Subtraction: a - b. Adds 2*p first to ensure no underflow.
    pub fn sub(&self, b: &Fe) -> Fe {
        Fe([
            self.0[0] + 2 * P[0] - b.0[0],
            self.0[1] + 2 * P[1] - b.0[1],
            self.0[2] + 2 * P[2] - b.0[2],
            self.0[3] + 2 * P[3] - b.0[3],
            self.0[4] + 2 * P[4] - b.0[4],
        ])
        .reduce()
    }

    /// Multiplication: a * b using 128-bit intermediates.
    pub fn mul(&self, b: &Fe) -> Fe {
        let a = &self.0;
        let b = &b.0;

        // Precompute 19*b[i] for reduction
        let b1_19 = b[1] * 19;
        let b2_19 = b[2] * 19;
        let b3_19 = b[3] * 19;
        let b4_19 = b[4] * 19;

        // Schoolbook multiply with reduction
        let r0 = (a[0] as u128) * (b[0] as u128)
            + (a[1] as u128) * (b4_19 as u128)
            + (a[2] as u128) * (b3_19 as u128)
            + (a[3] as u128) * (b2_19 as u128)
            + (a[4] as u128) * (b1_19 as u128);

        let r1 = (a[0] as u128) * (b[1] as u128)
            + (a[1] as u128) * (b[0] as u128)
            + (a[2] as u128) * (b4_19 as u128)
            + (a[3] as u128) * (b3_19 as u128)
            + (a[4] as u128) * (b2_19 as u128);

        let r2 = (a[0] as u128) * (b[2] as u128)
            + (a[1] as u128) * (b[1] as u128)
            + (a[2] as u128) * (b[0] as u128)
            + (a[3] as u128) * (b4_19 as u128)
            + (a[4] as u128) * (b3_19 as u128);

        let r3 = (a[0] as u128) * (b[3] as u128)
            + (a[1] as u128) * (b[2] as u128)
            + (a[2] as u128) * (b[1] as u128)
            + (a[3] as u128) * (b[0] as u128)
            + (a[4] as u128) * (b4_19 as u128);

        let r4 = (a[0] as u128) * (b[4] as u128)
            + (a[1] as u128) * (b[3] as u128)
            + (a[2] as u128) * (b[2] as u128)
            + (a[3] as u128) * (b[1] as u128)
            + (a[4] as u128) * (b[0] as u128);

        // Carry propagation
        let mut h = [0u64; 5];
        let mut carry;

        h[0] = r0 as u64 & 0x7FFFFFFFFFFFF;
        carry = (r0 >> 51) as u64;

        let r1 = r1 + carry as u128;
        h[1] = r1 as u64 & 0x7FFFFFFFFFFFF;
        carry = (r1 >> 51) as u64;

        let r2 = r2 + carry as u128;
        h[2] = r2 as u64 & 0x7FFFFFFFFFFFF;
        carry = (r2 >> 51) as u64;

        let r3 = r3 + carry as u128;
        h[3] = r3 as u64 & 0x7FFFFFFFFFFFF;
        carry = (r3 >> 51) as u64;

        let r4 = r4 + carry as u128;
        h[4] = r4 as u64 & 0x7FFFFFFFFFFFF;
        carry = (r4 >> 51) as u64;

        h[0] += carry * 19;
        // One more carry
        let c = h[0] >> 51;
        h[0] &= 0x7FFFFFFFFFFFF;
        h[1] += c;

        Fe(h)
    }

    /// Squaring: a^2 (slightly faster than mul).
    pub fn sq(&self) -> Fe {
        self.mul(self)
    }

    /// Compute self^(2^n) by repeated squaring.
    pub fn sq_n(&self, n: u32) -> Fe {
        let mut r = *self;
        for _ in 0..n {
            r = r.sq();
        }
        r
    }

    /// Multiply by a small constant.
    pub fn mul_small(&self, c: u64) -> Fe {
        let mut h = [0u128; 5];
        for i in 0..5 {
            h[i] = self.0[i] as u128 * c as u128;
        }
        let mut r = [0u64; 5];
        r[0] = h[0] as u64 & 0x7FFFFFFFFFFFF;
        h[1] += h[0] >> 51;
        r[1] = h[1] as u64 & 0x7FFFFFFFFFFFF;
        h[2] += h[1] >> 51;
        r[2] = h[2] as u64 & 0x7FFFFFFFFFFFF;
        h[3] += h[2] >> 51;
        r[3] = h[3] as u64 & 0x7FFFFFFFFFFFF;
        h[4] += h[3] >> 51;
        r[4] = h[4] as u64 & 0x7FFFFFFFFFFFF;
        r[0] += (h[4] >> 51) as u64 * 19;
        let c = r[0] >> 51;
        r[0] &= 0x7FFFFFFFFFFFF;
        r[1] += c;
        Fe(r)
    }

    /// Negate: -a mod p.
    pub fn neg(&self) -> Fe {
        Fe::ZERO.sub(self)
    }

    /// Modular inversion: a^(-1) mod p using Fermat's little theorem.
    /// a^(-1) = a^(p-2) where p = 2^255 - 19.
    pub fn invert(&self) -> Fe {
        // p-2 = 2^255 - 21
        // Using addition chain from donna
        let z2 = self.sq(); // z^2
        let t = z2.sq_n(2); // z^8
        let z9 = t.mul(self); // z^9
        let z11 = z9.mul(&z2); // z^11
        let t = z11.sq(); // z^22
        let z_5_0 = t.mul(&z9); // z^(2^5 - 1)
        let t = z_5_0.sq_n(5); // z^(2^10 - 2^5)
        let z_10_0 = t.mul(&z_5_0); // z^(2^10 - 1)
        let t = z_10_0.sq_n(10); // z^(2^20 - 2^10)
        let z_20_0 = t.mul(&z_10_0); // z^(2^20 - 1)
        let t = z_20_0.sq_n(20); // z^(2^40 - 2^20)
        let t = t.mul(&z_20_0); // z^(2^40 - 1)
        let t = t.sq_n(10); // z^(2^50 - 2^10)
        let z_50_0 = t.mul(&z_10_0); // z^(2^50 - 1)
        let t = z_50_0.sq_n(50); // z^(2^100 - 2^50)
        let z_100_0 = t.mul(&z_50_0); // z^(2^100 - 1)
        let t = z_100_0.sq_n(100); // z^(2^200 - 2^100)
        let t = t.mul(&z_100_0); // z^(2^200 - 1)
        let t = t.sq_n(50); // z^(2^250 - 2^50)
        let t = t.mul(&z_50_0); // z^(2^250 - 1)
        let t = t.sq_n(5); // z^(2^255 - 2^5)
        t.mul(&z11) // z^(2^255 - 21) = z^(p-2)
    }

    /// Compute the square root: returns sqrt(self) if it exists.
    /// Uses the fact that p = 5 mod 8, so sqrt(a) = a^((p+3)/8) * correction.
    pub fn sqrt(&self) -> Option<Fe> {
        // For p = 2^255 - 19, (p+3)/8 = 2^252 - 3
        let a = *self;

        // a^((p-5)/8) = a^(2^252 - 4)
        let a2 = a.sq();
        let a4 = a2.sq();
        let a8 = a4.sq();
        let a9 = a8.mul(&a);
        let a11 = a9.mul(&a2);

        // Compute a^(2^252 - 3) = a * a^(2^252 - 4)
        // Start with a^(p-5)/8 using the same chain as invert but different exponent
        let pow_p58 = self.pow_p58();

        // beta = a * a^((p-5)/8) = a^((p+3)/8)
        let beta = a.mul(&pow_p58);

        // Check: beta^2 == a?
        let beta2 = beta.sq();
        let check = beta2.sub(&a);
        let check_bytes = check.to_bytes();
        let is_zero = check_bytes.iter().all(|&b| b == 0);

        if is_zero {
            return Some(beta);
        }

        // Try beta * sqrt(-1)
        let sqrt_m1 = Fe::SQRT_M1;
        let beta_i = beta.mul(&sqrt_m1);
        let beta_i2 = beta_i.sq();
        let check2 = beta_i2.sub(&a);
        let check2_bytes = check2.to_bytes();
        let is_zero2 = check2_bytes.iter().all(|&b| b == 0);

        if is_zero2 {
            Some(beta_i)
        } else {
            None
        }
    }

    /// a^((p-5)/8) using addition chain.
    fn pow_p58(&self) -> Fe {
        // (p-5)/8 = 2^252 - 3
        // This is the same exponent chain as invert but to 2^252-3 instead of 2^255-21
        let z2 = self.sq();
        let t = z2.sq_n(2);
        let z9 = t.mul(self);
        let z11 = z9.mul(&z2);
        let t = z11.sq();
        let z_5_0 = t.mul(&z9);
        let t = z_5_0.sq_n(5);
        let z_10_0 = t.mul(&z_5_0);
        let t = z_10_0.sq_n(10);
        let z_20_0 = t.mul(&z_10_0);
        let t = z_20_0.sq_n(20);
        let t = t.mul(&z_20_0);
        let t = t.sq_n(10);
        let z_50_0 = t.mul(&z_10_0);
        let t = z_50_0.sq_n(50);
        let z_100_0 = t.mul(&z_50_0);
        let t = z_100_0.sq_n(100);
        let t = t.mul(&z_100_0);
        let t = t.sq_n(50);
        let t = t.mul(&z_50_0);
        let t = t.sq_n(2);
        t.mul(self) // a^(2^252 - 3)
    }

    /// sqrt(-1) mod p = 2^((p-1)/4) mod p.
    /// Precomputed constant.
    pub const SQRT_M1: Fe = Fe([
        0x61B274A0EA0B0,
        0x0D5A5FC8F189D,
        0x7EF5E9CBD0C60,
        0x78595A6804C9E,
        0x2B8324804FC1D,
    ]);

    /// Check if the field element is zero.
    pub fn is_zero(&self) -> bool {
        let bytes = self.to_bytes();
        bytes.iter().all(|&b| b == 0)
    }

    /// Check if the field element is negative (LSB of encoded form is 1).
    pub fn is_negative(&self) -> bool {
        let bytes = self.to_bytes();
        bytes[0] & 1 == 1
    }

    /// Constant-time conditional swap: if flag != 0, swap self and other.
    pub fn cswap(&mut self, other: &mut Fe, flag: u64) {
        let mask = 0u64.wrapping_sub(flag); // 0 or 0xFFFFFFFFFFFFFFFF
        for i in 0..5 {
            let t = mask & (self.0[i] ^ other.0[i]);
            self.0[i] ^= t;
            other.0[i] ^= t;
        }
    }

    /// Constant-time conditional move: if flag != 0, set self = src.
    pub fn cmov(&mut self, src: &Fe, flag: u64) {
        let mask = 0u64.wrapping_sub(flag);
        for i in 0..5 {
            self.0[i] ^= mask & (self.0[i] ^ src.0[i]);
        }
    }

    /// Absolute value: if self is negative, negate it.
    pub fn abs(&self) -> Fe {
        if self.is_negative() {
            self.neg()
        } else {
            *self
        }
    }
}

impl PartialEq for Fe {
    fn eq(&self, other: &Fe) -> bool {
        self.to_bytes() == other.to_bytes()
    }
}

impl Eq for Fe {}

impl core::fmt::Debug for Fe {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bytes = self.to_bytes();
        write!(f, "Fe(")?;
        for b in &bytes[..4] {
            write!(f, "{:02x}", b)?;
        }
        write!(f, "...)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fe_zero() {
        let z = Fe::ZERO;
        let bytes = z.to_bytes();
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn fe_one() {
        let one = Fe::ONE;
        let bytes = one.to_bytes();
        assert_eq!(bytes[0], 1);
        assert!(bytes[1..].iter().all(|&b| b == 0));
    }

    #[test]
    fn fe_roundtrip() {
        let mut input = [0u8; 32];
        input[0] = 42;
        input[10] = 0xFF;
        input[31] = 0x7F; // Keep MSB clear to avoid reduction
        let fe = Fe::from_bytes(&input);
        let output = fe.to_bytes();
        assert_eq!(input, output);
    }

    #[test]
    fn fe_add_sub() {
        let a = Fe::from_bytes(&{
            let mut b = [0u8; 32];
            b[0] = 100;
            b
        });
        let b = Fe::from_bytes(&{
            let mut b = [0u8; 32];
            b[0] = 50;
            b
        });
        let c = a.add(&b);
        let d = c.sub(&b);
        assert_eq!(a, d);
    }

    #[test]
    fn fe_mul_one() {
        let a = Fe::from_bytes(&{
            let mut b = [0u8; 32];
            b[0] = 42;
            b
        });
        let r = a.mul(&Fe::ONE);
        assert_eq!(a, r);
    }

    #[test]
    fn fe_mul_zero() {
        let a = Fe::from_bytes(&{
            let mut b = [0u8; 32];
            b[0] = 42;
            b
        });
        let r = a.mul(&Fe::ZERO);
        assert_eq!(Fe::ZERO, r);
    }

    #[test]
    fn fe_invert() {
        let a = Fe::from_bytes(&{
            let mut b = [0u8; 32];
            b[0] = 42;
            b
        });
        let a_inv = a.invert();
        let product = a.mul(&a_inv);
        assert_eq!(product, Fe::ONE);
    }

    #[test]
    fn fe_sq_vs_mul() {
        let a = Fe::from_bytes(&{
            let mut b = [0u8; 32];
            b[0] = 7;
            b[5] = 0x33;
            b
        });
        let sq = a.sq();
        let mul = a.mul(&a);
        assert_eq!(sq, mul);
    }

    #[test]
    fn fe_neg() {
        let a = Fe::from_bytes(&{
            let mut b = [0u8; 32];
            b[0] = 42;
            b
        });
        let neg_a = a.neg();
        let sum = a.add(&neg_a);
        assert_eq!(sum, Fe::ZERO);
    }

    #[test]
    fn fe_sqrt_m1_squared() {
        let i = Fe::SQRT_M1;
        let i2 = i.sq();
        let neg_one = Fe::ONE.neg();
        assert_eq!(i2, neg_one, "sqrt(-1)^2 should equal -1");
    }
}
