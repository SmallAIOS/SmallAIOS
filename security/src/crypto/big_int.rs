// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal variable-width unsigned big-integer arithmetic for RSA.
//!
//! Private support module for [`super::rsa_pss`] — not re-exported. Backs
//! a `Vec<u64>` little-endian limb representation and provides only what
//! RSA signature *verification* (`s^e mod n`) needs: add/sub/mul,
//! Montgomery reduction, a constant-time-in-the-exponent Montgomery-ladder
//! `mod_exp`, and a strict DER INTEGER parser.
//!
//! **Timing posture.** Verification handles no secret: the signature `s`,
//! modulus `n`, and exponent `e` are all public. Variable-time bigint
//! operations on these are therefore acceptable, and no attempt is made to
//! hide the operand *sizes*. The modular exponentiation is nonetheless
//! written as a Montgomery ladder whose per-bit operation sequence does not
//! depend on exponent bit values, so the primitive stays reusable for a
//! future signing change where the exponent would be secret (see
//! `openspec/changes/security-rsa-pss-v1/design.md`, D3).

// Limb-indexed loops (CIOS Montgomery multiplication, long division) read
// more clearly with explicit indices than with iterator adapters.
#![allow(clippy::needless_range_loop)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// Errors from DER INTEGER parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BigIntError {
    /// Structurally invalid DER (bad tag, truncation, non-minimal length,
    /// negative INTEGER).
    MalformedDer,
}

/// An arbitrary-precision unsigned integer, little-endian `u64` limbs,
/// always normalized (no most-significant zero limb except for the value
/// zero, represented as an empty limb vector).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigUint {
    limbs: Vec<u64>,
}

impl BigUint {
    /// Zero.
    pub fn zero() -> Self {
        BigUint { limbs: Vec::new() }
    }

    /// A single-limb value.
    pub fn from_u64(v: u64) -> Self {
        let mut b = BigUint { limbs: vec![v] };
        b.normalize();
        b
    }

    fn normalize(&mut self) {
        while let Some(&0) = self.limbs.last() {
            self.limbs.pop();
        }
    }

    /// True if the value is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// True if the value is odd (RSA moduli are odd — checked by callers).
    pub fn is_odd(&self) -> bool {
        self.limbs.first().is_some_and(|&l| l & 1 == 1)
    }

    /// Number of significant bits (0 for zero).
    pub fn bit_len(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => self.limbs.len() * 64 - top.leading_zeros() as usize,
        }
    }

    fn limb_len(&self) -> usize {
        self.limbs.len()
    }

    /// Parse a big-endian byte string as an unsigned integer.
    pub fn from_bytes_be(bytes: &[u8]) -> Self {
        let mut limbs = Vec::new();
        // Walk from the least-significant end in 8-byte groups.
        let mut i = bytes.len();
        while i > 0 {
            let start = i.saturating_sub(8);
            let mut buf = [0u8; 8];
            let chunk = &bytes[start..i];
            buf[8 - chunk.len()..].copy_from_slice(chunk);
            limbs.push(u64::from_be_bytes(buf));
            i = start;
        }
        let mut b = BigUint { limbs };
        b.normalize();
        b
    }

    /// Big-endian byte encoding, left-padded with zeros to exactly `len`
    /// bytes. Returns `None` if the value does not fit in `len` bytes.
    pub fn to_bytes_be_fixed(&self, len: usize) -> Option<Vec<u8>> {
        if self.bit_len() > len * 8 {
            return None;
        }
        let mut out = vec![0u8; len];
        for (i, &limb) in self.limbs.iter().enumerate() {
            let le = limb.to_be_bytes();
            // Limb i occupies bytes [len - 8*(i+1), len - 8*i) from the right.
            for (j, &byte) in le.iter().rev().enumerate() {
                let pos = i * 8 + j;
                if pos < len {
                    out[len - 1 - pos] = byte;
                }
            }
        }
        Some(out)
    }

    /// Compare `self` and `other`: -1 (<), 0 (=), 1 (>).
    pub fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for i in (0..self.limbs.len()).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }

    /// `self + other`. Part of the spec-required bigint surface; exercised
    /// by tests and available for a future signing change.
    #[allow(dead_code)]
    pub fn add(&self, other: &Self) -> Self {
        let n = self.limbs.len().max(other.limbs.len());
        let mut limbs = Vec::with_capacity(n + 1);
        let mut carry = 0u128;
        for i in 0..n {
            let a = *self.limbs.get(i).unwrap_or(&0) as u128;
            let b = *other.limbs.get(i).unwrap_or(&0) as u128;
            let s = a + b + carry;
            limbs.push(s as u64);
            carry = s >> 64;
        }
        if carry != 0 {
            limbs.push(carry as u64);
        }
        let mut r = BigUint { limbs };
        r.normalize();
        r
    }

    /// `self - other`, requires `self >= other`.
    pub fn sub(&self, other: &Self) -> Self {
        debug_assert!(self.cmp(other) != core::cmp::Ordering::Less);
        let mut limbs = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0i128;
        for i in 0..self.limbs.len() {
            let a = self.limbs[i] as i128;
            let b = *other.limbs.get(i).unwrap_or(&0) as i128;
            let mut d = a - b - borrow;
            if d < 0 {
                d += 1i128 << 64;
                borrow = 1;
            } else {
                borrow = 0;
            }
            limbs.push(d as u64);
        }
        let mut r = BigUint { limbs };
        r.normalize();
        r
    }

    /// `self * other` (schoolbook). Part of the spec-required bigint
    /// surface; exercised by tests and available for a future signing change.
    #[allow(dead_code)]
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return BigUint::zero();
        }
        let mut limbs = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &b) in other.limbs.iter().enumerate() {
                let cur = limbs[i + j] as u128 + (a as u128) * (b as u128) + carry;
                limbs[i + j] = cur as u64;
                carry = cur >> 64;
            }
            limbs[i + other.limbs.len()] += carry as u64;
        }
        let mut r = BigUint { limbs };
        r.normalize();
        r
    }

    /// `self mod modulus` via schoolbook bit-at-a-time long division.
    /// Used only for Montgomery-constant setup, on public values.
    pub fn rem(&self, modulus: &Self) -> Self {
        use core::cmp::Ordering;
        debug_assert!(!modulus.is_zero());
        if self.cmp(modulus) == Ordering::Less {
            return self.clone();
        }
        let mut r = BigUint::zero();
        for bit in (0..self.bit_len()).rev() {
            // r = (r << 1) | bit_of_self
            r = r.shl1();
            if self.test_bit(bit) {
                r.set_bit0();
            }
            if r.cmp(modulus) != Ordering::Less {
                r = r.sub(modulus);
            }
        }
        r
    }

    fn test_bit(&self, bit: usize) -> bool {
        let limb = bit / 64;
        let off = bit % 64;
        self.limbs.get(limb).is_some_and(|&l| (l >> off) & 1 == 1)
    }

    fn shl1(&self) -> Self {
        let mut limbs = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry = 0u64;
        for &l in &self.limbs {
            limbs.push((l << 1) | carry);
            carry = l >> 63;
        }
        if carry != 0 {
            limbs.push(carry);
        }
        let mut r = BigUint { limbs };
        r.normalize();
        r
    }

    fn set_bit0(&mut self) {
        if self.limbs.is_empty() {
            self.limbs.push(1);
        } else {
            self.limbs[0] |= 1;
        }
    }

    /// Modular exponentiation `self^exp mod modulus` for an **odd**
    /// modulus, via Montgomery multiplication and a Montgomery ladder that
    /// is constant-time in the exponent bits. `self` must be < `modulus`.
    pub fn mod_exp(&self, exp: &Self, modulus: &Self) -> Self {
        debug_assert!(modulus.is_odd());
        let ctx = MontCtx::new(modulus);
        // R0 = 1 (Montgomery form), R1 = self (Montgomery form).
        let mut r0 = ctx.one_mont.clone();
        let mut r1 = ctx.to_mont(self);
        // Ladder from the most-significant exponent bit down. The two
        // multiplications happen every iteration; only the operand roles
        // swap, selected by a constant-time cswap on the bit value — no
        // exponent-bit-dependent branch or memory index.
        let ebits = exp.bit_len();
        for i in (0..ebits).rev() {
            let bit = exp.test_bit(i) as u64;
            cswap(bit, &mut r0, &mut r1, ctx.n_limbs);
            r1 = ctx.mont_mul(&r0, &r1);
            r0 = ctx.mont_mul(&r0, &r0);
            cswap(bit, &mut r0, &mut r1, ctx.n_limbs);
        }
        ctx.demont(&r0)
    }

    /// Parse a DER INTEGER (`0x02 len content`) as an unsigned big integer,
    /// enforcing minimal encoding and rejecting negatives. Returns the
    /// value and the number of bytes consumed.
    pub fn parse_der_integer(input: &[u8]) -> Result<(Self, usize), BigIntError> {
        let mut r = DerReader::new(input);
        let content = r.read_tag_len(0x02)?;
        if content.is_empty() {
            return Err(BigIntError::MalformedDer);
        }
        if content[0] & 0x80 != 0 {
            // Negative INTEGER — not a valid unsigned value.
            return Err(BigIntError::MalformedDer);
        }
        let digits = if content[0] == 0x00 {
            if content.len() == 1 {
                // INTEGER 0 is well-formed; value zero.
                content
            } else if content[1] & 0x80 == 0 {
                // Redundant leading zero: non-minimal.
                return Err(BigIntError::MalformedDer);
            } else {
                &content[1..]
            }
        } else {
            content
        };
        Ok((BigUint::from_bytes_be(digits), r.pos))
    }
}

// ── Montgomery context (per-modulus) ────────────────────────────────────────

struct MontCtx {
    n: Vec<u64>,       // modulus limbs (fixed length n_limbs)
    n_limbs: usize,    // limb count of the modulus
    n0inv: u64,        // -n^{-1} mod 2^64
    one_mont: BigUint, // R mod n  (Montgomery form of 1)
}

impl MontCtx {
    fn new(modulus: &BigUint) -> Self {
        let n_limbs = modulus.limb_len();
        let mut n = modulus.limbs.clone();
        n.resize(n_limbs, 0);
        let n0inv = neg_inv_u64(n[0]);
        // R = 2^(64*n_limbs); R mod n and R^2 mod n via long division.
        let r = shift_left_limbs(&BigUint::from_u64(1), n_limbs); // 2^(64*n_limbs)
        let one_mont = r.rem(modulus);
        MontCtx {
            n,
            n_limbs,
            n0inv,
            one_mont,
        }
    }

    fn r2(&self) -> BigUint {
        // R^2 mod n = 2^(128*n_limbs) mod n.
        let r2 = shift_left_limbs(&BigUint::from_u64(1), 2 * self.n_limbs);
        let modulus = BigUint {
            limbs: {
                let mut m = self.n.clone();
                while let Some(&0) = m.last() {
                    m.pop();
                }
                m
            },
        };
        r2.rem(&modulus)
    }

    /// Convert `a` (< n) into Montgomery form: `a * R mod n`.
    fn to_mont(&self, a: &BigUint) -> BigUint {
        self.mont_mul(a, &self.r2())
    }

    /// Convert out of Montgomery form: `a * R^{-1} mod n`.
    fn demont(&self, a: &BigUint) -> BigUint {
        let one = BigUint::from_u64(1);
        self.mont_mul(a, &one)
    }

    /// CIOS Montgomery multiplication: `a * b * R^{-1} mod n`, inputs < n.
    fn mont_mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let s = self.n_limbs;
        let a_l = padded(&a.limbs, s);
        let b_l = padded(&b.limbs, s);
        let mut t = vec![0u64; s + 2];
        for i in 0..s {
            // t += a[i] * b
            let mut carry = 0u128;
            for j in 0..s {
                let cur = t[j] as u128 + (a_l[i] as u128) * (b_l[j] as u128) + carry;
                t[j] = cur as u64;
                carry = cur >> 64;
            }
            let sum = t[s] as u128 + carry;
            t[s] = sum as u64;
            t[s + 1] += (sum >> 64) as u64;
            // m = t[0] * n0inv mod 2^64; t += m * n; then shift right one limb.
            let m = t[0].wrapping_mul(self.n0inv);
            let mut carry2 = 0u128;
            for j in 0..s {
                let cur = t[j] as u128 + (m as u128) * (self.n[j] as u128) + carry2;
                t[j] = cur as u64;
                carry2 = cur >> 64;
            }
            let sum = t[s] as u128 + carry2;
            t[s] = sum as u64;
            t[s + 1] += (sum >> 64) as u64;
            // Shift down by one limb (t[0] is now zero).
            for j in 0..=s {
                t[j] = t[j + 1];
            }
            t[s + 1] = 0;
        }
        // t is < 2n in s+1 limbs; conditionally subtract n.
        let mut res = BigUint {
            limbs: t[..=s].to_vec(),
        };
        res.normalize();
        let modulus = BigUint {
            limbs: {
                let mut m = self.n.clone();
                while let Some(&0) = m.last() {
                    m.pop();
                }
                m
            },
        };
        if res.cmp(&modulus) != core::cmp::Ordering::Less {
            res = res.sub(&modulus);
        }
        res
    }
}

/// Constant-time-in-the-exponent conditional swap of two limb vectors.
fn cswap(bit: u64, a: &mut BigUint, b: &mut BigUint, n_limbs: usize) {
    let mask = 0u64.wrapping_sub(bit);
    let mut al = padded(&a.limbs, n_limbs);
    let mut bl = padded(&b.limbs, n_limbs);
    for i in 0..n_limbs {
        let t = mask & (al[i] ^ bl[i]);
        al[i] ^= t;
        bl[i] ^= t;
    }
    a.limbs = al;
    a.normalize();
    b.limbs = bl;
    b.normalize();
}

fn padded(limbs: &[u64], n: usize) -> Vec<u64> {
    let mut v = limbs.to_vec();
    v.resize(n, 0);
    v
}

fn shift_left_limbs(a: &BigUint, limbs: usize) -> BigUint {
    let mut v = vec![0u64; limbs];
    v.extend_from_slice(&a.limbs);
    let mut r = BigUint { limbs: v };
    r.normalize();
    r
}

/// `-m0^{-1} mod 2^64` for odd `m0`, via Newton iteration.
fn neg_inv_u64(m0: u64) -> u64 {
    let mut inv = m0;
    for _ in 0..5 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}

// ── strict DER reader ───────────────────────────────────────────────────────

struct DerReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> DerReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        DerReader { buf, pos: 0 }
    }

    fn byte(&mut self) -> Result<u8, BigIntError> {
        let b = *self.buf.get(self.pos).ok_or(BigIntError::MalformedDer)?;
        self.pos += 1;
        Ok(b)
    }

    fn read_tag_len(&mut self, tag: u8) -> Result<&'a [u8], BigIntError> {
        if self.byte()? != tag {
            return Err(BigIntError::MalformedDer);
        }
        let l0 = self.byte()?;
        let len = match l0 {
            0x00..=0x7F => l0 as usize,
            0x81 => {
                let v = self.byte()? as usize;
                if v < 0x80 {
                    return Err(BigIntError::MalformedDer);
                }
                v
            }
            0x82 => {
                let hi = self.byte()? as usize;
                let lo = self.byte()? as usize;
                let v = (hi << 8) | lo;
                if v < 0x100 {
                    return Err(BigIntError::MalformedDer);
                }
                v
            }
            _ => return Err(BigIntError::MalformedDer),
        };
        if self.pos + len > self.buf.len() {
            return Err(BigIntError::MalformedDer);
        }
        let content = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(v: u64) -> BigUint {
        BigUint::from_u64(v)
    }

    #[test]
    fn add_sub_mul_known_answers() {
        assert_eq!(u(2).add(&u(3)), u(5));
        assert_eq!(u(10).sub(&u(7)), u(3));
        assert_eq!(u(6).mul(&u(7)), u(42));
        // Carry across limbs: (2^64 - 1) + 1 = 2^64.
        let max = BigUint::from_bytes_be(&[0xFF; 8]);
        let two_64 = max.add(&u(1));
        assert_eq!(two_64.bit_len(), 65);
        // Multiply across the limb boundary.
        let prod = two_64.mul(&two_64); // 2^128
        assert_eq!(prod.bit_len(), 129);
    }

    #[test]
    fn bytes_roundtrip() {
        let bytes: &[u8] = &[0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x00, 0x11];
        let n = BigUint::from_bytes_be(bytes);
        assert_eq!(n.to_bytes_be_fixed(10).unwrap(), bytes);
        // Left-pad to a wider width.
        let padded = n.to_bytes_be_fixed(16).unwrap();
        assert_eq!(&padded[..6], &[0, 0, 0, 0, 0, 0]);
        assert_eq!(&padded[6..], bytes);
        // Too-narrow width fails.
        assert!(n.to_bytes_be_fixed(9).is_none());
    }

    #[test]
    fn rem_and_cmp() {
        let n = BigUint::from_u64(1000);
        let m = BigUint::from_u64(7);
        assert_eq!(n.rem(&m), u(6)); // 1000 mod 7 = 6
        assert_eq!(u(5).cmp(&u(9)), core::cmp::Ordering::Less);
        assert_eq!(u(9).cmp(&u(9)), core::cmp::Ordering::Equal);
    }

    #[test]
    fn mod_exp_small_known_answers() {
        // 3^4 mod 7 = 81 mod 7 = 4
        assert_eq!(u(3).mod_exp(&u(4), &u(7)), u(4));
        // 2^10 mod 1001 = 1024 mod 1001 = 23 (odd modulus — Montgomery
        // is only defined for odd moduli, which every RSA n satisfies)
        assert_eq!(u(2).mod_exp(&u(10), &BigUint::from_u64(1001)), u(23));
        // 7^0 mod 13 = 1
        assert_eq!(u(7).mod_exp(&BigUint::zero(), &BigUint::from_u64(13)), u(1));
        // RSA-shaped small case: n = 3233 (61*53), e = 17, m = 65.
        // c = 65^17 mod 3233 = 2790; then 2790^413 mod 3233 = 65 (d = 413).
        let n = BigUint::from_u64(3233);
        let c = u(65).mod_exp(&u(17), &n);
        assert_eq!(c, BigUint::from_u64(2790));
        assert_eq!(c.mod_exp(&BigUint::from_u64(413), &n), u(65));
    }

    #[test]
    fn mod_exp_multi_limb() {
        // Exercise the multi-limb Montgomery path with a >64-bit odd modulus.
        // n = 2^127 - 1 (a 127-bit Mersenne prime), base = 2^70, e = 5.
        let n = BigUint::from_bytes_be(&{
            let mut b = [0xFFu8; 16];
            b[0] = 0x7F;
            b
        });
        assert!(n.is_odd());
        let base = BigUint::from_u64(1).mul(&{
            // 2^70
            let mut two70 = vec![0u64; 1];
            two70[0] = 1u64 << 6;
            let mut bu = BigUint { limbs: two70 };
            bu = shift_left_limbs(&bu, 1); // * 2^64  → 2^70
            bu.normalize();
            bu
        });
        let e = u(5);
        // Cross-check against repeated multiply mod n.
        let mut expected = u(1);
        for _ in 0..5 {
            expected = expected.mul(&base).rem(&n);
        }
        assert_eq!(base.mod_exp(&e, &n), expected);
    }

    #[test]
    fn der_integer_parse_and_refuse() {
        // 0x02 0x01 0x05 → 5
        let (v, used) = BigUint::parse_der_integer(&[0x02, 0x01, 0x05]).unwrap();
        assert_eq!(v, u(5));
        assert_eq!(used, 3);
        // Leading 0x00 pad required for high-bit value 0x80.
        let (v, _) = BigUint::parse_der_integer(&[0x02, 0x02, 0x00, 0x80]).unwrap();
        assert_eq!(v, BigUint::from_u64(0x80));
        // Refusals:
        assert_eq!(
            BigUint::parse_der_integer(&[0x02, 0x01, 0x80]).unwrap_err(),
            BigIntError::MalformedDer // negative
        );
        assert_eq!(
            BigUint::parse_der_integer(&[0x02, 0x02, 0x00, 0x05]).unwrap_err(),
            BigIntError::MalformedDer // non-minimal leading zero
        );
        assert_eq!(
            BigUint::parse_der_integer(&[0x02, 0x05, 0x01]).unwrap_err(),
            BigIntError::MalformedDer // truncated
        );
        assert_eq!(
            BigUint::parse_der_integer(&[0x03, 0x01, 0x05]).unwrap_err(),
            BigIntError::MalformedDer // wrong tag
        );
    }
}
