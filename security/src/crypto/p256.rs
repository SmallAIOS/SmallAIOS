// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! NIST P-256 (secp256r1) field and curve arithmetic.
//!
//! Support module for [`super::ecdsa_p256`] (verification only — no
//! signing, no key generation). Implements:
//!
//! - Modular arithmetic for both P-256 moduli — the field prime `p` and
//!   the group order `n` — as 4×u64 little-endian limbs in Montgomery
//!   form. One shared CIOS multiplication routine serves both moduli
//!   (see `openspec/changes/security-ecdsa-p256-v1/design.md`, D2).
//! - Jacobian-coordinate point doubling and addition with constant-time
//!   handling of the incomplete cases (D3).
//! - Constant-time double-and-add-always scalar multiplication: the
//!   sequence of field operations is independent of scalar bit values —
//!   no scalar-bit-dependent branches or memory indexing (D4).
//!
//! Inversions use Fermat exponentiation with a fixed public exponent
//! (`m - 2`), so their operation schedule is data-independent.
//!
//! Curve parameters are from SEC 2 v2.0 §2.4.2; the Montgomery constants
//! (`R² mod m`, `-m⁻¹ mod 2^64`) are derived from the modulus by `const`
//! evaluation rather than transcribed, so a wrong constant cannot pass
//! the arithmetic identity tests below.

#![allow(clippy::needless_range_loop)]

// ── Limb primitives ─────────────────────────────────────────────────────────

/// `a + b + carry` → (sum, carry-out). Carry-in/out are 0 or 1.
#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = (a as u128) + (b as u128) + (carry as u128);
    (t as u64, (t >> 64) as u64)
}

/// `a - b - borrow` → (diff, borrow-out). Borrow-in/out are 0 or 1.
#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let (t1, o1) = a.overflowing_sub(b);
    let (t2, o2) = t1.overflowing_sub(borrow);
    (t2, (o1 as u64) | (o2 as u64))
}

/// `acc + a * b + carry` → (low, high).
#[inline(always)]
fn mac(acc: u64, a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = (acc as u128) + (a as u128) * (b as u128) + (carry as u128);
    (t as u64, (t >> 64) as u64)
}

fn add4(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], u64) {
    let mut r = [0u64; 4];
    let mut carry = 0;
    for i in 0..4 {
        let (s, c) = adc(a[i], b[i], carry);
        r[i] = s;
        carry = c;
    }
    (r, carry)
}

fn sub4(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], u64) {
    let mut r = [0u64; 4];
    let mut borrow = 0;
    for i in 0..4 {
        let (d, bo) = sbb(a[i], b[i], borrow);
        r[i] = d;
        borrow = bo;
    }
    (r, borrow)
}

// ── Constant-time helpers ───────────────────────────────────────────────────

/// Expand a 0/1 bit into an all-zeros / all-ones mask.
#[inline(always)]
fn ct_mask(bit: u64) -> u64 {
    0u64.wrapping_sub(bit)
}

/// Select `a` if `choice == 1`, else `b`. `choice` must be 0 or 1.
#[inline(always)]
fn ct_select4(a: &[u64; 4], b: &[u64; 4], choice: u64) -> [u64; 4] {
    let mask = ct_mask(choice);
    let mut r = [0u64; 4];
    for i in 0..4 {
        r[i] = (a[i] & mask) | (b[i] & !mask);
    }
    r
}

/// 1 if all limbs are zero, else 0. Constant-time.
#[inline(always)]
fn ct_is_zero4(a: &[u64; 4]) -> u64 {
    let x = a[0] | a[1] | a[2] | a[3];
    // For x == 0 both x and -x have a clear top bit; otherwise one is set.
    1 ^ ((x | x.wrapping_neg()) >> 63)
}

/// 1 if `a == b`, else 0. Constant-time.
#[inline(always)]
fn ct_eq4(a: &[u64; 4], b: &[u64; 4]) -> u64 {
    let x = [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]];
    ct_is_zero4(&x)
}

/// 1 if `a < b` as 256-bit integers, else 0. Constant-time.
#[inline(always)]
fn ct_lt4(a: &[u64; 4], b: &[u64; 4]) -> u64 {
    let (_, borrow) = sub4(a, b);
    borrow
}

// ── Montgomery parameter derivation (compile-time) ──────────────────────────

/// `-m₀⁻¹ mod 2^64` for odd `m₀`, via Newton's iteration.
const fn neg_inv_u64(m0: u64) -> u64 {
    // m0 * m0 ≡ 1 (mod 8) for odd m0; each step doubles the precision.
    let mut inv = m0;
    let mut i = 0;
    while i < 5 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg()
}

/// `2 * x mod m` for `x < m`, const-evaluable.
const fn shl1_mod(x: [u64; 4], m: [u64; 4]) -> [u64; 4] {
    // Shift left by one.
    let mut s = [0u64; 4];
    let mut carry = 0u64;
    let mut i = 0;
    while i < 4 {
        s[i] = (x[i] << 1) | carry;
        carry = x[i] >> 63;
        i += 1;
    }
    // Subtract m; keep the difference if `2x` overflowed 2^256 or `2x >= m`.
    let mut d = [0u64; 4];
    let mut borrow = 0u64;
    i = 0;
    while i < 4 {
        let (t1, o1) = s[i].overflowing_sub(m[i]);
        let (t2, o2) = t1.overflowing_sub(borrow);
        d[i] = t2;
        borrow = (o1 as u64) | (o2 as u64);
        i += 1;
    }
    if carry == 1 || borrow == 0 {
        d
    } else {
        s
    }
}

/// `2^k mod m` for `m > 2^255` (holds for both P-256 moduli).
const fn pow2_mod(k: u32, m: [u64; 4]) -> [u64; 4] {
    let mut x = [1u64, 0, 0, 0];
    let mut i = 0;
    while i < k {
        x = shl1_mod(x, m);
        i += 1;
    }
    x
}

// ── Montgomery field, generated for both moduli ─────────────────────────────

/// CIOS Montgomery multiplication: `a * b * R⁻¹ mod m`, inputs < m.
fn mont_mul(a: &[u64; 4], b: &[u64; 4], m: &[u64; 4], m0inv: u64) -> [u64; 4] {
    let mut t = [0u64; 6];
    for i in 0..4 {
        // t += a[i] * b
        let mut carry = 0;
        for j in 0..4 {
            let (lo, hi) = mac(t[j], a[i], b[j], carry);
            t[j] = lo;
            carry = hi;
        }
        let (s, c) = adc(t[4], carry, 0);
        t[4] = s;
        t[5] = c;
        // Reduce: add k*m so the low limb cancels, then shift down one limb.
        let k = t[0].wrapping_mul(m0inv);
        let (_, hi) = mac(t[0], k, m[0], 0);
        let mut carry = hi;
        for j in 1..4 {
            let (lo, hi) = mac(t[j], k, m[j], carry);
            t[j - 1] = lo;
            carry = hi;
        }
        let (s, c) = adc(t[4], carry, 0);
        t[3] = s;
        t[4] = t[5] + c;
        t[5] = 0;
    }
    // Result is < 2m; a single conditional subtraction normalizes it.
    let r = [t[0], t[1], t[2], t[3]];
    let (d, borrow) = sub4(&r, m);
    ct_select4(&d, &r, t[4] | (borrow ^ 1))
}

macro_rules! mont_field {
    ($name:ident, $modulus:expr, $doc:expr) => {
        #[doc = $doc]
        ///
        /// Values are held in Montgomery form (`x·R mod m`, `R = 2^256`).
        #[derive(Clone, Copy, Debug)]
        pub struct $name([u64; 4]);

        impl $name {
            /// The modulus, as little-endian u64 limbs.
            pub const MODULUS: [u64; 4] = $modulus;
            const M0INV: u64 = neg_inv_u64(Self::MODULUS[0]);
            /// `R mod m` — the Montgomery form of 1.
            const R: [u64; 4] = pow2_mod(256, Self::MODULUS);
            /// `R² mod m` — converts into Montgomery form via one `mont_mul`.
            const R2: [u64; 4] = pow2_mod(512, Self::MODULUS);

            pub const ZERO: Self = Self([0, 0, 0, 0]);
            pub const ONE: Self = Self(Self::R);

            /// Convert a canonical integer (< m) into Montgomery form.
            pub fn from_raw(x: [u64; 4]) -> Self {
                Self::to_mont(&x)
            }

            fn to_mont(x: &[u64; 4]) -> Self {
                Self(mont_mul(x, &Self::R2, &Self::MODULUS, Self::M0INV))
            }

            fn demont(&self) -> [u64; 4] {
                mont_mul(&self.0, &[1, 0, 0, 0], &Self::MODULUS, Self::M0INV)
            }

            /// Parse a canonical big-endian 32-byte integer. Returns `None`
            /// when the value is ≥ m (non-canonical).
            pub fn from_bytes_be(bytes: &[u8; 32]) -> Option<Self> {
                let limbs = be_bytes_to_limbs(bytes);
                if ct_lt4(&limbs, &Self::MODULUS) == 0 {
                    return None;
                }
                Some(Self::to_mont(&limbs))
            }

            /// Interpret 32 big-endian bytes as an integer reduced mod m.
            /// Valid because both P-256 moduli exceed 2^255, so a single
            /// conditional subtraction fully reduces any 256-bit value.
            pub fn from_bytes_be_reduced(bytes: &[u8; 32]) -> Self {
                let limbs = be_bytes_to_limbs(bytes);
                let (d, borrow) = sub4(&limbs, &Self::MODULUS);
                let reduced = ct_select4(&d, &limbs, borrow ^ 1);
                Self::to_mont(&reduced)
            }

            /// Canonical big-endian byte encoding.
            pub fn to_bytes_be(&self) -> [u8; 32] {
                limbs_to_be_bytes(&self.demont())
            }

            /// The canonical integer value, as little-endian limbs.
            pub fn to_raw_limbs(&self) -> [u64; 4] {
                self.demont()
            }

            pub fn add(&self, rhs: &Self) -> Self {
                let (s, carry) = add4(&self.0, &rhs.0);
                let (d, borrow) = sub4(&s, &Self::MODULUS);
                Self(ct_select4(&d, &s, carry | (borrow ^ 1)))
            }

            pub fn sub(&self, rhs: &Self) -> Self {
                let (d, borrow) = sub4(&self.0, &rhs.0);
                let (fixed, _) = add4(&d, &Self::MODULUS);
                Self(ct_select4(&fixed, &d, borrow))
            }

            pub fn neg(&self) -> Self {
                Self::ZERO.sub(self)
            }

            pub fn double(&self) -> Self {
                self.add(self)
            }

            pub fn mul(&self, rhs: &Self) -> Self {
                Self(mont_mul(&self.0, &rhs.0, &Self::MODULUS, Self::M0INV))
            }

            pub fn square(&self) -> Self {
                self.mul(self)
            }

            /// `self^exp` for a PUBLIC exponent. The branch on exponent bits
            /// is data-independent because the exponent is a fixed constant,
            /// never secret material.
            pub fn pow_public(&self, exp: &[u64; 4]) -> Self {
                let mut acc = Self::ONE;
                for limb in exp.iter().rev() {
                    for bit in (0..64).rev() {
                        acc = acc.square();
                        if (limb >> bit) & 1 == 1 {
                            acc = acc.mul(self);
                        }
                    }
                }
                acc
            }

            /// Multiplicative inverse via Fermat: `self^(m-2)`. Returns zero
            /// for zero input (callers reject zero scalars beforehand).
            pub fn invert(&self) -> Self {
                let (m_minus_2, _) = sub4(&Self::MODULUS, &[2, 0, 0, 0]);
                self.pow_public(&m_minus_2)
            }

            /// 1 if zero, else 0. Constant-time (zero is a fixed point of
            /// the Montgomery map).
            pub fn is_zero(&self) -> u64 {
                ct_is_zero4(&self.0)
            }

            /// 1 if equal, else 0. Constant-time; Montgomery representation
            /// of values < m is unique, so limb equality is value equality.
            pub fn ct_eq(&self, rhs: &Self) -> u64 {
                ct_eq4(&self.0, &rhs.0)
            }

            /// Select `a` if `choice == 1`, else `b`.
            pub fn conditional_select(a: &Self, b: &Self, choice: u64) -> Self {
                Self(ct_select4(&a.0, &b.0, choice))
            }
        }
    };
}

// The field prime p = 2^256 - 2^224 + 2^192 + 2^96 - 1.
mont_field!(
    FieldElement,
    [
        0xFFFF_FFFF_FFFF_FFFF,
        0x0000_0000_FFFF_FFFF,
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001,
    ],
    "An element of GF(p), the P-256 base field."
);

// The group order n.
mont_field!(
    Scalar,
    [
        0xF3B9_CAC2_FC63_2551,
        0xBCE6_FAAD_A717_9E84,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
    ],
    "An element of GF(n), integers modulo the P-256 group order."
);

fn be_bytes_to_limbs(bytes: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for i in 0..4 {
        let chunk: [u8; 8] = bytes[(3 - i) * 8..(4 - i) * 8].try_into().unwrap();
        limbs[i] = u64::from_be_bytes(chunk);
    }
    limbs
}

fn limbs_to_be_bytes(limbs: &[u64; 4]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        bytes[(3 - i) * 8..(4 - i) * 8].copy_from_slice(&limbs[i].to_be_bytes());
    }
    bytes
}

// ── Curve parameters (SEC 2 v2.0 §2.4.2), canonical form ────────────────────

const B_RAW: [u64; 4] = [
    0x3BCE_3C3E_27D2_604B,
    0x651D_06B0_CC53_B0F6,
    0xB3EB_BD55_7698_86BC,
    0x5AC6_35D8_AA3A_93E7,
];
const GX_RAW: [u64; 4] = [
    0xF4A1_3945_D898_C296,
    0x7703_7D81_2DEB_33A0,
    0xF8BC_E6E5_63A4_40F2,
    0x6B17_D1F2_E12C_4247,
];
const GY_RAW: [u64; 4] = [
    0xCBB6_4068_37BF_51F5,
    0x2BCE_3357_6B31_5ECE,
    0x8EE7_EB4A_7C0F_9E16,
    0x4FE3_42E2_FE1A_7F9B,
];

fn curve_b() -> FieldElement {
    FieldElement::from_raw(B_RAW)
}

// ── Points ──────────────────────────────────────────────────────────────────

/// A point on P-256 in affine coordinates. Guaranteed on-curve by
/// construction (`new_checked`).
#[derive(Clone, Copy, Debug)]
pub struct AffinePoint {
    pub x: FieldElement,
    pub y: FieldElement,
}

impl AffinePoint {
    /// Construct from canonical coordinates, validating the curve
    /// equation `y² = x³ - 3x + b (mod p)`. Returns `None` for
    /// non-canonical coordinates or off-curve points.
    pub fn new_checked(x_bytes: &[u8; 32], y_bytes: &[u8; 32]) -> Option<Self> {
        let x = FieldElement::from_bytes_be(x_bytes)?;
        let y = FieldElement::from_bytes_be(y_bytes)?;
        let p = AffinePoint { x, y };
        if p.is_on_curve() {
            Some(p)
        } else {
            None
        }
    }

    fn is_on_curve(&self) -> bool {
        let three = FieldElement::ONE.double().add(&FieldElement::ONE);
        // x³ - 3x + b = x(x² - 3) + b
        let rhs = self.x.mul(&self.x.square().sub(&three)).add(&curve_b());
        self.y.square().ct_eq(&rhs) == 1
    }

    /// The SEC 2 base point G.
    pub fn generator() -> Self {
        AffinePoint {
            x: FieldElement::from_raw(GX_RAW),
            y: FieldElement::from_raw(GY_RAW),
        }
    }

    pub fn neg(&self) -> Self {
        AffinePoint {
            x: self.x,
            y: self.y.neg(),
        }
    }
}

/// A point in Jacobian coordinates `(X, Y, Z)` representing the affine
/// point `(X/Z², Y/Z³)`; the identity is any point with `Z = 0`.
#[derive(Clone, Copy, Debug)]
pub struct ProjectivePoint {
    x: FieldElement,
    y: FieldElement,
    z: FieldElement,
}

impl ProjectivePoint {
    pub fn identity() -> Self {
        ProjectivePoint {
            x: FieldElement::ONE,
            y: FieldElement::ONE,
            z: FieldElement::ZERO,
        }
    }

    pub fn from_affine(p: &AffinePoint) -> Self {
        ProjectivePoint {
            x: p.x,
            y: p.y,
            z: FieldElement::ONE,
        }
    }

    /// 1 if this is the identity, else 0. Constant-time.
    pub fn is_identity(&self) -> u64 {
        self.z.is_zero()
    }

    fn conditional_select(a: &Self, b: &Self, choice: u64) -> Self {
        ProjectivePoint {
            x: FieldElement::conditional_select(&a.x, &b.x, choice),
            y: FieldElement::conditional_select(&a.y, &b.y, choice),
            z: FieldElement::conditional_select(&a.z, &b.z, choice),
        }
    }

    /// Point doubling (`dbl-2001-b`, specialized for a = -3). Maps the
    /// identity to the identity (Z stays zero).
    pub fn double(&self) -> Self {
        let delta = self.z.square();
        let gamma = self.y.square();
        let beta = self.x.mul(&gamma);
        // alpha = 3 * (X - delta) * (X + delta)
        let t = self.x.sub(&delta).mul(&self.x.add(&delta));
        let alpha = t.double().add(&t);
        // X3 = alpha² - 8*beta
        let beta4 = beta.double().double();
        let x3 = alpha.square().sub(&beta4.double());
        // Z3 = (Y + Z)² - gamma - delta
        let z3 = self.y.add(&self.z).square().sub(&gamma).sub(&delta);
        // Y3 = alpha * (4*beta - X3) - 8*gamma²
        let y3 = alpha
            .mul(&beta4.sub(&x3))
            .sub(&gamma.square().double().double().double());
        ProjectivePoint {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// Point addition (`add-2007-bl`) with constant-time selection of the
    /// incomplete cases: doubling (P == Q), inverse inputs (result is the
    /// identity), and identity operands.
    pub fn add(&self, other: &Self) -> Self {
        let z1z1 = self.z.square();
        let z2z2 = other.z.square();
        let u1 = self.x.mul(&z2z2);
        let u2 = other.x.mul(&z1z1);
        let s1 = self.y.mul(&other.z).mul(&z2z2);
        let s2 = other.y.mul(&self.z).mul(&z1z1);
        let h = u2.sub(&u1);
        let rr = s2.sub(&s1);

        let h_zero = h.is_zero();
        let r_zero = rr.is_zero();

        // General case (garbage when a special case applies; overridden below).
        let i = h.double().square();
        let j = h.mul(&i);
        let r2 = rr.double();
        let v = u1.mul(&i);
        let x3 = r2.square().sub(&j).sub(&v.double());
        let y3 = r2.mul(&v.sub(&x3)).sub(&s1.mul(&j).double());
        let z3 = self.z.add(&other.z).square().sub(&z1z1).sub(&z2z2).mul(&h);
        let general = ProjectivePoint {
            x: x3,
            y: y3,
            z: z3,
        };

        // Special-case ladder of constant-time selects; later lines override
        // earlier ones, so the identity-operand cases win over the H/r flags
        // (whose values are garbage when an operand is the identity).
        let mut result = general;
        let doubled = self.double();
        result = Self::conditional_select(&doubled, &result, h_zero & r_zero);
        result = Self::conditional_select(&Self::identity(), &result, h_zero & (r_zero ^ 1));
        result = Self::conditional_select(self, &result, other.is_identity());
        result = Self::conditional_select(other, &result, self.is_identity());
        result
    }

    /// Constant-time scalar multiplication (double-and-add-always).
    /// `scalar` is a canonical little-endian integer; every iteration
    /// performs the same double + add + select sequence regardless of the
    /// bit value, and bits are extracted arithmetically (no
    /// scalar-dependent indexing).
    pub fn mul(&self, scalar: &[u64; 4]) -> Self {
        let mut acc = Self::identity();
        for i in (0..256).rev() {
            acc = acc.double();
            let sum = acc.add(self);
            let bit = (scalar[i / 64] >> (i % 64)) & 1;
            acc = Self::conditional_select(&sum, &acc, bit);
        }
        acc
    }

    /// Convert to affine coordinates. Returns `None` for the identity.
    /// (Verification inputs are public; this branch is not secret-dependent.)
    pub fn to_affine(&self) -> Option<AffinePoint> {
        if self.is_identity() == 1 {
            return None;
        }
        let zinv = self.z.invert();
        let zinv2 = zinv.square();
        Some(AffinePoint {
            x: self.x.mul(&zinv2),
            y: self.y.mul(&zinv2).mul(&zinv),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(k: u64) -> FieldElement {
        FieldElement::from_raw([k, 0, 0, 0])
    }

    /// Re-evaluate the compile-time Montgomery parameter derivations at
    /// runtime and pin them against their defining properties (const
    /// evaluation is invisible to coverage instrumentation).
    #[test]
    fn montgomery_constant_derivation() {
        for m in [FieldElement::MODULUS, Scalar::MODULUS] {
            // -m0⁻¹: m0 * -(neg_inv) ≡ 1 (mod 2^64).
            let m0inv = neg_inv_u64(m[0]);
            assert_eq!(m[0].wrapping_mul(m0inv.wrapping_neg()), 1);
            // R = 2^256 mod m and R² = 2^512 mod m are reduced,
            // and R² · 1 · R⁻¹ = R under Montgomery multiplication.
            let r = pow2_mod(256, m);
            let r2 = pow2_mod(512, m);
            assert_eq!(ct_lt4(&r, &m), 1);
            assert_eq!(ct_lt4(&r2, &m), 1);
            assert_eq!(mont_mul(&r2, &[1, 0, 0, 0], &m, m0inv), r);
        }
        // shl1_mod branch coverage: 2·(m-1) mod m = m-2 (reducing) and
        // 2·1 = 2 (non-reducing).
        let mut m1 = FieldElement::MODULUS;
        m1[0] -= 1;
        let mut expect = FieldElement::MODULUS;
        expect[0] -= 2;
        assert_eq!(shl1_mod(m1, FieldElement::MODULUS), expect);
        assert_eq!(shl1_mod([1, 0, 0, 0], FieldElement::MODULUS), [2, 0, 0, 0]);
    }

    #[test]
    fn field_byte_roundtrip() {
        for v in [
            [1u64, 0, 0, 0],
            [0xDEAD_BEEF, 0x1234_5678, 0x9ABC_DEF0, 0x0FED_CBA9],
            {
                let mut m = FieldElement::MODULUS;
                m[0] -= 1;
                m
            },
        ] {
            let x: FieldElement = FieldElement::from_raw(v);
            let back = FieldElement::from_bytes_be(&x.to_bytes_be()).unwrap();
            assert_eq!(x.ct_eq(&back), 1);
            assert_eq!(x.to_raw_limbs(), v);
        }
    }

    #[test]
    fn field_rejects_non_canonical() {
        assert!(FieldElement::from_bytes_be(&limbs_to_be_bytes(&FieldElement::MODULUS)).is_none());
        assert!(Scalar::from_bytes_be(&limbs_to_be_bytes(&Scalar::MODULUS)).is_none());
        assert!(FieldElement::from_bytes_be(&[0xFF; 32]).is_none());
    }

    #[test]
    fn field_known_answers() {
        // 2 * 3 = 6, 5 - 7 = -2, (p-1) + 2 = 1
        assert_eq!(fe(2).mul(&fe(3)).ct_eq(&fe(6)), 1);
        assert_eq!(fe(5).sub(&fe(7)).ct_eq(&fe(2).neg()), 1);
        let mut pm1 = FieldElement::MODULUS;
        pm1[0] -= 1;
        let pm1: FieldElement = FieldElement::from_raw(pm1);
        assert_eq!(pm1.add(&fe(2)).ct_eq(&fe(1)), 1);
    }

    #[test]
    fn field_inversion() {
        for x in [fe(2), fe(0xABCD_EF12), AffinePoint::generator().x] {
            assert_eq!(x.mul(&x.invert()).ct_eq(&FieldElement::ONE), 1);
        }
        let s: Scalar = Scalar::from_raw([12345, 0, 0, 0]);
        assert_eq!(s.mul(&s.invert()).ct_eq(&Scalar::ONE), 1);
        // Zero inverts to zero under Fermat.
        assert_eq!(FieldElement::ZERO.invert().is_zero(), 1);
    }

    #[test]
    fn generator_is_on_curve() {
        assert!(AffinePoint::generator().is_on_curve());
    }

    #[test]
    fn group_law_identities() {
        let g = ProjectivePoint::from_affine(&AffinePoint::generator());

        // n * G = identity
        assert_eq!(g.mul(&Scalar::MODULUS).is_identity(), 1);

        // (n - 1) * G = -G
        let mut nm1 = Scalar::MODULUS;
        nm1[0] -= 1;
        let neg_g = g.mul(&nm1).to_affine().unwrap();
        let expected = AffinePoint::generator().neg();
        assert_eq!(neg_g.x.ct_eq(&expected.x), 1);
        assert_eq!(neg_g.y.ct_eq(&expected.y), 1);

        // 2G = G + G (doubling vs general-add special case)
        let dbl = g.double().to_affine().unwrap();
        let add = g.add(&g).to_affine().unwrap();
        assert_eq!(dbl.x.ct_eq(&add.x), 1);
        assert_eq!(dbl.y.ct_eq(&add.y), 1);

        // G + identity = G, identity + G = G
        for sum in [
            g.add(&ProjectivePoint::identity()),
            ProjectivePoint::identity().add(&g),
        ] {
            let a = sum.to_affine().unwrap();
            assert_eq!(a.x.ct_eq(&AffinePoint::generator().x), 1);
            assert_eq!(a.y.ct_eq(&AffinePoint::generator().y), 1);
        }

        // G + (-G) = identity
        let neg = ProjectivePoint::from_affine(&AffinePoint::generator().neg());
        assert_eq!(g.add(&neg).is_identity(), 1);
    }

    #[test]
    fn ladder_matches_repeated_addition() {
        let g = ProjectivePoint::from_affine(&AffinePoint::generator());
        let mut acc = ProjectivePoint::identity();
        for k in 1u64..=8 {
            acc = acc.add(&g);
            let ladder = g.mul(&[k, 0, 0, 0]).to_affine().unwrap();
            let direct = acc.to_affine().unwrap();
            assert_eq!(ladder.x.ct_eq(&direct.x), 1, "k = {k}");
            assert_eq!(ladder.y.ct_eq(&direct.y), 1, "k = {k}");
        }
        // 0 * G = identity
        assert_eq!(g.mul(&[0, 0, 0, 0]).is_identity(), 1);
    }
}
