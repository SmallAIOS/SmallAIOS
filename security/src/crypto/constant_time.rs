// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Constant-time comparison and selection utilities.
//!
//! These functions are designed to execute in time independent of the values
//! being compared, preventing timing side-channel attacks. They are critical
//! for cryptographic operations such as:
//!
//! - MAC tag verification (AES-GCM)
//! - Signature verification
//! - Password/key comparison
//! - Conditional selection without branching
//!
//! # Implementation Notes
//!
//! All operations use bitwise arithmetic only — no branches, no
//! early returns, and no table lookups that depend on secret data.
//! The `#[inline(never)]` attribute prevents the compiler from
//! optimizing away the constant-time properties.
//!
//! In production, these should be verified against compiler output
//! to ensure no branch instructions are emitted.

#![allow(unused)]

use core::fmt;

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from constant-time operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstantTimeError {
    /// Slices have different lengths.
    LengthMismatch,
}

impl fmt::Display for ConstantTimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch => write!(f, "slice length mismatch"),
        }
    }
}

// ─── Constant-Time Byte ─────────────────────────────────────────────────────

/// A byte-sized value suitable for constant-time conditional logic.
///
/// `CtBool` is either `0x00` (false) or `0xFF` (true), which allows
/// bitwise AND/OR to serve as constant-time selection.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CtBool(u8);

impl CtBool {
    /// Constant-time TRUE (all bits set).
    pub const TRUE: Self = Self(0xFF);

    /// Constant-time FALSE (no bits set).
    pub const FALSE: Self = Self(0x00);

    /// Create from a boolean (non-constant-time conversion, for setup only).
    #[inline]
    pub fn from_bool(b: bool) -> Self {
        // Convert bool to 0x00 or 0xFF without branching
        Self(0u8.wrapping_sub(b as u8))
    }

    /// Convert to a bool (non-constant-time, for final results only).
    #[inline]
    pub fn to_bool(self) -> bool {
        self.0 != 0
    }

    /// Bitwise AND of two CtBool values.
    #[inline]
    pub fn ct_and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Bitwise OR of two CtBool values.
    #[inline]
    pub fn ct_or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Bitwise NOT.
    #[inline]
    pub fn ct_not(self) -> Self {
        Self(!self.0)
    }

    /// Return the raw mask byte (0x00 or 0xFF).
    #[inline]
    pub fn mask(self) -> u8 {
        self.0
    }
}

impl fmt::Debug for CtBool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0xFF {
            write!(f, "CtBool(TRUE)")
        } else {
            write!(f, "CtBool(FALSE)")
        }
    }
}

// ─── Core Operations ─────────────────────────────────────────────────────────

/// Constant-time equality comparison of two bytes.
///
/// Returns `CtBool::TRUE` if `a == b`, `CtBool::FALSE` otherwise.
#[inline(never)]
pub fn ct_eq_byte(a: u8, b: u8) -> CtBool {
    // XOR gives 0 if equal, non-zero otherwise.
    // Collapse all bits: if any bit differs, result is non-zero.
    let diff = a ^ b;
    // Map 0 -> 0xFF (equal), non-zero -> 0x00 (not equal)
    // Uses the carry/borrow from subtraction:
    //   (0u16 - 0) >> 8 = 0, inverted = 0xFF
    //   (0u16 - N) >> 8 = 0xFF for N > 0, inverted = 0x00
    let diff16 = diff as u16;
    // Widen to avoid overflow, compute borrow
    let result = ((diff16.wrapping_sub(1)) >> 8) as u8;
    // If diff == 0: (0xFFFF) >> 8 = 0xFF (but we need to mask to 1 byte width)
    // If diff != 0: ((diff-1) >> 8) varies
    // Better approach: use OR-fold
    let is_zero = ct_is_zero(diff);
    CtBool(is_zero)
}

/// Internal: map 0 -> 0xFF, non-zero -> 0x00, constant-time.
#[inline(always)]
fn ct_is_zero(x: u8) -> u8 {
    // Trick: for x == 0, (x | -x) has MSB = 0
    //        for x != 0, (x | -x) has MSB = 1
    let neg_x = (x as i8).wrapping_neg() as u8;
    let v = x | neg_x;
    // Arithmetic right-shift the MSB to fill all bits
    // MSB=0 -> 0x00, MSB=1 -> 0xFF
    let mask = ((v >> 7) as i8).wrapping_neg() as u8;
    // Invert: we want 0 -> 0xFF, non-zero -> 0x00
    !mask
}

/// Constant-time equality comparison of two byte slices.
///
/// Returns `CtBool::TRUE` if the slices are equal in both length and content,
/// `CtBool::FALSE` otherwise. Always examines all bytes of both slices
/// (no early return).
///
/// # Panics
///
/// Does not panic. Returns `CtBool::FALSE` for different-length slices.
#[inline(never)]
pub fn ct_eq(a: &[u8], b: &[u8]) -> CtBool {
    if a.len() != b.len() {
        // Length mismatch is not secret — this branch is OK.
        // The attacker already knows the lengths of public parameters.
        return CtBool::FALSE;
    }

    // Accumulate XOR differences across all bytes
    let mut acc: u8 = 0;
    for i in 0..a.len() {
        acc |= a[i] ^ b[i];
    }

    // acc == 0 iff all bytes match
    CtBool(ct_is_zero(acc))
}

/// Constant-time equality comparison returning Result.
///
/// Convenience wrapper that returns `Ok(())` on match, `Err(LengthMismatch)`
/// on length mismatch, or `Err(LengthMismatch)` on content mismatch.
/// Note: in production, content mismatch should use a different error variant;
/// we reuse LengthMismatch here for the stub.
#[inline(never)]
pub fn ct_eq_check(a: &[u8], b: &[u8]) -> Result<(), ConstantTimeError> {
    if a.len() != b.len() {
        return Err(ConstantTimeError::LengthMismatch);
    }
    if ct_eq(a, b).to_bool() {
        Ok(())
    } else {
        // Content mismatch — but we don't reveal which byte differs
        Err(ConstantTimeError::LengthMismatch)
    }
}

/// Constant-time conditional selection: returns `a` if `condition` is TRUE, `b` otherwise.
///
/// Both `a` and `b` are always read; the selection is done via bitwise operations.
#[inline(never)]
pub fn ct_select_byte(condition: CtBool, a: u8, b: u8) -> u8 {
    // condition.mask() is 0xFF (true) or 0x00 (false)
    // result = (mask & a) | (!mask & b)
    let mask = condition.mask();
    (mask & a) | (!mask & b)
}

/// Constant-time conditional selection of byte slices.
///
/// Writes `a[i]` to `out[i]` if condition is TRUE, `b[i]` otherwise.
/// All three slices must have the same length.
///
/// # Returns
///
/// `Ok(())` on success, `Err(LengthMismatch)` if lengths differ.
#[inline(never)]
pub fn ct_select(condition: CtBool, a: &[u8], b: &[u8], out: &mut [u8]) -> Result<(), ConstantTimeError> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(ConstantTimeError::LengthMismatch);
    }
    let mask = condition.mask();
    for i in 0..a.len() {
        out[i] = (mask & a[i]) | (!mask & b[i]);
    }
    Ok(())
}

/// Constant-time conditional swap of two byte slices.
///
/// If `condition` is TRUE, swaps `a[i]` and `b[i]` for all i.
/// If FALSE, both slices are unchanged.
///
/// Both slices must have the same length.
#[inline(never)]
pub fn ct_swap(condition: CtBool, a: &mut [u8], b: &mut [u8]) -> Result<(), ConstantTimeError> {
    if a.len() != b.len() {
        return Err(ConstantTimeError::LengthMismatch);
    }
    let mask = condition.mask();
    for i in 0..a.len() {
        let t = mask & (a[i] ^ b[i]);
        a[i] ^= t;
        b[i] ^= t;
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use alloc::format;
    use super::*;

    #[test]
    fn ct_bool_true_false() {
        assert_eq!(CtBool::TRUE.mask(), 0xFF);
        assert_eq!(CtBool::FALSE.mask(), 0x00);
        assert!(CtBool::TRUE.to_bool());
        assert!(!CtBool::FALSE.to_bool());
    }

    #[test]
    fn ct_bool_from_bool() {
        assert_eq!(CtBool::from_bool(true), CtBool::TRUE);
        assert_eq!(CtBool::from_bool(false), CtBool::FALSE);
    }

    #[test]
    fn ct_bool_and() {
        assert_eq!(CtBool::TRUE.ct_and(CtBool::TRUE), CtBool::TRUE);
        assert_eq!(CtBool::TRUE.ct_and(CtBool::FALSE), CtBool::FALSE);
        assert_eq!(CtBool::FALSE.ct_and(CtBool::TRUE), CtBool::FALSE);
        assert_eq!(CtBool::FALSE.ct_and(CtBool::FALSE), CtBool::FALSE);
    }

    #[test]
    fn ct_bool_or() {
        assert_eq!(CtBool::TRUE.ct_or(CtBool::TRUE), CtBool::TRUE);
        assert_eq!(CtBool::TRUE.ct_or(CtBool::FALSE), CtBool::TRUE);
        assert_eq!(CtBool::FALSE.ct_or(CtBool::TRUE), CtBool::TRUE);
        assert_eq!(CtBool::FALSE.ct_or(CtBool::FALSE), CtBool::FALSE);
    }

    #[test]
    fn ct_bool_not() {
        assert_eq!(CtBool::TRUE.ct_not(), CtBool::FALSE);
        assert_eq!(CtBool::FALSE.ct_not(), CtBool::TRUE);
    }

    #[test]
    fn ct_eq_byte_equal() {
        for b in 0..=255u8 {
            assert!(ct_eq_byte(b, b).to_bool(), "ct_eq_byte({}, {}) should be true", b, b);
        }
    }

    #[test]
    fn ct_eq_byte_not_equal() {
        assert!(!ct_eq_byte(0, 1).to_bool());
        assert!(!ct_eq_byte(0xFF, 0xFE).to_bool());
        assert!(!ct_eq_byte(0x80, 0x00).to_bool());
    }

    #[test]
    fn ct_eq_slices_equal() {
        let a = [1u8, 2, 3, 4, 5];
        let b = [1u8, 2, 3, 4, 5];
        assert!(ct_eq(&a, &b).to_bool());
    }

    #[test]
    fn ct_eq_slices_not_equal() {
        let a = [1u8, 2, 3, 4, 5];
        let b = [1u8, 2, 3, 4, 6]; // Last byte differs
        assert!(!ct_eq(&a, &b).to_bool());
    }

    #[test]
    fn ct_eq_slices_first_byte_differs() {
        let a = [0u8, 2, 3, 4, 5];
        let b = [1u8, 2, 3, 4, 5];
        assert!(!ct_eq(&a, &b).to_bool());
    }

    #[test]
    fn ct_eq_slices_different_length() {
        let a = [1u8, 2, 3];
        let b = [1u8, 2, 3, 4];
        assert!(!ct_eq(&a, &b).to_bool());
    }

    #[test]
    fn ct_eq_empty_slices() {
        let a: [u8; 0] = [];
        let b: [u8; 0] = [];
        assert!(ct_eq(&a, &b).to_bool());
    }

    #[test]
    fn ct_eq_check_ok() {
        let a = [0xAA; 32];
        let b = [0xAA; 32];
        assert!(ct_eq_check(&a, &b).is_ok());
    }

    #[test]
    fn ct_eq_check_mismatch() {
        let a = [0xAA; 32];
        let b = [0xBB; 32];
        assert!(ct_eq_check(&a, &b).is_err());
    }

    #[test]
    fn ct_select_byte_true() {
        assert_eq!(ct_select_byte(CtBool::TRUE, 0xAA, 0xBB), 0xAA);
    }

    #[test]
    fn ct_select_byte_false() {
        assert_eq!(ct_select_byte(CtBool::FALSE, 0xAA, 0xBB), 0xBB);
    }

    #[test]
    fn ct_select_slices_true() {
        let a = [1u8, 2, 3];
        let b = [4u8, 5, 6];
        let mut out = [0u8; 3];
        ct_select(CtBool::TRUE, &a, &b, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3]);
    }

    #[test]
    fn ct_select_slices_false() {
        let a = [1u8, 2, 3];
        let b = [4u8, 5, 6];
        let mut out = [0u8; 3];
        ct_select(CtBool::FALSE, &a, &b, &mut out).unwrap();
        assert_eq!(out, [4, 5, 6]);
    }

    #[test]
    fn ct_select_length_mismatch() {
        let a = [1u8, 2];
        let b = [4u8, 5, 6];
        let mut out = [0u8; 3];
        assert_eq!(ct_select(CtBool::TRUE, &a, &b, &mut out), Err(ConstantTimeError::LengthMismatch));
    }

    #[test]
    fn ct_swap_true() {
        let mut a = [1u8, 2, 3];
        let mut b = [4u8, 5, 6];
        ct_swap(CtBool::TRUE, &mut a, &mut b).unwrap();
        assert_eq!(a, [4, 5, 6]);
        assert_eq!(b, [1, 2, 3]);
    }

    #[test]
    fn ct_swap_false() {
        let mut a = [1u8, 2, 3];
        let mut b = [4u8, 5, 6];
        ct_swap(CtBool::FALSE, &mut a, &mut b).unwrap();
        assert_eq!(a, [1, 2, 3]);
        assert_eq!(b, [4, 5, 6]);
    }

    #[test]
    fn ct_swap_length_mismatch() {
        let mut a = [1u8, 2];
        let mut b = [4u8, 5, 6];
        assert_eq!(ct_swap(CtBool::TRUE, &mut a, &mut b), Err(ConstantTimeError::LengthMismatch));
    }

    #[test]
    fn ct_is_zero_all_values() {
        // Verify internal helper for boundary values
        assert_eq!(ct_is_zero(0), 0xFF);
        assert_eq!(ct_is_zero(1), 0x00);
        assert_eq!(ct_is_zero(127), 0x00);
        assert_eq!(ct_is_zero(128), 0x00);
        assert_eq!(ct_is_zero(255), 0x00);
    }

    #[test]
    fn ct_bool_debug() {
        assert_eq!(format!("{:?}", CtBool::TRUE), "CtBool(TRUE)");
        assert_eq!(format!("{:?}", CtBool::FALSE), "CtBool(FALSE)");
    }
}
