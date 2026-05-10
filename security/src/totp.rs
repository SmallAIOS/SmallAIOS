// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RFC 6238 Time-Based One-Time Password (TOTP) — `#![no_std]` clean room.
//!
//! TOTP is the second factor SmallAIOS exposes to operators via
//! [`auth_login`] and [`auth_totp_setup`] (kernel syscalls 0x90 and 0x95).
//! Per spec
//! `openspec/changes/management-login-v1/specs/security/spec.md`,
//! the SHA-1 HMAC variant is the one this module ships — that is what
//! every common authenticator app (Google Authenticator, Authy, Aegis,
//! 1Password, etc.) speaks. A SHA-3-based variant MAY be added in a
//! future change without breaking the SHA-1 path (spec Q21).
//!
//! ## Algorithm summary (RFC 6238 §4 / RFC 4226)
//!
//! ```text
//! T  = (unix_time - T0) / X            // typically T0=0, X=30 seconds
//! HS = HMAC-SHA1(K, T_be64)            // 8-byte big-endian counter
//! offset = HS[19] & 0x0F               // dynamic truncation
//! P  = (HS[offset]   & 0x7F) << 24
//!    | (HS[offset+1] & 0xFF) << 16
//!    | (HS[offset+2] & 0xFF) <<  8
//!    | (HS[offset+3] & 0xFF)
//! TOTP = P mod 10^digits
//! ```
//!
//! ## RFC 6238 Appendix B test vectors
//!
//! ASCII secret "12345678901234567890" (20 bytes) at
//! `t = 59 / 1111111109 / 1234567890 / 2000000000 / 20000000000`
//! produces the 8-digit codes
//! `94287082 / 07081804 / 14050471 / 89005924 / 69279037`. Pinned in
//! [`crate::totp_test_vectors`].
//!
//! ## API
//!
//! - [`totp_generate`] — produce a code at a given Unix time.
//! - [`totp_verify`] — accept a code if it lies within `±window` steps.

use crate::hmac_sha1::hmac_sha1;

/// Default time step in seconds — 30, per RFC 6238 §5.2 recommendation.
pub const DEFAULT_PERIOD_SECONDS: u32 = 30;

/// Default digit count — 6, the most widely-supported authenticator
/// length.
pub const DEFAULT_DIGITS: u32 = 6;

/// Maximum digits supported. Larger values overflow the dynamic-truncation
/// 31-bit window and cease to be RFC-compliant.
pub const MAX_DIGITS: u32 = 9;

/// 10^k table for `k` in `0..=MAX_DIGITS`. Used by the `code mod 10^digits`
/// step. All values fit in a `u32` (10^9 = 1_000_000_000 ≤ 2^32-1).
const POW10: [u32; (MAX_DIGITS as usize) + 1] = [
    1,
    10,
    100,
    1000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
];

/// Compute a TOTP code for the given (secret, time, digits, period).
///
/// `secret` is the raw HMAC key — typically the 20 bytes returned by
/// [`auth_totp_setup`]. `unix_time` is the seconds-since-epoch at which
/// the code is being computed. `digits` selects the code length (1..=9).
/// `period` is the time step in seconds (≥ 1; 30 is the RFC 6238
/// default).
///
/// The return value is a numeric code — pad with leading zeros to
/// `digits` characters when displaying.
///
/// ## Edge cases
/// - `digits = 0` ⇒ returns 0 (degenerate, never useful in practice).
/// - `period = 0` ⇒ treated as 1 to avoid a division-by-zero panic.
/// - `digits > MAX_DIGITS` ⇒ clamped to `MAX_DIGITS`.
pub fn totp_generate(secret: &[u8], unix_time: u64, digits: u32, period: u32) -> u32 {
    let period = if period == 0 { 1 } else { period };
    let digits = digits.min(MAX_DIGITS);
    let counter = unix_time / (period as u64);
    let mut counter_be = [0u8; 8];
    counter_be.copy_from_slice(&counter.to_be_bytes());
    let hs = hmac_sha1(secret, &counter_be);
    truncate(&hs, digits)
}

/// Verify a code against the secret, accepting any time step in
/// `[t - window, t + window]`. RFC 6238 §6 recommends `window = 1`
/// (i.e. ±1 step / ±period seconds tolerance) for clock-skew
/// robustness; larger values widen the attack surface.
///
/// Returns `true` iff some step in the tolerance window produces the
/// supplied code. The full window is *always* walked even if an early
/// hit is found, to keep the check timing-equivalent regardless of
/// where the match falls. The match flag is OR-ed into a running
/// boolean so a constant amount of work is done per call.
///
/// ## Edge cases
/// - `period = 0` ⇒ treated as 1 (matches [`totp_generate`]).
/// - `window > 16` ⇒ clamped to 16 (DoS guard).
/// - Numerical underflow when `unix_time < window * period` is handled
///   via saturating subtraction; effectively the verifier treats the
///   pre-epoch window as time 0.
pub fn totp_verify(
    secret: &[u8],
    code: u32,
    unix_time: u64,
    digits: u32,
    period: u32,
    window: u32,
) -> bool {
    let period = if period == 0 { 1 } else { period };
    let digits = digits.min(MAX_DIGITS);
    let window = window.min(16);

    let period64 = period as u64;
    let window64 = window as u64;

    let center = unix_time / period64;
    let lo = center.saturating_sub(window64);
    let hi = center.saturating_add(window64);

    // Normalise the supplied `code` to the same modulus the truncation
    // step produces, so a caller passing `123456` and a caller passing
    // `1_000_000_123_456` both verify identically.
    let modulus = POW10[digits as usize];
    let normalised_code = code % modulus;

    let mut accept: u32 = 0; // 0 / 1; combined with bitwise OR.
    let mut step = lo;
    while step <= hi {
        let mut counter_be = [0u8; 8];
        counter_be.copy_from_slice(&step.to_be_bytes());
        let hs = hmac_sha1(secret, &counter_be);
        let candidate = truncate(&hs, digits);
        // Constant-time-equivalent equality check. Both sides have
        // already been reduced modulo `10^digits`, so the comparison
        // is over canonical values. The branchless `ct_eq_u32` keeps
        // the per-step latency uniform and lets us OR the result into
        // `accept` without a short-circuit.
        let m = ct_eq_u32(candidate, normalised_code);
        accept |= m;
        step += 1;
    }
    accept != 0
}

/// RFC 4226 §5.3 dynamic truncation. Pulled out so unit tests can
/// drive it directly with the RFC vectors.
fn truncate(hs: &[u8; 20], digits: u32) -> u32 {
    let offset = (hs[19] & 0x0F) as usize;
    let p = ((hs[offset] & 0x7F) as u32) << 24
        | (hs[offset + 1] as u32) << 16
        | (hs[offset + 2] as u32) << 8
        | (hs[offset + 3] as u32);
    let modulus = POW10[digits.min(MAX_DIGITS) as usize];
    p % modulus
}

/// Branchless equality check producing 1 on match, 0 otherwise.
/// Used by [`totp_verify`] to keep the per-step work uniform.
fn ct_eq_u32(a: u32, b: u32) -> u32 {
    // `a ^ b` is 0 iff a == b. We then collapse all bits down with a
    // saturating non-zero detector that produces 0/1 without a branch.
    let diff = a ^ b;
    // If `diff == 0`, returns 1; otherwise 0. `wrapping_sub(1) >> 31`
    // overflows when `diff == 0` to give all-ones; we shift the high
    // bit down.
    ((diff.wrapping_sub(1)) & !diff) >> 31
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "totp_test_vectors.rs"]
mod test_vectors;

#[cfg(test)]
mod tests {
    use super::test_vectors::*;
    use super::*;

    #[test]
    fn rfc6238_appendix_b_vectors_match_8_digit() {
        // Spec scenario "RFC 6238 vector matches" — pin all 5 SHA-1
        // entries from RFC 6238 Appendix B Table 1.
        for (t, expected) in RFC6238_VECTORS {
            let got = totp_generate(RFC6238_SECRET_SHA1, *t, 8, 30);
            assert_eq!(
                got, *expected,
                "TOTP({t}) expected {expected:08} got {got:08}"
            );
        }
    }

    #[test]
    fn classic_6_digit_at_t59_is_287082() {
        // RFC 6238 Appendix B's 8-digit code at t=59 truncates to a
        // 6-digit code of 287082 (the canonical "first" example).
        let code = totp_generate(RFC6238_SECRET_SHA1, 59, 6, 30);
        assert_eq!(code, 287082);
    }

    #[test]
    fn verify_accepts_correct_code_in_centre_window() {
        let t = 1_111_111_109u64;
        let code = totp_generate(RFC6238_SECRET_SHA1, t, 6, 30);
        assert!(totp_verify(RFC6238_SECRET_SHA1, code, t, 6, 30, 1));
    }

    #[test]
    fn verify_accepts_one_step_skew_within_window() {
        // Spec scenario "Clock-skew tolerance".
        //
        // The verifier accepts ±window TOTP steps. Pick a `t` aligned
        // to the start of a 30-second window so `t + 30` lands cleanly
        // in the next step, exercising the +1 tolerance — and
        // separately verify a verifier 30 s *earlier* still accepts
        // the same code (the −1 tolerance branch).
        let t = 1_111_111_080u64; // 30-aligned: 37_037_036 * 30
        let code = totp_generate(RFC6238_SECRET_SHA1, t, 6, 30);

        // +1 step: t + 30 lands one window later → window=1 accepts.
        let later = t + 30;
        assert!(totp_verify(RFC6238_SECRET_SHA1, code, later, 6, 30, 1));

        // -1 step: t - 30 lands one window earlier → window=1 accepts.
        let earlier = t.saturating_sub(30);
        assert!(totp_verify(RFC6238_SECRET_SHA1, code, earlier, 6, 30, 1));
    }

    #[test]
    fn verify_rejects_out_of_window_code() {
        // Spec scenario "Out-of-window code rejected".
        let t = 1_111_111_100u64;
        let code = totp_generate(RFC6238_SECRET_SHA1, t, 6, 30);
        // t + 90 is 3 steps later; window=1 must reject.
        let way_later = t + 90;
        assert!(!totp_verify(RFC6238_SECRET_SHA1, code, way_later, 6, 30, 1));
    }

    #[test]
    fn verify_rejects_obviously_wrong_code() {
        let t = 1_111_111_100u64;
        let code = totp_generate(RFC6238_SECRET_SHA1, t, 6, 30) ^ 0x000F_FFFF;
        // Mutated code SHALL fail. Tiny risk that the XOR happens to
        // hit another valid code at a different step within the
        // window; pick a window-0 verify to eliminate that.
        assert!(!totp_verify(RFC6238_SECRET_SHA1, code, t, 6, 30, 0));
    }

    #[test]
    fn verify_negative_window_step_handles_pre_epoch() {
        // Saturating-sub guard: window=5 at t=10 must not panic.
        let code = totp_generate(RFC6238_SECRET_SHA1, 10, 6, 30);
        let _ = totp_verify(RFC6238_SECRET_SHA1, code, 10, 6, 30, 5);
    }

    #[test]
    fn period_zero_treated_as_one() {
        // Defensive: period=0 must not divide by zero.
        let _ = totp_generate(RFC6238_SECRET_SHA1, 100, 6, 0);
        let _ = totp_verify(RFC6238_SECRET_SHA1, 0, 100, 6, 0, 1);
    }

    #[test]
    fn digits_clamped_to_max() {
        // digits > MAX_DIGITS must be silently clamped.
        let r = totp_generate(RFC6238_SECRET_SHA1, 59, 100, 30);
        let r2 = totp_generate(RFC6238_SECRET_SHA1, 59, MAX_DIGITS, 30);
        assert_eq!(r, r2);
    }

    #[test]
    fn truncation_offset_matches_rfc4226_example() {
        // RFC 4226 §5.4 worked example HMAC. The dynamic-truncation
        // step extracts `0x50ef7f19 = 1_357_872_921` (after masking
        // the high bit). The 6-digit truncation is the spec's
        // canonical "first" example.
        let hs: [u8; 20] = TRUNCATION_EXAMPLE_HS;
        assert_eq!(truncate(&hs, 6), 872_921);
        // 8-digit truncation: 1_357_872_921 mod 100_000_000 =
        // 57_872_921. Cross-check the wider window doesn't
        // accidentally collapse to a 6-digit value.
        assert_eq!(truncate(&hs, 8), 57_872_921);
    }

    #[test]
    fn ct_eq_u32_branchless() {
        assert_eq!(ct_eq_u32(0, 0), 1);
        assert_eq!(ct_eq_u32(7, 7), 1);
        assert_eq!(ct_eq_u32(0, 1), 0);
        assert_eq!(ct_eq_u32(u32::MAX, 0), 0);
        assert_eq!(ct_eq_u32(u32::MAX, u32::MAX), 1);
    }

    #[test]
    fn window_clamped_to_sixteen() {
        // window > 16 SHALL be clamped — i.e. behave the same as
        // window = 16. We assert the verify accepts any code at
        // window=16 vs window=10000 identically.
        let t = 1_000_000u64;
        let code = totp_generate(RFC6238_SECRET_SHA1, t, 6, 30);
        let a = totp_verify(RFC6238_SECRET_SHA1, code, t, 6, 30, 16);
        let b = totp_verify(RFC6238_SECRET_SHA1, code, t, 6, 30, 10_000);
        assert_eq!(a, b);
    }

    #[test]
    fn small_secret_does_not_panic() {
        // 1-byte secret — well below the 20-byte typical. Must still
        // produce a code without panicking.
        let s: &[u8] = &[0x42];
        let _ = totp_generate(s, 0, 6, 30);
        let _ = totp_verify(s, 0, 0, 6, 30, 1);
    }

    #[test]
    fn empty_secret_does_not_panic() {
        // RFC 4226 doesn't forbid empty secrets; the kernel-level
        // policy enforces a sane minimum, but the primitive must not
        // panic for hostile or fixture inputs.
        let _ = totp_generate(b"", 0, 6, 30);
    }
}
