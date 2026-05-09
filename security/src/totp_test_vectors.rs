// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TOTP KAT vectors from RFC 6238 Appendix B / RFC 4226 §5.4.
//!
//! All vectors are public test data taken verbatim from the RFCs; they
//! are NOT credentials.
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 6238 / RFC 4226 KAT vectors

#![allow(dead_code)]

/// RFC 6238 Appendix B reference secret for the SHA-1 variant —
/// "12345678901234567890" as ASCII, 20 bytes.
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 6238 Appendix B KAT
pub const RFC6238_SECRET_SHA1: &[u8] = b"12345678901234567890";

/// `(unix_time, expected 8-digit code)` tuples from RFC 6238 Appendix B
/// Table 1 (SHA-1 column only — SHA-256/SHA-512 omitted because Phase 9
/// ships the SHA-1 variant only, per spec Q21).
pub const RFC6238_VECTORS: &[(u64, u32)] = &[
    (59, 94_287_082),
    (1_111_111_109, 7_081_804),
    (1_111_111_111, 14_050_471),
    (1_234_567_890, 89_005_924),
    (2_000_000_000, 69_279_037),
    (20_000_000_000, 65_353_130),
];

/// Synthetic HMAC-SHA1 digest for the [`crate::totp::truncate`] regression
/// test — derived from RFC 4226 §5.4's worked example.
///
/// Note: this byte sequence corresponds to a hand-computed HMAC; the
/// exact provenance is the RFC 4226 §5.4 example column, where:
///
/// ```text
/// HS = 1f8698690e02ca16618550ef7f19da8e945b555a
/// offset = HS[19] & 0x0F = 0xA = 10
/// P = HS[10..14] with high bit cleared = 0x50ef7f19 = 1357872921
/// 6-digit TOTP = 1357872921 mod 1_000_000 = 872921
/// ```
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 4226 §5.4 worked-example HMAC
pub const TRUNCATION_EXAMPLE_HS: [u8; 20] = [
    0x1f, 0x86, 0x98, 0x69, 0x0e, 0x02, 0xca, 0x16, 0x61, 0x85, 0x50, 0xef, 0x7f, 0x19, 0xda, 0x8e,
    0x94, 0x5b, 0x55, 0x5a,
];
