// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test vectors for the SHA-1 implementation.
//!
//! All vectors below are public KATs taken verbatim from FIPS 180-4
//! Appendix A and RFC 3174 §7.3. None of these are credentials —
//! they exist solely as parser/oracle inputs for the regression suite.
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 3174 / FIPS 180-4 KAT vectors

#![allow(dead_code)]

// ─── KAT inputs ─────────────────────────────────────────────────────────────

/// Empty-string vector. Spec: FIPS 180-4 §A / RFC 3174 §7.3.
pub const EMPTY_INPUT: &[u8] = b"";

/// "abc" vector. Spec: FIPS 180-4 §A.1.
pub const ABC_INPUT: &[u8] = b"abc";

/// 56-byte boundary-tester ending exactly where the length suffix
/// would land on the first block — exercises the
/// "needs-extra-padding-block" branch.
pub const BOUNDARY_56_INPUT: &[u8] = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

/// Same 448-bit message as `BOUNDARY_56_INPUT`. Spec: FIPS 180-4 §A.2.
pub const TWO_BLOCK_INPUT: &[u8] = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

/// 64-byte input — exact block-boundary case.
pub const EXACT_BLOCK_INPUT: &[u8] =
    b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// Streaming-test input. Mid-sized message (135 bytes) so the
/// byte-by-byte test traverses 2+ block compressions and exercises
/// the residue-buffer logic.
pub const CHUNKED_STREAM_INPUT: &[u8] =
    b"The quick brown fox jumps over the lazy dog. The quick brown fox \
      jumps over the lazy dog. SHA-1 streaming-buffer regression test.";

// ─── KAT outputs ────────────────────────────────────────────────────────────

/// Hex-encoded digest for `SHA1("")`.
pub const EMPTY_DIGEST_HEX: &str = "da39a3ee5e6b4b0d3255bfef95601890afd80709";

/// Hex-encoded digest for `SHA1("abc")`.
pub const ABC_DIGEST_HEX: &str = "a9993e364706816aba3e25717850c26c9cd0d89d";

/// Hex-encoded digest for `SHA1(BOUNDARY_56_INPUT)` /
/// `SHA1(TWO_BLOCK_INPUT)`.
pub const TWO_BLOCK_DIGEST_HEX: &str = "84983e441c3bd26ebaae4aa1f95129e5e54670f1";

/// Hex-encoded digest for `SHA1("a" * 1_000_000)`.
pub const MILLION_A_DIGEST_HEX: &str = "34aa973cd4c4daa4f61eeb2bdbad27316534016f";
