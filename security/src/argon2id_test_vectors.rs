// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic byte-array fixtures for `argon2id` unit tests.
//!
//! These constants are extracted from inline `#[cfg(test)]` literals in
//! `argon2id.rs` so GitHub CodeQL's `rust/hard-coded-cryptographic-value`
//! heuristic does not re-fire on every scan. They are NOT loaded by any
//! production code path; the module is gated behind `#[cfg(test)]`.
//!
//! The corresponding `paths-ignore` entry lives in
//! `.github/codeql/codeql-config.yml` (`**/*_test_vectors.rs`). See
//! `openspec/changes/codeql-quality-cleanup-v1/specs/codeql-suppression-policy/spec.md`
//! for the full policy, and `net/src/quic/protection_test_vectors.rs`
//! for the canonical pattern.
//!
//! References:
//! - RFC 9106 §5.3 — Argon2id reference test vector

#![cfg(test)]

// ── RFC 9106 §5.3 reference vector ─────────────────────────────────────────
//
// Parameters: t=3, m=32 KiB, p=4, version=0x13, type=Argon2id, tag length=32.
// The expected tag below is the ground truth from RFC 9106 §5.3.

/// RFC 9106 §5.3 password input — 32 bytes of `0x01`.
pub(super) const RFC9106_PASSWORD: [u8; 32] = [0x01u8; 32];

/// RFC 9106 §5.3 salt input — 16 bytes of `0x02`.
pub(super) const RFC9106_SALT: [u8; 16] = [0x02u8; 16];

/// RFC 9106 §5.3 secret (key) input — 8 bytes of `0x03`.
pub(super) const RFC9106_SECRET: [u8; 8] = [0x03u8; 8];

/// RFC 9106 §5.3 associated-data input — 12 bytes of `0x04`.
pub(super) const RFC9106_AD: [u8; 12] = [0x04u8; 12];

/// RFC 9106 §5.3 expected 32-byte Argon2id tag:
///
/// ```text
/// 0d 64 0d f5 8d 78 76 6c  08 c0 37 a3 4a 8b 53 c9
/// d0 1e f0 45 2d 75 b6 5e  b5 25 20 e9 6b 01 e6 59
/// ```
pub(super) const RFC9106_EXPECTED_TAG: [u8; 32] = [
    0x0D, 0x64, 0x0D, 0xF5, 0x8D, 0x78, 0x76, 0x6C, 0x08, 0xC0, 0x37, 0xA3, 0x4A, 0x8B, 0x53, 0xC9,
    0xD0, 0x1E, 0xF0, 0x45, 0x2D, 0x75, 0xB6, 0x5E, 0xB5, 0x25, 0x20, 0xE9, 0x6B, 0x01, 0xE6, 0x59,
];

// ── PHC round-trip fixtures ────────────────────────────────────────────────

/// Plaintext password used by `phc_round_trip` to drive a hash → format →
/// parse → verify cycle. Not a credential.
pub(super) const PHC_ROUND_TRIP_PASSWORD: &[u8] = b"correct horse battery staple";

/// Salt used by `phc_round_trip`. Sixteen-byte ASCII pattern.
pub(super) const PHC_ROUND_TRIP_SALT: &[u8] = b"saltsalt12345678";

/// Negative-case password for `phc_round_trip` — verify must reject this
/// against a PHC produced from `PHC_ROUND_TRIP_PASSWORD`.
pub(super) const PHC_ROUND_TRIP_WRONG_PASSWORD: &[u8] = b"wrong password";

// ── Short-salt rejection fixture ───────────────────────────────────────────

/// Plaintext password for `short_salt_rejected`.
pub(super) const SHORT_SALT_PASSWORD: &[u8] = b"pw";

/// Salt that is shorter than 8 bytes and must be rejected.
pub(super) const SHORT_SALT_INVALID: &[u8] = b"short";

// ── Oracle-comparison fixtures ─────────────────────────────────────────────

/// Plaintext password used by `argon2_oracle_matches` to cross-check
/// against the upstream `argon2` crate.
pub(super) const ORACLE_PASSWORD: &[u8] = b"oracle-test-password";

/// 17-byte salt (≥ 8) for `argon2_oracle_matches`.
pub(super) const ORACLE_SALT: &[u8] = b"oracle-test-saltX";

// ── PHC interoperability fixtures ──────────────────────────────────────────

/// Plaintext password used by `phc_interoperates_with_oracle`.
pub(super) const INTEROP_PASSWORD: &[u8] = b"interop-test";

/// 20-byte salt for `phc_interoperates_with_oracle`.
pub(super) const INTEROP_SALT: &[u8] = b"interop-saltAAAAAAAA";

// ── Parameter-sweep fixtures ───────────────────────────────────────────────

/// Plaintext password used by `parameter_sweep`.
pub(super) const SWEEP_PASSWORD: &[u8] = b"sweep";

/// 16-byte salt for `parameter_sweep`.
pub(super) const SWEEP_SALT: &[u8] = b"sweepsalt0000000";
