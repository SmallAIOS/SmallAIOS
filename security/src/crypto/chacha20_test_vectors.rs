// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic byte-array fixtures for `chacha20` unit tests.
//!
//! These constants are extracted from inline `#[cfg(test)]` literals in
//! `chacha20.rs` so GitHub CodeQL's `rust/hard-coded-cryptographic-value`
//! heuristic does not re-fire on every scan. They are NOT loaded by any
//! production code path; the module is gated behind `#[cfg(test)]`.
//!
//! The corresponding `paths-ignore` entry lives in
//! `.github/codeql/codeql-config.yml` (`**/*_test_vectors.rs`). See
//! `openspec/changes/codeql-quality-cleanup-v1/specs/codeql-suppression-policy/spec.md`
//! for the full policy, and `security/src/argon2id_test_vectors.rs`
//! for the canonical pattern.
//!
//! References:
//! - RFC 8439 §2.3.2 — ChaCha20 block-function test vector
//! - RFC 8439 §2.4.2 — ChaCha20 encryption test vector

#![cfg(test)]

use super::{KEY_LEN, NONCE_LEN};

// ── RFC 8439 §2.3.2 block-function vector ──────────────────────────────────

/// RFC 8439 §2.3.2 256-bit key — incrementing bytes `0x00..=0x1f`.
pub(super) const RFC8439_2_3_2_KEY: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

/// RFC 8439 §2.3.2 96-bit nonce.
pub(super) const RFC8439_2_3_2_NONCE: [u8; 12] = [0, 0, 0, 9, 0, 0, 0, 0x4a, 0, 0, 0, 0];

/// RFC 8439 §2.3.2 expected 64-byte keystream block (counter = 1).
pub(super) const RFC8439_2_3_2_EXPECTED: [u8; 64] = [
    0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4,
    0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e,
    0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09, 0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2,
    0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9, 0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
];

// ── RFC 8439 §2.4.2 encryption vector ──────────────────────────────────────

/// RFC 8439 §2.4.2 256-bit key — incrementing bytes `0x00..=0x1f`.
pub(super) const RFC8439_2_4_2_KEY: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

/// RFC 8439 §2.4.2 96-bit nonce.
pub(super) const RFC8439_2_4_2_NONCE: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0x4a, 0, 0, 0, 0];

/// RFC 8439 §2.4.2 expected first 16 ciphertext bytes (counter = 1).
pub(super) const RFC8439_2_4_2_EXPECTED_FIRST: [u8; 16] = [
    0x6e, 0x2e, 0x35, 0x9a, 0x25, 0x68, 0xf9, 0x80, 0x41, 0xba, 0x07, 0x28, 0xdd, 0x0d, 0x69, 0x81,
];

// ── Involution fixture (ad-hoc, not an RFC vector) ─────────────────────────

/// Ad-hoc fill key for the keystream-involution property test.
pub(super) const INVOLUTION_KEY: [u8; KEY_LEN] = [0x42u8; KEY_LEN];

/// Ad-hoc fill nonce for the keystream-involution property test.
pub(super) const INVOLUTION_NONCE: [u8; NONCE_LEN] = [0x9eu8; NONCE_LEN];
