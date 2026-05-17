// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic byte-array fixtures for `chacha20_poly1305` unit tests.
//!
//! These constants are extracted from inline `#[cfg(test)]` literals in
//! `chacha20_poly1305.rs` so GitHub CodeQL's
//! `rust/hard-coded-cryptographic-value` heuristic does not re-fire on
//! every scan. They are NOT loaded by any production code path; the
//! module is gated behind `#[cfg(test)]`.
//!
//! The corresponding `paths-ignore` entry lives in
//! `.github/codeql/codeql-config.yml` (`**/*_test_vectors.rs`). See
//! `openspec/changes/codeql-quality-cleanup-v1/specs/codeql-suppression-policy/spec.md`
//! for the full policy, and `security/src/argon2id_test_vectors.rs`
//! for the canonical pattern.
//!
//! References:
//! - RFC 8439 §2.8.2 — ChaCha20-Poly1305 AEAD test vector

#![cfg(test)]

use super::{KEY_LEN, NONCE_LEN};

// ── RFC 8439 §2.8.2 AEAD vector ────────────────────────────────────────────

/// RFC 8439 §2.8.2 12-byte additional authenticated data.
pub(super) const RFC8439_2_8_2_AAD: [u8; 12] = [
    0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
];

/// RFC 8439 §2.8.2 256-bit AEAD key.
pub(super) const RFC8439_2_8_2_KEY: [u8; 32] = [
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
];

/// RFC 8439 §2.8.2 96-bit nonce.
pub(super) const RFC8439_2_8_2_NONCE: [u8; 12] = [
    0x07, 0, 0, 0, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
];

/// RFC 8439 §2.8.2 expected first 16 ciphertext bytes.
pub(super) const RFC8439_2_8_2_EXPECTED_CT_FIRST: [u8; 16] = [
    0xd3, 0x1a, 0x8d, 0x34, 0x64, 0x8e, 0x60, 0xdb, 0x7b, 0x86, 0xaf, 0xbc, 0x53, 0xef, 0x7e, 0xc2,
];

/// RFC 8439 §2.8.2 expected 16-byte Poly1305 tag.
pub(super) const RFC8439_2_8_2_EXPECTED_TAG: [u8; 16] = [
    0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60, 0x06, 0x91,
];

// ── Tamper / round-trip fixtures (ad-hoc fill values) ──────────────────────

/// Ad-hoc fill key used by the tamper-rejection round-trip tests.
pub(super) const FILL_KEY_42: [u8; KEY_LEN] = [0x42u8; KEY_LEN];

/// Ad-hoc fill nonce used by the tamper-rejection round-trip tests.
pub(super) const FILL_NONCE_9E: [u8; NONCE_LEN] = [0x9eu8; NONCE_LEN];

/// All-zero key used by the tampered-tag / tampered-aad tests.
pub(super) const ZERO_KEY: [u8; KEY_LEN] = [0u8; KEY_LEN];

/// All-zero nonce used by the tampered-tag / tampered-aad tests.
pub(super) const ZERO_NONCE: [u8; NONCE_LEN] = [0u8; NONCE_LEN];
