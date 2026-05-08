// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic byte-array fixtures for `quic::protection` unit tests.
//!
//! These are not real cryptographic keys, IVs, or header-protection masks —
//! they are repeating-byte patterns chosen so test assertions remain easy
//! to read (e.g. `nonce[11] == 0xBB ^ 0x01`). They are NOT loaded by any
//! production code path; the module is gated behind `#[cfg(test)]`.
//!
//! Extracted from inline `#[cfg(test)]` constants in `protection.rs` so
//! that GitHub CodeQL's `rust/hard-coded-cryptographic-value` heuristic
//! does not re-fire on every scan. The corresponding `paths-ignore` entry
//! lives in `.github/codeql/codeql-config.yml` (`**/*_test_vectors.rs`).
//! See `openspec/changes/codeql-quality-cleanup-v1/specs/codeql-suppression-policy/spec.md`
//! for the full policy.
//!
//! References:
//! - RFC 9001 §5  — QUIC packet protection (key/IV/hp_key shape)
//! - RFC 9001 §5.3 — AEAD nonce derivation (this is what test_compute_nonce exercises)

#![cfg(test)]

use super::HP_SAMPLE_LEN;

// ── Default test_keys() fixture ────────────────────────────────────────────
//
// The test helper `test_keys()` builds a `PacketProtectionKeys` using these
// three byte patterns. The `0xBB` IV pattern is chosen so the nonce
// derivation tests can assert `nonce[11] == 0xBB ^ 0x01`.

/// Traffic key for the default test_keys() fixture (32 bytes of `0xAA`).
pub(super) const TEST_KEY_AA: [u8; 32] = [0xAA; 32];

/// Traffic IV for the default test_keys() fixture (12 bytes of `0xBB`).
pub(super) const TEST_IV_BB: [u8; 12] = [0xBB; 12];

/// Header-protection key for the default test_keys() fixture
/// (16 bytes of `0xCC`).
pub(super) const TEST_HP_KEY_CC: [u8; 16] = [0xCC; 16];

// ── "Wrong key" fixture for test_aead_decrypt_wrong_key ────────────────────

/// Traffic key for the wrong-key roundtrip test (32 bytes of `0x01`).
/// Used to verify AEAD authentication rejects a ciphertext when decrypted
/// with a different traffic key.
pub(super) const TEST_WRONG_KEY_01: [u8; 32] = [0x01; 32];

// ── Header protection roundtrip fixtures ───────────────────────────────────

/// Header-protection key for the roundtrip tests (16 bytes of `0xDD`).
pub(super) const TEST_HP_KEY_DD: [u8; 16] = [0xDD; 16];

/// Header-protection sample for the roundtrip tests
/// (`HP_SAMPLE_LEN` bytes of `0xEE`). Acts as a fixed input to the mask
/// derivation so XOR-self-inverse can be observed deterministically.
pub(super) const TEST_HP_SAMPLE_EE: [u8; HP_SAMPLE_LEN] = [0xEE; HP_SAMPLE_LEN];
