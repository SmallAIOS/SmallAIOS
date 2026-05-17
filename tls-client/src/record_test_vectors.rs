// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic byte-array fixtures for `record` unit tests.
//!
//! These constants are extracted from inline `#[cfg(test)]` literals in
//! `record.rs` so GitHub CodeQL's `rust/hard-coded-cryptographic-value`
//! heuristic does not re-fire on every scan. They are NOT loaded by any
//! production code path; the module is gated behind `#[cfg(test)]`.
//!
//! The corresponding `paths-ignore` entry lives in
//! `.github/codeql/codeql-config.yml` (`**/*_test_vectors.rs`). See
//! `openspec/changes/codeql-quality-cleanup-v1/specs/codeql-suppression-policy/spec.md`
//! for the full policy, and `security/src/argon2id_test_vectors.rs`
//! for the canonical pattern.
//!
//! These are ad-hoc fill keys / IVs for AEAD seal/open round-trip and
//! tamper-rejection tests of the TLS 1.3 record layer (RFC 8446 §5.2,
//! §5.3). None is real key material.

#![cfg(test)]

use super::NONCE_LEN;

/// Ad-hoc fill key (`0x42` repeated) for round-trip seal/open tests.
pub(super) const FILL_KEY_42: [u8; 32] = [0x42u8; 32];

/// Ad-hoc fill static IV (`0x9e` repeated) for round-trip seal/open tests.
pub(super) const FILL_IV_9E: [u8; NONCE_LEN] = [0x9eu8; NONCE_LEN];

/// All-zero key for tamper / mismatch rejection tests.
pub(super) const ZERO_KEY: [u8; 32] = [0u8; 32];

/// All-zero static IV for tamper / mismatch rejection tests.
pub(super) const ZERO_IV: [u8; NONCE_LEN] = [0u8; NONCE_LEN];
