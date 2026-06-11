// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic fixtures for `hkdf` unit tests.
//!
//! These constants are extracted from inline `#[cfg(test)]` literals in
//! `hkdf.rs` so GitHub CodeQL's `rust/hard-coded-cryptographic-value`
//! heuristic does not re-fire on every scan. They are NOT loaded by any
//! production code path; the module is gated behind `#[cfg(test)]`.
//!
//! The corresponding `paths-ignore` entry lives in
//! `.github/codeql/codeql-config.yml` (`**/*_test_vectors.rs`). See the
//! `codeql-suppression-policy` spec (archived change
//! `2026-05-10-codeql-quality-cleanup-v1`) for the policy and
//! `security/src/argon2id_test_vectors.rs` for the canonical pattern.
//!
//! `IKM`/`SALT`/`INFO` are the RFC 5869 A.1 test-case-1 inputs; the
//! zero salts exercise the §2.2 empty-salt default. None is real key
//! material.

#![cfg(test)]

use crate::sha2::{DIGEST_LEN, SHA384_DIGEST_LEN};

/// RFC 5869 A.1 test case 1 input keying material (`0x0b` × 22).
pub(super) const IKM: [u8; 22] = [0x0b; 22];

/// RFC 5869 A.1 test case 1 salt (`0x00..0x0c`).
pub(super) const SALT: [u8; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x0a, 0x0b, 0x0c];

/// RFC 5869 A.1 test case 1 info (`0xf0..0xf9`).
pub(super) const INFO: [u8; 10] = [0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9];

/// Explicit all-zero SHA-256-length salt — RFC 5869 §2.2's default.
pub(super) const ZERO_SALT_SHA256: [u8; DIGEST_LEN] = [0u8; DIGEST_LEN];

/// Explicit all-zero SHA-384-length salt — RFC 5869 §2.2's default.
pub(super) const ZERO_SALT_SHA384: [u8; SHA384_DIGEST_LEN] = [0u8; SHA384_DIGEST_LEN];
