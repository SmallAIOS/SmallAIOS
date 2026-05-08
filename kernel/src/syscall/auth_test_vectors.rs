// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic byte-array fixtures for the auth-syscall handler unit
//! tests in `auth.rs`.
//!
//! These constants are extracted from inline `#[cfg(test)]` literals in
//! `auth.rs` so GitHub CodeQL's `rust/hard-coded-cryptographic-value`
//! heuristic does not re-fire on every scan. They are NOT loaded by any
//! production code path; the module is gated behind `#[cfg(test)]`.
//!
//! The corresponding `paths-ignore` entry lives in
//! `.github/codeql/codeql-config.yml` (`**/*_test_vectors.rs`). See
//! `openspec/changes/codeql-quality-cleanup-v1/specs/codeql-suppression-policy/spec.md`
//! for the full policy, and `net/src/quic/protection_test_vectors.rs`
//! for the canonical pattern.
//!
//! All passwords here are synthetic test fixtures driven through Argon2id
//! at the `tiny` tier — they are NOT shipped credentials.

#![cfg(test)]

// ── Salt fixtures ──────────────────────────────────────────────────────────

/// Sixteen-byte zero salt for [`super::tests::make_phc`]. The PHC strings
/// produced by hashing test passwords against this salt are never used as
/// real credentials.
pub(super) const ZERO_SALT_16: [u8; 16] = [0u8; 16];

/// Three-byte zero padding for `WhoamiOut::_pad`.
pub(super) const WHOAMI_PAD_3: [u8; 3] = [0; 3];

/// Four-byte zero padding for `WhoamiOut::_pad2`.
pub(super) const WHOAMI_PAD_4: [u8; 4] = [0; 4];

// ── Test passwords ─────────────────────────────────────────────────────────
//
// All entries are synthetic ASCII patterns used to exercise login,
// change-password, create-user, and whoami code paths. None are real
// secrets.

/// Canonical "happy-path" password used by most login tests.
pub(super) const PASSWORD_CORRECT_HORSE: &[u8] = b"correct-horse";

/// Negative-case password — must not match `PASSWORD_CORRECT_HORSE`.
pub(super) const PASSWORD_WRONG: &[u8] = b"wrong-password";

/// Generic short password used in capacity / EAGAIN / negative tests.
pub(super) const PASSWORD_PW: &[u8] = b"pw";

/// Old password for self-rotate flow.
pub(super) const PASSWORD_OLD_PW: &[u8] = b"old-pw";

/// New password for self-rotate flow (≥ minimum length).
pub(super) const PASSWORD_NEW_PW_LONG: &[u8] = b"new-pw-much-longer";

/// Old password for cross-rotate flow.
pub(super) const PASSWORD_CORRECT_OLD: &[u8] = b"correct-old";

/// Wrong-old password for cross-rotate negative flow.
pub(super) const PASSWORD_WRONG_OLD: &[u8] = b"wrong-old";

/// Truncated new password for negative flow.
pub(super) const PASSWORD_NEW_PW: &[u8] = b"new-pw";

/// Operator-tier login password.
pub(super) const PASSWORD_OP_PW: &[u8] = b"op-pw";

/// Viewer-tier login password.
pub(super) const PASSWORD_VIEWER_PW: &[u8] = b"viewer-pw";

/// Root-tier login password.
pub(super) const PASSWORD_ROOT_PW: &[u8] = b"root-pw";

/// Stand-in for a meaningless old-password input on a cross-rotate.
pub(super) const PASSWORD_UNUSED: &[u8] = b"unused";

/// Generic per-user password for duplicate-create test fixtures.
pub(super) const PASSWORD_X_PW: &[u8] = b"x-pw";

/// Initial password for `create_user` flow.
pub(super) const PASSWORD_INITIAL_PW: &[u8] = b"initial-pw";

/// Wrong-password input for `auth_login`.
pub(super) const PASSWORD_ANY_FOR_LOGIN: &[u8] = b"any-password";

// ── Username fixtures (test-only, bound to passwords above) ────────────────

/// Filler bytes used to occupy session-table slots.
pub(super) const FILLER_USERNAME: &[u8] = b"filler";

/// Username for a victim of a cross-rotate force-logout test.
pub(super) const VICTIM_USERNAME: &[u8] = b"victim";

/// Phase-3 placeholder factor-2 input — must be rejected with EINVAL.
pub(super) const PHASE3_FACTOR2: &[u8] = b"123456";
