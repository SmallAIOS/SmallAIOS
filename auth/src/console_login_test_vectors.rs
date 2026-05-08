// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test fixtures for [`super::console_login`] and [`super::recovery`].
//!
//! All byte and string constants in this file are **synthetic** — they
//! exist only as parser oracles for unit tests, are deterministic, and
//! are not used to authenticate anything in production. The
//! `paths-ignore` rule in `.github/codeql/codeql-config.yml` excludes
//! `**/*_test_vectors.rs` from CodeQL scanning entirely.

#![allow(dead_code)] // shared fixture file — not every consumer uses every constant

use alloc::string::String;
use alloc::vec::Vec;

// ─── Synthetic Argon2id PHC strings ──────────────────────────────────────────

/// Single-byte synthetic test passwords used by the `lookup_user_*`
/// fixture set. Pre-extracted here (rather than inline in the parent
/// test module) so the `paths-ignore` rule covers them.
pub const LOOKUP_USER_PW_A: &[u8] = b"a";
pub const LOOKUP_USER_PW_B: &[u8] = b"b";

/// Synthetic Argon2id PHC over `b"hunter2"` with the `tiny` tier
/// parameters. Hash is fictional; tests only round-trip the *shape* and
/// rely on `argon2id_verify` for true matching.
///
/// lgtm[rust/hard-coded-cryptographic-value] — synthetic test fixture
pub const SYNTHETIC_TINY_PHC: &str =
    "$argon2id$v=19$m=8192,t=3,p=1$YWFhYWFhYWFhYWFhYWFhYQ$YWJjZGVmZ2hpamtsbW5vcA";

/// Second synthetic PHC for round-trip tests.
///
/// lgtm[rust/hard-coded-cryptographic-value] — synthetic test fixture
pub const SYNTHETIC_TINY_PHC_2: &str =
    "$argon2id$v=19$m=8192,t=3,p=1$c2FsdHNhbHRzYWx0$dGFndGFndGFn";

/// Plaintext password whose Argon2id-tiny hash is computed at test time
/// to install in the mock shadow provider. This is **not** a credential
/// — it is a deterministic literal whose only purpose is to drive the
/// console_login state machine.
///
/// lgtm[rust/hard-coded-credentials] — synthetic test fixture
pub const SYNTHETIC_GOOD_PASSWORD: &[u8] = b"correct horse battery staple";

/// Wrong-password fixture for negative tests.
///
/// lgtm[rust/hard-coded-credentials] — synthetic test fixture
pub const SYNTHETIC_WRONG_PASSWORD: &[u8] = b"hunter2";

/// Synthetic salt — 16 bytes, `0x00..0x0F`. Real Argon2id callers
/// generate a fresh CSPRNG salt; this is only used to make the test
/// hash deterministic.
///
/// lgtm[rust/hard-coded-cryptographic-value] — synthetic test fixture
pub const SYNTHETIC_SALT: [u8; 16] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
];

// ─── Expected prompt strings (for assertion) ─────────────────────────────────

/// First-boot prompt issued when `/data/auth/shadow` is absent.
pub const PROMPT_FIRST_BOOT_SET: &str = "Set initial root password: ";

/// First-boot confirmation prompt.
pub const PROMPT_FIRST_BOOT_CONFIRM: &str = "Confirm: ";

/// Steady-state username prompt.
pub const PROMPT_USERNAME: &str = "username: ";

/// Steady-state password prompt.
pub const PROMPT_PASSWORD: &str = "password: ";

/// Banner printed on successful login.
pub const BANNER_AUTH_OK: &str = "auth: ok";

/// Banner printed on authentication failure (string-form fixture).
pub const BANNER_AUTH_FAIL_FIXTURE: &str = "Authentication failed.";

/// Banner printed when the source is locked out (string-form fixture).
pub const BANNER_LOCKED_OUT_FIXTURE: &str = "Authentication locked: try again later.";

/// Banner printed when the first-boot retry budget is exhausted.
pub const BANNER_FIRST_BOOT_HALT: &str =
    "First-boot password setup failed; halting. Recovery: reboot with auth.skip-firstboot.";

/// Banner printed after `auth.skip-firstboot` injects a one-time root
/// password.
pub const BANNER_RECOVERY_HEADER: &str =
    "auth: recovery — temporary root password (rotate immediately):";

// ─── Synthetic recovery fixtures ─────────────────────────────────────────────

/// Pre-computed Argon2id PHC accepted via
/// `auth.skip-firstboot=<phc>`.
///
/// lgtm[rust/hard-coded-cryptographic-value] — synthetic test fixture
pub const SYNTHETIC_RECOVERY_PHC: &str =
    "$argon2id$v=19$m=8192,t=3,p=1$cmVjb3ZlcnlzYWx0aA$cmVjb3ZlcnktdGFndGFn";

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build a script of bytes that types `s` followed by `\n`.
pub fn typed_with_newline(s: &str) -> Vec<u8> {
    let mut v = Vec::from(s.as_bytes());
    v.push(b'\n');
    v
}

/// Concatenate any number of byte slices.
pub fn concat_bytes(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

/// Append a typed line plus newline to `script`.
pub fn append_line(script: &mut Vec<u8>, s: &str) {
    script.extend_from_slice(s.as_bytes());
    script.push(b'\n');
}

/// Build a [`String`] echoing the same prompts that the console_login
/// flow emits, for assertion against the captured echo trace.
pub fn expected_login_prompt() -> String {
    let mut s = String::new();
    s.push_str(PROMPT_USERNAME);
    s.push_str(PROMPT_PASSWORD);
    s
}
