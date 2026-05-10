// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared POSIX errno table for the admin keyspace.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-zenoh-admin/spec.md`
//! Requirement "Admin keyspace contract" — every error response carries
//! `code: <negative-errno>` and the same numeric value the kernel
//! returns at the syscall boundary. `error_string()` is the **single**
//! mapping from numeric code to the human-readable string returned in
//! the `reason` field.
//!
//! This module deliberately re-exports the kernel's `ERRNO_*`
//! constants instead of redefining them. The kernel session table and
//! the admin wrapper share the exact same numeric space — when the
//! kernel returns `-EAGAIN` from `auth_login`, the wrapper formats it
//! as `{ code: -11, reason: "Account temporarily locked out" }` (the
//! verb adds context above the bare `error_string`).

use smallaios_kernel::auth::{
    ERRNO_EACCES, ERRNO_EAGAIN, ERRNO_EAUTHEXPIRED, ERRNO_EEXIST, ERRNO_EFAULT, ERRNO_EINVAL,
    ERRNO_ENOENT, ERRNO_ENOSPC, ERRNO_ENOSYS,
};

/// `EPERM` — operation not permitted. Used for peer-mismatch and any
/// "you do not have the role for this action" rejection. The kernel
/// uses `EACCES` for credential failures; we keep `EPERM = -1` here
/// for capability-denial errors that aren't credential failures.
pub const ERRNO_EPERM: i64 = -1;

/// `EAUTH` — synthetic authentication-required error. Returned when a
/// non-`login` admin verb arrives with an empty or malformed token.
/// Distinct from `EACCES` (which means "credentials wrong") and
/// `EAUTHEXPIRED` (which means "credentials valid but expired").
pub const ERRNO_EAUTH: i64 = -80;

/// Convert a numeric errno to the canonical `reason` string used in
/// the JSON error response.
///
/// Verb handlers MAY override the reason for additional context (e.g.
/// `"Per-identity session cap reached"` instead of `"Try again"`).
/// This table is the fallback mapping.
pub const fn error_string(code: i64) -> &'static str {
    match code {
        // -1
        ERRNO_EPERM => "Operation not permitted",
        // -2
        ERRNO_ENOENT => "No such file or directory",
        // -11
        ERRNO_EAGAIN => "Try again",
        // -13
        ERRNO_EACCES => "Authentication failed",
        // -14
        ERRNO_EFAULT => "Bad address",
        // -17
        ERRNO_EEXIST => "Already exists",
        // -22
        ERRNO_EINVAL => "Invalid argument",
        // -28
        ERRNO_ENOSPC => "No space left",
        // -38
        ERRNO_ENOSYS => "Function not implemented",
        // -80
        ERRNO_EAUTH => "Authentication required",
        // -81
        ERRNO_EAUTHEXPIRED => "Session expired",
        _ => "Unknown error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_codes_map_to_distinct_strings() {
        // Every "live" errno must produce a non-default string.
        for code in [
            ERRNO_EPERM,
            ERRNO_ENOENT,
            ERRNO_EAGAIN,
            ERRNO_EACCES,
            ERRNO_EFAULT,
            ERRNO_EEXIST,
            ERRNO_EINVAL,
            ERRNO_ENOSPC,
            ERRNO_ENOSYS,
            ERRNO_EAUTH,
            ERRNO_EAUTHEXPIRED,
        ] {
            assert_ne!(error_string(code), "Unknown error", "code {code}");
        }
    }

    #[test]
    fn unknown_code_falls_back() {
        assert_eq!(error_string(-9999), "Unknown error");
        assert_eq!(error_string(0), "Unknown error");
        assert_eq!(error_string(42), "Unknown error");
    }

    #[test]
    fn known_numeric_values_match_kernel() {
        // These are the same constants the kernel exports — guard
        // against accidental drift.
        assert_eq!(ERRNO_EAGAIN, -11);
        assert_eq!(ERRNO_EACCES, -13);
        assert_eq!(ERRNO_EINVAL, -22);
        assert_eq!(ERRNO_EAUTHEXPIRED, -81);
    }

    #[test]
    fn synthesized_codes_are_unique_in_table() {
        // `EPERM` and `EAUTH` are picked so they don't collide with
        // any kernel-defined errno used in the auth subsystem.
        let kernel_codes: &[i64] = &[
            ERRNO_ENOENT,
            ERRNO_EAGAIN,
            ERRNO_EACCES,
            ERRNO_EFAULT,
            ERRNO_EEXIST,
            ERRNO_EINVAL,
            ERRNO_ENOSPC,
            ERRNO_ENOSYS,
            ERRNO_EAUTHEXPIRED,
        ];
        assert!(!kernel_codes.contains(&ERRNO_EPERM));
        assert!(!kernel_codes.contains(&ERRNO_EAUTH));
    }
}
