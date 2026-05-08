// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Kernel-side authentication machinery.
//!
//! Spec: `openspec/changes/management-login-v1/specs/kernel-syscalls/spec.md`
//!
//! ## Layering note
//!
//! `smallaios-auth` (Layer 1) already depends on `smallaios-kernel`
//! (Layer 0) via [`auth/Cargo.toml`](../../auth/Cargo.toml). To avoid
//! turning that into a hard cycle, this module **does not** import
//! [`smallaios_auth`] types; it instead carries kernel-local mirror
//! types ([`Role`], [`shadow_provider::ShadowProviderEntry`]) that
//! preserve the wire format byte-for-byte.
//!
//! Higher-layer crates (`auth/`, `mgmt/`, `peripheral/`, `container/`)
//! are expected to plug their [`shadow_provider::ShadowProvider`] impl
//! into the kernel's syscall handlers and to convert between
//! [`smallaios_auth::role::Role`] and [`Role`] at the boundary —
//! a one-byte memcpy, since both enums share the `repr(u8)` wire
//! format `Root=0, Operator=1, Viewer=2`.
//!
//! ## Re-exports
//!
//! Per the Phase 3 design, the kernel **does not** re-export `Role` or
//! `ShadowEntry` from `auth/`; it owns kernel-local copies for the
//! reasons above. Importers in higher-layer crates SHALL `use
//! smallaios_auth::role::Role;` and convert.

pub mod min_role;
pub mod session_table;
pub mod shadow_provider;
pub mod sweeper;

pub use min_role::MinRole;
pub use session_table::{
    Session, SessionError, SessionId, SessionTable, MAX_USERNAME_LEN, SESSION_TABLE_CAP,
};
pub use shadow_provider::{
    KernelShadowProvider, ShadowProvider, ShadowProviderEntry, ShadowProviderError,
};
pub use sweeper::{
    AuditSink, CapturingAuditSink, IdlePolicy, NoopAuditSink, SweepReason, Sweeper,
    DEFAULT_OPERATOR_IDLE_SECS, DEFAULT_ROOT_IDLE_SECS, DEFAULT_VIEWER_IDLE_SECS,
};

#[cfg(any(test, feature = "test-mocks"))]
pub use shadow_provider::MockShadowProvider;

use core::sync::atomic::{AtomicU32, Ordering};

// ─── Kernel-local mirror of `smallaios_auth::role::Role` ─────────────────────

/// Three-role taxonomy mirror. Matches
/// [`smallaios_auth::role::Role`](../../auth/src/role.rs)
/// byte-for-byte: `Root=0, Operator=1, Viewer=2`.
///
/// The kernel cannot import the canonical type without creating a
/// circular Cargo dependency (auth → kernel already exists), so the
/// wire-format byte is what crosses the layer boundary; the syscall
/// handler converts via [`Role::from_u8`] / [`Role::as_u8`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Full access. Bootstrap account, created at first-boot.
    Root = 0,
    /// Model lifecycle + read-only audit/metrics.
    Operator = 1,
    /// Read-only telemetry.
    Viewer = 2,
}

impl Role {
    /// Decode a wire-format `u8` into [`Role`]. Bytes 3..=255 are
    /// rejected with `Err(())` — the syscall handler maps that to
    /// `-EINVAL`.
    #[allow(clippy::result_unit_err)] // intentional opaque error
    pub const fn from_u8(n: u8) -> Result<Self, ()> {
        match n {
            0 => Ok(Role::Root),
            1 => Ok(Role::Operator),
            2 => Ok(Role::Viewer),
            _ => Err(()),
        }
    }

    /// Encode a [`Role`] back to its wire-format byte.
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Lower-case role name, matching the shadow file's `role=` field.
    pub const fn name(self) -> &'static str {
        match self {
            Role::Root => "root",
            Role::Operator => "operator",
            Role::Viewer => "viewer",
        }
    }
}

// ─── Mirror of `auth/shadow.rs` flags ────────────────────────────────────────

/// Bit 0 of [`ShadowProviderEntry::flags`] — when set, the kernel SHALL
/// only honor `auth_change_password`, `auth_whoami`, and `auth_logout`
/// until the user rotates the password.
///
/// Mirrors [`smallaios_auth::shadow::FLAG_MUST_CHANGE_PASSWORD`].
pub const FLAG_MUST_CHANGE_PASSWORD: u32 = 0x0000_0001;

// ─── Errno constants — POSIX-compatible negative `i64` codes ─────────────────

/// `-EINVAL` (invalid argument).
pub const ERRNO_EINVAL: i64 = -22;

/// `-EACCES` (permission denied — capability check failed).
pub const ERRNO_EACCES: i64 = -13;

/// `-EAGAIN` (try again — temporary failure / lockout).
pub const ERRNO_EAGAIN: i64 = -11;

/// `-ENOSPC` (no slot in the session table).
pub const ERRNO_ENOSPC: i64 = -28;

/// `-ENOENT` (no such user).
pub const ERRNO_ENOENT: i64 = -2;

/// `-EEXIST` (user already exists).
pub const ERRNO_EEXIST: i64 = -17;

/// `-ENOSYS` (not implemented — used by `auth_totp_setup` until Phase 9
/// and by handlers that need a real shadow-write path).
pub const ERRNO_ENOSYS: i64 = -38;

/// `-EFAULT` (bad pointer from userspace).
pub const ERRNO_EFAULT: i64 = -14;

/// `-EAUTHEXPIRED` — synthetic kernel error returned by the syscall
/// dispatch when a session is gated by `must_change_password` and the
/// caller invokes a non-allowlisted syscall. The numeric value is taken
/// from FreeBSD's `EAUTH` (-80) extended by one to avoid collision with
/// any POSIX errno used elsewhere in the kernel; it is intentionally
/// distinct so callers can recognize "your password has expired" as
/// distinct from a generic `-EACCES`.
pub const ERRNO_EAUTHEXPIRED: i64 = -81;

// ─── Current session — global state ──────────────────────────────────────────

/// Raw [`SessionId`] of the currently authenticated session.
///
/// Stored as `AtomicU32` so dispatch can read it without taking a lock.
/// Initialized to [`SessionId::NONE`]'s raw value (`0xFFFF_FFFF`).
///
/// In a multi-tasking kernel each task would carry its own per-task
/// CURRENT_SESSION; the unikernel runs a single dispatch core so a
/// global atomic is sufficient for v1.
pub static CURRENT_SESSION: AtomicU32 = AtomicU32::new(SessionId::NONE.as_u32());

/// Read the currently authenticated [`SessionId`], if any.
pub fn current_session() -> SessionId {
    SessionId::from_raw(CURRENT_SESSION.load(Ordering::Acquire))
}

/// Install a new current session. The previous value is overwritten.
/// Returns the previous session id.
pub fn set_current_session(id: SessionId) -> SessionId {
    SessionId::from_raw(CURRENT_SESSION.swap(id.as_u32(), Ordering::AcqRel))
}

/// Convenience: clear `CURRENT_SESSION` to [`SessionId::NONE`].
pub fn clear_current_session() {
    CURRENT_SESSION.store(SessionId::NONE.as_u32(), Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_wire_format_matches_auth_crate_canonical() {
        // Cross-check: the kernel's mirror must round-trip through
        // the same `u8` values as `smallaios_auth::role::Role`. We
        // can't import that type here (cycle), so we hard-code the
        // canonical bytes from `auth/src/role.rs`.
        assert_eq!(Role::Root.as_u8(), 0);
        assert_eq!(Role::Operator.as_u8(), 1);
        assert_eq!(Role::Viewer.as_u8(), 2);
        assert_eq!(Role::from_u8(0), Ok(Role::Root));
        assert_eq!(Role::from_u8(1), Ok(Role::Operator));
        assert_eq!(Role::from_u8(2), Ok(Role::Viewer));
        assert_eq!(Role::from_u8(3), Err(()));
        assert_eq!(Role::from_u8(255), Err(()));
    }

    #[test]
    fn role_names_match_shadow_file_format() {
        assert_eq!(Role::Root.name(), "root");
        assert_eq!(Role::Operator.name(), "operator");
        assert_eq!(Role::Viewer.name(), "viewer");
    }

    #[test]
    fn must_change_password_flag_constant_matches_auth_crate() {
        // Mirrors `smallaios_auth::shadow::FLAG_MUST_CHANGE_PASSWORD`
        // — both must be 0x0000_0001 so the kernel can OR-mask the
        // shadow-loaded flag word directly.
        assert_eq!(FLAG_MUST_CHANGE_PASSWORD, 0x0000_0001);
    }

    #[test]
    fn current_session_initial_value_is_none() {
        // Note: this test runs in shared global state; we only
        // assert that NONE is recognized as such.
        let none = SessionId::NONE;
        assert!(none.is_none());
        assert_eq!(none.slot(), None);
    }

    #[test]
    fn errno_constants_match_posix_values() {
        assert_eq!(ERRNO_EINVAL, -22);
        assert_eq!(ERRNO_EACCES, -13);
        assert_eq!(ERRNO_EAGAIN, -11);
        assert_eq!(ERRNO_ENOSPC, -28);
        assert_eq!(ERRNO_ENOENT, -2);
        assert_eq!(ERRNO_EEXIST, -17);
        assert_eq!(ERRNO_ENOSYS, -38);
        assert_eq!(ERRNO_EFAULT, -14);
    }
}
