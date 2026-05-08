// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-syscall minimum-role declarations.
//!
//! Spec: `openspec/changes/management-login-v1/specs/auth-roles/spec.md`
//! Requirement "Per-syscall minimum-role table".
//!
//! [`MinRole`] is a four-rung lattice that classifies the *capability
//! gate* applied to each syscall before it is dispatched to its handler:
//!
//! | Rung              | Admit predicate                                   |
//! |-------------------|---------------------------------------------------|
//! | `Unauthenticated` | Always                                            |
//! | `Authenticated`   | Caller holds a session (any role)                 |
//! | `Operator`        | Caller's role is `Operator` or `Root`             |
//! | `Root`            | Caller's role is `Root`                           |
//!
//! The kernel-local [`Role`](super::Role) mirrors
//! [`smallaios_auth::role::Role`] byte-for-byte (`Root=0, Operator=1,
//! Viewer=2`), so the same wire format is shared on the syscall ABI.

use super::Role;

/// Minimum-role rungs admitted by a syscall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinRole {
    /// Anyone may call the syscall, including before login (e.g.
    /// `auth_login`, `sys_info`, `sys_time`).
    Unauthenticated,
    /// Caller must hold a session of any role.
    Authenticated,
    /// Caller must be `Operator` or `Root`.
    Operator,
    /// Caller must be `Root`.
    Root,
}

impl MinRole {
    /// Returns `true` iff a caller with the given (optional) role is
    /// admitted under this minimum-role rung.
    ///
    /// `None` represents the *unauthenticated* caller (no session
    /// installed). All four rungs treat `None` consistently:
    ///
    /// - `Unauthenticated` → always admitted
    /// - `Authenticated`/`Operator`/`Root` → reject `None`
    pub const fn admits(self, caller: Option<Role>) -> bool {
        match (self, caller) {
            (Self::Unauthenticated, _) => true,
            (Self::Authenticated, Some(_)) => true,
            (Self::Authenticated, None) => false,
            (Self::Operator, Some(Role::Root)) => true,
            (Self::Operator, Some(Role::Operator)) => true,
            (Self::Operator, _) => false,
            (Self::Root, Some(Role::Root)) => true,
            (Self::Root, _) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unauthenticated_admits_everyone_including_no_session() {
        assert!(MinRole::Unauthenticated.admits(None));
        assert!(MinRole::Unauthenticated.admits(Some(Role::Root)));
        assert!(MinRole::Unauthenticated.admits(Some(Role::Operator)));
        assert!(MinRole::Unauthenticated.admits(Some(Role::Viewer)));
    }

    #[test]
    fn authenticated_rejects_no_session_admits_all_roles() {
        assert!(!MinRole::Authenticated.admits(None));
        assert!(MinRole::Authenticated.admits(Some(Role::Root)));
        assert!(MinRole::Authenticated.admits(Some(Role::Operator)));
        assert!(MinRole::Authenticated.admits(Some(Role::Viewer)));
    }

    #[test]
    fn operator_admits_root_and_operator_rejects_viewer_and_no_session() {
        assert!(!MinRole::Operator.admits(None));
        assert!(MinRole::Operator.admits(Some(Role::Root)));
        assert!(MinRole::Operator.admits(Some(Role::Operator)));
        assert!(!MinRole::Operator.admits(Some(Role::Viewer)));
    }

    #[test]
    fn root_admits_only_root() {
        assert!(!MinRole::Root.admits(None));
        assert!(MinRole::Root.admits(Some(Role::Root)));
        assert!(!MinRole::Root.admits(Some(Role::Operator)));
        assert!(!MinRole::Root.admits(Some(Role::Viewer)));
    }
}
