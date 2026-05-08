// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Three-role taxonomy: `Root`, `Operator`, `Viewer`.
//!
//! Spec: `openspec/changes/management-login-v1/specs/auth-roles/spec.md`.
//!
//! The role enum is a `#[repr(u8)]` enum so it can cross the syscall ABI as a
//! raw `u8`. Values 0..=2 are valid in v1; 3..=255 are reserved for future
//! roles and SHALL be rejected with `-EINVAL` per spec Requirement
//! "Three-role taxonomy".

use core::fmt;

/// Three-role taxonomy for SmallAIOS v1.
///
/// `Root` (0) — full access: account management, model load/unload,
/// system power, audit, all config namespaces.
///
/// `Operator` (1) — model lifecycle and read-only access to logs/metrics.
///
/// `Viewer` (2) — read-only telemetry. May not load models or change config.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Full access. The bootstrap account; created at first-boot.
    Root = 0,
    /// Model lifecycle + read-only audit/metrics.
    Operator = 1,
    /// Read-only telemetry.
    Viewer = 2,
}

/// Errors raised by [`Role`] decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleError {
    /// Wire-format byte does not correspond to a known role. Future roles
    /// may legitimately appear here on a downgrade — callers SHALL reject
    /// with `-EINVAL`.
    Unknown(u8),
    /// String form did not match `"root"`, `"operator"`, or `"viewer"`.
    UnknownName,
}

impl fmt::Display for RoleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(n) => write!(f, "auth: unknown role byte 0x{n:02x}"),
            Self::UnknownName => write!(f, "auth: unknown role name"),
        }
    }
}

impl Role {
    /// Decode a `u8` into a [`Role`]. Reject 3..=255 with
    /// [`RoleError::Unknown`].
    pub const fn from_u8(n: u8) -> Result<Self, RoleError> {
        match n {
            0 => Ok(Role::Root),
            1 => Ok(Role::Operator),
            2 => Ok(Role::Viewer),
            _ => Err(RoleError::Unknown(n)),
        }
    }

    /// Encode a [`Role`] back to its wire-format byte.
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Lower-case role name as stored in `/data/auth/shadow`'s `role=` field.
    pub const fn name(self) -> &'static str {
        match self {
            Role::Root => "root",
            Role::Operator => "operator",
            Role::Viewer => "viewer",
        }
    }

    /// Parse a role name as it appears in the shadow file (`root`,
    /// `operator`, `viewer`). Case-sensitive per spec.
    pub fn from_name(name: &str) -> Result<Self, RoleError> {
        match name {
            "root" => Ok(Role::Root),
            "operator" => Ok(Role::Operator),
            "viewer" => Ok(Role::Viewer),
            _ => Err(RoleError::UnknownName),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_u8_accepts_known_values() {
        assert_eq!(Role::from_u8(0), Ok(Role::Root));
        assert_eq!(Role::from_u8(1), Ok(Role::Operator));
        assert_eq!(Role::from_u8(2), Ok(Role::Viewer));
    }

    #[test]
    fn from_u8_rejects_unknown_values() {
        assert_eq!(Role::from_u8(3), Err(RoleError::Unknown(3)));
        assert_eq!(Role::from_u8(7), Err(RoleError::Unknown(7)));
        assert_eq!(Role::from_u8(255), Err(RoleError::Unknown(255)));
    }

    #[test]
    fn as_u8_round_trips_for_every_known_value() {
        for n in 0u8..=2 {
            let r = Role::from_u8(n).unwrap();
            assert_eq!(r.as_u8(), n);
        }
    }

    #[test]
    fn name_matches_shadow_field_convention() {
        assert_eq!(Role::Root.name(), "root");
        assert_eq!(Role::Operator.name(), "operator");
        assert_eq!(Role::Viewer.name(), "viewer");
    }

    #[test]
    fn from_name_accepts_canonical_strings() {
        assert_eq!(Role::from_name("root"), Ok(Role::Root));
        assert_eq!(Role::from_name("operator"), Ok(Role::Operator));
        assert_eq!(Role::from_name("viewer"), Ok(Role::Viewer));
    }

    #[test]
    fn from_name_is_case_sensitive() {
        assert_eq!(Role::from_name("Root"), Err(RoleError::UnknownName));
        assert_eq!(Role::from_name("ROOT"), Err(RoleError::UnknownName));
        assert_eq!(Role::from_name(""), Err(RoleError::UnknownName));
        assert_eq!(Role::from_name("admin"), Err(RoleError::UnknownName));
    }

    #[test]
    fn name_round_trip() {
        for r in [Role::Root, Role::Operator, Role::Viewer] {
            assert_eq!(Role::from_name(r.name()), Ok(r));
        }
    }
}
