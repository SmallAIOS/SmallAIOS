// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Enforcement modes and gate verdicts for the formal methods firewall.
//!
//! Controls how the SecurityGate responds to verification failures:
//! - Enforcing: hard reject, data does not cross the boundary
//! - Permissive: log the violation, data flows through
//!
//! Modes are resolved hierarchically: per-type > per-boundary > global.

/// Enforcement mode for the security gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EnforcementMode {
    /// Hard reject on verification failure. Data does not cross the boundary.
    Enforcing = 0,
    /// Log the violation but allow data to flow through.
    Permissive = 1,
}

impl EnforcementMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Enforcing => "enforcing",
            Self::Permissive => "permissive",
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Enforcing),
            1 => Some(Self::Permissive),
            _ => None,
        }
    }

    /// Whether this mode blocks on failure.
    pub fn is_enforcing(self) -> bool {
        matches!(self, Self::Enforcing)
    }
}

/// Reason a gate check denied (or would deny) a crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenyReason {
    /// Task lacks required capability for the resource.
    MissingCapability,
    /// Data classification exceeds destination maximum.
    ClassificationViolation,
    /// Data integrity level too low for destination (Biba).
    IntegrityViolation,
    /// Message type ID not found in the registry.
    UnknownMessageType,
    /// A specific invariant failed (index into type's invariant array).
    InvariantFailed(u8),
    /// Model hash not present in policy whitelist.
    ModelNotWhitelisted,
    /// Policy blob signature verification failed.
    PolicySignatureInvalid,
}

impl DenyReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MissingCapability => "missing-capability",
            Self::ClassificationViolation => "classification-violation",
            Self::IntegrityViolation => "integrity-violation",
            Self::UnknownMessageType => "unknown-message-type",
            Self::InvariantFailed(_) => "invariant-failed",
            Self::ModelNotWhitelisted => "model-not-whitelisted",
            Self::PolicySignatureInvalid => "policy-signature-invalid",
        }
    }
}

/// Verdict from a SecurityGate check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    /// All verification layers passed. Data may cross the boundary.
    Allowed,
    /// A verification layer failed and mode is Enforcing. Data is blocked.
    Denied {
        /// The verification layer that failed (1-indexed: 1=cap, 2=class, 3=integrity, 4=type).
        layer: u8,
        /// Why the check failed.
        reason: DenyReason,
    },
    /// A verification layer failed but mode is Permissive. Data flows through.
    PermissivePass {
        /// The verification layer that would have denied.
        layer: u8,
        /// Why it would have been denied.
        reason: DenyReason,
    },
}

impl GateVerdict {
    /// Whether data is allowed to cross the boundary.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed | Self::PermissivePass { .. })
    }

    /// Whether the verdict is a hard denial.
    pub fn is_denied(&self) -> bool {
        matches!(self, Self::Denied { .. })
    }

    /// Whether this was a permissive pass (would have been denied).
    pub fn is_permissive_pass(&self) -> bool {
        matches!(self, Self::PermissivePass { .. })
    }
}

/// Resolve enforcement mode hierarchically: type > boundary > global.
///
/// - `type_mode`: mode from the VerifiedMessageType (None if untyped)
/// - `boundary_mode`: mode from the BoundaryDefinition
/// - `global_mode`: system-wide default
pub fn resolve_mode(
    type_mode: Option<EnforcementMode>,
    boundary_mode: EnforcementMode,
    _global_mode: EnforcementMode,
) -> EnforcementMode {
    // Type mode takes highest priority, then boundary, then global.
    // boundary_mode already incorporates the global default (set at policy load),
    // so we only need to check type_mode vs boundary_mode here.
    if let Some(tm) = type_mode {
        tm
    } else {
        boundary_mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── EnforcementMode ──

    #[test]
    fn enforcement_mode_as_str() {
        assert_eq!(EnforcementMode::Enforcing.as_str(), "enforcing");
        assert_eq!(EnforcementMode::Permissive.as_str(), "permissive");
    }

    #[test]
    fn enforcement_mode_from_u8() {
        assert_eq!(
            EnforcementMode::from_u8(0),
            Some(EnforcementMode::Enforcing)
        );
        assert_eq!(
            EnforcementMode::from_u8(1),
            Some(EnforcementMode::Permissive)
        );
        assert_eq!(EnforcementMode::from_u8(2), None);
    }

    #[test]
    fn enforcement_mode_is_enforcing() {
        assert!(EnforcementMode::Enforcing.is_enforcing());
        assert!(!EnforcementMode::Permissive.is_enforcing());
    }

    // ── DenyReason ──

    #[test]
    fn deny_reason_strings() {
        assert_eq!(DenyReason::MissingCapability.as_str(), "missing-capability");
        assert_eq!(
            DenyReason::ClassificationViolation.as_str(),
            "classification-violation"
        );
        assert_eq!(
            DenyReason::IntegrityViolation.as_str(),
            "integrity-violation"
        );
        assert_eq!(
            DenyReason::UnknownMessageType.as_str(),
            "unknown-message-type"
        );
        assert_eq!(DenyReason::InvariantFailed(3).as_str(), "invariant-failed");
        assert_eq!(
            DenyReason::ModelNotWhitelisted.as_str(),
            "model-not-whitelisted"
        );
        assert_eq!(
            DenyReason::PolicySignatureInvalid.as_str(),
            "policy-signature-invalid"
        );
    }

    // ── GateVerdict ──

    #[test]
    fn verdict_allowed() {
        let v = GateVerdict::Allowed;
        assert!(v.is_allowed());
        assert!(!v.is_denied());
        assert!(!v.is_permissive_pass());
    }

    #[test]
    fn verdict_denied() {
        let v = GateVerdict::Denied {
            layer: 3,
            reason: DenyReason::IntegrityViolation,
        };
        assert!(!v.is_allowed());
        assert!(v.is_denied());
        assert!(!v.is_permissive_pass());
    }

    #[test]
    fn verdict_permissive_pass() {
        let v = GateVerdict::PermissivePass {
            layer: 4,
            reason: DenyReason::UnknownMessageType,
        };
        assert!(v.is_allowed());
        assert!(!v.is_denied());
        assert!(v.is_permissive_pass());
    }

    #[test]
    fn verdict_denied_captures_layer_and_reason() {
        let v = GateVerdict::Denied {
            layer: 2,
            reason: DenyReason::ClassificationViolation,
        };
        if let GateVerdict::Denied { layer, reason } = v {
            assert_eq!(layer, 2);
            assert_eq!(reason, DenyReason::ClassificationViolation);
        } else {
            panic!("expected Denied");
        }
    }

    // ── Mode resolution ──

    #[test]
    fn resolve_type_overrides_boundary() {
        let mode = resolve_mode(
            Some(EnforcementMode::Enforcing),
            EnforcementMode::Permissive,
            EnforcementMode::Permissive,
        );
        assert_eq!(mode, EnforcementMode::Enforcing);
    }

    #[test]
    fn resolve_boundary_when_no_type() {
        let mode = resolve_mode(
            None,
            EnforcementMode::Enforcing,
            EnforcementMode::Permissive,
        );
        assert_eq!(mode, EnforcementMode::Enforcing);
    }

    #[test]
    fn resolve_permissive_type_overrides_enforcing_boundary() {
        let mode = resolve_mode(
            Some(EnforcementMode::Permissive),
            EnforcementMode::Enforcing,
            EnforcementMode::Enforcing,
        );
        assert_eq!(mode, EnforcementMode::Permissive);
    }
}
