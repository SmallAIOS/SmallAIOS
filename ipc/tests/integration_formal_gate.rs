// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for IPC security label enforcement via the formal gate.
//!
//! Tests the SecurityGate's 5-layer verification pipeline from the IPC
//! crate's perspective, covering matched/mismatched labels in both
//! Enforcing and Permissive modes.

#![cfg(feature = "formal-gate")]

extern crate alloc;

use alloc::boxed::Box;

use smallaios_security::boundary::trust_boundaries::{DataFlowDirection, TrustBoundary};
use smallaios_security::capability::{CapRegistry, Permissions, ResourceRef, ResourceType};
use smallaios_security::compliance::classification::ClassificationLevel;
use smallaios_security::enforcement::{DenyReason, EnforcementMode, GateVerdict};
use smallaios_security::gate::{BoundaryCrossing, SecurityGate};
use smallaios_security::labels::{IntegrityLevel, MessageTypeId, SecurityLabel};
use smallaios_security::message_types::{default_registry, MessageMetadata, TensorDtype};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a capability registry granting task `task_id` READ on NetworkSocket 1.
fn setup_cap_registry(task_id: u64) -> Box<CapRegistry> {
    let mut reg = Box::new(CapRegistry::new());
    let resource = ResourceRef::new(ResourceType::NetworkSocket, 1);
    let _ = reg.create(task_id, resource, Permissions::READ, 0);
    reg
}

/// Valid metadata for InferenceTensorInput (type 0x0001): rank-4 float32 tensor.
fn valid_tensor_metadata() -> MessageMetadata {
    let mut meta = MessageMetadata::empty();
    meta.rank = 4;
    meta.dtype = TensorDtype::Float32 as u8;
    meta.element_count = 1 * 3 * 224 * 224;
    meta.payload_bytes = meta.element_count * 4;
    meta.num_dimensions = 4;
    meta.dimensions = [1, 3, 224, 224, 0, 0, 0, 0];
    meta
}

/// Build a BoundaryCrossing with matched labels (Internal classification,
/// Medium integrity) flowing into a Restricted/Medium destination.
fn matched_crossing(metadata: &MessageMetadata) -> BoundaryCrossing<'_> {
    BoundaryCrossing {
        boundary: TrustBoundary::Network,
        direction: DataFlowDirection::Inbound,
        task_id: 1,
        label: SecurityLabel::new(
            ClassificationLevel::Internal,
            IntegrityLevel::Medium,
            Some(MessageTypeId(0x0001)),
        ),
        resource: ResourceRef::new(ResourceType::NetworkSocket, 1),
        required_permissions: Permissions::READ,
        metadata,
        dest_integrity: IntegrityLevel::Medium,
        dest_max_classification: ClassificationLevel::Restricted,
        now: 0,
    }
}

// ---------------------------------------------------------------------------
// 1. Enforcing mode — matched labels allow delivery
// ---------------------------------------------------------------------------

#[test]
fn enforcing_matched_labels_allowed() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Enforcing);

    let meta = valid_tensor_metadata();
    let crossing = matched_crossing(&meta);

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(
        verdict.is_allowed(),
        "matched labels in Enforcing mode must be allowed"
    );
    assert_eq!(verdict, GateVerdict::Allowed);
    assert_eq!(gate.total_allowed(), 1);
    assert_eq!(gate.total_denied(), 0);
}

#[test]
fn enforcing_matched_labels_high_to_low_integrity_allowed() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Enforcing);

    let meta = valid_tensor_metadata();
    // High integrity flowing to Low destination is fine (Biba: write-down OK).
    let crossing = BoundaryCrossing {
        label: SecurityLabel::new(
            ClassificationLevel::Internal,
            IntegrityLevel::High,
            Some(MessageTypeId(0x0001)),
        ),
        dest_integrity: IntegrityLevel::Low,
        ..matched_crossing(&meta)
    };

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(verdict.is_allowed());
}

// ---------------------------------------------------------------------------
// 2. Enforcing mode — integrity violation rejects
// ---------------------------------------------------------------------------

#[test]
fn enforcing_integrity_violation_denied() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
    // Set boundary to Enforcing so it takes effect for untyped messages.
    gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);

    let meta = valid_tensor_metadata();
    // Low integrity data trying to flow to High destination (Biba violation).
    // Use None message type so global/boundary Enforcing mode applies
    // (typed messages in the default registry have Permissive per-type mode).
    let crossing = BoundaryCrossing {
        label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Low, None),
        dest_integrity: IntegrityLevel::High,
        ..matched_crossing(&meta)
    };

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(verdict.is_denied(), "integrity violation must be denied");
    assert_eq!(
        verdict,
        GateVerdict::Denied {
            layer: 3,
            reason: DenyReason::IntegrityViolation,
        }
    );
    assert_eq!(gate.total_denied(), 1);
}

#[test]
fn enforcing_low_to_medium_integrity_denied() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
    gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);

    let meta = valid_tensor_metadata();
    let crossing = BoundaryCrossing {
        label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Low, None),
        dest_integrity: IntegrityLevel::Medium,
        ..matched_crossing(&meta)
    };

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(verdict.is_denied());
    if let GateVerdict::Denied { layer, reason } = verdict {
        assert_eq!(layer, 3);
        assert_eq!(reason, DenyReason::IntegrityViolation);
    } else {
        panic!("expected Denied verdict");
    }
}

// ---------------------------------------------------------------------------
// 3. Enforcing mode — classification violation rejects
// ---------------------------------------------------------------------------

#[test]
fn enforcing_classification_violation_denied() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
    gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);

    let meta = valid_tensor_metadata();
    // Restricted data flowing to a Public destination (Bell-LaPadula: no write-down).
    let crossing = BoundaryCrossing {
        label: SecurityLabel::new(
            ClassificationLevel::Restricted,
            IntegrityLevel::Medium,
            None,
        ),
        dest_max_classification: ClassificationLevel::Public,
        ..matched_crossing(&meta)
    };

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(
        verdict.is_denied(),
        "classification violation must be denied"
    );
    assert_eq!(
        verdict,
        GateVerdict::Denied {
            layer: 2,
            reason: DenyReason::ClassificationViolation,
        }
    );
}

#[test]
fn enforcing_restricted_to_internal_classification_denied() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
    gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);

    let meta = valid_tensor_metadata();
    let crossing = BoundaryCrossing {
        label: SecurityLabel::new(
            ClassificationLevel::Restricted,
            IntegrityLevel::Medium,
            None,
        ),
        dest_max_classification: ClassificationLevel::Internal,
        ..matched_crossing(&meta)
    };

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(verdict.is_denied());
    if let GateVerdict::Denied { layer, reason } = verdict {
        assert_eq!(layer, 2);
        assert_eq!(reason, DenyReason::ClassificationViolation);
    } else {
        panic!("expected Denied verdict");
    }
}

// ---------------------------------------------------------------------------
// 4. Permissive mode — violations are logged but allowed
// ---------------------------------------------------------------------------

#[test]
fn permissive_integrity_violation_passes() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Permissive);

    let meta = valid_tensor_metadata();
    // Same Biba violation as the Enforcing test, but Permissive mode.
    let crossing = BoundaryCrossing {
        label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Low, None),
        dest_integrity: IntegrityLevel::High,
        ..matched_crossing(&meta)
    };

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(
        verdict.is_allowed(),
        "permissive mode must allow even on violation"
    );
    assert!(
        verdict.is_permissive_pass(),
        "should be PermissivePass, not clean Allowed"
    );
    if let GateVerdict::PermissivePass { layer, reason } = verdict {
        assert_eq!(layer, 3);
        assert_eq!(reason, DenyReason::IntegrityViolation);
    } else {
        panic!("expected PermissivePass verdict");
    }
    assert_eq!(gate.total_permissive_passes(), 1);
    assert_eq!(gate.total_denied(), 0);
}

#[test]
fn permissive_classification_violation_passes() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Permissive);

    let meta = valid_tensor_metadata();
    let crossing = BoundaryCrossing {
        label: SecurityLabel::new(
            ClassificationLevel::Restricted,
            IntegrityLevel::Medium,
            None,
        ),
        dest_max_classification: ClassificationLevel::Public,
        ..matched_crossing(&meta)
    };

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(verdict.is_allowed());
    assert!(verdict.is_permissive_pass());
    if let GateVerdict::PermissivePass { layer, reason } = verdict {
        assert_eq!(layer, 2);
        assert_eq!(reason, DenyReason::ClassificationViolation);
    }
}

#[test]
fn permissive_unknown_type_passes() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Permissive);

    let meta = valid_tensor_metadata();
    let crossing = BoundaryCrossing {
        label: SecurityLabel::new(
            ClassificationLevel::Internal,
            IntegrityLevel::Medium,
            Some(MessageTypeId(0xBEEF)), // not in registry
        ),
        ..matched_crossing(&meta)
    };

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(verdict.is_allowed());
    assert!(verdict.is_permissive_pass());
}

// ---------------------------------------------------------------------------
// 5. Mixed scenarios — audit and statistics
// ---------------------------------------------------------------------------

#[test]
fn mixed_flow_statistics_and_audit() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();
    let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
    gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);

    let meta = valid_tensor_metadata();

    // Check 1: matched labels → Allowed
    let c1 = matched_crossing(&meta);
    let v1 = gate.check(&c1, &cap_reg, &type_reg);
    assert!(v1.is_allowed());

    // Check 2: integrity violation (untyped, Enforcing) → Denied
    let c2 = BoundaryCrossing {
        label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Low, None),
        dest_integrity: IntegrityLevel::High,
        ..matched_crossing(&meta)
    };
    let v2 = gate.check(&c2, &cap_reg, &type_reg);
    assert!(v2.is_denied());

    // Check 3: classification violation (untyped, Enforcing) → Denied
    let c3 = BoundaryCrossing {
        label: SecurityLabel::new(
            ClassificationLevel::Restricted,
            IntegrityLevel::Medium,
            None,
        ),
        dest_max_classification: ClassificationLevel::Public,
        ..matched_crossing(&meta)
    };
    let v3 = gate.check(&c3, &cap_reg, &type_reg);
    assert!(v3.is_denied());

    assert_eq!(gate.total_checks(), 3);
    assert_eq!(gate.total_allowed(), 1);
    assert_eq!(gate.total_denied(), 2);
    assert_eq!(gate.total_permissive_passes(), 0);
    assert_eq!(gate.audit_count(), 3);
}

#[test]
fn per_boundary_mode_override() {
    let cap_reg = setup_cap_registry(1);
    let type_reg = default_registry();

    // Global is Permissive, but Network boundary is Enforcing.
    let mut gate = SecurityGate::new(EnforcementMode::Permissive);
    gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);

    let meta = valid_tensor_metadata();
    // Unknown type on an Enforcing boundary → Denied (boundary overrides global).
    let crossing = BoundaryCrossing {
        label: SecurityLabel::new(
            ClassificationLevel::Internal,
            IntegrityLevel::Medium,
            Some(MessageTypeId(0xDEAD)),
        ),
        ..matched_crossing(&meta)
    };

    let verdict = gate.check(&crossing, &cap_reg, &type_reg);
    assert!(verdict.is_denied());
    if let GateVerdict::Denied { layer, reason } = verdict {
        assert_eq!(layer, 4);
        assert_eq!(reason, DenyReason::UnknownMessageType);
    }
}
