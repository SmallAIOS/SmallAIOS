// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SecurityGate: 5-layer verification pipeline at trust boundaries.
//!
//! Every data crossing a trust boundary passes through the gate:
//! 1. Capability check — does the task hold the required capability?
//! 2. Classification check — does the data classification permit this flow?
//! 3. Integrity check — does the Biba model allow this flow direction?
//! 4. Message type check — does the data match a verified type with valid invariants?
//! 5. Enforcement mode — Enforcing (hard deny) or Permissive (log and allow)?

use crate::boundary::trust_boundaries::{DataFlowDirection, TrustBoundary};
use crate::capability::{CapRegistry, Permissions, ResourceRef, TaskId};
use crate::compliance::classification::ClassificationLevel;
use crate::enforcement::{resolve_mode, DenyReason, EnforcementMode, GateVerdict};
use crate::labels::{IntegrityLevel, MessageTypeId, SecurityLabel};
use crate::message_types::{InvariantChecker, MessageMetadata, MessageTypeRegistry};

/// Number of trust boundaries (for per-boundary mode array).
const NUM_BOUNDARIES: usize = 5;

/// A data crossing at a trust boundary, to be checked by the gate.
#[derive(Debug)]
pub struct BoundaryCrossing<'a> {
    /// Which trust boundary is being crossed.
    pub boundary: TrustBoundary,
    /// Direction of data flow.
    pub direction: DataFlowDirection,
    /// Task attempting the crossing.
    pub task_id: TaskId,
    /// Security label on the data.
    pub label: SecurityLabel,
    /// The resource being accessed (for capability check).
    pub resource: ResourceRef,
    /// Required permissions on the resource.
    pub required_permissions: Permissions,
    /// Message metadata for invariant checking.
    pub metadata: &'a MessageMetadata,
    /// Destination integrity level (for Biba check).
    pub dest_integrity: IntegrityLevel,
    /// Destination maximum classification level.
    pub dest_max_classification: ClassificationLevel,
    /// Current time (for capability expiry checks).
    pub now: u64,
}

/// The formal methods firewall gate.
pub struct SecurityGate {
    /// Per-boundary enforcement modes.
    boundary_modes: [EnforcementMode; NUM_BOUNDARIES],
    /// Global default enforcement mode.
    global_mode: EnforcementMode,
    /// Invariant checker (holds per-type stateful tracking).
    invariant_checker: InvariantChecker,
    /// Statistics.
    total_checks: u64,
    total_allowed: u64,
    total_denied: u64,
    total_permissive_passes: u64,
}

impl SecurityGate {
    /// Create a new gate with the given global mode.
    pub fn new(global_mode: EnforcementMode) -> Self {
        Self {
            boundary_modes: [global_mode; NUM_BOUNDARIES],
            global_mode,
            invariant_checker: InvariantChecker::new(),
            total_checks: 0,
            total_allowed: 0,
            total_denied: 0,
            total_permissive_passes: 0,
        }
    }

    /// Set the enforcement mode for a specific boundary.
    pub fn set_boundary_mode(&mut self, boundary: TrustBoundary, mode: EnforcementMode) {
        let idx = boundary as usize;
        if idx < NUM_BOUNDARIES {
            self.boundary_modes[idx] = mode;
        }
    }

    /// Get the enforcement mode for a specific boundary.
    pub fn boundary_mode(&self, boundary: TrustBoundary) -> EnforcementMode {
        let idx = boundary as usize;
        if idx < NUM_BOUNDARIES {
            self.boundary_modes[idx]
        } else {
            self.global_mode
        }
    }

    /// Run the 5-layer verification pipeline.
    ///
    /// Layers:
    /// 1. Capability check
    /// 2. Classification check (Bell-LaPadula)
    /// 3. Integrity check (Biba)
    /// 4. Message type + invariant check
    /// 5. Enforcement mode resolution → verdict
    pub fn check(
        &mut self,
        crossing: &BoundaryCrossing,
        cap_registry: &CapRegistry,
        type_registry: &MessageTypeRegistry,
    ) -> GateVerdict {
        self.total_checks += 1;

        // Layer 1: Capability check
        if cap_registry
            .check(
                crossing.task_id,
                &crossing.resource,
                crossing.required_permissions,
                crossing.now,
            )
            .is_err()
        {
            return self.make_verdict(crossing, 1, DenyReason::MissingCapability, type_registry);
        }

        // Layer 2: Classification check (Bell-LaPadula — no write-down)
        if crossing.label.classification > crossing.dest_max_classification {
            return self.make_verdict(
                crossing,
                2,
                DenyReason::ClassificationViolation,
                type_registry,
            );
        }

        // Layer 3: Integrity check (Biba — no write-up)
        if !crossing.label.integrity.may_flow_to(crossing.dest_integrity) {
            return self.make_verdict(crossing, 3, DenyReason::IntegrityViolation, type_registry);
        }

        // Layer 4: Message type + invariant check
        if let Some(type_id) = crossing.label.message_type {
            if let Some(msg_type) = type_registry.lookup(type_id) {
                if let Some(inv_idx) = self.invariant_checker.check(msg_type, crossing.metadata) {
                    return self.make_verdict(
                        crossing,
                        4,
                        DenyReason::InvariantFailed(inv_idx),
                        type_registry,
                    );
                }
            } else {
                return self.make_verdict(
                    crossing,
                    4,
                    DenyReason::UnknownMessageType,
                    type_registry,
                );
            }
        }
        // Untyped messages (message_type = None) skip layer 4 —
        // enforcement mode determines if that's OK.

        // All layers passed
        self.total_allowed += 1;
        GateVerdict::Allowed
    }

    /// Resolve enforcement mode and produce verdict for a failure.
    fn make_verdict(
        &mut self,
        crossing: &BoundaryCrossing,
        layer: u8,
        reason: DenyReason,
        type_registry: &MessageTypeRegistry,
    ) -> GateVerdict {
        // Resolve mode: type > boundary > global
        let type_mode = crossing
            .label
            .message_type
            .and_then(|id| type_registry.lookup(id))
            .map(|mt| mt.mode);
        let boundary_mode = self.boundary_mode(crossing.boundary);
        let mode = resolve_mode(type_mode, boundary_mode, self.global_mode);

        match mode {
            EnforcementMode::Enforcing => {
                self.total_denied += 1;
                GateVerdict::Denied { layer, reason }
            }
            EnforcementMode::Permissive => {
                self.total_permissive_passes += 1;
                GateVerdict::PermissivePass { layer, reason }
            }
        }
    }

    /// Reset rate limit window for a message type.
    pub fn reset_rate_window(&mut self, type_id: MessageTypeId) {
        self.invariant_checker.reset_rate_window(type_id);
    }

    // ── Statistics ──

    pub fn total_checks(&self) -> u64 {
        self.total_checks
    }

    pub fn total_allowed(&self) -> u64 {
        self.total_allowed
    }

    pub fn total_denied(&self) -> u64 {
        self.total_denied
    }

    pub fn total_permissive_passes(&self) -> u64 {
        self.total_permissive_passes
    }
}

impl Default for SecurityGate {
    fn default() -> Self {
        Self::new(EnforcementMode::Permissive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use crate::capability::{CapRegistry, ResourceRef, ResourceType};
    use crate::message_types::{default_registry, MessageMetadata, TensorDtype};

    /// Helper: create a cap registry with a pre-granted capability for task 1.
    fn setup_cap_registry() -> Box<CapRegistry> {
        let mut reg = Box::new(CapRegistry::new());
        let resource = ResourceRef::new(ResourceType::NetworkSocket, 1);
        let _ = reg.create(1, resource, Permissions::READ, 0);
        reg
    }

    fn base_crossing(metadata: &MessageMetadata) -> BoundaryCrossing {
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

    // ── Layer 1: Capability ──

    #[test]
    fn layer1_capability_pass() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let mut meta = MessageMetadata::empty();
        meta.rank = 2;
        meta.dtype = TensorDtype::Float32 as u8;
        meta.element_count = 100;
        meta.num_dimensions = 2;
        meta.dimensions = [10, 10, 0, 0, 0, 0, 0, 0];

        let crossing = base_crossing(&meta);
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert!(verdict.is_allowed());
    }

    #[test]
    fn layer1_capability_denied() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let meta = MessageMetadata::empty();

        // Task 99 has no capabilities; use untyped so global Enforcing mode applies
        let crossing = BoundaryCrossing {
            task_id: 99,
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                None,
            ),
            ..base_crossing(&meta)
        };
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert_eq!(
            verdict,
            GateVerdict::Denied {
                layer: 1,
                reason: DenyReason::MissingCapability
            }
        );
    }

    // ── Layer 2: Classification ──

    #[test]
    fn layer2_classification_denied() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let meta = MessageMetadata::empty();

        // Restricted data trying to flow to Public destination; untyped for Enforcing
        let crossing = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Restricted,
                IntegrityLevel::Medium,
                None,
            ),
            dest_max_classification: ClassificationLevel::Public,
            ..base_crossing(&meta)
        };
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert_eq!(
            verdict,
            GateVerdict::Denied {
                layer: 2,
                reason: DenyReason::ClassificationViolation
            }
        );
    }

    // ── Layer 3: Integrity (Biba) ──

    #[test]
    fn layer3_integrity_denied() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let meta = MessageMetadata::empty();

        // Low integrity to High dest; untyped for Enforcing
        let crossing = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Low,
                None,
            ),
            dest_integrity: IntegrityLevel::High,
            ..base_crossing(&meta)
        };
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert_eq!(
            verdict,
            GateVerdict::Denied {
                layer: 3,
                reason: DenyReason::IntegrityViolation
            }
        );
    }

    #[test]
    fn layer3_integrity_pass_high_to_low() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let mut meta = MessageMetadata::empty();
        meta.rank = 2;
        meta.dtype = TensorDtype::Float32 as u8;
        meta.element_count = 100;
        meta.num_dimensions = 2;
        meta.dimensions = [10, 10, 0, 0, 0, 0, 0, 0];

        let crossing = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::High,
                Some(MessageTypeId(0x0001)),
            ),
            dest_integrity: IntegrityLevel::Low,
            ..base_crossing(&meta)
        };
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert!(verdict.is_allowed());
    }

    // ── Layer 4: Message type ──

    #[test]
    fn layer4_unknown_type_denied() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let meta = MessageMetadata::empty();

        let crossing = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                Some(MessageTypeId(0xFFFF)), // not in registry
            ),
            ..base_crossing(&meta)
        };
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert_eq!(
            verdict,
            GateVerdict::Denied {
                layer: 4,
                reason: DenyReason::UnknownMessageType
            }
        );
    }

    #[test]
    fn layer4_invariant_failed() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);

        // InferenceTensorInput (0x0001) has Permissive mode in default catalog,
        // so an invariant failure results in PermissivePass (not Denied).
        let mut meta = MessageMetadata::empty();
        meta.rank = 0; // fails MinRank(1)
        meta.dtype = TensorDtype::Float32 as u8;

        let crossing = base_crossing(&meta);
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        // Type mode (Permissive) takes priority over global (Enforcing)
        assert!(verdict.is_permissive_pass());
        if let GateVerdict::PermissivePass { layer, reason } = verdict {
            assert_eq!(layer, 4);
            assert!(matches!(reason, DenyReason::InvariantFailed(_)));
        }
    }

    #[test]
    fn untyped_message_passes_layer4() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let meta = MessageMetadata::empty();

        // No message type — skips layer 4
        let crossing = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                None,
            ),
            ..base_crossing(&meta)
        };
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert!(verdict.is_allowed());
    }

    // ── Enforcement mode ──

    #[test]
    fn permissive_mode_passes_on_failure() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Permissive);
        let meta = MessageMetadata::empty();

        // Unknown type, but Permissive mode → PermissivePass
        let crossing = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                Some(MessageTypeId(0xFFFF)),
            ),
            ..base_crossing(&meta)
        };
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert!(verdict.is_permissive_pass());
        assert!(verdict.is_allowed());
    }

    // ── Mode resolution ──

    #[test]
    fn per_boundary_mode_overrides_global() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Permissive);
        gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);
        let meta = MessageMetadata::empty();

        let crossing = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                Some(MessageTypeId(0xFFFF)),
            ),
            ..base_crossing(&meta)
        };
        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        // Network boundary is Enforcing even though global is Permissive
        assert!(verdict.is_denied());
    }

    // ── Statistics ──

    #[test]
    fn statistics_tracking() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);

        let mut meta = MessageMetadata::empty();
        meta.rank = 2;
        meta.dtype = TensorDtype::Float32 as u8;
        meta.element_count = 100;
        meta.num_dimensions = 2;
        meta.dimensions = [10, 10, 0, 0, 0, 0, 0, 0];

        // 1 allowed
        let crossing = base_crossing(&meta);
        gate.check(&crossing, &cap_reg, &type_reg);

        // 1 denied (bad task, untyped so global Enforcing applies)
        let crossing2 = BoundaryCrossing {
            task_id: 99,
            label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Medium, None),
            ..base_crossing(&meta)
        };
        gate.check(&crossing2, &cap_reg, &type_reg);

        assert_eq!(gate.total_checks(), 2);
        assert_eq!(gate.total_allowed(), 1);
        assert_eq!(gate.total_denied(), 1);
        assert_eq!(gate.total_permissive_passes(), 0);
    }

    #[test]
    fn boundary_mode_accessor() {
        let mut gate = SecurityGate::new(EnforcementMode::Permissive);
        assert_eq!(
            gate.boundary_mode(TrustBoundary::Network),
            EnforcementMode::Permissive
        );
        gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);
        assert_eq!(
            gate.boundary_mode(TrustBoundary::Network),
            EnforcementMode::Enforcing
        );
        // Other boundaries unchanged
        assert_eq!(
            gate.boundary_mode(TrustBoundary::Gpu),
            EnforcementMode::Permissive
        );
    }
}
