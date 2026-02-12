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

use crate::audit::entry::{AuditEntry, AuditResult, Operation};
use crate::audit::ring_buffer::AuditRingBuffer;
use crate::audit::taxonomy::AuditEventType;
use crate::boundary::trust_boundaries::{DataFlowDirection, TrustBoundary};
use crate::capability::{CapRegistry, Permissions, ResourceRef, TaskId};
use crate::compliance::classification::ClassificationLevel;
use crate::enforcement::{resolve_mode, DenyReason, EnforcementMode, GateVerdict};
use crate::labels::{IntegrityLevel, MessageTypeId, SecurityLabel};
use crate::message_types::{InvariantChecker, MessageMetadata, MessageTypeRegistry};

/// Number of trust boundaries (for per-boundary mode array).
const NUM_BOUNDARIES: usize = 5;

/// Capacity of the gate's internal audit ring buffer.
const GATE_AUDIT_CAPACITY: usize = 256;

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
    /// Audit ring buffer for gate decisions.
    audit_buffer: AuditRingBuffer<GATE_AUDIT_CAPACITY>,
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
            audit_buffer: AuditRingBuffer::new(),
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
        if !crossing
            .label
            .integrity
            .may_flow_to(crossing.dest_integrity)
        {
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
        let verdict = GateVerdict::Allowed;
        self.emit_audit(crossing, &verdict);
        verdict
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

        let verdict = match mode {
            EnforcementMode::Enforcing => {
                self.total_denied += 1;
                GateVerdict::Denied { layer, reason }
            }
            EnforcementMode::Permissive => {
                self.total_permissive_passes += 1;
                GateVerdict::PermissivePass { layer, reason }
            }
        };
        self.emit_audit(crossing, &verdict);
        verdict
    }

    /// Emit an audit entry for a gate decision.
    fn emit_audit(&mut self, crossing: &BoundaryCrossing, verdict: &GateVerdict) {
        let event_type = match verdict {
            GateVerdict::Allowed => AuditEventType::gate_allowed(),
            GateVerdict::Denied { .. } => AuditEventType::gate_denied(),
            GateVerdict::PermissivePass { .. } => AuditEventType::gate_permissive_pass(),
        };

        let result = if verdict.is_allowed() {
            AuditResult::Success
        } else {
            AuditResult::Failure
        };

        let op_str = match verdict {
            GateVerdict::Allowed => "gate-check",
            GateVerdict::Denied { reason, .. } | GateVerdict::PermissivePass { reason, .. } => {
                reason.as_str()
            }
        };

        let entry = AuditEntry {
            timestamp_ns: crossing.now,
            event_type,
            task_id: crossing.task_id,
            resource_type: crossing.boundary as u8,
            resource_instance: crossing.label.message_type.map_or(0, |id| id.0 as u64),
            operation: Operation::new(op_str),
            result,
            capability_id: 0,
        };

        self.audit_buffer.push(entry);
    }

    /// Reset rate limit window for a message type.
    pub fn reset_rate_window(&mut self, type_id: MessageTypeId) {
        self.invariant_checker.reset_rate_window(type_id);
    }

    // ── Audit ──

    /// Access the gate's audit ring buffer.
    pub fn audit_buffer(&self) -> &AuditRingBuffer<GATE_AUDIT_CAPACITY> {
        &self.audit_buffer
    }

    /// Access the gate's audit ring buffer mutably (for draining).
    pub fn audit_buffer_mut(&mut self) -> &mut AuditRingBuffer<GATE_AUDIT_CAPACITY> {
        &mut self.audit_buffer
    }

    /// Number of audit entries in the buffer.
    pub fn audit_count(&self) -> usize {
        self.audit_buffer.len()
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
    use crate::capability::{CapRegistry, ResourceRef, ResourceType};
    use crate::message_types::{default_registry, MessageMetadata, TensorDtype};
    use alloc::boxed::Box;

    /// Helper: create a cap registry with a pre-granted capability for task 1.
    fn setup_cap_registry() -> Box<CapRegistry> {
        let mut reg = Box::new(CapRegistry::new());
        let resource = ResourceRef::new(ResourceType::NetworkSocket, 1);
        let _ = reg.create(1, resource, Permissions::READ, 0);
        reg
    }

    fn base_crossing<'a>(metadata: &'a MessageMetadata) -> BoundaryCrossing<'a> {
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
            label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Medium, None),
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
            label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Low, None),
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
            label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Medium, None),
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

    // ── Audit emission ──

    #[test]
    fn audit_emitted_on_allowed() {
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
        gate.check(&crossing, &cap_reg, &type_reg);

        assert_eq!(gate.audit_count(), 1);
        let entry = gate.audit_buffer().get(0).unwrap();
        assert_eq!(
            entry.event_type,
            crate::audit::taxonomy::AuditEventType::gate_allowed()
        );
        assert_eq!(entry.result, crate::audit::entry::AuditResult::Success);
        assert_eq!(entry.task_id, 1);
        assert_eq!(entry.resource_type, TrustBoundary::Network as u8);
    }

    #[test]
    fn audit_emitted_on_denied() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let meta = MessageMetadata::empty();

        let crossing = BoundaryCrossing {
            task_id: 99,
            label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Medium, None),
            ..base_crossing(&meta)
        };
        gate.check(&crossing, &cap_reg, &type_reg);

        assert_eq!(gate.audit_count(), 1);
        let entry = gate.audit_buffer().get(0).unwrap();
        assert_eq!(
            entry.event_type,
            crate::audit::taxonomy::AuditEventType::gate_denied()
        );
        assert_eq!(entry.result, crate::audit::entry::AuditResult::Failure);
        assert_eq!(entry.task_id, 99);
    }

    #[test]
    fn audit_emitted_on_permissive_pass() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Permissive);
        let meta = MessageMetadata::empty();

        let crossing = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                Some(MessageTypeId(0xFFFF)),
            ),
            ..base_crossing(&meta)
        };
        gate.check(&crossing, &cap_reg, &type_reg);

        assert_eq!(gate.audit_count(), 1);
        let entry = gate.audit_buffer().get(0).unwrap();
        assert_eq!(
            entry.event_type,
            crate::audit::taxonomy::AuditEventType::gate_permissive_pass()
        );
        assert_eq!(entry.result, crate::audit::entry::AuditResult::Success);
    }

    #[test]
    fn audit_accumulates_multiple_checks() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let mut meta = MessageMetadata::empty();
        meta.rank = 2;
        meta.dtype = TensorDtype::Float32 as u8;
        meta.element_count = 100;
        meta.num_dimensions = 2;
        meta.dimensions = [10, 10, 0, 0, 0, 0, 0, 0];

        // 3 allowed checks
        for _ in 0..3 {
            let crossing = base_crossing(&meta);
            gate.check(&crossing, &cap_reg, &type_reg);
        }

        assert_eq!(gate.audit_count(), 3);
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

    // ── Integration tests: end-to-end boundary flows ──

    /// Helper: valid metadata for InferenceTensorInput (type 0x0001).
    fn valid_tensor_meta() -> MessageMetadata {
        let mut meta = MessageMetadata::empty();
        meta.rank = 4;
        meta.dtype = TensorDtype::Float32 as u8;
        meta.element_count = 1 * 3 * 224 * 224;
        meta.payload_bytes = meta.element_count * 4; // 4 bytes per f32
        meta.num_dimensions = 4;
        meta.dimensions = [1, 3, 224, 224, 0, 0, 0, 0];
        meta
    }

    #[test]
    fn integration_network_to_kernel_enforcing_allowed() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);

        let meta = valid_tensor_meta();
        let crossing = BoundaryCrossing {
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            task_id: 1,
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                Some(MessageTypeId(0x0001)), // InferenceTensorInput
            ),
            resource: ResourceRef::new(ResourceType::NetworkSocket, 1),
            required_permissions: Permissions::READ,
            metadata: &meta,
            dest_integrity: IntegrityLevel::Medium,
            dest_max_classification: ClassificationLevel::Restricted,
            now: 0,
        };

        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert!(verdict.is_allowed());
        assert_eq!(gate.total_allowed(), 1);
    }

    #[test]
    fn integration_bus_to_kernel_permissive_passthrough() {
        let mut cap_reg = Box::new(CapRegistry::new());
        let resource = ResourceRef::new(ResourceType::Device, 1);
        let _ = cap_reg.create(10, resource, Permissions::READ, 0);

        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Permissive);

        let mut meta = MessageMetadata::empty();
        meta.rank = 1;
        meta.dtype = TensorDtype::Float32 as u8;
        meta.element_count = 16;
        meta.payload_bytes = 64;
        meta.num_dimensions = 1;
        meta.dimensions = [16, 0, 0, 0, 0, 0, 0, 0];

        let crossing = BoundaryCrossing {
            boundary: TrustBoundary::BusProtocol,
            direction: DataFlowDirection::Inbound,
            task_id: 10,
            label: SecurityLabel::new(
                ClassificationLevel::Public,
                IntegrityLevel::Low,
                Some(MessageTypeId(0x0010)), // BusSensorFrame
            ),
            resource: ResourceRef::new(ResourceType::Device, 1),
            required_permissions: Permissions::READ,
            metadata: &meta,
            dest_integrity: IntegrityLevel::Medium,
            dest_max_classification: ClassificationLevel::Internal,
            now: 0,
        };

        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        // Low → Medium is a Biba violation, but Permissive → allowed
        assert!(verdict.is_allowed());
        assert!(verdict.is_permissive_pass());
    }

    #[test]
    fn integration_kernel_to_gpu_enforcing_allowed() {
        let mut cap_reg = Box::new(CapRegistry::new());
        let resource = ResourceRef::new(ResourceType::GpuDevice, 0);
        let _ = cap_reg.create(5, resource, Permissions::WRITE, 0);

        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);

        let mut meta = MessageMetadata::empty();
        meta.rank = 4;
        meta.dtype = TensorDtype::Float32 as u8;
        meta.element_count = 1024;
        meta.payload_bytes = 4096;
        meta.num_dimensions = 4;
        meta.dimensions = [1, 1, 32, 32, 0, 0, 0, 0];

        let crossing = BoundaryCrossing {
            boundary: TrustBoundary::Gpu,
            direction: DataFlowDirection::Outbound,
            task_id: 5,
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::High,
                Some(MessageTypeId(0x0030)), // GpuTensorTransfer
            ),
            resource: ResourceRef::new(ResourceType::GpuDevice, 0),
            required_permissions: Permissions::WRITE,
            metadata: &meta,
            dest_integrity: IntegrityLevel::Medium, // High→Medium OK for Biba
            dest_max_classification: ClassificationLevel::Restricted,
            now: 0,
        };

        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert!(verdict.is_allowed());
    }

    #[test]
    fn integration_enforcing_rejection_unknown_type() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        // Set Network boundary to Enforcing (it's already global Enforcing)
        gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);

        let meta = MessageMetadata::empty();
        let crossing = BoundaryCrossing {
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            task_id: 1,
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                Some(MessageTypeId(0xDEAD)), // not in registry
            ),
            resource: ResourceRef::new(ResourceType::NetworkSocket, 1),
            required_permissions: Permissions::READ,
            metadata: &meta,
            dest_integrity: IntegrityLevel::Medium,
            dest_max_classification: ClassificationLevel::Restricted,
            now: 0,
        };

        let verdict = gate.check(&crossing, &cap_reg, &type_reg);
        assert!(verdict.is_denied());
        if let GateVerdict::Denied { layer, reason } = verdict {
            assert_eq!(layer, 4);
            assert_eq!(reason, DenyReason::UnknownMessageType);
        }
    }

    #[test]
    fn integration_multi_flow_statistics() {
        let mut cap_reg = Box::new(CapRegistry::new());
        // Grant caps for tasks 1 and 2
        let _ = cap_reg.create(
            1,
            ResourceRef::new(ResourceType::NetworkSocket, 1),
            Permissions::READ,
            0,
        );
        let _ = cap_reg.create(
            2,
            ResourceRef::new(ResourceType::Device, 0),
            Permissions::READ,
            0,
        );

        let type_reg = default_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Permissive);

        let meta = valid_tensor_meta();

        // Flow 1: Network → Kernel (allowed)
        let c1 = BoundaryCrossing {
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
            metadata: &meta,
            dest_integrity: IntegrityLevel::Medium,
            dest_max_classification: ClassificationLevel::Restricted,
            now: 100,
        };
        gate.check(&c1, &cap_reg, &type_reg);

        // Flow 2: Bus → Kernel (permissive pass due to Biba)
        let mut bus_meta = MessageMetadata::empty();
        bus_meta.rank = 1;
        bus_meta.dtype = TensorDtype::Float32 as u8;
        bus_meta.element_count = 8;
        bus_meta.num_dimensions = 1;
        bus_meta.dimensions = [8, 0, 0, 0, 0, 0, 0, 0];

        let c2 = BoundaryCrossing {
            boundary: TrustBoundary::BusProtocol,
            direction: DataFlowDirection::Inbound,
            task_id: 2,
            label: SecurityLabel::new(
                ClassificationLevel::Public,
                IntegrityLevel::Low,
                Some(MessageTypeId(0x0010)),
            ),
            resource: ResourceRef::new(ResourceType::Device, 0),
            required_permissions: Permissions::READ,
            metadata: &bus_meta,
            dest_integrity: IntegrityLevel::Medium,
            dest_max_classification: ClassificationLevel::Internal,
            now: 200,
        };
        gate.check(&c2, &cap_reg, &type_reg);

        assert_eq!(gate.total_checks(), 2);
        // Both should be "allowed" (one clean, one permissive pass)
        assert_eq!(gate.audit_count(), 2);
    }

    // ── MC/DC coverage: gate decision paths (task 8.1) ──
    // Each test independently toggles one condition while holding others constant,
    // achieving Modified Condition/Decision Coverage for the 5-layer pipeline.
    //
    // NOTE: Default message types have mode=Permissive. To exercise Denied verdicts,
    // we use `None` message type so the boundary/global Enforcing mode takes effect
    // (type mode = None → boundary mode wins).

    #[test]
    fn mcdc_layer1_capability_present_vs_absent() {
        let type_reg = default_registry();
        let meta = valid_tensor_meta();

        // Use None message type so global Enforcing mode wins
        let crossing = BoundaryCrossing {
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            task_id: 1,
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                None, // No type → global mode controls enforcement
            ),
            resource: ResourceRef::new(ResourceType::NetworkSocket, 1),
            required_permissions: Permissions::READ,
            metadata: &meta,
            dest_integrity: IntegrityLevel::Medium,
            dest_max_classification: ClassificationLevel::Restricted,
            now: 0,
        };

        // Condition A: capability present → Allowed
        let cap_reg_ok = setup_cap_registry();
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);
        let v1 = gate.check(&crossing, &cap_reg_ok, &type_reg);
        assert!(v1.is_allowed());

        // Condition A toggled: capability absent → Denied at layer 1
        let cap_reg_empty = Box::new(CapRegistry::new());
        let v2 = gate.check(&crossing, &cap_reg_empty, &type_reg);
        assert!(v2.is_denied());
        if let GateVerdict::Denied { layer, reason } = v2 {
            assert_eq!(layer, 1);
            assert_eq!(reason, DenyReason::MissingCapability);
        }
    }

    #[test]
    fn mcdc_layer2_classification_pass_vs_fail() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let meta = valid_tensor_meta();

        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);

        // Classification OK: Internal ≤ Restricted
        let c1 = BoundaryCrossing {
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            task_id: 1,
            label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Medium, None),
            resource: ResourceRef::new(ResourceType::NetworkSocket, 1),
            required_permissions: Permissions::READ,
            metadata: &meta,
            dest_integrity: IntegrityLevel::Medium,
            dest_max_classification: ClassificationLevel::Restricted,
            now: 0,
        };
        assert!(gate.check(&c1, &cap_reg, &type_reg).is_allowed());

        // Classification FAIL: Restricted > Internal (Bell-LaPadula violation)
        let c2 = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Restricted,
                IntegrityLevel::Medium,
                None,
            ),
            dest_max_classification: ClassificationLevel::Internal,
            ..c1
        };
        let v2 = gate.check(&c2, &cap_reg, &type_reg);
        assert!(v2.is_denied());
        if let GateVerdict::Denied { layer, .. } = v2 {
            assert_eq!(layer, 2);
        }
    }

    #[test]
    fn mcdc_layer3_integrity_pass_vs_fail() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let meta = valid_tensor_meta();

        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);

        // Integrity OK: Medium → Medium (no write-up)
        let c1 = BoundaryCrossing {
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            task_id: 1,
            label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Medium, None),
            resource: ResourceRef::new(ResourceType::NetworkSocket, 1),
            required_permissions: Permissions::READ,
            metadata: &meta,
            dest_integrity: IntegrityLevel::Medium,
            dest_max_classification: ClassificationLevel::Restricted,
            now: 0,
        };
        assert!(gate.check(&c1, &cap_reg, &type_reg).is_allowed());

        // Integrity FAIL: Low → High (Biba violation)
        let c2 = BoundaryCrossing {
            label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Low, None),
            dest_integrity: IntegrityLevel::High,
            ..c1
        };
        let v2 = gate.check(&c2, &cap_reg, &type_reg);
        assert!(v2.is_denied());
        if let GateVerdict::Denied { layer, reason } = v2 {
            assert_eq!(layer, 3);
            assert_eq!(reason, DenyReason::IntegrityViolation);
        }
    }

    #[test]
    fn mcdc_layer4_known_vs_unknown_type() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let meta = valid_tensor_meta();

        // Known type → passes layer 4
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        gate.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);
        let c1 = base_crossing(&meta);
        assert!(gate.check(&c1, &cap_reg, &type_reg).is_allowed());

        // Unknown type → mode resolves: no type mode (not found), boundary Enforcing → Denied
        let c2 = BoundaryCrossing {
            label: SecurityLabel::new(
                ClassificationLevel::Internal,
                IntegrityLevel::Medium,
                Some(MessageTypeId(0xFFFF)),
            ),
            ..c1
        };
        let v2 = gate.check(&c2, &cap_reg, &type_reg);
        assert!(v2.is_denied());
        if let GateVerdict::Denied { layer, reason } = v2 {
            assert_eq!(layer, 4);
            assert_eq!(reason, DenyReason::UnknownMessageType);
        }
    }

    #[test]
    fn mcdc_layer5_enforcing_vs_permissive() {
        let cap_reg = setup_cap_registry();
        let type_reg = default_registry();
        let meta = valid_tensor_meta();

        // Same Biba violation with untyped message (None type), different global modes
        let crossing = BoundaryCrossing {
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            task_id: 1,
            label: SecurityLabel::new(ClassificationLevel::Internal, IntegrityLevel::Low, None),
            resource: ResourceRef::new(ResourceType::NetworkSocket, 1),
            required_permissions: Permissions::READ,
            metadata: &meta,
            dest_integrity: IntegrityLevel::High,
            dest_max_classification: ClassificationLevel::Restricted,
            now: 0,
        };

        // Enforcing → Denied
        let mut gate_e = SecurityGate::new(EnforcementMode::Enforcing);
        gate_e.set_boundary_mode(TrustBoundary::Network, EnforcementMode::Enforcing);
        assert!(gate_e.check(&crossing, &cap_reg, &type_reg).is_denied());

        // Permissive → PermissivePass
        let mut gate_p = SecurityGate::new(EnforcementMode::Permissive);
        let v = gate_p.check(&crossing, &cap_reg, &type_reg);
        assert!(v.is_allowed());
        assert!(v.is_permissive_pass());
    }
}
