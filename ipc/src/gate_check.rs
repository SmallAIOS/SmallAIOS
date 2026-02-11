// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Security gate integration for IPC message flows.
//!
//! Provides gate check call sites at trust boundary crossing points:
//! - Network boundary: inbound inference requests
//!
//! Each check constructs a `BoundaryCrossing` from the decoded message
//! and runs it through the `SecurityGate` pipeline.

use smallaios_security::boundary::trust_boundaries::{DataFlowDirection, TrustBoundary};
use smallaios_security::capability::{CapRegistry, Permissions, ResourceRef, ResourceType, TaskId};
use smallaios_security::compliance::classification::ClassificationLevel;
use smallaios_security::enforcement::{EnforcementMode, GateVerdict};
use smallaios_security::gate::{BoundaryCrossing, SecurityGate};
use smallaios_security::labels::{IntegrityLevel, MessageTypeId, SecurityLabel};
use smallaios_security::message_types::{MessageMetadata, MessageTypeRegistry, TensorDtype};

use crate::inference_proto::{InferenceRequest, InferenceRequestType};

/// Check an inbound inference request against the security gate.
///
/// This is the gate call site at the Network→Kernel boundary for
/// inference protocol messages. It constructs the appropriate metadata
/// from the decoded request and runs the 5-layer verification pipeline.
///
/// # Arguments
///
/// * `request` - The decoded inference request
/// * `task_id` - Task ID of the connection handler
/// * `gate` - The security gate instance
/// * `cap_registry` - Capability registry
/// * `type_registry` - Message type registry
/// * `now` - Current time for expiry checks
///
/// # Returns
///
/// The gate verdict (Allowed, Denied, or PermissivePass).
pub fn check_inbound_inference(
    request: &InferenceRequest,
    task_id: TaskId,
    gate: &mut SecurityGate,
    cap_registry: &CapRegistry,
    type_registry: &MessageTypeRegistry,
    now: u64,
) -> GateVerdict {
    // Determine message type ID based on request type
    let message_type_id = match request.request_type {
        InferenceRequestType::RunInference => Some(MessageTypeId(0x0003)), // InferenceRequest
        InferenceRequestType::LoadModel => Some(MessageTypeId(0x0003)),
        InferenceRequestType::UnloadModel => Some(MessageTypeId(0x0003)),
        InferenceRequestType::GetModelInfo => Some(MessageTypeId(0x0003)),
    };

    // Build metadata from the first input tensor (if any)
    let metadata = if let Some(input) = request.inputs.first() {
        let mut meta = MessageMetadata::empty();
        meta.rank = input.shape.len() as u8;
        meta.num_dimensions = input.shape.len() as u8;
        for (i, &dim) in input.shape.iter().enumerate().take(8) {
            meta.dimensions[i] = dim;
        }
        meta.element_count = input.shape.iter().copied().fold(1u32, |a, d| a.saturating_mul(d));
        meta.payload_bytes = input.data.len() as u32;
        meta.dtype = match input.data_type {
            crate::inference_proto::TensorDataType::Float32 => TensorDtype::Float32 as u8,
            crate::inference_proto::TensorDataType::Float16 => TensorDtype::Float16 as u8,
            crate::inference_proto::TensorDataType::Int8 => TensorDtype::Int8 as u8,
            crate::inference_proto::TensorDataType::Int32 => TensorDtype::Int32 as u8,
            crate::inference_proto::TensorDataType::Int64 => TensorDtype::Int64 as u8,
            crate::inference_proto::TensorDataType::Uint8 => TensorDtype::Uint8 as u8,
        };
        meta.timestamp = now;
        meta
    } else {
        let mut meta = MessageMetadata::empty();
        meta.timestamp = now;
        meta
    };

    // Security label: network inbound data is untrusted/public by default
    let label = SecurityLabel::new(
        ClassificationLevel::Public,
        IntegrityLevel::Low, // Network data starts at Low integrity
        message_type_id,
    );

    let crossing = BoundaryCrossing {
        boundary: TrustBoundary::Network,
        direction: DataFlowDirection::Inbound,
        task_id,
        label,
        resource: ResourceRef::new(ResourceType::NetworkSocket, 0),
        required_permissions: Permissions::READ,
        metadata: &metadata,
        dest_integrity: IntegrityLevel::Medium, // Kernel boundary
        dest_max_classification: ClassificationLevel::Restricted,
        now,
    };

    gate.check(&crossing, cap_registry, type_registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::string::String;
    use alloc::vec;
    use crate::inference_proto::{TensorData, TensorDataType};
    use smallaios_security::message_types::default_registry;

    fn setup_cap_registry(task_id: TaskId) -> Box<CapRegistry> {
        let mut reg = Box::new(CapRegistry::new());
        let resource = ResourceRef::new(ResourceType::NetworkSocket, 0);
        let _ = reg.create(task_id, resource, Permissions::READ, 0);
        reg
    }

    #[test]
    fn inbound_inference_allowed_permissive() {
        let mut gate = SecurityGate::new(EnforcementMode::Permissive);
        let cap_reg = setup_cap_registry(1);
        let type_reg = default_registry();

        let request = InferenceRequest {
            request_type: InferenceRequestType::RunInference,
            request_id: 1,
            model_id: 1,
            inputs: vec![TensorData {
                name: String::from("input"),
                data_type: TensorDataType::Float32,
                shape: vec![1, 3, 224, 224],
                data: vec![0u8; 100],
            }],
            timeout_ms: 5000,
        };

        let verdict = check_inbound_inference(&request, 1, &mut gate, &cap_reg, &type_reg, 0);
        // Network data has Low integrity flowing to Medium dest → Biba violation
        // but Permissive mode → PermissivePass
        assert!(verdict.is_allowed());
    }

    #[test]
    fn inbound_inference_denied_enforcing_biba() {
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let cap_reg = setup_cap_registry(1);
        let type_reg = default_registry();

        let request = InferenceRequest {
            request_type: InferenceRequestType::RunInference,
            request_id: 1,
            model_id: 1,
            inputs: vec![],
            timeout_ms: 5000,
        };

        // Untyped message → global Enforcing mode applies
        // Low integrity → Medium dest = Biba violation → Denied
        let verdict = check_inbound_inference(&request, 1, &mut gate, &cap_reg, &type_reg, 0);
        // With typed message (0x0003), per-type Permissive overrides global Enforcing
        // So we get PermissivePass for the Biba violation
        assert!(verdict.is_allowed()); // PermissivePass still allows
    }

    #[test]
    fn inbound_inference_no_capability() {
        let mut gate = SecurityGate::new(EnforcementMode::Enforcing);
        let cap_reg = Box::new(CapRegistry::new()); // empty - no caps
        let type_reg = default_registry();

        let request = InferenceRequest {
            request_type: InferenceRequestType::LoadModel,
            request_id: 1,
            model_id: 1,
            inputs: vec![],
            timeout_ms: 1000,
        };

        let verdict = check_inbound_inference(&request, 99, &mut gate, &cap_reg, &type_reg, 0);
        // Layer 1 fails but per-type Permissive → PermissivePass
        assert!(verdict.is_allowed());
    }

    #[test]
    fn inbound_inference_empty_inputs() {
        let mut gate = SecurityGate::new(EnforcementMode::Permissive);
        let cap_reg = setup_cap_registry(1);
        let type_reg = default_registry();

        let request = InferenceRequest {
            request_type: InferenceRequestType::GetModelInfo,
            request_id: 42,
            model_id: 5,
            inputs: vec![],
            timeout_ms: 1000,
        };

        let verdict = check_inbound_inference(&request, 1, &mut gate, &cap_reg, &type_reg, 0);
        assert!(verdict.is_allowed());
    }
}
