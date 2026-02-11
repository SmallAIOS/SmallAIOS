// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-boundary data flow authentication and integrity verification.
//!
//! Ensures all data flows across trust boundaries are authenticated
//! and integrity-protected using one of:
//! - Capability tokens (kernel-internal)
//! - Mutual TLS (network/K8s)
//! - Bus framing (CRC/checksum per protocol spec)

use super::trust_boundaries::{AuthMechanism, TrustBoundary};

/// A cross-boundary data flow to be verified.
#[derive(Debug, Clone, Copy)]
pub struct DataFlow {
    /// Source boundary.
    pub source: TrustBoundary,
    /// Destination boundary.
    pub destination: TrustBoundary,
    /// Description of the data flow.
    pub description: &'static str,
    /// Required authentication mechanism.
    pub required_auth: AuthMechanism,
    /// Whether integrity protection is required.
    pub integrity_required: bool,
    /// Whether confidentiality protection is required.
    pub confidentiality_required: bool,
}

/// All defined cross-boundary data flows.
pub const CROSS_BOUNDARY_FLOWS: &[DataFlow] = &[
    // Kernel <-> Network: inference requests via TLS
    DataFlow {
        source: TrustBoundary::Network,
        destination: TrustBoundary::Kernel,
        description: "Inference request via TLS",
        required_auth: AuthMechanism::MutualTls,
        integrity_required: true,
        confidentiality_required: true,
    },
    // Kernel -> Network: inference response via TLS
    DataFlow {
        source: TrustBoundary::Kernel,
        destination: TrustBoundary::Network,
        description: "Inference response via TLS",
        required_auth: AuthMechanism::MutualTls,
        integrity_required: true,
        confidentiality_required: true,
    },
    // K8s -> Kernel: pod management commands
    DataFlow {
        source: TrustBoundary::Kubernetes,
        destination: TrustBoundary::Kernel,
        description: "Pod management commands",
        required_auth: AuthMechanism::MutualTls,
        integrity_required: true,
        confidentiality_required: true,
    },
    // Kernel -> K8s: health and metrics
    DataFlow {
        source: TrustBoundary::Kernel,
        destination: TrustBoundary::Kubernetes,
        description: "Health status and metrics export",
        required_auth: AuthMechanism::MutualTls,
        integrity_required: true,
        confidentiality_required: false, // Metrics are Public classification
    },
    // Bus -> Kernel: sensor data ingestion
    DataFlow {
        source: TrustBoundary::BusProtocol,
        destination: TrustBoundary::Kernel,
        description: "Sensor data ingestion from bus",
        required_auth: AuthMechanism::BusFrameIntegrity,
        integrity_required: true,
        confidentiality_required: false,
    },
    // Kernel -> Bus: actuator commands
    DataFlow {
        source: TrustBoundary::Kernel,
        destination: TrustBoundary::BusProtocol,
        description: "Actuator commands to bus",
        required_auth: AuthMechanism::CapabilityToken,
        integrity_required: true,
        confidentiality_required: false,
    },
    // Kernel <-> GPU: tensor transfers
    DataFlow {
        source: TrustBoundary::Kernel,
        destination: TrustBoundary::Gpu,
        description: "Tensor data DMA to GPU",
        required_auth: AuthMechanism::DmaPermissions,
        integrity_required: true,
        confidentiality_required: false,
    },
    DataFlow {
        source: TrustBoundary::Gpu,
        destination: TrustBoundary::Kernel,
        description: "Inference results DMA from GPU",
        required_auth: AuthMechanism::DmaPermissions,
        integrity_required: true,
        confidentiality_required: false,
    },
];

/// Result of verifying a data flow's authentication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowVerificationResult {
    /// Flow is properly authenticated and integrity-protected.
    Verified,
    /// Flow is missing required authentication.
    MissingAuth {
        flow_index: usize,
        required: AuthMechanism,
    },
    /// Flow is missing integrity protection.
    MissingIntegrity { flow_index: usize },
    /// Flow is missing confidentiality protection.
    MissingConfidentiality { flow_index: usize },
}

/// Verify that a specific data flow has the required protections.
pub fn verify_flow(
    flow_index: usize,
    has_auth: bool,
    has_integrity: bool,
    has_confidentiality: bool,
) -> FlowVerificationResult {
    if flow_index >= CROSS_BOUNDARY_FLOWS.len() {
        return FlowVerificationResult::MissingAuth {
            flow_index,
            required: AuthMechanism::None,
        };
    }
    let flow = &CROSS_BOUNDARY_FLOWS[flow_index];

    if !has_auth {
        return FlowVerificationResult::MissingAuth {
            flow_index,
            required: flow.required_auth,
        };
    }
    if flow.integrity_required && !has_integrity {
        return FlowVerificationResult::MissingIntegrity { flow_index };
    }
    if flow.confidentiality_required && !has_confidentiality {
        return FlowVerificationResult::MissingConfidentiality { flow_index };
    }
    FlowVerificationResult::Verified
}

/// Verify all data flows. Returns the first failure, or Verified if all pass.
pub fn verify_all_flows(
    flow_states: &[(bool, bool, bool)], // (has_auth, has_integrity, has_confidentiality)
) -> FlowVerificationResult {
    for (i, &(auth, integrity, confidentiality)) in flow_states.iter().enumerate() {
        if i >= CROSS_BOUNDARY_FLOWS.len() {
            break;
        }
        let result = verify_flow(i, auth, integrity, confidentiality);
        if result != FlowVerificationResult::Verified {
            return result;
        }
    }
    FlowVerificationResult::Verified
}

/// Count of flows requiring confidentiality.
pub fn confidential_flow_count() -> usize {
    CROSS_BOUNDARY_FLOWS
        .iter()
        .filter(|f| f.confidentiality_required)
        .count()
}

/// Get all flows between two boundaries.
pub fn flows_between(
    source: TrustBoundary,
    destination: TrustBoundary,
) -> alloc::vec::Vec<&'static DataFlow> {
    CROSS_BOUNDARY_FLOWS
        .iter()
        .filter(|f| f.source == source && f.destination == destination)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flows_defined() {
        assert_eq!(CROSS_BOUNDARY_FLOWS.len(), 8);
    }

    #[test]
    fn all_flows_require_integrity() {
        for flow in CROSS_BOUNDARY_FLOWS {
            assert!(flow.integrity_required, "Flow '{}' should require integrity", flow.description);
        }
    }

    #[test]
    fn tls_flows_require_confidentiality() {
        let confidential: alloc::vec::Vec<_> = CROSS_BOUNDARY_FLOWS
            .iter()
            .filter(|f| f.confidentiality_required)
            .collect();
        assert!(confidential.len() >= 3);
        for flow in &confidential {
            assert_eq!(flow.required_auth, AuthMechanism::MutualTls);
        }
    }

    #[test]
    fn bus_flows_no_confidentiality() {
        let bus_flows: alloc::vec::Vec<_> = CROSS_BOUNDARY_FLOWS
            .iter()
            .filter(|f| f.source == TrustBoundary::BusProtocol || f.destination == TrustBoundary::BusProtocol)
            .collect();
        for flow in &bus_flows {
            assert!(!flow.confidentiality_required);
        }
    }

    #[test]
    fn verify_flow_success() {
        let result = verify_flow(0, true, true, true);
        assert_eq!(result, FlowVerificationResult::Verified);
    }

    #[test]
    fn verify_flow_missing_auth() {
        let result = verify_flow(0, false, true, true);
        assert!(matches!(result, FlowVerificationResult::MissingAuth { .. }));
    }

    #[test]
    fn verify_flow_missing_integrity() {
        let result = verify_flow(0, true, false, true);
        assert_eq!(result, FlowVerificationResult::MissingIntegrity { flow_index: 0 });
    }

    #[test]
    fn verify_flow_missing_confidentiality() {
        // Flow 0 requires confidentiality (TLS inference request)
        let result = verify_flow(0, true, true, false);
        assert_eq!(result, FlowVerificationResult::MissingConfidentiality { flow_index: 0 });
    }

    #[test]
    fn verify_flow_bus_no_confidentiality_ok() {
        // Flow 4 is bus sensor data (no confidentiality required)
        let result = verify_flow(4, true, true, false);
        assert_eq!(result, FlowVerificationResult::Verified);
    }

    #[test]
    fn verify_all_flows_success() {
        let states: alloc::vec::Vec<_> = CROSS_BOUNDARY_FLOWS
            .iter()
            .map(|f| (true, true, f.confidentiality_required))
            .collect();
        assert_eq!(verify_all_flows(&states), FlowVerificationResult::Verified);
    }

    #[test]
    fn verify_all_flows_first_failure() {
        // All authed/integrity, but flow 0 missing confidentiality
        let mut states: alloc::vec::Vec<_> = CROSS_BOUNDARY_FLOWS
            .iter()
            .map(|f| (true, true, f.confidentiality_required))
            .collect();
        states[0].2 = false; // Remove confidentiality from flow 0
        let result = verify_all_flows(&states);
        assert_eq!(result, FlowVerificationResult::MissingConfidentiality { flow_index: 0 });
    }

    #[test]
    fn confidential_flow_count_check() {
        assert!(confidential_flow_count() >= 3);
    }

    #[test]
    fn flows_between_boundaries() {
        let flows = flows_between(TrustBoundary::Network, TrustBoundary::Kernel);
        assert_eq!(flows.len(), 1);
        assert_eq!(flows[0].description, "Inference request via TLS");
    }

    #[test]
    fn flows_between_gpu_and_kernel() {
        let flows = flows_between(TrustBoundary::Kernel, TrustBoundary::Gpu);
        assert_eq!(flows.len(), 1);
    }
}
