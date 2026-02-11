// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Trust boundary definitions for kernel, K8s, network, bus, and GPU.
//!
//! Each boundary defines entry points, authentication mechanisms,
//! data flow direction, and connection limits.

/// Trust boundary types in the SmallAIOS architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TrustBoundary {
    /// Kernel boundary: syscall interface between userspace and kernel.
    Kernel = 0,
    /// K8s boundary: Virtual Kubelet management API.
    Kubernetes = 1,
    /// Network boundary: TLS 1.3 termination point.
    Network = 2,
    /// Bus protocol boundary: CAN/ARINC/1553/SpaceWire/CCSDS.
    BusProtocol = 3,
    /// GPU boundary: DMA and PCIe BAR mapping controls.
    Gpu = 4,
}

impl TrustBoundary {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::Kubernetes => "kubernetes",
            Self::Network => "network",
            Self::BusProtocol => "bus-protocol",
            Self::Gpu => "gpu",
        }
    }
}

/// Authentication mechanism used at a trust boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AuthMechanism {
    /// Capability token (kernel-internal).
    CapabilityToken = 0,
    /// Mutual TLS 1.3 with PQ hybrid key exchange.
    MutualTls = 1,
    /// Bus framing integrity (CRC/checksum per protocol).
    BusFrameIntegrity = 2,
    /// DMA memory region permissions (IOMMU/SMMU).
    DmaPermissions = 3,
    /// No authentication (public-facing, rate-limited only).
    None = 4,
}

impl AuthMechanism {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CapabilityToken => "capability-token",
            Self::MutualTls => "mutual-tls",
            Self::BusFrameIntegrity => "bus-frame-integrity",
            Self::DmaPermissions => "dma-permissions",
            Self::None => "none",
        }
    }
}

/// Data flow direction at a trust boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataFlowDirection {
    /// Data flows inward only (into the kernel).
    Inbound = 0,
    /// Data flows outward only (from the kernel).
    Outbound = 1,
    /// Data flows in both directions.
    Bidirectional = 2,
}

impl DataFlowDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Inbound => "inbound",
            Self::Outbound => "outbound",
            Self::Bidirectional => "bidirectional",
        }
    }
}

/// Trust boundary definition.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryDefinition {
    /// Boundary type.
    pub boundary: TrustBoundary,
    /// Authentication mechanism.
    pub auth: AuthMechanism,
    /// Data flow direction.
    pub flow: DataFlowDirection,
    /// Maximum concurrent connections (0 = unlimited).
    pub max_connections: u32,
    /// Maximum data rate in bytes/second (0 = unlimited).
    pub max_data_rate_bps: u64,
    /// Number of entry points.
    pub entry_point_count: u16,
    /// Default enforcement mode for the security gate at this boundary.
    /// When `formal-gate` is disabled, this field is still present but unused.
    /// Defaults to Permissive for all boundaries (can be overridden by policy).
    pub default_enforcement_mode: u8, // 0 = Enforcing, 1 = Permissive
}

/// Complete trust boundary definitions for all SmallAIOS boundaries.
pub const BOUNDARY_DEFINITIONS: &[BoundaryDefinition] = &[
    // Kernel boundary: 46 syscalls, capability-protected
    BoundaryDefinition {
        boundary: TrustBoundary::Kernel,
        auth: AuthMechanism::CapabilityToken,
        flow: DataFlowDirection::Bidirectional,
        max_connections: 0, // No connection concept; task-based
        max_data_rate_bps: 0,
        entry_point_count: 46, // ~46 syscalls
        default_enforcement_mode: 1, // Permissive
    },
    // K8s boundary: Virtual Kubelet API
    BoundaryDefinition {
        boundary: TrustBoundary::Kubernetes,
        auth: AuthMechanism::MutualTls,
        flow: DataFlowDirection::Bidirectional,
        max_connections: 16,
        max_data_rate_bps: 10_000_000, // 10 MB/s
        entry_point_count: 8, // Pod CRUD, health, metrics, logs, exec
        default_enforcement_mode: 1, // Permissive
    },
    // Network boundary: TLS 1.3
    BoundaryDefinition {
        boundary: TrustBoundary::Network,
        auth: AuthMechanism::MutualTls,
        flow: DataFlowDirection::Bidirectional,
        max_connections: 256,
        max_data_rate_bps: 100_000_000, // 100 MB/s
        entry_point_count: 4, // TCP, UDP, TLS, QUIC
        default_enforcement_mode: 1, // Permissive
    },
    // Bus protocol boundary
    BoundaryDefinition {
        boundary: TrustBoundary::BusProtocol,
        auth: AuthMechanism::BusFrameIntegrity,
        flow: DataFlowDirection::Bidirectional,
        max_connections: 32,          // Max bus channels
        max_data_rate_bps: 1_000_000, // 1 MB/s typical for CAN/1553
        entry_point_count: 6, // CAN, ARINC 429, ARINC 664, MIL-STD-1553, SpaceWire, CCSDS
        default_enforcement_mode: 1, // Permissive
    },
    // GPU boundary
    BoundaryDefinition {
        boundary: TrustBoundary::Gpu,
        auth: AuthMechanism::DmaPermissions,
        flow: DataFlowDirection::Bidirectional,
        max_connections: 1,   // Single GPU context
        max_data_rate_bps: 0, // DMA speed, not rate-limited
        entry_point_count: 3, // Command submission, DMA transfer, BAR access
        default_enforcement_mode: 1, // Permissive
    },
];

/// Look up the boundary definition for a trust boundary.
pub fn definition_for(boundary: TrustBoundary) -> Option<&'static BoundaryDefinition> {
    BOUNDARY_DEFINITIONS.iter().find(|d| d.boundary == boundary)
}

/// Count of boundaries using a specific auth mechanism.
pub fn count_by_auth(auth: AuthMechanism) -> usize {
    BOUNDARY_DEFINITIONS
        .iter()
        .filter(|d| d.auth == auth)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_boundaries_defined() {
        assert_eq!(BOUNDARY_DEFINITIONS.len(), 5);
    }

    #[test]
    fn kernel_boundary() {
        let def = definition_for(TrustBoundary::Kernel).unwrap();
        assert_eq!(def.auth, AuthMechanism::CapabilityToken);
        assert_eq!(def.entry_point_count, 46);
    }

    #[test]
    fn k8s_boundary() {
        let def = definition_for(TrustBoundary::Kubernetes).unwrap();
        assert_eq!(def.auth, AuthMechanism::MutualTls);
        assert_eq!(def.max_connections, 16);
    }

    #[test]
    fn network_boundary() {
        let def = definition_for(TrustBoundary::Network).unwrap();
        assert_eq!(def.auth, AuthMechanism::MutualTls);
        assert_eq!(def.max_connections, 256);
    }

    #[test]
    fn bus_boundary() {
        let def = definition_for(TrustBoundary::BusProtocol).unwrap();
        assert_eq!(def.auth, AuthMechanism::BusFrameIntegrity);
        assert_eq!(def.entry_point_count, 6);
    }

    #[test]
    fn gpu_boundary() {
        let def = definition_for(TrustBoundary::Gpu).unwrap();
        assert_eq!(def.auth, AuthMechanism::DmaPermissions);
        assert_eq!(def.max_connections, 1);
    }

    #[test]
    fn boundary_strings() {
        assert_eq!(TrustBoundary::Kernel.as_str(), "kernel");
        assert_eq!(TrustBoundary::Kubernetes.as_str(), "kubernetes");
        assert_eq!(TrustBoundary::Network.as_str(), "network");
        assert_eq!(TrustBoundary::BusProtocol.as_str(), "bus-protocol");
        assert_eq!(TrustBoundary::Gpu.as_str(), "gpu");
    }

    #[test]
    fn auth_mechanism_strings() {
        assert_eq!(AuthMechanism::CapabilityToken.as_str(), "capability-token");
        assert_eq!(AuthMechanism::MutualTls.as_str(), "mutual-tls");
        assert_eq!(
            AuthMechanism::BusFrameIntegrity.as_str(),
            "bus-frame-integrity"
        );
    }

    #[test]
    fn flow_direction_strings() {
        assert_eq!(DataFlowDirection::Inbound.as_str(), "inbound");
        assert_eq!(DataFlowDirection::Outbound.as_str(), "outbound");
        assert_eq!(DataFlowDirection::Bidirectional.as_str(), "bidirectional");
    }

    #[test]
    fn count_by_auth_mechanism() {
        assert_eq!(count_by_auth(AuthMechanism::MutualTls), 2);
        assert_eq!(count_by_auth(AuthMechanism::CapabilityToken), 1);
        assert_eq!(count_by_auth(AuthMechanism::BusFrameIntegrity), 1);
        assert_eq!(count_by_auth(AuthMechanism::DmaPermissions), 1);
        assert_eq!(count_by_auth(AuthMechanism::None), 0);
    }

    #[test]
    fn all_boundaries_bidirectional() {
        for def in BOUNDARY_DEFINITIONS {
            assert_eq!(def.flow, DataFlowDirection::Bidirectional);
        }
    }
}
