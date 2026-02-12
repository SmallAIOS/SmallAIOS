// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Attack surface inventory per trust boundary.
//!
//! Enumerates entry points, accepted data formats, validation
//! mechanisms, and known limitations for each boundary.

use super::trust_boundaries::TrustBoundary;
use alloc::vec::Vec;

/// An entry point in the attack surface.
#[derive(Debug, Clone, Copy)]
pub struct EntryPoint {
    /// Trust boundary this entry point belongs to.
    pub boundary: TrustBoundary,
    /// Entry point name.
    pub name: &'static str,
    /// Data format accepted.
    pub data_format: DataFormat,
    /// Validation mechanism applied.
    pub validation: ValidationMechanism,
    /// Known limitation (empty if none).
    pub limitation: &'static str,
}

/// Data formats accepted at entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataFormat {
    /// Raw binary syscall arguments (registers).
    SyscallRegisters = 0,
    /// Protobuf-encoded ONNX model.
    OnnxProtobuf = 1,
    /// TLS 1.3 record protocol.
    Tls13Record = 2,
    /// HTTP/2 or HTTP/3 frames.
    HttpFrames = 3,
    /// CAN 2.0B frame (11/29-bit ID + 0-8 bytes).
    CanFrame = 4,
    /// ARINC 429 word (32 bits).
    Arinc429Word = 5,
    /// ARINC 664 Part 7 Ethernet frame.
    Arinc664Frame = 6,
    /// MIL-STD-1553B command/data word.
    Mil1553Word = 7,
    /// SpaceWire packet.
    SpaceWirePacket = 8,
    /// CCSDS space packet.
    CcsdsPacket = 9,
    /// PCIe BAR mapped memory.
    PciBarMemory = 10,
    /// DMA descriptor.
    DmaDescriptor = 11,
    /// JSON configuration.
    JsonConfig = 12,
    /// Prometheus text exposition query.
    PrometheusQuery = 13,
}

impl DataFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SyscallRegisters => "syscall-registers",
            Self::OnnxProtobuf => "onnx-protobuf",
            Self::Tls13Record => "tls-1.3-record",
            Self::HttpFrames => "http-frames",
            Self::CanFrame => "can-frame",
            Self::Arinc429Word => "arinc-429-word",
            Self::Arinc664Frame => "arinc-664-frame",
            Self::Mil1553Word => "mil-1553-word",
            Self::SpaceWirePacket => "spacewire-packet",
            Self::CcsdsPacket => "ccsds-packet",
            Self::PciBarMemory => "pci-bar-memory",
            Self::DmaDescriptor => "dma-descriptor",
            Self::JsonConfig => "json-config",
            Self::PrometheusQuery => "prometheus-query",
        }
    }
}

/// Validation mechanisms applied at entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ValidationMechanism {
    /// Capability check (token must be valid + sufficient permissions).
    CapabilityCheck = 0,
    /// Input bounds validation (range, length, alignment).
    BoundsValidation = 1,
    /// Protobuf schema validation.
    SchemaValidation = 2,
    /// TLS certificate chain validation.
    CertificateValidation = 3,
    /// CRC / checksum verification.
    CrcChecksum = 4,
    /// IOMMU/SMMU memory region validation.
    MemoryRegionCheck = 5,
    /// Rate limiting.
    RateLimiting = 6,
    /// Combined: capability + bounds.
    CapabilityAndBounds = 7,
}

impl ValidationMechanism {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CapabilityCheck => "capability-check",
            Self::BoundsValidation => "bounds-validation",
            Self::SchemaValidation => "schema-validation",
            Self::CertificateValidation => "certificate-validation",
            Self::CrcChecksum => "crc-checksum",
            Self::MemoryRegionCheck => "memory-region-check",
            Self::RateLimiting => "rate-limiting",
            Self::CapabilityAndBounds => "capability-and-bounds",
        }
    }
}

/// Complete attack surface inventory.
pub const ATTACK_SURFACE: &[EntryPoint] = &[
    // Kernel boundary
    EntryPoint {
        boundary: TrustBoundary::Kernel,
        name: "syscall-dispatch",
        data_format: DataFormat::SyscallRegisters,
        validation: ValidationMechanism::CapabilityAndBounds,
        limitation: "Syscall handlers return NotSupported (not yet wired)",
    },
    EntryPoint {
        boundary: TrustBoundary::Kernel,
        name: "onnx-model-load",
        data_format: DataFormat::OnnxProtobuf,
        validation: ValidationMechanism::SchemaValidation,
        limitation: "ONNX operators are stubs",
    },
    // K8s boundary
    EntryPoint {
        boundary: TrustBoundary::Kubernetes,
        name: "kubelet-api",
        data_format: DataFormat::HttpFrames,
        validation: ValidationMechanism::CertificateValidation,
        limitation: "Virtual Kubelet not yet implemented",
    },
    EntryPoint {
        boundary: TrustBoundary::Kubernetes,
        name: "metrics-endpoint",
        data_format: DataFormat::PrometheusQuery,
        validation: ValidationMechanism::RateLimiting,
        limitation: "",
    },
    // Network boundary
    EntryPoint {
        boundary: TrustBoundary::Network,
        name: "tls-termination",
        data_format: DataFormat::Tls13Record,
        validation: ValidationMechanism::CertificateValidation,
        limitation: "TLS is stub; only TCP stack is production",
    },
    EntryPoint {
        boundary: TrustBoundary::Network,
        name: "tcp-accept",
        data_format: DataFormat::SyscallRegisters,
        validation: ValidationMechanism::RateLimiting,
        limitation: "",
    },
    // Bus protocol boundary
    EntryPoint {
        boundary: TrustBoundary::BusProtocol,
        name: "can-rx",
        data_format: DataFormat::CanFrame,
        validation: ValidationMechanism::CrcChecksum,
        limitation: "",
    },
    EntryPoint {
        boundary: TrustBoundary::BusProtocol,
        name: "arinc429-rx",
        data_format: DataFormat::Arinc429Word,
        validation: ValidationMechanism::CrcChecksum,
        limitation: "",
    },
    EntryPoint {
        boundary: TrustBoundary::BusProtocol,
        name: "arinc664-rx",
        data_format: DataFormat::Arinc664Frame,
        validation: ValidationMechanism::CrcChecksum,
        limitation: "",
    },
    EntryPoint {
        boundary: TrustBoundary::BusProtocol,
        name: "mil1553-rx",
        data_format: DataFormat::Mil1553Word,
        validation: ValidationMechanism::CrcChecksum,
        limitation: "",
    },
    EntryPoint {
        boundary: TrustBoundary::BusProtocol,
        name: "spacewire-rx",
        data_format: DataFormat::SpaceWirePacket,
        validation: ValidationMechanism::CrcChecksum,
        limitation: "",
    },
    EntryPoint {
        boundary: TrustBoundary::BusProtocol,
        name: "ccsds-rx",
        data_format: DataFormat::CcsdsPacket,
        validation: ValidationMechanism::CrcChecksum,
        limitation: "",
    },
    // GPU boundary
    EntryPoint {
        boundary: TrustBoundary::Gpu,
        name: "command-submission",
        data_format: DataFormat::DmaDescriptor,
        validation: ValidationMechanism::MemoryRegionCheck,
        limitation: "GPU subsystem is stub",
    },
    EntryPoint {
        boundary: TrustBoundary::Gpu,
        name: "dma-transfer",
        data_format: DataFormat::PciBarMemory,
        validation: ValidationMechanism::MemoryRegionCheck,
        limitation: "GPU subsystem is stub",
    },
];

/// Get all entry points for a trust boundary.
pub fn entry_points_for(boundary: TrustBoundary) -> Vec<&'static EntryPoint> {
    ATTACK_SURFACE
        .iter()
        .filter(|e| e.boundary == boundary)
        .collect()
}

/// Count entry points per boundary.
pub fn entry_point_count(boundary: TrustBoundary) -> usize {
    ATTACK_SURFACE
        .iter()
        .filter(|e| e.boundary == boundary)
        .count()
}

/// Get all entry points with known limitations.
pub fn entry_points_with_limitations() -> Vec<&'static EntryPoint> {
    ATTACK_SURFACE
        .iter()
        .filter(|e| !e.limitation.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attack_surface_not_empty() {
        assert!(ATTACK_SURFACE.len() >= 14);
    }

    #[test]
    fn kernel_entry_points() {
        let entries = entry_points_for(TrustBoundary::Kernel);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn network_entry_points() {
        let entries = entry_points_for(TrustBoundary::Network);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn bus_entry_points() {
        let entries = entry_points_for(TrustBoundary::BusProtocol);
        assert_eq!(entries.len(), 6);
    }

    #[test]
    fn gpu_entry_points() {
        let entries = entry_points_for(TrustBoundary::Gpu);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn k8s_entry_points() {
        let entries = entry_points_for(TrustBoundary::Kubernetes);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn entry_points_with_limitations_exist() {
        let limited = entry_points_with_limitations();
        assert!(limited.len() >= 4); // syscall, onnx, tls, gpu
    }

    #[test]
    fn data_format_strings() {
        assert_eq!(DataFormat::SyscallRegisters.as_str(), "syscall-registers");
        assert_eq!(DataFormat::OnnxProtobuf.as_str(), "onnx-protobuf");
        assert_eq!(DataFormat::CanFrame.as_str(), "can-frame");
        assert_eq!(DataFormat::Tls13Record.as_str(), "tls-1.3-record");
    }

    #[test]
    fn validation_strings() {
        assert_eq!(
            ValidationMechanism::CapabilityCheck.as_str(),
            "capability-check"
        );
        assert_eq!(ValidationMechanism::CrcChecksum.as_str(), "crc-checksum");
        assert_eq!(
            ValidationMechanism::CapabilityAndBounds.as_str(),
            "capability-and-bounds"
        );
    }

    #[test]
    fn bus_entries_use_crc() {
        for entry in entry_points_for(TrustBoundary::BusProtocol) {
            assert_eq!(entry.validation, ValidationMechanism::CrcChecksum);
        }
    }

    #[test]
    fn gpu_entries_use_memory_check() {
        for entry in entry_points_for(TrustBoundary::Gpu) {
            assert_eq!(entry.validation, ValidationMechanism::MemoryRegionCheck);
        }
    }
}
