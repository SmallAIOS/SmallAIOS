// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Clean-room attestation process data model.
//!
//! Documents the zero third-party code policy and development methodology.
//! Every source file in the SmallAIOS codebase must be attested as
//! original work with no copied code from external sources.

use alloc::string::String;
use alloc::vec::Vec;

/// Maximum attestation entries.
const MAX_ATTESTATIONS: usize = 256;

/// Clean-room development methodology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DevelopmentMethod {
    /// Written from specification only (no reference to existing code).
    FromSpecification = 0,
    /// Written using clean-room reverse engineering (specification derived
    /// from public interface documentation, not source code).
    CleanRoomReverseEngineering = 1,
    /// Written from RFC/standard specification.
    FromStandard = 2,
    /// Written as original design (no prior art).
    OriginalDesign = 3,
}

impl DevelopmentMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FromSpecification => "from-specification",
            Self::CleanRoomReverseEngineering => "clean-room-reverse-engineering",
            Self::FromStandard => "from-standard",
            Self::OriginalDesign => "original-design",
        }
    }
}

/// Clean-room attestation for a source file or module.
#[derive(Debug, Clone)]
pub struct CleanRoomAttestation {
    /// Crate name (e.g., "smallaios-security").
    pub crate_name: String,
    /// Module path (e.g., "crypto::sha3").
    pub module_path: String,
    /// Development methodology used.
    pub method: DevelopmentMethod,
    /// Reference specification (e.g., "FIPS 202", "RFC 9293").
    pub reference_spec: String,
    /// Attestation timestamp (nanoseconds).
    pub attested_ns: u64,
    /// Whether this attestation has been reviewed by a second party.
    pub peer_reviewed: bool,
}

/// Clean-room attestation registry.
pub struct CleanRoomRegistry {
    attestations: Vec<CleanRoomAttestation>,
}

impl CleanRoomRegistry {
    pub fn new() -> Self {
        Self {
            attestations: Vec::new(),
        }
    }

    /// Register a clean-room attestation.
    pub fn attest(
        &mut self,
        crate_name: String,
        module_path: String,
        method: DevelopmentMethod,
        reference_spec: String,
        attested_ns: u64,
    ) -> bool {
        if self.attestations.len() >= MAX_ATTESTATIONS {
            return false;
        }
        self.attestations.push(CleanRoomAttestation {
            crate_name,
            module_path,
            method,
            reference_spec,
            attested_ns,
            peer_reviewed: false,
        });
        true
    }

    /// Mark an attestation as peer-reviewed.
    pub fn mark_reviewed(&mut self, crate_name: &str, module_path: &str) -> bool {
        if let Some(att) = self
            .attestations
            .iter_mut()
            .find(|a| a.crate_name == crate_name && a.module_path == module_path)
        {
            att.peer_reviewed = true;
            true
        } else {
            false
        }
    }

    /// Get attestation for a module.
    pub fn get(&self, crate_name: &str, module_path: &str) -> Option<&CleanRoomAttestation> {
        self.attestations
            .iter()
            .find(|a| a.crate_name == crate_name && a.module_path == module_path)
    }

    /// Count of total attestations.
    pub fn total_count(&self) -> usize {
        self.attestations.len()
    }

    /// Count of peer-reviewed attestations.
    pub fn reviewed_count(&self) -> usize {
        self.attestations.iter().filter(|a| a.peer_reviewed).count()
    }

    /// Count of unreviewed attestations.
    pub fn unreviewed_count(&self) -> usize {
        self.attestations
            .iter()
            .filter(|a| !a.peer_reviewed)
            .count()
    }

    /// List attestations by development method.
    pub fn by_method(&self, method: DevelopmentMethod) -> Vec<&CleanRoomAttestation> {
        self.attestations
            .iter()
            .filter(|a| a.method == method)
            .collect()
    }
}

impl Default for CleanRoomRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Well-known SmallAIOS clean-room attestation entries.
///
/// These represent the major modules and their development provenance.
pub const KNOWN_ATTESTATIONS: &[(&str, &str, &str)] = &[
    ("smallaios-security", "crypto::sha3", "FIPS 202"),
    (
        "smallaios-security",
        "crypto::aes_gcm",
        "FIPS 197 + SP 800-38D",
    ),
    ("smallaios-security", "crypto::ml_kem", "FIPS 203"),
    ("smallaios-security", "crypto::ml_dsa", "FIPS 204"),
    ("smallaios-net", "tcp", "RFC 9293"),
    ("smallaios-net", "ipv4", "RFC 791"),
    ("smallaios-net", "ipv6", "RFC 8200"),
    ("smallaios-net", "arp", "RFC 826"),
    ("smallaios-onnx-rt", "parser", "ONNX IR Specification"),
    ("smallaios-kernel", "mem::buddy", "Original design"),
    ("smallaios-kernel", "scheduler", "Original design"),
    ("smallaios-kernel", "syscall", "Original design"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attest_and_retrieve() {
        let mut reg = CleanRoomRegistry::new();
        assert!(reg.attest(
            String::from("smallaios-security"),
            String::from("crypto::sha3"),
            DevelopmentMethod::FromStandard,
            String::from("FIPS 202"),
            1_000_000,
        ));

        let att = reg.get("smallaios-security", "crypto::sha3").unwrap();
        assert_eq!(att.method, DevelopmentMethod::FromStandard);
        assert_eq!(att.reference_spec, "FIPS 202");
        assert!(!att.peer_reviewed);
    }

    #[test]
    fn mark_reviewed() {
        let mut reg = CleanRoomRegistry::new();
        reg.attest(
            String::from("crate"),
            String::from("mod"),
            DevelopmentMethod::OriginalDesign,
            String::from("none"),
            0,
        );
        assert_eq!(reg.reviewed_count(), 0);
        assert!(reg.mark_reviewed("crate", "mod"));
        assert_eq!(reg.reviewed_count(), 1);
        assert_eq!(reg.unreviewed_count(), 0);
    }

    #[test]
    fn mark_reviewed_nonexistent() {
        let mut reg = CleanRoomRegistry::new();
        assert!(!reg.mark_reviewed("crate", "mod"));
    }

    #[test]
    fn counts() {
        let mut reg = CleanRoomRegistry::new();
        reg.attest(
            String::from("a"),
            String::from("b"),
            DevelopmentMethod::FromStandard,
            String::from("s"),
            0,
        );
        reg.attest(
            String::from("c"),
            String::from("d"),
            DevelopmentMethod::OriginalDesign,
            String::from("s"),
            0,
        );
        assert_eq!(reg.total_count(), 2);
        assert_eq!(reg.unreviewed_count(), 2);
    }

    #[test]
    fn by_method() {
        let mut reg = CleanRoomRegistry::new();
        reg.attest(
            String::from("a"),
            String::from("b"),
            DevelopmentMethod::FromStandard,
            String::from("s"),
            0,
        );
        reg.attest(
            String::from("c"),
            String::from("d"),
            DevelopmentMethod::OriginalDesign,
            String::from("s"),
            0,
        );
        reg.attest(
            String::from("e"),
            String::from("f"),
            DevelopmentMethod::FromStandard,
            String::from("s"),
            0,
        );

        assert_eq!(reg.by_method(DevelopmentMethod::FromStandard).len(), 2);
        assert_eq!(reg.by_method(DevelopmentMethod::OriginalDesign).len(), 1);
    }

    #[test]
    fn attestation_limit() {
        let mut reg = CleanRoomRegistry::new();
        for i in 0..MAX_ATTESTATIONS {
            assert!(reg.attest(
                alloc::format!("crate-{}", i),
                String::from("mod"),
                DevelopmentMethod::OriginalDesign,
                String::from("none"),
                0,
            ));
        }
        assert!(!reg.attest(
            String::from("overflow"),
            String::from("mod"),
            DevelopmentMethod::OriginalDesign,
            String::from("none"),
            0,
        ));
    }

    #[test]
    fn method_strings() {
        assert_eq!(
            DevelopmentMethod::FromSpecification.as_str(),
            "from-specification"
        );
        assert_eq!(DevelopmentMethod::FromStandard.as_str(), "from-standard");
        assert_eq!(
            DevelopmentMethod::OriginalDesign.as_str(),
            "original-design"
        );
    }

    #[test]
    fn known_attestations_not_empty() {
        assert!(KNOWN_ATTESTATIONS.len() >= 10);
    }
}
