// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! NIST SP 800-53 control cross-references to SmallAIOS security mechanisms.
//!
//! Maps each security subsystem to the NIST controls it satisfies:
//! - Capability system -> AC-3 (Access Enforcement), AC-6 (Least Privilege)
//! - Audit logging -> AU-2 (Event Logging), AU-12 (Audit Record Generation)
//! - Cryptography -> SC-12 (Key Management), SC-13 (Cryptographic Protection)
//! - Information flow -> AC-4 (Information Flow Enforcement)
//! - Monitoring -> SI-4 (System Monitoring)

/// NIST SP 800-53 control family identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ControlFamily {
    /// AC: Access Control
    AccessControl = 0,
    /// AU: Audit and Accountability
    Audit = 1,
    /// CA: Assessment, Authorization, and Monitoring
    Assessment = 2,
    /// CM: Configuration Management
    ConfigManagement = 3,
    /// CP: Contingency Planning
    Contingency = 4,
    /// IA: Identification and Authentication
    IdAuth = 5,
    /// IR: Incident Response
    IncidentResponse = 6,
    /// SC: System and Communications Protection
    SystemComm = 7,
    /// SI: System and Information Integrity
    SystemIntegrity = 8,
    /// SR: Supply Chain Risk Management
    SupplyChain = 9,
}

impl ControlFamily {
    pub fn code(&self) -> &'static str {
        match self {
            Self::AccessControl => "AC",
            Self::Audit => "AU",
            Self::Assessment => "CA",
            Self::ConfigManagement => "CM",
            Self::Contingency => "CP",
            Self::IdAuth => "IA",
            Self::IncidentResponse => "IR",
            Self::SystemComm => "SC",
            Self::SystemIntegrity => "SI",
            Self::SupplyChain => "SR",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::AccessControl => "Access Control",
            Self::Audit => "Audit and Accountability",
            Self::Assessment => "Assessment, Authorization, and Monitoring",
            Self::ConfigManagement => "Configuration Management",
            Self::Contingency => "Contingency Planning",
            Self::IdAuth => "Identification and Authentication",
            Self::IncidentResponse => "Incident Response",
            Self::SystemComm => "System and Communications Protection",
            Self::SystemIntegrity => "System and Information Integrity",
            Self::SupplyChain => "Supply Chain Risk Management",
        }
    }
}

/// A specific NIST control identifier (e.g., AC-3, AU-12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlId {
    pub family: ControlFamily,
    pub number: u16,
}

impl ControlId {
    pub const fn new(family: ControlFamily, number: u16) -> Self {
        Self { family, number }
    }
}

/// SmallAIOS security mechanism that satisfies NIST controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SecurityMechanism {
    /// Capability-based access control (capability.rs).
    CapabilitySystem = 0,
    /// Tamper-evident audit logging (audit/).
    AuditLogging = 1,
    /// Post-quantum cryptography (crypto/).
    Cryptography = 2,
    /// Information flow enforcement (boundary.rs).
    InformationFlow = 3,
    /// Continuous security monitoring (monitoring/).
    Monitoring = 4,
    /// Incident response (incident/).
    IncidentResponse = 5,
    /// Supply chain security (supply_chain/).
    SupplyChain = 6,
    /// Key management lifecycle (crypto/key_manager.rs).
    KeyManagement = 7,
    /// OT/ICS security hardening (ot/).
    OtSecurity = 8,
}

impl SecurityMechanism {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CapabilitySystem => "capability-system",
            Self::AuditLogging => "audit-logging",
            Self::Cryptography => "cryptography",
            Self::InformationFlow => "information-flow",
            Self::Monitoring => "monitoring",
            Self::IncidentResponse => "incident-response",
            Self::SupplyChain => "supply-chain",
            Self::KeyManagement => "key-management",
            Self::OtSecurity => "ot-security",
        }
    }
}

/// A cross-reference mapping a security mechanism to NIST controls.
#[derive(Debug, Clone, Copy)]
pub struct ControlMapping {
    pub mechanism: SecurityMechanism,
    pub control: ControlId,
    pub implementation_status: ImplementationStatus,
}

/// Implementation status of a control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ImplementationStatus {
    /// Fully implemented in code.
    Implemented = 0,
    /// Partially implemented (stub or incomplete).
    Partial = 1,
    /// Planned but not yet implemented.
    Planned = 2,
    /// Not applicable to this deployment mode.
    NotApplicable = 3,
}

impl ImplementationStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Implemented => "implemented",
            Self::Partial => "partial",
            Self::Planned => "planned",
            Self::NotApplicable => "n/a",
        }
    }
}

// Well-known NIST control IDs referenced by SmallAIOS
pub const AC_3: ControlId = ControlId::new(ControlFamily::AccessControl, 3);
pub const AC_4: ControlId = ControlId::new(ControlFamily::AccessControl, 4);
pub const AC_6: ControlId = ControlId::new(ControlFamily::AccessControl, 6);
pub const AU_2: ControlId = ControlId::new(ControlFamily::Audit, 2);
pub const AU_3: ControlId = ControlId::new(ControlFamily::Audit, 3);
pub const AU_9: ControlId = ControlId::new(ControlFamily::Audit, 9);
pub const AU_12: ControlId = ControlId::new(ControlFamily::Audit, 12);
pub const CA_2: ControlId = ControlId::new(ControlFamily::Assessment, 2);
pub const CA_7: ControlId = ControlId::new(ControlFamily::Assessment, 7);
pub const CA_8: ControlId = ControlId::new(ControlFamily::Assessment, 8);
pub const IR_4: ControlId = ControlId::new(ControlFamily::IncidentResponse, 4);
pub const IR_5: ControlId = ControlId::new(ControlFamily::IncidentResponse, 5);
pub const SC_12: ControlId = ControlId::new(ControlFamily::SystemComm, 12);
pub const SC_13: ControlId = ControlId::new(ControlFamily::SystemComm, 13);
pub const SI_4: ControlId = ControlId::new(ControlFamily::SystemIntegrity, 4);
pub const SI_7: ControlId = ControlId::new(ControlFamily::SystemIntegrity, 7);
pub const SR_3: ControlId = ControlId::new(ControlFamily::SupplyChain, 3);
pub const SR_4: ControlId = ControlId::new(ControlFamily::SupplyChain, 4);

/// Complete cross-reference table of SmallAIOS mechanisms to NIST 800-53 controls.
pub const CONTROL_MAPPINGS: &[ControlMapping] = &[
    // Capability system -> Access Control
    ControlMapping {
        mechanism: SecurityMechanism::CapabilitySystem,
        control: AC_3,
        implementation_status: ImplementationStatus::Implemented,
    },
    ControlMapping {
        mechanism: SecurityMechanism::CapabilitySystem,
        control: AC_6,
        implementation_status: ImplementationStatus::Implemented,
    },
    // Information flow -> AC-4
    ControlMapping {
        mechanism: SecurityMechanism::InformationFlow,
        control: AC_4,
        implementation_status: ImplementationStatus::Implemented,
    },
    // Audit logging -> AU family
    ControlMapping {
        mechanism: SecurityMechanism::AuditLogging,
        control: AU_2,
        implementation_status: ImplementationStatus::Implemented,
    },
    ControlMapping {
        mechanism: SecurityMechanism::AuditLogging,
        control: AU_3,
        implementation_status: ImplementationStatus::Implemented,
    },
    ControlMapping {
        mechanism: SecurityMechanism::AuditLogging,
        control: AU_9,
        implementation_status: ImplementationStatus::Implemented,
    },
    ControlMapping {
        mechanism: SecurityMechanism::AuditLogging,
        control: AU_12,
        implementation_status: ImplementationStatus::Implemented,
    },
    // Monitoring -> CA, SI
    ControlMapping {
        mechanism: SecurityMechanism::Monitoring,
        control: CA_7,
        implementation_status: ImplementationStatus::Implemented,
    },
    ControlMapping {
        mechanism: SecurityMechanism::Monitoring,
        control: SI_4,
        implementation_status: ImplementationStatus::Implemented,
    },
    // Incident response -> IR
    ControlMapping {
        mechanism: SecurityMechanism::IncidentResponse,
        control: IR_4,
        implementation_status: ImplementationStatus::Implemented,
    },
    ControlMapping {
        mechanism: SecurityMechanism::IncidentResponse,
        control: IR_5,
        implementation_status: ImplementationStatus::Implemented,
    },
    // Cryptography -> SC
    ControlMapping {
        mechanism: SecurityMechanism::Cryptography,
        control: SC_13,
        implementation_status: ImplementationStatus::Partial,
    },
    // Key management -> SC-12
    ControlMapping {
        mechanism: SecurityMechanism::KeyManagement,
        control: SC_12,
        implementation_status: ImplementationStatus::Implemented,
    },
    // Supply chain -> SR
    ControlMapping {
        mechanism: SecurityMechanism::SupplyChain,
        control: SR_3,
        implementation_status: ImplementationStatus::Implemented,
    },
    ControlMapping {
        mechanism: SecurityMechanism::SupplyChain,
        control: SR_4,
        implementation_status: ImplementationStatus::Implemented,
    },
    // OT security -> SI-7 (software integrity)
    ControlMapping {
        mechanism: SecurityMechanism::OtSecurity,
        control: SI_7,
        implementation_status: ImplementationStatus::Implemented,
    },
];

/// Look up all controls satisfied by a given mechanism.
pub fn controls_for_mechanism(
    mechanism: SecurityMechanism,
) -> impl Iterator<Item = &'static ControlMapping> {
    CONTROL_MAPPINGS
        .iter()
        .filter(move |m| m.mechanism == mechanism)
}

/// Look up all mechanisms that satisfy a given control.
pub fn mechanisms_for_control(control: ControlId) -> impl Iterator<Item = &'static ControlMapping> {
    CONTROL_MAPPINGS
        .iter()
        .filter(move |m| m.control == control)
}

/// Count of implemented controls.
pub fn implemented_count() -> usize {
    CONTROL_MAPPINGS
        .iter()
        .filter(|m| m.implementation_status == ImplementationStatus::Implemented)
        .count()
}

/// Count of partial/planned controls.
pub fn pending_count() -> usize {
    CONTROL_MAPPINGS
        .iter()
        .filter(|m| {
            matches!(
                m.implementation_status,
                ImplementationStatus::Partial | ImplementationStatus::Planned
            )
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_family_codes() {
        assert_eq!(ControlFamily::AccessControl.code(), "AC");
        assert_eq!(ControlFamily::Audit.code(), "AU");
        assert_eq!(ControlFamily::SystemComm.code(), "SC");
        assert_eq!(ControlFamily::SupplyChain.code(), "SR");
    }

    #[test]
    fn control_mappings_not_empty() {
        assert!(!CONTROL_MAPPINGS.is_empty());
        assert!(CONTROL_MAPPINGS.len() >= 15);
    }

    #[test]
    fn capability_maps_to_ac3_ac6() {
        let caps: alloc::vec::Vec<_> =
            controls_for_mechanism(SecurityMechanism::CapabilitySystem).collect();
        assert!(caps.iter().any(|m| m.control == AC_3));
        assert!(caps.iter().any(|m| m.control == AC_6));
    }

    #[test]
    fn audit_maps_to_au2_au12() {
        let audits: alloc::vec::Vec<_> =
            controls_for_mechanism(SecurityMechanism::AuditLogging).collect();
        assert!(audits.iter().any(|m| m.control == AU_2));
        assert!(audits.iter().any(|m| m.control == AU_12));
    }

    #[test]
    fn info_flow_maps_to_ac4() {
        let flows: alloc::vec::Vec<_> =
            controls_for_mechanism(SecurityMechanism::InformationFlow).collect();
        assert!(flows.iter().any(|m| m.control == AC_4));
    }

    #[test]
    fn key_management_maps_to_sc12() {
        let keys: alloc::vec::Vec<_> =
            controls_for_mechanism(SecurityMechanism::KeyManagement).collect();
        assert!(keys.iter().any(|m| m.control == SC_12));
    }

    #[test]
    fn supply_chain_maps_to_sr() {
        let sc: alloc::vec::Vec<_> =
            controls_for_mechanism(SecurityMechanism::SupplyChain).collect();
        assert!(sc.iter().any(|m| m.control == SR_3));
        assert!(sc.iter().any(|m| m.control == SR_4));
    }

    #[test]
    fn mechanisms_for_ac3() {
        let mechs: alloc::vec::Vec<_> = mechanisms_for_control(AC_3).collect();
        assert_eq!(mechs.len(), 1);
        assert_eq!(mechs[0].mechanism, SecurityMechanism::CapabilitySystem);
    }

    #[test]
    fn implemented_count_positive() {
        assert!(implemented_count() > 0);
    }

    #[test]
    fn implementation_status_strings() {
        assert_eq!(ImplementationStatus::Implemented.as_str(), "implemented");
        assert_eq!(ImplementationStatus::Partial.as_str(), "partial");
        assert_eq!(ImplementationStatus::Planned.as_str(), "planned");
        assert_eq!(ImplementationStatus::NotApplicable.as_str(), "n/a");
    }

    #[test]
    fn security_mechanism_strings() {
        assert_eq!(
            SecurityMechanism::CapabilitySystem.as_str(),
            "capability-system"
        );
        assert_eq!(SecurityMechanism::AuditLogging.as_str(), "audit-logging");
        assert_eq!(SecurityMechanism::KeyManagement.as_str(), "key-management");
    }

    #[test]
    fn security_mechanism_strings_all() {
        assert_eq!(SecurityMechanism::Cryptography.as_str(), "cryptography");
        assert_eq!(
            SecurityMechanism::InformationFlow.as_str(),
            "information-flow"
        );
        assert_eq!(SecurityMechanism::Monitoring.as_str(), "monitoring");
        assert_eq!(
            SecurityMechanism::IncidentResponse.as_str(),
            "incident-response"
        );
        assert_eq!(SecurityMechanism::SupplyChain.as_str(), "supply-chain");
        assert_eq!(SecurityMechanism::OtSecurity.as_str(), "ot-security");
    }

    #[test]
    fn control_family_names_all() {
        assert_eq!(ControlFamily::AccessControl.name(), "Access Control");
        assert_eq!(ControlFamily::Audit.name(), "Audit and Accountability");
        assert_eq!(
            ControlFamily::Assessment.name(),
            "Assessment, Authorization, and Monitoring"
        );
        assert_eq!(
            ControlFamily::ConfigManagement.name(),
            "Configuration Management"
        );
        assert_eq!(ControlFamily::Contingency.name(), "Contingency Planning");
        assert_eq!(
            ControlFamily::IdAuth.name(),
            "Identification and Authentication"
        );
        assert_eq!(ControlFamily::IncidentResponse.name(), "Incident Response");
        assert_eq!(
            ControlFamily::SystemComm.name(),
            "System and Communications Protection"
        );
        assert_eq!(
            ControlFamily::SystemIntegrity.name(),
            "System and Information Integrity"
        );
        assert_eq!(
            ControlFamily::SupplyChain.name(),
            "Supply Chain Risk Management"
        );
    }

    #[test]
    fn control_family_codes_all() {
        assert_eq!(ControlFamily::Assessment.code(), "CA");
        assert_eq!(ControlFamily::ConfigManagement.code(), "CM");
        assert_eq!(ControlFamily::Contingency.code(), "CP");
        assert_eq!(ControlFamily::IdAuth.code(), "IA");
        assert_eq!(ControlFamily::IncidentResponse.code(), "IR");
        assert_eq!(ControlFamily::SystemIntegrity.code(), "SI");
    }

    #[test]
    fn pending_count_value() {
        let count = pending_count();
        assert!(count >= 1); // SC-13 is Partial
    }

    #[test]
    fn mechanisms_for_various_controls() {
        let si4: alloc::vec::Vec<_> = mechanisms_for_control(SI_4).collect();
        assert_eq!(si4.len(), 1);
        assert_eq!(si4[0].mechanism, SecurityMechanism::Monitoring);

        let au9: alloc::vec::Vec<_> = mechanisms_for_control(AU_9).collect();
        assert_eq!(au9.len(), 1);
        assert_eq!(au9[0].mechanism, SecurityMechanism::AuditLogging);

        let ir4: alloc::vec::Vec<_> = mechanisms_for_control(IR_4).collect();
        assert_eq!(ir4.len(), 1);
        assert_eq!(ir4[0].mechanism, SecurityMechanism::IncidentResponse);
    }

    #[test]
    fn monitoring_maps_to_ca7_si4() {
        let mon: alloc::vec::Vec<_> =
            controls_for_mechanism(SecurityMechanism::Monitoring).collect();
        assert!(mon.iter().any(|m| m.control == CA_7));
        assert!(mon.iter().any(|m| m.control == SI_4));
    }

    #[test]
    fn ot_security_maps_to_si7() {
        let ot: alloc::vec::Vec<_> =
            controls_for_mechanism(SecurityMechanism::OtSecurity).collect();
        assert!(ot.iter().any(|m| m.control == SI_7));
    }

    #[test]
    fn crypto_maps_to_sc13_partial() {
        let crypto: alloc::vec::Vec<_> =
            controls_for_mechanism(SecurityMechanism::Cryptography).collect();
        assert!(crypto.iter().any(
            |m| m.control == SC_13 && m.implementation_status == ImplementationStatus::Partial
        ));
    }

    #[test]
    fn incident_response_maps_to_ir4_ir5() {
        let ir: alloc::vec::Vec<_> =
            controls_for_mechanism(SecurityMechanism::IncidentResponse).collect();
        assert!(ir.iter().any(|m| m.control == IR_4));
        assert!(ir.iter().any(|m| m.control == IR_5));
    }

    #[test]
    fn control_id_construction() {
        let id = ControlId::new(ControlFamily::Contingency, 1);
        assert_eq!(id.family, ControlFamily::Contingency);
        assert_eq!(id.number, 1);
    }
}
