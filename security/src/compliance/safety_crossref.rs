// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Safety-critical cross-reference tables.
//!
//! Maps kernel components to:
//! - DO-178C DAL (Design Assurance Level) A through E
//! - IEC 61508 SIL (Safety Integrity Level) 1 through 4
//! - ISO 26262 ASIL (Automotive SIL) A through D
//!
//! Also cross-references DO-178C verification objectives with
//! NIST SP 800-53 CA (Assessment) controls.

/// DO-178C Design Assurance Level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum DalLevel {
    /// Level E: no safety effect.
    E = 0,
    /// Level D: minor effect.
    D = 1,
    /// Level C: major effect.
    C = 2,
    /// Level B: hazardous effect.
    B = 3,
    /// Level A: catastrophic effect.
    A = 4,
}

impl DalLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::E => "DAL E",
            Self::D => "DAL D",
            Self::C => "DAL C",
            Self::B => "DAL B",
            Self::A => "DAL A",
        }
    }

    /// Required MC/DC coverage percentage for this DAL.
    pub fn mcdc_coverage_pct(&self) -> u32 {
        match self {
            Self::A => 100,
            Self::B => 100,
            Self::C => 100, // Statement + decision
            Self::D => 0,   // No structural coverage required
            Self::E => 0,
        }
    }
}

/// IEC 61508 Safety Integrity Level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SilLevel {
    /// No safety requirement.
    None = 0,
    /// SIL 1: lowest safety integrity.
    Sil1 = 1,
    /// SIL 2.
    Sil2 = 2,
    /// SIL 3.
    Sil3 = 3,
    /// SIL 4: highest safety integrity.
    Sil4 = 4,
}

impl SilLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "N/A",
            Self::Sil1 => "SIL 1",
            Self::Sil2 => "SIL 2",
            Self::Sil3 => "SIL 3",
            Self::Sil4 => "SIL 4",
        }
    }
}

/// ISO 26262 Automotive Safety Integrity Level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum AsilLevel {
    /// Quality Management (no ASIL).
    Qm = 0,
    /// ASIL A: lowest automotive safety level.
    A = 1,
    /// ASIL B.
    B = 2,
    /// ASIL C.
    C = 3,
    /// ASIL D: highest automotive safety level.
    D = 4,
}

impl AsilLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Qm => "QM",
            Self::A => "ASIL A",
            Self::B => "ASIL B",
            Self::C => "ASIL C",
            Self::D => "ASIL D",
        }
    }
}

/// Kernel component for safety classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KernelComponent {
    /// Memory allocator (buddy + slab).
    MemoryAllocator = 0,
    /// Task scheduler.
    Scheduler = 1,
    /// Syscall dispatch.
    SyscallDispatch = 2,
    /// Capability system.
    CapabilitySystem = 3,
    /// Cryptography module.
    CryptoModule = 4,
    /// ONNX inference runtime.
    OnnxRuntime = 5,
    /// Network stack.
    NetworkStack = 6,
    /// IPC messaging.
    IpcMessaging = 7,
    /// Audit subsystem.
    AuditSubsystem = 8,
    /// Watchdog / safe shutdown.
    Watchdog = 9,
}

impl KernelComponent {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MemoryAllocator => "memory-allocator",
            Self::Scheduler => "scheduler",
            Self::SyscallDispatch => "syscall-dispatch",
            Self::CapabilitySystem => "capability-system",
            Self::CryptoModule => "crypto-module",
            Self::OnnxRuntime => "onnx-runtime",
            Self::NetworkStack => "network-stack",
            Self::IpcMessaging => "ipc-messaging",
            Self::AuditSubsystem => "audit-subsystem",
            Self::Watchdog => "watchdog",
        }
    }
}

/// Cross-reference entry mapping a component to all three safety standards.
#[derive(Debug, Clone, Copy)]
pub struct SafetyCrossRef {
    pub component: KernelComponent,
    pub dal: DalLevel,
    pub sil: SilLevel,
    pub asil: AsilLevel,
    /// Required MC/DC coverage percentage (0-100).
    pub mcdc_target_pct: u32,
}

/// Complete safety cross-reference table for all kernel components.
pub const SAFETY_CROSSREF: &[SafetyCrossRef] = &[
    SafetyCrossRef {
        component: KernelComponent::MemoryAllocator,
        dal: DalLevel::A,
        sil: SilLevel::Sil4,
        asil: AsilLevel::D,
        mcdc_target_pct: 100,
    },
    SafetyCrossRef {
        component: KernelComponent::Scheduler,
        dal: DalLevel::A,
        sil: SilLevel::Sil4,
        asil: AsilLevel::D,
        mcdc_target_pct: 100,
    },
    SafetyCrossRef {
        component: KernelComponent::SyscallDispatch,
        dal: DalLevel::A,
        sil: SilLevel::Sil3,
        asil: AsilLevel::D,
        mcdc_target_pct: 100,
    },
    SafetyCrossRef {
        component: KernelComponent::CapabilitySystem,
        dal: DalLevel::A,
        sil: SilLevel::Sil4,
        asil: AsilLevel::D,
        mcdc_target_pct: 100,
    },
    SafetyCrossRef {
        component: KernelComponent::CryptoModule,
        dal: DalLevel::A,
        sil: SilLevel::Sil3,
        asil: AsilLevel::C,
        mcdc_target_pct: 100,
    },
    SafetyCrossRef {
        component: KernelComponent::OnnxRuntime,
        dal: DalLevel::B,
        sil: SilLevel::Sil2,
        asil: AsilLevel::B,
        mcdc_target_pct: 100,
    },
    SafetyCrossRef {
        component: KernelComponent::NetworkStack,
        dal: DalLevel::C,
        sil: SilLevel::Sil2,
        asil: AsilLevel::B,
        mcdc_target_pct: 100,
    },
    SafetyCrossRef {
        component: KernelComponent::IpcMessaging,
        dal: DalLevel::B,
        sil: SilLevel::Sil2,
        asil: AsilLevel::B,
        mcdc_target_pct: 100,
    },
    SafetyCrossRef {
        component: KernelComponent::AuditSubsystem,
        dal: DalLevel::B,
        sil: SilLevel::Sil2,
        asil: AsilLevel::B,
        mcdc_target_pct: 100,
    },
    SafetyCrossRef {
        component: KernelComponent::Watchdog,
        dal: DalLevel::A,
        sil: SilLevel::Sil4,
        asil: AsilLevel::D,
        mcdc_target_pct: 100,
    },
];

/// DO-178C verification objective to NIST 800-53 CA control mapping.
#[derive(Debug, Clone, Copy)]
pub struct VerificationCrossRef {
    /// DO-178C objective identifier.
    pub do178c_objective: &'static str,
    /// NIST 800-53 control (CA family).
    pub nist_control: &'static str,
    /// Description of the cross-reference.
    pub description: &'static str,
}

/// Cross-reference of DO-178C verification objectives to NIST CA controls.
pub const VERIFICATION_CROSSREF: &[VerificationCrossRef] = &[
    VerificationCrossRef {
        do178c_objective: "A-7.1",
        nist_control: "CA-2",
        description: "Requirements-based test coverage maps to security assessment",
    },
    VerificationCrossRef {
        do178c_objective: "A-7.2",
        nist_control: "CA-2",
        description: "Structural coverage analysis maps to assessment methodology",
    },
    VerificationCrossRef {
        do178c_objective: "A-7.3",
        nist_control: "CA-7",
        description: "Robustness testing maps to continuous monitoring",
    },
    VerificationCrossRef {
        do178c_objective: "A-7.5",
        nist_control: "CA-8",
        description: "Integration testing maps to penetration testing",
    },
    VerificationCrossRef {
        do178c_objective: "A-7.6",
        nist_control: "CA-7",
        description: "Requirements traceability maps to ongoing assessment",
    },
];

/// Look up the safety classification for a component.
pub fn classification_for(component: KernelComponent) -> Option<&'static SafetyCrossRef> {
    SAFETY_CROSSREF.iter().find(|r| r.component == component)
}

/// Count of components at DAL A.
pub fn dal_a_count() -> usize {
    SAFETY_CROSSREF
        .iter()
        .filter(|r| r.dal == DalLevel::A)
        .count()
}

/// Get all components requiring 100% MC/DC coverage.
pub fn full_mcdc_components() -> impl Iterator<Item = &'static SafetyCrossRef> {
    SAFETY_CROSSREF.iter().filter(|r| r.mcdc_target_pct == 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossref_table_not_empty() {
        assert!(!SAFETY_CROSSREF.is_empty());
        assert_eq!(SAFETY_CROSSREF.len(), 10);
    }

    #[test]
    fn memory_allocator_dal_a() {
        let r = classification_for(KernelComponent::MemoryAllocator).unwrap();
        assert_eq!(r.dal, DalLevel::A);
        assert_eq!(r.sil, SilLevel::Sil4);
        assert_eq!(r.asil, AsilLevel::D);
    }

    #[test]
    fn scheduler_dal_a() {
        let r = classification_for(KernelComponent::Scheduler).unwrap();
        assert_eq!(r.dal, DalLevel::A);
    }

    #[test]
    fn capability_dal_a() {
        let r = classification_for(KernelComponent::CapabilitySystem).unwrap();
        assert_eq!(r.dal, DalLevel::A);
        assert_eq!(r.mcdc_target_pct, 100);
    }

    #[test]
    fn crypto_dal_a() {
        let r = classification_for(KernelComponent::CryptoModule).unwrap();
        assert_eq!(r.dal, DalLevel::A);
        assert_eq!(r.mcdc_target_pct, 100);
    }

    #[test]
    fn onnx_runtime_dal_b() {
        let r = classification_for(KernelComponent::OnnxRuntime).unwrap();
        assert_eq!(r.dal, DalLevel::B);
    }

    #[test]
    fn dal_a_components_counted() {
        let count = dal_a_count();
        assert!(count >= 4); // At least allocator, scheduler, capability, crypto, syscall, watchdog
    }

    #[test]
    fn all_components_require_mcdc() {
        let count = full_mcdc_components().count();
        assert_eq!(count, SAFETY_CROSSREF.len());
    }

    #[test]
    fn verification_crossref_not_empty() {
        assert!(!VERIFICATION_CROSSREF.is_empty());
        assert!(VERIFICATION_CROSSREF.len() >= 5);
    }

    #[test]
    fn dal_level_ordering() {
        assert!(DalLevel::E < DalLevel::D);
        assert!(DalLevel::D < DalLevel::C);
        assert!(DalLevel::C < DalLevel::B);
        assert!(DalLevel::B < DalLevel::A);
    }

    #[test]
    fn sil_level_ordering() {
        assert!(SilLevel::None < SilLevel::Sil1);
        assert!(SilLevel::Sil1 < SilLevel::Sil4);
    }

    #[test]
    fn asil_level_ordering() {
        assert!(AsilLevel::Qm < AsilLevel::A);
        assert!(AsilLevel::A < AsilLevel::D);
    }

    #[test]
    fn dal_strings() {
        assert_eq!(DalLevel::A.as_str(), "DAL A");
        assert_eq!(DalLevel::E.as_str(), "DAL E");
    }

    #[test]
    fn sil_strings() {
        assert_eq!(SilLevel::Sil4.as_str(), "SIL 4");
        assert_eq!(SilLevel::None.as_str(), "N/A");
    }

    #[test]
    fn asil_strings() {
        assert_eq!(AsilLevel::D.as_str(), "ASIL D");
        assert_eq!(AsilLevel::Qm.as_str(), "QM");
    }

    #[test]
    fn dal_mcdc_requirements() {
        assert_eq!(DalLevel::A.mcdc_coverage_pct(), 100);
        assert_eq!(DalLevel::B.mcdc_coverage_pct(), 100);
        assert_eq!(DalLevel::D.mcdc_coverage_pct(), 0);
        assert_eq!(DalLevel::E.mcdc_coverage_pct(), 0);
    }

    #[test]
    fn component_strings() {
        assert_eq!(
            KernelComponent::MemoryAllocator.as_str(),
            "memory-allocator"
        );
        assert_eq!(KernelComponent::Scheduler.as_str(), "scheduler");
        assert_eq!(KernelComponent::Watchdog.as_str(), "watchdog");
    }
}
