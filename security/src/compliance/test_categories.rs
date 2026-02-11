// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Security test category definitions.
//!
//! Defines structured categories for security testing per
//! NIST SP 800-53 CA-8 (penetration testing) and SA-11 (developer testing):
//! - Capability bypass testing
//! - Crypto validation (NIST test vectors)
//! - Timing side-channel (dudect-style)
//! - Resource exhaustion

/// Security test categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SecurityTestCategory {
    /// Capability bypass: attempt to access resources without proper capabilities.
    CapabilityBypass = 0,
    /// Cryptographic validation: NIST test vectors, KATs, algorithm correctness.
    CryptoValidation = 1,
    /// Timing side-channel: dudect-style statistical testing for constant-time.
    TimingSideChannel = 2,
    /// Resource exhaustion: memory, file descriptors, capability slots, tasks.
    ResourceExhaustion = 3,
    /// Input fuzzing: malformed inputs to syscalls, IPC, ONNX parser.
    InputFuzzing = 4,
    /// Information flow: verify boundary enforcement blocks forbidden flows.
    InformationFlow = 5,
    /// Privilege escalation: attempt to gain higher privileges through delegation.
    PrivilegeEscalation = 6,
}

impl SecurityTestCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CapabilityBypass => "capability-bypass",
            Self::CryptoValidation => "crypto-validation",
            Self::TimingSideChannel => "timing-side-channel",
            Self::ResourceExhaustion => "resource-exhaustion",
            Self::InputFuzzing => "input-fuzzing",
            Self::InformationFlow => "information-flow",
            Self::PrivilegeEscalation => "privilege-escalation",
        }
    }

    /// NIST SP 800-53 control this test category supports.
    pub fn nist_control(&self) -> &'static str {
        match self {
            Self::CapabilityBypass => "AC-3, AC-6",
            Self::CryptoValidation => "SC-13",
            Self::TimingSideChannel => "SC-13",
            Self::ResourceExhaustion => "SI-10, SI-16",
            Self::InputFuzzing => "SI-10",
            Self::InformationFlow => "AC-4",
            Self::PrivilegeEscalation => "AC-6",
        }
    }

    /// Required test method.
    pub fn method(&self) -> TestMethod {
        match self {
            Self::CapabilityBypass => TestMethod::UnitAndIntegration,
            Self::CryptoValidation => TestMethod::KnownAnswerTest,
            Self::TimingSideChannel => TestMethod::StatisticalAnalysis,
            Self::ResourceExhaustion => TestMethod::StressTest,
            Self::InputFuzzing => TestMethod::Fuzz,
            Self::InformationFlow => TestMethod::UnitAndIntegration,
            Self::PrivilegeEscalation => TestMethod::UnitAndIntegration,
        }
    }
}

/// Test method types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TestMethod {
    /// Standard unit and integration tests.
    UnitAndIntegration = 0,
    /// Known Answer Test (KAT) with reference vectors.
    KnownAnswerTest = 1,
    /// Statistical analysis (e.g., dudect for timing).
    StatisticalAnalysis = 2,
    /// Stress testing with resource limits.
    StressTest = 3,
    /// Fuzzing with random/mutated inputs.
    Fuzz = 4,
}

impl TestMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UnitAndIntegration => "unit-and-integration",
            Self::KnownAnswerTest => "known-answer-test",
            Self::StatisticalAnalysis => "statistical-analysis",
            Self::StressTest => "stress-test",
            Self::Fuzz => "fuzz",
        }
    }
}

/// A security test requirement.
#[derive(Debug, Clone, Copy)]
pub struct SecurityTestRequirement {
    /// Test category.
    pub category: SecurityTestCategory,
    /// Target component/subsystem.
    pub target: &'static str,
    /// Specific test description.
    pub description: &'static str,
    /// Whether MC/DC coverage is required.
    pub mcdc_required: bool,
}

/// All defined security test requirements.
pub const SECURITY_TEST_REQUIREMENTS: &[SecurityTestRequirement] = &[
    // Capability bypass tests
    SecurityTestRequirement {
        category: SecurityTestCategory::CapabilityBypass,
        target: "capability-system",
        description: "Attempt resource access without capability",
        mcdc_required: true,
    },
    SecurityTestRequirement {
        category: SecurityTestCategory::CapabilityBypass,
        target: "capability-system",
        description: "Attempt access with expired capability",
        mcdc_required: true,
    },
    SecurityTestRequirement {
        category: SecurityTestCategory::CapabilityBypass,
        target: "capability-system",
        description: "Attempt access with revoked capability",
        mcdc_required: true,
    },
    // Crypto validation tests
    SecurityTestRequirement {
        category: SecurityTestCategory::CryptoValidation,
        target: "crypto-sha3",
        description: "SHA-3-256 NIST test vectors",
        mcdc_required: true,
    },
    SecurityTestRequirement {
        category: SecurityTestCategory::CryptoValidation,
        target: "crypto-aes-gcm",
        description: "AES-256-GCM NIST test vectors (when implemented)",
        mcdc_required: true,
    },
    SecurityTestRequirement {
        category: SecurityTestCategory::CryptoValidation,
        target: "crypto-ml-kem",
        description: "ML-KEM-768 known answer tests (when implemented)",
        mcdc_required: true,
    },
    SecurityTestRequirement {
        category: SecurityTestCategory::CryptoValidation,
        target: "crypto-ml-dsa",
        description: "ML-DSA-65 known answer tests (when implemented)",
        mcdc_required: true,
    },
    // Timing side-channel tests
    SecurityTestRequirement {
        category: SecurityTestCategory::TimingSideChannel,
        target: "crypto-constant-time",
        description: "Dudect-style statistical testing on comparison functions",
        mcdc_required: false,
    },
    // Resource exhaustion tests
    SecurityTestRequirement {
        category: SecurityTestCategory::ResourceExhaustion,
        target: "capability-registry",
        description: "Exhaust capability slot limit",
        mcdc_required: false,
    },
    SecurityTestRequirement {
        category: SecurityTestCategory::ResourceExhaustion,
        target: "key-manager",
        description: "Exhaust key registry slots",
        mcdc_required: false,
    },
    // Information flow tests
    SecurityTestRequirement {
        category: SecurityTestCategory::InformationFlow,
        target: "boundary",
        description: "Verify ONNX runtime cannot access network",
        mcdc_required: true,
    },
    SecurityTestRequirement {
        category: SecurityTestCategory::InformationFlow,
        target: "boundary",
        description: "Verify bus handler cannot access ONNX models",
        mcdc_required: true,
    },
    // Privilege escalation tests
    SecurityTestRequirement {
        category: SecurityTestCategory::PrivilegeEscalation,
        target: "capability-delegation",
        description: "Verify delegated permissions cannot exceed parent",
        mcdc_required: true,
    },
];

/// Count of test requirements by category.
pub fn count_by_category(category: SecurityTestCategory) -> usize {
    SECURITY_TEST_REQUIREMENTS
        .iter()
        .filter(|r| r.category == category)
        .count()
}

/// Count of test requirements requiring MC/DC coverage.
pub fn mcdc_required_count() -> usize {
    SECURITY_TEST_REQUIREMENTS
        .iter()
        .filter(|r| r.mcdc_required)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirements_not_empty() {
        assert!(!SECURITY_TEST_REQUIREMENTS.is_empty());
        assert!(SECURITY_TEST_REQUIREMENTS.len() >= 13);
    }

    #[test]
    fn capability_bypass_tests_exist() {
        assert!(count_by_category(SecurityTestCategory::CapabilityBypass) >= 3);
    }

    #[test]
    fn crypto_validation_tests_exist() {
        assert!(count_by_category(SecurityTestCategory::CryptoValidation) >= 4);
    }

    #[test]
    fn timing_tests_exist() {
        assert!(count_by_category(SecurityTestCategory::TimingSideChannel) >= 1);
    }

    #[test]
    fn resource_exhaustion_tests_exist() {
        assert!(count_by_category(SecurityTestCategory::ResourceExhaustion) >= 2);
    }

    #[test]
    fn information_flow_tests_exist() {
        assert!(count_by_category(SecurityTestCategory::InformationFlow) >= 2);
    }

    #[test]
    fn privilege_escalation_tests_exist() {
        assert!(count_by_category(SecurityTestCategory::PrivilegeEscalation) >= 1);
    }

    #[test]
    fn mcdc_required_count_positive() {
        assert!(mcdc_required_count() > 0);
    }

    #[test]
    fn category_strings() {
        assert_eq!(SecurityTestCategory::CapabilityBypass.as_str(), "capability-bypass");
        assert_eq!(SecurityTestCategory::CryptoValidation.as_str(), "crypto-validation");
        assert_eq!(SecurityTestCategory::TimingSideChannel.as_str(), "timing-side-channel");
        assert_eq!(SecurityTestCategory::ResourceExhaustion.as_str(), "resource-exhaustion");
    }

    #[test]
    fn category_nist_controls() {
        assert_eq!(SecurityTestCategory::CapabilityBypass.nist_control(), "AC-3, AC-6");
        assert_eq!(SecurityTestCategory::CryptoValidation.nist_control(), "SC-13");
        assert_eq!(SecurityTestCategory::InformationFlow.nist_control(), "AC-4");
    }

    #[test]
    fn category_methods() {
        assert_eq!(SecurityTestCategory::CryptoValidation.method(), TestMethod::KnownAnswerTest);
        assert_eq!(SecurityTestCategory::TimingSideChannel.method(), TestMethod::StatisticalAnalysis);
        assert_eq!(SecurityTestCategory::InputFuzzing.method(), TestMethod::Fuzz);
        assert_eq!(SecurityTestCategory::ResourceExhaustion.method(), TestMethod::StressTest);
    }

    #[test]
    fn method_strings() {
        assert_eq!(TestMethod::UnitAndIntegration.as_str(), "unit-and-integration");
        assert_eq!(TestMethod::KnownAnswerTest.as_str(), "known-answer-test");
        assert_eq!(TestMethod::Fuzz.as_str(), "fuzz");
    }
}
