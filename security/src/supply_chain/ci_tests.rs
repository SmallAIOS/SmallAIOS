// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CI integration test stubs for supply chain tooling.
//!
//! Provides test infrastructure for verifying:
//! - SBOM generation produces valid CycloneDX
//! - cargo-audit runs clean (no CVSS >= 7.0)
//! - Build attestation signature is verifiable
//! - Reproducible builds produce identical hashes

/// CI test categories for supply chain verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CiTestCategory {
    /// SBOM validation (CycloneDX schema conformance).
    SbomValidation = 0,
    /// Vulnerability scan (cargo-audit CVSS threshold).
    VulnerabilityScan = 1,
    /// Attestation verification (ML-DSA-65 signature check).
    AttestationVerification = 2,
    /// Reproducible build (hash comparison of two builds).
    ReproducibleBuild = 3,
    /// License compliance (all dependencies have approved licenses).
    LicenseCompliance = 4,
}

impl CiTestCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SbomValidation => "sbom-validation",
            Self::VulnerabilityScan => "vulnerability-scan",
            Self::AttestationVerification => "attestation-verification",
            Self::ReproducibleBuild => "reproducible-build",
            Self::LicenseCompliance => "license-compliance",
        }
    }

    /// Whether this test is blocking (must pass for release).
    pub fn is_blocking(&self) -> bool {
        match self {
            Self::SbomValidation => true,
            Self::VulnerabilityScan => true,
            Self::AttestationVerification => true,
            Self::ReproducibleBuild => true,
            Self::LicenseCompliance => true,
        }
    }
}

/// CI test result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiTestResult {
    /// Test passed.
    Pass,
    /// Test failed with a reason code.
    Fail(CiTestFailure),
    /// Test was skipped (not applicable to this build).
    Skipped,
}

/// CI test failure reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiTestFailure {
    /// SBOM did not validate against CycloneDX schema.
    SbomSchemaInvalid,
    /// SBOM has missing required fields.
    SbomMissingFields,
    /// Vulnerability with CVSS >= 7.0 found.
    HighCvssVulnerability,
    /// Attestation signature verification failed.
    SignatureInvalid,
    /// Builds produced different hashes.
    BuildNotReproducible,
    /// License not in approved list.
    UnapprovedLicense,
}

/// CI test suite result summary.
#[derive(Debug, Clone, Copy, Default)]
pub struct CiTestSummary {
    pub total: u32,
    pub passed: u32,
    pub failed: u32,
    pub skipped: u32,
}

impl CiTestSummary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a test result.
    pub fn record(&mut self, result: CiTestResult) {
        self.total += 1;
        match result {
            CiTestResult::Pass => self.passed += 1,
            CiTestResult::Fail(_) => self.failed += 1,
            CiTestResult::Skipped => self.skipped += 1,
        }
    }

    /// Whether all non-skipped tests passed.
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Whether the CI gate should pass (no blocking failures).
    pub fn gate_passes(&self) -> bool {
        self.all_passed()
    }
}

/// Stub: validate an SBOM against CycloneDX schema.
///
/// In production, this calls the CycloneDX validator. Here it
/// performs basic structural checks.
pub fn validate_sbom_stub(has_tool_name: bool, has_components: bool) -> CiTestResult {
    if !has_tool_name {
        return CiTestResult::Fail(CiTestFailure::SbomMissingFields);
    }
    if !has_components {
        return CiTestResult::Fail(CiTestFailure::SbomSchemaInvalid);
    }
    CiTestResult::Pass
}

/// Stub: run cargo-audit and check results.
pub fn audit_scan_stub(max_cvss_10x: u32) -> CiTestResult {
    if max_cvss_10x >= 70 {
        CiTestResult::Fail(CiTestFailure::HighCvssVulnerability)
    } else {
        CiTestResult::Pass
    }
}

/// Stub: verify attestation signature.
pub fn verify_attestation_stub(is_signed: bool, signature_valid: bool) -> CiTestResult {
    if !is_signed || !signature_valid {
        CiTestResult::Fail(CiTestFailure::SignatureInvalid)
    } else {
        CiTestResult::Pass
    }
}

/// Stub: verify reproducible build.
pub fn verify_reproducible_build_stub(hashes_match: bool) -> CiTestResult {
    if hashes_match {
        CiTestResult::Pass
    } else {
        CiTestResult::Fail(CiTestFailure::BuildNotReproducible)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_strings() {
        assert_eq!(CiTestCategory::SbomValidation.as_str(), "sbom-validation");
        assert_eq!(CiTestCategory::VulnerabilityScan.as_str(), "vulnerability-scan");
        assert_eq!(CiTestCategory::ReproducibleBuild.as_str(), "reproducible-build");
    }

    #[test]
    fn all_categories_blocking() {
        assert!(CiTestCategory::SbomValidation.is_blocking());
        assert!(CiTestCategory::VulnerabilityScan.is_blocking());
        assert!(CiTestCategory::AttestationVerification.is_blocking());
        assert!(CiTestCategory::ReproducibleBuild.is_blocking());
        assert!(CiTestCategory::LicenseCompliance.is_blocking());
    }

    #[test]
    fn sbom_validation_pass() {
        assert_eq!(validate_sbom_stub(true, true), CiTestResult::Pass);
    }

    #[test]
    fn sbom_validation_missing_tool() {
        assert_eq!(
            validate_sbom_stub(false, true),
            CiTestResult::Fail(CiTestFailure::SbomMissingFields)
        );
    }

    #[test]
    fn sbom_validation_no_components() {
        assert_eq!(
            validate_sbom_stub(true, false),
            CiTestResult::Fail(CiTestFailure::SbomSchemaInvalid)
        );
    }

    #[test]
    fn audit_scan_pass() {
        assert_eq!(audit_scan_stub(50), CiTestResult::Pass);
    }

    #[test]
    fn audit_scan_fail_high_cvss() {
        assert_eq!(
            audit_scan_stub(95),
            CiTestResult::Fail(CiTestFailure::HighCvssVulnerability)
        );
    }

    #[test]
    fn audit_scan_boundary() {
        assert_eq!(
            audit_scan_stub(70),
            CiTestResult::Fail(CiTestFailure::HighCvssVulnerability)
        );
        assert_eq!(audit_scan_stub(69), CiTestResult::Pass);
    }

    #[test]
    fn attestation_pass() {
        assert_eq!(verify_attestation_stub(true, true), CiTestResult::Pass);
    }

    #[test]
    fn attestation_unsigned() {
        assert_eq!(
            verify_attestation_stub(false, true),
            CiTestResult::Fail(CiTestFailure::SignatureInvalid)
        );
    }

    #[test]
    fn attestation_invalid_sig() {
        assert_eq!(
            verify_attestation_stub(true, false),
            CiTestResult::Fail(CiTestFailure::SignatureInvalid)
        );
    }

    #[test]
    fn reproducible_build_match() {
        assert_eq!(verify_reproducible_build_stub(true), CiTestResult::Pass);
    }

    #[test]
    fn reproducible_build_mismatch() {
        assert_eq!(
            verify_reproducible_build_stub(false),
            CiTestResult::Fail(CiTestFailure::BuildNotReproducible)
        );
    }

    #[test]
    fn summary_all_pass() {
        let mut summary = CiTestSummary::new();
        summary.record(CiTestResult::Pass);
        summary.record(CiTestResult::Pass);
        summary.record(CiTestResult::Skipped);
        assert!(summary.all_passed());
        assert!(summary.gate_passes());
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.skipped, 1);
    }

    #[test]
    fn summary_with_failure() {
        let mut summary = CiTestSummary::new();
        summary.record(CiTestResult::Pass);
        summary.record(CiTestResult::Fail(CiTestFailure::HighCvssVulnerability));
        assert!(!summary.all_passed());
        assert!(!summary.gate_passes());
        assert_eq!(summary.failed, 1);
    }
}
