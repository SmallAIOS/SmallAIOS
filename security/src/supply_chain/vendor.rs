// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Vendor assessment checklist data model.
//!
//! Provides structures for tracking hardware component vendors,
//! their security posture, firmware update processes, and lifecycle status.
//! Used for NIST SP 800-53 SR (Supply Chain Risk Management) controls.

use alloc::string::String;
use alloc::vec::Vec;

/// Maximum vendor assessments tracked.
const MAX_ASSESSMENTS: usize = 64;

/// Hardware component category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ComponentCategory {
    /// Graphics processing unit.
    Gpu = 0,
    /// Bus transceiver (CAN, ARINC, etc.).
    BusTransceiver = 1,
    /// Field-programmable gate array.
    Fpga = 2,
    /// System on chip.
    Soc = 3,
    /// Network interface controller.
    Nic = 4,
    /// Trusted Platform Module.
    Tpm = 5,
    /// Storage controller.
    StorageController = 6,
    /// Other hardware component.
    Other = 7,
}

impl ComponentCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gpu => "GPU",
            Self::BusTransceiver => "Bus Transceiver",
            Self::Fpga => "FPGA",
            Self::Soc => "SoC",
            Self::Nic => "NIC",
            Self::Tpm => "TPM",
            Self::StorageController => "Storage Controller",
            Self::Other => "Other",
        }
    }
}

/// Lifecycle status of a hardware component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LifecycleStatus {
    /// Actively supported with security patches.
    Active = 0,
    /// Reduced support, critical patches only.
    Maintenance = 1,
    /// End-of-life announced, deadline known.
    EndOfLifeAnnounced = 2,
    /// End-of-life reached, no further patches.
    EndOfLife = 3,
}

impl LifecycleStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Maintenance => "maintenance",
            Self::EndOfLifeAnnounced => "eol-announced",
            Self::EndOfLife => "end-of-life",
        }
    }

    /// Whether this status is still receiving security patches.
    pub fn receives_patches(&self) -> bool {
        matches!(self, Self::Active | Self::Maintenance)
    }
}

/// Firmware update capability assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FirmwareUpdateCapability {
    /// Signed, authenticated firmware updates supported.
    SignedUpdates = 0,
    /// Unsigned updates only (security concern).
    UnsignedUpdates = 1,
    /// No firmware update mechanism available.
    NoUpdateMechanism = 2,
    /// Unknown / not assessed.
    Unknown = 3,
}

impl FirmwareUpdateCapability {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SignedUpdates => "signed-updates",
            Self::UnsignedUpdates => "unsigned-updates",
            Self::NoUpdateMechanism => "no-update-mechanism",
            Self::Unknown => "unknown",
        }
    }

    /// Whether the firmware update process is considered secure.
    pub fn is_secure(&self) -> bool {
        matches!(self, Self::SignedUpdates)
    }
}

/// Risk level from vendor assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum VendorRisk {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl VendorRisk {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// Vendor assessment checklist for a hardware component.
#[derive(Debug, Clone)]
pub struct VendorAssessment {
    /// Component name / model number.
    pub component_name: String,
    /// Vendor / manufacturer name.
    pub vendor_name: String,
    /// Hardware component category.
    pub category: ComponentCategory,
    /// Country of origin (ISO 3166-1 alpha-2).
    pub country_of_origin: String,
    /// Firmware update capability.
    pub firmware_update: FirmwareUpdateCapability,
    /// Known CVE count for this component.
    pub known_cve_count: u32,
    /// Lifecycle status.
    pub lifecycle: LifecycleStatus,
    /// End-of-life date (year * 10000 + month * 100 + day, e.g., 20271231).
    /// Zero if not announced.
    pub eol_date: u32,
    /// Date of last security assessment (same format as eol_date).
    pub last_assessment_date: u32,
    /// Free-form notes from the assessor.
    pub notes: String,
    /// Computed risk level.
    pub risk_level: VendorRisk,
}

impl VendorAssessment {
    /// Create a new assessment with required fields.
    pub fn new(component_name: String, vendor_name: String, category: ComponentCategory) -> Self {
        Self {
            component_name,
            vendor_name,
            category,
            country_of_origin: String::new(),
            firmware_update: FirmwareUpdateCapability::Unknown,
            known_cve_count: 0,
            lifecycle: LifecycleStatus::Active,
            eol_date: 0,
            last_assessment_date: 0,
            notes: String::new(),
            risk_level: VendorRisk::Medium, // default until assessed
        }
    }

    /// Compute the risk level based on assessment fields.
    ///
    /// Risk factors:
    /// - End-of-life: Critical
    /// - No firmware updates + known CVEs: High
    /// - Unsigned firmware updates: Medium
    /// - Active lifecycle + signed updates + no CVEs: Low
    pub fn compute_risk(&mut self) {
        self.risk_level = if self.lifecycle >= LifecycleStatus::EndOfLife {
            VendorRisk::Critical
        } else if !self.firmware_update.is_secure() && self.known_cve_count > 0 {
            VendorRisk::High
        } else if !self.firmware_update.is_secure()
            || self.lifecycle >= LifecycleStatus::EndOfLifeAnnounced
        {
            VendorRisk::Medium
        } else {
            VendorRisk::Low
        };
    }

    /// Whether this assessment has any security concerns.
    pub fn has_concerns(&self) -> bool {
        self.risk_level >= VendorRisk::Medium
    }
}

/// Registry of vendor assessments.
#[derive(Debug)]
pub struct VendorRegistry {
    assessments: Vec<VendorAssessment>,
}

impl VendorRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            assessments: Vec::new(),
        }
    }

    /// Add an assessment. Returns false if limit reached.
    pub fn add(&mut self, assessment: VendorAssessment) -> bool {
        if self.assessments.len() >= MAX_ASSESSMENTS {
            return false;
        }
        self.assessments.push(assessment);
        true
    }

    /// Number of assessments.
    pub fn count(&self) -> usize {
        self.assessments.len()
    }

    /// Find assessments by vendor name.
    pub fn by_vendor(&self, vendor: &str) -> Vec<&VendorAssessment> {
        self.assessments
            .iter()
            .filter(|a| a.vendor_name == vendor)
            .collect()
    }

    /// Find assessments by category.
    pub fn by_category(&self, category: ComponentCategory) -> Vec<&VendorAssessment> {
        self.assessments
            .iter()
            .filter(|a| a.category == category)
            .collect()
    }

    /// Get all assessments at or above a given risk level.
    pub fn by_min_risk(&self, min_risk: VendorRisk) -> Vec<&VendorAssessment> {
        self.assessments
            .iter()
            .filter(|a| a.risk_level >= min_risk)
            .collect()
    }

    /// Count of end-of-life components (requires action).
    pub fn eol_count(&self) -> usize {
        self.assessments
            .iter()
            .filter(|a| a.lifecycle >= LifecycleStatus::EndOfLife)
            .count()
    }

    /// All assessments.
    pub fn all(&self) -> &[VendorAssessment] {
        &self.assessments
    }
}

impl Default for VendorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_assessment(name: &str, vendor: &str) -> VendorAssessment {
        VendorAssessment::new(
            String::from(name),
            String::from(vendor),
            ComponentCategory::Gpu,
        )
    }

    #[test]
    fn create_assessment() {
        let a = sample_assessment("RTX 4090", "NVIDIA");
        assert_eq!(a.component_name, "RTX 4090");
        assert_eq!(a.vendor_name, "NVIDIA");
        assert_eq!(a.category, ComponentCategory::Gpu);
        assert_eq!(a.risk_level, VendorRisk::Medium);
    }

    #[test]
    fn compute_risk_low() {
        let mut a = sample_assessment("Test", "Vendor");
        a.firmware_update = FirmwareUpdateCapability::SignedUpdates;
        a.lifecycle = LifecycleStatus::Active;
        a.known_cve_count = 0;
        a.compute_risk();
        assert_eq!(a.risk_level, VendorRisk::Low);
    }

    #[test]
    fn compute_risk_medium_unsigned_firmware() {
        let mut a = sample_assessment("Test", "Vendor");
        a.firmware_update = FirmwareUpdateCapability::UnsignedUpdates;
        a.lifecycle = LifecycleStatus::Active;
        a.known_cve_count = 0;
        a.compute_risk();
        assert_eq!(a.risk_level, VendorRisk::Medium);
    }

    #[test]
    fn compute_risk_medium_eol_announced() {
        let mut a = sample_assessment("Test", "Vendor");
        a.firmware_update = FirmwareUpdateCapability::SignedUpdates;
        a.lifecycle = LifecycleStatus::EndOfLifeAnnounced;
        a.compute_risk();
        assert_eq!(a.risk_level, VendorRisk::Medium);
    }

    #[test]
    fn compute_risk_high_no_update_with_cves() {
        let mut a = sample_assessment("Test", "Vendor");
        a.firmware_update = FirmwareUpdateCapability::NoUpdateMechanism;
        a.known_cve_count = 5;
        a.lifecycle = LifecycleStatus::Active;
        a.compute_risk();
        assert_eq!(a.risk_level, VendorRisk::High);
    }

    #[test]
    fn compute_risk_critical_eol() {
        let mut a = sample_assessment("Test", "Vendor");
        a.lifecycle = LifecycleStatus::EndOfLife;
        a.compute_risk();
        assert_eq!(a.risk_level, VendorRisk::Critical);
    }

    #[test]
    fn registry_operations() {
        let mut reg = VendorRegistry::new();
        assert_eq!(reg.count(), 0);

        reg.add(sample_assessment("GPU-A", "NVIDIA"));
        reg.add(sample_assessment("NIC-B", "Intel"));
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn registry_by_vendor() {
        let mut reg = VendorRegistry::new();
        reg.add(sample_assessment("GPU-A", "NVIDIA"));
        reg.add(sample_assessment("GPU-B", "NVIDIA"));
        reg.add(sample_assessment("NIC-A", "Intel"));

        assert_eq!(reg.by_vendor("NVIDIA").len(), 2);
        assert_eq!(reg.by_vendor("Intel").len(), 1);
        assert_eq!(reg.by_vendor("AMD").len(), 0);
    }

    #[test]
    fn registry_by_category() {
        let mut reg = VendorRegistry::new();
        let mut nic = sample_assessment("NIC-A", "Intel");
        nic.category = ComponentCategory::Nic;
        reg.add(sample_assessment("GPU-A", "NVIDIA"));
        reg.add(nic);

        assert_eq!(reg.by_category(ComponentCategory::Gpu).len(), 1);
        assert_eq!(reg.by_category(ComponentCategory::Nic).len(), 1);
        assert_eq!(reg.by_category(ComponentCategory::Fpga).len(), 0);
    }

    #[test]
    fn registry_by_min_risk() {
        let mut reg = VendorRegistry::new();

        let mut low = sample_assessment("A", "V");
        low.risk_level = VendorRisk::Low;
        let mut high = sample_assessment("B", "V");
        high.risk_level = VendorRisk::High;
        let mut critical = sample_assessment("C", "V");
        critical.risk_level = VendorRisk::Critical;

        reg.add(low);
        reg.add(high);
        reg.add(critical);

        assert_eq!(reg.by_min_risk(VendorRisk::Low).len(), 3);
        assert_eq!(reg.by_min_risk(VendorRisk::High).len(), 2);
        assert_eq!(reg.by_min_risk(VendorRisk::Critical).len(), 1);
    }

    #[test]
    fn registry_eol_count() {
        let mut reg = VendorRegistry::new();
        let mut eol = sample_assessment("Old", "V");
        eol.lifecycle = LifecycleStatus::EndOfLife;
        reg.add(sample_assessment("Active", "V"));
        reg.add(eol);
        assert_eq!(reg.eol_count(), 1);
    }

    #[test]
    fn registry_limit() {
        let mut reg = VendorRegistry::new();
        for i in 0..MAX_ASSESSMENTS {
            assert!(reg.add(sample_assessment(&alloc::format!("comp-{}", i), "vendor",)));
        }
        assert!(!reg.add(sample_assessment("overflow", "vendor")));
        assert_eq!(reg.count(), MAX_ASSESSMENTS);
    }

    #[test]
    fn component_category_strings() {
        assert_eq!(ComponentCategory::Gpu.as_str(), "GPU");
        assert_eq!(
            ComponentCategory::BusTransceiver.as_str(),
            "Bus Transceiver"
        );
        assert_eq!(ComponentCategory::Fpga.as_str(), "FPGA");
        assert_eq!(ComponentCategory::Soc.as_str(), "SoC");
        assert_eq!(ComponentCategory::Nic.as_str(), "NIC");
        assert_eq!(ComponentCategory::Tpm.as_str(), "TPM");
    }

    #[test]
    fn lifecycle_status_patches() {
        assert!(LifecycleStatus::Active.receives_patches());
        assert!(LifecycleStatus::Maintenance.receives_patches());
        assert!(!LifecycleStatus::EndOfLifeAnnounced.receives_patches());
        assert!(!LifecycleStatus::EndOfLife.receives_patches());
    }

    #[test]
    fn firmware_update_secure() {
        assert!(FirmwareUpdateCapability::SignedUpdates.is_secure());
        assert!(!FirmwareUpdateCapability::UnsignedUpdates.is_secure());
        assert!(!FirmwareUpdateCapability::NoUpdateMechanism.is_secure());
        assert!(!FirmwareUpdateCapability::Unknown.is_secure());
    }

    #[test]
    fn has_concerns() {
        let mut a = sample_assessment("Test", "V");
        a.risk_level = VendorRisk::Low;
        assert!(!a.has_concerns());
        a.risk_level = VendorRisk::Medium;
        assert!(a.has_concerns());
        a.risk_level = VendorRisk::High;
        assert!(a.has_concerns());
    }
}
