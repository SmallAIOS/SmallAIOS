// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CycloneDX SBOM (Software Bill of Materials) data model.
//!
//! Provides structs for representing CycloneDX 1.5 SBOM metadata.
//! Used by the build system to generate SBOM artifacts for every release.
//! Components are parsed from Cargo.lock during build.

use alloc::string::String;
use alloc::vec::Vec;

/// Maximum number of components in an SBOM.
const MAX_COMPONENTS: usize = 512;

/// CycloneDX SBOM document.
#[derive(Debug, Clone)]
pub struct Sbom {
    /// SBOM format version (e.g., "1.5").
    pub spec_version: SbomSpecVersion,
    /// Serial number (UUID-like identifier for this SBOM instance).
    pub serial_number: [u8; 16],
    /// Timestamp when the SBOM was generated (nanoseconds since epoch).
    pub timestamp_ns: u64,
    /// Tool that generated this SBOM.
    pub tool_name: String,
    /// Tool version.
    pub tool_version: String,
    /// Components listed in the SBOM.
    pub components: Vec<SbomComponent>,
}

/// CycloneDX specification version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SbomSpecVersion {
    /// CycloneDX 1.4
    V1_4 = 0,
    /// CycloneDX 1.5
    V1_5 = 1,
    /// CycloneDX 1.6
    V1_6 = 2,
}

impl SbomSpecVersion {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::V1_4 => "1.4",
            Self::V1_5 => "1.5",
            Self::V1_6 => "1.6",
        }
    }
}

/// Component type in CycloneDX terminology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ComponentType {
    /// A software library (Rust crate).
    Library = 0,
    /// An application (final binary).
    Application = 1,
    /// A framework.
    Framework = 2,
    /// An operating system component.
    OperatingSystem = 3,
}

impl ComponentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Library => "library",
            Self::Application => "application",
            Self::Framework => "framework",
            Self::OperatingSystem => "operating-system",
        }
    }
}

/// License identifier using SPDX expression.
#[derive(Debug, Clone)]
pub struct LicenseInfo {
    /// SPDX license expression (e.g., "Apache-2.0", "MIT OR Apache-2.0").
    pub spdx_expression: String,
}

/// Hash algorithm for component integrity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HashAlgorithm {
    Sha256 = 0,
    Sha3_256 = 1,
    Sha512 = 2,
}

impl HashAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sha256 => "SHA-256",
            Self::Sha3_256 => "SHA3-256",
            Self::Sha512 => "SHA-512",
        }
    }
}

/// A hash value for a component.
#[derive(Debug, Clone)]
pub struct ComponentHash {
    /// Hash algorithm used.
    pub algorithm: HashAlgorithm,
    /// Hex-encoded hash value.
    pub value: String,
}

/// A single component in the SBOM.
#[derive(Debug, Clone)]
pub struct SbomComponent {
    /// Component type.
    pub component_type: ComponentType,
    /// Component name (crate name).
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Package URL (purl) per the purl spec.
    /// Format: pkg:cargo/<name>@<version>
    pub purl: String,
    /// License information.
    pub license: Option<LicenseInfo>,
    /// Content hashes for integrity verification.
    pub hashes: Vec<ComponentHash>,
    /// Whether this is a direct (true) or transitive (false) dependency.
    pub is_direct: bool,
}

impl SbomComponent {
    /// Create a new component with required fields.
    pub fn new(name: String, version: String) -> Self {
        let purl = alloc::format!("pkg:cargo/{}@{}", name, version);
        Self {
            component_type: ComponentType::Library,
            name,
            version,
            purl,
            license: None,
            hashes: Vec::new(),
            is_direct: false,
        }
    }

    /// Set the component type.
    pub fn with_type(mut self, component_type: ComponentType) -> Self {
        self.component_type = component_type;
        self
    }

    /// Set the license.
    pub fn with_license(mut self, spdx: String) -> Self {
        self.license = Some(LicenseInfo {
            spdx_expression: spdx,
        });
        self
    }

    /// Add a hash.
    pub fn with_hash(mut self, algorithm: HashAlgorithm, value: String) -> Self {
        self.hashes.push(ComponentHash { algorithm, value });
        self
    }

    /// Mark as direct dependency.
    pub fn direct(mut self) -> Self {
        self.is_direct = true;
        self
    }
}

impl Sbom {
    /// Create a new empty SBOM.
    pub fn new(spec_version: SbomSpecVersion) -> Self {
        Self {
            spec_version,
            serial_number: [0u8; 16],
            timestamp_ns: 0,
            tool_name: String::new(),
            tool_version: String::new(),
            components: Vec::new(),
        }
    }

    /// Add a component to the SBOM.
    /// Returns false if the component limit has been reached.
    pub fn add_component(&mut self, component: SbomComponent) -> bool {
        if self.components.len() >= MAX_COMPONENTS {
            return false;
        }
        self.components.push(component);
        true
    }

    /// Number of components.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Number of direct dependencies.
    pub fn direct_count(&self) -> usize {
        self.components.iter().filter(|c| c.is_direct).count()
    }

    /// Number of transitive dependencies.
    pub fn transitive_count(&self) -> usize {
        self.components.iter().filter(|c| !c.is_direct).count()
    }

    /// Find a component by name.
    pub fn find_by_name(&self, name: &str) -> Option<&SbomComponent> {
        self.components.iter().find(|c| c.name == name)
    }

    /// Find a component by purl.
    pub fn find_by_purl(&self, purl: &str) -> Option<&SbomComponent> {
        self.components.iter().find(|c| c.purl == purl)
    }

    /// Validate the SBOM structure.
    /// Returns a list of validation errors (empty = valid).
    pub fn validate(&self) -> Vec<SbomValidationError> {
        let mut errors = Vec::new();

        if self.tool_name.is_empty() {
            errors.push(SbomValidationError::MissingToolName);
        }

        for (idx, comp) in self.components.iter().enumerate() {
            if comp.name.is_empty() {
                errors.push(SbomValidationError::EmptyComponentName { index: idx });
            }
            if comp.version.is_empty() {
                errors.push(SbomValidationError::EmptyComponentVersion { index: idx });
            }
            if comp.purl.is_empty() {
                errors.push(SbomValidationError::EmptyComponentPurl { index: idx });
            }
        }

        // Check for duplicate purls
        for i in 0..self.components.len() {
            for j in (i + 1)..self.components.len() {
                if self.components[i].purl == self.components[j].purl {
                    errors.push(SbomValidationError::DuplicatePurl {
                        purl: self.components[i].purl.clone(),
                    });
                }
            }
        }

        errors
    }
}

/// SBOM validation error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SbomValidationError {
    MissingToolName,
    EmptyComponentName { index: usize },
    EmptyComponentVersion { index: usize },
    EmptyComponentPurl { index: usize },
    DuplicatePurl { purl: String },
}

/// Parse a Cargo.lock-style entry into an SBOM component.
///
/// Expects lines in format: `name = "<crate_name>", version = "<version>", checksum = "<sha256>"`
/// This is a simplified parser for the build system integration point.
pub fn parse_cargo_lock_entry(
    name: &str,
    version: &str,
    checksum: Option<&str>,
) -> SbomComponent {
    let mut comp = SbomComponent::new(String::from(name), String::from(version));
    if let Some(hash) = checksum {
        comp = comp.with_hash(HashAlgorithm::Sha256, String::from(hash));
    }
    comp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_empty_sbom() {
        let sbom = Sbom::new(SbomSpecVersion::V1_5);
        assert_eq!(sbom.spec_version, SbomSpecVersion::V1_5);
        assert_eq!(sbom.component_count(), 0);
        assert_eq!(sbom.direct_count(), 0);
        assert_eq!(sbom.transitive_count(), 0);
    }

    #[test]
    fn add_component() {
        let mut sbom = Sbom::new(SbomSpecVersion::V1_5);
        let comp = SbomComponent::new(String::from("serde"), String::from("1.0.200"))
            .with_license(String::from("MIT OR Apache-2.0"))
            .with_hash(
                HashAlgorithm::Sha256,
                String::from("abcdef1234567890"),
            )
            .direct();

        assert!(sbom.add_component(comp));
        assert_eq!(sbom.component_count(), 1);
        assert_eq!(sbom.direct_count(), 1);
        assert_eq!(sbom.transitive_count(), 0);
    }

    #[test]
    fn component_purl_auto_generated() {
        let comp = SbomComponent::new(String::from("tokio"), String::from("1.37.0"));
        assert_eq!(comp.purl, "pkg:cargo/tokio@1.37.0");
    }

    #[test]
    fn find_by_name() {
        let mut sbom = Sbom::new(SbomSpecVersion::V1_5);
        sbom.add_component(SbomComponent::new(
            String::from("serde"),
            String::from("1.0.200"),
        ));
        sbom.add_component(SbomComponent::new(
            String::from("tokio"),
            String::from("1.37.0"),
        ));

        assert!(sbom.find_by_name("serde").is_some());
        assert!(sbom.find_by_name("tokio").is_some());
        assert!(sbom.find_by_name("missing").is_none());
    }

    #[test]
    fn find_by_purl() {
        let mut sbom = Sbom::new(SbomSpecVersion::V1_5);
        sbom.add_component(SbomComponent::new(
            String::from("serde"),
            String::from("1.0.200"),
        ));
        assert!(sbom.find_by_purl("pkg:cargo/serde@1.0.200").is_some());
        assert!(sbom.find_by_purl("pkg:cargo/serde@2.0.0").is_none());
    }

    #[test]
    fn validation_valid_sbom() {
        let mut sbom = Sbom::new(SbomSpecVersion::V1_5);
        sbom.tool_name = String::from("cargo-cyclonedx");
        sbom.tool_version = String::from("0.5.0");
        sbom.add_component(SbomComponent::new(
            String::from("serde"),
            String::from("1.0.200"),
        ));
        let errors = sbom.validate();
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn validation_missing_tool_name() {
        let sbom = Sbom::new(SbomSpecVersion::V1_5);
        let errors = sbom.validate();
        assert!(errors.contains(&SbomValidationError::MissingToolName));
    }

    #[test]
    fn validation_empty_component_name() {
        let mut sbom = Sbom::new(SbomSpecVersion::V1_5);
        sbom.tool_name = String::from("test");
        sbom.add_component(SbomComponent {
            component_type: ComponentType::Library,
            name: String::new(),
            version: String::from("1.0.0"),
            purl: String::from("pkg:cargo/x@1.0.0"),
            license: None,
            hashes: Vec::new(),
            is_direct: false,
        });
        let errors = sbom.validate();
        assert!(errors.contains(&SbomValidationError::EmptyComponentName { index: 0 }));
    }

    #[test]
    fn validation_duplicate_purl() {
        let mut sbom = Sbom::new(SbomSpecVersion::V1_5);
        sbom.tool_name = String::from("test");
        sbom.add_component(SbomComponent::new(
            String::from("serde"),
            String::from("1.0.0"),
        ));
        sbom.add_component(SbomComponent::new(
            String::from("serde"),
            String::from("1.0.0"),
        ));
        let errors = sbom.validate();
        assert!(errors.iter().any(|e| matches!(e, SbomValidationError::DuplicatePurl { .. })));
    }

    #[test]
    fn component_limit() {
        let mut sbom = Sbom::new(SbomSpecVersion::V1_5);
        for i in 0..MAX_COMPONENTS {
            assert!(sbom.add_component(SbomComponent::new(
                alloc::format!("crate-{}", i),
                String::from("1.0.0"),
            )));
        }
        // Exceeds limit
        assert!(!sbom.add_component(SbomComponent::new(
            String::from("overflow"),
            String::from("1.0.0"),
        )));
        assert_eq!(sbom.component_count(), MAX_COMPONENTS);
    }

    #[test]
    fn parse_cargo_lock_entry_basic() {
        let comp = parse_cargo_lock_entry("serde", "1.0.200", Some("abc123"));
        assert_eq!(comp.name, "serde");
        assert_eq!(comp.version, "1.0.200");
        assert_eq!(comp.purl, "pkg:cargo/serde@1.0.200");
        assert_eq!(comp.hashes.len(), 1);
        assert_eq!(comp.hashes[0].algorithm, HashAlgorithm::Sha256);
        assert_eq!(comp.hashes[0].value, "abc123");
    }

    #[test]
    fn parse_cargo_lock_entry_no_checksum() {
        let comp = parse_cargo_lock_entry("local-crate", "0.1.0", None);
        assert_eq!(comp.name, "local-crate");
        assert!(comp.hashes.is_empty());
    }

    #[test]
    fn component_type_strings() {
        assert_eq!(ComponentType::Library.as_str(), "library");
        assert_eq!(ComponentType::Application.as_str(), "application");
        assert_eq!(ComponentType::Framework.as_str(), "framework");
        assert_eq!(ComponentType::OperatingSystem.as_str(), "operating-system");
    }

    #[test]
    fn hash_algorithm_strings() {
        assert_eq!(HashAlgorithm::Sha256.as_str(), "SHA-256");
        assert_eq!(HashAlgorithm::Sha3_256.as_str(), "SHA3-256");
        assert_eq!(HashAlgorithm::Sha512.as_str(), "SHA-512");
    }

    #[test]
    fn spec_version_strings() {
        assert_eq!(SbomSpecVersion::V1_4.as_str(), "1.4");
        assert_eq!(SbomSpecVersion::V1_5.as_str(), "1.5");
        assert_eq!(SbomSpecVersion::V1_6.as_str(), "1.6");
    }

    #[test]
    fn component_builder_chain() {
        let comp = SbomComponent::new(String::from("test"), String::from("1.0.0"))
            .with_type(ComponentType::Application)
            .with_license(String::from("MIT"))
            .with_hash(HashAlgorithm::Sha3_256, String::from("deadbeef"))
            .direct();

        assert_eq!(comp.component_type, ComponentType::Application);
        assert!(comp.license.is_some());
        assert_eq!(comp.license.unwrap().spdx_expression, "MIT");
        assert_eq!(comp.hashes.len(), 1);
        assert!(comp.is_direct);
    }
}
