// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! OCI image SBOM label attachment.
//!
//! Provides data structures for attaching CycloneDX SBOMs as OCI
//! image labels and standalone build artifacts per NIST SR-4.

use alloc::string::String;

/// OCI annotation key for the SBOM.
pub const OCI_SBOM_ANNOTATION: &str = "org.cyclonedx.bom";

/// OCI annotation key for the SBOM format.
pub const OCI_SBOM_FORMAT_ANNOTATION: &str = "org.cyclonedx.bom.format";

/// OCI annotation key for the SBOM version.
pub const OCI_SBOM_VERSION_ANNOTATION: &str = "org.cyclonedx.bom.version";

/// SBOM media type for CycloneDX JSON.
pub const CYCLONEDX_JSON_MEDIA_TYPE: &str = "application/vnd.cyclonedx+json";

/// SBOM media type for CycloneDX XML.
pub const CYCLONEDX_XML_MEDIA_TYPE: &str = "application/vnd.cyclonedx+xml";

/// OCI SBOM artifact descriptor.
///
/// Represents an SBOM attached to an OCI image as either:
/// 1. An image label (embedded in image config)
/// 2. A standalone artifact (referencing the image via digest)
#[derive(Debug, Clone)]
pub struct OciSbomDescriptor {
    /// OCI image reference (e.g., "ghcr.io/smallaios/smallaios:v0.1.0").
    pub image_ref: String,
    /// Image digest (sha256:...).
    pub image_digest: String,
    /// SBOM format (json or xml).
    pub format: SbomFormat,
    /// CycloneDX spec version (e.g., "1.5").
    pub spec_version: String,
    /// SHA-256 hash of the SBOM content.
    pub sbom_hash: [u8; 32],
    /// Size of the SBOM content in bytes.
    pub sbom_size: u32,
}

/// SBOM serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SbomFormat {
    /// CycloneDX JSON.
    Json = 0,
    /// CycloneDX XML.
    Xml = 1,
}

impl SbomFormat {
    pub fn media_type(&self) -> &'static str {
        match self {
            Self::Json => CYCLONEDX_JSON_MEDIA_TYPE,
            Self::Xml => CYCLONEDX_XML_MEDIA_TYPE,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Xml => "xml",
        }
    }
}

impl OciSbomDescriptor {
    /// Create a new OCI SBOM descriptor.
    pub fn new(
        image_ref: String,
        image_digest: String,
        format: SbomFormat,
        sbom_hash: [u8; 32],
        sbom_size: u32,
    ) -> Self {
        Self {
            image_ref,
            image_digest,
            format,
            spec_version: String::from("1.5"),
            sbom_hash,
            sbom_size,
        }
    }

    /// Validate the descriptor.
    pub fn validate(&self) -> Result<(), OciSbomError> {
        if self.image_ref.is_empty() {
            return Err(OciSbomError::MissingImageRef);
        }
        if self.image_digest.is_empty() {
            return Err(OciSbomError::MissingImageDigest);
        }
        if self.sbom_hash == [0u8; 32] {
            return Err(OciSbomError::MissingSbomHash);
        }
        if self.sbom_size == 0 {
            return Err(OciSbomError::EmptySbom);
        }
        Ok(())
    }
}

/// OCI SBOM errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OciSbomError {
    MissingImageRef,
    MissingImageDigest,
    MissingSbomHash,
    EmptySbom,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hash() -> [u8; 32] {
        let mut h = [0u8; 32];
        for (i, byte) in h.iter_mut().enumerate() {
            *byte = (i + 1) as u8;
        }
        h
    }

    #[test]
    fn create_descriptor() {
        let desc = OciSbomDescriptor::new(
            String::from("ghcr.io/smallaios/smallaios:v0.1.0"),
            String::from("sha256:abc123"),
            SbomFormat::Json,
            sample_hash(),
            4096,
        );
        assert_eq!(desc.spec_version, "1.5");
        assert_eq!(desc.format, SbomFormat::Json);
    }

    #[test]
    fn validate_valid_descriptor() {
        let desc = OciSbomDescriptor::new(
            String::from("image:tag"),
            String::from("sha256:abc"),
            SbomFormat::Json,
            sample_hash(),
            100,
        );
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn validate_missing_image_ref() {
        let desc = OciSbomDescriptor::new(
            String::new(),
            String::from("sha256:abc"),
            SbomFormat::Json,
            sample_hash(),
            100,
        );
        assert_eq!(desc.validate(), Err(OciSbomError::MissingImageRef));
    }

    #[test]
    fn validate_missing_digest() {
        let desc = OciSbomDescriptor::new(
            String::from("image:tag"),
            String::new(),
            SbomFormat::Json,
            sample_hash(),
            100,
        );
        assert_eq!(desc.validate(), Err(OciSbomError::MissingImageDigest));
    }

    #[test]
    fn validate_missing_hash() {
        let desc = OciSbomDescriptor::new(
            String::from("image:tag"),
            String::from("sha256:abc"),
            SbomFormat::Json,
            [0u8; 32],
            100,
        );
        assert_eq!(desc.validate(), Err(OciSbomError::MissingSbomHash));
    }

    #[test]
    fn validate_empty_sbom() {
        let desc = OciSbomDescriptor::new(
            String::from("image:tag"),
            String::from("sha256:abc"),
            SbomFormat::Json,
            sample_hash(),
            0,
        );
        assert_eq!(desc.validate(), Err(OciSbomError::EmptySbom));
    }

    #[test]
    fn format_media_types() {
        assert_eq!(SbomFormat::Json.media_type(), CYCLONEDX_JSON_MEDIA_TYPE);
        assert_eq!(SbomFormat::Xml.media_type(), CYCLONEDX_XML_MEDIA_TYPE);
    }

    #[test]
    fn format_strings() {
        assert_eq!(SbomFormat::Json.as_str(), "json");
        assert_eq!(SbomFormat::Xml.as_str(), "xml");
    }

    #[test]
    fn oci_annotation_keys() {
        assert_eq!(OCI_SBOM_ANNOTATION, "org.cyclonedx.bom");
        assert_eq!(OCI_SBOM_FORMAT_ANNOTATION, "org.cyclonedx.bom.format");
    }
}
