// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! OCI container image metadata.
//!
//! Provides types for describing OCI-compatible container images tailored to
//! SmallAIOS unikernel packaging. Supports multi-architecture manifests and
//! simplified JSON serialisation for image registries.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use crate::ContainerError;

/// Processor architecture for a container image.
#[derive(Clone, Debug, PartialEq)]
pub enum Architecture {
    /// x86-64 / AMD64.
    Amd64,
    /// AArch64 / ARM64.
    Arm64,
    /// RISC-V 64-bit.
    RiscV64,
}

impl Architecture {
    /// Returns the OCI platform architecture string.
    fn as_str(&self) -> &str {
        match self {
            Architecture::Amd64 => "amd64",
            Architecture::Arm64 => "arm64",
            Architecture::RiscV64 => "riscv64",
        }
    }
}

/// Operating system type for a container image.
#[derive(Clone, Debug, PartialEq)]
pub enum OsType {
    /// SmallAIOS unikernel.
    SmallAIOS,
    /// Linux (for compatibility layers).
    Linux,
}

impl OsType {
    /// Returns the OCI platform OS string.
    fn as_str(&self) -> &str {
        match self {
            OsType::SmallAIOS => "smallaios",
            OsType::Linux => "linux",
        }
    }
}

/// A single layer within an OCI container image.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageLayer {
    /// Content-addressable digest (e.g., "sha256:abc123...").
    pub digest: String,
    /// Size of the layer in bytes.
    pub size_bytes: u64,
    /// MIME type (e.g., "application/vnd.oci.image.layer.v1.tar+gzip").
    pub media_type: String,
}

/// OCI image manifest describing a single-architecture image.
///
/// Schema version is always 2 (OCI Image Manifest v2).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageManifest {
    /// OCI schema version (always 2).
    pub schema_version: u8,
    /// Target processor architecture.
    pub architecture: Architecture,
    /// Target operating system.
    pub os: OsType,
    /// Ordered list of image layers.
    pub layers: Vec<ImageLayer>,
    /// Key-value annotation pairs.
    pub annotations: Vec<(String, String)>,
    /// Running total of all layer sizes in bytes.
    pub total_size: u64,
}

impl ImageManifest {
    /// Create a new image manifest for the given architecture.
    ///
    /// Defaults to [`OsType::SmallAIOS`], schema version 2, and empty layers.
    pub fn new(arch: Architecture) -> Self {
        ImageManifest {
            schema_version: 2,
            architecture: arch,
            os: OsType::SmallAIOS,
            layers: Vec::new(),
            annotations: Vec::new(),
            total_size: 0,
        }
    }

    /// Add a layer to the manifest.
    ///
    /// Updates the running total size automatically.
    pub fn add_layer(&mut self, digest: &str, size: u64, media_type: &str) {
        self.layers.push(ImageLayer {
            digest: String::from(digest),
            size_bytes: size,
            media_type: String::from(media_type),
        });
        self.total_size += size;
    }

    /// Add a key-value annotation to the manifest.
    pub fn add_annotation(&mut self, key: &str, value: &str) {
        self.annotations
            .push((String::from(key), String::from(value)));
    }

    /// Returns the number of layers in the manifest.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns the total size of all layers in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        self.total_size
    }

    /// Check whether the total image size is under the given limit in megabytes.
    pub fn is_under_size_limit(&self, limit_mb: u64) -> bool {
        self.total_size < limit_mb * 1024 * 1024
    }

    /// Produce a simplified JSON representation of the manifest.
    ///
    /// This is not a full OCI manifest JSON but contains the essential fields
    /// for registry interaction and debugging.
    pub fn to_json(&self) -> String {
        let mut json = String::new();
        let _ = write!(json, "{{");
        let _ = write!(json, "\"schemaVersion\":{}", self.schema_version);
        let _ = write!(json, ",\"architecture\":\"{}\"", self.architecture.as_str());
        let _ = write!(json, ",\"os\":\"{}\"", self.os.as_str());
        let _ = write!(json, ",\"layers\":[");
        for (i, layer) in self.layers.iter().enumerate() {
            if i > 0 {
                let _ = write!(json, ",");
            }
            let _ = write!(
                json,
                "{{\"digest\":\"{}\",\"size\":{},\"mediaType\":\"{}\"}}",
                layer.digest, layer.size_bytes, layer.media_type
            );
        }
        let _ = write!(json, "]");
        if !self.annotations.is_empty() {
            let _ = write!(json, ",\"annotations\":{{");
            for (i, (key, value)) in self.annotations.iter().enumerate() {
                if i > 0 {
                    let _ = write!(json, ",");
                }
                let _ = write!(json, "\"{}\":\"{}\"", key, value);
            }
            let _ = write!(json, "}}");
        }
        let _ = write!(json, ",\"totalSize\":{}", self.total_size);
        let _ = write!(json, "}}");
        json
    }
}

/// Builder for multi-architecture OCI container images.
///
/// Collects one or more [`ImageManifest`] entries (one per architecture) and
/// provides validation and aggregate statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageBuilder {
    /// Image repository name (e.g., "smallaios/inference-server").
    pub name: String,
    /// Image tag (e.g., "v1.0.0" or "latest").
    pub tag: String,
    /// Per-architecture manifests.
    pub manifests: Vec<ImageManifest>,
}

impl ImageBuilder {
    /// Create a new image builder with the given repository name and tag.
    pub fn new(name: &str, tag: &str) -> Self {
        ImageBuilder {
            name: String::from(name),
            tag: String::from(tag),
            manifests: Vec::new(),
        }
    }

    /// Add an architecture-specific manifest to the builder.
    pub fn add_manifest(&mut self, manifest: ImageManifest) {
        self.manifests.push(manifest);
    }

    /// Returns the number of manifests (architectures) in this image.
    pub fn manifest_count(&self) -> usize {
        self.manifests.len()
    }

    /// Returns `true` if the image targets more than one architecture.
    pub fn is_multi_arch(&self) -> bool {
        self.manifests.len() > 1
    }

    /// Returns the total size across all architecture manifests in bytes.
    pub fn total_size(&self) -> u64 {
        self.manifests.iter().map(|m| m.total_size).sum()
    }

    /// Validate the image builder configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ContainerError::InvalidManifest`] if no manifests have been
    /// added.
    pub fn validate(&self) -> Result<(), ContainerError> {
        if self.manifests.is_empty() {
            return Err(ContainerError::InvalidManifest(String::from(
                "at least one manifest is required",
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_manifest_defaults() {
        let manifest = ImageManifest::new(Architecture::Amd64);
        assert_eq!(manifest.schema_version, 2);
        assert_eq!(manifest.architecture, Architecture::Amd64);
        assert_eq!(manifest.os, OsType::SmallAIOS);
        assert_eq!(manifest.layer_count(), 0);
        assert_eq!(manifest.total_size_bytes(), 0);
    }

    #[test]
    fn add_layer_updates_size() {
        let mut manifest = ImageManifest::new(Architecture::Arm64);
        manifest.add_layer(
            "sha256:aaa",
            1000,
            "application/vnd.oci.image.layer.v1.tar+gzip",
        );
        assert_eq!(manifest.layer_count(), 1);
        assert_eq!(manifest.total_size_bytes(), 1000);

        manifest.add_layer(
            "sha256:bbb",
            2000,
            "application/vnd.oci.image.layer.v1.tar+gzip",
        );
        assert_eq!(manifest.layer_count(), 2);
        assert_eq!(manifest.total_size_bytes(), 3000);
    }

    #[test]
    fn add_annotation() {
        let mut manifest = ImageManifest::new(Architecture::Amd64);
        manifest.add_annotation("org.opencontainers.image.version", "1.0.0");
        assert_eq!(manifest.annotations.len(), 1);
        assert_eq!(
            manifest.annotations[0].0,
            "org.opencontainers.image.version"
        );
        assert_eq!(manifest.annotations[0].1, "1.0.0");
    }

    #[test]
    fn layer_count() {
        let mut manifest = ImageManifest::new(Architecture::RiscV64);
        assert_eq!(manifest.layer_count(), 0);
        manifest.add_layer("sha256:ccc", 500, "application/octet-stream");
        assert_eq!(manifest.layer_count(), 1);
        manifest.add_layer("sha256:ddd", 700, "application/octet-stream");
        assert_eq!(manifest.layer_count(), 2);
    }

    #[test]
    fn size_limit_under() {
        let mut manifest = ImageManifest::new(Architecture::Amd64);
        // 10 MB = 10 * 1024 * 1024 = 10_485_760 bytes
        manifest.add_layer("sha256:eee", 5_000_000, "application/octet-stream");
        assert!(manifest.is_under_size_limit(10)); // under 10 MB
    }

    #[test]
    fn size_limit_over() {
        let mut manifest = ImageManifest::new(Architecture::Amd64);
        // 15 MB limit = 15 * 1024 * 1024 = 15_728_640 bytes
        manifest.add_layer("sha256:fff", 16_000_000, "application/octet-stream");
        assert!(!manifest.is_under_size_limit(15)); // over 15 MB
    }

    #[test]
    fn multi_arch_build() {
        let mut builder = ImageBuilder::new("smallaios/inference", "v1.0.0");

        let mut amd64 = ImageManifest::new(Architecture::Amd64);
        amd64.add_layer("sha256:amd64_layer", 5000, "application/octet-stream");

        let mut arm64 = ImageManifest::new(Architecture::Arm64);
        arm64.add_layer("sha256:arm64_layer", 6000, "application/octet-stream");

        builder.add_manifest(amd64);
        builder.add_manifest(arm64);

        assert!(builder.is_multi_arch());
        assert_eq!(builder.manifest_count(), 2);
    }

    #[test]
    fn image_builder_validate_with_manifests() {
        let mut builder = ImageBuilder::new("smallaios/test", "latest");
        builder.add_manifest(ImageManifest::new(Architecture::Amd64));
        assert!(builder.validate().is_ok());
    }

    #[test]
    fn image_builder_validate_empty_returns_error() {
        let builder = ImageBuilder::new("smallaios/empty", "latest");
        let result = builder.validate();
        match result {
            Err(ContainerError::InvalidManifest(msg)) => {
                assert!(msg.contains("at least one manifest"));
            }
            other => panic!("expected InvalidManifest, got {:?}", other),
        }
    }

    #[test]
    fn architecture_variants() {
        let amd64 = Architecture::Amd64;
        let arm64 = Architecture::Arm64;
        let riscv = Architecture::RiscV64;

        assert_ne!(amd64, arm64);
        assert_ne!(arm64, riscv);
        assert_ne!(amd64, riscv);

        assert_eq!(amd64.as_str(), "amd64");
        assert_eq!(arm64.as_str(), "arm64");
        assert_eq!(riscv.as_str(), "riscv64");
    }

    #[test]
    fn json_output_contains_expected_fields() {
        let mut manifest = ImageManifest::new(Architecture::Amd64);
        manifest.add_layer(
            "sha256:abc123",
            4096,
            "application/vnd.oci.image.layer.v1.tar+gzip",
        );
        manifest.add_annotation("version", "1.0");

        let json = manifest.to_json();
        assert!(json.contains("\"schemaVersion\":2"));
        assert!(json.contains("\"architecture\":\"amd64\""));
        assert!(json.contains("\"os\":\"smallaios\""));
        assert!(json.contains("\"digest\":\"sha256:abc123\""));
        assert!(json.contains("\"size\":4096"));
        assert!(json.contains("\"mediaType\":\"application/vnd.oci.image.layer.v1.tar+gzip\""));
        assert!(json.contains("\"annotations\":{"));
        assert!(json.contains("\"version\":\"1.0\""));
        assert!(json.contains("\"totalSize\":4096"));
    }

    #[test]
    fn total_size_across_manifests() {
        let mut builder = ImageBuilder::new("test/image", "v2");

        let mut m1 = ImageManifest::new(Architecture::Amd64);
        m1.add_layer("sha256:a", 1000, "application/octet-stream");
        m1.add_layer("sha256:b", 2000, "application/octet-stream");

        let mut m2 = ImageManifest::new(Architecture::Arm64);
        m2.add_layer("sha256:c", 3000, "application/octet-stream");

        builder.add_manifest(m1);
        builder.add_manifest(m2);

        assert_eq!(builder.total_size(), 6000);
    }

    #[test]
    fn schema_version() {
        let manifest = ImageManifest::new(Architecture::RiscV64);
        assert_eq!(manifest.schema_version, 2);
    }

    #[test]
    fn os_type_defaults_to_smallaios() {
        let manifest = ImageManifest::new(Architecture::Amd64);
        assert_eq!(manifest.os, OsType::SmallAIOS);
        assert_ne!(manifest.os, OsType::Linux);
    }

    #[test]
    fn image_builder_single_arch_not_multi() {
        let mut builder = ImageBuilder::new("test/single", "v1");
        builder.add_manifest(ImageManifest::new(Architecture::Amd64));
        assert!(!builder.is_multi_arch());
        assert_eq!(builder.manifest_count(), 1);
    }

    #[test]
    fn image_layer_clone_and_eq() {
        let layer = ImageLayer {
            digest: String::from("sha256:test"),
            size_bytes: 512,
            media_type: String::from("application/octet-stream"),
        };
        let cloned = layer.clone();
        assert_eq!(layer, cloned);
    }
}
