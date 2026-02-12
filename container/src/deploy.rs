// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deployment and provisioning configuration.
//!
//! Provides data models and configuration for:
//! - UEFI Secure Boot with ML-DSA-65 signed kernel images (25.1)
//! - PXE/iPXE bare metal network boot provisioning (25.2)
//! - VM image generation configuration (raw, qcow2, VMDK) (25.3)
//! - Image signing with ML-DSA-65 post-quantum signatures (25.4)
//! - OCI multi-architecture image builds (amd64, arm64, riscv64) (25.5)

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::image::Architecture;
use crate::ContainerError;

// ---------------------------------------------------------------------------
// 25.1 — UEFI Secure Boot Configuration
// ---------------------------------------------------------------------------

/// Signature algorithm for secure boot and image signing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureAlgorithm {
    /// ML-DSA-65 (FIPS 204) — post-quantum digital signature.
    MlDsa65,
    /// ML-DSA-87 — higher security level.
    MlDsa87,
    /// Ed25519 — classical fallback.
    Ed25519,
    /// Hybrid: ML-DSA-65 + Ed25519.
    HybridMlDsa65Ed25519,
}

impl SignatureAlgorithm {
    /// Returns the OID-style identifier string.
    pub fn as_str(&self) -> &str {
        match self {
            SignatureAlgorithm::MlDsa65 => "ML-DSA-65",
            SignatureAlgorithm::MlDsa87 => "ML-DSA-87",
            SignatureAlgorithm::Ed25519 => "Ed25519",
            SignatureAlgorithm::HybridMlDsa65Ed25519 => "ML-DSA-65+Ed25519",
        }
    }
}

/// UEFI Secure Boot database entry.
#[derive(Debug, Clone, PartialEq)]
pub struct SecureBootKey {
    /// Key owner description.
    pub owner: String,
    /// Public key bytes.
    pub public_key: Vec<u8>,
    /// Signature algorithm.
    pub algorithm: SignatureAlgorithm,
}

/// UEFI Secure Boot configuration for kernel image verification.
#[derive(Debug, Clone, PartialEq)]
pub struct SecureBootConfig {
    /// Whether Secure Boot is enabled.
    pub enabled: bool,
    /// Signature algorithm for the kernel image.
    pub algorithm: SignatureAlgorithm,
    /// Platform key (PK) — root of trust.
    pub platform_key: Option<SecureBootKey>,
    /// Key Exchange Keys (KEK) — authorize db/dbx updates.
    pub key_exchange_keys: Vec<SecureBootKey>,
    /// Authorized signature database (db).
    pub authorized_keys: Vec<SecureBootKey>,
    /// Forbidden signature database (dbx) — revoked keys.
    pub forbidden_hashes: Vec<Vec<u8>>,
    /// Whether to require post-quantum signatures only.
    pub require_pqc: bool,
}

impl Default for SecureBootConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: SignatureAlgorithm::MlDsa65,
            platform_key: None,
            key_exchange_keys: Vec::new(),
            authorized_keys: Vec::new(),
            forbidden_hashes: Vec::new(),
            require_pqc: true,
        }
    }
}

impl SecureBootConfig {
    /// Validate the secure boot configuration.
    pub fn validate(&self) -> Result<(), ContainerError> {
        if self.enabled && self.platform_key.is_none() {
            return Err(ContainerError::InvalidConfig(String::from(
                "secure boot enabled but no platform key set",
            )));
        }
        if self.enabled && self.authorized_keys.is_empty() {
            return Err(ContainerError::InvalidConfig(String::from(
                "secure boot enabled but no authorized keys in db",
            )));
        }
        Ok(())
    }

    /// Set the platform key.
    pub fn set_platform_key(&mut self, owner: &str, public_key: &[u8]) {
        self.platform_key = Some(SecureBootKey {
            owner: String::from(owner),
            public_key: public_key.into(),
            algorithm: self.algorithm,
        });
    }

    /// Add a key exchange key.
    pub fn add_kek(&mut self, owner: &str, public_key: &[u8]) {
        self.key_exchange_keys.push(SecureBootKey {
            owner: String::from(owner),
            public_key: public_key.into(),
            algorithm: self.algorithm,
        });
    }

    /// Add an authorized key to the db.
    pub fn add_authorized_key(&mut self, owner: &str, public_key: &[u8]) {
        self.authorized_keys.push(SecureBootKey {
            owner: String::from(owner),
            public_key: public_key.into(),
            algorithm: self.algorithm,
        });
    }

    /// Add a forbidden hash to the dbx.
    pub fn add_forbidden_hash(&mut self, hash: &[u8]) {
        self.forbidden_hashes.push(hash.into());
    }
}

// ---------------------------------------------------------------------------
// 25.2 — PXE/iPXE Network Boot Configuration
// ---------------------------------------------------------------------------

/// PXE boot protocol variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PxeProtocol {
    /// Legacy PXE (DHCP + TFTP).
    LegacyPxe,
    /// iPXE with HTTP/HTTPS download.
    Ipxe,
    /// iPXE with iSCSI boot.
    IpxeIscsi,
}

/// DHCP option for PXE boot.
#[derive(Debug, Clone, PartialEq)]
pub struct DhcpOption {
    /// DHCP option number.
    pub option_number: u8,
    /// Option value.
    pub value: Vec<u8>,
}

/// PXE/iPXE network boot provisioning configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct PxeBootConfig {
    /// PXE protocol variant.
    pub protocol: PxeProtocol,
    /// TFTP/HTTP server address for kernel image download.
    pub server_address: String,
    /// Path to the kernel image on the server.
    pub kernel_path: String,
    /// Path to the initial ramdisk (if any).
    pub initrd_path: Option<String>,
    /// Kernel command line arguments.
    pub cmdline: String,
    /// Whether to verify image signature before boot.
    pub verify_signature: bool,
    /// DHCP options to send to the client.
    pub dhcp_options: Vec<DhcpOption>,
    /// iPXE script URL (for iPXE variants).
    pub ipxe_script_url: Option<String>,
    /// Target architecture.
    pub architecture: Architecture,
}

impl PxeBootConfig {
    /// Create a minimal PXE boot config for legacy PXE.
    pub fn new_legacy(server: &str, kernel_path: &str, arch: Architecture) -> Self {
        Self {
            protocol: PxeProtocol::LegacyPxe,
            server_address: String::from(server),
            kernel_path: String::from(kernel_path),
            initrd_path: None,
            cmdline: String::from("console=ttyS0"),
            verify_signature: true,
            dhcp_options: Vec::new(),
            ipxe_script_url: None,
            architecture: arch,
        }
    }

    /// Create an iPXE boot config.
    pub fn new_ipxe(server: &str, kernel_path: &str, script_url: &str, arch: Architecture) -> Self {
        Self {
            protocol: PxeProtocol::Ipxe,
            server_address: String::from(server),
            kernel_path: String::from(kernel_path),
            initrd_path: None,
            cmdline: String::from("console=ttyS0"),
            verify_signature: true,
            dhcp_options: Vec::new(),
            ipxe_script_url: Some(String::from(script_url)),
            architecture: arch,
        }
    }

    /// Validate the PXE boot configuration.
    pub fn validate(&self) -> Result<(), ContainerError> {
        if self.server_address.is_empty() {
            return Err(ContainerError::InvalidConfig(String::from(
                "PXE server address is empty",
            )));
        }
        if self.kernel_path.is_empty() {
            return Err(ContainerError::InvalidConfig(String::from(
                "PXE kernel path is empty",
            )));
        }
        if matches!(self.protocol, PxeProtocol::Ipxe | PxeProtocol::IpxeIscsi)
            && self.ipxe_script_url.is_none()
        {
            return Err(ContainerError::InvalidConfig(String::from(
                "iPXE protocol requires script URL",
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 25.3 — VM Image Generation Configuration
// ---------------------------------------------------------------------------

/// VM disk image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmImageFormat {
    /// Raw disk image (dd-style).
    Raw,
    /// QEMU qcow2 format.
    Qcow2,
    /// VMware VMDK format.
    Vmdk,
    /// VirtualBox VDI format.
    Vdi,
}

impl VmImageFormat {
    /// Returns the file extension for this format.
    pub fn extension(&self) -> &str {
        match self {
            VmImageFormat::Raw => "img",
            VmImageFormat::Qcow2 => "qcow2",
            VmImageFormat::Vmdk => "vmdk",
            VmImageFormat::Vdi => "vdi",
        }
    }
}

/// Partition table type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionTable {
    /// GUID Partition Table (GPT) — for UEFI.
    Gpt,
    /// Master Boot Record (MBR) — legacy.
    Mbr,
}

/// VM image generation configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct VmImageConfig {
    /// Output image format.
    pub format: VmImageFormat,
    /// Target architecture.
    pub architecture: Architecture,
    /// Disk size in megabytes.
    pub disk_size_mb: u64,
    /// Partition table type.
    pub partition_table: PartitionTable,
    /// Whether to include a UEFI ESP partition.
    pub include_esp: bool,
    /// Whether to compress the output image.
    pub compress: bool,
    /// Kernel image path to embed.
    pub kernel_path: String,
    /// Optional initrd path.
    pub initrd_path: Option<String>,
    /// Kernel command line.
    pub cmdline: String,
}

impl VmImageConfig {
    /// Create a default VM image config for QEMU qcow2.
    pub fn qcow2(arch: Architecture) -> Self {
        Self {
            format: VmImageFormat::Qcow2,
            architecture: arch,
            disk_size_mb: 64,
            partition_table: PartitionTable::Gpt,
            include_esp: true,
            compress: true,
            kernel_path: String::from("smallaios.elf"),
            initrd_path: None,
            cmdline: String::from("console=ttyS0"),
        }
    }

    /// Create a raw disk image config.
    pub fn raw(arch: Architecture, size_mb: u64) -> Self {
        Self {
            format: VmImageFormat::Raw,
            architecture: arch,
            disk_size_mb: size_mb,
            partition_table: PartitionTable::Gpt,
            include_esp: true,
            compress: false,
            kernel_path: String::from("smallaios.elf"),
            initrd_path: None,
            cmdline: String::from("console=ttyS0"),
        }
    }

    /// Create a VMDK config.
    pub fn vmdk(arch: Architecture) -> Self {
        Self {
            format: VmImageFormat::Vmdk,
            architecture: arch,
            disk_size_mb: 64,
            partition_table: PartitionTable::Gpt,
            include_esp: true,
            compress: true,
            kernel_path: String::from("smallaios.elf"),
            initrd_path: None,
            cmdline: String::from("console=ttyS0"),
        }
    }

    /// Validate the VM image configuration.
    pub fn validate(&self) -> Result<(), ContainerError> {
        if self.disk_size_mb == 0 {
            return Err(ContainerError::InvalidConfig(String::from(
                "disk size must be > 0",
            )));
        }
        if self.kernel_path.is_empty() {
            return Err(ContainerError::InvalidConfig(String::from(
                "kernel path is empty",
            )));
        }
        // Minimum 16 MB for kernel + ESP.
        if self.disk_size_mb < 16 {
            return Err(ContainerError::InvalidConfig(String::from(
                "disk size must be >= 16 MB",
            )));
        }
        Ok(())
    }

    /// Compute the output file name.
    pub fn output_filename(&self) -> String {
        alloc::format!(
            "smallaios-{}.{}",
            match &self.architecture {
                Architecture::Amd64 => "amd64",
                Architecture::Arm64 => "arm64",
                Architecture::RiscV64 => "riscv64",
            },
            self.format.extension()
        )
    }
}

// ---------------------------------------------------------------------------
// 25.4 — Image Signing
// ---------------------------------------------------------------------------

/// A digital signature over a kernel or container image.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSignature {
    /// Signature algorithm used.
    pub algorithm: SignatureAlgorithm,
    /// Signer identity (key owner).
    pub signer: String,
    /// Signature bytes.
    pub signature: Vec<u8>,
    /// SHA-3 (Keccak-256) hash of the signed content.
    pub content_hash: Vec<u8>,
    /// Signing timestamp in seconds since UNIX epoch.
    pub timestamp: u64,
}

/// Image signing configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSigningConfig {
    /// Signature algorithm.
    pub algorithm: SignatureAlgorithm,
    /// Signer identity.
    pub signer: String,
    /// Private key bytes (for signing operations).
    pub private_key: Vec<u8>,
    /// Public key bytes (for verification).
    pub public_key: Vec<u8>,
    /// Whether to include the timestamp in the signature.
    pub include_timestamp: bool,
}

impl ImageSigningConfig {
    /// Create a new signing config with ML-DSA-65.
    pub fn new_ml_dsa_65(signer: &str) -> Self {
        Self {
            algorithm: SignatureAlgorithm::MlDsa65,
            signer: String::from(signer),
            private_key: Vec::new(),
            public_key: Vec::new(),
            include_timestamp: true,
        }
    }

    /// Create a signing config with the hybrid ML-DSA-65 + Ed25519 scheme.
    pub fn new_hybrid(signer: &str) -> Self {
        Self {
            algorithm: SignatureAlgorithm::HybridMlDsa65Ed25519,
            signer: String::from(signer),
            private_key: Vec::new(),
            public_key: Vec::new(),
            include_timestamp: true,
        }
    }

    /// Set the key pair (for testing — real keys would be loaded from secure storage).
    pub fn set_keys(&mut self, private_key: &[u8], public_key: &[u8]) {
        self.private_key = private_key.into();
        self.public_key = public_key.into();
    }

    /// Create a stub signature for testing.
    ///
    /// In production, this would perform actual ML-DSA-65 or hybrid signing.
    pub fn sign(&self, content_hash: &[u8], timestamp: u64) -> ImageSignature {
        // Stub: create a deterministic "signature" from the hash and key.
        let mut sig_bytes = Vec::with_capacity(self.private_key.len() + content_hash.len());
        for (i, &b) in content_hash.iter().enumerate() {
            let key_byte = if self.private_key.is_empty() {
                0
            } else {
                self.private_key[i % self.private_key.len()]
            };
            sig_bytes.push(b ^ key_byte);
        }

        ImageSignature {
            algorithm: self.algorithm,
            signer: self.signer.clone(),
            signature: sig_bytes,
            content_hash: content_hash.into(),
            timestamp,
        }
    }

    /// Verify a stub signature (for testing).
    pub fn verify(&self, signature: &ImageSignature, content_hash: &[u8]) -> bool {
        if signature.algorithm != self.algorithm {
            return false;
        }
        if signature.content_hash != content_hash {
            return false;
        }
        // Stub verification: reconstruct the expected signature.
        for (i, &b) in content_hash.iter().enumerate() {
            let key_byte = if self.private_key.is_empty() {
                0
            } else {
                self.private_key[i % self.private_key.len()]
            };
            if i >= signature.signature.len() || signature.signature[i] != (b ^ key_byte) {
                return false;
            }
        }
        true
    }

    /// Validate the signing configuration.
    pub fn validate(&self) -> Result<(), ContainerError> {
        if self.signer.is_empty() {
            return Err(ContainerError::InvalidConfig(String::from(
                "signer identity is empty",
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 25.5 — OCI Multi-Arch Image Build
// ---------------------------------------------------------------------------

/// Multi-architecture OCI image build configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiArchBuildConfig {
    /// Image repository name.
    pub repository: String,
    /// Image tag.
    pub tag: String,
    /// Target architectures.
    pub architectures: Vec<Architecture>,
    /// Whether to sign the image.
    pub sign: bool,
    /// Signing configuration (if signing is enabled).
    pub signing_config: Option<ImageSigningConfig>,
    /// Whether to push to a registry after building.
    pub push: bool,
    /// Registry URL.
    pub registry_url: Option<String>,
    /// Maximum image size in MB (for validation).
    pub max_size_mb: u64,
}

impl MultiArchBuildConfig {
    /// Create a default multi-arch build config targeting amd64, arm64, and riscv64.
    pub fn default_triple(repository: &str, tag: &str) -> Self {
        Self {
            repository: String::from(repository),
            tag: String::from(tag),
            architectures: alloc::vec![
                Architecture::Amd64,
                Architecture::Arm64,
                Architecture::RiscV64,
            ],
            sign: true,
            signing_config: Some(ImageSigningConfig::new_ml_dsa_65("SmallAIOS Build")),
            push: false,
            registry_url: None,
            max_size_mb: 15,
        }
    }

    /// Create a build config for a single architecture.
    pub fn single_arch(repository: &str, tag: &str, arch: Architecture) -> Self {
        Self {
            repository: String::from(repository),
            tag: String::from(tag),
            architectures: alloc::vec![arch],
            sign: true,
            signing_config: Some(ImageSigningConfig::new_ml_dsa_65("SmallAIOS Build")),
            push: false,
            registry_url: None,
            max_size_mb: 15,
        }
    }

    /// Return the number of target architectures.
    pub fn arch_count(&self) -> usize {
        self.architectures.len()
    }

    /// Check if this is a multi-architecture build.
    pub fn is_multi_arch(&self) -> bool {
        self.architectures.len() > 1
    }

    /// Validate the build configuration.
    pub fn validate(&self) -> Result<(), ContainerError> {
        if self.repository.is_empty() {
            return Err(ContainerError::InvalidConfig(String::from(
                "repository name is empty",
            )));
        }
        if self.tag.is_empty() {
            return Err(ContainerError::InvalidConfig(String::from(
                "image tag is empty",
            )));
        }
        if self.architectures.is_empty() {
            return Err(ContainerError::InvalidConfig(String::from(
                "at least one architecture required",
            )));
        }
        if self.sign && self.signing_config.is_none() {
            return Err(ContainerError::InvalidConfig(String::from(
                "signing enabled but no signing config",
            )));
        }
        if self.push && self.registry_url.is_none() {
            return Err(ContainerError::InvalidConfig(String::from(
                "push enabled but no registry URL",
            )));
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- 25.1: UEFI Secure Boot ---------------------------------------------

    #[test]
    fn test_secure_boot_default() {
        let cfg = SecureBootConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.algorithm, SignatureAlgorithm::MlDsa65);
        assert!(cfg.require_pqc);
        assert!(cfg.platform_key.is_none());
        assert!(cfg.authorized_keys.is_empty());
    }

    #[test]
    fn test_secure_boot_validate_no_pk() {
        let cfg = SecureBootConfig::default();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_secure_boot_validate_no_db() {
        let mut cfg = SecureBootConfig::default();
        cfg.set_platform_key("SmallAIOS", &[0x01, 0x02, 0x03]);
        // No authorized keys in db.
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_secure_boot_validate_ok() {
        let mut cfg = SecureBootConfig::default();
        cfg.set_platform_key("SmallAIOS", &[0x01, 0x02, 0x03]);
        cfg.add_authorized_key("kernel-signer", &[0x04, 0x05, 0x06]);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_secure_boot_kek_and_dbx() {
        let mut cfg = SecureBootConfig::default();
        cfg.add_kek("kek-owner", &[0x10]);
        cfg.add_forbidden_hash(&[0xDE, 0xAD]);
        assert_eq!(cfg.key_exchange_keys.len(), 1);
        assert_eq!(cfg.forbidden_hashes.len(), 1);
    }

    #[test]
    fn test_signature_algorithm_as_str() {
        assert_eq!(SignatureAlgorithm::MlDsa65.as_str(), "ML-DSA-65");
        assert_eq!(SignatureAlgorithm::MlDsa87.as_str(), "ML-DSA-87");
        assert_eq!(SignatureAlgorithm::Ed25519.as_str(), "Ed25519");
        assert_eq!(
            SignatureAlgorithm::HybridMlDsa65Ed25519.as_str(),
            "ML-DSA-65+Ed25519"
        );
    }

    // -- 25.2: PXE Boot -----------------------------------------------------

    #[test]
    fn test_pxe_legacy_config() {
        let cfg =
            PxeBootConfig::new_legacy("192.168.1.100", "/tftpboot/smallaios", Architecture::Amd64);
        assert_eq!(cfg.protocol, PxeProtocol::LegacyPxe);
        assert_eq!(cfg.server_address, "192.168.1.100");
        assert!(cfg.verify_signature);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_pxe_ipxe_config() {
        let cfg = PxeBootConfig::new_ipxe(
            "192.168.1.100",
            "/boot/smallaios",
            "http://192.168.1.100/boot.ipxe",
            Architecture::Arm64,
        );
        assert_eq!(cfg.protocol, PxeProtocol::Ipxe);
        assert!(cfg.ipxe_script_url.is_some());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_pxe_validate_empty_server() {
        let mut cfg = PxeBootConfig::new_legacy("x", "/k", Architecture::Amd64);
        cfg.server_address = String::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_pxe_validate_ipxe_no_script() {
        let mut cfg = PxeBootConfig::new_ipxe("x", "/k", "http://s/boot.ipxe", Architecture::Amd64);
        cfg.ipxe_script_url = None;
        assert!(cfg.validate().is_err());
    }

    // -- 25.3: VM Image Generation ------------------------------------------

    #[test]
    fn test_vm_image_qcow2() {
        let cfg = VmImageConfig::qcow2(Architecture::Amd64);
        assert_eq!(cfg.format, VmImageFormat::Qcow2);
        assert_eq!(cfg.disk_size_mb, 64);
        assert!(cfg.include_esp);
        assert!(cfg.compress);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_vm_image_raw() {
        let cfg = VmImageConfig::raw(Architecture::Arm64, 128);
        assert_eq!(cfg.format, VmImageFormat::Raw);
        assert_eq!(cfg.disk_size_mb, 128);
        assert!(!cfg.compress);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_vm_image_vmdk() {
        let cfg = VmImageConfig::vmdk(Architecture::RiscV64);
        assert_eq!(cfg.format, VmImageFormat::Vmdk);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_vm_image_format_extension() {
        assert_eq!(VmImageFormat::Raw.extension(), "img");
        assert_eq!(VmImageFormat::Qcow2.extension(), "qcow2");
        assert_eq!(VmImageFormat::Vmdk.extension(), "vmdk");
        assert_eq!(VmImageFormat::Vdi.extension(), "vdi");
    }

    #[test]
    fn test_vm_image_output_filename() {
        let cfg = VmImageConfig::qcow2(Architecture::Arm64);
        assert_eq!(cfg.output_filename(), "smallaios-arm64.qcow2");

        let cfg = VmImageConfig::raw(Architecture::RiscV64, 64);
        assert_eq!(cfg.output_filename(), "smallaios-riscv64.img");
    }

    #[test]
    fn test_vm_image_validate_zero_size() {
        let mut cfg = VmImageConfig::qcow2(Architecture::Amd64);
        cfg.disk_size_mb = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_vm_image_validate_too_small() {
        let mut cfg = VmImageConfig::qcow2(Architecture::Amd64);
        cfg.disk_size_mb = 8;
        assert!(cfg.validate().is_err());
    }

    // -- 25.4: Image Signing ------------------------------------------------

    #[test]
    fn test_image_signing_ml_dsa65() {
        let cfg = ImageSigningConfig::new_ml_dsa_65("SmallAIOS");
        assert_eq!(cfg.algorithm, SignatureAlgorithm::MlDsa65);
        assert_eq!(cfg.signer, "SmallAIOS");
        assert!(cfg.include_timestamp);
    }

    #[test]
    fn test_image_signing_hybrid() {
        let cfg = ImageSigningConfig::new_hybrid("SmallAIOS");
        assert_eq!(cfg.algorithm, SignatureAlgorithm::HybridMlDsa65Ed25519);
    }

    #[test]
    fn test_image_sign_and_verify() {
        let mut cfg = ImageSigningConfig::new_ml_dsa_65("SmallAIOS");
        cfg.set_keys(&[0xAA, 0xBB, 0xCC, 0xDD], &[0x11, 0x22, 0x33, 0x44]);

        let content_hash = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let sig = cfg.sign(&content_hash, 1000000);

        assert_eq!(sig.algorithm, SignatureAlgorithm::MlDsa65);
        assert_eq!(sig.signer, "SmallAIOS");
        assert_eq!(sig.timestamp, 1000000);

        assert!(cfg.verify(&sig, &content_hash));
    }

    #[test]
    fn test_image_sign_verify_wrong_hash() {
        let mut cfg = ImageSigningConfig::new_ml_dsa_65("test");
        cfg.set_keys(&[0xAA], &[0x11]);

        let hash = [0x01, 0x02, 0x03, 0x04];
        let sig = cfg.sign(&hash, 1000);

        let wrong_hash = [0xFF, 0xFE, 0xFD, 0xFC];
        assert!(!cfg.verify(&sig, &wrong_hash));
    }

    #[test]
    fn test_image_signing_validate_empty_signer() {
        let mut cfg = ImageSigningConfig::new_ml_dsa_65("");
        cfg.signer = String::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_image_signing_validate_ok() {
        let cfg = ImageSigningConfig::new_ml_dsa_65("test-signer");
        assert!(cfg.validate().is_ok());
    }

    // -- 25.5: OCI Multi-Arch -----------------------------------------------

    #[test]
    fn test_multi_arch_default_triple() {
        let cfg = MultiArchBuildConfig::default_triple("smallaios/inference", "v1.0.0");
        assert_eq!(cfg.arch_count(), 3);
        assert!(cfg.is_multi_arch());
        assert!(cfg.sign);
        assert!(cfg.signing_config.is_some());
        assert_eq!(cfg.max_size_mb, 15);
    }

    #[test]
    fn test_multi_arch_single() {
        let cfg =
            MultiArchBuildConfig::single_arch("smallaios/test", "latest", Architecture::Amd64);
        assert_eq!(cfg.arch_count(), 1);
        assert!(!cfg.is_multi_arch());
    }

    #[test]
    fn test_multi_arch_validate_ok() {
        let cfg = MultiArchBuildConfig::default_triple("smallaios/inference", "v1.0.0");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_multi_arch_validate_empty_repo() {
        let mut cfg = MultiArchBuildConfig::default_triple("x", "v1");
        cfg.repository = String::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_multi_arch_validate_empty_tag() {
        let mut cfg = MultiArchBuildConfig::default_triple("x", "v1");
        cfg.tag = String::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_multi_arch_validate_no_arch() {
        let mut cfg = MultiArchBuildConfig::default_triple("x", "v1");
        cfg.architectures.clear();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_multi_arch_validate_sign_no_config() {
        let mut cfg = MultiArchBuildConfig::default_triple("x", "v1");
        cfg.sign = true;
        cfg.signing_config = None;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_multi_arch_validate_push_no_registry() {
        let mut cfg = MultiArchBuildConfig::default_triple("x", "v1");
        cfg.push = true;
        cfg.registry_url = None;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_multi_arch_push_with_registry() {
        let mut cfg = MultiArchBuildConfig::default_triple("x", "v1");
        cfg.push = true;
        cfg.registry_url = Some(String::from("ghcr.io"));
        assert!(cfg.validate().is_ok());
    }
}
