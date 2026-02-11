// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU identification -- architecture, execution units, resources.
//!
//! Given a PCI device ID this module returns a [`GpuInfo`] that describes the
//! GPU architecture, EU count, slice/subslice topology, VRAM size, and other
//! parameters required by the compute-launch and memory-management layers.

use crate::GpuError;

// ---------------------------------------------------------------------------
// GpuArchitecture
// ---------------------------------------------------------------------------

/// Intel GPU micro-architecture family.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuArchitecture {
    /// Xe-LP -- integrated GPUs (DG1, Tiger Lake, Rocket Lake).
    XeLP,
    /// Xe-HPG -- Arc A-series discrete GPUs (Alchemist).
    XeHPG,
    /// Xe-HPC -- Data Center GPU Max (Ponte Vecchio).
    XeHPC,
    /// Xe-LPG -- Meteor Lake integrated GPU.
    XeLPG,
    /// Xe2 -- Battlemage (next-generation discrete).
    Xe2,
    /// Architecture could not be determined.
    Unknown,
}

// ---------------------------------------------------------------------------
// XeVersion
// ---------------------------------------------------------------------------

/// Numeric Xe generation version for compatibility comparison.
///
/// Higher values represent newer architectures with more features.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct XeVersion(pub u8);

impl XeVersion {
    /// Xe-LP generation.
    pub const XE_LP: XeVersion = XeVersion(1);
    /// Xe-HPG generation.
    pub const XE_HPG: XeVersion = XeVersion(2);
    /// Xe-HPC generation.
    pub const XE_HPC: XeVersion = XeVersion(3);
    /// Xe-LPG generation.
    pub const XE_LPG: XeVersion = XeVersion(4);
    /// Xe2 generation.
    pub const XE2: XeVersion = XeVersion(5);

    /// Create a new Xe version.
    pub fn new(version: u8) -> Self {
        Self(version)
    }

    /// Return the raw version number.
    pub fn as_version(&self) -> u8 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// GpuInfo
// ---------------------------------------------------------------------------

/// Static description of a particular Intel GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuInfo {
    /// PCI device ID used to identify the GPU.
    pub device_id: u16,
    /// Micro-architecture family.
    pub architecture: GpuArchitecture,
    /// Xe generation version for compatibility checks.
    pub xe_version: XeVersion,
    /// Number of Execution Units (EUs).
    pub eu_count: u32,
    /// Number of slices.
    pub slices: u32,
    /// Number of subslices (dual subslices on Xe-HPG+).
    pub subslices: u32,
    /// Video RAM in megabytes (0 for integrated GPUs using system memory).
    pub vram_size_mb: u32,
    /// Maximum hardware threads per EU (always 8 on current Intel GPUs).
    pub max_threads_per_eu: u32,
    /// Human-readable product name.
    pub name: &'static str,
}

impl GpuInfo {
    /// Total hardware threads across all EUs.
    ///
    /// Each Intel EU supports 8 concurrent hardware threads.
    pub fn total_threads(&self) -> u32 {
        self.eu_count * self.max_threads_per_eu
    }

    /// Total EUs per subslice (derived).
    pub fn eus_per_subslice(&self) -> u32 {
        if self.subslices == 0 {
            return 0;
        }
        self.eu_count / self.subslices
    }

    /// Whether this GPU has dedicated VRAM (discrete GPU).
    pub fn has_dedicated_vram(&self) -> bool {
        self.vram_size_mb > 0
    }
}

// ---------------------------------------------------------------------------
// identify_gpu
// ---------------------------------------------------------------------------

/// Identify a GPU from its PCI device ID.
///
/// Returns a fully-populated [`GpuInfo`] for known Intel device IDs or
/// [`GpuError::UnsupportedDevice`] otherwise.
pub fn identify_gpu(device_id: u16) -> Result<GpuInfo, GpuError> {
    match device_id {
        // ----- Xe-LP (integrated / DG1) ----------------------------------
        0x4905..=0x4908 => {
            let (name, eus, slices, subslices, vram) = match device_id {
                0x4905 => ("Intel DG1 (Xe-LP)", 96, 1, 6, 4096),
                0x4906 => ("Intel Iris Xe (Tiger Lake)", 96, 1, 6, 0),
                0x4907 => ("Intel Iris Xe (Rocket Lake)", 32, 1, 2, 0),
                _ => ("Intel Xe-LP GPU", 96, 1, 6, 0),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeLP,
                xe_version: XeVersion::XE_LP,
                eu_count: eus,
                slices,
                subslices,
                vram_size_mb: vram,
                max_threads_per_eu: 8,
                name,
            })
        }

        // ----- Xe-HPG (Arc A-series, Alchemist) --------------------------
        // ACM-G10: Arc A770 (0x56A0), A750 (0x56A1), A580 (0x56A2)
        // ACM-G11: Arc A380 (0x56A5)
        0x56A0..=0x56AF => {
            let (name, eus, slices, subslices, vram) = match device_id {
                0x56A0 => ("Intel Arc A770", 512, 8, 32, 16384),
                0x56A1 => ("Intel Arc A750", 448, 7, 28, 8192),
                0x56A2 => ("Intel Arc A580", 384, 6, 24, 8192),
                0x56A5 => ("Intel Arc A380", 128, 2, 8, 6144),
                _ => ("Intel Arc GPU (Xe-HPG)", 512, 8, 32, 16384),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeHPG,
                xe_version: XeVersion::XE_HPG,
                eu_count: eus,
                slices,
                subslices,
                vram_size_mb: vram,
                max_threads_per_eu: 8,
                name,
            })
        }

        // ----- Xe-HPG (Flex series) --------------------------------------
        0x56C0..=0x56CF => {
            let (name, eus, slices, subslices, vram) = match device_id {
                0x56C0 => ("Intel Data Center GPU Flex 170", 512, 8, 32, 16384),
                0x56C1 => ("Intel Data Center GPU Flex 140", 256, 4, 16, 12288),
                _ => ("Intel Flex GPU (Xe-HPG)", 512, 8, 32, 16384),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeHPG,
                xe_version: XeVersion::XE_HPG,
                eu_count: eus,
                slices,
                subslices,
                vram_size_mb: vram,
                max_threads_per_eu: 8,
                name,
            })
        }

        // ----- Xe-HPC (Data Center GPU Max / Ponte Vecchio) --------------
        0x0BD0..=0x0BDF => {
            let (name, eus, slices, subslices, vram) = match device_id {
                0x0BD5 => ("Intel Data Center GPU Max 1550", 1024, 8, 64, 131072),
                0x0BD6 => ("Intel Data Center GPU Max 1100", 896, 7, 56, 49152),
                0x0BD9 => ("Intel Data Center GPU Max 1350", 1024, 8, 64, 98304),
                _ => ("Intel Data Center GPU Max", 1024, 8, 64, 131072),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeHPC,
                xe_version: XeVersion::XE_HPC,
                eu_count: eus,
                slices,
                subslices,
                vram_size_mb: vram,
                max_threads_per_eu: 8,
                name,
            })
        }

        // ----- Xe-LPG (Meteor Lake integrated) ---------------------------
        0x7D40..=0x7D6F => {
            let (name, eus, slices, subslices) = match device_id {
                0x7D45 => ("Intel Meteor Lake Iris Xe (GT2)", 128, 2, 8),
                0x7D55 => ("Intel Meteor Lake Iris Xe (GT1)", 64, 1, 4),
                _ => ("Intel Meteor Lake GPU (Xe-LPG)", 128, 2, 8),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeLPG,
                xe_version: XeVersion::XE_LPG,
                eu_count: eus,
                slices,
                subslices,
                vram_size_mb: 0, // Integrated, uses system memory.
                max_threads_per_eu: 8,
                name,
            })
        }

        // ----- Xe2 (Battlemage) ------------------------------------------
        0xE202..=0xE20F => {
            let (name, eus, slices, subslices, vram) = match device_id {
                0xE202 => ("Intel Battlemage B580", 320, 5, 20, 12288),
                0xE20B => ("Intel Battlemage B570", 288, 4, 18, 10240),
                _ => ("Intel Xe2 GPU (Battlemage)", 320, 5, 20, 12288),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Xe2,
                xe_version: XeVersion::XE2,
                eu_count: eus,
                slices,
                subslices,
                vram_size_mb: vram,
                max_threads_per_eu: 8,
                name,
            })
        }

        // ----- Unknown / unsupported -------------------------------------
        _ => Err(GpuError::UnsupportedDevice),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Individual GPU identification ------------------------------------

    #[test]
    fn identify_arc_a770() {
        let info = identify_gpu(0x56A0).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPG);
        assert_eq!(info.xe_version, XeVersion::XE_HPG);
        assert_eq!(info.eu_count, 512);
        assert_eq!(info.slices, 8);
        assert_eq!(info.subslices, 32);
        assert_eq!(info.vram_size_mb, 16384);
        assert_eq!(info.name, "Intel Arc A770");
    }

    #[test]
    fn identify_arc_a750() {
        let info = identify_gpu(0x56A1).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPG);
        assert_eq!(info.eu_count, 448);
        assert_eq!(info.vram_size_mb, 8192);
        assert_eq!(info.name, "Intel Arc A750");
    }

    #[test]
    fn identify_arc_a580() {
        let info = identify_gpu(0x56A2).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPG);
        assert_eq!(info.eu_count, 384);
        assert_eq!(info.vram_size_mb, 8192);
        assert_eq!(info.name, "Intel Arc A580");
    }

    #[test]
    fn identify_arc_a380() {
        let info = identify_gpu(0x56A5).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPG);
        assert_eq!(info.eu_count, 128);
        assert_eq!(info.vram_size_mb, 6144);
        assert_eq!(info.name, "Intel Arc A380");
    }

    #[test]
    fn identify_dc_max_1550() {
        let info = identify_gpu(0x0BD5).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPC);
        assert_eq!(info.xe_version, XeVersion::XE_HPC);
        assert_eq!(info.eu_count, 1024);
        assert_eq!(info.vram_size_mb, 131072);
        assert_eq!(info.name, "Intel Data Center GPU Max 1550");
    }

    #[test]
    fn identify_dc_max_1100() {
        let info = identify_gpu(0x0BD6).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPC);
        assert_eq!(info.eu_count, 896);
        assert_eq!(info.vram_size_mb, 49152);
        assert_eq!(info.name, "Intel Data Center GPU Max 1100");
    }

    #[test]
    fn identify_flex_170() {
        let info = identify_gpu(0x56C0).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPG);
        assert_eq!(info.eu_count, 512);
        assert_eq!(info.vram_size_mb, 16384);
        assert_eq!(info.name, "Intel Data Center GPU Flex 170");
    }

    #[test]
    fn identify_dg1_xe_lp() {
        let info = identify_gpu(0x4905).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeLP);
        assert_eq!(info.xe_version, XeVersion::XE_LP);
        assert_eq!(info.eu_count, 96);
        assert_eq!(info.vram_size_mb, 4096);
        assert_eq!(info.name, "Intel DG1 (Xe-LP)");
    }

    #[test]
    fn identify_iris_xe_tiger_lake() {
        let info = identify_gpu(0x4906).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeLP);
        assert_eq!(info.vram_size_mb, 0); // Integrated
        assert!(!info.has_dedicated_vram());
    }

    #[test]
    fn identify_meteor_lake_gt2() {
        let info = identify_gpu(0x7D45).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeLPG);
        assert_eq!(info.xe_version, XeVersion::XE_LPG);
        assert_eq!(info.eu_count, 128);
        assert_eq!(info.vram_size_mb, 0); // Integrated
    }

    #[test]
    fn identify_battlemage_b580() {
        let info = identify_gpu(0xE202).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Xe2);
        assert_eq!(info.xe_version, XeVersion::XE2);
        assert_eq!(info.eu_count, 320);
        assert_eq!(info.vram_size_mb, 12288);
        assert_eq!(info.name, "Intel Battlemage B580");
    }

    #[test]
    fn unknown_device_returns_error() {
        let result = identify_gpu(0xFFFF);
        assert_eq!(result, Err(GpuError::UnsupportedDevice));
    }

    #[test]
    fn unknown_device_in_gap_returns_error() {
        let result = identify_gpu(0x0001);
        assert_eq!(result, Err(GpuError::UnsupportedDevice));
    }

    // -- XeVersion --------------------------------------------------------

    #[test]
    fn xe_version_ordering() {
        assert!(XeVersion::XE_LP < XeVersion::XE_HPG);
        assert!(XeVersion::XE_HPG < XeVersion::XE_HPC);
        assert!(XeVersion::XE_HPC < XeVersion::XE_LPG);
        assert!(XeVersion::XE_LPG < XeVersion::XE2);
    }

    #[test]
    fn xe_version_as_version() {
        assert_eq!(XeVersion::XE_LP.as_version(), 1);
        assert_eq!(XeVersion::XE2.as_version(), 5);
    }

    // -- GpuInfo derived values -------------------------------------------

    #[test]
    fn total_threads_arc_a770() {
        let info = identify_gpu(0x56A0).unwrap();
        // 512 EUs * 8 threads/EU = 4096
        assert_eq!(info.total_threads(), 512 * 8);
    }

    #[test]
    fn total_threads_dc_max_1550() {
        let info = identify_gpu(0x0BD5).unwrap();
        // 1024 EUs * 8 threads/EU = 8192
        assert_eq!(info.total_threads(), 1024 * 8);
    }

    #[test]
    fn eus_per_subslice_arc_a770() {
        let info = identify_gpu(0x56A0).unwrap();
        // 512 EUs / 32 subslices = 16 EUs per subslice
        assert_eq!(info.eus_per_subslice(), 16);
    }

    #[test]
    fn has_dedicated_vram_discrete() {
        let info = identify_gpu(0x56A0).unwrap();
        assert!(info.has_dedicated_vram());
    }

    #[test]
    fn threads_per_eu_always_8() {
        for &dev_id in &[0x4905, 0x56A0, 0x0BD5, 0x7D45, 0xE202] {
            let info = identify_gpu(dev_id).unwrap();
            assert_eq!(
                info.max_threads_per_eu, 8,
                "threads per EU must be 8 for device 0x{:04X}",
                dev_id
            );
        }
    }

    // -- Architecture enum ------------------------------------------------

    #[test]
    fn architecture_enum_clone_debug() {
        let arch = GpuArchitecture::XeHPC;
        let cloned = arch.clone();
        assert_eq!(arch, cloned);
        let dbg = alloc::format!("{:?}", arch);
        assert!(dbg.contains("XeHPC"));
    }

    #[test]
    fn architecture_unknown_variant() {
        let arch = GpuArchitecture::Unknown;
        assert_eq!(arch, GpuArchitecture::Unknown);
    }
}
