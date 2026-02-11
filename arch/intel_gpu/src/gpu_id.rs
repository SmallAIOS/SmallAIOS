// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU identification -- architecture, EU count, memory, capabilities.
//!
//! Given a PCI device ID this module returns a [`GpuInfo`] that describes the
//! GPU architecture family (Xe-LP, Xe-HPG, Xe-HPC), EU count, local memory
//! size, SIMD width, and other parameters required by the compute and memory
//! management layers.

use crate::GpuError;

// ---------------------------------------------------------------------------
// GpuArchitecture
// ---------------------------------------------------------------------------

/// Intel GPU micro-architecture family.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuArchitecture {
    /// Xe-LP (Gen12) -- integrated GPUs in Tiger Lake, Alder Lake, Raptor Lake.
    XeLP,
    /// Xe-HPG -- discrete Arc GPUs (Alchemist: A770, A750, A580, A380).
    XeHPG,
    /// Xe-HPC -- data center GPUs (Ponte Vecchio / DC GPU Max series).
    XeHPC,
    /// Architecture could not be determined.
    Unknown,
}

// ---------------------------------------------------------------------------
// GpuGeneration
// ---------------------------------------------------------------------------

/// Intel GPU generation identifier.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuGeneration {
    /// Generation number (e.g. 12 for Gen12, 12.7 for Xe-HPG encoded as 127).
    pub version: u16,
}

impl GpuGeneration {
    /// Create a new generation identifier.
    pub fn new(version: u16) -> Self {
        Self { version }
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
    /// GPU generation.
    pub generation: GpuGeneration,
    /// Number of Execution Units (EUs).
    pub eu_count: u32,
    /// Number of Subslices (or Dual Subslices for Xe-HPG+).
    pub subslice_count: u32,
    /// Local memory (VRAM) in megabytes. 0 for shared-memory iGPUs.
    pub local_memory_mb: u32,
    /// SIMD width (8 for Xe-LP, 16 for Xe-HPG/HPC).
    pub simd_width: u32,
    /// Maximum threads per EU.
    pub max_threads_per_eu: u32,
    /// Maximum workgroup size (threads).
    pub max_workgroup_size: u32,
    /// Shared local memory per subslice in bytes.
    pub slm_per_subslice: u32,
    /// Whether XMX (matrix extension) engines are present.
    pub has_xmx: bool,
    /// Human-readable product name.
    pub name: &'static str,
}

impl GpuInfo {
    /// Total hardware threads across all EUs.
    pub fn total_threads(&self) -> u32 {
        self.eu_count * self.max_threads_per_eu
    }

    /// Total SIMD lanes across all EUs.
    ///
    /// Each EU can run `max_threads_per_eu` threads, each processing
    /// `simd_width` elements in parallel.
    pub fn total_simd_lanes(&self) -> u32 {
        self.eu_count * self.simd_width
    }

    /// Whether this GPU has dedicated local memory (discrete GPU).
    pub fn has_local_memory(&self) -> bool {
        self.local_memory_mb > 0
    }
}

// ---------------------------------------------------------------------------
// identify_gpu
// ---------------------------------------------------------------------------

/// Identify an Intel GPU from its PCI device ID.
///
/// Returns a fully-populated [`GpuInfo`] for known Intel GPU device IDs or
/// [`GpuError::UnsupportedDevice`] otherwise.
pub fn identify_gpu(device_id: u16) -> Result<GpuInfo, GpuError> {
    match device_id {
        // ----- Xe-LP (Gen12) -- Tiger Lake iGPU -------------------------
        0x9A40..=0x9A4F => {
            let (name, eu, ss) = match device_id {
                0x9A49 => ("Intel Iris Xe Graphics (Tiger Lake GT2)", 96, 6),
                0x9A40 => ("Intel UHD Graphics (Tiger Lake GT1)", 32, 2),
                _ => ("Intel Xe-LP GPU (Tiger Lake)", 64, 4),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeLP,
                generation: GpuGeneration::new(120),
                eu_count: eu,
                subslice_count: ss,
                local_memory_mb: 0, // shared system memory
                simd_width: 8,
                max_threads_per_eu: 7,
                max_workgroup_size: 512,
                slm_per_subslice: 64 * 1024,
                has_xmx: false,
                name,
            })
        }

        // ----- Xe-LP (Gen12) -- Alder Lake / Raptor Lake iGPU -----------
        0x4680..=0x468F | 0x4690..=0x469F | 0xA780..=0xA78F => {
            let (name, eu, ss) = match device_id {
                0x4680 => ("Intel Iris Xe Graphics (Alder Lake GT2)", 96, 6),
                0x4690 => ("Intel UHD Graphics (Alder Lake GT1)", 32, 2),
                0xA780 => ("Intel UHD Graphics (Raptor Lake)", 32, 2),
                _ => ("Intel Xe-LP GPU (Alder/Raptor Lake)", 64, 4),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeLP,
                generation: GpuGeneration::new(120),
                eu_count: eu,
                subslice_count: ss,
                local_memory_mb: 0,
                simd_width: 8,
                max_threads_per_eu: 7,
                max_workgroup_size: 512,
                slm_per_subslice: 64 * 1024,
                has_xmx: false,
                name,
            })
        }

        // ----- Xe-HPG (Alchemist / Arc A-series) -- Discrete ------------
        0x56A0..=0x56AF => {
            let (name, eu, ss, vram) = match device_id {
                0x56A0 => ("Intel Arc A770", 512, 32, 16384),
                0x56A1 => ("Intel Arc A750", 448, 28, 8192),
                0x56A5 => ("Intel Arc A580", 384, 24, 8192),
                _ => ("Intel Arc GPU (Xe-HPG)", 384, 24, 8192),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeHPG,
                generation: GpuGeneration::new(127),
                eu_count: eu,
                subslice_count: ss,
                local_memory_mb: vram,
                simd_width: 16,
                max_threads_per_eu: 8,
                max_workgroup_size: 1024,
                slm_per_subslice: 64 * 1024,
                has_xmx: true,
                name,
            })
        }

        // ----- Xe-HPG (Alchemist / Arc A-series lower tier) -------------
        0x56B0..=0x56BF => {
            let (name, eu, ss, vram) = match device_id {
                0x56B0 => ("Intel Arc A380", 128, 8, 6144),
                _ => ("Intel Arc GPU (Xe-HPG Entry)", 128, 8, 6144),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeHPG,
                generation: GpuGeneration::new(127),
                eu_count: eu,
                subslice_count: ss,
                local_memory_mb: vram,
                simd_width: 16,
                max_threads_per_eu: 8,
                max_workgroup_size: 1024,
                slm_per_subslice: 64 * 1024,
                has_xmx: true,
                name,
            })
        }

        // ----- Xe-HPC (Ponte Vecchio / DC GPU Max) ----------------------
        0x0BD0..=0x0BDF => {
            let (name, eu, ss, vram) = match device_id {
                0x0BD5 => ("Intel DC GPU Max 1550 (Ponte Vecchio)", 512, 32, 131072),
                0x0BD6 => ("Intel DC GPU Max 1100 (Ponte Vecchio)", 448, 28, 49152),
                0x0BD0 => ("Intel DC GPU Max 1350 (Ponte Vecchio)", 480, 30, 98304),
                _ => ("Intel DC GPU Max (Ponte Vecchio)", 448, 28, 49152),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::XeHPC,
                generation: GpuGeneration::new(128),
                eu_count: eu,
                subslice_count: ss,
                local_memory_mb: vram,
                simd_width: 16,
                max_threads_per_eu: 8,
                max_workgroup_size: 1024,
                slm_per_subslice: 128 * 1024,
                has_xmx: true,
                name,
            })
        }

        // ----- Unknown / unsupported ------------------------------------
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
    fn identify_tiger_lake_gt2() {
        let info = identify_gpu(0x9A49).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeLP);
        assert_eq!(info.generation.version, 120);
        assert_eq!(info.eu_count, 96);
        assert_eq!(info.local_memory_mb, 0);
        assert_eq!(info.simd_width, 8);
        assert_eq!(info.name, "Intel Iris Xe Graphics (Tiger Lake GT2)");
    }

    #[test]
    fn identify_tiger_lake_gt1() {
        let info = identify_gpu(0x9A40).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeLP);
        assert_eq!(info.eu_count, 32);
        assert_eq!(info.name, "Intel UHD Graphics (Tiger Lake GT1)");
    }

    #[test]
    fn identify_alder_lake_gt2() {
        let info = identify_gpu(0x4680).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeLP);
        assert_eq!(info.eu_count, 96);
        assert_eq!(info.name, "Intel Iris Xe Graphics (Alder Lake GT2)");
    }

    #[test]
    fn identify_arc_a770() {
        let info = identify_gpu(0x56A0).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPG);
        assert_eq!(info.eu_count, 512);
        assert_eq!(info.local_memory_mb, 16384);
        assert_eq!(info.simd_width, 16);
        assert!(info.has_xmx);
        assert_eq!(info.name, "Intel Arc A770");
    }

    #[test]
    fn identify_arc_a750() {
        let info = identify_gpu(0x56A1).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPG);
        assert_eq!(info.eu_count, 448);
        assert_eq!(info.local_memory_mb, 8192);
        assert_eq!(info.name, "Intel Arc A750");
    }

    #[test]
    fn identify_arc_a380() {
        let info = identify_gpu(0x56B0).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPG);
        assert_eq!(info.eu_count, 128);
        assert_eq!(info.local_memory_mb, 6144);
        assert_eq!(info.name, "Intel Arc A380");
    }

    #[test]
    fn identify_dc_gpu_max_1550() {
        let info = identify_gpu(0x0BD5).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPC);
        assert_eq!(info.eu_count, 512);
        assert_eq!(info.local_memory_mb, 131072);
        assert_eq!(info.simd_width, 16);
        assert!(info.has_xmx);
        assert_eq!(info.name, "Intel DC GPU Max 1550 (Ponte Vecchio)");
    }

    #[test]
    fn identify_dc_gpu_max_1100() {
        let info = identify_gpu(0x0BD6).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::XeHPC);
        assert_eq!(info.eu_count, 448);
        assert_eq!(info.local_memory_mb, 49152);
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

    // -- GpuGeneration ----------------------------------------------------

    #[test]
    fn generation_version() {
        let gen = GpuGeneration::new(120);
        assert_eq!(gen.version, 120);
    }

    // -- GpuInfo derived values -------------------------------------------

    #[test]
    fn total_threads_xe_lp() {
        let info = identify_gpu(0x9A49).unwrap();
        // 96 EUs * 7 threads/EU = 672
        assert_eq!(info.total_threads(), 96 * 7);
    }

    #[test]
    fn total_threads_xe_hpg() {
        let info = identify_gpu(0x56A0).unwrap();
        // 512 EUs * 8 threads/EU = 4096
        assert_eq!(info.total_threads(), 512 * 8);
    }

    #[test]
    fn total_threads_xe_hpc() {
        let info = identify_gpu(0x0BD5).unwrap();
        // 512 EUs * 8 threads/EU = 4096
        assert_eq!(info.total_threads(), 512 * 8);
    }

    #[test]
    fn total_simd_lanes_xe_lp() {
        let info = identify_gpu(0x9A49).unwrap();
        // 96 EUs * 8 SIMD width = 768
        assert_eq!(info.total_simd_lanes(), 96 * 8);
    }

    #[test]
    fn total_simd_lanes_xe_hpg() {
        let info = identify_gpu(0x56A0).unwrap();
        // 512 EUs * 16 SIMD width = 8192
        assert_eq!(info.total_simd_lanes(), 512 * 16);
    }

    #[test]
    fn has_local_memory_discrete() {
        let info = identify_gpu(0x56A0).unwrap();
        assert!(info.has_local_memory());
    }

    #[test]
    fn no_local_memory_integrated() {
        let info = identify_gpu(0x9A49).unwrap();
        assert!(!info.has_local_memory());
    }

    #[test]
    fn simd_width_xe_lp_is_8() {
        let info = identify_gpu(0x9A49).unwrap();
        assert_eq!(info.simd_width, 8);
    }

    #[test]
    fn simd_width_xe_hpg_is_16() {
        let info = identify_gpu(0x56A0).unwrap();
        assert_eq!(info.simd_width, 16);
    }

    #[test]
    fn simd_width_xe_hpc_is_16() {
        let info = identify_gpu(0x0BD5).unwrap();
        assert_eq!(info.simd_width, 16);
    }

    #[test]
    fn xmx_not_present_on_xe_lp() {
        let info = identify_gpu(0x9A49).unwrap();
        assert!(!info.has_xmx);
    }

    #[test]
    fn xmx_present_on_xe_hpg() {
        let info = identify_gpu(0x56A0).unwrap();
        assert!(info.has_xmx);
    }

    #[test]
    fn xmx_present_on_xe_hpc() {
        let info = identify_gpu(0x0BD5).unwrap();
        assert!(info.has_xmx);
    }

    // -- Architecture enum ------------------------------------------------

    #[test]
    fn architecture_enum_clone_debug() {
        let arch = GpuArchitecture::XeHPG;
        let cloned = arch.clone();
        assert_eq!(arch, cloned);
        let dbg = alloc::format!("{:?}", arch);
        assert!(dbg.contains("XeHPG"));
    }

    #[test]
    fn architecture_unknown_variant() {
        let arch = GpuArchitecture::Unknown;
        assert_eq!(arch, GpuArchitecture::Unknown);
    }
}
