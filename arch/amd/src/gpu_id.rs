// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU identification -- architecture, GFX version, resources.
//!
//! Given a PCI device ID this module returns a [`GpuInfo`] that describes the
//! GPU architecture, compute unit count, VRAM size, wavefront geometry, and
//! other parameters required by the compute-launch and memory-management layers.

use crate::GpuError;

// ---------------------------------------------------------------------------
// GpuArchitecture
// ---------------------------------------------------------------------------

/// AMD GPU micro-architecture family.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuArchitecture {
    /// RDNA 1 (Navi 10/14) -- RX 5600/5700 series.
    Rdna1,
    /// RDNA 2 (Navi 21/22/23) -- RX 6700/6800/6900 series.
    Rdna2,
    /// RDNA 3 (Navi 31/32/33) -- RX 7800/7900 series.
    Rdna3,
    /// CDNA 1 (Arcturus) -- Instinct MI100.
    Cdna1,
    /// CDNA 2 (Aldebaran) -- Instinct MI200 series.
    Cdna2,
    /// CDNA 3 -- Instinct MI300 series.
    Cdna3,
    /// Architecture could not be determined.
    Unknown,
}

impl GpuArchitecture {
    /// Returns `true` for CDNA architectures (datacenter compute).
    pub fn is_cdna(&self) -> bool {
        matches!(self, Self::Cdna1 | Self::Cdna2 | Self::Cdna3)
    }

    /// Returns `true` for RDNA architectures (consumer/gaming).
    pub fn is_rdna(&self) -> bool {
        matches!(self, Self::Rdna1 | Self::Rdna2 | Self::Rdna3)
    }
}

// ---------------------------------------------------------------------------
// GfxVersion
// ---------------------------------------------------------------------------

/// AMD GFX ISA version (e.g., gfx908, gfx942, gfx1100).
#[derive(Clone, Debug, PartialEq)]
pub struct GfxVersion {
    pub major: u16,
    pub minor: u16,
    pub stepping: u16,
}

impl GfxVersion {
    /// Create a new GFX version.
    pub fn new(major: u16, minor: u16, stepping: u16) -> Self {
        Self {
            major,
            minor,
            stepping,
        }
    }

    /// Encode as the `gfxNNN` version used by ROCm toolchains.
    ///
    /// Example: GFX 9.0.8 -> `gfx908` -> returns `908`.
    pub fn as_gfx_id(&self) -> u32 {
        self.major as u32 * 100 + self.minor as u32 * 10 + self.stepping as u32
    }
}

// ---------------------------------------------------------------------------
// GpuInfo
// ---------------------------------------------------------------------------

/// Static description of a particular AMD GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuInfo {
    /// PCI device ID used to identify the GPU.
    pub device_id: u16,
    /// Micro-architecture family.
    pub architecture: GpuArchitecture,
    /// GFX ISA version.
    pub gfx_version: GfxVersion,
    /// Number of Compute Units (CU).
    pub cu_count: u32,
    /// Video RAM in megabytes.
    pub vram_size_mb: u32,
    /// Maximum wavefronts per CU.
    pub max_wavefronts_per_cu: u32,
    /// Threads per wavefront (64 for CDNA, 32 for RDNA wave32).
    pub wavefront_size: u32,
    /// Shared memory (LDS) per CU in bytes.
    pub lds_per_cu: u32,
    /// Vector GPRs per CU.
    pub vgprs_per_cu: u32,
    /// Human-readable product name.
    pub name: &'static str,
}

impl GpuInfo {
    /// Total stream processors on the GPU.
    ///
    /// The number of stream processors per CU varies by architecture:
    /// - RDNA: 64 stream processors per CU (2 WGPs of 32 each)
    /// - CDNA: 64 stream processors per CU
    pub fn total_stream_processors(&self) -> u32 {
        self.cu_count * 64
    }

    /// Maximum number of threads that can be resident across all CUs.
    pub fn max_concurrent_threads(&self) -> u32 {
        self.cu_count * self.max_wavefronts_per_cu * self.wavefront_size
    }

    /// Whether this GPU supports wave32 mode (RDNA only).
    pub fn supports_wave32(&self) -> bool {
        self.architecture.is_rdna()
    }
}

// ---------------------------------------------------------------------------
// identify_gpu
// ---------------------------------------------------------------------------

/// Identify a GPU from its PCI device ID.
///
/// Returns a fully-populated [`GpuInfo`] for known AMD device IDs or
/// [`GpuError::UnsupportedDevice`] otherwise.
pub fn identify_gpu(device_id: u16) -> Result<GpuInfo, GpuError> {
    match device_id {
        // ----- RDNA 1 (gfx1010) -- RX 5000 series -----------------------
        0x7310..=0x737F => {
            let (name, cu, vram) = match device_id {
                0x7310 => ("AMD Radeon RX 5600 XT", 36, 6144),
                0x7312 => ("AMD Radeon RX 5700", 36, 8192),
                0x7318 => ("AMD Radeon RX 5700 XT", 40, 8192),
                _ => ("AMD RDNA 1 GPU", 36, 8192),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Rdna1,
                gfx_version: GfxVersion::new(10, 1, 0),
                cu_count: cu,
                vram_size_mb: vram,
                max_wavefronts_per_cu: 20,
                wavefront_size: 32,
                lds_per_cu: 128 * 1024,
                vgprs_per_cu: 1024,
                name,
            })
        }

        // ----- CDNA 1 (gfx908) -- MI100 ---------------------------------
        0x7380..=0x739F => {
            let (name, cu, vram) = match device_id {
                0x7388 => ("AMD Instinct MI100", 120, 32768),
                _ => ("AMD CDNA 1 GPU", 120, 32768),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Cdna1,
                gfx_version: GfxVersion::new(9, 0, 8),
                cu_count: cu,
                vram_size_mb: vram,
                max_wavefronts_per_cu: 40,
                wavefront_size: 64,
                lds_per_cu: 64 * 1024,
                vgprs_per_cu: 1024,
                name,
            })
        }

        // ----- RDNA 2 (gfx1030) -- RX 6000 series -----------------------
        0x73A0..=0x73EF => {
            let (name, cu, vram) = match device_id {
                0x73A2 => ("AMD Radeon RX 6700 XT", 40, 12288),
                0x73BF => ("AMD Radeon RX 6900 XT", 80, 16384),
                0x73C0 => ("AMD Radeon RX 6800 XT", 72, 16384),
                _ => ("AMD RDNA 2 GPU", 60, 16384),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Rdna2,
                gfx_version: GfxVersion::new(10, 3, 0),
                cu_count: cu,
                vram_size_mb: vram,
                max_wavefronts_per_cu: 20,
                wavefront_size: 32,
                lds_per_cu: 128 * 1024,
                vgprs_per_cu: 1024,
                name,
            })
        }

        // ----- CDNA 2 (gfx90a) -- MI200 series --------------------------
        0x7400..=0x743F => {
            let (name, cu, vram) = match device_id {
                0x7400 => ("AMD Instinct MI250X", 220, 131072),
                0x7402 => ("AMD Instinct MI250", 208, 131072),
                0x7408 => ("AMD Instinct MI210", 104, 65536),
                _ => ("AMD CDNA 2 GPU", 220, 131072),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Cdna2,
                gfx_version: GfxVersion::new(9, 0, 10),
                cu_count: cu,
                vram_size_mb: vram,
                max_wavefronts_per_cu: 40,
                wavefront_size: 64,
                lds_per_cu: 64 * 1024,
                vgprs_per_cu: 1024,
                name,
            })
        }

        // ----- RDNA 3 (gfx1100) -- RX 7000 series -----------------------
        0x7440..=0x74FF => {
            let (name, cu, vram) = match device_id {
                0x7440 => ("AMD Radeon RX 7900 XTX", 96, 24576),
                0x7448 => ("AMD Radeon RX 7900 XT", 84, 20480),
                0x7480 => ("AMD Radeon RX 7800 XT", 60, 16384),
                _ => ("AMD RDNA 3 GPU", 60, 16384),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Rdna3,
                gfx_version: GfxVersion::new(11, 0, 0),
                cu_count: cu,
                vram_size_mb: vram,
                max_wavefronts_per_cu: 20,
                wavefront_size: 32,
                lds_per_cu: 128 * 1024,
                vgprs_per_cu: 1536,
                name,
            })
        }

        // ----- CDNA 3 (gfx942) -- MI300 series --------------------------
        0x7500..=0x75FF => {
            let (name, cu, vram) = match device_id {
                0x7500 => ("AMD Instinct MI300X", 304, 196608),
                0x7502 => ("AMD Instinct MI300A", 228, 131072),
                0x7508 => ("AMD Instinct MI325X", 304, 262144),
                _ => ("AMD CDNA 3 GPU", 304, 196608),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Cdna3,
                gfx_version: GfxVersion::new(9, 4, 2),
                cu_count: cu,
                vram_size_mb: vram,
                max_wavefronts_per_cu: 40,
                wavefront_size: 64,
                lds_per_cu: 64 * 1024,
                vgprs_per_cu: 1024,
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
    fn identify_mi300x_cdna3() {
        let info = identify_gpu(0x7500).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Cdna3);
        assert_eq!(info.gfx_version, GfxVersion::new(9, 4, 2));
        assert_eq!(info.cu_count, 304);
        assert_eq!(info.vram_size_mb, 196608);
        assert_eq!(info.name, "AMD Instinct MI300X");
    }

    #[test]
    fn identify_mi250x_cdna2() {
        let info = identify_gpu(0x7400).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Cdna2);
        assert_eq!(info.gfx_version, GfxVersion::new(9, 0, 10));
        assert_eq!(info.cu_count, 220);
        assert_eq!(info.vram_size_mb, 131072);
        assert_eq!(info.name, "AMD Instinct MI250X");
    }

    #[test]
    fn identify_mi100_cdna1() {
        let info = identify_gpu(0x7388).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Cdna1);
        assert_eq!(info.gfx_version, GfxVersion::new(9, 0, 8));
        assert_eq!(info.cu_count, 120);
        assert_eq!(info.vram_size_mb, 32768);
        assert_eq!(info.name, "AMD Instinct MI100");
    }

    #[test]
    fn identify_rx7900xtx_rdna3() {
        let info = identify_gpu(0x7440).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Rdna3);
        assert_eq!(info.gfx_version, GfxVersion::new(11, 0, 0));
        assert_eq!(info.cu_count, 96);
        assert_eq!(info.vram_size_mb, 24576);
        assert_eq!(info.name, "AMD Radeon RX 7900 XTX");
    }

    #[test]
    fn identify_rx6900xt_rdna2() {
        let info = identify_gpu(0x73BF).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Rdna2);
        assert_eq!(info.gfx_version, GfxVersion::new(10, 3, 0));
        assert_eq!(info.cu_count, 80);
        assert_eq!(info.vram_size_mb, 16384);
        assert_eq!(info.name, "AMD Radeon RX 6900 XT");
    }

    #[test]
    fn identify_rx5700xt_rdna1() {
        let info = identify_gpu(0x7318).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Rdna1);
        assert_eq!(info.gfx_version, GfxVersion::new(10, 1, 0));
        assert_eq!(info.cu_count, 40);
        assert_eq!(info.vram_size_mb, 8192);
        assert_eq!(info.name, "AMD Radeon RX 5700 XT");
    }

    #[test]
    fn identify_mi325x_cdna3() {
        let info = identify_gpu(0x7508).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Cdna3);
        assert_eq!(info.cu_count, 304);
        assert_eq!(info.vram_size_mb, 262144);
        assert_eq!(info.name, "AMD Instinct MI325X");
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

    // -- GfxVersion -------------------------------------------------------

    #[test]
    fn gfx_version_id_908() {
        let gfx = GfxVersion::new(9, 0, 8);
        assert_eq!(gfx.as_gfx_id(), 908);
    }

    #[test]
    fn gfx_version_id_942() {
        let gfx = GfxVersion::new(9, 4, 2);
        assert_eq!(gfx.as_gfx_id(), 942);
    }

    #[test]
    fn gfx_version_id_1100() {
        let gfx = GfxVersion::new(11, 0, 0);
        assert_eq!(gfx.as_gfx_id(), 1100);
    }

    #[test]
    fn gfx_version_id_1030() {
        let gfx = GfxVersion::new(10, 3, 0);
        assert_eq!(gfx.as_gfx_id(), 1030);
    }

    // -- GpuInfo derived values -------------------------------------------

    #[test]
    fn total_stream_processors_cdna3() {
        let info = identify_gpu(0x7500).unwrap();
        // 304 CU * 64 SP = 19456
        assert_eq!(info.total_stream_processors(), 304 * 64);
    }

    #[test]
    fn total_stream_processors_rdna3() {
        let info = identify_gpu(0x7440).unwrap();
        // 96 CU * 64 SP = 6144
        assert_eq!(info.total_stream_processors(), 96 * 64);
    }

    #[test]
    fn max_concurrent_threads_mi300x() {
        let info = identify_gpu(0x7500).unwrap();
        // 304 CU * 40 wavefronts * 64 threads = 778240
        assert_eq!(info.max_concurrent_threads(), 304 * 40 * 64);
    }

    #[test]
    fn max_concurrent_threads_rdna3() {
        let info = identify_gpu(0x7440).unwrap();
        // 96 CU * 20 wavefronts * 32 threads = 61440
        assert_eq!(info.max_concurrent_threads(), 96 * 20 * 32);
    }

    #[test]
    fn wavefront_size_cdna_is_64() {
        for &dev_id in &[0x7388, 0x7400, 0x7500] {
            let info = identify_gpu(dev_id).unwrap();
            assert_eq!(
                info.wavefront_size, 64,
                "CDNA wavefront must be 64 for device 0x{:04X}",
                dev_id
            );
        }
    }

    #[test]
    fn wavefront_size_rdna_is_32() {
        for &dev_id in &[0x7318, 0x73BF, 0x7440] {
            let info = identify_gpu(dev_id).unwrap();
            assert_eq!(
                info.wavefront_size, 32,
                "RDNA wavefront must be 32 for device 0x{:04X}",
                dev_id
            );
        }
    }

    #[test]
    fn supports_wave32_rdna_only() {
        let rdna = identify_gpu(0x7440).unwrap();
        assert!(rdna.supports_wave32());
        let cdna = identify_gpu(0x7500).unwrap();
        assert!(!cdna.supports_wave32());
    }

    // -- Architecture classification --------------------------------------

    #[test]
    fn architecture_is_cdna() {
        assert!(GpuArchitecture::Cdna1.is_cdna());
        assert!(GpuArchitecture::Cdna2.is_cdna());
        assert!(GpuArchitecture::Cdna3.is_cdna());
        assert!(!GpuArchitecture::Rdna1.is_cdna());
        assert!(!GpuArchitecture::Rdna3.is_cdna());
        assert!(!GpuArchitecture::Unknown.is_cdna());
    }

    #[test]
    fn architecture_is_rdna() {
        assert!(GpuArchitecture::Rdna1.is_rdna());
        assert!(GpuArchitecture::Rdna2.is_rdna());
        assert!(GpuArchitecture::Rdna3.is_rdna());
        assert!(!GpuArchitecture::Cdna1.is_rdna());
        assert!(!GpuArchitecture::Cdna3.is_rdna());
        assert!(!GpuArchitecture::Unknown.is_rdna());
    }

    #[test]
    fn architecture_enum_clone_debug() {
        let arch = GpuArchitecture::Cdna3;
        let cloned = arch.clone();
        assert_eq!(arch, cloned);
        let dbg = alloc::format!("{:?}", arch);
        assert!(dbg.contains("Cdna3"));
    }

    #[test]
    fn architecture_unknown_variant() {
        let arch = GpuArchitecture::Unknown;
        assert_eq!(arch, GpuArchitecture::Unknown);
    }
}
