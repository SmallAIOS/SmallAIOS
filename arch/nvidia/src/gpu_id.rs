// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU identification — architecture, compute capability, resources.
//!
//! Given a PCI device ID this module returns a [`GpuInfo`] that describes the
//! GPU architecture, SM count, VRAM size, warp geometry, and other parameters
//! required by the compute-launch and memory-management layers.

#![allow(dead_code)]

use crate::GpuError;

// ---------------------------------------------------------------------------
// GpuArchitecture
// ---------------------------------------------------------------------------

/// NVIDIA GPU micro-architecture family.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuArchitecture {
    /// Maxwell (CC 5.x) — Jetson Nano, GTX 9xx.
    Maxwell,
    /// Pascal (CC 6.x) — GTX 10xx, Tesla P100.
    Pascal,
    /// Volta (CC 7.0) — Tesla V100, Titan V.
    Volta,
    /// Turing (CC 7.5) — Tesla T4, RTX 20xx.
    Turing,
    /// Ampere (CC 8.x) — A100, A30, Jetson Orin.
    Ampere,
    /// Hopper (CC 9.0) — H100, H200.
    Hopper,
    /// Blackwell (CC 10.0) — B200, DGX Spark.
    Blackwell,
    /// Architecture could not be determined.
    Unknown,
}

// ---------------------------------------------------------------------------
// ComputeCapability
// ---------------------------------------------------------------------------

/// NVIDIA compute-capability version (major.minor).
#[derive(Clone, Debug, PartialEq)]
pub struct ComputeCapability {
    pub major: u8,
    pub minor: u8,
}

impl ComputeCapability {
    /// Create a new compute-capability pair.
    pub fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// Encode as the `sm_XX` version used by PTX / SASS toolchains.
    ///
    /// Example: CC 7.5 → `sm_75` → returns `75`.
    pub fn as_sm_version(&self) -> u16 {
        self.major as u16 * 10 + self.minor as u16
    }
}

// ---------------------------------------------------------------------------
// GpuInfo
// ---------------------------------------------------------------------------

/// Static description of a particular NVIDIA GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuInfo {
    /// PCI device ID used to identify the GPU.
    pub device_id: u16,
    /// Micro-architecture family.
    pub architecture: GpuArchitecture,
    /// Compute capability (major.minor).
    pub compute_capability: ComputeCapability,
    /// Number of Streaming Multiprocessors.
    pub sm_count: u32,
    /// Video RAM in megabytes.
    pub vram_size_mb: u32,
    /// Maximum resident threads per SM.
    pub max_threads_per_sm: u32,
    /// Maximum resident warps per SM.
    pub max_warps_per_sm: u32,
    /// Threads per warp (always 32 on current NVIDIA hardware).
    pub warp_size: u32,
    /// Shared memory per SM in bytes.
    pub max_shared_memory_per_sm: u32,
    /// 32-bit registers per SM.
    pub max_registers_per_sm: u32,
    /// Human-readable product name.
    pub name: &'static str,
}

impl GpuInfo {
    /// Total CUDA cores on the GPU.
    ///
    /// The number of FP32 cores per SM varies by architecture:
    /// - Maxwell / Pascal: 128
    /// - Volta / Turing: 64
    /// - Ampere / Hopper / Blackwell: 128
    pub fn total_cuda_cores(&self) -> u32 {
        let cores_per_sm = match self.architecture {
            GpuArchitecture::Maxwell | GpuArchitecture::Pascal => 128,
            GpuArchitecture::Volta | GpuArchitecture::Turing => 64,
            GpuArchitecture::Ampere | GpuArchitecture::Hopper | GpuArchitecture::Blackwell => 128,
            GpuArchitecture::Unknown => 0,
        };
        self.sm_count * cores_per_sm
    }

    /// Maximum number of threads that can be resident across all SMs.
    pub fn max_concurrent_threads(&self) -> u32 {
        self.sm_count * self.max_threads_per_sm
    }
}

// ---------------------------------------------------------------------------
// identify_gpu
// ---------------------------------------------------------------------------

/// Identify a GPU from its PCI device ID.
///
/// Returns a fully-populated [`GpuInfo`] for known NVIDIA device IDs or
/// [`GpuError::UnsupportedDevice`] otherwise.
pub fn identify_gpu(device_id: u16) -> Result<GpuInfo, GpuError> {
    match device_id {
        // ----- Maxwell (CC 5.3) — Jetson Nano / TX1 ----------------------
        0x1340..=0x137F => {
            let (name, sm, vram) = match device_id {
                0x1340 => ("NVIDIA Jetson Nano (Maxwell)", 1, 4096),
                _ => ("NVIDIA Maxwell GPU", 4, 2048),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Maxwell,
                compute_capability: ComputeCapability::new(5, 3),
                sm_count: sm,
                vram_size_mb: vram,
                max_threads_per_sm: 2048,
                max_warps_per_sm: 64,
                warp_size: 32,
                max_shared_memory_per_sm: 96 * 1024,
                max_registers_per_sm: 65536,
                name,
            })
        }

        // ----- Volta (CC 7.0) — V100 ------------------------------------
        0x1DB0..=0x1DBF => {
            let (name, sm, vram) = match device_id {
                0x1DB1 => ("NVIDIA Tesla V100-SXM2-16GB", 80, 16384),
                0x1DB4 => ("NVIDIA Tesla V100-PCIE-16GB", 80, 16384),
                0x1DB5 => ("NVIDIA Tesla V100-SXM2-32GB", 80, 32768),
                _ => ("NVIDIA Volta GPU", 80, 16384),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Volta,
                compute_capability: ComputeCapability::new(7, 0),
                sm_count: sm,
                vram_size_mb: vram,
                max_threads_per_sm: 2048,
                max_warps_per_sm: 64,
                warp_size: 32,
                max_shared_memory_per_sm: 96 * 1024,
                max_registers_per_sm: 65536,
                name,
            })
        }

        // ----- Turing (CC 7.5) — Tesla T4, RTX 20xx ---------------------
        0x1B80..=0x1BFF => {
            let (name, sm, vram) = match device_id {
                0x1B80 => ("NVIDIA Tesla T4", 40, 16384),
                0x1B81 => ("NVIDIA GeForce GTX 1070", 15, 8192),
                0x1BA0 => ("NVIDIA GeForce GTX 1080", 20, 8192),
                _ => ("NVIDIA Turing GPU", 40, 16384),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Turing,
                compute_capability: ComputeCapability::new(7, 5),
                sm_count: sm,
                vram_size_mb: vram,
                max_threads_per_sm: 1024,
                max_warps_per_sm: 32,
                warp_size: 32,
                max_shared_memory_per_sm: 64 * 1024,
                max_registers_per_sm: 65536,
                name,
            })
        }

        // ----- Ampere (CC 8.0) — A100 -----------------------------------
        0x20B0..=0x20BF => {
            let (name, sm, vram) = match device_id {
                0x20B0 => ("NVIDIA A100-SXM4-40GB", 108, 40960),
                0x20B2 => ("NVIDIA A100-SXM4-80GB", 108, 81920),
                0x20B5 => ("NVIDIA A30", 56, 24576),
                _ => ("NVIDIA Ampere GPU", 108, 40960),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Ampere,
                compute_capability: ComputeCapability::new(8, 0),
                sm_count: sm,
                vram_size_mb: vram,
                max_threads_per_sm: 2048,
                max_warps_per_sm: 64,
                warp_size: 32,
                max_shared_memory_per_sm: 164 * 1024,
                max_registers_per_sm: 65536,
                name,
            })
        }

        // ----- Ampere (CC 8.7) — Jetson Orin -----------------------------
        0x2200..=0x22FF => {
            let (name, sm, vram) = match device_id {
                0x2204 => ("NVIDIA Jetson AGX Orin", 16, 32768),
                0x2206 => ("NVIDIA Jetson Orin NX 16GB", 8, 16384),
                0x2208 => ("NVIDIA Jetson Orin Nano 8GB", 4, 8192),
                _ => ("NVIDIA Orin GPU", 8, 8192),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Ampere,
                compute_capability: ComputeCapability::new(8, 7),
                sm_count: sm,
                vram_size_mb: vram,
                max_threads_per_sm: 1536,
                max_warps_per_sm: 48,
                warp_size: 32,
                max_shared_memory_per_sm: 164 * 1024,
                max_registers_per_sm: 65536,
                name,
            })
        }

        // ----- Hopper (CC 9.0) — H100, H200 -----------------------------
        0x2300..=0x23FF => {
            let (name, sm, vram) = match device_id {
                0x2330 => ("NVIDIA H100-SXM5-80GB", 132, 81920),
                0x2331 => ("NVIDIA H100-PCIE-80GB", 114, 81920),
                0x2336 => ("NVIDIA H200-SXM-141GB", 132, 144384),
                _ => ("NVIDIA Hopper GPU", 132, 81920),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Hopper,
                compute_capability: ComputeCapability::new(9, 0),
                sm_count: sm,
                vram_size_mb: vram,
                max_threads_per_sm: 2048,
                max_warps_per_sm: 64,
                warp_size: 32,
                max_shared_memory_per_sm: 228 * 1024,
                max_registers_per_sm: 65536,
                name,
            })
        }

        // ----- Blackwell (CC 10.0) — B200, DGX Spark --------------------
        0x2900..=0x29FF => {
            let (name, sm, vram) = match device_id {
                0x2900 => ("NVIDIA B200", 160, 196608),
                0x2901 => ("NVIDIA B100", 132, 196608),
                0x2910 => ("NVIDIA DGX Spark", 84, 131072),
                _ => ("NVIDIA Blackwell GPU", 160, 196608),
            };
            Ok(GpuInfo {
                device_id,
                architecture: GpuArchitecture::Blackwell,
                compute_capability: ComputeCapability::new(10, 0),
                sm_count: sm,
                vram_size_mb: vram,
                max_threads_per_sm: 2048,
                max_warps_per_sm: 64,
                warp_size: 32,
                max_shared_memory_per_sm: 228 * 1024,
                max_registers_per_sm: 65536,
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
    fn identify_tesla_t4_turing() {
        let info = identify_gpu(0x1B80).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Turing);
        assert_eq!(info.compute_capability, ComputeCapability::new(7, 5));
        assert_eq!(info.sm_count, 40);
        assert_eq!(info.vram_size_mb, 16384);
        assert_eq!(info.name, "NVIDIA Tesla T4");
    }

    #[test]
    fn identify_v100_volta() {
        let info = identify_gpu(0x1DB1).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Volta);
        assert_eq!(info.compute_capability, ComputeCapability::new(7, 0));
        assert_eq!(info.sm_count, 80);
        assert_eq!(info.vram_size_mb, 16384);
        assert_eq!(info.name, "NVIDIA Tesla V100-SXM2-16GB");
    }

    #[test]
    fn identify_a100_ampere() {
        let info = identify_gpu(0x20B0).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Ampere);
        assert_eq!(info.compute_capability, ComputeCapability::new(8, 0));
        assert_eq!(info.sm_count, 108);
        assert_eq!(info.vram_size_mb, 40960);
        assert_eq!(info.name, "NVIDIA A100-SXM4-40GB");
    }

    #[test]
    fn identify_h100_hopper() {
        let info = identify_gpu(0x2330).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Hopper);
        assert_eq!(info.compute_capability, ComputeCapability::new(9, 0));
        assert_eq!(info.sm_count, 132);
        assert_eq!(info.vram_size_mb, 81920);
        assert_eq!(info.name, "NVIDIA H100-SXM5-80GB");
    }

    #[test]
    fn identify_b200_blackwell() {
        let info = identify_gpu(0x2900).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Blackwell);
        assert_eq!(info.compute_capability, ComputeCapability::new(10, 0));
        assert_eq!(info.sm_count, 160);
        assert_eq!(info.vram_size_mb, 196608);
        assert_eq!(info.name, "NVIDIA B200");
    }

    #[test]
    fn identify_jetson_nano_maxwell() {
        let info = identify_gpu(0x1340).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Maxwell);
        assert_eq!(info.compute_capability, ComputeCapability::new(5, 3));
        assert_eq!(info.sm_count, 1);
        assert_eq!(info.vram_size_mb, 4096);
        assert_eq!(info.name, "NVIDIA Jetson Nano (Maxwell)");
    }

    #[test]
    fn identify_jetson_orin_ampere() {
        let info = identify_gpu(0x2204).unwrap();
        assert_eq!(info.architecture, GpuArchitecture::Ampere);
        assert_eq!(info.compute_capability, ComputeCapability::new(8, 7));
        assert_eq!(info.sm_count, 16);
        assert_eq!(info.vram_size_mb, 32768);
    }

    #[test]
    fn unknown_device_returns_error() {
        let result = identify_gpu(0xFFFF);
        assert_eq!(result, Err(GpuError::UnsupportedDevice));
    }

    #[test]
    fn unknown_device_in_gap_returns_error() {
        // 0x0001 is not in any known range.
        let result = identify_gpu(0x0001);
        assert_eq!(result, Err(GpuError::UnsupportedDevice));
    }

    // -- ComputeCapability ------------------------------------------------

    #[test]
    fn compute_capability_sm_version_75() {
        let cc = ComputeCapability::new(7, 5);
        assert_eq!(cc.as_sm_version(), 75);
    }

    #[test]
    fn compute_capability_sm_version_53() {
        let cc = ComputeCapability::new(5, 3);
        assert_eq!(cc.as_sm_version(), 53);
    }

    #[test]
    fn compute_capability_sm_version_100() {
        let cc = ComputeCapability::new(10, 0);
        assert_eq!(cc.as_sm_version(), 100);
    }

    // -- GpuInfo derived values -------------------------------------------

    #[test]
    fn total_cuda_cores_turing() {
        let info = identify_gpu(0x1B80).unwrap();
        // Turing: 64 cores/SM * 40 SM = 2560
        assert_eq!(info.total_cuda_cores(), 40 * 64);
    }

    #[test]
    fn total_cuda_cores_ampere() {
        let info = identify_gpu(0x20B0).unwrap();
        // Ampere: 128 cores/SM * 108 SM = 13824
        assert_eq!(info.total_cuda_cores(), 108 * 128);
    }

    #[test]
    fn total_cuda_cores_maxwell() {
        let info = identify_gpu(0x1340).unwrap();
        // Maxwell: 128 cores/SM * 1 SM = 128
        assert_eq!(info.total_cuda_cores(), 1 * 128);
    }

    #[test]
    fn total_cuda_cores_volta() {
        let info = identify_gpu(0x1DB1).unwrap();
        // Volta: 64 cores/SM * 80 SM = 5120
        assert_eq!(info.total_cuda_cores(), 80 * 64);
    }

    #[test]
    fn total_cuda_cores_hopper() {
        let info = identify_gpu(0x2330).unwrap();
        // Hopper: 128 cores/SM * 132 SM = 16896
        assert_eq!(info.total_cuda_cores(), 132 * 128);
    }

    #[test]
    fn total_cuda_cores_blackwell() {
        let info = identify_gpu(0x2900).unwrap();
        // Blackwell: 128 cores/SM * 160 SM = 20480
        assert_eq!(info.total_cuda_cores(), 160 * 128);
    }

    #[test]
    fn max_concurrent_threads_a100() {
        let info = identify_gpu(0x20B0).unwrap();
        assert_eq!(info.max_concurrent_threads(), 108 * 2048);
    }

    #[test]
    fn warp_size_always_32() {
        for &dev_id in &[0x1340, 0x1B80, 0x1DB1, 0x20B0, 0x2330, 0x2900] {
            let info = identify_gpu(dev_id).unwrap();
            assert_eq!(
                info.warp_size, 32,
                "warp size must be 32 for device 0x{:04X}",
                dev_id
            );
        }
    }

    // -- Architecture enum ------------------------------------------------

    #[test]
    fn architecture_enum_clone_debug() {
        let arch = GpuArchitecture::Hopper;
        let cloned = arch.clone();
        assert_eq!(arch, cloned);
        // Debug format should contain "Hopper"
        let dbg = alloc::format!("{:?}", arch);
        assert!(dbg.contains("Hopper"));
    }

    #[test]
    fn architecture_unknown_variant() {
        let arch = GpuArchitecture::Unknown;
        assert_eq!(arch, GpuArchitecture::Unknown);
    }
}
