// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HIP kernel definitions and registry.
//!
//! Maintains a registry of HIP/GCN kernels that implement ONNX operators on
//! AMD GPUs. Each [`HipKernel`] records its required GFX version, LDS
//! footprint, and workgroup geometry. The [`HipRegistry`] provides lookup by
//! operator type, precision, and hardware compatibility.

use alloc::string::String;
use alloc::vec::Vec;

use crate::gpu_id::GfxVersion;
use crate::GpuError;

// ---------------------------------------------------------------------------
// DataPrecision
// ---------------------------------------------------------------------------

/// Numeric precision for a HIP kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum DataPrecision {
    /// 32-bit IEEE 754 single-precision float.
    F32,
    /// 16-bit IEEE 754 half-precision float.
    F16,
    /// 16-bit brain floating-point.
    Bf16,
    /// 8-bit signed integer (quantised inference).
    Int8,
    /// 32-bit signed integer.
    Int32,
}

// ---------------------------------------------------------------------------
// HipKernelType
// ---------------------------------------------------------------------------

/// Category of GPU kernel, mapping 1-to-1 with ONNX operator families.
#[derive(Clone, Debug, PartialEq)]
pub enum HipKernelType {
    /// General matrix-matrix multiply (MatMul / Gemm).
    Gemm,
    /// 2-D convolution (Conv).
    Conv2d,
    /// Softmax normalisation.
    Softmax,
    /// Layer normalisation.
    LayerNorm,
    /// Point-wise element-wise ops (Relu, Sigmoid, Tanh, Add, Mul, Sub).
    Elementwise,
    /// Reduction along one or more axes (ReduceSum, ReduceMean, ReduceMax).
    Reduce,
    /// Transpose / permute dimensions.
    Transpose,
    /// Tensor concatenation along an axis.
    Concat,
    /// Pooling (MaxPool, AveragePool, GlobalAveragePool).
    Pool,
}

// ---------------------------------------------------------------------------
// HipKernel
// ---------------------------------------------------------------------------

/// A single HIP kernel that can be launched on an AMD GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct HipKernel {
    /// Human-readable kernel identifier (e.g. `"gemm_f32"`).
    pub name: String,
    /// Operator family this kernel implements.
    pub kernel_type: HipKernelType,
    /// Numeric precision.
    pub precision: DataPrecision,
    /// Minimum GFX ISA version required (e.g. 908 for MI100).
    pub min_gfx: u32,
    /// LDS (local data share) requirement in bytes.
    pub lds_size: u32,
    /// Maximum work-items per workgroup this kernel supports.
    pub max_workitems_per_workgroup: u32,
    /// VGPRs consumed per work-item.
    pub vgprs_per_workitem: u32,
}

impl HipKernel {
    /// Returns `true` when the given GFX version meets or exceeds the
    /// kernel's minimum GFX requirement.
    pub fn is_compatible(&self, gfx: &GfxVersion) -> bool {
        gfx.as_gfx_id() >= self.min_gfx
    }

    /// Theoretical occupancy ratio for the given `workitems_per_workgroup`.
    pub fn occupancy(&self, workitems_per_workgroup: u32) -> f64 {
        workitems_per_workgroup as f64 / self.max_workitems_per_workgroup as f64
    }
}

// ---------------------------------------------------------------------------
// HipRegistry
// ---------------------------------------------------------------------------

/// Maximum number of kernels the registry can hold.
pub const MAX_KERNELS: usize = 128;

/// Registry of HIP kernels available for dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct HipRegistry {
    /// Registered kernels.
    kernels: Vec<HipKernel>,
    /// Monotonically increasing id generator.
    next_id: u64,
}

impl Default for HipRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HipRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new HIP kernel and return its unique id.
    pub fn register(&mut self, kernel: HipKernel) -> Result<u64, GpuError> {
        if self.kernels.len() >= MAX_KERNELS {
            return Err(GpuError::QueueFull);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.kernels.push(kernel);
        Ok(id)
    }

    /// Find a registered kernel matching the requested type, precision, and
    /// hardware capability.
    pub fn find_kernel(
        &self,
        kernel_type: HipKernelType,
        precision: DataPrecision,
        gfx: &GfxVersion,
    ) -> Option<&HipKernel> {
        self.kernels.iter().find(|k| {
            k.kernel_type == kernel_type && k.precision == precision && k.is_compatible(gfx)
        })
    }

    /// Populate the registry with standard kernels covering the core ONNX
    /// operator set.
    ///
    /// Registers 11 default kernels:
    /// - `gemm_f32` (gfx900), `gemm_f16` (gfx900)
    /// - `conv2d_f32` (gfx900), `conv2d_f16` (gfx908)
    /// - `softmax_f32` (gfx900), `layernorm_f32` (gfx900)
    /// - `elementwise_f32` (gfx900), `reduce_f32` (gfx900)
    /// - `transpose_f32` (gfx900), `concat_f32` (gfx900)
    /// - `pool_f32` (gfx900)
    pub fn register_defaults(&mut self) -> Result<(), GpuError> {
        let defaults = alloc::vec![
            HipKernel {
                name: String::from("gemm_f32"),
                kernel_type: HipKernelType::Gemm,
                precision: DataPrecision::F32,
                min_gfx: 900,
                lds_size: 49152,
                max_workitems_per_workgroup: 256,
                vgprs_per_workitem: 32,
            },
            HipKernel {
                name: String::from("gemm_f16"),
                kernel_type: HipKernelType::Gemm,
                precision: DataPrecision::F16,
                min_gfx: 900,
                lds_size: 32768,
                max_workitems_per_workgroup: 256,
                vgprs_per_workitem: 24,
            },
            HipKernel {
                name: String::from("conv2d_f32"),
                kernel_type: HipKernelType::Conv2d,
                precision: DataPrecision::F32,
                min_gfx: 900,
                lds_size: 49152,
                max_workitems_per_workgroup: 256,
                vgprs_per_workitem: 32,
            },
            HipKernel {
                name: String::from("conv2d_f16"),
                kernel_type: HipKernelType::Conv2d,
                precision: DataPrecision::F16,
                min_gfx: 908,
                lds_size: 32768,
                max_workitems_per_workgroup: 256,
                vgprs_per_workitem: 24,
            },
            HipKernel {
                name: String::from("softmax_f32"),
                kernel_type: HipKernelType::Softmax,
                precision: DataPrecision::F32,
                min_gfx: 900,
                lds_size: 16384,
                max_workitems_per_workgroup: 256,
                vgprs_per_workitem: 16,
            },
            HipKernel {
                name: String::from("layernorm_f32"),
                kernel_type: HipKernelType::LayerNorm,
                precision: DataPrecision::F32,
                min_gfx: 900,
                lds_size: 16384,
                max_workitems_per_workgroup: 256,
                vgprs_per_workitem: 16,
            },
            HipKernel {
                name: String::from("elementwise_f32"),
                kernel_type: HipKernelType::Elementwise,
                precision: DataPrecision::F32,
                min_gfx: 900,
                lds_size: 0,
                max_workitems_per_workgroup: 1024,
                vgprs_per_workitem: 8,
            },
            HipKernel {
                name: String::from("reduce_f32"),
                kernel_type: HipKernelType::Reduce,
                precision: DataPrecision::F32,
                min_gfx: 900,
                lds_size: 32768,
                max_workitems_per_workgroup: 256,
                vgprs_per_workitem: 16,
            },
            HipKernel {
                name: String::from("transpose_f32"),
                kernel_type: HipKernelType::Transpose,
                precision: DataPrecision::F32,
                min_gfx: 900,
                lds_size: 32768,
                max_workitems_per_workgroup: 256,
                vgprs_per_workitem: 16,
            },
            HipKernel {
                name: String::from("concat_f32"),
                kernel_type: HipKernelType::Concat,
                precision: DataPrecision::F32,
                min_gfx: 900,
                lds_size: 0,
                max_workitems_per_workgroup: 1024,
                vgprs_per_workitem: 8,
            },
            HipKernel {
                name: String::from("pool_f32"),
                kernel_type: HipKernelType::Pool,
                precision: DataPrecision::F32,
                min_gfx: 900,
                lds_size: 16384,
                max_workitems_per_workgroup: 256,
                vgprs_per_workitem: 16,
            },
        ];

        for kernel in defaults {
            self.register(kernel)?;
        }

        Ok(())
    }

    /// Number of kernels currently registered.
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// Return references to all kernels compatible with the given GFX version.
    pub fn compatible_kernels(&self, gfx: &GfxVersion) -> Vec<&HipKernel> {
        self.kernels
            .iter()
            .filter(|k| k.is_compatible(gfx))
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_id::identify_gpu;

    fn test_gfx_cdna3() -> GfxVersion {
        GfxVersion::new(9, 4, 2) // gfx942
    }

    fn test_gfx_cdna1() -> GfxVersion {
        GfxVersion::new(9, 0, 8) // gfx908
    }

    fn test_gfx_vega() -> GfxVersion {
        GfxVersion::new(9, 0, 0) // gfx900
    }

    fn test_gfx_too_old() -> GfxVersion {
        GfxVersion::new(8, 0, 0) // gfx800, below any supported
    }

    fn make_kernel(name: &str, kt: HipKernelType, prec: DataPrecision, gfx: u32) -> HipKernel {
        HipKernel {
            name: String::from(name),
            kernel_type: kt,
            precision: prec,
            min_gfx: gfx,
            lds_size: 4096,
            max_workitems_per_workgroup: 256,
            vgprs_per_workitem: 16,
        }
    }

    #[test]
    fn test_register_kernel() {
        let mut reg = HipRegistry::new();
        let kernel = make_kernel("test_gemm", HipKernelType::Gemm, DataPrecision::F32, 900);
        let id = reg.register(kernel).unwrap();
        assert_eq!(id, 1);
        assert_eq!(reg.kernel_count(), 1);
    }

    #[test]
    fn test_find_kernel_by_type_and_precision() {
        let mut reg = HipRegistry::new();
        reg.register(make_kernel(
            "gemm_f32",
            HipKernelType::Gemm,
            DataPrecision::F32,
            900,
        ))
        .unwrap();
        reg.register(make_kernel(
            "gemm_f16",
            HipKernelType::Gemm,
            DataPrecision::F16,
            900,
        ))
        .unwrap();

        let gfx = test_gfx_cdna3();
        let found = reg.find_kernel(HipKernelType::Gemm, DataPrecision::F16, &gfx);
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "gemm_f16");
    }

    #[test]
    fn test_compatibility_check() {
        let kernel = make_kernel("conv2d_f16", HipKernelType::Conv2d, DataPrecision::F16, 908);
        assert!(kernel.is_compatible(&test_gfx_cdna1())); // gfx908 >= 908
        assert!(!kernel.is_compatible(&test_gfx_vega())); // gfx900 < 908
    }

    #[test]
    fn test_register_defaults_creates_11_kernels() {
        let mut reg = HipRegistry::new();
        reg.register_defaults().unwrap();
        assert_eq!(reg.kernel_count(), 11);
    }

    #[test]
    fn test_incompatible_gfx_returns_none() {
        let mut reg = HipRegistry::new();
        reg.register_defaults().unwrap();
        let gfx = test_gfx_too_old();
        let found = reg.find_kernel(HipKernelType::Gemm, DataPrecision::F32, &gfx);
        assert!(found.is_none());
    }

    #[test]
    fn test_compatible_kernels_filtering() {
        let mut reg = HipRegistry::new();
        reg.register_defaults().unwrap();

        // gfx900: everything except conv2d_f16 (gfx908)
        let compat = reg.compatible_kernels(&test_gfx_vega());
        assert_eq!(compat.len(), 10);

        // gfx942: all 11 kernels
        let compat = reg.compatible_kernels(&test_gfx_cdna3());
        assert_eq!(compat.len(), 11);
    }

    #[test]
    fn test_kernel_count() {
        let mut reg = HipRegistry::new();
        assert_eq!(reg.kernel_count(), 0);
        reg.register(make_kernel(
            "a",
            HipKernelType::Pool,
            DataPrecision::F32,
            900,
        ))
        .unwrap();
        assert_eq!(reg.kernel_count(), 1);
        reg.register(make_kernel(
            "b",
            HipKernelType::Pool,
            DataPrecision::F16,
            900,
        ))
        .unwrap();
        assert_eq!(reg.kernel_count(), 2);
    }

    #[test]
    fn test_occupancy_calculation() {
        let kernel = HipKernel {
            name: String::from("occ_test"),
            kernel_type: HipKernelType::Elementwise,
            precision: DataPrecision::F32,
            min_gfx: 900,
            lds_size: 0,
            max_workitems_per_workgroup: 1024,
            vgprs_per_workitem: 8,
        };
        let occ = kernel.occupancy(512);
        assert!((occ - 0.5).abs() < 1e-9);
        let occ_full = kernel.occupancy(1024);
        assert!((occ_full - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_registry_full_error() {
        let mut reg = HipRegistry::new();
        for i in 0..MAX_KERNELS {
            let name = alloc::format!("k{}", i);
            reg.register(make_kernel(
                &name,
                HipKernelType::Gemm,
                DataPrecision::F32,
                900,
            ))
            .unwrap();
        }
        assert_eq!(reg.kernel_count(), MAX_KERNELS);
        let result = reg.register(make_kernel(
            "overflow",
            HipKernelType::Gemm,
            DataPrecision::F32,
            900,
        ));
        assert_eq!(result, Err(GpuError::QueueFull));
    }

    #[test]
    fn test_all_kernel_types_in_defaults() {
        let mut reg = HipRegistry::new();
        reg.register_defaults().unwrap();
        let gfx = test_gfx_cdna3();

        let types = [
            HipKernelType::Gemm,
            HipKernelType::Conv2d,
            HipKernelType::Softmax,
            HipKernelType::LayerNorm,
            HipKernelType::Elementwise,
            HipKernelType::Reduce,
            HipKernelType::Transpose,
            HipKernelType::Concat,
            HipKernelType::Pool,
        ];
        for kt in &types {
            let found = reg.find_kernel(kt.clone(), DataPrecision::F32, &gfx);
            assert!(found.is_some(), "Expected default F32 kernel for {:?}", kt);
        }
    }

    #[test]
    fn test_f16_kernels_require_higher_gfx() {
        let mut reg = HipRegistry::new();
        reg.register_defaults().unwrap();

        // conv2d_f16 requires gfx908. gfx900 should fail.
        let found = reg.find_kernel(HipKernelType::Conv2d, DataPrecision::F16, &test_gfx_vega());
        assert!(
            found.is_none(),
            "conv2d_f16 should not be compatible with gfx900"
        );

        // gfx942 should succeed.
        let found = reg.find_kernel(HipKernelType::Conv2d, DataPrecision::F16, &test_gfx_cdna3());
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "conv2d_f16");
    }

    #[test]
    fn test_default_kernel_lds_values() {
        let mut reg = HipRegistry::new();
        reg.register_defaults().unwrap();
        let gfx = test_gfx_cdna3();

        let gemm = reg
            .find_kernel(HipKernelType::Gemm, DataPrecision::F32, &gfx)
            .unwrap();
        assert_eq!(gemm.lds_size, 49152);

        let eltwise = reg
            .find_kernel(HipKernelType::Elementwise, DataPrecision::F32, &gfx)
            .unwrap();
        assert_eq!(eltwise.lds_size, 0);

        let softmax = reg
            .find_kernel(HipKernelType::Softmax, DataPrecision::F32, &gfx)
            .unwrap();
        assert_eq!(softmax.lds_size, 16384);
    }

    #[test]
    fn test_hip_kernel_type_variants() {
        let variants = [
            HipKernelType::Gemm,
            HipKernelType::Conv2d,
            HipKernelType::Softmax,
            HipKernelType::LayerNorm,
            HipKernelType::Elementwise,
            HipKernelType::Reduce,
            HipKernelType::Transpose,
            HipKernelType::Concat,
            HipKernelType::Pool,
        ];
        assert_eq!(variants.len(), 9);
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_data_precision_variants() {
        let variants = [
            DataPrecision::F32,
            DataPrecision::F16,
            DataPrecision::Bf16,
            DataPrecision::Int8,
            DataPrecision::Int32,
        ];
        assert_eq!(variants.len(), 5);
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_kernel_clone_debug() {
        let kernel = make_kernel(
            "clone_test",
            HipKernelType::Softmax,
            DataPrecision::F32,
            900,
        );
        let cloned = kernel.clone();
        assert_eq!(kernel, cloned);
        let dbg = alloc::format!("{:?}", kernel);
        assert!(dbg.contains("clone_test"));
        assert!(dbg.contains("Softmax"));
    }

    #[test]
    fn test_find_nonexistent_precision_returns_none() {
        let mut reg = HipRegistry::new();
        reg.register_defaults().unwrap();
        let gfx = test_gfx_cdna3();
        let found = reg.find_kernel(HipKernelType::Gemm, DataPrecision::Bf16, &gfx);
        assert!(found.is_none());
    }

    #[test]
    fn test_identify_gpu_gfx_works_with_registry() {
        let gpu = identify_gpu(0x7500).unwrap();
        let mut reg = HipRegistry::new();
        reg.register_defaults().unwrap();
        let compat = reg.compatible_kernels(&gpu.gfx_version);
        assert_eq!(compat.len(), 11);
    }

    #[test]
    fn test_registry_initial_state() {
        let reg = HipRegistry::new();
        assert_eq!(reg.kernel_count(), 0);
        assert_eq!(reg.next_id, 1);
    }

    #[test]
    fn test_register_ids_increase() {
        let mut reg = HipRegistry::new();
        let id1 = reg
            .register(make_kernel(
                "k1",
                HipKernelType::Pool,
                DataPrecision::F32,
                900,
            ))
            .unwrap();
        let id2 = reg
            .register(make_kernel(
                "k2",
                HipKernelType::Pool,
                DataPrecision::F16,
                900,
            ))
            .unwrap();
        let id3 = reg
            .register(make_kernel(
                "k3",
                HipKernelType::Pool,
                DataPrecision::Int8,
                900,
            ))
            .unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }
}
