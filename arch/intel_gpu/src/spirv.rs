// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V kernel definitions and registry.
//!
//! Maintains a registry of SPIR-V compute kernels that implement ONNX
//! operators on Intel GPUs. Each [`SpirvKernel`] records its required GPU
//! generation, shared local memory footprint, and workgroup geometry. The
//! [`SpirvRegistry`] provides lookup by operator type, precision, and
//! hardware compatibility.
//!
//! SPIR-V is to Intel GPUs what PTX is to NVIDIA GPUs -- the standard
//! intermediate representation for compute shaders.

use alloc::string::String;
use alloc::vec::Vec;

use crate::gpu_id::GpuGeneration;
use crate::GpuError;

// ---------------------------------------------------------------------------
// DataPrecision
// ---------------------------------------------------------------------------

/// Numeric precision for a SPIR-V kernel.
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
// SpirvKernelType
// ---------------------------------------------------------------------------

/// Category of GPU kernel, mapping 1-to-1 with ONNX operator families.
#[derive(Clone, Debug, PartialEq)]
pub enum SpirvKernelType {
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
// SpirvKernel
// ---------------------------------------------------------------------------

/// A single SPIR-V kernel that can be dispatched on the GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct SpirvKernel {
    /// Human-readable kernel identifier (e.g. `"gemm_f32"`).
    pub name: String,
    /// Operator family this kernel implements.
    pub kernel_type: SpirvKernelType,
    /// Numeric precision.
    pub precision: DataPrecision,
    /// Minimum GPU generation required (e.g. 120 for Gen12).
    pub min_generation: u16,
    /// Shared local memory requirement in bytes.
    pub shared_memory: u32,
    /// Maximum work-items per workgroup this kernel supports.
    pub max_workgroup_size: u32,
    /// Whether this kernel uses XMX (matrix extension) instructions.
    pub uses_xmx: bool,
}

impl SpirvKernel {
    /// Returns `true` when the given GPU generation meets or exceeds the
    /// kernel's minimum generation requirement.
    pub fn is_compatible(&self, gen: &GpuGeneration) -> bool {
        gen.version >= self.min_generation
    }

    /// Theoretical occupancy ratio for the given `workgroup_size`.
    pub fn occupancy(&self, workgroup_size: u32) -> f64 {
        workgroup_size as f64 / self.max_workgroup_size as f64
    }
}

// ---------------------------------------------------------------------------
// SpirvRegistry
// ---------------------------------------------------------------------------

/// Maximum number of kernels the registry can hold.
pub const MAX_KERNELS: usize = 128;

/// Registry of SPIR-V kernels available for dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct SpirvRegistry {
    /// Registered kernels.
    kernels: Vec<SpirvKernel>,
    /// Monotonically increasing id generator.
    next_id: u64,
}

impl Default for SpirvRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SpirvRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new SPIR-V kernel and return its unique id.
    pub fn register(&mut self, kernel: SpirvKernel) -> Result<u64, GpuError> {
        if self.kernels.len() >= MAX_KERNELS {
            return Err(GpuError::QueueFull);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.kernels.push(kernel);
        Ok(id)
    }

    /// Find a registered kernel matching the requested type, precision, and
    /// hardware generation.
    pub fn find_kernel(
        &self,
        kernel_type: SpirvKernelType,
        precision: DataPrecision,
        gen: &GpuGeneration,
    ) -> Option<&SpirvKernel> {
        self.kernels.iter().find(|k| {
            k.kernel_type == kernel_type && k.precision == precision && k.is_compatible(gen)
        })
    }

    /// Populate the registry with standard kernels covering the core ONNX
    /// operator set.
    ///
    /// Registers 11 default kernels:
    /// - `gemm_f32` (gen 120), `gemm_f16` (gen 120)
    /// - `conv2d_f32` (gen 120), `conv2d_f16` (gen 127, XMX)
    /// - `softmax_f32` (gen 120), `layernorm_f32` (gen 120)
    /// - `elementwise_f32` (gen 120), `reduce_f32` (gen 120)
    /// - `transpose_f32` (gen 120), `concat_f32` (gen 120)
    /// - `pool_f32` (gen 120)
    pub fn register_defaults(&mut self) -> Result<(), GpuError> {
        let defaults = alloc::vec![
            SpirvKernel {
                name: String::from("gemm_f32"),
                kernel_type: SpirvKernelType::Gemm,
                precision: DataPrecision::F32,
                min_generation: 120,
                shared_memory: 49152,
                max_workgroup_size: 256,
                uses_xmx: false,
            },
            SpirvKernel {
                name: String::from("gemm_f16"),
                kernel_type: SpirvKernelType::Gemm,
                precision: DataPrecision::F16,
                min_generation: 120,
                shared_memory: 32768,
                max_workgroup_size: 256,
                uses_xmx: false,
            },
            SpirvKernel {
                name: String::from("conv2d_f32"),
                kernel_type: SpirvKernelType::Conv2d,
                precision: DataPrecision::F32,
                min_generation: 120,
                shared_memory: 49152,
                max_workgroup_size: 256,
                uses_xmx: false,
            },
            SpirvKernel {
                name: String::from("conv2d_f16"),
                kernel_type: SpirvKernelType::Conv2d,
                precision: DataPrecision::F16,
                min_generation: 127,
                shared_memory: 32768,
                max_workgroup_size: 256,
                uses_xmx: true,
            },
            SpirvKernel {
                name: String::from("softmax_f32"),
                kernel_type: SpirvKernelType::Softmax,
                precision: DataPrecision::F32,
                min_generation: 120,
                shared_memory: 16384,
                max_workgroup_size: 256,
                uses_xmx: false,
            },
            SpirvKernel {
                name: String::from("layernorm_f32"),
                kernel_type: SpirvKernelType::LayerNorm,
                precision: DataPrecision::F32,
                min_generation: 120,
                shared_memory: 16384,
                max_workgroup_size: 256,
                uses_xmx: false,
            },
            SpirvKernel {
                name: String::from("elementwise_f32"),
                kernel_type: SpirvKernelType::Elementwise,
                precision: DataPrecision::F32,
                min_generation: 120,
                shared_memory: 0,
                max_workgroup_size: 1024,
                uses_xmx: false,
            },
            SpirvKernel {
                name: String::from("reduce_f32"),
                kernel_type: SpirvKernelType::Reduce,
                precision: DataPrecision::F32,
                min_generation: 120,
                shared_memory: 32768,
                max_workgroup_size: 256,
                uses_xmx: false,
            },
            SpirvKernel {
                name: String::from("transpose_f32"),
                kernel_type: SpirvKernelType::Transpose,
                precision: DataPrecision::F32,
                min_generation: 120,
                shared_memory: 32768,
                max_workgroup_size: 256,
                uses_xmx: false,
            },
            SpirvKernel {
                name: String::from("concat_f32"),
                kernel_type: SpirvKernelType::Concat,
                precision: DataPrecision::F32,
                min_generation: 120,
                shared_memory: 0,
                max_workgroup_size: 1024,
                uses_xmx: false,
            },
            SpirvKernel {
                name: String::from("pool_f32"),
                kernel_type: SpirvKernelType::Pool,
                precision: DataPrecision::F32,
                min_generation: 120,
                shared_memory: 16384,
                max_workgroup_size: 256,
                uses_xmx: false,
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

    /// Return references to all kernels compatible with the given generation.
    pub fn compatible_kernels(&self, gen: &GpuGeneration) -> Vec<&SpirvKernel> {
        self.kernels
            .iter()
            .filter(|k| k.is_compatible(gen))
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

    fn test_gen_xe_lp() -> GpuGeneration {
        GpuGeneration::new(120)
    }

    fn test_gen_xe_hpg() -> GpuGeneration {
        GpuGeneration::new(127)
    }

    fn test_gen_too_old() -> GpuGeneration {
        GpuGeneration::new(90)
    }

    fn make_kernel(name: &str, kt: SpirvKernelType, prec: DataPrecision, gen: u16) -> SpirvKernel {
        SpirvKernel {
            name: String::from(name),
            kernel_type: kt,
            precision: prec,
            min_generation: gen,
            shared_memory: 4096,
            max_workgroup_size: 256,
            uses_xmx: false,
        }
    }

    #[test]
    fn test_register_kernel() {
        let mut reg = SpirvRegistry::new();
        let kernel = make_kernel("test_gemm", SpirvKernelType::Gemm, DataPrecision::F32, 120);
        let id = reg.register(kernel).unwrap();
        assert_eq!(id, 1);
        assert_eq!(reg.kernel_count(), 1);
    }

    #[test]
    fn test_find_kernel_by_type_and_precision() {
        let mut reg = SpirvRegistry::new();
        reg.register(make_kernel(
            "gemm_f32",
            SpirvKernelType::Gemm,
            DataPrecision::F32,
            120,
        ))
        .unwrap();
        reg.register(make_kernel(
            "gemm_f16",
            SpirvKernelType::Gemm,
            DataPrecision::F16,
            120,
        ))
        .unwrap();

        let gen = test_gen_xe_hpg();
        let found = reg.find_kernel(SpirvKernelType::Gemm, DataPrecision::F16, &gen);
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "gemm_f16");
    }

    #[test]
    fn test_compatibility_check() {
        let kernel = make_kernel(
            "conv2d_f16",
            SpirvKernelType::Conv2d,
            DataPrecision::F16,
            127,
        );
        assert!(kernel.is_compatible(&test_gen_xe_hpg()));
        assert!(!kernel.is_compatible(&test_gen_xe_lp()));
    }

    #[test]
    fn test_register_defaults_creates_11_kernels() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        assert_eq!(reg.kernel_count(), 11);
    }

    #[test]
    fn test_incompatible_gen_returns_none() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let gen = test_gen_too_old();
        let found = reg.find_kernel(SpirvKernelType::Gemm, DataPrecision::F32, &gen);
        assert!(found.is_none());
    }

    #[test]
    fn test_compatible_kernels_filtering() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();

        // Xe-LP (gen 120): everything except conv2d_f16 (gen 127).
        let compat = reg.compatible_kernels(&test_gen_xe_lp());
        assert_eq!(compat.len(), 10);

        // Xe-HPG (gen 127): all 11 kernels.
        let compat = reg.compatible_kernels(&test_gen_xe_hpg());
        assert_eq!(compat.len(), 11);
    }

    #[test]
    fn test_kernel_count() {
        let mut reg = SpirvRegistry::new();
        assert_eq!(reg.kernel_count(), 0);
        reg.register(make_kernel(
            "a",
            SpirvKernelType::Pool,
            DataPrecision::F32,
            120,
        ))
        .unwrap();
        assert_eq!(reg.kernel_count(), 1);
        reg.register(make_kernel(
            "b",
            SpirvKernelType::Pool,
            DataPrecision::F16,
            120,
        ))
        .unwrap();
        assert_eq!(reg.kernel_count(), 2);
    }

    #[test]
    fn test_occupancy_calculation() {
        let kernel = SpirvKernel {
            name: String::from("occ_test"),
            kernel_type: SpirvKernelType::Elementwise,
            precision: DataPrecision::F32,
            min_generation: 120,
            shared_memory: 0,
            max_workgroup_size: 1024,
            uses_xmx: false,
        };
        let occ = kernel.occupancy(512);
        assert!((occ - 0.5).abs() < 1e-9);
        let occ_full = kernel.occupancy(1024);
        assert!((occ_full - 1.0).abs() < 1e-9);
        let occ_quarter = kernel.occupancy(256);
        assert!((occ_quarter - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_registry_full_error() {
        let mut reg = SpirvRegistry::new();
        for i in 0..MAX_KERNELS {
            let name = alloc::format!("k{}", i);
            reg.register(make_kernel(
                &name,
                SpirvKernelType::Gemm,
                DataPrecision::F32,
                120,
            ))
            .unwrap();
        }
        assert_eq!(reg.kernel_count(), MAX_KERNELS);
        let result = reg.register(make_kernel(
            "overflow",
            SpirvKernelType::Gemm,
            DataPrecision::F32,
            120,
        ));
        assert_eq!(result, Err(GpuError::QueueFull));
    }

    #[test]
    fn test_all_kernel_types_in_defaults() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let gen = test_gen_xe_hpg();

        let types = [
            SpirvKernelType::Gemm,
            SpirvKernelType::Conv2d,
            SpirvKernelType::Softmax,
            SpirvKernelType::LayerNorm,
            SpirvKernelType::Elementwise,
            SpirvKernelType::Reduce,
            SpirvKernelType::Transpose,
            SpirvKernelType::Concat,
            SpirvKernelType::Pool,
        ];
        for kt in &types {
            let found = reg.find_kernel(kt.clone(), DataPrecision::F32, &gen);
            assert!(found.is_some(), "Expected default F32 kernel for {:?}", kt);
        }
    }

    #[test]
    fn test_f16_kernels_require_higher_gen() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();

        let found = reg.find_kernel(
            SpirvKernelType::Conv2d,
            DataPrecision::F16,
            &test_gen_xe_lp(),
        );
        assert!(
            found.is_none(),
            "conv2d_f16 should not be compatible with Xe-LP"
        );

        let found = reg.find_kernel(
            SpirvKernelType::Conv2d,
            DataPrecision::F16,
            &test_gen_xe_hpg(),
        );
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "conv2d_f16");
    }

    #[test]
    fn test_default_kernel_shared_memory() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let gen = test_gen_xe_hpg();

        let gemm = reg
            .find_kernel(SpirvKernelType::Gemm, DataPrecision::F32, &gen)
            .unwrap();
        assert_eq!(gemm.shared_memory, 49152);

        let eltwise = reg
            .find_kernel(SpirvKernelType::Elementwise, DataPrecision::F32, &gen)
            .unwrap();
        assert_eq!(eltwise.shared_memory, 0);

        let softmax = reg
            .find_kernel(SpirvKernelType::Softmax, DataPrecision::F32, &gen)
            .unwrap();
        assert_eq!(softmax.shared_memory, 16384);
    }

    #[test]
    fn test_spirv_kernel_type_variants() {
        let variants = [
            SpirvKernelType::Gemm,
            SpirvKernelType::Conv2d,
            SpirvKernelType::Softmax,
            SpirvKernelType::LayerNorm,
            SpirvKernelType::Elementwise,
            SpirvKernelType::Reduce,
            SpirvKernelType::Transpose,
            SpirvKernelType::Concat,
            SpirvKernelType::Pool,
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
            SpirvKernelType::Softmax,
            DataPrecision::F32,
            120,
        );
        let cloned = kernel.clone();
        assert_eq!(kernel, cloned);
        let dbg = alloc::format!("{:?}", kernel);
        assert!(dbg.contains("clone_test"));
        assert!(dbg.contains("Softmax"));
    }

    #[test]
    fn test_find_nonexistent_precision_returns_none() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let gen = test_gen_xe_hpg();
        let found = reg.find_kernel(SpirvKernelType::Gemm, DataPrecision::Bf16, &gen);
        assert!(found.is_none());
    }

    #[test]
    fn test_identify_gpu_gen_works_with_registry() {
        let gpu = identify_gpu(0x56A0).unwrap();
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();

        let compat = reg.compatible_kernels(&gpu.generation);
        assert_eq!(compat.len(), 11);
    }

    #[test]
    fn test_registry_initial_state() {
        let reg = SpirvRegistry::new();
        assert_eq!(reg.kernel_count(), 0);
        assert_eq!(reg.next_id, 1);
    }

    #[test]
    fn test_register_ids_increase() {
        let mut reg = SpirvRegistry::new();
        let id1 = reg
            .register(make_kernel(
                "k1",
                SpirvKernelType::Pool,
                DataPrecision::F32,
                120,
            ))
            .unwrap();
        let id2 = reg
            .register(make_kernel(
                "k2",
                SpirvKernelType::Pool,
                DataPrecision::F16,
                120,
            ))
            .unwrap();
        let id3 = reg
            .register(make_kernel(
                "k3",
                SpirvKernelType::Pool,
                DataPrecision::Int8,
                120,
            ))
            .unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[test]
    fn test_gemm_f16_compatible_with_xe_lp() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let found = reg.find_kernel(SpirvKernelType::Gemm, DataPrecision::F16, &test_gen_xe_lp());
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "gemm_f16");
    }

    #[test]
    fn test_xmx_kernel_flag() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let gen = test_gen_xe_hpg();
        let conv_f16 = reg
            .find_kernel(SpirvKernelType::Conv2d, DataPrecision::F16, &gen)
            .unwrap();
        assert!(conv_f16.uses_xmx, "conv2d_f16 should use XMX");

        let conv_f32 = reg
            .find_kernel(SpirvKernelType::Conv2d, DataPrecision::F32, &gen)
            .unwrap();
        assert!(!conv_f32.uses_xmx, "conv2d_f32 should not use XMX");
    }
}
