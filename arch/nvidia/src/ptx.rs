// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! PTX kernel definitions and registry.
//!
//! Maintains a registry of PTX kernels that implement ONNX operators on the
//! GPU.  Each [`PtxKernel`] records its required compute-capability, shared
//! memory footprint, and thread-block geometry.  The [`PtxRegistry`] provides
//! lookup by operator type, precision, and hardware compatibility.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::gpu_id::ComputeCapability;
use crate::GpuError;

// ---------------------------------------------------------------------------
// DataPrecision
// ---------------------------------------------------------------------------

/// Numeric precision for a PTX kernel.
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
// PtxKernelType
// ---------------------------------------------------------------------------

/// Category of GPU kernel, mapping 1-to-1 with ONNX operator families.
#[derive(Clone, Debug, PartialEq)]
pub enum PtxKernelType {
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
// PtxKernel
// ---------------------------------------------------------------------------

/// A single PTX kernel that can be launched on the GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct PtxKernel {
    /// Human-readable kernel identifier (e.g. `"gemm_f32"`).
    pub name: String,
    /// Operator family this kernel implements.
    pub kernel_type: PtxKernelType,
    /// Numeric precision.
    pub precision: DataPrecision,
    /// Minimum SM version required (e.g. `53` for Maxwell CC 5.3).
    pub min_sm: u16,
    /// Shared memory requirement in bytes.
    pub shared_memory: u32,
    /// Maximum threads per block this kernel supports.
    pub max_threads_per_block: u32,
    /// Registers consumed per thread.
    pub registers_per_thread: u32,
}

impl PtxKernel {
    /// Returns `true` when the given compute capability meets or exceeds the
    /// kernel's minimum SM requirement.
    pub fn is_compatible(&self, cc: &ComputeCapability) -> bool {
        cc.as_sm_version() >= self.min_sm
    }

    /// Theoretical occupancy ratio for the given `threads_per_block`.
    ///
    /// Returns a value in `0.0..=1.0` when `threads_per_block` is at most
    /// `max_threads_per_block`, or > 1.0 otherwise (caller should clamp).
    pub fn occupancy(&self, threads_per_block: u32) -> f64 {
        threads_per_block as f64 / self.max_threads_per_block as f64
    }
}

// ---------------------------------------------------------------------------
// PtxRegistry
// ---------------------------------------------------------------------------

/// Maximum number of kernels the registry can hold.
pub const MAX_KERNELS: usize = 128;

/// Registry of PTX kernels available for dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct PtxRegistry {
    /// Registered kernels.
    kernels: Vec<PtxKernel>,
    /// Monotonically increasing id generator.
    next_id: u64,
}

impl PtxRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new PTX kernel and return its unique id.
    ///
    /// Fails with [`GpuError::QueueFull`] when the registry has reached
    /// [`MAX_KERNELS`].
    pub fn register(&mut self, kernel: PtxKernel) -> Result<u64, GpuError> {
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
    ///
    /// Returns the first kernel that exactly matches `kernel_type` and
    /// `precision` and whose `min_sm` does not exceed the device's SM version.
    pub fn find_kernel(
        &self,
        kernel_type: PtxKernelType,
        precision: DataPrecision,
        cc: &ComputeCapability,
    ) -> Option<&PtxKernel> {
        self.kernels.iter().find(|k| {
            k.kernel_type == kernel_type && k.precision == precision && k.is_compatible(cc)
        })
    }

    /// Populate the registry with standard kernels covering the core ONNX
    /// operator set.
    ///
    /// Registers 11 default kernels:
    /// - `gemm_f32` (sm 53), `gemm_f16` (sm 53)
    /// - `conv2d_f32` (sm 53), `conv2d_f16` (sm 70)
    /// - `softmax_f32` (sm 53), `layernorm_f32` (sm 53)
    /// - `elementwise_f32` (sm 53), `reduce_f32` (sm 53)
    /// - `transpose_f32` (sm 53), `concat_f32` (sm 53)
    /// - `pool_f32` (sm 53)
    pub fn register_defaults(&mut self) -> Result<(), GpuError> {
        let defaults = alloc::vec![
            PtxKernel {
                name: String::from("gemm_f32"),
                kernel_type: PtxKernelType::Gemm,
                precision: DataPrecision::F32,
                min_sm: 53,
                shared_memory: 49152,
                max_threads_per_block: 256,
                registers_per_thread: 32,
            },
            PtxKernel {
                name: String::from("gemm_f16"),
                kernel_type: PtxKernelType::Gemm,
                precision: DataPrecision::F16,
                min_sm: 53,
                shared_memory: 32768,
                max_threads_per_block: 256,
                registers_per_thread: 24,
            },
            PtxKernel {
                name: String::from("conv2d_f32"),
                kernel_type: PtxKernelType::Conv2d,
                precision: DataPrecision::F32,
                min_sm: 53,
                shared_memory: 49152,
                max_threads_per_block: 256,
                registers_per_thread: 32,
            },
            PtxKernel {
                name: String::from("conv2d_f16"),
                kernel_type: PtxKernelType::Conv2d,
                precision: DataPrecision::F16,
                min_sm: 70,
                shared_memory: 32768,
                max_threads_per_block: 256,
                registers_per_thread: 24,
            },
            PtxKernel {
                name: String::from("softmax_f32"),
                kernel_type: PtxKernelType::Softmax,
                precision: DataPrecision::F32,
                min_sm: 53,
                shared_memory: 16384,
                max_threads_per_block: 256,
                registers_per_thread: 16,
            },
            PtxKernel {
                name: String::from("layernorm_f32"),
                kernel_type: PtxKernelType::LayerNorm,
                precision: DataPrecision::F32,
                min_sm: 53,
                shared_memory: 16384,
                max_threads_per_block: 256,
                registers_per_thread: 16,
            },
            PtxKernel {
                name: String::from("elementwise_f32"),
                kernel_type: PtxKernelType::Elementwise,
                precision: DataPrecision::F32,
                min_sm: 53,
                shared_memory: 0,
                max_threads_per_block: 1024,
                registers_per_thread: 8,
            },
            PtxKernel {
                name: String::from("reduce_f32"),
                kernel_type: PtxKernelType::Reduce,
                precision: DataPrecision::F32,
                min_sm: 53,
                shared_memory: 32768,
                max_threads_per_block: 256,
                registers_per_thread: 16,
            },
            PtxKernel {
                name: String::from("transpose_f32"),
                kernel_type: PtxKernelType::Transpose,
                precision: DataPrecision::F32,
                min_sm: 53,
                shared_memory: 32768,
                max_threads_per_block: 256,
                registers_per_thread: 16,
            },
            PtxKernel {
                name: String::from("concat_f32"),
                kernel_type: PtxKernelType::Concat,
                precision: DataPrecision::F32,
                min_sm: 53,
                shared_memory: 0,
                max_threads_per_block: 1024,
                registers_per_thread: 8,
            },
            PtxKernel {
                name: String::from("pool_f32"),
                kernel_type: PtxKernelType::Pool,
                precision: DataPrecision::F32,
                min_sm: 53,
                shared_memory: 16384,
                max_threads_per_block: 256,
                registers_per_thread: 16,
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

    /// Return references to all kernels compatible with the given compute
    /// capability.
    pub fn compatible_kernels(&self, cc: &ComputeCapability) -> Vec<&PtxKernel> {
        self.kernels
            .iter()
            .filter(|k| k.is_compatible(cc))
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

    // -- helpers ------------------------------------------------------------

    /// Tesla T4 (Turing, SM 7.5).
    fn test_cc_turing() -> ComputeCapability {
        ComputeCapability::new(7, 5)
    }

    /// Maxwell (SM 5.3).
    fn test_cc_maxwell() -> ComputeCapability {
        ComputeCapability::new(5, 3)
    }

    /// A very old / hypothetical device below any supported SM.
    fn test_cc_too_old() -> ComputeCapability {
        ComputeCapability::new(3, 0)
    }

    /// Build a simple test kernel with sensible defaults.
    fn make_kernel(name: &str, kt: PtxKernelType, prec: DataPrecision, sm: u16) -> PtxKernel {
        PtxKernel {
            name: String::from(name),
            kernel_type: kt,
            precision: prec,
            min_sm: sm,
            shared_memory: 4096,
            max_threads_per_block: 256,
            registers_per_thread: 16,
        }
    }

    // -- 1. Register kernel -------------------------------------------------

    #[test]
    fn test_register_kernel() {
        let mut reg = PtxRegistry::new();
        let kernel = make_kernel("test_gemm", PtxKernelType::Gemm, DataPrecision::F32, 53);
        let id = reg.register(kernel).unwrap();
        assert_eq!(id, 1);
        assert_eq!(reg.kernel_count(), 1);
    }

    // -- 2. Find kernel by type and precision -------------------------------

    #[test]
    fn test_find_kernel_by_type_and_precision() {
        let mut reg = PtxRegistry::new();
        reg.register(make_kernel(
            "gemm_f32",
            PtxKernelType::Gemm,
            DataPrecision::F32,
            53,
        ))
        .unwrap();
        reg.register(make_kernel(
            "gemm_f16",
            PtxKernelType::Gemm,
            DataPrecision::F16,
            53,
        ))
        .unwrap();

        let cc = test_cc_turing();
        let found = reg.find_kernel(PtxKernelType::Gemm, DataPrecision::F16, &cc);
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "gemm_f16");
    }

    // -- 3. Compatibility check with SM version -----------------------------

    #[test]
    fn test_compatibility_check() {
        let kernel = make_kernel("conv2d_f16", PtxKernelType::Conv2d, DataPrecision::F16, 70);
        // Turing (sm_75) >= 70 => compatible.
        assert!(kernel.is_compatible(&test_cc_turing()));
        // Maxwell (sm_53) < 70 => not compatible.
        assert!(!kernel.is_compatible(&test_cc_maxwell()));
    }

    // -- 4. Register defaults creates 11 kernels ---------------------------

    #[test]
    fn test_register_defaults_creates_11_kernels() {
        let mut reg = PtxRegistry::new();
        reg.register_defaults().unwrap();
        assert_eq!(reg.kernel_count(), 11);
    }

    // -- 5. Incompatible SM version returns None ----------------------------

    #[test]
    fn test_incompatible_sm_returns_none() {
        let mut reg = PtxRegistry::new();
        reg.register_defaults().unwrap();
        let cc = test_cc_too_old(); // SM 3.0, below everything
        let found = reg.find_kernel(PtxKernelType::Gemm, DataPrecision::F32, &cc);
        assert!(found.is_none());
    }

    // -- 6. Compatible kernels filtering ------------------------------------

    #[test]
    fn test_compatible_kernels_filtering() {
        let mut reg = PtxRegistry::new();
        reg.register_defaults().unwrap();

        // Maxwell (SM 5.3): everything except conv2d_f16 (sm 70).
        let compat = reg.compatible_kernels(&test_cc_maxwell());
        assert_eq!(compat.len(), 10);

        // Turing (SM 7.5): all 11 kernels.
        let compat = reg.compatible_kernels(&test_cc_turing());
        assert_eq!(compat.len(), 11);
    }

    // -- 7. Kernel count ----------------------------------------------------

    #[test]
    fn test_kernel_count() {
        let mut reg = PtxRegistry::new();
        assert_eq!(reg.kernel_count(), 0);
        reg.register(make_kernel(
            "a",
            PtxKernelType::Pool,
            DataPrecision::F32,
            53,
        ))
        .unwrap();
        assert_eq!(reg.kernel_count(), 1);
        reg.register(make_kernel(
            "b",
            PtxKernelType::Pool,
            DataPrecision::F16,
            53,
        ))
        .unwrap();
        assert_eq!(reg.kernel_count(), 2);
    }

    // -- 8. Occupancy calculation -------------------------------------------

    #[test]
    fn test_occupancy_calculation() {
        let kernel = PtxKernel {
            name: String::from("occ_test"),
            kernel_type: PtxKernelType::Elementwise,
            precision: DataPrecision::F32,
            min_sm: 53,
            shared_memory: 0,
            max_threads_per_block: 1024,
            registers_per_thread: 8,
        };
        // Half occupancy.
        let occ = kernel.occupancy(512);
        assert!((occ - 0.5).abs() < 1e-9);
        // Full occupancy.
        let occ_full = kernel.occupancy(1024);
        assert!((occ_full - 1.0).abs() < 1e-9);
        // Quarter occupancy.
        let occ_quarter = kernel.occupancy(256);
        assert!((occ_quarter - 0.25).abs() < 1e-9);
    }

    // -- 9. Registry full error ---------------------------------------------

    #[test]
    fn test_registry_full_error() {
        let mut reg = PtxRegistry::new();
        for i in 0..MAX_KERNELS {
            let name = alloc::format!("k{}", i);
            reg.register(make_kernel(
                &name,
                PtxKernelType::Gemm,
                DataPrecision::F32,
                53,
            ))
            .unwrap();
        }
        assert_eq!(reg.kernel_count(), MAX_KERNELS);
        let result = reg.register(make_kernel(
            "overflow",
            PtxKernelType::Gemm,
            DataPrecision::F32,
            53,
        ));
        assert_eq!(result, Err(GpuError::QueueFull));
    }

    // -- 10. All kernel types exist in defaults -----------------------------

    #[test]
    fn test_all_kernel_types_in_defaults() {
        let mut reg = PtxRegistry::new();
        reg.register_defaults().unwrap();
        let cc = test_cc_turing();

        // Every PtxKernelType variant should have at least one kernel.
        let types = [
            PtxKernelType::Gemm,
            PtxKernelType::Conv2d,
            PtxKernelType::Softmax,
            PtxKernelType::LayerNorm,
            PtxKernelType::Elementwise,
            PtxKernelType::Reduce,
            PtxKernelType::Transpose,
            PtxKernelType::Concat,
            PtxKernelType::Pool,
        ];
        for kt in &types {
            let found = reg.find_kernel(kt.clone(), DataPrecision::F32, &cc);
            assert!(found.is_some(), "Expected default F32 kernel for {:?}", kt);
        }
    }

    // -- 11. F16 kernels require higher SM ----------------------------------

    #[test]
    fn test_f16_kernels_require_higher_sm() {
        let mut reg = PtxRegistry::new();
        reg.register_defaults().unwrap();

        // conv2d_f16 requires sm_70. Maxwell (sm_53) should fail.
        let found = reg.find_kernel(
            PtxKernelType::Conv2d,
            DataPrecision::F16,
            &test_cc_maxwell(),
        );
        assert!(
            found.is_none(),
            "conv2d_f16 should not be compatible with Maxwell"
        );

        // But Turing (sm_75) should succeed.
        let found = reg.find_kernel(PtxKernelType::Conv2d, DataPrecision::F16, &test_cc_turing());
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "conv2d_f16");
    }

    // -- 12. Default kernel shared memory values ----------------------------

    #[test]
    fn test_default_kernel_shared_memory() {
        let mut reg = PtxRegistry::new();
        reg.register_defaults().unwrap();
        let cc = test_cc_turing();

        // GEMM kernels should use substantial shared memory.
        let gemm = reg
            .find_kernel(PtxKernelType::Gemm, DataPrecision::F32, &cc)
            .unwrap();
        assert_eq!(gemm.shared_memory, 49152); // 48 KiB

        // Elementwise should use zero shared memory.
        let eltwise = reg
            .find_kernel(PtxKernelType::Elementwise, DataPrecision::F32, &cc)
            .unwrap();
        assert_eq!(eltwise.shared_memory, 0);

        // Softmax uses moderate shared memory.
        let softmax = reg
            .find_kernel(PtxKernelType::Softmax, DataPrecision::F32, &cc)
            .unwrap();
        assert_eq!(softmax.shared_memory, 16384); // 16 KiB
    }

    // -- 13. PtxKernelType variants -----------------------------------------

    #[test]
    fn test_ptx_kernel_type_variants() {
        // Verify all nine variants can be constructed and compared.
        let variants = [
            PtxKernelType::Gemm,
            PtxKernelType::Conv2d,
            PtxKernelType::Softmax,
            PtxKernelType::LayerNorm,
            PtxKernelType::Elementwise,
            PtxKernelType::Reduce,
            PtxKernelType::Transpose,
            PtxKernelType::Concat,
            PtxKernelType::Pool,
        ];
        assert_eq!(variants.len(), 9);
        // Each variant is distinct.
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

    // -- 14. DataPrecision variants -----------------------------------------

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

    // -- 15. Clone and Debug derive work ------------------------------------

    #[test]
    fn test_kernel_clone_debug() {
        let kernel = make_kernel("clone_test", PtxKernelType::Softmax, DataPrecision::F32, 53);
        let cloned = kernel.clone();
        assert_eq!(kernel, cloned);
        let dbg = alloc::format!("{:?}", kernel);
        assert!(dbg.contains("clone_test"));
        assert!(dbg.contains("Softmax"));
    }

    // -- 16. Find non-existent type returns None ----------------------------

    #[test]
    fn test_find_nonexistent_precision_returns_none() {
        let mut reg = PtxRegistry::new();
        reg.register_defaults().unwrap();
        let cc = test_cc_turing();

        // There is no Bf16 kernel in the defaults.
        let found = reg.find_kernel(PtxKernelType::Gemm, DataPrecision::Bf16, &cc);
        assert!(found.is_none());
    }

    // -- 17. identify_gpu produces usable ComputeCapability -----------------

    #[test]
    fn test_identify_gpu_cc_works_with_registry() {
        let gpu = identify_gpu(0x1B80).unwrap();
        let mut reg = PtxRegistry::new();
        reg.register_defaults().unwrap();

        // Tesla T4 (SM 7.5) should be compatible with all 11 default kernels.
        let compat = reg.compatible_kernels(&gpu.compute_capability);
        assert_eq!(compat.len(), 11);
    }

    // -- 18. Registry new starts empty with id 1 ----------------------------

    #[test]
    fn test_registry_initial_state() {
        let reg = PtxRegistry::new();
        assert_eq!(reg.kernel_count(), 0);
        assert_eq!(reg.next_id, 1);
    }

    // -- 19. Multiple registrations yield increasing ids --------------------

    #[test]
    fn test_register_ids_increase() {
        let mut reg = PtxRegistry::new();
        let id1 = reg
            .register(make_kernel(
                "k1",
                PtxKernelType::Pool,
                DataPrecision::F32,
                53,
            ))
            .unwrap();
        let id2 = reg
            .register(make_kernel(
                "k2",
                PtxKernelType::Pool,
                DataPrecision::F16,
                53,
            ))
            .unwrap();
        let id3 = reg
            .register(make_kernel(
                "k3",
                PtxKernelType::Pool,
                DataPrecision::Int8,
                53,
            ))
            .unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    // -- 20. Default Gemm F16 is compatible with sm_53 ----------------------

    #[test]
    fn test_gemm_f16_compatible_with_maxwell() {
        let mut reg = PtxRegistry::new();
        reg.register_defaults().unwrap();
        // gemm_f16 has min_sm=53, Maxwell is sm_53 => exactly compatible.
        let found = reg.find_kernel(PtxKernelType::Gemm, DataPrecision::F16, &test_cc_maxwell());
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "gemm_f16");
    }
}
