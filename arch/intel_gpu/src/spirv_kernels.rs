// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V kernel definitions and registry.
//!
//! Maintains a registry of SPIR-V kernels that implement ONNX operators on
//! Intel GPUs. Each [`SpirvKernel`] records its required Xe generation,
//! shared local memory footprint, and data precision. The [`SpirvRegistry`]
//! provides lookup by operator type, precision, and hardware compatibility.
//!
//! SPIR-V is the standard intermediate representation for Intel GPU compute
//! kernels, consumed by the Intel Graphics Compiler (IGC) and dispatched
//! through the Level Zero API.

use alloc::string::String;
use alloc::vec::Vec;

use crate::gpu_id::XeVersion;
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
    /// 16-bit brain floating-point (supported natively on Xe-HPG+).
    Bf16,
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

/// A single SPIR-V kernel that can be dispatched on an Intel GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct SpirvKernel {
    /// Human-readable kernel identifier (e.g. `"gemm_f32"`).
    pub name: String,
    /// Operator family this kernel implements.
    pub kernel_type: SpirvKernelType,
    /// Numeric precision.
    pub precision: DataPrecision,
    /// Minimum Xe generation required (1=LP, 2=HPG, 3=HPC, 4=LPG, 5=Xe2).
    pub min_xe_version: XeVersion,
    /// Shared local memory requirement in bytes.
    pub shared_local_memory: u32,
    /// Maximum threads per workgroup this kernel supports.
    pub max_threads_per_workgroup: u32,
}

impl SpirvKernel {
    /// Returns `true` when the given Xe version meets or exceeds the
    /// kernel's minimum requirement.
    pub fn is_compatible(&self, xe_ver: &XeVersion) -> bool {
        xe_ver.as_version() >= self.min_xe_version.as_version()
    }

    /// Theoretical occupancy ratio for the given `threads_per_workgroup`.
    ///
    /// Returns a value in `0.0..=1.0` when `threads_per_workgroup` is at most
    /// `max_threads_per_workgroup`, or > 1.0 otherwise (caller should clamp).
    pub fn occupancy(&self, threads_per_workgroup: u32) -> f64 {
        threads_per_workgroup as f64 / self.max_threads_per_workgroup as f64
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
    ///
    /// Fails with [`GpuError::QueueFull`] when the registry has reached
    /// [`MAX_KERNELS`].
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
    /// hardware capability.
    ///
    /// Returns the first kernel that exactly matches `kernel_type` and
    /// `precision` and whose `min_xe_version` does not exceed the device's
    /// Xe version.
    pub fn find_kernel(
        &self,
        kernel_type: SpirvKernelType,
        precision: DataPrecision,
        xe_ver: &XeVersion,
    ) -> Option<&SpirvKernel> {
        self.kernels.iter().find(|k| {
            k.kernel_type == kernel_type && k.precision == precision && k.is_compatible(xe_ver)
        })
    }

    /// Populate the registry with standard kernels covering the core ONNX
    /// operator set.
    ///
    /// Registers 11 default kernels:
    /// - `gemm_f32` (Xe-LP), `gemm_f16` (Xe-LP)
    /// - `conv2d_f32` (Xe-LP), `conv2d_f16` (Xe-HPG)
    /// - `softmax_f32` (Xe-LP), `layernorm_f32` (Xe-LP)
    /// - `elementwise_f32` (Xe-LP), `reduce_f32` (Xe-LP)
    /// - `transpose_f32` (Xe-LP), `concat_f32` (Xe-LP)
    /// - `pool_f32` (Xe-LP)
    pub fn register_defaults(&mut self) -> Result<(), GpuError> {
        let defaults = alloc::vec![
            SpirvKernel {
                name: String::from("gemm_f32"),
                kernel_type: SpirvKernelType::Gemm,
                precision: DataPrecision::F32,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 49152,
                max_threads_per_workgroup: 256,
            },
            SpirvKernel {
                name: String::from("gemm_f16"),
                kernel_type: SpirvKernelType::Gemm,
                precision: DataPrecision::F16,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 32768,
                max_threads_per_workgroup: 256,
            },
            SpirvKernel {
                name: String::from("conv2d_f32"),
                kernel_type: SpirvKernelType::Conv2d,
                precision: DataPrecision::F32,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 49152,
                max_threads_per_workgroup: 256,
            },
            SpirvKernel {
                name: String::from("conv2d_f16"),
                kernel_type: SpirvKernelType::Conv2d,
                precision: DataPrecision::F16,
                min_xe_version: XeVersion::XE_HPG,
                shared_local_memory: 32768,
                max_threads_per_workgroup: 256,
            },
            SpirvKernel {
                name: String::from("softmax_f32"),
                kernel_type: SpirvKernelType::Softmax,
                precision: DataPrecision::F32,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 16384,
                max_threads_per_workgroup: 256,
            },
            SpirvKernel {
                name: String::from("layernorm_f32"),
                kernel_type: SpirvKernelType::LayerNorm,
                precision: DataPrecision::F32,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 16384,
                max_threads_per_workgroup: 256,
            },
            SpirvKernel {
                name: String::from("elementwise_f32"),
                kernel_type: SpirvKernelType::Elementwise,
                precision: DataPrecision::F32,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 0,
                max_threads_per_workgroup: 1024,
            },
            SpirvKernel {
                name: String::from("reduce_f32"),
                kernel_type: SpirvKernelType::Reduce,
                precision: DataPrecision::F32,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 32768,
                max_threads_per_workgroup: 256,
            },
            SpirvKernel {
                name: String::from("transpose_f32"),
                kernel_type: SpirvKernelType::Transpose,
                precision: DataPrecision::F32,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 32768,
                max_threads_per_workgroup: 256,
            },
            SpirvKernel {
                name: String::from("concat_f32"),
                kernel_type: SpirvKernelType::Concat,
                precision: DataPrecision::F32,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 0,
                max_threads_per_workgroup: 1024,
            },
            SpirvKernel {
                name: String::from("pool_f32"),
                kernel_type: SpirvKernelType::Pool,
                precision: DataPrecision::F32,
                min_xe_version: XeVersion::XE_LP,
                shared_local_memory: 16384,
                max_threads_per_workgroup: 256,
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

    /// Return references to all kernels compatible with the given Xe version.
    pub fn compatible_kernels(&self, xe_ver: &XeVersion) -> Vec<&SpirvKernel> {
        self.kernels
            .iter()
            .filter(|k| k.is_compatible(xe_ver))
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

    /// Xe-HPG (Arc A-series).
    fn test_xe_hpg() -> XeVersion {
        XeVersion::XE_HPG
    }

    /// Xe-LP (integrated).
    fn test_xe_lp() -> XeVersion {
        XeVersion::XE_LP
    }

    /// A very old / hypothetical device below any supported generation.
    fn test_xe_too_old() -> XeVersion {
        XeVersion::new(0)
    }

    /// Build a simple test kernel with sensible defaults.
    fn make_kernel(
        name: &str,
        kt: SpirvKernelType,
        prec: DataPrecision,
        min_xe: XeVersion,
    ) -> SpirvKernel {
        SpirvKernel {
            name: String::from(name),
            kernel_type: kt,
            precision: prec,
            min_xe_version: min_xe,
            shared_local_memory: 4096,
            max_threads_per_workgroup: 256,
        }
    }

    // -- 1. Register kernel -------------------------------------------------

    #[test]
    fn test_register_kernel() {
        let mut reg = SpirvRegistry::new();
        let kernel = make_kernel(
            "test_gemm",
            SpirvKernelType::Gemm,
            DataPrecision::F32,
            XeVersion::XE_LP,
        );
        let id = reg.register(kernel).unwrap();
        assert_eq!(id, 1);
        assert_eq!(reg.kernel_count(), 1);
    }

    // -- 2. Find kernel by type and precision -------------------------------

    #[test]
    fn test_find_kernel_by_type_and_precision() {
        let mut reg = SpirvRegistry::new();
        reg.register(make_kernel(
            "gemm_f32",
            SpirvKernelType::Gemm,
            DataPrecision::F32,
            XeVersion::XE_LP,
        ))
        .unwrap();
        reg.register(make_kernel(
            "gemm_f16",
            SpirvKernelType::Gemm,
            DataPrecision::F16,
            XeVersion::XE_LP,
        ))
        .unwrap();

        let xe = test_xe_hpg();
        let found = reg.find_kernel(SpirvKernelType::Gemm, DataPrecision::F16, &xe);
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "gemm_f16");
    }

    // -- 3. Compatibility check with Xe version -----------------------------

    #[test]
    fn test_compatibility_check() {
        let kernel = make_kernel(
            "conv2d_f16",
            SpirvKernelType::Conv2d,
            DataPrecision::F16,
            XeVersion::XE_HPG,
        );
        // Xe-HPG (2) >= Xe-HPG (2) => compatible.
        assert!(kernel.is_compatible(&test_xe_hpg()));
        // Xe-LP (1) < Xe-HPG (2) => not compatible.
        assert!(!kernel.is_compatible(&test_xe_lp()));
    }

    // -- 4. Register defaults creates 11 kernels ---------------------------

    #[test]
    fn test_register_defaults_creates_11_kernels() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        assert_eq!(reg.kernel_count(), 11);
    }

    // -- 5. Incompatible Xe version returns None ----------------------------

    #[test]
    fn test_incompatible_xe_returns_none() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let xe = test_xe_too_old();
        let found = reg.find_kernel(SpirvKernelType::Gemm, DataPrecision::F32, &xe);
        assert!(found.is_none());
    }

    // -- 6. Compatible kernels filtering ------------------------------------

    #[test]
    fn test_compatible_kernels_filtering() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();

        // Xe-LP: everything except conv2d_f16 (requires Xe-HPG).
        let compat = reg.compatible_kernels(&test_xe_lp());
        assert_eq!(compat.len(), 10);

        // Xe-HPG: all 11 kernels.
        let compat = reg.compatible_kernels(&test_xe_hpg());
        assert_eq!(compat.len(), 11);
    }

    // -- 7. Kernel count ----------------------------------------------------

    #[test]
    fn test_kernel_count() {
        let mut reg = SpirvRegistry::new();
        assert_eq!(reg.kernel_count(), 0);
        reg.register(make_kernel(
            "a",
            SpirvKernelType::Pool,
            DataPrecision::F32,
            XeVersion::XE_LP,
        ))
        .unwrap();
        assert_eq!(reg.kernel_count(), 1);
        reg.register(make_kernel(
            "b",
            SpirvKernelType::Pool,
            DataPrecision::F16,
            XeVersion::XE_LP,
        ))
        .unwrap();
        assert_eq!(reg.kernel_count(), 2);
    }

    // -- 8. Occupancy calculation -------------------------------------------

    #[test]
    fn test_occupancy_calculation() {
        let kernel = SpirvKernel {
            name: String::from("occ_test"),
            kernel_type: SpirvKernelType::Elementwise,
            precision: DataPrecision::F32,
            min_xe_version: XeVersion::XE_LP,
            shared_local_memory: 0,
            max_threads_per_workgroup: 1024,
        };
        let occ = kernel.occupancy(512);
        assert!((occ - 0.5).abs() < 1e-9);
        let occ_full = kernel.occupancy(1024);
        assert!((occ_full - 1.0).abs() < 1e-9);
        let occ_quarter = kernel.occupancy(256);
        assert!((occ_quarter - 0.25).abs() < 1e-9);
    }

    // -- 9. Registry full error ---------------------------------------------

    #[test]
    fn test_registry_full_error() {
        let mut reg = SpirvRegistry::new();
        for i in 0..MAX_KERNELS {
            let name = alloc::format!("k{}", i);
            reg.register(make_kernel(
                &name,
                SpirvKernelType::Gemm,
                DataPrecision::F32,
                XeVersion::XE_LP,
            ))
            .unwrap();
        }
        assert_eq!(reg.kernel_count(), MAX_KERNELS);
        let result = reg.register(make_kernel(
            "overflow",
            SpirvKernelType::Gemm,
            DataPrecision::F32,
            XeVersion::XE_LP,
        ));
        assert_eq!(result, Err(GpuError::QueueFull));
    }

    // -- 10. All kernel types exist in defaults -----------------------------

    #[test]
    fn test_all_kernel_types_in_defaults() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let xe = test_xe_hpg();

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
            let found = reg.find_kernel(kt.clone(), DataPrecision::F32, &xe);
            assert!(found.is_some(), "Expected default F32 kernel for {:?}", kt);
        }
    }

    // -- 11. F16 kernels require higher Xe version --------------------------

    #[test]
    fn test_f16_kernels_require_higher_xe() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();

        // conv2d_f16 requires Xe-HPG. Xe-LP should fail.
        let found = reg.find_kernel(SpirvKernelType::Conv2d, DataPrecision::F16, &test_xe_lp());
        assert!(
            found.is_none(),
            "conv2d_f16 should not be compatible with Xe-LP"
        );

        // But Xe-HPG should succeed.
        let found = reg.find_kernel(SpirvKernelType::Conv2d, DataPrecision::F16, &test_xe_hpg());
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "conv2d_f16");
    }

    // -- 12. Default kernel shared memory values ----------------------------

    #[test]
    fn test_default_kernel_shared_memory() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let xe = test_xe_hpg();

        let gemm = reg
            .find_kernel(SpirvKernelType::Gemm, DataPrecision::F32, &xe)
            .unwrap();
        assert_eq!(gemm.shared_local_memory, 49152); // 48 KiB

        let eltwise = reg
            .find_kernel(SpirvKernelType::Elementwise, DataPrecision::F32, &xe)
            .unwrap();
        assert_eq!(eltwise.shared_local_memory, 0);

        let softmax = reg
            .find_kernel(SpirvKernelType::Softmax, DataPrecision::F32, &xe)
            .unwrap();
        assert_eq!(softmax.shared_local_memory, 16384); // 16 KiB
    }

    // -- 13. SpirvKernelType variants -----------------------------------------

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

    // -- 14. DataPrecision variants -----------------------------------------

    #[test]
    fn test_data_precision_variants() {
        let variants = [DataPrecision::F32, DataPrecision::F16, DataPrecision::Bf16];
        assert_eq!(variants.len(), 3);
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
        let kernel = make_kernel(
            "clone_test",
            SpirvKernelType::Softmax,
            DataPrecision::F32,
            XeVersion::XE_LP,
        );
        let cloned = kernel.clone();
        assert_eq!(kernel, cloned);
        let dbg = alloc::format!("{:?}", kernel);
        assert!(dbg.contains("clone_test"));
        assert!(dbg.contains("Softmax"));
    }

    // -- 16. Find non-existent type returns None ----------------------------

    #[test]
    fn test_find_nonexistent_precision_returns_none() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let xe = test_xe_hpg();

        // There is no Bf16 kernel in the defaults.
        let found = reg.find_kernel(SpirvKernelType::Gemm, DataPrecision::Bf16, &xe);
        assert!(found.is_none());
    }

    // -- 17. identify_gpu produces usable XeVersion -------------------------

    #[test]
    fn test_identify_gpu_xe_works_with_registry() {
        let gpu = identify_gpu(0x56A0).unwrap();
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();

        // Arc A770 (Xe-HPG) should be compatible with all 11 default kernels.
        let compat = reg.compatible_kernels(&gpu.xe_version);
        assert_eq!(compat.len(), 11);
    }

    // -- 18. Registry new starts empty with id 1 ----------------------------

    #[test]
    fn test_registry_initial_state() {
        let reg = SpirvRegistry::new();
        assert_eq!(reg.kernel_count(), 0);
        assert_eq!(reg.next_id, 1);
    }

    // -- 19. Multiple registrations yield increasing ids --------------------

    #[test]
    fn test_register_ids_increase() {
        let mut reg = SpirvRegistry::new();
        let id1 = reg
            .register(make_kernel(
                "k1",
                SpirvKernelType::Pool,
                DataPrecision::F32,
                XeVersion::XE_LP,
            ))
            .unwrap();
        let id2 = reg
            .register(make_kernel(
                "k2",
                SpirvKernelType::Pool,
                DataPrecision::F16,
                XeVersion::XE_LP,
            ))
            .unwrap();
        let id3 = reg
            .register(make_kernel(
                "k3",
                SpirvKernelType::Pool,
                DataPrecision::Bf16,
                XeVersion::XE_LP,
            ))
            .unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    // -- 20. Default Gemm F16 is compatible with Xe-LP ----------------------

    #[test]
    fn test_gemm_f16_compatible_with_xe_lp() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        // gemm_f16 has min_xe=XE_LP, Xe-LP should match.
        let found = reg.find_kernel(SpirvKernelType::Gemm, DataPrecision::F16, &test_xe_lp());
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "gemm_f16");
    }

    // -- 21. Xe-HPC compatible with all defaults ----------------------------

    #[test]
    fn test_xe_hpc_compatible_with_all_defaults() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let xe_hpc = XeVersion::XE_HPC;
        let compat = reg.compatible_kernels(&xe_hpc);
        assert_eq!(compat.len(), 11);
    }

    // -- 22. Xe2 compatible with all defaults --------------------------------

    #[test]
    fn test_xe2_compatible_with_all_defaults() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let xe2 = XeVersion::XE2;
        let compat = reg.compatible_kernels(&xe2);
        assert_eq!(compat.len(), 11);
    }

    // -- 23. XeVersion equality ---------------------------------------------

    #[test]
    fn test_xe_version_equality() {
        assert_eq!(XeVersion::XE_LP, XeVersion::new(1));
        assert_eq!(XeVersion::XE_HPG, XeVersion::new(2));
        assert_ne!(XeVersion::XE_LP, XeVersion::XE_HPG);
    }

    // -- 24. Max threads per workgroup in defaults --------------------------

    #[test]
    fn test_max_threads_in_defaults() {
        let mut reg = SpirvRegistry::new();
        reg.register_defaults().unwrap();
        let xe = test_xe_hpg();

        // Elementwise and concat should support 1024 threads.
        let eltwise = reg
            .find_kernel(SpirvKernelType::Elementwise, DataPrecision::F32, &xe)
            .unwrap();
        assert_eq!(eltwise.max_threads_per_workgroup, 1024);

        let concat = reg
            .find_kernel(SpirvKernelType::Concat, DataPrecision::F32, &xe)
            .unwrap();
        assert_eq!(concat.max_threads_per_workgroup, 1024);

        // GEMM should support 256 threads.
        let gemm = reg
            .find_kernel(SpirvKernelType::Gemm, DataPrecision::F32, &xe)
            .unwrap();
        assert_eq!(gemm.max_threads_per_workgroup, 256);
    }

    // -- 25. Empty registry find returns None --------------------------------

    #[test]
    fn test_empty_registry_find_returns_none() {
        let reg = SpirvRegistry::new();
        let found = reg.find_kernel(SpirvKernelType::Gemm, DataPrecision::F32, &XeVersion::XE_LP);
        assert!(found.is_none());
    }
}
