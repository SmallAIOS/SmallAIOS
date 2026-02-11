// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ROCm execution provider for ONNX inference.
//!
//! [`RocmProvider`] is the top-level entry point that wires together the
//! VRAM allocator, SDMA engine, compute engine, and HIP kernel registry to
//! execute ONNX operators on an AMD GPU. It translates high-level ONNX
//! operator names into HIP kernel launches with appropriate grid/workgroup sizing.

use alloc::string::String;
use alloc::vec::Vec;

use crate::compute::{ComputeEngine, Dim3, LaunchConfig, WavefrontMode};
use crate::dma::{DmaDirection, SdmaEngine};
use crate::gpu_id::GpuInfo;
use crate::hip_kernels::{DataPrecision, HipKernelType, HipRegistry};
use crate::memory::{MemoryRegion, VramAllocator};
use crate::GpuError;

// ---------------------------------------------------------------------------
// OperatorMapping
// ---------------------------------------------------------------------------

/// Maps an ONNX operator name to a GPU kernel type and precision.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorMapping {
    /// ONNX operator name (e.g. `"MatMul"`, `"Conv"`).
    pub op_name: String,
    /// HIP kernel family to use.
    pub kernel_type: HipKernelType,
    /// Numeric precision for the kernel.
    pub precision: DataPrecision,
}

// ---------------------------------------------------------------------------
// ExecutionStep / ExecutionPlan
// ---------------------------------------------------------------------------

/// A single step in an execution plan.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionStep {
    /// ONNX operator name.
    pub op_name: String,
    /// HIP kernel type to dispatch.
    pub kernel_type: HipKernelType,
    /// VRAM allocation ids for input tensors.
    pub input_allocs: Vec<u64>,
    /// VRAM allocation id for the output tensor.
    pub output_alloc: u64,
    /// Launch configuration for this step.
    pub config: LaunchConfig,
}

/// A complete execution plan for a model inference pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionPlan {
    /// Ordered sequence of execution steps.
    pub steps: Vec<ExecutionStep>,
    /// Total workspace memory required in bytes.
    pub total_workspace: u64,
}

// ---------------------------------------------------------------------------
// ProviderStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of the ROCm execution provider.
#[derive(Clone, Debug, PartialEq)]
pub enum ProviderStatus {
    /// Provider has not been initialised.
    Uninitialized,
    /// Ready to accept work.
    Ready,
    /// Currently executing a kernel or model.
    Executing,
    /// An unrecoverable error has occurred.
    Error,
}

// ---------------------------------------------------------------------------
// RocmProvider
// ---------------------------------------------------------------------------

/// ROCm execution provider that dispatches ONNX operators to AMD GPU kernels.
///
/// Owns all GPU-side resources: memory allocator, compute and SDMA engines,
/// and the HIP kernel registry.
pub struct RocmProvider {
    /// Static description of the GPU.
    gpu_info: GpuInfo,
    /// VRAM allocator (70% static / 30% dynamic).
    allocator: VramAllocator,
    /// Kernel launch engine.
    compute: ComputeEngine,
    /// Host<->device SDMA engine.
    sdma: SdmaEngine,
    /// Registered HIP kernels.
    registry: HipRegistry,
    /// Current provider state.
    status: ProviderStatus,
    /// Number of models whose weights have been loaded.
    models_loaded: u32,
}

impl RocmProvider {
    /// Create a new ROCm provider for the given GPU.
    ///
    /// `vram_size` is the total usable VRAM in bytes. The allocator is
    /// configured with 70% static (weights) / 30% dynamic (workspace).
    /// Default HIP kernels are registered and the provider moves to
    /// [`ProviderStatus::Ready`].
    pub fn new(gpu_info: GpuInfo, vram_size: u64) -> Self {
        let allocator = VramAllocator::new(vram_size, 0.7);
        let compute = ComputeEngine::new();
        let sdma = SdmaEngine::new(0);
        let mut registry = HipRegistry::new();
        let _ = registry.register_defaults();

        Self {
            gpu_info,
            allocator,
            compute,
            sdma,
            registry,
            status: ProviderStatus::Ready,
            models_loaded: 0,
        }
    }

    /// Current provider status.
    pub fn status(&self) -> &ProviderStatus {
        &self.status
    }

    /// Static GPU information.
    pub fn gpu_info(&self) -> &GpuInfo {
        &self.gpu_info
    }

    /// Load model weights into VRAM (static region).
    pub fn load_weights(&mut self, size: u64) -> Result<u64, GpuError> {
        let id = self.allocator.alloc(size, MemoryRegion::Static)?;
        self.models_loaded += 1;
        Ok(id)
    }

    /// Allocate workspace memory (dynamic region) for intermediate tensors.
    pub fn allocate_workspace(&mut self, size: u64) -> Result<u64, GpuError> {
        self.allocator.alloc(size, MemoryRegion::Dynamic)
    }

    /// Free a previous VRAM allocation (static or dynamic).
    pub fn free_allocation(&mut self, id: u64) -> Result<(), GpuError> {
        self.allocator.free(id)
    }

    /// Submit a host-to-device DMA transfer.
    pub fn transfer_to_device(&mut self, src: u64, dst: u64, size: u64) -> Result<u64, GpuError> {
        let tid = self
            .sdma
            .submit(DmaDirection::HostToDevice, src, dst, size)?;
        Ok(tid.0)
    }

    /// Submit a device-to-host DMA transfer.
    pub fn transfer_from_device(&mut self, src: u64, dst: u64, size: u64) -> Result<u64, GpuError> {
        let tid = self
            .sdma
            .submit(DmaDirection::DeviceToHost, src, dst, size)?;
        Ok(tid.0)
    }

    /// Map an ONNX operator name to a GPU kernel type and precision.
    pub fn map_operator(op_name: &str) -> Option<OperatorMapping> {
        let kernel_type = match op_name {
            "MatMul" | "Gemm" => HipKernelType::Gemm,
            "Conv" => HipKernelType::Conv2d,
            "Relu" | "Sigmoid" | "Tanh" | "Add" | "Mul" | "Sub" => HipKernelType::Elementwise,
            "Softmax" => HipKernelType::Softmax,
            "LayerNormalization" => HipKernelType::LayerNorm,
            "ReduceSum" | "ReduceMean" | "ReduceMax" => HipKernelType::Reduce,
            "Transpose" => HipKernelType::Transpose,
            "Concat" => HipKernelType::Concat,
            "MaxPool" | "AveragePool" | "GlobalAveragePool" => HipKernelType::Pool,
            "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" => return None,
            _ => return None,
        };

        Some(OperatorMapping {
            op_name: String::from(op_name),
            kernel_type,
            precision: DataPrecision::F32,
        })
    }

    /// Determine wavefront mode based on GPU architecture.
    fn wavefront_mode(&self) -> WavefrontMode {
        if self.gpu_info.architecture.is_rdna() {
            WavefrontMode::Wave32
        } else {
            WavefrontMode::Wave64
        }
    }

    /// Launch a GPU kernel for the named ONNX operator.
    pub fn launch_kernel(&mut self, op_name: &str, elements: u32) -> Result<u64, GpuError> {
        let mapping = Self::map_operator(op_name).ok_or(GpuError::NotFound)?;

        let hip_kernel = self
            .registry
            .find_kernel(
                mapping.kernel_type,
                mapping.precision,
                &self.gpu_info.gfx_version,
            )
            .ok_or(GpuError::LaunchError)?;

        let workgroup_size: u32 = 256;
        let grid_size = elements.div_ceil(workgroup_size);

        let config = LaunchConfig {
            grid: Dim3::linear(grid_size),
            workgroup: Dim3::linear(workgroup_size),
            lds_size: hip_kernel.lds_size,
            queue: 0,
            wavefront_mode: self.wavefront_mode(),
        };

        let kid = self.compute.launch(op_name, config)?;
        self.compute.dispatch(kid)?;
        Ok(kid.0)
    }

    /// Synchronise the compute engine.
    pub fn synchronize(&mut self) {
        self.compute.synchronize();
    }

    /// Total VRAM currently in use (bytes).
    pub fn vram_used(&self) -> u64 {
        self.allocator.total_used()
    }

    /// Total VRAM available for new allocations (bytes).
    pub fn vram_free(&self) -> u64 {
        self.allocator.total_free()
    }

    /// Number of models whose weights have been loaded.
    pub fn models_loaded(&self) -> u32 {
        self.models_loaded
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::KernelStatus;
    use crate::gpu_id::identify_gpu;
    use crate::memory::GPU_PAGE_SIZE;

    fn test_gpu_info() -> GpuInfo {
        identify_gpu(0x7500).unwrap() // MI300X (CDNA 3, gfx942)
    }

    fn test_rdna_gpu_info() -> GpuInfo {
        identify_gpu(0x7440).unwrap() // RX 7900 XTX (RDNA 3, gfx1100)
    }

    fn test_provider() -> RocmProvider {
        let gpu = test_gpu_info();
        RocmProvider::new(gpu, 1024 * 1024 * 1024) // 1 GiB
    }

    fn test_rdna_provider() -> RocmProvider {
        let gpu = test_rdna_gpu_info();
        RocmProvider::new(gpu, 1024 * 1024 * 1024)
    }

    const PAGE: u64 = GPU_PAGE_SIZE as u64;

    #[test]
    fn test_new_provider_is_ready() {
        let prov = test_provider();
        assert_eq!(*prov.status(), ProviderStatus::Ready);
    }

    #[test]
    fn test_load_weights_static_region() {
        let mut prov = test_provider();
        let initial_used = prov.allocator.used_bytes(MemoryRegion::Static);
        let id = prov.load_weights(PAGE).unwrap();
        assert!(id >= 1);
        assert_eq!(
            prov.allocator.used_bytes(MemoryRegion::Static),
            initial_used + PAGE
        );
    }

    #[test]
    fn test_allocate_workspace_dynamic_region() {
        let mut prov = test_provider();
        let initial_used = prov.allocator.used_bytes(MemoryRegion::Dynamic);
        let id = prov.allocate_workspace(PAGE * 2).unwrap();
        assert!(id >= 1);
        assert_eq!(
            prov.allocator.used_bytes(MemoryRegion::Dynamic),
            initial_used + PAGE * 2
        );
    }

    #[test]
    fn test_free_allocation() {
        let mut prov = test_provider();
        let id = prov.load_weights(PAGE).unwrap();
        let used_before_free = prov.vram_used();
        prov.free_allocation(id).unwrap();
        assert_eq!(prov.vram_used(), used_before_free - PAGE);
    }

    #[test]
    fn test_transfer_to_device() {
        let mut prov = test_provider();
        let tid = prov.transfer_to_device(0x1000, 0x2000, 4096).unwrap();
        assert!(tid >= 1);
    }

    #[test]
    fn test_transfer_from_device() {
        let mut prov = test_provider();
        let tid = prov.transfer_from_device(0x2000, 0x1000, 8192).unwrap();
        assert!(tid >= 1);
    }

    #[test]
    fn test_map_known_operators() {
        let cases = [
            ("MatMul", HipKernelType::Gemm),
            ("Gemm", HipKernelType::Gemm),
            ("Conv", HipKernelType::Conv2d),
            ("Relu", HipKernelType::Elementwise),
            ("Sigmoid", HipKernelType::Elementwise),
            ("Tanh", HipKernelType::Elementwise),
            ("Add", HipKernelType::Elementwise),
            ("Mul", HipKernelType::Elementwise),
            ("Sub", HipKernelType::Elementwise),
            ("Softmax", HipKernelType::Softmax),
            ("LayerNormalization", HipKernelType::LayerNorm),
            ("ReduceSum", HipKernelType::Reduce),
            ("ReduceMean", HipKernelType::Reduce),
            ("ReduceMax", HipKernelType::Reduce),
            ("Transpose", HipKernelType::Transpose),
            ("Concat", HipKernelType::Concat),
            ("MaxPool", HipKernelType::Pool),
            ("AveragePool", HipKernelType::Pool),
            ("GlobalAveragePool", HipKernelType::Pool),
        ];
        for (op, expected_type) in &cases {
            let mapping = RocmProvider::map_operator(op);
            assert!(mapping.is_some(), "Expected mapping for {}", op);
            let m = mapping.unwrap();
            assert_eq!(m.kernel_type, *expected_type, "Wrong type for {}", op);
            assert_eq!(m.precision, DataPrecision::F32);
            assert_eq!(m.op_name, *op);
        }
    }

    #[test]
    fn test_map_unknown_operator_returns_none() {
        assert!(RocmProvider::map_operator("UnknownOp").is_none());
        assert!(RocmProvider::map_operator("").is_none());
    }

    #[test]
    fn test_map_reshape_returns_none() {
        assert!(RocmProvider::map_operator("Reshape").is_none());
        assert!(RocmProvider::map_operator("Flatten").is_none());
        assert!(RocmProvider::map_operator("Squeeze").is_none());
        assert!(RocmProvider::map_operator("Unsqueeze").is_none());
    }

    #[test]
    fn test_launch_matmul_kernel() {
        let mut prov = test_provider();
        let launch_id = prov.launch_kernel("MatMul", 1024).unwrap();
        assert!(launch_id >= 1);
    }

    #[test]
    fn test_launch_elementwise_kernel() {
        let mut prov = test_provider();
        let launch_id = prov.launch_kernel("Relu", 512).unwrap();
        assert!(launch_id >= 1);
    }

    #[test]
    fn test_synchronize() {
        let mut prov = test_provider();
        let kid_raw = prov.launch_kernel("Add", 256).unwrap();
        let kid = crate::compute::KernelId(kid_raw);
        assert_eq!(prov.compute.status(kid), Ok(&KernelStatus::Running));
        prov.synchronize();
        assert_eq!(prov.compute.status(kid), Ok(&KernelStatus::Completed));
    }

    #[test]
    fn test_vram_tracking() {
        let mut prov = test_provider();
        let total = prov.vram_used() + prov.vram_free();
        assert_eq!(total, 1024 * 1024 * 1024);

        let initial_used = prov.vram_used();
        prov.load_weights(PAGE).unwrap();
        assert_eq!(prov.vram_used(), initial_used + PAGE);
        assert_eq!(prov.vram_free(), total - prov.vram_used());
    }

    #[test]
    fn test_provider_gpu_info_access() {
        let prov = test_provider();
        let info = prov.gpu_info();
        assert_eq!(info.device_id, 0x7500);
        assert_eq!(info.name, "AMD Instinct MI300X");
        assert_eq!(info.gfx_version.as_gfx_id(), 942);
    }

    #[test]
    fn test_launch_config_grid_workgroup_sizing() {
        let mut prov = test_provider();

        let kid_raw = prov.launch_kernel("MatMul", 1000).unwrap();
        let kid = crate::compute::KernelId(kid_raw);
        let kernel = prov.compute.kernels.iter().find(|k| k.id == kid).unwrap();
        assert_eq!(kernel.config.grid.x, 4);
        assert_eq!(kernel.config.workgroup.x, 256);

        let kid_raw2 = prov.launch_kernel("Relu", 256).unwrap();
        let kid2 = crate::compute::KernelId(kid_raw2);
        let kernel2 = prov.compute.kernels.iter().find(|k| k.id == kid2).unwrap();
        assert_eq!(kernel2.config.grid.x, 1);
        assert_eq!(kernel2.config.workgroup.x, 256);
    }

    #[test]
    fn test_provider_status_enum() {
        assert_ne!(ProviderStatus::Uninitialized, ProviderStatus::Ready);
        assert_ne!(ProviderStatus::Ready, ProviderStatus::Executing);
        assert_ne!(ProviderStatus::Executing, ProviderStatus::Error);
        let cloned = ProviderStatus::Ready.clone();
        assert_eq!(cloned, ProviderStatus::Ready);
        let dbg = alloc::format!("{:?}", ProviderStatus::Executing);
        assert!(dbg.contains("Executing"));
    }

    #[test]
    fn test_models_loaded_counter() {
        let mut prov = test_provider();
        assert_eq!(prov.models_loaded(), 0);
        prov.load_weights(PAGE).unwrap();
        assert_eq!(prov.models_loaded(), 1);
        prov.load_weights(PAGE).unwrap();
        assert_eq!(prov.models_loaded(), 2);
    }

    #[test]
    fn test_launch_unknown_op_returns_not_found() {
        let mut prov = test_provider();
        let result = prov.launch_kernel("UnknownOp", 100);
        assert_eq!(result, Err(GpuError::NotFound));
    }

    #[test]
    fn test_launch_cpu_only_op_returns_not_found() {
        let mut prov = test_provider();
        let result = prov.launch_kernel("Reshape", 100);
        assert_eq!(result, Err(GpuError::NotFound));
    }

    #[test]
    fn test_execution_plan_derives() {
        let step = ExecutionStep {
            op_name: String::from("MatMul"),
            kernel_type: HipKernelType::Gemm,
            input_allocs: alloc::vec![1, 2],
            output_alloc: 3,
            config: LaunchConfig {
                grid: Dim3::linear(4),
                workgroup: Dim3::linear(256),
                lds_size: 49152,
                queue: 0,
                wavefront_mode: WavefrontMode::Wave64,
            },
        };
        let plan = ExecutionPlan {
            steps: alloc::vec![step.clone()],
            total_workspace: 1024 * 1024,
        };
        let plan2 = plan.clone();
        assert_eq!(plan, plan2);
        let dbg = alloc::format!("{:?}", plan);
        assert!(dbg.contains("MatMul"));
    }

    #[test]
    fn test_multiple_launches_and_sync() {
        let mut prov = test_provider();
        let id1 = prov.launch_kernel("MatMul", 1024).unwrap();
        let id2 = prov.launch_kernel("Relu", 2048).unwrap();
        let id3 = prov.launch_kernel("Softmax", 512).unwrap();
        assert!(id1 < id2);
        assert!(id2 < id3);
        prov.synchronize();
        for kid_raw in [id1, id2, id3] {
            let kid = crate::compute::KernelId(kid_raw);
            assert_eq!(prov.compute.status(kid), Ok(&KernelStatus::Completed));
        }
    }

    #[test]
    fn test_lds_passed_through() {
        let mut prov = test_provider();
        let kid_raw = prov.launch_kernel("MatMul", 256).unwrap();
        let kid = crate::compute::KernelId(kid_raw);
        let kernel = prov.compute.kernels.iter().find(|k| k.id == kid).unwrap();
        assert_eq!(kernel.config.lds_size, 49152);

        let kid_raw2 = prov.launch_kernel("Relu", 256).unwrap();
        let kid2 = crate::compute::KernelId(kid_raw2);
        let kernel2 = prov.compute.kernels.iter().find(|k| k.id == kid2).unwrap();
        assert_eq!(kernel2.config.lds_size, 0);
    }

    #[test]
    fn test_cdna_uses_wave64() {
        let mut prov = test_provider();
        let kid_raw = prov.launch_kernel("MatMul", 256).unwrap();
        let kid = crate::compute::KernelId(kid_raw);
        let kernel = prov.compute.kernels.iter().find(|k| k.id == kid).unwrap();
        assert_eq!(kernel.config.wavefront_mode, WavefrontMode::Wave64);
    }

    #[test]
    fn test_rdna_uses_wave32() {
        let mut prov = test_rdna_provider();
        let kid_raw = prov.launch_kernel("MatMul", 256).unwrap();
        let kid = crate::compute::KernelId(kid_raw);
        let kernel = prov.compute.kernels.iter().find(|k| k.id == kid).unwrap();
        assert_eq!(kernel.config.wavefront_mode, WavefrontMode::Wave32);
    }
}
