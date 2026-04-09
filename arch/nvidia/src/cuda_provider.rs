// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CUDA execution provider for ONNX inference.
//!
//! [`CudaProvider`] is the top-level entry point that wires together the
//! VRAM allocator, DMA engine, compute engine, and PTX kernel registry to
//! execute ONNX operators on an NVIDIA GPU.  It translates high-level ONNX
//! operator names into PTX kernel launches with appropriate grid/block sizing.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::compute::{ComputeEngine, Dim3, LaunchConfig};
use crate::dma::{DmaDirection, DmaEngine};
use crate::gpu_id::GpuInfo;
use crate::memory::{MemoryRegion, VramAllocator};
use crate::ptx::{DataPrecision, PtxKernelType, PtxRegistry};
use crate::GpuError;

// ---------------------------------------------------------------------------
// OperatorMapping
// ---------------------------------------------------------------------------

/// Maps an ONNX operator name to a GPU kernel type and precision.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorMapping {
    /// ONNX operator name (e.g. `"MatMul"`, `"Conv"`).
    pub op_name: String,
    /// PTX kernel family to use.
    pub kernel_type: PtxKernelType,
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
    /// PTX kernel type to dispatch.
    pub kernel_type: PtxKernelType,
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

/// Lifecycle state of the CUDA execution provider.
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
// CudaProvider
// ---------------------------------------------------------------------------

/// CUDA execution provider that dispatches ONNX operators to GPU kernels.
///
/// Owns all GPU-side resources: memory allocator, compute and DMA engines,
/// and the PTX kernel registry.
pub struct CudaProvider {
    /// Static description of the GPU.
    gpu_info: GpuInfo,
    /// VRAM allocator (70 % static / 30 % dynamic).
    allocator: VramAllocator,
    /// Kernel launch engine.
    compute: ComputeEngine,
    /// Host<->device DMA engine.
    dma: DmaEngine,
    /// Registered PTX kernels.
    registry: PtxRegistry,
    /// Current provider state.
    status: ProviderStatus,
    /// Number of models whose weights have been loaded.
    models_loaded: u32,
}

impl CudaProvider {
    /// Default DRAM budget for Tegra T210 — 256 MiB carved from shared
    /// system memory for GPU use (weights + workspace).
    #[cfg(feature = "tegra")]
    const TEGRA_DEFAULT_DRAM_BUDGET: u64 = 256 * 1024 * 1024;

    /// Create a new CUDA provider pre-configured for the Tegra T210 GM20B.
    ///
    /// The GM20B is the SoC-integrated Maxwell GPU on the Jetson Nano
    /// (CC 5.3, 2 SMs, 256 CUDA cores).  It has no discrete VRAM — the
    /// allocator is backed by a 256 MiB carve-out from shared system DRAM
    /// with the standard 70 % static / 30 % dynamic split.
    #[cfg(feature = "tegra")]
    pub fn new_tegra() -> Self {
        let gpu_info = crate::gpu_id::identify_gpu(0x12B1)
            .expect("GM20B device ID 0x12B1 must be in the GPU ID table");
        Self::new(gpu_info, Self::TEGRA_DEFAULT_DRAM_BUDGET)
    }

    /// Create a new CUDA provider for the given GPU.
    ///
    /// `vram_size` is the total usable VRAM in bytes.  The allocator is
    /// configured with 70 % static (weights) / 30 % dynamic (workspace).
    /// Default PTX kernels are registered and the provider moves to
    /// [`ProviderStatus::Ready`].
    pub fn new(gpu_info: GpuInfo, vram_size: u64) -> Self {
        let allocator = VramAllocator::new(vram_size, 0.7);
        let compute = ComputeEngine::new();
        let dma = DmaEngine::new();
        let mut registry = PtxRegistry::new();
        // Registering defaults should not fail on a fresh registry.
        let _ = registry.register_defaults();

        Self {
            gpu_info,
            allocator,
            compute,
            dma,
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
    ///
    /// Returns the VRAM allocation id on success and increments the loaded
    /// model counter.
    pub fn load_weights(&mut self, size: u64) -> Result<u64, GpuError> {
        let id = self.allocator.alloc(size, MemoryRegion::Static)?;
        self.models_loaded += 1;
        Ok(id)
    }

    /// Allocate workspace memory (dynamic region) for intermediate tensors.
    ///
    /// Returns the VRAM allocation id.
    pub fn allocate_workspace(&mut self, size: u64) -> Result<u64, GpuError> {
        self.allocator.alloc(size, MemoryRegion::Dynamic)
    }

    /// Free a previous VRAM allocation (static or dynamic).
    pub fn free_allocation(&mut self, id: u64) -> Result<(), GpuError> {
        self.allocator.free(id)
    }

    /// Submit a host-to-device DMA transfer.
    ///
    /// Returns the DMA transfer id.
    pub fn transfer_to_device(&mut self, src: u64, dst: u64, size: u64) -> Result<u64, GpuError> {
        let tid = self
            .dma
            .submit(DmaDirection::HostToDevice, src, dst, size)?;
        Ok(tid.0)
    }

    /// Submit a device-to-host DMA transfer.
    ///
    /// Returns the DMA transfer id.
    pub fn transfer_from_device(&mut self, src: u64, dst: u64, size: u64) -> Result<u64, GpuError> {
        let tid = self
            .dma
            .submit(DmaDirection::DeviceToHost, src, dst, size)?;
        Ok(tid.0)
    }

    /// Map an ONNX operator name to a GPU kernel type and precision.
    ///
    /// Returns `None` for operators that are handled on the CPU (reshape-like
    /// ops) or that are not recognised.
    pub fn map_operator(op_name: &str) -> Option<OperatorMapping> {
        let kernel_type = match op_name {
            "MatMul" | "Gemm" => PtxKernelType::Gemm,
            "Conv" => PtxKernelType::Conv2d,
            "Relu" | "Sigmoid" | "Tanh" | "Add" | "Mul" | "Sub" => PtxKernelType::Elementwise,
            "Softmax" => PtxKernelType::Softmax,
            "LayerNormalization" => PtxKernelType::LayerNorm,
            "ReduceSum" | "ReduceMean" | "ReduceMax" => PtxKernelType::Reduce,
            "Transpose" => PtxKernelType::Transpose,
            "Concat" => PtxKernelType::Concat,
            "MaxPool" | "AveragePool" | "GlobalAveragePool" => PtxKernelType::Pool,
            // CPU-only shape manipulation — no GPU kernel needed.
            "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" => return None,
            // Unknown operator.
            _ => return None,
        };

        Some(OperatorMapping {
            op_name: String::from(op_name),
            kernel_type,
            precision: DataPrecision::F32,
        })
    }

    /// Launch a GPU kernel for the named ONNX operator.
    ///
    /// Looks up the operator mapping, finds the matching PTX kernel, builds a
    /// launch configuration with block size 256, and dispatches to the compute
    /// engine.  Returns the kernel launch id.
    pub fn launch_kernel(&mut self, op_name: &str, elements: u32) -> Result<u64, GpuError> {
        // 1. Map ONNX op to kernel type.
        let mapping = Self::map_operator(op_name).ok_or(GpuError::NotFound)?;

        // 2. Find a compatible PTX kernel.
        let ptx_kernel = self
            .registry
            .find_kernel(
                mapping.kernel_type,
                mapping.precision,
                &self.gpu_info.compute_capability,
            )
            .ok_or(GpuError::LaunchError)?;

        // 3. Build launch config: block = 256, grid = ceil(elements / 256).
        let block_size: u32 = 256;
        let grid_size = elements.div_ceil(block_size);

        let config = LaunchConfig {
            grid: Dim3::linear(grid_size),
            block: Dim3::linear(block_size),
            shared_memory: ptx_kernel.shared_memory,
            stream: 0,
        };

        // 4. Submit to compute engine and dispatch immediately.
        let kid = self.compute.launch(op_name, config)?;
        self.compute.dispatch(kid)?;
        Ok(kid.0)
    }

    /// Synchronise the compute engine — wait for all running kernels to
    /// complete.
    pub fn synchronize(&mut self) {
        self.compute.synchronize();
    }

    /// Total VRAM currently in use (bytes, across both regions).
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

// ---------------------------------------------------------------------------
// ComputeProvider implementation
// ---------------------------------------------------------------------------

impl smallaios_compute::ComputeProvider for CudaProvider {
    type Buffer = u64; // VRAM allocation ID
    type Kernel = u64; // Kernel launch ID (name index)
    type Error = GpuError;

    fn device_info(&self) -> smallaios_compute::DeviceInfo {
        smallaios_compute::DeviceInfo {
            name: String::from(self.gpu_info.name),
            memory_bytes: self.gpu_info.vram_size_mb as u64 * 1024 * 1024,
            compute_units: self.gpu_info.sm_count,
            backend_type: smallaios_compute::BackendType::Cuda,
        }
    }

    fn init(&mut self) -> Result<(), Self::Error> {
        self.status = ProviderStatus::Ready;
        Ok(())
    }

    fn alloc(&mut self, size: usize) -> Result<Self::Buffer, Self::Error> {
        self.allocate_workspace(size as u64)
    }

    fn free(&mut self, buf: Self::Buffer) -> Result<(), Self::Error> {
        self.free_allocation(buf)
    }

    fn copy_host_to_device(
        &mut self,
        src: &[u8],
        dst: &mut Self::Buffer,
    ) -> Result<(), Self::Error> {
        // Stub: submit a DMA transfer using placeholder addresses.
        let _ = self.transfer_to_device(0, *dst, src.len() as u64)?;
        Ok(())
    }

    fn copy_device_to_host(&self, _src: &Self::Buffer, _dst: &mut [u8]) -> Result<(), Self::Error> {
        // Stub: device-to-host requires mutable DMA engine; return Ok
        // since this is an architectural stub.
        Ok(())
    }

    fn load_kernel(&mut self, name: &str, _source: &[u8]) -> Result<Self::Kernel, Self::Error> {
        // Check if the operator is supported by looking up the mapping.
        if Self::map_operator(name).is_some() {
            Ok(0) // Stub kernel handle
        } else {
            Err(GpuError::NotFound)
        }
    }

    fn launch(
        &mut self,
        _kernel: &Self::Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        _args: &[&Self::Buffer],
    ) -> Result<(), Self::Error> {
        let config = LaunchConfig {
            grid: Dim3 {
                x: grid[0],
                y: grid[1],
                z: grid[2],
            },
            block: Dim3 {
                x: block[0],
                y: block[1],
                z: block[2],
            },
            shared_memory: 0,
            stream: 0,
        };
        let kid = self.compute.launch("compute_provider", config)?;
        self.compute.dispatch(kid)?;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        self.compute.synchronize();
        Ok(())
    }

    fn supports_op(&self, op: &str) -> bool {
        Self::map_operator(op).is_some()
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

    // -- helpers ------------------------------------------------------------

    /// Construct a GpuInfo for Tesla T4 (Turing, SM 7.5).
    fn test_gpu_info() -> GpuInfo {
        identify_gpu(0x1B80).unwrap()
    }

    /// Build a CudaProvider backed by 1 GiB of VRAM.
    fn test_provider() -> CudaProvider {
        let gpu = test_gpu_info();
        CudaProvider::new(gpu, 1024 * 1024 * 1024) // 1 GiB
    }

    const PAGE: u64 = GPU_PAGE_SIZE as u64;

    // -- 1. New provider is Ready -------------------------------------------

    #[test]
    fn test_new_provider_is_ready() {
        let prov = test_provider();
        assert_eq!(*prov.status(), ProviderStatus::Ready);
    }

    // -- 2. Load weights allocates in static region -------------------------

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

    // -- 3. Allocate workspace in dynamic region ----------------------------

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

    // -- 4. Free allocation -------------------------------------------------

    #[test]
    fn test_free_allocation() {
        let mut prov = test_provider();
        let id = prov.load_weights(PAGE).unwrap();
        let used_before_free = prov.vram_used();
        prov.free_allocation(id).unwrap();
        assert_eq!(prov.vram_used(), used_before_free - PAGE);
    }

    // -- 5. Transfer to device ----------------------------------------------

    #[test]
    fn test_transfer_to_device() {
        let mut prov = test_provider();
        let tid = prov.transfer_to_device(0x1000, 0x2000, 4096).unwrap();
        assert!(tid >= 1);
    }

    // -- 6. Transfer from device --------------------------------------------

    #[test]
    fn test_transfer_from_device() {
        let mut prov = test_provider();
        let tid = prov.transfer_from_device(0x2000, 0x1000, 8192).unwrap();
        assert!(tid >= 1);
    }

    // -- 7. Map known operators: MatMul, Conv, Relu, Softmax ----------------

    #[test]
    fn test_map_known_operators() {
        let cases = [
            ("MatMul", PtxKernelType::Gemm),
            ("Gemm", PtxKernelType::Gemm),
            ("Conv", PtxKernelType::Conv2d),
            ("Relu", PtxKernelType::Elementwise),
            ("Sigmoid", PtxKernelType::Elementwise),
            ("Tanh", PtxKernelType::Elementwise),
            ("Add", PtxKernelType::Elementwise),
            ("Mul", PtxKernelType::Elementwise),
            ("Sub", PtxKernelType::Elementwise),
            ("Softmax", PtxKernelType::Softmax),
            ("LayerNormalization", PtxKernelType::LayerNorm),
            ("ReduceSum", PtxKernelType::Reduce),
            ("ReduceMean", PtxKernelType::Reduce),
            ("ReduceMax", PtxKernelType::Reduce),
            ("Transpose", PtxKernelType::Transpose),
            ("Concat", PtxKernelType::Concat),
            ("MaxPool", PtxKernelType::Pool),
            ("AveragePool", PtxKernelType::Pool),
            ("GlobalAveragePool", PtxKernelType::Pool),
        ];
        for (op, expected_type) in &cases {
            let mapping = CudaProvider::map_operator(op);
            assert!(mapping.is_some(), "Expected mapping for {}", op);
            let m = mapping.unwrap();
            assert_eq!(m.kernel_type, *expected_type, "Wrong type for {}", op);
            assert_eq!(
                m.precision,
                DataPrecision::F32,
                "Default precision should be F32"
            );
            assert_eq!(m.op_name, *op);
        }
    }

    // -- 8. Map unknown operator returns None -------------------------------

    #[test]
    fn test_map_unknown_operator_returns_none() {
        assert!(CudaProvider::map_operator("UnknownOp").is_none());
        assert!(CudaProvider::map_operator("").is_none());
        assert!(CudaProvider::map_operator("CustomAttention").is_none());
    }

    // -- 9. Map reshape returns None (CPU-only) -----------------------------

    #[test]
    fn test_map_reshape_returns_none() {
        assert!(CudaProvider::map_operator("Reshape").is_none());
        assert!(CudaProvider::map_operator("Flatten").is_none());
        assert!(CudaProvider::map_operator("Squeeze").is_none());
        assert!(CudaProvider::map_operator("Unsqueeze").is_none());
    }

    // -- 10. Launch MatMul kernel -------------------------------------------

    #[test]
    fn test_launch_matmul_kernel() {
        let mut prov = test_provider();
        let launch_id = prov.launch_kernel("MatMul", 1024).unwrap();
        assert!(launch_id >= 1);
    }

    // -- 11. Launch Elementwise kernel --------------------------------------

    #[test]
    fn test_launch_elementwise_kernel() {
        let mut prov = test_provider();
        let launch_id = prov.launch_kernel("Relu", 512).unwrap();
        assert!(launch_id >= 1);
    }

    // -- 12. Synchronize ----------------------------------------------------

    #[test]
    fn test_synchronize() {
        let mut prov = test_provider();
        let kid_raw = prov.launch_kernel("Add", 256).unwrap();
        // Before sync, the kernel is Running.
        let kid = crate::compute::KernelId(kid_raw);
        assert_eq!(prov.compute.status(kid), Ok(&KernelStatus::Running));
        prov.synchronize();
        // After sync, it should be Completed.
        assert_eq!(prov.compute.status(kid), Ok(&KernelStatus::Completed));
    }

    // -- 13. VRAM tracking --------------------------------------------------

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

    // -- 14. Provider GPU info access ---------------------------------------

    #[test]
    fn test_provider_gpu_info_access() {
        let prov = test_provider();
        let info = prov.gpu_info();
        assert_eq!(info.device_id, 0x1B80);
        assert_eq!(info.name, "NVIDIA Tesla T4");
        assert_eq!(info.compute_capability.as_sm_version(), 75);
    }

    // -- 15. Launch config calculation (grid/block sizing) ------------------

    #[test]
    fn test_launch_config_grid_block_sizing() {
        // We test that the grid is ceil(elements / 256).
        // For 1000 elements: ceil(1000/256) = 4.
        // For 256 elements: ceil(256/256) = 1.
        // For 1 element: ceil(1/256) = 1.
        // For 512 elements: ceil(512/256) = 2.

        // Verify by launching and checking the compute engine's kernel.
        let mut prov = test_provider();

        // 1000 elements => grid should be 4.
        let kid_raw = prov.launch_kernel("MatMul", 1000).unwrap();
        let kid = crate::compute::KernelId(kid_raw);
        let kernel = prov.compute.kernels.iter().find(|k| k.id == kid).unwrap();
        assert_eq!(kernel.config.grid.x, 4);
        assert_eq!(kernel.config.block.x, 256);

        // 256 elements => grid should be 1.
        let kid_raw2 = prov.launch_kernel("Relu", 256).unwrap();
        let kid2 = crate::compute::KernelId(kid_raw2);
        let kernel2 = prov.compute.kernels.iter().find(|k| k.id == kid2).unwrap();
        assert_eq!(kernel2.config.grid.x, 1);
        assert_eq!(kernel2.config.block.x, 256);
    }

    // -- 16. Provider status enum -------------------------------------------

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

    // -- 17. Models loaded counter ------------------------------------------

    #[test]
    fn test_models_loaded_counter() {
        let mut prov = test_provider();
        assert_eq!(prov.models_loaded(), 0);
        prov.load_weights(PAGE).unwrap();
        assert_eq!(prov.models_loaded(), 1);
        prov.load_weights(PAGE).unwrap();
        assert_eq!(prov.models_loaded(), 2);
    }

    // -- 18. Launch unknown op returns NotFound -----------------------------

    #[test]
    fn test_launch_unknown_op_returns_not_found() {
        let mut prov = test_provider();
        let result = prov.launch_kernel("UnknownOp", 100);
        assert_eq!(result, Err(GpuError::NotFound));
    }

    // -- 19. Launch CPU-only op returns NotFound ----------------------------

    #[test]
    fn test_launch_cpu_only_op_returns_not_found() {
        let mut prov = test_provider();
        let result = prov.launch_kernel("Reshape", 100);
        assert_eq!(result, Err(GpuError::NotFound));
    }

    // -- 20. ExecutionPlan and ExecutionStep derive traits -------------------

    #[test]
    fn test_execution_plan_derives() {
        let step = ExecutionStep {
            op_name: String::from("MatMul"),
            kernel_type: PtxKernelType::Gemm,
            input_allocs: alloc::vec![1, 2],
            output_alloc: 3,
            config: LaunchConfig {
                grid: Dim3::linear(4),
                block: Dim3::linear(256),
                shared_memory: 49152,
                stream: 0,
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

    // -- 21. Multiple launches and sync -------------------------------------

    #[test]
    fn test_multiple_launches_and_sync() {
        let mut prov = test_provider();
        let id1 = prov.launch_kernel("MatMul", 1024).unwrap();
        let id2 = prov.launch_kernel("Relu", 2048).unwrap();
        let id3 = prov.launch_kernel("Softmax", 512).unwrap();
        assert!(id1 < id2);
        assert!(id2 < id3);
        prov.synchronize();
        // All should be completed after sync.
        for kid_raw in [id1, id2, id3] {
            let kid = crate::compute::KernelId(kid_raw);
            assert_eq!(prov.compute.status(kid), Ok(&KernelStatus::Completed));
        }
    }

    // -- 22. new_tegra() convenience constructor (tegra feature) ------------

    #[cfg(feature = "tegra")]
    #[test]
    fn test_new_tegra_provider() {
        let prov = CudaProvider::new_tegra();
        assert_eq!(*prov.status(), ProviderStatus::Ready);

        let info = prov.gpu_info();
        assert_eq!(info.device_id, 0x12B1);
        assert_eq!(info.name, "NVIDIA GM20B (Tegra X1)");
        assert_eq!(info.compute_capability.as_sm_version(), 53);
        assert_eq!(info.sm_count, 2);
        assert_eq!(info.vram_size_mb, 0); // shared DRAM, no discrete VRAM

        // 256 MiB DRAM budget
        let total = prov.vram_used() + prov.vram_free();
        assert_eq!(total, 256 * 1024 * 1024);
    }

    // -- 23. Shared memory passed through from PTX kernel -------------------

    #[test]
    fn test_shared_memory_passed_through() {
        let mut prov = test_provider();
        // Launch a GEMM kernel — should use gemm_f32 with 49152 bytes shared.
        let kid_raw = prov.launch_kernel("MatMul", 256).unwrap();
        let kid = crate::compute::KernelId(kid_raw);
        let kernel = prov.compute.kernels.iter().find(|k| k.id == kid).unwrap();
        assert_eq!(kernel.config.shared_memory, 49152);

        // Launch an Elementwise — should use 0 bytes shared.
        let kid_raw2 = prov.launch_kernel("Relu", 256).unwrap();
        let kid2 = crate::compute::KernelId(kid_raw2);
        let kernel2 = prov.compute.kernels.iter().find(|k| k.id == kid2).unwrap();
        assert_eq!(kernel2.config.shared_memory, 0);
    }

    // -- 24. Tegra allocation round-trip (tegra feature) --------------------

    #[cfg(feature = "tegra")]
    #[test]
    fn test_tegra_alloc_free_round_trip() {
        let mut prov = CudaProvider::new_tegra();
        assert_eq!(*prov.status(), ProviderStatus::Ready);

        let baseline = prov.vram_used();
        let one_mb: u64 = 1024 * 1024;

        // Allocate 1 MiB of workspace.
        let id = prov.allocate_workspace(one_mb).unwrap();
        assert!(id >= 1);
        assert_eq!(prov.vram_used(), baseline + one_mb);

        // Free the allocation.
        prov.free_allocation(id).unwrap();
        assert_eq!(prov.vram_used(), baseline);
    }

    // -- 25. ComputeProvider trait: device_info ----------------------------

    #[test]
    fn test_compute_provider_device_info() {
        use smallaios_compute::ComputeProvider;
        let prov = test_provider();
        let info = ComputeProvider::device_info(&prov);
        assert_eq!(info.name, "NVIDIA Tesla T4");
        assert_eq!(info.backend_type, smallaios_compute::BackendType::Cuda);
        assert!(info.compute_units > 0);
    }

    // -- 26. ComputeProvider trait: init -----------------------------------

    #[test]
    fn test_compute_provider_init() {
        use smallaios_compute::ComputeProvider;
        let mut prov = test_provider();
        assert!(ComputeProvider::init(&mut prov).is_ok());
    }

    // -- 27. ComputeProvider trait: alloc / free ---------------------------

    #[test]
    fn test_compute_provider_alloc_free() {
        use smallaios_compute::ComputeProvider;
        let mut prov = test_provider();
        let buf = ComputeProvider::alloc(&mut prov, PAGE as usize).unwrap();
        assert!(ComputeProvider::free(&mut prov, buf).is_ok());
    }

    // -- 28. ComputeProvider trait: supports_op ---------------------------

    #[test]
    fn test_compute_provider_supports_op() {
        use smallaios_compute::ComputeProvider;
        let prov = test_provider();
        assert!(prov.supports_op("MatMul"));
        assert!(prov.supports_op("Relu"));
        assert!(prov.supports_op("Conv"));
        assert!(!prov.supports_op("Reshape"));
        assert!(!prov.supports_op("UnknownOp"));
    }

    // -- 29. ComputeProvider trait: launch / synchronize ------------------

    #[test]
    fn test_compute_provider_launch_sync() {
        use smallaios_compute::ComputeProvider;
        let mut prov = test_provider();
        let kernel = prov.load_kernel("MatMul", &[]).unwrap();
        assert!(prov.launch(&kernel, [1, 1, 1], [256, 1, 1], &[]).is_ok());
        assert!(ComputeProvider::synchronize(&mut prov).is_ok());
    }

    // -- 30. ComputeProvider trait: load_kernel unknown returns err -------

    #[test]
    fn test_compute_provider_load_kernel_unknown() {
        use smallaios_compute::ComputeProvider;
        let mut prov = test_provider();
        assert!(prov.load_kernel("UnknownOp", &[]).is_err());
    }
}
