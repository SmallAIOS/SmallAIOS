// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Level Zero execution provider for ONNX inference.
//!
//! [`LevelZeroProvider`] is the top-level entry point that wires together the
//! local memory allocator, DMA engine, compute engine, and SPIR-V kernel
//! registry to execute ONNX operators on an Intel GPU. It translates
//! high-level ONNX operator names into SPIR-V kernel dispatches with
//! appropriate workgroup sizing.
//!
//! Named after Intel's Level Zero API (part of oneAPI), which provides
//! low-level GPU access similar to CUDA for NVIDIA.

use alloc::string::String;
use alloc::vec::Vec;

use crate::compute::{ComputeEngine, DispatchConfig, WorkgroupSize};
use crate::dma::{DmaDirection, DmaEngine};
use crate::gpu_id::GpuInfo;
use crate::memory::{LocalMemoryAllocator, MemoryRegion};
use crate::spirv::{DataPrecision, SpirvKernelType, SpirvRegistry};
use crate::GpuError;

// ---------------------------------------------------------------------------
// OperatorMapping
// ---------------------------------------------------------------------------

/// Maps an ONNX operator name to a GPU kernel type and precision.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorMapping {
    /// ONNX operator name (e.g. `"MatMul"`, `"Conv"`).
    pub op_name: String,
    /// SPIR-V kernel family to use.
    pub kernel_type: SpirvKernelType,
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
    /// SPIR-V kernel type to dispatch.
    pub kernel_type: SpirvKernelType,
    /// Memory allocation ids for input tensors.
    pub input_allocs: Vec<u64>,
    /// Memory allocation id for the output tensor.
    pub output_alloc: u64,
    /// Dispatch configuration for this step.
    pub config: DispatchConfig,
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

/// Lifecycle state of the Level Zero execution provider.
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
// LevelZeroProvider
// ---------------------------------------------------------------------------

/// Level Zero execution provider that dispatches ONNX operators to Intel GPU
/// kernels.
///
/// Owns all GPU-side resources: memory allocator, compute and DMA engines,
/// and the SPIR-V kernel registry.
pub struct LevelZeroProvider {
    /// Static description of the GPU.
    gpu_info: GpuInfo,
    /// Memory allocator (70% static / 30% dynamic).
    allocator: LocalMemoryAllocator,
    /// Kernel dispatch engine.
    compute: ComputeEngine,
    /// Host<->device DMA engine (blitter).
    dma: DmaEngine,
    /// Registered SPIR-V kernels.
    registry: SpirvRegistry,
    /// Current provider state.
    status: ProviderStatus,
    /// Number of models whose weights have been loaded.
    models_loaded: u32,
}

impl LevelZeroProvider {
    /// Create a new Level Zero provider for the given GPU.
    ///
    /// `memory_size` is the total usable memory in bytes. The allocator is
    /// configured with 70% static (weights) / 30% dynamic (workspace).
    /// Default SPIR-V kernels are registered and the provider moves to
    /// [`ProviderStatus::Ready`].
    pub fn new(gpu_info: GpuInfo, memory_size: u64) -> Self {
        let allocator = LocalMemoryAllocator::new(memory_size, 0.7);
        let compute = ComputeEngine::new();
        let dma = DmaEngine::new();
        let mut registry = SpirvRegistry::new();
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

    /// Load model weights into memory (static region).
    pub fn load_weights(&mut self, size: u64) -> Result<u64, GpuError> {
        let id = self.allocator.alloc(size, MemoryRegion::Static)?;
        self.models_loaded += 1;
        Ok(id)
    }

    /// Allocate workspace memory (dynamic region) for intermediate tensors.
    pub fn allocate_workspace(&mut self, size: u64) -> Result<u64, GpuError> {
        self.allocator.alloc(size, MemoryRegion::Dynamic)
    }

    /// Free a previous memory allocation (static or dynamic).
    pub fn free_allocation(&mut self, id: u64) -> Result<(), GpuError> {
        self.allocator.free(id)
    }

    /// Submit a host-to-device DMA transfer.
    pub fn transfer_to_device(&mut self, src: u64, dst: u64, size: u64) -> Result<u64, GpuError> {
        let tid = self
            .dma
            .submit(DmaDirection::HostToDevice, src, dst, size)?;
        Ok(tid.0)
    }

    /// Submit a device-to-host DMA transfer.
    pub fn transfer_from_device(&mut self, src: u64, dst: u64, size: u64) -> Result<u64, GpuError> {
        let tid = self
            .dma
            .submit(DmaDirection::DeviceToHost, src, dst, size)?;
        Ok(tid.0)
    }

    /// Map an ONNX operator name to a GPU kernel type and precision.
    pub fn map_operator(op_name: &str) -> Option<OperatorMapping> {
        let kernel_type = match op_name {
            "MatMul" | "Gemm" => SpirvKernelType::Gemm,
            "Conv" => SpirvKernelType::Conv2d,
            "Relu" | "Sigmoid" | "Tanh" | "Add" | "Mul" | "Sub" => SpirvKernelType::Elementwise,
            "Softmax" => SpirvKernelType::Softmax,
            "LayerNormalization" => SpirvKernelType::LayerNorm,
            "ReduceSum" | "ReduceMean" | "ReduceMax" => SpirvKernelType::Reduce,
            "Transpose" => SpirvKernelType::Transpose,
            "Concat" => SpirvKernelType::Concat,
            "MaxPool" | "AveragePool" | "GlobalAveragePool" => SpirvKernelType::Pool,
            "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" => return None,
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
    /// Looks up the operator mapping, finds the matching SPIR-V kernel, builds
    /// a dispatch configuration with workgroup size 256, and dispatches to the
    /// compute engine. Returns the kernel launch id.
    pub fn launch_kernel(&mut self, op_name: &str, elements: u32) -> Result<u64, GpuError> {
        let mapping = Self::map_operator(op_name).ok_or(GpuError::NotFound)?;

        let spirv_kernel = self
            .registry
            .find_kernel(
                mapping.kernel_type,
                mapping.precision,
                &self.gpu_info.generation,
            )
            .ok_or(GpuError::LaunchError)?;

        let workgroup_size: u32 = 256;
        let grid_size = elements.div_ceil(workgroup_size);

        let config = DispatchConfig {
            grid: WorkgroupSize::linear(grid_size),
            workgroup: WorkgroupSize::linear(workgroup_size),
            shared_memory: spirv_kernel.shared_memory,
            queue: 0,
        };

        let kid = self.compute.launch(op_name, config)?;
        self.compute.dispatch(kid)?;
        Ok(kid.0)
    }

    /// Synchronise the compute engine.
    pub fn synchronize(&mut self) {
        self.compute.synchronize();
    }

    /// Total memory currently in use (bytes).
    pub fn memory_used(&self) -> u64 {
        self.allocator.total_used()
    }

    /// Total memory available for new allocations (bytes).
    pub fn memory_free(&self) -> u64 {
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

impl smallaios_compute::ComputeProvider for LevelZeroProvider {
    type Buffer = u64; // Memory allocation ID
    type Kernel = u64; // Stub kernel handle
    type Error = GpuError;

    fn device_info(&self) -> smallaios_compute::DeviceInfo {
        smallaios_compute::DeviceInfo {
            name: String::from(self.gpu_info.name),
            memory_bytes: self.gpu_info.local_memory_mb as u64 * 1024 * 1024,
            compute_units: self.gpu_info.eu_count,
            backend_type: smallaios_compute::BackendType::LevelZero,
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
        let _ = self.transfer_to_device(0, *dst, src.len() as u64)?;
        Ok(())
    }

    fn copy_device_to_host(&self, _src: &Self::Buffer, _dst: &mut [u8]) -> Result<(), Self::Error> {
        Ok(())
    }

    fn load_kernel(&mut self, name: &str, _source: &[u8]) -> Result<Self::Kernel, Self::Error> {
        if Self::map_operator(name).is_some() {
            Ok(0)
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
        let config = DispatchConfig {
            grid: WorkgroupSize {
                x: grid[0],
                y: grid[1],
                z: grid[2],
            },
            workgroup: WorkgroupSize {
                x: block[0],
                y: block[1],
                z: block[2],
            },
            shared_memory: 0,
            queue: 0,
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

    fn test_gpu_info() -> GpuInfo {
        identify_gpu(0x56A0).unwrap()
    }

    fn test_provider() -> LevelZeroProvider {
        let gpu = test_gpu_info();
        LevelZeroProvider::new(gpu, 1024 * 1024 * 1024)
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
        let used_before_free = prov.memory_used();
        prov.free_allocation(id).unwrap();
        assert_eq!(prov.memory_used(), used_before_free - PAGE);
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
            ("MatMul", SpirvKernelType::Gemm),
            ("Gemm", SpirvKernelType::Gemm),
            ("Conv", SpirvKernelType::Conv2d),
            ("Relu", SpirvKernelType::Elementwise),
            ("Sigmoid", SpirvKernelType::Elementwise),
            ("Tanh", SpirvKernelType::Elementwise),
            ("Add", SpirvKernelType::Elementwise),
            ("Mul", SpirvKernelType::Elementwise),
            ("Sub", SpirvKernelType::Elementwise),
            ("Softmax", SpirvKernelType::Softmax),
            ("LayerNormalization", SpirvKernelType::LayerNorm),
            ("ReduceSum", SpirvKernelType::Reduce),
            ("ReduceMean", SpirvKernelType::Reduce),
            ("ReduceMax", SpirvKernelType::Reduce),
            ("Transpose", SpirvKernelType::Transpose),
            ("Concat", SpirvKernelType::Concat),
            ("MaxPool", SpirvKernelType::Pool),
            ("AveragePool", SpirvKernelType::Pool),
            ("GlobalAveragePool", SpirvKernelType::Pool),
        ];
        for (op, expected_type) in &cases {
            let mapping = LevelZeroProvider::map_operator(op);
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

    #[test]
    fn test_map_unknown_operator_returns_none() {
        assert!(LevelZeroProvider::map_operator("UnknownOp").is_none());
        assert!(LevelZeroProvider::map_operator("").is_none());
        assert!(LevelZeroProvider::map_operator("CustomAttention").is_none());
    }

    #[test]
    fn test_map_reshape_returns_none() {
        assert!(LevelZeroProvider::map_operator("Reshape").is_none());
        assert!(LevelZeroProvider::map_operator("Flatten").is_none());
        assert!(LevelZeroProvider::map_operator("Squeeze").is_none());
        assert!(LevelZeroProvider::map_operator("Unsqueeze").is_none());
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
    fn test_memory_tracking() {
        let mut prov = test_provider();
        let total = prov.memory_used() + prov.memory_free();
        assert_eq!(total, 1024 * 1024 * 1024);

        let initial_used = prov.memory_used();
        prov.load_weights(PAGE).unwrap();
        assert_eq!(prov.memory_used(), initial_used + PAGE);
        assert_eq!(prov.memory_free(), total - prov.memory_used());
    }

    #[test]
    fn test_provider_gpu_info_access() {
        let prov = test_provider();
        let info = prov.gpu_info();
        assert_eq!(info.device_id, 0x56A0);
        assert_eq!(info.name, "Intel Arc A770");
        assert_eq!(info.eu_count, 512);
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
            kernel_type: SpirvKernelType::Gemm,
            input_allocs: alloc::vec![1, 2],
            output_alloc: 3,
            config: DispatchConfig {
                grid: WorkgroupSize::linear(4),
                workgroup: WorkgroupSize::linear(256),
                shared_memory: 49152,
                queue: 0,
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
    fn test_shared_memory_passed_through() {
        let mut prov = test_provider();
        let kid_raw = prov.launch_kernel("MatMul", 256).unwrap();
        let kid = crate::compute::KernelId(kid_raw);
        let kernel = prov.compute.kernels.iter().find(|k| k.id == kid).unwrap();
        assert_eq!(kernel.config.shared_memory, 49152);

        let kid_raw2 = prov.launch_kernel("Relu", 256).unwrap();
        let kid2 = crate::compute::KernelId(kid_raw2);
        let kernel2 = prov.compute.kernels.iter().find(|k| k.id == kid2).unwrap();
        assert_eq!(kernel2.config.shared_memory, 0);
    }

    // -- ComputeProvider trait tests --

    #[test]
    fn test_compute_provider_device_info() {
        use smallaios_compute::ComputeProvider;
        let prov = test_provider();
        let info = ComputeProvider::device_info(&prov);
        assert_eq!(info.name, "Intel Arc A770");
        assert_eq!(info.backend_type, smallaios_compute::BackendType::LevelZero);
        assert!(info.compute_units > 0);
    }

    #[test]
    fn test_compute_provider_init() {
        use smallaios_compute::ComputeProvider;
        let mut prov = test_provider();
        assert!(ComputeProvider::init(&mut prov).is_ok());
    }

    #[test]
    fn test_compute_provider_alloc_free() {
        use smallaios_compute::ComputeProvider;
        let mut prov = test_provider();
        let buf = ComputeProvider::alloc(&mut prov, PAGE as usize).unwrap();
        assert!(ComputeProvider::free(&mut prov, buf).is_ok());
    }

    #[test]
    fn test_compute_provider_supports_op() {
        use smallaios_compute::ComputeProvider;
        let prov = test_provider();
        assert!(prov.supports_op("MatMul"));
        assert!(prov.supports_op("Relu"));
        assert!(!prov.supports_op("Reshape"));
        assert!(!prov.supports_op("UnknownOp"));
    }

    #[test]
    fn test_compute_provider_launch_sync() {
        use smallaios_compute::ComputeProvider;
        let mut prov = test_provider();
        let kernel = prov.load_kernel("MatMul", &[]).unwrap();
        assert!(prov.launch(&kernel, [1, 1, 1], [256, 1, 1], &[]).is_ok());
        assert!(ComputeProvider::synchronize(&mut prov).is_ok());
    }

    #[test]
    fn test_compute_provider_load_kernel_unknown() {
        use smallaios_compute::ComputeProvider;
        let mut prov = test_provider();
        assert!(prov.load_kernel("UnknownOp", &[]).is_err());
    }
}
