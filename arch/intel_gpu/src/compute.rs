// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Compute engine -- EU-based workgroup dispatch, synchronization.
//!
//! Provides a stub compute engine that manages kernel lifecycle on Intel
//! GPU Execution Units (EUs). Kernels follow the state machine:
//! `Queued -> Running -> Completed | Failed`.
//!
//! Intel GPUs dispatch work as workgroups to subslices containing EUs,
//! using SIMD threads rather than CUDA warps.

use alloc::string::String;
use alloc::vec::Vec;

use crate::GpuError;

/// Three-dimensional size used for workgroup and dispatch dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupSize {
    /// Create a new `WorkgroupSize` with explicit x/y/z components.
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Convenience for one-dimensional dispatches: `(n, 1, 1)`.
    pub fn linear(n: u32) -> Self {
        Self { x: n, y: 1, z: 1 }
    }

    /// Total number of items represented.
    pub fn total(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

/// Configuration for a compute dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchConfig {
    /// Number of workgroups in the dispatch grid.
    pub grid: WorkgroupSize,
    /// Number of work-items (threads) per workgroup.
    pub workgroup: WorkgroupSize,
    /// Bytes of shared local memory (SLM) per workgroup.
    pub shared_memory: u32,
    /// Command queue ordinal.
    pub queue: u32,
}

/// Maximum work-items per workgroup (Intel Xe hardware limit).
const MAX_WORKGROUP_SIZE: u64 = 1024;

/// Maximum shared local memory per workgroup (128 KiB for Xe-HPC).
const MAX_SHARED_MEMORY: u32 = 131072;

impl DispatchConfig {
    /// Validate that this configuration is within hardware limits.
    pub fn validate(&self) -> Result<(), GpuError> {
        if self.workgroup.total() > MAX_WORKGROUP_SIZE {
            return Err(GpuError::InvalidConfig);
        }
        if self.grid.x == 0 || self.grid.y == 0 || self.grid.z == 0 {
            return Err(GpuError::InvalidConfig);
        }
        if self.workgroup.x == 0 || self.workgroup.y == 0 || self.workgroup.z == 0 {
            return Err(GpuError::InvalidConfig);
        }
        if self.shared_memory > MAX_SHARED_MEMORY {
            return Err(GpuError::InvalidConfig);
        }
        Ok(())
    }
}

/// Unique identifier for a submitted kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelId(pub u64);

/// Lifecycle state of a kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum KernelStatus {
    /// Submitted and waiting in the queue.
    Queued,
    /// Currently executing on the GPU.
    Running,
    /// Finished successfully.
    Completed,
    /// Terminated with an error.
    Failed,
}

/// A single kernel tracked by the compute engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    /// Unique kernel identifier.
    pub id: KernelId,
    /// Human-readable kernel name (e.g. ONNX op name).
    pub name: String,
    /// Dispatch configuration.
    pub config: DispatchConfig,
    /// Current status.
    pub status: KernelStatus,
}

/// Maximum number of kernels that may be queued at once.
pub const MAX_QUEUED_KERNELS: usize = 512;

/// Intel GPU compute engine (render/compute command streamer).
#[derive(Clone, Debug, PartialEq)]
pub struct ComputeEngine {
    /// All tracked kernels.
    pub(crate) kernels: Vec<Kernel>,
    /// Monotonically increasing id generator.
    next_id: u64,
    /// Running count of successfully completed kernels.
    completed_count: u64,
    /// Running count of failed kernels.
    failed_count: u64,
}

impl Default for ComputeEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeEngine {
    /// Create a new, empty compute engine.
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            next_id: 1,
            completed_count: 0,
            failed_count: 0,
        }
    }

    /// Submit a kernel for execution.
    ///
    /// The dispatch configuration is validated first. On success, the kernel
    /// is placed in [`KernelStatus::Queued`] and its [`KernelId`] is returned.
    pub fn launch(&mut self, name: &str, config: DispatchConfig) -> Result<KernelId, GpuError> {
        config.validate()?;

        if self.kernels.len() >= MAX_QUEUED_KERNELS {
            return Err(GpuError::QueueFull);
        }

        let id = KernelId(self.next_id);
        self.next_id += 1;

        self.kernels.push(Kernel {
            id,
            name: String::from(name),
            config,
            status: KernelStatus::Queued,
        });

        Ok(id)
    }

    /// Dispatch a queued kernel to the GPU (Queued -> Running).
    pub fn dispatch(&mut self, id: KernelId) -> Result<(), GpuError> {
        let kernel = self.find_mut(id)?;
        if kernel.status != KernelStatus::Queued {
            return Err(GpuError::InvalidState);
        }
        kernel.status = KernelStatus::Running;
        Ok(())
    }

    /// Mark a running kernel as completed (Running -> Completed).
    pub fn complete(&mut self, id: KernelId) -> Result<(), GpuError> {
        let kernel = self.find_mut(id)?;
        if kernel.status != KernelStatus::Running {
            return Err(GpuError::InvalidState);
        }
        kernel.status = KernelStatus::Completed;
        self.completed_count += 1;
        Ok(())
    }

    /// Mark a running kernel as failed (Running -> Failed).
    pub fn fail(&mut self, id: KernelId) -> Result<(), GpuError> {
        let kernel = self.find_mut(id)?;
        if kernel.status != KernelStatus::Running {
            return Err(GpuError::InvalidState);
        }
        kernel.status = KernelStatus::Failed;
        self.failed_count += 1;
        Ok(())
    }

    /// Query the current status of a kernel.
    pub fn status(&self, id: KernelId) -> Result<&KernelStatus, GpuError> {
        let kernel = self
            .kernels
            .iter()
            .find(|k| k.id == id)
            .ok_or(GpuError::NotFound)?;
        Ok(&kernel.status)
    }

    /// Device-wide synchronization barrier.
    ///
    /// All `Running` kernels are moved to `Completed`.
    pub fn synchronize(&mut self) {
        for kernel in &mut self.kernels {
            if kernel.status == KernelStatus::Running {
                kernel.status = KernelStatus::Completed;
                self.completed_count += 1;
            }
        }
    }

    /// Total number of kernels that have completed successfully.
    pub fn completed_count(&self) -> u64 {
        self.completed_count
    }

    /// Number of kernels currently in Queued state.
    pub fn queued_count(&self) -> usize {
        self.kernels
            .iter()
            .filter(|k| k.status == KernelStatus::Queued)
            .count()
    }

    fn find_mut(&mut self, id: KernelId) -> Result<&mut Kernel, GpuError> {
        self.kernels
            .iter_mut()
            .find(|k| k.id == id)
            .ok_or(GpuError::NotFound)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> DispatchConfig {
        DispatchConfig {
            grid: WorkgroupSize::linear(128),
            workgroup: WorkgroupSize::linear(256),
            shared_memory: 0,
            queue: 0,
        }
    }

    #[test]
    fn test_workgroup_size_creation_and_total() {
        let d = WorkgroupSize::new(4, 8, 2);
        assert_eq!(d.x, 4);
        assert_eq!(d.y, 8);
        assert_eq!(d.z, 2);
        assert_eq!(d.total(), 64);
    }

    #[test]
    fn test_workgroup_size_linear() {
        let d = WorkgroupSize::linear(512);
        assert_eq!(d.x, 512);
        assert_eq!(d.y, 1);
        assert_eq!(d.z, 1);
        assert_eq!(d.total(), 512);
    }

    #[test]
    fn test_dispatch_config_valid() {
        assert!(valid_config().validate().is_ok());
    }

    #[test]
    fn test_dispatch_config_workgroup_too_large() {
        let cfg = DispatchConfig {
            grid: WorkgroupSize::linear(1),
            workgroup: WorkgroupSize::new(32, 32, 2), // 2048 > 1024
            shared_memory: 0,
            queue: 0,
        };
        assert_eq!(cfg.validate(), Err(GpuError::InvalidConfig));
    }

    #[test]
    fn test_dispatch_config_zero_grid() {
        let cfg = DispatchConfig {
            grid: WorkgroupSize::new(0, 1, 1),
            workgroup: WorkgroupSize::linear(256),
            shared_memory: 0,
            queue: 0,
        };
        assert_eq!(cfg.validate(), Err(GpuError::InvalidConfig));
    }

    #[test]
    fn test_dispatch_config_shared_memory_limit() {
        let cfg = DispatchConfig {
            grid: WorkgroupSize::linear(1),
            workgroup: WorkgroupSize::linear(256),
            shared_memory: MAX_SHARED_MEMORY + 1,
            queue: 0,
        };
        assert_eq!(cfg.validate(), Err(GpuError::InvalidConfig));
    }

    #[test]
    fn test_launch_creates_queued() {
        let mut eng = ComputeEngine::new();
        let id = eng.launch("matmul", valid_config()).unwrap();
        assert_eq!(eng.status(id), Ok(&KernelStatus::Queued));
    }

    #[test]
    fn test_dispatch_queued() {
        let mut eng = ComputeEngine::new();
        let id = eng.launch("relu", valid_config()).unwrap();
        eng.dispatch(id).unwrap();
        assert_eq!(eng.status(id), Ok(&KernelStatus::Running));
    }

    #[test]
    fn test_complete_running() {
        let mut eng = ComputeEngine::new();
        let id = eng.launch("conv2d", valid_config()).unwrap();
        eng.dispatch(id).unwrap();
        eng.complete(id).unwrap();
        assert_eq!(eng.status(id), Ok(&KernelStatus::Completed));
    }

    #[test]
    fn test_fail_running() {
        let mut eng = ComputeEngine::new();
        let id = eng.launch("softmax", valid_config()).unwrap();
        eng.dispatch(id).unwrap();
        eng.fail(id).unwrap();
        assert_eq!(eng.status(id), Ok(&KernelStatus::Failed));
    }

    #[test]
    fn test_cannot_dispatch_completed() {
        let mut eng = ComputeEngine::new();
        let id = eng.launch("add", valid_config()).unwrap();
        eng.dispatch(id).unwrap();
        eng.complete(id).unwrap();
        assert_eq!(eng.dispatch(id), Err(GpuError::InvalidState));
    }

    #[test]
    fn test_synchronize() {
        let mut eng = ComputeEngine::new();
        let id1 = eng.launch("k1", valid_config()).unwrap();
        let id2 = eng.launch("k2", valid_config()).unwrap();
        let id3 = eng.launch("k3", valid_config()).unwrap();
        eng.dispatch(id1).unwrap();
        eng.dispatch(id2).unwrap();
        eng.synchronize();
        assert_eq!(eng.status(id1), Ok(&KernelStatus::Completed));
        assert_eq!(eng.status(id2), Ok(&KernelStatus::Completed));
        assert_eq!(eng.status(id3), Ok(&KernelStatus::Queued));
        assert_eq!(eng.completed_count(), 2);
    }

    #[test]
    fn test_status_not_found() {
        let eng = ComputeEngine::new();
        assert_eq!(eng.status(KernelId(999)), Err(GpuError::NotFound));
    }

    #[test]
    fn test_completed_queued_counts() {
        let mut eng = ComputeEngine::new();
        let id1 = eng.launch("a", valid_config()).unwrap();
        let _id2 = eng.launch("b", valid_config()).unwrap();
        assert_eq!(eng.queued_count(), 2);
        eng.dispatch(id1).unwrap();
        eng.complete(id1).unwrap();
        assert_eq!(eng.completed_count(), 1);
        assert_eq!(eng.queued_count(), 1);
    }

    #[test]
    fn test_queue_full() {
        let mut eng = ComputeEngine::new();
        for i in 0..MAX_QUEUED_KERNELS {
            eng.launch(&alloc::format!("k{}", i), valid_config())
                .unwrap();
        }
        assert_eq!(
            eng.launch("overflow", valid_config()),
            Err(GpuError::QueueFull)
        );
    }

    #[test]
    fn test_zero_workgroup_dimension() {
        let cfg = DispatchConfig {
            grid: WorkgroupSize::linear(1),
            workgroup: WorkgroupSize::new(256, 0, 1),
            shared_memory: 0,
            queue: 0,
        };
        assert_eq!(cfg.validate(), Err(GpuError::InvalidConfig));
    }

    #[test]
    fn test_exact_max_workgroup_size() {
        let cfg = DispatchConfig {
            grid: WorkgroupSize::linear(1),
            workgroup: WorkgroupSize::new(32, 32, 1), // 1024 == MAX
            shared_memory: 0,
            queue: 0,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_exact_max_shared_memory() {
        let cfg = DispatchConfig {
            grid: WorkgroupSize::linear(1),
            workgroup: WorkgroupSize::linear(64),
            shared_memory: MAX_SHARED_MEMORY,
            queue: 0,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_failed_count() {
        let mut eng = ComputeEngine::new();
        let id = eng.launch("bad_kernel", valid_config()).unwrap();
        eng.dispatch(id).unwrap();
        eng.fail(id).unwrap();
        assert_eq!(eng.failed_count, 1);
        assert_eq!(eng.completed_count(), 0);
    }

    #[test]
    fn test_cannot_complete_queued() {
        let mut eng = ComputeEngine::new();
        let id = eng.launch("skip", valid_config()).unwrap();
        assert_eq!(eng.complete(id), Err(GpuError::InvalidState));
    }
}
