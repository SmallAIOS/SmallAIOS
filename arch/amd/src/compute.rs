// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Wavefront-based compute engine -- kernel launch, grid/block config, synchronization.
//!
//! AMD GPUs use wavefronts (analogous to NVIDIA warps):
//! - CDNA: 64-wide wavefronts exclusively
//! - RDNA: 32-wide wavefronts (wave32) with wave64 compatibility mode
//!
//! Kernels follow the state machine:
//! `Queued -> Running -> Completed | Failed`.

use alloc::string::String;
use alloc::vec::Vec;

use crate::GpuError;

/// Three-dimensional size used for grid and workgroup dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    /// Create a new `Dim3` with explicit x/y/z components.
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Convenience for one-dimensional launches: `(n, 1, 1)`.
    pub fn linear(n: u32) -> Self {
        Self { x: n, y: 1, z: 1 }
    }

    /// Total number of threads/workgroups represented by this `Dim3`.
    pub fn total(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

/// Wavefront mode for kernel dispatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WavefrontMode {
    /// 32-thread wavefronts (RDNA native, wave32).
    Wave32,
    /// 64-thread wavefronts (CDNA native, GCN compatible).
    Wave64,
}

/// Configuration for a kernel launch.
#[derive(Clone, Debug, PartialEq)]
pub struct LaunchConfig {
    /// Number of workgroups in the grid.
    pub grid: Dim3,
    /// Number of work-items per workgroup.
    pub workgroup: Dim3,
    /// Bytes of LDS (local data share) per workgroup.
    pub lds_size: u32,
    /// Queue/stream ordinal.
    pub queue: u32,
    /// Wavefront mode for this dispatch.
    pub wavefront_mode: WavefrontMode,
}

/// Maximum work-items per workgroup (hardware limit).
const MAX_WORKITEMS_PER_WORKGROUP: u64 = 1024;

/// Maximum LDS per workgroup (128 KiB).
const MAX_LDS_PER_WORKGROUP: u32 = 131072;

impl LaunchConfig {
    /// Validate that this configuration is within hardware limits.
    pub fn validate(&self) -> Result<(), GpuError> {
        if self.workgroup.total() > MAX_WORKITEMS_PER_WORKGROUP {
            return Err(GpuError::InvalidConfig);
        }
        if self.grid.x == 0 || self.grid.y == 0 || self.grid.z == 0 {
            return Err(GpuError::InvalidConfig);
        }
        if self.workgroup.x == 0 || self.workgroup.y == 0 || self.workgroup.z == 0 {
            return Err(GpuError::InvalidConfig);
        }
        if self.lds_size > MAX_LDS_PER_WORKGROUP {
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
    /// Launch configuration.
    pub config: LaunchConfig,
    /// Current status.
    pub status: KernelStatus,
}

/// Maximum number of kernels that may be queued at once.
pub const MAX_QUEUED_KERNELS: usize = 512;

/// Wavefront-based GPU compute engine.
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
    pub fn launch(&mut self, name: &str, config: LaunchConfig) -> Result<KernelId, GpuError> {
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

    /// Number of kernels currently in [`KernelStatus::Queued`] state.
    pub fn queued_count(&self) -> usize {
        self.kernels
            .iter()
            .filter(|k| k.status == KernelStatus::Queued)
            .count()
    }

    // -- internal helpers ---------------------------------------------------

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

    fn valid_config() -> LaunchConfig {
        LaunchConfig {
            grid: Dim3::linear(128),
            workgroup: Dim3::linear(256),
            lds_size: 0,
            queue: 0,
            wavefront_mode: WavefrontMode::Wave64,
        }
    }

    fn valid_wave32_config() -> LaunchConfig {
        LaunchConfig {
            grid: Dim3::linear(128),
            workgroup: Dim3::linear(256),
            lds_size: 0,
            queue: 0,
            wavefront_mode: WavefrontMode::Wave32,
        }
    }

    #[test]
    fn test_dim3_creation_and_total() {
        let d = Dim3::new(4, 8, 2);
        assert_eq!(d.x, 4);
        assert_eq!(d.y, 8);
        assert_eq!(d.z, 2);
        assert_eq!(d.total(), 64);
    }

    #[test]
    fn test_dim3_linear() {
        let d = Dim3::linear(512);
        assert_eq!(d.x, 512);
        assert_eq!(d.y, 1);
        assert_eq!(d.z, 1);
        assert_eq!(d.total(), 512);
    }

    #[test]
    fn test_launch_config_valid() {
        assert!(valid_config().validate().is_ok());
    }

    #[test]
    fn test_launch_config_workgroup_too_large() {
        let cfg = LaunchConfig {
            grid: Dim3::linear(1),
            workgroup: Dim3::new(32, 32, 2), // 2048 > 1024
            lds_size: 0,
            queue: 0,
            wavefront_mode: WavefrontMode::Wave64,
        };
        assert_eq!(cfg.validate(), Err(GpuError::InvalidConfig));
    }

    #[test]
    fn test_launch_config_zero_grid() {
        let cfg = LaunchConfig {
            grid: Dim3::new(0, 1, 1),
            workgroup: Dim3::linear(256),
            lds_size: 0,
            queue: 0,
            wavefront_mode: WavefrontMode::Wave64,
        };
        assert_eq!(cfg.validate(), Err(GpuError::InvalidConfig));
    }

    #[test]
    fn test_launch_config_lds_limit() {
        let cfg = LaunchConfig {
            grid: Dim3::linear(1),
            workgroup: Dim3::linear(256),
            lds_size: MAX_LDS_PER_WORKGROUP + 1,
            queue: 0,
            wavefront_mode: WavefrontMode::Wave64,
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
        let cfg = LaunchConfig {
            grid: Dim3::linear(1),
            workgroup: Dim3::new(256, 0, 1),
            lds_size: 0,
            queue: 0,
            wavefront_mode: WavefrontMode::Wave64,
        };
        assert_eq!(cfg.validate(), Err(GpuError::InvalidConfig));
    }

    #[test]
    fn test_exact_max_workitems() {
        let cfg = LaunchConfig {
            grid: Dim3::linear(1),
            workgroup: Dim3::new(32, 32, 1), // 1024 == MAX
            lds_size: 0,
            queue: 0,
            wavefront_mode: WavefrontMode::Wave64,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_exact_max_lds() {
        let cfg = LaunchConfig {
            grid: Dim3::linear(1),
            workgroup: Dim3::linear(64),
            lds_size: MAX_LDS_PER_WORKGROUP,
            queue: 0,
            wavefront_mode: WavefrontMode::Wave64,
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

    #[test]
    fn test_wavefront_modes() {
        let mut eng = ComputeEngine::new();
        let id1 = eng.launch("wave64_kernel", valid_config()).unwrap();
        let id2 = eng.launch("wave32_kernel", valid_wave32_config()).unwrap();
        let k1 = eng.kernels.iter().find(|k| k.id == id1).unwrap();
        let k2 = eng.kernels.iter().find(|k| k.id == id2).unwrap();
        assert_eq!(k1.config.wavefront_mode, WavefrontMode::Wave64);
        assert_eq!(k2.config.wavefront_mode, WavefrontMode::Wave32);
    }
}
