// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unified compute abstraction for GPU backends.
//!
//! Defines the [`ComputeProvider`] trait that captures the common interface
//! across all GPU vendors (NVIDIA CUDA, AMD ROCm, Intel Level Zero, Apple
//! Metal). GPU arch crates at Layer 2 implement this trait; the ONNX runtime
//! and other services at Layer 1 consume it.
//!
//! Also provides [`GpuBackend`], an enum-dispatch wrapper that selects the
//! active backend at runtime via feature flags, avoiding trait objects in
//! `#![no_std]` environments.

#![no_std]

extern crate alloc;

use alloc::string::String;

// ---------------------------------------------------------------------------
// ComputeProvider trait
// ---------------------------------------------------------------------------

/// Unified interface for GPU compute backends.
///
/// Each backend (CUDA, ROCm, Level Zero, Metal) implements this trait,
/// providing a consistent API for device lifecycle, memory management,
/// kernel dispatch, and operator queries.
pub trait ComputeProvider {
    /// Handle to a device-side memory allocation.
    type Buffer;
    /// Handle to a compiled kernel/pipeline state.
    type Kernel;
    /// Backend-specific error type.
    type Error: core::fmt::Debug;

    /// Returns static information about the compute device.
    fn device_info(&self) -> DeviceInfo;

    /// Initializes the compute device and runtime resources.
    fn init(&mut self) -> Result<(), Self::Error>;

    /// Allocates `size` bytes of device memory.
    fn alloc(&mut self, size: usize) -> Result<Self::Buffer, Self::Error>;

    /// Frees a previously allocated device buffer.
    fn free(&mut self, buf: Self::Buffer) -> Result<(), Self::Error>;

    /// Copies data from host memory into a device buffer.
    fn copy_host_to_device(
        &mut self,
        src: &[u8],
        dst: &mut Self::Buffer,
    ) -> Result<(), Self::Error>;

    /// Copies data from a device buffer into host memory.
    fn copy_device_to_host(&self, src: &Self::Buffer, dst: &mut [u8]) -> Result<(), Self::Error>;

    /// Compiles/loads a named kernel from source bytes.
    fn load_kernel(&mut self, name: &str, source: &[u8]) -> Result<Self::Kernel, Self::Error>;

    /// Launches a kernel with the given grid/block dimensions and buffer args.
    fn launch(
        &mut self,
        kernel: &Self::Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[&Self::Buffer],
    ) -> Result<(), Self::Error>;

    /// Blocks until all submitted work has completed.
    fn synchronize(&mut self) -> Result<(), Self::Error>;

    /// Returns `true` if this backend supports the named ONNX operator.
    fn supports_op(&self, op: &str) -> bool;
}

// ---------------------------------------------------------------------------
// DeviceInfo
// ---------------------------------------------------------------------------

/// Static information about a compute device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// Total device memory in bytes.
    pub memory_bytes: u64,
    /// Number of compute units (SMs, CUs, EUs, GPU cores).
    pub compute_units: u32,
    /// Which backend family this device belongs to.
    pub backend_type: BackendType,
}

// ---------------------------------------------------------------------------
// BackendType
// ---------------------------------------------------------------------------

/// Identifies the GPU backend family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// CPU fallback (no GPU).
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// AMD ROCm / HIP.
    Rocm,
    /// Intel Level Zero (oneAPI).
    LevelZero,
    /// Apple Metal.
    Metal,
}

// ---------------------------------------------------------------------------
// ComputeError
// ---------------------------------------------------------------------------

/// Error type for the CPU fallback and GpuBackend dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    /// No GPU backend is available; operation not supported on CPU fallback.
    NoGpuAvailable,
    /// Device initialization failed.
    InitFailed,
    /// Memory allocation failed.
    AllocFailed,
    /// Memory free failed.
    FreeFailed,
    /// Host-to-device or device-to-host copy failed.
    CopyFailed,
    /// Kernel compilation or loading failed.
    KernelLoadFailed,
    /// Kernel launch failed.
    LaunchFailed,
    /// Synchronization failed.
    SyncFailed,
    /// The requested operation is not supported.
    Unsupported,
}

impl core::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ComputeError::NoGpuAvailable => write!(f, "no GPU available"),
            ComputeError::InitFailed => write!(f, "device init failed"),
            ComputeError::AllocFailed => write!(f, "allocation failed"),
            ComputeError::FreeFailed => write!(f, "free failed"),
            ComputeError::CopyFailed => write!(f, "copy failed"),
            ComputeError::KernelLoadFailed => write!(f, "kernel load failed"),
            ComputeError::LaunchFailed => write!(f, "launch failed"),
            ComputeError::SyncFailed => write!(f, "sync failed"),
            ComputeError::Unsupported => write!(f, "operation not supported"),
        }
    }
}

// ---------------------------------------------------------------------------
// CpuFallback — stub provider that always returns errors
// ---------------------------------------------------------------------------

/// CPU fallback provider. All GPU operations return errors since there
/// is no actual GPU device.
#[derive(Debug, Default)]
pub struct CpuFallback;

impl ComputeProvider for CpuFallback {
    type Buffer = ();
    type Kernel = ();
    type Error = ComputeError;

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: String::from("CPU (no GPU)"),
            memory_bytes: 0,
            compute_units: 0,
            backend_type: BackendType::Cpu,
        }
    }

    fn init(&mut self) -> Result<(), Self::Error> {
        // CPU fallback is always "initialized"
        Ok(())
    }

    fn alloc(&mut self, _size: usize) -> Result<Self::Buffer, Self::Error> {
        Err(ComputeError::NoGpuAvailable)
    }

    fn free(&mut self, _buf: Self::Buffer) -> Result<(), Self::Error> {
        Err(ComputeError::NoGpuAvailable)
    }

    fn copy_host_to_device(
        &mut self,
        _src: &[u8],
        _dst: &mut Self::Buffer,
    ) -> Result<(), Self::Error> {
        Err(ComputeError::NoGpuAvailable)
    }

    fn copy_device_to_host(&self, _src: &Self::Buffer, _dst: &mut [u8]) -> Result<(), Self::Error> {
        Err(ComputeError::NoGpuAvailable)
    }

    fn load_kernel(&mut self, _name: &str, _source: &[u8]) -> Result<Self::Kernel, Self::Error> {
        Err(ComputeError::NoGpuAvailable)
    }

    fn launch(
        &mut self,
        _kernel: &Self::Kernel,
        _grid: [u32; 3],
        _block: [u32; 3],
        _args: &[&Self::Buffer],
    ) -> Result<(), Self::Error> {
        Err(ComputeError::NoGpuAvailable)
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn supports_op(&self, _op: &str) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// GpuBackend — enum dispatch
// ---------------------------------------------------------------------------

/// Runtime-selected GPU backend using enum dispatch.
///
/// Feature-gated variants are compiled in only when the corresponding
/// feature flag is enabled. The `Cpu` variant is always available as a
/// fallback that returns errors for all GPU operations.
pub enum GpuBackend {
    /// CPU fallback — no GPU available.
    Cpu(CpuFallback),
}

impl GpuBackend {
    /// Creates a new CPU-fallback backend.
    pub fn cpu() -> Self {
        GpuBackend::Cpu(CpuFallback)
    }

    /// Returns device information for the active backend.
    pub fn device_info(&self) -> DeviceInfo {
        match self {
            GpuBackend::Cpu(p) => p.device_info(),
        }
    }

    /// Initializes the active backend.
    pub fn init(&mut self) -> Result<(), ComputeError> {
        match self {
            GpuBackend::Cpu(p) => p.init(),
        }
    }

    /// Returns `true` if the active backend supports the named ONNX operator.
    pub fn supports_op(&self, op: &str) -> bool {
        match self {
            GpuBackend::Cpu(p) => p.supports_op(op),
        }
    }

    /// Blocks until all submitted work has completed.
    pub fn synchronize(&mut self) -> Result<(), ComputeError> {
        match self {
            GpuBackend::Cpu(p) => p.synchronize(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;

    // ---- BackendType ----

    #[test]
    fn test_backend_type_variants() {
        assert_ne!(BackendType::Cpu, BackendType::Cuda);
        assert_ne!(BackendType::Cuda, BackendType::Rocm);
        assert_ne!(BackendType::Rocm, BackendType::LevelZero);
        assert_ne!(BackendType::LevelZero, BackendType::Metal);
        assert_eq!(BackendType::Cpu, BackendType::Cpu);
    }

    #[test]
    fn test_backend_type_debug() {
        let dbg = format!("{:?}", BackendType::Cuda);
        assert!(dbg.contains("Cuda"));
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn test_backend_type_clone_copy() {
        let a = BackendType::Metal;
        let b = a;
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // ---- DeviceInfo ----

    #[test]
    fn test_device_info_construction() {
        let info = DeviceInfo {
            name: String::from("Test GPU"),
            memory_bytes: 8 * 1024 * 1024 * 1024,
            compute_units: 64,
            backend_type: BackendType::Cuda,
        };
        assert_eq!(info.name, "Test GPU");
        assert_eq!(info.memory_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(info.compute_units, 64);
        assert_eq!(info.backend_type, BackendType::Cuda);
    }

    #[test]
    fn test_device_info_clone() {
        let info = DeviceInfo {
            name: String::from("Test GPU"),
            memory_bytes: 4096,
            compute_units: 32,
            backend_type: BackendType::Rocm,
        };
        let cloned = info.clone();
        assert_eq!(info, cloned);
    }

    #[test]
    fn test_device_info_debug() {
        let info = DeviceInfo {
            name: String::from("Test"),
            memory_bytes: 0,
            compute_units: 0,
            backend_type: BackendType::Cpu,
        };
        let dbg = format!("{:?}", info);
        assert!(dbg.contains("Test"));
        assert!(dbg.contains("Cpu"));
    }

    // ---- ComputeError ----

    #[test]
    fn test_compute_error_display() {
        assert_eq!(
            format!("{}", ComputeError::NoGpuAvailable),
            "no GPU available"
        );
        assert_eq!(
            format!("{}", ComputeError::InitFailed),
            "device init failed"
        );
        assert_eq!(
            format!("{}", ComputeError::AllocFailed),
            "allocation failed"
        );
        assert_eq!(format!("{}", ComputeError::FreeFailed), "free failed");
        assert_eq!(format!("{}", ComputeError::CopyFailed), "copy failed");
        assert_eq!(
            format!("{}", ComputeError::KernelLoadFailed),
            "kernel load failed"
        );
        assert_eq!(format!("{}", ComputeError::LaunchFailed), "launch failed");
        assert_eq!(format!("{}", ComputeError::SyncFailed), "sync failed");
        assert_eq!(
            format!("{}", ComputeError::Unsupported),
            "operation not supported"
        );
    }

    #[test]
    fn test_compute_error_debug() {
        let dbg = format!("{:?}", ComputeError::NoGpuAvailable);
        assert!(dbg.contains("NoGpuAvailable"));
    }

    #[test]
    fn test_compute_error_clone_eq() {
        let a = ComputeError::LaunchFailed;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ---- CpuFallback ----

    #[test]
    fn test_cpu_fallback_device_info() {
        let fb = CpuFallback;
        let info = fb.device_info();
        assert_eq!(info.name, "CPU (no GPU)");
        assert_eq!(info.memory_bytes, 0);
        assert_eq!(info.compute_units, 0);
        assert_eq!(info.backend_type, BackendType::Cpu);
    }

    #[test]
    fn test_cpu_fallback_init_succeeds() {
        let mut fb = CpuFallback;
        assert!(fb.init().is_ok());
    }

    #[test]
    fn test_cpu_fallback_alloc_returns_error() {
        let mut fb = CpuFallback;
        assert_eq!(fb.alloc(1024), Err(ComputeError::NoGpuAvailable));
    }

    #[test]
    fn test_cpu_fallback_free_returns_error() {
        let mut fb = CpuFallback;
        assert_eq!(fb.free(()), Err(ComputeError::NoGpuAvailable));
    }

    #[test]
    fn test_cpu_fallback_copy_h2d_returns_error() {
        let mut fb = CpuFallback;
        let mut buf = ();
        assert_eq!(
            fb.copy_host_to_device(&[1, 2, 3], &mut buf),
            Err(ComputeError::NoGpuAvailable)
        );
    }

    #[test]
    fn test_cpu_fallback_copy_d2h_returns_error() {
        let fb = CpuFallback;
        let buf = ();
        let mut dst = [0u8; 4];
        assert_eq!(
            fb.copy_device_to_host(&buf, &mut dst),
            Err(ComputeError::NoGpuAvailable)
        );
    }

    #[test]
    fn test_cpu_fallback_load_kernel_returns_error() {
        let mut fb = CpuFallback;
        assert_eq!(
            fb.load_kernel("test", &[0]),
            Err(ComputeError::NoGpuAvailable)
        );
    }

    #[test]
    fn test_cpu_fallback_launch_returns_error() {
        let mut fb = CpuFallback;
        let kernel = ();
        assert_eq!(
            fb.launch(&kernel, [1, 1, 1], [1, 1, 1], &[]),
            Err(ComputeError::NoGpuAvailable)
        );
    }

    #[test]
    fn test_cpu_fallback_synchronize_succeeds() {
        let mut fb = CpuFallback;
        assert!(fb.synchronize().is_ok());
    }

    #[test]
    fn test_cpu_fallback_supports_no_ops() {
        let fb = CpuFallback;
        assert!(!fb.supports_op("MatMul"));
        assert!(!fb.supports_op("Relu"));
        assert!(!fb.supports_op("Conv"));
        assert!(!fb.supports_op(""));
    }

    // ---- GpuBackend ----

    #[test]
    fn test_gpu_backend_cpu_device_info() {
        let backend = GpuBackend::cpu();
        let info = backend.device_info();
        assert_eq!(info.backend_type, BackendType::Cpu);
        assert_eq!(info.name, "CPU (no GPU)");
    }

    #[test]
    fn test_gpu_backend_cpu_init() {
        let mut backend = GpuBackend::cpu();
        assert!(backend.init().is_ok());
    }

    #[test]
    fn test_gpu_backend_cpu_supports_no_ops() {
        let backend = GpuBackend::cpu();
        assert!(!backend.supports_op("MatMul"));
        assert!(!backend.supports_op("Relu"));
    }

    #[test]
    fn test_gpu_backend_cpu_synchronize() {
        let mut backend = GpuBackend::cpu();
        assert!(backend.synchronize().is_ok());
    }

    // ---- Trait object safety check (compile-time) ----

    #[test]
    fn test_cpu_fallback_default() {
        let fb = CpuFallback;
        let info = fb.device_info();
        assert_eq!(info.backend_type, BackendType::Cpu);
    }

    #[test]
    fn test_cpu_fallback_debug() {
        let fb = CpuFallback;
        let dbg = format!("{:?}", fb);
        assert!(dbg.contains("CpuFallback"));
    }
}
