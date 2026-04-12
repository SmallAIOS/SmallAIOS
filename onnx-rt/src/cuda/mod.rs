// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CUDA execution provider for ONNX inference.
//!
//! Provides safe Rust wrappers around CUDA runtime, cuBLAS, and cuDNN FFI
//! for GPU-accelerated ONNX operator dispatch. All code is behind
//! `#[cfg(feature = "cuda")]`.
//!
//! ## Architecture
//!
//! ```text
//! cuda/mod.rs      — safe wrappers, CudaRuntime, CublasHandle, CudnnHandle
//! cuda/ffi.rs      — raw extern "C" declarations
//! cuda/memory.rs   — DeviceBuffer (RAII), DeviceWeightStore
//! ```

extern crate alloc;
use alloc::string::String;

pub mod conv;
pub mod dispatch;
pub mod ffi;
pub mod gpu_executor;
pub mod kv_cache;
pub mod memory;

pub use gpu_executor::{
    execute_graph_gpu, execute_graph_gpu_with_weights, gpu_conv2d_device, gpu_gemm_device,
    gpu_gemm_int8_device, initializers_to_gpu, tensor_to_device, DeviceTensor,
};
pub use kv_cache::{GpuKvCache, KvView, LayerKind};
pub use memory::{DeviceBuffer, DeviceWeightStore};

// ── Error type ──────────────────────────────────────────────────────

/// CUDA-related errors.
#[derive(Debug)]
pub enum CudaError {
    /// No CUDA-capable GPU found.
    NoDevice,
    /// CUDA runtime version mismatch.
    VersionMismatch {
        expected_major: i32,
        actual_major: i32,
    },
    /// cudaMalloc failed.
    AllocFailed { size: usize, code: i32 },
    /// cudaMemcpy failed.
    CopyFailed { msg: &'static str, code: i32 },
    /// cuBLAS operation failed.
    BlasError { op: &'static str, code: i32 },
    /// cuDNN operation failed.
    DnnError { op: &'static str, code: i32 },
    /// Generic CUDA runtime error.
    RuntimeError { op: &'static str, code: i32 },
}

impl core::fmt::Display for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no CUDA device found"),
            Self::VersionMismatch {
                expected_major,
                actual_major,
            } => write!(
                f,
                "CUDA version mismatch: expected major {}, got {}",
                expected_major, actual_major
            ),
            Self::AllocFailed { size, code } => {
                write!(f, "cudaMalloc({} bytes) failed: code {}", size, code)
            }
            Self::CopyFailed { msg, code } => write!(f, "cudaMemcpy {} failed: code {}", msg, code),
            Self::BlasError { op, code } => write!(f, "cuBLAS {} failed: code {}", op, code),
            Self::DnnError { op, code } => write!(f, "cuDNN {} failed: code {}", op, code),
            Self::RuntimeError { op, code } => write!(f, "CUDA {} failed: code {}", op, code),
        }
    }
}

// ── Compile-time CUDA major version ─────────────────────────────────

/// CUDA major version these bindings target.
/// Used for runtime compatibility check.
const TARGET_CUDA_MAJOR: i32 = 13;

// ── Device discovery ────────────────────────────────────────────────

/// Information about a CUDA device.
#[derive(Debug)]
pub struct CudaDeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub total_mem_bytes: usize,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
}

/// Query the number of CUDA devices.
pub fn device_count() -> Result<i32, CudaError> {
    let mut count: i32 = 0;
    let err = unsafe { ffi::cudaGetDeviceCount(&mut count) };
    if err != ffi::CUDA_SUCCESS {
        return Err(CudaError::RuntimeError {
            op: "cudaGetDeviceCount",
            code: err,
        });
    }
    Ok(count)
}

/// Query properties of a specific CUDA device.
///
/// Uses `cudaDeviceGetAttribute` for reliable access across CUDA versions
/// (the `cudaDeviceProp` struct layout changes between major versions).
pub fn device_info(device: i32) -> Result<CudaDeviceInfo, CudaError> {
    fn get_attr(attr: i32, device: i32) -> Result<i32, CudaError> {
        let mut val: i32 = 0;
        let err = unsafe { ffi::cudaDeviceGetAttribute(&mut val, attr, device) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::RuntimeError {
                op: "cudaDeviceGetAttribute",
                code: err,
            });
        }
        Ok(val)
    }

    // Attribute IDs (from cuda_runtime_api.h).
    const COMPUTE_MAJOR: i32 = 75;
    const COMPUTE_MINOR: i32 = 76;
    const MAX_THREADS_PER_BLOCK: i32 = 1;
    const WARP_SIZE: i32 = 10;

    let compute_major = get_attr(COMPUTE_MAJOR, device)?;
    let compute_minor = get_attr(COMPUTE_MINOR, device)?;
    let max_threads_per_block = get_attr(MAX_THREADS_PER_BLOCK, device)?;
    let warp_size = get_attr(WARP_SIZE, device)?;

    // Get total memory via cudaMemGetInfo (need to set device first).
    let prev_err = unsafe { ffi::cudaSetDevice(device) };
    if prev_err != ffi::CUDA_SUCCESS {
        return Err(CudaError::RuntimeError {
            op: "cudaSetDevice",
            code: prev_err,
        });
    }
    let mut free_mem: usize = 0;
    let mut total_mem: usize = 0;
    let err = unsafe { ffi::cudaMemGetInfo(&mut free_mem, &mut total_mem) };
    if err != ffi::CUDA_SUCCESS {
        return Err(CudaError::RuntimeError {
            op: "cudaMemGetInfo",
            code: err,
        });
    }

    // Get device name from cudaDeviceProp (only the name field at offset 0).
    let mut props = ffi::cudaDeviceProp::default();
    let _ = unsafe { ffi::cudaGetDeviceProperties(&mut props, device) };
    let name_len = props.name.iter().position(|&b| b == 0).unwrap_or(255);
    let name = core::str::from_utf8(&props.name[..name_len])
        .unwrap_or("unknown")
        .into();

    Ok(CudaDeviceInfo {
        device_id: device,
        name,
        total_mem_bytes: total_mem,
        compute_major,
        compute_minor,
        max_threads_per_block,
        warp_size,
    })
}

/// Get the CUDA runtime version and validate major version compatibility.
pub fn runtime_version() -> Result<i32, CudaError> {
    let mut version: i32 = 0;
    let err = unsafe { ffi::cudaRuntimeGetVersion(&mut version) };
    if err != ffi::CUDA_SUCCESS {
        return Err(CudaError::RuntimeError {
            op: "cudaRuntimeGetVersion",
            code: err,
        });
    }
    Ok(version)
}

/// Check that the CUDA runtime major version matches compiled bindings.
pub fn check_version() -> Result<i32, CudaError> {
    let version = runtime_version()?;
    // CUDA version encoding: major * 1000 + minor * 10
    let major = version / 1000;
    if major != TARGET_CUDA_MAJOR {
        return Err(CudaError::VersionMismatch {
            expected_major: TARGET_CUDA_MAJOR,
            actual_major: major,
        });
    }
    Ok(version)
}

/// Set the active CUDA device.
pub fn set_device(device: i32) -> Result<(), CudaError> {
    let err = unsafe { ffi::cudaSetDevice(device) };
    if err != ffi::CUDA_SUCCESS {
        return Err(CudaError::RuntimeError {
            op: "cudaSetDevice",
            code: err,
        });
    }
    Ok(())
}

/// Synchronize the current CUDA device (wait for all pending work).
pub fn synchronize() -> Result<(), CudaError> {
    let err = unsafe { ffi::cudaDeviceSynchronize() };
    if err != ffi::CUDA_SUCCESS {
        return Err(CudaError::RuntimeError {
            op: "cudaDeviceSynchronize",
            code: err,
        });
    }
    Ok(())
}

// ── cuBLAS handle ───────────────────────────────────────────────────

/// RAII wrapper around a cuBLAS handle.
pub struct CublasHandle {
    handle: ffi::cublasHandle_t,
}

unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    /// Create a new cuBLAS handle.
    pub fn new() -> Result<Self, CudaError> {
        let mut handle: ffi::cublasHandle_t = core::ptr::null_mut();
        let err = unsafe { ffi::cublasCreate_v2(&mut handle) };
        if err != ffi::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::BlasError {
                op: "cublasCreate",
                code: err,
            });
        }
        Ok(Self { handle })
    }

    /// Perform single-precision GEMM: C = alpha * op(A) * op(B) + beta * C
    ///
    /// Note: cuBLAS uses column-major layout. For row-major data (as used in
    /// ONNX tensors), swap A/B and transpose flags.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        &self,
        transa: ffi::cublasOperation_t,
        transb: ffi::cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &DeviceBuffer,
        lda: i32,
        b: &DeviceBuffer,
        ldb: i32,
        beta: f32,
        c: &DeviceBuffer,
        ldc: i32,
    ) -> Result<(), CudaError> {
        let err = unsafe {
            ffi::cublasSgemm_v2(
                self.handle,
                transa,
                transb,
                m,
                n,
                k,
                &alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                &beta,
                c.as_mut_ptr(),
                ldc,
            )
        };
        if err != ffi::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::BlasError {
                op: "cublasSgemm",
                code: err,
            });
        }
        Ok(())
    }

    /// Perform mixed-precision GEMM via cublasGemmEx (e.g. INT8 with INT32 accumulation).
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn gemm_ex(
        &self,
        transa: ffi::cublasOperation_t,
        transb: ffi::cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const core::ffi::c_void,
        a: &DeviceBuffer,
        a_type: ffi::cudaDataType_t,
        lda: i32,
        b: &DeviceBuffer,
        b_type: ffi::cudaDataType_t,
        ldb: i32,
        beta: *const core::ffi::c_void,
        c: &DeviceBuffer,
        c_type: ffi::cudaDataType_t,
        ldc: i32,
        compute_type: ffi::cublasComputeType_t,
    ) -> Result<(), CudaError> {
        let err = unsafe {
            ffi::cublasGemmEx(
                self.handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                a_type,
                lda,
                b.as_ptr(),
                b_type,
                ldb,
                beta,
                c.as_mut_ptr(),
                c_type,
                ldc,
                compute_type,
                -1, // CUBLAS_GEMM_DEFAULT
            )
        };
        if err != ffi::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::BlasError {
                op: "cublasGemmEx",
                code: err,
            });
        }
        Ok(())
    }

    /// Raw cuBLAS handle (for advanced use).
    pub fn raw(&self) -> ffi::cublasHandle_t {
        self.handle
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            ffi::cublasDestroy_v2(self.handle);
        }
    }
}

// ── cuDNN handle ────────────────────────────────────────────────────

/// RAII wrapper around a cuDNN handle.
pub struct CudnnHandle {
    handle: ffi::cudnnHandle_t,
}

unsafe impl Send for CudnnHandle {}
unsafe impl Sync for CudnnHandle {}

impl CudnnHandle {
    /// Create a new cuDNN handle.
    pub fn new() -> Result<Self, CudaError> {
        let mut handle: ffi::cudnnHandle_t = core::ptr::null_mut();
        let err = unsafe { ffi::cudnnCreate(&mut handle) };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::DnnError {
                op: "cudnnCreate",
                code: err,
            });
        }
        Ok(Self { handle })
    }

    /// Raw cuDNN handle.
    pub fn raw(&self) -> ffi::cudnnHandle_t {
        self.handle
    }
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroy(self.handle);
        }
    }
}

// ── cuBLASLt handle ─────────────────────────────────────────────────

/// RAII wrapper around a cuBLASLt handle (needed for FP8 GEMM).
pub struct CublasLtHandle {
    handle: ffi::cublasLtHandle_t,
}

unsafe impl Send for CublasLtHandle {}
unsafe impl Sync for CublasLtHandle {}

impl CublasLtHandle {
    /// Create a new cuBLASLt handle.
    pub fn new() -> Result<Self, CudaError> {
        let mut handle: ffi::cublasLtHandle_t = core::ptr::null_mut();
        let err = unsafe { ffi::cublasLtCreate(&mut handle) };
        if err != ffi::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::BlasError {
                op: "cublasLtCreate",
                code: err,
            });
        }
        Ok(Self { handle })
    }

    /// Raw handle.
    pub fn raw(&self) -> ffi::cublasLtHandle_t {
        self.handle
    }
}

impl Drop for CublasLtHandle {
    fn drop(&mut self) {
        unsafe {
            ffi::cublasLtDestroy(self.handle);
        }
    }
}

// ── Initialization ──────────────────────────────────────────────────

/// Complete CUDA runtime state for inference.
///
/// Created once at session initialization. Holds cuBLAS/cuDNN handles
/// and device info for the lifetime of the session.
// Safety: CudaRuntime contains cuBLAS/cuDNN handles which are safe to share
// across threads when CUDA operations are synchronized (which we do via
// cudaDeviceSynchronize after each kernel launch).
unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

/// GPU compute precision for GEMM operators.
///
/// Controls which `cublasComputeType_t` is used for f32 GEMM dispatch.
/// Higher-throughput modes (TF32, FP16, BF16) use tensor cores with
/// reduced internal precision but f32 I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPrecision {
    /// Standard f32 compute — no tensor cores.
    F32,
    /// TF32 tensor cores (19-bit mantissa) — best default for Ampere+.
    /// ~2x throughput vs F32, negligible accuracy loss for inference.
    Tf32,
    /// FP16 tensor cores with f32 accumulation.
    /// ~4x throughput vs F32, slight accuracy reduction.
    Fp16,
    /// BF16 tensor cores with f32 accumulation.
    /// ~4x throughput vs F32, slightly better dynamic range than FP16.
    Bf16,
}

impl GpuPrecision {
    /// Parse from environment variable string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "f32" | "fp32" => Self::F32,
            "tf32" => Self::Tf32,
            "fp16" | "f16" => Self::Fp16,
            "bf16" => Self::Bf16,
            _ => Self::Tf32, // default
        }
    }

    /// Convert to the corresponding cuBLAS compute type.
    pub fn to_cublas_compute_type(self) -> ffi::cublasComputeType_t {
        match self {
            Self::F32 => ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            Self::Tf32 => ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            Self::Fp16 => ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
            Self::Bf16 => ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
        }
    }
}

pub struct CudaRuntime {
    pub device: CudaDeviceInfo,
    pub cublas: CublasHandle,
    pub cublas_lt: CublasLtHandle,
    pub cudnn: CudnnHandle,
    pub cuda_version: i32,
    pub weight_store: DeviceWeightStore,
    pub precision: GpuPrecision,
}

impl CudaRuntime {
    /// Initialize the CUDA runtime: probe device, check version, create handles.
    ///
    /// Returns `Err(CudaError::NoDevice)` if no GPU is available.
    pub fn init() -> Result<Self, CudaError> {
        Self::init_with_precision(GpuPrecision::Tf32)
    }

    /// Initialize with a specific GPU precision mode.
    pub fn init_with_precision(precision: GpuPrecision) -> Result<Self, CudaError> {
        // Check version compatibility first.
        let cuda_version = check_version()?;

        // Probe for devices.
        let count = device_count()?;
        if count == 0 {
            return Err(CudaError::NoDevice);
        }

        // Use device 0.
        set_device(0)?;
        let device = device_info(0)?;

        // Create library handles.
        let cublas = CublasHandle::new()?;
        let cublas_lt = CublasLtHandle::new()?;
        let cudnn = CudnnHandle::new()?;

        Ok(Self {
            device,
            cublas,
            cublas_lt,
            cudnn,
            cuda_version,
            weight_store: DeviceWeightStore::new(),
            precision,
        })
    }

    /// Check whether a given ONNX operator can be dispatched to GPU.
    pub fn supports_op(op_type: &str) -> bool {
        matches!(op_type, "MatMul" | "Gemm" | "MatMulInteger" | "Conv")
    }
}
