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
use alloc::collections::BTreeMap;
use alloc::string::String;

pub mod conv;
pub mod dispatch;
pub mod ffi;
pub mod gpu_executor;
pub mod kernels;
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
    /// NVRTC failed to compile a kernel; `log` carries the NVRTC build log.
    KernelCompileFailed { name: String, log: String },
    /// `cuModuleLoadData` or `cuModuleGetFunction` failed for a compiled
    /// kernel.
    KernelLoadFailed { name: String, cuda_result: i32 },
    /// `cuLaunchKernel` failed for a previously compiled kernel.
    KernelLaunchFailed { name: String, cuda_result: i32 },
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
            Self::KernelCompileFailed { name, log } => {
                write!(f, "NVRTC failed to compile kernel {:?}: {}", name, log)
            }
            Self::KernelLoadFailed { name, cuda_result } => write!(
                f,
                "cuModule load of kernel {:?} failed: CUresult {}",
                name, cuda_result
            ),
            Self::KernelLaunchFailed { name, cuda_result } => write!(
                f,
                "cuLaunchKernel for {:?} failed: CUresult {}",
                name, cuda_result
            ),
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

    /// Strided batched mixed-precision GEMM via `cublasGemmStridedBatchedEx`.
    ///
    /// Each of `batch_count` independent `m×n×k` GEMMs is laid out
    /// contiguously in memory with the given per-batch element strides.
    /// Used by `gpu_gemm_strided_batched_ex` to drive the per-head
    /// QK^T and softmax·V matmuls inside `gpu_gqa`.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn gemm_strided_batched_ex(
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
        stride_a: i64,
        b: &DeviceBuffer,
        b_type: ffi::cudaDataType_t,
        ldb: i32,
        stride_b: i64,
        beta: *const core::ffi::c_void,
        c: &DeviceBuffer,
        c_type: ffi::cudaDataType_t,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
        compute_type: ffi::cublasComputeType_t,
    ) -> Result<(), CudaError> {
        let err = unsafe {
            ffi::cublasGemmStridedBatchedEx(
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
                stride_a,
                b.as_ptr(),
                b_type,
                ldb,
                stride_b,
                beta,
                c.as_mut_ptr(),
                c_type,
                ldc,
                stride_c,
                batch_count,
                compute_type,
                -1, // CUBLAS_GEMM_DEFAULT
            )
        };
        if err != ffi::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::BlasError {
                op: "cublasGemmStridedBatchedEx",
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
    /// Registry of NVRTC-compiled kernels, keyed by stable symbol name.
    ///
    /// Populated eagerly by [`CudaRuntime::init_kernels`] after runtime
    /// construction. Operator dispatch looks up kernels via
    /// [`CudaRuntime::with_kernel`]. Interior mutability via `Mutex` lets
    /// `&CudaRuntime` mutate the map while keeping the runtime `Sync`.
    pub kernel_registry: std::sync::Mutex<BTreeMap<&'static str, kernels::Kernel>>,
    /// Hard cap on the per-call attention-scratch buffer size in bytes.
    ///
    /// `gpu_gqa` checks the size of the `[H, seq_q, seq_kv]` F32 score
    /// matrix against this cap and returns `RuntimeError` rather than
    /// allocate beyond it. Defaults to 256 MiB; safetensors sessions
    /// can override via [`CudaRuntime::set_attention_scratch_cap`] once
    /// they know the layer-specific window / max sequence length.
    pub attention_scratch_cap_bytes: std::sync::atomic::AtomicUsize,
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
            kernel_registry: std::sync::Mutex::new(BTreeMap::new()),
            attention_scratch_cap_bytes: std::sync::atomic::AtomicUsize::new(256 * 1024 * 1024),
        })
    }

    /// Override the per-call attention-scratch cap (in bytes).
    ///
    /// Sessions that know their `max_seq_len` / sliding window can size
    /// the cap precisely, e.g. `num_heads * window^2 * 4` for a local
    /// layer. The cap is checked by `gpu_gqa` before allocating the
    /// `[H, seq_q, seq_kv]` F32 score matrix.
    pub fn set_attention_scratch_cap(&self, bytes: usize) {
        self.attention_scratch_cap_bytes
            .store(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Read the current attention-scratch cap.
    pub fn attention_scratch_cap(&self) -> usize {
        self.attention_scratch_cap_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Eagerly compile and register all built-in NVRTC kernels.
    ///
    /// This is **not** called automatically from [`CudaRuntime::init`] —
    /// callers must invoke it explicitly once per runtime instance
    /// before dispatching any operator that depends on a JIT kernel.
    /// Compiling up-front lets the first inference call avoid NVRTC
    /// latency.
    ///
    /// Kernels registered by Section 2 of `transformer-gpu-kernels-v1`:
    /// `add_f32`, `add_bf16`, `mul_f32`, `mul_bf16`, `silu_f32`,
    /// `silu_bf16`.
    pub fn init_kernels(&self) -> Result<(), CudaError> {
        // Ensure the driver-API context exists before any kernel compile.
        kernels::lazy_context_init()?;

        let mut registry = self
            .kernel_registry
            .lock()
            .map_err(|_| CudaError::RuntimeError {
                op: "kernel_registry poisoned",
                code: -1,
            })?;

        registry.insert(
            "add_f32",
            kernels::compile_kernel("add_f32", kernels::elementwise::ADD_F32_SRC, &[])?,
        );
        registry.insert(
            "add_bf16",
            kernels::compile_kernel("add_bf16", kernels::elementwise::ADD_BF16_SRC, &[])?,
        );
        registry.insert(
            "mul_f32",
            kernels::compile_kernel("mul_f32", kernels::elementwise::MUL_F32_SRC, &[])?,
        );
        registry.insert(
            "mul_bf16",
            kernels::compile_kernel("mul_bf16", kernels::elementwise::MUL_BF16_SRC, &[])?,
        );
        registry.insert(
            "silu_f32",
            kernels::compile_kernel("silu_f32", kernels::elementwise::SILU_F32_SRC, &[])?,
        );
        registry.insert(
            "silu_bf16",
            kernels::compile_kernel("silu_bf16", kernels::elementwise::SILU_BF16_SRC, &[])?,
        );
        registry.insert(
            "gather_f32",
            kernels::compile_kernel("gather_f32", kernels::gather::GATHER_F32_SRC, &[])?,
        );
        registry.insert(
            "gather_bf16",
            kernels::compile_kernel("gather_bf16", kernels::gather::GATHER_BF16_SRC, &[])?,
        );
        registry.insert(
            "rms_norm_f32",
            kernels::compile_kernel("rms_norm_f32", kernels::rms_norm::RMS_NORM_F32_SRC, &[])?,
        );
        registry.insert(
            "rms_norm_bf16",
            kernels::compile_kernel("rms_norm_bf16", kernels::rms_norm::RMS_NORM_BF16_SRC, &[])?,
        );
        registry.insert(
            "rotary_f32",
            kernels::compile_kernel("rotary_f32", kernels::rotary::ROTARY_F32_SRC, &[])?,
        );
        registry.insert(
            "rotary_bf16",
            kernels::compile_kernel("rotary_bf16", kernels::rotary::ROTARY_BF16_SRC, &[])?,
        );
        registry.insert(
            "gqa_softmax_mask_f32",
            kernels::compile_kernel(
                "gqa_softmax_mask_f32",
                kernels::attention::GQA_SOFTMAX_MASK_F32_SRC,
                &[],
            )?,
        );
        registry.insert(
            "gqa_kv_expand_f32",
            kernels::compile_kernel(
                "gqa_kv_expand_f32",
                kernels::attention::GQA_KV_EXPAND_F32_SRC,
                &[],
            )?,
        );
        registry.insert(
            "gqa_kv_expand_bf16",
            kernels::compile_kernel(
                "gqa_kv_expand_bf16",
                kernels::attention::GQA_KV_EXPAND_BF16_SRC,
                &[],
            )?,
        );
        registry.insert(
            "gqa_merge_heads_f32",
            kernels::compile_kernel(
                "gqa_merge_heads_f32",
                kernels::attention::GQA_MERGE_HEADS_F32_SRC,
                &[],
            )?,
        );
        registry.insert(
            "gqa_merge_heads_bf16",
            kernels::compile_kernel(
                "gqa_merge_heads_bf16",
                kernels::attention::GQA_MERGE_HEADS_BF16_SRC,
                &[],
            )?,
        );
        registry.insert(
            "softmax_cast_f32_to_bf16",
            kernels::compile_kernel(
                "softmax_cast_f32_to_bf16",
                kernels::attention::SOFTMAX_CAST_F32_TO_BF16_SRC,
                &[],
            )?,
        );
        registry.insert(
            "kv_cache_view_to_head_major_f32",
            kernels::compile_kernel(
                "kv_cache_view_to_head_major_f32",
                kernels::attention::KV_CACHE_VIEW_TO_HEAD_MAJOR_F32_SRC,
                &[],
            )?,
        );
        registry.insert(
            "kv_cache_view_to_head_major_bf16",
            kernels::compile_kernel(
                "kv_cache_view_to_head_major_bf16",
                kernels::attention::KV_CACHE_VIEW_TO_HEAD_MAJOR_BF16_SRC,
                &[],
            )?,
        );
        registry.insert(
            "cast_bf16_to_f32",
            kernels::compile_kernel(
                "cast_bf16_to_f32",
                kernels::attention::CAST_BF16_TO_F32_SRC,
                &[],
            )?,
        );

        Ok(())
    }

    /// Invoke `f` with an immutable reference to a named kernel, if the
    /// registry contains it. Returns `None` when the key is missing; the
    /// lock is released before returning.
    pub fn with_kernel<R>(&self, name: &str, f: impl FnOnce(&kernels::Kernel) -> R) -> Option<R> {
        let guard = self.kernel_registry.lock().ok()?;
        guard.get(name).map(f)
    }

    /// Check whether a given ONNX operator can be dispatched to GPU.
    pub fn supports_op(op_type: &str) -> bool {
        matches!(op_type, "MatMul" | "Gemm" | "MatMulInteger" | "Conv")
    }
}
