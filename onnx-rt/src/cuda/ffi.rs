// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Raw `extern "C"` FFI bindings to CUDA runtime, cuBLAS, and cuDNN.
//!
//! Minimal surface: only the ~20 functions needed for ONNX inference dispatch.
//! These are hand-rolled instead of using `cuda-sys` to stay `#![no_std]`-compatible
//! and avoid pulling in 2000+ unused declarations.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

// ── CUDA Runtime types ──────────────────────────────────────────────

/// CUDA error codes (subset).
pub type cudaError_t = i32;
pub const CUDA_SUCCESS: cudaError_t = 0;

/// Memory copy direction.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
}

/// Device properties (minimal subset).
#[repr(C)]
#[derive(Debug)]
pub struct cudaDeviceProp {
    pub name: [u8; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub clock_rate: i32,
    pub total_const_mem: usize,
    pub major: i32,
    pub minor: i32,
    // Pad to the full struct size (varies by CUDA version, ~800+ bytes).
    // We only read fields above; the kernel fills the rest.
    pub _padding: [u8; 512],
}

impl Default for cudaDeviceProp {
    fn default() -> Self {
        // Safety: cudaDeviceProp is a C struct of integers and byte arrays.
        // Zero-initialized is a valid (empty) state.
        unsafe { core::mem::zeroed() }
    }
}

// ── cuBLAS types ────────────────────────────────────────────────────

/// Opaque cuBLAS handle.
pub type cublasHandle_t = *mut core::ffi::c_void;

/// cuBLAS status codes.
pub type cublasStatus_t = i32;
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

/// cuBLAS operation (transpose).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0, // No transpose
    CUBLAS_OP_T = 1, // Transpose
    CUBLAS_OP_C = 2, // Conjugate transpose
}

/// cuBLAS compute type for GemmEx.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_32I = 72,
}

/// CUDA data type.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaDataType_t {
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_R_16F = 2,
    CUDA_R_8I = 3,
    CUDA_R_8U = 8,
    CUDA_R_32I = 10,
    CUDA_R_16BF = 14,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29,
}

// ── cuBLASLt types ──────────────────────────────────────────────────

/// Opaque cuBLASLt handles.
pub type cublasLtHandle_t = *mut core::ffi::c_void;
pub type cublasLtMatmulDesc_t = *mut core::ffi::c_void;
pub type cublasLtMatrixLayout_t = *mut core::ffi::c_void;

// ── cuDNN types ─────────────────────────────────────────────────────

/// Opaque cuDNN handle.
pub type cudnnHandle_t = *mut core::ffi::c_void;

/// cuDNN status codes.
pub type cudnnStatus_t = i32;
pub const CUDNN_STATUS_SUCCESS: cudnnStatus_t = 0;

/// Opaque descriptor handles.
pub type cudnnTensorDescriptor_t = *mut core::ffi::c_void;
pub type cudnnFilterDescriptor_t = *mut core::ffi::c_void;
pub type cudnnConvolutionDescriptor_t = *mut core::ffi::c_void;

/// cuDNN data type.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnDataType_t {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_BFLOAT16 = 9,
}

/// cuDNN tensor format.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
}

/// cuDNN convolution mode.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnConvolutionMode_t {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
}

/// cuDNN convolution forward algorithm.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnConvolutionFwdAlgo_t {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
}

// ── CUDA Runtime FFI ────────────────────────────────────────────────

#[link(name = "cudart")]
extern "C" {
    pub fn cudaGetDeviceCount(count: *mut i32) -> cudaError_t;
    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: i32) -> cudaError_t;
    pub fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> cudaError_t;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    pub fn cudaSetDevice(device: i32) -> cudaError_t;
    pub fn cudaMalloc(devPtr: *mut *mut core::ffi::c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut core::ffi::c_void) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaRuntimeGetVersion(runtimeVersion: *mut i32) -> cudaError_t;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const u8;
}

// ── cuBLAS FFI ──────────────────────────────────────────────────────

#[link(name = "cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const core::ffi::c_void,
        lda: i32,
        b: *const core::ffi::c_void,
        ldb: i32,
        beta: *const f32,
        c: *mut core::ffi::c_void,
        ldc: i32,
    ) -> cublasStatus_t;
    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const core::ffi::c_void,
        a: *const core::ffi::c_void,
        a_type: cudaDataType_t,
        lda: i32,
        b: *const core::ffi::c_void,
        b_type: cudaDataType_t,
        ldb: i32,
        beta: *const core::ffi::c_void,
        c: *mut core::ffi::c_void,
        c_type: cudaDataType_t,
        ldc: i32,
        compute_type: cublasComputeType_t,
        algo: i32, // cublasGemmAlgo_t, -1 = default
    ) -> cublasStatus_t;
}

// ── cuBLASLt FFI ────────────────────────────────────────────────────

#[link(name = "cublasLt")]
extern "C" {
    pub fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> cublasStatus_t;
    pub fn cublasLtDestroy(handle: cublasLtHandle_t) -> cublasStatus_t;
    pub fn cublasLtMatmulDescCreate(
        desc: *mut cublasLtMatmulDesc_t,
        compute_type: cublasComputeType_t,
        scale_type: cudaDataType_t,
    ) -> cublasStatus_t;
    pub fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> cublasStatus_t;
    pub fn cublasLtMatrixLayoutCreate(
        layout: *mut cublasLtMatrixLayout_t,
        data_type: cudaDataType_t,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> cublasStatus_t;
    pub fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> cublasStatus_t;
    pub fn cublasLtMatmul(
        handle: cublasLtHandle_t,
        matmul_desc: cublasLtMatmulDesc_t,
        alpha: *const core::ffi::c_void,
        a: *const core::ffi::c_void,
        a_layout: cublasLtMatrixLayout_t,
        b: *const core::ffi::c_void,
        b_layout: cublasLtMatrixLayout_t,
        beta: *const core::ffi::c_void,
        c: *const core::ffi::c_void,
        c_layout: cublasLtMatrixLayout_t,
        d: *mut core::ffi::c_void,
        d_layout: cublasLtMatrixLayout_t,
        algo: *const core::ffi::c_void, // NULL for default
        workspace: *mut core::ffi::c_void,
        workspace_size: usize,
        stream: *mut core::ffi::c_void, // cudaStream_t = NULL for default
    ) -> cublasStatus_t;
}

// ── cuDNN FFI ───────────────────────────────────────────────────────

#[link(name = "cudnn")]
extern "C" {
    pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;

    pub fn cudnnCreateTensorDescriptor(desc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyTensorDescriptor(desc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetTensor4dDescriptor(
        desc: cudnnTensorDescriptor_t,
        format: cudnnTensorFormat_t,
        data_type: cudnnDataType_t,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> cudnnStatus_t;

    pub fn cudnnCreateFilterDescriptor(desc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyFilterDescriptor(desc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetFilter4dDescriptor(
        desc: cudnnFilterDescriptor_t,
        data_type: cudnnDataType_t,
        format: cudnnTensorFormat_t,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> cudnnStatus_t;

    pub fn cudnnCreateConvolutionDescriptor(
        desc: *mut cudnnConvolutionDescriptor_t,
    ) -> cudnnStatus_t;
    pub fn cudnnDestroyConvolutionDescriptor(desc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetConvolution2dDescriptor(
        desc: cudnnConvolutionDescriptor_t,
        pad_h: i32,
        pad_w: i32,
        u: i32, // vertical stride
        v: i32, // horizontal stride
        dilation_h: i32,
        dilation_w: i32,
        mode: cudnnConvolutionMode_t,
        compute_type: cudnnDataType_t,
    ) -> cudnnStatus_t;

    pub fn cudnnConvolutionForward(
        handle: cudnnHandle_t,
        alpha: *const core::ffi::c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const core::ffi::c_void,
        w_desc: cudnnFilterDescriptor_t,
        w: *const core::ffi::c_void,
        conv_desc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        workspace: *mut core::ffi::c_void,
        workspace_size: usize,
        beta: *const core::ffi::c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut core::ffi::c_void,
    ) -> cudnnStatus_t;
}
