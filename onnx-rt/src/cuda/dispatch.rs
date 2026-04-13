// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU operator dispatch via cuBLAS and cuDNN.
//!
//! Translates ONNX operator calls (with host-side input/output tensors)
//! into GPU-accelerated execution: transfer to device → compute → transfer back.

extern crate alloc;
use alloc::string::String;
use alloc::vec;

use super::ffi;
use super::memory::DeviceBuffer;
use super::{CudaError, CudaRuntime};
use crate::tensor::{DataType, Tensor, TensorShape};

/// Execute a MatMul/Gemm on GPU via cuBLAS, returning the result tensor.
///
/// Handles 2D matrix multiply: C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C_bias.
/// For plain MatMul, use alpha=1.0, beta=0.0, c_bias=None.
#[allow(clippy::too_many_arguments)]
pub fn gpu_gemm(
    runtime: &CudaRuntime,
    a: &Tensor,
    b: &Tensor,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
    c_bias: Option<&Tensor>,
) -> Result<Tensor, CudaError> {
    let a_dims = &a.shape.dims;
    let b_dims = &b.shape.dims;

    if a_dims.len() < 2 || b_dims.len() < 2 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm: need 2D inputs",
            code: -1,
        });
    }

    let (m, k_a) = if trans_a {
        (
            a_dims[a_dims.len() - 1] as usize,
            a_dims[a_dims.len() - 2] as usize,
        )
    } else {
        (
            a_dims[a_dims.len() - 2] as usize,
            a_dims[a_dims.len() - 1] as usize,
        )
    };
    let (k_b, n) = if trans_b {
        (
            b_dims[b_dims.len() - 1] as usize,
            b_dims[b_dims.len() - 2] as usize,
        )
    } else {
        (
            b_dims[b_dims.len() - 2] as usize,
            b_dims[b_dims.len() - 1] as usize,
        )
    };

    if k_a != k_b {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm: k mismatch",
            code: -2,
        });
    }
    let k = k_a;

    let a_bytes = a.raw_data.len();
    let b_bytes = b.raw_data.len();
    let c_bytes = m * n * 4; // f32 output

    let a_buf = DeviceBuffer::alloc(a_bytes)?;
    let b_buf = DeviceBuffer::alloc(b_bytes)?;
    let c_buf = DeviceBuffer::alloc(c_bytes)?;

    a_buf.copy_from_host(&a.raw_data)?;
    b_buf.copy_from_host(&b.raw_data)?;

    // Initialize C with bias or zeros.
    if beta != 0.0 {
        if let Some(bias) = c_bias {
            let bias_elements = bias.shape.total_elements();
            if bias_elements == n {
                // Row-broadcast bias vector to full C matrix.
                let mut c_data = vec![0u8; c_bytes];
                for row in 0..m {
                    let dst_start = row * n * 4;
                    c_data[dst_start..dst_start + n * 4].copy_from_slice(&bias.raw_data[..n * 4]);
                }
                c_buf.copy_from_host(&c_data)?;
            } else if bias.raw_data.len() == c_bytes {
                c_buf.copy_from_host(&bias.raw_data)?;
            } else {
                c_buf.copy_from_host(&vec![0u8; c_bytes])?;
            }
        } else {
            c_buf.copy_from_host(&vec![0u8; c_bytes])?;
        }
    } else {
        c_buf.copy_from_host(&vec![0u8; c_bytes])?;
    }

    // cuBLAS uses column-major. For row-major ONNX tensors:
    //   C_row = alpha * op(A) * op(B) + beta * C
    // becomes in cuBLAS column-major:
    //   C_col = alpha * B^T_adj * A^T_adj + beta * C_col
    // where we swap A↔B and adjust transposes.
    let transa = if trans_b {
        ffi::cublasOperation_t::CUBLAS_OP_T
    } else {
        ffi::cublasOperation_t::CUBLAS_OP_N
    };
    let transb = if trans_a {
        ffi::cublasOperation_t::CUBLAS_OP_T
    } else {
        ffi::cublasOperation_t::CUBLAS_OP_N
    };

    let lda = if trans_b { k as i32 } else { n as i32 };
    let ldb = if trans_a { m as i32 } else { k as i32 };
    let ldc = n as i32;

    // Use GemmEx with the runtime's precision mode for tensor core acceleration.
    let compute_type = runtime.precision.to_cublas_compute_type();
    runtime.cublas.gemm_ex(
        transa,
        transb,
        n as i32,
        m as i32,
        k as i32,
        &alpha as *const f32 as *const core::ffi::c_void,
        &b_buf,
        ffi::cudaDataType_t::CUDA_R_32F,
        lda,
        &a_buf,
        ffi::cudaDataType_t::CUDA_R_32F,
        ldb,
        &beta as *const f32 as *const core::ffi::c_void,
        &c_buf,
        ffi::cudaDataType_t::CUDA_R_32F,
        ldc,
        compute_type,
    )?;

    super::synchronize()?;

    let mut result_bytes = vec![0u8; c_bytes];
    c_buf.copy_to_host(&mut result_bytes)?;

    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![m as i64, n as i64]),
        name: String::new(),
        raw_data: result_bytes,
    })
}

/// Execute an INT8 MatMul on GPU via cublasGemmEx, returning INT32 result tensor.
///
/// C[M,N] (i32) = A[M,K] (i8) * B[K,N] (i8)
/// Uses CUBLAS_COMPUTE_32I with INT32 accumulation.
///
/// Note: cuBLAS INT8 GEMM requires dimensions to be multiples of 4.
/// If dimensions aren't aligned, falls back to None to let CPU handle it.
pub fn gpu_gemm_int8(
    runtime: &CudaRuntime,
    a: &Tensor,
    b: &Tensor,
) -> Result<Option<Tensor>, CudaError> {
    let a_dims = &a.shape.dims;
    let b_dims = &b.shape.dims;

    if a_dims.len() < 2 || b_dims.len() < 2 {
        return Ok(None);
    }

    let m = a_dims[a_dims.len() - 2] as usize;
    let k = a_dims[a_dims.len() - 1] as usize;
    let n = b_dims[b_dims.len() - 1] as usize;

    // cuBLAS INT8 GEMM requires 4-aligned dimensions.
    if !m.is_multiple_of(4) || !n.is_multiple_of(4) || !k.is_multiple_of(4) {
        return Ok(None); // fall back to CPU
    }

    let a_bytes = a.raw_data.len();
    let b_bytes = b.raw_data.len();
    let c_bytes = m * n * 4; // i32 output = 4 bytes per element

    let a_buf = DeviceBuffer::alloc(a_bytes)?;
    let b_buf = DeviceBuffer::alloc(b_bytes)?;
    let c_buf = DeviceBuffer::alloc(c_bytes)?;

    a_buf.copy_from_host(&a.raw_data)?;
    b_buf.copy_from_host(&b.raw_data)?;
    c_buf.copy_from_host(&vec![0u8; c_bytes])?;

    // Row-major → column-major for INT8 GemmEx: same swap trick as sgemm.
    // C_row[M,N] = A_row[M,K] * B_row[K,N] becomes in column-major:
    // C_col^T[N,M] = B_col^T[N,K] * A_col^T[K,M]
    // i.e. GemmEx(OP_N, OP_N, N, M, K, B_raw, ld=N, A_raw, ld=K, C, ld=N)
    let alpha: i32 = 1;
    let beta: i32 = 0;

    runtime.cublas.gemm_ex(
        ffi::cublasOperation_t::CUBLAS_OP_N, // B^T already in memory
        ffi::cublasOperation_t::CUBLAS_OP_N, // A^T already in memory
        n as i32,                            // rows of op(B^T) = N
        m as i32,                            // cols of op(A^T) = M
        k as i32,
        &alpha as *const i32 as *const core::ffi::c_void,
        &b_buf, // "A" in cuBLAS = B_row
        ffi::cudaDataType_t::CUDA_R_8I,
        n as i32, // lda = N
        &a_buf,   // "B" in cuBLAS = A_row
        ffi::cudaDataType_t::CUDA_R_8I,
        k as i32, // ldb = K
        &beta as *const i32 as *const core::ffi::c_void,
        &c_buf,
        ffi::cudaDataType_t::CUDA_R_32I,
        n as i32, // ldc = N
        ffi::cublasComputeType_t::CUBLAS_COMPUTE_32I,
    )?;

    super::synchronize()?;

    let mut result_bytes = vec![0u8; c_bytes];
    c_buf.copy_to_host(&mut result_bytes)?;

    Ok(Some(Tensor {
        data_type: DataType::Int32,
        shape: TensorShape::new(vec![m as i64, n as i64]),
        name: String::new(),
        raw_data: result_bytes,
    }))
}

/// Execute an FP8 GEMM on GPU via cuBLASLt, returning f32 result tensor.
///
/// C[M,N] (f32) = A[M,K] (fp8) * B[K,N] (fp8)
/// Uses cublasLtMatmul since cublasGemmEx does not support FP8 on Blackwell.
///
/// `fp8_type` selects E4M3 or E5M2 format.
///
/// Returns None if dimensions aren't suitable.
pub fn gpu_gemm_fp8(
    runtime: &CudaRuntime,
    a: &Tensor,
    b: &Tensor,
    fp8_type: ffi::cudaDataType_t,
) -> Result<Option<Tensor>, CudaError> {
    let a_dims = &a.shape.dims;
    let b_dims = &b.shape.dims;

    if a_dims.len() < 2 || b_dims.len() < 2 {
        return Ok(None);
    }

    let m = a_dims[a_dims.len() - 2] as usize;
    let k = a_dims[a_dims.len() - 1] as usize;
    let n = b_dims[b_dims.len() - 1] as usize;

    // FP8 works best with 16-aligned dimensions, but let cuBLASLt handle smaller.
    let a_bytes = a.raw_data.len(); // 1 byte per FP8 element
    let b_bytes = b.raw_data.len();
    let c_bytes = m * n * 4; // f32 output

    let a_buf = DeviceBuffer::alloc(a_bytes)?;
    let b_buf = DeviceBuffer::alloc(b_bytes)?;
    let c_buf = DeviceBuffer::alloc(c_bytes)?;

    a_buf.copy_from_host(&a.raw_data)?;
    b_buf.copy_from_host(&b.raw_data)?;
    c_buf.copy_from_host(&vec![0u8; c_bytes])?;

    // Create cuBLASLt descriptors.
    // Column-major trick: for row-major C = A * B, we compute
    // col-major C^T = B^T * A^T (swap A↔B, same memory layout trick).
    let mut matmul_desc: ffi::cublasLtMatmulDesc_t = core::ptr::null_mut();
    let err = unsafe {
        ffi::cublasLtMatmulDescCreate(
            &mut matmul_desc,
            ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            ffi::cudaDataType_t::CUDA_R_32F,
        )
    };
    if err != ffi::CUBLAS_STATUS_SUCCESS {
        return Err(CudaError::BlasError {
            op: "LtMatmulDescCreate",
            code: err,
        });
    }

    let mut b_layout: ffi::cublasLtMatrixLayout_t = core::ptr::null_mut();
    let mut a_layout: ffi::cublasLtMatrixLayout_t = core::ptr::null_mut();
    let mut c_layout: ffi::cublasLtMatrixLayout_t = core::ptr::null_mut();

    // B^T is [N, K] in column-major (B row-major [K, N] reinterpreted)
    let err = unsafe {
        ffi::cublasLtMatrixLayoutCreate(&mut b_layout, fp8_type, n as u64, k as u64, n as i64)
    };
    if err != ffi::CUBLAS_STATUS_SUCCESS {
        unsafe {
            ffi::cublasLtMatmulDescDestroy(matmul_desc);
        }
        return Err(CudaError::BlasError {
            op: "LtLayoutCreate B",
            code: err,
        });
    }

    // A^T is [K, M] in column-major (A row-major [M, K] reinterpreted)
    let err = unsafe {
        ffi::cublasLtMatrixLayoutCreate(&mut a_layout, fp8_type, k as u64, m as u64, k as i64)
    };
    if err != ffi::CUBLAS_STATUS_SUCCESS {
        unsafe {
            ffi::cublasLtMatrixLayoutDestroy(b_layout);
            ffi::cublasLtMatmulDescDestroy(matmul_desc);
        }
        return Err(CudaError::BlasError {
            op: "LtLayoutCreate A",
            code: err,
        });
    }

    // C^T is [N, M] in column-major → C row-major [M, N]
    let err = unsafe {
        ffi::cublasLtMatrixLayoutCreate(
            &mut c_layout,
            ffi::cudaDataType_t::CUDA_R_32F,
            n as u64,
            m as u64,
            n as i64,
        )
    };
    if err != ffi::CUBLAS_STATUS_SUCCESS {
        unsafe {
            ffi::cublasLtMatrixLayoutDestroy(a_layout);
            ffi::cublasLtMatrixLayoutDestroy(b_layout);
            ffi::cublasLtMatmulDescDestroy(matmul_desc);
        }
        return Err(CudaError::BlasError {
            op: "LtLayoutCreate C",
            code: err,
        });
    }

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let err = unsafe {
        ffi::cublasLtMatmul(
            runtime.cublas_lt.raw(),
            matmul_desc,
            &alpha as *const f32 as *const core::ffi::c_void,
            b_buf.as_ptr(),
            b_layout,
            a_buf.as_ptr(),
            a_layout,
            &beta as *const f32 as *const core::ffi::c_void,
            c_buf.as_ptr(),
            c_layout,
            c_buf.as_mut_ptr(),
            c_layout,
            core::ptr::null(),     // NULL algo = default
            core::ptr::null_mut(), // no workspace
            0,                     // workspace size
            core::ptr::null_mut(), // default stream
        )
    };

    // Clean up descriptors.
    unsafe {
        ffi::cublasLtMatrixLayoutDestroy(c_layout);
        ffi::cublasLtMatrixLayoutDestroy(a_layout);
        ffi::cublasLtMatrixLayoutDestroy(b_layout);
        ffi::cublasLtMatmulDescDestroy(matmul_desc);
    }

    if err != ffi::CUBLAS_STATUS_SUCCESS {
        return Err(CudaError::BlasError {
            op: "cublasLtMatmul FP8",
            code: err,
        });
    }

    super::synchronize()?;

    let mut result_bytes = vec![0u8; c_bytes];
    c_buf.copy_to_host(&mut result_bytes)?;

    Ok(Some(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![m as i64, n as i64]),
        name: String::new(),
        raw_data: result_bytes,
    }))
}
