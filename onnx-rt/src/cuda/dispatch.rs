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
use crate::tensor::{f32_to_bf16, DataType, Tensor, TensorShape};

/// Output dtype selection for [`gpu_gemm`].
struct GemmDtype {
    is_bf16: bool,
    elem_size: usize,
    io_dtype: ffi::cudaDataType_t,
    out_data_type: DataType,
}

impl GemmDtype {
    fn pick(a: &Tensor, b: &Tensor) -> Self {
        let is_bf16 = a.data_type == DataType::BFloat16 && b.data_type == DataType::BFloat16;
        if is_bf16 {
            Self {
                is_bf16,
                elem_size: 2,
                io_dtype: ffi::cudaDataType_t::CUDA_R_16BF,
                out_data_type: DataType::BFloat16,
            }
        } else {
            Self {
                is_bf16,
                elem_size: 4,
                io_dtype: ffi::cudaDataType_t::CUDA_R_32F,
                out_data_type: DataType::Float,
            }
        }
    }
}

/// Extract the trailing `[M, K]` (or transposed) dims from a Gemm input.
fn gemm_mk_dims(dims: &[i64], transpose: bool) -> (usize, usize) {
    let last = dims.len() - 1;
    if transpose {
        (dims[last] as usize, dims[last - 1] as usize)
    } else {
        (dims[last - 1] as usize, dims[last] as usize)
    }
}

/// Re-encode a bias byte slice into the cuBLAS output dtype.
///
/// For BF16 output: F32 bias is converted; BF16 bias is passed through;
/// anything else is an error. For F32 output the bias is returned as-is
/// (since the executor only routes F32/BF16 to this path).
fn encode_bias_for_output(bias: &Tensor, is_bf16: bool) -> Result<alloc::vec::Vec<u8>, CudaError> {
    if !is_bf16 {
        return Ok(bias.raw_data.clone());
    }
    match bias.data_type {
        DataType::BFloat16 => Ok(bias.raw_data.clone()),
        DataType::Float => {
            let f32_vals: alloc::vec::Vec<f32> = bias
                .raw_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(f32_to_bf16(&f32_vals))
        }
        _ => Err(CudaError::RuntimeError {
            op: "gpu_gemm: unsupported bias dtype for BF16 output",
            code: -3,
        }),
    }
}

/// Initialise the cuBLAS C buffer to either a (broadcast) bias or zeros.
///
/// Mirrors the original semantics:
/// - `beta == 0` or no bias → zero-fill.
/// - bias is `[N]` (row vector) → broadcast across `M` rows.
/// - bias byte length matches `c_bytes` → copy as-is.
/// - any other mismatch → zero-fill (silent fallback).
#[allow(clippy::too_many_arguments)]
fn init_c_buffer(
    c_buf: &DeviceBuffer,
    c_bias: Option<&Tensor>,
    beta: f32,
    m: usize,
    n: usize,
    c_bytes: usize,
    elem_size: usize,
    is_bf16: bool,
) -> Result<(), CudaError> {
    if beta == 0.0 {
        return c_buf.copy_from_host(&vec![0u8; c_bytes]);
    }
    let Some(bias) = c_bias else {
        return c_buf.copy_from_host(&vec![0u8; c_bytes]);
    };

    let bias_bytes = encode_bias_for_output(bias, is_bf16)?;
    let bias_elements = bias.shape.total_elements();
    if bias_elements == n {
        // Row-broadcast bias vector to full C matrix.
        let row_bytes = n * elem_size;
        let mut c_data = vec![0u8; c_bytes];
        for row in 0..m {
            let dst_start = row * row_bytes;
            c_data[dst_start..dst_start + row_bytes].copy_from_slice(&bias_bytes[..row_bytes]);
        }
        c_buf.copy_from_host(&c_data)
    } else if bias_bytes.len() == c_bytes {
        c_buf.copy_from_host(&bias_bytes)
    } else {
        c_buf.copy_from_host(&vec![0u8; c_bytes])
    }
}

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

    let (m, k_a) = gemm_mk_dims(a_dims, trans_a);
    let (k_b, n) = gemm_mk_dims(b_dims, trans_b);

    if k_a != k_b {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm: k mismatch",
            code: -2,
        });
    }
    let k = k_a;

    // Detect BF16 native path: BF16 in → BF16 out, f32 accumulation via
    // CUBLAS_COMPUTE_32F_FAST_16BF. Falls back to f32 path for all other
    // (and mixed) input combinations — the executor routes non-f32/non-BF16
    // combinations to the CPU before reaching this function.
    let dtype = GemmDtype::pick(a, b);

    let a_bytes = a.raw_data.len();
    let b_bytes = b.raw_data.len();
    let c_bytes = m * n * dtype.elem_size;

    let a_buf = DeviceBuffer::alloc(a_bytes)?;
    let b_buf = DeviceBuffer::alloc(b_bytes)?;
    let c_buf = DeviceBuffer::alloc(c_bytes)?;

    a_buf.copy_from_host(&a.raw_data)?;
    b_buf.copy_from_host(&b.raw_data)?;

    // Initialize C with bias or zeros. For BF16 dispatch the bias must be
    // materialized in BF16 inside the output buffer before the cuBLAS call
    // — convert from f32 if necessary so callers can pass either dtype.
    init_c_buffer(
        &c_buf,
        c_bias,
        beta,
        m,
        n,
        c_bytes,
        dtype.elem_size,
        dtype.is_bf16,
    )?;

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
    // For BF16 inputs we hard-pin the compute type to 32F_FAST_16BF (f32
    // accumulation, BF16 tensor cores) regardless of runtime.precision, since
    // BF16 inputs cannot be computed in a non-BF16 tensor-core mode anyway.
    let compute_type = if dtype.is_bf16 {
        ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
    } else {
        runtime.precision.to_cublas_compute_type()
    };
    runtime.cublas.gemm_ex(
        transa,
        transb,
        n as i32,
        m as i32,
        k as i32,
        &alpha as *const f32 as *const core::ffi::c_void,
        &b_buf,
        dtype.io_dtype,
        lda,
        &a_buf,
        dtype.io_dtype,
        ldb,
        &beta as *const f32 as *const core::ffi::c_void,
        &c_buf,
        dtype.io_dtype,
        ldc,
        compute_type,
    )?;

    super::synchronize()?;

    let mut result_bytes = vec![0u8; c_bytes];
    c_buf.copy_to_host(&mut result_bytes)?;

    Ok(Tensor {
        data_type: dtype.out_data_type,
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

// ── Strided batched GEMM (transformer-gpu-kernels-v1 §6.1) ──────────

/// Strided batched mixed-precision GEMM operating on `DeviceTensor`s.
///
/// For every batch slice `i in 0..batch_count`, computes
///
/// ```text
/// C[i] = op(A[i]) * op(B[i])
/// ```
///
/// where `op` applies optional transposition. All three operands are
/// laid out as contiguous row-major 3-D tensors:
///
/// - `a`: `[batch, M, K]` if `trans_a == false`, else `[batch, K, M]`
/// - `b`: `[batch, K, N]` if `trans_b == false`, else `[batch, N, K]`
/// - output: `[batch, M, N]` row-major in the chosen `out_dtype`
///
/// The per-batch element strides are derived from the tensor shapes
/// (no padding). Compute is always F32 — for BF16 inputs this means
/// `CUBLAS_COMPUTE_32F` with BF16 tensor cores; for F32 inputs it
/// uses the runtime's configured precision (F32 or TF32).
///
/// # Errors
/// - `a.dtype != b.dtype` — `RuntimeError`
/// - dtype not in `{Float, BFloat16}` — `RuntimeError`
/// - `out_dtype` not in `{Float, BFloat16}` — `RuntimeError`
/// - rank != 3 on either input — `RuntimeError`
/// - mismatched batch dim or `K` — `RuntimeError`
/// - cuBLAS error — `BlasError`
///
/// The call synchronizes on the default stream before returning so
/// callers can chain it with non-cuBLAS kernels safely.
/// Map a Float/BFloat16 `DataType` to its cuBLAS dtype + element size.
///
/// `op` is embedded into the error message so the dispatcher's two callers
/// (`a/b dtype` and `out_dtype`) get distinct labels.
fn map_gemm_dtype(
    dt: DataType,
    op: &'static str,
) -> Result<(ffi::cudaDataType_t, usize), CudaError> {
    match dt {
        DataType::BFloat16 => Ok((ffi::cudaDataType_t::CUDA_R_16BF, 2)),
        DataType::Float => Ok((ffi::cudaDataType_t::CUDA_R_32F, 4)),
        _ => Err(CudaError::RuntimeError { op, code: -1 }),
    }
}

/// Validated batched-GEMM shape parameters.
struct BatchedGemmShape {
    batch_i64: i64,
    m_i64: i64,
    n_i64: i64,
    k_i64: i64,
}

impl BatchedGemmShape {
    fn validate(
        a: &super::gpu_executor::DeviceTensor,
        b: &super::gpu_executor::DeviceTensor,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<Self, CudaError> {
        if a.shape.len() != 3 || b.shape.len() != 3 {
            return Err(CudaError::RuntimeError {
                op: "gpu_gemm_strided_batched_ex: both inputs must be rank-3",
                code: -1,
            });
        }
        let batch_i64 = a.shape[0];
        if b.shape[0] != batch_i64 {
            return Err(CudaError::RuntimeError {
                op: "gpu_gemm_strided_batched_ex: batch dimensions differ",
                code: -1,
            });
        }
        let (m_i64, k_a_i64) = if trans_a {
            (a.shape[2], a.shape[1])
        } else {
            (a.shape[1], a.shape[2])
        };
        let (k_b_i64, n_i64) = if trans_b {
            (b.shape[2], b.shape[1])
        } else {
            (b.shape[1], b.shape[2])
        };
        if k_a_i64 != k_b_i64 {
            return Err(CudaError::RuntimeError {
                op: "gpu_gemm_strided_batched_ex: K dimensions differ",
                code: -1,
            });
        }
        if batch_i64 < 0 || m_i64 < 0 || n_i64 < 0 || k_a_i64 < 0 {
            return Err(CudaError::RuntimeError {
                op: "gpu_gemm_strided_batched_ex: negative dimension",
                code: -1,
            });
        }
        Ok(Self {
            batch_i64,
            m_i64,
            n_i64,
            k_i64: k_a_i64,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_gemm_strided_batched_ex(
    runtime: &CudaRuntime,
    a: &super::gpu_executor::DeviceTensor,
    b: &super::gpu_executor::DeviceTensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: DataType,
) -> Result<super::gpu_executor::DeviceTensor, CudaError> {
    if a.dtype != b.dtype {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm_strided_batched_ex: a/b dtypes must match",
            code: -1,
        });
    }
    let (a_cuda, _a_elem_size) = map_gemm_dtype(
        a.dtype,
        "gpu_gemm_strided_batched_ex: a/b must be Float or BFloat16",
    )?;
    let (c_cuda, c_elem_size) = map_gemm_dtype(
        out_dtype,
        "gpu_gemm_strided_batched_ex: out_dtype must be Float or BFloat16",
    )?;
    let shape = BatchedGemmShape::validate(a, b, trans_a, trans_b)?;
    let batch_i64 = shape.batch_i64;
    let m_i64 = shape.m_i64;
    let n_i64 = shape.n_i64;
    let k_a_i64 = shape.k_i64;

    let batch = batch_i64 as i32;
    let m = m_i64 as i32;
    let n = n_i64 as i32;
    let k = k_a_i64 as i32;

    let out_shape = vec![batch_i64, m_i64, n_i64];
    let out = super::gpu_executor::DeviceTensor::alloc(out_shape, out_dtype)?;

    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return Ok(out);
    }

    // cuBLAS column-major mapping. Same swap trick as `gpu_gemm`: pass
    // (B, A) instead of (A, B), and reverse the transpose flags relative
    // to the row-major view. Per-batch element strides are M*K, K*N,
    // M*N (row-major contiguous).
    let stride_a = a.shape[1] * a.shape[2];
    let stride_b = b.shape[1] * b.shape[2];
    let stride_c = m_i64 * n_i64;

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

    let lda = if trans_b { k } else { n };
    let ldb = if trans_a { m } else { k };
    let ldc = n;

    // Compute type. BF16 inputs always use 32F_FAST_16BF (BF16 tensor
    // cores, F32 accumulation). F32 inputs honor the runtime precision
    // mode (F32 vs TF32) — same convention as `gpu_gemm`.
    let compute_type = if a.dtype == DataType::BFloat16 {
        ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
    } else {
        runtime.precision.to_cublas_compute_type()
    };

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    runtime.cublas.gemm_strided_batched_ex(
        transa,
        transb,
        n,
        m,
        k,
        &alpha as *const f32 as *const core::ffi::c_void,
        &b.buffer,
        a_cuda,
        lda,
        stride_b,
        &a.buffer,
        a_cuda,
        ldb,
        stride_a,
        &beta as *const f32 as *const core::ffi::c_void,
        &out.buffer,
        c_cuda,
        ldc,
        stride_c,
        batch,
        compute_type,
    )?;

    super::synchronize()?;
    let _ = c_elem_size; // computed for future host-side debug logging
    Ok(out)
}
