// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `Gather` GPU operator — the embedding-lookup kernel.
//!
//! `Gather(data, indices, axis=0)` copies the rows of `data` selected by
//! `indices` into a contiguous output tensor. For embedding lookup on
//! modern transformers this is the very first operator of the forward
//! pass:
//!
//! - `data` = embedding table, shape `[vocab_size, hidden_size]`
//!   (`BFloat16` for Gemma 4, `Float` for small-scale tests).
//! - `indices` = input token IDs, `Int64` tensor of arbitrary rank.
//! - `output` shape = `indices.shape ++ data.shape[1..]`.
//!
//! Only `axis == 0` is supported in v1 — this is the shape Gemma / Llama
//! / Qwen all emit. Other axes and dtypes return
//! [`CudaError::RuntimeError`] *before* any kernel launch, so the
//! dispatcher can fall back to a CPU path without tearing down the device
//! state.
//!
//! ## Kernel layout (design D5)
//!
//! One thread block per output row. Threads inside the block cooperate
//! to copy the `hidden_size` elements of that row from `data[idx]` into
//! the output, where `idx = indices[row]` (clamped to `[0, vocab_size)`
//! — out-of-range indices read row 0 rather than invoking undefined
//! behaviour; this matches the safetensors path, which does not validate
//! token IDs).
//!
//! Block size is `min(hidden_size, 1024)`, the maximum CUDA block size
//! on all current hardware. Larger hidden dimensions are handled by the
//! in-kernel grid-stride loop over `threadIdx.x`.

extern crate alloc;

use alloc::string::ToString;
use alloc::vec::Vec;

use super::{launch_kernel, Kernel};
use crate::cuda::gpu_executor::DeviceTensor;
use crate::cuda::{CudaError, CudaRuntime};
use crate::tensor::DataType;

// ── Kernel sources ─────────────────────────────────────────────────

/// `gather_f32`: copy `indices[row]`-th row of a `[vocab, hidden]` F32
/// table into the row-th output row. One block per output row.
pub const GATHER_F32_SRC: &str = r#"
extern "C" __global__ void gather_f32(
    const float* __restrict__ data,
    const long long* __restrict__ indices,
    float* __restrict__ output,
    int num_rows,
    int vocab_size,
    int hidden_size
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;
    long long idx = indices[row];
    if (idx < 0 || idx >= (long long)vocab_size) idx = 0;
    const float* src = data + (size_t)idx * (size_t)hidden_size;
    float*       dst = output + (size_t)row * (size_t)hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}
"#;

/// `gather_bf16`: BF16 variant using `__nv_bfloat16`. Copy is a pure
/// memory move — no arithmetic — so we do not need the BF16 intrinsics.
pub const GATHER_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void gather_bf16(
    const __nv_bfloat16* __restrict__ data,
    const long long* __restrict__ indices,
    __nv_bfloat16* __restrict__ output,
    int num_rows,
    int vocab_size,
    int hidden_size
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;
    long long idx = indices[row];
    if (idx < 0 || idx >= (long long)vocab_size) idx = 0;
    const __nv_bfloat16* src = data + (size_t)idx * (size_t)hidden_size;
    __nv_bfloat16*       dst = output + (size_t)row * (size_t)hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}
"#;

// ── Public wrapper ─────────────────────────────────────────────────

/// GPU implementation of ONNX `Gather` for `axis == 0`.
///
/// # Parameters
/// - `data`: source tensor, must be rank ≥ 2 with dtype `Float` or
///   `BFloat16`. Shape is treated as `[vocab, tail_dims...]`; the tail
///   dimensions are flattened into a single row-size count.
/// - `indices`: `Int64` tensor of arbitrary rank. Its shape becomes the
///   leading dims of the output.
/// - `axis`: must be `0`. Non-zero values return
///   [`CudaError::RuntimeError`] without touching the device.
///
/// # Output
/// A new [`DeviceTensor`] of shape `indices.shape ++ data.shape[1..]`
/// and dtype matching `data.dtype`.
///
/// # Errors
/// - `axis != 0` — "gather_gpu: only axis=0 supported"
/// - `indices.dtype != Int64` — "gather_gpu: indices must be Int64"
/// - `data.dtype` not in `{Float, BFloat16}` — "gather_gpu: unsupported
///   data dtype"
/// - `data.shape.len() < 2` — "gather_gpu: data must be rank >= 2"
/// - kernel not registered in the runtime — `KernelLoadFailed`
/// - kernel launch failed — `KernelLaunchFailed`
///
/// The call is asynchronous on the default stream; use
/// [`super::synchronize`] before reading the output on the host.
pub fn gather_gpu(
    runtime: &CudaRuntime,
    data: &DeviceTensor,
    indices: &DeviceTensor,
    axis: i64,
) -> Result<DeviceTensor, CudaError> {
    if axis != 0 {
        return Err(CudaError::RuntimeError {
            op: "gather_gpu: only axis=0 supported",
            code: -1,
        });
    }
    if indices.dtype != DataType::Int64 {
        return Err(CudaError::RuntimeError {
            op: "gather_gpu: indices must be Int64",
            code: -1,
        });
    }
    let kernel_name = match data.dtype {
        DataType::Float => "gather_f32",
        DataType::BFloat16 => "gather_bf16",
        _ => {
            return Err(CudaError::RuntimeError {
                op: "gather_gpu: unsupported data dtype",
                code: -1,
            });
        }
    };
    if data.shape.len() < 2 {
        return Err(CudaError::RuntimeError {
            op: "gather_gpu: data must be rank >= 2",
            code: -1,
        });
    }

    // Dimensions. `hidden_size` is the product of all non-leading dims
    // of `data` — so `[vocab, hidden]` yields `hidden`, and higher-rank
    // tables degrade sensibly to "row byte count / element_size".
    let vocab_size_i64: i64 = data.shape[0];
    let hidden_size_i64: i64 = data.shape[1..].iter().copied().product::<i64>().max(0);
    let num_rows_i64: i64 = indices.shape.iter().copied().product::<i64>().max(0);
    // An empty vocab with non-empty indices would clamp every index to 0 and
    // read from a zero-byte buffer (OOB device read). Reject up front.
    if vocab_size_i64 == 0 && num_rows_i64 > 0 {
        return Err(CudaError::RuntimeError {
            op: "gather_gpu: data axis 0 (vocab) must be non-empty",
            code: -1,
        });
    }
    // Launch geometry and kernel args are i32. Reject tensors that would
    // silently truncate.
    if vocab_size_i64 > i32::MAX as i64
        || hidden_size_i64 > i32::MAX as i64
        || num_rows_i64 > i32::MAX as i64
    {
        return Err(CudaError::RuntimeError {
            op: "gather_gpu: tensor too large for i32 kernel launch parameters",
            code: -1,
        });
    }
    let vocab_size: i32 = vocab_size_i64 as i32;
    let hidden_size: i32 = hidden_size_i64 as i32;
    let num_rows: i32 = num_rows_i64 as i32;

    // Output shape: indices.shape ++ data.shape[1..].
    let mut out_shape: Vec<i64> = Vec::with_capacity(indices.shape.len() + data.shape.len() - 1);
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&data.shape[1..]);

    let out = DeviceTensor::alloc(out_shape, data.dtype)?;

    // Early out on empty tensors: nothing to launch.
    if num_rows == 0 || hidden_size == 0 {
        return Ok(out);
    }

    // Argument pack — each slot is the address of a local holding the
    // value, per the contract of `launch_kernel`.
    let mut data_ptr = data.buffer.as_mut_ptr();
    let mut indices_ptr = indices.buffer.as_mut_ptr();
    let mut out_ptr = out.buffer.as_mut_ptr();
    let mut num_rows_arg: i32 = num_rows;
    let mut vocab_size_arg: i32 = vocab_size;
    let mut hidden_size_arg: i32 = hidden_size;

    let args: [*mut core::ffi::c_void; 6] = [
        &mut data_ptr as *mut _ as *mut core::ffi::c_void,
        &mut indices_ptr as *mut _ as *mut core::ffi::c_void,
        &mut out_ptr as *mut _ as *mut core::ffi::c_void,
        &mut num_rows_arg as *mut _ as *mut core::ffi::c_void,
        &mut vocab_size_arg as *mut _ as *mut core::ffi::c_void,
        &mut hidden_size_arg as *mut _ as *mut core::ffi::c_void,
    ];

    // One block per output row (design D5), one thread per element up
    // to 1024. Larger `hidden_size` values are handled by the kernel's
    // internal stride loop.
    let block_x: u32 = core::cmp::min(hidden_size as u32, 1024);
    let grid = (num_rows as u32, 1, 1);
    let block = (block_x.max(1), 1, 1);

    runtime
        .with_kernel(kernel_name, |k: &Kernel| {
            launch_kernel(k, grid, block, &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: ("gather_gpu: kernel ".to_string() + kernel_name + " not registered"),
            cuda_result: -1,
        })??;

    Ok(out)
}
