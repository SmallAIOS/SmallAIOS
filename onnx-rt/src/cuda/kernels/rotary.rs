// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `RotaryEmbedding` (RoPE) GPU operator.
//!
//! Applies rotary positional embedding to a Q or K tensor of shape
//! `[B, H, Sq, head_dim]`, using precomputed cos/sin tables of shape
//! `[max_seq_len, head_dim / 2]`. Both BF16 and F32 variants are
//! provided. The kernel supports both pair layouts:
//!
//! - **Interleaved** (`(x[2i], x[2i+1])`): the textbook RoPE layout,
//!   used by LLaMA-style models.
//! - **Split-half** (`(x[i], x[i + half])`): the layout Gemma 4 emits
//!   (see `model_loader::gemma::rotary_embedding(.., interleaved=false)`).
//!
//! Both layouts produce numerically identical outputs *given the same
//! cos/sin tables* — they only differ in how the input pairs are
//! addressed. The kernel branches on a uniform `interleaved` argument,
//! so warps never diverge.
//!
//! ## Position handling
//!
//! `position` is interpreted as a **starting** position. Sequence
//! element `qi` (0-indexed within each row's `Sq` axis) reads cos/sin
//! from row `position + qi` of the table. This handles both:
//!
//! - Prefill (`Sq = N`, `position = 0`) → rows `0..N`
//! - Decode (`Sq = 1`, `position = current_step`) → row `current_step`
//!
//! ## Kernel layout (design D7)
//!
//! One CUDA block per `(b, h, qi)` row. Threads inside the block
//! cooperate via a grid-stride loop to apply rotation to each
//! `head_dim / 2` pair of elements. Block size is `min(half, 1024)`.
//! Elements with index `>= rotary_dim` (when partial RoPE is used) are
//! left unchanged. v1 only supports full-head RoPE
//! (`rotary_dim == head_dim`); the wrapper rejects partial cases.

extern crate alloc;

use alloc::string::ToString;

use super::{launch_kernel, Kernel};
use crate::cuda::gpu_executor::DeviceTensor;
use crate::cuda::{CudaError, CudaRuntime};
use crate::tensor::DataType;

// ── Kernel sources ─────────────────────────────────────────────────

/// `rotary_f32`: F32 input / cos / sin / output.
///
/// Both pair layouts are handled by branching on `interleaved` (a
/// uniform argument — no warp divergence).
pub const ROTARY_F32_SRC: &str = r#"
extern "C" __global__ void rotary_f32(
    const float* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ y,
    int seq_len,
    int head_dim,
    int position,
    int interleaved
) {
    int row = blockIdx.x;
    int qi  = row % seq_len;
    int pos = position + qi;
    int half = head_dim >> 1;

    const float* x_row  = x + (size_t)row * (size_t)head_dim;
    float*       y_row  = y + (size_t)row * (size_t)head_dim;
    const float* cos_row = cos_table + (size_t)pos * (size_t)half;
    const float* sin_row = sin_table + (size_t)pos * (size_t)half;

    for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
        float c = cos_row[pair];
        float s = sin_row[pair];
        int i0, i1;
        if (interleaved) {
            i0 = 2 * pair;
            i1 = 2 * pair + 1;
        } else {
            i0 = pair;
            i1 = pair + half;
        }
        float x0 = x_row[i0];
        float x1 = x_row[i1];
        y_row[i0] = c * x0 - s * x1;
        y_row[i1] = s * x0 + c * x1;
    }
}
"#;

/// `rotary_bf16`: BF16 input / output, F32 cos / sin tables. Tables
/// stay in F32 because (1) they are tiny relative to weights and (2)
/// the rotation is sensitive to the trig precision.
pub const ROTARY_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void rotary_bf16(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ y,
    int seq_len,
    int head_dim,
    int position,
    int interleaved
) {
    int row = blockIdx.x;
    int qi  = row % seq_len;
    int pos = position + qi;
    int half = head_dim >> 1;

    const __nv_bfloat16* x_row = x + (size_t)row * (size_t)head_dim;
    __nv_bfloat16*       y_row = y + (size_t)row * (size_t)head_dim;
    const float* cos_row = cos_table + (size_t)pos * (size_t)half;
    const float* sin_row = sin_table + (size_t)pos * (size_t)half;

    for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
        float c = cos_row[pair];
        float s = sin_row[pair];
        int i0, i1;
        if (interleaved) {
            i0 = 2 * pair;
            i1 = 2 * pair + 1;
        } else {
            i0 = pair;
            i1 = pair + half;
        }
        float x0 = __bfloat162float(x_row[i0]);
        float x1 = __bfloat162float(x_row[i1]);
        y_row[i0] = __float2bfloat16(c * x0 - s * x1);
        y_row[i1] = __float2bfloat16(s * x0 + c * x1);
    }
}
"#;

// ── Public wrapper ─────────────────────────────────────────────────

/// GPU implementation of RoPE on a `[B, H, Sq, head_dim]` tensor.
///
/// # Parameters
/// - `x`: input tensor of shape `[B, H, Sq, head_dim]`. Dtype must be
///   `Float` or `BFloat16`. `head_dim` must be even.
/// - `cos_table`, `sin_table`: precomputed tables of shape
///   `[max_seq_len, head_dim / 2]`, dtype `Float`. Row `position + qi`
///   is read for sequence element `qi`.
/// - `position`: starting absolute position (the position index of the
///   first sequence element in `x`). Sequence element `qi` uses table
///   row `position + qi`. For decode this is the current token index;
///   for prefill it is `0`.
/// - `interleaved`: pair layout. `true` uses `(x[2i], x[2i+1])` pairs
///   (textbook RoPE / LLaMA), `false` uses `(x[i], x[i + half])` pairs
///   (Gemma 4 layout — `model_loader::gemma` passes `interleaved=false`).
///
/// # Output
/// A new [`DeviceTensor`] of the same shape and dtype as `x`.
///
/// # Errors
/// - dtype not `Float` or `BFloat16` — `RuntimeError`
/// - `cos_table.dtype != Float` or `sin_table.dtype != Float` — `RuntimeError`
/// - `x.shape.len() != 4` — `RuntimeError`
/// - `head_dim` is odd — `RuntimeError`
/// - cos/sin table inner dim `!= head_dim / 2` — `RuntimeError`
/// - `position + Sq > max_seq_len` — `RuntimeError`
/// - kernel not registered — `KernelLoadFailed`
/// - kernel launch failed — `KernelLaunchFailed`
///
/// The call is asynchronous on the default stream; use
/// [`super::synchronize`] before reading the output on the host.
pub fn rotary_gpu(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    cos_table: &DeviceTensor,
    sin_table: &DeviceTensor,
    position: i32,
    interleaved: bool,
) -> Result<DeviceTensor, CudaError> {
    let kernel_name = match x.dtype {
        DataType::Float => "rotary_f32",
        DataType::BFloat16 => "rotary_bf16",
        _ => {
            return Err(CudaError::RuntimeError {
                op: "rotary_gpu: unsupported dtype (need Float or BFloat16)",
                code: -1,
            });
        }
    };
    if cos_table.dtype != DataType::Float || sin_table.dtype != DataType::Float {
        return Err(CudaError::RuntimeError {
            op: "rotary_gpu: cos/sin tables must be Float",
            code: -1,
        });
    }
    if x.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "rotary_gpu: x must be rank-4 [B, H, Sq, head_dim]",
            code: -1,
        });
    }
    let batch_i64 = x.shape[0];
    let heads_i64 = x.shape[1];
    let seq_len_i64 = x.shape[2];
    let head_dim_i64 = x.shape[3];

    if head_dim_i64 <= 0 || head_dim_i64 % 2 != 0 {
        return Err(CudaError::RuntimeError {
            op: "rotary_gpu: head_dim must be a positive even integer",
            code: -1,
        });
    }
    let half_i64 = head_dim_i64 / 2;

    if cos_table.shape.len() != 2 || sin_table.shape.len() != 2 {
        return Err(CudaError::RuntimeError {
            op: "rotary_gpu: cos/sin tables must be rank-2",
            code: -1,
        });
    }
    if cos_table.shape[1] != half_i64 || sin_table.shape[1] != half_i64 {
        return Err(CudaError::RuntimeError {
            op: "rotary_gpu: cos/sin inner dim must equal head_dim / 2",
            code: -1,
        });
    }
    let max_seq_len = cos_table.shape[0];
    if sin_table.shape[0] != max_seq_len {
        return Err(CudaError::RuntimeError {
            op: "rotary_gpu: cos/sin tables must share max_seq_len",
            code: -1,
        });
    }
    if (position as i64) < 0 || (position as i64) + seq_len_i64 > max_seq_len {
        return Err(CudaError::RuntimeError {
            op: "rotary_gpu: position + seq_len exceeds cos/sin table length",
            code: -1,
        });
    }

    let out = DeviceTensor::alloc(x.shape.clone(), x.dtype)?;

    let total_rows: i64 = batch_i64 * heads_i64 * seq_len_i64;
    if total_rows <= 0 || head_dim_i64 == 0 {
        return Ok(out);
    }

    // Argument pack — each slot is the address of a local holding the
    // value, per the contract of `launch_kernel`.
    let mut x_ptr = x.buffer.as_mut_ptr();
    let mut cos_ptr = cos_table.buffer.as_mut_ptr();
    let mut sin_ptr = sin_table.buffer.as_mut_ptr();
    let mut y_ptr = out.buffer.as_mut_ptr();
    let mut seq_len_arg: i32 = seq_len_i64 as i32;
    let mut head_dim_arg: i32 = head_dim_i64 as i32;
    let mut position_arg: i32 = position;
    let mut interleaved_arg: i32 = i32::from(interleaved);

    let args: [*mut core::ffi::c_void; 8] = [
        &mut x_ptr as *mut _ as *mut core::ffi::c_void,
        &mut cos_ptr as *mut _ as *mut core::ffi::c_void,
        &mut sin_ptr as *mut _ as *mut core::ffi::c_void,
        &mut y_ptr as *mut _ as *mut core::ffi::c_void,
        &mut seq_len_arg as *mut _ as *mut core::ffi::c_void,
        &mut head_dim_arg as *mut _ as *mut core::ffi::c_void,
        &mut position_arg as *mut _ as *mut core::ffi::c_void,
        &mut interleaved_arg as *mut _ as *mut core::ffi::c_void,
    ];

    let half_u32 = (half_i64 as u32).max(1);
    let block_x: u32 = core::cmp::min(half_u32, 1024);
    let grid = (total_rows as u32, 1, 1);
    let block = (block_x, 1, 1);

    runtime
        .with_kernel(kernel_name, |k: &Kernel| {
            launch_kernel(k, grid, block, &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: ("rotary_gpu: kernel ".to_string() + kernel_name + " not registered"),
            cuda_result: -1,
        })??;

    Ok(out)
}
