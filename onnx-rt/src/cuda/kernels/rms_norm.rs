// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `RMSNormalization` GPU operator.
//!
//! Root-mean-square normalization is the pre-attention / pre-MLP
//! normalization layer used by Gemma, Llama, Qwen, and DeepSeek. For
//! every row `x[i, :]` of shape `[hidden_size]` the kernel computes:
//!
//! ```text
//! y[i, :] = x[i, :] * rsqrt(mean(x[i, :]^2) + eps) * weight
//! ```
//!
//! Gemma's specification actually multiplies by `(1 + weight)`, but the
//! safetensors graph builder (`model_loader::gemma`) pre-adds 1 to the
//! stored weight at load time, so the GPU kernel executes the plain
//! formula unconditionally.
//!
//! ## Kernel layout (design D6)
//!
//! One CUDA block per outer element (where `outer_size` is the product
//! of all dimensions except the last). Threads within the block
//! cooperate via warp-level shuffles (`__shfl_down_sync`) to reduce the
//! sum of squares over the hidden dimension:
//!
//! 1. Each thread accumulates a partial `sum(x*x)` over its strided slice.
//! 2. A per-warp reduction (`__shfl_down_sync`) reduces each warp's lane
//!    values into lane 0.
//! 3. Lane 0 of each warp writes into a shared `warp_sums[32]` array.
//! 4. Warp 0 loads those partials and performs one final warp reduction.
//! 5. The resulting `total_sum` is broadcast via `warp_sums[0]` and
//!    `__syncthreads`, producing `inv_rms = rsqrt(mean + eps)`.
//! 6. Each thread then scales its slice of the row and writes the output.
//!
//! Accumulation is always in F32 — even for the BF16 variant — to keep
//! RMS numerically stable at long hidden dimensions. Block size is
//! clamped at 1024 (Blackwell's `maxThreadsPerBlock`). Rows with
//! `hidden_size > 1024` are handled by the in-kernel grid-stride loop.

extern crate alloc;

use alloc::string::ToString;

use super::{launch_kernel, Kernel};
use crate::cuda::gpu_executor::DeviceTensor;
use crate::cuda::{CudaError, CudaRuntime};
use crate::tensor::DataType;

// ── Kernel sources ─────────────────────────────────────────────────

/// `rms_norm_f32`: F32 input / weight / output. Accumulation in F32.
pub const RMS_NORM_F32_SRC: &str = r#"
extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    const float* x_row = x + (size_t)row * (size_t)hidden_size;
    float*       y_row = y + (size_t)row * (size_t)hidden_size;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x_row[i];
        local_sum += v * v;
    }

    unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    __shared__ float warp_sums[32];
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0) {
        float partial = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        if (lane == 0) warp_sums[0] = partial;
    }
    __syncthreads();

    float total_sum = warp_sums[0];
    float mean      = total_sum / (float)hidden_size;
    float inv_rms   = rsqrtf(mean + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y_row[i] = x_row[i] * inv_rms * weight[i];
    }
}
"#;

/// `rms_norm_bf16`: BF16 input / weight / output with F32 accumulation
/// and F32 intermediate scaling. Conversion via `__bfloat162float` /
/// `__float2bfloat16`.
pub const RMS_NORM_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void rms_norm_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + (size_t)row * (size_t)hidden_size;
    __nv_bfloat16*       y_row = y + (size_t)row * (size_t)hidden_size;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __bfloat162float(x_row[i]);
        local_sum += v * v;
    }

    unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    __shared__ float warp_sums[32];
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0) {
        float partial = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        if (lane == 0) warp_sums[0] = partial;
    }
    __syncthreads();

    float total_sum = warp_sums[0];
    float mean      = total_sum / (float)hidden_size;
    float inv_rms   = rsqrtf(mean + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float xv = __bfloat162float(x_row[i]);
        float wv = __bfloat162float(weight[i]);
        y_row[i] = __float2bfloat16(xv * inv_rms * wv);
    }
}
"#;

// ── Public wrapper ─────────────────────────────────────────────────

/// GPU implementation of ONNX `RMSNormalization`.
///
/// # Parameters
/// - `x`: input tensor, shape `[..., hidden_size]`. Dtype must be
///   `Float` or `BFloat16`.
/// - `weight`: learned scale, shape `[hidden_size]`. Must share `x`'s
///   dtype.
/// - `eps`: small constant added before the reciprocal square root
///   (`1e-6` is typical for Gemma).
///
/// # Output
/// A new [`DeviceTensor`] with the same shape and dtype as `x`.
///
/// # Errors
/// - `x.dtype != weight.dtype` — "rms_norm_gpu: x and weight dtypes must match"
/// - dtype not in `{Float, BFloat16}` — "rms_norm_gpu: unsupported dtype"
/// - `x.shape.len() < 1` — "rms_norm_gpu: x must have rank >= 1"
/// - `weight.shape != [hidden_size]` — "rms_norm_gpu: weight shape mismatch"
/// - kernel not registered — `KernelLoadFailed`
/// - kernel launch failed — `KernelLaunchFailed`
///
/// The call is asynchronous on the default stream; use
/// [`super::synchronize`] before reading the output on the host.
pub fn rms_norm_gpu(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    weight: &DeviceTensor,
    eps: f32,
) -> Result<DeviceTensor, CudaError> {
    if x.dtype != weight.dtype {
        return Err(CudaError::RuntimeError {
            op: "rms_norm_gpu: x and weight dtypes must match",
            code: -1,
        });
    }
    let kernel_name = match x.dtype {
        DataType::Float => "rms_norm_f32",
        DataType::BFloat16 => "rms_norm_bf16",
        _ => {
            return Err(CudaError::RuntimeError {
                op: "rms_norm_gpu: unsupported dtype",
                code: -1,
            });
        }
    };
    if x.shape.is_empty() {
        return Err(CudaError::RuntimeError {
            op: "rms_norm_gpu: x must have rank >= 1",
            code: -1,
        });
    }

    let hidden_size_i64: i64 = x.shape[x.shape.len() - 1];
    if weight.shape.len() != 1 || weight.shape[0] != hidden_size_i64 {
        return Err(CudaError::RuntimeError {
            op: "rms_norm_gpu: weight shape mismatch",
            code: -1,
        });
    }
    let hidden_size: i32 = hidden_size_i64 as i32;
    let outer_size_i64: i64 = x.shape[..x.shape.len() - 1]
        .iter()
        .copied()
        .product::<i64>()
        .max(0);
    let outer_size: i32 = outer_size_i64 as i32;

    let out = DeviceTensor::alloc(x.shape.clone(), x.dtype)?;

    // Empty tensors: nothing to launch.
    if outer_size == 0 || hidden_size == 0 {
        return Ok(out);
    }

    // Argument pack — each slot is the address of a local holding the
    // value, per the contract of `launch_kernel`.
    let mut x_ptr = x.buffer.as_mut_ptr();
    let mut weight_ptr = weight.buffer.as_mut_ptr();
    let mut y_ptr = out.buffer.as_mut_ptr();
    let mut hidden_size_arg: i32 = hidden_size;
    let mut eps_arg: f32 = eps;

    let args: [*mut core::ffi::c_void; 5] = [
        &mut x_ptr as *mut _ as *mut core::ffi::c_void,
        &mut weight_ptr as *mut _ as *mut core::ffi::c_void,
        &mut y_ptr as *mut _ as *mut core::ffi::c_void,
        &mut hidden_size_arg as *mut _ as *mut core::ffi::c_void,
        &mut eps_arg as *mut _ as *mut core::ffi::c_void,
    ];

    // Block size: cap at 1024 (Blackwell max). Warp reductions work
    // correctly regardless of whether hidden_size <, ==, or > blockDim.x.
    let block_x: u32 = core::cmp::min(hidden_size as u32, 1024).max(1);
    let grid = (outer_size as u32, 1, 1);
    let block = (block_x, 1, 1);

    runtime
        .with_kernel(kernel_name, |k: &Kernel| {
            launch_kernel(k, grid, block, &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: ("rms_norm_gpu: kernel ".to_string() + kernel_name + " not registered"),
            cuda_result: -1,
        })??;

    Ok(out)
}
