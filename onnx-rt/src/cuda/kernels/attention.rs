// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Attention support kernels for `gpu_gqa`.
//!
//! Section 6 of `transformer-gpu-kernels-v1` decomposes
//! GroupQueryAttention into a sequence of cuBLAS calls and a handful of
//! custom kernels. This module hosts those custom kernels:
//!
//! - [`GQA_SOFTMAX_MASK_F32_SRC`] — fused scale + causal mask +
//!   sliding-window mask + softmax, in-place on F32 scores.
//! - [`GQA_KV_EXPAND_*_SRC`] — replicate KV heads for GQA. (Added in
//!   §6.10.)
//! - [`GQA_MERGE_HEADS_*_SRC`] — final transpose / reshape from
//!   `[H, Sq, head_dim]` to `[B, Sq, H*head_dim]`. (Added in §6.15.)
//!
//! Public wrappers that compose into [`crate::cuda::kernels::attention`]
//! live alongside their kernel sources. The top-level `gpu_gqa` driver
//! ties everything together in a separate module (added in §6.12).

extern crate alloc;

use alloc::string::ToString;

use super::{launch_kernel, Kernel};
use crate::cuda::gpu_executor::DeviceTensor;
use crate::cuda::{CudaError, CudaRuntime};
use crate::tensor::DataType;

// ── Section 6.5: gqa_softmax_mask kernel ───────────────────────────

/// In-place fused softmax on F32 attention scores `[H, seq_q, seq_kv]`.
///
/// One block per `(head, q_idx)` row. Each thread cooperates via
/// warp-level shuffles for two reductions:
///
/// 1. Row max over allowed positions (masked positions contribute
///    `__int_as_float(0xff800000)`).
/// 2. Row sum of `exp(score * scale - row_max)` (masked positions
///    contribute `0.0`).
///
/// The kernel applies a causal mask (positions `k > q_pos` are masked)
/// and an optional sliding-window mask (positions `k < q_pos - window`
/// are also masked). `q_pos = causal_offset + qi` — `causal_offset`
/// equals `seq_kv - seq_q` for the standard "query positions are the
/// last `seq_q` of the cached `seq_kv`" layout.
///
/// `window <= 0` disables the sliding-window mask. The scale factor
/// (`1 / sqrt(head_dim)`) is applied inside the kernel so callers do
/// not have to scale the scores beforehand.
pub const GQA_SOFTMAX_MASK_F32_SRC: &str = r#"
extern "C" __global__ void gqa_softmax_mask_f32(
    float* __restrict__ scores,
    int seq_q,
    int seq_kv,
    int causal_offset,
    int window,
    float scale
) {
    int head = blockIdx.x;
    int qi   = blockIdx.y;
    if (qi >= seq_q) return;

    float* row = scores + ((size_t)head * (size_t)seq_q + (size_t)qi) * (size_t)seq_kv;
    int q_pos = causal_offset + qi;

    unsigned mask = 0xFFFFFFFFu;
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    // ---- Phase 1: row max over allowed positions ----
    float local_max = __int_as_float(0xff800000);
    for (int k = threadIdx.x; k < seq_kv; k += blockDim.x) {
        bool allowed = (k <= q_pos);
        if (window > 0 && k < q_pos - window) allowed = false;
        float v = allowed ? (row[k] * scale) : __int_as_float(0xff800000);
        if (v > local_max) local_max = v;
    }
    for (int off = 16; off > 0; off /= 2) {
        float other = __shfl_down_sync(mask, local_max, off);
        if (other > local_max) local_max = other;
    }
    __shared__ float warp_max[32];
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float partial = (lane < num_warps) ? warp_max[lane] : __int_as_float(0xff800000);
        for (int off = 16; off > 0; off /= 2) {
            float other = __shfl_down_sync(mask, partial, off);
            if (other > partial) partial = other;
        }
        if (lane == 0) warp_max[0] = partial;
    }
    __syncthreads();
    float row_max = warp_max[0];

    // ---- Phase 2: write exp(scaled_score - row_max), accumulate row sum ----
    // If the row is fully masked, row_max stays __int_as_float(0xff800000) and we write 0.0
    // everywhere (matches the convention of softmax over an empty set).
    bool empty_row = !isfinite(row_max);

    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < seq_kv; k += blockDim.x) {
        bool allowed = (k <= q_pos);
        if (window > 0 && k < q_pos - window) allowed = false;
        float v;
        if (!allowed || empty_row) {
            v = 0.0f;
        } else {
            v = expf(row[k] * scale - row_max);
        }
        row[k] = v;
        local_sum += v;
    }
    for (int off = 16; off > 0; off /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, off);
    }
    __shared__ float warp_sum[32];
    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float partial = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        for (int off = 16; off > 0; off /= 2) {
            partial += __shfl_down_sync(mask, partial, off);
        }
        if (lane == 0) warp_sum[0] = partial;
    }
    __syncthreads();
    float total = warp_sum[0];

    // ---- Phase 3: divide by sum ----
    if (total > 0.0f) {
        float inv = 1.0f / total;
        for (int k = threadIdx.x; k < seq_kv; k += blockDim.x) {
            row[k] *= inv;
        }
    }
}
"#;

/// In-place fused causal-and-window softmax on F32 attention scores.
///
/// `scores` must be `[num_heads, seq_q, seq_kv]` F32. The kernel
/// modifies the buffer in place — there is no separate output tensor.
///
/// `window` follows the convention used by `ops/microsoft.rs::group_query_attention`:
/// `0` (or negative) means "no sliding-window mask" (full causal), any
/// positive value `w` means each query at position `q_pos` only attends
/// to keys in the closed range `[q_pos - w, q_pos]`. `causal_offset` is
/// the absolute position of `qi == 0`; for the typical "queries are the
/// last `seq_q` of the cached `seq_kv`" layout it is `seq_kv - seq_q`.
///
/// `scale` is multiplied into every score before the softmax — pass
/// `1 / sqrt(head_dim)` for the standard scaled-dot-product attention.
///
/// # Errors
/// - `scores.dtype != Float` — `RuntimeError`
/// - `scores.shape.len() != 3` — `RuntimeError`
/// - `scores.shape[1] != seq_q` or `scores.shape[2] != seq_kv` — `RuntimeError`
/// - kernel not registered — `KernelLoadFailed`
/// - kernel launch failed — `KernelLaunchFailed`
///
/// The call is asynchronous on the default stream; use
/// [`super::synchronize`] before reading the output on the host.
pub fn masked_softmax_gpu(
    runtime: &CudaRuntime,
    scores: &DeviceTensor,
    seq_q: i32,
    seq_kv: i32,
    window: Option<i32>,
    scale: f32,
) -> Result<(), CudaError> {
    if scores.dtype != DataType::Float {
        return Err(CudaError::RuntimeError {
            op: "masked_softmax_gpu: scores must be Float",
            code: -1,
        });
    }
    if scores.shape.len() != 3 {
        return Err(CudaError::RuntimeError {
            op: "masked_softmax_gpu: scores must be rank-3 [H, seq_q, seq_kv]",
            code: -1,
        });
    }
    let heads_i64 = scores.shape[0];
    if scores.shape[1] != seq_q as i64 || scores.shape[2] != seq_kv as i64 {
        return Err(CudaError::RuntimeError {
            op: "masked_softmax_gpu: shape does not match seq_q/seq_kv",
            code: -1,
        });
    }
    if heads_i64 <= 0 || seq_q <= 0 || seq_kv <= 0 {
        return Ok(());
    }

    let causal_offset: i32 = seq_kv - seq_q;
    let window_arg: i32 = window.unwrap_or(0);

    let mut scores_ptr = scores.buffer.as_mut_ptr();
    let mut seq_q_arg: i32 = seq_q;
    let mut seq_kv_arg: i32 = seq_kv;
    let mut causal_offset_arg: i32 = causal_offset;
    let mut window_arg_local: i32 = window_arg;
    let mut scale_arg: f32 = scale;

    let args: [*mut core::ffi::c_void; 6] = [
        &mut scores_ptr as *mut _ as *mut core::ffi::c_void,
        &mut seq_q_arg as *mut _ as *mut core::ffi::c_void,
        &mut seq_kv_arg as *mut _ as *mut core::ffi::c_void,
        &mut causal_offset_arg as *mut _ as *mut core::ffi::c_void,
        &mut window_arg_local as *mut _ as *mut core::ffi::c_void,
        &mut scale_arg as *mut _ as *mut core::ffi::c_void,
    ];

    let block_x: u32 = core::cmp::min(seq_kv as u32, 1024).max(1);
    let grid = (heads_i64 as u32, seq_q as u32, 1);
    let block = (block_x, 1, 1);

    runtime
        .with_kernel("gqa_softmax_mask_f32", |k: &Kernel| {
            launch_kernel(k, grid, block, &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: "masked_softmax_gpu: kernel gqa_softmax_mask_f32 not registered".to_string(),
            cuda_result: -1,
        })??;

    Ok(())
}

// ── Section 6.10-6.11: KV head expansion ───────────────────────────

/// `gqa_kv_expand_f32`: replicate each KV head `expand` times along the
/// head axis. Input `[B, num_kv_heads, S, head_dim]` → output
/// `[B, num_kv_heads * expand, S, head_dim]`. One thread per output
/// element; each thread copies `out[b, h_q, s, d]` from
/// `in[b, h_q / expand, s, d]`.
pub const GQA_KV_EXPAND_F32_SRC: &str = r#"
extern "C" __global__ void gqa_kv_expand_f32(
    const float* __restrict__ in_kv,
    float* __restrict__ out_q,
    int batch,
    int num_kv_heads,
    int expand,
    int seq,
    int head_dim
) {
    int total = batch * (num_kv_heads * expand) * seq * head_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int hd  = head_dim;
    int sd  = seq * hd;
    int hqd = (num_kv_heads * expand) * sd;

    int b   = tid / hqd;
    int rem = tid - b * hqd;
    int hq  = rem / sd;
    int rem2 = rem - hq * sd;
    int s   = rem2 / hd;
    int d   = rem2 - s * hd;

    int hk  = hq / expand;
    int src_idx = ((b * num_kv_heads + hk) * seq + s) * hd + d;
    out_q[tid] = in_kv[src_idx];
}
"#;

/// `gqa_kv_expand_bf16`: BF16 variant of the same kernel. The copy is a
/// pure load/store so we read the BF16 bits directly without converting
/// to/from F32 — half the kernel of the F32 path.
pub const GQA_KV_EXPAND_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void gqa_kv_expand_bf16(
    const __nv_bfloat16* __restrict__ in_kv,
    __nv_bfloat16* __restrict__ out_q,
    int batch,
    int num_kv_heads,
    int expand,
    int seq,
    int head_dim
) {
    int total = batch * (num_kv_heads * expand) * seq * head_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int hd  = head_dim;
    int sd  = seq * hd;
    int hqd = (num_kv_heads * expand) * sd;

    int b   = tid / hqd;
    int rem = tid - b * hqd;
    int hq  = rem / sd;
    int rem2 = rem - hq * sd;
    int s   = rem2 / hd;
    int d   = rem2 - s * hd;

    int hk  = hq / expand;
    int src_idx = ((b * num_kv_heads + hk) * seq + s) * hd + d;
    out_q[tid] = in_kv[src_idx];
}
"#;

/// Replicate each KV head `expand` times along the head axis.
///
/// Input `kv` has shape `[B, num_kv_heads, S, head_dim]`; output has
/// shape `[B, num_kv_heads * expand, S, head_dim]` with each KV head
/// `h_kv` mapped to attention heads
/// `[h_kv * expand, .., h_kv * expand + expand - 1]`.
///
/// `expand == 1` is a no-op fast path: a fresh device tensor is
/// allocated and the input bytes are D2D-copied directly.
///
/// # Errors
/// - dtype not in `{Float, BFloat16}` — `RuntimeError`
/// - `kv.shape.len() != 4` — `RuntimeError`
/// - `expand <= 0` — `RuntimeError`
/// - kernel not registered — `KernelLoadFailed`
/// - kernel launch failed — `KernelLaunchFailed`
pub fn kv_expand_gpu(
    runtime: &CudaRuntime,
    kv: &DeviceTensor,
    expand: i32,
) -> Result<DeviceTensor, CudaError> {
    let _ = runtime;
    let kernel_name = match kv.dtype {
        DataType::Float => "gqa_kv_expand_f32",
        DataType::BFloat16 => "gqa_kv_expand_bf16",
        _ => {
            return Err(CudaError::RuntimeError {
                op: "kv_expand_gpu: unsupported dtype",
                code: -1,
            });
        }
    };
    if kv.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "kv_expand_gpu: kv must be rank-4 [B, num_kv_heads, S, head_dim]",
            code: -1,
        });
    }
    if expand <= 0 {
        return Err(CudaError::RuntimeError {
            op: "kv_expand_gpu: expand must be > 0",
            code: -1,
        });
    }

    let batch_i64 = kv.shape[0];
    let num_kv_i64 = kv.shape[1];
    let seq_i64 = kv.shape[2];
    let hd_i64 = kv.shape[3];

    let out_shape = alloc::vec![batch_i64, num_kv_i64 * expand as i64, seq_i64, hd_i64];
    let out = DeviceTensor::alloc(out_shape, kv.dtype)?;

    let total: i64 = batch_i64 * num_kv_i64 * expand as i64 * seq_i64 * hd_i64;
    if total <= 0 {
        return Ok(out);
    }

    // Fast path: expand == 1 is a straight D2D copy (no need to launch
    // the kernel for the standard MHA case).
    if expand == 1 {
        let bytes = out.byte_size();
        if bytes > 0 {
            let err = unsafe {
                crate::cuda::ffi::cudaMemcpy(
                    out.buffer.as_mut_ptr(),
                    kv.buffer.as_ptr(),
                    bytes,
                    crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                )
            };
            if err != crate::cuda::ffi::CUDA_SUCCESS {
                return Err(CudaError::CopyFailed {
                    msg: "kv_expand_gpu: device-to-device fast path",
                    code: err,
                });
            }
        }
        return Ok(out);
    }

    let mut in_ptr = kv.buffer.as_mut_ptr();
    let mut out_ptr = out.buffer.as_mut_ptr();
    let mut batch_arg: i32 = batch_i64 as i32;
    let mut num_kv_arg: i32 = num_kv_i64 as i32;
    let mut expand_arg: i32 = expand;
    let mut seq_arg: i32 = seq_i64 as i32;
    let mut hd_arg: i32 = hd_i64 as i32;

    let args: [*mut core::ffi::c_void; 7] = [
        &mut in_ptr as *mut _ as *mut core::ffi::c_void,
        &mut out_ptr as *mut _ as *mut core::ffi::c_void,
        &mut batch_arg as *mut _ as *mut core::ffi::c_void,
        &mut num_kv_arg as *mut _ as *mut core::ffi::c_void,
        &mut expand_arg as *mut _ as *mut core::ffi::c_void,
        &mut seq_arg as *mut _ as *mut core::ffi::c_void,
        &mut hd_arg as *mut _ as *mut core::ffi::c_void,
    ];

    let block_x: u32 = 256;
    let grid_x: u32 = (total as u64).div_ceil(block_x as u64) as u32;
    let grid = (grid_x.max(1), 1, 1);
    let block = (block_x, 1, 1);

    runtime
        .with_kernel(kernel_name, |k: &Kernel| {
            launch_kernel(k, grid, block, &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: ("kv_expand_gpu: kernel ".to_string() + kernel_name + " not registered"),
            cuda_result: -1,
        })??;

    Ok(out)
}
