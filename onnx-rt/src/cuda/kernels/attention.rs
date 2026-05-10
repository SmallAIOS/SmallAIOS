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

    // All threads in the block reach this point with no divergent control
    // flow above, so __activemask() returns the actual set of active lanes
    // in each warp. Using a literal 0xFFFFFFFFu would be UB on partial
    // warps (e.g. blockDim.x < 32 or blockDim.x not a multiple of 32).
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
    unsigned mask = __activemask();
    for (int off = 16; off > 0; off /= 2) {
        float other = __shfl_down_sync(mask, local_max, off);
        if (other > local_max) local_max = other;
    }
    __shared__ float warp_max[32];
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float partial = (lane < num_warps) ? warp_max[lane] : __int_as_float(0xff800000);
        unsigned mask2 = __activemask();
        for (int off = 16; off > 0; off /= 2) {
            float other = __shfl_down_sync(mask2, partial, off);
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
    unsigned mask_sum = __activemask();
    for (int off = 16; off > 0; off /= 2) {
        local_sum += __shfl_down_sync(mask_sum, local_sum, off);
    }
    __shared__ float warp_sum[32];
    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float partial = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        unsigned mask_sum2 = __activemask();
        for (int off = 16; off > 0; off /= 2) {
            partial += __shfl_down_sync(mask_sum2, partial, off);
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
    // seq_q > seq_kv would push `causal_offset` negative and silently mask
    // the leading query rows to all-zero probabilities. Reject up front.
    if seq_q > seq_kv {
        return Err(CudaError::RuntimeError {
            op: "masked_softmax_gpu: seq_q must be <= seq_kv",
            code: -1,
        });
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

// ── Section 6.15: gqa_merge_heads kernel ───────────────────────────

/// `gqa_merge_heads_f32`: transpose `[H, Sq, head_dim]` (head-major)
/// into `[B=1, Sq, H * head_dim]` (sequence-major). One thread per
/// output element.
pub const GQA_MERGE_HEADS_F32_SRC: &str = r#"
extern "C" __global__ void gqa_merge_heads_f32(
    const float* __restrict__ in_hsd,
    float* __restrict__ out_bsd,
    int heads,
    int seq,
    int head_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * seq * head_dim;
    if (tid >= total) return;

    int hd = head_dim;
    int sd = seq * hd;

    int h    = tid / sd;
    int rem  = tid - h * sd;
    int s    = rem / hd;
    int d    = rem - s * hd;

    int dst_idx = (s * heads + h) * hd + d;
    out_bsd[dst_idx] = in_hsd[tid];
}
"#;

/// `gqa_merge_heads_bf16`: BF16 variant of the merge transpose.
pub const GQA_MERGE_HEADS_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void gqa_merge_heads_bf16(
    const __nv_bfloat16* __restrict__ in_hsd,
    __nv_bfloat16* __restrict__ out_bsd,
    int heads,
    int seq,
    int head_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * seq * head_dim;
    if (tid >= total) return;

    int hd = head_dim;
    int sd = seq * hd;

    int h    = tid / sd;
    int rem  = tid - h * sd;
    int s    = rem / hd;
    int d    = rem - s * hd;

    int dst_idx = (s * heads + h) * hd + d;
    out_bsd[dst_idx] = in_hsd[tid];
}
"#;

/// Transpose `[H, Sq, head_dim]` → `[1, Sq, H * head_dim]`.
///
/// Final reshape after the softmax·V GEMM to produce the canonical
/// attention output layout.
pub fn merge_heads_gpu(
    runtime: &CudaRuntime,
    in_hsd: &DeviceTensor,
) -> Result<DeviceTensor, CudaError> {
    let kernel_name = match in_hsd.dtype {
        DataType::Float => "gqa_merge_heads_f32",
        DataType::BFloat16 => "gqa_merge_heads_bf16",
        _ => {
            return Err(CudaError::RuntimeError {
                op: "merge_heads_gpu: unsupported dtype",
                code: -1,
            });
        }
    };
    if in_hsd.shape.len() != 3 {
        return Err(CudaError::RuntimeError {
            op: "merge_heads_gpu: input must be rank-3 [H, Sq, head_dim]",
            code: -1,
        });
    }
    let heads_i64 = in_hsd.shape[0];
    let seq_i64 = in_hsd.shape[1];
    let hd_i64 = in_hsd.shape[2];

    let out_shape = alloc::vec![1i64, seq_i64, heads_i64 * hd_i64];
    let out = DeviceTensor::alloc(out_shape, in_hsd.dtype)?;

    let total: i64 = heads_i64 * seq_i64 * hd_i64;
    if total <= 0 {
        return Ok(out);
    }

    let mut in_ptr = in_hsd.buffer.as_mut_ptr();
    let mut out_ptr = out.buffer.as_mut_ptr();
    let mut heads_arg: i32 = heads_i64 as i32;
    let mut seq_arg: i32 = seq_i64 as i32;
    let mut hd_arg: i32 = hd_i64 as i32;

    let args: [*mut core::ffi::c_void; 5] = [
        &mut in_ptr as *mut _ as *mut core::ffi::c_void,
        &mut out_ptr as *mut _ as *mut core::ffi::c_void,
        &mut heads_arg as *mut _ as *mut core::ffi::c_void,
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
            name: ("merge_heads_gpu: kernel ".to_string() + kernel_name + " not registered"),
            cuda_result: -1,
        })??;

    Ok(out)
}

// ── Helper: cast F32 scratch to BF16 in place into a fresh tensor ──

/// `softmax_cast_f32_to_bf16`: cast every element of an F32 buffer to
/// BF16 in a fresh output buffer. Used to bridge the F32 softmax output
/// into the BF16 V GEMM where `gpu_gemm_strided_batched_ex` requires
/// matching A/B dtypes.
pub const SOFTMAX_CAST_F32_TO_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void softmax_cast_f32_to_bf16(
    const float* __restrict__ in_f32,
    __nv_bfloat16* __restrict__ out_bf16,
    int total
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    out_bf16[tid] = __float2bfloat16(in_f32[tid]);
}
"#;

/// `bf16_to_f32_gpu`: cast a BF16 buffer to F32 (mirror of
/// [`SOFTMAX_CAST_F32_TO_BF16_SRC`]). Used by `rotary_gpu` to accept
/// BF16 cos/sin tables (the rope kernels require F32 tables).
pub const CAST_BF16_TO_F32_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void cast_bf16_to_f32(
    const __nv_bfloat16* __restrict__ in_bf16,
    float* __restrict__ out_f32,
    int total
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    out_f32[tid] = __bfloat162float(in_bf16[tid]);
}
"#;

/// Cast a contiguous BF16 device tensor to F32.
pub fn cast_bf16_to_f32_gpu(
    runtime: &CudaRuntime,
    in_bf16: &DeviceTensor,
) -> Result<DeviceTensor, CudaError> {
    if in_bf16.dtype != DataType::BFloat16 {
        return Err(CudaError::RuntimeError {
            op: "cast_bf16_to_f32_gpu: input must be BFloat16",
            code: -1,
        });
    }
    let out = DeviceTensor::alloc(in_bf16.shape.clone(), DataType::Float)?;
    let total = in_bf16.total_elements();
    if total == 0 {
        return Ok(out);
    }
    let mut in_ptr = in_bf16.buffer.as_mut_ptr();
    let mut out_ptr = out.buffer.as_mut_ptr();
    let mut total_arg: i32 = total as i32;
    let args: [*mut core::ffi::c_void; 3] = [
        &mut in_ptr as *mut _ as *mut core::ffi::c_void,
        &mut out_ptr as *mut _ as *mut core::ffi::c_void,
        &mut total_arg as *mut _ as *mut core::ffi::c_void,
    ];
    let block_x: u32 = 256;
    let grid_x: u32 = (total as u64).div_ceil(block_x as u64) as u32;
    runtime
        .with_kernel("cast_bf16_to_f32", |k: &Kernel| {
            launch_kernel(k, (grid_x.max(1), 1, 1), (block_x, 1, 1), &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: "cast_bf16_to_f32_gpu: kernel cast_bf16_to_f32 not registered".to_string(),
            cuda_result: -1,
        })??;
    Ok(out)
}

/// Cast a contiguous F32 [`DeviceTensor`] to BF16, returning a new
/// tensor with the same shape. Used by `gpu_gqa` to bridge the F32
/// softmax scores into the BF16 V GEMM.
pub fn cast_f32_to_bf16_gpu(
    runtime: &CudaRuntime,
    in_f32: &DeviceTensor,
) -> Result<DeviceTensor, CudaError> {
    if in_f32.dtype != DataType::Float {
        return Err(CudaError::RuntimeError {
            op: "cast_f32_to_bf16_gpu: input must be Float",
            code: -1,
        });
    }
    let out = DeviceTensor::alloc(in_f32.shape.clone(), DataType::BFloat16)?;
    let total: usize = in_f32.total_elements();
    if total == 0 {
        return Ok(out);
    }

    let mut in_ptr = in_f32.buffer.as_mut_ptr();
    let mut out_ptr = out.buffer.as_mut_ptr();
    let mut total_arg: i32 = total as i32;
    let args: [*mut core::ffi::c_void; 3] = [
        &mut in_ptr as *mut _ as *mut core::ffi::c_void,
        &mut out_ptr as *mut _ as *mut core::ffi::c_void,
        &mut total_arg as *mut _ as *mut core::ffi::c_void,
    ];

    let block_x: u32 = 256;
    let grid_x: u32 = (total as u64).div_ceil(block_x as u64) as u32;
    let grid = (grid_x.max(1), 1, 1);
    let block = (block_x, 1, 1);

    runtime
        .with_kernel("softmax_cast_f32_to_bf16", |k: &Kernel| {
            launch_kernel(k, grid, block, &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: "cast_f32_to_bf16_gpu: kernel softmax_cast_f32_to_bf16 not registered"
                .to_string(),
            cuda_result: -1,
        })??;

    Ok(out)
}

// ── Section 6.12: top-level gpu_gqa ────────────────────────────────

/// Validated rank-4 GQA shape parameters extracted from Q/K/V.
struct GqaShape {
    num_q_heads_i64: i64,
    seq_q_i64: i64,
    head_dim_i64: i64,
    num_kv_heads_i64: i64,
    seq_kv_i64: i64,
}

/// Reject rank-3 Q/K/V at this entry point — `gpu_gqa_rank3` is the
/// supported path for the gemma builder's `[1, Sq, H*head_dim]` layout.
fn reject_rank3_gqa(q: &DeviceTensor, k: &DeviceTensor, v: &DeviceTensor) -> Result<(), CudaError> {
    if q.shape.len() != 3 || k.shape.len() != 3 || v.shape.len() != 3 {
        return Ok(());
    }
    let q_hidden = q.shape[2];
    let kv_hidden = k.shape[2];
    // Validate the q/kv hidden ratio before suggesting the rank-3 entry
    // point — preserves the original error precedence.
    if q_hidden == 0 || kv_hidden == 0 || q_hidden % kv_hidden != 0 {
        let kv_divides = kv_hidden != 0 && q_hidden % kv_hidden == 0;
        let mha = q_hidden == kv_hidden;
        if !kv_divides && !mha {
            return Err(CudaError::RuntimeError {
                op: "gpu_gqa: rank-3 q/kv hidden ratio is not integer",
                code: -1,
            });
        }
    }
    Err(CudaError::RuntimeError {
        op: "gpu_gqa: rank-3 input requires head-count attributes — use gpu_gqa_rank3",
        code: -1,
    })
}

/// Validate dtype + rank-4 shape for Q/K/V and return the GQA shape descriptor.
fn validate_gqa_rank4(
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
) -> Result<GqaShape, CudaError> {
    if q.dtype != k.dtype || q.dtype != v.dtype {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa: q/k/v dtypes must match",
            code: -1,
        });
    }
    if q.dtype != DataType::Float && q.dtype != DataType::BFloat16 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa: dtype must be Float or BFloat16",
            code: -1,
        });
    }
    if q.shape.len() != 4 || k.shape.len() != 4 || v.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa: q/k/v must each be rank-4 [B, H, S, head_dim]",
            code: -1,
        });
    }
    if q.shape[0] != 1 || k.shape[0] != 1 || v.shape[0] != 1 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa: only batch=1 is supported in v1",
            code: -1,
        });
    }
    if k.shape != v.shape {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa: k and v must share shape",
            code: -1,
        });
    }
    if v.shape[3] != q.shape[3] {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa: head_dim mismatch",
            code: -1,
        });
    }
    let num_q_heads_i64 = q.shape[1];
    let num_kv_heads_i64 = k.shape[1];
    if num_q_heads_i64 <= 0 || num_kv_heads_i64 <= 0 || num_q_heads_i64 % num_kv_heads_i64 != 0 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa: num_q_heads must be a positive multiple of num_kv_heads",
            code: -1,
        });
    }
    Ok(GqaShape {
        num_q_heads_i64,
        seq_q_i64: q.shape[2],
        head_dim_i64: q.shape[3],
        num_kv_heads_i64,
        seq_kv_i64: k.shape[2],
    })
}

/// Reject attention requests whose F32 scratch buffer exceeds the runtime cap.
fn check_attention_scratch_cap(runtime: &CudaRuntime, shape: &GqaShape) -> Result<(), CudaError> {
    let scratch_bytes: usize = (shape.num_q_heads_i64 as usize)
        .saturating_mul(shape.seq_q_i64 as usize)
        .saturating_mul(shape.seq_kv_i64 as usize)
        .saturating_mul(4);
    if scratch_bytes > runtime.attention_scratch_cap() {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa: attention scratch exceeds runtime cap (set_attention_scratch_cap)",
            code: -1,
        });
    }
    Ok(())
}

/// Group-query attention on the GPU.
///
/// Composes the cuBLAS strided batched GEMMs and the custom kernels
/// (`gqa_kv_expand`, `gqa_softmax_mask_f32`, `softmax_cast_f32_to_bf16`,
/// `gqa_merge_heads`) into a single forward pass. Inputs:
///
/// - `q`: `[B=1, num_q_heads, Sq, head_dim]`
/// - `k`, `v`: `[B=1, num_kv_heads, Sk, head_dim]`
///
/// Returns `[B=1, Sq, num_q_heads * head_dim]` (the canonical
/// attention output layout). All inputs must share the same dtype
/// (`Float` or `BFloat16`); the output matches.
///
/// `window`:
/// - `None` (default for global layers) — full causal mask.
/// - `Some(w)` — sliding-window mask over the last `w` keys.
///
/// `scale` defaults to `1 / sqrt(head_dim)` when `None`.
///
/// **No cache wiring.** This entry point treats `k` / `v` as the full
/// cached history. The cache append + view + transpose-to-head-major
/// flow lives in a follow-up wrapper (§6.14) — that wrapper will call
/// this function after materializing the merged K/V tensors.
///
/// # Errors
/// - `q.dtype != k.dtype` or `q.dtype != v.dtype` — `RuntimeError`
/// - dtype not in `{Float, BFloat16}` — `RuntimeError`
/// - rank mismatches or B != 1 — `RuntimeError`
/// - `num_q_heads % num_kv_heads != 0` — `RuntimeError`
/// - `head_dim`s differ — `RuntimeError`
/// - any intermediate kernel / cuBLAS error
#[allow(clippy::too_many_arguments)]
pub fn gpu_gqa(
    runtime: &CudaRuntime,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    window: Option<i32>,
    scale: Option<f32>,
) -> Result<DeviceTensor, CudaError> {
    // Rank-3 fallback: the gemma builder emits Q/K/V directly from its
    // projection Gemms in `[1, Sq, H*head_dim]` layout. Defer to the
    // dedicated rank-3 entry point — it needs explicit head counts.
    reject_rank3_gqa(q, k, v)?;

    let shape = validate_gqa_rank4(q, k, v)?;
    let dt = q.dtype;

    // §6.13: enforce the attention-scratch cap up front so a
    // pathological seq_kv doesn't try to allocate hundreds of GiB.
    check_attention_scratch_cap(runtime, &shape)?;

    let GqaShape {
        num_q_heads_i64,
        seq_q_i64,
        head_dim_i64,
        num_kv_heads_i64,
        seq_kv_i64,
    } = shape;
    let expand_factor = (num_q_heads_i64 / num_kv_heads_i64) as i32;
    let seq_q = seq_q_i64 as i32;
    let seq_kv = seq_kv_i64 as i32;
    let head_dim = head_dim_i64 as i32;

    // 1) Expand KV heads to per-attention-head replicas.
    let k_full = kv_expand_gpu(runtime, k, expand_factor)?;
    let v_full = kv_expand_gpu(runtime, v, expand_factor)?;

    // 2) QK^T via batched GEMM. Treat the leading B=1 as a no-op: pass
    //    [num_q_heads, Sq, head_dim] and [num_q_heads, Sk, head_dim]
    //    with trans_b=true → output [num_q_heads, Sq, Sk] in F32.
    let q_3d = view_as_3d(q, num_q_heads_i64, seq_q_i64, head_dim_i64)?;
    let k_full_3d = view_as_3d(&k_full, num_q_heads_i64, seq_kv_i64, head_dim_i64)?;
    let v_full_3d = view_as_3d(&v_full, num_q_heads_i64, seq_kv_i64, head_dim_i64)?;

    let scores_f32 = crate::cuda::dispatch::gpu_gemm_strided_batched_ex(
        runtime,
        &q_3d,
        &k_full_3d,
        false,
        true,
        DataType::Float,
    )?;

    // 3) Masked softmax in place.
    let scale_value: f32 = scale.unwrap_or_else(|| 1.0f32 / (head_dim as f32).sqrt());
    masked_softmax_gpu(runtime, &scores_f32, seq_q, seq_kv, window, scale_value)?;

    // 4) Softmax · V via batched GEMM. Output dtype matches V.
    //    For BF16 V we cast the F32 scratch to BF16 first because
    //    gpu_gemm_strided_batched_ex requires matching A/B dtypes.
    let context_3d = if dt == DataType::BFloat16 {
        let scores_bf16 = cast_f32_to_bf16_gpu(runtime, &scores_f32)?;
        crate::cuda::dispatch::gpu_gemm_strided_batched_ex(
            runtime,
            &scores_bf16,
            &v_full_3d,
            false,
            false,
            DataType::BFloat16,
        )?
    } else {
        crate::cuda::dispatch::gpu_gemm_strided_batched_ex(
            runtime,
            &scores_f32,
            &v_full_3d,
            false,
            false,
            DataType::Float,
        )?
    };
    // context_3d has shape [num_q_heads, seq_q, head_dim].

    // 5) Merge heads into [1, seq_q, num_q_heads * head_dim].
    merge_heads_gpu(runtime, &context_3d)
}

/// Reinterpret a `[1, H, S, D]` device tensor as `[H, S, D]`.
///
/// Wraps the input buffer in a new [`DeviceTensor`] header without
/// copying — needed because [`crate::cuda::dispatch::gpu_gemm_strided_batched_ex`]
/// takes rank-3 inputs and we want to keep the leading B=1 implicit.
fn view_as_3d(src: &DeviceTensor, h: i64, s: i64, d: i64) -> Result<DeviceTensor, CudaError> {
    if src.shape != alloc::vec![1i64, h, s, d] {
        return Err(CudaError::RuntimeError {
            op: "view_as_3d: shape mismatch",
            code: -1,
        });
    }
    // We need a DeviceTensor that *aliases* src.buffer. Since DeviceBuffer
    // doesn't provide a borrowing constructor, we copy via D2D — fine for
    // the v1 perf budget (the alternative is to refactor DeviceBuffer to
    // expose an alias-borrow API, which is out of scope here).
    let dst = DeviceTensor::alloc(alloc::vec![h, s, d], src.dtype)?;
    let bytes = src.byte_size();
    if bytes > 0 {
        let err = unsafe {
            crate::cuda::ffi::cudaMemcpy(
                dst.buffer.as_mut_ptr(),
                src.buffer.as_ptr(),
                bytes,
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            )
        };
        if err != crate::cuda::ffi::CUDA_SUCCESS {
            return Err(CudaError::CopyFailed {
                msg: "view_as_3d: D2D",
                code: err,
            });
        }
    }
    Ok(dst)
}

// ── Rank-3 adapter (transformer-gpu-kernels-v1 §7 gemma integration) ──

/// Rank-3 front-door to [`gpu_gqa`].
///
/// The gemma builder emits Q/K/V directly from its Gemm projection
/// layer in `[1, Sq, H*head_dim]` layout. Callers that already have
/// rank-4 `[B, H, Sq, head_dim]` tensors should use [`gpu_gqa`]
/// directly; this adapter transposes each operand to head-major
/// layout then delegates.
///
/// `num_q_heads` / `num_kv_heads` are read from the `GroupQueryAttention`
/// node attributes; the head_dim is derived as `q_hidden / num_q_heads`.
#[allow(clippy::too_many_arguments)]
pub fn gpu_gqa_rank3(
    runtime: &CudaRuntime,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    num_q_heads: i64,
    num_kv_heads: i64,
    window: Option<i32>,
    scale: Option<f32>,
) -> Result<DeviceTensor, CudaError> {
    let q_bhs = bsh_to_bhs_gpu(runtime, q, num_q_heads)?;
    let k_bhs = bsh_to_bhs_gpu(runtime, k, num_kv_heads)?;
    let v_bhs = bsh_to_bhs_gpu(runtime, v, num_kv_heads)?;
    gpu_gqa(runtime, &q_bhs, &k_bhs, &v_bhs, window, scale)
}

// ── Section 6.14: cache view → head-major transpose + gpu_gqa wrapper ──

/// `kv_cache_view_to_head_major_*`: transpose
/// `[token_count, num_kv_heads, head_dim]` (the [`crate::cuda::KvView`]
/// memory layout) into `[1, num_kv_heads, token_count, head_dim]` so it
/// can be fed straight into [`gpu_gqa`].
///
/// One thread per output element; each thread maps
/// `out[b=0, h, t, d]` ← `in[t, h, d]`.
/// `bsh_to_bhs_*`: transpose a `[B=1, Sq, H, head_dim]` tensor into
/// `[B=1, H, Sq, head_dim]` head-major layout. Identical memory reads
/// to [`KV_CACHE_VIEW_TO_HEAD_MAJOR_F32_SRC`] but accepts an explicit
/// batch axis so it can be used on the rank-3 Q/K/V tensors emitted by
/// the gemma builder's Gemm projection layer.
pub const BSH_TO_BHS_F32_SRC: &str = r#"
extern "C" __global__ void bsh_to_bhs_f32(
    const float* __restrict__ in_bshd,
    float* __restrict__ out_bhsd,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int total = seq_len * num_heads * head_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int hd = head_dim;
    int sd = seq_len * hd;

    int h    = tid / sd;
    int rem  = tid - h * sd;
    int s    = rem / hd;
    int d    = rem - s * hd;

    int src = (s * num_heads + h) * hd + d;
    out_bhsd[tid] = in_bshd[src];
}
"#;

pub const BSH_TO_BHS_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void bsh_to_bhs_bf16(
    const __nv_bfloat16* __restrict__ in_bshd,
    __nv_bfloat16* __restrict__ out_bhsd,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int total = seq_len * num_heads * head_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int hd = head_dim;
    int sd = seq_len * hd;

    int h    = tid / sd;
    int rem  = tid - h * sd;
    int s    = rem / hd;
    int d    = rem - s * hd;

    int src = (s * num_heads + h) * hd + d;
    out_bhsd[tid] = in_bshd[src];
}
"#;

/// Transpose a rank-3 `[1, Sq, num_heads * head_dim]` tensor into the
/// rank-4 head-major layout `[1, num_heads, Sq, head_dim]` expected by
/// [`gpu_gqa`]. The input buffer must be contiguous
/// `[B=1, Sq, H, head_dim]`.
pub fn bsh_to_bhs_gpu(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    num_heads: i64,
) -> Result<DeviceTensor, CudaError> {
    if x.shape.len() != 3 {
        return Err(CudaError::RuntimeError {
            op: "bsh_to_bhs_gpu: input must be rank-3 [B, Sq, H*head_dim]",
            code: -1,
        });
    }
    if x.shape[0] != 1 {
        return Err(CudaError::RuntimeError {
            op: "bsh_to_bhs_gpu: only batch=1 is supported",
            code: -1,
        });
    }
    let kernel_name = match x.dtype {
        DataType::Float => "bsh_to_bhs_f32",
        DataType::BFloat16 => "bsh_to_bhs_bf16",
        _ => {
            return Err(CudaError::RuntimeError {
                op: "bsh_to_bhs_gpu: unsupported dtype",
                code: -1,
            });
        }
    };
    let seq_len = x.shape[1];
    let hidden = x.shape[2];
    if num_heads <= 0 || hidden % num_heads != 0 {
        return Err(CudaError::RuntimeError {
            op: "bsh_to_bhs_gpu: hidden not divisible by num_heads",
            code: -1,
        });
    }
    let head_dim = hidden / num_heads;
    let out = DeviceTensor::alloc(alloc::vec![1i64, num_heads, seq_len, head_dim], x.dtype)?;

    let total: i64 = seq_len * num_heads * head_dim;
    if total <= 0 {
        return Ok(out);
    }

    let mut in_ptr = x.buffer.as_mut_ptr();
    let mut out_ptr = out.buffer.as_mut_ptr();
    let mut seq_arg: i32 = seq_len as i32;
    let mut heads_arg: i32 = num_heads as i32;
    let mut hd_arg: i32 = head_dim as i32;
    let args: [*mut core::ffi::c_void; 5] = [
        &mut in_ptr as *mut _ as *mut core::ffi::c_void,
        &mut out_ptr as *mut _ as *mut core::ffi::c_void,
        &mut seq_arg as *mut _ as *mut core::ffi::c_void,
        &mut heads_arg as *mut _ as *mut core::ffi::c_void,
        &mut hd_arg as *mut _ as *mut core::ffi::c_void,
    ];
    let block_x: u32 = 256;
    let grid_x: u32 = (total as u64).div_ceil(block_x as u64) as u32;
    runtime
        .with_kernel(kernel_name, |k: &Kernel| {
            launch_kernel(k, (grid_x.max(1), 1, 1), (block_x, 1, 1), &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: ("bsh_to_bhs_gpu: kernel ".to_string() + kernel_name + " not registered"),
            cuda_result: -1,
        })??;
    Ok(out)
}

pub const KV_CACHE_VIEW_TO_HEAD_MAJOR_F32_SRC: &str = r#"
extern "C" __global__ void kv_cache_view_to_head_major_f32(
    const float* __restrict__ in_thd,
    float* __restrict__ out_bhsd,
    int token_count,
    int num_kv_heads,
    int head_dim
) {
    int total = token_count * num_kv_heads * head_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int hd  = head_dim;
    int sd  = token_count * hd;

    int h    = tid / sd;
    int rem  = tid - h * sd;
    int t    = rem / hd;
    int d    = rem - t * hd;

    int src  = (t * num_kv_heads + h) * hd + d;
    out_bhsd[tid] = in_thd[src];
}
"#;

pub const KV_CACHE_VIEW_TO_HEAD_MAJOR_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void kv_cache_view_to_head_major_bf16(
    const __nv_bfloat16* __restrict__ in_thd,
    __nv_bfloat16* __restrict__ out_bhsd,
    int token_count,
    int num_kv_heads,
    int head_dim
) {
    int total = token_count * num_kv_heads * head_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int hd  = head_dim;
    int sd  = token_count * hd;

    int h    = tid / sd;
    int rem  = tid - h * sd;
    int t    = rem / hd;
    int d    = rem - t * hd;

    int src  = (t * num_kv_heads + h) * hd + d;
    out_bhsd[tid] = in_thd[src];
}
"#;

/// Materialize a [`crate::cuda::KvView`] as a head-major
/// `[1, num_kv_heads, token_count, head_dim]` device tensor.
///
/// The cache stores K/V in token-major layout (one token's
/// `[num_kv_heads, head_dim]` block at a time). The attention kernels
/// want head-major batches — this kernel is the transpose bridge.
fn cache_view_to_head_major(
    runtime: &CudaRuntime,
    view: &crate::cuda::KvView,
    target_ptr_field: WhichPtr,
) -> Result<DeviceTensor, CudaError> {
    let token_count = view.token_count;
    let num_kv_heads = view.num_kv_heads;
    let head_dim = view.head_dim;
    let dtype = view.dtype;

    let out = DeviceTensor::alloc(
        alloc::vec![
            1i64,
            num_kv_heads as i64,
            token_count as i64,
            head_dim as i64
        ],
        dtype,
    )?;

    if token_count == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Ok(out);
    }

    let kernel_name = match dtype {
        DataType::Float => "kv_cache_view_to_head_major_f32",
        DataType::BFloat16 => "kv_cache_view_to_head_major_bf16",
        _ => {
            return Err(CudaError::RuntimeError {
                op: "cache_view_to_head_major: unsupported dtype",
                code: -1,
            });
        }
    };

    let mut in_ptr: *mut core::ffi::c_void = match target_ptr_field {
        WhichPtr::K => view.k_ptr as *mut core::ffi::c_void,
        WhichPtr::V => view.v_ptr as *mut core::ffi::c_void,
    };
    let mut out_ptr = out.buffer.as_mut_ptr();
    let mut tc_arg: i32 = token_count as i32;
    let mut nh_arg: i32 = num_kv_heads as i32;
    let mut hd_arg: i32 = head_dim as i32;

    let args: [*mut core::ffi::c_void; 5] = [
        &mut in_ptr as *mut _ as *mut core::ffi::c_void,
        &mut out_ptr as *mut _ as *mut core::ffi::c_void,
        &mut tc_arg as *mut _ as *mut core::ffi::c_void,
        &mut nh_arg as *mut _ as *mut core::ffi::c_void,
        &mut hd_arg as *mut _ as *mut core::ffi::c_void,
    ];

    let total: usize = token_count * num_kv_heads * head_dim;
    let block_x: u32 = 256;
    let grid_x: u32 = (total as u64).div_ceil(block_x as u64) as u32;
    let grid = (grid_x.max(1), 1, 1);
    let block = (block_x, 1, 1);

    runtime
        .with_kernel(kernel_name, |k: &Kernel| {
            launch_kernel(k, grid, block, &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: ("cache_view_to_head_major: kernel ".to_string()
                + kernel_name
                + " not registered"),
            cuda_result: -1,
        })??;

    Ok(out)
}

/// Discriminator selecting K vs V inside [`cache_view_to_head_major`].
#[derive(Clone, Copy)]
enum WhichPtr {
    K,
    V,
}

/// Cache-aware GQA: append the new K/V to `kv_cache[layer_idx]`, view
/// the full per-layer history, transpose to head-major, then call
/// [`gpu_gqa`].
///
/// `new_k` and `new_v` must each have shape
/// `[1, num_kv_heads, 1, head_dim]` (single-token append). After the
/// call, `kv_cache.current_position()` for this layer has been
/// incremented by one — callers wiring multi-layer stacks must call
/// [`crate::cuda::GpuKvCache::advance_position`] exactly once after
/// every layer of a forward pass has appended its K/V (the existing
/// cache contract).
///
/// `window` is `Some(w)` for sliding-window layers, `None` for global
/// layers.
#[allow(clippy::too_many_arguments)]
pub fn gpu_gqa_with_cache(
    runtime: &CudaRuntime,
    q: &DeviceTensor,
    new_k: &DeviceTensor,
    new_v: &DeviceTensor,
    kv_cache: &mut crate::cuda::GpuKvCache,
    layer_idx: usize,
    window: Option<i32>,
    scale: Option<f32>,
) -> Result<DeviceTensor, CudaError> {
    if new_k.shape.len() != 4 || new_v.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa_with_cache: new_k/new_v must be rank-4",
            code: -1,
        });
    }
    if new_k.shape[2] != 1 || new_v.shape[2] != 1 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gqa_with_cache: only Sq=1 (decode) appends are supported in v1",
            code: -1,
        });
    }

    // append() writes to slot current_position but does NOT commit.
    // view_pending() returns [..., current_position + 1) so the new
    // token is visible to self-attention. The executor calls
    // advance_position() exactly once per forward pass after the last
    // layer has appended.
    kv_cache.append(layer_idx, new_k, new_v)?;
    let view = kv_cache.view_pending(layer_idx)?;

    let k_full = cache_view_to_head_major(runtime, &view, WhichPtr::K)?;
    let v_full = cache_view_to_head_major(runtime, &view, WhichPtr::V)?;

    gpu_gqa(runtime, q, &k_full, &v_full, window, scale)
}
