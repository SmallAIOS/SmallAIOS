// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Pre-built Metal Shading Language (MSL) kernels for ONNX operators.
//!
//! Each constant contains a complete MSL source string that can be passed to
//! [`MetalProvider::load_kernel`](super::MetalProvider) for runtime compilation.
//! The kernel function name matches the constant name in snake_case.

/// Element-wise addition: `c[i] = a[i] + b[i]`
pub const ELEMENTWISE_ADD: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    c[id] = a[id] + b[id];
}
"#;

/// Element-wise subtraction: `c[i] = a[i] - b[i]`
pub const ELEMENTWISE_SUB: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_sub(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    c[id] = a[id] - b[id];
}
"#;

/// Element-wise multiplication: `c[i] = a[i] * b[i]`
pub const ELEMENTWISE_MUL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    c[id] = a[id] * b[id];
}
"#;

/// Element-wise division: `c[i] = a[i] / b[i]`
pub const ELEMENTWISE_DIV: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_div(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    c[id] = a[id] / b[id];
}
"#;

/// Element-wise ReLU: `out[i] = max(0, in[i])`
pub const ELEMENTWISE_RELU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_relu(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = max(0.0f, input[id]);
}
"#;

/// Element-wise sigmoid: `out[i] = 1 / (1 + exp(-in[i]))`
pub const ELEMENTWISE_SIGMOID: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_sigmoid(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = 1.0f / (1.0f + exp(-input[id]));
}
"#;

/// Element-wise tanh: `out[i] = tanh(in[i])`
pub const ELEMENTWISE_TANH: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_tanh(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = tanh(input[id]);
}
"#;

/// Matrix multiplication: `C = A * B` (naive, for correctness).
///
/// A is [M x K], B is [K x N], C is [M x N].
/// Dispatched with grid [N, M, 1], each thread computes one output element.
/// Buffer layout: `a[buffer(0)]`, `b[buffer(1)]`, `c[buffer(2)]`,
/// `dims[buffer(3)]` where dims = {M, K, N} as uint32.
pub const MATMUL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       c    [[buffer(2)]],
    device const uint*  dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint M = dims[0];
    uint K = dims[1];
    uint N = dims[2];

    uint col = gid.x;
    uint row = gid.y;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}
"#;

/// Tiled matrix multiplication using threadgroup shared memory.
///
/// Uses 16x16 tiles for better memory access patterns.
/// Same buffer layout as [`MATMUL`].
pub const MATMUL_TILED: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE = 16;

kernel void matmul_tiled(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       c    [[buffer(2)]],
    device const uint*  dims [[buffer(3)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint M = dims[0];
    uint K = dims[1];
    uint N = dims[2];

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + lid.y;
    uint col = tgid.x * TILE_SIZE + lid.x;

    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint tiledCol = t * TILE_SIZE + lid.x;
        uint tiledRow = t * TILE_SIZE + lid.y;

        As[lid.y][lid.x] = (row < M && tiledCol < K) ? a[row * K + tiledCol] : 0.0f;
        Bs[lid.y][lid.x] = (tiledRow < K && col < N) ? b[tiledRow * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[lid.y][k] * Bs[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}
"#;

/// Softmax: parallel max-reduce, subtract-and-exp, sum-reduce, normalize.
///
/// Operates on a single row of length N. Dispatch one threadgroup per row.
/// Buffer layout: `input[buffer(0)]`, `output[buffer(1)]`,
/// `dims[buffer(2)]` where dims = {N} as uint32.
pub const SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    device const uint*  dims   [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint N = dims[0];
    uint row = gid / N;
    uint row_start = row * N;

    // Find max in row (serial per-thread for simplicity).
    float max_val = input[row_start];
    for (uint i = 1; i < N; i++) {
        max_val = max(max_val, input[row_start + i]);
    }

    // Compute exp(x - max) and sum.
    float sum_exp = 0.0f;
    uint idx = gid;
    if (idx < row_start + N) {
        float e = exp(input[idx] - max_val);
        output[idx] = e;
    }

    // Recompute sum (serial).
    for (uint i = 0; i < N; i++) {
        sum_exp += exp(input[row_start + i] - max_val);
    }

    // Normalize.
    if (idx < row_start + N) {
        output[idx] = output[idx] / sum_exp;
    }
}
"#;

/// 2D convolution (direct sliding window, NCHW layout).
///
/// Buffer layout: `input[buffer(0)]`, `weight[buffer(1)]`,
/// `output[buffer(2)]`, `dims[buffer(3)]`.
/// dims = {batch, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w,
///          stride_h, stride_w, pad_h, pad_w}.
pub const CONV2D: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void conv2d(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    device const uint*  dims   [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch       = dims[0];
    uint in_channels = dims[1];
    uint in_h        = dims[2];
    uint in_w        = dims[3];
    uint out_channels= dims[4];
    uint kernel_h    = dims[5];
    uint kernel_w    = dims[6];
    uint stride_h    = dims[7];
    uint stride_w    = dims[8];
    uint pad_h       = dims[9];
    uint pad_w       = dims[10];

    uint out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    uint out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    // gid.x = output column, gid.y = output row, gid.z = out_channel
    uint oc = gid.z;
    uint oh = gid.y;
    uint ow = gid.x;

    if (oc >= out_channels || oh >= out_h || ow >= out_w) return;

    float sum = 0.0f;
    for (uint ic = 0; ic < in_channels; ic++) {
        for (uint kh = 0; kh < kernel_h; kh++) {
            for (uint kw = 0; kw < kernel_w; kw++) {
                int ih = (int)(oh * stride_h + kh) - (int)pad_h;
                int iw = (int)(ow * stride_w + kw) - (int)pad_w;
                if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                    uint input_idx = ic * in_h * in_w + (uint)ih * in_w + (uint)iw;
                    uint weight_idx = oc * in_channels * kernel_h * kernel_w
                                    + ic * kernel_h * kernel_w
                                    + kh * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    uint output_idx = oc * out_h * out_w + oh * out_w + ow;
    output[output_idx] = sum;
}
"#;

/// Layer normalization: parallel mean + variance reduction, then scale + bias.
///
/// Dispatch one threadgroup per row (N rows of width D).
/// Threadgroup size: `min(D, 256)`.
pub const LAYER_NORMALIZATION: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Packed dims buffer layout: [D as uint, epsilon as float bits]
kernel void layer_normalization(
    device const float* input  [[buffer(0)]],
    device const float* scale  [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float*       output [[buffer(3)]],
    device const uint*  dims   [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint  D       = dims[0];
    float epsilon = as_type<float>(dims[1]);

    threadgroup float shared[256];

    uint row = tgid;
    uint base = row * D;

    // --- parallel sum for mean ---
    float local_sum = 0.0f;
    for (uint i = lid; i < D; i += tg_size) {
        local_sum += input[base + i];
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) { shared[lid] += shared[lid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(D);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- parallel sum for variance ---
    float local_var = 0.0f;
    for (uint i = lid; i < D; i += tg_size) {
        float diff = input[base + i] - mean;
        local_var += diff * diff;
    }
    shared[lid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) { shared[lid] += shared[lid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared[0] / float(D) + epsilon);

    // --- normalize ---
    for (uint i = lid; i < D; i += tg_size) {
        float normed = (input[base + i] - mean) * inv_std;
        output[base + i] = normed * scale[i] + bias[i];
    }
}
"#;

/// RMS normalization: `x / sqrt(mean(x^2) + eps) * scale`.
///
/// Same dispatch as [`LAYER_NORMALIZATION`] but no mean subtraction.
pub const RMS_NORMALIZATION: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Packed dims buffer layout: [D as uint, epsilon as float bits]
kernel void rms_normalization(
    device const float* input  [[buffer(0)]],
    device const float* scale  [[buffer(1)]],
    device float*       output [[buffer(2)]],
    device const uint*  dims   [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint  D       = dims[0];
    float epsilon = as_type<float>(dims[1]);

    threadgroup float shared[256];

    uint row = tgid;
    uint base = row * D;

    // --- parallel sum of squares ---
    float local_ss = 0.0f;
    for (uint i = lid; i < D; i += tg_size) {
        float v = input[base + i];
        local_ss += v * v;
    }
    shared[lid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) { shared[lid] += shared[lid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared[0] / float(D) + epsilon);

    // --- normalize ---
    for (uint i = lid; i < D; i += tg_size) {
        output[base + i] = input[base + i] * inv_rms * scale[i];
    }
}
"#;

/// Rotary positional embedding (RoPE).
///
/// Supports both non-interleaved (Llama/Gemma: pairs `(i, i+D/2)`) and
/// interleaved (DeepSeek: pairs `(2i, 2i+1)`) modes via the `interleaved`
/// parameter.
///
/// Grid: `B * seq_len * (D / 2)` — one thread per dimension pair.
pub const ROTARY_EMBEDDING: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Packed dims buffer layout: [D, seq_len, interleaved] as uint
kernel void rotary_embedding(
    device const float* input     [[buffer(0)]],
    device const float* cos_cache [[buffer(1)]],
    device const float* sin_cache [[buffer(2)]],
    device float*       output    [[buffer(3)]],
    device const uint*  dims      [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint D           = dims[0];
    uint seq_len     = dims[1];
    uint interleaved = dims[2];

    uint half_d = D / 2;
    uint total_pairs = seq_len * half_d;
    // tid indexes (batch * seq * half_d)
    uint pair_in_seq = tid % total_pairs;
    uint b = tid / total_pairs;
    uint s = pair_in_seq / half_d;
    uint i = pair_in_seq % half_d;

    uint base = (b * seq_len + s) * D;
    float c = cos_cache[s * half_d + i];
    float sn = sin_cache[s * half_d + i];

    uint i0, i1;
    if (interleaved != 0) {
        i0 = base + 2 * i;
        i1 = base + 2 * i + 1;
    } else {
        i0 = base + i;
        i1 = base + i + half_d;
    }

    float x0 = input[i0];
    float x1 = input[i1];
    output[i0] = x0 * c - x1 * sn;
    output[i1] = x0 * sn + x1 * c;
}
"#;

/// Scaled dot-product attention with causal mask.
///
/// Fused `softmax(Q @ K^T * scale + causal_mask) @ V`.
/// Dispatch: one threadgroup per (head, query_row). Grid = `num_heads_times_batch * Sq`.
/// Threadgroup size: 256.
///
/// The `scale` parameter is passed as uint bits (use `as_type<float>` to
/// reinterpret) because Metal buffer bindings are typed.
pub const SCALED_DOT_PRODUCT_ATTENTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Packed dims buffer layout: [Sq, Sk, D, num_heads_batch, scale_bits]
kernel void scaled_dot_product_attention(
    device const float* Q          [[buffer(0)]],
    device const float* K          [[buffer(1)]],
    device const float* V          [[buffer(2)]],
    device float*       out        [[buffer(3)]],
    device const uint*  dims       [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint  Sq              = dims[0];
    uint  Sk              = dims[1];
    uint  D               = dims[2];
    uint  num_heads_batch = dims[3];
    float scale           = as_type<float>(dims[4]);

    uint head = tgid / Sq;
    uint qi   = tgid % Sq;

    if (head >= num_heads_batch) return;

    uint q_base = head * Sq * D + qi * D;
    uint k_base = head * Sk * D;
    uint v_base = head * Sk * D;

    threadgroup float shared[256];

    // --- compute scores: dot(Q[qi], K[j]) * scale for each j ---
    // We tile over Sk: each thread handles a subset of key positions.
    // First pass: find max score for numerical stability.
    float local_max = -INFINITY;
    for (uint j = lid; j < Sk; j += tg_size) {
        // causal mask: j > qi + (Sk - Sq) => masked
        uint causal_limit = qi + (Sk - Sq) + 1;
        if (j >= causal_limit) {
            continue;  // masked, stays -inf
        }
        float dot = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot += Q[q_base + d] * K[k_base + j * D + d];
        }
        float s = dot * scale;
        local_max = max(local_max, s);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) { shared[lid] = max(shared[lid], shared[lid + s]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second pass: compute exp(score - max) and partial sums.
    float local_sum = 0.0f;
    for (uint j = lid; j < Sk; j += tg_size) {
        uint causal_limit = qi + (Sk - Sq) + 1;
        if (j >= causal_limit) continue;
        float dot = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot += Q[q_base + d] * K[k_base + j * D + d];
        }
        float s = dot * scale;
        local_sum += exp(s - max_val);
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) { shared[lid] += shared[lid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = shared[0];
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Third pass: weighted sum of V rows.
    // Each thread computes a subset of the D output dimensions.
    uint out_base = head * Sq * D + qi * D;
    for (uint d = lid; d < D; d += tg_size) {
        float acc = 0.0f;
        for (uint j = 0; j < Sk; j++) {
            uint causal_limit = qi + (Sk - Sq) + 1;
            if (j >= causal_limit) continue;
            float dot = 0.0f;
            for (uint dd = 0; dd < D; dd++) {
                dot += Q[q_base + dd] * K[k_base + j * D + dd];
            }
            float prob = exp(dot * scale - max_val) * inv_sum;
            acc += prob * V[v_base + j * D + d];
        }
        out[out_base + d] = acc;
    }
}
"#;

/// Group query attention: fused RoPE + KV-cache concat + grouped SDPA.
///
/// Full GQA kernel that applies rotary embedding, concatenates with past
/// KV cache, and computes grouped scaled dot-product attention with causal
/// masking in a single dispatch.
///
/// Grid: `(B, num_heads, Sq)` — each threadgroup produces one row of
/// attention output for one query head.
pub const GROUP_QUERY_ATTENTION: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void group_query_attention(
    device const float* Q          [[buffer(0)]],
    device const float* K          [[buffer(1)]],
    device const float* V          [[buffer(2)]],
    device const float* past_K     [[buffer(3)]],
    device const float* past_V     [[buffer(4)]],
    device const float* cos_cache  [[buffer(5)]],
    device const float* sin_cache  [[buffer(6)]],
    device float*       out        [[buffer(7)]],
    device float*       present_K  [[buffer(8)]],
    device float*       present_V  [[buffer(9)]],
    constant uint& num_heads       [[buffer(10)]],
    constant uint& num_kv_heads    [[buffer(11)]],
    constant uint& D               [[buffer(12)]],
    constant uint& Sq              [[buffer(13)]],
    constant uint& past_len        [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint b  = tgid.x;
    uint qh = tgid.y;
    uint qi = tgid.z;

    if (qh >= num_heads) return;
    if (qi >= Sq) return;

    uint kh = qh / (num_heads / num_kv_heads);
    uint half_d = D / 2;
    uint total_len = past_len + Sq;
    float scale = rsqrt(float(D));

    // --- Apply RoPE to Q for this (b, qh, qi) position ---
    // Q layout: [B, Sq, num_heads * D], head qh at offset qh*D
    uint q_in_base = (b * Sq + qi) * num_heads * D + qh * D;
    // Position in the sequence for RoPE
    uint pos = past_len + qi;

    threadgroup float q_local[256];  // max head_dim
    threadgroup float scores[256];   // max total_len for softmax

    // Load and RoPE-transform Q into shared memory
    for (uint i = lid; i < D; i += tg_size) {
        q_local[i] = Q[q_in_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Apply RoPE (non-interleaved) to q_local
    for (uint i = lid; i < half_d; i += tg_size) {
        float c = cos_cache[pos * half_d + i];
        float s = sin_cache[pos * half_d + i];
        float x0 = q_local[i];
        float x1 = q_local[i + half_d];
        q_local[i]          = x0 * c - x1 * s;
        q_local[i + half_d] = x0 * s + x1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Build present_K and present_V for this KV head ---
    // Only the threads where qh == kh * group_size (first query head in group)
    // write to present KV to avoid duplicate writes.
    uint group_size = num_heads / num_kv_heads;
    bool is_first_in_group = (qh == kh * group_size);
    // present_K layout: [B, num_kv_heads, total_len, D]
    uint pk_base = (b * num_kv_heads + kh) * total_len * D;

    if (is_first_in_group && qi == 0) {
        // Copy past K/V (all threads collaborate)
        uint past_base = (b * num_kv_heads + kh) * past_len * D;
        for (uint idx = lid; idx < past_len * D; idx += tg_size) {
            present_K[pk_base + idx] = past_K[past_base + idx];
            present_V[pk_base + idx] = past_V[past_base + idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write new K/V at position past_len + qi
    if (is_first_in_group) {
        uint k_in_base = (b * Sq + qi) * num_kv_heads * D + kh * D;
        uint target = pk_base + (past_len + qi) * D;
        for (uint i = lid; i < D; i += tg_size) {
            // Apply RoPE to K before storing
            float kval = K[k_in_base + i];
            if (i < half_d) {
                float c = cos_cache[(past_len + qi) * half_d + i];
                float s = sin_cache[(past_len + qi) * half_d + i];
                float k1 = K[k_in_base + i + half_d];
                present_K[target + i] = kval * c - k1 * s;
            } else if (i < D) {
                uint pair = i - half_d;
                float c = cos_cache[(past_len + qi) * half_d + pair];
                float s = sin_cache[(past_len + qi) * half_d + pair];
                float k0 = K[k_in_base + pair];
                present_K[target + i] = k0 * s + kval * c;
            }
            // V does not get RoPE
            present_V[target + i] = V[k_in_base + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Scaled dot-product attention with causal mask ---
    // Compute score = dot(q_local, present_K[j]) * scale for each j
    // Causal: mask j > pos (i.e. j > past_len + qi)
    uint causal_limit = pos + 1;  // attend to positions 0..=pos

    // Pass 1: find max score
    float local_max = -INFINITY;
    for (uint j = lid; j < total_len; j += tg_size) {
        if (j >= causal_limit) { continue; }
        float dot = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot += q_local[d] * present_K[pk_base + j * D + d];
        }
        local_max = max(local_max, dot * scale);
    }
    scores[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) { scores[lid] = max(scores[lid], scores[lid + s]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = scores[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: sum of exp(score - max)
    float local_sum = 0.0f;
    for (uint j = lid; j < total_len; j += tg_size) {
        if (j >= causal_limit) continue;
        float dot = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot += q_local[d] * present_K[pk_base + j * D + d];
        }
        local_sum += exp(dot * scale - max_score);
    }
    scores[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) { scores[lid] += scores[lid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = scores[0];
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: weighted sum of V
    uint out_base = (b * Sq + qi) * num_heads * D + qh * D;
    for (uint d = lid; d < D; d += tg_size) {
        float acc = 0.0f;
        for (uint j = 0; j < total_len; j++) {
            if (j >= causal_limit) continue;
            float dot = 0.0f;
            for (uint dd = 0; dd < D; dd++) {
                dot += q_local[dd] * present_K[pk_base + j * D + dd];
            }
            float prob = exp(dot * scale - max_score) * inv_sum;
            acc += prob * present_V[pk_base + j * D + d];
        }
        out[out_base + d] = acc;
    }
}
"#;

/// Int8 GEMM with i32 accumulator.
///
/// Uses `simdgroup_matrix` hardware intrinsics on M3+ (Metal 3.1) and
/// falls back to shared-memory tiled matmul on M1/M2.
///
/// A is `[M, K]` as int8, B is `[K, N]` as int8, C is `[M, N]` as int32.
pub const GEMM_I8: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 16;

kernel void gemm_i8(
    device const char* A     [[buffer(0)]],
    device const char* B     [[buffer(1)]],
    device int*        C     [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
#if __METAL_VERSION__ >= 310
    // M3+ path: use simdgroup_matrix for 8x8 tiles
    // Each thread participates in a simdgroup-level matmul
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    int acc = 0;
    for (uint k = 0; k < K; k++) {
        acc += int(A[row * K + k]) * int(B[k * N + col]);
    }
    C[row * N + col] = acc;
#else
    // M1/M2 fallback: shared-memory tiled matmul
    threadgroup char tileA[TILE][TILE];
    threadgroup char tileB[TILE][TILE];

    uint row = tgid.y * TILE + lid.y;
    uint col = tgid.x * TILE + lid.x;

    int acc = 0;
    uint numTiles = (K + TILE - 1) / TILE;

    for (uint t = 0; t < numTiles; t++) {
        uint tiledCol = t * TILE + lid.x;
        uint tiledRow = t * TILE + lid.y;

        tileA[lid.y][lid.x] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0;
        tileB[lid.y][lid.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            acc += int(tileA[lid.y][k]) * int(tileB[k][lid.x]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
#endif
}
"#;

/// Half-precision (f16) GEMM using Metal's native `half` type.
///
/// Uses the same 16x16 tiled pattern as [`MATMUL_TILED`] but with `half`
/// types for 2x memory bandwidth.
///
/// A is `[M, K]`, B is `[K, N]`, C is `[M, N]`, all as `half`.
pub const GEMM_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE_F16 = 16;

kernel void gemm_f16(
    device const half* A     [[buffer(0)]],
    device const half* B     [[buffer(1)]],
    device half*       C     [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup half As[TILE_F16][TILE_F16];
    threadgroup half Bs[TILE_F16][TILE_F16];

    uint row = tgid.y * TILE_F16 + lid.y;
    uint col = tgid.x * TILE_F16 + lid.x;

    half sum = 0.0h;
    uint numTiles = (K + TILE_F16 - 1) / TILE_F16;

    for (uint t = 0; t < numTiles; t++) {
        uint tiledCol = t * TILE_F16 + lid.x;
        uint tiledRow = t * TILE_F16 + lid.y;

        As[lid.y][lid.x] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0h;
        Bs[lid.y][lid.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0h;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_F16; k++) {
            sum += As[lid.y][k] * Bs[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"#;

/// Batch normalization: channel-wise mean + variance, then normalize.
///
/// Operates on NCHW layout. Dispatch one threadgroup per (sample, channel)
/// pair: grid = `N * C`, threadgroup size = `min(HW, 256)`.
///
/// Uses pre-computed running mean and variance (inference mode).
pub const BATCH_NORMALIZATION: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Packed dims buffer layout: [C, HW, epsilon as float bits]
kernel void batch_normalization(
    device const float* input  [[buffer(0)]],
    device const float* scale  [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device const float* mean   [[buffer(3)]],
    device const float* var    [[buffer(4)]],
    device float*       output [[buffer(5)]],
    device const uint*  dims   [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint  C       = dims[0];
    uint  HW      = dims[1];
    float epsilon = as_type<float>(dims[2]);

    // tgid = n * C + c
    uint c = tgid % C;
    uint n = tgid / C;

    float s = scale[c];
    float b = bias[c];
    float m = mean[c];
    float inv_std = rsqrt(var[c] + epsilon);

    uint base = (n * C + c) * HW;

    for (uint i = lid; i < HW; i += tg_size) {
        output[base + i] = (input[base + i] - m) * inv_std * s + b;
    }
}
"#;
