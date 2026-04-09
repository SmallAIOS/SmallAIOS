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
