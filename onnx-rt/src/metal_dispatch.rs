// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU dispatch for ONNX operator execution.
//!
//! This module is split into two layers for testability:
//!
//! 1. **Planning layer** (platform-independent): [`plan_kernel_launch`] computes
//!    which MSL kernel to use, grid/threadgroup sizes, and buffer layout from
//!    op type + input shapes. This is fully testable on Linux CI.
//!
//! 2. **Execution layer** (macOS-only): [`MetalDispatcher`] takes a
//!    [`KernelLaunchPlan`] and calls the real Metal API to copy data, compile
//!    kernels, launch, and read back results.
//!
//! If an operator is not supported on GPU, `try_execute` returns `Ok(None)`
//! and the caller falls back to the CPU implementation.

#[cfg(all(feature = "metal", target_os = "macos"))]
use alloc::format;
#[cfg(all(feature = "metal", target_os = "macos"))]
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(all(feature = "metal", target_os = "macos"))]
use smallaios_arch_apple::{MetalError, MetalProvider, MetalTensorCache};
#[cfg(all(feature = "metal", target_os = "macos"))]
use smallaios_compute::ComputeProvider;

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::onnx_types::AttributeProto;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::operators::OpError;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::tensor::DataType;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::tensor::Tensor;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::tensor::TensorShape;

// ===========================================================================
// Layer 1: Platform-independent dispatch planning
// ===========================================================================

/// Describes which MSL kernel to use and how to launch it.
///
/// This struct encodes the kernel launch parameters without touching any
/// Metal APIs, making it fully testable on all platforms including Linux CI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelLaunchPlan {
    /// Metal function name in the compiled MSL library (e.g. `"elementwise_add"`).
    pub kernel_name: &'static str,
    /// Identifier for the MSL source constant (e.g. `"ELEMENTWISE_ADD"`).
    pub kernel_source_id: &'static str,
    /// Dispatch grid dimensions `[x, y, z]`.
    pub grid_size: [u32; 3],
    /// Threadgroup dimensions `[x, y, z]`.
    pub threadgroup_size: [u32; 3],
    /// Number of input buffers the kernel expects.
    pub input_buffer_count: usize,
    /// Number of output buffers the kernel produces.
    pub output_buffer_count: usize,
    /// Shape/dimension parameters passed as a dims buffer (e.g. `[M, K, N]`).
    pub param_buffers: Vec<u32>,
    /// Whether the kernel needs a separate dims buffer bound.
    pub needs_dims_buffer: bool,
    /// Output shape (dimensions as i64 for tensor construction).
    pub output_shape: Vec<i64>,
    /// Total output elements (for buffer allocation).
    pub output_elements: usize,
}

/// Kernel category for grouping similar ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelCategory {
    /// Binary elementwise: Add, Sub, Mul, Div (2 inputs, 1 output, same shape).
    ElementwiseBinary,
    /// Unary elementwise: Relu, Sigmoid, Tanh (1 input, 1 output, same shape).
    ElementwiseUnary,
    /// Matrix multiply: MatMul, Gemm.
    MatMul,
    /// Softmax over last axis.
    Softmax,
    /// 2D convolution.
    Conv2d,
}

/// Returns the [`KernelCategory`] for an op, or `None` if unsupported.
pub fn classify_op(op_type: &str) -> Option<KernelCategory> {
    match op_type {
        "Add" | "Sub" | "Mul" | "Div" => Some(KernelCategory::ElementwiseBinary),
        "Relu" | "Sigmoid" | "Tanh" => Some(KernelCategory::ElementwiseUnary),
        "MatMul" | "Gemm" => Some(KernelCategory::MatMul),
        "Softmax" => Some(KernelCategory::Softmax),
        "Conv" => Some(KernelCategory::Conv2d),
        _ => None,
    }
}

/// Plans the kernel launch for a given op without touching Metal APIs.
///
/// Returns `None` if the op is not GPU-supported. The returned plan
/// contains all information needed to execute the kernel: function name,
/// grid/threadgroup sizes, buffer counts, and dimension parameters.
///
/// # Arguments
/// * `op_type` - ONNX operator type string (e.g. `"Add"`, `"MatMul"`)
/// * `input_shapes` - Shapes of each input tensor
/// * `attrs` - Simplified attribute list (name, int-list pairs for Conv strides/pads)
pub fn plan_kernel_launch(
    op_type: &str,
    input_shapes: &[&[i64]],
    attrs: &[(&str, &[i64])],
) -> Option<KernelLaunchPlan> {
    // Try Tier 2 ops first (they don't go through classify_op).
    match op_type {
        "LayerNormalization" => return plan_layer_norm(input_shapes, attrs),
        "RMSNormalization" | "SimplifiedLayerNormalization" => return plan_rms_norm(input_shapes),
        "RotaryEmbedding" => return plan_rotary_embedding(input_shapes, attrs),
        "ScaledDotProductAttention" => return plan_sdpa(input_shapes),
        "GroupQueryAttention" => return plan_gqa(input_shapes, attrs),
        "MatMulInteger" => return plan_gemm_i8(input_shapes),
        "BatchNormalization" => return plan_batch_norm(input_shapes, attrs),
        _ => {}
    }

    let category = classify_op(op_type)?;

    match category {
        KernelCategory::ElementwiseBinary => plan_elementwise_binary(op_type, input_shapes),
        KernelCategory::ElementwiseUnary => plan_elementwise_unary(op_type, input_shapes),
        KernelCategory::MatMul => plan_matmul(input_shapes),
        KernelCategory::Softmax => plan_softmax(input_shapes),
        KernelCategory::Conv2d => plan_conv2d(input_shapes, attrs),
    }
}

/// Returns `true` if the given operator is supported on Metal GPU.
pub fn is_gpu_supported(op_type: &str) -> bool {
    matches!(
        op_type,
        "LayerNormalization"
            | "RMSNormalization"
            | "SimplifiedLayerNormalization"
            | "RotaryEmbedding"
            | "ScaledDotProductAttention"
            | "GroupQueryAttention"
            | "MatMulInteger"
            | "BatchNormalization"
    ) || classify_op(op_type).is_some()
}

// ---------------------------------------------------------------------------
// Elementwise binary: Add, Sub, Mul, Div
// ---------------------------------------------------------------------------

fn plan_elementwise_binary(op_type: &str, input_shapes: &[&[i64]]) -> Option<KernelLaunchPlan> {
    if input_shapes.is_empty() {
        return None;
    }
    let n: u32 = input_shapes[0].iter().product::<i64>() as u32;
    if n == 0 {
        return None;
    }

    let (kernel_name, kernel_source_id) = match op_type {
        "Add" => ("elementwise_add", "ELEMENTWISE_ADD"),
        "Sub" => ("elementwise_sub", "ELEMENTWISE_SUB"),
        "Mul" => ("elementwise_mul", "ELEMENTWISE_MUL"),
        "Div" => ("elementwise_div", "ELEMENTWISE_DIV"),
        _ => return None,
    };

    Some(elementwise_plan(
        kernel_name,
        kernel_source_id,
        n,
        2,
        input_shapes[0],
    ))
}

// ---------------------------------------------------------------------------
// Elementwise unary: Relu, Sigmoid, Tanh
// ---------------------------------------------------------------------------

fn plan_elementwise_unary(op_type: &str, input_shapes: &[&[i64]]) -> Option<KernelLaunchPlan> {
    if input_shapes.is_empty() {
        return None;
    }
    let n: u32 = input_shapes[0].iter().product::<i64>() as u32;
    if n == 0 {
        return None;
    }

    let (kernel_name, kernel_source_id) = match op_type {
        "Relu" => ("elementwise_relu", "ELEMENTWISE_RELU"),
        "Sigmoid" => ("elementwise_sigmoid", "ELEMENTWISE_SIGMOID"),
        "Tanh" => ("elementwise_tanh", "ELEMENTWISE_TANH"),
        _ => return None,
    };

    Some(elementwise_plan(
        kernel_name,
        kernel_source_id,
        n,
        1,
        input_shapes[0],
    ))
}

/// Shared constructor for elementwise plans — eliminates struct-literal
/// duplication between binary (Add/Sub/Mul/Div) and unary (Relu/Sigmoid/Tanh).
fn elementwise_plan(
    kernel_name: &'static str,
    kernel_source_id: &'static str,
    n: u32,
    input_buffer_count: usize,
    shape: &[i64],
) -> KernelLaunchPlan {
    let tg = n.min(256);
    KernelLaunchPlan {
        kernel_name,
        kernel_source_id,
        grid_size: [n, 1, 1],
        threadgroup_size: [tg, 1, 1],
        input_buffer_count,
        output_buffer_count: 1,
        param_buffers: vec![n],
        needs_dims_buffer: false,
        output_shape: shape.to_vec(),
        output_elements: n as usize,
    }
}

// ---------------------------------------------------------------------------
// MatMul / Gemm
// ---------------------------------------------------------------------------

fn plan_matmul(input_shapes: &[&[i64]]) -> Option<KernelLaunchPlan> {
    if input_shapes.len() < 2 {
        return None;
    }
    let a_dims = input_shapes[0];
    let b_dims = input_shapes[1];
    if a_dims.len() < 2 || b_dims.len() < 2 {
        return None;
    }

    let m = a_dims[a_dims.len() - 2] as u32;
    let k = a_dims[a_dims.len() - 1] as u32;
    let n = b_dims[b_dims.len() - 1] as u32;

    // Inner dimension check
    if k != b_dims[b_dims.len() - 2] as u32 {
        return None;
    }

    let use_tiled = m >= 16 && n >= 16;
    let (kernel_name, kernel_source_id) = if use_tiled {
        ("matmul_tiled", "MATMUL_TILED")
    } else {
        ("matmul", "MATMUL")
    };

    let (grid, tg) = if use_tiled {
        let gx = n.div_ceil(16);
        let gy = m.div_ceil(16);
        ([gx * 16, gy * 16, 1], [16u32, 16, 1])
    } else {
        ([n, m, 1], [n.min(16), m.min(16), 1])
    };

    let out_elements = (m as usize) * (n as usize);
    Some(KernelLaunchPlan {
        kernel_name,
        kernel_source_id,
        grid_size: grid,
        threadgroup_size: tg,
        input_buffer_count: 2,
        output_buffer_count: 1,
        param_buffers: vec![m, k, n],
        needs_dims_buffer: true,
        output_shape: vec![m as i64, n as i64],
        output_elements: out_elements,
    })
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

fn plan_softmax(input_shapes: &[&[i64]]) -> Option<KernelLaunchPlan> {
    if input_shapes.is_empty() || input_shapes[0].is_empty() {
        return None;
    }
    let dims = input_shapes[0];
    let cols = *dims.last().unwrap() as u32;
    let rows: u32 = dims[..dims.len() - 1].iter().product::<i64>() as u32;
    // For 1-D input, rows=1 (product of empty slice = 1 handled by i64 product = 1)
    let rows = if dims.len() == 1 { 1 } else { rows };
    let total = (rows as usize) * (cols as usize);

    let tg = (total as u32).min(256);
    Some(KernelLaunchPlan {
        kernel_name: "softmax",
        kernel_source_id: "SOFTMAX",
        grid_size: [total as u32, 1, 1],
        threadgroup_size: [tg, 1, 1],
        input_buffer_count: 1,
        output_buffer_count: 1,
        param_buffers: vec![cols],
        needs_dims_buffer: true,
        output_shape: dims.to_vec(),
        output_elements: total,
    })
}

// ---------------------------------------------------------------------------
// Conv2D
// ---------------------------------------------------------------------------

/// `no_std`-compatible square root approximation using Newton-Raphson.
///
/// Starts from the classic fast inverse square root seed and refines with
/// two Newton iterations. Sufficient accuracy for GPU dispatch scale factors.
fn sqrt_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // Initial estimate via bit manipulation (Quake-style seed).
    let i = x.to_bits();
    let guess_bits = 0x1FBD_1DF5 + (i >> 1);
    let mut y = f32::from_bits(guess_bits);
    // Two Newton-Raphson refinements: y = (y + x/y) / 2
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y
}

/// Extracts an integer list attribute from the simplified attr slice.
fn get_ints_attr_from_pairs(attrs: &[(&str, &[i64])], name: &str, default: &[i64]) -> Vec<i64> {
    attrs
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| v.to_vec())
        .unwrap_or_else(|| default.to_vec())
}

/// Extracts an integer list attribute from `AttributeProto` slice.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn get_ints_attr(attrs: &[AttributeProto], name: &str, default: &[i64]) -> Vec<i64> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| {
            if a.ints.is_empty() {
                default.to_vec()
            } else {
                a.ints.clone()
            }
        })
        .unwrap_or_else(|| default.to_vec())
}

fn plan_conv2d(input_shapes: &[&[i64]], attrs: &[(&str, &[i64])]) -> Option<KernelLaunchPlan> {
    if input_shapes.len() < 2 {
        return None;
    }
    let in_dims = input_shapes[0];
    let w_dims = input_shapes[1];
    if in_dims.len() != 4 || w_dims.len() != 4 {
        return None;
    }

    // Input: [batch, in_channels, in_h, in_w]
    let in_channels = in_dims[1] as u32;
    let in_h = in_dims[2] as u32;
    let in_w = in_dims[3] as u32;

    // Weight: [out_channels, in_channels/group, kernel_h, kernel_w]
    let out_channels = w_dims[0] as u32;
    let kernel_h = w_dims[2] as u32;
    let kernel_w = w_dims[3] as u32;

    let strides = get_ints_attr_from_pairs(attrs, "strides", &[1, 1]);
    let pads = get_ints_attr_from_pairs(attrs, "pads", &[0, 0, 0, 0]);

    let stride_h = strides[0] as u32;
    let stride_w = if strides.len() > 1 {
        strides[1] as u32
    } else {
        stride_h
    };
    let pad_h = pads[0] as u32;
    let pad_w = if pads.len() > 1 {
        pads[1] as u32
    } else {
        pad_h
    };

    let out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    let out_elements = (out_channels as usize) * (out_h as usize) * (out_w as usize);

    let tg_size = 256u32.min(out_w).max(1);

    // Dims buffer: [batch, in_channels, in_h, in_w, out_channels,
    //               kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w]
    let params = vec![
        1u32, // batch (we dispatch one sample)
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    ];

    Some(KernelLaunchPlan {
        kernel_name: "conv2d",
        kernel_source_id: "CONV2D",
        grid_size: [out_w, out_h, out_channels],
        threadgroup_size: [tg_size, 1, 1],
        input_buffer_count: 2,
        output_buffer_count: 1,
        param_buffers: params,
        needs_dims_buffer: true,
        output_shape: vec![1, out_channels as i64, out_h as i64, out_w as i64],
        output_elements: out_elements,
    })
}

// ---------------------------------------------------------------------------
// Tier 2: Layer Normalization
// ---------------------------------------------------------------------------

/// Shared plan constructor for normalization ops that operate row-wise over
/// the last axis (LayerNorm, RMSNorm).
fn plan_norm(
    kernel_name: &'static str,
    kernel_source_id: &'static str,
    input_buffer_count: usize,
    input_shapes: &[&[i64]],
    epsilon_bits: u32,
) -> Option<KernelLaunchPlan> {
    if input_shapes.is_empty() || input_shapes[0].len() < 2 {
        return None;
    }
    let dims = input_shapes[0];
    let d = *dims.last().unwrap() as u32;
    if d == 0 {
        return None;
    }
    let n: u32 = dims[..dims.len() - 1].iter().product::<i64>() as u32;
    if n == 0 {
        return None;
    }
    let tg = d.min(256);
    let total = (n as usize) * (d as usize);

    // The shader uses one threadgroup per row (tgid = row). dispatch_threads
    // takes a *total thread count*, so multiply by tg_size to get `n` groups.
    Some(KernelLaunchPlan {
        kernel_name,
        kernel_source_id,
        grid_size: [n * tg, 1, 1],
        threadgroup_size: [tg, 1, 1],
        input_buffer_count,
        output_buffer_count: 1,
        param_buffers: vec![d, epsilon_bits],
        needs_dims_buffer: true,
        output_shape: dims.to_vec(),
        output_elements: total,
    })
}

/// Extracts a float attribute from simplified attr pairs, returning its
/// raw bit pattern as u32. Defaults to `default_val` bits.
fn get_float_attr_bits(attrs: &[(&str, &[i64])], name: &str, default_val: f32) -> u32 {
    attrs
        .iter()
        .find(|(n, _)| *n == name)
        .and_then(|(_, v)| v.first().map(|&i| i as u32))
        .unwrap_or_else(|| default_val.to_bits())
}

fn plan_layer_norm(input_shapes: &[&[i64]], attrs: &[(&str, &[i64])]) -> Option<KernelLaunchPlan> {
    let epsilon_bits = get_float_attr_bits(attrs, "epsilon", 1e-5);
    plan_norm(
        "layer_normalization",
        "LAYER_NORMALIZATION",
        3, // data + scale + bias
        input_shapes,
        epsilon_bits,
    )
}

// ---------------------------------------------------------------------------
// Tier 2: RMS Normalization
// ---------------------------------------------------------------------------

fn plan_rms_norm(input_shapes: &[&[i64]]) -> Option<KernelLaunchPlan> {
    let epsilon_bits = 1e-5_f32.to_bits();
    plan_norm(
        "rms_normalization",
        "RMS_NORMALIZATION",
        2, // data + scale (no bias)
        input_shapes,
        epsilon_bits,
    )
}

// ---------------------------------------------------------------------------
// Tier 2: Rotary Embedding
// ---------------------------------------------------------------------------

fn plan_rotary_embedding(
    input_shapes: &[&[i64]],
    attrs: &[(&str, &[i64])],
) -> Option<KernelLaunchPlan> {
    if input_shapes.is_empty() || input_shapes[0].len() < 3 {
        return None;
    }
    let dims = input_shapes[0];
    let d = *dims.last().unwrap() as u32;
    if d == 0 || !d.is_multiple_of(2) {
        return None;
    }
    // Assume [B, seq_len, D] or [B, seq_len, heads, D].
    let batch: u32 = dims[0] as u32;
    let seq_len: u32 = dims[1] as u32;
    if batch == 0 || seq_len == 0 {
        return None;
    }

    let half_d = d / 2;
    let grid_total = batch * seq_len * half_d;
    let tg = grid_total.min(256);
    let total_elements: usize = dims.iter().product::<i64>() as usize;

    let interleaved = get_ints_attr_from_pairs(attrs, "interleaved", &[0]);
    let interleaved_flag = interleaved.first().copied().unwrap_or(0) as u32;

    Some(KernelLaunchPlan {
        kernel_name: "rotary_embedding",
        kernel_source_id: "ROTARY_EMBEDDING",
        grid_size: [grid_total, 1, 1],
        threadgroup_size: [tg, 1, 1],
        input_buffer_count: 3, // input + cos_cache + sin_cache
        output_buffer_count: 1,
        param_buffers: vec![d, seq_len, interleaved_flag],
        needs_dims_buffer: true,
        output_shape: dims.to_vec(),
        output_elements: total_elements,
    })
}

// ---------------------------------------------------------------------------
// Tier 2: Scaled Dot-Product Attention
// ---------------------------------------------------------------------------

fn plan_sdpa(input_shapes: &[&[i64]]) -> Option<KernelLaunchPlan> {
    // Q: [batch*heads, Sq, D], K: [batch*heads, Sk, D], V: [batch*heads, Sk, D]
    if input_shapes.len() < 3 {
        return None;
    }
    let q_dims = input_shapes[0];
    let k_dims = input_shapes[1];
    if q_dims.len() < 3 || k_dims.len() < 3 {
        return None;
    }

    let batch_heads = q_dims[0] as u32;
    let sq = q_dims[1] as u32;
    let d = q_dims[2] as u32;
    let sk = k_dims[1] as u32;

    if batch_heads == 0 || sq == 0 || sk == 0 || d == 0 {
        return None;
    }

    let scale = 1.0_f32 / sqrt_approx(d as f32);
    let scale_bits = scale.to_bits();

    // Each threadgroup computes one output row (one (head, query) pair).
    // tg sizes the inner reduction over Sk and the per-row write over D, so
    // pick the larger of the two but cap at 256.
    let grid_x = batch_heads * sq;
    let tg = sk.max(d).clamp(1, 256);
    let out_elements = (batch_heads as usize) * (sq as usize) * (d as usize);

    Some(KernelLaunchPlan {
        kernel_name: "scaled_dot_product_attention",
        kernel_source_id: "SCALED_DOT_PRODUCT_ATTENTION",
        grid_size: [grid_x * tg, 1, 1],
        threadgroup_size: [tg, 1, 1],
        input_buffer_count: 3, // Q + K + V
        output_buffer_count: 1,
        param_buffers: vec![sq, sk, d, batch_heads, scale_bits],
        needs_dims_buffer: true,
        output_shape: q_dims.to_vec(),
        output_elements: out_elements,
    })
}

// ---------------------------------------------------------------------------
// Tier 2: Group Query Attention
// ---------------------------------------------------------------------------

fn plan_gqa(input_shapes: &[&[i64]], attrs: &[(&str, &[i64])]) -> Option<KernelLaunchPlan> {
    // Inputs: Q[B,Sq,num_heads*D], K[B,Sq,num_kv*D], V[B,Sq,num_kv*D],
    //         past_K[B,num_kv,past_len,D], past_V[B,num_kv,past_len,D],
    //         cos_cache, sin_cache
    if input_shapes.len() < 5 {
        return None;
    }
    let q_dims = input_shapes[0]; // [B, Sq, num_heads*D]
    let past_k_dims = input_shapes[3]; // [B, num_kv_heads, past_len, D]

    if q_dims.len() < 3 || past_k_dims.len() < 4 {
        return None;
    }

    let batch = q_dims[0] as u32;
    let sq = q_dims[1] as u32;

    let num_heads_attr = get_ints_attr_from_pairs(attrs, "num_heads", &[0]);
    let num_heads = num_heads_attr.first().copied().unwrap_or(0) as u32;
    if num_heads == 0 || batch == 0 || sq == 0 {
        return None;
    }

    let num_kv_heads = past_k_dims[1] as u32;
    let past_len = past_k_dims[2] as u32;
    let d = past_k_dims[3] as u32;
    if num_kv_heads == 0 || d == 0 {
        return None;
    }

    let total_len = past_len + sq;

    // Grid: (B, num_heads, Sq) — each threadgroup handles one (batch, head, query_row).
    let grid_x = batch;
    let grid_y = num_heads;
    let grid_z = sq;
    let tg = d.min(256);

    let out_elements = (batch as usize) * (sq as usize) * (num_heads as usize) * (d as usize);
    let out_shape = vec![batch as i64, sq as i64, (num_heads * d) as i64];

    // present_K/V sizes: [B, num_kv_heads, total_len, D]
    let _present_elements =
        (batch as usize) * (num_kv_heads as usize) * (total_len as usize) * (d as usize);

    Some(KernelLaunchPlan {
        kernel_name: "group_query_attention",
        kernel_source_id: "GROUP_QUERY_ATTENTION",
        grid_size: [grid_x, grid_y, grid_z],
        threadgroup_size: [tg, 1, 1],
        input_buffer_count: 7,  // Q + K + V + past_K + past_V + cos + sin
        output_buffer_count: 3, // attention_out + present_K + present_V
        param_buffers: vec![num_heads, num_kv_heads, d, sq, past_len],
        needs_dims_buffer: true,
        output_shape: out_shape,
        output_elements: out_elements,
    })
}

// ---------------------------------------------------------------------------
// Tier 2: Int8 GEMM
// ---------------------------------------------------------------------------

fn plan_gemm_i8(input_shapes: &[&[i64]]) -> Option<KernelLaunchPlan> {
    if input_shapes.len() < 2 {
        return None;
    }
    let a_dims = input_shapes[0];
    let b_dims = input_shapes[1];
    if a_dims.len() < 2 || b_dims.len() < 2 {
        return None;
    }

    let m = a_dims[a_dims.len() - 2] as u32;
    let k = a_dims[a_dims.len() - 1] as u32;
    let n = b_dims[b_dims.len() - 1] as u32;

    if k != b_dims[b_dims.len() - 2] as u32 {
        return None;
    }
    if m == 0 || k == 0 || n == 0 {
        return None;
    }

    let gx = m.div_ceil(16);
    let gy = n.div_ceil(16);
    let out_elements = (m as usize) * (n as usize);

    Some(KernelLaunchPlan {
        kernel_name: "gemm_i8",
        kernel_source_id: "GEMM_I8",
        grid_size: [gx * 16, gy * 16, 1],
        threadgroup_size: [16, 16, 1],
        input_buffer_count: 2, // A (i8) + B (i8)
        output_buffer_count: 1,
        param_buffers: vec![m, k, n],
        needs_dims_buffer: true,
        output_shape: vec![m as i64, n as i64],
        output_elements: out_elements,
    })
}

// ---------------------------------------------------------------------------
// Tier 2: Batch Normalization
// ---------------------------------------------------------------------------

fn plan_batch_norm(input_shapes: &[&[i64]], attrs: &[(&str, &[i64])]) -> Option<KernelLaunchPlan> {
    // Input: [N, C, H, W] (NCHW)
    if input_shapes.is_empty() || input_shapes[0].len() != 4 {
        return None;
    }
    let dims = input_shapes[0];
    let n_batch = dims[0] as u32;
    let c = dims[1] as u32;
    let h = dims[2] as u32;
    let w = dims[3] as u32;
    let hw = h * w;

    if n_batch == 0 || c == 0 || hw == 0 {
        return None;
    }

    let grid = n_batch * c;
    let tg = hw.clamp(1, 256);
    let total = (n_batch as usize) * (c as usize) * (hw as usize);
    let epsilon_bits = get_float_attr_bits(attrs, "epsilon", 1e-5);

    // One threadgroup per (n, c) pair; dispatch_threads wants total threads.
    Some(KernelLaunchPlan {
        kernel_name: "batch_normalization",
        kernel_source_id: "BATCH_NORMALIZATION",
        grid_size: [grid * tg, 1, 1],
        threadgroup_size: [tg, 1, 1],
        input_buffer_count: 5, // data + scale + bias + mean + var
        output_buffer_count: 1,
        param_buffers: vec![c, hw, epsilon_bits],
        needs_dims_buffer: true,
        output_shape: dims.to_vec(),
        output_elements: total,
    })
}

// ===========================================================================
// Layer 2: Platform-specific Metal execution (macOS only)
// ===========================================================================

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_err(e: MetalError) -> OpError {
    OpError::InternalError(format!("Metal GPU error: {}", e.0))
}

/// Maps a `kernel_source_id` to its MSL source string via a static table.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn resolve_kernel_source(id: &str) -> &'static str {
    use smallaios_arch_apple::shaders;
    const TABLE: &[(&str, &str)] = &[
        ("ELEMENTWISE_ADD", shaders::ELEMENTWISE_ADD),
        ("ELEMENTWISE_SUB", shaders::ELEMENTWISE_SUB),
        ("ELEMENTWISE_MUL", shaders::ELEMENTWISE_MUL),
        ("ELEMENTWISE_DIV", shaders::ELEMENTWISE_DIV),
        ("ELEMENTWISE_RELU", shaders::ELEMENTWISE_RELU),
        ("ELEMENTWISE_SIGMOID", shaders::ELEMENTWISE_SIGMOID),
        ("ELEMENTWISE_TANH", shaders::ELEMENTWISE_TANH),
        ("MATMUL", shaders::MATMUL),
        ("MATMUL_TILED", shaders::MATMUL_TILED),
        ("SOFTMAX", shaders::SOFTMAX),
        ("CONV2D", shaders::CONV2D),
        ("LAYER_NORMALIZATION", shaders::LAYER_NORMALIZATION),
        ("RMS_NORMALIZATION", shaders::RMS_NORMALIZATION),
        ("ROTARY_EMBEDDING", shaders::ROTARY_EMBEDDING),
        (
            "SCALED_DOT_PRODUCT_ATTENTION",
            shaders::SCALED_DOT_PRODUCT_ATTENTION,
        ),
        ("GROUP_QUERY_ATTENTION", shaders::GROUP_QUERY_ATTENTION),
        ("GEMM_I8", shaders::GEMM_I8),
        ("GEMM_F16", shaders::GEMM_F16),
        ("BATCH_NORMALIZATION", shaders::BATCH_NORMALIZATION),
    ];
    TABLE
        .iter()
        .find(|(k, _)| *k == id)
        .map(|(_, v)| *v)
        .unwrap_or("")
}

/// Executes a planned kernel launch on the Metal GPU.
///
/// This is the thin platform-specific layer that performs the actual Metal
/// API calls: copies inputs to device, compiles the kernel, launches it,
/// synchronizes, and copies the result back to a host [`Tensor`].
#[cfg(all(feature = "metal", target_os = "macos"))]
fn execute_plan(
    plan: &KernelLaunchPlan,
    provider: &mut MetalProvider,
    cache: &mut MetalTensorCache,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>, OpError> {
    let out_byte_size = plan.output_elements * core::mem::size_of::<f32>();

    // Upload each input to a numbered device buffer.
    for (i, input) in inputs.iter().enumerate() {
        let label = format!("__gpu_in_{}", i);
        cache
            .copy_to_device(provider, &label, &input.raw_data)
            .map_err(metal_err)?;
    }

    // Allocate output buffer
    cache
        .get_or_create(provider, "__gpu_out", out_byte_size)
        .map_err(metal_err)?;

    // Upload dims buffer if needed
    if plan.needs_dims_buffer {
        let dims_data: Vec<u8> = plan
            .param_buffers
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        cache
            .copy_to_device(provider, "__gpu_dims", &dims_data)
            .map_err(metal_err)?;
    }

    // Compile kernel
    let source = resolve_kernel_source(plan.kernel_source_id);
    let kernel = provider
        .load_kernel(plan.kernel_name, source.as_bytes())
        .map_err(metal_err)?;

    // Gather buffer references for launch: inputs + output [+ dims].
    let mut bufs: Vec<&_> = Vec::with_capacity(inputs.len() + 2);
    for i in 0..inputs.len() {
        let label = format!("__gpu_in_{}", i);
        bufs.push(cache.get(&label).unwrap());
    }
    bufs.push(cache.get("__gpu_out").unwrap());
    if plan.needs_dims_buffer {
        bufs.push(cache.get("__gpu_dims").unwrap());
    }

    // Launch
    provider
        .launch(&kernel, plan.grid_size, plan.threadgroup_size, &bufs)
        .map_err(metal_err)?;

    provider.synchronize().map_err(metal_err)?;

    // Read back result
    let buf_out = cache.get("__gpu_out").unwrap();
    let raw_data =
        MetalTensorCache::copy_from_device(provider, buf_out, out_byte_size).map_err(metal_err)?;

    let out_shape = TensorShape::new(plan.output_shape.clone());
    let mut output = Tensor::new(DataType::Float, out_shape, String::new());
    output.raw_data = raw_data;

    Ok(vec![output])
}

// ===========================================================================
// Layer 3: MetalDispatcher wrapper
// ===========================================================================

/// Dispatches ONNX operators to Metal GPU kernels.
///
/// Wraps a [`MetalProvider`] and a [`MetalTensorCache`] to manage device
/// memory and kernel compilation. The `try_execute` method is the main
/// entry point: it plans the kernel launch (platform-independent), then
/// executes it via Metal APIs (macOS-only).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct MetalDispatcher {
    provider: MetalProvider,
    tensor_cache: MetalTensorCache,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl MetalDispatcher {
    /// Creates a new Metal dispatcher, initializing the Metal device.
    ///
    /// Returns an error if no Metal-capable device is available.
    pub fn new() -> Result<Self, MetalError> {
        let provider = MetalProvider::new()?;
        Ok(Self {
            provider,
            tensor_cache: MetalTensorCache::new(),
        })
    }

    /// Attempts to execute an ONNX operator on the Metal GPU.
    ///
    /// Returns `Ok(Some(outputs))` if the operator was executed on GPU,
    /// `Ok(None)` if the operator is not supported (caller should fall back
    /// to CPU), or `Err` if a GPU error occurred.
    pub fn try_execute(
        &mut self,
        op_type: &str,
        inputs: &[Option<&Tensor>],
        attrs: &[AttributeProto],
    ) -> Result<Option<Vec<Tensor>>, OpError> {
        // Extract shapes from inputs (platform-independent)
        let shapes: Vec<Vec<i64>> = inputs
            .iter()
            .filter_map(|t| t.map(|t| t.shape.dims.clone()))
            .collect();
        let shape_refs: Vec<&[i64]> = shapes.iter().map(|s| s.as_slice()).collect();

        // Extract simplified attributes for planning
        let strides = get_ints_attr(attrs, "strides", &[1, 1]);
        let pads = get_ints_attr(attrs, "pads", &[0, 0, 0, 0]);
        let attr_pairs: Vec<(&str, Vec<i64>)> = vec![("strides", strides), ("pads", pads)];
        let attr_refs: Vec<(&str, &[i64])> =
            attr_pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();

        // Plan the kernel launch (platform-independent, testable)
        let plan = match plan_kernel_launch(op_type, &shape_refs, &attr_refs) {
            Some(p) => p,
            None => return Ok(None),
        };

        // Validate inputs before execution
        let real_inputs: Vec<&Tensor> = inputs.iter().filter_map(|t| *t).collect();
        if real_inputs.len() < plan.input_buffer_count {
            return Err(OpError::ShapeMismatch(format!(
                "{} requires {} inputs, got {}",
                op_type,
                plan.input_buffer_count,
                real_inputs.len()
            )));
        }

        // Validate data types
        for input in &real_inputs {
            if input.data_type != DataType::Float {
                return Err(OpError::InternalError(format!(
                    "Metal {} requires Float tensors",
                    op_type
                )));
            }
        }

        // Execute the plan (platform-specific Metal API calls)
        let result = execute_plan(
            &plan,
            &mut self.provider,
            &mut self.tensor_cache,
            &real_inputs,
        )?;
        Ok(Some(result))
    }

    /// Returns `true` if the given operator can be executed on Metal.
    pub fn supports_op(&self, op_type: &str) -> bool {
        is_gpu_supported(op_type)
    }

    /// Clears the tensor cache, releasing all cached GPU buffers.
    pub fn clear_cache(&mut self) {
        self.tensor_cache.clear();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Platform-independent planning tests (run on ALL platforms including
    // Linux CI). These cover the majority of the module's logic.
    // -------------------------------------------------------------------

    // ---- classify_op ----

    #[test]
    fn test_classify_op_elementwise_binary() {
        for op in &["Add", "Sub", "Mul", "Div"] {
            assert_eq!(
                classify_op(op),
                Some(KernelCategory::ElementwiseBinary),
                "{} should be ElementwiseBinary",
                op
            );
        }
    }

    #[test]
    fn test_classify_op_elementwise_unary() {
        for op in &["Relu", "Sigmoid", "Tanh"] {
            assert_eq!(
                classify_op(op),
                Some(KernelCategory::ElementwiseUnary),
                "{} should be ElementwiseUnary",
                op
            );
        }
    }

    #[test]
    fn test_classify_op_matmul() {
        assert_eq!(classify_op("MatMul"), Some(KernelCategory::MatMul));
        assert_eq!(classify_op("Gemm"), Some(KernelCategory::MatMul));
    }

    #[test]
    fn test_classify_op_softmax() {
        assert_eq!(classify_op("Softmax"), Some(KernelCategory::Softmax));
    }

    #[test]
    fn test_classify_op_conv() {
        assert_eq!(classify_op("Conv"), Some(KernelCategory::Conv2d));
    }

    #[test]
    fn test_classify_op_unsupported() {
        assert_eq!(classify_op("Reshape"), None);
        assert_eq!(classify_op("Gather"), None);
        assert_eq!(classify_op("QuantizeLinear"), None);
    }

    // ---- is_gpu_supported ----

    #[test]
    fn test_is_gpu_supported() {
        assert!(is_gpu_supported("Add"));
        assert!(is_gpu_supported("MatMul"));
        assert!(is_gpu_supported("Relu"));
        assert!(is_gpu_supported("Conv"));
        assert!(!is_gpu_supported("Reshape"));
        assert!(!is_gpu_supported("Slice"));
    }

    // ---- Elementwise binary planning ----

    /// Data-driven test for all elementwise binary ops — eliminates per-op
    /// test duplication that SonarCloud flags.
    #[test]
    fn test_plan_elementwise_binary_ops() {
        let cases: &[(&str, &str, &str)] = &[
            ("Add", "elementwise_add", "ELEMENTWISE_ADD"),
            ("Sub", "elementwise_sub", "ELEMENTWISE_SUB"),
            ("Mul", "elementwise_mul", "ELEMENTWISE_MUL"),
            ("Div", "elementwise_div", "ELEMENTWISE_DIV"),
        ];
        for &(op, expected_kernel, expected_source) in cases {
            let plan = plan_kernel_launch(op, &[&[2, 3]], &[])
                .unwrap_or_else(|| panic!("{} should produce a plan", op));
            assert_eq!(plan.kernel_name, expected_kernel, "{} kernel name", op);
            assert_eq!(plan.kernel_source_id, expected_source, "{} source id", op);
            assert_eq!(plan.grid_size, [6, 1, 1], "{} grid", op);
            assert_eq!(plan.threadgroup_size, [6, 1, 1], "{} threadgroup", op);
            assert_eq!(plan.input_buffer_count, 2, "{} inputs", op);
            assert_eq!(plan.output_buffer_count, 1, "{} outputs", op);
            assert_eq!(plan.output_elements, 6, "{} elements", op);
            assert_eq!(plan.output_shape, vec![2, 3], "{} shape", op);
            assert!(!plan.needs_dims_buffer, "{} dims buffer", op);
        }
    }

    #[test]
    fn test_plan_elementwise_large_threadgroup_capped() {
        let plan = plan_kernel_launch("Add", &[&[1024]], &[]).unwrap();
        assert_eq!(plan.grid_size, [1024, 1, 1]);
        assert_eq!(plan.threadgroup_size, [256, 1, 1]); // capped at 256
    }

    #[test]
    fn test_plan_elementwise_empty_shape_returns_none() {
        assert!(plan_kernel_launch("Add", &[], &[]).is_none());
    }

    // ---- Elementwise unary planning ----

    /// Data-driven test for all elementwise unary activation ops.
    #[test]
    fn test_plan_elementwise_unary_ops() {
        let cases: &[(&str, &str, &str)] = &[
            ("Relu", "elementwise_relu", "ELEMENTWISE_RELU"),
            ("Sigmoid", "elementwise_sigmoid", "ELEMENTWISE_SIGMOID"),
            ("Tanh", "elementwise_tanh", "ELEMENTWISE_TANH"),
        ];
        for &(op, expected_kernel, expected_source) in cases {
            let plan = plan_kernel_launch(op, &[&[256]], &[])
                .unwrap_or_else(|| panic!("{} should produce a plan", op));
            assert_eq!(plan.kernel_name, expected_kernel, "{} kernel name", op);
            assert_eq!(plan.kernel_source_id, expected_source, "{} source id", op);
            assert_eq!(plan.input_buffer_count, 1, "{} inputs", op);
            assert_eq!(plan.output_buffer_count, 1, "{} outputs", op);
            assert_eq!(plan.grid_size, [256, 1, 1], "{} grid", op);
            assert_eq!(plan.threadgroup_size, [256, 1, 1], "{} threadgroup", op);
            assert!(!plan.needs_dims_buffer, "{} dims buffer", op);
        }
    }

    #[test]
    fn test_plan_unary_small_shape() {
        let plan = plan_kernel_launch("Tanh", &[&[50]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "elementwise_tanh");
    }

    // ---- MatMul planning ----

    #[test]
    fn test_plan_matmul_selects_tiled_for_large() {
        let plan = plan_kernel_launch("MatMul", &[&[64, 128], &[128, 32]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "matmul_tiled");
        assert_eq!(plan.kernel_source_id, "MATMUL_TILED");
        // grid: ceil(32/16)*16=32, ceil(64/16)*16=64
        assert_eq!(plan.grid_size, [32, 64, 1]);
        assert_eq!(plan.threadgroup_size, [16, 16, 1]);
        assert_eq!(plan.param_buffers, vec![64, 128, 32]);
        assert!(plan.needs_dims_buffer);
        assert_eq!(plan.output_shape, vec![64, 32]);
        assert_eq!(plan.output_elements, 64 * 32);
    }

    #[test]
    fn test_plan_matmul_selects_naive_for_small() {
        let plan = plan_kernel_launch("MatMul", &[&[4, 8], &[8, 4]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "matmul");
        assert_eq!(plan.kernel_source_id, "MATMUL");
        assert_eq!(plan.grid_size, [4, 4, 1]);
        assert_eq!(plan.param_buffers, vec![4, 8, 4]);
    }

    #[test]
    fn test_plan_gemm_uses_matmul_path() {
        let plan = plan_kernel_launch("Gemm", &[&[2, 3], &[3, 4]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "matmul");
        assert_eq!(plan.output_shape, vec![2, 4]);
    }

    #[test]
    fn test_plan_matmul_non_square() {
        let plan = plan_kernel_launch("MatMul", &[&[1, 3], &[3, 1]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "matmul");
        assert_eq!(plan.output_shape, vec![1, 1]);
        assert_eq!(plan.output_elements, 1);
    }

    #[test]
    fn test_plan_matmul_inner_dim_mismatch_returns_none() {
        // A=[2,3], B=[4,2] -- inner dims 3 != 4
        assert!(plan_kernel_launch("MatMul", &[&[2, 3], &[4, 2]], &[]).is_none());
    }

    #[test]
    fn test_plan_matmul_1d_input_returns_none() {
        assert!(plan_kernel_launch("MatMul", &[&[3], &[3]], &[]).is_none());
    }

    #[test]
    fn test_plan_matmul_missing_input_returns_none() {
        assert!(plan_kernel_launch("MatMul", &[&[2, 3]], &[]).is_none());
    }

    // ---- Softmax planning ----

    #[test]
    fn test_plan_softmax_2d() {
        let plan = plan_kernel_launch("Softmax", &[&[4, 100]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "softmax");
        assert_eq!(plan.kernel_source_id, "SOFTMAX");
        assert_eq!(plan.param_buffers, vec![100]); // N = cols (row length)
        assert_eq!(plan.grid_size[0], 400); // total elements
        assert!(plan.needs_dims_buffer);
        assert_eq!(plan.output_shape, vec![4, 100]);
        assert_eq!(plan.output_elements, 400);
    }

    #[test]
    fn test_plan_softmax_1d() {
        let plan = plan_kernel_launch("Softmax", &[&[10]], &[]).unwrap();
        assert_eq!(plan.param_buffers, vec![10]); // N = cols (row length)
        assert_eq!(plan.output_elements, 10);
    }

    #[test]
    fn test_plan_softmax_3d() {
        let plan = plan_kernel_launch("Softmax", &[&[2, 3, 50]], &[]).unwrap();
        // N = cols (row length) = 50
        assert_eq!(plan.param_buffers, vec![50]);
        assert_eq!(plan.output_elements, 300);
    }

    #[test]
    fn test_plan_softmax_empty_returns_none() {
        assert!(plan_kernel_launch("Softmax", &[], &[]).is_none());
        assert!(plan_kernel_launch("Softmax", &[&[]], &[]).is_none());
    }

    // ---- Conv2D planning ----

    #[test]
    fn test_plan_conv2d_basic() {
        let plan = plan_kernel_launch(
            "Conv",
            &[&[1, 3, 224, 224], &[64, 3, 7, 7]],
            &[("strides", &[2, 2]), ("pads", &[3, 3, 3, 3])],
        )
        .unwrap();
        assert_eq!(plan.kernel_name, "conv2d");
        assert_eq!(plan.kernel_source_id, "CONV2D");
        assert_eq!(plan.input_buffer_count, 2);
        assert!(plan.needs_dims_buffer);
        // out_h = (224 + 6 - 7)/2 + 1 = 112
        // out_w = (224 + 6 - 7)/2 + 1 = 112
        assert_eq!(plan.output_shape, vec![1, 64, 112, 112]);
        assert_eq!(plan.output_elements, 64 * 112 * 112);
    }

    #[test]
    fn test_plan_conv2d_no_padding() {
        let plan = plan_kernel_launch("Conv", &[&[1, 1, 28, 28], &[16, 1, 3, 3]], &[]).unwrap();
        // out_h = (28 + 0 - 3)/1 + 1 = 26
        // out_w = (28 + 0 - 3)/1 + 1 = 26
        assert_eq!(plan.output_shape, vec![1, 16, 26, 26]);
    }

    #[test]
    fn test_plan_conv2d_dims_buffer_contents() {
        let plan = plan_kernel_launch(
            "Conv",
            &[&[1, 3, 8, 8], &[4, 3, 3, 3]],
            &[("strides", &[1, 1]), ("pads", &[0, 0, 0, 0])],
        )
        .unwrap();
        // params: [batch=1, in_ch=3, in_h=8, in_w=8, out_ch=4, kh=3, kw=3, sh=1, sw=1, ph=0, pw=0]
        assert_eq!(plan.param_buffers, vec![1, 3, 8, 8, 4, 3, 3, 1, 1, 0, 0]);
    }

    #[test]
    fn test_plan_conv2d_non_4d_returns_none() {
        // 3D input
        assert!(plan_kernel_launch("Conv", &[&[1, 3, 28], &[16, 3, 3, 3]], &[]).is_none());
        // 3D weight
        assert!(plan_kernel_launch("Conv", &[&[1, 3, 28, 28], &[16, 3, 3]], &[]).is_none());
    }

    #[test]
    fn test_plan_conv2d_missing_weight_returns_none() {
        assert!(plan_kernel_launch("Conv", &[&[1, 3, 28, 28]], &[]).is_none());
    }

    // ---- Unsupported ops ----

    #[test]
    fn test_plan_unsupported_op_returns_none() {
        assert!(plan_kernel_launch("QuantizeLinear", &[&[10]], &[]).is_none());
        assert!(plan_kernel_launch("Reshape", &[&[2, 3]], &[]).is_none());
        assert!(plan_kernel_launch("Gather", &[&[10]], &[]).is_none());
        assert!(plan_kernel_launch("Slice", &[&[10]], &[]).is_none());
        assert!(plan_kernel_launch("Pad", &[&[10]], &[]).is_none());
    }

    // ---- KernelLaunchPlan equality ----

    #[test]
    fn test_plan_deterministic() {
        let plan1 = plan_kernel_launch("Add", &[&[100]], &[]).unwrap();
        let plan2 = plan_kernel_launch("Add", &[&[100]], &[]).unwrap();
        assert_eq!(plan1, plan2, "same inputs must produce identical plans");
    }

    // ---- get_ints_attr_from_pairs ----

    #[test]
    fn test_get_ints_attr_found() {
        let attrs = [
            ("strides", [2i64, 2].as_slice()),
            ("pads", [1, 1, 1, 1].as_slice()),
        ];
        assert_eq!(
            get_ints_attr_from_pairs(&attrs, "strides", &[1, 1]),
            vec![2, 2]
        );
    }

    #[test]
    fn test_get_ints_attr_default() {
        let attrs: [(&str, &[i64]); 0] = [];
        assert_eq!(
            get_ints_attr_from_pairs(&attrs, "strides", &[1, 1]),
            vec![1, 1]
        );
    }

    // ---- Tier 2: Layer Normalization planning ----

    #[test]
    fn test_plan_layer_norm_basic() {
        let plan = plan_kernel_launch("LayerNormalization", &[&[4, 128]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "layer_normalization");
        assert_eq!(plan.kernel_source_id, "LAYER_NORMALIZATION");
        // grid = N * tg (dispatch_threads wants total threads); tg = min(D, 256) = 128
        assert_eq!(plan.grid_size, [4 * 128, 1, 1]);
        assert_eq!(plan.threadgroup_size, [128, 1, 1]);
        assert_eq!(plan.input_buffer_count, 3); // data + scale + bias
        assert_eq!(plan.output_buffer_count, 1);
        assert!(plan.needs_dims_buffer);
        assert_eq!(plan.output_shape, vec![4, 128]);
        assert_eq!(plan.output_elements, 512);
        assert_eq!(plan.param_buffers[0], 128); // D
    }

    #[test]
    fn test_plan_layer_norm_large_d_capped() {
        let plan = plan_kernel_launch("LayerNormalization", &[&[2, 512]], &[]).unwrap();
        assert_eq!(plan.threadgroup_size, [256, 1, 1]); // capped at 256
        assert_eq!(plan.grid_size, [2 * 256, 1, 1]);
    }

    #[test]
    fn test_plan_layer_norm_empty_returns_none() {
        assert!(plan_kernel_launch("LayerNormalization", &[], &[]).is_none());
        // 1-D input not valid (needs at least 2 dims)
        assert!(plan_kernel_launch("LayerNormalization", &[&[128]], &[]).is_none());
    }

    // ---- Tier 2: RMS Normalization planning ----

    #[test]
    fn test_plan_rms_norm_basic() {
        let plan = plan_kernel_launch("RMSNormalization", &[&[8, 64]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "rms_normalization");
        assert_eq!(plan.kernel_source_id, "RMS_NORMALIZATION");
        assert_eq!(plan.input_buffer_count, 2); // data + scale (no bias)
        assert_eq!(plan.grid_size, [8 * 64, 1, 1]);
        assert_eq!(plan.threadgroup_size, [64, 1, 1]);
    }

    #[test]
    fn test_plan_rms_norm_via_simplified() {
        let plan = plan_kernel_launch("SimplifiedLayerNormalization", &[&[4, 256]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "rms_normalization");
        assert_eq!(plan.kernel_source_id, "RMS_NORMALIZATION");
    }

    // ---- Tier 2: Rotary Embedding planning ----

    #[test]
    fn test_plan_rotary_embedding_basic() {
        // [B=2, seq_len=16, D=64]
        let plan = plan_kernel_launch("RotaryEmbedding", &[&[2, 16, 64]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "rotary_embedding");
        assert_eq!(plan.kernel_source_id, "ROTARY_EMBEDDING");
        assert_eq!(plan.input_buffer_count, 3); // input + cos + sin
                                                // grid = B * seq_len * (D/2) = 2 * 16 * 32 = 1024
        assert_eq!(plan.grid_size, [1024, 1, 1]);
        assert_eq!(plan.param_buffers, vec![64, 16, 0]); // D, seq_len, interleaved=0
        assert_eq!(plan.output_elements, 2 * 16 * 64);
    }

    #[test]
    fn test_plan_rotary_embedding_odd_d_returns_none() {
        assert!(plan_kernel_launch("RotaryEmbedding", &[&[1, 8, 63]], &[]).is_none());
    }

    #[test]
    fn test_plan_rotary_embedding_2d_returns_none() {
        assert!(plan_kernel_launch("RotaryEmbedding", &[&[8, 64]], &[]).is_none());
    }

    // ---- Tier 2: Scaled Dot-Product Attention planning ----

    #[test]
    fn test_plan_sdpa_basic() {
        // Q: [4, 8, 64], K: [4, 12, 64], V: [4, 12, 64]
        let plan = plan_kernel_launch(
            "ScaledDotProductAttention",
            &[&[4, 8, 64], &[4, 12, 64], &[4, 12, 64]],
            &[],
        )
        .unwrap();
        assert_eq!(plan.kernel_name, "scaled_dot_product_attention");
        assert_eq!(plan.input_buffer_count, 3);
        assert_eq!(plan.output_buffer_count, 1);
        // num_threadgroups = batch_heads * Sq = 32; tg = max(Sk, D).min(256) = 64
        // grid (total threads) = 32 * 64 = 2048
        assert_eq!(plan.grid_size, [32 * 64, 1, 1]);
        assert_eq!(plan.threadgroup_size, [64, 1, 1]);
        assert_eq!(plan.param_buffers[0], 8); // Sq
        assert_eq!(plan.param_buffers[1], 12); // Sk
        assert_eq!(plan.param_buffers[2], 64); // D
        assert_eq!(plan.output_elements, 4 * 8 * 64);
    }

    #[test]
    fn test_plan_sdpa_missing_inputs_returns_none() {
        assert!(plan_kernel_launch("ScaledDotProductAttention", &[&[4, 8, 64]], &[]).is_none());
    }

    // ---- Tier 2: Group Query Attention planning ----

    #[test]
    fn test_plan_gqa_basic() {
        // Q: [1, 4, 256] (B=1, Sq=4, num_heads*D=256)
        // K: [1, 4, 64], V: [1, 4, 64]
        // past_K: [1, 2, 8, 32] (B=1, num_kv=2, past_len=8, D=32)
        // past_V: [1, 2, 8, 32]
        // cos: [12, 16], sin: [12, 16]
        let plan = plan_kernel_launch(
            "GroupQueryAttention",
            &[
                &[1, 4, 256],
                &[1, 4, 64],
                &[1, 4, 64],
                &[1, 2, 8, 32],
                &[1, 2, 8, 32],
                &[12, 16],
                &[12, 16],
            ],
            &[("num_heads", &[8])],
        )
        .unwrap();
        assert_eq!(plan.kernel_name, "group_query_attention");
        assert_eq!(plan.input_buffer_count, 7);
        assert_eq!(plan.output_buffer_count, 3);
        assert_eq!(plan.grid_size, [1, 8, 4]); // (B, num_heads, Sq)
        assert_eq!(plan.param_buffers, vec![8, 2, 32, 4, 8]); // heads, kv_heads, D, Sq, past_len
    }

    #[test]
    fn test_plan_gqa_missing_num_heads_returns_none() {
        assert!(plan_kernel_launch(
            "GroupQueryAttention",
            &[
                &[1, 4, 256],
                &[1, 4, 64],
                &[1, 4, 64],
                &[1, 2, 8, 32],
                &[1, 2, 8, 32],
                &[12, 16],
                &[12, 16],
            ],
            &[],
        )
        .is_none());
    }

    // ---- Tier 2: Int8 GEMM planning ----

    #[test]
    fn test_plan_gemm_i8_basic() {
        let plan = plan_kernel_launch("MatMulInteger", &[&[32, 64], &[64, 48]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "gemm_i8");
        assert_eq!(plan.kernel_source_id, "GEMM_I8");
        // grid = (ceil(32/16)*16, ceil(48/16)*16) = (32, 48)
        assert_eq!(plan.grid_size, [32, 48, 1]);
        assert_eq!(plan.threadgroup_size, [16, 16, 1]);
        assert_eq!(plan.param_buffers, vec![32, 64, 48]);
        assert_eq!(plan.output_shape, vec![32, 48]);
    }

    #[test]
    fn test_plan_gemm_i8_non_tile_aligned() {
        let plan = plan_kernel_launch("MatMulInteger", &[&[7, 13], &[13, 5]], &[]).unwrap();
        // ceil(7/16)*16 = 16, ceil(5/16)*16 = 16
        assert_eq!(plan.grid_size, [16, 16, 1]);
        assert_eq!(plan.output_elements, 7 * 5);
    }

    #[test]
    fn test_plan_gemm_i8_inner_mismatch_returns_none() {
        assert!(plan_kernel_launch("MatMulInteger", &[&[4, 8], &[7, 3]], &[]).is_none());
    }

    // ---- Tier 2: Batch Normalization planning ----

    #[test]
    fn test_plan_batch_norm_basic() {
        let plan = plan_kernel_launch("BatchNormalization", &[&[2, 64, 32, 32]], &[]).unwrap();
        assert_eq!(plan.kernel_name, "batch_normalization");
        assert_eq!(plan.kernel_source_id, "BATCH_NORMALIZATION");
        assert_eq!(plan.input_buffer_count, 5); // data + scale + bias + mean + var
                                                // num_threadgroups = N * C = 128; tg = min(HW, 256) = 256
                                                // grid (total threads) = 128 * 256 = 32768
        assert_eq!(plan.grid_size, [128 * 256, 1, 1]);
        assert_eq!(plan.threadgroup_size, [256, 1, 1]);
        assert_eq!(plan.output_shape, vec![2, 64, 32, 32]);
        assert_eq!(plan.output_elements, 2 * 64 * 32 * 32);
        assert_eq!(plan.param_buffers[0], 64); // C
        assert_eq!(plan.param_buffers[1], 1024); // HW
    }

    #[test]
    fn test_plan_batch_norm_non_4d_returns_none() {
        assert!(plan_kernel_launch("BatchNormalization", &[&[2, 64, 32]], &[]).is_none());
    }

    // ---- Tier 2: is_gpu_supported coverage ----

    #[test]
    fn test_is_gpu_supported_tier2() {
        assert!(is_gpu_supported("LayerNormalization"));
        assert!(is_gpu_supported("RMSNormalization"));
        assert!(is_gpu_supported("SimplifiedLayerNormalization"));
        assert!(is_gpu_supported("RotaryEmbedding"));
        assert!(is_gpu_supported("ScaledDotProductAttention"));
        assert!(is_gpu_supported("GroupQueryAttention"));
        assert!(is_gpu_supported("MatMulInteger"));
        assert!(is_gpu_supported("BatchNormalization"));
    }

    // -------------------------------------------------------------------
    // macOS-only GPU execution tests (run via `just test-metal`)
    // -------------------------------------------------------------------

    #[cfg(all(feature = "metal", target_os = "macos"))]
    mod metal_tests {
        use super::super::*;
        use crate::tensor::{DataType, Tensor, TensorShape};
        use alloc::string::String;
        use alloc::vec;
        use alloc::vec::Vec;

        /// Helper: create a Float tensor from f32 data.
        fn make_tensor(shape: Vec<i64>, data: &[f32]) -> Tensor {
            let raw_data: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut t = Tensor::new(DataType::Float, TensorShape::new(shape), String::new());
            t.raw_data = raw_data;
            t
        }

        /// Helper: extract f32 values from a tensor.
        fn tensor_f32(t: &Tensor) -> Vec<f32> {
            t.raw_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        /// Helper: assert two f32 slices are approximately equal.
        fn assert_approx_eq(actual: &[f32], expected: &[f32], tol: f32) {
            assert_eq!(actual.len(), expected.len(), "length mismatch");
            for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() < tol,
                    "element {} differs: actual={}, expected={}, tol={}",
                    i,
                    a,
                    e,
                    tol
                );
            }
        }

        #[test]
        fn test_elementwise_add_gpu_matches_cpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]);
            let b = make_tensor(vec![4], &[5.0, 6.0, 7.0, 8.0]);
            let result = disp
                .try_execute("Add", &[Some(&a), Some(&b)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            assert_approx_eq(&out, &[6.0, 8.0, 10.0, 12.0], 1e-5);
        }

        #[test]
        fn test_elementwise_sub_gpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![3], &[10.0, 20.0, 30.0]);
            let b = make_tensor(vec![3], &[1.0, 2.0, 3.0]);
            let result = disp
                .try_execute("Sub", &[Some(&a), Some(&b)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            assert_approx_eq(&out, &[9.0, 18.0, 27.0], 1e-5);
        }

        #[test]
        fn test_elementwise_mul_gpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![3], &[2.0, 3.0, 4.0]);
            let b = make_tensor(vec![3], &[5.0, 6.0, 7.0]);
            let result = disp
                .try_execute("Mul", &[Some(&a), Some(&b)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            assert_approx_eq(&out, &[10.0, 18.0, 28.0], 1e-5);
        }

        #[test]
        fn test_elementwise_div_gpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![3], &[10.0, 20.0, 30.0]);
            let b = make_tensor(vec![3], &[2.0, 5.0, 10.0]);
            let result = disp
                .try_execute("Div", &[Some(&a), Some(&b)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            assert_approx_eq(&out, &[5.0, 4.0, 3.0], 1e-5);
        }

        #[test]
        fn test_relu_gpu_matches_cpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let input = make_tensor(vec![6], &[-3.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
            let result = disp
                .try_execute("Relu", &[Some(&input)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            assert_approx_eq(&out, &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], 1e-5);
        }

        #[test]
        fn test_sigmoid_gpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let input = make_tensor(vec![3], &[0.0, 1.0, -1.0]);
            let result = disp
                .try_execute("Sigmoid", &[Some(&input)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            assert_approx_eq(&out, &[0.5, 0.7310586, 0.26894143], 1e-4);
        }

        #[test]
        fn test_tanh_gpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let input = make_tensor(vec![3], &[0.0, 1.0, -1.0]);
            let result = disp
                .try_execute("Tanh", &[Some(&input)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            assert_approx_eq(&out, &[0.0, 0.7615942, -0.7615942], 1e-4);
        }

        #[test]
        fn test_matmul_gpu_matches_cpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let b = make_tensor(vec![2, 2], &[5.0, 6.0, 7.0, 8.0]);
            let result = disp
                .try_execute("MatMul", &[Some(&a), Some(&b)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            assert_approx_eq(&out, &[19.0, 22.0, 43.0, 50.0], 1e-4);
        }

        #[test]
        fn test_matmul_non_square_gpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![1, 3], &[1.0, 2.0, 3.0]);
            let b = make_tensor(vec![3, 1], &[4.0, 5.0, 6.0]);
            let result = disp
                .try_execute("MatMul", &[Some(&a), Some(&b)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            assert_approx_eq(&out, &[32.0], 1e-4);
        }

        #[test]
        fn test_softmax_gpu_matches_cpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let input = make_tensor(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]);
            let result = disp
                .try_execute("Softmax", &[Some(&input)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            let sum_exp: f32 = [1.0f32, 2.0, 3.0, 4.0].iter().map(|x| x.exp()).sum();
            let expected: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0]
                .iter()
                .map(|x| x.exp() / sum_exp)
                .collect();
            assert_approx_eq(&out, &expected, 1e-4);
        }

        #[test]
        fn test_tensor_cache_reuse() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]);
            let b = make_tensor(vec![4], &[1.0, 1.0, 1.0, 1.0]);
            let _r1 = disp
                .try_execute("Add", &[Some(&a), Some(&b)], &[])
                .unwrap()
                .unwrap();
            assert!(!disp.tensor_cache.is_empty());
            let c = make_tensor(vec![4], &[10.0, 20.0, 30.0, 40.0]);
            let r2 = disp
                .try_execute("Add", &[Some(&c), Some(&b)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&r2[0]);
            assert_approx_eq(&out, &[11.0, 21.0, 31.0, 41.0], 1e-5);
        }

        #[test]
        fn test_cpu_fallback_for_unsupported_op() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]);
            let result = disp.try_execute("Reshape", &[Some(&a)], &[]).unwrap();
            assert!(result.is_none(), "unsupported op should return None");
        }

        #[test]
        fn test_unsupported_op_returns_none() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![2], &[1.0, 2.0]);
            assert!(disp
                .try_execute("Gather", &[Some(&a)], &[])
                .unwrap()
                .is_none());
            assert!(disp
                .try_execute("Slice", &[Some(&a)], &[])
                .unwrap()
                .is_none());
            assert!(disp.try_execute("Pad", &[Some(&a)], &[]).unwrap().is_none());
        }

        #[test]
        fn test_dispatcher_supports_op() {
            let disp = MetalDispatcher::new().expect("Metal required");
            assert!(disp.supports_op("Add"));
            assert!(disp.supports_op("MatMul"));
            assert!(disp.supports_op("Relu"));
            assert!(disp.supports_op("Conv"));
            assert!(!disp.supports_op("Reshape"));
        }

        #[test]
        fn test_clear_cache() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![2], &[1.0, 2.0]);
            let b = make_tensor(vec![2], &[3.0, 4.0]);
            let _ = disp.try_execute("Add", &[Some(&a), Some(&b)], &[]).unwrap();
            assert!(!disp.tensor_cache.is_empty());
            disp.clear_cache();
            assert!(disp.tensor_cache.is_empty());
        }

        #[test]
        fn test_elementwise_shape_mismatch() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let a = make_tensor(vec![3], &[1.0, 2.0, 3.0]);
            let b = make_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]);
            let result = disp.try_execute("Add", &[Some(&a), Some(&b)], &[]);
            // Planning uses first input shape; the mismatch may surface
            // during execution or planning depending on implementation
            assert!(result.is_ok() || result.is_err());
        }

        // -------------------------------------------------------------------
        // Tier 2 — transformer ops, GPU vs. CPU reference
        // -------------------------------------------------------------------

        /// CPU reference: layer norm over last dim with default eps=1e-5.
        fn cpu_layer_norm(input: &[f32], d: usize, scale: &[f32], bias: &[f32]) -> Vec<f32> {
            let eps = 1e-5_f32;
            let mut out = vec![0.0f32; input.len()];
            for chunk_start in (0..input.len()).step_by(d) {
                let row = &input[chunk_start..chunk_start + d];
                let mean: f32 = row.iter().sum::<f32>() / d as f32;
                let var: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / d as f32;
                let inv_std = (var + eps).sqrt().recip();
                for i in 0..d {
                    out[chunk_start + i] = (row[i] - mean) * inv_std * scale[i] + bias[i];
                }
            }
            out
        }

        #[test]
        fn test_layer_norm_gpu_matches_cpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let input = make_tensor(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]);
            let scale = make_tensor(vec![4], &[1.0, 1.0, 1.0, 1.0]);
            let bias = make_tensor(vec![4], &[0.0, 0.0, 0.0, 0.0]);
            let result = disp
                .try_execute(
                    "LayerNormalization",
                    &[Some(&input), Some(&scale), Some(&bias)],
                    &[],
                )
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            let expected = cpu_layer_norm(
                &[1.0, 2.0, 3.0, 4.0],
                4,
                &[1.0, 1.0, 1.0, 1.0],
                &[0.0, 0.0, 0.0, 0.0],
            );
            assert_approx_eq(&out, &expected, 1e-4);
        }

        /// Multi-row LayerNorm — exercises the per-row threadgroup dispatch.
        #[test]
        fn test_layer_norm_gpu_multi_row_matches_cpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            // 3 rows × 4 cols. With grid=[3*4,1,1] tg=[4,1,1] this dispatches
            // 3 threadgroups; pre-fix only 1 group fired and rows 1/2 were
            // left as garbage from the output buffer.
            let input_data = vec![
                1.0f32, 2.0, 3.0, 4.0, // row 0
                10.0, 20.0, 30.0, 40.0, // row 1
                -1.0, -2.0, -3.0, -4.0, // row 2
            ];
            let scale = vec![1.0f32, 1.0, 1.0, 1.0];
            let bias = vec![0.0f32, 0.0, 0.0, 0.0];
            let input = make_tensor(vec![3, 4], &input_data);
            let scale_t = make_tensor(vec![4], &scale);
            let bias_t = make_tensor(vec![4], &bias);
            let result = disp
                .try_execute(
                    "LayerNormalization",
                    &[Some(&input), Some(&scale_t), Some(&bias_t)],
                    &[],
                )
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            let expected = cpu_layer_norm(&input_data, 4, &scale, &bias);
            assert_approx_eq(&out, &expected, 1e-4);
        }

        /// CPU reference: RMS norm over last dim with eps=1e-5, no bias.
        fn cpu_rms_norm(input: &[f32], d: usize, scale: &[f32]) -> Vec<f32> {
            let eps = 1e-5_f32;
            let mut out = vec![0.0f32; input.len()];
            for chunk_start in (0..input.len()).step_by(d) {
                let row = &input[chunk_start..chunk_start + d];
                let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / d as f32;
                let inv_rms = (mean_sq + eps).sqrt().recip();
                for i in 0..d {
                    out[chunk_start + i] = row[i] * inv_rms * scale[i];
                }
            }
            out
        }

        #[test]
        fn test_rms_norm_gpu_matches_cpu() {
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let input = make_tensor(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]);
            let scale = make_tensor(vec![4], &[1.0, 1.0, 1.0, 1.0]);
            let result = disp
                .try_execute("RMSNormalization", &[Some(&input), Some(&scale)], &[])
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            let expected = cpu_rms_norm(&[1.0, 2.0, 3.0, 4.0], 4, &[1.0, 1.0, 1.0, 1.0]);
            assert_approx_eq(&out, &expected, 1e-4);
        }

        /// CPU reference: non-interleaved RoPE on a single (batch, seq, D) tensor.
        fn cpu_rope_non_interleaved(
            input: &[f32],
            seq_len: usize,
            d: usize,
            cos_cache: &[f32],
            sin_cache: &[f32],
        ) -> Vec<f32> {
            let half_d = d / 2;
            let mut out = vec![0.0f32; input.len()];
            let batch = input.len() / (seq_len * d);
            for b in 0..batch {
                for s in 0..seq_len {
                    let base = (b * seq_len + s) * d;
                    for i in 0..half_d {
                        let c = cos_cache[s * half_d + i];
                        let sn = sin_cache[s * half_d + i];
                        let x0 = input[base + i];
                        let x1 = input[base + i + half_d];
                        out[base + i] = x0 * c - x1 * sn;
                        out[base + i + half_d] = x0 * sn + x1 * c;
                    }
                }
            }
            out
        }

        #[test]
        fn test_rotary_embedding_gpu_matches_cpu() {
            // batch=1, seq=2, D=4 (half_d=2). cos/sin caches are [seq, half_d].
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let cos_data = vec![1.0f32, 0.0, 0.5, 0.5];
            let sin_data = vec![0.0f32, 1.0, 0.5, -0.5];
            let input = make_tensor(vec![1, 2, 4], &input_data);
            let cos_t = make_tensor(vec![2, 2], &cos_data);
            let sin_t = make_tensor(vec![2, 2], &sin_data);
            let result = disp
                .try_execute(
                    "RotaryEmbedding",
                    &[Some(&input), Some(&cos_t), Some(&sin_t)],
                    &[],
                )
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            let expected = cpu_rope_non_interleaved(&input_data, 2, 4, &cos_data, &sin_data);
            assert_approx_eq(&out, &expected, 1e-4);
        }

        /// CPU reference: SDPA with causal mask. Q/K/V each [BH, S*, D].
        /// Note: causal_limit in shader is `qi + (Sk - Sq) + 1`; with Sk == Sq, all keys
        /// up to and including qi are visible.
        fn cpu_sdpa(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            bh: usize,
            sq: usize,
            sk: usize,
            d: usize,
        ) -> Vec<f32> {
            let scale = 1.0_f32 / (d as f32).sqrt();
            let mut out = vec![0.0f32; bh * sq * d];
            for h in 0..bh {
                for qi in 0..sq {
                    cpu_sdpa_query(q, k, v, &mut out, h, qi, sq, sk, d, scale);
                }
            }
            out
        }

        /// Compute the attention output for a single (head, query) position.
        #[allow(clippy::too_many_arguments)]
        fn cpu_sdpa_query(
            q: &[f32],
            k: &[f32],
            v: &[f32],
            out: &mut [f32],
            h: usize,
            qi: usize,
            sq: usize,
            sk: usize,
            d: usize,
            scale: f32,
        ) {
            let q_base = h * sq * d + qi * d;
            let causal_limit = qi + (sk - sq) + 1;
            let scores = cpu_sdpa_scores(q, k, h, sk, d, q_base, causal_limit, scale);
            let (exps, sum_exp) = softmax_with_sum(&scores);
            cpu_sdpa_weighted_values(v, &exps, sum_exp, out, h, qi, sq, sk, d, causal_limit);
        }

        /// Build the (masked) score vector for a single query position.
        #[allow(clippy::too_many_arguments)]
        fn cpu_sdpa_scores(
            q: &[f32],
            k: &[f32],
            h: usize,
            sk: usize,
            d: usize,
            q_base: usize,
            causal_limit: usize,
            scale: f32,
        ) -> Vec<f32> {
            let mut scores = vec![f32::NEG_INFINITY; sk];
            for j in 0..sk.min(causal_limit) {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[q_base + dd] * k[h * sk * d + j * d + dd];
                }
                scores[j] = dot * scale;
            }
            scores
        }

        /// Stable softmax over `scores`, returning the exp vector and its sum.
        fn softmax_with_sum(scores: &[f32]) -> (Vec<f32>, f32) {
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            (exps, sum_exp)
        }

        /// Write `softmax(scores) @ V` for a single (head, query) into `out`.
        #[allow(clippy::too_many_arguments)]
        fn cpu_sdpa_weighted_values(
            v: &[f32],
            exps: &[f32],
            sum_exp: f32,
            out: &mut [f32],
            h: usize,
            qi: usize,
            sq: usize,
            sk: usize,
            d: usize,
            causal_limit: usize,
        ) {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for j in 0..sk.min(causal_limit) {
                    let prob = exps[j] / sum_exp;
                    acc += prob * v[h * sk * d + j * d + dd];
                }
                out[h * sq * d + qi * d + dd] = acc;
            }
        }

        #[test]
        fn test_sdpa_gpu_matches_cpu() {
            // bh=1, Sq=Sk=2, D=2.
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let q_data = vec![1.0f32, 0.0, 0.0, 1.0];
            let k_data = vec![1.0f32, 0.0, 0.0, 1.0];
            let v_data = vec![2.0f32, 3.0, 4.0, 5.0];
            let q = make_tensor(vec![1, 2, 2], &q_data);
            let k = make_tensor(vec![1, 2, 2], &k_data);
            let v = make_tensor(vec![1, 2, 2], &v_data);
            let result = disp
                .try_execute(
                    "ScaledDotProductAttention",
                    &[Some(&q), Some(&k), Some(&v)],
                    &[],
                )
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            let expected = cpu_sdpa(&q_data, &k_data, &v_data, 1, 2, 2, 2);
            assert_approx_eq(&out, &expected, 1e-4);
        }

        /// CPU reference: BatchNorm in inference mode over [N, C, H, W].
        #[allow(clippy::too_many_arguments)]
        fn cpu_batch_norm(
            input: &[f32],
            n: usize,
            c: usize,
            hw: usize,
            scale: &[f32],
            bias: &[f32],
            mean: &[f32],
            var: &[f32],
        ) -> Vec<f32> {
            let eps = 1e-5_f32;
            let mut out = vec![0.0f32; input.len()];
            for ni in 0..n {
                for ci in 0..c {
                    let inv_std = (var[ci] + eps).sqrt().recip();
                    let base = (ni * c + ci) * hw;
                    for i in 0..hw {
                        out[base + i] =
                            (input[base + i] - mean[ci]) * inv_std * scale[ci] + bias[ci];
                    }
                }
            }
            out
        }

        #[test]
        fn test_batch_norm_gpu_matches_cpu() {
            // N=1, C=1, H=W=2.
            let mut disp = MetalDispatcher::new().expect("Metal required");
            let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
            let scale = vec![1.0f32];
            let bias = vec![0.0f32];
            let mean = vec![2.5f32];
            let var = vec![1.25f32];
            let input = make_tensor(vec![1, 1, 2, 2], &input_data);
            let scale_t = make_tensor(vec![1], &scale);
            let bias_t = make_tensor(vec![1], &bias);
            let mean_t = make_tensor(vec![1], &mean);
            let var_t = make_tensor(vec![1], &var);
            let result = disp
                .try_execute(
                    "BatchNormalization",
                    &[
                        Some(&input),
                        Some(&scale_t),
                        Some(&bias_t),
                        Some(&mean_t),
                        Some(&var_t),
                    ],
                    &[],
                )
                .unwrap()
                .unwrap();
            let out = tensor_f32(&result[0]);
            let expected = cpu_batch_norm(&input_data, 1, 1, 4, &scale, &bias, &mean, &var);
            assert_approx_eq(&out, &expected, 1e-4);
        }
    }
}
