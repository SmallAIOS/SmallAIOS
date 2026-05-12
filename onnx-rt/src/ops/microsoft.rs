// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `com.microsoft`-domain fused ONNX operators for canonical HuggingFace
//! LLM exports (Llama / Gemma / DeepSeek).
//!
//! This module lands as part of the `microsoft-fused-ops-v1` OpenSpec
//! change. It implements the five `com.microsoft`-domain operators that
//! the HuggingFace `optimum-cli export onnx` pipeline emits by default
//! for modern decoder-only LLMs:
//!
//! - [`op_simplified_layer_normalization`] — thin alias for RMSNorm.
//! - [`op_skip_simplified_layer_normalization`] — fused residual-add +
//!   RMSNorm (Llama / DeepSeek).
//! - [`op_rotary_embedding`] — standalone fused RoPE (DeepSeek upstream
//!   of `MultiHeadAttention`).
//! - [`op_multi_head_attention`] — non-grouped attention fusion
//!   (DeepSeek).
//! - [`op_group_query_attention`] — grouped-query fused attention with
//!   in-op RoPE, KV-cache concat, and causal masking (Llama / Gemma).
//!
//! Two shared helpers power the attention ops:
//!
//! - [`apply_rope_in_place`] — applies rotary positional embedding to a
//!   4D tensor in both interleaved (DeepSeek) and non-interleaved
//!   (Llama / Gemma) variants.
//! - [`scaled_dot_product_attention`] — masked softmax attention over
//!   per-head Q/K/V blocks. Used by both `MultiHeadAttention` and
//!   `GroupQueryAttention`.
//!
//! All operators operate on `f32` tensors. Integer / fp16 support is out
//! of scope for this change (see `design.md` § Non-Goals).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, read_i32, read_i64, write_f32};
use crate::operators::{expf_approx, OpError};
use crate::ops::common::require_float;
use crate::ops::generative::op_rms_normalization;
use crate::tensor::{DataType, Tensor, TensorShape};

// ---------------------------------------------------------------------------
// Local math approximations
// ---------------------------------------------------------------------------

/// `sqrtf` via Newton-Raphson. Duplicated from `ops::generative` to keep
/// this module self-contained.
#[inline]
fn sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = x;
    for _ in 0..16 {
        guess = 0.5 * (guess + x / guess);
    }
    guess
}

// ---------------------------------------------------------------------------
// Shared helper: apply RoPE in place
// ---------------------------------------------------------------------------

/// Validate `rotary_dim` and the cos/sin cache layout. Returns the
/// number of rows (max_seq) in the cache and `half = rotary_dim/2`.
fn validate_rope_caches(
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    rotary_dim: usize,
    head_dim: usize,
) -> Result<(usize, usize), OpError> {
    require_float(cos_cache, "RotaryEmbedding")?;
    require_float(sin_cache, "RotaryEmbedding")?;
    if rotary_dim == 0 || rotary_dim > head_dim {
        return Err(OpError::InvalidAttribute(String::from(
            "apply_rope_in_place: rotary_dim must be in 1..=head_dim",
        )));
    }
    if !rotary_dim.is_multiple_of(2) {
        return Err(OpError::InvalidAttribute(String::from(
            "apply_rope_in_place: rotary_dim must be even",
        )));
    }
    let half = rotary_dim / 2;
    let cos_dims = &cos_cache.shape.dims;
    let sin_dims = &sin_cache.shape.dims;
    if cos_dims.len() != 2 || sin_dims.len() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "apply_rope_in_place: cos/sin caches must be rank-2",
        )));
    }
    let cache_cols = cos_dims[1] as usize;
    if sin_dims[1] as usize != cache_cols {
        return Err(OpError::ShapeMismatch(String::from(
            "apply_rope_in_place: cos and sin caches must share the inner dim",
        )));
    }
    if cache_cols != half {
        return Err(OpError::ShapeMismatch(format!(
            "apply_rope_in_place: cache inner dim {} != rotary_dim/2 = {}",
            cache_cols, half
        )));
    }
    Ok((cos_dims[0] as usize, half))
}

/// Resolve the position for a `(batch, query)` pair given a flat
/// `position_ids` buffer that is either per-sequence or per-(batch, sequence).
#[inline]
fn rope_position_for(position_ids: &[i64], bi: usize, qi: usize, sq: usize) -> i64 {
    if position_ids.len() == sq {
        position_ids[qi]
    } else {
        position_ids[bi * sq + qi]
    }
}

/// Validate that `pos` is a valid row in the cos/sin cache. Returns `pos` as a `usize`.
#[inline]
fn check_rope_position(pos: i64, cache_rows: usize) -> Result<usize, OpError> {
    if pos < 0 || (pos as usize) >= cache_rows {
        return Err(OpError::ShapeMismatch(format!(
            "apply_rope_in_place: position {} out of cache range {}",
            pos, cache_rows
        )));
    }
    Ok(pos as usize)
}

/// Rotate a single `(query, head)` block in place using the RoPE pair layout.
/// `interleaved == true` pairs `(x[2i], x[2i+1])`; otherwise `(x[i], x[i+half])`.
fn rotate_rope_block(
    raw: &mut [u8],
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    base: usize,
    pos: usize,
    cache_cols: usize,
    half: usize,
    interleaved: bool,
) {
    for pair in 0..half {
        let c = read_f32(&cos_cache.raw_data, pos * cache_cols + pair);
        let s = read_f32(&sin_cache.raw_data, pos * cache_cols + pair);
        let (i0, i1) = if interleaved {
            (base + 2 * pair, base + 2 * pair + 1)
        } else {
            (base + pair, base + pair + half)
        };
        let x0 = read_f32(raw, i0);
        let x1 = read_f32(raw, i1);
        write_f32(raw, i0, c * x0 - s * x1);
        write_f32(raw, i1, s * x0 + c * x1);
    }
}

/// Applies rotary positional embedding to a 4-D tensor in place.
///
/// The tensor is expected to have shape `(B, num_heads, Sq, head_dim)`.
/// `cos_cache` / `sin_cache` have shape `(max_seq_len, head_dim/2)` and
/// are indexed by `position_ids[b * Sq + i]` per query position.
///
/// Two layouts are supported:
///
/// - `interleaved = false` (HuggingFace / Llama / Gemma): the first
///   `head_dim/2` elements pair with the second `head_dim/2` elements.
///   `(x[i], x[i+D/2]) -> (cos*x[i] - sin*x[i+D/2], sin*x[i] + cos*x[i+D/2])`.
/// - `interleaved = true` (DeepSeek): adjacent pairs
///   `(x[2i], x[2i+1])` rotate by the same `(cos, sin)` at pair index `i`.
///
/// The input tensor's raw data is mutated directly.
pub(crate) fn apply_rope_in_place(
    raw: &mut [u8],
    shape: &[i64],
    cos_cache: &Tensor,
    sin_cache: &Tensor,
    position_ids: &[i64],
    interleaved: bool,
    rotary_dim: usize,
) -> Result<(), OpError> {
    if shape.len() != 4 {
        return Err(OpError::ShapeMismatch(String::from(
            "apply_rope_in_place: tensor must be rank-4 (B, H, Sq, head_dim)",
        )));
    }
    let b = shape[0] as usize;
    let num_heads = shape[1] as usize;
    let sq = shape[2] as usize;
    let head_dim = shape[3] as usize;

    if position_ids.len() != b * sq && position_ids.len() != sq {
        return Err(OpError::ShapeMismatch(format!(
            "apply_rope_in_place: position_ids length {} != B*Sq = {}",
            position_ids.len(),
            b * sq
        )));
    }

    let (cache_rows, half) = validate_rope_caches(cos_cache, sin_cache, rotary_dim, head_dim)?;
    let cache_cols = half;

    // Strides into the raw buffer.
    let head_stride = sq * head_dim;
    let batch_stride = num_heads * head_stride;

    for bi in 0..b {
        for hi in 0..num_heads {
            for qi in 0..sq {
                let pos =
                    check_rope_position(rope_position_for(position_ids, bi, qi, sq), cache_rows)?;
                let base = bi * batch_stride + hi * head_stride + qi * head_dim;
                rotate_rope_block(
                    raw,
                    cos_cache,
                    sin_cache,
                    base,
                    pos,
                    cache_cols,
                    half,
                    interleaved,
                );
                // Elements >= rotary_dim pass through unchanged.
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helper: scaled dot-product attention with on-the-fly causal mask
// ---------------------------------------------------------------------------

/// Compute the masked, scaled dot-product `scores[ki] = (Q_row · K_row[ki]) * scale`.
/// Positions `ki >= max_k` are filled with `-inf`.
fn sdpa_scores_row(
    q_raw: &[u8],
    k_raw: &[u8],
    q_row: usize,
    k_base: usize,
    sk: usize,
    head_dim: usize,
    max_k: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores: Vec<f32> = Vec::with_capacity(sk);
    for ki in 0..sk {
        if ki >= max_k {
            scores.push(f32::NEG_INFINITY);
            continue;
        }
        let k_row = k_base + ki * head_dim;
        let mut acc = 0.0f32;
        for d in 0..head_dim {
            acc += read_f32(q_raw, q_row + d) * read_f32(k_raw, k_row + d);
        }
        scores.push(acc * scale);
    }
    scores
}

/// Numerically-stable softmax of `scores`. Returns `None` when every entry
/// is `-inf` or the denominator underflows to zero — both cases are treated
/// by the caller as "no valid key" and produce a zero output row.
fn sdpa_softmax(scores: &[f32]) -> Option<Vec<f32>> {
    let mut max_val = f32::NEG_INFINITY;
    for &s in scores {
        if s > max_val {
            max_val = s;
        }
    }
    if !max_val.is_finite() {
        return None;
    }
    let mut denom = 0.0f32;
    let mut probs: Vec<f32> = Vec::with_capacity(scores.len());
    for &s in scores {
        let e = if s.is_finite() {
            expf_approx(s - max_val)
        } else {
            0.0
        };
        denom += e;
        probs.push(e);
    }
    if denom == 0.0 {
        return None;
    }
    let inv = 1.0 / denom;
    for p in probs.iter_mut() {
        *p *= inv;
    }
    Some(probs)
}

/// Write `probs · V` into `out[q_row..q_row+head_dim]`.
fn sdpa_write_weighted_v(
    out: &mut [u8],
    v_raw: &[u8],
    probs: &[f32],
    q_row: usize,
    v_base: usize,
    sk: usize,
    head_dim: usize,
) {
    for d in 0..head_dim {
        let mut acc = 0.0f32;
        for ki in 0..sk {
            let v_row = v_base + ki * head_dim;
            acc += probs[ki] * read_f32(v_raw, v_row + d);
        }
        write_f32(out, q_row + d, acc);
    }
}

/// Write `head_dim` zeros into `out[q_row..q_row+head_dim]`.
fn sdpa_write_zero_row(out: &mut [u8], q_row: usize, head_dim: usize) {
    for d in 0..head_dim {
        write_f32(out, q_row + d, 0.0);
    }
}

/// Number of keys query position `qi` may attend to under the (optional)
/// causal mask. Returns `sk` when `causal == false`.
#[inline]
fn sdpa_max_k(causal: bool, past_sk: usize, qi: usize, sk: usize) -> usize {
    if causal {
        (past_sk + qi + 1).min(sk)
    } else {
        sk
    }
}

/// Compute the SDPA output for one `(batch, q-head, q-pos)` row.
#[allow(clippy::too_many_arguments)]
fn sdpa_row(
    out: &mut [u8],
    q_raw: &[u8],
    k_raw: &[u8],
    v_raw: &[u8],
    q_row: usize,
    k_base: usize,
    v_base: usize,
    sk: usize,
    head_dim: usize,
    max_k: usize,
    scale: f32,
) {
    let scores = sdpa_scores_row(q_raw, k_raw, q_row, k_base, sk, head_dim, max_k, scale);
    match sdpa_softmax(&scores) {
        Some(probs) => sdpa_write_weighted_v(out, v_raw, &probs, q_row, v_base, sk, head_dim),
        None => sdpa_write_zero_row(out, q_row, head_dim),
    }
}

/// Computes `softmax(Q @ K^T * scale + causal_mask) @ V` over a single
/// batched attention block.
///
/// Inputs are laid out as:
/// - `q_raw`: `(B, Hq, Sq, head_dim)` row-major f32
/// - `k_raw`: `(B, Hk, Sk, head_dim)` row-major f32
/// - `v_raw`: `(B, Hk, Sk, head_dim)` row-major f32
///
/// Each query head is associated with a KV head by grouping: head `qh`
/// uses KV head `qh / group_size`. `group_size == 1` reproduces standard
/// multi-head attention; `group_size == Hq / Hk` reproduces GQA.
///
/// The on-the-fly causal mask sends positions with
/// `k_idx > past_sk + q_idx` to `-inf` before the softmax. `past_sk` is
/// the number of already-cached key positions, so the first query slot
/// attends over `past_sk + 1` keys and so on.
///
/// Output: `(B, Hq, Sq, head_dim)` row-major f32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scaled_dot_product_attention(
    q_raw: &[u8],
    k_raw: &[u8],
    v_raw: &[u8],
    batch: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    sq: usize,
    sk: usize,
    head_dim: usize,
    past_sk: usize,
    scale: f32,
    causal: bool,
) -> Result<Vec<u8>, OpError> {
    if num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "SDPA: num_heads / head_dim must be > 0",
        )));
    }
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(OpError::ShapeMismatch(String::from(
            "SDPA: num_q_heads must be a multiple of num_kv_heads",
        )));
    }
    let group_size = num_q_heads / num_kv_heads;

    let out_len = batch * num_q_heads * sq * head_dim;
    let mut out = allocate_tensor_data(out_len, DataType::Float);

    if sk == 0 {
        // Defensive: zero keys means the softmax is undefined. Return zeros.
        return Ok(out);
    }

    let q_head_stride = sq * head_dim;
    let q_batch_stride = num_q_heads * q_head_stride;
    let k_head_stride = sk * head_dim;
    let k_batch_stride = num_kv_heads * k_head_stride;

    for bi in 0..batch {
        for qh in 0..num_q_heads {
            let kh = qh / group_size;
            let q_base = bi * q_batch_stride + qh * q_head_stride;
            let k_base = bi * k_batch_stride + kh * k_head_stride;
            let v_base = k_base;

            for qi in 0..sq {
                let q_row = q_base + qi * head_dim;
                let max_k = sdpa_max_k(causal, past_sk, qi, sk);
                sdpa_row(
                    &mut out, q_raw, k_raw, v_raw, q_row, k_base, v_base, sk, head_dim, max_k,
                    scale,
                );
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// SimplifiedLayerNormalization
// ---------------------------------------------------------------------------

/// `com.microsoft.SimplifiedLayerNormalization`.
///
/// Mathematically identical to RMSNorm. Delegates to
/// [`op_rms_normalization`] and adds the optional bias as a post-step.
pub fn op_simplified_layer_normalization(
    inputs: &[Option<&Tensor>],
    epsilon: f32,
    axis: i64,
) -> Result<Tensor, OpError> {
    let x = inputs.first().and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from("SimplifiedLayerNormalization: missing input"))
    })?;
    let scale = inputs.get(1).and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from("SimplifiedLayerNormalization: missing scale"))
    })?;
    let bias = inputs.get(2).and_then(|t| *t);

    let mut out = op_rms_normalization(x, scale, axis, epsilon)?;
    if let Some(b) = bias {
        require_float(b, "SimplifiedLayerNormalization")?;
        // Broadcast the bias across the outer axes: the bias has the same
        // element count as the normalized region.
        let norm_size = b.shape.total_elements();
        let total = out.shape.total_elements();
        if total % norm_size != 0 {
            return Err(OpError::ShapeMismatch(String::from(
                "SimplifiedLayerNormalization: bias size incompatible with output",
            )));
        }
        for i in 0..total {
            let v = read_f32(&out.raw_data, i);
            let bv = read_f32(&b.raw_data, i % norm_size);
            write_f32(&mut out.raw_data, i, v + bv);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// SkipSimplifiedLayerNormalization
// ---------------------------------------------------------------------------

/// `com.microsoft.SkipSimplifiedLayerNormalization`.
///
/// Computes `pre = x + skip + bias` and `y = rms_norm(pre) * scale`.
/// Returns both `y` (the normalized output) and `pre` (the pre-norm sum,
/// used as the skip input for the next transformer layer).
pub fn op_skip_simplified_layer_normalization(
    inputs: &[Option<&Tensor>],
    epsilon: f32,
) -> Result<(Tensor, Tensor), OpError> {
    let x = inputs.first().and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from(
            "SkipSimplifiedLayerNormalization: missing input",
        ))
    })?;
    let skip = inputs.get(1).and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from(
            "SkipSimplifiedLayerNormalization: missing skip",
        ))
    })?;
    let scale = inputs.get(2).and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from(
            "SkipSimplifiedLayerNormalization: missing scale",
        ))
    })?;
    let bias = inputs.get(3).and_then(|t| *t);

    require_float(x, "SkipSimplifiedLayerNormalization")?;
    require_float(skip, "SkipSimplifiedLayerNormalization")?;
    if x.shape.dims != skip.shape.dims {
        return Err(OpError::ShapeMismatch(String::from(
            "SkipSimplifiedLayerNormalization: x and skip must have matching shapes",
        )));
    }

    let total = x.shape.total_elements();
    let mut pre_raw = allocate_tensor_data(total, DataType::Float);
    let bias_n = bias.map(|b| b.shape.total_elements()).unwrap_or(0);
    if let Some(b) = bias {
        require_float(b, "SkipSimplifiedLayerNormalization")?;
        if bias_n == 0 || total % bias_n != 0 {
            return Err(OpError::ShapeMismatch(String::from(
                "SkipSimplifiedLayerNormalization: bias size incompatible",
            )));
        }
    }
    for i in 0..total {
        let mut v = read_f32(&x.raw_data, i) + read_f32(&skip.raw_data, i);
        if let Some(b) = bias {
            v += read_f32(&b.raw_data, i % bias_n);
        }
        write_f32(&mut pre_raw, i, v);
    }

    let pre = Tensor {
        data_type: DataType::Float,
        shape: x.shape.clone(),
        name: String::new(),
        raw_data: pre_raw,
    };

    // Normalize the pre-sum and apply the scale via the RMSNorm kernel.
    let y = op_rms_normalization(&pre, scale, -1, epsilon)?;
    Ok((y, pre))
}

// ---------------------------------------------------------------------------
// RotaryEmbedding (standalone)
// ---------------------------------------------------------------------------

/// `com.microsoft.RotaryEmbedding` — standalone fused rotary embedding.
pub fn op_rotary_embedding(
    inputs: &[Option<&Tensor>],
    interleaved: bool,
    rotary_embedding_dim: i64,
    num_heads_attr: i64,
) -> Result<Tensor, OpError> {
    let input = inputs
        .first()
        .and_then(|t| *t)
        .ok_or_else(|| OpError::ShapeMismatch(String::from("RotaryEmbedding: missing input")))?;
    let position_ids = inputs.get(1).and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from("RotaryEmbedding: missing position_ids"))
    })?;
    let cos_cache = inputs.get(2).and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from("RotaryEmbedding: missing cos_cache"))
    })?;
    let sin_cache = inputs.get(3).and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from("RotaryEmbedding: missing sin_cache"))
    })?;
    require_float(input, "RotaryEmbedding")?;

    // Accept either rank-3 (B, Sq, hidden) or rank-4 (B, H, Sq, head_dim).
    let (view_shape, out_shape) = match input.shape.dims.len() {
        3 => {
            let b = input.shape.dims[0];
            let sq = input.shape.dims[1];
            let hidden = input.shape.dims[2];
            if num_heads_attr <= 0 {
                return Err(OpError::InvalidAttribute(String::from(
                    "RotaryEmbedding: num_heads must be > 0 for rank-3 input",
                )));
            }
            if hidden % num_heads_attr != 0 {
                return Err(OpError::ShapeMismatch(String::from(
                    "RotaryEmbedding: hidden not divisible by num_heads",
                )));
            }
            let head_dim = hidden / num_heads_attr;
            (
                vec![b, num_heads_attr, sq, head_dim],
                input.shape.dims.clone(),
            )
        }
        4 => (input.shape.dims.clone(), input.shape.dims.clone()),
        _ => {
            return Err(OpError::ShapeMismatch(String::from(
                "RotaryEmbedding: input must be rank-3 or rank-4",
            )))
        }
    };

    let head_dim = view_shape[3] as usize;
    let rotary_dim = if rotary_embedding_dim <= 0 {
        head_dim
    } else {
        rotary_embedding_dim as usize
    };

    // Flatten position_ids.
    let pos_total = position_ids.shape.total_elements();
    let mut positions: Vec<i64> = Vec::with_capacity(pos_total);
    match position_ids.data_type {
        DataType::Int64 => {
            for i in 0..pos_total {
                positions.push(read_i64(&position_ids.raw_data, i));
            }
        }
        DataType::Int32 => {
            for i in 0..pos_total {
                positions.push(read_i32(&position_ids.raw_data, i) as i64);
            }
        }
        _ => {
            return Err(OpError::ShapeMismatch(String::from(
                "RotaryEmbedding: position_ids must be int32 or int64",
            )))
        }
    }

    let mut out_raw = input.raw_data.clone();
    apply_rope_in_place(
        &mut out_raw,
        &view_shape,
        cos_cache,
        sin_cache,
        &positions,
        interleaved,
        rotary_dim,
    )?;

    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(out_shape),
        name: String::new(),
        raw_data: out_raw,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers for MHA / GQA
// ---------------------------------------------------------------------------

/// Reshape `(B, Sq, num_heads * head_dim)` row-major into
/// `(B, num_heads, Sq, head_dim)` row-major. Input is read-only.
fn reshape_to_bhsd(src: &[u8], b: usize, sq: usize, num_heads: usize, head_dim: usize) -> Vec<u8> {
    let n = b * num_heads * sq * head_dim;
    let mut out = allocate_tensor_data(n, DataType::Float);
    for bi in 0..b {
        for si in 0..sq {
            for hi in 0..num_heads {
                for d in 0..head_dim {
                    // Source layout: [b, si, hi, d] with inner stride head_dim.
                    let src_idx = ((bi * sq + si) * num_heads + hi) * head_dim + d;
                    // Dest layout: [b, hi, si, d]
                    let dst_idx = ((bi * num_heads + hi) * sq + si) * head_dim + d;
                    let v = read_f32(src, src_idx);
                    write_f32(&mut out, dst_idx, v);
                }
            }
        }
    }
    out
}

/// Inverse of [`reshape_to_bhsd`]: `(B, num_heads, Sq, head_dim)` back to
/// `(B, Sq, num_heads * head_dim)`.
fn reshape_from_bhsd(
    src: &[u8],
    b: usize,
    num_heads: usize,
    sq: usize,
    head_dim: usize,
) -> Vec<u8> {
    let n = b * sq * num_heads * head_dim;
    let mut out = allocate_tensor_data(n, DataType::Float);
    for bi in 0..b {
        for hi in 0..num_heads {
            for si in 0..sq {
                for d in 0..head_dim {
                    let src_idx = ((bi * num_heads + hi) * sq + si) * head_dim + d;
                    let dst_idx = ((bi * sq + si) * num_heads + hi) * head_dim + d;
                    let v = read_f32(src, src_idx);
                    write_f32(&mut out, dst_idx, v);
                }
            }
        }
    }
    out
}

/// Validate a (potentially absent) past KV cache tensor and return its
/// sequence-length axis (`past_sk`). Returns 0 when `past` is `None`.
fn validate_past_kv(
    past: Option<&Tensor>,
    batch: usize,
    heads: usize,
    head_dim: usize,
) -> Result<usize, OpError> {
    let Some(p) = past else {
        return Ok(0);
    };
    require_float(p, "GroupQueryAttention")?;
    if p.shape.dims.len() != 4 {
        return Err(OpError::ShapeMismatch(String::from(
            "GQA/MHA: past_key/past_value must be rank-4",
        )));
    }
    if p.shape.dims[0] as usize != batch
        || p.shape.dims[1] as usize != heads
        || p.shape.dims[3] as usize != head_dim
    {
        return Err(OpError::ShapeMismatch(String::from(
            "GQA/MHA: past cache batch/heads/head_dim mismatch",
        )));
    }
    Ok(p.shape.dims[2] as usize)
}

/// Copy `src_sk` rows of `(head_dim)` from `src_raw` into the `(B,H,total_sk,head_dim)`
/// destination starting at `dst_start_si` along the sequence axis.
fn kv_copy_block(
    dst: &mut [u8],
    src_raw: &[u8],
    batch: usize,
    heads: usize,
    src_sk: usize,
    total_sk: usize,
    head_dim: usize,
    dst_start_si: usize,
) {
    for bi in 0..batch {
        for hi in 0..heads {
            for si in 0..src_sk {
                for d in 0..head_dim {
                    let src = ((bi * heads + hi) * src_sk + si) * head_dim + d;
                    let dst_idx = ((bi * heads + hi) * total_sk + dst_start_si + si) * head_dim + d;
                    write_f32(dst, dst_idx, read_f32(src_raw, src));
                }
            }
        }
    }
}

/// Concatenate past + new along the sequence axis of a
/// `(B, H, Sk, head_dim)` buffer. Produces `(B, H, past_sk + new_sk,
/// head_dim)`. If `past_sk == 0`, returns a clone of `new`.
fn kv_concat(
    past: Option<&Tensor>,
    new_raw: &[u8],
    batch: usize,
    heads: usize,
    new_sk: usize,
    head_dim: usize,
) -> Result<(Vec<u8>, usize), OpError> {
    let past_sk = validate_past_kv(past, batch, heads, head_dim)?;
    let total_sk = past_sk + new_sk;
    let out_len = batch * heads * total_sk * head_dim;
    let mut out = allocate_tensor_data(out_len, DataType::Float);

    if let Some(p) = past {
        kv_copy_block(
            &mut out,
            &p.raw_data,
            batch,
            heads,
            past_sk,
            total_sk,
            head_dim,
            0,
        );
    }
    kv_copy_block(
        &mut out, new_raw, batch, heads, new_sk, total_sk, head_dim, past_sk,
    );

    Ok((out, past_sk))
}

// ---------------------------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------------------------

/// `com.microsoft.MultiHeadAttention` (unpacked Q/K/V form).
///
/// Inputs: `query`, `key`, `value`, optional `bias` (ignored — present
/// in some exports as a zero tensor), optional `key_padding_mask`
/// (unsupported — returns error), optional `relative_position_bias`
/// (unsupported), optional `past_key`, optional `past_value`.
///
/// Returns three tensors: `attention_out`, `present_key`, `present_value`.
pub fn op_multi_head_attention(
    inputs: &[Option<&Tensor>],
    num_heads: i64,
    scale_attr: Option<f32>,
) -> Result<(Tensor, Tensor, Tensor), OpError> {
    let query = inputs
        .first()
        .and_then(|t| *t)
        .ok_or_else(|| OpError::ShapeMismatch(String::from("MultiHeadAttention: missing query")))?;
    let key = inputs
        .get(1)
        .and_then(|t| *t)
        .ok_or_else(|| OpError::ShapeMismatch(String::from("MultiHeadAttention: missing key")))?;
    let value = inputs
        .get(2)
        .and_then(|t| *t)
        .ok_or_else(|| OpError::ShapeMismatch(String::from("MultiHeadAttention: missing value")))?;

    // inputs[3] = bias, inputs[4] = key_padding_mask, inputs[5] = relative_position_bias
    // inputs[6] = past_key, inputs[7] = past_value
    let past_key = inputs.get(6).and_then(|t| *t);
    let past_value = inputs.get(7).and_then(|t| *t);

    require_float(query, "MultiHeadAttention")?;
    require_float(key, "MultiHeadAttention")?;
    require_float(value, "MultiHeadAttention")?;

    if num_heads <= 0 {
        return Err(OpError::InvalidAttribute(String::from(
            "MultiHeadAttention: num_heads must be > 0",
        )));
    }
    let num_heads = num_heads as usize;

    if query.shape.dims.len() != 3 || key.shape.dims.len() != 3 || value.shape.dims.len() != 3 {
        return Err(OpError::ShapeMismatch(String::from(
            "MultiHeadAttention: Q/K/V must be rank-3 (B, Sq, hidden)",
        )));
    }
    let b = query.shape.dims[0] as usize;
    let sq = query.shape.dims[1] as usize;
    let hidden = query.shape.dims[2] as usize;
    if !hidden.is_multiple_of(num_heads) {
        return Err(OpError::ShapeMismatch(String::from(
            "MultiHeadAttention: hidden not divisible by num_heads",
        )));
    }
    let head_dim = hidden / num_heads;

    if key.shape.dims[0] as usize != b
        || key.shape.dims[1] as usize != sq
        || key.shape.dims[2] as usize != hidden
    {
        return Err(OpError::ShapeMismatch(String::from(
            "MultiHeadAttention: K shape must match Q (unpacked form)",
        )));
    }
    if value.shape.dims[0] as usize != b
        || value.shape.dims[1] as usize != sq
        || value.shape.dims[2] as usize != hidden
    {
        return Err(OpError::ShapeMismatch(String::from(
            "MultiHeadAttention: V shape must match Q",
        )));
    }

    // Reshape Q/K/V into (B, H, Sq, head_dim).
    let q_bhsd = reshape_to_bhsd(&query.raw_data, b, sq, num_heads, head_dim);
    let k_new = reshape_to_bhsd(&key.raw_data, b, sq, num_heads, head_dim);
    let v_new = reshape_to_bhsd(&value.raw_data, b, sq, num_heads, head_dim);

    // Concat past KV with new.
    let (k_full, past_sk) = kv_concat(past_key, &k_new, b, num_heads, sq, head_dim)?;
    let (v_full, past_sk_v) = kv_concat(past_value, &v_new, b, num_heads, sq, head_dim)?;
    if past_sk != past_sk_v {
        return Err(OpError::ShapeMismatch(String::from(
            "MultiHeadAttention: past_key/past_value sequence length mismatch",
        )));
    }
    let total_sk = past_sk + sq;

    let scale = scale_attr.unwrap_or_else(|| 1.0 / sqrt_f32(head_dim as f32));

    let out_bhsd = scaled_dot_product_attention(
        &q_bhsd, &k_full, &v_full, b, num_heads, num_heads, sq, total_sk, head_dim, past_sk, scale,
        true,
    )?;

    let out_raw = reshape_from_bhsd(&out_bhsd, b, num_heads, sq, head_dim);

    let attn_out = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![b as i64, sq as i64, hidden as i64]),
        name: String::new(),
        raw_data: out_raw,
    };
    let present_k = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![
            b as i64,
            num_heads as i64,
            total_sk as i64,
            head_dim as i64,
        ]),
        name: String::new(),
        raw_data: k_full,
    };
    let present_v = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![
            b as i64,
            num_heads as i64,
            total_sk as i64,
            head_dim as i64,
        ]),
        name: String::new(),
        raw_data: v_full,
    };
    Ok((attn_out, present_k, present_v))
}

// ---------------------------------------------------------------------------
// GroupQueryAttention
// ---------------------------------------------------------------------------

/// Validate the per-attribute and per-tensor shape invariants for GQA.
/// Returns `(num_heads, kv_heads, b, sq, q_hidden, head_dim)`.
fn gqa_validate(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_heads: i64,
    kv_num_heads: i64,
) -> Result<(usize, usize, usize, usize, usize, usize), OpError> {
    if num_heads <= 0 || kv_num_heads <= 0 {
        return Err(OpError::InvalidAttribute(String::from(
            "GroupQueryAttention: num_heads and kv_num_heads must be > 0",
        )));
    }
    if num_heads % kv_num_heads != 0 {
        return Err(OpError::InvalidAttribute(String::from(
            "GroupQueryAttention: num_heads must be a multiple of kv_num_heads",
        )));
    }
    let num_heads = num_heads as usize;
    let kv_heads = kv_num_heads as usize;

    if query.shape.dims.len() != 3 || key.shape.dims.len() != 3 || value.shape.dims.len() != 3 {
        return Err(OpError::ShapeMismatch(String::from(
            "GroupQueryAttention: Q/K/V must be rank-3",
        )));
    }
    let b = query.shape.dims[0] as usize;
    let sq = query.shape.dims[1] as usize;
    let q_hidden = query.shape.dims[2] as usize;
    let kv_hidden = key.shape.dims[2] as usize;
    if !q_hidden.is_multiple_of(num_heads) {
        return Err(OpError::ShapeMismatch(String::from(
            "GroupQueryAttention: q_hidden not divisible by num_heads",
        )));
    }
    if !kv_hidden.is_multiple_of(kv_heads) {
        return Err(OpError::ShapeMismatch(String::from(
            "GroupQueryAttention: kv_hidden not divisible by kv_num_heads",
        )));
    }
    let head_dim = q_hidden / num_heads;
    if kv_hidden / kv_heads != head_dim {
        return Err(OpError::ShapeMismatch(String::from(
            "GroupQueryAttention: Q and KV head_dim must match",
        )));
    }
    if value.shape.dims != key.shape.dims {
        return Err(OpError::ShapeMismatch(String::from(
            "GroupQueryAttention: key and value must have the same shape",
        )));
    }
    Ok((num_heads, kv_heads, b, sq, q_hidden, head_dim))
}

/// Past-key sequence length for GQA's pre-validation step (before `kv_concat`
/// runs the deeper checks).
fn gqa_past_sk(past_key: Option<&Tensor>) -> Result<usize, OpError> {
    match past_key {
        Some(p) => {
            if p.shape.dims.len() != 4 {
                return Err(OpError::ShapeMismatch(String::from(
                    "GroupQueryAttention: past_key must be rank-4",
                )));
            }
            Ok(p.shape.dims[2] as usize)
        }
        None => Ok(0),
    }
}

/// Apply fused RoPE to Q and K when `do_rotary` is set.
#[allow(clippy::too_many_arguments)]
fn gqa_apply_rotary(
    q_bhsd: &mut [u8],
    k_bhsd: &mut [u8],
    cos_cache: Option<&Tensor>,
    sin_cache: Option<&Tensor>,
    b: usize,
    num_heads: usize,
    kv_heads: usize,
    sq: usize,
    head_dim: usize,
    past_sk: usize,
    rotary_interleaved: bool,
) -> Result<(), OpError> {
    let cos = cos_cache.ok_or_else(|| {
        OpError::ShapeMismatch(String::from(
            "GroupQueryAttention: do_rotary=1 but cos_cache is missing",
        ))
    })?;
    let sin = sin_cache.ok_or_else(|| {
        OpError::ShapeMismatch(String::from(
            "GroupQueryAttention: do_rotary=1 but sin_cache is missing",
        ))
    })?;
    let mut positions: Vec<i64> = Vec::with_capacity(sq);
    for i in 0..sq {
        positions.push((past_sk + i) as i64);
    }
    let q_shape = vec![b as i64, num_heads as i64, sq as i64, head_dim as i64];
    let k_shape = vec![b as i64, kv_heads as i64, sq as i64, head_dim as i64];
    apply_rope_in_place(
        q_bhsd,
        &q_shape,
        cos,
        sin,
        &positions,
        rotary_interleaved,
        head_dim,
    )?;
    apply_rope_in_place(
        k_bhsd,
        &k_shape,
        cos,
        sin,
        &positions,
        rotary_interleaved,
        head_dim,
    )?;
    Ok(())
}

/// `com.microsoft.GroupQueryAttention`.
///
/// Inputs (per ORT contrib op spec):
/// - `query`  `(B, Sq, num_heads * head_dim)`
/// - `key`    `(B, Sq, kv_num_heads * head_dim)`
/// - `value`  `(B, Sq, kv_num_heads * head_dim)`
/// - `past_key`   `(B, kv_num_heads, past_sk, head_dim)` — may be empty.
/// - `past_value` `(B, kv_num_heads, past_sk, head_dim)`
/// - `seqlens_k` — int32, per-batch current total key length (unused in
///   this simplified impl; we honour it only to the extent that the
///   concat length determines attention).
/// - `total_sequence_length` — int32 scalar, unused.
/// - `cos_cache`  `(max_seq_len, head_dim/2)` — required when `do_rotary`.
/// - `sin_cache`  `(max_seq_len, head_dim/2)` — required when `do_rotary`.
///
/// Outputs: `attention_out`, `present_key`, `present_value`.
///
/// **KV cache lifecycle (deviation from spec D8):** the spec describes
/// in-place mutation of the KV cache, but `dispatch_node` passes inputs as
/// `&[Option<&Tensor>]` (read-only) and returns owned outputs. Rather than
/// rework the dispatcher to support interior mutability, this op returns
/// fresh `present_key` / `present_value` tensors that the executor inserts
/// into the value map. When called inside a `Loop` body via the Phase 2
/// sub-graph executor, these become loop-carried values rotated into the
/// next iteration's input slot — observably identical to in-place mutation,
/// at the cost of one full KV-cache copy per iteration. The pending
/// `kv-cache-quantization-v1` work compresses this cost ~6x.
#[allow(clippy::too_many_arguments)]
pub fn op_group_query_attention(
    inputs: &[Option<&Tensor>],
    num_heads: i64,
    kv_num_heads: i64,
    scale_attr: Option<f32>,
    local_window_size: i64,
    do_rotary: bool,
    rotary_interleaved: bool,
) -> Result<(Tensor, Tensor, Tensor), OpError> {
    if local_window_size != -1 {
        return Err(OpError::InvalidAttribute(String::from(
            "GroupQueryAttention: local_window_size != -1 not supported in this tier",
        )));
    }
    let query = inputs.first().and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from("GroupQueryAttention: missing query"))
    })?;
    let key = inputs
        .get(1)
        .and_then(|t| *t)
        .ok_or_else(|| OpError::ShapeMismatch(String::from("GroupQueryAttention: missing key")))?;
    let value = inputs.get(2).and_then(|t| *t).ok_or_else(|| {
        OpError::ShapeMismatch(String::from("GroupQueryAttention: missing value"))
    })?;
    let past_key = inputs.get(3).and_then(|t| *t);
    let past_value = inputs.get(4).and_then(|t| *t);
    // inputs[5] = seqlens_k, inputs[6] = total_sequence_length — advisory only.
    let cos_cache = inputs.get(7).and_then(|t| *t);
    let sin_cache = inputs.get(8).and_then(|t| *t);

    require_float(query, "GroupQueryAttention")?;
    require_float(key, "GroupQueryAttention")?;
    require_float(value, "GroupQueryAttention")?;

    let (num_heads, kv_heads, b, sq, q_hidden, head_dim) =
        gqa_validate(query, key, value, num_heads, kv_num_heads)?;

    let past_sk = gqa_past_sk(past_key)?;

    // Reshape Q and KV into (B, H*, Sq, head_dim).
    let mut q_bhsd = reshape_to_bhsd(&query.raw_data, b, sq, num_heads, head_dim);
    let mut k_bhsd = reshape_to_bhsd(&key.raw_data, b, sq, kv_heads, head_dim);
    let v_bhsd = reshape_to_bhsd(&value.raw_data, b, sq, kv_heads, head_dim);

    if do_rotary {
        gqa_apply_rotary(
            &mut q_bhsd,
            &mut k_bhsd,
            cos_cache,
            sin_cache,
            b,
            num_heads,
            kv_heads,
            sq,
            head_dim,
            past_sk,
            rotary_interleaved,
        )?;
    }

    // Concat past with new along the key-sequence axis.
    let (k_full, past_sk_k) = kv_concat(past_key, &k_bhsd, b, kv_heads, sq, head_dim)?;
    let (v_full, past_sk_v) = kv_concat(past_value, &v_bhsd, b, kv_heads, sq, head_dim)?;
    if past_sk_k != past_sk || past_sk_v != past_sk {
        return Err(OpError::ShapeMismatch(String::from(
            "GroupQueryAttention: past_key / past_value sequence mismatch",
        )));
    }
    let total_sk = past_sk + sq;

    // Attention dispatch via the shared SDPA helper.
    let scale = scale_attr.unwrap_or_else(|| 1.0 / sqrt_f32(head_dim as f32));
    let out_bhsd = scaled_dot_product_attention(
        &q_bhsd, &k_full, &v_full, b, num_heads, kv_heads, sq, total_sk, head_dim, past_sk, scale,
        true,
    )?;

    let out_raw = reshape_from_bhsd(&out_bhsd, b, num_heads, sq, head_dim);

    let attn_out = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![b as i64, sq as i64, q_hidden as i64]),
        name: String::new(),
        raw_data: out_raw,
    };
    let present_k = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![
            b as i64,
            kv_heads as i64,
            total_sk as i64,
            head_dim as i64,
        ]),
        name: String::new(),
        raw_data: k_full,
    };
    let present_v = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![
            b as i64,
            kv_heads as i64,
            total_sk as i64,
            head_dim as i64,
        ]),
        name: String::new(),
        raw_data: v_full,
    };
    Ok((attn_out, present_k, present_v))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_io::write_i64;
    use crate::ops::common::test_helpers::{make_f32, make_i64, read_all_f32};

    // ---------- sqrt_f32 ----------
    #[test]
    fn sqrt_f32_basic() {
        assert!((sqrt_f32(4.0) - 2.0).abs() < 1e-5);
        assert!((sqrt_f32(2.0) - core::f32::consts::SQRT_2).abs() < 1e-5);
        assert_eq!(sqrt_f32(0.0), 0.0);
        assert_eq!(sqrt_f32(-1.0), 0.0);
    }

    // ---------- SimplifiedLayerNormalization ----------
    #[test]
    fn simplified_layer_norm_matches_rmsnorm() {
        let x = make_f32(&[2, 4], &[1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25, 2.0]);
        let scale = make_f32(&[4], &[1.0, 1.0, 1.0, 1.0]);
        let inputs = [Some(&x), Some(&scale)];
        let got = op_simplified_layer_normalization(&inputs, 1e-5, -1).unwrap();
        let want = op_rms_normalization(&x, &scale, -1, 1e-5).unwrap();
        let g = read_all_f32(&got);
        let w = read_all_f32(&want);
        for i in 0..g.len() {
            assert!(
                (g[i] - w[i]).abs() < 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                g[i],
                w[i]
            );
        }
    }

    #[test]
    fn simplified_layer_norm_with_bias() {
        let x = make_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        let scale = make_f32(&[3], &[1.0, 1.0, 1.0]);
        let bias = make_f32(&[3], &[0.1, 0.2, 0.3]);
        let inputs = [Some(&x), Some(&scale), Some(&bias)];
        let got = op_simplified_layer_normalization(&inputs, 1e-5, -1).unwrap();
        let base = op_rms_normalization(&x, &scale, -1, 1e-5).unwrap();
        let g = read_all_f32(&got);
        let b = read_all_f32(&base);
        assert!((g[0] - (b[0] + 0.1)).abs() < 1e-6);
        assert!((g[1] - (b[1] + 0.2)).abs() < 1e-6);
        assert!((g[2] - (b[2] + 0.3)).abs() < 1e-6);
    }

    #[test]
    fn simplified_layer_norm_rejects_missing_inputs() {
        let x = make_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        let inputs = [Some(&x)];
        assert!(op_simplified_layer_normalization(&inputs, 1e-5, -1).is_err());
    }

    #[test]
    fn simplified_layer_norm_single_element() {
        let x = make_f32(&[1, 1], &[5.0]);
        let scale = make_f32(&[1], &[2.0]);
        let inputs = [Some(&x), Some(&scale)];
        let got = op_simplified_layer_normalization(&inputs, 1e-5, -1).unwrap();
        let g = read_all_f32(&got);
        // rms = sqrt(25 + eps) ~= 5; x / 5 * 2 = 2.
        assert!((g[0] - 2.0).abs() < 1e-4);
    }

    // ---------- SkipSimplifiedLayerNormalization ----------
    #[test]
    fn skip_simplified_layer_norm_fused_add() {
        let x = make_f32(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let skip = make_f32(&[1, 4], &[0.5, 0.5, 0.5, 0.5]);
        let scale = make_f32(&[4], &[1.0, 1.0, 1.0, 1.0]);
        let inputs = [Some(&x), Some(&skip), Some(&scale)];
        let (y, pre) = op_skip_simplified_layer_normalization(&inputs, 1e-5).unwrap();
        let pre_vals = read_all_f32(&pre);
        assert!((pre_vals[0] - 1.5).abs() < 1e-6);
        assert!((pre_vals[1] - 2.5).abs() < 1e-6);
        assert!((pre_vals[2] - 3.5).abs() < 1e-6);
        assert!((pre_vals[3] - 4.5).abs() < 1e-6);
        // y should equal rmsnorm(pre).
        let want = op_rms_normalization(&pre, &scale, -1, 1e-5).unwrap();
        let y_vals = read_all_f32(&y);
        let w_vals = read_all_f32(&want);
        for i in 0..y_vals.len() {
            assert!((y_vals[i] - w_vals[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn skip_simplified_layer_norm_with_bias() {
        let x = make_f32(&[1, 3], &[1.0, 1.0, 1.0]);
        let skip = make_f32(&[1, 3], &[1.0, 1.0, 1.0]);
        let scale = make_f32(&[3], &[1.0, 1.0, 1.0]);
        let bias = make_f32(&[3], &[0.5, 0.5, 0.5]);
        let inputs = [Some(&x), Some(&skip), Some(&scale), Some(&bias)];
        let (_, pre) = op_skip_simplified_layer_normalization(&inputs, 1e-5).unwrap();
        let p = read_all_f32(&pre);
        assert!((p[0] - 2.5).abs() < 1e-6);
        assert!((p[1] - 2.5).abs() < 1e-6);
        assert!((p[2] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn skip_simplified_layer_norm_shape_mismatch() {
        let x = make_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        let skip = make_f32(&[1, 4], &[1.0, 1.0, 1.0, 1.0]);
        let scale = make_f32(&[3], &[1.0, 1.0, 1.0]);
        let inputs = [Some(&x), Some(&skip), Some(&scale)];
        assert!(op_skip_simplified_layer_normalization(&inputs, 1e-5).is_err());
    }

    #[test]
    fn skip_simplified_layer_norm_missing_scale() {
        let x = make_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        let skip = make_f32(&[1, 3], &[1.0, 1.0, 1.0]);
        let inputs = [Some(&x), Some(&skip)];
        assert!(op_skip_simplified_layer_normalization(&inputs, 1e-5).is_err());
    }

    // ---------- RotaryEmbedding ----------
    fn make_rope_caches(max_seq: usize, half: usize) -> (Tensor, Tensor) {
        // Pre-computed deterministic cos/sin tables. At position 0 every
        // entry is (cos=1, sin=0) so row-0 is the identity rotation.
        let mut cos_vals = Vec::with_capacity(max_seq * half);
        let mut sin_vals = Vec::with_capacity(max_seq * half);
        for p in 0..max_seq {
            for i in 0..half {
                let theta = p as f32 * (0.1 + i as f32 * 0.01);
                cos_vals.push(taylor_cos(theta));
                sin_vals.push(taylor_sin(theta));
            }
        }
        let dims = [max_seq as i64, half as i64];
        (make_f32(&dims, &cos_vals), make_f32(&dims, &sin_vals))
    }

    fn taylor_cos(x: f32) -> f32 {
        // Small-x Taylor (tests use |x| < 4).
        let x2 = x * x;
        1.0 - x2 / 2.0 + x2 * x2 / 24.0 - x2 * x2 * x2 / 720.0
    }
    fn taylor_sin(x: f32) -> f32 {
        let x2 = x * x;
        x * (1.0 - x2 / 6.0 + x2 * x2 / 120.0 - x2 * x2 * x2 / 5040.0)
    }

    #[test]
    fn rope_position_zero_is_identity() {
        // At position 0, cos=1, sin=0 -> no rotation.
        let max_seq = 4;
        let half = 2;
        let (cos_c, sin_c) = make_rope_caches(max_seq, half);
        let input = make_f32(&[1, 1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let pos = make_i64(&[1, 1], &[0]);
        let inputs = [Some(&input), Some(&pos), Some(&cos_c), Some(&sin_c)];
        let out = op_rotary_embedding(&inputs, false, 0, 1).unwrap();
        let v = read_all_f32(&out);
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[1] - 2.0).abs() < 1e-5);
        assert!((v[2] - 3.0).abs() < 1e-5);
        assert!((v[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn rope_interleaved_vs_noninterleaved_differ() {
        // Distinct layouts must produce distinct outputs at a non-zero pos.
        let max_seq = 4;
        let half = 2;
        let (cos_c, sin_c) = make_rope_caches(max_seq, half);
        let input = make_f32(&[1, 1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let pos = make_i64(&[1, 1], &[1]);

        let inputs = [Some(&input), Some(&pos), Some(&cos_c), Some(&sin_c)];
        let a = op_rotary_embedding(&inputs, false, 0, 1).unwrap();
        let b = op_rotary_embedding(&inputs, true, 0, 1).unwrap();
        let av = read_all_f32(&a);
        let bv = read_all_f32(&b);
        let diff = av.iter().zip(bv.iter()).any(|(x, y)| (x - y).abs() > 1e-4);
        assert!(diff, "interleaved and non-interleaved RoPE should differ");
    }

    #[test]
    fn rope_non_interleaved_manual_calc() {
        // head_dim=4, half=2, pos=1. cos[1,0]=taylor_cos(0.1), etc.
        let max_seq = 2;
        let half = 2;
        let (cos_c, sin_c) = make_rope_caches(max_seq, half);
        let input = make_f32(&[1, 1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let pos = make_i64(&[1, 1], &[1]);
        let inputs = [Some(&input), Some(&pos), Some(&cos_c), Some(&sin_c)];
        let out = op_rotary_embedding(&inputs, false, 0, 1).unwrap();
        let v = read_all_f32(&out);

        // Manual: non-interleaved pair (x[0], x[2]) with (c0, s0) and
        // (x[1], x[3]) with (c1, s1). With theta = p * (0.1 + i*0.01)
        // and p=1, pair 0 -> theta=0.1, pair 1 -> theta=0.11.
        let c0 = taylor_cos(0.1);
        let s0 = taylor_sin(0.1);
        let c1 = taylor_cos(0.11);
        let s1 = taylor_sin(0.11);
        let expected_0 = c0 * 1.0 - s0 * 3.0;
        let expected_2 = s0 * 1.0 + c0 * 3.0;
        let expected_1 = c1 * 2.0 - s1 * 4.0;
        let expected_3 = s1 * 2.0 + c1 * 4.0;
        assert!((v[0] - expected_0).abs() < 1e-4);
        assert!((v[1] - expected_1).abs() < 1e-4);
        assert!((v[2] - expected_2).abs() < 1e-4);
        assert!((v[3] - expected_3).abs() < 1e-4);
    }

    #[test]
    fn rope_rejects_bad_rank() {
        let (cos_c, sin_c) = make_rope_caches(2, 2);
        // rank-2 input is invalid.
        let input = make_f32(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let pos = make_i64(&[1], &[0]);
        let inputs = [Some(&input), Some(&pos), Some(&cos_c), Some(&sin_c)];
        assert!(op_rotary_embedding(&inputs, false, 0, 1).is_err());
    }

    // ---------- SDPA helper ----------
    #[test]
    fn sdpa_single_head_non_causal_matches_manual() {
        // head_dim=2, sq=1, sk=2, single batch, single head.
        // Q = [1, 0]; K = [[1, 0], [0, 1]]; V = [[1, 2], [3, 4]]; scale = 1.0
        // scores = [1, 0]; softmax = [e/(e+1), 1/(e+1)]
        // E = exp(1), s0 = E/(E+1), s1 = 1/(E+1)
        // out = s0 * [1,2] + s1 * [3,4]
        let q = make_f32(&[1, 1, 1, 2], &[1.0, 0.0]);
        let k = make_f32(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let v = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let out = scaled_dot_product_attention(
            &q.raw_data,
            &k.raw_data,
            &v.raw_data,
            1,
            1,
            1,
            1,
            2,
            2,
            0,
            1.0,
            false,
        )
        .unwrap();
        let e = crate::operators::expf_approx(1.0);
        let denom = e + 1.0;
        let s0 = e / denom;
        let s1 = 1.0 / denom;
        let expected_0 = s0 * 1.0 + s1 * 3.0;
        let expected_1 = s0 * 2.0 + s1 * 4.0;
        assert!(
            (read_f32(&out, 0) - expected_0).abs() < 1e-4,
            "got {}, want {}",
            read_f32(&out, 0),
            expected_0
        );
        assert!((read_f32(&out, 1) - expected_1).abs() < 1e-4);
    }

    #[test]
    fn sdpa_causal_mask_exact_zero() {
        // sq=3, sk=3 causal. q=0 attends only to k=0; q=1 attends to k=0..1;
        // q=2 attends to all. All Q/K/V entries = 1 so attention pattern
        // after softmax is uniform over the unmasked keys.
        let q = make_f32(&[1, 1, 3, 1], &[1.0, 1.0, 1.0]);
        let k = make_f32(&[1, 1, 3, 1], &[1.0, 1.0, 1.0]);
        let v = make_f32(&[1, 1, 3, 1], &[2.0, 4.0, 8.0]);
        let out = scaled_dot_product_attention(
            &q.raw_data,
            &k.raw_data,
            &v.raw_data,
            1,
            1,
            1,
            3,
            3,
            1,
            0,
            1.0,
            true,
        )
        .unwrap();
        // q=0 -> v[0] = 2; q=1 -> (2+4)/2 = 3; q=2 -> (2+4+8)/3 ≈ 4.666.
        assert!((read_f32(&out, 0) - 2.0).abs() < 1e-4);
        assert!((read_f32(&out, 1) - 3.0).abs() < 1e-4);
        assert!((read_f32(&out, 2) - 14.0 / 3.0).abs() < 1e-3);
    }

    #[test]
    fn sdpa_gqa_grouping() {
        // num_q_heads=2, num_kv_heads=1 (group_size=2). Both q-heads see the
        // same KV. Use simple zeros queries -> uniform softmax.
        let q = make_f32(&[1, 2, 1, 2], &[0.0; 4]);
        let k = make_f32(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let v = make_f32(&[1, 1, 2, 2], &[2.0, 3.0, 4.0, 5.0]);
        let out = scaled_dot_product_attention(
            &q.raw_data,
            &k.raw_data,
            &v.raw_data,
            1,
            2,
            1,
            1,
            2,
            2,
            0,
            1.0,
            false,
        )
        .unwrap();
        // Uniform softmax = [0.5, 0.5]; output = (v[0] + v[1]) / 2 = [3, 4].
        for h in 0..2 {
            assert!((read_f32(&out, h * 2) - 3.0).abs() < 1e-4);
            assert!((read_f32(&out, h * 2 + 1) - 4.0).abs() < 1e-4);
        }
    }

    #[test]
    fn sdpa_rejects_bad_grouping() {
        let q = make_f32(&[1, 3, 1, 2], &[0.0; 6]);
        let k = make_f32(&[1, 2, 1, 2], &[0.0; 4]);
        let v = make_f32(&[1, 2, 1, 2], &[0.0; 4]);
        // num_q_heads=3, num_kv_heads=2 -> not a multiple.
        let r = scaled_dot_product_attention(
            &q.raw_data,
            &k.raw_data,
            &v.raw_data,
            1,
            3,
            2,
            1,
            1,
            2,
            0,
            1.0,
            false,
        );
        assert!(r.is_err());
    }

    // ---------- MultiHeadAttention ----------
    #[test]
    fn mha_basic_two_heads_no_cache() {
        // B=1, Sq=2, num_heads=2, head_dim=2 -> hidden=4.
        let q = make_f32(&[1, 2, 4], &[0.0; 8]);
        let k = make_f32(&[1, 2, 4], &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let v = make_f32(&[1, 2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let inputs = [Some(&q), Some(&k), Some(&v), None, None, None, None, None];
        let (attn, pk, pv) = op_multi_head_attention(&inputs, 2, None).unwrap();
        // Check output shape.
        assert_eq!(attn.shape.dims, vec![1, 2, 4]);
        // With Q=0 and causal mask, output at q=0 is only v[0], at q=1 is
        // mean of v[0], v[1]. Verify present cache has correct length.
        assert_eq!(pk.shape.dims, vec![1, 2, 2, 2]);
        assert_eq!(pv.shape.dims, vec![1, 2, 2, 2]);
    }

    #[test]
    fn mha_with_past_cache() {
        let q = make_f32(&[1, 1, 4], &[0.0; 4]);
        let k = make_f32(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let v = make_f32(&[1, 1, 4], &[5.0, 6.0, 7.0, 8.0]);
        // past: (1, 2, 1, 2)
        let past_k = make_f32(&[1, 2, 1, 2], &[0.1, 0.2, 0.3, 0.4]);
        let past_v = make_f32(&[1, 2, 1, 2], &[0.5, 0.6, 0.7, 0.8]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            Some(&past_k),
            Some(&past_v),
        ];
        let (_attn, pk, _pv) = op_multi_head_attention(&inputs, 2, None).unwrap();
        // present_sk = 1 past + 1 new = 2
        assert_eq!(pk.shape.dims, vec![1, 2, 2, 2]);
    }

    #[test]
    fn mha_rejects_rank_mismatch() {
        let q = make_f32(&[1, 2], &[0.0; 2]);
        let k = make_f32(&[1, 2, 4], &[0.0; 8]);
        let v = make_f32(&[1, 2, 4], &[0.0; 8]);
        let inputs = [Some(&q), Some(&k), Some(&v), None, None, None, None, None];
        assert!(op_multi_head_attention(&inputs, 2, None).is_err());
    }

    #[test]
    fn mha_rejects_bad_num_heads() {
        let q = make_f32(&[1, 2, 4], &[0.0; 8]);
        let k = make_f32(&[1, 2, 4], &[0.0; 8]);
        let v = make_f32(&[1, 2, 4], &[0.0; 8]);
        let inputs = [Some(&q), Some(&k), Some(&v), None, None, None, None, None];
        // hidden=4 not divisible by num_heads=3
        assert!(op_multi_head_attention(&inputs, 3, None).is_err());
    }

    // ---------- GroupQueryAttention ----------
    #[test]
    fn gqa_basic_no_rope_no_cache() {
        // num_heads=4, kv_num_heads=2, head_dim=2 -> q_hidden=8, kv_hidden=4.
        let q = make_f32(&[1, 1, 8], &[0.0; 8]);
        let k = make_f32(&[1, 1, 4], &[1.0, 0.0, 0.0, 1.0]);
        let v = make_f32(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        let (attn, pk, pv) =
            op_group_query_attention(&inputs, 4, 2, None, -1, false, false).unwrap();
        assert_eq!(attn.shape.dims, vec![1, 1, 8]);
        assert_eq!(pk.shape.dims, vec![1, 2, 1, 2]);
        assert_eq!(pv.shape.dims, vec![1, 2, 1, 2]);
    }

    #[test]
    fn gqa_kv_cache_mutation_across_calls() {
        // First call Sq=2, second call Sq=1, verify second call has
        // present_sk = 3.
        let q1 = make_f32(&[1, 2, 8], &[0.0; 16]);
        let k1 = make_f32(&[1, 2, 4], &[1.0; 8]);
        let v1 = make_f32(&[1, 2, 4], &[2.0; 8]);
        let inputs1 = [
            Some(&q1),
            Some(&k1),
            Some(&v1),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        let (_, pk1, pv1) =
            op_group_query_attention(&inputs1, 4, 2, None, -1, false, false).unwrap();
        assert_eq!(pk1.shape.dims, vec![1, 2, 2, 2]);

        // Second call with Sq=1, past=pk1, pv1.
        let q2 = make_f32(&[1, 1, 8], &[0.0; 8]);
        let k2 = make_f32(&[1, 1, 4], &[3.0; 4]);
        let v2 = make_f32(&[1, 1, 4], &[4.0; 4]);
        let inputs2 = [
            Some(&q2),
            Some(&k2),
            Some(&v2),
            Some(&pk1),
            Some(&pv1),
            None,
            None,
            None,
            None,
        ];
        let (_, pk2, _pv2) =
            op_group_query_attention(&inputs2, 4, 2, None, -1, false, false).unwrap();
        assert_eq!(pk2.shape.dims, vec![1, 2, 3, 2]);
    }

    #[test]
    fn gqa_do_rotary_requires_caches() {
        let q = make_f32(&[1, 1, 8], &[0.0; 8]);
        let k = make_f32(&[1, 1, 4], &[0.0; 4]);
        let v = make_f32(&[1, 1, 4], &[0.0; 4]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        assert!(op_group_query_attention(&inputs, 4, 2, None, -1, true, false).is_err());
    }

    #[test]
    fn gqa_do_rotary_zero_skips_rope() {
        // With do_rotary=0, cos/sin caches are unused even if missing.
        let q = make_f32(&[1, 1, 8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let k = make_f32(&[1, 1, 4], &[1.0, 0.0, 0.0, 1.0]);
        let v = make_f32(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        let (attn, _, _) = op_group_query_attention(&inputs, 4, 2, None, -1, false, false).unwrap();
        assert_eq!(attn.shape.dims, vec![1, 1, 8]);
    }

    #[test]
    fn gqa_rejects_local_window() {
        let q = make_f32(&[1, 1, 8], &[0.0; 8]);
        let k = make_f32(&[1, 1, 4], &[0.0; 4]);
        let v = make_f32(&[1, 1, 4], &[0.0; 4]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        assert!(op_group_query_attention(&inputs, 4, 2, None, 512, false, false).is_err());
    }

    #[test]
    fn gqa_rejects_bad_head_divisibility() {
        let q = make_f32(&[1, 1, 8], &[0.0; 8]);
        let k = make_f32(&[1, 1, 4], &[0.0; 4]);
        let v = make_f32(&[1, 1, 4], &[0.0; 4]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        // num_heads=3, kv_num_heads=2 — not a multiple.
        assert!(op_group_query_attention(&inputs, 3, 2, None, -1, false, false).is_err());
    }

    // ---------- Integration tests (4) ----------
    #[test]
    fn integration_rmsnorm_mha_pipeline() {
        // RMSNorm -> MHA produces a valid shape.
        let x = make_f32(&[1, 2, 4], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let scale = make_f32(&[4], &[1.0, 1.0, 1.0, 1.0]);
        let n_inputs = [Some(&x), Some(&scale)];
        let normalized = op_simplified_layer_normalization(&n_inputs, 1e-5, -1).unwrap();
        // MHA(x=normalized as Q=K=V).
        let mha_inputs = [
            Some(&normalized),
            Some(&normalized),
            Some(&normalized),
            None,
            None,
            None,
            None,
            None,
        ];
        let (attn, _, _) = op_multi_head_attention(&mha_inputs, 2, None).unwrap();
        assert_eq!(attn.shape.dims, vec![1, 2, 4]);
    }

    #[test]
    fn integration_skip_norm_into_mha() {
        let x = make_f32(&[1, 2, 4], &[1.0; 8]);
        let skip = make_f32(&[1, 2, 4], &[0.5; 8]);
        let scale = make_f32(&[4], &[1.0, 1.0, 1.0, 1.0]);
        let n_inputs = [Some(&x), Some(&skip), Some(&scale)];
        let (y, pre) = op_skip_simplified_layer_normalization(&n_inputs, 1e-5).unwrap();
        assert_eq!(y.shape.dims, vec![1, 2, 4]);
        assert_eq!(pre.shape.dims, vec![1, 2, 4]);
        // Feed y into MHA.
        let mha_inputs = [Some(&y), Some(&y), Some(&y), None, None, None, None, None];
        let (attn, _, _) = op_multi_head_attention(&mha_inputs, 2, None).unwrap();
        assert_eq!(attn.shape.dims, vec![1, 2, 4]);
    }

    #[test]
    fn integration_gqa_two_step_generation() {
        // Emulate a two-step generation: prefill (Sq=3) then decode (Sq=1).
        let q1 = make_f32(&[1, 3, 8], &[0.1; 24]);
        let k1 = make_f32(&[1, 3, 4], &[0.2; 12]);
        let v1 = make_f32(&[1, 3, 4], &[0.3; 12]);
        let in1 = [
            Some(&q1),
            Some(&k1),
            Some(&v1),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        let (out1, pk, pv) = op_group_query_attention(&in1, 4, 2, None, -1, false, false).unwrap();
        assert_eq!(out1.shape.dims, vec![1, 3, 8]);
        assert_eq!(pk.shape.dims[2], 3);

        // Decode step.
        let q2 = make_f32(&[1, 1, 8], &[0.4; 8]);
        let k2 = make_f32(&[1, 1, 4], &[0.5; 4]);
        let v2 = make_f32(&[1, 1, 4], &[0.6; 4]);
        let in2 = [
            Some(&q2),
            Some(&k2),
            Some(&v2),
            Some(&pk),
            Some(&pv),
            None,
            None,
            None,
            None,
        ];
        let (out2, pk2, _) = op_group_query_attention(&in2, 4, 2, None, -1, false, false).unwrap();
        assert_eq!(out2.shape.dims, vec![1, 1, 8]);
        // Past 3 + new 1 = 4
        assert_eq!(pk2.shape.dims[2], 4);
    }

    // ---------- Additional coverage tests ----------
    #[test]
    fn domain_round_trip() {
        use crate::operators::{Domain, OpKind};
        assert_eq!(Domain::parse_str(""), Some(Domain::StandardOnnx));
        assert_eq!(Domain::parse_str("ai.onnx"), Some(Domain::StandardOnnx));
        assert_eq!(
            Domain::parse_str("com.microsoft"),
            Some(Domain::MicrosoftFused)
        );
        assert_eq!(Domain::parse_str("com.other"), None);
        assert_eq!(OpKind::Add.domain(), Domain::StandardOnnx);
        assert_eq!(OpKind::MultiHeadAttention.domain(), Domain::MicrosoftFused);
    }

    #[test]
    fn domain_aware_lookup() {
        use crate::operators::OpKind;
        // Microsoft-domain name in the standard domain is unknown.
        assert_eq!(
            OpKind::lookup_by_domain_and_name("", "MultiHeadAttention"),
            None
        );
        assert_eq!(
            OpKind::lookup_by_domain_and_name("com.microsoft", "MultiHeadAttention"),
            Some(OpKind::MultiHeadAttention)
        );
        assert_eq!(
            OpKind::lookup_by_domain_and_name("", "Add"),
            Some(OpKind::Add)
        );
        assert_eq!(
            OpKind::lookup_by_domain_and_name("com.microsoft", "Add"),
            None
        );
    }

    #[test]
    fn op_kind_round_trip_microsoft() {
        use crate::operators::OpKind;
        for op in [
            OpKind::SimplifiedLayerNormalization,
            OpKind::SkipSimplifiedLayerNormalization,
            OpKind::RotaryEmbedding,
            OpKind::MultiHeadAttention,
            OpKind::GroupQueryAttention,
        ] {
            let name = op.name();
            assert_eq!(OpKind::parse_str(name), Some(op));
        }
    }

    #[test]
    fn rope_interleaved_position_zero_identity() {
        let (cos_c, sin_c) = make_rope_caches(4, 2);
        let input = make_f32(&[1, 1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let pos = make_i64(&[1, 1], &[0]);
        let inputs = [Some(&input), Some(&pos), Some(&cos_c), Some(&sin_c)];
        let out = op_rotary_embedding(&inputs, true, 0, 1).unwrap();
        let v = read_all_f32(&out);
        for (i, expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!((v[i] - *expected).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_int32_positions() {
        // Same as rope_position_zero_is_identity but with int32 positions.
        let (cos_c, sin_c) = make_rope_caches(4, 2);
        let input = make_f32(&[1, 1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let mut pos_bytes = vec![0u8; 4];
        crate::byte_io::write_i32(&mut pos_bytes, 0, 0);
        let pos = Tensor {
            data_type: DataType::Int32,
            shape: TensorShape::new(vec![1, 1]),
            name: String::new(),
            raw_data: pos_bytes,
        };
        let inputs = [Some(&input), Some(&pos), Some(&cos_c), Some(&sin_c)];
        let out = op_rotary_embedding(&inputs, false, 0, 1).unwrap();
        let v = read_all_f32(&out);
        for (i, expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!((v[i] - *expected).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_rank3_reshape_path() {
        // rank-3 input (B, Sq, hidden) with num_heads=2.
        let (cos_c, sin_c) = make_rope_caches(4, 1);
        let input = make_f32(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let pos = make_i64(&[1, 1], &[0]);
        let inputs = [Some(&input), Some(&pos), Some(&cos_c), Some(&sin_c)];
        let out = op_rotary_embedding(&inputs, false, 0, 2).unwrap();
        assert_eq!(out.shape.dims, vec![1, 1, 4]);
    }

    #[test]
    fn rope_rejects_bad_num_heads_for_rank3() {
        let (cos_c, sin_c) = make_rope_caches(2, 2);
        let input = make_f32(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let pos = make_i64(&[1, 1], &[0]);
        let inputs = [Some(&input), Some(&pos), Some(&cos_c), Some(&sin_c)];
        // num_heads=0 is invalid for rank-3 input.
        assert!(op_rotary_embedding(&inputs, false, 0, 0).is_err());
    }

    #[test]
    fn rope_rejects_missing_inputs() {
        let input = make_f32(&[1, 1, 1, 4], &[0.0; 4]);
        let inputs = [Some(&input)];
        assert!(op_rotary_embedding(&inputs, false, 0, 1).is_err());
    }

    #[test]
    fn sdpa_empty_keys_returns_zeros() {
        let q = make_f32(&[1, 1, 1, 2], &[1.0, 2.0]);
        let out =
            scaled_dot_product_attention(&q.raw_data, &[], &[], 1, 1, 1, 1, 0, 2, 0, 1.0, false)
                .unwrap();
        assert_eq!(read_f32(&out, 0), 0.0);
        assert_eq!(read_f32(&out, 1), 0.0);
    }

    #[test]
    fn sdpa_causal_first_query_sees_only_first_key() {
        // Sq=1, sk=4, causal, past_sk=0 -> only k=0 visible.
        let q = make_f32(&[1, 1, 1, 1], &[1.0]);
        let k = make_f32(&[1, 1, 4, 1], &[10.0, 20.0, 30.0, 40.0]);
        let v = make_f32(&[1, 1, 4, 1], &[1.0, 2.0, 3.0, 4.0]);
        let out = scaled_dot_product_attention(
            &q.raw_data,
            &k.raw_data,
            &v.raw_data,
            1,
            1,
            1,
            1,
            4,
            1,
            0,
            1.0,
            true,
        )
        .unwrap();
        // Only k=0 is unmasked, so result = v[0] = 1.0.
        assert!((read_f32(&out, 0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sdpa_past_sk_shifts_causal_window() {
        // past_sk=2 means first query slot has absolute position 2,
        // so it attends over k=0..=2 (three keys).
        let q = make_f32(&[1, 1, 1, 1], &[1.0]);
        let k = make_f32(&[1, 1, 4, 1], &[1.0, 1.0, 1.0, 1.0]);
        let v = make_f32(&[1, 1, 4, 1], &[1.0, 2.0, 3.0, 4.0]);
        let out = scaled_dot_product_attention(
            &q.raw_data,
            &k.raw_data,
            &v.raw_data,
            1,
            1,
            1,
            1,
            4,
            1,
            2,
            1.0,
            true,
        )
        .unwrap();
        // Uniform softmax over 3 unmasked keys: (1+2+3)/3 = 2.0.
        assert!((read_f32(&out, 0) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn mha_missing_past_key_rejects_only_one() {
        let q = make_f32(&[1, 1, 4], &[0.0; 4]);
        let k = make_f32(&[1, 1, 4], &[0.0; 4]);
        let v = make_f32(&[1, 1, 4], &[0.0; 4]);
        let past_k = make_f32(&[1, 2, 1, 2], &[0.0; 4]);
        // past_v missing — kv_concat past_sk mismatch (0 vs 1).
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            Some(&past_k),
            None,
        ];
        assert!(op_multi_head_attention(&inputs, 2, None).is_err());
    }

    #[test]
    fn mha_explicit_scale() {
        let q = make_f32(&[1, 1, 2], &[1.0, 0.0]);
        let k = make_f32(&[1, 1, 2], &[1.0, 0.0]);
        let v = make_f32(&[1, 1, 2], &[1.0, 2.0]);
        let inputs = [Some(&q), Some(&k), Some(&v), None, None, None, None, None];
        let (attn, _, _) = op_multi_head_attention(&inputs, 1, Some(0.5)).unwrap();
        assert_eq!(attn.shape.dims, vec![1, 1, 2]);
    }

    #[test]
    fn gqa_past_cache_rank_rejected() {
        let q = make_f32(&[1, 1, 8], &[0.0; 8]);
        let k = make_f32(&[1, 1, 4], &[0.0; 4]);
        let v = make_f32(&[1, 1, 4], &[0.0; 4]);
        // rank-3 past_key is invalid (must be rank-4).
        let bad_past = make_f32(&[1, 2, 2], &[0.0; 4]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            Some(&bad_past),
            None,
            None,
            None,
            None,
            None,
        ];
        assert!(op_group_query_attention(&inputs, 4, 2, None, -1, false, false).is_err());
    }

    #[test]
    fn gqa_kv_shape_mismatch() {
        let q = make_f32(&[1, 1, 8], &[0.0; 8]);
        let k = make_f32(&[1, 1, 4], &[0.0; 4]);
        let v = make_f32(&[1, 1, 6], &[0.0; 6]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        assert!(op_group_query_attention(&inputs, 4, 2, None, -1, false, false).is_err());
    }

    #[test]
    fn gqa_head_dim_mismatch_between_q_and_kv() {
        // q_hidden=8 / 4 = 2; kv_hidden=8 / 2 = 4. Mismatch.
        let q = make_f32(&[1, 1, 8], &[0.0; 8]);
        let k = make_f32(&[1, 1, 8], &[0.0; 8]);
        let v = make_f32(&[1, 1, 8], &[0.0; 8]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        assert!(op_group_query_attention(&inputs, 4, 2, None, -1, false, false).is_err());
    }

    #[test]
    fn gqa_rejects_zero_num_heads() {
        let q = make_f32(&[1, 1, 8], &[0.0; 8]);
        let k = make_f32(&[1, 1, 4], &[0.0; 4]);
        let v = make_f32(&[1, 1, 4], &[0.0; 4]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        assert!(op_group_query_attention(&inputs, 0, 2, None, -1, false, false).is_err());
        assert!(op_group_query_attention(&inputs, 4, 0, None, -1, false, false).is_err());
    }

    #[test]
    fn gqa_with_rope_applies_rotation() {
        // Prove the RoPE path runs and produces present_key values that
        // differ from the raw (non-rotated) K. Use head_dim=4 (half=2) so
        // both pair indices participate and the rotation is non-trivial
        // even at position 0.
        let q_raw: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.05).collect();
        let k_raw: Vec<f32> = (0..8).map(|i| (i as f32) * 0.2 + 0.1).collect();
        let v_raw: Vec<f32> = (0..8).map(|i| (i as f32) * 0.3).collect();
        let q = make_f32(&[1, 1, 16], &q_raw);
        let k = make_f32(&[1, 1, 8], &k_raw);
        let v = make_f32(&[1, 1, 8], &v_raw);
        // past_sk=1 so positions are [1] (row 1 of cache, non-identity).
        let past_k = make_f32(&[1, 2, 1, 4], &[0.0; 8]);
        let past_v = make_f32(&[1, 2, 1, 4], &[0.0; 8]);
        let (cos_c, sin_c) = make_rope_caches(8, 2);
        let in_with = [
            Some(&q),
            Some(&k),
            Some(&v),
            Some(&past_k),
            Some(&past_v),
            None,
            None,
            Some(&cos_c),
            Some(&sin_c),
        ];
        // num_heads=4, kv_num_heads=2, q_hidden=16 -> head_dim=4
        let (_, pk_rot, _) =
            op_group_query_attention(&in_with, 4, 2, None, -1, true, false).unwrap();
        let pk_vals = read_all_f32(&pk_rot);
        // present_key has shape (1, 2, 2, 4) = 16 elements. The first 4
        // (head 0 past) are zeros; the first 4 of head 0's new slot (idx
        // 4..8) are the rotated K values and must differ from the raw K.
        let raw_head0: [f32; 4] = [k_raw[0], k_raw[1], k_raw[2], k_raw[3]];
        let rotated: [f32; 4] = [pk_vals[4], pk_vals[5], pk_vals[6], pk_vals[7]];
        let differs = raw_head0
            .iter()
            .zip(rotated.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "RoPE should rotate the new key past position 0");
    }

    #[test]
    fn gqa_present_key_has_correct_shape_with_cache() {
        let q = make_f32(&[1, 2, 8], &[0.0; 16]);
        let k = make_f32(&[1, 2, 4], &[0.0; 8]);
        let v = make_f32(&[1, 2, 4], &[0.0; 8]);
        // past: (B=1, kv_heads=2, past_sk=3, head_dim=2).
        let past_k = make_f32(&[1, 2, 3, 2], &[0.0; 12]);
        let past_v = make_f32(&[1, 2, 3, 2], &[0.0; 12]);
        let inputs = [
            Some(&q),
            Some(&k),
            Some(&v),
            Some(&past_k),
            Some(&past_v),
            None,
            None,
            None,
            None,
        ];
        let (_, pk, pv) = op_group_query_attention(&inputs, 4, 2, None, -1, false, false).unwrap();
        // total_sk = 3 + 2 = 5
        assert_eq!(pk.shape.dims, vec![1, 2, 5, 2]);
        assert_eq!(pv.shape.dims, vec![1, 2, 5, 2]);
    }

    #[test]
    fn simplified_layer_norm_rank3_input() {
        let x = make_f32(
            &[2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let scale = make_f32(&[3], &[1.0, 1.0, 1.0]);
        let inputs = [Some(&x), Some(&scale)];
        let got = op_simplified_layer_normalization(&inputs, 1e-5, -1).unwrap();
        assert_eq!(got.shape.dims, vec![2, 2, 3]);
    }

    #[test]
    fn apply_rope_rejects_odd_rotary_dim() {
        let input = make_f32(&[1, 1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let (cos_c, sin_c) = make_rope_caches(2, 2);
        let mut raw = input.raw_data.clone();
        let r = apply_rope_in_place(&mut raw, &input.shape.dims, &cos_c, &sin_c, &[0], false, 3);
        assert!(r.is_err(), "rotary_dim=3 (odd) must be rejected");
    }

    #[test]
    fn skip_norm_rank3_input() {
        let x = make_f32(&[1, 2, 4], &[1.0; 8]);
        let skip = make_f32(&[1, 2, 4], &[2.0; 8]);
        let scale = make_f32(&[4], &[1.0, 1.0, 1.0, 1.0]);
        let inputs = [Some(&x), Some(&skip), Some(&scale)];
        let (y, pre) = op_skip_simplified_layer_normalization(&inputs, 1e-5).unwrap();
        assert_eq!(y.shape.dims, vec![1, 2, 4]);
        assert_eq!(pre.shape.dims, vec![1, 2, 4]);
        // Every pre element should be 3.0.
        for v in read_all_f32(&pre) {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn integration_rotary_then_mha() {
        // Apply RoPE to Q/K, then feed into MHA.
        // num_heads=2, head_dim=2 -> half=1, so cache cols = 1.
        let (cos_c, sin_c) = make_rope_caches(8, 1);
        let q_pre = make_f32(&[1, 2, 4], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        // Positions [0, 1] (non-interleaved, num_heads=2)
        let mut pos_bytes = vec![0u8; 16];
        write_i64(&mut pos_bytes, 0, 0);
        write_i64(&mut pos_bytes, 1, 1);
        let pos = Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(vec![1, 2]),
            name: String::new(),
            raw_data: pos_bytes,
        };
        let rope_inputs = [Some(&q_pre), Some(&pos), Some(&cos_c), Some(&sin_c)];
        let q_rot = op_rotary_embedding(&rope_inputs, false, 0, 2).unwrap();
        let k_rot = q_rot.clone();
        let v = make_f32(&[1, 2, 4], &[1.0; 8]);
        let mha_inputs = [
            Some(&q_rot),
            Some(&k_rot),
            Some(&v),
            None,
            None,
            None,
            None,
            None,
        ];
        let (attn, _, _) = op_multi_head_attention(&mha_inputs, 2, None).unwrap();
        assert_eq!(attn.shape.dims, vec![1, 2, 4]);
    }
}
