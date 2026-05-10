// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Phase 2 generative / normalization / sampler operators.
//!
//! This module lands as part of the `generative-llm-v1` change and provides
//! the 19 non-control-flow operators required to run generative LLMs end-to-end:
//!
//! - RMSNormalization, LpNormalization, MeanVarianceNormalization
//! - MatMulInteger, DynamicQuantizeLinear
//! - RandomNormal / RandomNormalLike / RandomUniform / RandomUniformLike
//! - Multinomial, Bernoulli, Dropout, EyeLike
//! - ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceSumSquare
//! - Softplus
//!
//! Random operators use a deterministic xorshift32 PRNG seeded from the
//! `seed` attribute (float) and `shape` attribute (ints) so repeated calls
//! with the same seed produce reproducible results. This is critical for
//! DO-178C DAL A certification — "random" is a misnomer; every draw must be
//! deterministically replayable from (seed, call-site).
//!
//! The real int8 GEMM kernel used by `op_mat_mul_integer` also powers the
//! `op_qlinear_matmul` re-implementation in `ops::quantized` (see `gemm_i8`
//! in this module).

#![allow(clippy::needless_range_loop)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32, write_i64};
use crate::operators::OpError;
use crate::ops::common::{next_coord, require_float};
use crate::tensor::{bf16_to_f32, f32_to_bf16, DataType, Tensor, TensorShape};

// ---------------------------------------------------------------------------
// Deterministic xorshift32 PRNG (for the random-sampling ops)
// ---------------------------------------------------------------------------

/// Deterministic 32-bit xorshift PRNG.
///
/// Used by every Phase 2 random operator so that model runs are
/// reproducible: the same `(seed, attribute)` combination always yields the
/// same output. `no_std`-friendly, zero-dependency, `Copy`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    /// Seed the PRNG. Zero seeds are remapped to a non-zero constant so
    /// the xorshift recurrence doesn't collapse to all zeros.
    pub(crate) fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9 } else { seed },
        }
    }

    /// Returns the next uniformly-distributed u32.
    pub(crate) fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Returns a uniform f32 in `[0, 1)`.
    pub(crate) fn next_f32_unit(&mut self) -> f32 {
        // Use the top 24 bits (f32 mantissa width) for maximal precision.
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Returns an f32 sample from a standard normal distribution using the
    /// Box-Muller transform.
    pub(crate) fn next_f32_normal(&mut self) -> f32 {
        // Draw two uniforms in (0, 1] (reject exact zero to avoid log(0)).
        let mut u1 = self.next_f32_unit();
        if u1 <= 1e-7 {
            u1 = 1e-7;
        }
        let u2 = self.next_f32_unit();
        let r = libm_sqrt(-2.0 * libm_ln(u1));
        let theta = 2.0 * core::f32::consts::PI * u2;
        r * libm_cos(theta)
    }
}

// ---------------------------------------------------------------------------
// Minimal no_std math helpers
// ---------------------------------------------------------------------------

/// `sqrtf` via Newton-Raphson.
#[inline]
fn libm_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // Newton's method with a reasonable seed from bit twiddling.
    let mut guess = x;
    for _ in 0..16 {
        guess = 0.5 * (guess + x / guess);
    }
    guess
}

/// Natural log approximation using the identity `ln(x) = 2 * atanh((x-1)/(x+1))`
/// after range reduction via the IEEE exponent.
#[inline]
fn libm_ln(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa_bits = (bits & 0x007f_ffff) | 0x3f80_0000;
    let m = f32::from_bits(mantissa_bits); // m in [1.0, 2.0)
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    // Sum of odd powers: 2*(y + y^3/3 + y^5/5 + y^7/7 + y^9/9).
    let series = y * (1.0 + y2 * (1.0 / 3.0 + y2 * (1.0 / 5.0 + y2 * (1.0 / 7.0 + y2 / 9.0))));
    2.0 * series + (exp as f32) * core::f32::consts::LN_2
}

/// Cosine via Taylor polynomial after range reduction to `[-pi, pi]`.
#[inline]
fn libm_cos(mut x: f32) -> f32 {
    let two_pi = 2.0 * core::f32::consts::PI;
    // Range reduce to [-pi, pi].
    while x > core::f32::consts::PI {
        x -= two_pi;
    }
    while x < -core::f32::consts::PI {
        x += two_pi;
    }
    let x2 = x * x;
    // Degree-8 Taylor polynomial: 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8!.
    1.0 - x2 / 2.0 + x2 * x2 / 24.0 - x2 * x2 * x2 / 720.0 + x2 * x2 * x2 * x2 / 40320.0
}

/// Softplus `ln(1 + e^x)` with numerical guard against overflow.
#[inline]
fn softplus_scalar(x: f32) -> f32 {
    // For large positive x, ln(1+e^x) ~= x; for large negative, ~= e^x.
    if x > 20.0 {
        x
    } else if x < -20.0 {
        // e^x is tiny; ln(1 + e^x) ~= e^x ~= 0.
        0.0
    } else {
        libm_ln(1.0 + crate::operators::expf_approx(x))
    }
}

// ---------------------------------------------------------------------------
// Softplus
// ---------------------------------------------------------------------------

/// Element-wise softplus: `ln(1 + e^x)`.
pub fn op_softplus(input: &Tensor) -> Result<Tensor, OpError> {
    require_float(input, "Softplus")?;
    let n = input.shape.total_elements();
    let mut data = allocate_tensor_data(n, DataType::Float);
    for i in 0..n {
        write_f32(&mut data, i, softplus_scalar(read_f32(&input.raw_data, i)));
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

// ---------------------------------------------------------------------------
// EyeLike
// ---------------------------------------------------------------------------

/// Identity-like output: a 2D tensor matching the input's shape with 1s on
/// a shifted diagonal (controlled by `k`). Element type is Float (the `dtype`
/// attribute is currently fixed to f32 in SmallAIOS).
pub fn op_eye_like(input: &Tensor, k: i64) -> Result<Tensor, OpError> {
    if input.shape.ndim() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "EyeLike requires a 2D input",
        )));
    }
    let rows = input.shape.dims[0];
    let cols = input.shape.dims[1];
    let n = (rows as usize) * (cols as usize);
    let mut data = allocate_tensor_data(n, DataType::Float);
    for r in 0..rows {
        let c = r + k;
        if c >= 0 && c < cols {
            let idx = (r as usize) * (cols as usize) + c as usize;
            write_f32(&mut data, idx, 1.0);
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

// ---------------------------------------------------------------------------
// Dropout (inference pass-through)
// ---------------------------------------------------------------------------

/// Dropout in inference mode (`training_mode = false`, the ONNX default)
/// is the identity function. SmallAIOS is an inference-only runtime so
/// training-mode dropout is intentionally unsupported.
pub fn op_dropout(input: &Tensor) -> Result<Tensor, OpError> {
    require_float(input, "Dropout")?;
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: input.raw_data.clone(),
    })
}

// ---------------------------------------------------------------------------
// Random ops
// ---------------------------------------------------------------------------

/// Fill a tensor of the given shape with standard-normal samples seeded
/// from `seed` (f32 attribute). Shape is provided via the `shape` ints attr.
pub fn op_random_normal(
    shape: &[i64],
    seed: f32,
    mean: f32,
    scale: f32,
) -> Result<Tensor, OpError> {
    random_fill(shape, seed, |rng| mean + scale * rng.next_f32_normal())
}

/// `RandomNormalLike`: same as `RandomNormal` but takes shape from an input
/// tensor rather than an attribute.
pub fn op_random_normal_like(
    input: &Tensor,
    seed: f32,
    mean: f32,
    scale: f32,
) -> Result<Tensor, OpError> {
    op_random_normal(&input.shape.dims, seed, mean, scale)
}

/// Fill a tensor with uniform samples in `[low, high)`.
pub fn op_random_uniform(shape: &[i64], seed: f32, low: f32, high: f32) -> Result<Tensor, OpError> {
    if high <= low {
        return Err(OpError::InvalidAttribute(String::from(
            "RandomUniform: high must be > low",
        )));
    }
    let range = high - low;
    random_fill(shape, seed, |rng| low + range * rng.next_f32_unit())
}

/// `RandomUniformLike`: shape derived from input.
pub fn op_random_uniform_like(
    input: &Tensor,
    seed: f32,
    low: f32,
    high: f32,
) -> Result<Tensor, OpError> {
    op_random_uniform(&input.shape.dims, seed, low, high)
}

/// Shared helper: allocate an f32 tensor of the given shape and fill it
/// from a closure that pulls draws from a seeded [`XorShift32`].
fn random_fill<F: FnMut(&mut XorShift32) -> f32>(
    shape: &[i64],
    seed: f32,
    mut gen: F,
) -> Result<Tensor, OpError> {
    if shape.iter().any(|&d| d < 0) {
        return Err(OpError::InvalidAttribute(String::from(
            "random: negative dim",
        )));
    }
    let total: usize = shape.iter().map(|&d| d as usize).product();
    let mut data = allocate_tensor_data(total, DataType::Float);
    let mut rng = XorShift32::new(seeded_rng_state(seed, 0x1234_5678, 0xA5A5_A5A5));
    for i in 0..total {
        write_f32(&mut data, i, gen(&mut rng));
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: data,
    })
}

/// Derives a 32-bit seed: if the ONNX seed attribute is exactly 0.0 we
/// fall back to `default_seed` (each random op uses a distinct constant);
/// otherwise we XOR the raw bits of the seed with `xor_mask` so different
/// ops with the same input seed produce independent streams.
fn seeded_rng_state(seed: f32, default_seed: u32, xor_mask: u32) -> u32 {
    if seed == 0.0 {
        default_seed
    } else {
        seed.to_bits() ^ xor_mask
    }
}

/// Bernoulli op: draws an i64 sample in {0, 1} per element where the input
/// tensor encodes per-element success probability in [0, 1]. Output dtype is
/// Float (ONNX permits several output dtypes; we pick Float for consistency
/// with the rest of the Phase 2 surface).
pub fn op_bernoulli(input: &Tensor, seed: f32) -> Result<Tensor, OpError> {
    require_float(input, "Bernoulli")?;
    let n = input.shape.total_elements();
    let mut data = allocate_tensor_data(n, DataType::Float);
    let mut rng = XorShift32::new(seeded_rng_state(seed, 0xDEAD_BEEF, 0x5A5A_5A5A));
    for i in 0..n {
        let p = read_f32(&input.raw_data, i).clamp(0.0, 1.0);
        let u = rng.next_f32_unit();
        write_f32(&mut data, i, if u < p { 1.0 } else { 0.0 });
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

/// Multinomial op: interprets `input` as `[batch, classes]` unnormalized
/// log-probabilities (logits) and draws `sample_size` categorical samples per
/// batch row. Output shape is `[batch, sample_size]`, dtype Int64.
///
/// This implementation assumes logits (ONNX convention) and uses the
/// Gumbel-max trick for single-sample stability.
pub fn op_multinomial(input: &Tensor, sample_size: i64, seed: f32) -> Result<Tensor, OpError> {
    require_float(input, "Multinomial")?;
    if input.shape.ndim() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "Multinomial requires 2D [batch, classes] input",
        )));
    }
    if sample_size <= 0 {
        return Err(OpError::InvalidAttribute(String::from(
            "Multinomial: sample_size must be positive",
        )));
    }
    let batch = input.shape.dims[0] as usize;
    let classes = input.shape.dims[1] as usize;
    let samples = sample_size as usize;
    let out_total = batch * samples;
    let mut data = allocate_tensor_data(out_total, DataType::Int64);
    let mut rng = XorShift32::new(seeded_rng_state(seed, 0xCAFE_BABE, 0x3333_3333));

    for b in 0..batch {
        for s in 0..samples {
            let pick = gumbel_max_argmax(&input.raw_data, b * classes, classes, &mut rng);
            write_i64(&mut data, b * samples + s, pick);
        }
    }
    Ok(Tensor {
        data_type: DataType::Int64,
        shape: TensorShape::new(alloc::vec![batch as i64, samples as i64]),
        name: String::new(),
        raw_data: data,
    })
}

/// Picks a class index via Gumbel-max sampling over `classes` logits
/// stored contiguously at `raw[row_off..row_off + classes]`.
fn gumbel_max_argmax(raw: &[u8], row_off: usize, classes: usize, rng: &mut XorShift32) -> i64 {
    let mut best = i64::MIN;
    let mut best_score = f32::NEG_INFINITY;
    for c in 0..classes {
        let logit = read_f32(raw, row_off + c);
        let mut u = rng.next_f32_unit();
        if u <= 1e-7 {
            u = 1e-7;
        }
        // Gumbel noise: -ln(-ln(u))
        let gumbel = -libm_ln(-libm_ln(u));
        let score = logit + gumbel;
        if score > best_score {
            best_score = score;
            best = c as i64;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// RMSNormalization / LpNormalization / MeanVarianceNormalization
// ---------------------------------------------------------------------------

/// BFloat16 implementation of RMSNormalization. Converts inputs from BF16
/// to `f32`, runs the full-precision normalization, and writes BF16 outputs.
fn rms_normalization_bf16(
    input: &Tensor,
    scale: &Tensor,
    axis: i64,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    let scale_f32 = widen_rms_scale_to_f32(scale)?;
    let axis = resolve_rms_axis(input, axis)?;
    let (outer, norm_size, total) = rms_layout_sizes(input, axis)?;

    let scale_n = scale_f32.len();
    if scale_n != 1 && scale_n != norm_size {
        return Err(OpError::ShapeMismatch(String::from(
            "RMSNormalization: scale must match normalized region",
        )));
    }

    let input_f32 = bf16_to_f32(&input.raw_data);
    if input_f32.len() < total {
        return Err(OpError::ShapeMismatch(String::from(
            "RMSNormalization: BF16 input buffer smaller than shape",
        )));
    }

    let out_f32 = apply_rms_norm(&input_f32, &scale_f32, outer, norm_size, epsilon);
    Ok(Tensor {
        data_type: DataType::BFloat16,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: f32_to_bf16(&out_f32),
    })
}

/// Widens an RMSNormalization scale tensor (BF16 or Float) to a flat
/// `Vec<f32>`. Errors on unsupported dtypes.
fn widen_rms_scale_to_f32(scale: &Tensor) -> Result<Vec<f32>, OpError> {
    match scale.data_type {
        DataType::BFloat16 => Ok(bf16_to_f32(&scale.raw_data)),
        DataType::Float => Ok((0..scale.shape.total_elements())
            .map(|i| read_f32(&scale.raw_data, i))
            .collect()),
        _ => Err(OpError::InternalError(alloc::format!(
            "RMSNormalization does not support {} scale",
            scale.data_type
        ))),
    }
}

/// Normalizes a potentially-negative axis into the valid `[0, ndim)` range
/// for the given input tensor.
fn resolve_rms_axis(input: &Tensor, axis: i64) -> Result<usize, OpError> {
    let ndim = input.shape.ndim() as i64;
    if ndim == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "RMSNormalization: input must have at least 1 dim",
        )));
    }
    let axis = if axis < 0 { ndim + axis } else { axis };
    if axis < 0 || axis >= ndim {
        return Err(OpError::InvalidAttribute(String::from(
            "RMSNormalization: axis out of range",
        )));
    }
    Ok(axis as usize)
}

/// Returns `(outer, norm_size, total)` for the RMSNorm row layout.
fn rms_layout_sizes(input: &Tensor, axis: usize) -> Result<(usize, usize, usize), OpError> {
    let dims = &input.shape.dims;
    let norm_size: usize = dims[axis..].iter().map(|&d| d as usize).product();
    if norm_size == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "RMSNormalization: normalized region is empty",
        )));
    }
    let outer: usize = dims[..axis]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    Ok((outer, norm_size, outer * norm_size))
}

/// Core RMSNorm transform: for each of `outer` rows of length `norm_size`,
/// computes `x / sqrt(mean(x^2) + eps) * scale`.
fn apply_rms_norm(
    input_f32: &[f32],
    scale_f32: &[f32],
    outer: usize,
    norm_size: usize,
    epsilon: f32,
) -> Vec<f32> {
    let scale_n = scale_f32.len();
    let mut out_f32 = alloc::vec![0.0f32; outer * norm_size];
    for o in 0..outer {
        let base = o * norm_size;
        let mut ss = 0.0f32;
        for i in 0..norm_size {
            let v = input_f32[base + i];
            ss += v * v;
        }
        let rms_inv = 1.0 / libm_sqrt(ss / norm_size as f32 + epsilon);
        for i in 0..norm_size {
            let v = input_f32[base + i];
            let s = if scale_n == 1 {
                scale_f32[0]
            } else {
                scale_f32[i]
            };
            out_f32[base + i] = v * rms_inv * s;
        }
    }
    out_f32
}

/// RMS normalization (LLaMA/Mistral): `y = x / sqrt(mean(x^2) + eps) * scale`
/// over the last `axis`-many dimensions (ONNX opset 23 default: normalize
/// the single innermost axis).
pub fn op_rms_normalization(
    input: &Tensor,
    scale: &Tensor,
    axis: i64,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    // Accept Float or BFloat16. For BF16, convert -> compute in f32 ->
    // convert back so the output tensor keeps the same dtype as the input.
    if input.data_type == DataType::BFloat16 {
        return rms_normalization_bf16(input, scale, axis, epsilon);
    }
    require_float(input, "RMSNormalization")?;
    require_float(scale, "RMSNormalization")?;
    let ndim = input.shape.ndim() as i64;
    if ndim == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "RMSNormalization: input must have at least 1 dim",
        )));
    }
    let axis = if axis < 0 { ndim + axis } else { axis };
    if axis < 0 || axis >= ndim {
        return Err(OpError::InvalidAttribute(String::from(
            "RMSNormalization: axis out of range",
        )));
    }
    let axis = axis as usize;
    let dims = &input.shape.dims;
    // Number of contiguous elements to normalize over (dims[axis..]).
    let norm_size: usize = dims[axis..].iter().map(|&d| d as usize).product();
    if norm_size == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "RMSNormalization: normalized region is empty",
        )));
    }
    let outer: usize = dims[..axis]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let total = outer * norm_size;
    // Scale must be broadcastable to the normalized region. Require that
    // its total element count is either 1 or equal to norm_size.
    let scale_n = scale.shape.total_elements();
    if scale_n != 1 && scale_n != norm_size {
        return Err(OpError::ShapeMismatch(String::from(
            "RMSNormalization: scale must match normalized region",
        )));
    }
    let mut data = allocate_tensor_data(total, DataType::Float);
    for o in 0..outer {
        let base = o * norm_size;
        let mut ss = 0.0f32;
        for i in 0..norm_size {
            let v = read_f32(&input.raw_data, base + i);
            ss += v * v;
        }
        let rms_inv = 1.0 / libm_sqrt(ss / norm_size as f32 + epsilon);
        for i in 0..norm_size {
            let v = read_f32(&input.raw_data, base + i);
            let s = if scale_n == 1 {
                read_f32(&scale.raw_data, 0)
            } else {
                read_f32(&scale.raw_data, i)
            };
            write_f32(&mut data, base + i, v * rms_inv * s);
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

/// Lp normalization along a single axis. `p == 1` gives L1, `p == 2` gives L2.
pub fn op_lp_normalization(input: &Tensor, axis: i64, p: i64) -> Result<Tensor, OpError> {
    require_float(input, "LpNormalization")?;
    if p != 1 && p != 2 {
        return Err(OpError::InvalidAttribute(String::from(
            "LpNormalization: p must be 1 or 2",
        )));
    }
    let ndim = input.shape.ndim() as i64;
    let axis = if axis < 0 { ndim + axis } else { axis };
    if axis < 0 || axis >= ndim {
        return Err(OpError::InvalidAttribute(String::from(
            "LpNormalization: axis out of range",
        )));
    }
    let axis = axis as usize;
    let dims = &input.shape.dims;
    let axis_size = dims[axis] as usize;
    let outer: usize = dims[..axis]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let inner: usize = dims[axis + 1..]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let total = outer * axis_size * inner;
    let mut data = allocate_tensor_data(total, DataType::Float);
    for o in 0..outer {
        for n in 0..inner {
            let mut norm = 0.0f32;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + n;
                let v = read_f32(&input.raw_data, idx);
                norm += if p == 1 { abs32(v) } else { v * v };
            }
            let denom = if p == 1 { norm } else { libm_sqrt(norm) };
            let inv = if denom > 1e-12 { 1.0 / denom } else { 0.0 };
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + n;
                let v = read_f32(&input.raw_data, idx);
                write_f32(&mut data, idx, v * inv);
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

#[inline]
fn abs32(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

#[inline]
fn round_f32(x: f32) -> f32 {
    // `no_std` round-half-away-from-zero without pulling in libm.
    if x >= 0.0 {
        (x + 0.5) as i64 as f32
    } else {
        (x - 0.5) as i64 as f32
    }
}

/// Mean-variance normalization: subtract the mean and divide by std over
/// the listed `axes` (or all axes if empty).
pub fn op_mean_variance_normalization(input: &Tensor, axes: &[i64]) -> Result<Tensor, OpError> {
    require_float(input, "MeanVarianceNormalization")?;
    let ndim = input.shape.ndim();
    let n = input.shape.total_elements();
    if n == 0 {
        return Ok(input.clone());
    }
    // Resolve axes: empty means all.
    let resolved: Vec<usize> = if axes.is_empty() {
        (0..ndim).collect()
    } else {
        let nd = ndim as i64;
        axes.iter()
            .map(|&a| {
                let r = if a < 0 { nd + a } else { a };
                if r < 0 || r >= nd {
                    Err(OpError::InvalidAttribute(String::from(
                        "MeanVarianceNormalization: axis out of range",
                    )))
                } else {
                    Ok(r as usize)
                }
            })
            .collect::<Result<_, _>>()?
    };
    // Compute grouped mean + variance; fall back to global when every axis
    // is included. For simplicity we handle the common "normalize across
    // the entire tensor" case exactly and reject the more exotic grouped
    // case with an InvalidAttribute error.
    let is_full = resolved.len() == ndim;
    if !is_full {
        return Err(OpError::InvalidAttribute(String::from(
            "MeanVarianceNormalization: partial-axis reduction not yet supported",
        )));
    }
    let mut mean = 0.0f32;
    for i in 0..n {
        mean += read_f32(&input.raw_data, i);
    }
    mean /= n as f32;
    let mut var = 0.0f32;
    for i in 0..n {
        let d = read_f32(&input.raw_data, i) - mean;
        var += d * d;
    }
    var /= n as f32;
    let inv = 1.0 / libm_sqrt(var + 1e-9);
    let mut data = allocate_tensor_data(n, DataType::Float);
    for i in 0..n {
        let v = read_f32(&input.raw_data, i);
        write_f32(&mut data, i, (v - mean) * inv);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

// ---------------------------------------------------------------------------
// Reductions: ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceSumSquare
// ---------------------------------------------------------------------------

fn reduction_out_dims(dims: &[i64], axes: &[usize], keepdims: bool) -> Vec<i64> {
    let mut out: Vec<i64> = (0..dims.len())
        .filter_map(|d| {
            if axes.contains(&d) {
                if keepdims {
                    Some(1)
                } else {
                    None
                }
            } else {
                Some(dims[d])
            }
        })
        .collect();
    if out.is_empty() {
        out.push(1);
    }
    out
}

fn resolve_axes(ndim: usize, axes: &[i64]) -> Result<Vec<usize>, OpError> {
    if axes.is_empty() {
        return Ok((0..ndim).collect());
    }
    let nd = ndim as i64;
    axes.iter()
        .map(|&a| {
            let r = if a < 0 { nd + a } else { a };
            if r < 0 || r >= nd {
                Err(OpError::InvalidAttribute(String::from(
                    "reduction: axis out of range",
                )))
            } else {
                Ok(r as usize)
            }
        })
        .collect()
}

fn compute_strides(shape: &[i64]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = alloc::vec![0usize; ndim];
    if ndim == 0 {
        return strides;
    }
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        let dim = shape[i + 1].max(1) as usize;
        strides[i] = strides[i + 1] * dim;
    }
    strides
}

/// Generic fold over a reduction. `init` is the starting accumulator value,
/// `combine(acc, x)` folds element `x` in, and `finalize(acc, count)` is
/// applied once per output element.
fn reduce_fold<Fold: Fn(f32, f32) -> f32, Fin: Fn(f32, usize) -> f32>(
    input: &Tensor,
    axes: &[i64],
    keepdims: bool,
    op: &str,
    init: f32,
    combine: Fold,
    finalize: Fin,
) -> Result<Tensor, OpError> {
    require_float(input, op)?;
    let ndim = input.shape.ndim();
    let resolved = resolve_axes(ndim, axes)?;
    let out_dims = reduction_out_dims(&input.shape.dims, &resolved, keepdims);
    let out_shape = TensorShape::new(out_dims);
    let total_out = out_shape.total_elements();
    let out_strides = compute_strides(&out_shape.dims);
    let (acc, count_per_out) = reduce_accumulate(
        input,
        ndim,
        &resolved,
        keepdims,
        &out_strides,
        total_out,
        init,
        combine,
    );
    let raw = reduce_finalize_buffer(&acc, &count_per_out, total_out, finalize);
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data: raw,
    })
}

/// Single forward pass over `input` projecting each element into the
/// output index space and applying `combine` to accumulate.
#[allow(clippy::too_many_arguments)]
fn reduce_accumulate<Fold: Fn(f32, f32) -> f32>(
    input: &Tensor,
    ndim: usize,
    resolved: &[usize],
    keepdims: bool,
    out_strides: &[usize],
    total_out: usize,
    init: f32,
    combine: Fold,
) -> (Vec<f32>, Vec<usize>) {
    let mut acc = alloc::vec![init; total_out];
    let mut count_per_out = alloc::vec![0usize; total_out];
    let in_total = input.shape.total_elements();
    let mut in_coord = alloc::vec![0usize; ndim];
    for i in 0..in_total {
        if i > 0 {
            next_coord(&mut in_coord, &input.shape.dims);
        }
        let out_idx = project_out_index(&in_coord, ndim, resolved, keepdims, out_strides);
        let v = read_f32(&input.raw_data, i);
        acc[out_idx] = combine(acc[out_idx], v);
        count_per_out[out_idx] += 1;
    }
    (acc, count_per_out)
}

/// Applies the finalizer to every accumulator cell and packs the result
/// into a fresh Float buffer.
fn reduce_finalize_buffer<Fin: Fn(f32, usize) -> f32>(
    acc: &[f32],
    count_per_out: &[usize],
    total_out: usize,
    finalize: Fin,
) -> Vec<u8> {
    let mut raw = allocate_tensor_data(total_out, DataType::Float);
    for i in 0..total_out {
        write_f32(&mut raw, i, finalize(acc[i], count_per_out[i]));
    }
    raw
}

/// `ReduceL1`: sum of absolute values.
pub fn op_reduce_l1(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    reduce_fold(
        input,
        axes,
        keepdims,
        "ReduceL1",
        0.0,
        |a, x| a + abs32(x),
        |a, _| a,
    )
}

/// `ReduceL2`: Euclidean norm.
pub fn op_reduce_l2(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    reduce_fold(
        input,
        axes,
        keepdims,
        "ReduceL2",
        0.0,
        |a, x| a + x * x,
        |a, _| libm_sqrt(a),
    )
}

/// `ReduceLogSum`: log of sum.
pub fn op_reduce_log_sum(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    reduce_fold(
        input,
        axes,
        keepdims,
        "ReduceLogSum",
        0.0,
        |a, x| a + x,
        |a, _| libm_ln(a),
    )
}

/// `ReduceLogSumExp`: log of sum of exponentials. Implemented naively in two
/// passes (max-subtract for stability). Because the generic `reduce_fold`
/// helper only supports one pass, this op uses a direct implementation.
pub fn op_reduce_log_sum_exp(
    input: &Tensor,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor, OpError> {
    require_float(input, "ReduceLogSumExp")?;
    let ndim = input.shape.ndim();
    let resolved = resolve_axes(ndim, axes)?;
    let out_dims = reduction_out_dims(&input.shape.dims, &resolved, keepdims);
    let out_shape = TensorShape::new(out_dims);
    let total_out = out_shape.total_elements();
    let in_total = input.shape.total_elements();
    let out_strides = compute_strides(&out_shape.dims);
    // Pass 1: find max per output cell.
    let mut maxs = alloc::vec![f32::NEG_INFINITY; total_out];
    let mut in_coord = alloc::vec![0usize; ndim];
    for i in 0..in_total {
        if i > 0 {
            next_coord(&mut in_coord, &input.shape.dims);
        }
        let out_idx = project_out_index(&in_coord, ndim, &resolved, keepdims, &out_strides);
        let v = read_f32(&input.raw_data, i);
        if v > maxs[out_idx] {
            maxs[out_idx] = v;
        }
    }
    // Pass 2: sum of exp(x - max).
    let mut sums = alloc::vec![0.0f32; total_out];
    let mut in_coord = alloc::vec![0usize; ndim];
    for i in 0..in_total {
        if i > 0 {
            next_coord(&mut in_coord, &input.shape.dims);
        }
        let out_idx = project_out_index(&in_coord, ndim, &resolved, keepdims, &out_strides);
        let v = read_f32(&input.raw_data, i);
        sums[out_idx] += crate::operators::expf_approx(v - maxs[out_idx]);
    }
    let mut raw = allocate_tensor_data(total_out, DataType::Float);
    for i in 0..total_out {
        let lse = if sums[i] > 0.0 {
            maxs[i] + libm_ln(sums[i])
        } else {
            f32::NEG_INFINITY
        };
        write_f32(&mut raw, i, lse);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data: raw,
    })
}

fn project_out_index(
    in_coord: &[usize],
    ndim: usize,
    resolved: &[usize],
    keepdims: bool,
    out_strides: &[usize],
) -> usize {
    let mut out_idx = 0usize;
    let mut od = 0usize;
    for d in 0..ndim {
        if resolved.contains(&d) {
            if keepdims {
                od += 1;
            }
        } else {
            if od < out_strides.len() {
                out_idx += in_coord[d] * out_strides[od];
            }
            od += 1;
        }
    }
    out_idx
}

/// `ReduceSumSquare`: sum of `x^2`.
pub fn op_reduce_sum_square(
    input: &Tensor,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor, OpError> {
    reduce_fold(
        input,
        axes,
        keepdims,
        "ReduceSumSquare",
        0.0,
        |a, x| a + x * x,
        |a, _| a,
    )
}

// ---------------------------------------------------------------------------
// DynamicQuantizeLinear
// ---------------------------------------------------------------------------

/// Computes an asymmetric (u8) quantization of a Float tensor at runtime.
/// Returns the quantized tensor only; the scale and zero_point (ONNX
/// secondary outputs) can be derived from the dispatch glue via a
/// separate helper. This simplified API returns all three values.
pub fn op_dynamic_quantize_linear(input: &Tensor) -> Result<(Tensor, Tensor, Tensor), OpError> {
    require_float(input, "DynamicQuantizeLinear")?;
    let n = input.shape.total_elements();
    if n == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "DynamicQuantizeLinear: empty input",
        )));
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for i in 0..n {
        let v = read_f32(&input.raw_data, i);
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    // Include 0 in the range so the encoded zero is exactly representable.
    if lo > 0.0 {
        lo = 0.0;
    }
    if hi < 0.0 {
        hi = 0.0;
    }
    let scale = if hi - lo > 0.0 {
        (hi - lo) / 255.0
    } else {
        1.0
    };
    let zp_f = -lo / scale;
    let zp = round_f32(zp_f).clamp(0.0, 255.0) as u8;

    let mut data = alloc::vec![0u8; n];
    for i in 0..n {
        let v = read_f32(&input.raw_data, i);
        let q = round_f32(v / scale) as i32 + zp as i32;
        data[i] = q.clamp(0, 255) as u8;
    }
    let qtensor = Tensor {
        data_type: DataType::Uint8,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    };
    let mut s_raw = alloc::vec![0u8; 4];
    write_f32(&mut s_raw, 0, scale);
    let stensor = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![1]),
        name: String::new(),
        raw_data: s_raw,
    };
    let ztensor = Tensor {
        data_type: DataType::Uint8,
        shape: TensorShape::new(alloc::vec![1]),
        name: String::new(),
        raw_data: alloc::vec![zp],
    };
    Ok((qtensor, stensor, ztensor))
}

// ---------------------------------------------------------------------------
// Integer GEMM kernel (shared by MatMulInteger and QLinearMatMul)
// ---------------------------------------------------------------------------

/// Tiled i32-accumulator matrix-multiply for 8-bit operands.
///
/// Computes `C[i,j] = sum_k (A[i,k] - a_zp) * (B[k,j] - b_zp)` with the
/// accumulation performed in `i32`. Zero-points are folded at the edges
/// (D6 in `docs/generative-llm-v1/design.md`):
///
/// ```text
///   sum_k (a - a_zp) * (b - b_zp)
/// = sum_k a*b - a_zp*sum_k b - b_zp*sum_k a + K*a_zp*b_zp
/// ```
///
/// so we precompute `row_a = sum_k a[i,k]` and `col_b = sum_k b[k,j]`,
/// then correct once per output element. The hot inner loop is a pure
/// `i32` multiply-accumulate.
///
/// Inputs are passed as raw `i8` slices (use `i8::from(u8 as i8)` for the
/// signed interpretation).
///
/// Returns the raw i32 result buffer of length `m * n`, row-major.
pub(crate) fn gemm_i8(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    a_zp: i32,
    b_zp: i32,
) -> Vec<i32> {
    let row_a_sum = i8_row_sums(a, m, k);
    let col_b_sum = i8_col_sums(b, k, n);
    let mut c = alloc::vec![0i32; m * n];
    gemm_i8_blocked_kernel(a, b, m, k, n, &mut c);
    apply_zero_point_correction(&mut c, m, n, k, a_zp, b_zp, &row_a_sum, &col_b_sum);
    c
}

/// Row sums of an `m x k` row-major i8 matrix, as i32.
fn i8_row_sums(a: &[i8], m: usize, k: usize) -> Vec<i32> {
    let mut row_a_sum = alloc::vec![0i32; m];
    for i in 0..m {
        let mut s = 0i32;
        for kk in 0..k {
            s += a[i * k + kk] as i32;
        }
        row_a_sum[i] = s;
    }
    row_a_sum
}

/// Column sums of a `k x n` row-major i8 matrix, as i32.
fn i8_col_sums(b: &[i8], k: usize, n: usize) -> Vec<i32> {
    let mut col_b_sum = alloc::vec![0i32; n];
    for j in 0..n {
        let mut s = 0i32;
        for kk in 0..k {
            s += b[kk * n + j] as i32;
        }
        col_b_sum[j] = s;
    }
    col_b_sum
}

/// Cache-blocked i32-accumulator multiply: `c += a * b`.
fn gemm_i8_blocked_kernel(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, c: &mut [i32]) {
    // Block sizes match the f32 gemm helper.
    const BM: usize = 64;
    const BN: usize = 64;
    const BK: usize = 64;
    for i0 in (0..m).step_by(BM) {
        let i_end = (i0 + BM).min(m);
        for j0 in (0..n).step_by(BN) {
            let j_end = (j0 + BN).min(n);
            for k0 in (0..k).step_by(BK) {
                let k_end = (k0 + BK).min(k);
                gemm_i8_inner_block(a, b, c, k, n, i0..i_end, j0..j_end, k0..k_end);
            }
        }
    }
}

/// Innermost triple-loop multiply-accumulate over a single block.
#[allow(clippy::too_many_arguments)]
fn gemm_i8_inner_block(
    a: &[i8],
    b: &[i8],
    c: &mut [i32],
    k: usize,
    n: usize,
    i_range: core::ops::Range<usize>,
    j_range: core::ops::Range<usize>,
    k_range: core::ops::Range<usize>,
) {
    for i in i_range {
        for j in j_range.clone() {
            let mut acc = c[i * n + j];
            for kk in k_range.clone() {
                acc += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
            }
            c[i * n + j] = acc;
        }
    }
}

/// Folds in the per-element zero-point correction:
/// `c[i,j] += k*a_zp*b_zp - a_zp*col_b_sum[j] - b_zp*row_a_sum[i]`.
#[allow(clippy::too_many_arguments)]
fn apply_zero_point_correction(
    c: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
    a_zp: i32,
    b_zp: i32,
    row_a_sum: &[i32],
    col_b_sum: &[i32],
) {
    let zp_const = (k as i32) * a_zp * b_zp;
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] += zp_const - a_zp * col_b_sum[j] - b_zp * row_a_sum[i];
        }
    }
}

/// `MatMulInteger`: 2D i8 x i8 matrix multiply returning an i32 tensor.
pub fn op_matmul_integer(
    a: &Tensor,
    b: &Tensor,
    a_zp: Option<&Tensor>,
    b_zp: Option<&Tensor>,
) -> Result<Tensor, OpError> {
    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "MatMulInteger requires 2D inputs",
        )));
    }
    if !matches!(a.data_type, DataType::Int8 | DataType::Uint8) {
        return Err(OpError::ShapeMismatch(String::from(
            "MatMulInteger: A must be Int8 or Uint8",
        )));
    }
    if !matches!(b.data_type, DataType::Int8 | DataType::Uint8) {
        return Err(OpError::ShapeMismatch(String::from(
            "MatMulInteger: B must be Int8 or Uint8",
        )));
    }
    let m = a.shape.dims[0] as usize;
    let ka = a.shape.dims[1] as usize;
    let kb = b.shape.dims[0] as usize;
    let n = b.shape.dims[1] as usize;
    if ka != kb {
        return Err(OpError::ShapeMismatch(String::from(
            "MatMulInteger: inner dim mismatch",
        )));
    }
    let a_zp_v = read_int_zp(a_zp, a.data_type)?;
    let b_zp_v = read_int_zp(b_zp, b.data_type)?;

    let a_signed = to_signed_vec(&a.raw_data, a.data_type);
    let b_signed = to_signed_vec(&b.raw_data, b.data_type);
    let c = gemm_i8(&a_signed, &b_signed, m, ka, n, a_zp_v, b_zp_v);

    let mut data = allocate_tensor_data(m * n, DataType::Int32);
    for (i, v) in c.iter().enumerate() {
        crate::byte_io::write_i32(&mut data, i, *v);
    }
    Ok(Tensor {
        data_type: DataType::Int32,
        shape: TensorShape::new(alloc::vec![m as i64, n as i64]),
        name: String::new(),
        raw_data: data,
    })
}

fn to_signed_vec(raw: &[u8], dtype: DataType) -> Vec<i8> {
    match dtype {
        DataType::Int8 => raw.iter().map(|&b| b as i8).collect(),
        // Uint8 interpreted with zp-folding treats bytes as i8 after shift.
        // The MatMulInteger kernel already absorbs the u8-vs-i8 interpretation
        // through the `a_zp`/`b_zp` values provided by the caller, so here
        // we simply reinterpret the bytes. Note: for u8 inputs the caller
        // is expected to pass `a_zp = 128` if they want a symmetric frame.
        DataType::Uint8 => raw.iter().map(|&b| b as i8).collect(),
        _ => Vec::new(),
    }
}

fn read_int_zp(zp: Option<&Tensor>, _dtype: DataType) -> Result<i32, OpError> {
    match zp {
        None => Ok(0),
        Some(t) => {
            if t.shape.total_elements() > 1 {
                return Err(OpError::InvalidAttribute(String::from(
                    "MatMulInteger: per-axis zero_point not yet supported",
                )));
            }
            match t.data_type {
                DataType::Int8 => Ok(t.raw_data.first().map(|&b| b as i8 as i32).unwrap_or(0)),
                DataType::Uint8 => Ok(t.raw_data.first().map(|&b| b as i32).unwrap_or(0)),
                _ => Err(OpError::ShapeMismatch(String::from(
                    "MatMulInteger: zero_point must be Int8/Uint8",
                ))),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::common::test_helpers::{make_f32, read_all_f32};

    fn read_i32_vec(t: &Tensor) -> Vec<i32> {
        (0..t.shape.total_elements())
            .map(|i| crate::byte_io::read_i32(&t.raw_data, i))
            .collect()
    }

    fn read_i64_vec(t: &Tensor) -> Vec<i64> {
        (0..t.shape.total_elements())
            .map(|i| crate::byte_io::read_i64(&t.raw_data, i))
            .collect()
    }

    // ---- Softplus ----

    #[test]
    fn softplus_zero_is_ln2() {
        let t = make_f32(&[1], &[0.0]);
        let out = op_softplus(&t).unwrap();
        let v = read_all_f32(&out)[0];
        assert!((v - core::f32::consts::LN_2).abs() < 1e-4);
    }

    #[test]
    fn softplus_large_positive_identity() {
        let t = make_f32(&[1], &[50.0]);
        let out = op_softplus(&t).unwrap();
        assert!((read_all_f32(&out)[0] - 50.0).abs() < 1e-3);
    }

    // ---- EyeLike ----

    #[test]
    fn eye_like_basic() {
        let t = make_f32(&[3, 3], &[0.0; 9]);
        let out = op_eye_like(&t, 0).unwrap();
        let v = read_all_f32(&out);
        assert_eq!(v, alloc::vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn eye_like_offset_diagonal() {
        let t = make_f32(&[3, 3], &[0.0; 9]);
        let out = op_eye_like(&t, 1).unwrap();
        let v = read_all_f32(&out);
        // k=1: superdiagonal.
        assert_eq!(v, alloc::vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    // ---- Dropout ----

    #[test]
    fn dropout_inference_is_identity() {
        let t = make_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let out = op_dropout(&t).unwrap();
        assert_eq!(read_all_f32(&out), alloc::vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dropout_rejects_non_float() {
        let t = Tensor {
            data_type: DataType::Int32,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: alloc::vec![0u8; 4],
        };
        assert!(op_dropout(&t).is_err());
    }

    // ---- Random ops ----

    #[test]
    fn random_uniform_in_range() {
        let out = op_random_uniform(&[100], 42.0, -1.0, 1.0).unwrap();
        for v in read_all_f32(&out) {
            assert!((-1.0..1.0).contains(&v));
        }
    }

    #[test]
    fn random_uniform_is_deterministic() {
        let a = op_random_uniform(&[8], 3.5, 0.0, 1.0).unwrap();
        let b = op_random_uniform(&[8], 3.5, 0.0, 1.0).unwrap();
        assert_eq!(read_all_f32(&a), read_all_f32(&b));
    }

    #[test]
    fn random_uniform_rejects_bad_range() {
        assert!(op_random_uniform(&[4], 1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn random_normal_statistics() {
        let out = op_random_normal(&[4096], 7.0, 0.0, 1.0).unwrap();
        let v = read_all_f32(&out);
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        // Mean should be close to 0 for enough samples.
        assert!(mean.abs() < 0.15, "mean={}", mean);
    }

    #[test]
    fn random_normal_like_shape_matches_input() {
        let shape = make_f32(&[2, 3], &[0.0; 6]);
        let out = op_random_normal_like(&shape, 1.0, 0.0, 1.0).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![2, 3]);
    }

    #[test]
    fn random_uniform_like_shape_matches_input() {
        let shape = make_f32(&[5], &[0.0; 5]);
        let out = op_random_uniform_like(&shape, 1.0, -2.0, 2.0).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![5]);
    }

    // ---- Bernoulli ----

    #[test]
    fn bernoulli_p_zero_is_all_zeros() {
        let t = make_f32(&[10], &[0.0; 10]);
        let out = op_bernoulli(&t, 1.0).unwrap();
        for v in read_all_f32(&out) {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn bernoulli_p_one_is_all_ones() {
        let t = make_f32(&[10], &[1.0; 10]);
        let out = op_bernoulli(&t, 1.0).unwrap();
        for v in read_all_f32(&out) {
            assert_eq!(v, 1.0);
        }
    }

    // ---- Multinomial ----

    #[test]
    fn multinomial_peaked_logits_pick_peak() {
        // logits [100, 0, 0]: should always pick class 0.
        let t = make_f32(&[1, 3], &[100.0, 0.0, 0.0]);
        let out = op_multinomial(&t, 4, 2.0).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 4]);
        for v in read_i64_vec(&out) {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn multinomial_rejects_non_2d() {
        let t = make_f32(&[3], &[1.0, 2.0, 3.0]);
        assert!(op_multinomial(&t, 1, 0.0).is_err());
    }

    // ---- RMSNormalization ----

    #[test]
    fn rms_norm_constant_input() {
        // All-ones input: RMS = 1, so output = 1 * scale.
        let x = make_f32(&[1, 4], &[1.0; 4]);
        let scale = make_f32(&[4], &[0.5, 0.5, 0.5, 0.5]);
        let out = op_rms_normalization(&x, &scale, -1, 1e-6).unwrap();
        for v in read_all_f32(&out) {
            assert!((v - 0.5).abs() < 1e-4);
        }
    }

    #[test]
    fn rms_norm_preserves_zero_mean_direction() {
        // x = [1, -1, 1, -1]: RMS = 1, so output matches input * scale.
        let x = make_f32(&[1, 4], &[1.0, -1.0, 1.0, -1.0]);
        let scale = make_f32(&[1], &[1.0]);
        let out = op_rms_normalization(&x, &scale, 1, 1e-6).unwrap();
        let v = read_all_f32(&out);
        for (o, &ex) in v.iter().zip([1.0f32, -1.0, 1.0, -1.0].iter()) {
            assert!((o - ex).abs() < 1e-3);
        }
    }

    // ---- LpNormalization ----

    #[test]
    fn lp_normalization_l2_unit_vector() {
        // [3, 4] should normalize to [0.6, 0.8].
        let x = make_f32(&[1, 2], &[3.0, 4.0]);
        let out = op_lp_normalization(&x, 1, 2).unwrap();
        let v = read_all_f32(&out);
        assert!((v[0] - 0.6).abs() < 1e-4);
        assert!((v[1] - 0.8).abs() < 1e-4);
    }

    #[test]
    fn lp_normalization_l1_sums_to_one() {
        let x = make_f32(&[1, 3], &[1.0, 2.0, 1.0]);
        let out = op_lp_normalization(&x, -1, 1).unwrap();
        let v = read_all_f32(&out);
        let sum: f32 = v.iter().map(|x| abs32(*x)).sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    // ---- MeanVarianceNormalization ----

    #[test]
    fn mvn_zero_mean_unit_variance() {
        let x = make_f32(&[5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = op_mean_variance_normalization(&x, &[]).unwrap();
        let v = read_all_f32(&out);
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn mvn_rejects_partial_axes() {
        let x = make_f32(&[2, 3], &[1.0; 6]);
        assert!(op_mean_variance_normalization(&x, &[0]).is_err());
    }

    // ---- Reduction ops ----

    #[test]
    fn reduce_l1_basic() {
        let x = make_f32(&[4], &[1.0, -2.0, 3.0, -4.0]);
        let out = op_reduce_l1(&x, &[], true).unwrap();
        assert!((read_all_f32(&out)[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_l1_axis_rejected_out_of_range() {
        let x = make_f32(&[2, 3], &[1.0; 6]);
        assert!(op_reduce_l1(&x, &[5], false).is_err());
    }

    #[test]
    fn reduce_l2_basic() {
        let x = make_f32(&[3], &[3.0, 4.0, 0.0]);
        let out = op_reduce_l2(&x, &[], true).unwrap();
        // sqrt(9+16) = 5
        assert!((read_all_f32(&out)[0] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn reduce_l2_over_axis() {
        let x = make_f32(&[2, 2], &[3.0, 4.0, 0.0, 5.0]);
        let out = op_reduce_l2(&x, &[1], false).unwrap();
        let v = read_all_f32(&out);
        assert!((v[0] - 5.0).abs() < 1e-4);
        assert!((v[1] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn reduce_log_sum_basic() {
        let x = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let out = op_reduce_log_sum(&x, &[], true).unwrap();
        // ln(6) ≈ 1.7917
        assert!((read_all_f32(&out)[0] - 1.791_759_5).abs() < 1e-3);
    }

    #[test]
    fn reduce_log_sum_exp_stable_for_large_values() {
        // Without max-subtract, sum would overflow.
        let x = make_f32(&[3], &[80.0, 80.0, 80.0]);
        let out = op_reduce_log_sum_exp(&x, &[], true).unwrap();
        // Expected: 80 + ln(3) ≈ 81.0986
        let v = read_all_f32(&out)[0];
        assert!((v - 81.098_6).abs() < 1e-2, "got {}", v);
    }

    #[test]
    fn reduce_log_sum_exp_axis_reduction() {
        let x = make_f32(&[2, 2], &[0.0, 0.0, 1.0, 1.0]);
        let out = op_reduce_log_sum_exp(&x, &[1], false).unwrap();
        let v = read_all_f32(&out);
        // ln(e^0 + e^0) = ln 2 for row 0
        assert!((v[0] - core::f32::consts::LN_2).abs() < 1e-3);
    }

    #[test]
    fn reduce_sum_square_basic() {
        let x = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let out = op_reduce_sum_square(&x, &[], true).unwrap();
        assert!((read_all_f32(&out)[0] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_sum_square_axis() {
        let x = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let out = op_reduce_sum_square(&x, &[0], false).unwrap();
        let v = read_all_f32(&out);
        // [1+9, 4+16] = [10, 20]
        assert!((v[0] - 10.0).abs() < 1e-5);
        assert!((v[1] - 20.0).abs() < 1e-5);
    }

    // ---- DynamicQuantizeLinear ----

    #[test]
    fn dynamic_quantize_roundtrip_zero_in_range() {
        let x = make_f32(&[4], &[-1.0, 0.0, 1.0, 2.0]);
        let (q, s, z) = op_dynamic_quantize_linear(&x).unwrap();
        assert_eq!(q.data_type, DataType::Uint8);
        let scale = read_f32(&s.raw_data, 0);
        let zp = z.raw_data[0] as i32;
        // Dequantize and compare.
        for i in 0..4 {
            let dq = (q.raw_data[i] as i32 - zp) as f32 * scale;
            let orig = read_f32(&x.raw_data, i);
            assert!(
                (dq - orig).abs() < scale * 1.5 + 1e-3,
                "roundtrip {} vs {} (scale {})",
                dq,
                orig,
                scale
            );
        }
    }

    #[test]
    fn dynamic_quantize_all_positive() {
        // lo > 0 should be clamped to 0 for symmetric range.
        let x = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let (q, s, z) = op_dynamic_quantize_linear(&x).unwrap();
        let scale = read_f32(&s.raw_data, 0);
        let zp = z.raw_data[0] as i32;
        // zero point should encode the value 0 exactly.
        assert_eq!(zp, 0);
        let _ = (q, scale);
    }

    // ---- MatMulInteger + gemm_i8 kernel ----

    fn make_i8_tensor(dims: &[i64], vals: &[i8]) -> Tensor {
        let raw: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
        Tensor {
            data_type: DataType::Int8,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: raw,
        }
    }

    #[test]
    fn matmul_integer_basic_no_zp() {
        // 2x3 * 3x2 with easy values.
        // A = [[1, 2, 3], [4, 5, 6]]
        // B = [[1, 0], [0, 1], [1, 1]]
        // C = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4, 5], [10, 11]]
        let a = make_i8_tensor(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        let b = make_i8_tensor(&[3, 2], &[1, 0, 0, 1, 1, 1]);
        let out = op_matmul_integer(&a, &b, None, None).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![2, 2]);
        let v = read_i32_vec(&out);
        assert_eq!(v, alloc::vec![4, 5, 10, 11]);
    }

    #[test]
    fn matmul_integer_with_zero_points() {
        // Verify the zero-point fold produces the same result as an
        // f32 subtract-then-multiply reference.
        let a_raw = [10i8, 20, 30, 40];
        let b_raw = [5i8, 6, 7, 8];
        let a = make_i8_tensor(&[2, 2], &a_raw);
        let b = make_i8_tensor(&[2, 2], &b_raw);
        let a_zp = Tensor {
            data_type: DataType::Int8,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: alloc::vec![5i8 as u8],
        };
        let b_zp = Tensor {
            data_type: DataType::Int8,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: alloc::vec![3i8 as u8],
        };
        let out = op_matmul_integer(&a, &b, Some(&a_zp), Some(&b_zp)).unwrap();
        let v = read_i32_vec(&out);

        // Reference: (a - 5) @ (b - 3)
        let a_ref: [i32; 4] = [5, 15, 25, 35];
        let b_ref: [i32; 4] = [2, 3, 4, 5];
        let exp: [i32; 4] = [
            a_ref[0] * b_ref[0] + a_ref[1] * b_ref[2],
            a_ref[0] * b_ref[1] + a_ref[1] * b_ref[3],
            a_ref[2] * b_ref[0] + a_ref[3] * b_ref[2],
            a_ref[2] * b_ref[1] + a_ref[3] * b_ref[3],
        ];
        assert_eq!(v, exp.to_vec());
    }

    #[test]
    fn matmul_integer_tile_boundary_k() {
        // K larger than the 64-sized inner block to exercise cache tiling.
        let k = 70;
        let m = 2;
        let n = 2;
        let a: Vec<i8> = (0..(m * k)).map(|i| ((i % 5) as i8) - 2).collect();
        let b: Vec<i8> = (0..(k * n)).map(|i| ((i % 7) as i8) - 3).collect();
        let at = make_i8_tensor(&[m as i64, k as i64], &a);
        let bt = make_i8_tensor(&[k as i64, n as i64], &b);
        let out = op_matmul_integer(&at, &bt, None, None).unwrap();
        let v = read_i32_vec(&out);
        // Naive reference
        let mut ref_c = alloc::vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0i32;
                for kk in 0..k {
                    s += (a[i * k + kk] as i32) * (b[kk * n + j] as i32);
                }
                ref_c[i * n + j] = s;
            }
        }
        assert_eq!(v, ref_c);
    }

    #[test]
    fn matmul_integer_rejects_shape_mismatch() {
        let a = make_i8_tensor(&[2, 3], &[0; 6]);
        let b = make_i8_tensor(&[4, 2], &[0; 8]);
        assert!(op_matmul_integer(&a, &b, None, None).is_err());
    }

    // ---- XorShift32 determinism ----

    #[test]
    fn xorshift_is_deterministic() {
        let mut a = XorShift32::new(42);
        let mut b = XorShift32::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    // ---- BFloat16 RMSNormalization (safetensors-model-loader-v1 §3) ----

    fn make_bf16(dims: &[i64], vals: &[f32]) -> Tensor {
        Tensor {
            data_type: DataType::BFloat16,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: f32_to_bf16(vals),
        }
    }

    #[test]
    fn rms_normalization_bf16_matches_f32_reference() {
        // Values deliberately chosen with BF16-exact representations so
        // round-tripping doesn't introduce additional error.
        let vals = [1.0f32, 2.0, 3.0, 4.0, 0.5, -1.5, 2.5, -3.0];
        let scale_vals = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let input_f32 = make_f32(&[1, 8], &vals);
        let scale_f32 = make_f32(&[8], &scale_vals);
        let ref_out = op_rms_normalization(&input_f32, &scale_f32, -1, 1e-5).unwrap();
        let ref_vals = read_all_f32(&ref_out);

        let input_bf = make_bf16(&[1, 8], &vals);
        let scale_bf = make_bf16(&[8], &scale_vals);
        let bf_out = op_rms_normalization(&input_bf, &scale_bf, -1, 1e-5).unwrap();
        assert_eq!(bf_out.data_type, DataType::BFloat16);
        let decoded = bf16_to_f32(&bf_out.raw_data);

        for (r, b) in ref_vals.iter().zip(decoded.iter()) {
            // BF16 mantissa has ~8 bits -> relative error ~1/128 = 0.78%.
            let rel = (r - b).abs() / r.abs().max(1e-3);
            assert!(
                rel < 2e-2,
                "rms_norm bf16 mismatch: ref={} bf={} rel={}",
                r,
                b,
                rel
            );
        }
    }

    #[test]
    fn rms_normalization_bf16_preserves_shape_and_dtype() {
        let x = make_bf16(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let scale = make_bf16(&[4], &[1.0, 1.0, 1.0, 1.0]);
        let out = op_rms_normalization(&x, &scale, -1, 1e-5).unwrap();
        assert_eq!(out.data_type, DataType::BFloat16);
        assert_eq!(out.shape.dims, alloc::vec![2, 4]);
        assert_eq!(out.raw_data.len(), 2 * 4 * 2);
    }
}
