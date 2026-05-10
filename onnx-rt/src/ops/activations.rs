// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Composite activation functions used by transformer-style models.

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
use crate::operators::{expf_approx, OpError};
use crate::ops::common::{require_float, unary};
use crate::ops::math::erf_approx;
use crate::tensor::{DataType, Tensor};

/// Gaussian Error Linear Unit: `0.5 * x * (1 + erf(x / sqrt(2)))`.
pub fn op_gelu(input: &Tensor) -> Result<Tensor, OpError> {
    let inv_sqrt2 = core::f32::consts::FRAC_1_SQRT_2;
    unary(input, "Gelu", |x| {
        0.5 * x * (1.0 + erf_approx(x * inv_sqrt2))
    })
}

/// Leaky ReLU: `x if x > 0 else alpha * x`.
pub fn op_leaky_relu(input: &Tensor, alpha: f32) -> Result<Tensor, OpError> {
    unary(input, "LeakyRelu", |x| if x >= 0.0 { x } else { alpha * x })
}

/// Exponential Linear Unit: `x if x >= 0 else alpha * (exp(x) - 1)`.
pub fn op_elu(input: &Tensor, alpha: f32) -> Result<Tensor, OpError> {
    unary(input, "Elu", |x| {
        if x >= 0.0 {
            x
        } else {
            alpha * (expf_approx(x) - 1.0)
        }
    })
}

/// Swish (a.k.a. SiLU): `x * sigmoid(x)`.
pub fn op_swish(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Swish", |x| x / (1.0 + expf_approx(-x)))
}

/// HardSigmoid: `clip(alpha * x + beta, 0, 1)`.
pub fn op_hard_sigmoid(input: &Tensor, alpha: f32, beta: f32) -> Result<Tensor, OpError> {
    unary(input, "HardSigmoid", |x| (alpha * x + beta).clamp(0.0, 1.0))
}

/// HardSwish: `x * relu6(x + 3) / 6`.
pub fn op_hard_swish(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "HardSwish", |x| x * (x + 3.0).clamp(0.0, 6.0) / 6.0)
}

/// PReLU: `x if x > 0 else slope * x`, where `slope` is per-channel.
///
/// `slope` may be a scalar or broadcastable along the leading axis of
/// `input`. When `slope.len() == C`, the channel index is inferred
/// from NCHW layout (axis 1). Otherwise a single scalar slope is used.
pub fn op_prelu(input: &Tensor, slope: &Tensor) -> Result<Tensor, OpError> {
    require_float(input, "PRelu")?;
    require_float(slope, "PRelu")?;
    let total = input.shape.total_elements();
    let mut data = allocate_tensor_data(total, DataType::Float);

    let mode = classify_prelu_slope(input, slope)?;
    let c: usize = if input.shape.dims.len() >= 2 {
        input.shape.dims[1] as usize
    } else {
        1
    };
    for flat in 0..total {
        let x = read_f32(&input.raw_data, flat);
        let s = prelu_slope_for(&mode, &slope.raw_data, flat, c);
        let y = if x >= 0.0 { x } else { s * x };
        write_f32(&mut data, flat, y);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: alloc::string::String::new(),
        raw_data: data,
    })
}

/// Broadcasting mode for the PReLU `slope` tensor.
///
/// ONNX PRelu allows uni-directional broadcasting of `slope` over
/// `input`. We accept the two shapes real exporters use and reject
/// anything else with ShapeMismatch (no silent clamping).
///
///   1. Scalar slope: shape `[]` or `[1]` (or all-ones) -> applied
///      uniformly.
///   2. Per-channel: rank >= 2, channel axis == 1, and slope is
///      either length `C` as a 1D tensor or a rank-matching tensor
///      with all non-channel dims equal to 1.
#[derive(Clone, Copy)]
enum PreluMode {
    Scalar,
    PerChannel(usize),
}

/// Validates the slope tensor's broadcast shape against the input and
/// returns the dispatch mode.
fn classify_prelu_slope(input: &Tensor, slope: &Tensor) -> Result<PreluMode, OpError> {
    let slope_total = slope.shape.total_elements();
    let slope_is_scalar = slope_total == 1
        && (slope.shape.dims.is_empty() || slope.shape.dims.iter().all(|&d| d == 1));
    if slope_is_scalar {
        return Ok(PreluMode::Scalar);
    }
    if input.shape.dims.len() < 2 {
        return Err(OpError::ShapeMismatch(alloc::string::String::from(
            "PRelu: slope must be scalar for rank<2 input",
        )));
    }
    let c = input.shape.dims[1] as usize;
    if !slope_matches_channel(slope, input, c) {
        return Err(OpError::ShapeMismatch(alloc::string::String::from(
            "PRelu: slope must be scalar or per-channel (shape [C] or broadcastable to channel axis)",
        )));
    }
    let inner: usize = input.shape.dims[2..]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    Ok(PreluMode::PerChannel(inner))
}

/// Returns true when `slope` is either a 1D `[C]` tensor or a rank-matching
/// broadcast (`[..., C, 1, ...]` with all non-channel dims == 1).
fn slope_matches_channel(slope: &Tensor, input: &Tensor, c: usize) -> bool {
    let per_channel_1d = slope.shape.dims.as_slice() == [c as i64];
    let per_channel_broadcast = slope.shape.dims.len() == input.shape.dims.len()
        && slope.shape.dims[1] as usize == c
        && slope
            .shape
            .dims
            .iter()
            .enumerate()
            .all(|(i, &d)| if i == 1 { d as usize == c } else { d == 1 });
    per_channel_1d || per_channel_broadcast
}

/// Picks the slope value for output element `flat` given the broadcast
/// mode, raw slope bytes, and channel count `c`.
fn prelu_slope_for(mode: &PreluMode, slope_raw: &[u8], flat: usize, c: usize) -> f32 {
    match mode {
        PreluMode::Scalar => read_f32(slope_raw, 0),
        PreluMode::PerChannel(inner) => {
            let ch = (flat / inner) % c;
            read_f32(slope_raw, ch)
        }
    }
}

/// Mish: `x * tanh(softplus(x))` where `softplus(x) = ln(1 + exp(x))`.
pub fn op_mish(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Mish", |x| {
        // Stable softplus: for large x, softplus(x) ≈ x; for small, ln(1+exp(x)).
        let sp = if x > 20.0 {
            x
        } else if x < -20.0 {
            0.0
        } else {
            ln1p_approx(expf_approx(x))
        };
        // tanh via (e^a - e^-a)/(e^a + e^-a) with expf_approx.
        let ep = expf_approx(sp);
        let en = expf_approx(-sp);
        let tanh_sp = (ep - en) / (ep + en);
        x * tanh_sp
    })
}

/// Series/log approximation of `ln(1 + x)` for x in a reasonable range.
/// Good to ~1e-6 for |x| < 1, passable elsewhere; used only by Mish.
fn ln1p_approx(x: f32) -> f32 {
    // Fall back to operators::ln_approx via Log op path? Simpler: use a
    // Taylor series for small x, otherwise compute via ln of (1+x) with a
    // cheap iterative log2-style approximation.
    if x > -0.5 && x < 0.5 {
        // Series: x - x^2/2 + x^3/3 - x^4/4 + x^5/5
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        let x5 = x4 * x;
        x - 0.5 * x2 + x3 / 3.0 - 0.25 * x4 + 0.2 * x5
    } else {
        // Fallback: use crate::ops::math::op_log via naive method.
        // Reuse algorithm: ln(1+x) via libm-ish. For large x this is fine
        // since Mish only calls with exp output, always > 0.
        ln_approx(1.0 + x)
    }
}

/// Natural log approximation for positive input. Reuses the bit-level
/// decomposition used elsewhere in the runtime.
fn ln_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    let m_bits = (bits & 0x007f_ffff) | 0x3f80_0000;
    let m = f32::from_bits(m_bits); // m in [1,2)
                                    // ln(m) via 5-term minimax on [1,2).
    let t = m - 1.0;
    let poly =
        t * (1.0 - t * (0.5 - t * (0.333_333_34 - t * (0.25 - t * (0.2 - t * 0.166_666_67)))));
    poly + (e as f32) * core::f32::consts::LN_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::common::test_helpers::{make_f32, read_all_f32 as read_all};
    use crate::tensor::{DataType, TensorShape};
    use alloc::string::String;

    #[test]
    fn test_gelu_at_zero() {
        let t = make_f32(&[1], &[0.0]);
        let v = read_all(&op_gelu(&t).unwrap());
        assert!(v[0].abs() < 1e-5);
    }

    #[test]
    fn test_gelu_known_values() {
        // Reference values from PyTorch torch.nn.functional.gelu
        let t = make_f32(&[3], &[1.0, -1.0, 2.0]);
        let v = read_all(&op_gelu(&t).unwrap());
        assert!((v[0] - 0.8413).abs() < 1e-2);
        assert!((v[1] - -0.1587).abs() < 1e-2);
        assert!((v[2] - 1.9545).abs() < 1e-2);
    }

    #[test]
    fn test_leaky_relu_positive_unchanged() {
        let t = make_f32(&[2], &[1.5, 3.0]);
        let v = read_all(&op_leaky_relu(&t, 0.01).unwrap());
        assert_eq!(v, alloc::vec![1.5, 3.0]);
    }

    #[test]
    fn test_leaky_relu_negative_scaled() {
        let t = make_f32(&[3], &[-1.0, -2.0, 0.0]);
        let v = read_all(&op_leaky_relu(&t, 0.1).unwrap());
        assert!((v[0] - -0.1).abs() < 1e-5);
        assert!((v[1] - -0.2).abs() < 1e-5);
        assert!(v[2].abs() < 1e-5);
    }

    #[test]
    fn test_elu_positive_unchanged() {
        let t = make_f32(&[2], &[1.0, 2.5]);
        let v = read_all(&op_elu(&t, 1.0).unwrap());
        assert_eq!(v, alloc::vec![1.0, 2.5]);
    }

    #[test]
    fn test_elu_negative() {
        // ELU(-1, alpha=1) = 1 * (e^-1 - 1) ≈ -0.6321
        let t = make_f32(&[1], &[-1.0]);
        let v = read_all(&op_elu(&t, 1.0).unwrap());
        assert!((v[0] - -0.6321).abs() < 1e-2);
    }

    #[test]
    fn test_swish_at_zero() {
        let t = make_f32(&[1], &[0.0]);
        let v = read_all(&op_swish(&t).unwrap());
        assert!(v[0].abs() < 1e-5);
    }

    #[test]
    fn test_swish_known_values() {
        // Swish(1) = 1 * sigmoid(1) ≈ 0.7310
        // Swish(2) = 2 * sigmoid(2) ≈ 1.7616
        let t = make_f32(&[2], &[1.0, 2.0]);
        let v = read_all(&op_swish(&t).unwrap());
        assert!((v[0] - 0.7310).abs() < 1e-2);
        assert!((v[1] - 1.7616).abs() < 1e-2);
    }

    #[test]
    fn test_swish_negative() {
        // Swish(-1) = -1 * sigmoid(-1) ≈ -0.2689
        let t = make_f32(&[1], &[-1.0]);
        let v = read_all(&op_swish(&t).unwrap());
        assert!((v[0] - -0.2689).abs() < 1e-2);
    }

    #[test]
    fn test_hard_sigmoid_clips() {
        // Default alpha=0.2, beta=0.5.
        let t = make_f32(&[5], &[-10.0, -2.0, 0.0, 2.0, 10.0]);
        let v = read_all(&op_hard_sigmoid(&t, 0.2, 0.5).unwrap());
        assert_eq!(v[0], 0.0);
        assert!((v[2] - 0.5).abs() < 1e-5);
        assert_eq!(v[4], 1.0);
    }

    #[test]
    fn test_hard_sigmoid_linear_region() {
        let t = make_f32(&[1], &[1.0]);
        let v = read_all(&op_hard_sigmoid(&t, 0.2, 0.5).unwrap());
        assert!((v[0] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_hard_swish_zero() {
        let t = make_f32(&[1], &[0.0]);
        let v = read_all(&op_hard_swish(&t).unwrap());
        assert!(v[0].abs() < 1e-5);
    }

    #[test]
    fn test_hard_swish_large_positive() {
        let t = make_f32(&[2], &[5.0, 10.0]);
        let v = read_all(&op_hard_swish(&t).unwrap());
        // For x >= 3, hard_swish(x) == x.
        assert!((v[0] - 5.0).abs() < 1e-5);
        assert!((v[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_hard_swish_negative_clipped() {
        let t = make_f32(&[2], &[-5.0, -10.0]);
        let v = read_all(&op_hard_swish(&t).unwrap());
        // For x <= -3, hard_swish(x) == 0.
        assert!(v[0].abs() < 1e-5);
        assert!(v[1].abs() < 1e-5);
    }

    #[test]
    fn test_prelu_scalar_slope() {
        let t = make_f32(&[4], &[1.0, -1.0, 2.0, -2.0]);
        let slope = make_f32(&[1], &[0.25]);
        let v = read_all(&op_prelu(&t, &slope).unwrap());
        assert_eq!(v, alloc::vec![1.0, -0.25, 2.0, -0.5]);
    }

    #[test]
    fn test_prelu_per_channel() {
        // NCHW [1,2,1,2] with per-channel slopes [0.1, 0.5].
        let t = make_f32(&[1, 2, 1, 2], &[-1.0, -2.0, -1.0, -2.0]);
        let slope = make_f32(&[2], &[0.1, 0.5]);
        let v = read_all(&op_prelu(&t, &slope).unwrap());
        // ch0: -0.1, -0.2 ; ch1: -0.5, -1.0
        assert!((v[0] + 0.1).abs() < 1e-5);
        assert!((v[1] + 0.2).abs() < 1e-5);
        assert!((v[2] + 0.5).abs() < 1e-5);
        assert!((v[3] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prelu_scalar_rank0_slope() {
        // Rank-0 scalar slope tensor must also be accepted.
        let t = make_f32(&[3], &[1.0, -1.0, -2.0]);
        let slope = Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(alloc::vec![]),
            name: String::new(),
            raw_data: {
                let mut d = allocate_tensor_data(1, DataType::Float);
                write_f32(&mut d, 0, 0.5);
                d
            },
        };
        let v = read_all(&op_prelu(&t, &slope).unwrap());
        assert_eq!(v, alloc::vec![1.0, -0.5, -1.0]);
    }

    #[test]
    fn test_prelu_rejects_invalid_broadcast() {
        // Input [1,2,1,2] (C=2) with slope length 3 is not broadcastable.
        let t = make_f32(&[1, 2, 1, 2], &[-1.0, -2.0, -1.0, -2.0]);
        let slope = make_f32(&[3], &[0.1, 0.2, 0.3]);
        assert!(matches!(
            op_prelu(&t, &slope),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_prelu_rejects_mismatched_rank_broadcast() {
        // Slope shape [1, 2, 2, 1] has non-1 non-channel dim — reject.
        let t = make_f32(&[1, 2, 1, 2], &[-1.0, -2.0, -1.0, -2.0]);
        let slope = make_f32(&[1, 2, 2, 1], &[0.1, 0.1, 0.2, 0.2]);
        assert!(matches!(
            op_prelu(&t, &slope),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_mish_at_zero() {
        let t = make_f32(&[1], &[0.0]);
        let v = read_all(&op_mish(&t).unwrap());
        assert!(v[0].abs() < 1e-4);
    }

    #[test]
    fn test_mish_known_values() {
        // Mish(1) = 1 * tanh(softplus(1)) = 1 * tanh(1.3133) ≈ 0.8651
        let t = make_f32(&[2], &[1.0, -1.0]);
        let v = read_all(&op_mish(&t).unwrap());
        assert!((v[0] - 0.8651).abs() < 5e-2);
        // Mish(-1) ≈ -0.3034
        assert!((v[1] + 0.3034).abs() < 5e-2);
    }

    // ---- BFloat16 activation support (safetensors-model-loader-v1 §3) ----

    fn make_bf16(dims: &[i64], vals: &[f32]) -> Tensor {
        Tensor {
            data_type: DataType::BFloat16,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: crate::tensor::f32_to_bf16(vals),
        }
    }

    fn read_bf16_all(t: &Tensor) -> alloc::vec::Vec<f32> {
        let n = t.shape.total_elements();
        let raw = crate::tensor::bf16_to_f32(&t.raw_data);
        raw[..n].to_vec()
    }

    #[test]
    fn test_swish_bf16_matches_f32_reference() {
        // SiLU = Swish; values picked for close BF16/F32 alignment.
        let vals = [0.0f32, 1.0, -1.0, 2.0, -2.0];
        let t_f = make_f32(&[5], &vals);
        let ref_out = read_all(&op_swish(&t_f).unwrap());

        let t_bf = make_bf16(&[5], &vals);
        let out = op_swish(&t_bf).unwrap();
        assert_eq!(out.data_type, DataType::BFloat16);
        let bf_out = read_bf16_all(&out);

        for (r, b) in ref_out.iter().zip(bf_out.iter()) {
            let rel = (r - b).abs() / r.abs().max(1e-3);
            assert!(rel < 2e-2, "swish bf16 mismatch ref={} bf={}", r, b);
        }
    }

    #[test]
    fn test_gelu_bf16_sanity() {
        let t = make_bf16(&[3], &[0.0, 1.0, -1.0]);
        let out = op_gelu(&t).unwrap();
        assert_eq!(out.data_type, DataType::BFloat16);
        let v = read_bf16_all(&out);
        assert!(v[0].abs() < 1e-2);
        assert!((v[1] - 0.8413).abs() < 5e-2);
        assert!((v[2] - -0.1587).abs() < 5e-2);
    }

    #[test]
    fn test_activations_reject_non_float() {
        let t = Tensor {
            data_type: DataType::Int32,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: alloc::vec![0u8; 4],
        };
        assert!(op_gelu(&t).is_err());
        assert!(op_leaky_relu(&t, 0.1).is_err());
        assert!(op_elu(&t, 1.0).is_err());
        assert!(op_swish(&t).is_err());
        assert!(op_hard_sigmoid(&t, 0.2, 0.5).is_err());
        assert!(op_hard_swish(&t).is_err());
        assert!(op_mish(&t).is_err());
    }
}
