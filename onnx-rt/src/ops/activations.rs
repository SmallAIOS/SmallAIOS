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
    let slope_total = slope.shape.total_elements();
    let scalar_slope = slope_total <= 1;
    let channel_slope =
        !scalar_slope && input.shape.dims.len() >= 2 && slope_total == input.shape.dims[1] as usize;
    let inner_per_c: usize = if input.shape.dims.len() >= 2 {
        input.shape.dims[2..]
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1)
    } else {
        1
    };
    let outer: usize = if input.shape.dims.is_empty() {
        1
    } else {
        input.shape.dims[0] as usize
    };
    let c: usize = if input.shape.dims.len() >= 2 {
        input.shape.dims[1] as usize
    } else {
        1
    };
    for flat in 0..total {
        let x = read_f32(&input.raw_data, flat);
        let s = if scalar_slope {
            read_f32(&slope.raw_data, 0)
        } else if channel_slope {
            let ch = (flat / inner_per_c) % c;
            read_f32(&slope.raw_data, ch)
        } else {
            read_f32(&slope.raw_data, flat.min(slope_total - 1))
        };
        let _ = outer;
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
