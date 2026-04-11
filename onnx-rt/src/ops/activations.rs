// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Composite activation functions used by transformer-style models.

use crate::operators::{expf_approx, OpError};
use crate::ops::common::unary;
use crate::ops::math::erf_approx;
use crate::tensor::Tensor;

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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
    use crate::tensor::{DataType, TensorShape};

    fn make_f32(dims: &[i64], vals: &[f32]) -> Tensor {
        let mut data = allocate_tensor_data(vals.len(), DataType::Float);
        for (i, &v) in vals.iter().enumerate() {
            write_f32(&mut data, i, v);
        }
        Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: data,
        }
    }

    fn read_all(t: &Tensor) -> alloc::vec::Vec<f32> {
        (0..t.shape.total_elements())
            .map(|i| read_f32(&t.raw_data, i))
            .collect()
    }

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
    }
}
