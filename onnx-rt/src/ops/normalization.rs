// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GroupNormalization and InstanceNormalization.
//!
//! These normalize NCHW-style tensors across spatial dims grouped by
//! channel slices. InstanceNorm is `GroupNorm` with `num_groups == C`.

use alloc::string::String;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
use crate::operators::{sqrt_approx, OpError};
use crate::tensor::{DataType, Tensor};

fn require_float(t: &Tensor, op: &str) -> Result<(), OpError> {
    if t.data_type != DataType::Float {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{} requires Float input",
            op
        )));
    }
    Ok(())
}

/// Group normalization over an NCHW-like input.
///
/// Channels are split into `num_groups` contiguous groups; statistics
/// are computed per (N, group) over (C/num_groups * H * W ... spatial).
/// `scale` and `bias` are length-C vectors applied after normalization.
pub fn op_group_normalization(
    x: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    num_groups: i64,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    require_float(x, "GroupNormalization")?;
    require_float(scale, "GroupNormalization")?;
    if x.shape.dims.len() < 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "GroupNormalization requires rank >= 2",
        )));
    }
    let n = x.shape.dims[0] as usize;
    let c = x.shape.dims[1] as usize;
    let spatial: usize = x.shape.dims[2..]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    if num_groups <= 0 || !c.is_multiple_of(num_groups as usize) {
        return Err(OpError::InvalidAttribute(String::from(
            "GroupNormalization: C must be divisible by num_groups",
        )));
    }
    if scale.shape.total_elements() != c {
        return Err(OpError::ShapeMismatch(String::from(
            "GroupNormalization scale must have length C",
        )));
    }
    if let Some(b) = bias {
        require_float(b, "GroupNormalization")?;
        if b.shape.total_elements() != c {
            return Err(OpError::ShapeMismatch(String::from(
                "GroupNormalization bias must have length C",
            )));
        }
    }
    let groups = num_groups as usize;
    let channels_per_group = c / groups;
    let group_size = channels_per_group * spatial;
    let total = x.shape.total_elements();
    let mut data = allocate_tensor_data(total, DataType::Float);

    for bn in 0..n {
        for g in 0..groups {
            let base = bn * c * spatial + g * channels_per_group * spatial;
            // Mean
            let mut sum = 0.0f32;
            for i in 0..group_size {
                sum += read_f32(&x.raw_data, base + i);
            }
            let mean = sum / group_size as f32;
            // Variance
            let mut var_sum = 0.0f32;
            for i in 0..group_size {
                let d = read_f32(&x.raw_data, base + i) - mean;
                var_sum += d * d;
            }
            let inv_std = 1.0 / sqrt_approx(var_sum / group_size as f32 + epsilon);
            // Apply per-channel scale/bias.
            for ci in 0..channels_per_group {
                let c_abs = g * channels_per_group + ci;
                let s = read_f32(&scale.raw_data, c_abs);
                let b = bias.map(|bt| read_f32(&bt.raw_data, c_abs)).unwrap_or(0.0);
                for sp in 0..spatial {
                    let idx = base + ci * spatial + sp;
                    let v = read_f32(&x.raw_data, idx);
                    write_f32(&mut data, idx, s * (v - mean) * inv_std + b);
                }
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: x.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

/// Instance normalization = GroupNormalization with `num_groups == C`.
pub fn op_instance_normalization(
    x: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    if x.shape.dims.len() < 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "InstanceNormalization requires rank >= 2",
        )));
    }
    let c = x.shape.dims[1];
    op_group_normalization(x, scale, Some(bias), c, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

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
    fn test_group_norm_one_group_zero_mean() {
        // [1,2,1,2] with values [1,2, 3,4]; one group over all 4 values.
        let x = make_f32(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let scale = make_f32(&[2], &[1.0, 1.0]);
        let out = op_group_normalization(&x, &scale, None, 1, 1e-5).unwrap();
        let v = read_all(&out);
        // Normalized values should sum to ~0.
        let s: f32 = v.iter().sum();
        assert!(s.abs() < 1e-3);
    }

    #[test]
    fn test_group_norm_scale_bias_applied() {
        let x = make_f32(&[1, 2, 1, 1], &[1.0, 3.0]);
        let scale = make_f32(&[2], &[2.0, 2.0]);
        let bias = make_f32(&[2], &[0.5, 0.5]);
        let out = op_group_normalization(&x, &scale, Some(&bias), 1, 1e-5).unwrap();
        let v = read_all(&out);
        // Mean 2, var 1 → normalized [-1,1] * 2 + 0.5 = [-1.5, 2.5]
        assert!((v[0] + 1.5).abs() < 1e-2);
        assert!((v[1] - 2.5).abs() < 1e-2);
    }

    #[test]
    fn test_group_norm_rejects_bad_group_count() {
        let x = make_f32(&[1, 3, 1, 1], &[1.0, 2.0, 3.0]);
        let scale = make_f32(&[3], &[1.0, 1.0, 1.0]);
        assert!(op_group_normalization(&x, &scale, None, 2, 1e-5).is_err());
    }

    #[test]
    fn test_instance_norm_per_channel() {
        // Two channels with distinct stats — each should be normalized
        // independently.
        let x = make_f32(&[1, 2, 1, 2], &[1.0, 3.0, 10.0, 20.0]);
        let scale = make_f32(&[2], &[1.0, 1.0]);
        let bias = make_f32(&[2], &[0.0, 0.0]);
        let out = op_instance_normalization(&x, &scale, &bias, 1e-5).unwrap();
        let v = read_all(&out);
        // Channel 0: mean 2 → [-1, 1]. Channel 1: mean 15 → [-1, 1].
        assert!((v[0] + 1.0).abs() < 1e-2);
        assert!((v[1] - 1.0).abs() < 1e-2);
        assert!((v[2] + 1.0).abs() < 1e-2);
        assert!((v[3] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_instance_norm_rejects_bad_rank() {
        let x = make_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let scale = make_f32(&[1], &[1.0]);
        let bias = make_f32(&[1], &[0.0]);
        assert!(op_instance_normalization(&x, &scale, &bias, 1e-5).is_err());
    }
}
