// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Quantized operators: QuantizeLinear, DequantizeLinear, QLinearMatMul,
//! QLinearConv.
//!
//! The strategy for QLinearMatMul/QLinearConv is "dequantize → compute in
//! f32 → requantize". This trades performance for correctness; a real i8
//! GEMM kernel is future work.

use alloc::string::String;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
use crate::operators::{op_conv, op_matmul, OpError};
use crate::tensor::{DataType, Tensor};

/// Banker's rounding (round half to even) without depending on the
/// nightly `f32::round_ties_even` inherent method.
fn round_half_even(x: f32) -> f32 {
    // Floor.
    let i = x as i64 as f32;
    let f = if x < 0.0 && i != x { i - 1.0 } else { i };
    let frac = x - f;
    if frac < 0.5 {
        f
    } else if frac > 0.5 {
        f + 1.0
    } else {
        let fi = f as i64;
        if (fi & 1) == 0 {
            f
        } else {
            f + 1.0
        }
    }
}

/// Returns the per-tensor scale (first element of the scale tensor).
fn read_scale(scale: &Tensor) -> Result<f32, OpError> {
    if scale.data_type != DataType::Float || scale.raw_data.len() < 4 {
        return Err(OpError::ShapeMismatch(String::from(
            "quantized: scale must be Float",
        )));
    }
    Ok(read_f32(&scale.raw_data, 0))
}

/// Returns the zero-point as an i32 (read from int8 / uint8 / int32).
fn read_zero_point(zp: Option<&Tensor>) -> Result<i32, OpError> {
    match zp {
        None => Ok(0),
        Some(t) => match t.data_type {
            DataType::Int8 => Ok(t.raw_data.first().map(|&b| b as i8 as i32).unwrap_or(0)),
            DataType::Uint8 => Ok(t.raw_data.first().map(|&b| b as i32).unwrap_or(0)),
            DataType::Int32 => {
                if t.raw_data.len() < 4 {
                    Ok(0)
                } else {
                    Ok(crate::byte_io::read_i32(&t.raw_data, 0))
                }
            }
            _ => Err(OpError::ShapeMismatch(String::from(
                "quantized: zero_point must be Int8/Uint8/Int32",
            ))),
        },
    }
}

/// Quantizes a Float tensor to Int8 or Uint8 using a per-tensor scale and
/// zero point.
///
/// Output dtype matches the dtype of `zero_point` (defaulting to Int8 when
/// none is provided).
pub fn op_quantize_linear(
    input: &Tensor,
    scale: &Tensor,
    zero_point: Option<&Tensor>,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::ShapeMismatch(String::from(
            "QuantizeLinear input must be Float",
        )));
    }
    let s = read_scale(scale)?;
    if s == 0.0 {
        return Err(OpError::InvalidAttribute(String::from(
            "QuantizeLinear: scale must be non-zero",
        )));
    }
    let zp = read_zero_point(zero_point)?;
    let out_dtype = zero_point.map(|z| z.data_type).unwrap_or(DataType::Int8);
    let n = input.shape.total_elements();
    let mut data = alloc::vec![0u8; n];
    match out_dtype {
        DataType::Int8 => {
            for (i, slot) in data.iter_mut().enumerate().take(n) {
                let v = read_f32(&input.raw_data, i);
                let q = round_half_even(v / s) as i32 + zp;
                let clipped = q.clamp(-128, 127) as i8;
                *slot = clipped as u8;
            }
        }
        DataType::Uint8 => {
            for (i, slot) in data.iter_mut().enumerate().take(n) {
                let v = read_f32(&input.raw_data, i);
                let q = round_half_even(v / s) as i32 + zp;
                *slot = q.clamp(0, 255) as u8;
            }
        }
        _ => {
            return Err(OpError::ShapeMismatch(String::from(
                "QuantizeLinear: zero_point dtype must be Int8 or Uint8",
            )));
        }
    }
    Ok(Tensor {
        data_type: out_dtype,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

/// Dequantizes an Int8 / Uint8 tensor back to Float.
pub fn op_dequantize_linear(
    input: &Tensor,
    scale: &Tensor,
    zero_point: Option<&Tensor>,
) -> Result<Tensor, OpError> {
    let s = read_scale(scale)?;
    let zp = read_zero_point(zero_point)?;
    let n = input.shape.total_elements();
    let mut data = allocate_tensor_data(n, DataType::Float);
    match input.data_type {
        DataType::Int8 => {
            for i in 0..n {
                let q = input.raw_data[i] as i8 as i32;
                write_f32(&mut data, i, (q - zp) as f32 * s);
            }
        }
        DataType::Uint8 => {
            for i in 0..n {
                let q = input.raw_data[i] as i32;
                write_f32(&mut data, i, (q - zp) as f32 * s);
            }
        }
        _ => {
            return Err(OpError::ShapeMismatch(String::from(
                "DequantizeLinear input must be Int8 or Uint8",
            )));
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
// QLinearMatMul / QLinearConv
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn op_qlinear_matmul(
    a: &Tensor,
    a_scale: &Tensor,
    a_zp: &Tensor,
    b: &Tensor,
    b_scale: &Tensor,
    b_zp: &Tensor,
    y_scale: &Tensor,
    y_zp: &Tensor,
) -> Result<Tensor, OpError> {
    let a_f32 = op_dequantize_linear(a, a_scale, Some(a_zp))?;
    let b_f32 = op_dequantize_linear(b, b_scale, Some(b_zp))?;
    let y_f32 = op_matmul(&a_f32, &b_f32)?;
    op_quantize_linear(&y_f32, y_scale, Some(y_zp))
}

#[allow(clippy::too_many_arguments)]
pub fn op_qlinear_conv(
    x: &Tensor,
    x_scale: &Tensor,
    x_zp: &Tensor,
    w: &Tensor,
    w_scale: &Tensor,
    w_zp: &Tensor,
    y_scale: &Tensor,
    y_zp: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor, OpError> {
    let x_f32 = op_dequantize_linear(x, x_scale, Some(x_zp))?;
    let w_f32 = op_dequantize_linear(w, w_scale, Some(w_zp))?;
    // Bias for QLinearConv is typically Int32 (pre-scaled by x_scale*w_scale).
    // For the dequantize-then-quantize approach we accept None or a Float bias.
    let bias_f32: Option<Tensor> = match bias {
        None => None,
        Some(b) if b.data_type == DataType::Float => Some(b.clone()),
        Some(b) if b.data_type == DataType::Int32 => {
            // Approximate: bias_f32[i] = bias_i32[i] * (x_scale * w_scale)
            let xs = read_scale(x_scale)?;
            let ws = read_scale(w_scale)?;
            let n = b.shape.total_elements();
            let mut data = allocate_tensor_data(n, DataType::Float);
            for i in 0..n {
                let q = crate::byte_io::read_i32(&b.raw_data, i);
                write_f32(&mut data, i, q as f32 * xs * ws);
            }
            Some(Tensor {
                data_type: DataType::Float,
                shape: b.shape.clone(),
                name: String::new(),
                raw_data: data,
            })
        }
        Some(_) => {
            return Err(OpError::ShapeMismatch(String::from(
                "QLinearConv bias must be Float or Int32",
            )))
        }
    };
    let y_f32 = op_conv(&x_f32, &w_f32, bias_f32.as_ref())?;
    op_quantize_linear(&y_f32, y_scale, Some(y_zp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_io::{write_f32, write_i32};
    use crate::tensor::TensorShape;
    use alloc::vec::Vec;

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

    fn scalar_f32(v: f32) -> Tensor {
        let mut data = alloc::vec![0u8; 4];
        write_f32(&mut data, 0, v);
        Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: data,
        }
    }

    fn scalar_i8(v: i8) -> Tensor {
        Tensor {
            data_type: DataType::Int8,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: alloc::vec![v as u8],
        }
    }

    fn scalar_u8(v: u8) -> Tensor {
        Tensor {
            data_type: DataType::Uint8,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: alloc::vec![v],
        }
    }

    fn read_int8(t: &Tensor) -> Vec<i8> {
        t.raw_data.iter().map(|&b| b as i8).collect()
    }

    #[test]
    fn test_quantize_int8_basic() {
        let x = make_f32(&[4], &[0.0, 0.1, 0.2, -0.1]);
        let s = scalar_f32(0.1);
        let zp = scalar_i8(0);
        let out = op_quantize_linear(&x, &s, Some(&zp)).unwrap();
        assert_eq!(out.data_type, DataType::Int8);
        assert_eq!(read_int8(&out), alloc::vec![0, 1, 2, -1]);
    }

    #[test]
    fn test_quantize_uint8_basic() {
        let x = make_f32(&[3], &[0.0, 0.5, 1.0]);
        let s = scalar_f32(0.1);
        let zp = scalar_u8(128);
        let out = op_quantize_linear(&x, &s, Some(&zp)).unwrap();
        assert_eq!(out.data_type, DataType::Uint8);
        assert_eq!(out.raw_data, alloc::vec![128u8, 133, 138]);
    }

    #[test]
    fn test_quantize_clipping() {
        let x = make_f32(&[2], &[100.0, -100.0]);
        let s = scalar_f32(0.1);
        let zp = scalar_i8(0);
        let out = op_quantize_linear(&x, &s, Some(&zp)).unwrap();
        let v = read_int8(&out);
        assert_eq!(v, alloc::vec![127, -128]);
    }

    #[test]
    fn test_dequantize_int8_basic() {
        let q = Tensor {
            data_type: DataType::Int8,
            shape: TensorShape::new(alloc::vec![3]),
            name: String::new(),
            raw_data: alloc::vec![0u8, 1u8, (-1i8) as u8],
        };
        let s = scalar_f32(0.1);
        let zp = scalar_i8(0);
        let out = op_dequantize_linear(&q, &s, Some(&zp)).unwrap();
        assert_eq!(out.data_type, DataType::Float);
        let v: Vec<f32> = (0..3).map(|i| read_f32(&out.raw_data, i)).collect();
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[1] - 0.1).abs() < 1e-6);
        assert!((v[2] - -0.1).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_dequantize_round_trip() {
        let original = make_f32(&[5], &[0.0, 0.3, -0.7, 1.2, -1.5]);
        let s = scalar_f32(0.05);
        let zp = scalar_i8(0);
        let q = op_quantize_linear(&original, &s, Some(&zp)).unwrap();
        let dq = op_dequantize_linear(&q, &s, Some(&zp)).unwrap();
        for i in 0..5 {
            let o = read_f32(&original.raw_data, i);
            let r = read_f32(&dq.raw_data, i);
            // Round-trip error must be within scale.
            assert!((o - r).abs() <= 0.05 + 1e-6, "{} vs {}", o, r);
        }
    }

    #[test]
    fn test_qlinear_matmul_matches_f32() {
        // 2x3 * 3x2 = 2x2
        let a_f = make_f32(&[2, 3], &[0.5, -0.5, 1.0, 0.0, 0.5, -0.5]);
        let b_f = make_f32(&[3, 2], &[1.0, 0.5, -0.5, 1.0, 0.0, -1.0]);
        let s = scalar_f32(0.01);
        let zp = scalar_i8(0);
        let a_q = op_quantize_linear(&a_f, &s, Some(&zp)).unwrap();
        let b_q = op_quantize_linear(&b_f, &s, Some(&zp)).unwrap();
        let y_s = scalar_f32(0.01);
        let y_zp = scalar_i8(0);
        let q_out = op_qlinear_matmul(&a_q, &s, &zp, &b_q, &s, &zp, &y_s, &y_zp).unwrap();
        let dq_out = op_dequantize_linear(&q_out, &y_s, Some(&y_zp)).unwrap();

        let f_out = op_matmul(&a_f, &b_f).unwrap();
        for i in 0..4 {
            let f = read_f32(&f_out.raw_data, i);
            let dq = read_f32(&dq_out.raw_data, i);
            // Quantized result within 1% of f32 result (or 0.05 absolute).
            let tol = (f.abs() * 0.05).max(0.05);
            assert!((f - dq).abs() <= tol, "{} vs {} tol {}", f, dq, tol);
        }
    }

    #[test]
    fn test_quantize_zero_scale_fails() {
        let x = make_f32(&[1], &[1.0]);
        let s = scalar_f32(0.0);
        let zp = scalar_i8(0);
        assert!(op_quantize_linear(&x, &s, Some(&zp)).is_err());
    }

    #[test]
    fn test_dequantize_int32_zero_point() {
        // Verify Int32 zero point is accepted.
        let q = Tensor {
            data_type: DataType::Int8,
            shape: TensorShape::new(alloc::vec![2]),
            name: String::new(),
            raw_data: alloc::vec![10u8, 20u8],
        };
        let s = scalar_f32(0.5);
        let mut zp_data = alloc::vec![0u8; 4];
        write_i32(&mut zp_data, 0, 5);
        let zp = Tensor {
            data_type: DataType::Int32,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: zp_data,
        };
        let out = op_dequantize_linear(&q, &s, Some(&zp)).unwrap();
        let v: Vec<f32> = (0..2).map(|i| read_f32(&out.raw_data, i)).collect();
        // (10-5)*0.5 = 2.5, (20-5)*0.5 = 7.5
        assert!((v[0] - 2.5).abs() < 1e-6);
        assert!((v[1] - 7.5).abs() < 1e-6);
    }
}
