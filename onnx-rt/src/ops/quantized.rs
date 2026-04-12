// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Quantized operators: QuantizeLinear, DequantizeLinear, QLinearMatMul,
//! QLinearConv.
//!
//! The strategy for QLinearMatMul/QLinearConv is "dequantize → compute in
//! f32 → requantize". This trades performance for correctness; a real i8
//! GEMM kernel is future work.

use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
#[cfg(test)]
use crate::operators::op_matmul;
use crate::operators::{op_conv, OpError};
use crate::tensor::{DataType, Tensor};

/// Round-half-away-from-zero without depending on `f32::round` (unavailable
/// in `no_std`).
#[inline]
fn round_f32(x: f32) -> f32 {
    if x >= 0.0 {
        (x + 0.5) as i64 as f32
    } else {
        (x - 0.5) as i64 as f32
    }
}

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
///
/// Per-axis quantization is not yet implemented; any scale tensor with
/// more than one element is rejected explicitly rather than silently
/// collapsing to the first element. Non-positive scales are also
/// rejected.
fn read_scale(scale: &Tensor) -> Result<f32, OpError> {
    if scale.data_type != DataType::Float || scale.raw_data.len() < 4 {
        return Err(OpError::ShapeMismatch(String::from(
            "quantized: scale must be Float",
        )));
    }
    if scale.shape.total_elements() > 1 {
        return Err(OpError::InvalidAttribute(String::from(
            "quantized: per-axis scale not yet implemented",
        )));
    }
    let s = read_f32(&scale.raw_data, 0);
    if s.is_nan() || s <= 0.0 {
        return Err(OpError::InvalidAttribute(String::from(
            "quantized: scale must be positive and finite",
        )));
    }
    Ok(s)
}

/// Returns the zero-point as an i32 (read from int8 / uint8 / int32).
///
/// Per-axis zero points are rejected with the same error as per-axis
/// scales, since a correct per-axis implementation would require
/// coordinated changes to the matmul/conv kernels.
fn read_zero_point(zp: Option<&Tensor>) -> Result<i32, OpError> {
    match zp {
        None => Ok(0),
        Some(t) => {
            if t.shape.total_elements() > 1 {
                return Err(OpError::InvalidAttribute(String::from(
                    "quantized: per-axis zero_point not yet implemented",
                )));
            }
            match t.data_type {
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
            }
        }
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

/// Real int8 GEMM path for `QLinearMatMul`.
///
/// Phase 2 (`generative-llm-v1`) replaces the previous
/// "dequantize -> f32 MatMul -> requantize" shim with a tiled i32-accumulator
/// kernel that reads bytes as i8/u8, folds zero-points at the edges, and
/// saturates on store. The shared `gemm_i8` implementation lives in
/// `ops::generative` (see `docs/sub-graph-executor-design.md` D6 for the
/// mathematical derivation).
///
/// Only 2D inputs are supported; per-axis scales/zero-points are still
/// rejected. Output dtype is `Int8` if `y_zp` is `Int8`, otherwise `Uint8`.
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
    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "QLinearMatMul requires 2D inputs",
        )));
    }
    if !matches!(a.data_type, DataType::Int8 | DataType::Uint8) {
        return Err(OpError::ShapeMismatch(String::from(
            "QLinearMatMul: A must be Int8 or Uint8",
        )));
    }
    if !matches!(b.data_type, DataType::Int8 | DataType::Uint8) {
        return Err(OpError::ShapeMismatch(String::from(
            "QLinearMatMul: B must be Int8 or Uint8",
        )));
    }
    let m = a.shape.dims[0] as usize;
    let ka = a.shape.dims[1] as usize;
    let kb = b.shape.dims[0] as usize;
    let n = b.shape.dims[1] as usize;
    if ka != kb {
        return Err(OpError::ShapeMismatch(String::from(
            "QLinearMatMul: inner dim mismatch",
        )));
    }

    let a_s = read_scale(a_scale)?;
    let b_s = read_scale(b_scale)?;
    let y_s = read_scale(y_scale)?;
    let a_zp_i = read_zero_point(Some(a_zp))?;
    let b_zp_i = read_zero_point(Some(b_zp))?;
    let y_zp_i = read_zero_point(Some(y_zp))?;

    // Shift u8 inputs into the signed [-128, 127] domain so the i8 kernel
    // sees them correctly. For u8 with zero-point `zp_u`, subtracting 128
    // from both the byte and the zero-point is equivalent to keeping the
    // original arithmetic (the `zp` term in the kernel's subtraction just
    // rebases).
    let (a_bytes, a_zp_kernel): (Vec<i8>, i32) = match a.data_type {
        DataType::Int8 => (a.raw_data.iter().map(|&b| b as i8).collect(), a_zp_i),
        DataType::Uint8 => (
            a.raw_data
                .iter()
                .map(|&b| ((b as i32) - 128) as i8)
                .collect(),
            a_zp_i - 128,
        ),
        _ => unreachable!(),
    };
    let (b_bytes, b_zp_kernel): (Vec<i8>, i32) = match b.data_type {
        DataType::Int8 => (b.raw_data.iter().map(|&b| b as i8).collect(), b_zp_i),
        DataType::Uint8 => (
            b.raw_data
                .iter()
                .map(|&b| ((b as i32) - 128) as i8)
                .collect(),
            b_zp_i - 128,
        ),
        _ => unreachable!(),
    };

    let c_int =
        crate::ops::generative::gemm_i8(&a_bytes, &b_bytes, m, ka, n, a_zp_kernel, b_zp_kernel);

    // Scale the i32 accumulator to the output domain and requantize.
    //
    //   y = clamp(round(c_int * a_s * b_s / y_s) + y_zp)
    //
    // We compute this in f32 which is good enough for the target ±1
    // quantized-step accuracy. A future pass can replace with a fixed-
    // point multiplier for DO-178C determinism.
    let scale_ratio = (a_s * b_s) / y_s;
    let mut data = alloc::vec![0u8; m * n];
    let out_dtype = y_zp.data_type;
    match out_dtype {
        DataType::Int8 => {
            for (i, slot) in data.iter_mut().enumerate().take(m * n) {
                let scaled = round_f32(c_int[i] as f32 * scale_ratio) as i32 + y_zp_i;
                *slot = scaled.clamp(-128, 127) as i8 as u8;
            }
        }
        DataType::Uint8 => {
            for (i, slot) in data.iter_mut().enumerate().take(m * n) {
                let scaled = round_f32(c_int[i] as f32 * scale_ratio) as i32 + y_zp_i;
                *slot = scaled.clamp(0, 255) as u8;
            }
        }
        _ => {
            return Err(OpError::ShapeMismatch(String::from(
                "QLinearMatMul: y_zero_point must be Int8 or Uint8",
            )));
        }
    }

    Ok(Tensor {
        data_type: out_dtype,
        shape: crate::tensor::TensorShape::new(alloc::vec![m as i64, n as i64]),
        name: String::new(),
        raw_data: data,
    })
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
    pads: Option<&[i64]>,
    strides: Option<&[i64]>,
    dilations: Option<&[i64]>,
    group: i64,
) -> Result<Tensor, OpError> {
    // op_conv currently only supports unpadded, unit-stride, unit-dilation,
    // group=1 convolutions. Rather than silently dropping these attributes
    // (the previous behavior), reject non-default values up-front so
    // callers know the result would be wrong. Phase 4 will widen op_conv.
    if let Some(p) = pads {
        if p.iter().any(|&v| v != 0) {
            return Err(OpError::InvalidAttribute(String::from(
                "QLinearConv: non-zero pads not yet supported",
            )));
        }
    }
    if let Some(s) = strides {
        if s.iter().any(|&v| v != 1) {
            return Err(OpError::InvalidAttribute(String::from(
                "QLinearConv: non-unit strides not yet supported",
            )));
        }
    }
    if let Some(d) = dilations {
        if d.iter().any(|&v| v != 1) {
            return Err(OpError::InvalidAttribute(String::from(
                "QLinearConv: non-unit dilations not yet supported",
            )));
        }
    }
    if group != 1 {
        return Err(OpError::InvalidAttribute(String::from(
            "QLinearConv: group != 1 not yet supported",
        )));
    }
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
    use crate::byte_io::write_i32;
    use crate::ops::common::test_helpers::{make_f32, scalar_f32, scalar_i8, scalar_u8};
    use crate::tensor::TensorShape;
    use alloc::vec::Vec;

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
    fn test_quantize_per_axis_scale_rejected() {
        // Multi-element scale must be rejected, not silently truncated.
        let x = make_f32(&[4], &[0.0, 0.1, 0.2, 0.3]);
        let mut raw = alloc::vec![0u8; 8];
        write_f32(&mut raw, 0, 0.1);
        write_f32(&mut raw, 1, 0.2);
        let s = Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(alloc::vec![2]),
            name: String::new(),
            raw_data: raw,
        };
        let zp = scalar_i8(0);
        assert!(matches!(
            op_quantize_linear(&x, &s, Some(&zp)),
            Err(OpError::InvalidAttribute(_))
        ));
    }

    #[test]
    fn test_quantize_negative_scale_rejected() {
        let x = make_f32(&[1], &[1.0]);
        let s = scalar_f32(-0.1);
        let zp = scalar_i8(0);
        assert!(matches!(
            op_quantize_linear(&x, &s, Some(&zp)),
            Err(OpError::InvalidAttribute(_))
        ));
    }

    #[test]
    fn test_qlinear_conv_rejects_non_default_attrs() {
        let x = make_f32(&[1, 1, 3, 3], &alloc::vec![0.0f32; 9]);
        let w = make_f32(&[1, 1, 2, 2], &alloc::vec![0.0f32; 4]);
        let s = scalar_f32(0.1);
        let zp = scalar_i8(0);
        let xq = op_quantize_linear(&x, &s, Some(&zp)).unwrap();
        let wq = op_quantize_linear(&w, &s, Some(&zp)).unwrap();
        // Non-unit strides must be rejected (not silently dropped).
        assert!(op_qlinear_conv(
            &xq,
            &s,
            &zp,
            &wq,
            &s,
            &zp,
            &s,
            &zp,
            None,
            None,
            Some(&[2, 2]),
            None,
            1
        )
        .is_err());
        // Non-zero pads must be rejected.
        assert!(op_qlinear_conv(
            &xq,
            &s,
            &zp,
            &wq,
            &s,
            &zp,
            &s,
            &zp,
            None,
            Some(&[1, 1, 1, 1]),
            None,
            None,
            1
        )
        .is_err());
    }

    // ---- Real i8 kernel correctness tests ----

    #[test]
    fn test_qlinear_matmul_symmetric_int8_matches_f32_reference() {
        // Uses a larger K that crosses the 64-wide tile boundary so the
        // blocked kernel is exercised.
        let m = 4;
        let k = 80;
        let n = 3;
        let a_f: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b_f: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02).cos()).collect();
        let a_tensor = Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(alloc::vec![m as i64, k as i64]),
            name: String::new(),
            raw_data: {
                let mut r = alloc::vec![0u8; 4 * m * k];
                for (i, &v) in a_f.iter().enumerate() {
                    write_f32(&mut r, i, v);
                }
                r
            },
        };
        let b_tensor = Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(alloc::vec![k as i64, n as i64]),
            name: String::new(),
            raw_data: {
                let mut r = alloc::vec![0u8; 4 * k * n];
                for (i, &v) in b_f.iter().enumerate() {
                    write_f32(&mut r, i, v);
                }
                r
            },
        };

        let s = scalar_f32(0.01);
        let zp = scalar_i8(0);
        let a_q = op_quantize_linear(&a_tensor, &s, Some(&zp)).unwrap();
        let b_q = op_quantize_linear(&b_tensor, &s, Some(&zp)).unwrap();

        // Pick a y_scale that keeps outputs in [-128, 127] after requantize
        // for the K=80 dot-product range (~ ±15 with these inputs).
        let y_s = scalar_f32(0.2);
        let y_zp = scalar_i8(0);
        let y_q = op_qlinear_matmul(&a_q, &s, &zp, &b_q, &s, &zp, &y_s, &y_zp).unwrap();

        let y_f_ref = op_matmul(&a_tensor, &b_tensor).unwrap();
        let y_dq = op_dequantize_linear(&y_q, &y_s, Some(&y_zp)).unwrap();
        for i in 0..(m * n) {
            let r = read_f32(&y_f_ref.raw_data, i);
            let d = read_f32(&y_dq.raw_data, i);
            // Error tolerance includes quantization noise plus requantization.
            let tol = r.abs() * 0.1 + 0.4;
            assert!((r - d).abs() <= tol, "r={} dq={} tol={}", r, d, tol);
        }
    }

    #[test]
    fn test_qlinear_matmul_saturates_overflow() {
        // Values that blow past i8 range in the accumulator should saturate.
        let a = Tensor {
            data_type: DataType::Int8,
            shape: TensorShape::new(alloc::vec![1, 4]),
            name: String::new(),
            raw_data: alloc::vec![127u8, 127, 127, 127],
        };
        let b = Tensor {
            data_type: DataType::Int8,
            shape: TensorShape::new(alloc::vec![4, 1]),
            name: String::new(),
            raw_data: alloc::vec![127u8, 127, 127, 127],
        };
        // Use a small y_scale so the requantized value blows past i8.
        let s = scalar_f32(1.0);
        let zp = scalar_i8(0);
        let y_s = scalar_f32(0.001);
        let y_zp = scalar_i8(0);
        let out = op_qlinear_matmul(&a, &s, &zp, &b, &s, &zp, &y_s, &y_zp).unwrap();
        assert_eq!(out.raw_data[0] as i8, 127);
    }

    #[test]
    fn test_qlinear_matmul_asymmetric_zero_points() {
        // A quantized with u8/zero_point=128, B with i8/zero_point=0.
        // We verify the dequant-quant path still matches via roundtrip.
        let a_f = Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(alloc::vec![2, 2]),
            name: String::new(),
            raw_data: {
                let mut r = alloc::vec![0u8; 16];
                write_f32(&mut r, 0, 0.5);
                write_f32(&mut r, 1, -0.5);
                write_f32(&mut r, 2, 1.0);
                write_f32(&mut r, 3, 0.0);
                r
            },
        };
        let b_f = Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(alloc::vec![2, 2]),
            name: String::new(),
            raw_data: {
                let mut r = alloc::vec![0u8; 16];
                write_f32(&mut r, 0, 1.0);
                write_f32(&mut r, 1, 0.0);
                write_f32(&mut r, 2, 0.0);
                write_f32(&mut r, 3, 1.0);
                r
            },
        };
        let a_s = scalar_f32(0.01);
        let a_zp = scalar_u8(128);
        let b_s = scalar_f32(0.01);
        let b_zp = scalar_i8(0);
        let a_q = op_quantize_linear(&a_f, &a_s, Some(&a_zp)).unwrap();
        let b_q = op_quantize_linear(&b_f, &b_s, Some(&b_zp)).unwrap();

        let y_s = scalar_f32(0.01);
        let y_zp = scalar_i8(0);
        let y_q = op_qlinear_matmul(&a_q, &a_s, &a_zp, &b_q, &b_s, &b_zp, &y_s, &y_zp).unwrap();
        let y_f_ref = op_matmul(&a_f, &b_f).unwrap();
        let y_dq = op_dequantize_linear(&y_q, &y_s, Some(&y_zp)).unwrap();
        for i in 0..4 {
            let r = read_f32(&y_f_ref.raw_data, i);
            let d = read_f32(&y_dq.raw_data, i);
            assert!((r - d).abs() <= 0.1, "r={} dq={}", r, d);
        }
    }

    #[test]
    fn test_qlinear_matmul_tile_boundary_k() {
        // K exactly equal to and just past the 64-wide tile; validates both
        // the tile-complete and tile-remainder code paths.
        for k in [64usize, 65, 128, 129] {
            let a: Vec<i8> = (0..(2 * k)).map(|i| ((i % 7) as i8) - 3).collect();
            let b: Vec<i8> = (0..(k * 2)).map(|i| ((i % 5) as i8) - 2).collect();
            let a_t = Tensor {
                data_type: DataType::Int8,
                shape: TensorShape::new(alloc::vec![2, k as i64]),
                name: String::new(),
                raw_data: a.iter().map(|&v| v as u8).collect(),
            };
            let b_t = Tensor {
                data_type: DataType::Int8,
                shape: TensorShape::new(alloc::vec![k as i64, 2]),
                name: String::new(),
                raw_data: b.iter().map(|&v| v as u8).collect(),
            };
            let s = scalar_f32(1.0);
            let zp = scalar_i8(0);
            let y_s = scalar_f32(1.0);
            let y_zp = scalar_i8(0);
            let out = op_qlinear_matmul(&a_t, &s, &zp, &b_t, &s, &zp, &y_s, &y_zp).unwrap();

            // Naive reference (clamp to i8 since y_zp = 0 and y_s = 1).
            let mut expected = alloc::vec![0i32; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut acc = 0i32;
                    for kk in 0..k {
                        acc += (a[i * k + kk] as i32) * (b[kk * 2 + j] as i32);
                    }
                    expected[i * 2 + j] = acc;
                }
            }
            for (i, &exp_raw) in expected.iter().enumerate() {
                let got = out.raw_data[i] as i8 as i32;
                let exp = exp_raw.clamp(-128, 127);
                assert_eq!(got, exp, "k={} i={} exp={} got={}", k, i, exp, got);
            }
        }
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
