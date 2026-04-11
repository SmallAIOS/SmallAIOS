// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tier 2 element-wise math primitives.
//!
//! Implements Pow, Sqrt, Exp, Log, Erf, Neg, Abs, Floor, Ceil, Round.
//! All operators consume f32 tensors and produce f32 tensors. Pow supports
//! NumPy-style broadcasting; the rest are unary element-wise.

use alloc::string::String;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
use crate::operators::{expf_approx, sqrt_approx, OpError};
use crate::tensor::{DataType, Tensor, TensorShape};

/// Returns the broadcast output shape for two NumPy-style shapes.
fn broadcast_shape(a: &[i64], b: &[i64]) -> Result<alloc::vec::Vec<i64>, OpError> {
    let max = a.len().max(b.len());
    let mut out = alloc::vec![1i64; max];
    for i in 0..max {
        let ad = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let bd = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        let dim = if ad == bd {
            ad
        } else if ad == 1 {
            bd
        } else if bd == 1 {
            ad
        } else {
            return Err(OpError::ShapeMismatch(String::from(
                "incompatible shapes for broadcast",
            )));
        };
        out[max - 1 - i] = dim;
    }
    Ok(out)
}

/// Computes the linear index for a coordinate in a (broadcast) tensor.
fn broadcast_index(coord: &[usize], shape: &[i64]) -> usize {
    // Right-aligned. coord is in output space; shape is the input shape.
    let off = coord.len() - shape.len();
    let mut idx = 0usize;
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        let c = if shape[i] == 1 { 0 } else { coord[off + i] };
        idx += c * stride;
        stride *= shape[i] as usize;
    }
    idx
}

fn next_coord(coord: &mut [usize], dims: &[i64]) {
    for i in (0..coord.len()).rev() {
        coord[i] += 1;
        if (coord[i] as i64) < dims[i] {
            return;
        }
        coord[i] = 0;
    }
}

fn require_float(t: &Tensor, op: &str) -> Result<(), OpError> {
    if t.data_type != DataType::Float {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{} requires Float input",
            op
        )));
    }
    Ok(())
}

/// Element-wise unary helper that allocates an output tensor with the same
/// shape and applies a function to each element.
fn unary<F: Fn(f32) -> f32>(input: &Tensor, op: &str, f: F) -> Result<Tensor, OpError> {
    require_float(input, op)?;
    let n = input.shape.total_elements();
    let mut out = allocate_tensor_data(n, DataType::Float);
    for i in 0..n {
        write_f32(&mut out, i, f(read_f32(&input.raw_data, i)));
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: out,
    })
}

/// Natural log via `ln(x) = log2(x) * ln(2)`. Uses the IEEE-754 exponent
/// for `log2` and a 5-term polynomial in the mantissa. Accurate to ~2 ULP
/// for typical inference ranges.
fn lnf_approx(x: f32) -> f32 {
    if x <= 0.0 {
        // Return -inf (or NaN for negatives) — match libm semantics loosely.
        if x == 0.0 {
            return f32::NEG_INFINITY;
        }
        return f32::NAN;
    }
    let bits = x.to_bits();
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa_bits = (bits & 0x7f_ffff) | 0x3f80_0000; // m in [1, 2)
    let mut m = f32::from_bits(mantissa_bits);
    // Range-reduce m to [sqrt(0.5), sqrt(2)] ≈ [0.707, 1.414] so the
    // artanh series converges much faster.
    if m > core::f32::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    // ln(m) = 2 * artanh((m-1)/(m+1))
    // artanh(y) = y + y^3/3 + y^5/5 + y^7/7 + y^9/9 + ...
    // For m in [0.707, 1.414], y is in [-0.172, 0.172]; a 5-term series is
    // accurate to ~1e-7.
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let y3 = y2 * y;
    let y5 = y3 * y2;
    let y7 = y5 * y2;
    let y9 = y7 * y2;
    let ln_m = 2.0 * (y + y3 / 3.0 + y5 / 5.0 + y7 / 7.0 + y9 / 9.0);
    (exp as f32) * core::f32::consts::LN_2 + ln_m
}

/// Error function via the Abramowitz & Stegun 7.1.26 polynomial
/// approximation. Maximum error ~1.5e-7.
pub(crate) fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = if x < 0.0 { -x } else { x };
    // Abramowitz & Stegun 7.1.26
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_74_f32;
    let a3 = 1.421_413_7_f32;
    let a4 = -1.453_152_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.327_591_1_f32;
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * expf_approx(-ax * ax);
    sign * y
}

// ---------------------------------------------------------------------------
// Public operators
// ---------------------------------------------------------------------------

/// Element-wise power with broadcasting: `out[i] = base[i] ^ exp[i]`.
pub fn op_pow(base: &Tensor, exponent: &Tensor) -> Result<Tensor, OpError> {
    require_float(base, "Pow")?;
    require_float(exponent, "Pow")?;
    let out_dims = broadcast_shape(&base.shape.dims, &exponent.shape.dims)?;
    let total: usize = out_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let mut data = allocate_tensor_data(total, DataType::Float);
    let ndim = out_dims.len();
    let mut coord = alloc::vec![0usize; ndim];
    for flat in 0..total {
        let bi = broadcast_index(&coord, &base.shape.dims);
        let ei = broadcast_index(&coord, &exponent.shape.dims);
        let b = read_f32(&base.raw_data, bi);
        let e = read_f32(&exponent.raw_data, ei);
        // pow(x, y) = exp(y * ln(x)) for positive x; handle integer y for negatives.
        let v = if b == 0.0 {
            if e == 0.0 {
                1.0
            } else {
                0.0
            }
        } else if b < 0.0 {
            // Only support integer exponents for negative bases.
            let e_int = e as i32;
            if (e_int as f32 - e).abs() < 1e-6 {
                let mut p = 1.0_f32;
                let n = e_int.unsigned_abs();
                for _ in 0..n {
                    p *= b;
                }
                if e_int < 0 {
                    1.0 / p
                } else {
                    p
                }
            } else {
                f32::NAN
            }
        } else {
            expf_approx(e * lnf_approx(b))
        };
        write_f32(&mut data, flat, v);
        if flat + 1 < total {
            next_coord(&mut coord, &out_dims);
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(out_dims),
        name: String::new(),
        raw_data: data,
    })
}

/// Element-wise square root.
pub fn op_sqrt(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Sqrt", sqrt_approx)
}

/// Element-wise natural exponential.
pub fn op_exp(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Exp", expf_approx)
}

/// Element-wise natural logarithm.
pub fn op_log(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Log", lnf_approx)
}

/// Element-wise error function.
pub fn op_erf(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Erf", erf_approx)
}

/// Element-wise negation.
pub fn op_neg(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Neg", |x| -x)
}

/// Element-wise absolute value.
pub fn op_abs(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Abs", |x| if x < 0.0 { -x } else { x })
}

/// Element-wise floor.
pub fn op_floor(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Floor", floor_f32)
}

/// Element-wise ceiling.
pub fn op_ceil(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Ceil", ceil_f32)
}

/// Element-wise round (banker's rounding to nearest, ties to even).
pub fn op_round(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Round", round_half_even)
}

fn floor_f32(x: f32) -> f32 {
    let i = x as i64 as f32;
    if x < 0.0 && i != x {
        i - 1.0
    } else {
        i
    }
}

fn ceil_f32(x: f32) -> f32 {
    let i = x as i64 as f32;
    if x > 0.0 && i != x {
        i + 1.0
    } else {
        i
    }
}

fn round_half_even(x: f32) -> f32 {
    let f = floor_f32(x);
    let frac = x - f;
    if frac < 0.5 {
        f
    } else if frac > 0.5 {
        f + 1.0
    } else {
        // Tie: round to even
        let fi = f as i64;
        if fi % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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
    fn test_sqrt_basic() {
        let t = make_f32(&[3], &[4.0, 9.0, 16.0]);
        let out = op_sqrt(&t).unwrap();
        let v = read_all(&out);
        assert!((v[0] - 2.0).abs() < 1e-3);
        assert!((v[1] - 3.0).abs() < 1e-3);
        assert!((v[2] - 4.0).abs() < 1e-3);
    }

    #[test]
    fn test_sqrt_zero() {
        let t = make_f32(&[1], &[0.0]);
        let out = op_sqrt(&t).unwrap();
        assert_eq!(read_f32(&out.raw_data, 0), 0.0);
    }

    #[test]
    fn test_exp_basic() {
        let t = make_f32(&[3], &[0.0, 1.0, -1.0]);
        let out = op_exp(&t).unwrap();
        let v = read_all(&out);
        assert!((v[0] - 1.0).abs() < 1e-3);
        assert!((v[1] - core::f32::consts::E).abs() < 1e-3);
        assert!((v[2] - 1.0 / core::f32::consts::E).abs() < 1e-3);
    }

    #[test]
    fn test_log_basic() {
        let t = make_f32(&[3], &[1.0, core::f32::consts::E, 4.0]);
        let out = op_log(&t).unwrap();
        let v = read_all(&out);
        assert!(v[0].abs() < 1e-3);
        assert!((v[1] - 1.0).abs() < 1e-3);
        assert!((v[2] - 4.0_f32.ln()).abs() < 1e-3);
    }

    #[test]
    fn test_log_zero_returns_neg_inf() {
        let t = make_f32(&[1], &[0.0]);
        let out = op_log(&t).unwrap();
        assert!(read_f32(&out.raw_data, 0).is_infinite());
    }

    #[test]
    fn test_log_exp_round_trip() {
        let t = make_f32(&[5], &[0.5, 1.0, 2.0, 3.5, 7.7]);
        let l = op_log(&t).unwrap();
        let e = op_exp(&l).unwrap();
        let original = read_all(&t);
        let restored = read_all(&e);
        for i in 0..5 {
            assert!(
                (original[i] - restored[i]).abs() / original[i] < 1e-2,
                "round trip failed: {} vs {}",
                original[i],
                restored[i]
            );
        }
    }

    #[test]
    fn test_erf_known_values() {
        // erf(0)=0, erf(1)≈0.8427, erf(-1)≈-0.8427
        let t = make_f32(&[3], &[0.0, 1.0, -1.0]);
        let out = op_erf(&t).unwrap();
        let v = read_all(&out);
        assert!(v[0].abs() < 1e-4);
        assert!((v[1] - 0.8427).abs() < 1e-3);
        assert!((v[2] + 0.8427).abs() < 1e-3);
    }

    #[test]
    fn test_neg() {
        let t = make_f32(&[4], &[1.0, -2.0, 0.0, 3.5]);
        let v = read_all(&op_neg(&t).unwrap());
        assert_eq!(v, vec![-1.0, 2.0, -0.0, -3.5]);
    }

    #[test]
    fn test_abs() {
        let t = make_f32(&[4], &[1.0, -2.0, 0.0, -3.5]);
        let v = read_all(&op_abs(&t).unwrap());
        assert_eq!(v, vec![1.0, 2.0, 0.0, 3.5]);
    }

    #[test]
    fn test_floor() {
        let t = make_f32(&[5], &[1.7, -1.7, 0.0, 2.0, -2.0]);
        let v = read_all(&op_floor(&t).unwrap());
        assert_eq!(v, vec![1.0, -2.0, 0.0, 2.0, -2.0]);
    }

    #[test]
    fn test_ceil() {
        let t = make_f32(&[5], &[1.3, -1.3, 0.0, 2.0, -2.0]);
        let v = read_all(&op_ceil(&t).unwrap());
        assert_eq!(v, vec![2.0, -1.0, 0.0, 2.0, -2.0]);
    }

    #[test]
    fn test_round_half_even() {
        let t = make_f32(&[6], &[0.5, 1.5, 2.5, -0.5, 1.4, 1.6]);
        let v = read_all(&op_round(&t).unwrap());
        // 0.5 → 0 (even), 1.5 → 2 (even), 2.5 → 2 (even), -0.5 → 0
        assert_eq!(v, vec![0.0, 2.0, 2.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_pow_scalar() {
        let base = make_f32(&[3], &[2.0, 3.0, 4.0]);
        let exp = make_f32(&[3], &[2.0, 2.0, 2.0]);
        let v = read_all(&op_pow(&base, &exp).unwrap());
        assert!((v[0] - 4.0).abs() < 1e-2);
        assert!((v[1] - 9.0).abs() < 1e-2);
        assert!((v[2] - 16.0).abs() < 1e-2);
    }

    #[test]
    fn test_pow_broadcast() {
        let base = make_f32(&[2, 2], &[2.0, 3.0, 4.0, 5.0]);
        let exp = make_f32(&[1], &[2.0]);
        let out = op_pow(&base, &exp).unwrap();
        let v = read_all(&out);
        assert!((v[0] - 4.0).abs() < 1e-2);
        assert!((v[1] - 9.0).abs() < 1e-2);
        assert!((v[2] - 16.0).abs() < 1e-2);
        assert!((v[3] - 25.0).abs() < 1e-2);
    }

    #[test]
    fn test_pow_negative_integer_exponent() {
        let base = make_f32(&[2], &[-2.0, -3.0]);
        let exp = make_f32(&[2], &[3.0, 2.0]);
        let v = read_all(&op_pow(&base, &exp).unwrap());
        assert!((v[0] - -8.0).abs() < 1e-3);
        assert!((v[1] - 9.0).abs() < 1e-3);
    }

    #[test]
    fn test_pow_zero_zero_is_one() {
        let base = make_f32(&[1], &[0.0]);
        let exp = make_f32(&[1], &[0.0]);
        let v = read_all(&op_pow(&base, &exp).unwrap());
        assert_eq!(v[0], 1.0);
    }

    #[test]
    fn test_unary_rejects_non_float() {
        let t = Tensor {
            data_type: DataType::Int32,
            shape: TensorShape::new(vec![1]),
            name: String::new(),
            raw_data: alloc::vec![0u8; 4],
        };
        assert!(op_sqrt(&t).is_err());
        assert!(op_exp(&t).is_err());
        assert!(op_neg(&t).is_err());
    }
}
