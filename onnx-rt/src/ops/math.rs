// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tier 2 / Tier 3 element-wise math primitives.
//!
//! Implements Pow, Sqrt, Exp, Log, Erf, Neg, Abs, Floor, Ceil, Round and
//! the Phase 1 transformer math additions (Mod, Sin, Cos, Reciprocal,
//! Sign, Sum, Mean, And, Or, LogSoftmax). All operators consume f32 (or
//! Bool) tensors. Binary/variadic ops use NumPy-style broadcasting; unary
//! ops preserve input shape.

use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
use crate::operators::{expf_approx, sqrt_approx, OpError};
use crate::ops::common::{broadcast_index, broadcast_shape, next_coord, require_float, unary};
use crate::tensor::{DataType, Tensor, TensorShape};

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

/// Computes `b ^ e` for the special case where `b == 0`, matching the
/// behavior expected by ONNX Pow: `0^0 = 1`, `0^neg = +inf`, `0^pos = 0`.
fn pow_zero_base(e: f32) -> f32 {
    if e == 0.0 {
        1.0
    } else if e < 0.0 {
        f32::INFINITY
    } else {
        0.0
    }
}

/// Computes `b ^ e` for negative `b`, supporting only integer exponents.
/// Non-integer exponents on a negative base return NaN.
fn pow_negative_base(b: f32, e: f32) -> f32 {
    let e_int = e as i32;
    if (e_int as f32 - e).abs() >= 1e-6 {
        return f32::NAN;
    }
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
}

/// Computes a single element-wise `pow(b, e)` matching ONNX semantics.
fn pow_one(b: f32, e: f32) -> f32 {
    if b == 0.0 {
        pow_zero_base(e)
    } else if b < 0.0 {
        pow_negative_base(b, e)
    } else {
        expf_approx(e * lnf_approx(b))
    }
}

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
        write_f32(&mut data, flat, pow_one(b, e));
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
///
/// Returns `NaN` for negative inputs (matching IEEE semantics). The
/// internal `sqrt_approx` clamps negatives to 0 for other callers, so
/// we handle the negative case here before delegating.
pub fn op_sqrt(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Sqrt", |x| {
        if x < 0.0 {
            f32::NAN
        } else {
            sqrt_approx(x)
        }
    })
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

// ---------------------------------------------------------------------------
// Phase 1 transformer math ops
// ---------------------------------------------------------------------------

/// Range-reduced degree-7 sine approximation on the full real line.
/// Accurate to ~1e-5 for typical transformer activation ranges.
pub(crate) fn sinf_approx(x: f32) -> f32 {
    // Reduce to [-pi, pi] using floor (truncation toward zero is wrong
    // for negative x).
    let two_pi = 2.0 * core::f32::consts::PI;
    let mut y = x - floor_f32((x + core::f32::consts::PI) / two_pi) * two_pi;
    // y is now in [-pi, pi]; clamp any numeric fuzz.
    if y > core::f32::consts::PI {
        y -= two_pi;
    } else if y < -core::f32::consts::PI {
        y += two_pi;
    }
    // Reduce further using sin(pi - y) = sin(y)
    if y > core::f32::consts::FRAC_PI_2 {
        y = core::f32::consts::PI - y;
    } else if y < -core::f32::consts::FRAC_PI_2 {
        y = -core::f32::consts::PI - y;
    }
    // Taylor series around 0: y - y^3/6 + y^5/120 - y^7/5040
    let y2 = y * y;
    let y3 = y2 * y;
    let y5 = y3 * y2;
    let y7 = y5 * y2;
    y - y3 / 6.0 + y5 / 120.0 - y7 / 5040.0
}

/// Cosine via cos(x) = sin(x + pi/2).
pub(crate) fn cosf_approx(x: f32) -> f32 {
    sinf_approx(x + core::f32::consts::FRAC_PI_2)
}

/// Python-style modulo: result has the sign of the divisor.
fn pymod(a: f32, b: f32) -> f32 {
    let r = a - floor_f32(a / b) * b;
    if r != 0.0 && ((r < 0.0) != (b < 0.0)) {
        r + b
    } else {
        r
    }
}

/// C-style fmod: result has the sign of the dividend.
fn cfmod(a: f32, b: f32) -> f32 {
    let q = (a / b) as i64 as f32;
    a - q * b
}

/// Element-wise modulo with broadcasting.
///
/// If `fmod` is true, uses C-style fmod semantics (result sign follows
/// dividend). If false, uses Python-style modulo (result sign follows
/// divisor) — this is the ONNX default for integer Mod.
pub fn op_mod(a: &Tensor, b: &Tensor, fmod: bool) -> Result<Tensor, OpError> {
    require_float(a, "Mod")?;
    require_float(b, "Mod")?;
    let out_dims = broadcast_shape(&a.shape.dims, &b.shape.dims)?;
    let total: usize = out_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let mut data = allocate_tensor_data(total, DataType::Float);
    let ndim = out_dims.len();
    let mut coord = alloc::vec![0usize; ndim];
    for flat in 0..total {
        let ai = broadcast_index(&coord, &a.shape.dims);
        let bi = broadcast_index(&coord, &b.shape.dims);
        let av = read_f32(&a.raw_data, ai);
        let bv = read_f32(&b.raw_data, bi);
        let v = if bv == 0.0 {
            f32::NAN
        } else if fmod {
            cfmod(av, bv)
        } else {
            pymod(av, bv)
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

/// Element-wise sine.
pub fn op_sin(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Sin", sinf_approx)
}

/// Element-wise cosine.
pub fn op_cos(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Cos", cosf_approx)
}

/// Element-wise reciprocal (1/x). Returns +/-inf for zero inputs.
pub fn op_reciprocal(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Reciprocal", |x| 1.0 / x)
}

/// Element-wise sign: -1, 0, or +1.
pub fn op_sign(input: &Tensor) -> Result<Tensor, OpError> {
    unary(input, "Sign", |x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

/// Variadic element-wise reducer helper. `acc_init` is the starting value
/// at every output cell; `f` combines a value with the accumulator.
fn variadic_broadcast<F: Fn(f32, f32) -> f32>(
    inputs: &[&Tensor],
    op: &str,
    acc_init: f32,
    f: F,
) -> Result<(Vec<i64>, Vec<u8>), OpError> {
    if inputs.is_empty() {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{} requires at least one input",
            op
        )));
    }
    for t in inputs {
        require_float(t, op)?;
    }
    // Compute output broadcast shape by folding.
    let mut out_dims = inputs[0].shape.dims.clone();
    for t in inputs.iter().skip(1) {
        out_dims = broadcast_shape(&out_dims, &t.shape.dims)?;
    }
    let total: usize = out_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let mut data = allocate_tensor_data(total, DataType::Float);
    let ndim = out_dims.len();
    let mut coord = alloc::vec![0usize; ndim];
    for flat in 0..total {
        let mut acc = acc_init;
        for t in inputs {
            let idx = broadcast_index(&coord, &t.shape.dims);
            acc = f(acc, read_f32(&t.raw_data, idx));
        }
        write_f32(&mut data, flat, acc);
        if flat + 1 < total {
            next_coord(&mut coord, &out_dims);
        }
    }
    Ok((out_dims, data))
}

/// Variadic element-wise sum with broadcasting.
pub fn op_sum(inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    let (out_dims, data) = variadic_broadcast(inputs, "Sum", 0.0, |acc, v| acc + v)?;
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(out_dims),
        name: String::new(),
        raw_data: data,
    })
}

/// Variadic element-wise mean with broadcasting.
pub fn op_mean(inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    let (out_dims, mut data) = variadic_broadcast(inputs, "Mean", 0.0, |acc, v| acc + v)?;
    let n = inputs.len() as f32;
    let total = data.len() / 4;
    for i in 0..total {
        let v = read_f32(&data, i) / n;
        write_f32(&mut data, i, v);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(out_dims),
        name: String::new(),
        raw_data: data,
    })
}

/// Element-wise Bool helper with broadcasting.
fn bool_binary<F: Fn(bool, bool) -> bool>(
    a: &Tensor,
    b: &Tensor,
    op: &str,
    f: F,
) -> Result<Tensor, OpError> {
    if a.data_type != DataType::Bool || b.data_type != DataType::Bool {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{} requires Bool inputs",
            op
        )));
    }
    let out_dims = broadcast_shape(&a.shape.dims, &b.shape.dims)?;
    let total: usize = out_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let mut data = alloc::vec![0u8; total];
    let ndim = out_dims.len();
    let mut coord = alloc::vec![0usize; ndim];
    for (flat, slot) in data.iter_mut().enumerate().take(total) {
        let ai = broadcast_index(&coord, &a.shape.dims);
        let bi = broadcast_index(&coord, &b.shape.dims);
        let av = a.raw_data[ai] != 0;
        let bv = b.raw_data[bi] != 0;
        *slot = u8::from(f(av, bv));
        if flat + 1 < total {
            next_coord(&mut coord, &out_dims);
        }
    }
    Ok(Tensor {
        data_type: DataType::Bool,
        shape: TensorShape::new(out_dims),
        name: String::new(),
        raw_data: data,
    })
}

/// Element-wise boolean AND with broadcasting.
pub fn op_and(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    bool_binary(a, b, "And", |x, y| x && y)
}

/// Element-wise boolean OR with broadcasting.
pub fn op_or(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    bool_binary(a, b, "Or", |x, y| x || y)
}

/// Returns the maximum value along one (outer, inner) lane of `axis_size`
/// elements with stride `inner` starting at `base`.
fn lane_max(input: &[u8], base: usize, axis_size: usize, inner: usize) -> f32 {
    let mut maxv = f32::NEG_INFINITY;
    for a in 0..axis_size {
        let v = read_f32(input, base + a * inner);
        if v > maxv {
            maxv = v;
        }
    }
    maxv
}

/// Returns `sum(exp(x - maxv))` along one lane.
fn lane_exp_sum(input: &[u8], base: usize, axis_size: usize, inner: usize, maxv: f32) -> f32 {
    let mut sum = 0.0_f32;
    for a in 0..axis_size {
        sum += expf_approx(read_f32(input, base + a * inner) - maxv);
    }
    sum
}

/// Writes `x - maxv - log_sum` into `out` along one lane.
fn lane_write_logsoftmax(
    input: &[u8],
    out: &mut [u8],
    base: usize,
    axis_size: usize,
    inner: usize,
    maxv: f32,
    log_sum: f32,
) {
    for a in 0..axis_size {
        let idx = base + a * inner;
        let v = read_f32(input, idx) - maxv - log_sum;
        write_f32(out, idx, v);
    }
}

/// Numerically-stable log-softmax along a single axis.
///
/// Computes `x - max - log(sum(exp(x - max)))` along `axis`.
pub fn op_log_softmax(input: &Tensor, axis: i64) -> Result<Tensor, OpError> {
    require_float(input, "LogSoftmax")?;
    let ndim = input.shape.ndim() as i64;
    let axis = if axis < 0 { ndim + axis } else { axis };
    if axis < 0 || axis >= ndim {
        return Err(OpError::InvalidAttribute(String::from(
            "LogSoftmax axis out of range",
        )));
    }
    let dims = &input.shape.dims;
    let axis = axis as usize;
    let outer: usize = dims[..axis]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let axis_size = dims[axis] as usize;
    let inner: usize = dims[axis + 1..]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let total = outer * axis_size * inner;
    let mut out = allocate_tensor_data(total, DataType::Float);

    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_size * inner + i;
            let maxv = lane_max(&input.raw_data, base, axis_size, inner);
            let sum = lane_exp_sum(&input.raw_data, base, axis_size, inner, maxv);
            let log_sum = lnf_approx(sum);
            lane_write_logsoftmax(
                &input.raw_data,
                &mut out,
                base,
                axis_size,
                inner,
                maxv,
                log_sum,
            );
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: out,
    })
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
    use crate::ops::common::test_helpers::{
        make_bool, make_f32, read_all_bool as read_bool_all, read_all_f32 as read_all,
    };
    use alloc::vec;

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

    // ---- Mod ----
    #[test]
    fn test_mod_python_style() {
        let a = make_f32(&[4], &[7.0, -7.0, 7.0, -7.0]);
        let b = make_f32(&[4], &[3.0, 3.0, -3.0, -3.0]);
        let v = read_all(&op_mod(&a, &b, false).unwrap());
        // Python: 7%3=1, -7%3=2, 7%-3=-2, -7%-3=-1
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[1] - 2.0).abs() < 1e-5);
        assert!((v[2] - -2.0).abs() < 1e-5);
        assert!((v[3] - -1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mod_fmod_style() {
        let a = make_f32(&[2], &[7.0, -7.0]);
        let b = make_f32(&[2], &[3.0, 3.0]);
        let v = read_all(&op_mod(&a, &b, true).unwrap());
        // fmod: 7%3=1, -7%3=-1 (sign of dividend)
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[1] - -1.0).abs() < 1e-5);
    }

    // ---- Sin/Cos ----
    #[test]
    fn test_sin_known_values() {
        let t = make_f32(
            &[4],
            &[
                0.0,
                core::f32::consts::FRAC_PI_2,
                core::f32::consts::PI,
                -core::f32::consts::FRAC_PI_2,
            ],
        );
        let v = read_all(&op_sin(&t).unwrap());
        assert!(v[0].abs() < 1e-3);
        assert!((v[1] - 1.0).abs() < 1e-3, "sin(pi/2) = {}", v[1]);
        assert!(v[2].abs() < 1e-3);
        assert!((v[3] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_sin_range_reduction_large() {
        // sin(10*pi) = 0
        let t = make_f32(&[1], &[10.0 * core::f32::consts::PI]);
        let v = read_all(&op_sin(&t).unwrap());
        assert!(v[0].abs() < 1e-3);
    }

    #[test]
    fn test_cos_known_values() {
        let t = make_f32(
            &[3],
            &[0.0, core::f32::consts::FRAC_PI_2, core::f32::consts::PI],
        );
        let v = read_all(&op_cos(&t).unwrap());
        assert!((v[0] - 1.0).abs() < 1e-3, "cos(0) = {}", v[0]);
        assert!(v[1].abs() < 1e-3);
        assert!((v[2] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_cos_scalar() {
        let t = make_f32(&[1], &[0.0]);
        let v = read_all(&op_cos(&t).unwrap());
        assert!((v[0] - 1.0).abs() < 1e-3);
    }

    // ---- Reciprocal ----
    #[test]
    fn test_reciprocal() {
        let t = make_f32(&[3], &[1.0, 2.0, 4.0]);
        let v = read_all(&op_reciprocal(&t).unwrap());
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 0.5).abs() < 1e-6);
        assert!((v[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_reciprocal_negative() {
        let t = make_f32(&[2], &[-1.0, -2.0]);
        let v = read_all(&op_reciprocal(&t).unwrap());
        assert!((v[0] + 1.0).abs() < 1e-6);
        assert!((v[1] + 0.5).abs() < 1e-6);
    }

    // ---- Sign ----
    #[test]
    fn test_sign_basic() {
        let t = make_f32(&[5], &[3.0, -2.0, 0.0, 7.5, -0.1]);
        let v = read_all(&op_sign(&t).unwrap());
        assert_eq!(v, vec![1.0, -1.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn test_sign_all_zero() {
        let t = make_f32(&[3], &[0.0, 0.0, 0.0]);
        let v = read_all(&op_sign(&t).unwrap());
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    // ---- Sum/Mean ----
    #[test]
    fn test_sum_variadic() {
        let a = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let b = make_f32(&[3], &[4.0, 5.0, 6.0]);
        let c = make_f32(&[3], &[7.0, 8.0, 9.0]);
        let v = read_all(&op_sum(&[&a, &b, &c]).unwrap());
        assert_eq!(v, vec![12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_sum_broadcast() {
        let a = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = make_f32(&[1], &[10.0]);
        let v = read_all(&op_sum(&[&a, &b]).unwrap());
        assert_eq!(v, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_mean_variadic() {
        let a = make_f32(&[2], &[2.0, 4.0]);
        let b = make_f32(&[2], &[4.0, 8.0]);
        let v = read_all(&op_mean(&[&a, &b]).unwrap());
        assert_eq!(v, vec![3.0, 6.0]);
    }

    #[test]
    fn test_mean_single_input() {
        let a = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let v = read_all(&op_mean(&[&a]).unwrap());
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    // ---- And/Or ----
    #[test]
    fn test_and_basic() {
        let a = make_bool(&[4], &[true, true, false, false]);
        let b = make_bool(&[4], &[true, false, true, false]);
        let v = read_bool_all(&op_and(&a, &b).unwrap());
        assert_eq!(v, vec![true, false, false, false]);
    }

    #[test]
    fn test_or_broadcast() {
        let a = make_bool(&[2, 2], &[true, false, false, true]);
        let b = make_bool(&[1], &[true]);
        let v = read_bool_all(&op_or(&a, &b).unwrap());
        assert_eq!(v, vec![true, true, true, true]);
    }

    #[test]
    fn test_and_rejects_float() {
        let a = make_f32(&[1], &[1.0]);
        let b = make_bool(&[1], &[true]);
        assert!(op_and(&a, &b).is_err());
    }

    // ---- LogSoftmax ----
    #[test]
    fn test_log_softmax_axis_last() {
        // For [1.0, 2.0, 3.0] log-softmax = x - log(sum(exp(x)))
        let t = make_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        let v = read_all(&op_log_softmax(&t, -1).unwrap());
        // sum = e^1+e^2+e^3 ≈ 30.19; log ≈ 3.407
        // log_softmax ≈ [-2.407, -1.407, -0.407]
        assert!((v[0] + 2.407).abs() < 1e-2);
        assert!((v[1] + 1.407).abs() < 1e-2);
        assert!((v[2] + 0.407).abs() < 1e-2);
    }

    #[test]
    fn test_log_softmax_sums_to_zero_exp() {
        // exp of log_softmax should sum to 1 along the axis.
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
        let out = op_log_softmax(&t, 1).unwrap();
        let v = read_all(&out);
        let row0: f32 = (0..3).map(|i| v[i].exp()).sum();
        let row1: f32 = (3..6).map(|i| v[i].exp()).sum();
        assert!((row0 - 1.0).abs() < 1e-3);
        assert!((row1 - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_log_softmax_invalid_axis() {
        let t = make_f32(&[2], &[1.0, 2.0]);
        assert!(op_log_softmax(&t, 5).is_err());
    }

    #[test]
    fn test_pow_zero_negative_exponent_is_infinity() {
        // pow(0, -1) must be +inf, not 0.
        let base = make_f32(&[1], &[0.0]);
        let exp = make_f32(&[1], &[-1.0]);
        let out = op_pow(&base, &exp).unwrap();
        let v = read_f32(&out.raw_data, 0);
        assert!(v.is_infinite() && v > 0.0);
    }

    #[test]
    fn test_pow_zero_positive_exponent_is_zero() {
        let base = make_f32(&[1], &[0.0]);
        let exp = make_f32(&[1], &[2.0]);
        let out = op_pow(&base, &exp).unwrap();
        assert_eq!(read_f32(&out.raw_data, 0), 0.0);
    }

    #[test]
    fn test_pow_zero_zero_still_one_after_fix() {
        // Regression guard: ensure the fix didn't regress the 0^0 case.
        let base = make_f32(&[1], &[0.0]);
        let exp = make_f32(&[1], &[0.0]);
        let out = op_pow(&base, &exp).unwrap();
        assert_eq!(read_f32(&out.raw_data, 0), 1.0);
    }

    #[test]
    fn test_sqrt_negative_is_nan() {
        let t = make_f32(&[2], &[-1.0, -4.0]);
        let out = op_sqrt(&t).unwrap();
        assert!(read_f32(&out.raw_data, 0).is_nan());
        assert!(read_f32(&out.raw_data, 1).is_nan());
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
