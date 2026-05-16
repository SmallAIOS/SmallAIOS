// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Phase 1 shape / data-movement operators.
//!
//! Includes graph-plumbing ops (Shape, Size, Identity, Constant) as well
//! as the tensor constructors (ConstantOfShape, Range), masking/cumulative
//! ops (Trilu, CumSum) and the N-dimensional Gather/Scatter variants
//! (GatherND, ScatterND).

use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{
    allocate_tensor_data, read_f32, read_i64, write_f32, write_i64, F32_SIZE, I64_SIZE,
};
use crate::operators::OpError;
use crate::ops::common::require_float_attr as require_float;
use crate::tensor::{DataType, Tensor, TensorShape};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn require_int64(t: &Tensor, op: &str) -> Result<(), OpError> {
    if t.data_type != DataType::Int64 {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "{} requires Int64 indices",
            op
        )));
    }
    Ok(())
}

fn compute_strides(shape: &[i64]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = alloc::vec![0usize; ndim];
    if ndim == 0 {
        return strides;
    }
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        let dim = if shape[i + 1] < 1 {
            1
        } else {
            shape[i + 1] as usize
        };
        strides[i] = strides[i + 1] * dim;
    }
    strides
}

// ---------------------------------------------------------------------------
// Shape / Size / Identity / Constant
// ---------------------------------------------------------------------------

/// Returns the shape of the input as a 1-D Int64 tensor.
pub fn op_shape(input: &Tensor) -> Result<Tensor, OpError> {
    let ndim = input.shape.ndim();
    let mut raw = alloc::vec![0u8; ndim * I64_SIZE];
    for (i, &d) in input.shape.dims.iter().enumerate() {
        write_i64(&mut raw, i, d);
    }
    Ok(Tensor {
        data_type: DataType::Int64,
        shape: TensorShape::new(alloc::vec![ndim as i64]),
        name: String::new(),
        raw_data: raw,
    })
}

/// Returns total element count as a scalar Int64 tensor.
pub fn op_size(input: &Tensor) -> Result<Tensor, OpError> {
    let total = input.shape.total_elements() as i64;
    let mut raw = alloc::vec![0u8; I64_SIZE];
    write_i64(&mut raw, 0, total);
    Ok(Tensor {
        data_type: DataType::Int64,
        shape: TensorShape::new(alloc::vec![]),
        name: String::new(),
        raw_data: raw,
    })
}

/// Identity pass-through: clones the input tensor.
pub fn op_identity(input: &Tensor) -> Result<Tensor, OpError> {
    Ok(Tensor {
        data_type: input.data_type,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: input.raw_data.clone(),
    })
}

/// Constant op: returns the attribute tensor unchanged. Typically folded
/// at graph build time; this function exists as the runtime fallback.
pub fn op_constant(value: &Tensor) -> Result<Tensor, OpError> {
    op_identity(value)
}

/// ConstantOfShape: produce a tensor of the given shape filled with a
/// scalar value. The current runtime stores float32 fills; the executor
/// dispatcher accepts the standard tensor-typed `value` attribute as well
/// as the `value_float` / `value_int` scalar shortcuts and projects the
/// fill down to f32 before invoking this function.
pub fn op_constant_of_shape(shape: &[i64], value: f32) -> Result<Tensor, OpError> {
    // Reject negative dims rather than silently absorbing them.
    if shape.iter().any(|&d| d < 0) {
        return Err(OpError::InvalidAttribute(String::from(
            "ConstantOfShape: shape must be non-negative",
        )));
    }
    // Honor zero-element shapes (e.g. [0, 3]) — do NOT collapse to 1.
    let total: usize = shape.iter().map(|&d| d as usize).product::<usize>();
    let mut data = allocate_tensor_data(total, DataType::Float);
    for i in 0..total {
        write_f32(&mut data, i, value);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: data,
    })
}

/// Range: `[start, start+delta, ..., < limit]` (float32 output).
pub fn op_range(start: f32, limit: f32, delta: f32) -> Result<Tensor, OpError> {
    if delta == 0.0 {
        return Err(OpError::InvalidAttribute(String::from(
            "Range delta must be nonzero",
        )));
    }
    let mut values = Vec::new();
    if delta > 0.0 {
        let mut v = start;
        while v < limit {
            values.push(v);
            let next = v + delta;
            // f32 rounding: when `v` is large and `delta` is small,
            // `v + delta == v`; break rather than spin forever.
            if next == v {
                break;
            }
            v = next;
        }
    } else {
        let mut v = start;
        while v > limit {
            values.push(v);
            let next = v + delta;
            if next == v {
                break;
            }
            v = next;
        }
    }
    let n = values.len();
    let mut data = allocate_tensor_data(n, DataType::Float);
    for (i, &v) in values.iter().enumerate() {
        write_f32(&mut data, i, v);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![n as i64]),
        name: String::new(),
        raw_data: data,
    })
}

/// Returns `true` when element `(r, c)` is kept by a Trilu mask with
/// diagonal offset `k`. `upper=true` keeps `col-row >= k`; otherwise keeps
/// `col-row <= k`.
fn trilu_keep(r: i64, c: i64, k: i64, upper: bool) -> bool {
    let diff = c - r;
    if upper {
        diff >= k
    } else {
        diff <= k
    }
}

/// Writes the Trilu mask for one matrix slice (rows × cols) starting at
/// `base` into the flat output buffer.
fn trilu_fill_matrix(
    input: &[u8],
    out: &mut [u8],
    base: usize,
    rows: i64,
    cols: i64,
    k: i64,
    upper: bool,
) {
    for r in 0..rows {
        for c in 0..cols {
            let flat = base + (r * cols + c) as usize;
            let v = if trilu_keep(r, c, k, upper) {
                read_f32(input, flat)
            } else {
                0.0
            };
            write_f32(out, flat, v);
        }
    }
}

/// Trilu: upper or lower triangular mask for 2D+ tensors.
///
/// For higher-rank inputs, the masking is applied to the last two dims
/// (the "matrix" dimensions) and broadcast across leading batch dims.
/// `k` offsets the diagonal: `k=0` is the main diagonal, `k>0` shifts
/// toward the upper-right, `k<0` toward the lower-left. When `upper` is
/// true, elements where `col - row >= k` are kept; otherwise elements
/// where `col - row <= k` are kept.
pub fn op_trilu(input: &Tensor, k: i64, upper: bool) -> Result<Tensor, OpError> {
    require_float(input, "Trilu")?;
    let ndim = input.shape.ndim();
    if ndim < 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "Trilu requires at least 2D input",
        )));
    }
    let rows = input.shape.dims[ndim - 2];
    let cols = input.shape.dims[ndim - 1];
    let matrix_elems = (rows * cols) as usize;
    let batches: usize = input.shape.dims[..ndim - 2]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let total = batches * matrix_elems;
    let mut out = allocate_tensor_data(total, DataType::Float);
    for b in 0..batches {
        trilu_fill_matrix(
            &input.raw_data,
            &mut out,
            b * matrix_elems,
            rows,
            cols,
            k,
            upper,
        );
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: out,
    })
}

/// Normalizes a signed axis index against an `ndim`-rank tensor. Returns
/// the non-negative axis or an `InvalidAttribute` error when out of range.
fn normalize_axis(axis: i64, ndim: usize, op: &str) -> Result<usize, OpError> {
    let nd = ndim as i64;
    let normalized = if axis < 0 { nd + axis } else { axis };
    if normalized < 0 || normalized >= nd {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "{} axis out of range",
            op
        )));
    }
    Ok(normalized as usize)
}

/// Splits a shape into `(outer, axis_size, inner)` factor sizes around
/// `axis`. Each factor is clamped to `max(1)` so scalar/empty inputs still
/// step exactly once.
fn factor_sizes(dims: &[i64], axis: usize) -> (usize, usize, usize) {
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
    (outer, axis_size, inner)
}

/// Scans a single (outer, inner) lane of a CumSum along `axis_size`
/// elements with stride `inner`. `exclusive=true` emits the running total
/// *before* adding the current element.
fn cumsum_scan_lane(
    input: &[u8],
    out: &mut [u8],
    base: usize,
    axis_size: usize,
    inner: usize,
    exclusive: bool,
    reverse: bool,
) {
    let mut acc = 0.0_f32;
    let order: alloc::vec::Vec<usize> = if reverse {
        (0..axis_size).rev().collect()
    } else {
        (0..axis_size).collect()
    };
    for a in order {
        let idx = base + a * inner;
        let v = read_f32(input, idx);
        if exclusive {
            write_f32(out, idx, acc);
            acc += v;
        } else {
            acc += v;
            write_f32(out, idx, acc);
        }
    }
}

/// CumSum along a single axis. `exclusive=true` omits the current element
/// from each partial sum; `reverse=true` scans from the end.
pub fn op_cumsum(
    input: &Tensor,
    axis: i64,
    exclusive: bool,
    reverse: bool,
) -> Result<Tensor, OpError> {
    require_float(input, "CumSum")?;
    let ndim = input.shape.ndim();
    let axis = normalize_axis(axis, ndim, "CumSum")?;
    let (outer, axis_size, inner) = factor_sizes(&input.shape.dims, axis);
    let total = outer * axis_size * inner;
    let mut out = allocate_tensor_data(total, DataType::Float);

    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_size * inner + i;
            cumsum_scan_lane(
                &input.raw_data,
                &mut out,
                base,
                axis_size,
                inner,
                exclusive,
                reverse,
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

// ---------------------------------------------------------------------------
// GatherND / ScatterND
// ---------------------------------------------------------------------------

/// ONNX GatherND-13.
///
/// The last dim of `indices` (call it `k`) determines how deep into the
/// input each index tuple reaches. `batch_dims` is the number of leading
/// dimensions that are shared (zipped) between input and indices. The
/// output has shape `indices.shape[:-1] + input.shape[batch_dims + k:]`.
///
/// Float32 data only. Int64 indices only.
/// Folds an index tuple (`k` elements starting at `idx_base` in
/// `indices`) into a flat input offset using `in_strides`. The tuple
/// addresses dimensions `[axis_start, axis_start + k)` of `input_dims`.
/// Negative indices are normalized to the non-negative range; out-of-range
/// values produce an error tagged with `op`.
fn gather_index_offset(
    indices: &[u8],
    idx_base: usize,
    k: usize,
    axis_start: usize,
    input_dims: &[i64],
    in_strides: &[usize],
    op: &str,
) -> Result<usize, OpError> {
    let mut off = 0usize;
    for j in 0..k {
        let raw = read_i64(indices, idx_base + j);
        let pos = axis_start + j;
        let dim = input_dims[pos];
        let normalized = if raw < 0 { raw + dim } else { raw };
        if normalized < 0 || normalized >= dim {
            return Err(OpError::ShapeMismatch(alloc::format!(
                "{} index {} out of range [0,{})",
                op,
                raw,
                dim
            )));
        }
        off += (normalized as usize) * in_strides[pos];
    }
    Ok(off)
}

/// Returns the flat offset into a tensor with `in_strides` corresponding
/// to the row-major batch index `batch` over the first `batch_dims` of
/// `input_dims`.
fn batch_offset(
    batch: usize,
    batch_dims: usize,
    input_dims: &[i64],
    in_strides: &[usize],
) -> usize {
    let mut off = 0usize;
    let mut rem = batch;
    for d in (0..batch_dims).rev() {
        let dim = input_dims[d] as usize;
        let c = rem % dim;
        rem /= dim;
        off += c * in_strides[d];
    }
    off
}

/// Product of a dim slice, clamped to at least 1.
fn product_dims(dims: &[i64]) -> usize {
    dims.iter().map(|&d| d as usize).product::<usize>().max(1)
}

/// Validates `indices`/`batch_dims` for GatherND and returns the
/// `(idx_ndim, batch_dims, k)` triple expected by the main loop.
fn gather_nd_validate(
    input: &Tensor,
    indices: &Tensor,
    batch_dims: i64,
) -> Result<(usize, usize, usize), OpError> {
    let idx_ndim = indices.shape.ndim();
    if idx_ndim == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "GatherND indices must have at least 1 dimension",
        )));
    }
    if batch_dims < 0 || (batch_dims as usize) >= idx_ndim {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "GatherND batch_dims {} out of range [0, {})",
            batch_dims,
            idx_ndim
        )));
    }
    let batch_dims = batch_dims as usize;
    let k = indices.shape.dims[idx_ndim - 1] as usize;
    if batch_dims + k > input.shape.ndim() {
        return Err(OpError::ShapeMismatch(String::from(
            "GatherND batch_dims+k exceeds input rank",
        )));
    }
    Ok((idx_ndim, batch_dims, k))
}

pub fn op_gather_nd(input: &Tensor, indices: &Tensor, batch_dims: i64) -> Result<Tensor, OpError> {
    require_float(input, "GatherND")?;
    require_int64(indices, "GatherND")?;
    let (idx_ndim, batch_dims, k) = gather_nd_validate(input, indices, batch_dims)?;

    // Output shape = indices.shape[:-1] + input.shape[batch_dims+k:]
    let mut out_shape: Vec<i64> = indices.shape.dims[..idx_ndim - 1].to_vec();
    let slice_dims = &input.shape.dims[batch_dims + k..];
    out_shape.extend_from_slice(slice_dims);
    let slice_size = product_dims(slice_dims);

    let total_out = product_dims(&out_shape);
    let mut out = allocate_tensor_data(total_out, DataType::Float);

    let tuple_count = product_dims(&indices.shape.dims[..idx_ndim - 1]);
    let batch_count = product_dims(&indices.shape.dims[..batch_dims]);
    let tuples_per_batch = tuple_count / batch_count;

    let in_strides = compute_strides(&input.shape.dims);

    for batch in 0..batch_count {
        let in_batch_offset = batch_offset(batch, batch_dims, &input.shape.dims, &in_strides);
        for t in 0..tuples_per_batch {
            let idx_base = (batch * tuples_per_batch + t) * k;
            let tuple_off = gather_index_offset(
                &indices.raw_data,
                idx_base,
                k,
                batch_dims,
                &input.shape.dims,
                &in_strides,
                "GatherND",
            )?;
            let in_off = in_batch_offset + tuple_off;
            let out_off = (batch * tuples_per_batch + t) * slice_size;
            for s in 0..slice_size {
                let v = read_f32(&input.raw_data, in_off + s);
                write_f32(&mut out, out_off + s, v);
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(out_shape),
        name: String::new(),
        raw_data: out,
    })
}

/// ONNX ScatterND-16 (no `reduction` attribute — "none" only; the common
/// BERT case).
///
/// Copies `input` and overwrites slices at the positions specified by
/// `indices` with values from `updates`. `indices.shape[-1] = k` and
/// `updates.shape == indices.shape[:-1] + input.shape[k:]`.
/// Validates the ScatterND shape contracts and returns `(idx_ndim, k)`
/// for the main loop.
fn scatter_nd_validate(
    input: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
) -> Result<(usize, usize), OpError> {
    let idx_ndim = indices.shape.ndim();
    if idx_ndim == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "ScatterND indices must have at least 1 dimension",
        )));
    }
    let k = indices.shape.dims[idx_ndim - 1] as usize;
    let in_ndim = input.shape.ndim();
    if k > in_ndim {
        return Err(OpError::ShapeMismatch(String::from(
            "ScatterND index depth exceeds input rank",
        )));
    }
    let expected_updates_rank = (idx_ndim - 1) + (in_ndim - k);
    if updates.shape.ndim() != expected_updates_rank {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "ScatterND: updates rank {} != expected {}",
            updates.shape.ndim(),
            expected_updates_rank
        )));
    }
    scatter_nd_check_updates_dims(input, indices, updates, idx_ndim, k)?;
    Ok((idx_ndim, k))
}

/// Verifies `updates.shape == indices.shape[:-1] + input.shape[k:]`.
fn scatter_nd_check_updates_dims(
    input: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
    idx_ndim: usize,
    k: usize,
) -> Result<(), OpError> {
    for (i, &d) in indices.shape.dims[..idx_ndim - 1].iter().enumerate() {
        if updates.shape.dims[i] != d {
            return Err(OpError::ShapeMismatch(String::from(
                "ScatterND: updates leading dims must match indices.shape[:-1]",
            )));
        }
    }
    for (i, &d) in input.shape.dims[k..].iter().enumerate() {
        if updates.shape.dims[(idx_ndim - 1) + i] != d {
            return Err(OpError::ShapeMismatch(String::from(
                "ScatterND: updates trailing dims must match input.shape[k:]",
            )));
        }
    }
    Ok(())
}

/// Copies `slice_size` f32 elements from `updates[upd_off..]` into
/// `out_raw[in_off..]` (element indices).
fn scatter_nd_write_slice(
    out_raw: &mut [u8],
    updates: &[u8],
    in_off: usize,
    upd_off: usize,
    slice_size: usize,
) {
    for s in 0..slice_size {
        let v = read_f32(updates, upd_off + s);
        let byte_off = (in_off + s) * F32_SIZE;
        out_raw[byte_off..byte_off + F32_SIZE].copy_from_slice(&v.to_le_bytes());
    }
}

pub fn op_scatter_nd(
    input: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
) -> Result<Tensor, OpError> {
    require_float(input, "ScatterND")?;
    require_int64(indices, "ScatterND")?;
    require_float(updates, "ScatterND")?;

    let (idx_ndim, k) = scatter_nd_validate(input, indices, updates)?;
    let slice_size = product_dims(&input.shape.dims[k..]);

    let mut out_raw = input.raw_data.clone();
    let in_strides = compute_strides(&input.shape.dims);
    let tuple_count = product_dims(&indices.shape.dims[..idx_ndim - 1]);

    for t in 0..tuple_count {
        let in_off = gather_index_offset(
            &indices.raw_data,
            t * k,
            k,
            0,
            &input.shape.dims,
            &in_strides,
            "ScatterND",
        )?;
        scatter_nd_write_slice(
            &mut out_raw,
            &updates.raw_data,
            in_off,
            t * slice_size,
            slice_size,
        );
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: out_raw,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::common::test_helpers::{make_f32, make_i64, read_all_f32, read_all_i64};
    use alloc::vec;

    #[test]
    fn test_shape_basic() {
        let t = make_f32(&[2, 3, 4], &alloc::vec![0.0; 24]);
        let s = op_shape(&t).unwrap();
        assert_eq!(s.shape.dims, vec![3]);
        assert_eq!(read_all_i64(&s), vec![2, 3, 4]);
    }

    #[test]
    fn test_shape_scalar() {
        let t = make_f32(&[], &[1.0]);
        let s = op_shape(&t).unwrap();
        assert_eq!(s.shape.dims, vec![0]);
    }

    #[test]
    fn test_size_basic() {
        let t = make_f32(&[2, 3, 4], &alloc::vec![0.0; 24]);
        let s = op_size(&t).unwrap();
        assert_eq!(read_i64(&s.raw_data, 0), 24);
    }

    #[test]
    fn test_size_scalar() {
        let t = make_f32(&[1], &[7.0]);
        let s = op_size(&t).unwrap();
        assert_eq!(read_i64(&s.raw_data, 0), 1);
    }

    #[test]
    fn test_identity_preserves_data() {
        let t = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let out = op_identity(&t).unwrap();
        assert_eq!(out.shape.dims, t.shape.dims);
        assert_eq!(read_all_f32(&out), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_constant_returns_value() {
        let t = make_f32(&[2], &[5.0, 6.0]);
        let out = op_constant(&t).unwrap();
        assert_eq!(read_all_f32(&out), vec![5.0, 6.0]);
    }

    #[test]
    fn test_constant_of_shape_basic() {
        let out = op_constant_of_shape(&[2, 3], 7.0).unwrap();
        assert_eq!(out.shape.dims, vec![2, 3]);
        assert_eq!(read_all_f32(&out), vec![7.0; 6]);
    }

    #[test]
    fn test_constant_of_shape_zero_fill() {
        let out = op_constant_of_shape(&[4], 0.0).unwrap();
        assert_eq!(read_all_f32(&out), vec![0.0; 4]);
    }

    #[test]
    fn test_range_positive_delta() {
        let out = op_range(0.0, 5.0, 1.0).unwrap();
        assert_eq!(read_all_f32(&out), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_range_negative_delta() {
        let out = op_range(5.0, 0.0, -1.0).unwrap();
        assert_eq!(read_all_f32(&out), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_range_fractional() {
        let out = op_range(0.0, 1.0, 0.25).unwrap();
        assert_eq!(read_all_f32(&out), vec![0.0, 0.25, 0.5, 0.75]);
    }

    #[test]
    fn test_range_zero_delta_error() {
        assert!(op_range(0.0, 5.0, 0.0).is_err());
    }

    #[test]
    fn test_trilu_upper() {
        let t = make_f32(&[3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let out = op_trilu(&t, 0, true).unwrap();
        assert_eq!(
            read_all_f32(&out),
            vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
        );
    }

    #[test]
    fn test_trilu_lower() {
        let t = make_f32(&[3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let out = op_trilu(&t, 0, false).unwrap();
        assert_eq!(
            read_all_f32(&out),
            vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_trilu_offset_k() {
        // Upper with k=1 -> strictly above main diagonal.
        let t = make_f32(&[3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let out = op_trilu(&t, 1, true).unwrap();
        assert_eq!(
            read_all_f32(&out),
            vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_trilu_requires_2d() {
        let t = make_f32(&[3], &[1.0, 2.0, 3.0]);
        assert!(op_trilu(&t, 0, true).is_err());
    }

    #[test]
    fn test_cumsum_axis0_inclusive() {
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = op_cumsum(&t, 0, false, false).unwrap();
        assert_eq!(read_all_f32(&out), vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_cumsum_axis1_exclusive() {
        let t = make_f32(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let out = op_cumsum(&t, 1, true, false).unwrap();
        assert_eq!(read_all_f32(&out), vec![0.0, 1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_cumsum_reverse() {
        let t = make_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let out = op_cumsum(&t, 0, false, true).unwrap();
        assert_eq!(read_all_f32(&out), vec![10.0, 9.0, 7.0, 4.0]);
    }

    #[test]
    fn test_cumsum_invalid_axis() {
        let t = make_f32(&[3], &[1.0, 2.0, 3.0]);
        assert!(op_cumsum(&t, 5, false, false).is_err());
    }

    #[test]
    fn test_gather_nd_simple_2d() {
        // input[[0,0], [1,1]] -> pick [0,0] and [1,1] from 2x2 matrix
        let t = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let idx = make_i64(&[2, 2], &[0, 0, 1, 1]);
        let out = op_gather_nd(&t, &idx, 0).unwrap();
        assert_eq!(out.shape.dims, vec![2]);
        assert_eq!(read_all_f32(&out), vec![1.0, 4.0]);
    }

    #[test]
    fn test_gather_nd_slice_output() {
        // k=1 on a 2x3 matrix: each tuple picks one row.
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let idx = make_i64(&[2, 1], &[1, 0]);
        let out = op_gather_nd(&t, &idx, 0).unwrap();
        assert_eq!(out.shape.dims, vec![2, 3]);
        assert_eq!(read_all_f32(&out), vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_gather_nd_out_of_range_errors() {
        let t = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let idx = make_i64(&[1, 2], &[5, 0]);
        assert!(op_gather_nd(&t, &idx, 0).is_err());
    }

    #[test]
    fn test_scatter_nd_point_updates() {
        // Base 2x2, overwrite element (0,1) and (1,0).
        let t = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let idx = make_i64(&[2, 2], &[0, 1, 1, 0]);
        let upd = make_f32(&[2], &[20.0, 30.0]);
        let out = op_scatter_nd(&t, &idx, &upd).unwrap();
        assert_eq!(read_all_f32(&out), vec![1.0, 20.0, 30.0, 4.0]);
    }

    #[test]
    fn test_scatter_nd_row_update() {
        // k=1 on 2x3 — overwrite row 0 with a new row.
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let idx = make_i64(&[1, 1], &[0]);
        let upd = make_f32(&[1, 3], &[10.0, 20.0, 30.0]);
        let out = op_scatter_nd(&t, &idx, &upd).unwrap();
        assert_eq!(read_all_f32(&out), vec![10.0, 20.0, 30.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_constant_of_shape_rejects_negative_dim() {
        assert!(op_constant_of_shape(&[-1, 3], 0.0).is_err());
    }

    #[test]
    fn test_constant_of_shape_zero_element_shape() {
        // Shape [0, 3] must produce a zero-element tensor, NOT a 1-element one.
        let out = op_constant_of_shape(&[0, 3], 7.0).unwrap();
        assert_eq!(out.shape.dims, vec![0, 3]);
        assert_eq!(out.shape.total_elements(), 0);
        assert!(out.raw_data.is_empty());
    }

    #[test]
    fn test_range_non_advancing_breaks() {
        // start=1e20, delta=1.0 in f32 → 1e20 + 1.0 rounds back to 1e20.
        // Must terminate without spinning, not return an error for this case.
        let out = op_range(1.0e20, 1.0e21, 1.0).unwrap();
        // We just require it terminates and produces a small finite output
        // (exactly one element, start, before we bail out).
        assert_eq!(read_all_f32(&out), vec![1.0e20]);
    }

    #[test]
    fn test_range_zero_delta_still_errors() {
        assert!(op_range(0.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn test_gather_nd_rejects_oob_batch_dims() {
        let t = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let idx = make_i64(&[2, 2], &[0, 0, 1, 1]);
        // idx_ndim == 2, so batch_dims must be in [0, 2). 2 is out of range.
        assert!(op_gather_nd(&t, &idx, 2).is_err());
        // Negative batch_dims also rejected.
        assert!(op_gather_nd(&t, &idx, -1).is_err());
    }

    #[test]
    fn test_scatter_nd_rejects_mismatched_updates_shape() {
        // input [2,3], idx shape [1,1] (k=1), expected updates shape [1,3]
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let idx = make_i64(&[1, 1], &[0]);
        // Wrong updates shape: [1, 2] instead of [1, 3]
        let upd = make_f32(&[1, 2], &[10.0, 20.0]);
        assert!(op_scatter_nd(&t, &idx, &upd).is_err());
    }

    #[test]
    fn test_scatter_nd_out_of_range_errors() {
        let t = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let idx = make_i64(&[1, 2], &[3, 0]);
        let upd = make_f32(&[1], &[99.0]);
        assert!(op_scatter_nd(&t, &idx, &upd).is_err());
    }
}
