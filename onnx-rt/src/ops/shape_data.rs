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
use crate::tensor::{DataType, Tensor, TensorShape};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn require_float(t: &Tensor, op: &str) -> Result<(), OpError> {
    if t.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "{} only supports float32",
            op
        )));
    }
    Ok(())
}

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
/// scalar value. The current runtime stores float32 fills; if the
/// fill-value tensor is Int64 we emit a Float32 with the integer
/// converted to float (good enough for BERT-style integer masks).
pub fn op_constant_of_shape(shape: &[i64], value: f32) -> Result<Tensor, OpError> {
    let total: usize = shape
        .iter()
        .map(|&d| d.max(0) as usize)
        .product::<usize>()
        .max(1);
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
            v += delta;
        }
    } else {
        let mut v = start;
        while v > limit {
            values.push(v);
            v += delta;
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
        for r in 0..rows {
            for c in 0..cols {
                let flat = b * matrix_elems + (r * cols + c) as usize;
                let diff = c - r;
                let keep = if upper { diff >= k } else { diff <= k };
                let v = if keep {
                    read_f32(&input.raw_data, flat)
                } else {
                    0.0
                };
                write_f32(&mut out, flat, v);
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: out,
    })
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
    let ndim = input.shape.ndim() as i64;
    let axis = if axis < 0 { ndim + axis } else { axis };
    if axis < 0 || axis >= ndim {
        return Err(OpError::InvalidAttribute(String::from(
            "CumSum axis out of range",
        )));
    }
    let axis = axis as usize;
    let dims = &input.shape.dims;
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
            let mut acc = 0.0_f32;
            if !reverse {
                for a in 0..axis_size {
                    let idx = (o * axis_size + a) * inner + i;
                    let v = read_f32(&input.raw_data, idx);
                    if exclusive {
                        write_f32(&mut out, idx, acc);
                        acc += v;
                    } else {
                        acc += v;
                        write_f32(&mut out, idx, acc);
                    }
                }
            } else {
                for a in (0..axis_size).rev() {
                    let idx = (o * axis_size + a) * inner + i;
                    let v = read_f32(&input.raw_data, idx);
                    if exclusive {
                        write_f32(&mut out, idx, acc);
                        acc += v;
                    } else {
                        acc += v;
                        write_f32(&mut out, idx, acc);
                    }
                }
            }
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
pub fn op_gather_nd(input: &Tensor, indices: &Tensor, batch_dims: i64) -> Result<Tensor, OpError> {
    require_float(input, "GatherND")?;
    require_int64(indices, "GatherND")?;
    let batch_dims = batch_dims as usize;
    let idx_ndim = indices.shape.ndim();
    if idx_ndim == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "GatherND indices must have at least 1 dimension",
        )));
    }
    let k = indices.shape.dims[idx_ndim - 1] as usize;
    let in_ndim = input.shape.ndim();
    if batch_dims + k > in_ndim {
        return Err(OpError::ShapeMismatch(String::from(
            "GatherND batch_dims+k exceeds input rank",
        )));
    }

    // Output shape = indices.shape[:-1] + input.shape[batch_dims+k:]
    let mut out_shape: Vec<i64> = indices.shape.dims[..idx_ndim - 1].to_vec();
    let slice_dims = &input.shape.dims[batch_dims + k..];
    out_shape.extend_from_slice(slice_dims);
    let slice_size: usize = slice_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);

    let total_out: usize = out_shape
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let mut out = allocate_tensor_data(total_out, DataType::Float);

    // Total number of index tuples = product of indices.shape[:-1]
    let tuple_count: usize = indices.shape.dims[..idx_ndim - 1]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    // Number of batch tuples = product of first batch_dims of indices
    let batch_count: usize = indices.shape.dims[..batch_dims]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let tuples_per_batch = tuple_count / batch_count;

    let in_strides = compute_strides(&input.shape.dims);

    for batch in 0..batch_count {
        // Leading batch offset into input (flattened).
        let in_batch_offset = {
            let mut off = 0usize;
            let mut rem = batch;
            for d in (0..batch_dims).rev() {
                let dim = input.shape.dims[d] as usize;
                let c = rem % dim;
                rem /= dim;
                off += c * in_strides[d];
            }
            off
        };
        for t in 0..tuples_per_batch {
            let idx_base = (batch * tuples_per_batch + t) * k;
            // Compute linear offset into input.
            let mut in_off = in_batch_offset;
            for (j, idx_pos) in (batch_dims..batch_dims + k).enumerate() {
                let raw = read_i64(&indices.raw_data, idx_base + j);
                let dim = input.shape.dims[idx_pos];
                let normalized = if raw < 0 { raw + dim } else { raw };
                if normalized < 0 || normalized >= dim {
                    return Err(OpError::ShapeMismatch(alloc::format!(
                        "GatherND index {} out of range [0,{})",
                        raw,
                        dim
                    )));
                }
                in_off += (normalized as usize) * in_strides[idx_pos];
            }
            // Copy slice_size elements.
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
pub fn op_scatter_nd(
    input: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
) -> Result<Tensor, OpError> {
    require_float(input, "ScatterND")?;
    require_int64(indices, "ScatterND")?;
    require_float(updates, "ScatterND")?;

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
    let slice_dims = &input.shape.dims[k..];
    let slice_size: usize = slice_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);

    let mut out_raw = input.raw_data.clone();
    let in_strides = compute_strides(&input.shape.dims);

    let tuple_count: usize = indices.shape.dims[..idx_ndim - 1]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);

    for t in 0..tuple_count {
        let idx_base = t * k;
        let mut in_off = 0usize;
        #[allow(clippy::needless_range_loop)]
        for j in 0..k {
            let raw = read_i64(&indices.raw_data, idx_base + j);
            let dim = input.shape.dims[j];
            let normalized = if raw < 0 { raw + dim } else { raw };
            if normalized < 0 || normalized >= dim {
                return Err(OpError::ShapeMismatch(alloc::format!(
                    "ScatterND index {} out of range [0,{})",
                    raw,
                    dim
                )));
            }
            in_off += (normalized as usize) * in_strides[j];
        }
        let upd_off = t * slice_size;
        for s in 0..slice_size {
            let v = read_f32(&updates.raw_data, upd_off + s);
            // Write into out_raw at position in_off + s (in float32 element idx).
            let byte_off = (in_off + s) * F32_SIZE;
            out_raw[byte_off..byte_off + F32_SIZE].copy_from_slice(&v.to_le_bytes());
        }
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

    fn make_i64(dims: &[i64], vals: &[i64]) -> Tensor {
        let mut data = alloc::vec![0u8; vals.len() * I64_SIZE];
        for (i, &v) in vals.iter().enumerate() {
            write_i64(&mut data, i, v);
        }
        Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: data,
        }
    }

    fn read_all_f32(t: &Tensor) -> Vec<f32> {
        (0..t.shape.total_elements())
            .map(|i| read_f32(&t.raw_data, i))
            .collect()
    }

    fn read_all_i64(t: &Tensor) -> Vec<i64> {
        (0..t.shape.total_elements())
            .map(|i| read_i64(&t.raw_data, i))
            .collect()
    }

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
    fn test_scatter_nd_out_of_range_errors() {
        let t = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let idx = make_i64(&[1, 2], &[3, 0]);
        let upd = make_f32(&[1], &[99.0]);
        assert!(op_scatter_nd(&t, &idx, &upd).is_err());
    }
}
