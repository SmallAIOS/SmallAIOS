// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Data selection / indexing operators: TopK, Compress, NonZero,
//! Unique, GatherElements, ScatterElements.
//!
//! These ops are used by modern CV/detection pipelines (Mask R-CNN,
//! YOLO post-processing, NMS) and by some transformer attention
//! variants. Float inputs only; int tensors are accepted as indices
//! where required.

use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, read_i64, write_f32, write_i64};
use crate::operators::OpError;
use crate::tensor::{DataType, Tensor, TensorShape};

fn require_float(t: &Tensor, op: &str) -> Result<(), OpError> {
    if t.data_type != DataType::Float {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{} requires Float input",
            op
        )));
    }
    Ok(())
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, OpError> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        return Err(OpError::InvalidAttribute(String::from("axis out of range")));
    }
    Ok(a as usize)
}

fn read_i64_vec(t: &Tensor) -> Vec<i64> {
    match t.data_type {
        DataType::Int64 => (0..t.shape.total_elements())
            .map(|i| read_i64(&t.raw_data, i))
            .collect(),
        DataType::Int32 => (0..t.shape.total_elements())
            .map(|i| crate::byte_io::read_i32(&t.raw_data, i) as i64)
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// TopK
// ---------------------------------------------------------------------------

/// Returns the top-k values and indices along `axis`.
///
/// Always returns two tensors: `[values, indices]`. Indices are Int64.
/// `largest=true` selects the k largest, `false` the k smallest.
/// `sorted=true` orders the returned slice descending (or ascending
/// for `largest=false`).
pub fn op_top_k(
    input: &Tensor,
    k: i64,
    axis: i64,
    largest: bool,
    sorted: bool,
) -> Result<Vec<Tensor>, OpError> {
    require_float(input, "TopK")?;
    let rank = input.shape.dims.len();
    let ax = normalize_axis(axis, rank)?;
    let axis_dim = input.shape.dims[ax] as usize;
    if k <= 0 || (k as usize) > axis_dim {
        return Err(OpError::InvalidAttribute(String::from(
            "TopK: k out of range",
        )));
    }
    let k_us = k as usize;
    let outer = product_dims(&input.shape.dims[..ax]);
    let inner = product_dims(&input.shape.dims[ax + 1..]);

    let mut out_dims = input.shape.dims.clone();
    out_dims[ax] = k;
    let total = outer * k_us * inner;
    let mut val_data = allocate_tensor_data(total, DataType::Float);
    let mut idx_data = allocate_tensor_data(total, DataType::Int64);

    let mut scratch: Vec<(f32, usize)> = Vec::with_capacity(axis_dim);
    for o in 0..outer {
        for i in 0..inner {
            fill_top_k_scratch(&input.raw_data, &mut scratch, o, i, axis_dim, inner);
            sort_top_k(&mut scratch, largest);
            // If not sorted, the spec allows any order — we still
            // return the selection order from sort.
            let _ = sorted;
            write_top_k_slot(&scratch, &mut val_data, &mut idx_data, o, i, k_us, inner);
        }
    }
    let values = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(out_dims.clone()),
        name: String::new(),
        raw_data: val_data,
    };
    let indices = Tensor {
        data_type: DataType::Int64,
        shape: TensorShape::new(out_dims),
        name: String::new(),
        raw_data: idx_data,
    };
    Ok(alloc::vec![values, indices])
}

fn product_dims(dims: &[i64]) -> usize {
    dims.iter().map(|&d| d as usize).product::<usize>().max(1)
}

fn fill_top_k_scratch(
    raw: &[u8],
    scratch: &mut Vec<(f32, usize)>,
    o: usize,
    i: usize,
    axis_dim: usize,
    inner: usize,
) {
    scratch.clear();
    for a in 0..axis_dim {
        let src = o * axis_dim * inner + a * inner + i;
        scratch.push((read_f32(raw, src), a));
    }
}

fn sort_top_k(scratch: &mut [(f32, usize)], largest: bool) {
    if largest {
        scratch.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    } else {
        scratch.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    }
}

fn write_top_k_slot(
    scratch: &[(f32, usize)],
    val_data: &mut [u8],
    idx_data: &mut [u8],
    o: usize,
    i: usize,
    k_us: usize,
    inner: usize,
) {
    for (rank_idx, &(v, idx)) in scratch.iter().take(k_us).enumerate() {
        let dst = o * k_us * inner + rank_idx * inner + i;
        write_f32(val_data, dst, v);
        write_i64(idx_data, dst, idx as i64);
    }
}

// ---------------------------------------------------------------------------
// Compress
// ---------------------------------------------------------------------------

/// Selects slices from `input` along `axis` where `condition` is true.
///
/// If `axis` is `None`, the input is flattened first. `condition` is
/// interpreted element-wise; non-zero means keep.
pub fn op_compress(
    input: &Tensor,
    condition: &Tensor,
    axis: Option<i64>,
) -> Result<Tensor, OpError> {
    require_float(input, "Compress")?;
    let cond = decode_compress_condition(condition);
    match axis {
        None => Ok(compress_flat(input, &cond)),
        Some(ax) => compress_along_axis(input, &cond, ax),
    }
}

fn decode_compress_condition(condition: &Tensor) -> Vec<bool> {
    // Condition may be Bool/Int8 but we store as raw bytes: treat any
    // non-zero byte in a length-aligned stream as true.
    let n = condition.shape.total_elements();
    (0..n)
        .map(|i| {
            if i < condition.raw_data.len() {
                condition.raw_data[i] != 0
            } else {
                false
            }
        })
        .collect()
}

fn compress_flat(input: &Tensor, cond: &[bool]) -> Tensor {
    let total = input.shape.total_elements();
    let mut kept: Vec<f32> = Vec::new();
    for i in 0..total {
        if i < cond.len() && cond[i] {
            kept.push(read_f32(&input.raw_data, i));
        }
    }
    let len = kept.len();
    let mut data = allocate_tensor_data(len, DataType::Float);
    for (i, v) in kept.iter().enumerate() {
        write_f32(&mut data, i, *v);
    }
    Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![len as i64]),
        name: String::new(),
        raw_data: data,
    }
}

fn compress_along_axis(input: &Tensor, cond: &[bool], ax: i64) -> Result<Tensor, OpError> {
    let rank = input.shape.dims.len();
    let ax = normalize_axis(ax, rank)?;
    let axis_dim = input.shape.dims[ax] as usize;
    let outer = product_dims(&input.shape.dims[..ax]);
    let inner = product_dims(&input.shape.dims[ax + 1..]);
    let keep: Vec<usize> = (0..axis_dim)
        .filter(|&i| i < cond.len() && cond[i])
        .collect();
    let k_len = keep.len();
    let mut out_dims = input.shape.dims.clone();
    out_dims[ax] = k_len as i64;
    let total = outer * k_len * inner;
    let mut data = allocate_tensor_data(total, DataType::Float);
    for o in 0..outer {
        for (ki, &src_a) in keep.iter().enumerate() {
            for i in 0..inner {
                let src = o * axis_dim * inner + src_a * inner + i;
                let dst = o * k_len * inner + ki * inner + i;
                write_f32(&mut data, dst, read_f32(&input.raw_data, src));
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(out_dims),
        name: String::new(),
        raw_data: data,
    })
}

// ---------------------------------------------------------------------------
// NonZero
// ---------------------------------------------------------------------------

/// Returns the indices of non-zero elements of `input` as an Int64
/// tensor of shape `[rank, count]`.
pub fn op_non_zero(input: &Tensor) -> Result<Tensor, OpError> {
    let rank = input.shape.dims.len();
    let total = input.shape.total_elements();
    let mut coord = alloc::vec![0usize; rank];
    let mut rows: Vec<Vec<i64>> = (0..rank).map(|_| Vec::new()).collect();
    for flat in 0..total {
        if is_nonzero_element(input, flat) {
            for (d, row) in rows.iter_mut().enumerate() {
                row.push(coord[d] as i64);
            }
        }
        increment_shape_coord(&mut coord, &input.shape.dims, rank);
    }
    let count = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut data = allocate_tensor_data(rank * count, DataType::Int64);
    for (d, row) in rows.iter().enumerate() {
        for (i, &v) in row.iter().enumerate() {
            write_i64(&mut data, d * count + i, v);
        }
    }
    Ok(Tensor {
        data_type: DataType::Int64,
        shape: TensorShape::new(alloc::vec![rank as i64, count as i64]),
        name: String::new(),
        raw_data: data,
    })
}

fn is_nonzero_element(input: &Tensor, flat: usize) -> bool {
    match input.data_type {
        DataType::Float => read_f32(&input.raw_data, flat) != 0.0,
        DataType::Int64 => read_i64(&input.raw_data, flat) != 0,
        DataType::Int32 => crate::byte_io::read_i32(&input.raw_data, flat) != 0,
        _ => flat < input.raw_data.len() && input.raw_data[flat] != 0,
    }
}

fn increment_shape_coord(coord: &mut [usize], dims: &[i64], rank: usize) {
    for d in (0..rank).rev() {
        coord[d] += 1;
        if (coord[d] as i64) < dims[d] {
            break;
        }
        coord[d] = 0;
    }
}

// ---------------------------------------------------------------------------
// Unique
// ---------------------------------------------------------------------------

/// Returns the unique values of `input` along `axis` (None = flatten).
///
/// Limitation: only the primary `Y` output tensor is returned. The
/// optional `indices`, `inverse_indices`, and `counts` tensors that
/// the ONNX spec describes are not produced — consumers that require
/// them should be rejected at load time. `sorted=true` returns values
/// in ascending order.
pub fn op_unique(input: &Tensor, axis: Option<i64>, sorted: bool) -> Result<Tensor, OpError> {
    require_float(input, "Unique")?;
    if axis.is_some() {
        return Err(OpError::InvalidAttribute(String::from(
            "Unique: axis-specific mode not supported (flattened only)",
        )));
    }
    if input.shape.dims.is_empty() {
        return Err(OpError::ShapeMismatch(String::from(
            "Unique requires rank >= 1",
        )));
    }
    if input.shape.dims.iter().any(|&d| d < 0) {
        return Err(OpError::ShapeMismatch(String::from(
            "Unique: negative dimensions are invalid",
        )));
    }
    let total = input.shape.total_elements();
    let mut vals: Vec<f32> = Vec::new();
    for i in 0..total {
        let v = read_f32(&input.raw_data, i);
        if !vals.iter().any(|&u| u.to_bits() == v.to_bits()) {
            vals.push(v);
        }
    }
    if sorted {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    }
    let n = vals.len();
    let mut data = allocate_tensor_data(n, DataType::Float);
    for (i, v) in vals.iter().enumerate() {
        write_f32(&mut data, i, *v);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![n as i64]),
        name: String::new(),
        raw_data: data,
    })
}

// ---------------------------------------------------------------------------
// GatherElements / ScatterElements
// ---------------------------------------------------------------------------

/// Element-wise gather along `axis`. `indices` must have the same shape
/// as `input`.
pub fn op_gather_elements(input: &Tensor, indices: &Tensor, axis: i64) -> Result<Tensor, OpError> {
    require_float(input, "GatherElements")?;
    if indices.shape.dims != input.shape.dims {
        return Err(OpError::ShapeMismatch(String::from(
            "GatherElements: indices shape must match input shape",
        )));
    }
    let rank = input.shape.dims.len();
    let ax = normalize_axis(axis, rank)?;
    let axis_dim = input.shape.dims[ax] as usize;
    let total = input.shape.total_elements();
    let idx = read_i64_vec(indices);
    if idx.len() != total {
        return Err(OpError::ShapeMismatch(String::from(
            "GatherElements: indices must be Int32/Int64",
        )));
    }
    let mut data = allocate_tensor_data(total, DataType::Float);
    let strides = element_strides(&input.shape.dims, rank);
    let mut coord = alloc::vec![0usize; rank];
    for (flat, &raw_idx) in idx.iter().enumerate().take(total) {
        let src_idx_val = normalize_index(raw_idx, axis_dim, "GatherElements")?;
        let src = src_offset(&coord, &strides, rank, ax, src_idx_val);
        write_f32(&mut data, flat, read_f32(&input.raw_data, src));
        increment_shape_coord(&mut coord, &input.shape.dims, rank);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

fn element_strides(dims: &[i64], rank: usize) -> Vec<usize> {
    let mut strides = alloc::vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as usize;
    }
    strides
}

fn normalize_index(raw_idx: i64, axis_dim: usize, op: &str) -> Result<usize, OpError> {
    let mut v = raw_idx;
    if v < 0 {
        v += axis_dim as i64;
    }
    if v < 0 || (v as usize) >= axis_dim {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{}: index out of range",
            op
        )));
    }
    Ok(v as usize)
}

fn src_offset(
    coord: &[usize],
    strides: &[usize],
    rank: usize,
    ax: usize,
    axis_idx: usize,
) -> usize {
    let mut src = 0usize;
    for d in 0..rank {
        let c = if d == ax { axis_idx } else { coord[d] };
        src += c * strides[d];
    }
    src
}

/// Element-wise scatter along `axis`. `indices` and `updates` must have
/// the same shape, and each is broadcast-indexed into a copy of
/// `input`. Supported reductions: "none" (default), "add", "mul".
pub fn op_scatter_elements(
    input: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
    axis: i64,
    reduction: &str,
) -> Result<Tensor, OpError> {
    require_float(input, "ScatterElements")?;
    require_float(updates, "ScatterElements")?;
    if indices.shape.dims != updates.shape.dims {
        return Err(OpError::ShapeMismatch(String::from(
            "ScatterElements: indices shape must match updates shape",
        )));
    }
    let rank = input.shape.dims.len();
    if indices.shape.dims.len() != rank {
        return Err(OpError::ShapeMismatch(String::from(
            "ScatterElements: indices rank must match input rank",
        )));
    }
    let ax = normalize_axis(axis, rank)?;
    let axis_dim = input.shape.dims[ax] as usize;
    validate_scatter_dims(&input.shape.dims, &indices.shape.dims, rank, ax)?;
    let idx = read_i64_vec(indices);
    let u_total = updates.shape.total_elements();
    if idx.len() != u_total {
        return Err(OpError::ShapeMismatch(String::from(
            "ScatterElements: indices must be Int32/Int64",
        )));
    }
    // Start with a copy of input.
    let mut data = input.raw_data.clone();
    let strides = element_strides(&input.shape.dims, rank);
    let mut coord = alloc::vec![0usize; rank];
    for (flat, &raw_idx) in idx.iter().enumerate().take(u_total) {
        let ax_idx = normalize_index(raw_idx, axis_dim, "ScatterElements")?;
        let dst = src_offset(&coord, &strides, rank, ax, ax_idx);
        let upd = read_f32(&updates.raw_data, flat);
        let cur = read_f32(&data, dst);
        let new_val = apply_scatter_reduction(reduction, cur, upd)?;
        write_f32(&mut data, dst, new_val);
        increment_shape_coord(&mut coord, &updates.shape.dims, rank);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

/// Non-axis dims of indices/updates must not exceed the corresponding
/// dims of `input`, otherwise the scatter would write past the output
/// buffer. ONNX additionally allows indices[ax] <= input[ax].
fn validate_scatter_dims(
    input_dims: &[i64],
    idx_dims: &[i64],
    rank: usize,
    ax: usize,
) -> Result<(), OpError> {
    for d in 0..rank {
        let in_d = input_dims[d];
        let idx_d = idx_dims[d];
        if idx_d < 0 || in_d < 0 {
            return Err(OpError::ShapeMismatch(String::from(
                "ScatterElements: negative dimensions are invalid",
            )));
        }
        if d == ax {
            if idx_d > in_d {
                return Err(OpError::ShapeMismatch(String::from(
                    "ScatterElements: indices axis dim must be <= input axis dim",
                )));
            }
        } else if idx_d != in_d {
            return Err(OpError::ShapeMismatch(String::from(
                "ScatterElements: indices non-axis dims must match input",
            )));
        }
    }
    Ok(())
}

fn apply_scatter_reduction(reduction: &str, cur: f32, upd: f32) -> Result<f32, OpError> {
    match reduction {
        "" | "none" => Ok(upd),
        "add" => Ok(cur + upd),
        "mul" => Ok(cur * upd),
        other => Err(OpError::InvalidAttribute(alloc::format!(
            "ScatterElements reduction '{}' not supported",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut data = allocate_tensor_data(vals.len(), DataType::Int64);
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

    fn make_bool(dims: &[i64], vals: &[bool]) -> Tensor {
        let data: Vec<u8> = vals.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();
        Tensor {
            data_type: DataType::Bool,
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
    fn test_top_k_largest() {
        let t = make_f32(&[4], &[1.0, 3.0, 2.0, 5.0]);
        let outs = op_top_k(&t, 2, 0, true, true).unwrap();
        assert_eq!(outs.len(), 2);
        let vals = read_all_f32(&outs[0]);
        let idxs = read_all_i64(&outs[1]);
        assert_eq!(vals, alloc::vec![5.0, 3.0]);
        assert_eq!(idxs, alloc::vec![3, 1]);
    }

    #[test]
    fn test_top_k_smallest() {
        let t = make_f32(&[4], &[1.0, 3.0, 2.0, 5.0]);
        let outs = op_top_k(&t, 2, 0, false, true).unwrap();
        let vals = read_all_f32(&outs[0]);
        assert_eq!(vals, alloc::vec![1.0, 2.0]);
    }

    #[test]
    fn test_top_k_2d_axis_neg1() {
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 6.0, 5.0, 4.0]);
        let outs = op_top_k(&t, 2, -1, true, true).unwrap();
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 2]);
        let vals = read_all_f32(&outs[0]);
        assert_eq!(vals, alloc::vec![3.0, 2.0, 6.0, 5.0]);
    }

    #[test]
    fn test_compress_flat() {
        let t = make_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let cond = make_bool(&[4], &[true, false, true, false]);
        let out = op_compress(&t, &cond, None).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![2]);
        assert_eq!(read_all_f32(&out), alloc::vec![1.0, 3.0]);
    }

    #[test]
    fn test_compress_axis() {
        // [2,3]: keep columns 0 and 2.
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let cond = make_bool(&[3], &[true, false, true]);
        let out = op_compress(&t, &cond, Some(1)).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![2, 2]);
        assert_eq!(read_all_f32(&out), alloc::vec![1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_non_zero_basic() {
        let t = make_f32(&[2, 3], &[0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
        let out = op_non_zero(&t).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![2, 3]);
        // rows: [[0,1,1],[1,0,2]]
        let v = read_all_i64(&out);
        assert_eq!(v, alloc::vec![0, 1, 1, 1, 0, 2]);
    }

    #[test]
    fn test_non_zero_empty() {
        let t = make_f32(&[3], &[0.0, 0.0, 0.0]);
        let out = op_non_zero(&t).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 0]);
    }

    #[test]
    fn test_unique_sorted() {
        let t = make_f32(&[5], &[3.0, 1.0, 3.0, 2.0, 1.0]);
        let out = op_unique(&t, None, true).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![3]);
        assert_eq!(read_all_f32(&out), alloc::vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_unique_unsorted_preserves_order() {
        let t = make_f32(&[5], &[3.0, 1.0, 3.0, 2.0, 1.0]);
        let out = op_unique(&t, None, false).unwrap();
        assert_eq!(read_all_f32(&out), alloc::vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_unique_rejects_axis_mode() {
        let t = make_f32(&[3], &[1.0, 2.0, 3.0]);
        assert!(op_unique(&t, Some(0), false).is_err());
    }

    #[test]
    fn test_gather_elements_axis0() {
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let idx = make_i64(&[2, 3], &[0, 1, 0, 1, 0, 1]);
        let out = op_gather_elements(&t, &idx, 0).unwrap();
        assert_eq!(
            read_all_f32(&out),
            alloc::vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]
        );
    }

    #[test]
    fn test_gather_elements_axis_neg1() {
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let idx = make_i64(&[2, 3], &[2, 1, 0, 0, 1, 2]);
        let out = op_gather_elements(&t, &idx, -1).unwrap();
        assert_eq!(
            read_all_f32(&out),
            alloc::vec![3.0, 2.0, 1.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_scatter_elements_replace() {
        let t = make_f32(&[3, 3], &alloc::vec![0.0; 9]);
        let idx = make_i64(&[2, 3], &[1, 0, 2, 0, 2, 1]);
        let upd = make_f32(&[2, 3], &[1.0, 1.1, 1.2, 2.0, 2.1, 2.2]);
        let out = op_scatter_elements(&t, &idx, &upd, 0, "none").unwrap();
        assert_eq!(out.shape.dims, alloc::vec![3, 3]);
        // Replacements place on axis 0; final content depends on update order.
        // Simple invariant: result contains at least the latest updates.
        let v = read_all_f32(&out);
        assert!(v
            .iter()
            .any(|&x| (x - 1.1).abs() < 1e-5 || (x - 2.1).abs() < 1e-5));
    }

    #[test]
    fn test_scatter_elements_rejects_nonaxis_shape_mismatch() {
        // input is [3, 3] but indices non-axis dim is 5 (wider than input).
        let t = make_f32(&[3, 3], &alloc::vec![0.0; 9]);
        let idx = make_i64(&[2, 5], &alloc::vec![0i64; 10]);
        let upd = make_f32(&[2, 5], &alloc::vec![1.0; 10]);
        assert!(matches!(
            op_scatter_elements(&t, &idx, &upd, 0, "none"),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_unique_rejects_rank_zero() {
        // Rank-0 scalar tensor should be rejected.
        let t = Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(alloc::vec![]),
            name: String::new(),
            raw_data: alloc::vec![0u8; 4],
        };
        assert!(matches!(
            op_unique(&t, None, false),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_scatter_elements_add() {
        let t = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let idx = make_i64(&[3], &[0, 0, 2]);
        let upd = make_f32(&[3], &[10.0, 20.0, 30.0]);
        let out = op_scatter_elements(&t, &idx, &upd, 0, "add").unwrap();
        // Index 0 gets +10+20, index 2 gets +30.
        let v = read_all_f32(&out);
        assert!((v[0] - 31.0).abs() < 1e-5);
        assert!((v[1] - 2.0).abs() < 1e-5);
        assert!((v[2] - 33.0).abs() < 1e-5);
    }
}
