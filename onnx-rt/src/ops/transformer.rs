// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Transformer building-block operators: Split, Expand, Tile, OneHot,
//! Einsum.
//!
//! Einsum supports the equation patterns commonly used in transformer
//! attention (e.g. `bij,bjk->bik`, `bhij,bhkj->bhik`, `ij,jk->ik`).
//! Generic einsum implementations are out of scope; the parser dispatches
//! to a small set of contraction templates.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, read_i64, write_f32};
use crate::operators::OpError;
use crate::ops::common::{broadcast_index, next_coord, require_float};
use crate::tensor::{DataType, Tensor, TensorShape};

fn product(dims: &[i64]) -> usize {
    dims.iter().map(|&d| d as usize).product::<usize>().max(1)
}

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

/// Splits a float tensor along the given axis.
///
/// If `split_sizes` is `Some`, the input is split according to those sizes.
/// If `split_sizes` is `None`, `num_outputs` (derived from the node's output
/// arity by the dispatcher) is required and the axis dim must be evenly
/// divisible by it. This matches ONNX Split-13/Split-18 semantics.
pub fn op_split(
    input: &Tensor,
    axis: i64,
    split_sizes: Option<&[i64]>,
    num_outputs: Option<usize>,
) -> Result<Vec<Tensor>, OpError> {
    require_float(input, "Split")?;
    let ndim = input.shape.dims.len() as i64;
    let axis = if axis < 0 { axis + ndim } else { axis };
    if axis < 0 || axis >= ndim {
        return Err(OpError::InvalidAttribute(String::from(
            "Split: axis out of range",
        )));
    }
    let axis = axis as usize;
    let axis_dim = input.shape.dims[axis];

    let sizes: Vec<i64> = match split_sizes {
        Some(s) => s.to_vec(),
        None => {
            // Equal-split path: we need to know how many outputs the node
            // declares. ONNX derives this from the node's output arity.
            let n = num_outputs.ok_or_else(|| {
                OpError::InvalidAttribute(String::from(
                    "Split: equal-split path requires explicit num_outputs",
                ))
            })?;
            if n == 0 {
                return Err(OpError::InvalidAttribute(String::from(
                    "Split: num_outputs must be > 0",
                )));
            }
            if axis_dim < 0 || !(axis_dim as usize).is_multiple_of(n) {
                return Err(OpError::InvalidAttribute(String::from(
                    "Split: axis dim not divisible by num_outputs",
                )));
            }
            let each = axis_dim / (n as i64);
            alloc::vec![each; n]
        }
    };
    let sum: i64 = sizes.iter().sum();
    if sum != axis_dim {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "Split sizes sum {} != axis dim {}",
            sum,
            axis_dim
        )));
    }

    // Compute outer/inner sizes around axis.
    let outer: usize = input.shape.dims[..axis]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let inner: usize = input.shape.dims[axis + 1..]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let axis_dim_us = axis_dim as usize;

    let mut outputs = Vec::with_capacity(sizes.len());
    let mut axis_offset = 0usize;
    for &size in &sizes {
        let size_us = size as usize;
        let mut new_dims = input.shape.dims.clone();
        new_dims[axis] = size;
        let total = outer * size_us * inner;
        let mut data = allocate_tensor_data(total, DataType::Float);
        for o in 0..outer {
            for s in 0..size_us {
                for i in 0..inner {
                    let src_idx = o * axis_dim_us * inner + (axis_offset + s) * inner + i;
                    let dst_idx = o * size_us * inner + s * inner + i;
                    write_f32(&mut data, dst_idx, read_f32(&input.raw_data, src_idx));
                }
            }
        }
        outputs.push(Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(new_dims),
            name: String::new(),
            raw_data: data,
        });
        axis_offset += size_us;
    }
    Ok(outputs)
}

// ---------------------------------------------------------------------------
// Expand
// ---------------------------------------------------------------------------

/// Broadcasts the input to the target shape using ONNX Expand semantics.
///
/// ONNX Expand requires that the output shape exactly matches the (rank-
/// aligned) target shape. For each aligned dim, the input dim must equal
/// the target dim or be 1 (broadcast); any other mismatch is rejected.
pub fn op_expand(input: &Tensor, target: &[i64]) -> Result<Tensor, OpError> {
    require_float(input, "Expand")?;
    // Validate input shape can broadcast to target.
    let in_dims = &input.shape.dims;
    let max = in_dims.len().max(target.len());
    let mut out_dims = alloc::vec![1i64; max];
    for i in 0..max {
        let id = if i < in_dims.len() {
            in_dims[in_dims.len() - 1 - i]
        } else {
            1
        };
        let td = if i < target.len() {
            target[target.len() - 1 - i]
        } else {
            1
        };
        if td < 0 {
            return Err(OpError::InvalidAttribute(String::from(
                "Expand: target shape must be non-negative",
            )));
        }
        // ONNX Expand: output is exactly the (rank-aligned) target shape;
        // the input dim must be either equal to it or 1.
        if !(id == td || id == 1) {
            return Err(OpError::ShapeMismatch(String::from(
                "Expand: input dim must equal target or be 1",
            )));
        }
        out_dims[max - 1 - i] = td;
    }
    let total = product(&out_dims);
    let mut data = allocate_tensor_data(total, DataType::Float);
    let mut coord = alloc::vec![0usize; out_dims.len()];
    for flat in 0..total {
        let src = broadcast_index(&coord, in_dims);
        write_f32(&mut data, flat, read_f32(&input.raw_data, src));
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

// ---------------------------------------------------------------------------
// Tile
// ---------------------------------------------------------------------------

/// Repeats the input tensor along each axis according to `repeats`.
pub fn op_tile(input: &Tensor, repeats: &[i64]) -> Result<Tensor, OpError> {
    require_float(input, "Tile")?;
    if repeats.len() != input.shape.dims.len() {
        return Err(OpError::ShapeMismatch(String::from(
            "Tile: repeats length must match input rank",
        )));
    }
    if repeats.iter().any(|&r| r < 0) {
        return Err(OpError::InvalidAttribute(String::from(
            "Tile: repeats must be non-negative",
        )));
    }
    let in_dims = &input.shape.dims;
    let out_dims: Vec<i64> = in_dims
        .iter()
        .zip(repeats.iter())
        .map(|(&d, &r)| d * r)
        .collect();
    let total = product(&out_dims);
    let mut data = allocate_tensor_data(total, DataType::Float);
    let mut coord = alloc::vec![0usize; out_dims.len()];
    for flat in 0..total {
        // Map output coord to input coord by modulo input dim.
        let src_idx = {
            let mut idx = 0usize;
            let mut stride = 1usize;
            for i in (0..in_dims.len()).rev() {
                let c = coord[i] % in_dims[i] as usize;
                idx += c * stride;
                stride *= in_dims[i] as usize;
            }
            idx
        };
        write_f32(&mut data, flat, read_f32(&input.raw_data, src_idx));
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

// ---------------------------------------------------------------------------
// OneHot
// ---------------------------------------------------------------------------

/// Produces a one-hot tensor from integer indices.
///
/// `values[0]` is `off_value`, `values[1]` is `on_value`. New axis is
/// inserted at `axis` of the output.
pub fn op_one_hot(
    indices: &Tensor,
    depth: i64,
    values: &Tensor,
    axis: i64,
) -> Result<Tensor, OpError> {
    if values.data_type != DataType::Float || values.shape.total_elements() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "OneHot values must be Float tensor of length 2",
        )));
    }
    if depth <= 0 {
        return Err(OpError::InvalidAttribute(String::from(
            "OneHot depth must be > 0",
        )));
    }
    let off_value = read_f32(&values.raw_data, 0);
    let on_value = read_f32(&values.raw_data, 1);

    // Decode indices as i64. Accept Int64 only for simplicity.
    let n_indices = indices.shape.total_elements();
    let idx_vals: Vec<i64> = match indices.data_type {
        DataType::Int64 => (0..n_indices)
            .map(|i| read_i64(&indices.raw_data, i))
            .collect(),
        DataType::Int32 => (0..n_indices)
            .map(|i| crate::byte_io::read_i32(&indices.raw_data, i) as i64)
            .collect(),
        _ => {
            return Err(OpError::ShapeMismatch(String::from(
                "OneHot indices must be Int32 or Int64",
            )));
        }
    };

    // Build output shape: insert depth at axis position.
    let in_dims = &indices.shape.dims;
    let in_rank = in_dims.len() as i64;
    let axis_norm = if axis < 0 { axis + in_rank + 1 } else { axis };
    if axis_norm < 0 || axis_norm > in_rank {
        return Err(OpError::InvalidAttribute(String::from(
            "OneHot axis out of range",
        )));
    }
    let axis_pos = axis_norm as usize;
    let mut out_dims = in_dims.clone();
    out_dims.insert(axis_pos, depth);
    let total = product(&out_dims);
    let mut data = allocate_tensor_data(total, DataType::Float);

    // Outer/inner sizes around axis_pos in the output.
    let outer: usize = out_dims[..axis_pos]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let inner: usize = out_dims[axis_pos + 1..]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let depth_us = depth as usize;
    // Fill with off_value first.
    for i in 0..total {
        write_f32(&mut data, i, off_value);
    }
    // For each index, set on_value at the right position.
    for o in 0..outer {
        for i in 0..inner {
            // The index for this (outer, inner) cell is at position o*inner + i
            // in the flattened indices tensor.
            let idx_flat = o * inner + i;
            if idx_flat >= idx_vals.len() {
                continue;
            }
            let idx = idx_vals[idx_flat];
            if idx >= 0 && (idx as usize) < depth_us {
                let dst = o * depth_us * inner + (idx as usize) * inner + i;
                write_f32(&mut data, dst, on_value);
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
// Einsum
// ---------------------------------------------------------------------------

/// Computes Einstein summation for a small set of well-known patterns
/// commonly used in transformer attention.
///
/// Supported patterns:
/// - `ij,jk->ik` — 2D matrix multiply
/// - `bij,bjk->bik` — batched matmul
/// - `bhij,bhkj->bhik` — batched/headed Q·K^T
pub fn op_einsum(equation: &str, inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    let eq = equation.replace(' ', "");
    let eq = eq.as_str();
    match (eq, inputs.len()) {
        ("ij,jk->ik", 2) => einsum_matmul(inputs[0], inputs[1]),
        ("bij,bjk->bik", 2) => einsum_batched_matmul(inputs[0], inputs[1]),
        ("bhij,bhkj->bhik", 2) => einsum_qkt(inputs[0], inputs[1]),
        _ => Err(OpError::InvalidAttribute(alloc::format!(
            "Einsum equation not supported: {}",
            equation
        ))),
    }
}

fn einsum_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    require_float(a, "Einsum")?;
    require_float(b, "Einsum")?;
    if a.shape.dims.len() != 2 || b.shape.dims.len() != 2 {
        return Err(OpError::ShapeMismatch(
            "Einsum ij,jk requires 2D".to_string(),
        ));
    }
    let m = a.shape.dims[0] as usize;
    let k = a.shape.dims[1] as usize;
    let k2 = b.shape.dims[0] as usize;
    let n = b.shape.dims[1] as usize;
    if k != k2 {
        return Err(OpError::ShapeMismatch("Einsum dim mismatch".to_string()));
    }
    let mut data = allocate_tensor_data(m * n, DataType::Float);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += read_f32(&a.raw_data, i * k + kk) * read_f32(&b.raw_data, kk * n + j);
            }
            write_f32(&mut data, i * n + j, sum);
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![m as i64, n as i64]),
        name: String::new(),
        raw_data: data,
    })
}

fn einsum_batched_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    require_float(a, "Einsum")?;
    require_float(b, "Einsum")?;
    if a.shape.dims.len() != 3 || b.shape.dims.len() != 3 {
        return Err(OpError::ShapeMismatch(
            "Einsum bij,bjk requires 3D".to_string(),
        ));
    }
    let bsz = a.shape.dims[0] as usize;
    let m = a.shape.dims[1] as usize;
    let k = a.shape.dims[2] as usize;
    if b.shape.dims[0] as usize != bsz || b.shape.dims[1] as usize != k {
        return Err(OpError::ShapeMismatch("Einsum dim mismatch".to_string()));
    }
    let n = b.shape.dims[2] as usize;
    let mut data = allocate_tensor_data(bsz * m * n, DataType::Float);
    for batch in 0..bsz {
        let a_off = batch * m * k;
        let b_off = batch * k * n;
        let o_off = batch * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += read_f32(&a.raw_data, a_off + i * k + kk)
                        * read_f32(&b.raw_data, b_off + kk * n + j);
                }
                write_f32(&mut data, o_off + i * n + j, sum);
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![bsz as i64, m as i64, n as i64]),
        name: String::new(),
        raw_data: data,
    })
}

fn einsum_qkt(q: &Tensor, k: &Tensor) -> Result<Tensor, OpError> {
    // bhij,bhkj->bhik : Q is (b,h,i,j), K is (b,h,k,j), out is (b,h,i,k)
    require_float(q, "Einsum")?;
    require_float(k, "Einsum")?;
    if q.shape.dims.len() != 4 || k.shape.dims.len() != 4 {
        return Err(OpError::ShapeMismatch(
            "Einsum bhij requires 4D".to_string(),
        ));
    }
    let b = q.shape.dims[0] as usize;
    let h = q.shape.dims[1] as usize;
    let i_dim = q.shape.dims[2] as usize;
    let j_dim = q.shape.dims[3] as usize;
    if k.shape.dims[0] as usize != b
        || k.shape.dims[1] as usize != h
        || k.shape.dims[3] as usize != j_dim
    {
        return Err(OpError::ShapeMismatch("Einsum dim mismatch".to_string()));
    }
    let k_dim = k.shape.dims[2] as usize;
    let total = b * h * i_dim * k_dim;
    let mut data = allocate_tensor_data(total, DataType::Float);
    for bb in 0..b {
        for hh in 0..h {
            let q_off = ((bb * h) + hh) * i_dim * j_dim;
            let k_off = ((bb * h) + hh) * k_dim * j_dim;
            let o_off = ((bb * h) + hh) * i_dim * k_dim;
            for ii in 0..i_dim {
                for kk in 0..k_dim {
                    let mut sum = 0.0f32;
                    for jj in 0..j_dim {
                        sum += read_f32(&q.raw_data, q_off + ii * j_dim + jj)
                            * read_f32(&k.raw_data, k_off + kk * j_dim + jj);
                    }
                    write_f32(&mut data, o_off + ii * k_dim + kk, sum);
                }
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![b as i64, h as i64, i_dim as i64, k_dim as i64]),
        name: String::new(),
        raw_data: data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::common::test_helpers::{make_f32, make_i64, read_all_f32 as read_all};

    #[test]
    fn test_split_equal() {
        // [2,6] split axis=1 into 3 → three [2,2]
        let t = make_f32(
            &[2, 6],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let outs = op_split(&t, 1, Some(&[2, 2, 2]), None).unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 2]);
        assert_eq!(read_all(&outs[0]), alloc::vec![1.0, 2.0, 7.0, 8.0]);
        assert_eq!(read_all(&outs[1]), alloc::vec![3.0, 4.0, 9.0, 10.0]);
        assert_eq!(read_all(&outs[2]), alloc::vec![5.0, 6.0, 11.0, 12.0]);
    }

    #[test]
    fn test_split_custom_sizes() {
        let t = make_f32(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let outs = op_split(&t, 0, Some(&[2, 3, 1]), None).unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].shape.dims, alloc::vec![2]);
        assert_eq!(outs[1].shape.dims, alloc::vec![3]);
        assert_eq!(outs[2].shape.dims, alloc::vec![1]);
        assert_eq!(read_all(&outs[2]), alloc::vec![6.0]);
    }

    #[test]
    fn test_split_axis_validation() {
        let t = make_f32(&[3], &[1.0, 2.0, 3.0]);
        assert!(op_split(&t, 5, None, Some(2)).is_err());
        let bad = op_split(&t, 0, Some(&[1, 1]), None).unwrap_err();
        assert!(matches!(bad, OpError::ShapeMismatch(_)));
    }

    #[test]
    fn test_expand_scalar_to_matrix() {
        let t = make_f32(&[1], &[7.0]);
        let out = op_expand(&t, &[2, 3]).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![2, 3]);
        assert_eq!(read_all(&out), alloc::vec![7.0; 6]);
    }

    #[test]
    fn test_expand_broadcast_axis() {
        // [1,3] → [2,3]
        let t = make_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        let out = op_expand(&t, &[2, 3]).unwrap();
        assert_eq!(read_all(&out), alloc::vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tile_2d() {
        // [2,3] tiled by [2,1] → [4,3]
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = op_tile(&t, &[2, 1]).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![4, 3]);
        assert_eq!(
            read_all(&out),
            alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_tile_inner() {
        let t = make_f32(&[2], &[1.0, 2.0]);
        let out = op_tile(&t, &[3]).unwrap();
        assert_eq!(read_all(&out), alloc::vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_one_hot_basic() {
        let indices = make_i64(&[3], &[0, 1, 2]);
        let values = make_f32(&[2], &[0.0, 1.0]);
        let out = op_one_hot(&indices, 3, &values, -1).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![3, 3]);
        // Identity-like
        let v = read_all(&out);
        assert_eq!(v, alloc::vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_one_hot_oob_index_no_op() {
        let indices = make_i64(&[3], &[0, 5, -1]);
        let values = make_f32(&[2], &[0.0, 1.0]);
        let out = op_one_hot(&indices, 3, &values, -1).unwrap();
        let v = read_all(&out);
        // First row hit; rows for 5 and -1 stay at off_value
        assert_eq!(v[0], 1.0);
        assert_eq!(v[3], 0.0);
        assert_eq!(v[6], 0.0);
    }

    #[test]
    fn test_einsum_2d_matmul() {
        let a = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_f32(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = op_einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![2, 2]);
        // [[22,28],[49,64]]
        assert_eq!(read_all(&out), alloc::vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_einsum_batched_matmul() {
        // batch=2, [2,2,3] @ [2,3,2] → [2,2,2]
        let a = make_f32(
            &[2, 2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let b = make_f32(
            &[2, 3, 2],
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        let out = op_einsum("bij,bjk->bik", &[&a, &b]).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![2, 2, 2]);
        // First batch: [[1,2],[4,5]]
        // Second batch: [[1,0],[0,1]]
        let v = read_all(&out);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 4.0);
        assert_eq!(v[3], 5.0);
        assert_eq!(v[4], 1.0);
        assert_eq!(v[7], 1.0);
    }

    #[test]
    fn test_einsum_qkt() {
        // Q,K shape (1,1,2,3), expected QK^T shape (1,1,2,2)
        let q = make_f32(&[1, 1, 2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let k = make_f32(&[1, 1, 2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let out = op_einsum("bhij,bhkj->bhik", &[&q, &k]).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 1, 2, 2]);
        // Identity-like dot products: [[1,0],[0,1]]
        assert_eq!(read_all(&out), alloc::vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_split_equal_2way() {
        // [6] split into 2 equal parts
        let t = make_f32(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let outs = op_split(&t, 0, None, Some(2)).unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].shape.dims, alloc::vec![3]);
        assert_eq!(read_all(&outs[0]), alloc::vec![1.0, 2.0, 3.0]);
        assert_eq!(read_all(&outs[1]), alloc::vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_split_equal_3way_qkv() {
        // QKV pattern: [2,6] split axis=1 into 3 equal [2,2]
        let t = make_f32(
            &[2, 6],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let outs = op_split(&t, 1, None, Some(3)).unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 2]);
    }

    #[test]
    fn test_split_equal_4way() {
        let t = make_f32(&[8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let outs = op_split(&t, 0, None, Some(4)).unwrap();
        assert_eq!(outs.len(), 4);
        for out in &outs {
            assert_eq!(out.shape.dims, alloc::vec![2]);
        }
    }

    #[test]
    fn test_split_equal_requires_num_outputs() {
        // Without num_outputs and without sizes, must error.
        let t = make_f32(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let err = op_split(&t, 0, None, None).unwrap_err();
        assert!(matches!(err, OpError::InvalidAttribute(_)));
    }

    #[test]
    fn test_split_equal_not_divisible() {
        let t = make_f32(&[7], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert!(op_split(&t, 0, None, Some(3)).is_err());
    }

    #[test]
    fn test_expand_valid_broadcast() {
        // [1,3] -> [2,3] exactly, no "reverse-broadcast" allowed
        let t = make_f32(&[1, 3], &[1.0, 2.0, 3.0]);
        let out = op_expand(&t, &[2, 3]).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![2, 3]);
    }

    #[test]
    fn test_expand_rejects_incompatible_target() {
        // [4,3] cannot expand to [1,3] — that would shrink the output.
        let t = make_f32(&[4, 3], &alloc::vec![0.0f32; 12]);
        let err = op_expand(&t, &[1, 3]).unwrap_err();
        assert!(matches!(err, OpError::ShapeMismatch(_)));
    }

    #[test]
    fn test_tile_rejects_negative_repeats() {
        let t = make_f32(&[2], &[1.0, 2.0]);
        let err = op_tile(&t, &[-1]).unwrap_err();
        assert!(matches!(err, OpError::InvalidAttribute(_)));
    }

    #[test]
    fn test_one_hot_rejects_wrong_values_length() {
        let indices = make_i64(&[2], &[0, 1]);
        let three_values = make_f32(&[3], &[0.0, 1.0, 2.0]);
        let err = op_one_hot(&indices, 3, &three_values, -1).unwrap_err();
        assert!(matches!(err, OpError::ShapeMismatch(_)));
    }

    #[test]
    fn test_einsum_unsupported_equation() {
        let a = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert!(op_einsum("ii->", &[&a]).is_err());
    }
}
