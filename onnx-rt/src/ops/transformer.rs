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

fn product(dims: &[i64]) -> usize {
    dims.iter().map(|&d| d as usize).product::<usize>().max(1)
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

fn broadcast_index(coord: &[usize], shape: &[i64]) -> usize {
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

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

/// Splits a float tensor along the given axis.
///
/// If `split_sizes` is `None`, the input is split into equal parts based on
/// the axis dimension. The number of outputs is determined by the length of
/// `split_sizes` (or by axis dim when None and dim is divisible).
pub fn op_split(
    input: &Tensor,
    axis: i64,
    split_sizes: Option<&[i64]>,
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
            // Equal split into 2 by default if no info — this matches a common
            // case where the model has num_outputs implicit.
            if axis_dim % 2 == 0 {
                alloc::vec![axis_dim / 2, axis_dim / 2]
            } else {
                return Err(OpError::InvalidAttribute(String::from(
                    "Split requires explicit sizes for non-even dim",
                )));
            }
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

/// Broadcasts the input to the target shape using NumPy semantics.
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
        let d = if id == td || id == 1 {
            td
        } else if td == 1 {
            id
        } else {
            return Err(OpError::ShapeMismatch(String::from(
                "Expand: shapes incompatible",
            )));
        };
        out_dims[max - 1 - i] = d;
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
    if values.data_type != DataType::Float || values.shape.total_elements() < 2 {
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
            crate::byte_io::write_i64(&mut data, i, v);
        }
        Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: data,
        }
    }

    fn read_all(t: &Tensor) -> Vec<f32> {
        (0..t.shape.total_elements())
            .map(|i| read_f32(&t.raw_data, i))
            .collect()
    }

    #[test]
    fn test_split_equal() {
        // [2,6] split axis=1 into 3 → three [2,2]
        let t = make_f32(
            &[2, 6],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let outs = op_split(&t, 1, Some(&[2, 2, 2])).unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 2]);
        assert_eq!(read_all(&outs[0]), alloc::vec![1.0, 2.0, 7.0, 8.0]);
        assert_eq!(read_all(&outs[1]), alloc::vec![3.0, 4.0, 9.0, 10.0]);
        assert_eq!(read_all(&outs[2]), alloc::vec![5.0, 6.0, 11.0, 12.0]);
    }

    #[test]
    fn test_split_custom_sizes() {
        let t = make_f32(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let outs = op_split(&t, 0, Some(&[2, 3, 1])).unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].shape.dims, alloc::vec![2]);
        assert_eq!(outs[1].shape.dims, alloc::vec![3]);
        assert_eq!(outs[2].shape.dims, alloc::vec![1]);
        assert_eq!(read_all(&outs[2]), alloc::vec![6.0]);
    }

    #[test]
    fn test_split_axis_validation() {
        let t = make_f32(&[3], &[1.0, 2.0, 3.0]);
        assert!(op_split(&t, 5, None).is_err());
        let bad = op_split(&t, 0, Some(&[1, 1])).unwrap_err();
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
    fn test_einsum_unsupported_equation() {
        let a = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert!(op_einsum("ii->", &[&a]).is_err());
    }
}
