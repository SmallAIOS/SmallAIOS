// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Phase 1 reduction operators: ReduceMax, ReduceMin, ReduceProd,
//! ArgMax, ArgMin.
//!
//! ArgMax/ArgMin produce Int64 index tensors; ReduceMax/ReduceMin/
//! ReduceProd preserve the input dtype (f32 only in this runtime).

use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32, write_i64, I64_SIZE};
use crate::operators::OpError;
use crate::tensor::{DataType, Tensor, TensorShape};

fn require_float(t: &Tensor, op: &str) -> Result<(), OpError> {
    if t.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "{} only supports float32",
            op
        )));
    }
    Ok(())
}

fn resolved_axes(ndim: usize, axes: &[i64]) -> Vec<usize> {
    if axes.is_empty() {
        (0..ndim).collect()
    } else {
        axes.iter()
            .map(|&a| {
                if a < 0 {
                    (ndim as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    }
}

fn compute_out_dims(input_dims: &[i64], resolved: &[usize], keepdims: bool) -> Vec<i64> {
    let mut out_dims: Vec<i64> = (0..input_dims.len())
        .filter_map(|d| {
            if resolved.contains(&d) {
                if keepdims {
                    Some(1)
                } else {
                    None
                }
            } else {
                Some(input_dims[d])
            }
        })
        .collect();
    if out_dims.is_empty() {
        out_dims.push(1);
    }
    out_dims
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

/// Generic reduction helper for f32 reductions with an initial value and a
/// combine function.
fn reduce_float<F: Fn(f32, f32) -> f32>(
    input: &Tensor,
    axes: &[i64],
    keepdims: bool,
    op: &str,
    init: f32,
    combine: F,
) -> Result<Tensor, OpError> {
    require_float(input, op)?;
    let ndim = input.shape.ndim();
    let resolved = resolved_axes(ndim, axes);
    let out_dims = compute_out_dims(&input.shape.dims, &resolved, keepdims);
    let out_shape = TensorShape::new(out_dims);
    let total_out = out_shape.total_elements();
    let mut raw = allocate_tensor_data(total_out, DataType::Float);
    // Fill with init.
    for i in 0..total_out {
        write_f32(&mut raw, i, init);
    }

    let in_total = input.shape.total_elements();
    let out_strides = compute_strides(&out_shape.dims);
    let mut in_coord = alloc::vec![0usize; ndim];
    for i in 0..in_total {
        if i > 0 {
            next_coord(&mut in_coord, &input.shape.dims);
        }
        let mut out_idx = 0usize;
        let mut od = 0usize;
        #[allow(clippy::needless_range_loop)]
        for d in 0..ndim {
            if resolved.contains(&d) {
                if keepdims {
                    od += 1;
                }
            } else {
                if od < out_strides.len() {
                    out_idx += in_coord[d] * out_strides[od];
                }
                od += 1;
            }
        }
        let prev = read_f32(&raw, out_idx);
        let v = read_f32(&input.raw_data, i);
        write_f32(&mut raw, out_idx, combine(prev, v));
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data: raw,
    })
}

/// ReduceMax along specified axes.
pub fn op_reduce_max(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    reduce_float(
        input,
        axes,
        keepdims,
        "ReduceMax",
        f32::NEG_INFINITY,
        |a, b| if b > a { b } else { a },
    )
}

/// ReduceMin along specified axes.
pub fn op_reduce_min(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    reduce_float(input, axes, keepdims, "ReduceMin", f32::INFINITY, |a, b| {
        if b < a {
            b
        } else {
            a
        }
    })
}

/// ReduceProd along specified axes.
pub fn op_reduce_prod(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    reduce_float(input, axes, keepdims, "ReduceProd", 1.0, |a, b| a * b)
}

/// ArgMax/ArgMin core. Operates along a single axis and emits Int64 indices.
/// `select_last_index` controls whether ties pick the last matching index.
fn argreduce<F: Fn(f32, f32) -> bool>(
    input: &Tensor,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
    op: &str,
    better: F,
) -> Result<Tensor, OpError> {
    require_float(input, op)?;
    let ndim = input.shape.ndim() as i64;
    let axis = if axis < 0 { ndim + axis } else { axis };
    if axis < 0 || axis >= ndim {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "{} axis out of range",
            op
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

    let out_dims: Vec<i64> = dims
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            if i == axis {
                if keepdims {
                    Some(1)
                } else {
                    None
                }
            } else {
                Some(d)
            }
        })
        .collect();
    let out_shape = TensorShape::new(if out_dims.is_empty() {
        alloc::vec![1]
    } else {
        out_dims
    });
    let total_out = outer * inner;
    let mut raw = alloc::vec![0u8; total_out * I64_SIZE];

    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx: usize = 0;
            let mut best_val = read_f32(&input.raw_data, (o * axis_size) * inner + i);
            for a in 1..axis_size {
                let idx = (o * axis_size + a) * inner + i;
                let v = read_f32(&input.raw_data, idx);
                if better(v, best_val) || (select_last_index && v == best_val) {
                    best_val = v;
                    best_idx = a;
                }
            }
            write_i64(&mut raw, o * inner + i, best_idx as i64);
        }
    }
    Ok(Tensor {
        data_type: DataType::Int64,
        shape: out_shape,
        name: String::new(),
        raw_data: raw,
    })
}

/// ArgMax returns the index of the maximum value along `axis`.
pub fn op_arg_max(
    input: &Tensor,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
) -> Result<Tensor, OpError> {
    argreduce(
        input,
        axis,
        keepdims,
        select_last_index,
        "ArgMax",
        |v, b| v > b,
    )
}

/// ArgMin returns the index of the minimum value along `axis`.
pub fn op_arg_min(
    input: &Tensor,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
) -> Result<Tensor, OpError> {
    argreduce(
        input,
        axis,
        keepdims,
        select_last_index,
        "ArgMin",
        |v, b| v < b,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_io::read_i64;
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
    fn test_reduce_max_axis0() {
        let t = make_f32(&[2, 3], &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let out = op_reduce_max(&t, &[0], false).unwrap();
        assert_eq!(read_all_f32(&out), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reduce_max_all() {
        let t = make_f32(&[2, 2], &[1.0, 7.0, 3.0, 4.0]);
        let out = op_reduce_max(&t, &[], true).unwrap();
        assert_eq!(read_all_f32(&out), vec![7.0]);
    }

    #[test]
    fn test_reduce_min_axis1() {
        let t = make_f32(&[2, 3], &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let out = op_reduce_min(&t, &[1], false).unwrap();
        assert_eq!(read_all_f32(&out), vec![1.0, 2.0]);
    }

    #[test]
    fn test_reduce_min_negative() {
        let t = make_f32(&[3], &[-1.0, -5.0, -3.0]);
        let out = op_reduce_min(&t, &[], false).unwrap();
        assert_eq!(read_all_f32(&out), vec![-5.0]);
    }

    #[test]
    fn test_reduce_prod_axis0() {
        let t = make_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = op_reduce_prod(&t, &[0], false).unwrap();
        assert_eq!(read_all_f32(&out), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_reduce_prod_empty_axes() {
        let t = make_f32(&[3], &[2.0, 3.0, 4.0]);
        let out = op_reduce_prod(&t, &[], false).unwrap();
        assert_eq!(read_all_f32(&out), vec![24.0]);
    }

    #[test]
    fn test_arg_max_axis_last() {
        let t = make_f32(&[2, 3], &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);
        let out = op_arg_max(&t, -1, false, false).unwrap();
        assert_eq!(read_all_i64(&out), vec![1, 2]);
    }

    #[test]
    fn test_arg_max_keepdims() {
        let t = make_f32(&[2, 3], &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);
        let out = op_arg_max(&t, 1, true, false).unwrap();
        assert_eq!(out.shape.dims, vec![2, 1]);
        assert_eq!(read_all_i64(&out), vec![1, 2]);
    }

    #[test]
    fn test_arg_max_select_last_index() {
        let t = make_f32(&[4], &[1.0, 3.0, 3.0, 2.0]);
        let first = op_arg_max(&t, 0, false, false).unwrap();
        let last = op_arg_max(&t, 0, false, true).unwrap();
        assert_eq!(read_all_i64(&first), vec![1]);
        assert_eq!(read_all_i64(&last), vec![2]);
    }

    #[test]
    fn test_arg_min_basic() {
        let t = make_f32(&[2, 3], &[1.0, 3.0, 2.0, 5.0, 4.0, 0.5]);
        let out = op_arg_min(&t, 1, false, false).unwrap();
        assert_eq!(read_all_i64(&out), vec![0, 2]);
    }

    #[test]
    fn test_arg_min_invalid_axis() {
        let t = make_f32(&[3], &[1.0, 2.0, 3.0]);
        assert!(op_arg_min(&t, 5, false, false).is_err());
    }

    #[test]
    fn test_reduce_max_rejects_non_float() {
        let t = Tensor {
            data_type: DataType::Int32,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: alloc::vec![0u8; 4],
        };
        assert!(op_reduce_max(&t, &[], false).is_err());
    }
}
