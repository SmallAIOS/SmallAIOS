// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tier 2 element-wise comparison and selection operators.
//!
//! Equal, NotEqual, Less, LessOrEqual, Greater, GreaterOrEqual return Bool
//! tensors. Where selects between two float tensors based on a Bool
//! condition. Min/Max are element-wise reductions over two float tensors.
//! Not negates a Bool tensor.

use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
use crate::operators::OpError;
use crate::ops::common::{broadcast_index, broadcast_shape, next_coord, require_float};
use crate::tensor::{DataType, Tensor, TensorShape};

/// Element-wise comparison helper that returns a Bool tensor.
fn compare<F: Fn(f32, f32) -> bool>(
    a: &Tensor,
    b: &Tensor,
    op: &str,
    f: F,
) -> Result<Tensor, OpError> {
    require_float(a, op)?;
    require_float(b, op)?;
    let out_dims = broadcast_shape(&a.shape.dims, &b.shape.dims)?;
    let total: usize = out_dims.iter().map(|&d| d as usize).product::<usize>();
    // Honor zero-element tensors: allocate empty buffer and return early
    // (reading at coord 0 on a 0-total tensor would be undefined).
    if total == 0 {
        return Ok(Tensor {
            data_type: DataType::Bool,
            shape: TensorShape::new(out_dims),
            name: String::new(),
            raw_data: Vec::new(),
        });
    }
    let mut data = alloc::vec![0u8; total];
    let ndim = out_dims.len();
    let mut coord = alloc::vec![0usize; ndim];
    for (flat, slot) in data.iter_mut().enumerate().take(total) {
        let ai = broadcast_index(&coord, &a.shape.dims);
        let bi = broadcast_index(&coord, &b.shape.dims);
        let av = read_f32(&a.raw_data, ai);
        let bv = read_f32(&b.raw_data, bi);
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

fn elementwise_min_max<F: Fn(f32, f32) -> f32>(
    a: &Tensor,
    b: &Tensor,
    op: &str,
    f: F,
) -> Result<Tensor, OpError> {
    require_float(a, op)?;
    require_float(b, op)?;
    let out_dims = broadcast_shape(&a.shape.dims, &b.shape.dims)?;
    let total: usize = out_dims.iter().map(|&d| d as usize).product::<usize>();
    if total == 0 {
        return Ok(Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(out_dims),
            name: String::new(),
            raw_data: Vec::new(),
        });
    }
    let mut data = allocate_tensor_data(total, DataType::Float);
    let ndim = out_dims.len();
    let mut coord = alloc::vec![0usize; ndim];
    for flat in 0..total {
        let ai = broadcast_index(&coord, &a.shape.dims);
        let bi = broadcast_index(&coord, &b.shape.dims);
        let av = read_f32(&a.raw_data, ai);
        let bv = read_f32(&b.raw_data, bi);
        write_f32(&mut data, flat, f(av, bv));
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
// Public operators
// ---------------------------------------------------------------------------

pub fn op_equal(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    compare(a, b, "Equal", |x, y| x == y)
}

pub fn op_not_equal(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    compare(a, b, "NotEqual", |x, y| x != y)
}

pub fn op_less(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    compare(a, b, "Less", |x, y| x < y)
}

pub fn op_less_or_equal(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    compare(a, b, "LessOrEqual", |x, y| x <= y)
}

pub fn op_greater(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    compare(a, b, "Greater", |x, y| x > y)
}

pub fn op_greater_or_equal(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    compare(a, b, "GreaterOrEqual", |x, y| x >= y)
}

pub fn op_min(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    elementwise_min_max(a, b, "Min", |x, y| if x < y { x } else { y })
}

pub fn op_max(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    elementwise_min_max(a, b, "Max", |x, y| if x > y { x } else { y })
}

/// Element-wise ternary select: `out[i] = if cond[i] { x[i] } else { y[i] }`.
/// Supports broadcasting across all three inputs.
pub fn op_where(cond: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, OpError> {
    if cond.data_type != DataType::Bool {
        return Err(OpError::ShapeMismatch(String::from(
            "Where requires Bool condition",
        )));
    }
    require_float(x, "Where")?;
    require_float(y, "Where")?;

    let xy = broadcast_shape(&x.shape.dims, &y.shape.dims)?;
    let out_dims = broadcast_shape(&xy, &cond.shape.dims)?;
    let total: usize = out_dims.iter().map(|&d| d as usize).product::<usize>();
    if total == 0 {
        return Ok(Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(out_dims),
            name: String::new(),
            raw_data: Vec::new(),
        });
    }
    let mut data = allocate_tensor_data(total, DataType::Float);
    let ndim = out_dims.len();
    let mut coord = alloc::vec![0usize; ndim];
    for flat in 0..total {
        let ci = broadcast_index(&coord, &cond.shape.dims);
        let xi = broadcast_index(&coord, &x.shape.dims);
        let yi = broadcast_index(&coord, &y.shape.dims);
        let v = if cond.raw_data[ci] != 0 {
            read_f32(&x.raw_data, xi)
        } else {
            read_f32(&y.raw_data, yi)
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

/// Element-wise boolean negation. Input must be a Bool tensor.
pub fn op_not(input: &Tensor) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Bool {
        return Err(OpError::ShapeMismatch(String::from(
            "Not requires Bool input",
        )));
    }
    let mut out = input.raw_data.clone();
    for byte in out.iter_mut() {
        *byte = if *byte == 0 { 1 } else { 0 };
    }
    Ok(Tensor {
        data_type: DataType::Bool,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: out,
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

    fn make_bool(dims: &[i64], vals: &[bool]) -> Tensor {
        let data: Vec<u8> = vals.iter().map(|&b| if b { 1 } else { 0 }).collect();
        Tensor {
            data_type: DataType::Bool,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: data,
        }
    }

    fn read_bool_all(t: &Tensor) -> Vec<bool> {
        t.raw_data.iter().map(|&b| b != 0).collect()
    }

    fn read_f32_all(t: &Tensor) -> Vec<f32> {
        (0..t.shape.total_elements())
            .map(|i| read_f32(&t.raw_data, i))
            .collect()
    }

    #[test]
    fn test_equal() {
        let a = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let b = make_f32(&[3], &[1.0, 5.0, 3.0]);
        let v = read_bool_all(&op_equal(&a, &b).unwrap());
        assert_eq!(v, alloc::vec![true, false, true]);
    }

    #[test]
    fn test_not_equal() {
        let a = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let b = make_f32(&[3], &[1.0, 5.0, 3.0]);
        let v = read_bool_all(&op_not_equal(&a, &b).unwrap());
        assert_eq!(v, alloc::vec![false, true, false]);
    }

    #[test]
    fn test_less_and_greater() {
        let a = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let b = make_f32(&[3], &[2.0, 2.0, 1.0]);
        assert_eq!(
            read_bool_all(&op_less(&a, &b).unwrap()),
            alloc::vec![true, false, false]
        );
        assert_eq!(
            read_bool_all(&op_greater(&a, &b).unwrap()),
            alloc::vec![false, false, true]
        );
        assert_eq!(
            read_bool_all(&op_less_or_equal(&a, &b).unwrap()),
            alloc::vec![true, true, false]
        );
        assert_eq!(
            read_bool_all(&op_greater_or_equal(&a, &b).unwrap()),
            alloc::vec![false, true, true]
        );
    }

    #[test]
    fn test_min_max() {
        let a = make_f32(&[3], &[1.0, 5.0, 3.0]);
        let b = make_f32(&[3], &[2.0, 4.0, 3.0]);
        assert_eq!(
            read_f32_all(&op_min(&a, &b).unwrap()),
            alloc::vec![1.0, 4.0, 3.0]
        );
        assert_eq!(
            read_f32_all(&op_max(&a, &b).unwrap()),
            alloc::vec![2.0, 5.0, 3.0]
        );
    }

    #[test]
    fn test_min_max_broadcast() {
        let a = make_f32(&[2, 2], &[1.0, 5.0, 3.0, 7.0]);
        let b = make_f32(&[1], &[4.0]);
        assert_eq!(
            read_f32_all(&op_min(&a, &b).unwrap()),
            alloc::vec![1.0, 4.0, 3.0, 4.0]
        );
        assert_eq!(
            read_f32_all(&op_max(&a, &b).unwrap()),
            alloc::vec![4.0, 5.0, 4.0, 7.0]
        );
    }

    #[test]
    fn test_where() {
        let cond = make_bool(&[3], &[true, false, true]);
        let x = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let y = make_f32(&[3], &[10.0, 20.0, 30.0]);
        let out = op_where(&cond, &x, &y).unwrap();
        assert_eq!(read_f32_all(&out), alloc::vec![1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_where_broadcast() {
        let cond = make_bool(&[1], &[true]);
        let x = make_f32(&[3], &[1.0, 2.0, 3.0]);
        let y = make_f32(&[3], &[10.0, 20.0, 30.0]);
        let out = op_where(&cond, &x, &y).unwrap();
        assert_eq!(read_f32_all(&out), alloc::vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_not() {
        let t = make_bool(&[3], &[true, false, true]);
        let out = op_not(&t).unwrap();
        assert_eq!(read_bool_all(&out), alloc::vec![false, true, false]);
    }

    #[test]
    fn test_compare_broadcast() {
        let a = make_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = make_f32(&[1], &[2.0]);
        let v = read_bool_all(&op_less(&a, &b).unwrap());
        assert_eq!(v, alloc::vec![true, false, false, false]);
    }

    #[test]
    fn test_compare_zero_element_tensor() {
        // Shape [0] → output should be 0 elements, NOT 1.
        let a = make_f32(&[0], &[]);
        let b = make_f32(&[0], &[]);
        let out = op_equal(&a, &b).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![0]);
        assert!(out.raw_data.is_empty());
    }

    #[test]
    fn test_min_max_zero_element_tensor() {
        let a = make_f32(&[0, 3], &[]);
        let b = make_f32(&[0, 3], &[]);
        let out = op_min(&a, &b).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![0, 3]);
        assert!(out.raw_data.is_empty());
    }

    #[test]
    fn test_compare_rejects_non_float() {
        let t = Tensor {
            data_type: DataType::Int32,
            shape: TensorShape::new(alloc::vec![1]),
            name: String::new(),
            raw_data: alloc::vec![0u8; 4],
        };
        let f = make_f32(&[1], &[1.0]);
        assert!(op_equal(&t, &f).is_err());
    }
}
