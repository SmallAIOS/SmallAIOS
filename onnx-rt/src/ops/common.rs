// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the Tier 2 / Phase 1 operator modules.
//!
//! These helpers were originally copy-pasted across `math.rs`, `compare.rs`,
//! `transformer.rs`, `recurrent.rs`, `activations.rs`, `shape_data.rs`, and
//! `reduction.rs`. Centralizing them here both removes the duplication (which
//! SonarCloud flags on new code) and gives us one canonical implementation to
//! test and reason about.
//!
//! All helpers are `pub(crate)` so they are visible to every op submodule but
//! not re-exported as public API.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
use crate::operators::OpError;
use crate::tensor::{bf16_to_f32, f32_to_bf16, DataType, Tensor};

/// Returns the broadcast output shape of two NumPy-style shape vectors.
///
/// Dimensions are aligned right-to-left. A dimension of `1` broadcasts to the
/// other side's size; mismatched non-`1` dims produce
/// [`OpError::ShapeMismatch`].
pub(crate) fn broadcast_shape(a: &[i64], b: &[i64]) -> Result<Vec<i64>, OpError> {
    let max = a.len().max(b.len());
    let mut out = vec![1i64; max];
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

/// Computes the linear element index for a coordinate in a broadcast tensor.
///
/// `coord` is expressed in the *output* (broadcast) coordinate space and
/// `shape` is the input tensor's (possibly smaller) shape. Size-1 dims are
/// clamped to index `0`.
pub(crate) fn broadcast_index(coord: &[usize], shape: &[i64]) -> usize {
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

/// Advances `coord` in row-major order with respect to `dims`.
///
/// After the final coordinate the vector wraps back to zero; callers that
/// iterate over the full range should track the iteration count externally.
pub(crate) fn next_coord(coord: &mut [usize], dims: &[i64]) {
    for i in (0..coord.len()).rev() {
        coord[i] += 1;
        if (coord[i] as i64) < dims[i] {
            return;
        }
        coord[i] = 0;
    }
}

/// Returns `Ok(())` iff `t.data_type == DataType::Float`, otherwise a
/// [`OpError::ShapeMismatch`] describing the offending operator.
///
/// This flavor matches the error variant used by the math/compare/
/// transformer/recurrent/activations op modules.
pub(crate) fn require_float(t: &Tensor, op: &str) -> Result<(), OpError> {
    if t.data_type != DataType::Float {
        return Err(OpError::ShapeMismatch(format!(
            "{} requires Float input",
            op
        )));
    }
    Ok(())
}

/// Returns `Ok(())` iff `t.data_type == DataType::Float`, otherwise a
/// [`OpError::InvalidAttribute`] using the legacy "only supports float32"
/// wording from Tier 1.
///
/// This flavor matches the reduction / shape_data modules which surface
/// float-type failures as attribute errors.
pub(crate) fn require_float_attr(t: &Tensor, op: &str) -> Result<(), OpError> {
    if t.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(format!(
            "{} only supports float32",
            op
        )));
    }
    Ok(())
}

/// Element-wise unary helper. Allocates an output tensor with the same shape
/// and dtype as `input` and applies `f` to every element. BFloat16 inputs are
/// widened to `f32` for the computation and packed back to BF16 on write.
/// Non-float inputs are rejected via [`require_float`].
pub(crate) fn unary<F: Fn(f32) -> f32>(input: &Tensor, op: &str, f: F) -> Result<Tensor, OpError> {
    if input.data_type == DataType::BFloat16 {
        let values = bf16_to_f32(&input.raw_data);
        let n = input.shape.total_elements();
        if values.len() < n {
            return Err(OpError::ShapeMismatch(format!(
                "{} BF16 input buffer smaller than shape",
                op
            )));
        }
        let mut out = alloc::vec![0.0f32; n];
        for i in 0..n {
            out[i] = f(values[i]);
        }
        return Ok(Tensor {
            data_type: DataType::BFloat16,
            shape: input.shape.clone(),
            name: String::new(),
            raw_data: f32_to_bf16(&out),
        });
    }
    require_float(input, op)?;
    let n = input.shape.total_elements();
    let mut data = allocate_tensor_data(n, DataType::Float);
    for i in 0..n {
        write_f32(&mut data, i, f(read_f32(&input.raw_data, i)));
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: data,
    })
}

/// Shared test fixture constructors.
///
/// Every op submodule used to redefine `make_f32`, `read_all`, `make_i64`,
/// `make_bool`, and assorted scalar constructors in its own `#[cfg(test)]
/// mod tests` block. SonarCloud flagged the copies as duplicated code on new
/// PRs. This module centralizes them so each test module can just
/// `use crate::ops::common::test_helpers::*;`.
#[cfg(test)]
pub(crate) mod test_helpers {
    use crate::byte_io::{allocate_tensor_data, read_f32, read_i64, write_f32, write_i64};
    use crate::tensor::{DataType, Tensor, TensorShape};
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Build a Float tensor from dims + values.
    pub(crate) fn make_f32(dims: &[i64], vals: &[f32]) -> Tensor {
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

    /// Build an Int64 tensor from dims + values.
    pub(crate) fn make_i64(dims: &[i64], vals: &[i64]) -> Tensor {
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

    /// Build a Bool tensor from dims + values.
    pub(crate) fn make_bool(dims: &[i64], vals: &[bool]) -> Tensor {
        let data: Vec<u8> = vals.iter().map(|&b| u8::from(b)).collect();
        Tensor {
            data_type: DataType::Bool,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: data,
        }
    }

    /// Build a Float scalar tensor with shape `[1]`.
    pub(crate) fn scalar_f32(v: f32) -> Tensor {
        let mut data = vec![0u8; 4];
        write_f32(&mut data, 0, v);
        Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(vec![1]),
            name: String::new(),
            raw_data: data,
        }
    }

    /// Build an Int8 scalar tensor with shape `[1]`.
    pub(crate) fn scalar_i8(v: i8) -> Tensor {
        Tensor {
            data_type: DataType::Int8,
            shape: TensorShape::new(vec![1]),
            name: String::new(),
            raw_data: vec![v as u8],
        }
    }

    /// Build a Uint8 scalar tensor with shape `[1]`.
    pub(crate) fn scalar_u8(v: u8) -> Tensor {
        Tensor {
            data_type: DataType::Uint8,
            shape: TensorShape::new(vec![1]),
            name: String::new(),
            raw_data: vec![v],
        }
    }

    /// Read all f32 elements from a Float tensor's raw data.
    pub(crate) fn read_all_f32(t: &Tensor) -> Vec<f32> {
        (0..t.shape.total_elements())
            .map(|i| read_f32(&t.raw_data, i))
            .collect()
    }

    /// Read all i64 elements from an Int64 tensor's raw data.
    pub(crate) fn read_all_i64(t: &Tensor) -> Vec<i64> {
        (0..t.shape.total_elements())
            .map(|i| read_i64(&t.raw_data, i))
            .collect()
    }

    /// Read all bool elements from a Bool tensor.
    pub(crate) fn read_all_bool(t: &Tensor) -> Vec<bool> {
        t.raw_data.iter().map(|&b| b != 0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    fn tensor_f32(dims: &[i64], data: &[f32]) -> Tensor {
        let mut raw = allocate_tensor_data(data.len(), DataType::Float);
        for (i, &v) in data.iter().enumerate() {
            write_f32(&mut raw, i, v);
        }
        Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: raw,
        }
    }

    #[test]
    fn broadcast_shape_equal() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_shape_size_one_expansion() {
        assert_eq!(broadcast_shape(&[2, 1], &[1, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_shape_rank_differs() {
        assert_eq!(broadcast_shape(&[5, 4], &[4]).unwrap(), vec![5, 4]);
    }

    #[test]
    fn broadcast_shape_incompatible_rejected() {
        assert!(broadcast_shape(&[2, 3], &[2, 4]).is_err());
    }

    #[test]
    fn broadcast_index_handles_size_one_dim() {
        // Output coord [1, 2] into an input of shape [1, 3] -> index 2.
        assert_eq!(broadcast_index(&[1, 2], &[1, 3]), 2);
    }

    #[test]
    fn broadcast_index_respects_row_major_stride() {
        // shape [2, 3]; coord [1, 2] -> 1*3 + 2 = 5.
        assert_eq!(broadcast_index(&[1, 2], &[2, 3]), 5);
    }

    #[test]
    fn next_coord_increments_last_dim() {
        let mut c = [0usize, 0];
        next_coord(&mut c, &[2, 3]);
        assert_eq!(c, [0, 1]);
    }

    #[test]
    fn next_coord_carries_into_prev_dim() {
        let mut c = [0usize, 2];
        next_coord(&mut c, &[2, 3]);
        assert_eq!(c, [1, 0]);
    }

    #[test]
    fn next_coord_wraps_at_end() {
        let mut c = [1usize, 2];
        next_coord(&mut c, &[2, 3]);
        assert_eq!(c, [0, 0]);
    }

    #[test]
    fn require_float_accepts_float() {
        let t = tensor_f32(&[2], &[1.0, 2.0]);
        assert!(require_float(&t, "Op").is_ok());
    }

    #[test]
    fn require_float_rejects_int() {
        let t = Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(vec![1]),
            name: String::new(),
            raw_data: vec![0u8; 8],
        };
        let err = require_float(&t, "Op").unwrap_err();
        matches!(err, OpError::ShapeMismatch(_));
    }

    #[test]
    fn require_float_attr_rejects_int() {
        let t = Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(vec![1]),
            name: String::new(),
            raw_data: vec![0u8; 8],
        };
        let err = require_float_attr(&t, "Op").unwrap_err();
        matches!(err, OpError::InvalidAttribute(_));
    }

    #[test]
    fn unary_applies_function_elementwise() {
        let t = tensor_f32(&[3], &[1.0, 2.0, 3.0]);
        let out = unary(&t, "Double", |x| x * 2.0).unwrap();
        assert_eq!(out.shape.dims, vec![3]);
        assert_eq!(read_f32(&out.raw_data, 0), 2.0);
        assert_eq!(read_f32(&out.raw_data, 1), 4.0);
        assert_eq!(read_f32(&out.raw_data, 2), 6.0);
    }
}
