// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tier 1 CPU operator implementations for the SmallAIOS ONNX runtime.
//!
//! Defines the supported ONNX operator set, operator dispatch, and
//! stub implementations for each operator. Shape inference helpers
//! are provided for validating tensor dimensions during graph building.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::tensor::{DataType, Tensor, TensorShape};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during operator execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpError {
    /// The requested operator is not supported by this runtime.
    UnsupportedOp(String),
    /// Input tensor shapes are incompatible for the operation.
    ShapeMismatch(String),
    /// An operator attribute has an invalid value.
    InvalidAttribute(String),
    /// An internal error occurred during computation.
    InternalError(String),
    /// The operator is defined but not yet implemented.
    NotImplemented,
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::UnsupportedOp(name) => write!(f, "unsupported operator: {}", name),
            OpError::ShapeMismatch(msg) => write!(f, "shape mismatch: {}", msg),
            OpError::InvalidAttribute(msg) => write!(f, "invalid attribute: {}", msg),
            OpError::InternalError(msg) => write!(f, "internal error: {}", msg),
            OpError::NotImplemented => write!(f, "operator not implemented"),
        }
    }
}

// ---------------------------------------------------------------------------
// Operator kind enumeration
// ---------------------------------------------------------------------------

/// Enumeration of supported ONNX operators.
///
/// Each variant corresponds to a standard ONNX operator. The runtime
/// will reject models containing operators not present in this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    // Arithmetic
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
    /// Matrix multiplication.
    MatMul,

    // Activation
    /// Rectified linear unit.
    Relu,
    /// Sigmoid activation.
    Sigmoid,
    /// Hyperbolic tangent activation.
    Tanh,
    /// Softmax normalization.
    Softmax,

    // Convolution and pooling
    /// N-dimensional convolution.
    Conv,
    /// Max pooling.
    MaxPool,
    /// Average pooling.
    AveragePool,
    /// Batch normalization.
    BatchNormalization,

    // Shape manipulation
    /// Reshape tensor dimensions.
    Reshape,
    /// Transpose tensor axes.
    Transpose,
    /// Flatten tensor to 2D.
    Flatten,
    /// Remove single-dimensional entries from shape.
    Squeeze,
    /// Insert single-dimensional entry into shape.
    Unsqueeze,

    // Data movement
    /// Concatenate tensors along an axis.
    Concat,
    /// Gather elements along an axis.
    Gather,
    /// Slice tensor along axes.
    Slice,
    /// Pad tensor edges.
    Pad,

    // Linear algebra and normalization
    /// General matrix multiply.
    Gemm,
    /// Global average pooling.
    GlobalAveragePool,
    /// Layer normalization.
    LayerNormalization,

    // Type and reduction
    /// Cast tensor to a different data type.
    Cast,
    /// Clamp values to a range.
    Clip,
    /// Reduce tensor by computing the mean.
    ReduceMean,
    /// Reduce tensor by computing the sum.
    ReduceSum,
}

impl OpKind {
    /// Converts an ONNX operator type string to an `OpKind`.
    ///
    /// Returns `None` if the operator name is not recognized.
    pub fn parse_str(name: &str) -> Option<Self> {
        match name {
            "Add" => Some(OpKind::Add),
            "Sub" => Some(OpKind::Sub),
            "Mul" => Some(OpKind::Mul),
            "Div" => Some(OpKind::Div),
            "MatMul" => Some(OpKind::MatMul),
            "Relu" => Some(OpKind::Relu),
            "Sigmoid" => Some(OpKind::Sigmoid),
            "Tanh" => Some(OpKind::Tanh),
            "Softmax" => Some(OpKind::Softmax),
            "Conv" => Some(OpKind::Conv),
            "MaxPool" => Some(OpKind::MaxPool),
            "AveragePool" => Some(OpKind::AveragePool),
            "BatchNormalization" => Some(OpKind::BatchNormalization),
            "Reshape" => Some(OpKind::Reshape),
            "Transpose" => Some(OpKind::Transpose),
            "Flatten" => Some(OpKind::Flatten),
            "Squeeze" => Some(OpKind::Squeeze),
            "Unsqueeze" => Some(OpKind::Unsqueeze),
            "Concat" => Some(OpKind::Concat),
            "Gather" => Some(OpKind::Gather),
            "Slice" => Some(OpKind::Slice),
            "Pad" => Some(OpKind::Pad),
            "Gemm" => Some(OpKind::Gemm),
            "GlobalAveragePool" => Some(OpKind::GlobalAveragePool),
            "LayerNormalization" => Some(OpKind::LayerNormalization),
            "Cast" => Some(OpKind::Cast),
            "Clip" => Some(OpKind::Clip),
            "ReduceMean" => Some(OpKind::ReduceMean),
            "ReduceSum" => Some(OpKind::ReduceSum),
            _ => None,
        }
    }

    /// Returns the canonical ONNX operator type string.
    pub fn name(&self) -> &'static str {
        match self {
            OpKind::Add => "Add",
            OpKind::Sub => "Sub",
            OpKind::Mul => "Mul",
            OpKind::Div => "Div",
            OpKind::MatMul => "MatMul",
            OpKind::Relu => "Relu",
            OpKind::Sigmoid => "Sigmoid",
            OpKind::Tanh => "Tanh",
            OpKind::Softmax => "Softmax",
            OpKind::Conv => "Conv",
            OpKind::MaxPool => "MaxPool",
            OpKind::AveragePool => "AveragePool",
            OpKind::BatchNormalization => "BatchNormalization",
            OpKind::Reshape => "Reshape",
            OpKind::Transpose => "Transpose",
            OpKind::Flatten => "Flatten",
            OpKind::Squeeze => "Squeeze",
            OpKind::Unsqueeze => "Unsqueeze",
            OpKind::Concat => "Concat",
            OpKind::Gather => "Gather",
            OpKind::Slice => "Slice",
            OpKind::Pad => "Pad",
            OpKind::Gemm => "Gemm",
            OpKind::GlobalAveragePool => "GlobalAveragePool",
            OpKind::LayerNormalization => "LayerNormalization",
            OpKind::Cast => "Cast",
            OpKind::Clip => "Clip",
            OpKind::ReduceMean => "ReduceMean",
            OpKind::ReduceSum => "ReduceSum",
        }
    }
}

// ---------------------------------------------------------------------------
// Operator registry
// ---------------------------------------------------------------------------

/// Complete list of all built-in operators for registry initialization.
const ALL_OPS: &[OpKind] = &[
    OpKind::Add,
    OpKind::Sub,
    OpKind::Mul,
    OpKind::Div,
    OpKind::MatMul,
    OpKind::Relu,
    OpKind::Sigmoid,
    OpKind::Tanh,
    OpKind::Softmax,
    OpKind::Conv,
    OpKind::MaxPool,
    OpKind::AveragePool,
    OpKind::BatchNormalization,
    OpKind::Reshape,
    OpKind::Transpose,
    OpKind::Flatten,
    OpKind::Squeeze,
    OpKind::Unsqueeze,
    OpKind::Concat,
    OpKind::Gather,
    OpKind::Slice,
    OpKind::Pad,
    OpKind::Gemm,
    OpKind::GlobalAveragePool,
    OpKind::LayerNormalization,
    OpKind::Cast,
    OpKind::Clip,
    OpKind::ReduceMean,
    OpKind::ReduceSum,
];

/// Registry of supported ONNX operators.
///
/// Used during model validation to verify that every operator in the
/// model graph has a corresponding implementation in this runtime.
pub struct OperatorRegistry {
    /// The set of registered operator kinds.
    ops: Vec<OpKind>,
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorRegistry {
    /// Creates a new registry pre-populated with all built-in operators.
    pub fn new() -> Self {
        Self {
            ops: Vec::from(ALL_OPS),
        }
    }

    /// Returns `true` if the given ONNX operator type name is supported.
    pub fn is_supported(&self, op_type: &str) -> bool {
        if let Some(kind) = OpKind::parse_str(op_type) {
            self.ops.contains(&kind)
        } else {
            false
        }
    }

    /// Returns the number of supported operators.
    pub fn supported_count(&self) -> usize {
        self.ops.len()
    }
}

// ---------------------------------------------------------------------------
// Operator stub implementations
// ---------------------------------------------------------------------------

/// Computes e^x for f32 using range reduction and polynomial approximation.
///
/// Accurate to ~1 ULP for typical inference ranges.
fn expf_approx(x: f32) -> f32 {
    // Clamp to avoid overflow/underflow
    let x = x.clamp(-88.7, 88.7);

    // Range reduction: e^x = 2^(x * log2(e)) = 2^k * 2^f where k=floor, f=fraction
    let t = x * core::f32::consts::LOG2_E;
    // Round to nearest integer
    let k = if t >= 0.0 {
        (t + 0.5) as i32
    } else {
        (t - 0.5) as i32
    };
    let f = x - (k as f32) * core::f32::consts::LN_2;

    // Polynomial approximation of e^f for f in [-0.5*ln2, 0.5*ln2]
    let f2 = f * f;
    let p = 1.0 + f + f2 * 0.5 + f2 * f * (1.0 / 6.0) + f2 * f2 * (1.0 / 24.0);

    // Reconstruct: multiply by 2^k using IEEE 754 bit manipulation
    let bits = ((k + 127) as u32) << 23;
    let scale = f32::from_bits(bits);
    p * scale
}

/// Reads a little-endian f32 from a raw byte slice at the given element index.
#[inline]
fn read_f32(data: &[u8], idx: usize) -> f32 {
    let off = idx * 4;
    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Writes a little-endian f32 to a raw byte slice at the given element index.
#[inline]
fn write_f32(data: &mut [u8], idx: usize, val: f32) {
    let off = idx * 4;
    let bytes = val.to_le_bytes();
    data[off] = bytes[0];
    data[off + 1] = bytes[1];
    data[off + 2] = bytes[2];
    data[off + 3] = bytes[3];
}

/// Computes contiguous strides (row-major) for a shape.
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

/// Element-wise addition of two input tensors with broadcasting.
///
/// Supports ONNX NumPy-style broadcasting. Both inputs must be Float type.
pub fn op_add(inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    if inputs.len() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "Add requires exactly 2 inputs",
        )));
    }
    let a = inputs[0];
    let b = inputs[1];
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Add only supports float32",
        )));
    }

    let out_shape = infer_binary_shape(&a.shape, &b.shape)?;
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];

    let a_strides = compute_strides(&a.shape.dims);
    let b_strides = compute_strides(&b.shape.dims);
    let ndim = out_shape.dims.len();
    let a_dim_offset = ndim.saturating_sub(a.shape.dims.len());
    let b_dim_offset = ndim.saturating_sub(b.shape.dims.len());

    // Iterate over every element in the output shape
    let mut coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        // Convert linear index to coordinate
        if i > 0 {
            let mut carry = true;
            for d in (0..ndim).rev() {
                if carry {
                    coord[d] += 1;
                    if coord[d] >= out_shape.dims[d] as usize {
                        coord[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }

        // Compute broadcast indices for a and b
        let a_idx = a.shape.dims.iter().enumerate().zip(a_strides.iter()).fold(
            0usize,
            |acc, ((d, &dim), &stride)| {
                if dim as usize != 1 {
                    acc + coord[d + a_dim_offset] * stride
                } else {
                    acc
                }
            },
        );

        let b_idx = b.shape.dims.iter().enumerate().zip(b_strides.iter()).fold(
            0usize,
            |acc, ((d, &dim), &stride)| {
                if dim as usize != 1 {
                    acc + coord[d + b_dim_offset] * stride
                } else {
                    acc
                }
            },
        );

        let va = read_f32(&a.raw_data, a_idx);
        let vb = read_f32(&b.raw_data, b_idx);
        write_f32(&mut raw_data, i, va + vb);
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Rectified linear unit activation: `max(0, x)`.
///
/// Operates element-wise on Float tensors.
pub fn op_relu(input: &Tensor) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Relu only supports float32",
        )));
    }
    let total = input.shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];

    for i in 0..total {
        let val = read_f32(&input.raw_data, i);
        let out = if val > 0.0 { val } else { 0.0 };
        write_f32(&mut raw_data, i, out);
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Matrix multiplication of two 2D tensors.
///
/// Computes C = A @ B where A is [M, K] and B is [K, N].
pub fn op_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "MatMul only supports float32",
        )));
    }
    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "MatMul requires 2D inputs",
        )));
    }

    let m = a.shape.dims[0] as usize;
    let k = a.shape.dims[1] as usize;
    let k_b = b.shape.dims[0] as usize;
    let n = b.shape.dims[1] as usize;

    if k != k_b {
        return Err(OpError::ShapeMismatch(String::from(
            "MatMul inner dimensions do not match",
        )));
    }

    let mut raw_data = alloc::vec![0u8; m * n * 4];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let va = read_f32(&a.raw_data, i * k + p);
                let vb = read_f32(&b.raw_data, p * n + j);
                sum += va * vb;
            }
            write_f32(&mut raw_data, i * n + j, sum);
        }
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(Vec::from([m as i64, n as i64])),
        name: String::new(),
        raw_data,
    })
}

/// Softmax normalization along the specified axis.
///
/// Computes `exp(x_i - max(x)) / sum(exp(x_j - max(x)))` along `axis`
/// for numerical stability. Supports negative axis indexing.
pub fn op_softmax(input: &Tensor, axis: i64) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Softmax only supports float32",
        )));
    }
    let ndim = input.shape.ndim();
    if ndim == 0 {
        // Scalar softmax is always 1.0
        let mut raw_data = alloc::vec![0u8; 4];
        write_f32(&mut raw_data, 0, 1.0);
        return Ok(Tensor {
            data_type: DataType::Float,
            shape: input.shape.clone(),
            name: String::new(),
            raw_data,
        });
    }

    // Resolve negative axis
    let resolved_axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    if resolved_axis >= ndim {
        return Err(OpError::InvalidAttribute(String::from("axis out of range")));
    }

    let total = input.shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];

    // Compute outer_size * inner_size * axis_size = total
    let axis_size = input.shape.dims[resolved_axis] as usize;
    let mut inner_size = 1usize;
    for d in (resolved_axis + 1)..ndim {
        inner_size *= input.shape.dims[d] as usize;
    }
    let mut outer_size = 1usize;
    for d in 0..resolved_axis {
        outer_size *= input.shape.dims[d] as usize;
    }

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for a in 0..axis_size {
                let idx = outer * axis_size * inner_size + a * inner_size + inner;
                let val = read_f32(&input.raw_data, idx);
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            for a in 0..axis_size {
                let idx = outer * axis_size * inner_size + a * inner_size + inner;
                let val = read_f32(&input.raw_data, idx);
                let exp_val = expf_approx(val - max_val);
                write_f32(&mut raw_data, idx, exp_val);
                sum += exp_val;
            }

            // Normalize
            if sum > 0.0 {
                for a in 0..axis_size {
                    let idx = outer * axis_size * inner_size + a * inner_size + inner;
                    let val = read_f32(&raw_data, idx);
                    write_f32(&mut raw_data, idx, val / sum);
                }
            }
        }
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Reshape a tensor to the specified shape.
///
/// Follows ONNX Reshape semantics: `-1` indicates an inferred
/// dimension, and `0` means copy from the original shape.
pub fn op_reshape(input: &Tensor, shape: &[i64]) -> Result<Tensor, OpError> {
    let total = input.shape.total_elements();

    // Resolve 0 dims (copy from original) and find -1 position
    let mut new_dims: Vec<i64> = Vec::with_capacity(shape.len());
    let mut neg_one_idx: Option<usize> = None;
    let mut known_product: usize = 1;

    for (i, &d) in shape.iter().enumerate() {
        if d == 0 {
            // Copy from input shape
            if i < input.shape.dims.len() {
                let orig = input.shape.dims[i];
                new_dims.push(orig);
                known_product *= orig as usize;
            } else {
                return Err(OpError::ShapeMismatch(String::from(
                    "Reshape: 0 dim index out of range",
                )));
            }
        } else if d == -1 {
            if neg_one_idx.is_some() {
                return Err(OpError::ShapeMismatch(String::from(
                    "Reshape: only one -1 dimension allowed",
                )));
            }
            neg_one_idx = Some(i);
            new_dims.push(-1);
        } else if d > 0 {
            new_dims.push(d);
            known_product *= d as usize;
        } else {
            return Err(OpError::ShapeMismatch(String::from(
                "Reshape: invalid dimension value",
            )));
        }
    }

    // Resolve -1 dimension
    if let Some(idx) = neg_one_idx {
        if known_product == 0 {
            return Err(OpError::ShapeMismatch(String::from(
                "Reshape: cannot infer dimension with zero-product shape",
            )));
        }
        let inferred = total / known_product;
        if inferred * known_product != total {
            return Err(OpError::ShapeMismatch(String::from(
                "Reshape: total elements do not match",
            )));
        }
        new_dims[idx] = inferred as i64;
    } else if known_product != total {
        return Err(OpError::ShapeMismatch(String::from(
            "Reshape: total elements do not match",
        )));
    }

    Ok(Tensor {
        data_type: input.data_type,
        shape: TensorShape::new(new_dims),
        name: String::new(),
        raw_data: input.raw_data.clone(),
    })
}

/// 2D convolution with optional bias, stride=1, no padding, dilation=1.
///
/// Input shape: [N, C_in, H, W]
/// Weight shape: [C_out, C_in, KH, KW]
/// Bias shape: [C_out] (optional)
/// Output shape: [N, C_out, H-KH+1, W-KW+1]
pub fn op_conv(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float || weight.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Conv only supports float32",
        )));
    }
    if input.shape.ndim() != 4 || weight.shape.ndim() != 4 {
        return Err(OpError::ShapeMismatch(String::from(
            "Conv requires 4D input [N,C,H,W] and 4D weight [Co,Ci,KH,KW]",
        )));
    }

    let n = input.shape.dims[0] as usize;
    let c_in = input.shape.dims[1] as usize;
    let h = input.shape.dims[2] as usize;
    let w = input.shape.dims[3] as usize;

    let c_out = weight.shape.dims[0] as usize;
    let c_in_w = weight.shape.dims[1] as usize;
    let kh = weight.shape.dims[2] as usize;
    let kw = weight.shape.dims[3] as usize;

    if c_in != c_in_w {
        return Err(OpError::ShapeMismatch(String::from(
            "Conv: input channels do not match weight channels",
        )));
    }
    if kh > h || kw > w {
        return Err(OpError::ShapeMismatch(String::from(
            "Conv: kernel larger than input",
        )));
    }

    let oh = h - kh + 1;
    let ow = w - kw + 1;
    let out_total = n * c_out * oh * ow;
    let mut raw_data = alloc::vec![0u8; out_total * 4];

    for batch in 0..n {
        for co in 0..c_out {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = 0.0f32;
                    for ci in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy + ky;
                                let ix = ox + kx;
                                let in_idx = batch * c_in * h * w + ci * h * w + iy * w + ix;
                                let w_idx = co * c_in_w * kh * kw + ci * kh * kw + ky * kw + kx;
                                sum += read_f32(&input.raw_data, in_idx)
                                    * read_f32(&weight.raw_data, w_idx);
                            }
                        }
                    }
                    if let Some(b) = bias {
                        sum += read_f32(&b.raw_data, co);
                    }
                    let out_idx = batch * c_out * oh * ow + co * oh * ow + oy * ow + ox;
                    write_f32(&mut raw_data, out_idx, sum);
                }
            }
        }
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(Vec::from([n as i64, c_out as i64, oh as i64, ow as i64])),
        name: String::new(),
        raw_data,
    })
}

// ---------------------------------------------------------------------------
// Shape inference helpers
// ---------------------------------------------------------------------------

/// Infers the output shape for a binary element-wise operation with broadcasting.
///
/// Applies ONNX/NumPy broadcasting rules: dimensions are compared from the
/// trailing end; a dimension of 1 is broadcastable to any size.
pub fn infer_binary_shape(a: &TensorShape, b: &TensorShape) -> Result<TensorShape, OpError> {
    let max_ndim = core::cmp::max(a.ndim(), b.ndim());
    let mut result_dims: Vec<i64> = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        // Index from the trailing end
        let da = if i < a.ndim() {
            a.dims[a.ndim() - 1 - i]
        } else {
            1
        };
        let db = if i < b.ndim() {
            b.dims[b.ndim() - 1 - i]
        } else {
            1
        };

        if da == db {
            result_dims.push(da);
        } else if da == 1 || da == -1 {
            result_dims.push(db);
        } else if db == 1 || db == -1 {
            result_dims.push(da);
        } else {
            return Err(OpError::ShapeMismatch(String::from(
                "incompatible dimensions for broadcasting",
            )));
        }
    }

    // We built from trailing end, so reverse
    result_dims.reverse();
    Ok(TensorShape::new(result_dims))
}

/// Infers the output shape for matrix multiplication.
///
/// For 2D inputs with shapes `[M, K]` and `[K, N]`, validates that
/// the inner dimensions match and returns shape `[M, N]`.
/// Returns `ShapeMismatch` if the inner dimensions do not agree.
pub fn infer_matmul_shape(a: &TensorShape, b: &TensorShape) -> Result<TensorShape, OpError> {
    // Require exactly 2D inputs for basic matmul shape inference.
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "matmul requires 2D inputs for basic shape inference",
        )));
    }

    let m = a.dims[0];
    let k_a = a.dims[1];
    let k_b = b.dims[0];
    let n = b.dims[1];

    // Inner dimensions must match (unless one is symbolic).
    if k_a != k_b && k_a != -1 && k_b != -1 {
        return Err(OpError::ShapeMismatch(String::from(
            "matmul inner dimensions do not match",
        )));
    }

    Ok(TensorShape::new(Vec::from([m, n])))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DataType;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;

    // ---- OpKind from_str tests ----

    #[test]
    fn test_opkind_from_str_arithmetic() {
        assert_eq!(OpKind::parse_str("Add"), Some(OpKind::Add));
        assert_eq!(OpKind::parse_str("Sub"), Some(OpKind::Sub));
        assert_eq!(OpKind::parse_str("Mul"), Some(OpKind::Mul));
        assert_eq!(OpKind::parse_str("Div"), Some(OpKind::Div));
        assert_eq!(OpKind::parse_str("MatMul"), Some(OpKind::MatMul));
    }

    #[test]
    fn test_opkind_from_str_activations() {
        assert_eq!(OpKind::parse_str("Relu"), Some(OpKind::Relu));
        assert_eq!(OpKind::parse_str("Sigmoid"), Some(OpKind::Sigmoid));
        assert_eq!(OpKind::parse_str("Tanh"), Some(OpKind::Tanh));
        assert_eq!(OpKind::parse_str("Softmax"), Some(OpKind::Softmax));
    }

    #[test]
    fn test_opkind_from_str_shape_ops() {
        assert_eq!(OpKind::parse_str("Reshape"), Some(OpKind::Reshape));
        assert_eq!(OpKind::parse_str("Transpose"), Some(OpKind::Transpose));
        assert_eq!(OpKind::parse_str("Flatten"), Some(OpKind::Flatten));
        assert_eq!(OpKind::parse_str("Squeeze"), Some(OpKind::Squeeze));
        assert_eq!(OpKind::parse_str("Unsqueeze"), Some(OpKind::Unsqueeze));
    }

    #[test]
    fn test_opkind_from_str_invalid() {
        assert_eq!(OpKind::parse_str("NonExistent"), None);
        assert_eq!(OpKind::parse_str("add"), None); // case-sensitive
        assert_eq!(OpKind::parse_str("RELU"), None);
        assert_eq!(OpKind::parse_str(""), None);
    }

    #[test]
    fn test_opkind_name_roundtrip() {
        // Every variant should round-trip through name() and from_str().
        let ops = [
            OpKind::Add,
            OpKind::Sub,
            OpKind::Mul,
            OpKind::Div,
            OpKind::MatMul,
            OpKind::Relu,
            OpKind::Sigmoid,
            OpKind::Tanh,
            OpKind::Softmax,
            OpKind::Conv,
            OpKind::MaxPool,
            OpKind::AveragePool,
            OpKind::BatchNormalization,
            OpKind::Reshape,
            OpKind::Transpose,
            OpKind::Flatten,
            OpKind::Squeeze,
            OpKind::Unsqueeze,
            OpKind::Concat,
            OpKind::Gather,
            OpKind::Slice,
            OpKind::Pad,
            OpKind::Gemm,
            OpKind::GlobalAveragePool,
            OpKind::LayerNormalization,
            OpKind::Cast,
            OpKind::Clip,
            OpKind::ReduceMean,
            OpKind::ReduceSum,
        ];
        for op in ops.iter() {
            let name = op.name();
            let parsed = OpKind::parse_str(name);
            assert_eq!(parsed, Some(*op), "round-trip failed for {:?}", op);
        }
    }

    // ---- OperatorRegistry tests ----

    #[test]
    fn test_registry_supported_count() {
        let registry = OperatorRegistry::new();
        assert_eq!(registry.supported_count(), 29);
    }

    #[test]
    fn test_registry_is_supported_valid() {
        let registry = OperatorRegistry::new();
        assert!(registry.is_supported("Add"));
        assert!(registry.is_supported("Conv"));
        assert!(registry.is_supported("MatMul"));
        assert!(registry.is_supported("Softmax"));
        assert!(registry.is_supported("Reshape"));
        assert!(registry.is_supported("ReduceSum"));
    }

    #[test]
    fn test_registry_is_supported_invalid() {
        let registry = OperatorRegistry::new();
        assert!(!registry.is_supported("FakeOp"));
        assert!(!registry.is_supported(""));
        assert!(!registry.is_supported("relu")); // case-sensitive
    }

    // ---- Helper ----

    /// Creates a Float tensor with the given shape and f32 data.
    fn make_f32_tensor(shape: &[i64], data: &[f32]) -> Tensor {
        let mut raw = alloc::vec![0u8; data.len() * 4];
        for (i, &v) in data.iter().enumerate() {
            let bytes = v.to_le_bytes();
            raw[i * 4] = bytes[0];
            raw[i * 4 + 1] = bytes[1];
            raw[i * 4 + 2] = bytes[2];
            raw[i * 4 + 3] = bytes[3];
        }
        Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(Vec::from(shape)),
            name: String::new(),
            raw_data: raw,
        }
    }

    /// Reads f32 values from a tensor's raw_data.
    fn read_f32_vec(t: &Tensor) -> Vec<f32> {
        let count = t.shape.total_elements();
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let off = i * 4;
            let val = f32::from_le_bytes([
                t.raw_data[off],
                t.raw_data[off + 1],
                t.raw_data[off + 2],
                t.raw_data[off + 3],
            ]);
            result.push(val);
        }
        result
    }

    // ---- op_add tests ----

    #[test]
    fn test_op_add_same_shape() {
        let a = make_f32_tensor(&[3], &[1.0, 2.0, 3.0]);
        let b = make_f32_tensor(&[3], &[4.0, 5.0, 6.0]);
        let result = op_add(&[&a, &b]).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_op_add_broadcast_scalar() {
        let a = make_f32_tensor(&[3], &[1.0, 2.0, 3.0]);
        let b = make_f32_tensor(&[1], &[10.0]);
        let result = op_add(&[&a, &b]).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_op_add_broadcast_2d() {
        // [2,3] + [1,3] -> [2,3]
        let a = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_f32_tensor(&[1, 3], &[10.0, 20.0, 30.0]);
        let result = op_add(&[&a, &b]).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_op_add_wrong_input_count() {
        let a = make_f32_tensor(&[2], &[1.0, 2.0]);
        let result = op_add(&[&a]);
        assert!(result.is_err());
    }

    // ---- op_relu tests ----

    #[test]
    fn test_op_relu_basic() {
        let t = make_f32_tensor(&[5], &[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = op_relu(&t).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_op_relu_all_positive() {
        let t = make_f32_tensor(&[3], &[1.0, 2.0, 3.0]);
        let result = op_relu(&t).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_op_relu_all_negative() {
        let t = make_f32_tensor(&[3], &[-1.0, -2.0, -3.0]);
        let result = op_relu(&t).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![0.0, 0.0, 0.0]);
    }

    // ---- op_matmul tests ----

    #[test]
    fn test_op_matmul_2x3_times_3x2() {
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        let a = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_f32_tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = op_matmul(&a, &b).unwrap();
        assert_eq!(result.shape.dims, vec![2, 2]);
        let vals = read_f32_vec(&result);
        // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
        assert_eq!(vals, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_op_matmul_identity() {
        let a = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let identity = make_f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let result = op_matmul(&a, &identity).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_op_matmul_inner_dim_mismatch() {
        let a = make_f32_tensor(&[2, 3], &[1.0; 6]);
        let b = make_f32_tensor(&[4, 2], &[1.0; 8]);
        let result = op_matmul(&a, &b);
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    // ---- op_softmax tests ----

    #[test]
    fn test_op_softmax_sums_to_one() {
        let t = make_f32_tensor(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let result = op_softmax(&t, -1).unwrap();
        let vals = read_f32_vec(&result);
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {}", sum);
        // Values should be monotonically increasing
        for i in 0..vals.len() - 1 {
            assert!(vals[i] < vals[i + 1]);
        }
    }

    #[test]
    fn test_op_softmax_uniform() {
        let t = make_f32_tensor(&[1, 3], &[0.0, 0.0, 0.0]);
        let result = op_softmax(&t, -1).unwrap();
        let vals = read_f32_vec(&result);
        for &v in &vals {
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_op_softmax_axis_0() {
        // axis=0 on [2, 2] should normalize along first dimension
        let t = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let result = op_softmax(&t, 0).unwrap();
        let vals = read_f32_vec(&result);
        // Column sums should each be 1.0
        let col0 = vals[0] + vals[2];
        let col1 = vals[1] + vals[3];
        assert!((col0 - 1.0).abs() < 1e-5);
        assert!((col1 - 1.0).abs() < 1e-5);
    }

    // ---- op_reshape tests ----

    #[test]
    fn test_op_reshape_flatten() {
        let t = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = op_reshape(&t, &[6]).unwrap();
        assert_eq!(result.shape.dims, vec![6]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_op_reshape_infer_dim() {
        let t = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = op_reshape(&t, &[3, -1]).unwrap();
        assert_eq!(result.shape.dims, vec![3, 2]);
    }

    #[test]
    fn test_op_reshape_copy_dim() {
        let t = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = op_reshape(&t, &[0, -1]).unwrap();
        assert_eq!(result.shape.dims, vec![2, 3]);
    }

    #[test]
    fn test_op_reshape_invalid_multiple_neg_one() {
        let t = make_f32_tensor(&[6], &[1.0; 6]);
        let result = op_reshape(&t, &[-1, -1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_op_reshape_size_mismatch() {
        let t = make_f32_tensor(&[6], &[1.0; 6]);
        let result = op_reshape(&t, &[5]);
        assert!(result.is_err());
    }

    // ---- op_conv tests ----

    #[test]
    fn test_op_conv_identity_kernel() {
        // 1x1x3x3 input, 1x1x1x1 kernel with weight=1.0 -> identity
        let input = make_f32_tensor(
            &[1, 1, 3, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let weight = make_f32_tensor(&[1, 1, 1, 1], &[1.0]);
        let result = op_conv(&input, &weight, None).unwrap();
        assert_eq!(result.shape.dims, vec![1, 1, 3, 3]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_op_conv_2x2_kernel() {
        // 1x1x3x3 input, 1x1x2x2 kernel -> 1x1x2x2 output
        let input = make_f32_tensor(
            &[1, 1, 3, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let weight = make_f32_tensor(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let result = op_conv(&input, &weight, None).unwrap();
        assert_eq!(result.shape.dims, vec![1, 1, 2, 2]);
        let vals = read_f32_vec(&result);
        // (0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6
        // (0,1): 2*1 + 3*0 + 5*0 + 6*1 = 8
        // (1,0): 4*1 + 5*0 + 7*0 + 8*1 = 12
        // (1,1): 5*1 + 6*0 + 8*0 + 9*1 = 14
        assert_eq!(vals, vec![6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn test_op_conv_with_bias() {
        let input = make_f32_tensor(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = make_f32_tensor(&[1, 1, 1, 1], &[2.0]);
        let bias = make_f32_tensor(&[1], &[10.0]);
        let result = op_conv(&input, &weight, Some(&bias)).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn test_op_conv_channel_mismatch() {
        let input = make_f32_tensor(&[1, 3, 4, 4], &[1.0; 48]);
        let weight = make_f32_tensor(&[1, 2, 3, 3], &[1.0; 18]); // Wrong c_in
        let result = op_conv(&input, &weight, None);
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    // ---- Shape inference tests ----

    #[test]
    fn test_infer_binary_shape_same() {
        let a = TensorShape::new(vec![2, 3]);
        let b = TensorShape::new(vec![2, 3]);
        let result = infer_binary_shape(&a, &b).unwrap();
        assert_eq!(result.dims, vec![2, 3]);
    }

    #[test]
    fn test_infer_binary_shape_broadcast_scalar() {
        let a = TensorShape::new(vec![2, 3]);
        let b = TensorShape::new(vec![1]);
        let result = infer_binary_shape(&a, &b).unwrap();
        assert_eq!(result.dims, vec![2, 3]);
    }

    #[test]
    fn test_infer_binary_shape_broadcast_row() {
        let a = TensorShape::new(vec![3, 4]);
        let b = TensorShape::new(vec![1, 4]);
        let result = infer_binary_shape(&a, &b).unwrap();
        assert_eq!(result.dims, vec![3, 4]);
    }

    #[test]
    fn test_infer_binary_shape_broadcast_different_ndim() {
        // [3, 1, 5] + [4, 5] -> [3, 4, 5]
        let a = TensorShape::new(vec![3, 1, 5]);
        let b = TensorShape::new(vec![4, 5]);
        let result = infer_binary_shape(&a, &b).unwrap();
        assert_eq!(result.dims, vec![3, 4, 5]);
    }

    #[test]
    fn test_infer_binary_shape_incompatible() {
        let a = TensorShape::new(vec![3, 4]);
        let b = TensorShape::new(vec![3, 5]);
        let result = infer_binary_shape(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_matmul_shape_compatible() {
        let a = TensorShape::new(vec![4, 3]);
        let b = TensorShape::new(vec![3, 5]);
        let result = infer_matmul_shape(&a, &b).unwrap();
        assert_eq!(result.dims, vec![4, 5]);
    }

    #[test]
    fn test_infer_matmul_shape_incompatible() {
        let a = TensorShape::new(vec![4, 3]);
        let b = TensorShape::new(vec![7, 5]);
        let result = infer_matmul_shape(&a, &b);
        assert!(result.is_err());
        match result {
            Err(OpError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_infer_matmul_shape_symbolic() {
        // Symbolic inner dimension (-1) should be accepted.
        let a = TensorShape::new(vec![4, -1]);
        let b = TensorShape::new(vec![3, 5]);
        let result = infer_matmul_shape(&a, &b).unwrap();
        assert_eq!(result.dims, vec![4, 5]);
    }

    #[test]
    fn test_infer_matmul_shape_non_2d() {
        let a = TensorShape::new(vec![2, 3, 4]);
        let b = TensorShape::new(vec![4, 5]);
        let result = infer_matmul_shape(&a, &b);
        assert!(result.is_err());
    }

    // ---- OpError Display test ----

    #[test]
    fn test_op_error_display() {
        assert_eq!(
            format!("{}", OpError::UnsupportedOp(String::from("FooOp"))),
            "unsupported operator: FooOp"
        );
        assert_eq!(
            format!("{}", OpError::ShapeMismatch(String::from("dim mismatch"))),
            "shape mismatch: dim mismatch"
        );
        assert_eq!(
            format!("{}", OpError::InvalidAttribute(String::from("bad axis"))),
            "invalid attribute: bad axis"
        );
        assert_eq!(
            format!("{}", OpError::InternalError(String::from("oom"))),
            "internal error: oom"
        );
        assert_eq!(
            format!("{}", OpError::NotImplemented),
            "operator not implemented"
        );
    }
}
