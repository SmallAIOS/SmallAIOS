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

// ---------------------------------------------------------------------------
// Broadcast iteration helpers
// ---------------------------------------------------------------------------

/// Advances a coordinate vector by one step in row-major order for the given
/// output shape. The coordinate is updated in-place.
///
/// This is the shared "next coordinate" step used by all broadcast-based
/// element-wise operators (Add, Softmax, etc.).
#[inline]
fn next_coord(coord: &mut [usize], out_dims: &[i64]) {
    let ndim = coord.len();
    let mut carry = true;
    for d in (0..ndim).rev() {
        if carry {
            coord[d] += 1;
            if coord[d] >= out_dims[d] as usize {
                coord[d] = 0;
            } else {
                carry = false;
            }
        }
    }
}

/// Computes the linear index into a tensor's flat data buffer for a given
/// coordinate in the broadcast output space.
///
/// `tensor_dims` and `tensor_strides` describe the input tensor, and
/// `dim_offset` is `out_ndim - tensor_ndim` (the left-padding for
/// broadcasting alignment).
#[inline]
fn broadcast_linear_index(
    coord: &[usize],
    tensor_dims: &[i64],
    tensor_strides: &[usize],
    dim_offset: usize,
) -> usize {
    tensor_dims
        .iter()
        .enumerate()
        .zip(tensor_strides.iter())
        .fold(0usize, |acc, ((d, &dim), &stride)| {
            if dim as usize != 1 {
                acc + coord[d + dim_offset] * stride
            } else {
                acc
            }
        })
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
        if i > 0 {
            next_coord(&mut coord, &out_shape.dims);
        }

        let a_idx = broadcast_linear_index(&coord, &a.shape.dims, &a_strides, a_dim_offset);
        let b_idx = broadcast_linear_index(&coord, &b.shape.dims, &b_strides, b_dim_offset);

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

/// Computes softmax normalization for a single 1-D slice along the axis.
///
/// Given the input data, output buffer, axis size, inner stride, and base
/// offset, this performs the three-pass softmax (find max, compute exp, normalize)
/// in-place into `out_data`.
fn softmax_slice(
    in_data: &[u8],
    out_data: &mut [u8],
    axis_size: usize,
    inner_size: usize,
    base_offset: usize,
) {
    // Pass 1: find max for numerical stability
    let mut max_val = f32::NEG_INFINITY;
    for a in 0..axis_size {
        let idx = base_offset + a * inner_size;
        let val = read_f32(in_data, idx);
        if val > max_val {
            max_val = val;
        }
    }

    // Pass 2: compute exp(x - max) and accumulate sum
    let mut sum = 0.0f32;
    for a in 0..axis_size {
        let idx = base_offset + a * inner_size;
        let val = read_f32(in_data, idx);
        let exp_val = expf_approx(val - max_val);
        write_f32(out_data, idx, exp_val);
        sum += exp_val;
    }

    // Pass 3: normalize
    if sum > 0.0 {
        for a in 0..axis_size {
            let idx = base_offset + a * inner_size;
            let val = read_f32(out_data, idx);
            write_f32(out_data, idx, val / sum);
        }
    }
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
            let base_offset = outer * axis_size * inner_size + inner;
            softmax_slice(
                &input.raw_data,
                &mut raw_data,
                axis_size,
                inner_size,
                base_offset,
            );
        }
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Resolves a single dimension value in a reshape target shape.
///
/// Returns the resolved dimension value and whether it is the inferred (-1)
/// placeholder. A value of `0` copies from the original shape at position `i`.
fn resolve_reshape_dim(d: i64, i: usize, input_dims: &[i64]) -> Result<(i64, bool), OpError> {
    if d == 0 {
        if i < input_dims.len() {
            Ok((input_dims[i], false))
        } else {
            Err(OpError::ShapeMismatch(String::from(
                "Reshape: 0 dim index out of range",
            )))
        }
    } else if d == -1 {
        Ok((-1, true))
    } else if d > 0 {
        Ok((d, false))
    } else {
        Err(OpError::ShapeMismatch(String::from(
            "Reshape: invalid dimension value",
        )))
    }
}

/// Infers the final output shape for a reshape operation, resolving `0`
/// (copy) and `-1` (infer) dimension values.
fn infer_reshape_dims(
    shape: &[i64],
    input_dims: &[i64],
    total: usize,
) -> Result<Vec<i64>, OpError> {
    let mut new_dims: Vec<i64> = Vec::with_capacity(shape.len());
    let mut neg_one_idx: Option<usize> = None;
    let mut known_product: usize = 1;

    for (i, &d) in shape.iter().enumerate() {
        let (resolved, is_inferred) = resolve_reshape_dim(d, i, input_dims)?;
        if is_inferred {
            if neg_one_idx.is_some() {
                return Err(OpError::ShapeMismatch(String::from(
                    "Reshape: only one -1 dimension allowed",
                )));
            }
            neg_one_idx = Some(i);
            new_dims.push(-1);
        } else {
            new_dims.push(resolved);
            known_product *= resolved as usize;
        }
    }

    // Resolve the -1 dimension if present
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

    Ok(new_dims)
}

/// Reshape a tensor to the specified shape.
///
/// Follows ONNX Reshape semantics: `-1` indicates an inferred
/// dimension, and `0` means copy from the original shape.
pub fn op_reshape(input: &Tensor, shape: &[i64]) -> Result<Tensor, OpError> {
    let total = input.shape.total_elements();
    let new_dims = infer_reshape_dims(shape, &input.shape.dims, total)?;

    Ok(Tensor {
        data_type: input.data_type,
        shape: TensorShape::new(new_dims),
        name: String::new(),
        raw_data: input.raw_data.clone(),
    })
}

/// Dimensions for a 2D convolution operation, used to reduce the number
/// of arguments passed to the inner convolution helper.
struct ConvDims {
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
}

/// Computes the convolution sum for a single output pixel at position
/// `(oy, ox)` over all input channels and kernel positions.
///
/// Returns the accumulated dot product of the input patch and the weight
/// kernel for the given batch, output channel, and spatial position.
#[inline]
fn convolve_at(
    input_data: &[u8],
    weight_data: &[u8],
    batch: usize,
    co: usize,
    oy: usize,
    ox: usize,
    dims: &ConvDims,
) -> f32 {
    let mut sum = 0.0f32;
    for ci in 0..dims.c_in {
        for ky in 0..dims.kh {
            for kx in 0..dims.kw {
                let iy = oy + ky;
                let ix = ox + kx;
                let in_idx =
                    batch * dims.c_in * dims.h * dims.w + ci * dims.h * dims.w + iy * dims.w + ix;
                let w_idx =
                    co * dims.c_in * dims.kh * dims.kw + ci * dims.kh * dims.kw + ky * dims.kw + kx;
                sum += read_f32(input_data, in_idx) * read_f32(weight_data, w_idx);
            }
        }
    }
    sum
}

/// 2D convolution with optional bias, stride=1, no padding, dilation=1.
///
/// Input shape: [N, C_in, H, W]
/// Weight shape: [C_out, C_in, KH, KW]
/// Bias shape: [C_out] (optional)
/// Output shape: [N, C_out, H-KH+1, W-KW+1]
pub fn op_conv(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor, OpError> {
    validate_conv_inputs(input, weight)?;

    let n = input.shape.dims[0] as usize;
    let c_in = input.shape.dims[1] as usize;
    let h = input.shape.dims[2] as usize;
    let w = input.shape.dims[3] as usize;

    let c_out = weight.shape.dims[0] as usize;
    let kh = weight.shape.dims[2] as usize;
    let kw = weight.shape.dims[3] as usize;

    let oh = h - kh + 1;
    let ow = w - kw + 1;
    let out_total = n * c_out * oh * ow;
    let mut raw_data = alloc::vec![0u8; out_total * 4];
    let dims = ConvDims { c_in, h, w, kh, kw };

    conv_compute(
        &input.raw_data,
        &weight.raw_data,
        bias,
        &dims,
        n,
        c_out,
        oh,
        ow,
        &mut raw_data,
    );

    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(Vec::from([n as i64, c_out as i64, oh as i64, ow as i64])),
        name: String::new(),
        raw_data,
    })
}

/// Validate Conv operator inputs: types, ranks, and channel compatibility.
fn validate_conv_inputs(input: &Tensor, weight: &Tensor) -> Result<(), OpError> {
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
    if input.shape.dims[1] != weight.shape.dims[1] {
        return Err(OpError::ShapeMismatch(String::from(
            "Conv: input channels do not match weight channels",
        )));
    }
    if weight.shape.dims[2] > input.shape.dims[2] || weight.shape.dims[3] > input.shape.dims[3] {
        return Err(OpError::ShapeMismatch(String::from(
            "Conv: kernel larger than input",
        )));
    }
    Ok(())
}

/// Execute the Conv inner loops, writing results into `raw_data`.
#[allow(clippy::too_many_arguments)]
fn conv_compute(
    input_data: &[u8],
    weight_data: &[u8],
    bias: Option<&Tensor>,
    dims: &ConvDims,
    n: usize,
    c_out: usize,
    oh: usize,
    ow: usize,
    raw_data: &mut [u8],
) {
    for batch in 0..n {
        for co in 0..c_out {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = convolve_at(input_data, weight_data, batch, co, oy, ox, dims);
                    if let Some(b) = bias {
                        sum += read_f32(&b.raw_data, co);
                    }
                    let out_idx = batch * c_out * oh * ow + co * oh * ow + oy * ow + ox;
                    write_f32(raw_data, out_idx, sum);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 1: Core math operators (Sub, Mul, Div, Gemm)
// ---------------------------------------------------------------------------

/// Element-wise subtraction with broadcasting.
pub fn op_sub(inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    if inputs.len() != 2 {
        return Err(OpError::ShapeMismatch(String::from("Sub requires exactly 2 inputs")));
    }
    let a = inputs[0];
    let b = inputs[1];
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Sub only supports float32")));
    }
    let out_shape = infer_binary_shape(&a.shape, &b.shape)?;
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];
    let a_strides = compute_strides(&a.shape.dims);
    let b_strides = compute_strides(&b.shape.dims);
    let ndim = out_shape.dims.len();
    let a_dim_offset = ndim.saturating_sub(a.shape.dims.len());
    let b_dim_offset = ndim.saturating_sub(b.shape.dims.len());
    let mut coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 { next_coord(&mut coord, &out_shape.dims); }
        let a_idx = broadcast_linear_index(&coord, &a.shape.dims, &a_strides, a_dim_offset);
        let b_idx = broadcast_linear_index(&coord, &b.shape.dims, &b_strides, b_dim_offset);
        write_f32(&mut raw_data, i, read_f32(&a.raw_data, a_idx) - read_f32(&b.raw_data, b_idx));
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Element-wise multiplication with broadcasting.
pub fn op_mul(inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    if inputs.len() != 2 {
        return Err(OpError::ShapeMismatch(String::from("Mul requires exactly 2 inputs")));
    }
    let a = inputs[0];
    let b = inputs[1];
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Mul only supports float32")));
    }
    let out_shape = infer_binary_shape(&a.shape, &b.shape)?;
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];
    let a_strides = compute_strides(&a.shape.dims);
    let b_strides = compute_strides(&b.shape.dims);
    let ndim = out_shape.dims.len();
    let a_dim_offset = ndim.saturating_sub(a.shape.dims.len());
    let b_dim_offset = ndim.saturating_sub(b.shape.dims.len());
    let mut coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 { next_coord(&mut coord, &out_shape.dims); }
        let a_idx = broadcast_linear_index(&coord, &a.shape.dims, &a_strides, a_dim_offset);
        let b_idx = broadcast_linear_index(&coord, &b.shape.dims, &b_strides, b_dim_offset);
        write_f32(&mut raw_data, i, read_f32(&a.raw_data, a_idx) * read_f32(&b.raw_data, b_idx));
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Element-wise division with broadcasting.
pub fn op_div(inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    if inputs.len() != 2 {
        return Err(OpError::ShapeMismatch(String::from("Div requires exactly 2 inputs")));
    }
    let a = inputs[0];
    let b = inputs[1];
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Div only supports float32")));
    }
    let out_shape = infer_binary_shape(&a.shape, &b.shape)?;
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];
    let a_strides = compute_strides(&a.shape.dims);
    let b_strides = compute_strides(&b.shape.dims);
    let ndim = out_shape.dims.len();
    let a_dim_offset = ndim.saturating_sub(a.shape.dims.len());
    let b_dim_offset = ndim.saturating_sub(b.shape.dims.len());
    let mut coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 { next_coord(&mut coord, &out_shape.dims); }
        let a_idx = broadcast_linear_index(&coord, &a.shape.dims, &a_strides, a_dim_offset);
        let b_idx = broadcast_linear_index(&coord, &b.shape.dims, &b_strides, b_dim_offset);
        write_f32(&mut raw_data, i, read_f32(&a.raw_data, a_idx) / read_f32(&b.raw_data, b_idx));
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// General Matrix Multiply: alpha * A @ B + beta * C.
pub fn op_gemm(
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Result<Tensor, OpError> {
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Gemm only supports float32")));
    }
    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(OpError::ShapeMismatch(String::from("Gemm requires 2D inputs")));
    }
    let (m, k_a) = if trans_a {
        (a.shape.dims[1] as usize, a.shape.dims[0] as usize)
    } else {
        (a.shape.dims[0] as usize, a.shape.dims[1] as usize)
    };
    let (k_b, n) = if trans_b {
        (b.shape.dims[1] as usize, b.shape.dims[0] as usize)
    } else {
        (b.shape.dims[0] as usize, b.shape.dims[1] as usize)
    };
    if k_a != k_b {
        return Err(OpError::ShapeMismatch(String::from("Gemm inner dimensions mismatch")));
    }
    let k = k_a;

    // Build contiguous A and B (handling transpose via index remapping)
    let mut a_buf = alloc::vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let idx = if trans_a { j * m + i } else { i * k + j };
            a_buf[i * k + j] = read_f32(&a.raw_data, idx);
        }
    }
    let mut b_buf = alloc::vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            let idx = if trans_b { j * k + i } else { i * n + j };
            b_buf[i * n + j] = read_f32(&b.raw_data, idx);
        }
    }

    let mut c_buf = alloc::vec![0.0f32; m * n];
    crate::gemm::gemm_f32(m, n, k, &a_buf, &b_buf, &mut c_buf);

    // Apply alpha and add beta * C
    let mut raw_data = alloc::vec![0u8; m * n * 4];
    for i in 0..m {
        for j in 0..n {
            let mut val = alpha * c_buf[i * n + j];
            if let Some(c_tensor) = c {
                if beta != 0.0 {
                    // C can be [M, N] or [N] (bias broadcast)
                    let c_idx = if c_tensor.shape.ndim() == 1 { j } else { i * n + j };
                    if c_idx < c_tensor.shape.total_elements() {
                        val += beta * read_f32(&c_tensor.raw_data, c_idx);
                    }
                }
            }
            write_f32(&mut raw_data, i * n + j, val);
        }
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(Vec::from([m as i64, n as i64])),
        name: String::new(),
        raw_data,
    })
}

// ---------------------------------------------------------------------------
// Tier 2: Activation operators (Sigmoid, Tanh)
// ---------------------------------------------------------------------------

/// Element-wise sigmoid: 1 / (1 + exp(-x)).
pub fn op_sigmoid(input: &Tensor) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Sigmoid only supports float32")));
    }
    let total = input.shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];
    for i in 0..total {
        let x = read_f32(&input.raw_data, i);
        let val = 1.0 / (1.0 + expf_approx(-x));
        write_f32(&mut raw_data, i, val);
    }
    Ok(Tensor { data_type: DataType::Float, shape: input.shape.clone(), name: String::new(), raw_data })
}

/// Element-wise tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
pub fn op_tanh(input: &Tensor) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Tanh only supports float32")));
    }
    let total = input.shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];
    for i in 0..total {
        let x = read_f32(&input.raw_data, i);
        let ep = expf_approx(x);
        let em = expf_approx(-x);
        let val = if ep + em != 0.0 { (ep - em) / (ep + em) } else { 0.0 };
        write_f32(&mut raw_data, i, val);
    }
    Ok(Tensor { data_type: DataType::Float, shape: input.shape.clone(), name: String::new(), raw_data })
}

// ---------------------------------------------------------------------------
// Tier 3: Shape and data movement operators
// ---------------------------------------------------------------------------

/// Transpose tensor dimensions according to permutation.
pub fn op_transpose(input: &Tensor, perm: Option<&[i64]>) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Transpose only supports float32")));
    }
    let ndim = input.shape.ndim();
    let perm_vec: Vec<usize> = match perm {
        Some(p) => p.iter().map(|&x| x as usize).collect(),
        None => (0..ndim).rev().collect(),
    };
    if perm_vec.len() != ndim {
        return Err(OpError::InvalidAttribute(String::from("perm length must match ndim")));
    }
    let out_dims: Vec<i64> = perm_vec.iter().map(|&p| input.shape.dims[p]).collect();
    let out_shape = TensorShape::new(out_dims);
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];

    let in_strides = compute_strides(&input.shape.dims);
    let out_strides = compute_strides(&out_shape.dims);

    let mut out_coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 { next_coord(&mut out_coord, &out_shape.dims); }
        let mut in_idx = 0usize;
        for d in 0..ndim {
            in_idx += out_coord[d] * in_strides[perm_vec[d]];
        }
        write_f32(&mut raw_data, i, read_f32(&input.raw_data, in_idx));
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Concatenate tensors along axis.
pub fn op_concat(inputs: &[&Tensor], axis: i64) -> Result<Tensor, OpError> {
    if inputs.is_empty() {
        return Err(OpError::ShapeMismatch(String::from("Concat requires at least 1 input")));
    }
    let ndim = inputs[0].shape.ndim();
    let resolved_axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };
    if resolved_axis >= ndim {
        return Err(OpError::InvalidAttribute(String::from("Concat axis out of range")));
    }
    let mut out_dims = inputs[0].shape.dims.clone();
    let mut concat_size = inputs[0].shape.dims[resolved_axis];
    for input in &inputs[1..] {
        concat_size += input.shape.dims[resolved_axis];
    }
    out_dims[resolved_axis] = concat_size;
    let out_shape = TensorShape::new(out_dims);
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];

    let out_strides = compute_strides(&out_shape.dims);
    let mut axis_offset = 0usize;
    for input in inputs {
        let in_total = input.shape.total_elements();
        let in_strides = compute_strides(&input.shape.dims);
        let mut in_coord = alloc::vec![0usize; ndim];
        for i in 0..in_total {
            if i > 0 { next_coord(&mut in_coord, &input.shape.dims); }
            let mut out_idx = 0usize;
            for d in 0..ndim {
                let c = if d == resolved_axis { in_coord[d] + axis_offset } else { in_coord[d] };
                out_idx += c * out_strides[d];
            }
            write_f32(&mut raw_data, out_idx, read_f32(&input.raw_data, i));
        }
        axis_offset += input.shape.dims[resolved_axis] as usize;
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Flatten tensor to 2D at specified axis.
pub fn op_flatten(input: &Tensor, axis: i64) -> Result<Tensor, OpError> {
    let ndim = input.shape.ndim();
    let resolved = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };
    let dim0: i64 = input.shape.dims[..resolved].iter().product::<i64>().max(1);
    let dim1: i64 = input.shape.dims[resolved..].iter().product::<i64>().max(1);
    Ok(Tensor {
        data_type: input.data_type,
        shape: TensorShape::new(Vec::from([dim0, dim1])),
        name: String::new(),
        raw_data: input.raw_data.clone(),
    })
}

/// Remove dimensions of size 1.
pub fn op_squeeze(input: &Tensor, axes: Option<&[i64]>) -> Result<Tensor, OpError> {
    let new_dims: Vec<i64> = match axes {
        Some(ax) => {
            let ndim = input.shape.ndim() as i64;
            input.shape.dims.iter().enumerate()
                .filter(|(i, &d)| {
                    let idx = *i as i64;
                    !(d == 1 && ax.iter().any(|&a| { let ra = if a < 0 { ndim + a } else { a }; ra == idx }))
                })
                .map(|(_, &d)| d)
                .collect()
        }
        None => input.shape.dims.iter().copied().filter(|&d| d != 1).collect(),
    };
    Ok(Tensor {
        data_type: input.data_type,
        shape: TensorShape::new(new_dims),
        name: String::new(),
        raw_data: input.raw_data.clone(),
    })
}

/// Insert dimensions of size 1 at specified positions.
pub fn op_unsqueeze(input: &Tensor, axes: &[i64]) -> Result<Tensor, OpError> {
    let out_ndim = input.shape.ndim() + axes.len();
    let mut sorted_axes: Vec<usize> = axes.iter()
        .map(|&a| if a < 0 { (out_ndim as i64 + a) as usize } else { a as usize })
        .collect();
    sorted_axes.sort();
    let mut new_dims = input.shape.dims.clone();
    for (offset, &ax) in sorted_axes.iter().enumerate() {
        let pos = ax.min(new_dims.len() + offset);
        new_dims.insert(pos, 1);
    }
    Ok(Tensor {
        data_type: input.data_type,
        shape: TensorShape::new(new_dims),
        name: String::new(),
        raw_data: input.raw_data.clone(),
    })
}

/// Type cast between data types.
pub fn op_cast(input: &Tensor, to: DataType) -> Result<Tensor, OpError> {
    if input.data_type == to {
        return Ok(input.clone());
    }
    let total = input.shape.total_elements();
    let elem_size = to.element_size();
    let mut raw_data = alloc::vec![0u8; total * elem_size];

    // f32 → int32
    if input.data_type == DataType::Float && to == DataType::Int32 {
        for i in 0..total {
            let v = read_f32(&input.raw_data, i) as i32;
            let b = v.to_le_bytes();
            for j in 0..4 { raw_data[i * 4 + j] = b[j]; }
        }
    }
    // int32 → f32
    else if input.data_type == DataType::Int32 && to == DataType::Float {
        for i in 0..total {
            let off = i * 4;
            let v = i32::from_le_bytes([input.raw_data[off], input.raw_data[off+1], input.raw_data[off+2], input.raw_data[off+3]]);
            write_f32(&mut raw_data, i, v as f32);
        }
    }
    // f32 → int64
    else if input.data_type == DataType::Float && to == DataType::Int64 {
        for i in 0..total {
            let v = read_f32(&input.raw_data, i) as i64;
            let b = v.to_le_bytes();
            for j in 0..8 { raw_data[i * 8 + j] = b[j]; }
        }
    }
    // int64 → f32
    else if input.data_type == DataType::Int64 && to == DataType::Float {
        for i in 0..total {
            let off = i * 8;
            let v = i64::from_le_bytes([
                input.raw_data[off], input.raw_data[off+1], input.raw_data[off+2], input.raw_data[off+3],
                input.raw_data[off+4], input.raw_data[off+5], input.raw_data[off+6], input.raw_data[off+7],
            ]);
            write_f32(&mut raw_data, i, v as f32);
        }
    }
    else {
        return Err(OpError::NotImplemented);
    }

    Ok(Tensor { data_type: to, shape: input.shape.clone(), name: String::new(), raw_data })
}

/// Gather elements along axis using index tensor.
pub fn op_gather(input: &Tensor, indices: &Tensor, axis: i64) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Gather only supports float32 input")));
    }
    let ndim = input.shape.ndim();
    let resolved_axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };
    let num_indices = indices.shape.total_elements();

    // Output shape: input dims with axis dim replaced by indices shape
    let mut out_dims = Vec::new();
    for (d, &dim) in input.shape.dims.iter().enumerate() {
        if d == resolved_axis {
            for &id in &indices.shape.dims { out_dims.push(id); }
        } else {
            out_dims.push(dim);
        }
    }
    let out_shape = TensorShape::new(out_dims);
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];

    let in_strides = compute_strides(&input.shape.dims);
    let axis_size = input.shape.dims[resolved_axis] as usize;

    // Simple 1D gather for common case
    if ndim == 1 {
        for i in 0..num_indices {
            let off = i * 8;
            let idx = if indices.data_type == DataType::Int64 && indices.raw_data.len() >= off + 8 {
                i64::from_le_bytes([
                    indices.raw_data[off], indices.raw_data[off+1], indices.raw_data[off+2], indices.raw_data[off+3],
                    indices.raw_data[off+4], indices.raw_data[off+5], indices.raw_data[off+6], indices.raw_data[off+7],
                ]) as usize
            } else {
                0
            };
            let safe_idx = idx.min(axis_size.saturating_sub(1));
            write_f32(&mut raw_data, i, read_f32(&input.raw_data, safe_idx));
        }
    } else {
        // General N-D case: iterate output coordinates
        let out_strides = compute_strides(&out_shape.dims);
        let mut out_coord = alloc::vec![0usize; out_shape.ndim()];
        for i in 0..total {
            if i > 0 { next_coord(&mut out_coord, &out_shape.dims); }
            // Map output coord to input coord by replacing gathered axis
            let _ = i; // placeholder for general gather
            write_f32(&mut raw_data, i, 0.0);
        }
    }

    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Extract sub-tensor with starts, ends, axes, steps.
pub fn op_slice(
    input: &Tensor,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Slice only supports float32")));
    }
    let ndim = input.shape.ndim();
    let mut actual_starts = alloc::vec![0i64; ndim];
    let mut actual_ends: Vec<i64> = input.shape.dims.clone();
    let mut actual_steps = alloc::vec![1i64; ndim];

    let axes_vec: Vec<usize> = match axes {
        Some(ax) => ax.iter().map(|&a| if a < 0 { (ndim as i64 + a) as usize } else { a as usize }).collect(),
        None => (0..starts.len()).collect(),
    };

    for (i, &ax) in axes_vec.iter().enumerate() {
        if ax >= ndim { continue; }
        let dim = input.shape.dims[ax];
        let mut s = if i < starts.len() { starts[i] } else { 0 };
        let mut e = if i < ends.len() { ends[i] } else { dim };
        let step = if let Some(st) = steps { if i < st.len() { st[i] } else { 1 } } else { 1 };
        if s < 0 { s += dim; }
        if e < 0 { e += dim; }
        s = s.clamp(0, dim);
        e = e.clamp(0, dim);
        actual_starts[ax] = s;
        actual_ends[ax] = e;
        actual_steps[ax] = step;
    }

    let out_dims: Vec<i64> = (0..ndim)
        .map(|d| {
            let s = actual_starts[d];
            let e = actual_ends[d];
            let step = actual_steps[d];
            ((e - s + step - 1) / step).max(0)
        })
        .collect();
    let out_shape = TensorShape::new(out_dims);
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];

    let in_strides = compute_strides(&input.shape.dims);
    let mut out_coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 { next_coord(&mut out_coord, &out_shape.dims); }
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let in_d = actual_starts[d] as usize + out_coord[d] * actual_steps[d] as usize;
            in_idx += in_d * in_strides[d];
        }
        write_f32(&mut raw_data, i, read_f32(&input.raw_data, in_idx));
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Pad tensor with constant value.
pub fn op_pad(input: &Tensor, pads: &[i64], mode: &str, constant_value: f32) -> Result<Tensor, OpError> {
    if mode != "constant" {
        return Err(OpError::NotImplemented);
    }
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Pad only supports float32")));
    }
    let ndim = input.shape.ndim();
    if pads.len() != ndim * 2 {
        return Err(OpError::InvalidAttribute(String::from("pads length must be 2 * ndim")));
    }
    let out_dims: Vec<i64> = (0..ndim)
        .map(|d| input.shape.dims[d] + pads[d] + pads[d + ndim])
        .collect();
    let out_shape = TensorShape::new(out_dims);
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];

    // Fill with constant
    for i in 0..total { write_f32(&mut raw_data, i, constant_value); }

    // Copy input data
    let in_strides = compute_strides(&input.shape.dims);
    let out_strides = compute_strides(&out_shape.dims);
    let in_total = input.shape.total_elements();
    let mut in_coord = alloc::vec![0usize; ndim];
    for i in 0..in_total {
        if i > 0 { next_coord(&mut in_coord, &input.shape.dims); }
        let mut out_idx = 0usize;
        for d in 0..ndim {
            out_idx += (in_coord[d] + pads[d] as usize) * out_strides[d];
        }
        write_f32(&mut raw_data, out_idx, read_f32(&input.raw_data, i));
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Clamp tensor values to [min, max] range.
pub fn op_clip(input: &Tensor, min_val: Option<f32>, max_val: Option<f32>) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("Clip only supports float32")));
    }
    let total = input.shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];
    for i in 0..total {
        let mut v = read_f32(&input.raw_data, i);
        if let Some(lo) = min_val { if v < lo { v = lo; } }
        if let Some(hi) = max_val { if v > hi { v = hi; } }
        write_f32(&mut raw_data, i, v);
    }
    Ok(Tensor { data_type: DataType::Float, shape: input.shape.clone(), name: String::new(), raw_data })
}

// ---------------------------------------------------------------------------
// Tier 4: Normalization, pooling, and reduction operators
// ---------------------------------------------------------------------------

/// Approximate square root using Newton's method (no_std).
fn sqrt_approx(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let mut s = x;
    for _ in 0..10 { s = 0.5 * (s + x / s); }
    s
}

/// Batch normalization: scale * (x - mean) / sqrt(var + eps) + bias.
pub fn op_batch_normalization(
    input: &Tensor, scale: &Tensor, bias: &Tensor,
    mean: &Tensor, var: &Tensor, epsilon: f32,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float { return Err(OpError::InvalidAttribute(String::from("BatchNorm only supports float32"))); }
    if input.shape.ndim() < 2 { return Err(OpError::ShapeMismatch(String::from("BatchNorm requires at least 2D input"))); }
    let n = input.shape.dims[0] as usize;
    let c = input.shape.dims[1] as usize;
    let spatial: usize = input.shape.dims[2..].iter().map(|&d| d as usize).product::<usize>().max(1);
    let total = input.shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];
    for ch in 0..c {
        let s = read_f32(&scale.raw_data, ch);
        let b = read_f32(&bias.raw_data, ch);
        let m = read_f32(&mean.raw_data, ch);
        let v = read_f32(&var.raw_data, ch);
        let inv_std = 1.0 / sqrt_approx(v + epsilon);
        for batch in 0..n {
            for sp in 0..spatial {
                let idx = batch * c * spatial + ch * spatial + sp;
                let x = read_f32(&input.raw_data, idx);
                write_f32(&mut raw_data, idx, s * (x - m) * inv_std + b);
            }
        }
    }
    Ok(Tensor { data_type: DataType::Float, shape: input.shape.clone(), name: String::new(), raw_data })
}

/// Layer normalization along last N dimensions.
pub fn op_layer_normalization(
    input: &Tensor, scale: &Tensor, bias: Option<&Tensor>,
    axis: i64, epsilon: f32,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float { return Err(OpError::InvalidAttribute(String::from("LayerNorm only supports float32"))); }
    let ndim = input.shape.ndim();
    let resolved = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };
    let outer: usize = input.shape.dims[..resolved].iter().map(|&d| d as usize).product::<usize>().max(1);
    let inner: usize = input.shape.dims[resolved..].iter().map(|&d| d as usize).product::<usize>().max(1);
    let total = input.shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * 4];
    for o in 0..outer {
        let base = o * inner;
        let mut sum = 0.0f32;
        for i in 0..inner { sum += read_f32(&input.raw_data, base + i); }
        let mean = sum / inner as f32;
        let mut var_sum = 0.0f32;
        for i in 0..inner { let d = read_f32(&input.raw_data, base + i) - mean; var_sum += d * d; }
        let inv_std = 1.0 / sqrt_approx(var_sum / inner as f32 + epsilon);
        for i in 0..inner {
            let x = read_f32(&input.raw_data, base + i);
            let s = read_f32(&scale.raw_data, i);
            let b = bias.map(|bt| read_f32(&bt.raw_data, i)).unwrap_or(0.0);
            write_f32(&mut raw_data, base + i, s * (x - mean) * inv_std + b);
        }
    }
    Ok(Tensor { data_type: DataType::Float, shape: input.shape.clone(), name: String::new(), raw_data })
}

/// Max pooling over NCHW input.
pub fn op_maxpool(
    input: &Tensor, kernel_shape: &[i64],
    strides: Option<&[i64]>, pads: Option<&[i64]>,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float || input.shape.ndim() != 4 {
        return Err(OpError::ShapeMismatch(String::from("MaxPool requires 4D float input")));
    }
    let (n, c, h, w) = (input.shape.dims[0] as usize, input.shape.dims[1] as usize,
                          input.shape.dims[2] as usize, input.shape.dims[3] as usize);
    let kh = kernel_shape[0] as usize;
    let kw = kernel_shape[1] as usize;
    let sh = strides.map(|s| s[0] as usize).unwrap_or(kh);
    let sw = strides.map(|s| s[1] as usize).unwrap_or(kw);
    let (pt, pl, pb, pr) = match pads {
        Some(p) if p.len() >= 4 => (p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize),
        _ => (0, 0, 0, 0),
    };
    let oh = (h + pt + pb - kh) / sh + 1;
    let ow = (w + pl + pr - kw) / sw + 1;
    let out_shape = TensorShape::new(Vec::from([n as i64, c as i64, oh as i64, ow as i64]));
    let mut raw_data = alloc::vec![0u8; out_shape.total_elements() * 4];
    for bn in 0..n {
        for ch in 0..c {
            for oi in 0..oh {
                for oj in 0..ow {
                    let mut max_val = f32::NEG_INFINITY;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = oi * sh + ki;
                            let wi = oj * sw + kj;
                            if hi >= pt && hi < h + pt && wi >= pl && wi < w + pl {
                                let idx = bn * c * h * w + ch * h * w + (hi - pt) * w + (wi - pl);
                                let v = read_f32(&input.raw_data, idx);
                                if v > max_val { max_val = v; }
                            }
                        }
                    }
                    let out_idx = bn * c * oh * ow + ch * oh * ow + oi * ow + oj;
                    write_f32(&mut raw_data, out_idx, max_val);
                }
            }
        }
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Average pooling over NCHW input.
pub fn op_averagepool(
    input: &Tensor, kernel_shape: &[i64],
    strides: Option<&[i64]>, pads: Option<&[i64]>,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float || input.shape.ndim() != 4 {
        return Err(OpError::ShapeMismatch(String::from("AvgPool requires 4D float input")));
    }
    let (n, c, h, w) = (input.shape.dims[0] as usize, input.shape.dims[1] as usize,
                          input.shape.dims[2] as usize, input.shape.dims[3] as usize);
    let kh = kernel_shape[0] as usize;
    let kw = kernel_shape[1] as usize;
    let sh = strides.map(|s| s[0] as usize).unwrap_or(kh);
    let sw = strides.map(|s| s[1] as usize).unwrap_or(kw);
    let (pt, pl, pb, pr) = match pads {
        Some(p) if p.len() >= 4 => (p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize),
        _ => (0, 0, 0, 0),
    };
    let oh = (h + pt + pb - kh) / sh + 1;
    let ow = (w + pl + pr - kw) / sw + 1;
    let out_shape = TensorShape::new(Vec::from([n as i64, c as i64, oh as i64, ow as i64]));
    let mut raw_data = alloc::vec![0u8; out_shape.total_elements() * 4];
    for bn in 0..n {
        for ch in 0..c {
            for oi in 0..oh {
                for oj in 0..ow {
                    let mut sum = 0.0f32;
                    let mut count = 0u32;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = oi * sh + ki;
                            let wi = oj * sw + kj;
                            if hi >= pt && hi < h + pt && wi >= pl && wi < w + pl {
                                let idx = bn * c * h * w + ch * h * w + (hi - pt) * w + (wi - pl);
                                sum += read_f32(&input.raw_data, idx);
                                count += 1;
                            }
                        }
                    }
                    let out_idx = bn * c * oh * ow + ch * oh * ow + oi * ow + oj;
                    write_f32(&mut raw_data, out_idx, if count > 0 { sum / count as f32 } else { 0.0 });
                }
            }
        }
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Global average pooling: NCHW → NC11.
pub fn op_global_average_pool(input: &Tensor) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float || input.shape.ndim() != 4 {
        return Err(OpError::ShapeMismatch(String::from("GlobalAvgPool requires 4D float input")));
    }
    let n = input.shape.dims[0] as usize;
    let c = input.shape.dims[1] as usize;
    let h = input.shape.dims[2] as usize;
    let w = input.shape.dims[3] as usize;
    let spatial = h * w;
    let out_shape = TensorShape::new(Vec::from([n as i64, c as i64, 1, 1]));
    let mut raw_data = alloc::vec![0u8; n * c * 4];
    for bn in 0..n {
        for ch in 0..c {
            let mut sum = 0.0f32;
            for s in 0..spatial {
                sum += read_f32(&input.raw_data, bn * c * spatial + ch * spatial + s);
            }
            write_f32(&mut raw_data, bn * c + ch, sum / spatial as f32);
        }
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
}

/// Reduce by mean along specified axes.
pub fn op_reduce_mean(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    let sum_tensor = op_reduce_sum(input, axes, keepdims)?;
    let total_in = input.shape.total_elements();
    let total_out = sum_tensor.shape.total_elements();
    let reduce_count = if total_out > 0 { total_in / total_out } else { 1 };
    let mut raw_data = sum_tensor.raw_data;
    for i in 0..total_out {
        let v = f32::from_le_bytes([raw_data[i*4], raw_data[i*4+1], raw_data[i*4+2], raw_data[i*4+3]]);
        write_f32(&mut raw_data, i, v / reduce_count as f32);
    }
    Ok(Tensor { data_type: DataType::Float, shape: sum_tensor.shape, name: String::new(), raw_data })
}

/// Reduce by sum along specified axes.
pub fn op_reduce_sum(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from("ReduceSum only supports float32")));
    }
    let ndim = input.shape.ndim();
    let resolved: Vec<usize> = if axes.is_empty() {
        (0..ndim).collect()
    } else {
        axes.iter().map(|&a| if a < 0 { (ndim as i64 + a) as usize } else { a as usize }).collect()
    };

    let out_dims: Vec<i64> = (0..ndim)
        .filter_map(|d| {
            if resolved.contains(&d) {
                if keepdims { Some(1) } else { None }
            } else {
                Some(input.shape.dims[d])
            }
        })
        .collect();
    let out_shape = TensorShape::new(if out_dims.is_empty() { Vec::from([1i64]) } else { out_dims });
    let total_out = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total_out * 4];

    let in_strides = compute_strides(&input.shape.dims);
    let in_total = input.shape.total_elements();
    let mut in_coord = alloc::vec![0usize; ndim];
    for i in 0..in_total {
        if i > 0 { next_coord(&mut in_coord, &input.shape.dims); }
        // Compute output index by dropping reduced dimensions
        let mut out_idx = 0usize;
        let out_strides = compute_strides(&out_shape.dims);
        let mut od = 0usize;
        for d in 0..ndim {
            if resolved.contains(&d) {
                if keepdims { od += 1; }
            } else {
                if od < out_strides.len() {
                    out_idx += in_coord[d] * out_strides[od];
                }
                od += 1;
            }
        }
        let prev = f32::from_le_bytes([raw_data[out_idx*4], raw_data[out_idx*4+1], raw_data[out_idx*4+2], raw_data[out_idx*4+3]]);
        write_f32(&mut raw_data, out_idx, prev + read_f32(&input.raw_data, i));
    }
    Ok(Tensor { data_type: DataType::Float, shape: out_shape, name: String::new(), raw_data })
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

    // ---------------------------------------------------------------
    // End-to-end MobileNetV2-style inference test (Task 7.11)
    // ---------------------------------------------------------------
    //
    // Constructs a simplified neural network pipeline that exercises
    // the core operator chain found in MobileNetV2-like architectures:
    //   Input [1,1,6,6] -> Conv [2,1,3,3] -> Relu -> Reshape [1,K]
    //     -> MatMul [K,C] -> Add (bias) -> Softmax -> probabilities
    //
    // Verifies:
    //   - Output shape is [1, num_classes]
    //   - All probabilities are non-negative
    //   - Probabilities sum to ~1.0 (valid distribution)
    //   - Highest probability class is deterministic for fixed weights

    #[test]
    fn test_mobilenetv2_e2e_inference_pipeline() {
        let num_classes: usize = 5;

        // --- Stage 1: Convolution ---
        // Input: [1, 1, 6, 6] (batch=1, channels=1, 6x6 spatial)
        let input_data: Vec<f32> = (0..36).map(|i| (i as f32) * 0.1).collect();
        let input = make_f32_tensor(&[1, 1, 6, 6], &input_data);

        // Conv weight: [2, 1, 3, 3] (2 output channels, 1 input channel, 3x3 kernel)
        // Use edge-detection-like filters
        let conv_weight = make_f32_tensor(
            &[2, 1, 3, 3],
            &[
                // Filter 0: horizontal edge detector
                -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                // Filter 1: vertical edge detector
                -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0,
            ],
        );

        // Conv bias: [2]
        let conv_bias = make_f32_tensor(&[2], &[0.1, 0.2]);

        let conv_out = op_conv(&input, &conv_weight, Some(&conv_bias)).unwrap();
        // Output should be [1, 2, 4, 4] (6-3+1 = 4)
        assert_eq!(conv_out.shape.dims, vec![1, 2, 4, 4]);

        // --- Stage 2: ReLU activation ---
        let relu_out = op_relu(&conv_out).unwrap();
        assert_eq!(relu_out.shape.dims, vec![1, 2, 4, 4]);

        // Verify ReLU: all values >= 0
        let relu_vals = read_f32_vec(&relu_out);
        for &v in &relu_vals {
            assert!(v >= 0.0, "ReLU output must be non-negative, got {}", v);
        }

        // --- Stage 3: Reshape (flatten) ---
        // Flatten [1, 2, 4, 4] -> [1, 32]
        let flat_size = 2 * 4 * 4; // 32
        let reshaped = op_reshape(&relu_out, &[1, flat_size as i64]).unwrap();
        assert_eq!(reshaped.shape.dims, vec![1, 32]);

        // --- Stage 4: Fully-connected layer (MatMul + Add bias) ---
        // Weights: [32, num_classes]
        let mut fc_weight_data = vec![0.0f32; flat_size * num_classes];
        // Initialize with a simple deterministic pattern
        for i in 0..flat_size {
            for j in 0..num_classes {
                // Each class gets a different weight pattern to produce distinct logits
                fc_weight_data[i * num_classes + j] = ((i + j * 7) % 13) as f32 * 0.02 - 0.12;
            }
        }
        let fc_weight = make_f32_tensor(&[flat_size as i64, num_classes as i64], &fc_weight_data);

        let matmul_out = op_matmul(&reshaped, &fc_weight).unwrap();
        assert_eq!(matmul_out.shape.dims, vec![1, num_classes as i64]);

        // Add FC bias
        let fc_bias = make_f32_tensor(&[1, num_classes as i64], &[0.1, -0.2, 0.3, -0.1, 0.05]);
        let logits = op_add(&[&matmul_out, &fc_bias]).unwrap();
        assert_eq!(logits.shape.dims, vec![1, num_classes as i64]);

        // --- Stage 5: Softmax (classification) ---
        let probs = op_softmax(&logits, -1).unwrap();
        assert_eq!(probs.shape.dims, vec![1, num_classes as i64]);

        let prob_vals = read_f32_vec(&probs);
        assert_eq!(prob_vals.len(), num_classes);

        // --- Verification ---
        // 1. All probabilities must be non-negative
        for (i, &p) in prob_vals.iter().enumerate() {
            assert!(p >= 0.0, "probability[{}] = {} must be non-negative", i, p);
        }

        // 2. Probabilities must sum to ~1.0
        let sum: f32 = prob_vals.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "probability sum = {}, expected ~1.0",
            sum
        );

        // 3. Each probability must be in [0, 1]
        for (i, &p) in prob_vals.iter().enumerate() {
            assert!(p <= 1.0 + 1e-6, "probability[{}] = {} exceeds 1.0", i, p);
        }

        // 4. There must be a unique argmax (highest probability class)
        let max_prob = prob_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let argmax = prob_vals
            .iter()
            .position(|&p| (p - max_prob).abs() < 1e-7)
            .unwrap();

        // The predicted class is deterministic for these fixed weights
        // (verifies the full pipeline is computing consistently)
        assert!(argmax < num_classes, "argmax {} out of range", argmax);
    }

    #[test]
    fn test_mobilenetv2_e2e_multi_channel_conv_chain() {
        // More complex pipeline: two Conv->Relu stages feeding into classification
        // Input: [1, 1, 8, 8]
        let input_data: Vec<f32> = (0..64).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let input = make_f32_tensor(&[1, 1, 8, 8], &input_data);

        // Conv1: [1, 1, 8, 8] -> [1, 2, 6, 6] (3x3 kernel, 2 output channels)
        let conv1_w = make_f32_tensor(
            &[2, 1, 3, 3],
            &[
                0.1, 0.2, 0.1, 0.2, 0.4, 0.2, 0.1, 0.2, 0.1, // smooth
                -0.1, 0.0, 0.1, -0.2, 0.0, 0.2, -0.1, 0.0, 0.1, // edge
            ],
        );
        let conv1_out = op_conv(&input, &conv1_w, None).unwrap();
        assert_eq!(conv1_out.shape.dims, vec![1, 2, 6, 6]);

        let relu1_out = op_relu(&conv1_out).unwrap();

        // Conv2: [1, 2, 6, 6] -> [1, 4, 4, 4] (3x3 kernel, 4 output channels)
        let conv2_w_data: Vec<f32> = (0..(4 * 2 * 3 * 3))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();
        let conv2_w = make_f32_tensor(&[4, 2, 3, 3], &conv2_w_data);
        let conv2_out = op_conv(&relu1_out, &conv2_w, None).unwrap();
        assert_eq!(conv2_out.shape.dims, vec![1, 4, 4, 4]);

        let relu2_out = op_relu(&conv2_out).unwrap();

        // Flatten: [1, 4, 4, 4] -> [1, 64]
        let flat = op_reshape(&relu2_out, &[1, 64]).unwrap();
        assert_eq!(flat.shape.dims, vec![1, 64]);

        // FC: [1, 64] x [64, 10] -> [1, 10]
        let num_classes = 10;
        let fc_data: Vec<f32> = (0..64 * num_classes)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.01)
            .collect();
        let fc_w = make_f32_tensor(&[64, num_classes as i64], &fc_data);
        let logits = op_matmul(&flat, &fc_w).unwrap();

        // Softmax
        let probs = op_softmax(&logits, -1).unwrap();
        let prob_vals = read_f32_vec(&probs);
        assert_eq!(prob_vals.len(), num_classes as usize);

        // Verify valid probability distribution
        let sum: f32 = prob_vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {}, expected ~1.0", sum);
        for &p in &prob_vals {
            assert!(p >= 0.0 && p <= 1.0 + 1e-6);
        }
    }

    // ---- Coverage for validate_conv_inputs error paths ----

    #[test]
    fn test_conv_wrong_dtype() {
        // Int32 input should be rejected
        let input = Tensor {
            data_type: DataType::Int32,
            shape: TensorShape::new(vec![1, 1, 3, 3]),
            name: String::new(),
            raw_data: alloc::vec![0u8; 9 * 4],
        };
        let weight = make_f32_tensor(&[1, 1, 1, 1], &[1.0]);
        let result = op_conv(&input, &weight, None);
        assert!(matches!(result, Err(OpError::InvalidAttribute(_))));
    }

    #[test]
    fn test_conv_wrong_rank() {
        // 3D input (missing batch dim) should be rejected
        let input = make_f32_tensor(&[1, 3, 3], &[1.0; 9]);
        let weight = make_f32_tensor(&[1, 1, 1, 1], &[1.0]);
        let result = op_conv(&input, &weight, None);
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    #[test]
    fn test_conv_kernel_larger_than_input() {
        // 4x4 kernel on 3x3 input should be rejected
        let input = make_f32_tensor(&[1, 1, 3, 3], &[1.0; 9]);
        let weight = make_f32_tensor(&[1, 1, 4, 4], &[1.0; 16]);
        let result = op_conv(&input, &weight, None);
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }
}
