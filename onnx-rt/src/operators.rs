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

use crate::tensor::{Tensor, TensorShape};

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
    pub fn from_str(name: &str) -> Option<Self> {
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

impl OperatorRegistry {
    /// Creates a new registry pre-populated with all built-in operators.
    pub fn new() -> Self {
        Self {
            ops: Vec::from(ALL_OPS),
        }
    }

    /// Returns `true` if the given ONNX operator type name is supported.
    pub fn is_supported(&self, op_type: &str) -> bool {
        if let Some(kind) = OpKind::from_str(op_type) {
            self.ops.iter().any(|op| *op == kind)
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

/// Element-wise addition of input tensors.
///
/// Supports broadcasting according to ONNX specification.
/// Currently returns `NotImplemented`.
pub fn op_add(inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    let _ = inputs;
    Err(OpError::NotImplemented)
}

/// Rectified linear unit activation: `max(0, x)`.
///
/// Currently returns `NotImplemented`.
pub fn op_relu(input: &Tensor) -> Result<Tensor, OpError> {
    let _ = input;
    Err(OpError::NotImplemented)
}

/// Matrix multiplication of two tensors.
///
/// Follows ONNX MatMul semantics: supports batched matrix multiply
/// and broadcasting of batch dimensions.
/// Currently returns `NotImplemented`.
pub fn op_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
    let _ = (a, b);
    Err(OpError::NotImplemented)
}

/// Softmax normalization along the specified axis.
///
/// Computes `exp(x_i) / sum(exp(x_j))` along `axis`.
/// Currently returns `NotImplemented`.
pub fn op_softmax(input: &Tensor, axis: i64) -> Result<Tensor, OpError> {
    let _ = (input, axis);
    Err(OpError::NotImplemented)
}

/// Reshape a tensor to the specified shape.
///
/// Follows ONNX Reshape semantics: `-1` indicates an inferred
/// dimension, and `0` means copy from the original shape.
/// Currently returns `NotImplemented`.
pub fn op_reshape(input: &Tensor, shape: &[i64]) -> Result<Tensor, OpError> {
    let _ = (input, shape);
    Err(OpError::NotImplemented)
}

/// N-dimensional convolution.
///
/// Supports optional bias. Attributes such as kernel size, strides,
/// padding, and dilations will be passed via operator attributes
/// when fully implemented.
/// Currently returns `NotImplemented`.
pub fn op_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor, OpError> {
    let _ = (input, weight, bias);
    Err(OpError::NotImplemented)
}

// ---------------------------------------------------------------------------
// Shape inference helpers
// ---------------------------------------------------------------------------

/// Infers the output shape for a binary element-wise operation with broadcasting.
///
/// Applies ONNX broadcasting rules: dimensions are compared from the
/// trailing end; a dimension of 1 is broadcastable to any size.
/// Currently returns `NotImplemented`.
pub fn infer_binary_shape(
    a: &TensorShape,
    b: &TensorShape,
) -> Result<TensorShape, OpError> {
    let _ = (a, b);
    Err(OpError::NotImplemented)
}

/// Infers the output shape for matrix multiplication.
///
/// For 2D inputs with shapes `[M, K]` and `[K, N]`, validates that
/// the inner dimensions match and returns shape `[M, N]`.
/// Returns `ShapeMismatch` if the inner dimensions do not agree.
pub fn infer_matmul_shape(
    a: &TensorShape,
    b: &TensorShape,
) -> Result<TensorShape, OpError> {
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
    use alloc::format;
    use crate::tensor::DataType;
    use alloc::string::String;
    use alloc::vec;

    // ---- OpKind from_str tests ----

    #[test]
    fn test_opkind_from_str_arithmetic() {
        assert_eq!(OpKind::from_str("Add"), Some(OpKind::Add));
        assert_eq!(OpKind::from_str("Sub"), Some(OpKind::Sub));
        assert_eq!(OpKind::from_str("Mul"), Some(OpKind::Mul));
        assert_eq!(OpKind::from_str("Div"), Some(OpKind::Div));
        assert_eq!(OpKind::from_str("MatMul"), Some(OpKind::MatMul));
    }

    #[test]
    fn test_opkind_from_str_activations() {
        assert_eq!(OpKind::from_str("Relu"), Some(OpKind::Relu));
        assert_eq!(OpKind::from_str("Sigmoid"), Some(OpKind::Sigmoid));
        assert_eq!(OpKind::from_str("Tanh"), Some(OpKind::Tanh));
        assert_eq!(OpKind::from_str("Softmax"), Some(OpKind::Softmax));
    }

    #[test]
    fn test_opkind_from_str_shape_ops() {
        assert_eq!(OpKind::from_str("Reshape"), Some(OpKind::Reshape));
        assert_eq!(OpKind::from_str("Transpose"), Some(OpKind::Transpose));
        assert_eq!(OpKind::from_str("Flatten"), Some(OpKind::Flatten));
        assert_eq!(OpKind::from_str("Squeeze"), Some(OpKind::Squeeze));
        assert_eq!(OpKind::from_str("Unsqueeze"), Some(OpKind::Unsqueeze));
    }

    #[test]
    fn test_opkind_from_str_invalid() {
        assert_eq!(OpKind::from_str("NonExistent"), None);
        assert_eq!(OpKind::from_str("add"), None); // case-sensitive
        assert_eq!(OpKind::from_str("RELU"), None);
        assert_eq!(OpKind::from_str(""), None);
    }

    #[test]
    fn test_opkind_name_roundtrip() {
        // Every variant should round-trip through name() and from_str().
        let ops = [
            OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div, OpKind::MatMul,
            OpKind::Relu, OpKind::Sigmoid, OpKind::Tanh, OpKind::Softmax,
            OpKind::Conv, OpKind::MaxPool, OpKind::AveragePool, OpKind::BatchNormalization,
            OpKind::Reshape, OpKind::Transpose, OpKind::Flatten, OpKind::Squeeze,
            OpKind::Unsqueeze, OpKind::Concat, OpKind::Gather, OpKind::Slice, OpKind::Pad,
            OpKind::Gemm, OpKind::GlobalAveragePool, OpKind::LayerNormalization,
            OpKind::Cast, OpKind::Clip, OpKind::ReduceMean, OpKind::ReduceSum,
        ];
        for op in ops.iter() {
            let name = op.name();
            let parsed = OpKind::from_str(name);
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

    // ---- Operator stub tests ----

    #[test]
    fn test_op_add_not_implemented() {
        let t = Tensor::new(DataType::Float, TensorShape::new(vec![2, 3]), String::from("a"));
        let result = op_add(&[&t, &t]);
        assert_eq!(result, Err(OpError::NotImplemented));
    }

    #[test]
    fn test_op_relu_not_implemented() {
        let t = Tensor::new(DataType::Float, TensorShape::new(vec![4]), String::from("x"));
        let result = op_relu(&t);
        assert_eq!(result, Err(OpError::NotImplemented));
    }

    #[test]
    fn test_op_matmul_not_implemented() {
        let a = Tensor::new(DataType::Float, TensorShape::new(vec![2, 3]), String::from("a"));
        let b = Tensor::new(DataType::Float, TensorShape::new(vec![3, 4]), String::from("b"));
        let result = op_matmul(&a, &b);
        assert_eq!(result, Err(OpError::NotImplemented));
    }

    #[test]
    fn test_op_softmax_not_implemented() {
        let t = Tensor::new(DataType::Float, TensorShape::new(vec![1, 10]), String::from("logits"));
        let result = op_softmax(&t, -1);
        assert_eq!(result, Err(OpError::NotImplemented));
    }

    #[test]
    fn test_op_reshape_not_implemented() {
        let t = Tensor::new(DataType::Float, TensorShape::new(vec![2, 3]), String::from("in"));
        let result = op_reshape(&t, &[6]);
        assert_eq!(result, Err(OpError::NotImplemented));
    }

    #[test]
    fn test_op_conv_not_implemented() {
        let input = Tensor::new(DataType::Float, TensorShape::new(vec![1, 3, 224, 224]), String::from("input"));
        let weight = Tensor::new(DataType::Float, TensorShape::new(vec![64, 3, 7, 7]), String::from("weight"));
        let result = op_conv(&input, &weight, None);
        assert_eq!(result, Err(OpError::NotImplemented));
    }

    // ---- Shape inference tests ----

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
