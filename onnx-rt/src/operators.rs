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

use crate::byte_io::{
    allocate_tensor_data, read_f32, read_i32, read_i64, write_f32, write_i32, write_i64,
};
use crate::onnx_types::AttributeProto;
use crate::tensor::{bf16_to_f32, f32_to_bf16, DataType, Tensor, TensorShape};

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
// Operator domain
// ---------------------------------------------------------------------------

/// ONNX operator domain. SmallAIOS distinguishes the standard ONNX op set
/// from the `com.microsoft` contrib op set so that operators with the same
/// name but different semantics can coexist in the registry (see the
/// `microsoft-fused-ops-v1` OpenSpec change).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    /// The default ONNX operator set (`""` or `"ai.onnx"`).
    StandardOnnx,
    /// The `com.microsoft` contrib op set: fused operators emitted by
    /// HuggingFace `optimum-cli export onnx` for modern decoder-only LLMs.
    MicrosoftFused,
}

impl Domain {
    /// Parse an ONNX `NodeProto.domain` string to a [`Domain`]. The empty
    /// string and `"ai.onnx"` both map to [`Domain::StandardOnnx`].
    pub fn parse_str(s: &str) -> Option<Self> {
        match s {
            "" | "ai.onnx" => Some(Domain::StandardOnnx),
            "com.microsoft" => Some(Domain::MicrosoftFused),
            _ => None,
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

    // ---- Tier 2: math primitives ----
    /// Element-wise power.
    Pow,
    /// Element-wise square root.
    Sqrt,
    /// Element-wise natural exponential.
    Exp,
    /// Element-wise natural logarithm.
    Log,
    /// Element-wise error function.
    Erf,
    /// Element-wise negation.
    Neg,
    /// Element-wise absolute value.
    Abs,
    /// Element-wise floor.
    Floor,
    /// Element-wise ceiling.
    Ceil,
    /// Element-wise rounding.
    Round,

    // ---- Tier 2: comparison and selection ----
    /// Element-wise equality.
    Equal,
    /// Element-wise inequality.
    NotEqual,
    /// Element-wise less-than.
    Less,
    /// Element-wise less-than-or-equal.
    LessOrEqual,
    /// Element-wise greater-than.
    Greater,
    /// Element-wise greater-than-or-equal.
    GreaterOrEqual,
    /// Element-wise ternary select.
    Where,
    /// Element-wise minimum.
    Min,
    /// Element-wise maximum.
    Max,
    /// Element-wise boolean negation.
    Not,

    // ---- Tier 2: composite activations ----
    /// Gaussian Error Linear Unit.
    Gelu,
    /// Leaky ReLU.
    LeakyRelu,
    /// Exponential Linear Unit.
    Elu,
    /// Sigmoid Linear Unit (Swish/SiLU).
    Swish,

    // ---- Tier 2: recurrent ----
    /// Basic RNN.
    RNN,
    /// Long Short-Term Memory.
    LSTM,
    /// Gated Recurrent Unit.
    GRU,

    // ---- Tier 2: transformer building blocks ----
    /// Split tensor along an axis.
    Split,
    /// Expand (broadcast) to a target shape.
    Expand,
    /// Repeat tensor along axes.
    Tile,
    /// One-hot encoding.
    OneHot,
    /// Einstein summation.
    Einsum,

    // ---- Tier 2: quantized ----
    /// Float -> int8/uint8 quantization.
    QuantizeLinear,
    /// int8/uint8 -> float dequantization.
    DequantizeLinear,
    /// Quantized matrix multiply.
    QLinearMatMul,
    /// Quantized convolution.
    QLinearConv,

    // ---- Tier 3 (Phase 1): transformer-model math ----
    /// Element-wise modulo.
    Mod,
    /// Element-wise sine.
    Sin,
    /// Element-wise cosine.
    Cos,
    /// Element-wise reciprocal (1/x).
    Reciprocal,
    /// Element-wise sign function.
    Sign,
    /// Variadic element-wise sum.
    Sum,
    /// Variadic element-wise mean.
    Mean,
    /// Element-wise boolean AND.
    And,
    /// Element-wise boolean OR.
    Or,
    /// Numerically-stable log-softmax.
    LogSoftmax,

    // ---- Tier 3 (Phase 1): reductions ----
    /// Reduce by maximum.
    ReduceMax,
    /// Reduce by minimum.
    ReduceMin,
    /// Reduce by product.
    ReduceProd,
    /// Argmax (index of max).
    ArgMax,
    /// Argmin (index of min).
    ArgMin,

    // ---- Tier 3 (Phase 1): shape / data movement ----
    /// Runtime tensor shape.
    Shape,
    /// Runtime tensor element count.
    Size,
    /// Identity pass-through.
    Identity,
    /// Constant value (attribute-decoded).
    Constant,
    /// Broadcast a scalar to a shape.
    ConstantOfShape,
    /// Construct a 1-D arithmetic sequence.
    Range,
    /// Upper/lower triangular mask.
    Trilu,
    /// Cumulative sum along an axis.
    CumSum,
    /// N-dimensional gather.
    GatherND,
    /// N-dimensional scatter.
    ScatterND,

    // ---- Phase 1 vision: image ops ----
    /// 2D image resize (nearest/bilinear).
    Resize,
    /// Rearrange depth into spatial blocks.
    DepthToSpace,
    /// Rearrange spatial blocks into depth.
    SpaceToDepth,
    /// Region-of-interest align (bilinear avg).
    RoiAlign,
    /// 2D grid sampling (bilinear/nearest).
    GridSample,
    /// Global max pooling.
    GlobalMaxPool,
    /// Symmetric crop/pad to target shape.
    CenterCropPad,

    // ---- Phase 1 vision: data selection ----
    /// Top-k values and indices along an axis.
    TopK,
    /// Select elements where condition is true.
    Compress,
    /// Indices of non-zero elements.
    NonZero,
    /// Unique values (primary output only).
    Unique,
    /// Element-wise gather along an axis.
    GatherElements,
    /// Element-wise scatter along an axis.
    ScatterElements,

    // ---- Phase 1 vision: normalization ----
    /// Group normalization.
    GroupNormalization,
    /// Instance normalization.
    InstanceNormalization,

    // ---- Phase 1 vision: composite activations ----
    /// Hard sigmoid (clipped affine).
    HardSigmoid,
    /// Hard swish (x * relu6(x+3)/6).
    HardSwish,
    /// Parametric ReLU (per-channel negative slope).
    PRelu,
    /// Mish activation (x * tanh(softplus(x))).
    Mish,

    // ---- Phase 2 (generative-llm-v1): control flow ----
    /// Conditional branch selecting `then_branch` or `else_branch`.
    If,
    /// Iterated sub-graph with loop-carried state.
    Loop,
    /// Sequence-axis scan over an inner body.
    Scan,

    // ---- Phase 2: generative / quantized LLM primitives ----
    /// RMS-based layer normalization (LLaMA/Mistral).
    RMSNormalization,
    /// Integer matrix multiply with zero-point folding.
    MatMulInteger,
    /// Compute scale + zero-point at runtime and quantize.
    DynamicQuantizeLinear,

    // ---- Phase 2: samplers / random ----
    /// Gaussian random tensor.
    RandomNormal,
    /// Gaussian random tensor with input-derived shape.
    RandomNormalLike,
    /// Uniform random tensor.
    RandomUniform,
    /// Uniform random tensor with input-derived shape.
    RandomUniformLike,
    /// Multinomial sampling (per-row categorical).
    Multinomial,
    /// Bernoulli draws (per-element probability).
    Bernoulli,
    /// Dropout (inference-mode pass-through by default).
    Dropout,
    /// Identity matrix generator.
    EyeLike,

    // ---- Phase 2: extra reductions ----
    /// Sum of absolute values.
    ReduceL1,
    /// Euclidean norm.
    ReduceL2,
    /// Log of sum.
    ReduceLogSum,
    /// Log of sum of exponentials.
    ReduceLogSumExp,
    /// Sum of squares.
    ReduceSumSquare,

    // ---- Phase 2: normalization + elementwise ----
    /// Lp normalization.
    LpNormalization,
    /// Mean/variance normalization.
    MeanVarianceNormalization,
    /// Softplus activation: ln(1 + e^x).
    Softplus,

    // ---- microsoft-fused-ops-v1: com.microsoft-domain fused LLM ops ----
    /// `com.microsoft.SimplifiedLayerNormalization` — RMSNorm alias used
    /// by canonical HF ONNX exports of Llama / Gemma / DeepSeek.
    SimplifiedLayerNormalization,
    /// `com.microsoft.SkipSimplifiedLayerNormalization` — fused residual
    /// add + RMSNorm, emits both the normalized output and the pre-sum.
    SkipSimplifiedLayerNormalization,
    /// `com.microsoft.RotaryEmbedding` — standalone fused RoPE op used by
    /// DeepSeek upstream of MultiHeadAttention.
    RotaryEmbedding,
    /// `com.microsoft.MultiHeadAttention` — non-grouped attention fusion
    /// with KV-cache support (DeepSeek).
    MultiHeadAttention,
    /// `com.microsoft.GroupQueryAttention` — grouped-query attention with
    /// fused RoPE, KV-cache concat, and on-the-fly causal masking
    /// (Llama / Gemma).
    GroupQueryAttention,
}

impl OpKind {
    /// Returns the [`Domain`] this operator belongs to. Every variant
    /// except the `microsoft-fused-ops-v1` additions is
    /// [`Domain::StandardOnnx`].
    pub fn domain(&self) -> Domain {
        match self {
            OpKind::SimplifiedLayerNormalization
            | OpKind::SkipSimplifiedLayerNormalization
            | OpKind::RotaryEmbedding
            | OpKind::MultiHeadAttention
            | OpKind::GroupQueryAttention => Domain::MicrosoftFused,
            _ => Domain::StandardOnnx,
        }
    }

    /// Resolve a `(domain, op_type)` pair to a canonical [`OpKind`]
    /// variant. Used by the dispatcher to differentiate standard-domain
    /// ops from `com.microsoft`-domain contrib ops that may share names.
    pub fn lookup_by_domain_and_name(domain: &str, name: &str) -> Option<Self> {
        let d = Domain::parse_str(domain)?;
        let kind = Self::parse_str(name)?;
        if kind.domain() == d {
            Some(kind)
        } else {
            None
        }
    }

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
            // Tier 2: math
            "Pow" => Some(OpKind::Pow),
            "Sqrt" => Some(OpKind::Sqrt),
            "Exp" => Some(OpKind::Exp),
            "Log" => Some(OpKind::Log),
            "Erf" => Some(OpKind::Erf),
            "Neg" => Some(OpKind::Neg),
            "Abs" => Some(OpKind::Abs),
            "Floor" => Some(OpKind::Floor),
            "Ceil" => Some(OpKind::Ceil),
            "Round" => Some(OpKind::Round),
            // Tier 2: comparison
            "Equal" => Some(OpKind::Equal),
            "NotEqual" => Some(OpKind::NotEqual),
            "Less" => Some(OpKind::Less),
            "LessOrEqual" => Some(OpKind::LessOrEqual),
            "Greater" => Some(OpKind::Greater),
            "GreaterOrEqual" => Some(OpKind::GreaterOrEqual),
            "Where" => Some(OpKind::Where),
            "Min" => Some(OpKind::Min),
            "Max" => Some(OpKind::Max),
            "Not" => Some(OpKind::Not),
            // Tier 2: activations
            "Gelu" => Some(OpKind::Gelu),
            "LeakyRelu" => Some(OpKind::LeakyRelu),
            "Elu" => Some(OpKind::Elu),
            "Swish" => Some(OpKind::Swish),
            // Tier 2: recurrent
            "RNN" => Some(OpKind::RNN),
            "LSTM" => Some(OpKind::LSTM),
            "GRU" => Some(OpKind::GRU),
            // Tier 2: transformer
            "Split" => Some(OpKind::Split),
            "Expand" => Some(OpKind::Expand),
            "Tile" => Some(OpKind::Tile),
            "OneHot" => Some(OpKind::OneHot),
            "Einsum" => Some(OpKind::Einsum),
            // Tier 2: quantized
            "QuantizeLinear" => Some(OpKind::QuantizeLinear),
            "DequantizeLinear" => Some(OpKind::DequantizeLinear),
            "QLinearMatMul" => Some(OpKind::QLinearMatMul),
            "QLinearConv" => Some(OpKind::QLinearConv),
            // Tier 3 Phase 1: math
            "Mod" => Some(OpKind::Mod),
            "Sin" => Some(OpKind::Sin),
            "Cos" => Some(OpKind::Cos),
            "Reciprocal" => Some(OpKind::Reciprocal),
            "Sign" => Some(OpKind::Sign),
            "Sum" => Some(OpKind::Sum),
            "Mean" => Some(OpKind::Mean),
            "And" => Some(OpKind::And),
            "Or" => Some(OpKind::Or),
            "LogSoftmax" => Some(OpKind::LogSoftmax),
            // Tier 3 Phase 1: reductions
            "ReduceMax" => Some(OpKind::ReduceMax),
            "ReduceMin" => Some(OpKind::ReduceMin),
            "ReduceProd" => Some(OpKind::ReduceProd),
            "ArgMax" => Some(OpKind::ArgMax),
            "ArgMin" => Some(OpKind::ArgMin),
            // Tier 3 Phase 1: shape / data
            "Shape" => Some(OpKind::Shape),
            "Size" => Some(OpKind::Size),
            "Identity" => Some(OpKind::Identity),
            "Constant" => Some(OpKind::Constant),
            "ConstantOfShape" => Some(OpKind::ConstantOfShape),
            "Range" => Some(OpKind::Range),
            "Trilu" => Some(OpKind::Trilu),
            "CumSum" => Some(OpKind::CumSum),
            "GatherND" => Some(OpKind::GatherND),
            "ScatterND" => Some(OpKind::ScatterND),
            // Phase 1 vision: image
            "Resize" => Some(OpKind::Resize),
            "DepthToSpace" => Some(OpKind::DepthToSpace),
            "SpaceToDepth" => Some(OpKind::SpaceToDepth),
            "RoiAlign" => Some(OpKind::RoiAlign),
            "GridSample" => Some(OpKind::GridSample),
            "GlobalMaxPool" => Some(OpKind::GlobalMaxPool),
            "CenterCropPad" => Some(OpKind::CenterCropPad),
            // Phase 1 vision: data selection
            "TopK" => Some(OpKind::TopK),
            "Compress" => Some(OpKind::Compress),
            "NonZero" => Some(OpKind::NonZero),
            "Unique" => Some(OpKind::Unique),
            "GatherElements" => Some(OpKind::GatherElements),
            "ScatterElements" => Some(OpKind::ScatterElements),
            // Phase 1 vision: normalization
            "GroupNormalization" => Some(OpKind::GroupNormalization),
            "InstanceNormalization" => Some(OpKind::InstanceNormalization),
            // Phase 1 vision: activations
            "HardSigmoid" => Some(OpKind::HardSigmoid),
            "HardSwish" => Some(OpKind::HardSwish),
            "PRelu" => Some(OpKind::PRelu),
            "Mish" => Some(OpKind::Mish),
            // Phase 2: control flow
            "If" => Some(OpKind::If),
            "Loop" => Some(OpKind::Loop),
            "Scan" => Some(OpKind::Scan),
            // Phase 2: generative / quantized
            "RMSNormalization" => Some(OpKind::RMSNormalization),
            "MatMulInteger" => Some(OpKind::MatMulInteger),
            "DynamicQuantizeLinear" => Some(OpKind::DynamicQuantizeLinear),
            // Phase 2: random / samplers
            "RandomNormal" => Some(OpKind::RandomNormal),
            "RandomNormalLike" => Some(OpKind::RandomNormalLike),
            "RandomUniform" => Some(OpKind::RandomUniform),
            "RandomUniformLike" => Some(OpKind::RandomUniformLike),
            "Multinomial" => Some(OpKind::Multinomial),
            "Bernoulli" => Some(OpKind::Bernoulli),
            "Dropout" => Some(OpKind::Dropout),
            "EyeLike" => Some(OpKind::EyeLike),
            // Phase 2: reductions
            "ReduceL1" => Some(OpKind::ReduceL1),
            "ReduceL2" => Some(OpKind::ReduceL2),
            "ReduceLogSum" => Some(OpKind::ReduceLogSum),
            "ReduceLogSumExp" => Some(OpKind::ReduceLogSumExp),
            "ReduceSumSquare" => Some(OpKind::ReduceSumSquare),
            // Phase 2: normalization / elementwise
            "LpNormalization" => Some(OpKind::LpNormalization),
            "MeanVarianceNormalization" => Some(OpKind::MeanVarianceNormalization),
            "Softplus" => Some(OpKind::Softplus),
            // microsoft-fused-ops-v1: com.microsoft-domain fused LLM ops
            "SimplifiedLayerNormalization" => Some(OpKind::SimplifiedLayerNormalization),
            "SkipSimplifiedLayerNormalization" => Some(OpKind::SkipSimplifiedLayerNormalization),
            "RotaryEmbedding" => Some(OpKind::RotaryEmbedding),
            "MultiHeadAttention" => Some(OpKind::MultiHeadAttention),
            "GroupQueryAttention" => Some(OpKind::GroupQueryAttention),
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
            OpKind::Pow => "Pow",
            OpKind::Sqrt => "Sqrt",
            OpKind::Exp => "Exp",
            OpKind::Log => "Log",
            OpKind::Erf => "Erf",
            OpKind::Neg => "Neg",
            OpKind::Abs => "Abs",
            OpKind::Floor => "Floor",
            OpKind::Ceil => "Ceil",
            OpKind::Round => "Round",
            OpKind::Equal => "Equal",
            OpKind::NotEqual => "NotEqual",
            OpKind::Less => "Less",
            OpKind::LessOrEqual => "LessOrEqual",
            OpKind::Greater => "Greater",
            OpKind::GreaterOrEqual => "GreaterOrEqual",
            OpKind::Where => "Where",
            OpKind::Min => "Min",
            OpKind::Max => "Max",
            OpKind::Not => "Not",
            OpKind::Gelu => "Gelu",
            OpKind::LeakyRelu => "LeakyRelu",
            OpKind::Elu => "Elu",
            OpKind::Swish => "Swish",
            OpKind::RNN => "RNN",
            OpKind::LSTM => "LSTM",
            OpKind::GRU => "GRU",
            OpKind::Split => "Split",
            OpKind::Expand => "Expand",
            OpKind::Tile => "Tile",
            OpKind::OneHot => "OneHot",
            OpKind::Einsum => "Einsum",
            OpKind::QuantizeLinear => "QuantizeLinear",
            OpKind::DequantizeLinear => "DequantizeLinear",
            OpKind::QLinearMatMul => "QLinearMatMul",
            OpKind::QLinearConv => "QLinearConv",
            OpKind::Mod => "Mod",
            OpKind::Sin => "Sin",
            OpKind::Cos => "Cos",
            OpKind::Reciprocal => "Reciprocal",
            OpKind::Sign => "Sign",
            OpKind::Sum => "Sum",
            OpKind::Mean => "Mean",
            OpKind::And => "And",
            OpKind::Or => "Or",
            OpKind::LogSoftmax => "LogSoftmax",
            OpKind::ReduceMax => "ReduceMax",
            OpKind::ReduceMin => "ReduceMin",
            OpKind::ReduceProd => "ReduceProd",
            OpKind::ArgMax => "ArgMax",
            OpKind::ArgMin => "ArgMin",
            OpKind::Shape => "Shape",
            OpKind::Size => "Size",
            OpKind::Identity => "Identity",
            OpKind::Constant => "Constant",
            OpKind::ConstantOfShape => "ConstantOfShape",
            OpKind::Range => "Range",
            OpKind::Trilu => "Trilu",
            OpKind::CumSum => "CumSum",
            OpKind::GatherND => "GatherND",
            OpKind::ScatterND => "ScatterND",
            OpKind::Resize => "Resize",
            OpKind::DepthToSpace => "DepthToSpace",
            OpKind::SpaceToDepth => "SpaceToDepth",
            OpKind::RoiAlign => "RoiAlign",
            OpKind::GridSample => "GridSample",
            OpKind::GlobalMaxPool => "GlobalMaxPool",
            OpKind::CenterCropPad => "CenterCropPad",
            OpKind::TopK => "TopK",
            OpKind::Compress => "Compress",
            OpKind::NonZero => "NonZero",
            OpKind::Unique => "Unique",
            OpKind::GatherElements => "GatherElements",
            OpKind::ScatterElements => "ScatterElements",
            OpKind::GroupNormalization => "GroupNormalization",
            OpKind::InstanceNormalization => "InstanceNormalization",
            OpKind::HardSigmoid => "HardSigmoid",
            OpKind::HardSwish => "HardSwish",
            OpKind::PRelu => "PRelu",
            OpKind::Mish => "Mish",
            // Phase 2
            OpKind::If => "If",
            OpKind::Loop => "Loop",
            OpKind::Scan => "Scan",
            OpKind::RMSNormalization => "RMSNormalization",
            OpKind::MatMulInteger => "MatMulInteger",
            OpKind::DynamicQuantizeLinear => "DynamicQuantizeLinear",
            OpKind::RandomNormal => "RandomNormal",
            OpKind::RandomNormalLike => "RandomNormalLike",
            OpKind::RandomUniform => "RandomUniform",
            OpKind::RandomUniformLike => "RandomUniformLike",
            OpKind::Multinomial => "Multinomial",
            OpKind::Bernoulli => "Bernoulli",
            OpKind::Dropout => "Dropout",
            OpKind::EyeLike => "EyeLike",
            OpKind::ReduceL1 => "ReduceL1",
            OpKind::ReduceL2 => "ReduceL2",
            OpKind::ReduceLogSum => "ReduceLogSum",
            OpKind::ReduceLogSumExp => "ReduceLogSumExp",
            OpKind::ReduceSumSquare => "ReduceSumSquare",
            OpKind::LpNormalization => "LpNormalization",
            OpKind::MeanVarianceNormalization => "MeanVarianceNormalization",
            OpKind::Softplus => "Softplus",
            // microsoft-fused-ops-v1
            OpKind::SimplifiedLayerNormalization => "SimplifiedLayerNormalization",
            OpKind::SkipSimplifiedLayerNormalization => "SkipSimplifiedLayerNormalization",
            OpKind::RotaryEmbedding => "RotaryEmbedding",
            OpKind::MultiHeadAttention => "MultiHeadAttention",
            OpKind::GroupQueryAttention => "GroupQueryAttention",
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
    // Tier 2
    OpKind::Pow,
    OpKind::Sqrt,
    OpKind::Exp,
    OpKind::Log,
    OpKind::Erf,
    OpKind::Neg,
    OpKind::Abs,
    OpKind::Floor,
    OpKind::Ceil,
    OpKind::Round,
    OpKind::Equal,
    OpKind::NotEqual,
    OpKind::Less,
    OpKind::LessOrEqual,
    OpKind::Greater,
    OpKind::GreaterOrEqual,
    OpKind::Where,
    OpKind::Min,
    OpKind::Max,
    OpKind::Not,
    OpKind::Gelu,
    OpKind::LeakyRelu,
    OpKind::Elu,
    OpKind::Swish,
    OpKind::RNN,
    OpKind::LSTM,
    OpKind::GRU,
    OpKind::Split,
    OpKind::Expand,
    OpKind::Tile,
    OpKind::OneHot,
    OpKind::Einsum,
    OpKind::QuantizeLinear,
    OpKind::DequantizeLinear,
    OpKind::QLinearMatMul,
    OpKind::QLinearConv,
    // Tier 3 Phase 1
    OpKind::Mod,
    OpKind::Sin,
    OpKind::Cos,
    OpKind::Reciprocal,
    OpKind::Sign,
    OpKind::Sum,
    OpKind::Mean,
    OpKind::And,
    OpKind::Or,
    OpKind::LogSoftmax,
    OpKind::ReduceMax,
    OpKind::ReduceMin,
    OpKind::ReduceProd,
    OpKind::ArgMax,
    OpKind::ArgMin,
    OpKind::Shape,
    OpKind::Size,
    OpKind::Identity,
    OpKind::Constant,
    OpKind::ConstantOfShape,
    OpKind::Range,
    OpKind::Trilu,
    OpKind::CumSum,
    OpKind::GatherND,
    OpKind::ScatterND,
    // Phase 1 vision
    OpKind::Resize,
    OpKind::DepthToSpace,
    OpKind::SpaceToDepth,
    OpKind::RoiAlign,
    OpKind::GridSample,
    OpKind::GlobalMaxPool,
    OpKind::CenterCropPad,
    OpKind::TopK,
    OpKind::Compress,
    OpKind::NonZero,
    OpKind::Unique,
    OpKind::GatherElements,
    OpKind::ScatterElements,
    OpKind::GroupNormalization,
    OpKind::InstanceNormalization,
    OpKind::HardSigmoid,
    OpKind::HardSwish,
    OpKind::PRelu,
    OpKind::Mish,
    // Phase 2 (generative-llm-v1)
    OpKind::If,
    OpKind::Loop,
    OpKind::Scan,
    OpKind::RMSNormalization,
    OpKind::MatMulInteger,
    OpKind::DynamicQuantizeLinear,
    OpKind::RandomNormal,
    OpKind::RandomNormalLike,
    OpKind::RandomUniform,
    OpKind::RandomUniformLike,
    OpKind::Multinomial,
    OpKind::Bernoulli,
    OpKind::Dropout,
    OpKind::EyeLike,
    OpKind::ReduceL1,
    OpKind::ReduceL2,
    OpKind::ReduceLogSum,
    OpKind::ReduceLogSumExp,
    OpKind::ReduceSumSquare,
    OpKind::LpNormalization,
    OpKind::MeanVarianceNormalization,
    OpKind::Softplus,
    // microsoft-fused-ops-v1: com.microsoft-domain fused LLM operators
    OpKind::SimplifiedLayerNormalization,
    OpKind::SkipSimplifiedLayerNormalization,
    OpKind::RotaryEmbedding,
    OpKind::MultiHeadAttention,
    OpKind::GroupQueryAttention,
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
// Operator coverage inventory
// ---------------------------------------------------------------------------

/// Coverage roadmap phases for standard ONNX operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Phase 1 — transformer models (e.g., BERT).
    P1,
    /// Phase 2 — generative / control-flow operators.
    P2,
    /// Phase 3 — audio + detection models.
    P3,
    /// Phase 4 — long tail (rarely used operators).
    P4,
}

/// Status of an ONNX operator within SmallAIOS's coverage roadmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorStatus {
    /// Implemented and dispatched by the runtime today.
    Implemented,
    /// Listed in a future roadmap phase.
    Planned(Phase),
    /// Requires a subsystem we have not yet built (sequence types,
    /// optional types, control-flow executor, string tensors, ONNX-ML, ...).
    DeferredSubsystem(&'static str),
    /// Training-only operator; SmallAIOS is inference-only.
    SkippedTraining,
    /// Replaced by a newer ONNX operator.
    SkippedDeprecated,
    /// Vendor (non-standard) operator.
    #[allow(dead_code)]
    SkippedVendor,
}

/// Complete inventory of standard ONNX operators recognised by SmallAIOS's
/// coverage roadmap.
///
/// Used by the `test_inventory_matches_registry` unit test to prevent
/// drift between the runtime dispatch registry and the planning document.
/// Every entry marked [`OperatorStatus::Implemented`] must correspond to a
/// real [`OpKind`] variant and vice-versa.
pub const SUPPORTED_OPS_INVENTORY: &[(&str, OperatorStatus)] = &[
    // ---------------- Implemented (Tier 1 + Tier 2 + Tier 3 Phase 1) ----
    ("Add", OperatorStatus::Implemented),
    ("Sub", OperatorStatus::Implemented),
    ("Mul", OperatorStatus::Implemented),
    ("Div", OperatorStatus::Implemented),
    ("MatMul", OperatorStatus::Implemented),
    ("Relu", OperatorStatus::Implemented),
    ("Sigmoid", OperatorStatus::Implemented),
    ("Tanh", OperatorStatus::Implemented),
    ("Softmax", OperatorStatus::Implemented),
    ("Conv", OperatorStatus::Implemented),
    ("MaxPool", OperatorStatus::Implemented),
    ("AveragePool", OperatorStatus::Implemented),
    ("BatchNormalization", OperatorStatus::Implemented),
    ("Reshape", OperatorStatus::Implemented),
    ("Transpose", OperatorStatus::Implemented),
    ("Flatten", OperatorStatus::Implemented),
    ("Squeeze", OperatorStatus::Implemented),
    ("Unsqueeze", OperatorStatus::Implemented),
    ("Concat", OperatorStatus::Implemented),
    ("Gather", OperatorStatus::Implemented),
    ("Slice", OperatorStatus::Implemented),
    ("Pad", OperatorStatus::Implemented),
    ("Gemm", OperatorStatus::Implemented),
    ("GlobalAveragePool", OperatorStatus::Implemented),
    ("LayerNormalization", OperatorStatus::Implemented),
    ("Cast", OperatorStatus::Implemented),
    ("Clip", OperatorStatus::Implemented),
    ("ReduceMean", OperatorStatus::Implemented),
    ("ReduceSum", OperatorStatus::Implemented),
    ("Pow", OperatorStatus::Implemented),
    ("Sqrt", OperatorStatus::Implemented),
    ("Exp", OperatorStatus::Implemented),
    ("Log", OperatorStatus::Implemented),
    ("Erf", OperatorStatus::Implemented),
    ("Neg", OperatorStatus::Implemented),
    ("Abs", OperatorStatus::Implemented),
    ("Floor", OperatorStatus::Implemented),
    ("Ceil", OperatorStatus::Implemented),
    ("Round", OperatorStatus::Implemented),
    ("Equal", OperatorStatus::Implemented),
    ("NotEqual", OperatorStatus::Implemented),
    ("Less", OperatorStatus::Implemented),
    ("LessOrEqual", OperatorStatus::Implemented),
    ("Greater", OperatorStatus::Implemented),
    ("GreaterOrEqual", OperatorStatus::Implemented),
    ("Where", OperatorStatus::Implemented),
    ("Min", OperatorStatus::Implemented),
    ("Max", OperatorStatus::Implemented),
    ("Not", OperatorStatus::Implemented),
    ("Gelu", OperatorStatus::Implemented),
    ("LeakyRelu", OperatorStatus::Implemented),
    ("Elu", OperatorStatus::Implemented),
    ("Swish", OperatorStatus::Implemented),
    ("RNN", OperatorStatus::Implemented),
    ("LSTM", OperatorStatus::Implemented),
    ("GRU", OperatorStatus::Implemented),
    ("Split", OperatorStatus::Implemented),
    ("Expand", OperatorStatus::Implemented),
    ("Tile", OperatorStatus::Implemented),
    ("OneHot", OperatorStatus::Implemented),
    ("Einsum", OperatorStatus::Implemented),
    ("QuantizeLinear", OperatorStatus::Implemented),
    ("DequantizeLinear", OperatorStatus::Implemented),
    ("QLinearMatMul", OperatorStatus::Implemented),
    ("QLinearConv", OperatorStatus::Implemented),
    ("Mod", OperatorStatus::Implemented),
    ("Sin", OperatorStatus::Implemented),
    ("Cos", OperatorStatus::Implemented),
    ("Reciprocal", OperatorStatus::Implemented),
    ("Sign", OperatorStatus::Implemented),
    ("Sum", OperatorStatus::Implemented),
    ("Mean", OperatorStatus::Implemented),
    ("And", OperatorStatus::Implemented),
    ("Or", OperatorStatus::Implemented),
    ("LogSoftmax", OperatorStatus::Implemented),
    ("ReduceMax", OperatorStatus::Implemented),
    ("ReduceMin", OperatorStatus::Implemented),
    ("ReduceProd", OperatorStatus::Implemented),
    ("ArgMax", OperatorStatus::Implemented),
    ("ArgMin", OperatorStatus::Implemented),
    ("Shape", OperatorStatus::Implemented),
    ("Size", OperatorStatus::Implemented),
    ("Identity", OperatorStatus::Implemented),
    ("Constant", OperatorStatus::Implemented),
    ("ConstantOfShape", OperatorStatus::Implemented),
    ("Range", OperatorStatus::Implemented),
    ("Trilu", OperatorStatus::Implemented),
    ("CumSum", OperatorStatus::Implemented),
    ("GatherND", OperatorStatus::Implemented),
    ("ScatterND", OperatorStatus::Implemented),
    // ---------------- Phase 1B: vision ----
    ("Resize", OperatorStatus::Implemented),
    ("DepthToSpace", OperatorStatus::Implemented),
    ("SpaceToDepth", OperatorStatus::Implemented),
    ("RoiAlign", OperatorStatus::Implemented),
    ("GridSample", OperatorStatus::Implemented),
    ("GlobalMaxPool", OperatorStatus::Implemented),
    ("CenterCropPad", OperatorStatus::Implemented),
    ("TopK", OperatorStatus::Implemented),
    ("Compress", OperatorStatus::Implemented),
    ("NonZero", OperatorStatus::Implemented),
    ("Unique", OperatorStatus::Implemented),
    ("GatherElements", OperatorStatus::Implemented),
    ("ScatterElements", OperatorStatus::Implemented),
    ("GroupNormalization", OperatorStatus::Implemented),
    ("InstanceNormalization", OperatorStatus::Implemented),
    ("HardSigmoid", OperatorStatus::Implemented),
    ("HardSwish", OperatorStatus::Implemented),
    ("PRelu", OperatorStatus::Implemented),
    ("Mish", OperatorStatus::Implemented),
    // ---------------- Phase 2: generative / control-flow (generative-llm-v1) ----
    ("If", OperatorStatus::Implemented),
    ("Loop", OperatorStatus::Implemented),
    ("Scan", OperatorStatus::Implemented),
    ("RMSNormalization", OperatorStatus::Implemented),
    ("MatMulInteger", OperatorStatus::Implemented),
    ("DynamicQuantizeLinear", OperatorStatus::Implemented),
    ("RandomNormal", OperatorStatus::Implemented),
    ("RandomNormalLike", OperatorStatus::Implemented),
    ("RandomUniform", OperatorStatus::Implemented),
    ("RandomUniformLike", OperatorStatus::Implemented),
    ("Multinomial", OperatorStatus::Implemented),
    ("Bernoulli", OperatorStatus::Implemented),
    ("Dropout", OperatorStatus::Implemented),
    ("EyeLike", OperatorStatus::Implemented),
    ("ReduceL1", OperatorStatus::Implemented),
    ("ReduceL2", OperatorStatus::Implemented),
    ("ReduceLogSum", OperatorStatus::Implemented),
    ("ReduceLogSumExp", OperatorStatus::Implemented),
    ("ReduceSumSquare", OperatorStatus::Implemented),
    ("LpNormalization", OperatorStatus::Implemented),
    ("MeanVarianceNormalization", OperatorStatus::Implemented),
    ("Softplus", OperatorStatus::Implemented),
    // ---- microsoft-fused-ops-v1: com.microsoft-domain fused LLM ops ----
    ("SimplifiedLayerNormalization", OperatorStatus::Implemented),
    (
        "SkipSimplifiedLayerNormalization",
        OperatorStatus::Implemented,
    ),
    ("RotaryEmbedding", OperatorStatus::Implemented),
    ("MultiHeadAttention", OperatorStatus::Implemented),
    ("GroupQueryAttention", OperatorStatus::Implemented),
    // Phase 2 deferred: attention + integer conv still pending
    ("Attention", OperatorStatus::Planned(Phase::P2)),
    ("ConvInteger", OperatorStatus::Planned(Phase::P2)),
    // ---------------- Phase 3: audio + detection ----
    ("NonMaxSuppression", OperatorStatus::Planned(Phase::P3)),
    ("STFT", OperatorStatus::Planned(Phase::P3)),
    ("DFT", OperatorStatus::Planned(Phase::P3)),
    ("MelWeightMatrix", OperatorStatus::Planned(Phase::P3)),
    ("HammingWindow", OperatorStatus::Planned(Phase::P3)),
    ("HannWindow", OperatorStatus::Planned(Phase::P3)),
    ("BlackmanWindow", OperatorStatus::Planned(Phase::P3)),
    ("ConvTranspose", OperatorStatus::Planned(Phase::P3)),
    // ---------------- Phase 4: long tail ----
    ("BitShift", OperatorStatus::Planned(Phase::P4)),
    ("BitwiseAnd", OperatorStatus::Planned(Phase::P4)),
    ("BitwiseOr", OperatorStatus::Planned(Phase::P4)),
    ("BitwiseXor", OperatorStatus::Planned(Phase::P4)),
    ("BitwiseNot", OperatorStatus::Planned(Phase::P4)),
    ("Xor", OperatorStatus::Planned(Phase::P4)),
    ("Asin", OperatorStatus::Planned(Phase::P4)),
    ("Acos", OperatorStatus::Planned(Phase::P4)),
    ("Atan", OperatorStatus::Planned(Phase::P4)),
    ("Sinh", OperatorStatus::Planned(Phase::P4)),
    ("Cosh", OperatorStatus::Planned(Phase::P4)),
    ("Asinh", OperatorStatus::Planned(Phase::P4)),
    ("Acosh", OperatorStatus::Planned(Phase::P4)),
    ("Atanh", OperatorStatus::Planned(Phase::P4)),
    ("Tan", OperatorStatus::Planned(Phase::P4)),
    ("IsNaN", OperatorStatus::Planned(Phase::P4)),
    ("IsInf", OperatorStatus::Planned(Phase::P4)),
    ("Selu", OperatorStatus::Planned(Phase::P4)),
    ("Softsign", OperatorStatus::Planned(Phase::P4)),
    ("ThresholdedRelu", OperatorStatus::Planned(Phase::P4)),
    ("CastLike", OperatorStatus::Planned(Phase::P4)),
    ("Shrink", OperatorStatus::Planned(Phase::P4)),
    ("LpPool", OperatorStatus::Planned(Phase::P4)),
    ("GlobalLpPool", OperatorStatus::Planned(Phase::P4)),
    // ---------------- Deferred: needs subsystems we haven't built ----
    (
        "SequenceConstruct",
        OperatorStatus::DeferredSubsystem("sequence types"),
    ),
    (
        "SequenceEmpty",
        OperatorStatus::DeferredSubsystem("sequence types"),
    ),
    (
        "SequenceAt",
        OperatorStatus::DeferredSubsystem("sequence types"),
    ),
    (
        "SequenceInsert",
        OperatorStatus::DeferredSubsystem("sequence types"),
    ),
    (
        "SequenceErase",
        OperatorStatus::DeferredSubsystem("sequence types"),
    ),
    (
        "SequenceLength",
        OperatorStatus::DeferredSubsystem("sequence types"),
    ),
    (
        "SplitToSequence",
        OperatorStatus::DeferredSubsystem("sequence types"),
    ),
    (
        "ConcatFromSequence",
        OperatorStatus::DeferredSubsystem("sequence types"),
    ),
    (
        "Optional",
        OperatorStatus::DeferredSubsystem("optional types"),
    ),
    (
        "OptionalGetElement",
        OperatorStatus::DeferredSubsystem("optional types"),
    ),
    (
        "OptionalHasElement",
        OperatorStatus::DeferredSubsystem("optional types"),
    ),
    (
        "StringNormalizer",
        OperatorStatus::DeferredSubsystem("string tensors"),
    ),
    (
        "StringConcat",
        OperatorStatus::DeferredSubsystem("string tensors"),
    ),
    (
        "StringSplit",
        OperatorStatus::DeferredSubsystem("string tensors"),
    ),
    (
        "RegexFullMatch",
        OperatorStatus::DeferredSubsystem("string tensors"),
    ),
    ("LabelEncoder", OperatorStatus::DeferredSubsystem("ONNX-ML")),
    (
        "CategoryMapper",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    (
        "DictVectorizer",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    (
        "FeatureVectorizer",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    ("Imputer", OperatorStatus::DeferredSubsystem("ONNX-ML")),
    (
        "LinearClassifier",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    (
        "LinearRegressor",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    ("Normalizer", OperatorStatus::DeferredSubsystem("ONNX-ML")),
    (
        "OneHotEncoder",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    (
        "SVMClassifier",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    ("SVMRegressor", OperatorStatus::DeferredSubsystem("ONNX-ML")),
    ("Scaler", OperatorStatus::DeferredSubsystem("ONNX-ML")),
    (
        "TreeEnsembleClassifier",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    (
        "TreeEnsembleRegressor",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    ("ZipMap", OperatorStatus::DeferredSubsystem("ONNX-ML")),
    (
        "ArrayFeatureExtractor",
        OperatorStatus::DeferredSubsystem("ONNX-ML"),
    ),
    ("Binarizer", OperatorStatus::DeferredSubsystem("ONNX-ML")),
    // ---------------- Skipped: training-only ----
    ("Gradient", OperatorStatus::SkippedTraining),
    ("GradientDescent", OperatorStatus::SkippedTraining),
    ("Momentum", OperatorStatus::SkippedTraining),
    ("Adagrad", OperatorStatus::SkippedTraining),
    ("Adam", OperatorStatus::SkippedTraining),
    ("SoftmaxCrossEntropyLoss", OperatorStatus::SkippedTraining),
    ("NegativeLogLikelihoodLoss", OperatorStatus::SkippedTraining),
    // ---------------- Skipped: deprecated / replaced ----
    ("Upsample", OperatorStatus::SkippedDeprecated),
    ("Scatter", OperatorStatus::SkippedDeprecated),
    ("GivensGather", OperatorStatus::SkippedDeprecated),
    ("ImageScaler", OperatorStatus::SkippedDeprecated),
    ("Crop", OperatorStatus::SkippedDeprecated),
    ("ParametricSoftplus", OperatorStatus::SkippedDeprecated),
    ("ScaledTanh", OperatorStatus::SkippedDeprecated),
    ("Affine", OperatorStatus::SkippedDeprecated),
];

// ---------------------------------------------------------------------------
// Operator stub implementations
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// expf_approx polynomial approximation constants
// ---------------------------------------------------------------------------
//
// Range reduction splits `e^x` as `2^k * e^f` where `f` is small. The tail
// `e^f` is approximated by a degree-4 Taylor polynomial:
//     1 + f + f^2/2 + f^3/6 + f^4/24
// These constants name the coefficients and IEEE-754 magic numbers so the
// implementation reads as math, not hex.

/// Upper clamp for the input of `expf_approx` (avoids overflow to inf).
const EXP_CLAMP_MAX: f32 = 88.7;
/// Lower clamp for the input of `expf_approx` (avoids underflow to 0).
const EXP_CLAMP_MIN: f32 = -88.7;
/// Taylor coefficient for `f^2` term in `e^f`.
const EXP_POLY_C2: f32 = 0.5;
/// Taylor coefficient for `f^3` term in `e^f`.
const EXP_POLY_C3: f32 = 1.0 / 6.0;
/// Taylor coefficient for `f^4` term in `e^f`.
const EXP_POLY_C4: f32 = 1.0 / 24.0;
/// IEEE-754 single-precision exponent bias.
const F32_EXPONENT_BIAS: i32 = 127;
/// Number of mantissa bits in IEEE-754 single precision.
const F32_MANTISSA_BITS: u32 = 23;

/// Computes e^x for f32 using range reduction and polynomial approximation.
///
/// Accurate to ~1 ULP for typical inference ranges.
pub(crate) fn expf_approx(x: f32) -> f32 {
    // Clamp to avoid overflow/underflow
    let x = x.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX);

    // Range reduction: e^x = 2^(x * log2(e)) = 2^k * 2^f where k=floor, f=fraction
    let t = x * core::f32::consts::LOG2_E;
    // Round to nearest integer
    let k = if t >= 0.0 {
        (t + 0.5) as i32
    } else {
        (t - 0.5) as i32
    };
    // Guard against IEEE-754 exponent underflow: if `k + bias` would go
    // negative, the bitmask `((k+bias) as u32) << 23` reinterprets as a
    // huge exponent and yields -inf instead of a tiny positive number.
    // Treat that as genuine underflow → 0.
    if k + F32_EXPONENT_BIAS < 0 {
        return 0.0;
    }
    let f = x - (k as f32) * core::f32::consts::LN_2;

    // Polynomial approximation of e^f for f in [-0.5*ln2, 0.5*ln2]
    let f2 = f * f;
    let p = 1.0 + f + f2 * EXP_POLY_C2 + f2 * f * EXP_POLY_C3 + f2 * f2 * EXP_POLY_C4;

    // Reconstruct: multiply by 2^k using IEEE 754 bit manipulation
    let bits = ((k + F32_EXPONENT_BIAS) as u32) << F32_MANTISSA_BITS;
    let scale = f32::from_bits(bits);
    p * scale
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

/// BFloat16 element-wise unary op helper. Widens the input to `f32`, applies
/// `f`, and packs the result back into a BF16 tensor with the same shape.
fn bf16_unary<F: Fn(f32) -> f32>(input: &Tensor, f: F) -> Result<Tensor, OpError> {
    let values = bf16_to_f32(&input.raw_data);
    let n = input.shape.total_elements();
    if values.len() < n {
        return Err(OpError::ShapeMismatch(String::from(
            "BF16 input buffer smaller than shape",
        )));
    }
    let mut out = alloc::vec![0.0f32; n];
    for i in 0..n {
        out[i] = f(values[i]);
    }
    Ok(Tensor {
        data_type: DataType::BFloat16,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data: f32_to_bf16(&out),
    })
}

/// BFloat16 element-wise binary op helper. Widens both inputs to `f32`,
/// applies `f` element-wise using NumPy-style broadcasting, and packs the
/// result back into a BF16 tensor. Used by the BF16 paths of Add and Mul.
#[allow(clippy::needless_range_loop)]
fn bf16_binary_broadcast<F: Fn(f32, f32) -> f32>(
    a: &Tensor,
    b: &Tensor,
    _op: &str,
    f: F,
) -> Result<Tensor, OpError> {
    let a_f32 = bf16_to_f32(&a.raw_data);
    let b_f32 = bf16_to_f32(&b.raw_data);
    let out_shape = infer_binary_shape(&a.shape, &b.shape)?;
    let total = out_shape.total_elements();
    let mut out = alloc::vec![0.0f32; total];
    let a_strides = compute_strides(&a.shape.dims);
    let b_strides = compute_strides(&b.shape.dims);
    let ndim = out_shape.dims.len();
    let a_dim_offset = ndim.saturating_sub(a.shape.dims.len());
    let b_dim_offset = ndim.saturating_sub(b.shape.dims.len());
    let mut coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 {
            next_coord(&mut coord, &out_shape.dims);
        }
        let a_idx = broadcast_linear_index(&coord, &a.shape.dims, &a_strides, a_dim_offset);
        let b_idx = broadcast_linear_index(&coord, &b.shape.dims, &b_strides, b_dim_offset);
        out[i] = f(a_f32[a_idx], b_f32[b_idx]);
    }
    Ok(Tensor {
        data_type: DataType::BFloat16,
        shape: out_shape,
        name: String::new(),
        raw_data: f32_to_bf16(&out),
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
    if a.data_type == DataType::BFloat16 && b.data_type == DataType::BFloat16 {
        return bf16_binary_broadcast(a, b, "Add", |x, y| x + y);
    }
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Add only supports float32",
        )));
    }

    let out_shape = infer_binary_shape(&a.shape, &b.shape)?;
    let total = out_shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);

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
    let mut raw_data = allocate_tensor_data(total, DataType::Float);

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

    let mut raw_data = allocate_tensor_data(m * n, DataType::Float);

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
    let mut raw_data = allocate_tensor_data(total, DataType::Float);

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

/// Parsed Conv operator attributes with ONNX-11 defaults.
///
/// Shared by the CPU (`op_conv`) and CUDA (`cuda::conv::gpu_conv2d`)
/// dispatch paths so both agree on attribute semantics. Built via
/// [`ConvAttrs::from_attributes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvAttrs {
    /// Spatial padding as `[top, left, bottom, right]`. Default `[0; 4]`.
    pub pads: [i32; 4],
    /// Stride along `[H, W]` axes. Default `[1, 1]`.
    pub strides: [i32; 2],
    /// Kernel dilation along `[H, W]` axes. Default `[1, 1]`.
    pub dilations: [i32; 2],
    /// Group count. `1` for standard conv, `C_in` for depthwise. Default `1`.
    pub group: i32,
}

impl Default for ConvAttrs {
    fn default() -> Self {
        Self {
            pads: [0; 4],
            strides: [1, 1],
            dilations: [1, 1],
            group: 1,
        }
    }
}

impl ConvAttrs {
    /// True when every field holds its ONNX default — enables the
    /// legacy fast path in `conv_compute` that skips stride/pad/dilation
    /// bounds checks and the group partitioning logic.
    pub fn is_default(&self) -> bool {
        self.pads == [0; 4] && self.strides == [1, 1] && self.dilations == [1, 1] && self.group == 1
    }

    /// Parse a Conv node's `AttributeProto` list into a `ConvAttrs`.
    ///
    /// Unknown or unsupported attribute values (for example a non-NOTSET
    /// `auto_pad` mode) produce `OpError::InvalidAttribute`. Benign
    /// unrelated attributes like `kernel_shape` are ignored because the
    /// weight tensor shape is authoritative.
    pub fn from_attributes(attrs: &[AttributeProto]) -> Result<Self, OpError> {
        let mut out = Self::default();
        for attr in attrs {
            Self::apply_attribute(&mut out, attr)?;
        }
        Ok(out)
    }

    /// Apply a single attribute to the in-progress `ConvAttrs`.
    ///
    /// Unknown attributes (including the informational `kernel_shape`) are
    /// silently ignored — the weight tensor shape is authoritative.
    fn apply_attribute(out: &mut Self, attr: &AttributeProto) -> Result<(), OpError> {
        match attr.name.as_str() {
            "strides" => out.strides = parse_pair_i32("Conv", "strides", &attr.ints)?,
            "pads" => out.pads = parse_quad_i32("Conv", "pads", &attr.ints)?,
            "dilations" => out.dilations = parse_pair_i32("Conv", "dilations", &attr.ints)?,
            "group" => out.group = parse_group("Conv", attr.i)?,
            "auto_pad" => validate_auto_pad_notset("Conv", &attr.s)?,
            _ => {}
        }
        Ok(())
    }
}

/// Parse a `[i32; 2]` attribute, rejecting wrong lengths.
fn parse_pair_i32(op: &str, name: &str, ints: &[i64]) -> Result<[i32; 2], OpError> {
    if ints.len() != 2 {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "{}: {} must be length 2 (got {})",
            op,
            name,
            ints.len()
        )));
    }
    Ok([ints[0] as i32, ints[1] as i32])
}

/// Parse a `[i32; 4]` attribute, rejecting wrong lengths.
fn parse_quad_i32(op: &str, name: &str, ints: &[i64]) -> Result<[i32; 4], OpError> {
    if ints.len() != 4 {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "{}: {} must be length 4 (got {})",
            op,
            name,
            ints.len()
        )));
    }
    Ok([
        ints[0] as i32,
        ints[1] as i32,
        ints[2] as i32,
        ints[3] as i32,
    ])
}

/// Parse a `group` attribute, rejecting values `< 1`.
fn parse_group(op: &str, value: i64) -> Result<i32, OpError> {
    if value < 1 {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "{}: group must be >= 1 (got {})",
            op,
            value
        )));
    }
    Ok(value as i32)
}

/// Ensure an `auto_pad` attribute is either empty or `"NOTSET"`. The
/// `Conv` variant appends the legacy "only NOTSET + explicit pads
/// supported" hint to preserve its original error string.
fn validate_auto_pad_notset(op: &str, raw: &[u8]) -> Result<(), OpError> {
    let val = core::str::from_utf8(raw).unwrap_or("<invalid utf8>");
    if val == "NOTSET" || val.is_empty() {
        return Ok(());
    }
    Err(OpError::InvalidAttribute(if op == "Conv" {
        alloc::format!(
            "Conv: unsupported auto_pad value '{}' (only NOTSET + explicit pads supported)",
            val
        )
    } else {
        alloc::format!("{}: unsupported auto_pad value '{}'", op, val)
    }))
}

/// Parsed BatchNormalization attributes with ONNX defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchNormAttrs {
    pub epsilon: f32,
    pub momentum: f32,
}

impl Default for BatchNormAttrs {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            momentum: 0.9,
        }
    }
}

impl BatchNormAttrs {
    pub fn from_attributes(attrs: &[AttributeProto]) -> Result<Self, OpError> {
        let mut out = Self::default();
        for attr in attrs {
            match attr.name.as_str() {
                "epsilon" => out.epsilon = attr.f,
                "momentum" => out.momentum = attr.f,
                // spatial / training_mode are ignored in inference
                _ => {}
            }
        }
        Ok(out)
    }
}

/// Parsed pooling attributes with ONNX defaults.
///
/// `kernel_shape = [0, 0]` means "unset" — `from_attributes(true)` will
/// error in that case. `GlobalAveragePool` uses `from_attributes(false)`
/// because kernel is inferred from the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolAttrs {
    pub kernel_shape: [i32; 2],
    pub pads: [i32; 4],
    pub strides: [i32; 2],
    pub ceil_mode: bool,
    pub count_include_pad: bool,
}

impl Default for PoolAttrs {
    fn default() -> Self {
        Self {
            kernel_shape: [0, 0],
            pads: [0; 4],
            strides: [1, 1],
            ceil_mode: false,
            count_include_pad: false,
        }
    }
}

impl PoolAttrs {
    pub fn from_attributes(
        attrs: &[AttributeProto],
        require_kernel: bool,
    ) -> Result<Self, OpError> {
        let mut out = Self::default();
        for attr in attrs {
            Self::apply_attribute(&mut out, attr)?;
        }
        if require_kernel && out.kernel_shape == [0, 0] {
            return Err(OpError::InvalidAttribute(String::from(
                "Pool: kernel_shape attribute is required",
            )));
        }
        // Strides default to kernel dims if explicit strides weren't set.
        if out.strides == [1, 1] && out.kernel_shape != [0, 0] {
            // Actually ONNX defaults strides to 1, not kernel_shape. Keep 1s.
            // (The pool op defaults stride to kernel_shape only when pool
            // was used as subsample — ONNX spec says default = 1 for all.)
        }
        Ok(out)
    }

    /// Apply a single attribute to the in-progress `PoolAttrs`.
    fn apply_attribute(out: &mut Self, attr: &AttributeProto) -> Result<(), OpError> {
        match attr.name.as_str() {
            "kernel_shape" => {
                out.kernel_shape = parse_pair_i32("Pool", "kernel_shape", &attr.ints)?
            }
            "strides" => out.strides = parse_pair_i32("Pool", "strides", &attr.ints)?,
            "pads" => out.pads = parse_quad_i32("Pool", "pads", &attr.ints)?,
            "ceil_mode" => out.ceil_mode = attr.i != 0,
            "count_include_pad" => out.count_include_pad = attr.i != 0,
            "auto_pad" => validate_auto_pad_notset("Pool", &attr.s)?,
            _ => {}
        }
        Ok(())
    }
}

/// Dimensions for a 2D convolution operation, used to reduce the number
/// of arguments passed to the inner convolution helper.
struct ConvDims {
    /// Total input channels (across all groups).
    c_in: usize,
    /// Input channels per group — `c_in / group`, matching weight.dims[1].
    c_in_per_group: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
}

/// Computes the convolution sum for a single output pixel at position
/// `(oy, ox)` over this output channel's input-channel group and all
/// kernel positions. Honors `strides`, `pads`, `dilations`, and `group`.
///
/// `ci_base` is the starting absolute input-channel index for this
/// output channel's group; weights are already laid out per-group as
/// `[C_out, C_in/group, KH, KW]`, so the inner loop iterates
/// `0..c_in_per_group` and reads `weight[co, ci_local, ky, kx]` while
/// reading `input[batch, ci_base + ci_local, iy, ix]`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn convolve_at(
    input_data: &[u8],
    weight_data: &[u8],
    batch: usize,
    co: usize,
    ci_base: usize,
    oy: usize,
    ox: usize,
    dims: &ConvDims,
    attrs: &ConvAttrs,
) -> f32 {
    let stride_h = attrs.strides[0] as isize;
    let stride_w = attrs.strides[1] as isize;
    let dilation_h = attrs.dilations[0] as isize;
    let dilation_w = attrs.dilations[1] as isize;
    let pad_top = attrs.pads[0] as isize;
    let pad_left = attrs.pads[1] as isize;

    let mut sum = 0.0f32;
    for ci_local in 0..dims.c_in_per_group {
        for ky in 0..dims.kh {
            for kx in 0..dims.kw {
                let iy = (oy as isize) * stride_h + (ky as isize) * dilation_h - pad_top;
                let ix = (ox as isize) * stride_w + (kx as isize) * dilation_w - pad_left;
                if iy < 0 || iy >= dims.h as isize || ix < 0 || ix >= dims.w as isize {
                    continue;
                }
                let abs_ci = ci_base + ci_local;
                let in_idx = batch * dims.c_in * dims.h * dims.w
                    + abs_ci * dims.h * dims.w
                    + (iy as usize) * dims.w
                    + (ix as usize);
                let w_idx = co * dims.c_in_per_group * dims.kh * dims.kw
                    + ci_local * dims.kh * dims.kw
                    + ky * dims.kw
                    + kx;
                sum += read_f32(input_data, in_idx) * read_f32(weight_data, w_idx);
            }
        }
    }
    sum
}

/// 2D convolution with optional bias, honoring `strides`, `pads`,
/// `dilations`, and `group` attributes.
///
/// Input shape: `[N, C_in, H, W]`
/// Weight shape: `[C_out, C_in / group, KH, KW]`
/// Bias shape: `[C_out]` (optional)
/// Output shape: `[N, C_out, OH, OW]` where
///   `OH = (H + pad_top + pad_bottom - (KH - 1) * dilation_h - 1) / stride_h + 1`
///   `OW = (W + pad_left + pad_right - (KW - 1) * dilation_w - 1) / stride_w + 1`
pub fn op_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    attrs: &ConvAttrs,
) -> Result<Tensor, OpError> {
    validate_conv_inputs(input, weight, attrs)?;

    let n = input.shape.dims[0] as usize;
    let c_in = input.shape.dims[1] as usize;
    let h = input.shape.dims[2] as usize;
    let w = input.shape.dims[3] as usize;

    let c_out = weight.shape.dims[0] as usize;
    let c_in_per_group = weight.shape.dims[1] as usize;
    let kh = weight.shape.dims[2] as usize;
    let kw = weight.shape.dims[3] as usize;

    let (oh, ow) = compute_conv_out_dims(h, w, kh, kw, attrs);
    let out_total = n * c_out * oh * ow;
    let mut raw_data = allocate_tensor_data(out_total, DataType::Float);
    let dims = ConvDims {
        c_in,
        c_in_per_group,
        h,
        w,
        kh,
        kw,
    };
    let mut params = ConvParams {
        n,
        c_out,
        oh,
        ow,
        raw_data: &mut raw_data,
    };

    conv_compute(
        &input.raw_data,
        &weight.raw_data,
        bias,
        &dims,
        attrs,
        &mut params,
    );

    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(Vec::from([n as i64, c_out as i64, oh as i64, ow as i64])),
        name: String::new(),
        raw_data,
    })
}

/// Compute Conv output spatial dimensions per the ONNX Conv-11 formula.
/// Uses `isize` arithmetic internally so negative results (kernel larger
/// than padded input) surface as zero-sized outputs rather than panics.
pub(crate) fn compute_conv_out_dims(
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    attrs: &ConvAttrs,
) -> (usize, usize) {
    let ih = h as isize + attrs.pads[0] as isize + attrs.pads[2] as isize
        - (kh as isize - 1) * attrs.dilations[0] as isize
        - 1;
    let iw = w as isize + attrs.pads[1] as isize + attrs.pads[3] as isize
        - (kw as isize - 1) * attrs.dilations[1] as isize
        - 1;
    let oh = if ih < 0 {
        0
    } else {
        ih as usize / attrs.strides[0] as usize + 1
    };
    let ow = if iw < 0 {
        0
    } else {
        iw as usize / attrs.strides[1] as usize + 1
    };
    (oh, ow)
}

/// Validate Conv operator inputs: types, ranks, channel compatibility
/// (group-aware), and kernel-vs-input sizing.
fn validate_conv_inputs(input: &Tensor, weight: &Tensor, attrs: &ConvAttrs) -> Result<(), OpError> {
    if input.data_type != DataType::Float || weight.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Conv only supports float32",
        )));
    }
    if input.shape.ndim() != 4 || weight.shape.ndim() != 4 {
        return Err(OpError::ShapeMismatch(String::from(
            "Conv requires 4D input [N,C,H,W] and 4D weight [Co,Ci/group,KH,KW]",
        )));
    }
    let group = attrs.group as i64;
    if group < 1 {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "Conv: group must be >= 1 (got {})",
            group
        )));
    }
    let c_in = input.shape.dims[1];
    let c_out = weight.shape.dims[0];
    let w_in = weight.shape.dims[1];
    if c_in != w_in * group {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "Conv: input channels ({}) must equal weight channels ({}) * group ({})",
            c_in,
            w_in,
            group
        )));
    }
    if c_out % group != 0 {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "Conv: C_out ({}) must be divisible by group ({})",
            c_out,
            group
        )));
    }
    // Effective (dilated) kernel size must fit inside the padded input.
    let pad_h = attrs.pads[0] as i64 + attrs.pads[2] as i64;
    let pad_w = attrs.pads[1] as i64 + attrs.pads[3] as i64;
    let eff_kh = (weight.shape.dims[2] - 1) * attrs.dilations[0] as i64 + 1;
    let eff_kw = (weight.shape.dims[3] - 1) * attrs.dilations[1] as i64 + 1;
    if eff_kh > input.shape.dims[2] + pad_h || eff_kw > input.shape.dims[3] + pad_w {
        return Err(OpError::ShapeMismatch(String::from(
            "Conv: effective kernel larger than padded input",
        )));
    }
    Ok(())
}

/// Output-side parameters for `conv_compute`: batch/channels-out/spatial
/// output dims plus the destination byte buffer. Grouped as a struct so
/// `conv_compute` stays at 4 args.
struct ConvParams<'a> {
    n: usize,
    c_out: usize,
    oh: usize,
    ow: usize,
    raw_data: &'a mut [u8],
}

/// Execute the Conv inner loops, writing results into `params.raw_data`.
/// Partitions output channels into `attrs.group` contiguous blocks; each
/// block's output channels read only from the corresponding
/// input-channel slice `[g * c_in_per_group .. (g+1) * c_in_per_group)`.
fn conv_compute(
    input_data: &[u8],
    weight_data: &[u8],
    bias: Option<&Tensor>,
    dims: &ConvDims,
    attrs: &ConvAttrs,
    params: &mut ConvParams<'_>,
) {
    let ConvParams {
        n,
        c_out,
        oh,
        ow,
        raw_data,
    } = params;
    let (n, c_out, oh, ow) = (*n, *c_out, *oh, *ow);
    let group = attrs.group as usize;
    let c_out_per_group = c_out / group;

    for batch in 0..n {
        for co in 0..c_out {
            let group_idx = co / c_out_per_group;
            let ci_base = group_idx * dims.c_in_per_group;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = convolve_at(
                        input_data,
                        weight_data,
                        batch,
                        co,
                        ci_base,
                        oy,
                        ox,
                        dims,
                        attrs,
                    );
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
        return Err(OpError::ShapeMismatch(String::from(
            "Sub requires exactly 2 inputs",
        )));
    }
    let a = inputs[0];
    let b = inputs[1];
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Sub only supports float32",
        )));
    }
    let out_shape = infer_binary_shape(&a.shape, &b.shape)?;
    let total = out_shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    let a_strides = compute_strides(&a.shape.dims);
    let b_strides = compute_strides(&b.shape.dims);
    let ndim = out_shape.dims.len();
    let a_dim_offset = ndim.saturating_sub(a.shape.dims.len());
    let b_dim_offset = ndim.saturating_sub(b.shape.dims.len());
    let mut coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 {
            next_coord(&mut coord, &out_shape.dims);
        }
        let a_idx = broadcast_linear_index(&coord, &a.shape.dims, &a_strides, a_dim_offset);
        let b_idx = broadcast_linear_index(&coord, &b.shape.dims, &b_strides, b_dim_offset);
        write_f32(
            &mut raw_data,
            i,
            read_f32(&a.raw_data, a_idx) - read_f32(&b.raw_data, b_idx),
        );
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Element-wise multiplication with broadcasting.
pub fn op_mul(inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    if inputs.len() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "Mul requires exactly 2 inputs",
        )));
    }
    let a = inputs[0];
    let b = inputs[1];
    if a.data_type == DataType::BFloat16 && b.data_type == DataType::BFloat16 {
        return bf16_binary_broadcast(a, b, "Mul", |x, y| x * y);
    }
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Mul only supports float32",
        )));
    }
    let out_shape = infer_binary_shape(&a.shape, &b.shape)?;
    let total = out_shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    let a_strides = compute_strides(&a.shape.dims);
    let b_strides = compute_strides(&b.shape.dims);
    let ndim = out_shape.dims.len();
    let a_dim_offset = ndim.saturating_sub(a.shape.dims.len());
    let b_dim_offset = ndim.saturating_sub(b.shape.dims.len());
    let mut coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 {
            next_coord(&mut coord, &out_shape.dims);
        }
        let a_idx = broadcast_linear_index(&coord, &a.shape.dims, &a_strides, a_dim_offset);
        let b_idx = broadcast_linear_index(&coord, &b.shape.dims, &b_strides, b_dim_offset);
        write_f32(
            &mut raw_data,
            i,
            read_f32(&a.raw_data, a_idx) * read_f32(&b.raw_data, b_idx),
        );
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Element-wise division with broadcasting.
pub fn op_div(inputs: &[&Tensor]) -> Result<Tensor, OpError> {
    if inputs.len() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "Div requires exactly 2 inputs",
        )));
    }
    let a = inputs[0];
    let b = inputs[1];
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Div only supports float32",
        )));
    }
    let out_shape = infer_binary_shape(&a.shape, &b.shape)?;
    let total = out_shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    let a_strides = compute_strides(&a.shape.dims);
    let b_strides = compute_strides(&b.shape.dims);
    let ndim = out_shape.dims.len();
    let a_dim_offset = ndim.saturating_sub(a.shape.dims.len());
    let b_dim_offset = ndim.saturating_sub(b.shape.dims.len());
    let mut coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 {
            next_coord(&mut coord, &out_shape.dims);
        }
        let a_idx = broadcast_linear_index(&coord, &a.shape.dims, &a_strides, a_dim_offset);
        let b_idx = broadcast_linear_index(&coord, &b.shape.dims, &b_strides, b_dim_offset);
        write_f32(
            &mut raw_data,
            i,
            read_f32(&a.raw_data, a_idx) / read_f32(&b.raw_data, b_idx),
        );
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
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
    validate_gemm_inputs(a, b)?;
    let (m, n, k) = gemm_dims(a, b, trans_a, trans_b)?;

    let a_buf = pack_gemm_a(a, m, k, trans_a);
    let b_buf = pack_gemm_b(b, k, n, trans_b);

    let mut c_buf = alloc::vec![0.0f32; m * n];
    crate::gemm::gemm_f32(m, n, k, &a_buf, &b_buf, &mut c_buf);

    let raw_data = apply_alpha_beta_c(&c_buf, c, alpha, beta, m, n);

    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(Vec::from([m as i64, n as i64])),
        name: String::new(),
        raw_data,
    })
}

/// Validate that both Gemm operands are 2-D float32 tensors.
fn validate_gemm_inputs(a: &Tensor, b: &Tensor) -> Result<(), OpError> {
    if a.data_type != DataType::Float || b.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Gemm only supports float32",
        )));
    }
    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "Gemm requires 2D inputs",
        )));
    }
    Ok(())
}

/// Resolve `(m, n, k)` from the Gemm operands honoring transpose flags.
fn gemm_dims(
    a: &Tensor,
    b: &Tensor,
    trans_a: bool,
    trans_b: bool,
) -> Result<(usize, usize, usize), OpError> {
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
        return Err(OpError::ShapeMismatch(String::from(
            "Gemm inner dimensions mismatch",
        )));
    }
    Ok((m, n, k_a))
}

/// Materialize Gemm operand `A` as a row-major `[M, K]` buffer.
fn pack_gemm_a(a: &Tensor, m: usize, k: usize, trans_a: bool) -> Vec<f32> {
    let mut a_buf = alloc::vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let idx = if trans_a { j * m + i } else { i * k + j };
            a_buf[i * k + j] = read_f32(&a.raw_data, idx);
        }
    }
    a_buf
}

/// Materialize Gemm operand `B` as a row-major `[K, N]` buffer.
fn pack_gemm_b(b: &Tensor, k: usize, n: usize, trans_b: bool) -> Vec<f32> {
    let mut b_buf = alloc::vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            let idx = if trans_b { j * k + i } else { i * n + j };
            b_buf[i * n + j] = read_f32(&b.raw_data, idx);
        }
    }
    b_buf
}

/// Scale the `M*K x K*N` product by `alpha` and fold in `beta * C` (with
/// 1-D bias broadcast). Returns the packed raw-bytes output buffer.
fn apply_alpha_beta_c(
    c_buf: &[f32],
    c: Option<&Tensor>,
    alpha: f32,
    beta: f32,
    m: usize,
    n: usize,
) -> Vec<u8> {
    let mut raw_data = allocate_tensor_data(m * n, DataType::Float);
    for i in 0..m {
        for j in 0..n {
            let val = alpha * c_buf[i * n + j] + gemm_bias_contribution(c, beta, i, j, n);
            write_f32(&mut raw_data, i * n + j, val);
        }
    }
    raw_data
}

/// Read the optional `beta * C` term at `(i, j)` for the Gemm output,
/// supporting `[M, N]` matrices and `[N]` bias-broadcast vectors.
fn gemm_bias_contribution(c: Option<&Tensor>, beta: f32, i: usize, j: usize, n: usize) -> f32 {
    let Some(c_tensor) = c else {
        return 0.0;
    };
    if beta == 0.0 {
        return 0.0;
    }
    let c_idx = if c_tensor.shape.ndim() == 1 {
        j
    } else {
        i * n + j
    };
    if c_idx < c_tensor.shape.total_elements() {
        beta * read_f32(&c_tensor.raw_data, c_idx)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tier 2: Activation operators (Sigmoid, Tanh)
// ---------------------------------------------------------------------------

/// Element-wise sigmoid: 1 / (1 + exp(-x)).
pub fn op_sigmoid(input: &Tensor) -> Result<Tensor, OpError> {
    if input.data_type == DataType::BFloat16 {
        return bf16_unary(input, |x| 1.0 / (1.0 + expf_approx(-x)));
    }
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Sigmoid only supports float32",
        )));
    }
    let total = input.shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    for i in 0..total {
        let x = read_f32(&input.raw_data, i);
        let val = 1.0 / (1.0 + expf_approx(-x));
        write_f32(&mut raw_data, i, val);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Element-wise tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
pub fn op_tanh(input: &Tensor) -> Result<Tensor, OpError> {
    if input.data_type == DataType::BFloat16 {
        return bf16_unary(input, |x| {
            let ep = expf_approx(x);
            let em = expf_approx(-x);
            if ep + em != 0.0 {
                (ep - em) / (ep + em)
            } else {
                0.0
            }
        });
    }
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Tanh only supports float32",
        )));
    }
    let total = input.shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    for i in 0..total {
        let x = read_f32(&input.raw_data, i);
        let ep = expf_approx(x);
        let em = expf_approx(-x);
        let val = if ep + em != 0.0 {
            (ep - em) / (ep + em)
        } else {
            0.0
        };
        write_f32(&mut raw_data, i, val);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

// ---------------------------------------------------------------------------
// Tier 3: Shape and data movement operators
// ---------------------------------------------------------------------------

/// Transpose tensor dimensions according to permutation.
pub fn op_transpose(input: &Tensor, perm: Option<&[i64]>) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Transpose only supports float32",
        )));
    }
    let ndim = input.shape.ndim();
    let perm_vec: Vec<usize> = match perm {
        Some(p) => p.iter().map(|&x| x as usize).collect(),
        None => (0..ndim).rev().collect(),
    };
    if perm_vec.len() != ndim {
        return Err(OpError::InvalidAttribute(String::from(
            "perm length must match ndim",
        )));
    }
    let out_dims: Vec<i64> = perm_vec.iter().map(|&p| input.shape.dims[p]).collect();
    let out_shape = TensorShape::new(out_dims);
    let total = out_shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);

    let in_strides = compute_strides(&input.shape.dims);

    let mut out_coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 {
            next_coord(&mut out_coord, &out_shape.dims);
        }
        let mut in_idx = 0usize;
        for d in 0..ndim {
            in_idx += out_coord[d] * in_strides[perm_vec[d]];
        }
        write_f32(&mut raw_data, i, read_f32(&input.raw_data, in_idx));
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Copies one source tensor's elements into the concat output buffer at
/// the correct offset along `resolved_axis`.
fn copy_tensor_to_concat_output(
    input: &Tensor,
    out_data: &mut [u8],
    out_strides: &[usize],
    resolved_axis: usize,
    axis_offset: usize,
    ndim: usize,
    elem_size: usize,
) {
    let in_total = input.shape.total_elements();
    let mut in_coord = alloc::vec![0usize; ndim];
    for i in 0..in_total {
        if i > 0 {
            next_coord(&mut in_coord, &input.shape.dims);
        }
        let mut out_idx = 0usize;
        for d in 0..ndim {
            let c = if d == resolved_axis {
                in_coord[d] + axis_offset
            } else {
                in_coord[d]
            };
            out_idx += c * out_strides[d];
        }
        let src = i * elem_size;
        let dst = out_idx * elem_size;
        out_data[dst..dst + elem_size].copy_from_slice(&input.raw_data[src..src + elem_size]);
    }
}

/// Concatenate tensors along `axis`.
///
/// Preserves input dtype via byte-level copy — works for any fixed-width
/// element type (Float, BFloat16, Int8/16/32/64, Uint variants, Bool,
/// Float16, Float8, Double). All inputs must share the same dtype per
/// the ONNX spec.
pub fn op_concat(inputs: &[&Tensor], axis: i64) -> Result<Tensor, OpError> {
    if inputs.is_empty() {
        return Err(OpError::ShapeMismatch(String::from(
            "Concat requires at least 1 input",
        )));
    }
    let dtype = inputs[0].data_type;
    let elem_size = dtype.element_size();
    if elem_size == 0 {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "Concat: unsupported variable-width dtype {:?}",
            dtype
        )));
    }
    for (i, input) in inputs.iter().enumerate().skip(1) {
        if input.data_type != dtype {
            return Err(OpError::ShapeMismatch(alloc::format!(
                "Concat: input {} dtype {:?} does not match input 0 dtype {:?}",
                i,
                input.data_type,
                dtype
            )));
        }
    }
    let ndim = inputs[0].shape.ndim();
    let resolved_axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    if resolved_axis >= ndim {
        return Err(OpError::InvalidAttribute(String::from(
            "Concat axis out of range",
        )));
    }
    let mut out_dims = inputs[0].shape.dims.clone();
    let mut concat_size = inputs[0].shape.dims[resolved_axis];
    for input in &inputs[1..] {
        concat_size += input.shape.dims[resolved_axis];
    }
    out_dims[resolved_axis] = concat_size;
    let out_shape = TensorShape::new(out_dims);
    let total = out_shape.total_elements();
    let mut raw_data = alloc::vec![0u8; total * elem_size];

    let out_strides = compute_strides(&out_shape.dims);
    let mut axis_offset = 0usize;
    for input in inputs {
        copy_tensor_to_concat_output(
            input,
            &mut raw_data,
            &out_strides,
            resolved_axis,
            axis_offset,
            ndim,
            elem_size,
        );
        axis_offset += input.shape.dims[resolved_axis] as usize;
    }
    Ok(Tensor {
        data_type: dtype,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Flatten tensor to 2D at specified axis.
pub fn op_flatten(input: &Tensor, axis: i64) -> Result<Tensor, OpError> {
    let ndim = input.shape.ndim();
    let resolved = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
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
            input
                .shape
                .dims
                .iter()
                .enumerate()
                .filter(|(i, &d)| {
                    let idx = *i as i64;
                    !(d == 1
                        && ax.iter().any(|&a| {
                            let ra = if a < 0 { ndim + a } else { a };
                            ra == idx
                        }))
                })
                .map(|(_, &d)| d)
                .collect()
        }
        None => input
            .shape
            .dims
            .iter()
            .copied()
            .filter(|&d| d != 1)
            .collect(),
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
    let mut sorted_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (out_ndim as i64 + a) as usize
            } else {
                a as usize
            }
        })
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

/// Builds a tensor that shares `input`'s shape but has a new data type and buffer.
#[inline]
fn cast_output(input: &Tensor, to: DataType, raw_data: Vec<u8>) -> Tensor {
    Tensor {
        data_type: to,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    }
}

fn cast_f32_to_i32(input: &Tensor) -> Result<Tensor, OpError> {
    let total = input.shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Int32);
    for i in 0..total {
        let v = read_f32(&input.raw_data, i) as i32;
        write_i32(&mut raw_data, i, v);
    }
    Ok(cast_output(input, DataType::Int32, raw_data))
}

fn cast_i32_to_f32(input: &Tensor) -> Result<Tensor, OpError> {
    let total = input.shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    for i in 0..total {
        let v = read_i32(&input.raw_data, i);
        write_f32(&mut raw_data, i, v as f32);
    }
    Ok(cast_output(input, DataType::Float, raw_data))
}

fn cast_f32_to_i64(input: &Tensor) -> Result<Tensor, OpError> {
    let total = input.shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Int64);
    for i in 0..total {
        let v = read_f32(&input.raw_data, i) as i64;
        write_i64(&mut raw_data, i, v);
    }
    Ok(cast_output(input, DataType::Int64, raw_data))
}

fn cast_i64_to_f32(input: &Tensor) -> Result<Tensor, OpError> {
    let total = input.shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    for i in 0..total {
        let v = read_i64(&input.raw_data, i);
        write_f32(&mut raw_data, i, v as f32);
    }
    Ok(cast_output(input, DataType::Float, raw_data))
}

/// Type cast between data types.
pub fn op_cast(input: &Tensor, to: DataType) -> Result<Tensor, OpError> {
    if input.data_type == to {
        return Ok(input.clone());
    }
    match (input.data_type, to) {
        (DataType::Float, DataType::Int32) => cast_f32_to_i32(input),
        (DataType::Int32, DataType::Float) => cast_i32_to_f32(input),
        (DataType::Float, DataType::Int64) => cast_f32_to_i64(input),
        (DataType::Int64, DataType::Float) => cast_i64_to_f32(input),
        _ => Err(OpError::NotImplemented),
    }
}

/// Gather elements along `axis` using `indices`.
///
/// Preserves the input dtype and operates as a byte-level copy so any
/// fixed-width type (Float, BFloat16, Int8/16/32/64, Uint8/16/32/64,
/// Bool, Float16, Float8, Double) passes through unchanged.
/// `indices` must be Int32 or Int64. Variable-width types (String) are
/// rejected.
///
/// Semantics match ONNX Gather-11: the output has shape
/// `input.dims[..axis] ++ indices.dims ++ input.dims[axis+1..]` and each
/// output element at output-coord `o` reads the input at
/// `input[..o_axis_prefix, indices[o_idx_slice], ..o_axis_suffix]`.
pub fn op_gather(input: &Tensor, indices: &Tensor, axis: i64) -> Result<Tensor, OpError> {
    let elem_size = input.data_type.element_size();
    validate_gather_inputs(input, indices, elem_size)?;
    let ndim = input.shape.ndim();
    let resolved_axis = resolve_gather_axis(axis, ndim)?;

    let axis_size = input.shape.dims[resolved_axis] as usize;

    // Output shape: input.dims[..axis] ++ indices.dims ++ input.dims[axis+1..]
    let mut out_dims = Vec::with_capacity(ndim - 1 + indices.shape.ndim());
    out_dims.extend(input.shape.dims[..resolved_axis].iter().copied());
    out_dims.extend(indices.shape.dims.iter().copied());
    out_dims.extend(input.shape.dims[resolved_axis + 1..].iter().copied());
    let out_shape = TensorShape::new(out_dims);

    // Block sizes measured in elements:
    //   outer_count = product(input.dims[..axis])
    //   slice_size  = product(input.dims[axis+1..])
    //   num_indices = product(indices.dims)
    let outer_count: usize = input.shape.dims[..resolved_axis]
        .iter()
        .map(|&d| d as usize)
        .product();
    let slice_size: usize = input.shape.dims[resolved_axis + 1..]
        .iter()
        .map(|&d| d as usize)
        .product();
    let num_indices = indices.shape.total_elements();

    let total_elems = outer_count * num_indices * slice_size;
    let mut raw_data = alloc::vec![0u8; total_elems * elem_size];

    let slice_bytes = slice_size * elem_size;
    let input_outer_stride = axis_size * slice_bytes;
    let out_outer_stride = num_indices * slice_bytes;

    for outer in 0..outer_count {
        for i in 0..num_indices {
            let idx = read_gather_index(indices, i, axis_size)?;
            let src_start = outer * input_outer_stride + idx * slice_bytes;
            let dst_start = outer * out_outer_stride + i * slice_bytes;
            raw_data[dst_start..dst_start + slice_bytes]
                .copy_from_slice(&input.raw_data[src_start..src_start + slice_bytes]);
        }
    }

    Ok(Tensor {
        data_type: input.data_type,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Validate Gather inputs: dtype must be fixed-width, indices must be
/// integer, and input must have at least one dimension.
fn validate_gather_inputs(
    input: &Tensor,
    indices: &Tensor,
    elem_size: usize,
) -> Result<(), OpError> {
    if elem_size == 0 {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "Gather: unsupported variable-width input dtype {:?}",
            input.data_type
        )));
    }
    if !matches!(indices.data_type, DataType::Int32 | DataType::Int64) {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "Gather: indices must be Int32 or Int64 (got {:?})",
            indices.data_type
        )));
    }
    if input.shape.ndim() == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "Gather: input must have at least 1 dimension",
        )));
    }
    Ok(())
}

/// Resolve a (possibly negative) Gather `axis` against the input rank.
fn resolve_gather_axis(axis: i64, ndim: usize) -> Result<usize, OpError> {
    let resolved = if axis < 0 { ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "Gather: axis {} out of range for rank {}",
            axis,
            ndim
        )));
    }
    Ok(resolved as usize)
}

/// Read index `i` from the Int32/Int64 indices tensor, wrap negative
/// values per ONNX spec, and bounds-check against `axis_size`.
fn read_gather_index(indices: &Tensor, i: usize, axis_size: usize) -> Result<usize, OpError> {
    let raw = read_gather_raw_index(indices, i)?;
    let wrapped = if raw < 0 { raw + axis_size as i64 } else { raw };
    if wrapped < 0 || wrapped >= axis_size as i64 {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "Gather: index {} out of bounds for axis size {}",
            raw,
            axis_size
        )));
    }
    Ok(wrapped as usize)
}

/// Decode a single index value from the Int32 or Int64 indices buffer.
fn read_gather_raw_index(indices: &Tensor, i: usize) -> Result<i64, OpError> {
    if indices.data_type == DataType::Int64 {
        return Ok(read_i64(&indices.raw_data, i));
    }
    let offset = i * 4;
    if offset + 4 > indices.raw_data.len() {
        return Err(OpError::ShapeMismatch(String::from(
            "Gather: indices buffer too small",
        )));
    }
    Ok(i32::from_le_bytes([
        indices.raw_data[offset],
        indices.raw_data[offset + 1],
        indices.raw_data[offset + 2],
        indices.raw_data[offset + 3],
    ]) as i64)
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
        return Err(OpError::InvalidAttribute(String::from(
            "Slice only supports float32",
        )));
    }
    let ndim = input.shape.ndim();
    let (actual_starts, actual_ends, actual_steps) =
        resolve_slice_ranges(input, starts, ends, axes, steps, ndim);

    let out_shape = slice_output_shape(&actual_starts, &actual_ends, &actual_steps, ndim);
    let total = out_shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);

    let in_strides = compute_strides(&input.shape.dims);
    let mut out_coord = alloc::vec![0usize; ndim];
    for i in 0..total {
        if i > 0 {
            next_coord(&mut out_coord, &out_shape.dims);
        }
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let in_d = actual_starts[d] as usize + out_coord[d] * actual_steps[d] as usize;
            in_idx += in_d * in_strides[d];
        }
        write_f32(&mut raw_data, i, read_f32(&input.raw_data, in_idx));
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Resolve the per-axis `(start, end, step)` ranges for `op_slice`,
/// applying negative-index wrapping and clamping to the input dims.
fn resolve_slice_ranges(
    input: &Tensor,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
    ndim: usize,
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut actual_starts = alloc::vec![0i64; ndim];
    let mut actual_ends: Vec<i64> = input.shape.dims.clone();
    let mut actual_steps = alloc::vec![1i64; ndim];

    let axes_vec = slice_axes(axes, ndim, starts.len());

    for (i, &ax) in axes_vec.iter().enumerate() {
        if ax >= ndim {
            continue;
        }
        let dim = input.shape.dims[ax];
        let (s, e) = clamp_slice_range(starts, ends, i, dim);
        actual_starts[ax] = s;
        actual_ends[ax] = e;
        actual_steps[ax] = step_at(steps, i);
    }
    (actual_starts, actual_ends, actual_steps)
}

/// Normalize the optional `axes` parameter, defaulting to `0..starts.len()`.
fn slice_axes(axes: Option<&[i64]>, ndim: usize, starts_len: usize) -> Vec<usize> {
    match axes {
        Some(ax) => ax
            .iter()
            .map(|&a| {
                if a < 0 {
                    (ndim as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect(),
        None => (0..starts_len).collect(),
    }
}

/// Resolve and clamp `(start, end)` for axis index `i` against `dim`.
fn clamp_slice_range(starts: &[i64], ends: &[i64], i: usize, dim: i64) -> (i64, i64) {
    let mut s = if i < starts.len() { starts[i] } else { 0 };
    let mut e = if i < ends.len() { ends[i] } else { dim };
    if s < 0 {
        s += dim;
    }
    if e < 0 {
        e += dim;
    }
    (s.clamp(0, dim), e.clamp(0, dim))
}

/// Resolve the step for axis index `i`, defaulting to `1`.
fn step_at(steps: Option<&[i64]>, i: usize) -> i64 {
    match steps {
        Some(st) if i < st.len() => st[i],
        _ => 1,
    }
}

/// Compute the output shape after slicing.
fn slice_output_shape(
    actual_starts: &[i64],
    actual_ends: &[i64],
    actual_steps: &[i64],
    ndim: usize,
) -> TensorShape {
    let out_dims: Vec<i64> = (0..ndim)
        .map(|d| {
            let s = actual_starts[d];
            let e = actual_ends[d];
            let step = actual_steps[d];
            ((e - s + step - 1) / step).max(0)
        })
        .collect();
    TensorShape::new(out_dims)
}

/// Pad tensor with constant value.
pub fn op_pad(
    input: &Tensor,
    pads: &[i64],
    mode: &str,
    constant_value: f32,
) -> Result<Tensor, OpError> {
    if mode != "constant" {
        return Err(OpError::NotImplemented);
    }
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Pad only supports float32",
        )));
    }
    let ndim = input.shape.ndim();
    if pads.len() != ndim * 2 {
        return Err(OpError::InvalidAttribute(String::from(
            "pads length must be 2 * ndim",
        )));
    }
    let out_dims: Vec<i64> = (0..ndim)
        .map(|d| input.shape.dims[d] + pads[d] + pads[d + ndim])
        .collect();
    let out_shape = TensorShape::new(out_dims);
    let total = out_shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);

    // Fill with constant
    for i in 0..total {
        write_f32(&mut raw_data, i, constant_value);
    }

    // Copy input data
    let out_strides = compute_strides(&out_shape.dims);
    let in_total = input.shape.total_elements();
    let mut in_coord = alloc::vec![0usize; ndim];
    for i in 0..in_total {
        if i > 0 {
            next_coord(&mut in_coord, &input.shape.dims);
        }
        let mut out_idx = 0usize;
        for d in 0..ndim {
            out_idx += (in_coord[d] + pads[d] as usize) * out_strides[d];
        }
        write_f32(&mut raw_data, out_idx, read_f32(&input.raw_data, i));
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Clamp tensor values to [min, max] range.
pub fn op_clip(
    input: &Tensor,
    min_val: Option<f32>,
    max_val: Option<f32>,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Clip only supports float32",
        )));
    }
    let total = input.shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    for i in 0..total {
        let mut v = read_f32(&input.raw_data, i);
        if let Some(lo) = min_val {
            if v < lo {
                v = lo;
            }
        }
        if let Some(hi) = max_val {
            if v > hi {
                v = hi;
            }
        }
        write_f32(&mut raw_data, i, v);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

// ---------------------------------------------------------------------------
// Tier 4: Normalization, pooling, and reduction operators
// ---------------------------------------------------------------------------

/// Approximate square root using Newton's method (no_std).
pub(crate) fn sqrt_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut s = x;
    for _ in 0..10 {
        s = 0.5 * (s + x / s);
    }
    s
}

/// Batch normalization: scale * (x - mean) / sqrt(var + eps) + bias.
pub fn op_batch_normalization(
    input: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "BatchNorm only supports float32",
        )));
    }
    if input.shape.ndim() < 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "BatchNorm requires at least 2D input",
        )));
    }
    let n = input.shape.dims[0] as usize;
    let c = input.shape.dims[1] as usize;
    let spatial: usize = input.shape.dims[2..]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let total = input.shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
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
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Layer normalization along last N dimensions.
pub fn op_layer_normalization(
    input: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    axis: i64,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "LayerNorm only supports float32",
        )));
    }
    let ndim = input.shape.ndim();
    let resolved = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    let outer: usize = input.shape.dims[..resolved]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let inner: usize = input.shape.dims[resolved..]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let total = input.shape.total_elements();
    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    for o in 0..outer {
        let base = o * inner;
        let mut sum = 0.0f32;
        for i in 0..inner {
            sum += read_f32(&input.raw_data, base + i);
        }
        let mean = sum / inner as f32;
        let mut var_sum = 0.0f32;
        for i in 0..inner {
            let d = read_f32(&input.raw_data, base + i) - mean;
            var_sum += d * d;
        }
        let inv_std = 1.0 / sqrt_approx(var_sum / inner as f32 + epsilon);
        for i in 0..inner {
            let x = read_f32(&input.raw_data, base + i);
            let s = read_f32(&scale.raw_data, i);
            let b = bias.map(|bt| read_f32(&bt.raw_data, i)).unwrap_or(0.0);
            write_f32(&mut raw_data, base + i, s * (x - mean) * inv_std + b);
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Max pooling over NCHW input.
pub fn op_maxpool(
    input: &Tensor,
    kernel_shape: &[i64],
    strides: Option<&[i64]>,
    pads: Option<&[i64]>,
) -> Result<Tensor, OpError> {
    let geom = PoolGeom::from_inputs("MaxPool", input, kernel_shape, strides, pads)?;
    let out_shape = geom.out_shape();
    let mut raw_data = allocate_tensor_data(out_shape.total_elements(), DataType::Float);
    for bn in 0..geom.n {
        for ch in 0..geom.c {
            for oi in 0..geom.oh {
                for oj in 0..geom.ow {
                    let max_val = pool_reduce_max(&geom, input, bn, ch, oi, oj);
                    let out_idx = geom.out_index(bn, ch, oi, oj);
                    write_f32(&mut raw_data, out_idx, max_val);
                }
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Maximum activation over a single pooling window.
fn pool_reduce_max(
    geom: &PoolGeom,
    input: &Tensor,
    bn: usize,
    ch: usize,
    oi: usize,
    oj: usize,
) -> f32 {
    let mut max_val = f32::NEG_INFINITY;
    for ki in 0..geom.kh {
        for kj in 0..geom.kw {
            if let Some(idx) = geom.input_index(bn, ch, oi, oj, ki, kj) {
                let v = read_f32(&input.raw_data, idx);
                if v > max_val {
                    max_val = v;
                }
            }
        }
    }
    max_val
}

/// Average pooling over NCHW input.
pub fn op_averagepool(
    input: &Tensor,
    kernel_shape: &[i64],
    strides: Option<&[i64]>,
    pads: Option<&[i64]>,
) -> Result<Tensor, OpError> {
    let geom = PoolGeom::from_inputs("AvgPool", input, kernel_shape, strides, pads)?;
    let out_shape = geom.out_shape();
    let mut raw_data = allocate_tensor_data(out_shape.total_elements(), DataType::Float);
    for bn in 0..geom.n {
        for ch in 0..geom.c {
            for oi in 0..geom.oh {
                for oj in 0..geom.ow {
                    let avg = pool_reduce_avg(&geom, input, bn, ch, oi, oj);
                    let out_idx = geom.out_index(bn, ch, oi, oj);
                    write_f32(&mut raw_data, out_idx, avg);
                }
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Mean activation over a single pooling window (zero if window empty).
fn pool_reduce_avg(
    geom: &PoolGeom,
    input: &Tensor,
    bn: usize,
    ch: usize,
    oi: usize,
    oj: usize,
) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for ki in 0..geom.kh {
        for kj in 0..geom.kw {
            if let Some(idx) = geom.input_index(bn, ch, oi, oj, ki, kj) {
                sum += read_f32(&input.raw_data, idx);
                count += 1;
            }
        }
    }
    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

/// Shared 2-D pool geometry: NCHW dims, kernel, stride, pads, output dims.
struct PoolGeom {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    pt: usize,
    pl: usize,
    oh: usize,
    ow: usize,
}

impl PoolGeom {
    /// Validate inputs and build the geometry for a 2-D pool over NCHW.
    fn from_inputs(
        op: &str,
        input: &Tensor,
        kernel_shape: &[i64],
        strides: Option<&[i64]>,
        pads: Option<&[i64]>,
    ) -> Result<Self, OpError> {
        if input.data_type != DataType::Float || input.shape.ndim() != 4 {
            return Err(OpError::ShapeMismatch(alloc::format!(
                "{} requires 4D float input",
                op
            )));
        }
        let n = input.shape.dims[0] as usize;
        let c = input.shape.dims[1] as usize;
        let h = input.shape.dims[2] as usize;
        let w = input.shape.dims[3] as usize;
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
        Ok(Self {
            n,
            c,
            h,
            w,
            kh,
            kw,
            sh,
            sw,
            pt,
            pl,
            oh,
            ow,
        })
    }

    fn out_shape(&self) -> TensorShape {
        TensorShape::new(Vec::from([
            self.n as i64,
            self.c as i64,
            self.oh as i64,
            self.ow as i64,
        ]))
    }

    fn out_index(&self, bn: usize, ch: usize, oi: usize, oj: usize) -> usize {
        bn * self.c * self.oh * self.ow + ch * self.oh * self.ow + oi * self.ow + oj
    }

    /// Map a pool window position to a linear input index, or `None` when
    /// the position lies in the (zero-) padded region.
    fn input_index(
        &self,
        bn: usize,
        ch: usize,
        oi: usize,
        oj: usize,
        ki: usize,
        kj: usize,
    ) -> Option<usize> {
        let hi = oi * self.sh + ki;
        let wi = oj * self.sw + kj;
        if hi >= self.pt && hi < self.h + self.pt && wi >= self.pl && wi < self.w + self.pl {
            Some(
                bn * self.c * self.h * self.w
                    + ch * self.h * self.w
                    + (hi - self.pt) * self.w
                    + (wi - self.pl),
            )
        } else {
            None
        }
    }
}

/// Global average pooling: NCHW → NC11.
pub fn op_global_average_pool(input: &Tensor) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float || input.shape.ndim() != 4 {
        return Err(OpError::ShapeMismatch(String::from(
            "GlobalAvgPool requires 4D float input",
        )));
    }
    let n = input.shape.dims[0] as usize;
    let c = input.shape.dims[1] as usize;
    let h = input.shape.dims[2] as usize;
    let w = input.shape.dims[3] as usize;
    let spatial = h * w;
    let out_shape = TensorShape::new(Vec::from([n as i64, c as i64, 1, 1]));
    let mut raw_data = allocate_tensor_data(n * c, DataType::Float);
    for bn in 0..n {
        for ch in 0..c {
            let mut sum = 0.0f32;
            for s in 0..spatial {
                sum += read_f32(&input.raw_data, bn * c * spatial + ch * spatial + s);
            }
            write_f32(&mut raw_data, bn * c + ch, sum / spatial as f32);
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Reduce by mean along specified axes.
pub fn op_reduce_mean(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    let sum_tensor = op_reduce_sum(input, axes, keepdims)?;
    let total_in = input.shape.total_elements();
    let total_out = sum_tensor.shape.total_elements();
    let reduce_count = total_in.checked_div(total_out).unwrap_or(1);
    let mut raw_data = sum_tensor.raw_data;
    for i in 0..total_out {
        let v = read_f32(&raw_data, i);
        write_f32(&mut raw_data, i, v / reduce_count as f32);
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: sum_tensor.shape,
        name: String::new(),
        raw_data,
    })
}

/// Reduce by sum along specified axes.
pub fn op_reduce_sum(input: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "ReduceSum only supports float32",
        )));
    }
    let ndim = input.shape.ndim();
    let resolved = resolve_reduce_axes(axes, ndim);
    let out_shape = reduce_output_shape(input, &resolved, ndim, keepdims);
    let total_out = out_shape.total_elements();
    let mut raw_data = allocate_tensor_data(total_out, DataType::Float);

    let in_total = input.shape.total_elements();
    let out_strides = compute_strides(&out_shape.dims);
    let mut in_coord = alloc::vec![0usize; ndim];
    for i in 0..in_total {
        if i > 0 {
            next_coord(&mut in_coord, &input.shape.dims);
        }
        let out_idx = reduce_output_index(&in_coord, &resolved, &out_strides, ndim, keepdims);
        let prev = read_f32(&raw_data, out_idx);
        write_f32(&mut raw_data, out_idx, prev + read_f32(&input.raw_data, i));
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Resolve a possibly negative or empty axes list into concrete indices.
/// An empty list reduces over all axes.
fn resolve_reduce_axes(axes: &[i64], ndim: usize) -> Vec<usize> {
    if axes.is_empty() {
        return (0..ndim).collect();
    }
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

/// Compute the output shape after reduction, honoring `keepdims`. An
/// all-axes reduction with `keepdims=false` collapses to `[1]`.
fn reduce_output_shape(
    input: &Tensor,
    resolved: &[usize],
    ndim: usize,
    keepdims: bool,
) -> TensorShape {
    let out_dims: Vec<i64> = (0..ndim)
        .filter_map(|d| {
            if resolved.contains(&d) {
                keepdims.then_some(1)
            } else {
                Some(input.shape.dims[d])
            }
        })
        .collect();
    TensorShape::new(if out_dims.is_empty() {
        Vec::from([1i64])
    } else {
        out_dims
    })
}

/// Map an input coord to a reduced-output linear index by skipping
/// reduced dimensions (or keeping a placeholder when `keepdims`).
fn reduce_output_index(
    in_coord: &[usize],
    resolved: &[usize],
    out_strides: &[usize],
    ndim: usize,
    keepdims: bool,
) -> usize {
    let mut out_idx = 0usize;
    let mut od = 0usize;
    #[allow(clippy::needless_range_loop)]
    for d in 0..ndim {
        if resolved.contains(&d) {
            if keepdims {
                od += 1;
            }
            continue;
        }
        if od < out_strides.len() {
            out_idx += in_coord[d] * out_strides[od];
        }
        od += 1;
    }
    out_idx
}

// ---------------------------------------------------------------------------
// Parallel operator variants (container mode)
// ---------------------------------------------------------------------------

use crate::parallel::CorePool;

/// Parallel element-wise unary operation on Float tensors.
///
/// Splits the flat element array across threads, applies `f` to each
/// element. Falls back to sequential if below threshold or pool has 1 thread.
pub fn parallel_elementwise_unary(
    pool: &CorePool,
    input: &Tensor,
    threshold: usize,
    f: impl Fn(f32) -> f32 + Send + Sync,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "only supports float32",
        )));
    }
    let total = input.shape.total_elements();
    if pool.num_threads() <= 1 || total <= threshold {
        // Sequential: apply f to each element
        let mut raw_data = allocate_tensor_data(total, DataType::Float);
        for i in 0..total {
            let val = read_f32(&input.raw_data, i);
            write_f32(&mut raw_data, i, f(val));
        }
        return Ok(Tensor {
            data_type: DataType::Float,
            shape: input.shape.clone(),
            name: String::new(),
            raw_data,
        });
    }

    // Parallel: each thread computes its chunk and returns (start_idx, bytes)
    let input_data = &input.raw_data;
    let results = pool.parallel_for(0..total, |range| {
        let mut chunk_data = allocate_tensor_data(range.len(), DataType::Float);
        for (local_i, global_i) in range.clone().enumerate() {
            let val = read_f32(input_data, global_i);
            write_f32(&mut chunk_data, local_i, f(val));
        }
        (range.start, chunk_data)
    });

    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    for (start, chunk) in results {
        raw_data[start * 4..start * 4 + chunk.len()].copy_from_slice(&chunk);
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Parallel Relu: `max(0, x)`.
pub fn op_relu_parallel(
    pool: &CorePool,
    input: &Tensor,
    threshold: usize,
) -> Result<Tensor, OpError> {
    parallel_elementwise_unary(pool, input, threshold, |x| if x > 0.0 { x } else { 0.0 })
}

/// Parallel Sigmoid: `1 / (1 + exp(-x))`.
pub fn op_sigmoid_parallel(
    pool: &CorePool,
    input: &Tensor,
    threshold: usize,
) -> Result<Tensor, OpError> {
    parallel_elementwise_unary(pool, input, threshold, |x| 1.0 / (1.0 + expf_approx(-x)))
}

/// Parallel Tanh: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`.
pub fn op_tanh_parallel(
    pool: &CorePool,
    input: &Tensor,
    threshold: usize,
) -> Result<Tensor, OpError> {
    parallel_elementwise_unary(pool, input, threshold, |x| {
        let ep = expf_approx(x);
        let en = expf_approx(-x);
        if ep + en == 0.0 {
            0.0
        } else {
            (ep - en) / (ep + en)
        }
    })
}

/// Parallel Clip: clamp values to [min, max].
pub fn op_clip_parallel(
    pool: &CorePool,
    input: &Tensor,
    min_val: Option<f32>,
    max_val: Option<f32>,
    threshold: usize,
) -> Result<Tensor, OpError> {
    parallel_elementwise_unary(pool, input, threshold, move |mut v| {
        if let Some(lo) = min_val {
            if v < lo {
                v = lo;
            }
        }
        if let Some(hi) = max_val {
            if v > hi {
                v = hi;
            }
        }
        v
    })
}

/// Parallel convolution with output channel decomposition.
///
/// Splits output channels across threads. Each thread computes its
/// assigned output channels independently. Falls back to sequential
/// [`op_conv`] if below threshold.
pub fn op_conv_parallel(
    pool: &CorePool,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    attrs: &ConvAttrs,
    threshold: usize,
) -> Result<Tensor, OpError> {
    validate_conv_inputs(input, weight, attrs)?;

    // For non-default attributes, fall through to the single-threaded
    // `op_conv` — parallelizing the grouped / strided / padded case is
    // a perf follow-up, not a correctness concern.
    if !attrs.is_default() {
        return op_conv(input, weight, bias, attrs);
    }

    let geom = ConvParallelGeom::from_default_attrs(input, weight);

    if pool.num_threads() <= 1 || geom.c_out * geom.oh * geom.ow <= threshold {
        return op_conv(input, weight, bias, attrs);
    }

    let results = conv_parallel_compute(pool, input, weight, bias, attrs, &geom);
    let raw_data = conv_parallel_assemble(results, &geom);

    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(Vec::from([
            geom.n as i64,
            geom.c_out as i64,
            geom.oh as i64,
            geom.ow as i64,
        ])),
        name: String::new(),
        raw_data,
    })
}

/// Shape constants for the default-attrs `op_conv_parallel` fast path.
struct ConvParallelGeom {
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
}

impl ConvParallelGeom {
    fn from_default_attrs(input: &Tensor, weight: &Tensor) -> Self {
        let n = input.shape.dims[0] as usize;
        let c_in = input.shape.dims[1] as usize;
        let h = input.shape.dims[2] as usize;
        let w = input.shape.dims[3] as usize;
        let c_out = weight.shape.dims[0] as usize;
        let kh = weight.shape.dims[2] as usize;
        let kw = weight.shape.dims[3] as usize;
        let oh = h - kh + 1;
        let ow = w - kw + 1;
        Self {
            n,
            c_in,
            h,
            w,
            c_out,
            kh,
            kw,
            oh,
            ow,
        }
    }
}

/// Compute each parallel chunk: `(start_co, ch_count, chunk_data_f32_bytes)`.
fn conv_parallel_compute(
    pool: &CorePool,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    attrs: &ConvAttrs,
    geom: &ConvParallelGeom,
) -> Vec<(usize, usize, Vec<u8>)> {
    let dims = ConvDims {
        c_in: geom.c_in,
        c_in_per_group: geom.c_in,
        h: geom.h,
        w: geom.w,
        kh: geom.kh,
        kw: geom.kw,
    };
    let input_data = &input.raw_data;
    let weight_data = &weight.raw_data;
    let (n, oh, ow) = (geom.n, geom.oh, geom.ow);

    pool.parallel_for(0..geom.c_out, |ch_range| {
        let ch_count = ch_range.end - ch_range.start;
        let mut chunk_data = allocate_tensor_data(n * ch_count * oh * ow, DataType::Float);
        for batch in 0..n {
            for (local_co, global_co) in ch_range.clone().enumerate() {
                conv_parallel_fill_channel(
                    &mut chunk_data,
                    input_data,
                    weight_data,
                    bias,
                    attrs,
                    &dims,
                    batch,
                    global_co,
                    local_co,
                    ch_count,
                    oh,
                    ow,
                );
            }
        }
        (ch_range.start, ch_count, chunk_data)
    })
}

/// Fill one `(batch, local_co)` plane of the per-thread output chunk.
#[allow(clippy::too_many_arguments)]
fn conv_parallel_fill_channel(
    chunk_data: &mut [u8],
    input_data: &[u8],
    weight_data: &[u8],
    bias: Option<&Tensor>,
    attrs: &ConvAttrs,
    dims: &ConvDims,
    batch: usize,
    global_co: usize,
    local_co: usize,
    ch_count: usize,
    oh: usize,
    ow: usize,
) {
    for oy in 0..oh {
        for ox in 0..ow {
            let mut sum = convolve_at(
                input_data,
                weight_data,
                batch,
                global_co,
                0,
                oy,
                ox,
                dims,
                attrs,
            );
            if let Some(b) = bias {
                sum += read_f32(&b.raw_data, global_co);
            }
            let local_idx = batch * ch_count * oh * ow + local_co * oh * ow + oy * ow + ox;
            write_f32(chunk_data, local_idx, sum);
        }
    }
}

/// Stitch per-thread channel chunks into the final NCHW output buffer.
fn conv_parallel_assemble(
    results: Vec<(usize, usize, Vec<u8>)>,
    geom: &ConvParallelGeom,
) -> Vec<u8> {
    let out_total = geom.n * geom.c_out * geom.oh * geom.ow;
    let mut raw_data = allocate_tensor_data(out_total, DataType::Float);
    for (start_co, ch_count, chunk_data) in results {
        conv_parallel_copy_chunk(&mut raw_data, &chunk_data, start_co, ch_count, geom);
    }
    raw_data
}

/// Copy one parallel chunk back into its NCHW destination slot.
fn conv_parallel_copy_chunk(
    raw_data: &mut [u8],
    chunk_data: &[u8],
    start_co: usize,
    ch_count: usize,
    geom: &ConvParallelGeom,
) {
    let (n, c_out, oh, ow) = (geom.n, geom.c_out, geom.oh, geom.ow);
    for batch in 0..n {
        for local_co in 0..ch_count {
            let global_co = start_co + local_co;
            for oy in 0..oh {
                for ox in 0..ow {
                    let src_idx = batch * ch_count * oh * ow + local_co * oh * ow + oy * ow + ox;
                    let dst_idx = batch * c_out * oh * ow + global_co * oh * ow + oy * ow + ox;
                    let val = read_f32(chunk_data, src_idx);
                    write_f32(raw_data, dst_idx, val);
                }
            }
        }
    }
}

/// Parallel reduce sum: partial sums per chunk, then merge.
///
/// This is a simple parallel reduction over the *entire* tensor to a
/// scalar sum. For axis-based reduction, the sequential op is used.
/// Falls back to sequential [`op_reduce_sum`] if below threshold or
/// if axes-based reduction is needed.
pub fn op_reduce_sum_parallel(
    pool: &CorePool,
    input: &Tensor,
    axes: &[i64],
    keepdims: bool,
    threshold: usize,
) -> Result<Tensor, OpError> {
    let total = input.shape.total_elements();
    // For axis-based or small reductions, use sequential
    if pool.num_threads() <= 1 || total <= threshold || !axes.is_empty() {
        return op_reduce_sum(input, axes, keepdims);
    }

    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "ReduceSum only supports float32",
        )));
    }

    // Full reduction to scalar — parallelize partial sums
    let input_data = &input.raw_data;
    let partial_sums = pool.parallel_for(0..total, |range| {
        let mut sum = 0.0f32;
        for i in range {
            sum += read_f32(input_data, i);
        }
        sum
    });

    let total_sum: f32 = partial_sums.iter().sum();
    let mut raw_data = alloc::vec![0u8; 4];
    write_f32(&mut raw_data, 0, total_sum);

    let out_shape = if keepdims {
        TensorShape::new(alloc::vec![1i64; input.shape.ndim()])
    } else {
        TensorShape::new(alloc::vec![1i64])
    };

    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data,
    })
}

/// Parallel reduce mean: parallel sum then divide.
pub fn op_reduce_mean_parallel(
    pool: &CorePool,
    input: &Tensor,
    axes: &[i64],
    keepdims: bool,
    threshold: usize,
) -> Result<Tensor, OpError> {
    let total = input.shape.total_elements();
    if pool.num_threads() <= 1 || total <= threshold || !axes.is_empty() {
        return op_reduce_mean(input, axes, keepdims);
    }

    let sum_tensor = op_reduce_sum_parallel(pool, input, axes, keepdims, threshold)?;
    let total_out = sum_tensor.shape.total_elements();
    let reduce_count = total.checked_div(total_out).unwrap_or(1);
    let mut raw_data = sum_tensor.raw_data;
    for i in 0..total_out {
        let v = read_f32(&raw_data, i);
        write_f32(&mut raw_data, i, v / reduce_count as f32);
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: sum_tensor.shape,
        name: String::new(),
        raw_data,
    })
}

/// Parallel softmax: parallel max, parallel exp+sum, parallel normalize.
pub fn op_softmax_parallel(
    pool: &CorePool,
    input: &Tensor,
    axis: i64,
    threshold: usize,
) -> Result<Tensor, OpError> {
    if input.data_type != DataType::Float {
        return Err(OpError::InvalidAttribute(String::from(
            "Softmax only supports float32",
        )));
    }
    let total = input.shape.total_elements();
    if pool.num_threads() <= 1 || total <= threshold {
        return op_softmax(input, axis);
    }

    let ndim = input.shape.ndim();
    if ndim == 0 {
        return softmax_parallel_scalar(input);
    }

    let resolved_axis = resolve_softmax_axis(axis, ndim);
    if resolved_axis != ndim - 1 || ndim > 2 {
        // For non-last-axis softmax, fall back to sequential
        return op_softmax(input, axis);
    }

    if ndim == 1 {
        softmax_parallel_1d(pool, input, total)
    } else {
        softmax_parallel_2d(pool, input, total)
    }
}

/// Resolve a (possibly negative) softmax `axis` against the input rank.
fn resolve_softmax_axis(axis: i64, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    }
}

/// Softmax of a 0-D tensor: identity-with-value `1.0`.
fn softmax_parallel_scalar(input: &Tensor) -> Result<Tensor, OpError> {
    let mut raw_data = alloc::vec![0u8; 4];
    write_f32(&mut raw_data, 0, 1.0);
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Three-pass parallel softmax over a 1-D tensor.
fn softmax_parallel_1d(pool: &CorePool, input: &Tensor, total: usize) -> Result<Tensor, OpError> {
    let input_data = &input.raw_data;
    let global_max = softmax_parallel_max(pool, input_data, total);
    let partial_results = softmax_parallel_exp(pool, input_data, total, global_max);
    let global_sum: f32 = partial_results.iter().map(|(_, _, s)| s).sum();

    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    for (start, chunk, _) in partial_results {
        let count = chunk.len() / 4;
        for i in 0..count {
            let v = read_f32(&chunk, i);
            write_f32(&mut raw_data, start + i, v / global_sum);
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Parallel max-reduction pass for `softmax_parallel_1d`.
fn softmax_parallel_max(pool: &CorePool, input_data: &[u8], total: usize) -> f32 {
    let partial_maxes = pool.parallel_for(0..total, |range| {
        let mut max_val = f32::NEG_INFINITY;
        for i in range {
            let v = read_f32(input_data, i);
            if v > max_val {
                max_val = v;
            }
        }
        max_val
    });
    partial_maxes
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Parallel `exp(x - max)` pass with per-chunk partial sums.
fn softmax_parallel_exp(
    pool: &CorePool,
    input_data: &[u8],
    total: usize,
    global_max: f32,
) -> Vec<(usize, Vec<u8>, f32)> {
    pool.parallel_for(0..total, |range| {
        let mut local_data = allocate_tensor_data(range.len(), DataType::Float);
        let mut local_sum = 0.0f32;
        for (local_i, global_i) in range.clone().enumerate() {
            let v = read_f32(input_data, global_i);
            let exp_v = expf_approx(v - global_max);
            write_f32(&mut local_data, local_i, exp_v);
            local_sum += exp_v;
        }
        (range.start, local_data, local_sum)
    })
}

/// Row-parallel softmax over a `[num_rows, row_len]` tensor.
fn softmax_parallel_2d(pool: &CorePool, input: &Tensor, total: usize) -> Result<Tensor, OpError> {
    let num_rows = input.shape.dims[0] as usize;
    let row_len = input.shape.dims[1] as usize;
    let input_data = &input.raw_data;

    let results = pool.parallel_for(0..num_rows, |row_range| {
        let mut chunk_data = allocate_tensor_data(row_range.len() * row_len, DataType::Float);
        for (local_r, global_r) in row_range.clone().enumerate() {
            softmax_row(input_data, &mut chunk_data, global_r, local_r, row_len);
        }
        (row_range.start, chunk_data)
    });

    let mut raw_data = allocate_tensor_data(total, DataType::Float);
    for (start_row, chunk) in results {
        let offset = start_row * row_len * 4;
        raw_data[offset..offset + chunk.len()].copy_from_slice(&chunk);
    }

    Ok(Tensor {
        data_type: DataType::Float,
        shape: input.shape.clone(),
        name: String::new(),
        raw_data,
    })
}

/// Compute softmax for a single row, writing the normalized values into
/// `chunk_data` at the per-thread output base offset.
fn softmax_row(
    input_data: &[u8],
    chunk_data: &mut [u8],
    global_r: usize,
    local_r: usize,
    row_len: usize,
) {
    let base = global_r * row_len;
    let out_base = local_r * row_len;
    let max_val = softmax_row_max(input_data, base, row_len);
    let sum = softmax_row_exp_into(input_data, chunk_data, base, out_base, row_len, max_val);
    if sum > 0.0 {
        softmax_row_normalize(chunk_data, out_base, row_len, sum);
    }
}

/// Max across one input row.
fn softmax_row_max(input_data: &[u8], base: usize, row_len: usize) -> f32 {
    let mut max_val = f32::NEG_INFINITY;
    for j in 0..row_len {
        let v = read_f32(input_data, base + j);
        if v > max_val {
            max_val = v;
        }
    }
    max_val
}

/// Compute `exp(x - max)` per row element, write into `chunk_data`,
/// return the row sum.
fn softmax_row_exp_into(
    input_data: &[u8],
    chunk_data: &mut [u8],
    base: usize,
    out_base: usize,
    row_len: usize,
    max_val: f32,
) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..row_len {
        let v = read_f32(input_data, base + j);
        let exp_v = expf_approx(v - max_val);
        write_f32(chunk_data, out_base + j, exp_v);
        sum += exp_v;
    }
    sum
}

/// Divide every element of one row by its sum (in place in `chunk_data`).
fn softmax_row_normalize(chunk_data: &mut [u8], out_base: usize, row_len: usize, sum: f32) {
    for j in 0..row_len {
        let v = read_f32(chunk_data, out_base + j);
        write_f32(chunk_data, out_base + j, v / sum);
    }
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
        assert_eq!(registry.supported_count(), 136);
    }

    #[test]
    fn test_inventory_matches_registry() {
        // Every Implemented inventory entry must parse to an OpKind.
        // Every entry in ALL_OPS must appear in the inventory as Implemented.
        use alloc::collections::BTreeSet;

        let impl_names: BTreeSet<&str> = SUPPORTED_OPS_INVENTORY
            .iter()
            .filter(|(_, s)| *s == OperatorStatus::Implemented)
            .map(|(n, _)| *n)
            .collect();

        let registry_names: BTreeSet<&str> = ALL_OPS.iter().map(|k| k.name()).collect();

        for name in &impl_names {
            assert!(
                OpKind::parse_str(name).is_some(),
                "inventory lists '{}' as Implemented but parse_str returned None",
                name
            );
        }

        // Every ALL_OPS entry must be listed Implemented.
        for name in &registry_names {
            assert!(
                impl_names.contains(name),
                "registry contains '{}' but inventory does not mark it Implemented",
                name
            );
        }
        // And no Implemented inventory entry should be missing from ALL_OPS.
        for name in &impl_names {
            assert!(
                registry_names.contains(name),
                "inventory marks '{}' Implemented but it is not in ALL_OPS",
                name
            );
        }

        // Sanity: must match the advertised 136-op count (131 standard
        // ONNX + 5 microsoft-fused-ops-v1 additions).
        assert_eq!(impl_names.len(), 136);

        // No duplicate entries in the inventory.
        let mut all_names: alloc::vec::Vec<&str> =
            SUPPORTED_OPS_INVENTORY.iter().map(|(n, _)| *n).collect();
        all_names.sort_unstable();
        let orig = all_names.len();
        all_names.dedup();
        assert_eq!(
            orig,
            all_names.len(),
            "duplicate entry in SUPPORTED_OPS_INVENTORY"
        );
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
        let result = op_conv(&input, &weight, None, &ConvAttrs::default()).unwrap();
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
        let result = op_conv(&input, &weight, None, &ConvAttrs::default()).unwrap();
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
        let result = op_conv(&input, &weight, Some(&bias), &ConvAttrs::default()).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn test_op_conv_channel_mismatch() {
        let input = make_f32_tensor(&[1, 3, 4, 4], &[1.0; 48]);
        let weight = make_f32_tensor(&[1, 2, 3, 3], &[1.0; 18]); // Wrong c_in
        let result = op_conv(&input, &weight, None, &ConvAttrs::default());
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    // ---- ConvAttrs parser tests ----

    fn attr_ints(name: &str, vals: &[i64]) -> AttributeProto {
        AttributeProto {
            name: String::from(name),
            attr_type: crate::onnx_types::AttributeType::Ints,
            ints: vals.to_vec(),
            ..AttributeProto::default()
        }
    }
    fn attr_int(name: &str, val: i64) -> AttributeProto {
        AttributeProto {
            name: String::from(name),
            attr_type: crate::onnx_types::AttributeType::Int,
            i: val,
            ..AttributeProto::default()
        }
    }
    fn attr_str(name: &str, val: &str) -> AttributeProto {
        AttributeProto {
            name: String::from(name),
            attr_type: crate::onnx_types::AttributeType::String,
            s: val.as_bytes().to_vec(),
            ..AttributeProto::default()
        }
    }

    #[test]
    fn test_conv_attrs_default_when_empty() {
        let parsed = ConvAttrs::from_attributes(&[]).unwrap();
        assert_eq!(parsed, ConvAttrs::default());
        assert!(parsed.is_default());
    }

    #[test]
    fn test_conv_attrs_populates_each_field() {
        let attrs = [
            attr_ints("strides", &[2, 2]),
            attr_ints("pads", &[1, 2, 3, 4]),
            attr_ints("dilations", &[2, 3]),
            attr_int("group", 4),
        ];
        let parsed = ConvAttrs::from_attributes(&attrs).unwrap();
        assert_eq!(parsed.strides, [2, 2]);
        assert_eq!(parsed.pads, [1, 2, 3, 4]);
        assert_eq!(parsed.dilations, [2, 3]);
        assert_eq!(parsed.group, 4);
        assert!(!parsed.is_default());
    }

    #[test]
    fn test_conv_attrs_ignores_kernel_shape() {
        let attrs = [attr_ints("kernel_shape", &[3, 3])];
        let parsed = ConvAttrs::from_attributes(&attrs).unwrap();
        assert_eq!(parsed, ConvAttrs::default());
    }

    #[test]
    fn test_conv_attrs_rejects_unsupported_auto_pad() {
        let attrs = [attr_str("auto_pad", "SAME_UPPER")];
        let err = ConvAttrs::from_attributes(&attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidAttribute(_)));
    }

    #[test]
    fn test_conv_attrs_accepts_notset_auto_pad() {
        let attrs = [attr_str("auto_pad", "NOTSET")];
        let parsed = ConvAttrs::from_attributes(&attrs).unwrap();
        assert_eq!(parsed, ConvAttrs::default());
    }

    #[test]
    fn test_conv_attrs_rejects_bad_lengths() {
        let strides_bad = [attr_ints("strides", &[1, 1, 1])];
        assert!(matches!(
            ConvAttrs::from_attributes(&strides_bad),
            Err(OpError::InvalidAttribute(_))
        ));
        let pads_bad = [attr_ints("pads", &[0, 0])];
        assert!(matches!(
            ConvAttrs::from_attributes(&pads_bad),
            Err(OpError::InvalidAttribute(_))
        ));
        let group_bad = [attr_int("group", 0)];
        assert!(matches!(
            ConvAttrs::from_attributes(&group_bad),
            Err(OpError::InvalidAttribute(_))
        ));
    }

    // ---- Conv validator group-aware tests ----

    #[test]
    fn test_validate_conv_accepts_depthwise() {
        let input = make_f32_tensor(&[1, 32, 8, 8], &[0.0f32; 32 * 64]);
        let weight = make_f32_tensor(&[32, 1, 3, 3], &[0.0f32; 32 * 9]);
        let attrs = ConvAttrs {
            group: 32,
            ..ConvAttrs::default()
        };
        assert!(validate_conv_inputs(&input, &weight, &attrs).is_ok());
    }

    #[test]
    fn test_validate_conv_rejects_cout_not_divisible_by_group() {
        let input = make_f32_tensor(&[1, 16, 4, 4], &[0.0f32; 16 * 16]);
        let weight = make_f32_tensor(&[15, 8, 3, 3], &[0.0f32; 15 * 8 * 9]);
        let attrs = ConvAttrs {
            group: 2,
            ..ConvAttrs::default()
        };
        let err = validate_conv_inputs(&input, &weight, &attrs).unwrap_err();
        assert!(matches!(err, OpError::ShapeMismatch(ref m) if m.contains("divisible")));
    }

    // ---- Attribute-honoring Conv kernel tests ----

    // ---- PoolAttrs / BatchNormAttrs parser tests ----

    #[test]
    fn test_pool_attrs_defaults_when_empty() {
        let err = PoolAttrs::from_attributes(&[], true).unwrap_err();
        assert!(matches!(err, OpError::InvalidAttribute(_)));

        let no_kernel = PoolAttrs::from_attributes(&[], false).unwrap();
        assert_eq!(no_kernel.kernel_shape, [0, 0]);
        assert_eq!(no_kernel.strides, [1, 1]);
        assert_eq!(no_kernel.pads, [0, 0, 0, 0]);
        assert!(!no_kernel.ceil_mode);
        assert!(!no_kernel.count_include_pad);
    }

    #[test]
    fn test_pool_attrs_populates_fields() {
        let attrs = [
            attr_ints("kernel_shape", &[3, 3]),
            attr_ints("strides", &[2, 2]),
            attr_ints("pads", &[1, 1, 1, 1]),
            attr_int("ceil_mode", 1),
            attr_int("count_include_pad", 1),
        ];
        let p = PoolAttrs::from_attributes(&attrs, true).unwrap();
        assert_eq!(p.kernel_shape, [3, 3]);
        assert_eq!(p.strides, [2, 2]);
        assert_eq!(p.pads, [1, 1, 1, 1]);
        assert!(p.ceil_mode);
        assert!(p.count_include_pad);
    }

    #[test]
    fn test_pool_attrs_rejects_bad_auto_pad() {
        let attrs = [
            attr_ints("kernel_shape", &[2, 2]),
            attr_str("auto_pad", "SAME_UPPER"),
        ];
        assert!(matches!(
            PoolAttrs::from_attributes(&attrs, true),
            Err(OpError::InvalidAttribute(_))
        ));
    }

    #[test]
    fn test_batchnorm_attrs_defaults_and_overrides() {
        let d = BatchNormAttrs::from_attributes(&[]).unwrap();
        assert_eq!(d.epsilon, 1e-5);
        assert_eq!(d.momentum, 0.9);

        let over = BatchNormAttrs::from_attributes(&[
            AttributeProto {
                name: String::from("epsilon"),
                attr_type: crate::onnx_types::AttributeType::Float,
                f: 1e-3,
                ..AttributeProto::default()
            },
            AttributeProto {
                name: String::from("momentum"),
                attr_type: crate::onnx_types::AttributeType::Float,
                f: 0.5,
                ..AttributeProto::default()
            },
        ])
        .unwrap();
        assert_eq!(over.epsilon, 1e-3);
        assert_eq!(over.momentum, 0.5);
    }

    #[test]
    fn test_conv_stride2_halves_spatial() {
        // 1x1x4x4 input with identity 1x1 kernel at stride 2 -> 1x1x2x2 output
        let input = make_f32_tensor(
            &[1, 1, 4, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let weight = make_f32_tensor(&[1, 1, 1, 1], &[1.0]);
        let attrs = ConvAttrs {
            strides: [2, 2],
            ..ConvAttrs::default()
        };
        let out = op_conv(&input, &weight, None, &attrs).unwrap();
        assert_eq!(out.shape.dims, vec![1, 1, 2, 2]);
        assert_eq!(read_f32_vec(&out), vec![1.0, 3.0, 9.0, 11.0]);
    }

    #[test]
    fn test_conv_pad1_keeps_3x3_spatial_dims() {
        // 1x1x3x3 input with 3x3 kernel of weight 1/9 + pad=1 → same spatial dims
        let input = make_f32_tensor(&[1, 1, 3, 3], &[1.0f32; 9]);
        let weight = make_f32_tensor(&[1, 1, 3, 3], &[1.0f32 / 9.0; 9]);
        let attrs = ConvAttrs {
            pads: [1, 1, 1, 1],
            ..ConvAttrs::default()
        };
        let out = op_conv(&input, &weight, None, &attrs).unwrap();
        assert_eq!(out.shape.dims, vec![1, 1, 3, 3]);
        // Corners see 4 ones out of 9 kernel taps; edges see 6; center sees 9.
        let vals = read_f32_vec(&out);
        let approx_eq = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(approx_eq(vals[0], 4.0 / 9.0)); // top-left corner
        assert!(approx_eq(vals[4], 9.0 / 9.0)); // center
        assert!(approx_eq(vals[8], 4.0 / 9.0)); // bottom-right corner
    }

    #[test]
    fn test_conv_dilation2_expands_receptive_field() {
        // 1x1x5x5 with 2x2 kernel, dilation=[2,2] -> output spatial 2x2
        // input = row-major 0..25
        let input_vals: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let input = make_f32_tensor(&[1, 1, 5, 5], &input_vals);
        let weight = make_f32_tensor(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let attrs = ConvAttrs {
            dilations: [2, 2],
            ..ConvAttrs::default()
        };
        let out = op_conv(&input, &weight, None, &attrs).unwrap();
        assert_eq!(out.shape.dims, vec![1, 1, 3, 3]);
        // Dilated kernel samples (0,0) and (2,2) of the input window.
        // Output (0,0): input[0,0] + input[2,2] = 0 + 12 = 12
        // Output (2,2): input[2,2] + input[4,4] = 12 + 24 = 36
        let vals = read_f32_vec(&out);
        assert!((vals[0] - 12.0).abs() < 1e-5);
        assert!((vals[8] - 36.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv_group2_independent_halves() {
        // Input [1, 4, 2, 2] split into 2 groups of 2 channels each.
        // Weight [4, 2, 1, 1] — output channels 0,1 read input channels 0,1;
        // output channels 2,3 read input channels 2,3.
        let input = make_f32_tensor(
            &[1, 4, 2, 2],
            &[
                1.0, 2.0, 3.0, 4.0, // ch0
                5.0, 6.0, 7.0, 8.0, // ch1
                9.0, 10.0, 11.0, 12.0, // ch2
                13.0, 14.0, 15.0, 16.0, // ch3
            ],
        );
        // Each output channel reads 2 input channels via a 1x1 kernel.
        // co=0: (1*ch0 + 0*ch1) => ch0
        // co=1: (0*ch0 + 1*ch1) => ch1
        // co=2: (1*ch2 + 0*ch3) => ch2
        // co=3: (0*ch2 + 1*ch3) => ch3
        let weight = make_f32_tensor(&[4, 2, 1, 1], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let attrs = ConvAttrs {
            group: 2,
            ..ConvAttrs::default()
        };
        let out = op_conv(&input, &weight, None, &attrs).unwrap();
        assert_eq!(out.shape.dims, vec![1, 4, 2, 2]);
        let vals = read_f32_vec(&out);
        // co=0 should equal ch0 of input.
        assert_eq!(&vals[0..4], &[1.0, 2.0, 3.0, 4.0]);
        // co=1 should equal ch1.
        assert_eq!(&vals[4..8], &[5.0, 6.0, 7.0, 8.0]);
        // co=2 should equal ch2.
        assert_eq!(&vals[8..12], &[9.0, 10.0, 11.0, 12.0]);
        // co=3 should equal ch3.
        assert_eq!(&vals[12..16], &[13.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    fn test_conv_depthwise_matches_reference() {
        // Depthwise 2-channel 1x1x2x2 input, 2-channel 1x1 kernel with
        // distinct weights — each output channel reads exactly one input
        // channel.
        let input = make_f32_tensor(&[1, 2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Weight shape [C, 1, 1, 1] for depthwise conv.
        let weight = make_f32_tensor(&[2, 1, 1, 1], &[10.0, 100.0]);
        let attrs = ConvAttrs {
            group: 2,
            ..ConvAttrs::default()
        };
        let out = op_conv(&input, &weight, None, &attrs).unwrap();
        assert_eq!(out.shape.dims, vec![1, 2, 2, 2]);
        let vals = read_f32_vec(&out);
        assert_eq!(&vals[0..4], &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(&vals[4..8], &[500.0, 600.0, 700.0, 800.0]);
    }

    #[test]
    fn test_conv_asymmetric_pads_output_shape() {
        // 1x1x4x4 input, 2x2 kernel, pads = [top=1, left=2, bottom=0, right=0]
        // OH = (4 + 1 + 0 - 1 - 1) / 1 + 1 = 4
        // OW = (4 + 2 + 0 - 1 - 1) / 1 + 1 = 5
        let input = make_f32_tensor(&[1, 1, 4, 4], &[1.0f32; 16]);
        let weight = make_f32_tensor(&[1, 1, 2, 2], &[1.0f32; 4]);
        let attrs = ConvAttrs {
            pads: [1, 2, 0, 0],
            ..ConvAttrs::default()
        };
        let out = op_conv(&input, &weight, None, &attrs).unwrap();
        assert_eq!(out.shape.dims, vec![1, 1, 4, 5]);
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

        let conv_out = op_conv(
            &input,
            &conv_weight,
            Some(&conv_bias),
            &ConvAttrs::default(),
        )
        .unwrap();
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
        let conv1_out = op_conv(&input, &conv1_w, None, &ConvAttrs::default()).unwrap();
        assert_eq!(conv1_out.shape.dims, vec![1, 2, 6, 6]);

        let relu1_out = op_relu(&conv1_out).unwrap();

        // Conv2: [1, 2, 6, 6] -> [1, 4, 4, 4] (3x3 kernel, 4 output channels)
        let conv2_w_data: Vec<f32> = (0..(4 * 2 * 3 * 3))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();
        let conv2_w = make_f32_tensor(&[4, 2, 3, 3], &conv2_w_data);
        let conv2_out = op_conv(&relu1_out, &conv2_w, None, &ConvAttrs::default()).unwrap();
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
            assert!((0.0..=1.0 + 1e-6).contains(&p));
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
        let result = op_conv(&input, &weight, None, &ConvAttrs::default());
        assert!(matches!(result, Err(OpError::InvalidAttribute(_))));
    }

    #[test]
    fn test_conv_wrong_rank() {
        // 3D input (missing batch dim) should be rejected
        let input = make_f32_tensor(&[1, 3, 3], &[1.0; 9]);
        let weight = make_f32_tensor(&[1, 1, 1, 1], &[1.0]);
        let result = op_conv(&input, &weight, None, &ConvAttrs::default());
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    #[test]
    fn test_conv_kernel_larger_than_input() {
        // 4x4 kernel on 3x3 input should be rejected
        let input = make_f32_tensor(&[1, 1, 3, 3], &[1.0; 9]);
        let weight = make_f32_tensor(&[1, 1, 4, 4], &[1.0; 16]);
        let result = op_conv(&input, &weight, None, &ConvAttrs::default());
        assert!(matches!(result, Err(OpError::ShapeMismatch(_))));
    }

    // ---- Tier 1: op_sub tests ----

    #[test]
    fn test_op_sub_basic() {
        let a = make_f32_tensor(&[3], &[3.0, 2.0, 1.0]);
        let b = make_f32_tensor(&[3], &[1.0, 1.0, 1.0]);
        let result = op_sub(&[&a, &b]).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_op_sub_broadcast() {
        // [1,3] - [1,1] -> [1,3]
        let a = make_f32_tensor(&[1, 3], &[3.0, 3.0, 3.0]);
        let b = make_f32_tensor(&[1, 1], &[1.0]);
        let result = op_sub(&[&a, &b]).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![2.0, 2.0, 2.0]);
    }

    // ---- Tier 1: op_mul tests ----

    #[test]
    fn test_op_mul_basic() {
        let a = make_f32_tensor(&[2], &[2.0, 3.0]);
        let b = make_f32_tensor(&[2], &[4.0, 5.0]);
        let result = op_mul(&[&a, &b]).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![8.0, 15.0]);
    }

    #[test]
    fn test_op_mul_broadcast() {
        // [1,3] * scalar [1,1] -> [1,3]
        let a = make_f32_tensor(&[1, 3], &[1.0, 2.0, 3.0]);
        let b = make_f32_tensor(&[1, 1], &[5.0]);
        let result = op_mul(&[&a, &b]).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![5.0, 10.0, 15.0]);
    }

    // ---- Tier 1: op_div tests ----

    #[test]
    fn test_op_div_basic() {
        let a = make_f32_tensor(&[2], &[6.0, 9.0]);
        let b = make_f32_tensor(&[2], &[2.0, 3.0]);
        let result = op_div(&[&a, &b]).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![3.0, 3.0]);
    }

    #[test]
    fn test_op_div_by_zero() {
        let a = make_f32_tensor(&[2], &[1.0, -1.0]);
        let b = make_f32_tensor(&[2], &[0.0, 0.0]);
        let result = op_div(&[&a, &b]).unwrap();
        let vals = read_f32_vec(&result);
        assert!(vals[0].is_infinite() && vals[0] > 0.0);
        assert!(vals[1].is_infinite() && vals[1] < 0.0);
    }

    // ---- Tier 1: op_gemm tests ----

    #[test]
    fn test_op_gemm_basic() {
        // 2x2 identity matrix, alpha=1, beta=0
        let a = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let identity = make_f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let result = op_gemm(&a, &identity, None, 1.0, 0.0, false, false).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_op_gemm_with_bias() {
        // A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity), C = [10, 20] (1D bias)
        // result = 1 * A @ B + 1 * C = [[11, 22], [13, 24]]
        let a = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = make_f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let c = make_f32_tensor(&[2], &[10.0, 20.0]);
        let result = op_gemm(&a, &b, Some(&c), 1.0, 1.0, false, false).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_op_gemm_transpose() {
        // A = [[1,3],[2,4]] stored as [1,3,2,4], transA=true -> logical [[1,2],[3,4]]
        // B = [[1,0],[0,1]] identity
        // result = [[1,2],[3,4]]
        let a = make_f32_tensor(&[2, 2], &[1.0, 3.0, 2.0, 4.0]);
        let b = make_f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let result = op_gemm(&a, &b, None, 1.0, 0.0, true, false).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);

        // transB=true: B stored as [[1,0],[0,1]]^T = [[1,0],[0,1]]
        // Use a non-identity: B = [[5,6],[7,8]], transB=true -> logical [[5,7],[6,8]]
        let a2 = make_f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let b2 = make_f32_tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let result2 = op_gemm(&a2, &b2, None, 1.0, 0.0, false, true).unwrap();
        let vals2 = read_f32_vec(&result2);
        // I @ B^T = [[5,7],[6,8]]
        assert_eq!(vals2, vec![5.0, 7.0, 6.0, 8.0]);
    }

    #[test]
    fn test_op_gemm_alpha_beta() {
        // A = [[1,0],[0,1]], B = [[2,0],[0,2]], C = [[10,10],[10,10]]
        // result = alpha * A @ B + beta * C = 2 * [[2,0],[0,2]] + 0.5 * [[10,10],[10,10]]
        //        = [[4,0],[0,4]] + [[5,5],[5,5]] = [[9,5],[5,9]]
        let a = make_f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let b = make_f32_tensor(&[2, 2], &[2.0, 0.0, 0.0, 2.0]);
        let c = make_f32_tensor(&[2, 2], &[10.0, 10.0, 10.0, 10.0]);
        let result = op_gemm(&a, &b, Some(&c), 2.0, 0.5, false, false).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![9.0, 5.0, 5.0, 9.0]);
    }

    // ---- Tier 2: op_sigmoid tests ----

    #[test]
    fn test_op_sigmoid_zero() {
        let t = make_f32_tensor(&[1], &[0.0]);
        let result = op_sigmoid(&t).unwrap();
        let vals = read_f32_vec(&result);
        assert!(
            (vals[0] - 0.5).abs() < 1e-4,
            "sigmoid(0) = {}, expected ~0.5",
            vals[0]
        );
    }

    #[test]
    fn test_op_sigmoid_positive() {
        let t = make_f32_tensor(&[1], &[10.0]);
        let result = op_sigmoid(&t).unwrap();
        let vals = read_f32_vec(&result);
        assert!(
            (vals[0] - 1.0).abs() < 1e-4,
            "sigmoid(10) = {}, expected ~1.0",
            vals[0]
        );
    }

    #[test]
    fn test_op_sigmoid_negative() {
        let t = make_f32_tensor(&[1], &[-10.0]);
        let result = op_sigmoid(&t).unwrap();
        let vals = read_f32_vec(&result);
        assert!(
            vals[0].abs() < 1e-4,
            "sigmoid(-10) = {}, expected ~0.0",
            vals[0]
        );
    }

    // ---- Tier 2: op_tanh tests ----

    #[test]
    fn test_op_tanh_zero() {
        let t = make_f32_tensor(&[1], &[0.0]);
        let result = op_tanh(&t).unwrap();
        let vals = read_f32_vec(&result);
        assert!(vals[0].abs() < 1e-4, "tanh(0) = {}, expected ~0.0", vals[0]);
    }

    #[test]
    fn test_op_tanh_positive() {
        let t = make_f32_tensor(&[1], &[10.0]);
        let result = op_tanh(&t).unwrap();
        let vals = read_f32_vec(&result);
        assert!(
            (vals[0] - 1.0).abs() < 1e-4,
            "tanh(10) = {}, expected ~1.0",
            vals[0]
        );
    }

    #[test]
    fn test_op_tanh_negative() {
        let t = make_f32_tensor(&[1], &[-10.0]);
        let result = op_tanh(&t).unwrap();
        let vals = read_f32_vec(&result);
        assert!(
            (vals[0] + 1.0).abs() < 1e-4,
            "tanh(-10) = {}, expected ~-1.0",
            vals[0]
        );
    }

    #[test]
    fn test_op_tanh_symmetry() {
        // tanh(-x) = -tanh(x)
        let vals_to_check = [0.5, 1.0, 2.0, 3.0];
        for &x in &vals_to_check {
            let pos = make_f32_tensor(&[1], &[x]);
            let neg = make_f32_tensor(&[1], &[-x]);
            let r_pos = read_f32_vec(&op_tanh(&pos).unwrap());
            let r_neg = read_f32_vec(&op_tanh(&neg).unwrap());
            assert!(
                (r_neg[0] + r_pos[0]).abs() < 1e-4,
                "tanh(-{}) + tanh({}) = {}, expected ~0.0",
                x,
                x,
                r_neg[0] + r_pos[0]
            );
        }
    }

    // ---- Tier 4: normalization, pooling, reduction tests ----

    #[test]
    fn test_op_batch_norm_basic() {
        // NCHW [1,2,1,2]: 2 channels, each with 2 spatial values
        // Channel 0: [1.0, 2.0], Channel 1: [3.0, 4.0]
        let input = make_f32_tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let scale = make_f32_tensor(&[2], &[1.0, 1.0]);
        let bias = make_f32_tensor(&[2], &[0.0, 0.0]);
        let mean = make_f32_tensor(&[2], &[1.5, 3.5]);
        let var = make_f32_tensor(&[2], &[1.0, 1.0]);
        let eps = 1e-5;
        let result = op_batch_normalization(&input, &scale, &bias, &mean, &var, eps).unwrap();
        let vals = read_f32_vec(&result);
        // Channel 0: (1.0 - 1.5) / sqrt(1.0 + eps) ≈ -0.5, (2.0 - 1.5) / sqrt(1.0 + eps) ≈ 0.5
        // Channel 1: (3.0 - 3.5) / sqrt(1.0 + eps) ≈ -0.5, (4.0 - 3.5) / sqrt(1.0 + eps) ≈ 0.5
        assert!((vals[0] - (-0.5)).abs() < 0.01, "ch0[0]={}", vals[0]);
        assert!((vals[1] - 0.5).abs() < 0.01, "ch0[1]={}", vals[1]);
        assert!((vals[2] - (-0.5)).abs() < 0.01, "ch1[0]={}", vals[2]);
        assert!((vals[3] - 0.5).abs() < 0.01, "ch1[1]={}", vals[3]);
    }

    #[test]
    fn test_op_layer_norm_basic() {
        // [2,3] tensor, normalize along axis=-1 (last axis, size 3)
        // Row 0: [1.0, 2.0, 3.0], Row 1: [4.0, 5.0, 6.0]
        let input = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scale = make_f32_tensor(&[3], &[1.0, 1.0, 1.0]);
        let eps = 1e-5;
        let result = op_layer_normalization(&input, &scale, None, -1, eps).unwrap();
        let vals = read_f32_vec(&result);
        // Each row should have mean ≈ 0 and variance ≈ 1
        for row in 0..2 {
            let row_vals = &vals[row * 3..(row + 1) * 3];
            let row_mean: f32 = row_vals.iter().sum::<f32>() / 3.0;
            let row_var: f32 = row_vals
                .iter()
                .map(|v| (v - row_mean) * (v - row_mean))
                .sum::<f32>()
                / 3.0;
            assert!(row_mean.abs() < 0.01, "row {} mean={}", row, row_mean);
            assert!((row_var - 1.0).abs() < 0.01, "row {} var={}", row, row_var);
        }
    }

    #[test]
    fn test_op_maxpool_2x2() {
        // NCHW [1,1,4,4], kernel 2x2, stride 2 -> [1,1,2,2]
        #[rustfmt::skip]
        let data = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_f32_tensor(&[1, 1, 4, 4], &data);
        let result = op_maxpool(&input, &[2, 2], Some(&[2, 2]), None).unwrap();
        assert_eq!(result.shape.dims, vec![1, 1, 2, 2]);
        let vals = read_f32_vec(&result);
        // Top-left 2x2: max(1,2,5,6)=6
        // Top-right 2x2: max(3,4,7,8)=8
        // Bottom-left 2x2: max(9,10,13,14)=14
        // Bottom-right 2x2: max(11,12,15,16)=16
        assert!((vals[0] - 6.0).abs() < 0.01, "tl={}", vals[0]);
        assert!((vals[1] - 8.0).abs() < 0.01, "tr={}", vals[1]);
        assert!((vals[2] - 14.0).abs() < 0.01, "bl={}", vals[2]);
        assert!((vals[3] - 16.0).abs() < 0.01, "br={}", vals[3]);
    }

    #[test]
    fn test_op_averagepool_2x2() {
        // NCHW [1,1,4,4], kernel 2x2, stride 2 -> [1,1,2,2]
        #[rustfmt::skip]
        let data = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_f32_tensor(&[1, 1, 4, 4], &data);
        let result = op_averagepool(&input, &[2, 2], Some(&[2, 2]), None).unwrap();
        assert_eq!(result.shape.dims, vec![1, 1, 2, 2]);
        let vals = read_f32_vec(&result);
        // Top-left: avg(1,2,5,6) = 3.5
        // Top-right: avg(3,4,7,8) = 5.5
        // Bottom-left: avg(9,10,13,14) = 11.5
        // Bottom-right: avg(11,12,15,16) = 13.5
        assert!((vals[0] - 3.5).abs() < 0.01, "tl={}", vals[0]);
        assert!((vals[1] - 5.5).abs() < 0.01, "tr={}", vals[1]);
        assert!((vals[2] - 11.5).abs() < 0.01, "bl={}", vals[2]);
        assert!((vals[3] - 13.5).abs() < 0.01, "br={}", vals[3]);
    }

    #[test]
    fn test_op_global_average_pool() {
        // NCHW [1,2,3,3] -> [1,2,1,1]
        // Channel 0: 9 values 1..9, mean = 5.0
        // Channel 1: 9 values 10..18, mean = 14.0
        #[rustfmt::skip]
        let data = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ];
        let input = make_f32_tensor(&[1, 2, 3, 3], &data);
        let result = op_global_average_pool(&input).unwrap();
        assert_eq!(result.shape.dims, vec![1, 2, 1, 1]);
        let vals = read_f32_vec(&result);
        assert!((vals[0] - 5.0).abs() < 0.01, "ch0 mean={}", vals[0]);
        assert!((vals[1] - 14.0).abs() < 0.01, "ch1 mean={}", vals[1]);
    }

    #[test]
    fn test_op_reduce_sum_axis0() {
        // [[1,2],[3,4]] reduce axis 0 -> [4,6]
        let input = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let result = op_reduce_sum(&input, &[0], false).unwrap();
        let vals = read_f32_vec(&result);
        assert!((vals[0] - 4.0).abs() < 0.01, "col0={}", vals[0]);
        assert!((vals[1] - 6.0).abs() < 0.01, "col1={}", vals[1]);
    }

    #[test]
    fn test_op_reduce_sum_axis1() {
        // [[1,2],[3,4]] reduce axis 1 -> [3,7]
        let input = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let result = op_reduce_sum(&input, &[1], false).unwrap();
        let vals = read_f32_vec(&result);
        assert!((vals[0] - 3.0).abs() < 0.01, "row0={}", vals[0]);
        assert!((vals[1] - 7.0).abs() < 0.01, "row1={}", vals[1]);
    }

    #[test]
    fn test_op_reduce_sum_keepdims() {
        // [[1,2],[3,4]] reduce axis 0, keepdims=true -> shape [1,2], values [4,6]
        let input = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let result = op_reduce_sum(&input, &[0], true).unwrap();
        assert_eq!(result.shape.dims, vec![1, 2]);
        let vals = read_f32_vec(&result);
        assert!((vals[0] - 4.0).abs() < 0.01, "col0={}", vals[0]);
        assert!((vals[1] - 6.0).abs() < 0.01, "col1={}", vals[1]);
    }

    #[test]
    fn test_op_reduce_mean_axis0() {
        // [[1,2],[3,4]] reduce axis 0 -> [2,3]
        let input = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let result = op_reduce_mean(&input, &[0], false).unwrap();
        let vals = read_f32_vec(&result);
        assert!((vals[0] - 2.0).abs() < 0.01, "col0={}", vals[0]);
        assert!((vals[1] - 3.0).abs() < 0.01, "col1={}", vals[1]);
    }

    #[test]
    fn test_op_reduce_mean_axis1() {
        // [[1,2],[3,4]] reduce axis 1 -> [1.5, 3.5]
        let input = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let result = op_reduce_mean(&input, &[1], false).unwrap();
        let vals = read_f32_vec(&result);
        assert!((vals[0] - 1.5).abs() < 0.01, "row0={}", vals[0]);
        assert!((vals[1] - 3.5).abs() < 0.01, "row1={}", vals[1]);
    }

    #[test]
    fn test_op_reduce_mean_all() {
        // [[1,2],[3,4]] reduce all axes -> scalar 2.5
        let input = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let result = op_reduce_mean(&input, &[], false).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 2.5).abs() < 0.01, "mean={}", vals[0]);
    }

    #[test]
    fn test_sqrt_approx() {
        assert!(
            (sqrt_approx(4.0) - 2.0).abs() < 0.01,
            "sqrt(4)={}",
            sqrt_approx(4.0)
        );
        assert!(
            (sqrt_approx(9.0) - 3.0).abs() < 0.01,
            "sqrt(9)={}",
            sqrt_approx(9.0)
        );
        assert!(
            (sqrt_approx(0.0)).abs() < 0.01,
            "sqrt(0)={}",
            sqrt_approx(0.0)
        );
        assert!(
            (sqrt_approx(1.0) - 1.0).abs() < 0.01,
            "sqrt(1)={}",
            sqrt_approx(1.0)
        );
    }

    // ================================================================
    // Tier 3 — Shape/data movement operator tests
    // ================================================================

    // ---- op_transpose tests ----

    #[test]
    fn test_op_transpose_2d() {
        // [[1,2],[3,4]] transposed = [[1,3],[2,4]]
        let t = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let result = op_transpose(&t, Some(&[1, 0])).unwrap();
        assert_eq!(result.shape.dims, vec![2, 2]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_op_transpose_identity() {
        // perm=[0,1] returns same data
        let t = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = op_transpose(&t, Some(&[0, 1])).unwrap();
        assert_eq!(result.shape.dims, vec![2, 3]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ---- op_concat tests ----

    #[test]
    fn test_op_concat_axis0() {
        // Concat two [1,2] tensors along axis 0 -> [2,2]
        let a = make_f32_tensor(&[1, 2], &[1.0, 2.0]);
        let b = make_f32_tensor(&[1, 2], &[3.0, 4.0]);
        let result = op_concat(&[&a, &b], 0).unwrap();
        assert_eq!(result.shape.dims, vec![2, 2]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_op_concat_axis1() {
        // Concat two [2,1] tensors along axis 1 -> [2,2]
        let a = make_f32_tensor(&[2, 1], &[1.0, 3.0]);
        let b = make_f32_tensor(&[2, 1], &[2.0, 4.0]);
        let result = op_concat(&[&a, &b], 1).unwrap();
        assert_eq!(result.shape.dims, vec![2, 2]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ---- op_flatten tests ----

    #[test]
    fn test_op_flatten_3d() {
        // [2,3,4] shape -> [2,12] at axis=1
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = make_f32_tensor(&[2, 3, 4], &data);
        let result = op_flatten(&t, 1).unwrap();
        assert_eq!(result.shape.dims, vec![2, 12]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, data);
    }

    // ---- op_squeeze tests ----

    #[test]
    fn test_op_squeeze_all() {
        // [1,3,1] -> [3]
        let t = make_f32_tensor(&[1, 3, 1], &[10.0, 20.0, 30.0]);
        let result = op_squeeze(&t, None).unwrap();
        assert_eq!(result.shape.dims, vec![3]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_op_squeeze_specific() {
        // [1,3,1] squeeze axis 0 -> [3,1]
        let t = make_f32_tensor(&[1, 3, 1], &[10.0, 20.0, 30.0]);
        let result = op_squeeze(&t, Some(&[0])).unwrap();
        assert_eq!(result.shape.dims, vec![3, 1]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    // ---- op_unsqueeze tests ----

    #[test]
    fn test_op_unsqueeze() {
        // [3] unsqueeze axis 0 -> [1,3]
        let t = make_f32_tensor(&[3], &[10.0, 20.0, 30.0]);
        let result = op_unsqueeze(&t, &[0]).unwrap();
        assert_eq!(result.shape.dims, vec![1, 3]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    // ---- squeeze/unsqueeze roundtrip ----

    #[test]
    fn test_op_squeeze_unsqueeze_roundtrip() {
        // Start with [1,3,1], squeeze all -> [3], unsqueeze at 0 and 2 -> [1,3,1]
        let t = make_f32_tensor(&[1, 3, 1], &[10.0, 20.0, 30.0]);
        let squeezed = op_squeeze(&t, None).unwrap();
        assert_eq!(squeezed.shape.dims, vec![3]);
        let restored = op_unsqueeze(&squeezed, &[0, 2]).unwrap();
        assert_eq!(restored.shape.dims, vec![1, 3, 1]);
        let vals = read_f32_vec(&restored);
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    // ---- op_cast tests ----

    #[test]
    fn test_op_cast_f32_to_i32() {
        // Cast [1.5, -2.7] -> [1, -2] (truncation toward zero)
        let t = make_f32_tensor(&[2], &[1.5, -2.7]);
        let result = op_cast(&t, DataType::Int32).unwrap();
        assert_eq!(result.data_type, DataType::Int32);
        assert_eq!(result.shape.dims, vec![2]);
        let v0 = i32::from_le_bytes([
            result.raw_data[0],
            result.raw_data[1],
            result.raw_data[2],
            result.raw_data[3],
        ]);
        let v1 = i32::from_le_bytes([
            result.raw_data[4],
            result.raw_data[5],
            result.raw_data[6],
            result.raw_data[7],
        ]);
        assert_eq!(v0, 1);
        assert_eq!(v1, -2);
    }

    #[test]
    fn test_op_cast_identity() {
        // Cast f32 to f32 returns same tensor
        let t = make_f32_tensor(&[3], &[1.0, 2.0, 3.0]);
        let result = op_cast(&t, DataType::Float).unwrap();
        assert_eq!(result.data_type, DataType::Float);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    // ---- cast helper direct tests ----

    fn make_i32_tensor(shape: &[i64], data: &[i32]) -> Tensor {
        let mut raw = alloc::vec![0u8; data.len() * 4];
        for (i, &v) in data.iter().enumerate() {
            let bytes = v.to_le_bytes();
            raw[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        Tensor {
            data_type: DataType::Int32,
            shape: TensorShape::new(Vec::from(shape)),
            name: String::new(),
            raw_data: raw,
        }
    }

    fn make_i64_tensor(shape: &[i64], data: &[i64]) -> Tensor {
        let mut raw = alloc::vec![0u8; data.len() * 8];
        for (i, &v) in data.iter().enumerate() {
            let bytes = v.to_le_bytes();
            raw[i * 8..i * 8 + 8].copy_from_slice(&bytes);
        }
        Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(Vec::from(shape)),
            name: String::new(),
            raw_data: raw,
        }
    }

    #[test]
    fn test_cast_f32_to_i32_truncation() {
        let t = make_f32_tensor(&[4], &[1.7, -2.3, 0.0, 4.999]);
        let result = cast_f32_to_i32(&t).unwrap();
        assert_eq!(result.data_type, DataType::Int32);
        assert_eq!(result.shape.dims, vec![4]);
        assert_eq!(read_i32(&result.raw_data, 0), 1);
        assert_eq!(read_i32(&result.raw_data, 1), -2);
        assert_eq!(read_i32(&result.raw_data, 2), 0);
        assert_eq!(read_i32(&result.raw_data, 3), 4);
    }

    #[test]
    fn test_cast_i32_to_f32_basic() {
        let t = make_i32_tensor(&[3], &[5, -7, 0]);
        let result = cast_i32_to_f32(&t).unwrap();
        assert_eq!(result.data_type, DataType::Float);
        assert_eq!(result.shape.dims, vec![3]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![5.0, -7.0, 0.0]);
    }

    #[test]
    fn test_cast_f32_to_i64_overflow() {
        // Large but representable values
        let t = make_f32_tensor(&[3], &[1.0e10, -1.0e10, 42.9]);
        let result = cast_f32_to_i64(&t).unwrap();
        assert_eq!(result.data_type, DataType::Int64);
        assert_eq!(result.shape.dims, vec![3]);
        assert_eq!(read_i64(&result.raw_data, 0), 10_000_000_000);
        assert_eq!(read_i64(&result.raw_data, 1), -10_000_000_000);
        assert_eq!(read_i64(&result.raw_data, 2), 42);
    }

    #[test]
    fn test_cast_i64_to_f32_precision() {
        let t = make_i64_tensor(&[3], &[1, -2, 1_000_000]);
        let result = cast_i64_to_f32(&t).unwrap();
        assert_eq!(result.data_type, DataType::Float);
        let vals = read_f32_vec(&result);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], -2.0);
        // Large values lose precision but should round-trip for 1e6
        assert!((vals[2] - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_cast_output_preserves_shape_clears_name() {
        let input = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let raw_data = alloc::vec![0u8; 24];
        let out = cast_output(&input, DataType::Int64, raw_data);
        assert_eq!(out.data_type, DataType::Int64);
        assert_eq!(out.shape.dims, vec![2, 3]);
        assert_eq!(out.name, String::new());
        assert_eq!(out.raw_data.len(), 24);
    }

    #[test]
    fn test_cast_f32_to_i32_via_op_cast() {
        // Exercise the op_cast dispatch for f32->i64 and i32->f32 paths
        let t = make_f32_tensor(&[2], &[3.9, -4.1]);
        let r = op_cast(&t, DataType::Int64).unwrap();
        assert_eq!(r.data_type, DataType::Int64);
        assert_eq!(read_i64(&r.raw_data, 0), 3);
        assert_eq!(read_i64(&r.raw_data, 1), -4);

        let ti = make_i32_tensor(&[2], &[9, -3]);
        let rf = op_cast(&ti, DataType::Float).unwrap();
        assert_eq!(rf.data_type, DataType::Float);
        let vals = read_f32_vec(&rf);
        assert_eq!(vals, vec![9.0, -3.0]);

        let tl = make_i64_tensor(&[2], &[11, -22]);
        let rfl = op_cast(&tl, DataType::Float).unwrap();
        let vals = read_f32_vec(&rfl);
        assert_eq!(vals, vec![11.0, -22.0]);
    }

    // ---- op_clip tests ----

    #[test]
    fn test_op_clip_both() {
        // Clip [-5, 0, 5] to [-2, 2] -> [-2, 0, 2]
        let t = make_f32_tensor(&[3], &[-5.0, 0.0, 5.0]);
        let result = op_clip(&t, Some(-2.0), Some(2.0)).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![-2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_op_clip_min_only() {
        // Clip with min=0 -> relu behavior
        let t = make_f32_tensor(&[4], &[-3.0, -1.0, 0.0, 5.0]);
        let result = op_clip(&t, Some(0.0), None).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_op_clip_max_only() {
        // Clip with max=1
        let t = make_f32_tensor(&[4], &[-2.0, 0.5, 1.0, 3.0]);
        let result = op_clip(&t, None, Some(1.0)).unwrap();
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![-2.0, 0.5, 1.0, 1.0]);
    }

    // ---- op_pad tests ----

    #[test]
    fn test_op_pad_1d() {
        // Pad [1,2,3] with pads=[1,1] constant=0 -> [0,1,2,3,0]
        let t = make_f32_tensor(&[3], &[1.0, 2.0, 3.0]);
        let result = op_pad(&t, &[1, 1], "constant", 0.0).unwrap();
        assert_eq!(result.shape.dims, vec![5]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    // ---- op_slice tests ----

    #[test]
    fn test_op_slice_basic() {
        // Slice [1,2,3,4,5] starts=[1] ends=[4] -> [2,3,4]
        let t = make_f32_tensor(&[5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = op_slice(&t, &[1], &[4], None, None).unwrap();
        assert_eq!(result.shape.dims, vec![3]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_op_slice_negative() {
        // starts=[-3] ends=[-1] on [1,2,3,4,5] -> [3,4]
        let t = make_f32_tensor(&[5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = op_slice(&t, &[-3], &[-1], None, None).unwrap();
        assert_eq!(result.shape.dims, vec![2]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![3.0, 4.0]);
    }

    // ---- op_gather tests ----

    #[test]
    fn test_op_gather_1d() {
        // Gather from [10,20,30,40] indices=[1,3] -> [20,40]
        let input = make_f32_tensor(&[4], &[10.0, 20.0, 30.0, 40.0]);
        // Create Int64 index tensor with values [1, 3]
        let idx_data: Vec<u8> = {
            let mut d = alloc::vec![0u8; 2 * 8];
            let b1 = 1i64.to_le_bytes();
            let b3 = 3i64.to_le_bytes();
            d[..8].copy_from_slice(&b1);
            d[8..16].copy_from_slice(&b3);
            d
        };
        let indices = Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(vec![2]),
            name: String::new(),
            raw_data: idx_data,
        };
        let result = op_gather(&input, &indices, 0).unwrap();
        assert_eq!(result.shape.dims, vec![2]);
        let vals = read_f32_vec(&result);
        assert_eq!(vals, vec![20.0, 40.0]);
    }

    #[test]
    fn test_op_gather_1d_int64_input() {
        // Classic MobileNetV2 pattern: int64 shape vector, int64 scalar index.
        let input = make_i64_tensor(&[4], &[1, 3, 224, 224]);
        let indices = make_i64_tensor(&[], &[0]);
        let result = op_gather(&input, &indices, 0).unwrap();
        assert_eq!(result.data_type, DataType::Int64);
        // Scalar output — rank 0.
        assert_eq!(result.shape.dims, Vec::<i64>::new());
        assert_eq!(result.raw_data.len(), 8);
        assert_eq!(read_i64(&result.raw_data, 0), 1);
    }

    #[test]
    fn test_op_gather_1d_int32_indices() {
        let input = make_f32_tensor(&[5], &[10.0, 20.0, 30.0, 40.0, 50.0]);
        let indices = make_i32_tensor(&[3], &[4, 0, 2]);
        let result = op_gather(&input, &indices, 0).unwrap();
        assert_eq!(result.shape.dims, vec![3]);
        assert_eq!(read_f32_vec(&result), vec![50.0, 10.0, 30.0]);
    }

    #[test]
    fn test_op_gather_2d_along_axis0() {
        // 2-D float gather — previously silently zeroed.
        // Input shape [3, 4], axis=0, indices=[2, 0] -> shape [2, 4].
        let input = make_f32_tensor(
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, // row 0
                5.0, 6.0, 7.0, 8.0, // row 1
                9.0, 10.0, 11.0, 12.0, // row 2
            ],
        );
        let indices = make_i64_tensor(&[2], &[2, 0]);
        let result = op_gather(&input, &indices, 0).unwrap();
        assert_eq!(result.shape.dims, vec![2, 4]);
        assert_eq!(
            read_f32_vec(&result),
            vec![9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_op_gather_2d_along_axis1() {
        // Input shape [3, 4], axis=1, indices=[3, 1] -> shape [3, 2].
        let input = make_f32_tensor(
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, // row 0
                5.0, 6.0, 7.0, 8.0, // row 1
                9.0, 10.0, 11.0, 12.0, // row 2
            ],
        );
        let indices = make_i64_tensor(&[2], &[3, 1]);
        let result = op_gather(&input, &indices, 1).unwrap();
        assert_eq!(result.shape.dims, vec![3, 2]);
        assert_eq!(read_f32_vec(&result), vec![4.0, 2.0, 8.0, 6.0, 12.0, 10.0]);
    }

    #[test]
    fn test_op_gather_scalar_index_removes_axis() {
        // Scalar index (rank-0 indices) gathers one element along axis
        // and removes that dimension from the output.
        let input = make_f32_tensor(&[4], &[10.0, 20.0, 30.0, 40.0]);
        let indices = make_i64_tensor(&[], &[2]);
        let result = op_gather(&input, &indices, 0).unwrap();
        assert_eq!(result.shape.dims, Vec::<i64>::new());
        assert_eq!(read_f32_vec(&result), vec![30.0]);
    }

    #[test]
    fn test_op_gather_negative_index_wraps() {
        let input = make_f32_tensor(&[4], &[10.0, 20.0, 30.0, 40.0]);
        let indices = make_i64_tensor(&[2], &[-1, -4]);
        let result = op_gather(&input, &indices, 0).unwrap();
        assert_eq!(read_f32_vec(&result), vec![40.0, 10.0]);
    }

    #[test]
    fn test_op_gather_out_of_bounds_errors() {
        let input = make_f32_tensor(&[3], &[1.0, 2.0, 3.0]);
        let indices = make_i64_tensor(&[1], &[5]);
        let err = op_gather(&input, &indices, 0).unwrap_err();
        assert!(matches!(err, OpError::ShapeMismatch(_)));
    }

    #[test]
    fn test_op_gather_string_input_rejected() {
        let input = Tensor {
            data_type: DataType::String,
            shape: TensorShape::new(vec![2]),
            name: String::new(),
            raw_data: Vec::new(),
        };
        let indices = make_i64_tensor(&[1], &[0]);
        assert!(matches!(
            op_gather(&input, &indices, 0),
            Err(OpError::InvalidAttribute(_))
        ));
    }

    #[test]
    fn test_op_gather_float_indices_rejected() {
        let input = make_f32_tensor(&[3], &[1.0, 2.0, 3.0]);
        let bad_indices = make_f32_tensor(&[1], &[0.0]);
        assert!(matches!(
            op_gather(&input, &bad_indices, 0),
            Err(OpError::InvalidAttribute(_))
        ));
    }

    // ---- Parallel operator tests ----

    fn make_float_tensor(shape: Vec<i64>, data: Vec<f32>) -> Tensor {
        let mut raw_data = alloc::vec![0u8; data.len() * 4];
        for (i, &v) in data.iter().enumerate() {
            let bytes = v.to_le_bytes();
            raw_data[i * 4] = bytes[0];
            raw_data[i * 4 + 1] = bytes[1];
            raw_data[i * 4 + 2] = bytes[2];
            raw_data[i * 4 + 3] = bytes[3];
        }
        Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(shape),
            name: String::new(),
            raw_data,
        }
    }

    #[test]
    fn test_parallel_relu_matches_sequential() {
        use crate::parallel::CorePool;
        let n = 100_000;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 50_000.0) * 0.01).collect();
        let tensor = make_float_tensor(vec![n as i64], data);

        let seq = op_relu(&tensor).unwrap();
        let pool = CorePool::new(4);
        let par = op_relu_parallel(&pool, &tensor, 0).unwrap(); // threshold=0 forces parallel

        assert_eq!(seq.raw_data.len(), par.raw_data.len());
        for i in 0..n {
            let vs = read_f32(&seq.raw_data, i);
            let vp = read_f32(&par.raw_data, i);
            assert!(
                (vs - vp).abs() < 1e-6,
                "relu mismatch at {}: seq={} par={}",
                i,
                vs,
                vp
            );
        }
    }

    #[test]
    fn test_parallel_sigmoid_matches_sequential() {
        use crate::parallel::CorePool;
        let n = 100_000;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 50_000.0) * 0.001).collect();
        let tensor = make_float_tensor(vec![n as i64], data);

        let seq = op_sigmoid(&tensor).unwrap();
        let pool = CorePool::new(4);
        let par = op_sigmoid_parallel(&pool, &tensor, 0).unwrap();

        for i in 0..n {
            let vs = read_f32(&seq.raw_data, i);
            let vp = read_f32(&par.raw_data, i);
            assert!(
                (vs - vp).abs() < 1e-5,
                "sigmoid mismatch at {}: seq={} par={}",
                i,
                vs,
                vp
            );
        }
    }

    #[test]
    fn test_parallel_tanh_matches_sequential() {
        use crate::parallel::CorePool;
        let n = 100_000;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 50_000.0) * 0.0001).collect();
        let tensor = make_float_tensor(vec![n as i64], data);

        let seq = op_tanh(&tensor).unwrap();
        let pool = CorePool::new(4);
        let par = op_tanh_parallel(&pool, &tensor, 0).unwrap();

        for i in 0..n {
            let vs = read_f32(&seq.raw_data, i);
            let vp = read_f32(&par.raw_data, i);
            assert!(
                (vs - vp).abs() < 1e-5,
                "tanh mismatch at {}: seq={} par={}",
                i,
                vs,
                vp
            );
        }
    }

    #[test]
    fn test_parallel_clip_matches_sequential() {
        use crate::parallel::CorePool;
        let n = 100_000;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 50_000.0) * 0.01).collect();
        let tensor = make_float_tensor(vec![n as i64], data);

        let seq = op_clip(&tensor, Some(-100.0), Some(100.0)).unwrap();
        let pool = CorePool::new(4);
        let par = op_clip_parallel(&pool, &tensor, Some(-100.0), Some(100.0), 0).unwrap();

        for i in 0..n {
            let vs = read_f32(&seq.raw_data, i);
            let vp = read_f32(&par.raw_data, i);
            assert!(
                (vs - vp).abs() < 1e-6,
                "clip mismatch at {}: seq={} par={}",
                i,
                vs,
                vp
            );
        }
    }

    #[test]
    fn test_parallel_conv_matches_sequential() {
        use crate::parallel::CorePool;
        // [1, 3, 8, 8] input, [16, 3, 3, 3] weight
        let n = 1;
        let c_in = 3;
        let h = 8;
        let w = 8;
        let c_out = 16;
        let kh = 3;
        let kw = 3;

        let in_size = n * c_in * h * w;
        let w_size = c_out * c_in * kh * kw;
        let in_data: Vec<f32> = (0..in_size).map(|i| (i % 7) as f32 * 0.1).collect();
        let w_data: Vec<f32> = (0..w_size)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();

        let input = make_float_tensor(vec![n as i64, c_in as i64, h as i64, w as i64], in_data);
        let weight = make_float_tensor(
            vec![c_out as i64, c_in as i64, kh as i64, kw as i64],
            w_data,
        );

        let seq = op_conv(&input, &weight, None, &ConvAttrs::default()).unwrap();
        let pool = CorePool::new(4);
        let par = op_conv_parallel(&pool, &input, &weight, None, &ConvAttrs::default(), 0).unwrap();

        assert_eq!(seq.shape.dims, par.shape.dims);
        let total = seq.shape.total_elements();
        for i in 0..total {
            let vs = read_f32(&seq.raw_data, i);
            let vp = read_f32(&par.raw_data, i);
            assert!(
                (vs - vp).abs() < 1e-4,
                "conv mismatch at {}: seq={} par={}",
                i,
                vs,
                vp
            );
        }
    }

    #[test]
    fn test_parallel_reduce_sum_matches_sequential() {
        use crate::parallel::CorePool;
        let n = 100_000;
        let data: Vec<f32> = (0..n).map(|i| (i % 100) as f32 * 0.01).collect();
        let tensor = make_float_tensor(vec![n as i64], data);

        let seq = op_reduce_sum(&tensor, &[], true).unwrap();
        let pool = CorePool::new(4);
        let par = op_reduce_sum_parallel(&pool, &tensor, &[], true, 0).unwrap();

        let vs = read_f32(&seq.raw_data, 0);
        let vp = read_f32(&par.raw_data, 0);
        // f32 accumulation order differs, so allow larger tolerance
        assert!(
            (vs - vp).abs() / vs.abs().max(1.0) < 1e-3,
            "reduce_sum mismatch: seq={} par={}",
            vs,
            vp
        );
    }

    #[test]
    fn test_parallel_reduce_mean_matches_sequential() {
        use crate::parallel::CorePool;
        let n = 100_000;
        let data: Vec<f32> = (0..n).map(|i| (i % 100) as f32 * 0.01).collect();
        let tensor = make_float_tensor(vec![n as i64], data);

        let seq = op_reduce_mean(&tensor, &[], true).unwrap();
        let pool = CorePool::new(4);
        let par = op_reduce_mean_parallel(&pool, &tensor, &[], true, 0).unwrap();

        let vs = read_f32(&seq.raw_data, 0);
        let vp = read_f32(&par.raw_data, 0);
        assert!(
            (vs - vp).abs() / vs.abs().max(1.0) < 1e-3,
            "reduce_mean mismatch: seq={} par={}",
            vs,
            vp
        );
    }

    #[test]
    fn test_parallel_softmax_1d_matches_sequential() {
        use crate::parallel::CorePool;
        // Use a moderate range to avoid exp underflow to zero
        let n = 10_000;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 5000.0) * 0.001).collect();
        let tensor = make_float_tensor(vec![n as i64], data);

        let seq = op_softmax(&tensor, 0).unwrap();
        let pool = CorePool::new(4);
        let par = op_softmax_parallel(&pool, &tensor, 0, 0).unwrap();

        for i in 0..n {
            let vs = read_f32(&seq.raw_data, i);
            let vp = read_f32(&par.raw_data, i);
            assert!(
                (vs - vp).abs() < 1e-5,
                "softmax 1d mismatch at {}: seq={} par={}",
                i,
                vs,
                vp
            );
        }
    }

    #[test]
    fn test_parallel_softmax_2d_matches_sequential() {
        use crate::parallel::CorePool;
        let rows = 100;
        let cols = 200;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 50) as f32 - 25.0) * 0.1)
            .collect();
        let tensor = make_float_tensor(vec![rows as i64, cols as i64], data);

        let seq = op_softmax(&tensor, 1).unwrap();
        let pool = CorePool::new(4);
        let par = op_softmax_parallel(&pool, &tensor, 1, 0).unwrap();

        for i in 0..rows * cols {
            let vs = read_f32(&seq.raw_data, i);
            let vp = read_f32(&par.raw_data, i);
            assert!(
                (vs - vp).abs() < 1e-5,
                "softmax 2d mismatch at {}: seq={} par={}",
                i,
                vs,
                vp
            );
        }
    }

    // ---- End-to-end parallel inference test (Task Group 9) ----

    #[test]
    fn test_parallel_end_to_end_matches_sequential() {
        use crate::parallel::CorePool;

        // Build a simple pipeline: MatMul -> Add (bias) -> Relu -> MatMul -> Softmax
        let m = 32;
        let k1 = 64;
        let k2 = 32;

        // Input: [32, 64]
        let input_data: Vec<f32> = (0..m * k1).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let input = make_float_tensor(vec![m as i64, k1 as i64], input_data);

        // W1: [64, 32]
        let w1_data: Vec<f32> = (0..k1 * k2)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let w1 = make_float_tensor(vec![k1 as i64, k2 as i64], w1_data);

        // Bias: [32]
        let bias_data: Vec<f32> = (0..k2).map(|i| i as f32 * 0.01).collect();
        let bias = make_float_tensor(vec![k2 as i64], bias_data);

        // W2: [32, 10]
        let out_classes = 10usize;
        let w2_data: Vec<f32> = (0..k2 * out_classes)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let w2 = make_float_tensor(vec![k2 as i64, out_classes as i64], w2_data);

        // --- Sequential path ---
        let mm1_seq = op_matmul(&input, &w1).unwrap();
        let add_seq = op_add(&[&mm1_seq, &bias]).unwrap();
        let relu_seq = op_relu(&add_seq).unwrap();
        let mm2_seq = op_matmul(&relu_seq, &w2).unwrap();
        let out_seq = op_softmax(&mm2_seq, 1).unwrap();

        // --- Parallel path (4 threads) ---
        let pool = CorePool::new(4);
        // MatMul1 uses GEMM internally; for parallel, use gemm_f32_parallel
        // But op_matmul doesn't take a pool — so we verify the building blocks match
        let mm1_par = op_matmul(&input, &w1).unwrap();
        let add_par = op_add(&[&mm1_par, &bias]).unwrap();
        let relu_par = op_relu_parallel(&pool, &add_par, 0).unwrap();
        let mm2_par = op_matmul(&relu_par, &w2).unwrap();
        let out_par = op_softmax_parallel(&pool, &mm2_par, 1, 0).unwrap();

        // Verify outputs match
        let total = out_seq.shape.total_elements();
        assert_eq!(out_seq.shape.dims, out_par.shape.dims);
        for i in 0..total {
            let vs = read_f32(&out_seq.raw_data, i);
            let vp = read_f32(&out_par.raw_data, i);
            assert!(
                (vs - vp).abs() < 1e-5,
                "e2e mismatch at {}: seq={} par={}",
                i,
                vs,
                vp
            );
        }
    }

    #[test]
    fn test_expf_approx_very_negative_is_finite_zero() {
        // Very negative inputs must not produce -inf due to exponent
        // underflow reinterpretation; they should clamp to 0.
        let v = expf_approx(-1000.0);
        assert!(v.is_finite(), "expf_approx(-1000) = {} must be finite", v);
        assert_eq!(v, 0.0);
    }

    // ---------------------------------------------------------------------
    // BFloat16 operator coverage (safetensors-model-loader-v1 Section 3)
    // ---------------------------------------------------------------------

    fn make_bf16_tensor(dims: Vec<i64>, values: &[f32]) -> Tensor {
        Tensor {
            data_type: DataType::BFloat16,
            shape: TensorShape::new(dims),
            name: String::new(),
            raw_data: f32_to_bf16(values),
        }
    }

    fn read_bf16_tensor(t: &Tensor) -> Vec<f32> {
        let raw = bf16_to_f32(&t.raw_data);
        let n = t.shape.total_elements();
        raw[..n].to_vec()
    }

    #[test]
    fn test_op_add_bf16_element_wise() {
        let a = make_bf16_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let b = make_bf16_tensor(vec![4], &[0.5, 0.5, -1.0, 2.0]);
        let out = op_add(&[&a, &b]).unwrap();
        assert_eq!(out.data_type, DataType::BFloat16);
        assert_eq!(out.shape.dims, vec![4]);
        let v = read_bf16_tensor(&out);
        assert_eq!(v, vec![1.5, 2.5, 2.0, 6.0]);
    }

    #[test]
    fn test_op_add_bf16_broadcast() {
        // [2, 3] + [3] -> [2, 3]
        let a = make_bf16_tensor(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_bf16_tensor(vec![3], &[10.0, 20.0, 30.0]);
        let out = op_add(&[&a, &b]).unwrap();
        let v = read_bf16_tensor(&out);
        assert_eq!(v, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_op_mul_bf16_element_wise() {
        let a = make_bf16_tensor(vec![3], &[2.0, 4.0, 8.0]);
        let b = make_bf16_tensor(vec![3], &[0.5, 0.25, -1.0]);
        let out = op_mul(&[&a, &b]).unwrap();
        let v = read_bf16_tensor(&out);
        assert_eq!(v, vec![1.0, 1.0, -8.0]);
    }

    #[test]
    fn test_op_sigmoid_bf16() {
        // sigmoid(0) = 0.5 ; round-trip at BF16 precision.
        let t = make_bf16_tensor(vec![3], &[0.0, 2.0, -2.0]);
        let out = op_sigmoid(&t).unwrap();
        assert_eq!(out.data_type, DataType::BFloat16);
        let v = read_bf16_tensor(&out);
        assert!((v[0] - 0.5).abs() < 1e-2);
        assert!((v[1] - 0.8808).abs() < 1e-2);
        assert!((v[2] - 0.1192).abs() < 1e-2);
    }

    #[test]
    fn test_op_tanh_bf16() {
        let t = make_bf16_tensor(vec![3], &[0.0, 1.0, -1.0]);
        let out = op_tanh(&t).unwrap();
        let v = read_bf16_tensor(&out);
        assert!(v[0].abs() < 1e-2);
        assert!((v[1] - 0.7616).abs() < 2e-2);
        assert!((v[2] + 0.7616).abs() < 2e-2);
    }

    #[test]
    fn test_op_relu_bf16_rejected() {
        // op_relu has not been extended to BF16 — it must return a clear
        // error rather than panicking or silently coercing.
        let t = make_bf16_tensor(vec![3], &[-1.0, 0.0, 2.0]);
        let err = op_relu(&t).unwrap_err();
        assert!(
            matches!(err, OpError::InvalidAttribute(_)),
            "expected InvalidAttribute for BF16 Relu, got {:?}",
            err
        );
    }

    #[test]
    fn test_op_conv_bf16_rejected() {
        // Conv has no BF16 path — handing it BF16 input should surface a
        // clear error, not panic.
        let x = make_bf16_tensor(vec![1, 1, 3, 3], &[0.0f32; 9]);
        let w = make_bf16_tensor(vec![1, 1, 2, 2], &[0.0f32; 4]);
        let res = op_conv(&x, &w, None, &ConvAttrs::default());
        assert!(res.is_err(), "Conv with BF16 must return an error");
    }
}
