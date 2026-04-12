// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-operator profiling, timing, and WCET budget enforcement.
//!
//! Shared primitive types (`OperatorClass`, `OperatorBudget`, `BudgetResult`)
//! live in the Layer 0 `smallaios-sched-types` crate and are re-exported here
//! so existing call sites keep working.

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use alloc::string::String;
use alloc::vec::Vec;

/// Time source for operator timing measurement.
///
/// `no_std`-compatible trait — different implementations for kernel mode
/// (architecture timer) and container mode (std::time::Instant).
pub trait TimeSource: Send + Sync {
    /// Current time in microseconds since some fixed epoch.
    /// Epoch is arbitrary — only differences matter.
    fn now_us(&self) -> u64;
}

/// Zero-cost TimeSource that always returns 0. Used when profiling is disabled.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullTimeSource;

impl TimeSource for NullTimeSource {
    fn now_us(&self) -> u64 {
        0
    }
}

/// std::time::Instant-based TimeSource for container mode.
#[cfg(feature = "std")]
pub struct StdTimeSource {
    epoch: std::time::Instant,
}

#[cfg(feature = "std")]
impl StdTimeSource {
    pub fn new() -> Self {
        Self {
            epoch: std::time::Instant::now(),
        }
    }
}

#[cfg(feature = "std")]
impl Default for StdTimeSource {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
impl TimeSource for StdTimeSource {
    fn now_us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }
}

pub use smallaios_sched_types::{BudgetResult, OperatorBudget, OperatorClass};

/// Classify an ONNX operator name into its budget category.
pub fn classify_op(op_type: &str) -> OperatorClass {
    match op_type {
        "Add" | "Sub" | "Mul" | "Div" | "Relu" | "Sigmoid" | "Tanh" | "Clip" | "Cast"
        | "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" | "Transpose" | "Concat" | "Slice"
        | "Pad" | "Gather"
        // Tier 2 element-wise math
        | "Pow" | "Sqrt" | "Exp" | "Log" | "Erf" | "Neg" | "Abs" | "Floor" | "Ceil" | "Round"
        // Tier 2 comparison/selection
        | "Equal" | "NotEqual" | "Less" | "LessOrEqual" | "Greater" | "GreaterOrEqual"
        | "Where" | "Min" | "Max" | "Not"
        // Tier 2 composite activations
        | "Gelu" | "LeakyRelu" | "Elu" | "Swish"
        // Tier 2 transformer shape ops
        | "Split" | "Expand" | "Tile" | "OneHot"
        // Tier 2 quantization conversion
        | "QuantizeLinear" | "DequantizeLinear"
        // Tier 3 Phase 1 element-wise math / bool
        | "Mod" | "Sin" | "Cos" | "Reciprocal" | "Sign" | "Sum" | "Mean"
        | "And" | "Or"
        // Tier 3 Phase 1 shape / data-movement
        | "Shape" | "Size" | "Identity" | "Constant" | "ConstantOfShape"
        | "Range" | "Trilu" | "CumSum" | "GatherND" | "ScatterND"
        // Phase 1 vision: shape/data-movement-style ops
        | "DepthToSpace" | "SpaceToDepth" | "CenterCropPad"
        // Phase 1 vision: data selection
        | "TopK" | "Compress" | "NonZero" | "Unique"
        | "GatherElements" | "ScatterElements"
        // Phase 1 vision: composite activations
        | "HardSigmoid" | "HardSwish" | "PRelu" | "Mish"
        // Phase 2 generative: elementwise / samplers / small utilities
        | "DynamicQuantizeLinear" | "EyeLike" | "RandomNormal" | "RandomNormalLike"
        | "RandomUniform" | "RandomUniformLike" | "Bernoulli" | "Multinomial"
        | "Dropout" | "Softplus"
        // microsoft-fused-ops-v1: RoPE is elementwise per query position.
        | "RotaryEmbedding" => {
            OperatorClass::Elementwise
        }
        "Softmax" | "LayerNormalization" | "BatchNormalization" | "MaxPool" | "AveragePool"
        | "GlobalAveragePool" | "ReduceMean" | "ReduceSum"
        // Tier 3 Phase 1 reductions
        | "ReduceMax" | "ReduceMin" | "ReduceProd" | "ArgMax" | "ArgMin"
        | "LogSoftmax"
        // Phase 1 vision: pooling/resize/align — reduction-class budget.
        | "GlobalMaxPool" | "Resize" | "RoiAlign" | "GridSample"
        | "GroupNormalization" | "InstanceNormalization"
        // Phase 2 reductions + RMS-style normalizations
        | "RMSNormalization" | "LpNormalization" | "MeanVarianceNormalization"
        | "ReduceL1" | "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp" | "ReduceSumSquare"
        // microsoft-fused-ops-v1: RMSNorm wrappers classify as reduction.
        | "SimplifiedLayerNormalization" | "SkipSimplifiedLayerNormalization"
            => OperatorClass::Reduction,
        "MatMul" | "Gemm" | "Conv"
        // Tier 2: Einsum is matmul-like; quantized GEMM/Conv same.
        | "Einsum" | "QLinearMatMul" | "QLinearConv"
        // Phase 2: integer matmul.
        | "MatMulInteger" => OperatorClass::Gemm,
        // Recurrent layers do many GEMMs across time; classify as Attention budget.
        "RNN" | "LSTM" | "GRU"
        // microsoft-fused-ops-v1: fused attention operators.
        | "MultiHeadAttention" | "GroupQueryAttention" => OperatorClass::Attention,
        // Phase 2 control-flow ops: aggregate budget.
        "If" | "Loop" | "Scan" => OperatorClass::ControlFlow,
        _ => OperatorClass::Elementwise,
    }
}

/// Per-operator timing measurement.
#[derive(Debug, Clone)]
pub struct OperatorMeasurement {
    pub op_type: String,
    pub class: OperatorClass,
    pub actual_us: u64,
    pub budget_result: BudgetResult,
}

/// Full inference profile.
#[derive(Debug, Clone, Default)]
pub struct InferenceProfile {
    pub total_us: u64,
    pub operators: Vec<OperatorMeasurement>,
    pub warnings_count: u32,
    pub soft_limit_count: u32,
    pub hard_limit_aborted: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_elementwise() {
        assert_eq!(classify_op("Add"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Sub"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Mul"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Div"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Relu"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Sigmoid"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Tanh"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Clip"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Cast"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Reshape"), OperatorClass::Elementwise);
        assert_eq!(classify_op("Transpose"), OperatorClass::Elementwise);
    }

    #[test]
    fn test_classify_reduction() {
        assert_eq!(classify_op("Softmax"), OperatorClass::Reduction);
        assert_eq!(classify_op("LayerNormalization"), OperatorClass::Reduction);
        assert_eq!(classify_op("BatchNormalization"), OperatorClass::Reduction);
        assert_eq!(classify_op("MaxPool"), OperatorClass::Reduction);
        assert_eq!(classify_op("AveragePool"), OperatorClass::Reduction);
        assert_eq!(classify_op("GlobalAveragePool"), OperatorClass::Reduction);
        assert_eq!(classify_op("ReduceMean"), OperatorClass::Reduction);
        assert_eq!(classify_op("ReduceSum"), OperatorClass::Reduction);
    }

    #[test]
    fn test_classify_gemm() {
        assert_eq!(classify_op("MatMul"), OperatorClass::Gemm);
        assert_eq!(classify_op("Gemm"), OperatorClass::Gemm);
        assert_eq!(classify_op("Conv"), OperatorClass::Gemm);
    }

    #[test]
    fn test_classify_unknown_defaults_to_elementwise() {
        assert_eq!(classify_op("MysteryOp"), OperatorClass::Elementwise);
        assert_eq!(classify_op(""), OperatorClass::Elementwise);
    }

    #[test]
    fn test_null_time_source_returns_zero() {
        let ts = NullTimeSource;
        assert_eq!(ts.now_us(), 0);
        assert_eq!(ts.now_us(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_std_time_source_monotonic() {
        let ts = StdTimeSource::new();
        let a = ts.now_us();
        // Busy-wait a tiny bit to guarantee a tick.
        for _ in 0..1000 {
            core::hint::spin_loop();
        }
        let b = ts.now_us();
        assert!(b >= a, "StdTimeSource not monotonic: {} -> {}", a, b);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_std_time_source_default() {
        let _ts = StdTimeSource::default();
    }

    #[test]
    fn test_budget_ok() {
        let budget = OperatorBudget::DEFAULT;
        // 500us is well under the 1000us elementwise budget.
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 500),
            BudgetResult::Ok
        );
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 1000),
            BudgetResult::Ok
        );
    }

    #[test]
    fn test_budget_warning() {
        let budget = OperatorBudget::DEFAULT;
        // 1500us > 1000 (budget), <= 2000 (soft). Warning.
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 1500),
            BudgetResult::Warning
        );
    }

    #[test]
    fn test_budget_soft_limit() {
        let budget = OperatorBudget::DEFAULT;
        // 5000us > 2000 (soft), <= 10000 (hard). SoftLimit.
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 5000),
            BudgetResult::SoftLimit
        );
    }

    #[test]
    fn test_budget_hard_limit() {
        let budget = OperatorBudget::DEFAULT;
        // 20000us > 10000 (hard). HardLimit.
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 20_000),
            BudgetResult::HardLimit
        );
    }

    #[test]
    fn test_budget_all_classes() {
        let budget = OperatorBudget::DEFAULT;
        assert_eq!(
            budget.check(OperatorClass::Reduction, 5_000),
            BudgetResult::Ok
        );
        assert_eq!(budget.check(OperatorClass::Gemm, 50_000), BudgetResult::Ok);
        assert_eq!(
            budget.check(OperatorClass::Attention, 100_000),
            BudgetResult::Ok
        );
        assert_eq!(
            budget.check(OperatorClass::GpuKernel, 500_000),
            BudgetResult::Ok
        );
    }

    #[test]
    fn test_budget_default_values() {
        let budget = OperatorBudget::DEFAULT;
        assert_eq!(budget.elementwise_us, 1_000);
        assert_eq!(budget.reduction_us, 10_000);
        assert_eq!(budget.gemm_us, 100_000);
        assert_eq!(budget.attention_us, 500_000);
        assert_eq!(budget.gpu_kernel_us, 1_000_000);
        assert_eq!(budget.soft_multiplier, 2);
        assert_eq!(budget.hard_multiplier, 10);
    }

    #[test]
    fn test_budget_default_trait() {
        let a = OperatorBudget::default();
        let b = OperatorBudget::DEFAULT;
        assert_eq!(a.elementwise_us, b.elementwise_us);
        assert_eq!(a.hard_multiplier, b.hard_multiplier);
    }

    #[test]
    fn test_inference_profile_default() {
        let p = InferenceProfile::default();
        assert_eq!(p.total_us, 0);
        assert!(p.operators.is_empty());
        assert_eq!(p.warnings_count, 0);
        assert_eq!(p.soft_limit_count, 0);
        assert!(!p.hard_limit_aborted);
    }

    #[test]
    fn test_operator_measurement_clone() {
        let m = OperatorMeasurement {
            op_type: String::from("Relu"),
            class: OperatorClass::Elementwise,
            actual_us: 42,
            budget_result: BudgetResult::Ok,
        };
        let m2 = m.clone();
        assert_eq!(m2.op_type, "Relu");
        assert_eq!(m2.actual_us, 42);
    }
}
