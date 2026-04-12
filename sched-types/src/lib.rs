// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared scheduler primitive types for SmallAIOS.
//!
//! This crate exists at Layer 0 (Foundation) so that types used by
//! both the kernel scheduler (Layer 0) and the ONNX runtime (Layer 1)
//! can be defined once. Without this crate, types like `OperatorClass`
//! and `OperatorBudget` would have to be duplicated, since `onnx-rt`
//! cannot depend on `kernel` per the layered architecture.
//!
//! All types here are `Copy` and `no_std` / alloc-free.

#![no_std]
#![forbid(unsafe_code)]

/// Operator category for budget lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OperatorClass {
    Elementwise = 0,
    Reduction = 1,
    Gemm = 2,
    Attention = 3,
    GpuKernel = 4,
    /// Control-flow operator (`If`, `Loop`, `Scan`).
    ///
    /// Phase 2 (`generative-llm-v1`) introduces this variant so the cooperative
    /// scheduler can account for sub-graph executor dispatches as a single
    /// aggregate budget unit even when the inner body runs thousands of inner
    /// operators. The budget is much larger than a single `Gemm` because a
    /// `Loop` body can accumulate work across up to
    /// [`crate::MAX_LOOP_ITERATIONS`]-equivalent iterations.
    ControlFlow = 5,
}

/// Result of checking an operator's execution time against its budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetResult {
    Ok,
    Warning,
    SoftLimit,
    HardLimit,
}

/// Per-operator execution time budgets in microseconds.
#[derive(Debug, Clone, Copy)]
pub struct OperatorBudget {
    pub elementwise_us: u64,
    pub reduction_us: u64,
    pub gemm_us: u64,
    pub attention_us: u64,
    pub gpu_kernel_us: u64,
    /// Baseline budget for `If`/`Loop`/`Scan` dispatch.
    ///
    /// Because control-flow operators aggregate the wall time of *every*
    /// inner operator across every iteration, the baseline is several orders
    /// of magnitude larger than the `gemm_us` budget. A typical autoregressive
    /// decode loop generating 512 tokens at ~4 ms/tok sits comfortably inside
    /// the default 2-second baseline.
    pub control_flow_us: u64,
    pub soft_multiplier: u64,
    pub hard_multiplier: u64,
}

impl OperatorBudget {
    pub const DEFAULT: Self = Self {
        elementwise_us: 1_000,
        reduction_us: 10_000,
        gemm_us: 100_000,
        attention_us: 500_000,
        gpu_kernel_us: 1_000_000,
        control_flow_us: 2_000_000,
        soft_multiplier: 2,
        hard_multiplier: 10,
    };

    pub const fn check(&self, class: OperatorClass, actual_us: u64) -> BudgetResult {
        let budget = match class {
            OperatorClass::Elementwise => self.elementwise_us,
            OperatorClass::Reduction => self.reduction_us,
            OperatorClass::Gemm => self.gemm_us,
            OperatorClass::Attention => self.attention_us,
            OperatorClass::GpuKernel => self.gpu_kernel_us,
            OperatorClass::ControlFlow => self.control_flow_us,
        };
        if actual_us > budget * self.hard_multiplier {
            BudgetResult::HardLimit
        } else if actual_us > budget * self.soft_multiplier {
            BudgetResult::SoftLimit
        } else if actual_us > budget {
            BudgetResult::Warning
        } else {
            BudgetResult::Ok
        }
    }
}

impl Default for OperatorBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budgets_match_spec() {
        let b = OperatorBudget::DEFAULT;
        assert_eq!(b.elementwise_us, 1_000);
        assert_eq!(b.reduction_us, 10_000);
        assert_eq!(b.gemm_us, 100_000);
        assert_eq!(b.attention_us, 500_000);
        assert_eq!(b.gpu_kernel_us, 1_000_000);
        assert_eq!(b.control_flow_us, 2_000_000);
        assert_eq!(b.soft_multiplier, 2);
        assert_eq!(b.hard_multiplier, 10);
    }

    #[test]
    fn control_flow_budget_check() {
        let b = OperatorBudget::DEFAULT;
        // 1 second: well under the 2s baseline.
        assert_eq!(
            b.check(OperatorClass::ControlFlow, 1_000_000),
            BudgetResult::Ok
        );
        // 3 seconds: over baseline, under soft (4 s).
        assert_eq!(
            b.check(OperatorClass::ControlFlow, 3_000_000),
            BudgetResult::Warning
        );
        // 10 seconds: over soft (4 s), under hard (20 s).
        assert_eq!(
            b.check(OperatorClass::ControlFlow, 10_000_000),
            BudgetResult::SoftLimit
        );
        // 30 seconds: over hard (20 s).
        assert_eq!(
            b.check(OperatorClass::ControlFlow, 30_000_000),
            BudgetResult::HardLimit
        );
    }

    #[test]
    fn control_flow_variant_distinct() {
        assert_ne!(OperatorClass::ControlFlow, OperatorClass::Gemm);
        assert_ne!(OperatorClass::ControlFlow, OperatorClass::Attention);
    }

    #[test]
    fn check_ok() {
        let b = OperatorBudget::DEFAULT;
        assert_eq!(b.check(OperatorClass::Elementwise, 500), BudgetResult::Ok);
        assert_eq!(b.check(OperatorClass::Gemm, 50_000), BudgetResult::Ok);
    }

    #[test]
    fn check_warning() {
        let b = OperatorBudget::DEFAULT;
        // Just over 1x
        assert_eq!(
            b.check(OperatorClass::Elementwise, 1_500),
            BudgetResult::Warning
        );
    }

    #[test]
    fn check_soft_limit() {
        let b = OperatorBudget::DEFAULT;
        // Just over 2x
        assert_eq!(
            b.check(OperatorClass::Elementwise, 3_000),
            BudgetResult::SoftLimit
        );
    }

    #[test]
    fn check_hard_limit() {
        let b = OperatorBudget::DEFAULT;
        // Just over 10x
        assert_eq!(
            b.check(OperatorClass::Elementwise, 12_000),
            BudgetResult::HardLimit
        );
    }

    #[test]
    fn enum_variants_are_distinct() {
        assert_ne!(OperatorClass::Elementwise, OperatorClass::Reduction);
        assert_ne!(BudgetResult::Ok, BudgetResult::Warning);
    }
}
