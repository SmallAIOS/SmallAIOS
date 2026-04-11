// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for WCET budget enforcement and operator profiling.

extern crate alloc;

use smallaios_onnx_rt::profile::{
    classify_op, BudgetResult, InferenceProfile, NullTimeSource, OperatorBudget, OperatorClass,
    TimeSource,
};

#[cfg(feature = "std")]
use smallaios_onnx_rt::profile::StdTimeSource;

#[test]
fn classify_known_ops_by_category() {
    assert_eq!(classify_op("Relu"), OperatorClass::Elementwise);
    assert_eq!(classify_op("MatMul"), OperatorClass::Gemm);
    assert_eq!(classify_op("Softmax"), OperatorClass::Reduction);
    assert_eq!(classify_op("Conv"), OperatorClass::Gemm);
}

#[test]
fn null_time_source_is_zero() {
    let ts = NullTimeSource;
    // Even across many calls NullTimeSource returns 0.
    for _ in 0..1_000 {
        assert_eq!(ts.now_us(), 0);
    }
}

#[cfg(feature = "std")]
#[test]
fn std_time_source_advances() {
    let ts = StdTimeSource::new();
    let a = ts.now_us();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let b = ts.now_us();
    assert!(b > a, "StdTimeSource did not advance after 2ms sleep");
}

#[test]
fn default_budget_matches_const() {
    let a = OperatorBudget::default();
    assert_eq!(a.elementwise_us, OperatorBudget::DEFAULT.elementwise_us);
    assert_eq!(a.reduction_us, OperatorBudget::DEFAULT.reduction_us);
    assert_eq!(a.gemm_us, OperatorBudget::DEFAULT.gemm_us);
}

#[test]
fn budget_boundary_ok() {
    let b = OperatorBudget::DEFAULT;
    // Exactly at budget — still Ok.
    assert_eq!(b.check(OperatorClass::Elementwise, 1_000), BudgetResult::Ok);
    assert_eq!(b.check(OperatorClass::Reduction, 10_000), BudgetResult::Ok);
    assert_eq!(b.check(OperatorClass::Gemm, 100_000), BudgetResult::Ok);
}

#[test]
fn budget_warning_path() {
    let b = OperatorBudget::DEFAULT;
    // Over budget but under 2x: Warning.
    assert_eq!(
        b.check(OperatorClass::Elementwise, 1_500),
        BudgetResult::Warning
    );
    assert_eq!(b.check(OperatorClass::Gemm, 150_000), BudgetResult::Warning);
}

#[test]
fn budget_soft_limit_path() {
    let b = OperatorBudget::DEFAULT;
    assert_eq!(
        b.check(OperatorClass::Elementwise, 5_000),
        BudgetResult::SoftLimit
    );
    assert_eq!(
        b.check(OperatorClass::Gemm, 500_000),
        BudgetResult::SoftLimit
    );
}

#[test]
fn budget_hard_limit_path() {
    let b = OperatorBudget::DEFAULT;
    assert_eq!(
        b.check(OperatorClass::Elementwise, 20_000),
        BudgetResult::HardLimit
    );
    assert_eq!(
        b.check(OperatorClass::Gemm, 2_000_000),
        BudgetResult::HardLimit
    );
}

#[test]
fn custom_budget_tighter_than_default() {
    let tight = OperatorBudget {
        elementwise_us: 10,
        reduction_us: 100,
        gemm_us: 1_000,
        attention_us: 5_000,
        gpu_kernel_us: 10_000,
        soft_multiplier: 2,
        hard_multiplier: 5,
    };
    assert_eq!(tight.check(OperatorClass::Elementwise, 5), BudgetResult::Ok);
    assert_eq!(
        tight.check(OperatorClass::Elementwise, 15),
        BudgetResult::Warning
    );
    assert_eq!(
        tight.check(OperatorClass::Elementwise, 25),
        BudgetResult::SoftLimit
    );
    assert_eq!(
        tight.check(OperatorClass::Elementwise, 100),
        BudgetResult::HardLimit
    );
}

#[test]
fn inference_profile_starts_empty() {
    let p = InferenceProfile::default();
    assert_eq!(p.total_us, 0);
    assert!(p.operators.is_empty());
    assert!(!p.hard_limit_aborted);
}
