## 1. Create sched-types Crate

- [x] 1.1 Create `sched-types/Cargo.toml` with no_std, no dependencies
- [x] 1.2 Create `sched-types/src/lib.rs` with `OperatorClass`, `BudgetResult`, `OperatorBudget`
- [x] 1.3 Move `check()` method and `DEFAULT` const verbatim from kernel/onnx-rt
- [x] 1.4 Add `#![no_std]` and `#![forbid(unsafe_code)]`
- [x] 1.5 Add unit tests for `OperatorBudget::check()` covering all 4 BudgetResult variants
- [x] 1.6 Add `"sched-types"` to workspace members in root `Cargo.toml`

## 2. Refactor kernel

- [x] 2.1 Add `smallaios-sched-types = { path = "../sched-types" }` to `kernel/Cargo.toml`
- [x] 2.2 Replace local definitions in `kernel/src/sched/executor.rs` with `pub use smallaios_sched_types::{OperatorClass, BudgetResult, OperatorBudget};`
- [x] 2.3 Run `cargo test -p smallaios-kernel` — all 525 tests must still pass
- [x] 2.4 Verify no API surface change: search for `kernel::sched::executor::OperatorClass` usages workspace-wide

## 3. Refactor onnx-rt

- [x] 3.1 Add `smallaios-sched-types = { path = "../sched-types" }` to `onnx-rt/Cargo.toml`
- [x] 3.2 Replace local definitions in `onnx-rt/src/profile.rs` with `pub use smallaios_sched_types::{OperatorClass, BudgetResult, OperatorBudget};`
- [x] 3.3 Keep the `TimeSource`, `NullTimeSource`, `StdTimeSource`, `InferenceProfile`, `OperatorMeasurement`, `classify_op` types in profile.rs (they're not duplicates)
- [x] 3.4 Run `cargo test -p smallaios-onnx-rt --features std` — all tests must pass

## 4. Remove SonarCloud Exclusion

- [x] 4.1 Remove the `sonar.cpd.exclusions=onnx-rt/src/profile.rs` rule from `sonar-project.properties`
- [x] 4.2 Remove the explanatory comment block

## 5. Documentation

- [x] 5.1 Update `docs/architecture.md` to add `sched-types` to Layer 0 alongside kernel/security/compute
- [x] 5.2 Update CLAUDE.md workspace description (20 → 21 crates, Layer 0 list)
- [x] 5.3 Update `Justfile` `host_crates` to include `smallaios-sched-types`

## 6. Validation

- [x] 6.1 `just fmt` clean
- [x] 6.2 `just clippy --all-targets` clean
- [x] 6.3 `just test` all passing
- [x] 6.4 Cycle check: sched-types is leaf (no deps), kernel/onnx-rt depend on it
- [x] 6.5 SonarCloud no longer flags profile.rs for duplication
