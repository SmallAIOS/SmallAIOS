## Why

In `timer-hal-wcet-v1` (PR #69), we duplicated `OperatorClass`, `BudgetResult`, and `OperatorBudget` from `kernel/src/sched/executor.rs` into `onnx-rt/src/profile.rs` because the 4-layer architecture prevents `onnx-rt` (Layer 1) from depending on `kernel` (Layer 0). This duplication caused a SonarCloud "Duplication on New Code" gate failure that we worked around by adding a `sonar.cpd.exclusions` rule.

The clean fix: extract the shared types into a new `sched-types` crate at Layer 0 (Foundation) that both `kernel` and `onnx-rt` depend on. This:
- Eliminates the duplication for real (no more Sonar exclusion needed)
- Establishes a pattern for future shared types between kernel and other Layer 1 crates
- Keeps the architecture layering clean

## What Changes

- Create new `sched-types/` crate at Layer 0 with `OperatorClass`, `BudgetResult`, `OperatorBudget` (and constants)
- Move the canonical definitions out of `kernel/src/sched/executor.rs` and into `sched-types`
- Re-export from kernel for backwards compatibility (`pub use sched_types::OperatorClass;`)
- Move the duplicated types out of `onnx-rt/src/profile.rs` and use `sched-types` directly
- Remove the `sonar.cpd.exclusions` rule for `profile.rs`
- Update `compute/Cargo.toml` and others if they need similar types in the future
- Update Layer 0 documentation in `docs/architecture.md` and CLAUDE.md

## Capabilities

### New Capabilities
- `sched-types`: Layer 0 crate exposing scheduler primitive types (OperatorClass, BudgetResult, OperatorBudget) for use by kernel, onnx-rt, and any other crate needing them

### Modified Capabilities
- `kernel-core`: re-exports types from sched-types instead of defining them
- `onnx-runtime`: depends on sched-types for budget enforcement types

## Impact

- **Code:** New `sched-types/` crate (~150 lines), refactor `kernel/src/sched/executor.rs` (re-exports), refactor `onnx-rt/src/profile.rs` (use sched-types instead of duplicating)
- **Cargo.toml:** New workspace member, new deps in kernel/onnx-rt/possibly compute
- **Architecture:** Layer 0 grows by one crate (3 → 4: kernel, security, compute, sched-types... wait, compute is at Layer 0 too. Let me recount.) Actually: Layer 0 = kernel, security, compute, sched-types (4 crates)
- **No behavior change:** pure refactor; all 3,200+ tests must continue to pass
- **SonarCloud:** can remove `sonar.cpd.exclusions` for profile.rs once duplication is gone
