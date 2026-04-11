## MODIFIED Requirements

### Requirement: Per-Operator Budget Enforcement
The ONNX runtime SHALL use `OperatorClass`, `BudgetResult`, and `OperatorBudget` from the shared `smallaios-sched-types` crate rather than maintaining duplicate type definitions.

#### Scenario: Re-export from sched-types
- **WHEN** code imports `onnx_rt::profile::OperatorClass` (or similar)
- **THEN** it MUST resolve to `smallaios_sched_types::OperatorClass`
- **AND** existing API surface MUST be preserved via `pub use`

#### Scenario: No type duplication remains
- **WHEN** SonarCloud analyzes the workspace
- **THEN** `onnx-rt/src/profile.rs` MUST NOT trigger duplication warnings against `kernel/src/sched/executor.rs`
- **AND** the `sonar.cpd.exclusions` rule for `profile.rs` MUST be removable
