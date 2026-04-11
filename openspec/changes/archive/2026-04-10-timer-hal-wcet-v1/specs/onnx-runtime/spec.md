## MODIFIED Requirements

### Requirement: Operator-Level Scheduler Integration
The runtime SHALL insert mandatory scheduler yield points between every operator in the execution graph and support per-operator time budgets, with budget enforcement actually taking effect when profiling is enabled.

#### Scenario: Yield between operators
- **WHEN** the runtime executes an inference graph
- **THEN** it MUST yield to the scheduler after each operator completes
- **AND** the yield MUST allow higher-priority tasks (SYSTEM, IPC) to execute before inference resumes

#### Scenario: Per-operator timing with profile enabled
- **WHEN** `Session::run_with_profile()` is called
- **THEN** the runtime MUST measure each operator's wall-clock execution time via `TimeSource`
- **AND** MUST call `OperatorBudget::check()` with the measured time and operator class

#### Scenario: Operator budget exceeded warning
- **WHEN** an operator's execution time triggers `BudgetResult::Warning`
- **THEN** the runtime MUST log a warning with operator name, actual time, and budget
- **AND** MUST continue inference normally

#### Scenario: Operator hard timeout
- **WHEN** an operator's execution time triggers `BudgetResult::HardLimit`
- **THEN** the runtime MUST abort the inference and return `SessionError::ExecutionFailed`
- **AND** the error message MUST identify the offending operator and its measured time

#### Scenario: Profile disabled is zero-overhead
- **WHEN** `Session::run()` is called (without profiling)
- **THEN** no operator timing measurement SHALL occur
- **AND** no `OperatorBudget::check()` calls SHALL happen
