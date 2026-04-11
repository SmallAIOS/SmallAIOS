## MODIFIED Requirements

### Requirement: Operator-Level Scheduler Integration
The kernel scheduler SHALL re-export `OperatorClass`, `BudgetResult`, and `OperatorBudget` from the `smallaios-sched-types` crate so existing kernel code can use them via the existing `kernel::sched::executor` paths.

#### Scenario: Re-export preserves existing imports
- **WHEN** kernel code imports `kernel::sched::executor::OperatorClass`
- **THEN** it MUST resolve to `smallaios_sched_types::OperatorClass`
- **AND** existing usage MUST compile without modification

#### Scenario: Yield between operators
- **WHEN** the runtime executes an inference graph
- **THEN** it MUST yield to the scheduler after each operator completes
- **AND** the yield MUST allow higher-priority tasks (SYSTEM, IPC) to execute before inference resumes
