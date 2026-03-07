## ADDED Requirements

### Requirement: ModeCodeProcessor process cognitive complexity ≤ 15
The `ModeCodeProcessor::process()` function in `bus/src/mil1553/mode_code.rs` SHALL have cognitive complexity ≤ 15. The repeated broadcast/unicast response pattern SHALL be extracted into a `broadcast_response()` helper. All existing MIL-STD-1553 mode code tests SHALL continue to pass.

#### Scenario: ModeCodeProcessor process refactored below threshold
- **WHEN** SonarCloud analyzes `ModeCodeProcessor::process()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 42)

#### Scenario: Mode code processing behavior preserved
- **WHEN** existing MIL-STD-1553 mode code tests execute against the refactored code
- **THEN** all broadcast and unicast responses SHALL match expected values

### Requirement: Scheduler poll cognitive complexity ≤ 15
The `Scheduler::poll()` function in `bus/src/arinc429/scheduler.rs` SHALL have cognitive complexity ≤ 15. Scheduling phases SHALL be extracted into helper methods. All existing ARINC 429 scheduler tests SHALL continue to pass.

#### Scenario: Scheduler poll refactored below threshold
- **WHEN** SonarCloud analyzes `Scheduler::poll()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 20)

### Requirement: CanZenohAdapter transmit cognitive complexity ≤ 15
The `CanZenohAdapter::transmit()` function in `bus/src/can/adapter.rs` SHALL have cognitive complexity ≤ 15. Frame construction logic SHALL be extracted into a helper. All existing CAN adapter tests SHALL continue to pass.

#### Scenario: CanZenohAdapter transmit refactored below threshold
- **WHEN** SonarCloud analyzes `CanZenohAdapter::transmit()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 17)

### Requirement: No public API changes in bus crate
All extracted helper functions SHALL be private. No existing public types, traits, or function signatures in the `bus` crate SHALL change.

#### Scenario: Public API unchanged
- **WHEN** downstream crates compile against the refactored bus crate
- **THEN** compilation SHALL succeed without modification
