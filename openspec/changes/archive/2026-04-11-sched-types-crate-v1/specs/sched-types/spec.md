## ADDED Requirements

### Requirement: Layer 0 Sched Types Crate
The system SHALL provide a `sched-types` crate at Layer 0 (Foundation) containing scheduler primitive types usable by any other crate.

#### Scenario: Crate is no_std and alloc-free
- **WHEN** the `smallaios-sched-types` crate is built
- **THEN** it MUST compile with `#![no_std]`
- **AND** MUST NOT depend on `alloc` or any other crate
- **AND** MUST NOT contain any `unsafe` code

#### Scenario: Exposes OperatorClass enum
- **WHEN** a crate depends on `smallaios-sched-types`
- **THEN** `OperatorClass` MUST be importable with variants `Elementwise`, `Reduction`, `Gemm`, `Attention`, `GpuKernel`
- **AND** the enum MUST be `#[repr(u8)]` for stable ABI

#### Scenario: Exposes BudgetResult enum
- **WHEN** a crate depends on `smallaios-sched-types`
- **THEN** `BudgetResult` MUST be importable with variants `Ok`, `Warning`, `SoftLimit`, `HardLimit`

#### Scenario: Exposes OperatorBudget struct
- **WHEN** a crate depends on `smallaios-sched-types`
- **THEN** `OperatorBudget` MUST be importable with all 7 fields and the `DEFAULT` const
- **AND** the `check()` method MUST return the correct `BudgetResult` for each input range
