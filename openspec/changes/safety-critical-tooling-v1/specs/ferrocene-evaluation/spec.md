## ADDED Requirements

### Requirement: Ferrocene compiler compatibility assessment
The project SHALL produce a compatibility report documenting Ferrocene's support for all build targets (x86_64-unknown-none, aarch64-unknown-none, riscv64gc-unknown-none-elf), nightly features used (`naked_functions`, `asm`, `build-std`), and any gaps requiring workarounds.

#### Scenario: Build all crates with Ferrocene
- **WHEN** the Ferrocene toolchain is used to build the workspace
- **THEN** the compatibility report SHALL document which crates build successfully and which fail, with specific error details

#### Scenario: Nightly feature audit
- **WHEN** the codebase is scanned for nightly-only features
- **THEN** each usage SHALL be documented with its Ferrocene support status (supported/unsupported/workaround-available)

### Requirement: Ferrocene migration path documentation
The project SHALL document a migration path from nightly Rust to Ferrocene, including estimated effort, feature gaps, and commercial license requirements.

#### Scenario: Migration decision documented
- **WHEN** the evaluation is complete
- **THEN** the report SHALL include a go/no-go recommendation with rationale
