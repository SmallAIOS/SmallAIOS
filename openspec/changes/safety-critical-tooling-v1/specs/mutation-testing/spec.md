## ADDED Requirements

### Requirement: Mutation testing on safety-critical modules
The crypto, memory management, and scheduler modules SHALL be tested with cargo-mutants. The mutation score SHALL be tracked over time.

#### Scenario: Mutation testing on crypto module
- **WHEN** cargo-mutants runs on `security/src/crypto/`
- **THEN** the mutation score SHALL be reported and any surviving mutants SHALL be documented

#### Scenario: Mutation testing results in CI
- **WHEN** mutation testing runs (on-demand or scheduled)
- **THEN** results SHALL be available as a CI artifact with per-function mutation scores
