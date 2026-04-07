## ADDED Requirements

### Requirement: Local Coverage Threshold Gate
The CI pipeline SHALL enforce a minimum line coverage threshold using cargo-llvm-cov independent of external coverage services.

#### Scenario: Coverage above threshold passes
- **WHEN** `cargo llvm-cov` reports line coverage at or above the configured threshold
- **THEN** the CI check MUST pass

#### Scenario: Coverage below threshold fails
- **WHEN** `cargo llvm-cov` reports line coverage below the configured threshold
- **THEN** the CI check MUST fail
- **AND** the failure message MUST report the actual coverage percentage and the required threshold

#### Scenario: Threshold ratcheting
- **WHEN** the actual project coverage exceeds the threshold by more than 5%
- **THEN** the threshold SHOULD be manually increased to within 2% of the actual coverage
- **AND** the increase MUST be committed as a deliberate change

### Requirement: Coverage Threshold Configuration
The coverage threshold SHALL be configurable and start conservatively.

#### Scenario: Initial threshold
- **WHEN** the coverage threshold gate is first introduced
- **THEN** the threshold MUST be set to 80% line coverage
- **AND** MUST be documented with a ratcheting schedule toward the Codecov 93% target

#### Scenario: Threshold stored in CI config
- **WHEN** the threshold needs to be updated
- **THEN** it MUST be stored as a clear value in the CI workflow file or a dedicated config file
- **AND** MUST NOT be embedded in opaque scripts
