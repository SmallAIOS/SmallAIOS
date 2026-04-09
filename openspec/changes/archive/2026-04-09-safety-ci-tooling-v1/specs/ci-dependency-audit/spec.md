## ADDED Requirements

### Requirement: Dependency Audit Trail Enforcement
The CI pipeline SHALL enforce that every third-party dependency version has a recorded audit entry using cargo-vet.

#### Scenario: All dependencies audited
- **WHEN** `cargo vet check` runs in CI
- **AND** all dependency versions have audit entries or exemptions
- **THEN** the check MUST pass

#### Scenario: Unaudited dependency blocks PR
- **WHEN** a PR adds or bumps a dependency that lacks an audit entry
- **AND** no exemption has been recorded for that version
- **THEN** `cargo vet check` MUST fail
- **AND** the failure message MUST identify the unaudited crate and version

#### Scenario: Trusted publisher imports
- **WHEN** a dependency has audits published by trusted organizations (Mozilla, Google)
- **THEN** cargo-vet MUST accept those audits via `imports` in the config
- **AND** the dependency MUST NOT require a local audit entry

### Requirement: Audit Bootstrap for Existing Dependencies
The project SHALL initialize cargo-vet with trust entries for all current dependencies.

#### Scenario: Initial bootstrap
- **WHEN** `cargo vet init` is run for the first time
- **THEN** a `supply-chain/` directory MUST be created with config.toml and audits.toml
- **AND** all existing dependencies MUST be certified or exempted
- **AND** the bootstrap MUST be committed as a single auditable commit
