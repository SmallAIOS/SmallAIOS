## ADDED Requirements

### Requirement: API Breakage Detection in CI
The CI pipeline SHALL detect accidental API-incompatible changes by comparing the PR branch against the base branch using cargo-semver-checks.

#### Scenario: Unintentional API breakage blocks PR
- **WHEN** a PR does not contain `!` in the title (non-breaking change)
- **AND** cargo-semver-checks detects removed or changed public API items
- **THEN** the CI check MUST fail
- **AND** the failure message MUST list the specific breaking changes detected

#### Scenario: Intentional breaking change passes
- **WHEN** a PR title contains `!` (e.g., `feat!: rename session API`)
- **AND** cargo-semver-checks detects API-incompatible changes
- **THEN** the CI check MUST pass with an advisory warning listing the changes

#### Scenario: No API changes passes cleanly
- **WHEN** cargo-semver-checks detects no API-incompatible changes
- **THEN** the CI check MUST pass regardless of PR title

### Requirement: Semver Check in Pre-Commit
The pre-commit hook SHALL optionally run cargo-semver-checks for fast local feedback.

#### Scenario: Local semver check
- **WHEN** a developer runs `just check` or the pre-commit hook triggers
- **AND** cargo-semver-checks is installed
- **THEN** the hook MUST run semver checks against the current base branch
- **AND** MUST report any detected breaking changes as a warning
