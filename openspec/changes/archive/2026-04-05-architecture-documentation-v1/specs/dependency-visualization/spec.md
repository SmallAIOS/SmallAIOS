## MODIFIED Requirements

### Requirement: CI acyclicity enforcement
The CI pipeline SHALL include a `cargo-modules dependencies --acyclic` check that verifies no module-level cyclic dependencies exist.

#### Scenario: Acyclicity check passes
- **WHEN** CI runs on a PR with no cyclic module dependencies
- **THEN** the acyclicity check passes

#### Scenario: Acyclicity check detects cycle
- **WHEN** a PR introduces a module-level cyclic dependency
- **THEN** the acyclicity check fails with a descriptive error

### Requirement: DSM metrics in CI artifacts
The CI dependency-analysis job SHALL run `scripts/dsm-analysis.py` and include the metrics report in the uploaded analysis artifacts.

#### Scenario: Metrics artifact
- **WHEN** CI dependency-analysis job completes
- **THEN** `build/analysis/dsm-metrics.json` is included in the uploaded artifact
