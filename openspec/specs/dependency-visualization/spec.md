# dependency-visualization Specification

## Purpose
TBD - created by archiving change dependency-analysis-v1. Update Purpose after archive.
## Requirements
### Requirement: Crate-level dependency graph generation
The system SHALL provide a Makefile target `make depgraph` that generates a workspace-level dependency graph using `cargo-depgraph`, filtered to workspace members only, and outputs DOT and SVG files to `build/analysis/`.

#### Scenario: Generate crate dependency graph
- **WHEN** developer runs `make depgraph`
- **THEN** `build/analysis/crate-deps.dot` and `build/analysis/crate-deps.svg` are created containing all 18 workspace crates and their normal dependencies

#### Scenario: GraphViz not installed
- **WHEN** developer runs `make depgraph` without GraphViz installed
- **THEN** DOT file is still generated and a warning is printed that SVG generation was skipped

### Requirement: Module-level dependency graph generation
The system SHALL provide a Makefile target `make modgraph` that generates module-level dependency graphs for each host-testable crate using `cargo-modules`, outputting DOT files to `build/analysis/modules/`.

#### Scenario: Generate module graphs for all crates
- **WHEN** developer runs `make modgraph`
- **THEN** a DOT file is generated for each host-testable crate at `build/analysis/modules/<crate-name>.dot`

#### Scenario: Generate module graph for single crate
- **WHEN** developer runs `make modgraph CRATE=smallaios-kernel`
- **THEN** only `build/analysis/modules/smallaios-kernel.dot` is generated

### Requirement: Module-level cycle detection
The system SHALL provide a Makefile target `make arch-check` that runs `cargo-modules` with `--acyclic` on each host-testable crate and reports any module-level cycles found.

#### Scenario: No cycles detected
- **WHEN** developer runs `make arch-check` and no module-level cycles exist
- **THEN** the command exits 0 and prints "OK: no module-level cycles detected"

#### Scenario: Cycles detected
- **WHEN** developer runs `make arch-check` and module-level cycles exist in a crate
- **THEN** the command prints the cycle path and the crate name, and exits non-zero

### Requirement: CI dependency graph artifacts
The system SHALL include a CI job that generates crate-level dependency graphs and runs module-level cycle detection on pull requests, uploading graph artifacts.

#### Scenario: PR triggers dependency analysis
- **WHEN** a pull request is opened or updated
- **THEN** CI generates crate dependency DOT/SVG, runs module-level cycle detection, and uploads graph files as job artifacts

#### Scenario: Module cycle detected in CI
- **WHEN** CI detects a module-level cycle
- **THEN** the job logs a warning but does not fail the build (continue-on-error)

### Requirement: Generated artifacts excluded from git
All generated dependency graphs, SVG files, and analysis outputs SHALL be written to `build/analysis/` which is excluded from version control.

#### Scenario: Generated files not tracked
- **WHEN** developer runs `make depgraph` or `make modgraph`
- **THEN** no generated files appear in `git status`

