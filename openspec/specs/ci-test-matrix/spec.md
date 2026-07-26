# ci-test-matrix Specification

## Purpose

Single-source crate/feature test matrix (ci/test-matrix.toml) driving the CI unit-test, clippy, and coverage gates and the Justfile test recipes, with executed-test floors so no workspace tests run dark.

## Requirements

### Requirement: Single Source of Truth for the Test Matrix
The workspace SHALL define its host-testable crate/feature matrix in exactly one checked-in file (`ci/test-matrix.toml`), and every consumer of a crate list for testing or linting (CI unit-test job, CI clippy job, Justfile test recipes, release pre-flight) SHALL derive its list from that file rather than carrying its own copy.

#### Scenario: CI and Justfile consume the same matrix
- **WHEN** the unit-test CI job and `just test` execute
- **THEN** both run the crate/feature groups defined in `ci/test-matrix.toml`, with no independently-maintained crate list in either consumer

#### Scenario: Matrix edit propagates everywhere
- **WHEN** a crate or feature flag is added to a group in `ci/test-matrix.toml`
- **THEN** the next CI run and the next local `just test-all` both include it without any other file being edited

### Requirement: Workspace Coverage Verification
The matrix tooling SHALL provide a verification mode that cross-checks `ci/test-matrix.toml` against the actual `cargo metadata` workspace members, and CI SHALL run it as a blocking step. Every workspace member MUST be either (a) covered by at least one matrix group or (b) listed in the matrix file's exclusion table with a non-empty human-readable reason.

#### Scenario: New crate cannot silently skip CI
- **WHEN** a new workspace member is added without classifying it in `ci/test-matrix.toml`
- **THEN** the matrix verification step fails the pipeline until the crate is added to a group or excluded with a reason

#### Scenario: Documented exclusions pass
- **WHEN** a crate is listed in the exclusion table with a reason (e.g. "inline asm requires matching host arch")
- **THEN** verification passes and the exclusion reason is visible in the verification step's log output

### Requirement: No Vacuous Test Gates
The matrix runner SHALL parse the executed-test counts of every cargo test invocation it performs and SHALL fail the group when the total executed count is zero. Groups MAY declare a `min_tests` floor; when declared, the runner SHALL fail the group if fewer tests execute.

#### Scenario: Zero-test regression turns the gate red
- **WHEN** a feature or filter change causes a matrix group to compile and execute zero tests
- **THEN** the group's CI job fails with a message stating the group name and the zero executed-test count

#### Scenario: Suite shrinks below its declared floor
- **WHEN** a group with `min_tests = 150` executes 40 tests
- **THEN** the group's CI job fails, reporting executed count versus the declared floor

### Requirement: Feature-Gated Suites Execute in CI
The recovered dark suites SHALL each execute in a blocking CI matrix group: fs with `fs-flash`, `fs-flash-mock`, and `fs-on-disk-mounts` features (littlefs, overlay, and squashfs conformance tests), posix with its fs-flash features, tls-client, audit-export with its `bearer` feature, and the host-runnable arch crate test targets.

#### Scenario: littlefs suite is CI-visible
- **WHEN** a change breaks a `fs-flash`-gated littlefs test
- **THEN** a blocking CI job fails before merge

#### Scenario: squashfs conformance gate is real
- **WHEN** the fs-interop gate runs
- **THEN** its squashfs conformance tests execute with a nonzero count asserted by the runner

#### Scenario: tls-client and audit-export are gated
- **WHEN** a change breaks a tls-client handshake test or an audit-export pipeline test
- **THEN** a blocking CI job fails before merge

### Requirement: Bounded CI Impact
New matrix groups SHALL run as parallel CI matrix entries with per-group caching, and the matrix design SHALL NOT introduce an additional full-workspace build.

#### Scenario: Groups parallelize
- **WHEN** the test matrix executes in CI
- **THEN** each group runs as its own matrix job in parallel, and no single job's wall-time exceeds the pre-change unit-test job by more than a small constant factor attributable to its own group's compilation
