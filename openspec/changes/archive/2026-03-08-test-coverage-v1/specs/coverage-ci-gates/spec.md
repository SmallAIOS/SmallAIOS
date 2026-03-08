# Delta for coverage-ci-gates

## ADDED Requirements

### Requirement: Codecov Configuration File
A codecov.yml file SHALL be present at the repository root configuring coverage targets, flags, and exclusions.

#### Scenario: Project coverage target
- GIVEN the codecov.yml configuration
- THEN the project coverage target MUST be set to at least 93%
- AND the threshold for acceptable fluctuation MUST be no more than 1%

#### Scenario: Patch coverage target
- GIVEN the codecov.yml configuration
- THEN the patch coverage target for new/changed lines MUST be set to at least 90%

#### Scenario: Per-crate coverage flags
- GIVEN the codecov.yml configuration
- THEN coverage flags MUST be defined for at least: kernel, security, onnx-rt, net, peripheral, container, ipc, bus, usb, sdr
- AND each flag MUST specify the correct source path for its crate
- AND carryforward MUST be enabled for each flag

#### Scenario: Path exclusions
- GIVEN the codecov.yml configuration
- THEN the following paths MUST be excluded from coverage reporting: arch/**, container/src/main.rs, bench/**, fuzz/**, docs/**
- AND exclusions MUST NOT inadvertently hide testable code

### Requirement: CI Coverage Gate
The CI pipeline SHALL fail PRs that cause coverage regression below configured thresholds.

#### Scenario: Coverage regression blocks PR
- GIVEN a PR that reduces overall project coverage below the codecov.yml project target
- WHEN the Codecov status check runs
- THEN the status check MUST report failure
- AND the PR MUST NOT be mergeable while the status check fails

#### Scenario: Low patch coverage blocks PR
- GIVEN a PR where newly added or changed lines have less than 90% coverage
- WHEN the Codecov status check runs
- THEN the patch coverage status check MUST report failure
- AND the failure message MUST identify which files have insufficient coverage

### Requirement: Coverage Reporting on PRs
Codecov SHALL post coverage reports as PR comments.

#### Scenario: PR comment content
- GIVEN a PR with coverage data uploaded to Codecov
- WHEN the coverage analysis completes
- THEN a PR comment MUST be posted showing the diff coverage, per-flag coverage, and affected files
- AND the comment MUST only appear when coverage changes exist (require_changes: true)

### Requirement: Coverage Target Ratcheting
The project coverage target SHALL be increased as overall coverage improves.

#### Scenario: Ratcheting mechanism
- GIVEN the current project coverage exceeds the codecov.yml target by more than 2%
- WHEN the next coverage improvement PR is merged
- THEN the codecov.yml project target SHOULD be updated to within 1% of the new actual coverage
- AND this update MUST be a deliberate manual commit (not automated)

### Requirement: Coverage Exclusion Annotations
Code that cannot be meaningfully tested SHALL be excluded with documented justification.

#### Scenario: Exclusion annotation format
- GIVEN a line or block of code that cannot be tested (e.g., hardware-only error path)
- WHEN the line is excluded from coverage
- THEN the exclusion MUST use the standard annotation (LCOV_EXCL_LINE or LCOV_EXCL_START/STOP)
- AND a comment MUST accompany the annotation explaining why the code is untestable
- AND exclusions MUST NOT be used to hide code that could be tested with mocks
