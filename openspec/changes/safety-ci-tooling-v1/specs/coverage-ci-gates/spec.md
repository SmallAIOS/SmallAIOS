## MODIFIED Requirements

### Requirement: CI Coverage Gate
The CI pipeline SHALL fail PRs that cause coverage regression below configured thresholds, using both Codecov external service and a local cargo-llvm-cov gate.

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

#### Scenario: Local coverage gate as backstop
- **WHEN** the Codecov external service is unavailable or misconfigured
- **THEN** the local `cargo-llvm-cov --fail-under-lines` check MUST still enforce the minimum threshold
- **AND** the PR MUST NOT be mergeable if the local gate fails
