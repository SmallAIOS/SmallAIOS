## Why

SmallAIOS targets DO-178C DAL A certification. The CI pipeline already has strong coverage (format, clippy, tests, cargo-deny, geiger, Kani, Miri, TLA+, SPIN, fuzz, mutation testing) but is missing several tools critical for safety-critical development processes: API breakage detection, dependency audit trail, extra UB checks, and hard coverage thresholds. These gaps mean accidental breaking changes, unreviewed dependencies, and coverage regressions can reach `develop` without detection.

## What Changes

- **Add `cargo-semver-checks`** CI job: detect accidental API breaking changes between the PR branch and base. Gate PRs — breaking changes must be intentional (annotated with `!` in PR title per semver rules).
- **Add `cargo-vet`** CI job: enforce that every third-party dependency has a recorded audit. Required for DO-178C traceability — all external code must have a review trail.
- **Add `cargo-careful`** CI job: run tests with extra UB checks (debug assertions in std, stricter than Miri for some patterns). Advisory initially, promote to gate.
- **Add `cargo-llvm-cov --fail-under`** threshold enforcement: fail CI if overall coverage drops below the configured minimum (93% per existing codecov.yml). Complements Codecov's external check with a local gate.
- **Add all new jobs to the `change-gates` `needs` list** so they block PR mergeability.
- **Update pre-commit hooks** to include `cargo-semver-checks` for fast local feedback.

## Capabilities

### New Capabilities
- `ci-semver-checks`: Automated API breakage detection in CI with PR gating
- `ci-dependency-audit`: Dependency audit trail enforcement via cargo-vet
- `ci-careful-testing`: Extra undefined behavior detection via cargo-careful
- `ci-coverage-threshold`: Local coverage threshold enforcement independent of Codecov

### Modified Capabilities
- `coverage-ci-gates`: Add local `cargo-llvm-cov --fail-under` threshold as a CI job independent of Codecov external service

## Impact

- **CI:** 4 new jobs in `.github/workflows/ci.yml`, updated `change-gates` dependencies
- **Config:** New `supply-chain/` directory for cargo-vet audit files, updated `deny.toml` if needed
- **Dev dependencies:** `cargo-semver-checks`, `cargo-vet`, `cargo-careful` added to recommended tooling
- **Pre-commit:** `cargo-semver-checks` added for local breakage detection
