# ci-test-gates-v1 Proposal

## Why

The 2026-07-03 structure/quality audit (adversarially verified, test-structure dimension) found that **~1,084 tests exist in the workspace but never execute in any CI gate** — roughly 18% growth over the 5,916 tests CI does run — and one gate (fs-interop) passes vacuously while running 0 of its 40 conformance tests. For a project targeting DO-178C DAL A, green gates that measure nothing are worse than missing gates: they create false verification evidence. The root cause is structural, not accidental: three hand-curated, independently-drifting crate lists (`.github/workflows/ci.yml`, `Justfile`, release workflow) plus feature-gated suites (`fs-flash`, overlay, on-disk mounts) that no invocation ever enables.

## What Changes

- **Single-source the host-testable crate/feature matrix.** One checked-in matrix definition (consumed by both the Justfile recipes and CI workflow) replaces the three divergent hand-curated lists, so a new crate or feature suite cannot silently miss CI.
- **Light up the six dark suites** (counts from the audit, all verified passing locally under `--all-features`):
  - fs overlay: 172 in-src + 184 conformance tests
  - arch crates: 224 in-module tests (host-runnable subset)
  - tls-client: 165 tests
  - fs-flash / littlefs: 162 tests (`fs-flash`, `fs-flash-mock` features)
  - audit-export: 137 tests (`bearer` feature)
  - squashfs conformance: 40 tests currently compiled out of the fs-interop gate
- **Make vacuous gates impossible:** test-executing gates assert a nonzero executed-test count (and, where cheap, a minimum expected count) so a feature/filter regression turns the gate red instead of silently green.
- **Keep CI wall-time bounded:** new suites run as parallel matrix entries reusing existing cache keys; no redundant full-workspace rebuilds.
- **Re-baseline the coverage gate:** enabling these suites moves the `cargo-llvm-cov` denominator; verify the 80% line threshold still passes and record the new baseline (ratchet plan unchanged).

## Capabilities

### New Capabilities
- `ci-test-matrix`: the workspace's host-testable crate/feature matrix is defined in exactly one place, every test suite in the workspace has an executing CI gate (or a documented exclusion with reason), and test-executing gates fail on zero executed tests.

### Modified Capabilities

<!-- none — coverage-ci-gates requirements (regression blocking, codecov config) are unchanged; only the measured baseline moves, which is not a spec-level behavior change -->

## Impact

- `.github/workflows/ci.yml` — unit-test/clippy jobs consume the shared matrix; new feature-matrix entries for fs/posix/tls-client/audit-export/arch; fs-interop gate gains a test-count assertion.
- `Justfile` — `test` recipe and `host_crates` derive from the same matrix source; new `test-features` recipe mirrors CI locally.
- New matrix definition file (e.g. `ci/test-matrix.*` or generated include) + small validation script.
- `codecov.yml` / coverage job — baseline shift only.
- No production crate code changes; no API changes. Risk is CI-config-only and reversible per-job.
- Relates to: PR #232 (verification-tooling honesty fixes), PR #233 (verified the dark suites pass under `--all-features`), `embedded-flash-fs-v1` (its 162 tests become CI-visible), `verifiable-audit-log-v1` (audit-export suite becomes CI-visible).
