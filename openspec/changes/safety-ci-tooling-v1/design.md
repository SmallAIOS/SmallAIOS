## Context

The CI pipeline (`.github/workflows/ci.yml`) runs ~27 jobs. The `change-gates` meta-job gates PR mergeability on a subset of these. Current gates: format, clippy, tests (3 variants), builds (4 targets), docker, semver title check, cargo-deny, cycle check. Advisory (non-gating): geiger, Kani, Miri, SPIN, fuzz, mutation, coverage.

The existing `codecov.yml` configures a 93% project target and 90% patch target via the Codecov external service. But Codecov is an external dependency — if the service is down or misconfigured, coverage gates silently pass.

## Goals / Non-Goals

**Goals:**
- Detect API breakage before merge (cargo-semver-checks)
- Enforce dependency audit trail for DO-178C traceability (cargo-vet)
- Catch additional UB beyond what Miri finds (cargo-careful)
- Local coverage gate independent of external services (cargo-llvm-cov)
- All new checks integrated into change-gates for PR blocking

**Non-Goals:**
- Replacing existing tools (these complement what's there)
- Making advisory checks into gates (Kani, Miri, fuzz stay advisory)
- MIRAI or Prusti integration (evaluate later — too experimental)
- Modifying the pre-1.0 semver rules (cargo-semver-checks enforces Rust API compatibility, not the project's semver policy)

## Decisions

### D1: cargo-semver-checks — Gate on Non-Breaking PRs, Advisory on Breaking

`cargo-semver-checks` compares the PR branch against the base branch to detect API-incompatible changes (removed public items, changed signatures, etc.).

**Behavior:**
- If the PR title contains `!` (breaking change indicator), the check is advisory (warn but pass)
- If the PR title does NOT contain `!`, the check gates — API breakage is unintentional and must be fixed or the PR title updated
- Uses `cargo-semver-checks check-release --baseline-rev origin/develop`

**Why conditional gating:** Pre-1.0, breaking changes happen. But they must be *intentional*. An accidental removal of a public function should be caught.

### D2: cargo-vet — Bootstrap with `cargo vet init`, Require Audits for New Deps

`cargo-vet` maintains an audit trail of who reviewed each dependency version. For DO-178C, all external code must have documented review.

**Bootstrap approach:**
1. Run `cargo vet init` to create `supply-chain/` directory
2. Run `cargo vet certify` to trust existing dependencies (bulk import — these are already in production)
3. Going forward, new dependencies or version bumps require `cargo vet certify` or an exemption

**CI behavior:** `cargo vet check` fails if any dependency lacks an audit entry. This is a gate — unvetted dependencies must not reach develop.

### D3: cargo-careful — Advisory Initially

`cargo-careful` runs the test suite with extra runtime checks enabled (debug assertions in std, extra alignment checks, etc.). It catches UB that Miri misses in some patterns.

**Why advisory:** It's slower than regular tests and may produce false positives on edge cases in `no_std` code. Start advisory, promote to gate after one release cycle of clean runs.

### D4: cargo-llvm-cov --fail-under — Hard Local Gate

Add a CI job that runs `cargo llvm-cov --fail-under-lines 80` as a backstop independent of Codecov. The threshold starts at 80% (below the Codecov 93% target) to avoid false failures while the new operators stabilize, then ratchet up.

**Why 80% not 93%:** The 93% target in codecov.yml applies to the existing codebase. New code (29 operators, executor) needs time to reach full coverage. Starting at 80% catches regressions without blocking the ONNX runtime work. Ratchet to 90% → 93% as tests are added.

### D5: Change Gates Integration

Add new jobs to the `change-gates` `needs` list:
- `cargo-semver-checks` — conditional gate (see D1)
- `cargo-vet` — hard gate
- `cargo-careful` — NOT in gates initially (advisory)
- `coverage-threshold` — hard gate

Updated needs: `[check-format, clippy, test, test-formal-gate, build-x86_64, build-aarch64, build-riscv64, build-jetson, docker-build-local, semver-check, cargo-deny, check-cycles, semver-api-check, cargo-vet-check, coverage-threshold]`

## Risks / Trade-offs

**[Risk] cargo-vet bootstrap is labor-intensive** — Many existing dependencies need initial certification. Mitigation: Use `cargo vet import` to trust crates.io audits from Mozilla, Google, and other trusted publishers. Then `cargo vet certify` the remainder in one batch.

**[Risk] cargo-semver-checks false positives** — Pre-1.0 workspace crates with `#[doc(hidden)]` items or internal-only APIs may trigger false breakage reports. Mitigation: Use `--package` flag to check only public-facing crates, exclude internal arch crates.

**[Risk] cargo-careful incompatibility with no_std** — `cargo-careful` rebuilds std with debug assertions, which doesn't apply to `no_std` crates. Mitigation: Run only on host-testable crates (same list as `just test`), not bare-metal targets.

**[Trade-off] Coverage threshold below Codecov target** — Starting at 80% feels lenient. But it's a floor, not a target. Codecov still enforces 93% externally. The local gate is a safety net for when Codecov is unavailable.

## Open Questions

- **Q1:** Should `cargo-vet` trust all existing deps immediately, or should we audit the top-10 critical deps (security, onnx-rt, kernel) individually?
- **Q2:** Should `cargo-careful` run on every PR or only on pushes to develop/main (to reduce CI cost)?
