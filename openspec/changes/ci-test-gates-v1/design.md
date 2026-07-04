# ci-test-gates-v1 Design

## Context

CI currently runs one `cargo test` invocation over a hand-curated 17-crate list (`.github/workflows/ci.yml` unit-test job), the Justfile `test` recipe carries a second copy of that list, and the release workflow a third. Feature-gated suites are never enabled anywhere: `fs-flash`/`fs-flash-mock` (littlefs, 162 tests), fs overlay + on-disk mounts (356 tests), `tls-client` (165), `audit-export` (137), arch in-module tests (224), and the fs-interop gate compiles 0 of its 40 squashfs conformance tests. All of these pass when actually run (verified 2026-07-03 with `--all-features` locally: 689 fs unit + ~800 integration/conformance tests green). The lists have already drifted twice before (#219 added auth/mgmt/fs/xtask after the same class of gap).

Constraints: DO-178C DAL-A posture means gates must produce true evidence; CI wall-time is already substantial (55 checks); arch crates contain `target_arch`-specific inline asm, so their host-testability depends on the runner architecture (verified: `arch/x86_64` fails `cargo check` on an ARM host).

## Goals / Non-Goals

**Goals:**
- One checked-in definition of the host-testable crate/feature matrix, consumed by both Justfile and CI.
- Structural un-driftability: a verification step that cross-checks the matrix against `cargo metadata` workspace members, so an unclassified crate fails CI.
- Every dark suite from the audit executes in a gate (or carries a documented exclusion with reason).
- Zero-executed-tests in any test group turns that gate red (vacuous-pass prevention).
- Bounded CI impact: new groups are parallel matrix entries with per-group caches.

**Non-Goals:**
- Fixing the 3 module cycles found by the repaired arch-check (separate backlog).
- Hardware-gated suites (Jetson GPU smoke) — unchanged.
- Advisory jobs (Kani, Miri, fuzz, mutation) — unchanged.
- Raising or lowering the coverage threshold — the 80% gate stays; only the measured baseline is re-recorded.

## Decisions

**D1 — Matrix source of truth: `ci/test-matrix.toml` + `scripts/test-matrix.py`.**
A declarative TOML file defines named *groups*: each group lists crates, feature flags, and optional `min_tests`. A single Python script (stdlib-only, mirroring `scripts/dsm-matrix.py` precedent) provides:
- `--emit gha` → JSON for a GitHub Actions `strategy.matrix` (setup job + `fromJSON`)
- `--run <group>` → executes the group's cargo invocations, parses `test result:` summary lines, fails on zero (or `< min_tests`) executed tests
- `--verify` → asserts every `cargo metadata` workspace member is either covered by some group or present in `[exclusions]` with a `reason` string
*Alternatives considered:* Justfile as source (CI calls `just`) — rejected: Actions can't build a parallel matrix from Justfile variables without shelling out anyway, and the release workflow doesn't use just. Generated YAML — rejected: generated-file drift is the disease we're curing.

**D2 — Groups (initial matrix).**
- `default`: today's 17-crate list, unchanged behavior
- `fs-features`: `smallaios-fs` with `fs-flash,fs-flash-mock,fs-on-disk-mounts` + overlay/conformance test targets (the 356 + 162 + 40)
- `posix-features`: `smallaios-posix` with `fs-flash-mock,fs-on-disk-mounts`
- `tls-client`: crate default features (165 tests; 9 network-e2e tests stay `#[ignore]`)
- `audit-export`: `bearer` feature (137 tests)
- `arch-x86_64-host`: `smallaios-arch-x86_64` in-module tests on the x86_64 ubuntu runner
- `arch-portable`: nvidia/amd/intel_gpu/apple/riscv64/aarch64 crates *iff* their test targets compile on x86_64 host — determined during implementation; non-compiling ones go to `[exclusions]` with reason "inline asm requires matching host arch" (revisit when an arm64 runner lands)
Clippy uses the union of group crates+features so lint coverage matches test coverage.

**D3 — Vacuous-gate prevention lives in the runner, not per-job shell.** `--run` counts executed tests from cargo's `test result:` lines; any group totaling 0 fails, and groups with `min_tests` fail below it. `min_tests` is set to ~90% of today's observed counts (drift headroom for legitimate test removals) for the six recovered suites. The fs-interop gate switches to `--run fs-features` (its 0/40 bug becomes structurally unrepresentable).

**D4 — Coverage job includes the new feature groups.** `cargo-llvm-cov` invocation extends with the fs/posix/tls-client/audit-export feature sets so the recovered tests count toward coverage. The 80% line gate is expected to hold (the recovered suites are test-dense); the observed new baseline is recorded in the change's tasks, and any surprise dip is resolved by scoped `codecov.yml`/llvm-cov exclusions — never by lowering the threshold.

**D5 — Release/Justfile convergence.** `just test` becomes `python3 scripts/test-matrix.py --run default`; a new `just test-all` runs every group locally. The release workflow's list is retired in favor of `--verify` + `just test-all` in the pre-release hook.

## Risks / Trade-offs

- [Dark tests fail on linux despite passing on macOS] → land groups as separate commits; a failing group ships behind `continue-on-error: true` with a tracking issue, never silently dropped from the matrix.
- [CI wall-time grows] → parallel matrix entries with per-group `rust-cache` keys; the six groups compile far less than the workspace build already cached; measure before/after on the PR itself.
- [Coverage gate dips below 80%] → measure on the PR; if it dips, exclude generated `*_test_vectors.rs` from the denominator (already Sonar-excluded precedent) rather than touching the threshold.
- [Python script becomes its own drift point] → `--verify` is itself a CI step; the script is ~150 lines stdlib-only with unit tests.
- [arch-portable group turns out empty on x86_64] → acceptable: exclusions are documented per-crate with reasons, and the requirement is "execute or document", not "execute unconditionally".

## Migration Plan

1. Land matrix file + script + `--verify` step (no behavior change to existing jobs).
2. Switch existing unit-test/clippy jobs to consume the matrix (`default` group) — output identical to today.
3. Add recovered-suite groups one commit each (fs-features, posix-features, tls-client, audit-export, arch-*).
4. Fix fs-interop gate to `--run fs-features`.
5. Extend coverage invocation; record new baseline.
6. Retire the release workflow's private list.
Rollback: each step is an independent CI-config commit; reverting any single commit restores the prior gate shape.

## Open Questions

- Does `arch-portable` yield any host-runnable crates on x86_64? (Resolved during implementation; drives exclusion list contents.)
- Should `xtask`/`coverage-probe`/`tools/dsm` join the matrix as a `tools` group? (Lean yes — cheap; decide at implementation.)
