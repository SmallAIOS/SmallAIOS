# ci-test-gates-v1 Tasks

## 1. Matrix foundation (no behavior change)

- [ ] 1.1 Create `ci/test-matrix.toml`: `default` group mirroring today's 17-crate unit-test list (incl. net `tegra-net` and peripheral `full-peripheral` feature args), empty `[exclusions]` table schema with `reason` field
- [ ] 1.2 Write `scripts/test-matrix.py` (stdlib-only): TOML load, `--emit gha`, `--run <group>` with `test result:` parsing + zero/`min_tests` enforcement, `--verify` against `cargo metadata` workspace members
- [ ] 1.3 Unit tests for the script (summary-line parsing incl. `0 passed; N ignored`, min_tests floor, verify pass/fail cases) runnable via `python3 -m unittest`
- [ ] 1.4 Classify all remaining workspace members: groups or exclusions-with-reasons (kernel-mode-only crates, bin-only tools) so `--verify` passes
- [ ] 1.5 Add `--verify` as a blocking CI step; confirm it fails on an unclassified synthetic crate locally

## 2. Converge existing consumers on the `default` group

- [ ] 2.1 Switch CI unit-test job to `python3 scripts/test-matrix.py --run default`; confirm identical crate set and green run
- [ ] 2.2 Switch CI clippy job to the matrix-derived crate/feature union (`--emit clippy-args` or equivalent)
- [ ] 2.3 Point `just test` at `--run default`; add `just test-all` (all groups) and `just test-group <name>`; delete the Justfile's duplicated crate list (derive `host_crates` for depgraph/arch-check recipes from the matrix too, or document why it stays)
- [ ] 2.4 Replace the release workflow's private crate list with `--verify` + `just test-all` in the pre-release hook

## 3. Recover the dark suites (one commit per group)

- [ ] 3.1 `fs-features` group: fs with `fs-flash,fs-flash-mock,fs-on-disk-mounts` — expect ~689 unit + overlay/conformance targets; set `min_tests` ≈ 90% of observed
- [ ] 3.2 Point the fs-interop gate at `--run fs-features`; verify the 40 squashfs conformance tests report a nonzero executed count in the job log
- [ ] 3.3 `posix-features` group: posix with `fs-flash-mock,fs-on-disk-mounts`
- [ ] 3.4 `tls-client` group (165 tests; the 9 network-e2e `#[ignore]` tests stay ignored)
- [ ] 3.5 `audit-export` group with `bearer` feature (137 tests)
- [ ] 3.6 `arch-x86_64-host` group on the x86_64 runner; probe nvidia/amd/intel_gpu/apple/riscv64/aarch64 test targets on x86_64 — add compiling ones to `arch-portable`, exclude the rest with reasons
- [ ] 3.7 Wire the new groups as parallel `strategy.matrix` entries (setup job + `fromJSON(--emit gha)`) with per-group rust-cache keys; record before/after CI wall-time on the PR

## 4. Coverage + verification

- [ ] 4.1 Extend the cargo-llvm-cov invocation with the recovered feature groups; record the new coverage baseline in this tasks file (was: 80% gate, ratcheting to 93%)
- [ ] 4.2 If the gate dips: exclude generated `*_test_vectors.rs` from the coverage denominator (Sonar-exclusion precedent) — do not touch the threshold; document outcome either way
- [ ] 4.3 Full pipeline green run on the PR with every group executing nonzero tests (screenshot/log links in PR description)
- [ ] 4.4 Update CLAUDE.md "Known quirks" (CI omits tls-client/audit-export; fs-flash never enabled — now stale) and the CI section's gate list
- [ ] 4.5 `openspec validate ci-test-gates-v1` clean; run `/opsx:verify` before archive
