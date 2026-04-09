## 1. cargo-semver-checks CI Job

- [x] 1.1 Add `semver-api-check` job to `ci.yml`: install `cargo-semver-checks`, run `cargo semver-checks check-release --baseline-rev origin/${{ github.base_ref }}` on host-testable crates
- [x] 1.2 Implement conditional gating: parse PR title for `!`, set `continue-on-error: true` if breaking change is intentional
- [x] 1.3 Add `semver-api-check` to `change-gates` `needs` list
- [x] 1.4 Add `cargo-semver-checks` advisory check to `.githooks/pre-commit` (runs if installed, non-blocking)
- [x] 1.5 Update `.pre-commit-config.yaml` with `cargo-semver-checks` hook
- [ ] 1.6 Test: create a PR that removes a public function, verify CI catches it; create a `feat!:` PR, verify it passes with warning

## 2. cargo-vet Dependency Audit

- [x] 2.1 Run `cargo vet init` to create `supply-chain/` directory with `config.toml` and `audits.toml`
- [x] 2.2 Import trusted publisher audits: `cargo vet import` from Mozilla, Google, Bytecode Alliance
- [x] 2.3 Run `cargo vet certify` or `cargo vet add-exemption` for all remaining unaudited dependencies
- [x] 2.4 Verify `cargo vet check` passes locally with all dependencies covered
- [x] 2.5 Add `cargo-vet-check` job to `ci.yml`: install `cargo-vet`, run `cargo vet check`
- [x] 2.6 Add `cargo-vet-check` to `change-gates` `needs` list (hard gate)
- [x] 2.7 Commit `supply-chain/` directory as a single auditable commit
- [x] 2.8 Update CLAUDE.md with `cargo vet certify` workflow for new dependencies

## 3. cargo-careful Testing

- [x] 3.1 Add `careful-test` job to `ci.yml`: install `cargo-careful`, run `cargo careful test` on host-testable crates
- [x] 3.2 Set `continue-on-error: true` (advisory initially)
- [x] 3.3 Upload careful test results as artifact for review
- [ ] 3.4 Test: verify job runs successfully on current codebase; document any false positives

## 4. Coverage Threshold Gate

- [x] 4.1 Add `coverage-threshold` job to `ci.yml`: install `cargo-llvm-cov`, run `cargo llvm-cov --fail-under-lines 80` on host-testable crates
- [x] 4.2 Add `coverage-threshold` to `change-gates` `needs` list (hard gate)
- [x] 4.3 Document threshold ratcheting schedule in a comment in `ci.yml`: 80% now → 85% after onnx-cpu-runtime-v1 tests → 90% → 93% parity with Codecov
- [ ] 4.4 Test: verify current coverage is above 80%; adjust threshold if needed

## 5. Change Gates Update

- [x] 5.1 Update `change-gates` job `needs` list to include: `semver-api-check`, `cargo-vet-check`, `coverage-threshold`
- [x] 5.2 Verify all gated jobs are required (not `continue-on-error`) except `careful-test`
- [ ] 5.3 Run full CI pipeline on a test branch, verify `change-gates` correctly blocks when any new gate fails

## 6. Documentation and Dev Setup

- [x] 6.1 Update CLAUDE.md CI/CD section with new jobs and their gate status
- [x] 6.2 Update `docs/scheduling-model.md` or create `docs/ci-safety-tooling.md` summarizing all safety tools, their DO-178C relevance, and gate vs advisory status
- [x] 6.3 Add `cargo install cargo-semver-checks cargo-vet cargo-careful --locked` to dev dependencies in CLAUDE.md
- [x] 6.4 Verify `just check` and `just audit` still pass; update recipes if needed
- [ ] 6.5 Run `just clippy` and `just fmt-check` on all CI config changes
