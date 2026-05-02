## 1. Phase 1 — Inventory the full Code Quality view

- [ ] 1.1 Open the GitHub Code Quality view for the repo (https://github.com/SmallAIOS/SmallAIOS/security/code-scanning?query=is%3Aopen+tool%3ACodeQL+ — try variations until you find the Quality tab) and capture every finding into `openspec/changes/codeql-quality-cleanup-v1/findings.md` with columns: `path`, `line`, `rule_id`, `severity`, `tool`, `message`, `triage` (fix / extract-to-test-vectors / inline-annotation / accept-as-noise / investigate).
- [ ] 1.2 Confirm the user-reported entries appear in the inventory:
  - `scripts/dsm-matrix.py:16,18` — `import os`, `import sys` unused
  - `scripts/lcov-to-sonar.py:14` — `import os` unused
  - `onnx-rt/src/session.rs:1`
  - `onnx-rt/tests/integration_inference.rs:1`
  - `openspec/changes/archive/2026-04-28-microsoft-fused-ops-v1/tasks.md:1`
  - `openspec/changes/async-multistream-v1/tasks.md:2`
  - `openspec/changes/formal-proving-and-redteam-v1/tasks.md:2`
- [ ] 1.3 For the markdown findings: open one in the UI and read the rule ID + tool name. Record whether they're CodeQL or another tool (markdownlint, etc.). If non-CodeQL, decide whether they're in scope for this change or get a follow-up.
- [ ] 1.4 For the rust findings on `session.rs:1` and `integration_inference.rs:1`: read the rule ID and message. These will likely turn out to be header / lint findings (line 1 of files almost always means a file-header or top-of-file rule).
- [ ] 1.5 **Scope-check gate:** if total Quality findings exceed 30, OR if more than 5 distinct rule IDs appear that are not anticipated in `proposal.md`, pause and re-scope before proceeding to Phase 2. Open a comment in the PR or notify the user.
- [ ] 1.6 Also confirm the 7 active `rust/hard-coded-cryptographic-value` findings from develop's latest analysis (analysis id `1207890049`) are present and that their locations match `proposal.md`:
  - `security/src/crypto/ml_kem.rs:804`
  - `security/src/crypto/ml_dsa.rs:1070,1157,1351,1480`
  - `net/src/quic/protection.rs:227,281`

## 2. Phase 2 — Trivial Python fixes

- [ ] 2.1 Edit `scripts/dsm-matrix.py`: delete `import os` (line 16) and `import sys` (line 18). Confirm by `python3 scripts/dsm-matrix.py --help` (or equivalent dry run) that the script still runs without ImportError.
- [ ] 2.2 Edit `scripts/lcov-to-sonar.py`: delete `import os` (line 14). Confirm by running it on a sample lcov file (or `python3 -c "import scripts.lcov_to_sonar"` if it's importable) that nothing breaks.
- [ ] 2.3 If the inventory from Phase 1 surfaces any other Python findings that are clear true positives (not just unused imports — could be unused variables, etc.), fix them now.
- [ ] 2.4 Run `just fmt-check` and `just clippy` (no Rust changes yet; just sanity-check the worktree is clean).

## 3. Phase 3 — Confirm CodeQL config strategy

- [ ] 3.1 Check the repo for existing `.github/codeql/` directory and config file. If absent, plan to create `.github/codeql/codeql-config.yml`.
- [ ] 3.2 Determine whether GitHub default-setup CodeQL honors a custom `codeql-config.yml`. (Typical: default-setup ignores it; workflow-driven setup uses it.) Visit repo Settings → Code security → Code scanning to check current setup mode.
- [ ] 3.3 If default-setup ignores config, decide between (a) migrating to workflow-driven CodeQL (out of scope for this change — would be a separate `codeql-workflow-migration-v1`), or (b) falling back to inline `// lgtm[rust/hard-coded-cryptographic-value]` annotations on the affected lines.
- [ ] 3.4 Document the chosen path in `findings.md` (and optionally update `design.md` Open Question #1 with the resolution).

## 4. Phase 4 — KAT-vector module extraction (path-ignore strategy)

Skip this phase if Phase 3 chose inline annotations instead.

- [ ] 4.1 Create `security/src/crypto/ml_kem_test_vectors.rs` with the byte-array constants currently inline at `ml_kem.rs:804` (and any nearby siblings — likely there are more KAT vectors in the same file even if CodeQL only flagged one). Mark each `pub(crate) const` and document the FIPS / NIST source.
- [ ] 4.2 Create `security/src/crypto/ml_dsa_test_vectors.rs` with the byte-array constants currently at `ml_dsa.rs:1070, 1157, 1351, 1480` and any nearby siblings.
- [ ] 4.3 Create `net/src/quic/protection_test_vectors.rs` (or `net/src/quic/test_vectors/protection.rs`) with the IVs currently at `protection.rs:227, 281` and any nearby siblings.
- [ ] 4.4 Update `mod` declarations in `security/src/crypto/mod.rs` (or wherever the parent `mod` is) and `net/src/quic/mod.rs` to declare the new test-vector modules, gated `#[cfg(any(test, feature = "kat-vectors"))]` or similar if the vectors are only used by tests.
- [ ] 4.5 Update `use` paths in code that previously referenced the inline constants. Ensure no public API names change. Run `just test` — all tests must still pass and produce identical results.
- [ ] 4.6 Add `paths-ignore` entry in `.github/codeql/codeql-config.yml`:
  ```yaml
  paths-ignore:
    - "**/*_test_vectors.rs"
    - "**/test_vectors/**"
  ```
- [ ] 4.7 Confirm `just clippy -D warnings`, `just fmt-check`, `just arch-check` all pass.
- [ ] 4.8 Confirm coverage gate (≥80%) is not regressed: `cargo llvm-cov --workspace --fail-under-lines 80`.

## 5. Phase 5 — Phase 4 fallback: inline annotations

Skip this phase if Phase 4 was completed.

- [ ] 5.1 Add `// lgtm[rust/hard-coded-cryptographic-value]` annotation to each of the 7 flagged lines (`ml_kem.rs:804`, `ml_dsa.rs:1070,1157,1351,1480`, `protection.rs:227,281`).
- [ ] 5.2 Above each annotation, add a comment explaining why the constant is not a real secret (e.g., `// NIST CAVP test vector — known-answer-test data, not a live key`). Cite the source RFC / FIPS document.
- [ ] 5.3 Run `just clippy -D warnings`, `just fmt-check`, `just arch-check`, `just test`. All must pass.

## 6. Phase 6 — One-off Quality-view triage

For each remaining finding from the Phase 1 inventory not addressed by Phases 2/4/5, apply the policy from `specs/codeql-suppression-policy/spec.md`:

- [ ] 6.1 For findings classified `fix` in the inventory: fix them. Examples likely include the `onnx-rt/src/session.rs:1` and `onnx-rt/tests/integration_inference.rs:1` rust findings if they turn out to be missing-license-header or similar trivial things.
- [ ] 6.2 For findings classified `inline-annotation` in the inventory: apply `// lgtm[rule-id]` with a rationale comment per the policy.
- [ ] 6.3 For findings classified `accept-as-noise` (typically: a Markdown finding that's truly stylistic, on an archived `tasks.md`): document the decision in `findings.md` and move on. Do not edit archived openspec changes' tasks.md files just to silence Quality findings — they're historical records.
- [ ] 6.4 For findings classified `investigate` that turned out to require structural changes (e.g., a real lint rule we should fix everywhere): split them off into a follow-up change `codeql-quality-cleanup-v2` rather than expanding this change's scope.

## 7. Phase 7 — Documentation

- [ ] 7.1 Write `docs/code-quality.md` with the four required sections (per `specs/documentation/spec.md`):
  - "Code Scanning Alerts vs Code Quality View"
  - "Suppression Policy"
  - "Test-Vector Module Convention"
  - "Triage Checklist"
- [ ] 7.2 Cross-reference `docs/code-quality.md` from `CLAUDE.md`'s CI/CD section.
- [ ] 7.3 If a separate `CONTRIBUTING.md` exists at repo root, also link from there.
- [ ] 7.4 In `docs/code-quality.md`, link to the `codeql-suppression-policy` spec at `openspec/specs/codeql-suppression-policy/spec.md` (post-archive path) so the doc and the spec stay in sync.

## 8. Phase 8 — Verification, push, monitor

- [ ] 8.1 Push the change branch and open a PR against `develop` titled `chore: codeql-quality-cleanup-v1 — triage Code Quality view findings`.
- [ ] 8.2 Wait for CI (CodeQL runs as part of standard analysis). Confirm:
  - Zero open `rust/hard-coded-cryptographic-value` findings on the affected paths
  - All previously-required CI checks still pass
  - Code Quality view findings expected to be addressed are gone
- [ ] 8.3 Address review feedback. Any reviewer-flagged findings that turn out to be real defects get follow-up tasks added here or split into a follow-up change.
- [ ] 8.4 Merge to `develop` once approved.
- [ ] 8.5 **One-week soak:** monitor daily CodeQL runs on `develop` for 7 days post-merge. If no recurrence of `rust/hard-coded-cryptographic-value` on affected paths, the suppression strategy is validated. If recurrence happens, open a follow-up change to revise the strategy.
- [ ] 8.6 After the soak, run `/opsx:archive codeql-quality-cleanup-v1` to archive this change and update main specs with the deltas from `specs/codeql-suppression-policy/spec.md` and `specs/documentation/spec.md`.
