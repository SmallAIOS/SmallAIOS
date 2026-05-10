## 1. Phase 1 — Inventory the Code Quality view

- [x] 1.1 Inventory captured. The Code Quality view aggregates two sources: CodeQL `code-quality` suite (true static-analysis lints) + GitHub Copilot Code Review (AI-authored review observations). 8 findings total: 3 unused-import (CodeQL) + 5 review observations (Copilot). API doesn't expose this surface; user provided UI content directly.
- [x] 1.2 Triaged: 4 actionable in this change (3 Python deletes + 1 test refactor); 2 accept-as-noise (archived material — see §9 Deferrals); 2 deferred to other owners (see §9).
- [x] 1.3 Scope-check gate: 8 findings, 2 distinct sources. Threshold (>30 findings or >5 unanticipated rule IDs) NOT exceeded. Continue to Phase 2.
- [x] 1.4 Confirmed 7 active `rust/hard-coded-cryptographic-value` findings on develop match `proposal.md` (analysis 1207890049, sarif 6dd2d066-45c3-11f1-9110-80dbf988e3e1) — these are Phase 3+ work.

## 2. Phase 2 — Python unused imports + test refactor

- [x] 2.1 `scripts/dsm-matrix.py`: deleted `import os` and `import sys` (genuinely unused per grep of `\b(os|sys)\.`). Script parses cleanly via `ast.parse`.
- [x] 2.2 `scripts/lcov-to-sonar.py`: deleted `import os` (only `sys` is used). Script parses cleanly.
- [x] 2.3 `onnx-rt/tests/integration_inference.rs::session_custom_config`: refactored to use `..SessionConfig::default()` so only the 4 fields under test are explicit. Test passes; `cargo fmt --check` clean. This addresses the Copilot review suggestion on that test.

## 3. Phase 3 — Confirm CodeQL config strategy

- [x] 3.1 Checked `.github/codeql/` — no directory existed pre-change. Created `.github/codeql/codeql-config.yml`.
- [x] 3.2 Repo uses **workflow-driven CodeQL** (`.github/workflows/codeql.yml` invokes `github/codeql-action/init@v4` for the rust/python/actions/go matrix). Workflow-driven setup honors `config-file:` passed to the init action, so a custom `codeql-config.yml` IS effective. (Default-setup would have ignored it — but that's not the path here.)
- [x] 3.3 N/A — workflow-driven setup honors the config; no migration or fallback needed.
- [x] 3.4 Resolved Open Question #1 from `design.md` here in tasks.md: workflow-driven; Phase 4 (config + module extraction) proceeds.

## 4. Phase 4 — KAT-vector module extraction (path-ignore strategy)

Phase 4 was the chosen path. Outcome differs in scope from the original plan:
the seven flagged sites split into two distinct classes once read in context.

- [x] 4.1 `security/src/crypto/ml_kem.rs:804` — investigation showed this is **production code**, not a KAT byte array: `let e2 = prf(random_coins, (2 * ML_KEM_768_K) as u8, ETA2);` (FIPS 203 §7.2 PRF domain-separation index). Cannot be extracted as a test vector. The pre-existing inline `// lgtm[rust/hard-coded-cryptographic-value]` comment cites the FIPS section and remains as documentation. No new `ml_kem_test_vectors.rs` file is created. Out of scope for path-ignore.
- [x] 4.2 `security/src/crypto/ml_dsa.rs:1070, 1157, 1351, 1480` — same finding as 4.1: these are **production code** lines (`sample_uniform(rho, (i * 256 + j) as u16)` in ExpandA, `sample_mask(&rho_pp, kappa + i as u16)` in ExpandMask). FIPS 204 §6.x mandates the call shape. Pre-existing `// lgtm` comments cite the FIPS sections. No `ml_dsa_test_vectors.rs` file is created.
- [x] 4.3 `net/src/quic/protection.rs:227, 281` — these ARE genuine test fixtures inside `#[cfg(test)] mod tests`. Created `net/src/quic/protection_test_vectors.rs` with six `pub(super) const` byte-pattern fixtures (`TEST_KEY_AA`, `TEST_IV_BB`, `TEST_HP_KEY_CC`, `TEST_WRONG_KEY_01`, `TEST_HP_KEY_DD`, `TEST_HP_SAMPLE_EE`). The module is `#![cfg(test)]`-gated so it does not affect production build size or surface.
- [x] 4.4 Added `#[cfg(test)] #[path = "protection_test_vectors.rs"] mod test_vectors;` declaration inside `protection.rs`. The `#[cfg(test)]` gate prevents the module from being included in production builds.
- [x] 4.5 Updated test bodies in `protection.rs` (`test_keys()`, `test_aead_decrypt_wrong_key`, `test_header_protection_roundtrip`, `test_header_protection_short_header`) to import the constants via `use super::test_vectors::*;`. No public API names changed. `cargo test -p smallaios-net --lib quic::protection` passes 13/13 with byte-identical assertions.
- [x] 4.6 Added `.github/codeql/codeql-config.yml`:
  ```yaml
  paths-ignore:
    - "**/*_test_vectors.rs"
    - "**/test_vectors/**"
  ```
  Wired into `.github/workflows/codeql.yml` via `config-file: ./.github/codeql/codeql-config.yml` on the `github/codeql-action/init@v4` step.
- [x] 4.7 Verified: `just fmt-check` clean, `just clippy` clean (zero warnings under `-D warnings`), `just arch-check` reports all 14 host crates acyclic at module level. `just test` passes (no failures across the workspace; `quic::protection` 13/13).
- [ ] 4.8 Coverage gate (≥80%) not regressed — deferred to CI run on the PR (the existing `Coverage Threshold` job will report; local `cargo llvm-cov` skipped to keep this PR scoped to suppression strategy).

### Phase 4 follow-up scope note

The five `ml_kem.rs` / `ml_dsa.rs` production-code findings will likely re-fire on the next CodeQL scan because they are not on a `*_test_vectors.rs` path. They are not KAT data; they are FIPS-mandated PRF domain-separation indices that cannot be moved out of the call site. Options for a follow-up (NOT in this PR):

1. Targeted `query-filters` block in `codeql-config.yml` excluding `rust/hard-coded-cryptographic-value` for those two paths, with rationale comment citing FIPS.
2. UI dismissal as "won't fix — false positive" once the suppression policy doc lands and reviewers can cite it.
3. Refactor the `prf` / `sample_uniform` / `sample_mask` signatures so the second argument is wrapped in a typed `DomainTag(u8)` / `DomainTag(u16)` newtype that CodeQL's heuristic does not flag.

Decision deferred to Phase 6 (one-off triage), where it can be batched with whatever the next CodeQL run surfaces.

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

## 10. Phase 10 — Query-filter the production-code irreducible false positives

Follow-up to Phase 4 footnote ("Five `ml_kem.rs` / `ml_dsa.rs` production-code findings will likely re-fire"). The May 2026 CodeQL scan on `develop` re-fired 13 `rust/hard-coded-cryptographic-value` alerts — but on a different set of files than expected:

- `security/src/argon2id.rs` — 8 alerts at lines 900, 902, 904, 906, 908, 910 (the RFC 4648 base64 alphabet inside `b64_decode`).
- `kernel/src/syscall/auth.rs` — 5 alerts at lines 70 (`DUMMY_PHC` constant), 420 + 509 (`[0u8; 16]` salt buffers passed to the kernel CSPRNG), and 429 + 518 (adjacent comment blocks the analyzer is over-matching).

Each call site already carries a standalone-line `// lgtm[rust/hard-coded-cryptographic-value]` annotation that the Rust analyzer does not honor for this rule on this codebase. Picked option (1) from the Phase 4 footnote: targeted `query-filters` block in `.github/codeql/codeql-config.yml`.

- [x] 10.1 Extended `.github/codeql/codeql-config.yml` `query-filters` to exclude `rust/hard-coded-cryptographic-value` for the two affected production files via `path: { regex: ^(security/src/argon2id\.rs|kernel/src/syscall/auth\.rs)$ }`. (Replaces the earlier `paths:` list form, which is not part of the documented CodeQL query-filter schema.)
- [x] 10.2 Updated the rationale doc comment in `codeql-config.yml` to enumerate the three pattern classes (RFC 4648 alphabet offsets, `DUMMY_PHC` anti-enumeration constant, CSPRNG-overwritten salt buffers) and to note that each call site retains a standalone `// lgtm[...]` comment as in-source documentation.
- [x] 10.3 No production code changes — the suppression is purely an analyzer-config change. Branch `change/codeql-prodcode-suppression-v1`.
- [ ] 10.4 Verification: after merge to `develop`, the 13 currently-open `rust/hard-coded-cryptographic-value` alerts close on the next CodeQL scan. (Tracked in PR description.)

## 9. Deferrals & accept-as-noise (out of scope for this change)

These items came out of Phase 1 triage but are intentionally not addressed in this change. Recorded here so they don't get lost when this change archives.

**Deferred to follow-up change:**
- `onnx-rt/src/session.rs` eager-vs-lazy `transfer_streams <= 2` validation. `Session::new(config: SessionConfig) -> Self` returns `Self`, not `Result`. Moving validation eager requires an API decision (return `Result`, add `try_new`, or surface a `SessionConfig::validate()` method). That belongs in its own change (`session-config-eager-validation-v1`), not a code-quality cleanup. Lazy validation in `ensure_stream_pool` works correctly today.

**Deferred to upstream change owner:**
- `openspec/changes/formal-proving-and-redteam-v1/tasks.md`: Copilot review suggested scripting the manual PR-revert verification step. That's a process improvement to the formal-proving change's plan, not a quality-view cleanup. Surface to PR #122 / the change's owner.

**Accept-as-noise (no action):**
- `openspec/changes/archive/2026-04-28-microsoft-fused-ops-v1/tasks.md`: Copilot questioned `SMALLAIOS_MODEL_DIR` (singular) vs `SMALLAIOS_MODELS_DIR` (plural). The singular form is the established convention documented in CLAUDE.md and is the actual env var read by `smallaios-container`. The archived doc is correct. Per Decision 4 in `design.md`, archived openspec changes are historical records and SHALL NOT be edited solely to silence Quality findings.
- `openspec/changes/async-multistream-v1/tasks.md`: Copilot flagged "≥1.3× / ≥1.5×" throughput target as ambiguous. This change is being archived in PR #126; the file moves to `openspec/changes/archive/2026-05-03-async-multistream-v1/tasks.md` and the same archived-material policy applies.
