## Context

The repo runs CodeQL via GitHub's default-setup with three language analyses (`/language:rust`, `/language:python`, `/language:actions`, `/language:go`). Findings reach engineers through two distinct surfaces:

1. **Code Scanning alerts** — `repos/{owner}/{repo}/code-scanning/alerts`. Surfaces severity-`error`/`warning` findings as managed alerts that can be dismissed / fixed / re-opened. Queryable via API.
2. **Code Quality view** (newer) — surfaces findings from the `code-quality` query suite (severity `note` / informational, things like unused imports, dead code, style). NOT exposed in the alerts API; only browsable via the GitHub UI.

State at the start of this change (as of develop tip `0267e16`):

- Code Scanning alerts API:
  - 0 open alerts
  - 13 dismissed (all "false positive": 9× `rust/cleartext-logging`, 4× `rust/hard-coded-cryptographic-value`)
  - 31 fixed
- Develop's latest CodeQL `language:rust` analysis (id `1207890049`, sarif `6dd2d066-45c3-11f1-9110-80dbf988e3e1`): **7 results, all `rust/hard-coded-cryptographic-value`, none surfaced as alerts** because they appear under a category that doesn't auto-open alerts. Locations:
  - `security/src/crypto/ml_kem.rs:804` (nonce)
  - `security/src/crypto/ml_dsa.rs:1070, 1157, 1351, 1480` (nonces)
  - `net/src/quic/protection.rs:227, 281` (IVs)
- Code Quality UI: contains additional findings on Python and Markdown files plus `onnx-rt/src/session.rs` and `onnx-rt/tests/integration_inference.rs` per user report. Cannot enumerate programmatically.

The 7 hard-coded-cryptographic-value findings are FIPS / NIST PQC known-answer-test (KAT) vectors. They **must remain in source** to satisfy NIST CAVP testing requirements. CodeQL's heuristic flags any byte-array constant whose name suggests cryptographic use (nonce / iv / key); it cannot tell that these specific constants are test fixtures. Thirteen prior dismissals of the same class did not prevent recurrence — every time a test vector is added or the file is re-scanned, the alerts re-fire.

**Stakeholders:**
- `security` crate maintainers — own the KAT-vector files
- `net/quic` maintainers — own the protection.rs IVs (likely RFC 9001 test vectors)
- `scripts` maintainers — own the Python tooling
- DO-178C compliance — NIST CAVP test vector accessibility is part of the safety-critical evidence chain; any restructuring must preserve that

## Goals / Non-Goals

**Goals:**
- Eliminate 7 active `rust/hard-coded-cryptographic-value` findings in a way that does **not recur** on subsequent CodeQL runs
- Eliminate 3 confirmed unused-import findings on Python scripts
- Document a triaged inventory of Code Quality UI findings and resolve each
- Preserve all KAT vectors in source (can be moved to separate files, but cannot be deleted or moved out of the repo)
- Preserve coverage, clippy, fmt, arch-check, and all required CI checks
- Establish a documented suppression policy so future similar findings have a clear playbook

**Non-Goals:**
- Adding new CodeQL queries
- Migrating away from GitHub default-setup CodeQL to the workflow-driven setup (out of scope; could be a future change if needed)
- Refactoring crypto implementations beyond what's needed for suppression
- Fixing security or correctness bugs (none found — every finding is quality / false-positive)
- Auditing or revisiting the 13 already-dismissed alerts (left as-is unless they recur)

## Decisions

### Decision 1: Suppression strategy — extract KAT vectors into dedicated `test_vectors` modules + path-filter in CodeQL config

For each affected file (`ml_kem.rs`, `ml_dsa.rs`, `quic/protection.rs`), move the inline KAT byte-array constants into a sibling submodule (`ml_kem_test_vectors.rs`, etc.) that contains **only** test data, no functional code. Add path-ignore entries in `.github/codeql/codeql-config.yml` for those files (or a `*_test_vectors.rs` glob). Production code accesses them via a normal `use super::test_vectors::KAT_NONCE_FOO;` style.

**Why:**
- Inline `// lgtm[rust/hard-coded-cryptographic-value]` annotations work but are easy to miss and accumulate as visual noise — a reviewer scanning the file sees a comment-laced wall.
- Path-filtering the *entire* crypto file would also exclude any *real* cryptographic-value findings the code might pick up later. Bad blast radius.
- Module extraction draws a clear, testable boundary: "if it's in a `test_vectors` file, CodeQL skips it; everything else is still scanned." It also doubles as documentation — "everything in here is reference data, not live secrets."

**Alternatives considered:**
- *Inline `// lgtm` annotations*: Simpler change, but doesn't generalize. Rejected for KAT-heavy files (it's fine for one-off cases — see Decision 2).
- *Whole-file `paths-ignore`*: Too broad. Rejected.
- *Refactor to load KAT vectors from external files at runtime*: Breaks `#![no_std]` (would need an embed mechanism), and the vectors are compile-time constants for a reason. Rejected.
- *Convert byte arrays to `hex!("...")` macro from `hex-literal`*: Doesn't help — CodeQL flags the result equally. Rejected.

### Decision 2: One-off non-KAT findings get inline annotations, not module extraction

For findings that are demonstrably false-positive but don't fit the "this is test-vector data" pattern (e.g., a one-off nonce constant used as a domain-separation tag), use an inline `// lgtm[rule-id]` annotation **with a comment explaining why it's not a real secret**. Module extraction would be overkill and obscure the comment.

**Why:** Reserve the heavy hammer (Decision 1) for the heavy class. Single-instance suppressions stay readable.

### Decision 3: First task in implementation is full Code Quality UI inventory

Because the alerts API does not expose Code Quality view findings, the only way to enumerate them is manual UI browsing. The first concrete task in this change is to capture the **complete** Quality view content into a tracking table inside the change directory (`findings.md` or extending `tasks.md`), with rule ID, severity, file, line, and one-line description.

**Why:** Without this inventory we don't know the actual scope. The user-reported set may be illustrative, not exhaustive. The change cannot complete its triage without the full list.

**Sub-decision:** If the inventory shows >30 findings or substantially different patterns from the starter set, the change pauses for re-scoping (the proposal's stated risk). Threshold: 30 findings or 5+ distinct rule IDs not anticipated in this proposal.

### Decision 4: Markdown-file findings get investigated before action

The user-reported `tasks.md` findings are unusual — CodeQL doesn't have query packs for Markdown. Possibilities: (a) GitHub UI mis-attributes a different tool's findings to "CodeQL"; (b) the findings target a code block fenced inside the markdown (CodeQL can analyze fenced code blocks in some configurations); (c) a third-party tool (e.g., a markdownlint integration) is feeding the Quality view.

**Action:** First-pass task includes screenshotting / opening one of the markdown findings in the UI to read the rule and tool name. Triage decision then follows from what we find.

### Decision 5: Python suppression — fix, not suppress

The three Python unused-import findings are real. `os` is genuinely unused; `sys` in `dsm-matrix.py` is genuinely unused. The right action is to delete the imports, not to suppress them. No suppression policy needed for these.

**Why:** Suppressions exist for false positives. These are true positives.

### Decision 6: Documentation lives in `docs/code-quality.md`

Add a new doc explaining: alerts vs Quality view, the suppression policy, the `test_vectors` module convention, and a "before you suppress, ask…" checklist. Cross-reference from `CONTRIBUTING.md` (or its equivalent in this repo) and from the relevant CLAUDE.md sections about CI gates.

**Why:** Future contributors hitting a CodeQL false positive should have a single page that tells them what to do, not have to re-derive it from this change's design.md.

## Risks / Trade-offs

- **[Risk] Module extraction (Decision 1) breaks public API of `security` crate** → Mitigation: KAT vectors are crate-internal (`pub(crate)` at most). Verify before edit; no public API change expected. If any vector is publicly exported, keep its public name and re-export from the new module.
- **[Risk] CodeQL `paths-ignore` config has a typo / glob mismatch and the suppressions don't take effect** → Mitigation: after landing, manually re-trigger the CodeQL workflow on the change branch and verify zero `rust/hard-coded-cryptographic-value` alerts on affected paths *before* merging. Acceptance criterion #1.
- **[Risk] Quality-view UI enumeration finds many more findings than expected; scope blows out** → Mitigation: Decision 3's sub-decision — pause and re-scope at thresholds (>30 findings or >5 unanticipated rule IDs).
- **[Risk] Markdown findings turn out to be from a tool we can't easily silence** → Mitigation: triage decision in Decision 4. If they're genuine markdownlint findings on `tasks.md` files, fix them; if they're spurious tooling errors, document and move on.
- **[Trade-off] Decision 1 means more files in the source tree** → Acceptable: each `*_test_vectors.rs` is a small file with a clear single purpose. The DSM cost is zero (same module path). The signal-to-noise win on code review is positive.
- **[Trade-off] Decision 2 (inline annotations for one-offs) means two suppression styles in the codebase** → Acceptable: the rule for which is explicit (Decision 1 for KAT-heavy, Decision 2 for one-off), documented in `docs/code-quality.md`.

## Migration Plan

1. **Phase 1 — Inventory (no code changes):** browse Code Quality UI, capture every finding into a tracking table at `openspec/changes/codeql-quality-cleanup-v1/findings.md`. Apply scope-check threshold from Decision 3.
2. **Phase 2 — Trivial fixes:** remove the 3 Python unused imports. Land as a small first PR or include in the main PR per reviewer preference.
3. **Phase 3 — KAT-vector module extraction:** create `*_test_vectors.rs` for the three affected files. Update `.github/codeql/codeql-config.yml`. Update `use` paths in calling code. Run `just test`, `just clippy`, `just fmt-check`, `just arch-check` — all must pass.
4. **Phase 4 — One-off suppressions:** apply inline `// lgtm` annotations to any remaining specific findings identified in inventory.
5. **Phase 5 — Documentation:** write `docs/code-quality.md`. Cross-reference from CONTRIBUTING / CLAUDE.md.
6. **Phase 6 — Verification:** push the branch, re-trigger CodeQL on the PR, confirm zero new alerts on previously-flagged paths. Confirm Code Quality UI shows expected reductions.

**Rollback:** Each phase is committed separately. Phase 3 (module extraction) is the only one with non-trivial code movement; if it breaks something, revert just that commit. Phases 2 and 4 are isolated edits with low blast radius.

## Resolved Decisions

- **Suppression strategy primary path: module extraction + path-ignore.** Inline annotations are reserved for one-offs.
- **KAT vectors stay in source.** NIST CAVP and DO-178C evidence chains depend on it.
- **Python findings are fixed (delete unused imports), not suppressed.**

## Open Questions

1. **Where exactly does `codeql-config.yml` live?** GitHub default-setup may not honor a custom config — switching to workflow-driven CodeQL adds complexity. Confirm during Phase 3; if default-setup ignores config, fall back to inline annotations for KAT vectors. **Default for tasks: try config first; fall back if needed.**
2. **Are the user-reported `tasks.md` findings truly from CodeQL, or another tool?** Resolved during Phase 1 inventory.
3. **Is there a single `CONTRIBUTING.md` or are docs distributed?** Quick check during Phase 5; cross-link wherever it lives.
4. **Should this change archive after Phase 6, or wait for one full week of green CodeQL runs to confirm no recurrence?** **Default for tasks: wait one week before archiving** to confirm suppressions hold across daily scans.
