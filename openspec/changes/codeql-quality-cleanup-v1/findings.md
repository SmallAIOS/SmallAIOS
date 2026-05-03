# Code Quality View — Findings Inventory

Phase 1 deliverable for `codeql-quality-cleanup-v1`. Snapshot taken 2026-05-03 from the GitHub Code Quality view (the newer surface that combines CodeQL `code-quality` query suite findings AND GitHub Copilot Code Review observations).

## Important reframing

The proposal originally assumed all Quality-view findings were CodeQL static-analysis lints (e.g., unused imports, dead code). Inventory revealed that the surface actually mixes two distinct sources:

1. **CodeQL `code-quality` suite** — true static-analysis lints (e.g., unused imports). Severity `note` / informational.
2. **GitHub Copilot Code Review** — AI-authored review observations on file content. Surfaced in the same UI but produced by a different system. These are *suggestions*, not static-analysis findings.

Both surfaces are addressed here; triage decisions differ by source.

## Findings

| # | Path | Source | Triage | Action |
|---|------|--------|--------|--------|
| 1 | `scripts/dsm-matrix.py:16` (unused `import os`) | CodeQL | **FIX** | Delete line in Phase 2 |
| 2 | `scripts/dsm-matrix.py:18` (unused `import sys`) | CodeQL | **FIX** | Delete line in Phase 2 |
| 3 | `scripts/lcov-to-sonar.py:14` (unused `import os`) | CodeQL | **FIX** | Delete line in Phase 2 |
| 4 | `onnx-rt/src/session.rs` (eager-vs-lazy `transfer_streams` validation) | Copilot Review | **DEFER** | Architectural — see deferral note |
| 5 | `onnx-rt/tests/integration_inference.rs` (use `..SessionConfig::default()`) | Copilot Review | **FIX** | Small bonus refactor in this PR |
| 6 | `openspec/changes/archive/2026-04-28-microsoft-fused-ops-v1/tasks.md` (env-var naming) | Copilot Review | **ACCEPT-AS-NOISE** | Archived material; established convention |
| 7 | `openspec/changes/async-multistream-v1/tasks.md` (throughput-target wording) | Copilot Review | **ACCEPT-AS-NOISE** | Resolved by archive in PR #126 |
| 8 | `openspec/changes/formal-proving-and-redteam-v1/tasks.md` (script manual PR-revert) | Copilot Review | **DEFER** | Owner-of-change concern, not quality cleanup |

## Deferral notes

### Finding #4 — `Session::new` validation timing

The Copilot review observes that `transfer_streams <= 2` is validated lazily in `Session::ensure_stream_pool` (called on first multi-stream inference) rather than eagerly in `Session::new`. The suggestion is reasonable on its merits but **out of scope** for this change because:

- `Session::new(config: SessionConfig) -> Self` currently returns `Self` directly, not `Result<Self, SessionError>`.
- Adding eager validation requires either (a) breaking the API by returning `Result`, (b) introducing a parallel `Session::try_new`, (c) panicking on invalid config (bad), or (d) moving validation to a `SessionConfig::validate()` method that callers must remember to call.
- All four options are non-trivial design decisions affecting public API surface and downstream callers (bench, container, integration tests).
- A code-quality cleanup PR is the wrong place to debate that. It belongs in its own change, e.g. `session-config-eager-validation-v1`.

**Action:** Document this deferral. Open a follow-up change at the appropriate time. Lazy validation continues to work correctly today — this is a usability improvement, not a defect.

### Finding #8 — Manual PR-revert verification in `formal-proving-and-redteam-v1`

The Copilot review suggests scripting the manual PR-revert verification step in `formal-proving-and-redteam-v1`'s task list. The suggestion is a real improvement to *that* change's plan, not a code-quality cleanup. The right place to surface it is on PR #122 (or in `formal-proving-and-redteam-v1`'s tasks.md as an enhancement).

**Action:** Add a comment to PR #122 referencing this finding. Do not modify `formal-proving-and-redteam-v1` from this change.

## Accept-as-noise notes

### Finding #6 — Archived `microsoft-fused-ops-v1` env-var naming

The Copilot review questions `SMALLAIOS_MODEL_DIR` (singular) vs `SMALLAIOS_MODELS_DIR` (plural). The singular form is the **established convention** documented in `CLAUDE.md` ("Container Environment Variables" section) and is the actual env var read by `smallaios-container`. The archived `tasks.md` correctly references the established name.

Per the change's design.md decision: archived openspec changes are historical records and SHALL NOT be edited solely to silence Quality findings.

**Action:** None. Document.

### Finding #7 — `async-multistream-v1` throughput-target wording

The Copilot review notes "≥1.3× / ≥1.5×" is ambiguous without configuration context. The change is being archived right now in PR #126; after merge the file moves to `openspec/changes/archive/2026-05-03-async-multistream-v1/tasks.md` and the same archived-material policy applies.

**Action:** None. Finding will stop appearing after PR #126's CodeQL run.

## Scope-check (per design.md Decision 3)

- Total findings: **8**
- Distinct rule IDs / sources: **2** (CodeQL unused-import + Copilot Code Review)
- Threshold (>30 findings or >5 unanticipated rule IDs): **NOT EXCEEDED**

Continue to Phase 2 without re-scoping.

## Phase 2 readiness

The 4 actionable items (3 Python deletes + 1 test refactor) are mechanical and can land as a single commit. Phase 3 (KAT-vector extraction) remains separate and still requires Phase 3.2's check on whether GitHub default-setup honors a custom `codeql-config.yml`.
