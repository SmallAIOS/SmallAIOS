## Why

GitHub recently rolled out a "Code Quality" view that surfaces CodeQL findings from a broader query suite (including the `code-quality` suite) without filing them as alerts in the standard Code Scanning alerts API. The repo's previous `codeql-remediation-v1` change only addressed alerts visible through the alerts API, leaving a backlog of UI-visible quality findings unaddressed. Separately, seven `rust/hard-coded-cryptographic-value` alerts have re-fired on `develop` despite thirteen identical alerts having already been dismissed as false positives — these are FIPS / NIST PQC known-answer-test (KAT) vectors that CodeQL's heuristics cannot distinguish from real keys. The dismiss-and-refire cycle wastes reviewer time and erodes trust in code-scanning signal.

This change triages the UI-visible Quality findings, fixes the trivially-correctable ones, and applies a durable suppression strategy to the recurring KAT-vector false positives so they stop coming back.

## What Changes

- Remove three confirmed unused-import findings on Python tooling scripts:
  - `scripts/dsm-matrix.py` — drop `import os` (line 16) and `import sys` (line 18); neither is referenced in the file
  - `scripts/lcov-to-sonar.py` — drop `import os` (line 14); only `sys` is used
- Apply a durable suppression strategy for `rust/hard-coded-cryptographic-value` false positives on KAT / test-vector data in:
  - `security/src/crypto/ml_kem.rs`
  - `security/src/crypto/ml_dsa.rs`
  - `net/src/quic/protection.rs`
  Strategy choice (config-level path filter, source-level `// lgtm` annotations, or extracting KAT vectors into separate modules excluded by config) is settled in `design.md`. The chosen approach SHALL prevent recurrence on subsequent CodeQL runs.
- Add a one-time task to enumerate the **full** set of Code Quality view findings via UI inspection (since the alerts API does not expose them), capture rule IDs / messages / paths in a tracking table inside the change, and triage each (fix / suppress with rationale / accept). User-reported starter set:
  - `onnx-rt/src/session.rs:1` (rule TBD)
  - `onnx-rt/tests/integration_inference.rs:1` (rule TBD)
  - `openspec/changes/archive/2026-04-28-microsoft-fused-ops-v1/tasks.md:1` (rule TBD — flagging on a `.md` file is unusual, may be a non-CodeQL tool)
  - `openspec/changes/async-multistream-v1/tasks.md:2`
  - `openspec/changes/formal-proving-and-redteam-v1/tasks.md:2`
- Add CI / contributor docs explaining: (a) the difference between the Code Scanning alerts surface and the Code Quality view, (b) how to interpret each, and (c) when to suppress vs fix.

## Capabilities

### New Capabilities

- `codeql-suppression-policy`: A documented, testable convention for handling recurring CodeQL false positives — what gets in-source `// lgtm` annotations, what gets path-filtered in `.github/codeql/codeql-config.yml`, what gets extracted into separate modules. Includes acceptance criteria so future similar findings can be triaged consistently.

### Modified Capabilities

- `documentation`: A new `docs/code-quality.md` (or section of an existing CONTRIBUTING-style doc) explaining the alerts vs quality-view distinction, how to navigate findings, and the suppression policy. The repo's existing `documentation` capability gets one new requirement covering this doc's existence and content.

## Impact

**Code:**
- 3 trivial deletions in `scripts/*.py` (Layer N/A — tooling)
- Suppression annotations or module restructuring in `security/src/crypto/{ml_kem,ml_dsa}.rs` and `net/src/quic/protection.rs` (no behavioral change; KAT vectors must remain functionally accessible to tests)
- Possibly new files under `security/src/crypto/test_vectors/` and `net/src/quic/test_vectors/` if the chosen suppression strategy is module extraction
- Possibly an updated `.github/codeql/codeql-config.yml` with `paths-ignore` entries for KAT modules

**Build / CI:**
- No new gates; preserve all existing CI checks
- Re-running CodeQL after this change SHALL produce zero open `rust/hard-coded-cryptographic-value` alerts on the affected paths
- Coverage gate (≥80%) MUST not regress; KAT vectors are still exercised by tests

**Dependencies:**
- No new runtime crate dependencies
- No new offline-tool dependencies

**Architecture:**
- 4-layer acyclic dependency model preserved (no new crate dependencies)
- DSM should report no new layering violations

**Pre-existing dismissed alerts:**
- Out of scope to revisit the 13 dismissed `rust/cleartext-logging` and historical `rust/hard-coded-cryptographic-value` alerts unless they re-fire after this change's suppression strategy lands. If they do, they're added to the same triage table.

**Risk to triage scope:**
- The Code Quality view findings beyond the user's screenshot are unenumerable via API. Reviewer must browse the UI manually. If the count is large (say >50), this change may need to be split — the enumeration task includes a "scope check" gate after the inventory.
