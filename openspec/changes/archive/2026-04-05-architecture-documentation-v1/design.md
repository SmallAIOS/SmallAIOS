## Context

SmallAIOS is an 18-crate Rust workspace with a clean acyclic dependency graph at the crate level. DSM analysis confirms 36 dependency edges (31 normal, 5 dev) with zero production cycles. The architecture follows a natural 4-layer model:

1. **Foundation:** `security` (zero workspace deps) → `kernel` (depends only on security)
2. **Core services:** `net`, `ipc`, `posix`, `onnx-rt`, `usb` (depend on kernel ± security)
3. **Hardware abstraction:** arch crates, `peripheral`, `bus`, `sdr` (depend on kernel)
4. **Integration:** `container` (7 deps, top-level entry point), `bench` (testing only)

The OpenSpec archive is split across `openspec/archived/` (older, no date prefix) and `openspec/changes/archive/` (recent, date-prefixed). Five active changes have only hardware-deferred or admin tasks remaining.

## Goals / Non-Goals

**Goals:**
- Document the 4-layer architecture with DSM evidence
- Provide automated coupling metrics (propagation cost, cluster detection)
- Enforce acyclicity in CI
- Consolidate OpenSpec archive to single location with consistent naming
- Close out changes that are effectively complete (only hardware-deferred tasks remain)

**Non-Goals:**
- Refactoring the dependency graph (it's already clean — no cycles to remove)
- Breaking the dev-dependency cycle (kernel↔security↔net — this is benign and Cargo-permitted)
- Completing hardware-dependent tests (those require physical devices)
- Changing the crate structure or splitting/merging crates

## Decisions

1. **Single archive location:** Move everything to `openspec/changes/archive/` with `YYYY-MM-DD-` prefix. Delete `openspec/archived/` after migration. Rationale: single source of truth, consistent with recent convention.

2. **DSM metrics script:** `scripts/dsm-analysis.py` reads `build/analysis/dsm.json` and computes:
   - Propagation cost (% of system affected by changes to each crate)
   - Fan-in/fan-out per crate
   - Coupling clusters (groups of tightly-coupled crates)
   - Layering violations (dependencies that skip layers)
   Output: JSON report + human-readable summary to stdout.

3. **Architecture doc location:** `docs/architecture.md` — alongside existing docs. Not auto-generated; hand-written with DSM data as evidence.

4. **CI acyclicity enforcement:** Add `cargo-modules dependencies --acyclic` check. Make it advisory (continue-on-error) initially since cargo-modules may not handle all workspace configurations. Promote to required after validation.

5. **Closing active changes:** Archive with explicit `DEFERRED` annotations in tasks.md listing what remains and why (hardware requirement). This preserves traceability without blocking the archive.

## Risks / Trade-offs

- **cargo-modules availability:** Tool may not be available on all CI runners. Mitigated by installing via `cargo install` and using `continue-on-error`.
- **Archive migration:** Moving `openspec/archived/` entries requires adding date prefixes. Use git log to determine original completion dates. Risk: minor date inaccuracy for older changes.
- **DSM metrics interpretation:** Propagation cost thresholds are project-specific. Document expected ranges rather than hard-coding pass/fail criteria.
