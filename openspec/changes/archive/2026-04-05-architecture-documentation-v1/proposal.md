## Why

The project has grown to 18 crates with a clean acyclic dependency graph, but there is no formal architecture documentation, no automated DSM metrics, and the OpenSpec archive is split across two locations. As we approach a release, we need architecture documentation that reflects the actual layered design, tooling to calculate coupling metrics from the DSM export, CI enforcement of acyclicity, and a consolidated archive structure. Active OpenSpec changes with only hardware-deferred tasks should also be formally closed out.

## What Changes

- Consolidate `openspec/archived/` into `openspec/changes/archive/` with date prefixes for consistency
- Close out active OpenSpec changes that have only hardware-deferred or admin-gate tasks remaining (codeql-remediation-v1, smallaios-kernel-v1, platform-expansion-v2, safety-critical-tooling-v1, release-semver-v1)
- Add `scripts/dsm-analysis.py` to compute coupling metrics (propagation cost, cluster detection, fan-in/fan-out analysis) from the DSM JSON export
- Create architecture documentation in `docs/architecture.md` covering the 4-layer dependency model, DSM interpretation, and design rationale
- Update CLAUDE.md workspace architecture section to match the documented layered model
- Add `cargo-modules --acyclic` enforcement to CI as a required check

## Capabilities

### New Capabilities
- `dsm-metrics`: DSM analysis tooling — propagation cost calculation, coupling cluster detection, fan-in/fan-out reporting from dsm.json
- `architecture-docs`: Formal architecture documentation — layered dependency model, design rationale, acyclicity guarantees

### Modified Capabilities
- `dsm-export`: Add integration with new dsm-metrics analysis (output format compatibility)
- `dependency-visualization`: Add acyclicity enforcement in CI via cargo-modules

## Impact

- `scripts/dsm-analysis.py` — new analysis script consuming `build/analysis/dsm.json`
- `docs/architecture.md` — new architecture documentation
- `CLAUDE.md` — updated architecture section
- `.github/workflows/ci.yml` — cargo-modules acyclicity check added
- `openspec/archived/` — contents moved to `openspec/changes/archive/`
- `openspec/changes/` — 5 active changes archived with deferred-task annotations
