## 1. OpenSpec Archive Consolidation

- [ ] 1.1 Move `openspec/archived/` entries to `openspec/changes/archive/` with date prefixes (use git log dates)
- [ ] 1.2 Remove empty `openspec/archived/` directory
- [ ] 1.3 Archive remaining active changes (codeql-remediation-v1, safety-critical-tooling-v1, release-semver-v1, smallaios-kernel-v1, platform-expansion-v2) with DEFERRED annotations

## 2. DSM Analysis Tooling (Rust)

- [ ] 2.1 Create `tools/dsm/` crate with `Cargo.toml` (binary, edition 2021, std)
- [ ] 2.2 Implement DSM matrix parser: read `dsm.json` into adjacency matrix struct
- [ ] 2.3 Implement transitive closure / reachability computation for propagation cost
- [ ] 2.4 Implement fan-in / fan-out calculation per crate
- [ ] 2.5 Implement coupling cluster detection (strongly connected components via Tarjan's algorithm)
- [ ] 2.6 Implement layering violation detection (define layers, check for skip-layer deps)
- [ ] 2.7 Implement JSON output to `dsm-metrics.json` with all computed metrics
- [ ] 2.8 Implement human-readable stdout summary (table format)
- [ ] 2.9 Add unit tests for DSM calculations (known small graphs with expected metrics)
- [ ] 2.10 Add `just dsm-analyze` recipe to Justfile that runs `just dsm` then the analysis tool

## 3. Architecture Documentation

- [ ] 3.1 Create `docs/architecture.md` with 4-layer model diagram and crate assignments
- [ ] 3.2 Add DSM evidence section: propagation cost table, fan-in/fan-out summary
- [ ] 3.3 Add design rationale section: unikernel constraints, no_std, size goals
- [ ] 3.4 Add dependency rules section: which layers may depend on which
- [ ] 3.5 Add acyclicity guarantee section with dev-dependency cycle explanation
- [ ] 3.6 Update CLAUDE.md workspace architecture section to reference layered model

## 4. CI Integration

- [ ] 4.1 Add DSM analysis step to CI dependency-analysis job (run dsm-analysis tool, upload metrics)
- [ ] 4.2 Add `cargo-modules --acyclic` check to CI (advisory, continue-on-error)
- [ ] 4.3 Update Justfile `arch-check` recipe to also run DSM analysis

## 5. Cleanup

- [ ] 5.1 Remove legacy `scripts/dsm-matrix.py` (replaced by Rust tool) or update it to call the Rust binary
- [ ] 5.2 Verify `just dsm` and `just dsm-analyze` work end-to-end locally
