## Context

SmallAIOS is an 18-crate Rust workspace. Dependency relationships are currently checked only for cycles via `scripts/check-cycles.sh` (a Python script parsing `cargo metadata`). There is no visualization, no module-level analysis, and no structured export for architectural review tools.

`cargo-depgraph` generates crate-level dependency graphs as DOT/GraphViz output. `cargo-modules` provides module-level structure visualization within individual crates and includes `--acyclic` cycle detection. Both are mature, maintained tools available via `cargo install`.

## Goals / Non-Goals

**Goals:**
- Crate-level dependency graph generation (DOT → SVG) via `cargo-depgraph`
- Module-level dependency graphs per crate via `cargo-modules`
- Module-level cycle detection using `cargo-modules dependencies --acyclic`
- DSM-style adjacency matrix export (JSON/CSV) from `cargo metadata` for future Lattix integration
- Makefile targets for local developer use
- CI job generating graph artifacts on PRs
- All generated artifacts excluded from git

**Non-Goals:**
- Full Lattix integration (future change)
- Bazel build migration (separate change)
- Automated dependency optimization recommendations
- Interactive web-based graph exploration

## Decisions

### 1. cargo-depgraph for crate-level graphs
**Choice**: Use `cargo-depgraph` over `cargo-deps` or `cargo-tree`
**Rationale**: cargo-depgraph uses `cargo metadata` (stable API), supports filtering workspace-only deps, and produces clean DOT output with visual differentiation for dependency kinds (dev, build, normal). cargo-deps is unmaintained. cargo-tree is text-only.

### 2. cargo-modules for module-level analysis
**Choice**: Use `cargo-modules` for per-crate module graphs and cycle detection
**Rationale**: It's the only tool that provides module-level dependency graphs (not just module trees). The `--acyclic` flag gives module-level cycle detection that our existing script can't do. Supports DOT output for consistent visualization.

### 3. Custom DSM script over existing tools
**Choice**: Write a Python script to generate DSM adjacency matrices from `cargo metadata`
**Rationale**: No existing open-source tool generates Lattix-compatible DSM output from Cargo workspaces. The data is available in `cargo metadata --format-version 1` JSON. A simple Python script can produce both JSON (for programmatic use) and CSV (for spreadsheet/Lattix import).

### 4. Generated artifacts go to build/ directory
**Choice**: Output all graphs and matrices to `build/analysis/` (already gitignored via `build/`)
**Rationale**: Consistent with existing convention (`build/` is used for QEMU artifacts). No changes to `.gitignore` needed. CI uploads as job artifacts.

### 5. CI as optional quality gate
**Choice**: Module-level cycle detection runs in CI but uses `continue-on-error: true` initially
**Rationale**: We may have existing module-level cycles that need to be addressed incrementally. Start as advisory, then tighten to required once clean.

## Risks / Trade-offs

- **[Tool availability in CI]** → Install via `cargo install` with `--locked` flag; cache cargo bin directory between CI runs
- **[cargo-modules slow on large crates]** → Only run on crates that have changed (use path filters) or accept ~30s overhead per crate
- **[DSM format compatibility]** → Start with generic JSON/CSV; validate against Lattix import format when that integration begins
- **[GraphViz not installed in CI]** → Use `sudo apt-get install graphviz` in CI step; DOT files are still useful without SVG rendering
