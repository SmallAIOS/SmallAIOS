## Why

The 18-crate workspace has grown complex enough that dependency relationships are no longer obvious from reading Cargo.toml files. The existing `scripts/check-cycles.sh` catches circular dependencies but provides no visualization, no module-level analysis, and no structured data for architectural review. Integrating `cargo-depgraph` and `cargo-modules` gives developers crate-level and module-level dependency graphs, automated cycle detection at the module granularity, and a DSM-compatible adjacency matrix export for future integration with tools like Lattix.

## What Changes

- Install and integrate `cargo-depgraph` for workspace-level dependency graph generation (DOT/SVG)
- Install and integrate `cargo-modules` for per-crate module-level dependency graphs and cycle detection
- Add a script to generate a DSM-style adjacency matrix (JSON/CSV) from `cargo metadata` for architectural analysis and future Lattix import
- Add Makefile targets: `make depgraph`, `make modgraph`, `make dsm`, `make arch-check`
- Add CI job to generate dependency graph artifacts on PRs and detect module-level cycles
- Enhance the existing cycle detection to use `cargo-modules --acyclic` for deeper module-level checks
- Add generated graph artifacts to `.gitignore`

## Capabilities

### New Capabilities
- `dependency-visualization`: Crate-level and module-level dependency graph generation using cargo-depgraph and cargo-modules, with DOT/SVG output and Makefile integration
- `dsm-export`: Design Structure Matrix adjacency matrix export from cargo metadata in JSON/CSV format for architectural analysis and Lattix compatibility

### Modified Capabilities

## Impact

- **Makefile**: New targets for graph generation, DSM export, and architecture checks
- **CI**: New job for dependency graph artifact generation and module-level cycle detection
- **scripts/**: New scripts for DSM matrix generation; enhanced cycle detection
- **.gitignore**: Exclude generated graph/SVG artifacts
- **Dev dependencies**: `cargo-depgraph` and `cargo-modules` as developer tools (not crate deps)
- **.pre-commit-config.yaml**: Optionally enhance cycle check hook with module-level detection
