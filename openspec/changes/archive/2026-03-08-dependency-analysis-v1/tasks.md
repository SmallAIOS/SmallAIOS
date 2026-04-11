## 1. Tool Setup and Verification

- [x] 1.1 Add `cargo-depgraph` and `cargo-modules` to documented dev tool requirements — update CLAUDE.md or a new `docs/dev-tools.md` with install instructions (`cargo install cargo-depgraph cargo-modules --locked`)
- [x] 1.2 Verify `cargo depgraph` runs against the workspace and produces valid DOT output — test locally, confirm all 18 crates appear in output
- [x] 1.3 Verify `cargo modules dependencies` runs against host-testable crates — test with `smallaios-kernel`, confirm DOT output and `--acyclic` flag works

## 2. Crate-Level Dependency Graph

- [x] 2.1 Add Makefile target `depgraph` — run `cargo depgraph --workspace-only | dot -Tsvg -o build/analysis/crate-deps.svg`, also save DOT file, create `build/analysis/` directory if needed
- [x] 2.2 Handle missing GraphViz gracefully — if `dot` command not found, save DOT file only and print warning about SVG generation skipped
- [x] 2.3 Add `--dedup-transitive-deps` flag to cargo-depgraph to simplify the graph output

## 3. Module-Level Dependency Graphs

- [x] 3.1 Add Makefile target `modgraph` — iterate over host-testable crates, run `cargo modules dependencies --package <pkg> --layout dot` for each, save to `build/analysis/modules/<crate-name>.dot`
- [x] 3.2 Support `CRATE=<name>` variable to generate a single crate's module graph — `make modgraph CRATE=smallaios-kernel`
- [x] 3.3 Add Makefile target `arch-check` — run `cargo modules dependencies --package <pkg> --acyclic` for each host-testable crate, report results summary

## 4. DSM Matrix Export

- [x] 4.1 Create `scripts/dsm-matrix.py` — parse `cargo metadata --no-deps --format-version 1`, build adjacency matrix with dependency kind classification (1=normal, 2=dev, 3=build), output JSON to `build/analysis/dsm-matrix.json`
- [x] 4.2 Add CSV output to `scripts/dsm-matrix.py` — generate `build/analysis/dsm-matrix.csv` with crate names as row/column headers, numeric dependency values in cells
- [x] 4.3 Add Makefile target `dsm` — run `scripts/dsm-matrix.py`, ensure `build/analysis/` directory exists

## 5. CI Integration

- [x] 5.1 Add `dependency-analysis` job to `.github/workflows/ci.yml` — install cargo-depgraph, cargo-modules, and graphviz; run `make depgraph` and `make arch-check`; upload `build/analysis/` as artifact
- [x] 5.2 Set module-level cycle detection to `continue-on-error: true` in CI — advisory only until all existing cycles are resolved
- [x] 5.3 Run DSM matrix generation in CI — include `make dsm` in the dependency-analysis job, upload matrix files as part of the artifact

## 6. Cleanup and Documentation

- [x] 6.1 Verify `build/analysis/` is already covered by existing `.gitignore` entry for `/build/` — if not, add the exclusion
- [x] 6.2 Add a `make arch` convenience target that runs `depgraph`, `modgraph`, and `dsm` together
- [x] 6.3 Update `.pre-commit-config.yaml` to optionally use `cargo-modules --acyclic` for the cycle check hook if the tool is installed, falling back to the existing Python script
