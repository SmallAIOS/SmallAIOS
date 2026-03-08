## Context

SmallAIOS uses a GNU Makefile as a task runner wrapping Cargo commands. The Makefile has ~60 targets covering builds (container, kernel, Jetson), QEMU, Docker, testing, formal verification, deployment, dependency analysis, and release management. All actual compilation is done by Cargo — Make only orchestrates.

`just` is a command runner (not a build system) purpose-built for this use case. It's the most popular Make replacement in the Rust ecosystem, with ~500K monthly installs from crates.io.

## Goals / Non-Goals

**Goals:**
- 1:1 conversion of all Makefile targets to Just recipes
- CI workflows use `just` instead of `make`
- Documentation reflects `just` commands
- Pre-commit hooks updated to use `just` where applicable
- Zero behavior changes — same commands, same outputs

**Non-Goals:**
- Changing the build system (Cargo stays)
- Restructuring or consolidating existing targets
- Adding new functionality beyond what Make provides
- Supporting both Make and Just simultaneously (clean cut)

## Decisions

### 1. Just syntax conventions

**Decision:** Use `just` features that improve on Make:
- Recipe doc comments (`# comment above recipe`) for `just --list` discoverability
- Named parameters instead of Make's `$(VAR)` pattern (e.g., `just deploy-rpi-sdcard /dev/sdX` instead of `make deploy-rpi-sdcard DEV=/dev/sdX`)
- `set shell` for consistent bash usage
- Group related recipes with `[group]` attributes (just 1.23+)

**Rationale:** Take advantage of Just's features rather than doing a mechanical 1:1 translation. The UX improvement is the main motivation for migrating.

### 2. CI installation

**Decision:** Use `taiki-e/install-action@just` in GitHub Actions.

**Rationale:** Single-line install, caches the binary, widely used in Rust CI pipelines. Alternative: `cargo install just` is slower (compiles from source).

### 3. Variable handling

**Decision:** Convert Make variables to Just variables and recipe parameters:
- `CARGO`, `DOCKER`, `QEMU_*` → Just variables at top of file
- `DEV=`, `BUMP=`, `L4T=`, `GPU=`, `CRATE=` → recipe parameters with defaults where appropriate
- `BUILD_STD`, `FEATURES` → internal Just variables

**Rationale:** Just has cleaner variable/parameter syntax than Make. Parameters with defaults eliminate the `ifdef` pattern.

### 4. Migration strategy

**Decision:** Single atomic commit — remove Makefile, add Justfile, update all references.

**Rationale:** No value in a transition period. The Justfile is a direct conversion, and all CI/docs references are updated in the same change.

## Risks / Trade-offs

- **[Developer muscle memory]** → `just` and `make` commands are nearly identical (`just test` vs `make test`). Shell aliases can bridge the gap.
- **[CI runner doesn't have just]** → Using `taiki-e/install-action` which is reliable and cached. Fallback: `curl` install from GitHub releases.
- **[Pre-commit hooks]** → Hooks that call `make` targets need updating. The `module-cycles` hook already uses `bash -c` so switching to `just` is straightforward.
