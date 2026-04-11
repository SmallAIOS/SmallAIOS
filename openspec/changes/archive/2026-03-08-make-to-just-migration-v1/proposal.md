## Why

The project uses GNU Make as a task runner wrapping Cargo commands. Make has tab-sensitivity issues, limited argument handling, and platform-specific behavior (BSD make vs GNU make). `just` is the Rust ecosystem standard for task running — simpler syntax, better argument passing, built-in help, and cross-platform consistency. Since SmallAIOS is pure Rust with Cargo as the build system, `just` is the natural fit.

## What Changes

- Create a `Justfile` with all recipes converted from existing Makefile targets
- Update `.github/workflows/ci.yml` to install and use `just` instead of `make`
- Update `CLAUDE.md` to document `just` commands instead of `make` commands
- Update `.pre-commit-config.yaml` hooks that reference `make` targets
- Remove `Makefile` after migration is verified
- Add `just` to documented dev tool requirements

## Capabilities

### New Capabilities
- `just-task-runner`: Convert all Make targets to Just recipes with improved argument handling, built-in `--list` for discoverability, and recipe documentation via comments

### Modified Capabilities

## Impact

- `Makefile` — removed entirely
- `Justfile` — new file, replaces Makefile
- `.github/workflows/ci.yml` — all `make` invocations become `just` invocations
- `CLAUDE.md` — build command documentation updated
- `.pre-commit-config.yaml` — hooks referencing make targets updated
- Developer workflow: `make test` becomes `just test`, etc.
- CI runners need `just` installed (single binary, no deps)
