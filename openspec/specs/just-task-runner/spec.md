# just-task-runner Specification

## Purpose
TBD - created by archiving change make-to-just-migration-v1. Update Purpose after archive.
## Requirements
### Requirement: All Make targets have equivalent Just recipes
The Justfile SHALL contain a recipe for every target currently in the Makefile. Recipe names SHALL match existing target names for continuity.

#### Scenario: Build targets exist
- **WHEN** a developer runs `just build-kernel-x86`, `just build-container-arm`, or any existing build target name
- **THEN** the same cargo command executes as the equivalent Make target

#### Scenario: Test and lint targets exist
- **WHEN** a developer runs `just test`, `just clippy`, `just fmt`, or `just fmt-check`
- **THEN** the same cargo commands execute with identical flags

### Requirement: Recipes are discoverable via just --list
All recipes SHALL have doc comments visible in `just --list` output. Related recipes SHALL be grouped logically.

#### Scenario: Listing available recipes
- **WHEN** a developer runs `just --list`
- **THEN** all recipes are displayed with brief descriptions, grouped by category

### Requirement: Parameterized recipes replace Make variable patterns
Recipes that accept user input (device paths, bump levels, crate names) SHALL use Just parameters with sensible defaults where applicable.

#### Scenario: Release with bump parameter
- **WHEN** a developer runs `just release minor`
- **THEN** `cargo release minor --execute` runs, equivalent to `make release BUMP=minor`

#### Scenario: Module graph for specific crate
- **WHEN** a developer runs `just modgraph smallaios-kernel`
- **THEN** only that crate's module graph is generated

#### Scenario: SD card deploy with device parameter
- **WHEN** a developer runs `just deploy-rpi-sdcard /dev/sdX`
- **THEN** the deploy script runs with the specified device

### Requirement: CI uses just for all task execution
The CI workflow SHALL install `just` and use it for all steps that currently invoke `make`.

#### Scenario: CI installs just
- **WHEN** a CI job needs to run a recipe
- **THEN** `just` is installed via `taiki-e/install-action` before use

#### Scenario: CI invokes just recipes
- **WHEN** CI runs build, test, lint, or analysis steps
- **THEN** `just <recipe>` is used instead of `make <target>`

### Requirement: Makefile is removed after migration
The Makefile SHALL be deleted. The project SHALL not maintain both files.

#### Scenario: No Makefile in repository
- **WHEN** the migration is complete
- **THEN** the Makefile no longer exists in the repository root

### Requirement: Documentation reflects just commands
CLAUDE.md and any other documentation referencing `make` commands SHALL be updated to show `just` equivalents.

#### Scenario: CLAUDE.md build commands
- **WHEN** a developer reads CLAUDE.md
- **THEN** all command examples use `just` syntax

