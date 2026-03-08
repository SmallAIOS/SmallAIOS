## ADDED Requirements

### Requirement: 4+1 View architecture documentation
The project SHALL maintain architecture documentation following the Kruchten 4+1 View Model, with diagrams in PlantUML source files stored in `docs/architecture/`.

#### Scenario: Logical view documents crate decomposition
- **WHEN** the logical view diagram is rendered
- **THEN** it SHALL show all 18 workspace crates, their public interfaces, and dependency relationships

#### Scenario: Process view documents runtime behavior
- **WHEN** the process view diagrams are rendered
- **THEN** they SHALL show the cooperative scheduler flow, interrupt handling, and ONNX inference pipeline

#### Scenario: Physical view documents deployment targets
- **WHEN** the physical view diagram is rendered
- **THEN** it SHALL show all target platforms (x86-64, AArch64, RISC-V, Jetson, container) and their build configurations

### Requirement: Acyclic crate dependency graph
The workspace crate dependency graph SHALL be a directed acyclic graph (DAG). CI SHALL verify no cycles exist.

#### Scenario: CI detects cyclic dependency
- **WHEN** a PR introduces a circular dependency between workspace crates
- **THEN** the dependency check CI job SHALL fail and report the cycle

#### Scenario: Clean DAG on develop
- **WHEN** the dependency graph is checked on the develop branch
- **THEN** the check SHALL pass confirming no cycles exist

### Requirement: PlantUML diagrams rendered in documentation
All architecture PlantUML diagrams SHALL be rendered as SVG in the Sphinx documentation site.

#### Scenario: PlantUML diagram renders in docs build
- **WHEN** `make docs` builds the Sphinx site
- **THEN** all `.puml` files in `docs/architecture/` SHALL be rendered as inline SVG images
