# dsm-metrics Specification

## Purpose
Provides DSM-based metrics analysis including propagation cost, fan-in/fan-out, coupling cluster detection, and layering violation detection for the SmallAIOS workspace.

## Requirements
### Requirement: DSM propagation cost calculation
`scripts/dsm-analysis.py` SHALL compute propagation cost for each crate as the percentage of workspace crates transitively affected by a change to that crate.

#### Scenario: Propagation cost for foundation crate
- **WHEN** dsm-analysis.py processes dsm.json containing the kernel crate with 16 dependents
- **THEN** kernel's propagation cost is reported as ~89% (16/18 crates affected)

### Requirement: Fan-in and fan-out metrics
The script SHALL report fan-in (number of crates depending on this crate) and fan-out (number of crates this crate depends on) for each workspace crate.

#### Scenario: Leaf crate metrics
- **WHEN** dsm-analysis.py processes a crate with zero dependents (e.g., bus)
- **THEN** fan-in is reported as 0 and fan-out reflects its direct dependencies

### Requirement: Coupling cluster detection
The script SHALL identify groups of crates with mutual dev-dependency relationships and report them as coupling clusters.

#### Scenario: Dev-dependency cycle cluster
- **WHEN** dsm-analysis.py detects kernel->security->net->kernel via dev deps
- **THEN** it reports a coupling cluster containing {kernel, security, net} with annotation "dev-dependency only"

### Requirement: JSON output format
The script SHALL output a structured JSON report to `build/analysis/dsm-metrics.json` containing propagation_cost, fan_in, fan_out, and clusters fields.

#### Scenario: Output file generation
- **WHEN** dsm-analysis.py runs successfully
- **THEN** `build/analysis/dsm-metrics.json` is created with all metrics
- **AND** a human-readable summary is printed to stdout

### Requirement: Layering violation detection
The script SHALL detect dependencies that skip architectural layers (e.g., an arch crate depending directly on container) and report them as layering violations.

#### Scenario: Clean architecture
- **WHEN** no dependencies skip layers in the current workspace
- **THEN** the layering violations list is empty
