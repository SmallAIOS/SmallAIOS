# dsm-export Specification

## Purpose
TBD - created by archiving change dependency-analysis-v1. Update Purpose after archive.
## Requirements
### Requirement: DSM adjacency matrix generation
The system SHALL provide a script `scripts/dsm-matrix.py` that reads `cargo metadata` and produces a Design Structure Matrix as both JSON and CSV, showing crate-to-crate dependencies for all workspace members.

#### Scenario: Generate DSM matrix
- **WHEN** developer runs `make dsm`
- **THEN** `build/analysis/dsm-matrix.json` and `build/analysis/dsm-matrix.csv` are created

#### Scenario: JSON format includes crate names and adjacency data
- **WHEN** DSM JSON is generated
- **THEN** the JSON contains a `crates` array of crate names and a `matrix` 2D array where `matrix[i][j] = 1` if crate `i` depends on crate `j`, and `0` otherwise

#### Scenario: CSV format is human-readable
- **WHEN** DSM CSV is generated
- **THEN** the first row and first column contain crate names, and cell values are `1` (depends) or empty (no dependency)

### Requirement: DSM includes dependency classification
The DSM matrix SHALL distinguish between normal, dev, and build dependencies using different numeric values.

#### Scenario: Dependency types encoded in matrix
- **WHEN** DSM JSON is generated
- **THEN** `matrix[i][j]` values use: `1` for normal dependency, `2` for dev dependency, `3` for build dependency, `0` for no dependency

### Requirement: DSM JSON output compatibility with analysis tooling
The existing `scripts/dsm-matrix.py` output format SHALL be compatible with `scripts/dsm-analysis.py` input. The JSON SHALL include crate names, adjacency matrix, and dependency kind encoding.

#### Scenario: Pipeline integration
- **WHEN** `just dsm` generates `build/analysis/dsm.json`
- **THEN** `scripts/dsm-analysis.py build/analysis/dsm.json` can consume it without transformation

### Requirement: DSM export compatible with Lattix import
The DSM CSV output SHALL use a format compatible with generic DSM import tools (row/column headers as component names, numeric cells for relationships).

#### Scenario: CSV loadable in spreadsheet
- **WHEN** the CSV file is opened in a spreadsheet application
- **THEN** it displays a square matrix with crate names on both axes and dependency indicators in cells

