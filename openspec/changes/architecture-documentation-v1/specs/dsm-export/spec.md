## MODIFIED Requirements

### Requirement: DSM JSON output compatibility with analysis tooling
The existing `scripts/dsm-matrix.py` output format SHALL be compatible with `scripts/dsm-analysis.py` input. The JSON SHALL include crate names, adjacency matrix, and dependency kind encoding.

#### Scenario: Pipeline integration
- **WHEN** `just dsm` generates `build/analysis/dsm.json`
- **THEN** `scripts/dsm-analysis.py build/analysis/dsm.json` can consume it without transformation
