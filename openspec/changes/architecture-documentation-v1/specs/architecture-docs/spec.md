## ADDED Requirements

### Requirement: Layered architecture documentation
`docs/architecture.md` SHALL document the 4-layer dependency model with descriptions of each layer's purpose, crate membership, and dependency rules.

#### Scenario: Layer definitions
- **WHEN** a developer reads docs/architecture.md
- **THEN** they can identify which layer each crate belongs to
- **AND** understand which layers a crate may depend on (only same or lower layers)

### Requirement: DSM evidence in documentation
The architecture document SHALL include the DSM matrix (or a summary) as evidence that the dependency structure matches the documented layers.

#### Scenario: DSM reference
- **WHEN** the architecture document is generated
- **THEN** it includes fan-in/fan-out data and propagation cost for key crates
- **AND** confirms zero production-dependency cycles

### Requirement: Design rationale
The document SHALL explain WHY the layered model was chosen, including unikernel constraints, no_std requirements, and size optimization goals.

#### Scenario: Rationale section
- **WHEN** a developer asks why crates are structured this way
- **THEN** the rationale section explains the design decisions

### Requirement: CLAUDE.md architecture update
The workspace architecture section in CLAUDE.md SHALL be updated to reference the layered model and docs/architecture.md.

#### Scenario: CLAUDE.md consistency
- **WHEN** CLAUDE.md is read by an AI assistant
- **THEN** the architecture section matches the documented 4-layer model

### Requirement: OpenSpec archive consolidation
All archived changes SHALL reside in `openspec/changes/archive/` with `YYYY-MM-DD-` date prefixes. The legacy `openspec/archived/` directory SHALL be removed.

#### Scenario: Archive migration
- **WHEN** archive consolidation is complete
- **THEN** `openspec/archived/` no longer exists
- **AND** all previously archived changes are in `openspec/changes/archive/` with date prefixes
