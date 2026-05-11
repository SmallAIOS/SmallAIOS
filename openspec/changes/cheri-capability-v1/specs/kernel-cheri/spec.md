## ADDED Requirements

### Requirement: CHERI alignment documentation maintained alongside the capability system

The repository SHALL maintain a CHERI alignment document (`docs/cheri-alignment.md`) that maps the SmallAIOS in-software capability model (`kernel/src/cap.rs`) to the CHERI hardware capability model, identifies gaps between them, and is refreshed whenever the SmallAIOS capability model changes substantively.

#### Scenario: Alignment document exists and opens with a research-stage disclaimer

- **GIVEN** the SmallAIOS repository
- **WHEN** a reader opens `docs/cheri-alignment.md`
- **THEN** the document's opening section SHALL contain the statement "This document is alignment-on-paper, not hardware-tested. SmallAIOS does not run on CHERI silicon today." (or substantively equivalent wording)
- **AND** the document SHALL contain at least three sections: capability-field mapping, permissions mapping, and gap analysis
- **AND** the document SHALL cite specific `kernel/src/cap.rs` line numbers when describing the SmallAIOS-side primitives

#### Scenario: Capability-field mapping table

- **GIVEN** the alignment document
- **WHEN** a reader inspects the field-mapping section
- **THEN** it SHALL include a table covering at least these CHERI capability fields: `tag`, `base`, `length`, `address`, `perms`, `otype`
- **AND** for each CHERI field, the table SHALL state the corresponding SmallAIOS-side primitive (or "no direct analog" with an explanation)
- **AND** the table SHALL be sourced from CHERI ISAv9 (or whatever version is current at doc-update time) with a citation

#### Scenario: Alignment doc refreshes on capability-model change

- **GIVEN** a future SmallAIOS change that modifies `kernel/src/cap.rs`'s capability shape
- **WHEN** that change is proposed
- **THEN** the change's tasks list SHALL include refreshing `docs/cheri-alignment.md` to reflect the new shape
- **AND** validation of the change SHALL fail if the doc is not updated

### Requirement: CHERI compile experiment evidence published

The repository SHALL publish empirical evidence from at least one attempt to compile the SmallAIOS capability primitives under the CHERI-Rust research toolchain, as a one-shot snapshot rather than a CI-gated activity.

#### Scenario: Compile experiment results stored under the change's notes

- **GIVEN** the change `cheri-capability-v1` has completed Phase 2
- **WHEN** a reviewer inspects the change directory
- **THEN** the directory SHALL contain `notes/cheri-compile-experiment.md` (or, when the change is archived, the equivalent file under `openspec/changes/archive/<date>-cheri-capability-v1/notes/`)
- **AND** the file SHALL document: the `cheri-rust` toolchain version used, the build invocation, the errors encountered (clustered by category), the hand-fixes applied if any, and the conclusion (% of the targeted code that compiled cleanly)

#### Scenario: Experiment is one-shot, not CI-gated

- **GIVEN** the SmallAIOS CI configuration
- **WHEN** the CI pipeline runs
- **THEN** no CI job SHALL depend on the `cheri-rust` toolchain
- **AND** the experiment SHALL NOT be automated to re-run on every PR — the published evidence is a snapshot, not a continuous regression check

### Requirement: CHERI implementation work is explicitly deferred

The change `cheri-capability-v1` SHALL produce documentation only, and SHALL NOT modify any Rust source code or Cargo manifest in the repository. Implementation work targeting CHERI silicon is explicitly deferred to a future change (`cheri-capability-v2` or successor) that is gated on production-grade CHERI hardware availability.

#### Scenario: This change introduces no Rust code

- **GIVEN** the diff of the `cheri-capability-v1` change against `develop`
- **WHEN** a reviewer inspects the diff
- **THEN** the diff SHALL contain no `.rs` files, no `Cargo.toml` modifications, no `.cargo/config.toml` modifications, and no CI workflow changes
- **AND** the diff SHALL consist of documentation (`docs/`, `notes/`) and the OpenSpec change directory only

#### Scenario: Implementation follow-up is named and tracked

- **GIVEN** the alignment doc's "roadmap if silicon matures" section
- **WHEN** a reader inspects it
- **THEN** the section SHALL name the follow-up change (`cheri-capability-v2`) and list its in-dependency-order tasks at a high level
- **AND** the section SHALL state explicitly that the follow-up is gated on production-grade CHERI silicon availability — it is NOT a committed roadmap item
