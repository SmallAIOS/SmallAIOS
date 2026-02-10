# Delta for Documentation System

## ADDED Requirements

### Requirement: Sphinx-needs Requirements Engineering
The project SHALL use Sphinx-needs to manage requirements, specifications, implementations, tests, and verification results with full bidirectional traceability.

#### Scenario: Define requirement with traceability
- **WHEN** a requirement is documented in the Sphinx-needs system
- **THEN** it MUST use one of the defined need types: REQ (requirement), SPEC (specification), IMPL (implementation), TEST (test case), VERIFY (verification result)
- **AND** each need MUST have a unique ID, title, description, and status
- **AND** each need MUST link to its upstream and downstream needs via `links` and `links_back`

#### Scenario: Traceability matrix is complete
- **WHEN** the Sphinx documentation is built
- **THEN** the generated traceability matrix MUST show all REQ → SPEC → IMPL → TEST → VERIFY chains
- **AND** orphan needs (needs with no upstream or downstream links) MUST be flagged as warnings
- **AND** the matrix MUST be exportable as HTML and CSV

#### Scenario: Need status tracking
- **WHEN** a need's implementation status changes
- **THEN** the need status MUST be updated to one of: open, in_progress, implemented, verified, rejected
- **AND** the documentation build MUST show a summary dashboard with counts per status

### Requirement: PlantUML Architecture Diagrams
The project SHALL include PlantUML diagrams for all major architectural views, generated from version-controlled text sources.

#### Scenario: Component diagram shows system decomposition
- **WHEN** the component diagram is generated
- **THEN** it MUST show all 10 Rust crates and their dependency relationships
- **AND** it MUST show the HAL boundary separating platform-specific from platform-independent code
- **AND** the diagram source MUST be stored as `.puml` files in the `docs/` directory

#### Scenario: Sequence diagram for inference request flow
- **WHEN** the inference sequence diagram is generated
- **THEN** it MUST show the complete message flow from external TCP client through IPC router, inference dispatcher, ONNX session, execution provider, and back
- **AND** the diagram MUST show parallel paths for CPU and GPU execution

#### Scenario: State machine diagram for task lifecycle
- **WHEN** the task state machine diagram is generated
- **THEN** it MUST show all task states (Created, Ready, Running, Waiting, Completed) and valid transitions
- **AND** it MUST match the actual state machine in the scheduler implementation

#### Scenario: Deployment diagram for container and bare metal
- **WHEN** the deployment diagram is generated
- **THEN** it MUST show both container mode (SmallAIOS as process in Docker/K8s) and VM/bare metal mode
- **AND** it MUST show GPU passthrough architecture in both modes

### Requirement: DO-178C Document Artifacts
The project SHALL produce all DO-178C required document artifacts for DAL A certification.

#### Scenario: Plan for Software Aspects of Certification (PSAC)
- **WHEN** the PSAC document is generated
- **THEN** it MUST describe the system overview, software overview, certification considerations, software lifecycle, schedule, and additional considerations
- **AND** it MUST reference all subordinate plans (SDP, SVP, SCMP, SQAP)

#### Scenario: Software Requirements Standards (SRS)
- **WHEN** the SRS document is generated
- **THEN** it MUST define the methods, rules, and tools for developing high-level software requirements
- **AND** it MUST reference the Sphinx-needs REQ need type as the requirements management mechanism

#### Scenario: Software Design Standards (SDD)
- **WHEN** the SDD document is generated
- **THEN** it MUST define the architecture, low-level requirements, and data flow for each software component
- **AND** it MUST reference the Sphinx-needs SPEC and IMPL need types

#### Scenario: Software Verification Results document
- **WHEN** the verification results document is generated
- **THEN** it MUST include test results, code review records, MC/DC coverage reports, static analysis results, and formal verification results
- **AND** all results MUST be traceable to their corresponding VERIFY needs in Sphinx-needs

### Requirement: Auto-generated API Documentation
The project SHALL generate API documentation from Rust doc comments integrated with the Sphinx-needs traceability system.

#### Scenario: Rust docs generated with cargo doc
- **WHEN** `cargo doc` is run on the workspace
- **THEN** it MUST generate HTML documentation for all public APIs in all crates
- **AND** each public function, struct, and trait MUST have a doc comment

#### Scenario: Cross-reference between Sphinx-needs and Rust docs
- **WHEN** an IMPL need references a Rust module or function
- **THEN** the Sphinx documentation MUST include a hyperlink to the corresponding cargo doc page
- **AND** the cargo doc page MUST include a reference back to the IMPL need ID

### Requirement: Documentation CI Pipeline
Documentation SHALL be built and validated automatically in CI.

#### Scenario: Sphinx build succeeds with zero warnings
- **WHEN** CI builds the Sphinx documentation
- **THEN** the build MUST succeed with zero warnings
- **AND** all PlantUML diagrams MUST render successfully
- **AND** all Sphinx-needs cross-references MUST resolve

#### Scenario: Traceability gap detection
- **WHEN** CI analyzes the Sphinx-needs traceability matrix
- **THEN** it MUST fail the build if any REQ has no linked SPEC
- **AND** MUST fail if any SPEC has no linked IMPL
- **AND** MUST fail if any IMPL has no linked TEST
