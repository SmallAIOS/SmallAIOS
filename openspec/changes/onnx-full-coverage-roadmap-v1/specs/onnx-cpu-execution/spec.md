## ADDED Requirements

### Requirement: Operator Coverage Roadmap
The ONNX runtime SHALL maintain a single authoritative roadmap document
at `docs/onnx-coverage-roadmap.md` that catalogs every standard ONNX
operator and assigns it to one of: an implemented tier, a planned
future tier, a deferred subsystem, or an explicitly skipped category.

#### Scenario: Roadmap is the source of truth
- **WHEN** a contributor needs to know whether an ONNX operator is
  implemented, planned, deferred, or skipped
- **THEN** they MUST be able to find the answer in
  `docs/onnx-coverage-roadmap.md`
- **AND** the roadmap MUST list a target tier and a target model class
  for every planned operator

#### Scenario: Roadmap stays in sync with the registry
- **WHEN** a new operator is added to `OperatorRegistry`
- **THEN** its entry in `docs/onnx-coverage-roadmap.md` MUST be
  updated to mark it implemented in the same PR

### Requirement: Tier Naming Convention
Each future operator-coverage OpenSpec change SHALL use a tier name
reserved by this roadmap. Tier names MUST follow the pattern
`<model-class>-models-v1` (e.g., `transformer-models-v1`,
`vision-transformers-v1`) or `<capability>-v1` for non-model tiers
(e.g., `int8-kernels-v1`, `control-flow-v1`).

#### Scenario: A new tier change uses a reserved name
- **WHEN** a contributor opens a new operator-coverage OpenSpec change
- **THEN** the change name MUST match a tier listed in the roadmap
  document
- **AND** the change proposal MUST cite the roadmap as its parent

### Requirement: Agent-Team Execution Playbook
The roadmap document SHALL include an "Agent-Team Execution"
section describing the worktree-per-tier pattern, the file-ownership
rules for parallel implementation agents, and the validation gates
each tier must pass before merging to `develop`.

#### Scenario: Agents work in parallel without merge conflicts
- **WHEN** two tiers are being implemented concurrently in separate
  worktrees
- **THEN** the playbook MUST specify which files are append-only and
  which require sequential ownership
- **AND** each tier's PR MUST pass `just fmt`, `just clippy`, and
  `just test` before merge
