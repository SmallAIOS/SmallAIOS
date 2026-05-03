## ADDED Requirements

### Requirement: Adversarial test playbook
The kernel SHALL maintain a red-team playbook at `docs/red-team-playbook.md` that enumerates every threat-model line with at least one corresponding adversarial test in `crates/red-team-tests/`.

#### Scenario: Every in-scope threat has a test
- **WHEN** the threat model lists a threat as in-scope
- **THEN** the playbook MUST cite a test module name and the test MUST exist under `crates/red-team-tests/tests/`

#### Scenario: Out-of-scope threats are explicitly marked
- **WHEN** a threat is out-of-scope (physical, host kernel, etc.)
- **THEN** the playbook MUST note the rationale rather than omit it

### Requirement: Capability invariants are property-tested
The kernel SHALL verify capability invariants via `proptest` properties covering delegation, revocation, ID uniqueness, aliasing, quota, and audit completeness.

#### Scenario: Delegated rights never exceed parent
- **WHEN** a parent capability delegates to a child
- **THEN** the child rights set MUST be a subset of the parent rights set, and the property test MUST exercise this on randomized chains

#### Scenario: Revocation cascades within one tick
- **WHEN** a parent capability is revoked
- **THEN** all transitively delegated children MUST be invalidated before the next scheduler tick

#### Scenario: Capability IDs are not reused
- **WHEN** a capability is revoked
- **THEN** its ID MUST NOT be issued to any future capability for the lifetime of the kernel instance

### Requirement: Adversarial inputs are refused or audited
All adversarial scenarios in the playbook MUST result in one of: refusal at the boundary, panic-halt with audit log entry, or rate-limited drop with audit log entry — never silent acceptance.

#### Scenario: Forged capability handle is refused
- **WHEN** a syscall presents a capability handle that was never issued by the kernel
- **THEN** the call MUST return an error and emit an audit log entry

#### Scenario: Malformed ONNX graph fails at load
- **WHEN** an ONNX model contains a cyclic graph, op-count overflow, or attribute type confusion
- **THEN** `onnx-rt` MUST refuse to load before any tensor allocation

#### Scenario: IPC topic flood does not panic
- **WHEN** N publishers flood a single topic past its queue depth
- **THEN** the kernel MUST apply back-pressure or drop with audit and MUST NOT panic

### Requirement: Attack-surface inventory is current
The kernel SHALL maintain `docs/attack-surface.md` enumerating every external interface (syscalls, listen ports, IPC topics, USB endpoint classes, CAN routes, GPU device ABI) with its current adversarial coverage status.

#### Scenario: New external interface adds an attack-surface row
- **WHEN** a PR introduces a new syscall, listen port, or IPC topic
- **THEN** the PR MUST add a row to `docs/attack-surface.md` with coverage status `NONE`, `FUZZ-ONLY`, `PROPERTY`, or `INTEGRATION`
