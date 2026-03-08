# formal-verification Specification

## Purpose
TBD - created by archiving change smallaios-kernel-v1. Update Purpose after archive.
## Requirements
### Requirement: TLA+ Scheduler Concurrency Model
The project SHALL maintain TLA+ models for the cooperative async scheduler proving absence of deadlock, starvation, and ensuring fairness.

#### Scenario: Prove no deadlock in scheduler
- WHEN the TLA+ model checker (TLC) verifies the scheduler model
- THEN the model MUST prove that no reachable state exists where all tasks are blocked and no progress is possible
- AND the model MUST cover work-stealing interactions between all core counts from 1 to 16

#### Scenario: Prove no starvation
- WHEN the TLA+ model includes tasks at all four priority levels (GPU, System, Normal, Low)
- THEN the model MUST prove that every runnable task is eventually scheduled
- AND fairness constraints MUST ensure low-priority tasks are not indefinitely starved by higher-priority tasks

#### Scenario: Prove work-stealing fairness
- WHEN the TLA+ model represents per-core run queues with work-stealing
- THEN the model MUST prove that load is distributed such that no core remains idle while another has two or more pending tasks

### Requirement: TLA+ Memory Allocator Model
The project SHALL maintain TLA+ models for the memory allocator proving absence of double-free, use-after-free, and memory leaks.

#### Scenario: Prove no double-free
- WHEN the TLA+ model checker verifies the buddy allocator model
- THEN the model MUST prove that no reachable state exists where the same page is freed twice
- AND the invariant MUST hold across all interleaving of concurrent allocation/free sequences

#### Scenario: Prove no use-after-free
- WHEN the TLA+ model represents allocation, use, and deallocation of tensor buffers
- THEN the model MUST prove that no reachable state exists where a freed buffer is accessed
- AND the model MUST include reference-counted shared buffers

#### Scenario: Prove no memory leak
- WHEN the TLA+ model simulates a sequence of allocations and frees
- THEN the model MUST prove that all allocated memory is eventually freed or accounted for
- AND the invariant MUST hold that free memory plus allocated memory equals total memory at all states

### Requirement: Lean 4 Type-Level Invariant Proofs
The project SHALL maintain Lean 4 proofs for type-level invariants including tensor shape correctness and capability non-forgery.

#### Scenario: Prove tensor shape correctness
- WHEN a Lean 4 proof is constructed for the tensor type system
- THEN the proof MUST demonstrate that tensor operations (reshape, matmul, concat) preserve shape consistency
- AND the proof MUST show that matmul of tensors with shapes [M,K] and [K,N] always produces shape [M,N]

#### Scenario: Prove capability non-forgery
- WHEN a Lean 4 proof is constructed for the capability system
- THEN the proof MUST demonstrate that capabilities can only be created by the kernel init or delegated from an existing capability with GRANT permission
- AND the proof MUST show that no sequence of operations can produce a capability with more permissions than its parent

### Requirement: SPIN IPC Protocol Model
The project SHALL maintain SPIN (Promela) models for the IPC messaging protocol proving message delivery guarantees and absence of message loss.

#### Scenario: Prove no message loss in pub/sub
- WHEN the SPIN model checker verifies the pub/sub protocol model
- THEN the model MUST prove that every message published to a key expression with at least one active subscriber is delivered
- AND the model MUST verify this under all interleaving of concurrent publish/subscribe operations

#### Scenario: Prove request/reply completeness
- WHEN the SPIN model represents the request/reply (queryable) pattern
- THEN the model MUST prove that every query sent to a registered queryable receives exactly one reply or a timeout error
- AND the model MUST prove that no reply is delivered to the wrong requester

### Requirement: SPIN TCP State Machine Model
The project SHALL maintain SPIN models for the TCP state machine verifying correct state transitions per RFC 9293.

#### Scenario: Verify TCP connection lifecycle
- WHEN the SPIN model checker verifies the TCP state machine model
- THEN the model MUST prove that all transitions from CLOSED through ESTABLISHED to CLOSED follow the RFC 9293 state diagram
- AND the model MUST verify correct behavior for simultaneous open, simultaneous close, and RST handling

#### Scenario: Verify no orphan connections
- WHEN the SPIN model simulates connection teardown sequences
- THEN the model MUST prove that TIME-WAIT state eventually transitions to CLOSED
- AND no connection remains in a non-terminal state indefinitely

### Requirement: Model Checking CI Integration
All formal verification models SHALL be executed as part of the continuous integration pipeline.

#### Scenario: TLA+ models run in CI
- WHEN a pull request modifies scheduler or memory allocator code
- THEN the CI pipeline MUST execute the corresponding TLA+ model checker
- AND the build MUST fail if any safety property violation is found

#### Scenario: SPIN models run in CI
- WHEN a pull request modifies IPC or TCP code
- THEN the CI pipeline MUST execute the corresponding SPIN model checker
- AND the build MUST fail if any assertion violation or invalid end state is found

#### Scenario: Lean 4 proofs checked in CI
- WHEN a pull request modifies tensor type definitions or capability system code
- THEN the CI pipeline MUST verify all Lean 4 proofs compile and check successfully
- AND the build MUST fail if any proof obligation remains unresolved

