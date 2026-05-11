# Capability: kernel-scheduling — Deterministic mode (delta)

## ADDED Requirements

### Requirement: Deterministic scheduler mode

The kernel SHALL provide a one-shot init-time scheduler mode setting (`SchedulerMode::Default` or `SchedulerMode::Deterministic`) such that, when `Deterministic` is selected, the dispatch order of tasks within a priority class is a pure function of `(priority_class, task_id, spawn_sequence_number)` and is independent of host wall-clock timing, of which core polled the run queue first, and of any work-stealing behavior.

#### Scenario: Deterministic mode is selectable at boot

- **GIVEN** a SmallAIOS kernel boot
- **WHEN** the kernel is started with the `--deterministic` boot argument (kernel mode) or with `SMALLAIOS_DETERMINISTIC=1` (container mode)
- **THEN** the scheduler SHALL initialize in `SchedulerMode::Deterministic` before the first task spawn
- **AND** subsequent attempts to change the mode after task spawning has begun SHALL return a typed error
- **AND** the mode setting SHALL be queryable via `Scheduler::mode()` so other subsystems (the CUDA executor, the `DeterministicRng`) can read the mode without a per-call host-time branch

#### Scenario: Deterministic dispatch order within a priority class

- **GIVEN** a scheduler running in `Deterministic` mode
- **AND** three tasks of equal `SchedulingClass::Inference` priority queued from two different cores in some host-timing-dependent order
- **WHEN** the run queue is dequeued
- **THEN** the dequeue order SHALL be the lexicographic order of `(task_id, spawn_sequence_number)` across the three tasks
- **AND** two identical kernel boots with the same input load SHALL produce the same dispatch sequence
- **AND** the dispatch order SHALL be independent of which core happened to spawn each task

#### Scenario: Work-stealing disabled in deterministic mode

- **GIVEN** a scheduler running in `Deterministic` mode with multiple cores
- **WHEN** an idle core checks the global executor for a steal candidate via `RunQueue::steal_task`
- **THEN** `steal_task` SHALL return `None` regardless of whether other cores have queued Inference tasks
- **AND** Inference tasks SHALL execute only on their AMP-assigned core
- **AND** the resulting per-core data-parallel decomposition SHALL be documented in `docs/scheduling-model.md` so contributors understand the determinism / throughput trade-off

#### Scenario: Default mode behavior preserved bit-for-bit

- **GIVEN** a scheduler initialized in `SchedulerMode::Default` (the default)
- **WHEN** the same workload runs as before this change
- **THEN** the dispatch order, work-stealing behavior, and throughput SHALL be bit-for-bit identical to the pre-`deterministic-scheduling-v1` implementation
- **AND** the regression-pinning unit tests in `kernel/src/sched/executor.rs` SHALL assert this property

### Requirement: Spawn-sequence numbering

The `Task` type SHALL carry a `spawn_sequence_number: u64` field assigned atomically at task spawn from a per-scheduler monotonic counter, and the dispatch tiebreaker in deterministic mode SHALL consume this field.

#### Scenario: Spawn-sequence is monotonic and unique

- **GIVEN** a sequence of `N` task spawns on a scheduler
- **WHEN** each spawn is inspected
- **THEN** each task SHALL have a unique `spawn_sequence_number`
- **AND** the sequence numbers SHALL be strictly monotonically increasing across spawns within the lifetime of the boot

#### Scenario: Spawn-sequence is stable across deterministic-mode runs

- **GIVEN** two identical kernel boots in deterministic mode with the same workload
- **WHEN** the spawn sequence of all tasks is captured in each boot
- **THEN** the captured sequence numbers SHALL be identical across the two boots for corresponding tasks
- **AND** this property SHALL be verifiable by a reproducibility integration test

### Requirement: Cooperative-yield semantics unchanged in deterministic mode

Deterministic mode SHALL NOT change the cooperative-yield contract documented in `docs/scheduling-model.md` (yield at ONNX operator boundaries, no mid-operator preemption, `OperatorBudget` enforcement preserved, priority preemption at yield points only).

#### Scenario: Operator-boundary yield still fires

- **GIVEN** a scheduler running in `Deterministic` mode
- **AND** an inference task running with a registered `yield_fn` callback
- **WHEN** the task completes an ONNX operator
- **THEN** the `yield_fn` SHALL be invoked exactly as in `Default` mode
- **AND** the scheduler SHALL check for pending SYSTEM and IPC tasks before resuming the inference

#### Scenario: `OperatorBudget` enforcement preserved

- **GIVEN** a scheduler running in `Deterministic` mode
- **AND** an operator whose measured wall-clock time exceeds its `hard_limit_multiplier * budget_ns`
- **WHEN** the operator completes
- **THEN** the scheduler SHALL abort the inference session with the same `SessionError::ExecutionFailed` it would have raised in `Default` mode
- **AND** the abort decision SHALL be made *after* the operator's outputs are computed, so the decision branch cannot leak host-timing state into subsequent operator behavior

#### Scenario: Priority preemption at yield points

- **GIVEN** an inference task running in deterministic mode
- **AND** a SYSTEM-class task (e.g. watchdog) becoming runnable
- **WHEN** the inference task reaches the next operator-boundary yield
- **THEN** the scheduler SHALL preempt the inference and run the SYSTEM task
- **AND** the order in which SYSTEM tasks are dispatched (if multiple are ready) SHALL follow the deterministic tiebreaker

### Requirement: Documentation of deterministic mode

`docs/scheduling-model.md` SHALL gain a "Deterministic mode" section that documents the ordering rule, the work-stealing disablement, the throughput cost, the certification claims unlocked, and the cross-link to `docs/determinism.md`.

#### Scenario: Contributor can discover deterministic-mode rules

- **GIVEN** a new contributor reading `docs/scheduling-model.md`
- **WHEN** they search for "deterministic"
- **THEN** they SHALL find an explicit section explaining (a) when to enable deterministic mode, (b) what changes about scheduling behavior, (c) what changes about throughput, and (d) what DO-178C / ISO 26262 claims the mode unlocks
- **AND** they SHALL find a cross-link to `docs/determinism.md` for the full contract
