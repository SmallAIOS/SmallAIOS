## ADDED Requirements

### Requirement: Backend-List Configuration

`SessionConfig` SHALL expose a `backends` field accepting an ordered list of `ExecutionBackend` implementations. The first backend in the list whose `can_run` returns true for an op SHALL be selected for that op at session-build time. The runtime SHALL preserve the existing CPU-only default behavior when `backends` is left at its default (a single `CpuBackend`).

#### Scenario: Default config remains CPU-only

- **WHEN** `SessionConfig::default()` is used to construct a session
- **THEN** the `backends` list SHALL contain exactly one entry, a `CpuBackend`
- **AND** all ops SHALL be bound to `CpuBackend`
- **AND** the resulting session SHALL produce byte-identical outputs to the pre-refactor implementation for any model previously runnable

#### Scenario: Custom backend list is honored

- **WHEN** `SessionConfig::backends` is set to `[FpgaBackend, CpuBackend]`
- **AND** `FpgaBackend::can_run` returns true for MatMul but false for Softmax
- **THEN** MatMul ops in the loaded graph SHALL be bound to `FpgaBackend`
- **AND** Softmax ops SHALL be bound to `CpuBackend`

### Requirement: Session-Build Dispatch Table

The runtime SHALL build a precomputed dispatch table mapping each op in the loaded graph to a selected backend at session creation time. The hot inference path SHALL dispatch ops via this table without revisiting capability checks.

#### Scenario: Table is fully populated before inference

- **WHEN** session creation succeeds
- **THEN** every op in the loaded graph SHALL have an entry in the dispatch table referencing exactly one backend
- **AND** no entry SHALL be `None` or `Pending`

#### Scenario: Inference loop performs no capability checks

- **WHEN** an inference run is profiled (e.g., via `gpu-profile`-style instrumentation extended to the HAL)
- **THEN** zero calls to `ExecutionBackend::can_run` SHALL appear in the inference loop's call graph
- **AND** all op execution SHALL be a direct call to `ExecutionBackend::dispatch` via the dispatch table

### Requirement: Per-Op Fallback to a Lower-Priority Backend

When a backend's `dispatch` returns `Err(ExecError::FallbackToCpu)` (or another defined fallback variant), the runtime SHALL retry the operator on the next-priority backend in `SessionConfig::backends`. If no fallback succeeds, the runtime SHALL return an inference error and SHALL NOT panic.

#### Scenario: Transient FPGA error falls back to CPU

- **WHEN** an op bound to `FpgaBackend` returns `Err(ExecError::FallbackToCpu)` at runtime
- **AND** `CpuBackend` is registered with lower priority and supports the op
- **THEN** the runtime SHALL re-dispatch the op to `CpuBackend`
- **AND** inference SHALL continue with the CPU result
- **AND** the fallback event SHALL be logged

#### Scenario: All backends exhausted

- **WHEN** an op bound to `FpgaBackend` returns `Err(ExecError::FallbackToCpu)` and `CpuBackend` also returns an error
- **THEN** the runtime SHALL return `OnnxError::DispatchExhausted` from the inference call
- **AND** SHALL NOT panic

### Requirement: Tensor Buffer Ownership Contract

Backends SHALL receive read access to input tensors and write access to output tensors via `TensorEnv` handles. The runtime SHALL retain ownership of all tensor buffers it allocates via the memory planner. Backends MAY allocate internal device-side memory (DMA buffers, scratchpad), but SHALL NOT return runtime-owned buffers.

#### Scenario: Backend cannot allocate runtime tensor

- **WHEN** an `ExecutionBackend` implementation source is reviewed
- **THEN** there SHALL be no public API on `TensorEnv` that allows the backend to allocate a runtime-owned tensor buffer
- **AND** internal device memory APIs SHALL NOT return types that the runtime treats as `Tensor`

#### Scenario: Memory planner is unaffected by backend choice

- **WHEN** the memory planner runs on a graph that will execute partly on `FpgaBackend` and partly on `CpuBackend`
- **THEN** buffer reuse decisions SHALL be identical to the CPU-only case for any tensor that crosses the FPGA/CPU boundary as a runtime tensor
- **AND** the planner SHALL NOT need to know which backend will execute each op

### Requirement: Session Build-Time Validation

If the configured backend list does not collectively cover every op in the loaded graph, session creation SHALL fail with `OnnxError::NoBackendForOp` reporting the offending operator.

#### Scenario: Missing CPU fallback flagged at build

- **WHEN** `SessionConfig::backends` is `[FpgaBackend]` (no CpuBackend) and the graph contains an op `FpgaBackend::can_run` returns false for
- **THEN** session creation SHALL return `Err(OnnxError::NoBackendForOp { op_type, op_name, .. })`
- **AND** no inference SHALL execute
