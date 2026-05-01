## ADDED Requirements

### Requirement: ExecutionBackend Trait

The `onnx-rt` crate SHALL define a public `ExecutionBackend` trait that abstracts per-operator dispatch to a concrete execution target (CPU, FPGA accelerator, GPU, etc.). The trait SHALL be `#![no_std]`-compatible and SHALL NOT reference vendor-specific concepts (no `.xmodel`, no DPU, no Vitis AI).

#### Scenario: Trait surface is vendor-neutral

- **WHEN** a reviewer reads the `ExecutionBackend` trait definition and its doc comments
- **THEN** the trait SHALL contain no symbols, type names, or doc text referring to AMD/Xilinx/Vitis/DPU/`.xmodel`/XRT/VART
- **AND** the trait SHALL contain no symbols referring to NVIDIA/CUDA/cuDNN, Intel oneAPI, Apple Metal, or any other vendor stack

#### Scenario: Trait is object-safe

- **WHEN** a user writes `let b: &dyn ExecutionBackend = &my_backend;`
- **THEN** the code SHALL compile without errors

#### Scenario: Trait can be implemented in `no_std`

- **WHEN** an out-of-tree crate implements `ExecutionBackend` in a `#![no_std]` context using only the public API of `onnx-rt`
- **THEN** the implementation SHALL compile without requiring `std`

### Requirement: Per-Op Capability Reporting

The `ExecutionBackend` trait SHALL expose a `can_run(op: &OpDescriptor) -> bool` method that reports whether the backend can execute a given operator. The runtime SHALL call this method exclusively at session-build time, never on the hot path.

#### Scenario: Backend reports support for known op

- **WHEN** a backend supporting INT8 MatMul receives `can_run` for an `OpDescriptor` describing INT8 MatMul with shape it can handle
- **THEN** the call SHALL return `true`

#### Scenario: Backend reports lack of support

- **WHEN** the same backend receives `can_run` for an `OpDescriptor` describing an operator it does not implement (e.g., `LayerNorm`)
- **THEN** the call SHALL return `false`

#### Scenario: `can_run` is not called during inference

- **WHEN** an inference session executes its bound graph
- **THEN** no calls to `ExecutionBackend::can_run` SHALL occur during the inference loop
- **AND** all dispatch decisions SHALL come from a precomputed table built at session creation

### Requirement: Per-Op Dispatch

The `ExecutionBackend` trait SHALL expose a `dispatch(op: &OpDescriptor, env: &mut TensorEnv) -> Result<(), ExecError>` method that executes a single operator. The runtime SHALL pass input and output tensor handles through `TensorEnv`; the backend SHALL NOT allocate runtime-owned tensor buffers.

#### Scenario: Successful dispatch produces output

- **WHEN** the runtime invokes `dispatch` for a supported op with valid input tensors in `env`
- **THEN** the backend SHALL write the result to the output tensor's buffer
- **AND** SHALL return `Ok(())`

#### Scenario: Dispatch reports recoverable error

- **WHEN** a backend encounters an error condition that should fall back to another backend (e.g., a transient resource shortage)
- **THEN** it SHALL return `Err(ExecError::FallbackToCpu)` (or another defined fallback variant)
- **AND** SHALL leave the output tensor unmodified

#### Scenario: Backend does not allocate runtime tensors

- **WHEN** any backend implementation is reviewed
- **THEN** it SHALL NOT call `TensorEnv::allocate_runtime_buffer` or any other API that creates a tensor buffer the runtime owns
- **AND** internal device-side memory (DMA buffers, scratchpad) MAY be allocated, but SHALL remain opaque to the runtime

### Requirement: CPU Backend as a First-Class Implementation

The existing host-CPU dispatch path (x86 SIMD, ARM NEON/SVE) SHALL be exposed as a concrete `CpuBackend` implementing `ExecutionBackend`. The runtime SHALL NOT contain a CPU-specific bypass path that does not go through the trait.

#### Scenario: All ops route through ExecutionBackend

- **WHEN** an inference session runs on a CPU-only configuration
- **THEN** every op execution SHALL be a call to `ExecutionBackend::dispatch` on a registered backend
- **AND** there SHALL be no direct CPU dispatch path that bypasses the trait

#### Scenario: CpuBackend supports all currently-supported ops

- **WHEN** the registered backend list contains only `CpuBackend`
- **THEN** every model previously runnable by `onnx-rt` (per existing `onnx-runtime` and `onnx-cpu-execution` specs) SHALL still run
- **AND** results SHALL be byte-identical to the pre-refactor implementation

### Requirement: Static Backend Binding at Session Creation

The runtime SHALL accept an ordered list of backends in `SessionConfig` and SHALL bind each operator in the loaded graph to the highest-priority backend whose `can_run` returns true. Binding SHALL occur at session creation; per-op dispatch decisions SHALL NOT be revisited during inference.

#### Scenario: First-match priority binding

- **WHEN** `SessionConfig::backends` is `[FpgaBackend, CpuBackend]` and an op is supported by both
- **THEN** the op SHALL be bound to `FpgaBackend`

#### Scenario: Fallback to CPU for unsupported op

- **WHEN** `SessionConfig::backends` is `[FpgaBackend, CpuBackend]` and an op is supported only by `CpuBackend`
- **THEN** the op SHALL be bound to `CpuBackend`

#### Scenario: No capable backend produces session error

- **WHEN** `SessionConfig::backends` is `[FpgaBackend]` (no CpuBackend) and the graph contains an op `FpgaBackend::can_run` reports false for
- **THEN** session creation SHALL fail with `OnnxError::NoBackendForOp` reporting the operator name and type
- **AND** no inference SHALL run

### Requirement: QEMU Stub Reference Backend

A reference `QemuStubBackend` SHALL be provided behind a `qemu-stub` Cargo feature in `onnx-rt`. The backend SHALL drive a deterministic AXI-mapped MMIO device exposed by QEMU, exercising the AXI/DMA driver framework end-to-end without requiring real FPGA hardware.

#### Scenario: Stub backend runs MatMul under QEMU

- **WHEN** SmallAIOS is started in QEMU with the stub accelerator device attached and a session is created with `[QemuStubBackend, CpuBackend]`
- **AND** the loaded model contains a MatMul op the stub claims via `can_run`
- **THEN** the MatMul SHALL be dispatched to `QemuStubBackend`
- **AND** the output SHALL be numerically equivalent to the `CpuBackend` reference (within f32 epsilon)

#### Scenario: Stub backend is disabled by default

- **WHEN** `onnx-rt` is built without the `qemu-stub` feature
- **THEN** `QemuStubBackend` SHALL NOT be present in the compiled binary
- **AND** there SHALL be no compile-time, link-time, or runtime dependency on QEMU device interfaces

#### Scenario: Stub backend gracefully degrades when device absent

- **WHEN** the `qemu-stub` feature is enabled but SmallAIOS is run on a host without the stub device (e.g., bare metal, or QEMU without the device)
- **THEN** `QemuStubBackend::probe` SHALL return `Err(BackendUnavailable)` at session creation
- **AND** session creation SHALL fall back to remaining backends without panic
