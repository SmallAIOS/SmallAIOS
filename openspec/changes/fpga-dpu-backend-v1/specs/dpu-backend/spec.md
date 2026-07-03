## ADDED Requirements

### Requirement: DpuBackend Implements ExecutionBackend

The `onnx-rt` crate SHALL provide a `DpuBackend` type implementing the `ExecutionBackend` trait, gated behind a non-default `dpu` Cargo feature. `DpuBackend` SHALL drive an AMD Zynq UltraScale+ DPU (variant `DPUCZDX8G`, B4096 configuration) via the AXI/AXI-DMA framework provided by `arch/aarch64-zynqmp`.

#### Scenario: DpuBackend implements the trait

- **WHEN** a reviewer reads the public API of `onnx-rt::backend::dpu`
- **THEN** `DpuBackend` SHALL implement `ExecutionBackend`
- **AND** the trait surface SHALL NOT be modified by this change

#### Scenario: dpu feature disables the backend

- **WHEN** `onnx-rt` is built without the `dpu` Cargo feature
- **THEN** the `DpuBackend` type SHALL NOT be present in the compiled binary
- **AND** there SHALL be no compile-time, link-time, or runtime dependency on Vitis AI, XRT, or VART
- **AND** the default workspace build SHALL succeed without the feature

#### Scenario: DpuBackend is constructed from a board-supplied handle

- **WHEN** `arch/aarch64-zynqmp::dpu` exposes a `DpuHandle` bound to the K26 stock bitstream's DPU AXI region
- **AND** `onnx-rt::backend::dpu::DpuBackend::new(handle)` is called with that handle
- **THEN** construction SHALL succeed
- **AND** the resulting backend SHALL hold the handle and not attempt to remap AXI itself
- **AND** no AXI addresses SHALL be hard-coded inside `onnx-rt`

### Requirement: DpuBackend Probe Detects DPU Presence

`DpuBackend` SHALL expose a `probe()` method that reads the DPU identification register and validates the reported variant against a list of supported configurations. When the DPU is absent or unsupported, `probe` SHALL return `Err(BackendUnavailable)`.

#### Scenario: Supported DPU variant probed successfully

- **WHEN** `probe()` is called on a board where the DPU identification register reports `DPUCZDX8G` B4096
- **THEN** the call SHALL return `Ok(())`
- **AND** subsequent `can_run` calls SHALL be valid

#### Scenario: Missing DPU returns BackendUnavailable

- **WHEN** `probe()` is called on a board where the DPU identification register is unreadable, returns zero, or reports an unknown variant
- **THEN** the call SHALL return `Err(BackendUnavailable)`
- **AND** session creation SHALL fall back to remaining backends without panic

#### Scenario: Unsupported DPU variant rejected with diagnostic

- **WHEN** `probe()` reads a known-but-unsupported DPU variant (e.g., DPUCZDX8G B1152)
- **THEN** the call SHALL return `Err(BackendUnavailable)` carrying a diagnostic with the observed variant name
- **AND** the diagnostic SHALL reference `docs/zynqmp-dpu.md` for the supported-variant list

### Requirement: `.xmodel` Parser Loads Vitis-AI-Compiled Subgraphs

`onnx-rt::backend::dpu` SHALL provide a `#![no_std]` parser that consumes a Vitis-AI-emitted `.xmodel` byte slice and produces a structured representation containing: DPU subgraph boundaries, per-subgraph DPU instruction blob, input/output tensor descriptors, weight/bias blob references, and the residual host-CPU op list. The parser SHALL reuse the existing `onnx-rt` hand-rolled protobuf decoder; it SHALL NOT introduce a new third-party protobuf crate.

#### Scenario: Tiny MatMul `.xmodel` round-trips through the parser

- **WHEN** the parser is fed a known-good `.xmodel` produced by Vitis AI for a single-MatMul model from the `tests/fixtures/dpu/` corpus
- **THEN** the parser SHALL identify exactly one DPU subgraph
- **AND** the subgraph SHALL carry a non-empty instruction blob
- **AND** the input and output tensor descriptors SHALL match the source ONNX model's shapes
- **AND** the residual host-op list SHALL be empty

#### Scenario: Mixed MatMul + LayerNorm `.xmodel` separates DPU and host ops

- **WHEN** the parser is fed a `.xmodel` whose source ONNX contains a MatMul followed by a LayerNorm and a Softmax (LayerNorm and Softmax not supported on this DPU configuration)
- **THEN** the parser SHALL emit a DPU subgraph for MatMul
- **AND** SHALL emit a residual host-op list containing LayerNorm and Softmax
- **AND** the residual ops SHALL carry the correct shapes and attribute values

#### Scenario: Unknown protobuf field tags do not abort parsing

- **WHEN** the parser encounters a `.xmodel` with field tags it does not recognize (e.g., from a newer Vitis AI minor version)
- **THEN** the parser SHALL skip the unknown fields per protobuf wire-format rules
- **AND** parsing SHALL succeed if all required fields are present
- **AND** a warning SHALL be logged listing the unknown tags for traceability

#### Scenario: Malformed `.xmodel` rejected with clear error

- **WHEN** the parser is fed bytes that are not a valid XIR `.xmodel` (truncated, wrong magic, missing required subgraph attributes)
- **THEN** the parser SHALL return `Err(XmodelError::*)` with a variant identifying the failure mode
- **AND** the error SHALL reference `docs/zynqmp-dpu.md` for diagnosis

### Requirement: DpuBackend Capability Reporting via Synthetic DpuSubgraph Op

`DpuBackend::can_run` SHALL return true only for the synthetic `OpDescriptor::DpuSubgraph` op produced by the `.xmodel` loader. It SHALL return false for raw ONNX ops (MatMul, Conv, etc.). The runtime's session-build path SHALL be responsible for synthesizing `DpuSubgraph` ops from `.xmodel` content.

#### Scenario: DpuBackend claims DpuSubgraph ops

- **WHEN** the runtime calls `can_run` on a `DpuBackend` for an `OpDescriptor::DpuSubgraph` whose declared variant matches the probed DPU
- **THEN** the call SHALL return true

#### Scenario: DpuBackend rejects raw ONNX ops

- **WHEN** the runtime calls `can_run` on a `DpuBackend` for an `OpDescriptor` describing a raw ONNX MatMul (not a synthetic subgraph)
- **THEN** the call SHALL return false
- **AND** the op SHALL be eligible to bind to a lower-priority backend (typically `CpuBackend`)

#### Scenario: DpuBackend rejects DpuSubgraph for a mismatched variant

- **WHEN** the runtime calls `can_run` on a `DpuBackend` (probed as `DPUCZDX8G` B4096) for an `OpDescriptor::DpuSubgraph` whose blob targets a different DPU configuration
- **THEN** the call SHALL return false
- **AND** session creation SHALL surface a clear `OnnxError::NoBackendForOp` if no other backend claims the op

### Requirement: DpuBackend Dispatch Uses IRQ-Driven Completion

`DpuBackend::dispatch` SHALL submit the DPU instruction stream via the AXI/AXI-DMA framework, register a completion waker against the GIC SPI line wired to the DPU IRQ, and yield until the IRQ fires. Polling SHALL NOT be used in default builds. A diagnostic polling path MAY exist behind a non-default `dpu-polling-debug` feature.

#### Scenario: Dispatch yields and resumes on IRQ

- **WHEN** `DpuBackend::dispatch` is called with valid input tensors
- **THEN** the dispatching task SHALL yield after submitting the instruction stream
- **AND** SHALL resume only after the DPU completion IRQ is observed by the GIC-400 driver
- **AND** SHALL NOT busy-wait or poll the DPU status register in a default build

#### Scenario: dpu-polling-debug feature off in default builds

- **WHEN** any default workspace build is produced
- **THEN** the `dpu-polling-debug` feature SHALL be off
- **AND** no polling fallback SHALL appear in the dispatch hot path

#### Scenario: Dispatch surfaces fault as FallbackToCpu

- **WHEN** the DPU completion handler observes a fault status (instruction error, timeout, or unrecognized result code)
- **THEN** `DpuBackend::dispatch` SHALL return `Err(ExecError::FallbackToCpu)`
- **AND** the runtime's existing fallback semantics SHALL retry the offending op on the next-priority backend
- **AND** the fault event SHALL be logged with the offending subgraph identifier

### Requirement: DpuBackend Honors Cache-Coherency Port Choice

`DpuBackend` SHALL place activation tensors on a coherent AXI port (HPC0 via `DmaBuffer<HpcPort>`), and SHALL place weight blobs and instruction blobs on a non-coherent AXI port (HP0 via `DmaBuffer<HpPort>`). Explicit `clean_for_device()` calls SHALL be performed on weight and instruction buffers before the DPU is allowed to read them; activations SHALL NOT require manual maintenance.

#### Scenario: Activation buffers are HpcPort

- **WHEN** the source of `DpuBackend::dispatch` is reviewed
- **THEN** activation tensor DMA buffers SHALL be typed `DmaBuffer<HpcPort>`
- **AND** there SHALL be no `clean_for_device` or `invalidate_for_cpu` call on activation buffers
- **AND** any such call SHALL fail to compile

#### Scenario: Weight and instruction buffers are HpPort with explicit maintenance

- **WHEN** weights or the instruction stream are loaded into DMA buffers
- **THEN** the buffers SHALL be typed `DmaBuffer<HpPort>`
- **AND** `clean_for_device()` SHALL be called once after CPU writes complete and before the DPU is signaled
- **AND** subsequent reads by the CPU (if any) SHALL precede an `invalidate_for_cpu()` call

#### Scenario: Debug cache-tracker confirms maintenance pattern

- **WHEN** the unit-test harness wraps the three DMA buffer roles with the AXI framework's debug cache-tracker (per `axi-dma-framework` spec)
- **AND** a representative `.xmodel` is dispatched in test
- **THEN** the tracker SHALL record exactly one `clean_for_device` per weight buffer and per instruction buffer per session
- **AND** SHALL record zero maintenance calls on activation buffers
- **AND** SHALL fail the test if any expectation is violated

### Requirement: Per-Op Fallback to CpuBackend for Residual Ops

When the `.xmodel` loader produces a residual host-op list, `DpuBackend` SHALL NOT execute those ops. The session-build path SHALL bind them to the next-priority backend (typically `CpuBackend`) per the existing HAL fallback semantics.

#### Scenario: Residual LayerNorm runs on CpuBackend

- **WHEN** a `.xmodel` carries a DPU subgraph plus a residual LayerNorm op
- **AND** `SessionConfig::backends` is `[DpuBackend, CpuBackend]`
- **THEN** the DpuSubgraph synthetic op SHALL bind to `DpuBackend`
- **AND** the LayerNorm SHALL bind to `CpuBackend`
- **AND** end-to-end inference SHALL produce a result numerically equivalent to a pure-CPU reference (within the model's quantization tolerance)

#### Scenario: Missing CpuBackend with residual ops fails session build

- **WHEN** a `.xmodel` carries a residual op
- **AND** `SessionConfig::backends` is `[DpuBackend]` only
- **THEN** session creation SHALL fail with `OnnxError::NoBackendForOp` reporting the residual op
- **AND** no inference SHALL execute

### Requirement: dpu-profile Instrumentation

`onnx-rt::backend::dpu` SHALL provide a `dpu-profile` Cargo feature that, when enabled, records per-dispatch DPU latency, DMA bytes in, DMA bytes out, completion-IRQ wait time, and residual-CPU op count and total host time. The aggregated summary SHALL be written to stderr at `DpuBackend::drop`. The feature SHALL be off by default. Production builds with the feature off SHALL pay zero overhead.

#### Scenario: dpu-profile off has zero overhead

- **WHEN** `onnx-rt` is built with the `dpu` feature on but the `dpu-profile` feature off
- **THEN** no profiling counters or timestamp captures SHALL exist in the dispatch hot path
- **AND** `DpuBackend::drop` SHALL emit no profile output

#### Scenario: dpu-profile on emits a session summary

- **WHEN** `onnx-rt` is built with `dpu-profile` on and a session runs at least one inference
- **THEN** at `DpuBackend::drop` a summary SHALL be written to stderr
- **AND** the summary SHALL include per-op-type DPU latency aggregates, DMA throughput, and host residual op time
- **AND** the summary SHALL be parseable by the analysis script in `tools/dpu-profile/` (added by this change)

### Requirement: Offline `.xmodel` Production Documented

The change SHALL deliver `docs/zynqmp-dpu.md` describing the offline pipeline that produces a `.xmodel`: ONNX → quantized ONNX (Brevitas) → Vitis AI compile → `.xmodel`. The document SHALL pin a specific Vitis AI version and a specific Brevitas version. The document SHALL state plainly that QEMU runs of the `dpu` recipe validate only software-side packaging and `.xmodel` parsing — DPU instruction execution requires real silicon.

#### Scenario: Documentation pins toolchain versions

- **WHEN** a reviewer reads `docs/zynqmp-dpu.md`
- **THEN** the document SHALL pin a Vitis AI version
- **AND** SHALL pin a Brevitas version
- **AND** SHALL state the BOOT.BIN packaging procedure for boards carrying the K26 stock DPU bitstream

#### Scenario: QEMU caveat is explicit

- **WHEN** the `just run-arm-zynqmp-dpu` recipe is executed
- **THEN** the recipe SHALL print a banner stating "DPU instructions do not execute under QEMU; this validates packaging only"
- **AND** the same caveat SHALL appear in `docs/zynqmp-dpu.md`
