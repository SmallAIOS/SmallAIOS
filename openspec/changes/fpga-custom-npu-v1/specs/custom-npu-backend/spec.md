## ADDED Requirements

### Requirement: CustomNpuBackend Implements ExecutionBackend

The `onnx-rt` crate SHALL provide a `CustomNpuBackend` type in `onnx-rt::backend::custom_npu` implementing the `ExecutionBackend` trait, gated behind a non-default `custom-npu` Cargo feature. `CustomNpuBackend` SHALL drive the SmallAIOS-native custom NPU on the Kria K26 SOM (KV260 / KR260, Zynq UltraScale+ PL) via the AXI/AXI-DMA framework provided by `arch/aarch64-zynqmp`. The backend SHALL have no compile-time, link-time, or runtime dependency on Vitis AI, the `.xmodel` format, or any external compiler — the SmallAIOS runtime is the source of truth for what the accelerator executes.

#### Scenario: CustomNpuBackend implements the trait

- **WHEN** a reviewer reads the public API of `onnx-rt::backend::custom_npu`
- **THEN** `CustomNpuBackend` SHALL implement `ExecutionBackend`
- **AND** the trait surface SHALL NOT be modified by this change

#### Scenario: custom-npu feature disables the backend

- **WHEN** `onnx-rt` is built without the `custom-npu` Cargo feature
- **THEN** the `CustomNpuBackend` type SHALL NOT be present in the compiled binary
- **AND** the default workspace build SHALL succeed without the feature

#### Scenario: No Vitis AI or .xmodel dependency

- **WHEN** `onnx-rt` is built with the `custom-npu` feature enabled
- **THEN** the build SHALL NOT depend on Vitis AI, XRT, VART, or any `.xmodel` parser
- **AND** no new third-party runtime crate dependencies SHALL be introduced

#### Scenario: CustomNpuBackend is constructed from a board-supplied handle

- **WHEN** `arch/aarch64-zynqmp::custom_npu` exposes a `CustomNpuHandle` bound to the custom NPU's AXI region
- **AND** `onnx-rt::backend::custom_npu::CustomNpuBackend::new(handle)` is called with that handle
- **THEN** construction SHALL succeed
- **AND** the resulting backend SHALL hold the handle and not attempt to remap AXI itself
- **AND** no AXI addresses SHALL be hard-coded inside `onnx-rt`

### Requirement: Per-Op Dispatch Claims in can_run

Unlike `DpuBackend`, which dispatches at synthetic-subgraph granularity, `CustomNpuBackend::can_run` SHALL claim individual ONNX ops from the finalized v1 op set. Each accelerated op SHALL be its own dispatch call, decomposed into AXI-DMA plus control-register sequences inside the backend. Ops the backend does not claim SHALL remain eligible to bind to lower-priority backends (typically `CpuBackend`) per the existing HAL fallback semantics. No synthetic `CustomNpuSubgraph` op and no custom IR layer SHALL be introduced in v1.

#### Scenario: Backend claims an op from the finalized op set

- **WHEN** the runtime calls `can_run` on a probed `CustomNpuBackend` for an `OpDescriptor` describing a MatMul in a supported dtype and shape
- **THEN** the call SHALL return true
- **AND** the op SHALL dispatch as an individual AXI-DMA + control-register sequence

#### Scenario: Unclaimed op falls back to CpuBackend

- **WHEN** the runtime calls `can_run` for an op outside the finalized op set (or in an unsupported dtype/shape)
- **AND** `SessionConfig::backends` is `[CustomNpuBackend, CpuBackend]`
- **THEN** `can_run` SHALL return false
- **AND** the op SHALL bind to `CpuBackend`
- **AND** end-to-end inference on a mixed model SHALL interleave NPU-claimed and CPU-residual ops without an extra parser or graph rewrite

### Requirement: Phase 0 Go/No-Go Gate Locks the Op Set

The set of ops `CustomNpuBackend` claims SHALL be finalized only after the DPU perf report (`docs/perf/dpu-baseline.md` from `fpga-dpu-backend-v1`) is reviewed in Phase 0. Candidate ops are matmul + GEMM-fused-bias, layernorm, RMSNorm, softmax, and gather for KV-cache lookup; the final list MAY add or drop candidates based on measurement. The floor SHALL be matmul + GEMM-fused-bias. Phase 0 SHALL land `docs/zynqmp-custom-npu-design-doc.md` capturing the op set, dtype mix, tile sizes, and scratchpad sizing with traceability to specific perf-report numbers. Downstream design and implementation phases SHALL remain DEFERRED until the gate closes "go".

#### Scenario: Design doc traces op-set decisions to perf data

- **WHEN** a reviewer reads `docs/zynqmp-custom-npu-design-doc.md`
- **THEN** every op in the finalized v1 op set SHALL cite the specific `docs/perf/dpu-baseline.md` measurement justifying its inclusion
- **AND** the op set SHALL include at least matmul + GEMM-fused-bias

#### Scenario: No-go verdict halts the change without RTL work

- **WHEN** the Phase 0 review concludes the DPU is good enough on the target models
- **THEN** no HLS/RTL design work SHALL start
- **AND** the change SHALL be eligible for cancellation with the proposal retained as a roadmap reference

### Requirement: Numeric Format Support — INT8 Floor, BF16 Stretch

The custom NPU SHALL support INT8 multiply-accumulate at minimum. BF16 SHALL be a stretch goal, included only if the PL resource budget permits without breaking tile-size targets. FP16 SHALL be considered only if the perf report shows BF16 alone is insufficient for target-model accuracy. The INT8 format SHALL match the format produced by the existing CPU INT8 quantized-inference path; no quantization-aware-training pipeline SHALL be introduced.

#### Scenario: INT8 ops execute on the NPU

- **WHEN** a session dispatches an INT8 matmul claimed by `CustomNpuBackend`
- **THEN** the NPU SHALL execute it using INT8 multiply-accumulate
- **AND** the result SHALL be bit-accurate against the existing CPU INT8 reference kernel

#### Scenario: BF16 dropped when the resource budget does not permit it

- **WHEN** post-synthesis results show BF16 support would exceed the PL resource budget or break tile-size targets
- **THEN** the v1 design SHALL ship INT8-only
- **AND** `can_run` SHALL return false for BF16 ops so they bind to a lower-priority backend

### Requirement: Probe Detects NPU Presence in the Loaded PL Configuration

`CustomNpuBackend` SHALL probe the running PL configuration for its expected NPU identification registers. When the expected NPU IDs are not present — for example, because the DPU bitstream (or no accelerator bitstream) was packed into BOOT.BIN instead — the probe SHALL return `Err(BackendUnavailable)` cleanly, matching the `DpuBackend` behavior, and session creation SHALL fall back to remaining backends without panic.

#### Scenario: NPU bitstream present probes successfully

- **WHEN** `probe()` is called on a board whose BOOT.BIN carried the custom NPU bitstream
- **THEN** the call SHALL return `Ok(())`
- **AND** subsequent `can_run` calls SHALL be valid

#### Scenario: DPU bitstream loaded instead of NPU

- **WHEN** `probe()` is called on a board whose loaded PL configuration does not expose the expected NPU identification registers (e.g., the DPU bitstream is loaded)
- **THEN** the call SHALL return `Err(BackendUnavailable)`
- **AND** session creation SHALL fall back to remaining backends without panic

### Requirement: Dispatch Reuses the AXI/DMA Framework With IRQ-Driven Completion

`CustomNpuBackend::dispatch` SHALL reuse the AXI/AXI-DMA framework, the typed cache-coherency discipline, and the IRQ-driven completion pattern provided by `fpga-accelerator-hal-v1` via `arch/aarch64-zynqmp`; it SHALL NOT introduce new DMA, coherency, or interrupt infrastructure. Weights SHALL be fetched via AXI from DDR with on-die BRAM as cache, pre-fetched via a separate AXI-DMA channel from activations.

#### Scenario: Dispatch yields and resumes on IRQ

- **WHEN** `CustomNpuBackend::dispatch` is called with valid input tensors
- **THEN** the dispatching task SHALL yield after submitting the AXI-DMA + control-register sequence
- **AND** SHALL resume only after the NPU completion IRQ is observed
- **AND** SHALL NOT busy-wait or poll NPU status registers in a default build

#### Scenario: DMA buffers use the framework's typed coherency API

- **WHEN** the source of `CustomNpuBackend::dispatch` is reviewed
- **THEN** all device-visible buffers SHALL be typed `DmaBuffer` instances from the `arch/aarch64-zynqmp` AXI framework
- **AND** any cache maintenance SHALL go through the framework's typed API only
- **AND** weight pre-fetch SHALL use an AXI-DMA channel separate from the activation channel

### Requirement: Board Driver Exposes CustomNpuHandle

`arch/aarch64-zynqmp` SHALL provide a `custom_npu` module exposing a `CustomNpuHandle` that wraps the AXI-mapped NPU peripheral: control registers, IRQ subscription, and DMA channel bindings. The handle SHALL mirror the `DpuHandle` and `StubHandle` construction pattern, preserving the 4-layer model rule that the runtime cannot know AXI addresses.

#### Scenario: Handle wraps the AXI peripheral

- **WHEN** a reviewer reads the public API of `arch/aarch64-zynqmp::custom_npu`
- **THEN** `CustomNpuHandle` SHALL expose the NPU control registers, IRQ subscription, and DMA channels
- **AND** the construction pattern SHALL mirror `DpuBackend::new(DpuHandle)` and `QemuStubBackend::new(StubHandle)`

#### Scenario: Layering keeps AXI knowledge out of the runtime

- **WHEN** the dependency graph is checked with `just arch-check`
- **THEN** `onnx-rt` SHALL NOT depend on `arch/aarch64-zynqmp` internals beyond the handle it consumes
- **AND** all AXI base addresses and IRQ line numbers SHALL live in `arch/aarch64-zynqmp`

### Requirement: Versioned Hardware Artifact Tree hw/custom-npu/

The change SHALL add a hardware-design artifact tree at `hw/custom-npu/`, outside the Cargo workspace, containing the HLS sources, generated RTL, simulation testbenches, and bitstream provenance metadata. The tree SHALL carry its own `VERSION` file, bumped on any RTL change. Provenance manifests SHALL record the HLS source git revision, the RTL synthesis log hash, and the bitstream MD5; the bitstream binary itself SHALL NOT be committed (produced offline per the pinned Vivado). The runtime feature-flag value SHALL be traceable to a bitstream provenance MD5.

#### Scenario: Artifact tree contents

- **WHEN** a reviewer inspects `hw/custom-npu/`
- **THEN** it SHALL contain HLS sources, generated RTL, simulation testbenches under `hw/custom-npu/sim/`, and provenance manifests
- **AND** it SHALL contain a `VERSION` file
- **AND** it SHALL NOT be a member of the Cargo workspace

#### Scenario: Bitstream provenance without committed binaries

- **WHEN** a bitstream is produced by the pinned offline Vivado build
- **THEN** the committed provenance manifest SHALL record the HLS source git revision, the synthesis log hash, and the bitstream MD5
- **AND** the bitstream binary SHALL NOT be committed to the repository

#### Scenario: VERSION bumps on RTL change

- **WHEN** any RTL or HLS source under `hw/custom-npu/` changes
- **THEN** the `hw/custom-npu/VERSION` file SHALL be bumped in the same PR

### Requirement: HLS-First Design With Hand-RTL Fallback

The first cut of the NPU design SHALL be written in Vitis HLS (C++): the matmul tile, layernorm/softmax pipelines, on-die activation and weight buffers, the layernorm/softmax statistics scratchpad, and the AXI-stream interfaces. Where post-synthesis Quality-of-Results (Fmax, LUT count, DSP utilization, latency) is unacceptable for a critical kernel, that kernel MAY be rewritten in hand-Verilog/SystemVerilog, with the substitution and its QoR justification recorded in the artifact tree. No MyHDL/SpinalHDL/Chisel toolchain SHALL be added.

#### Scenario: v1 kernels are HLS sources

- **WHEN** a reviewer inspects `hw/custom-npu/` at the end of Phase 1
- **THEN** the matmul tile, normalization/softmax pipelines, and AXI-stream interfaces SHALL exist as C++ Vitis HLS sources
- **AND** the design SHALL include on-die activation and weight buffers plus a scratchpad for layernorm/softmax statistics

#### Scenario: Hand-RTL substitution is justified and recorded

- **WHEN** a kernel (e.g., the matmul tile) is rewritten in hand-Verilog/SystemVerilog because HLS QoR was unacceptable
- **THEN** the artifact tree SHALL record the QoR numbers (Fmax, LUT, DSP, latency) that motivated the substitution
- **AND** the AXI plumbing MAY remain in HLS

### Requirement: 70% PL Resource Budget

The custom NPU SHALL target a ceiling of 70% per resource type of the K26 PL (~256K LUT / ~144 BRAM / ~1248 DSP). The remaining 30% SHALL be reserved for AXI-DMA controllers and AXI plumbing, ILA/VIO debug instrumentation during bring-up, and headroom for future `fpga-manager-v1` partial-reconfiguration overlays. If the design exceeds the budget, tile sizes SHALL be shrunk before ops are dropped, and any budget increase SHALL be an explicit recorded decision — never a silent bust.

#### Scenario: Post-synthesis utilization within budget

- **WHEN** the Vivado post-synthesis utilization report for the v1 bitstream is reviewed
- **THEN** LUT, BRAM, and DSP utilization SHALL each be at or below 70% of the K26 PL totals

#### Scenario: Budget overrun triggers tile shrink, then escalation

- **WHEN** a synthesis run reports any resource type above the 70% ceiling
- **THEN** the design SHALL first shrink tile sizes to fit
- **AND** if the smallest viable tile still exceeds the budget, the overrun SHALL be escalated as an explicit recorded decision (e.g., toward a Versal AI Edge follow-up) rather than silently exceeding 70%

### Requirement: Runtime-Driven Co-Simulation Verification

The `hw/custom-npu/sim/` testbench SHALL be drivable from the same `OpDescriptor` and tensor inputs the runtime would dispatch on real hardware. The harness SHALL run the NPU implementation in a Verilator/QuestaSim co-simulation and the SmallAIOS CPU reference in parallel, comparing outputs bit-accurately for INT8 ops and within a documented tolerance for BF16/FP16 ops. Host-side runtime unit tests SHALL exercise the co-sim path so runtime and RTL are validated together on a developer machine before any board run. Once silicon is available, any sim/silicon mismatch SHALL be recorded as a bug against the harness, not accepted as expected drift.

#### Scenario: INT8 op matches bit-accurately in co-sim

- **WHEN** the co-sim harness dispatches an INT8 op from the v1 op set with the same `OpDescriptor` and inputs to both the RTL co-simulation and the SmallAIOS CPU reference
- **THEN** the two outputs SHALL match bit-for-bit
- **AND** any mismatch SHALL fail the test

#### Scenario: BF16 op matches within documented tolerance

- **WHEN** the co-sim harness dispatches a BF16 op (if BF16 ships in v1)
- **THEN** the RTL output SHALL match the CPU reference within a tolerance documented in the artifact tree
- **AND** results outside tolerance SHALL fail the test

#### Scenario: Host unit tests exercise co-sim without a board

- **WHEN** the host-side `onnx-rt` test suite for the `custom-npu` feature runs on a developer machine with no FPGA attached
- **THEN** the runtime-driven co-sim tests SHALL execute against the RTL/HLS testbench
- **AND** SHALL validate runtime dispatch and RTL behavior together

### Requirement: Vivado-Produced Bitstream With Static BOOT.BIN Selection

The NPU bitstream SHALL be produced offline by a pinned Vivado/Vitis HLS version on Linux x86; the SmallAIOS runtime SHALL remain standalone with no Vivado dependency. For this change, bitstream selection SHALL be static: exactly one accelerator bitstream (DPU or custom NPU) is loaded per boot, chosen by which artifact is packed into BOOT.BIN. Runtime bitstream swap SHALL be out of scope, deferred to `fpga-manager-v1`. Open-source bitstream generation SHALL NOT be attempted (UltraScale+ has no open toolchain).

#### Scenario: Pinned offline toolchain

- **WHEN** a reviewer reads `docs/zynqmp-custom-npu.md`
- **THEN** the document SHALL pin a specific Vivado / Vitis HLS version
- **AND** the SmallAIOS runtime build SHALL succeed on a machine without Vivado installed

#### Scenario: One bitstream per boot

- **WHEN** a BOOT.BIN is packaged with the custom NPU bitstream
- **THEN** the boot SHALL load only the NPU bitstream (not the DPU)
- **AND** switching accelerators SHALL require repackaging BOOT.BIN, not a runtime mechanism

### Requirement: custom-npu-profile Instrumentation

`onnx-rt::backend::custom_npu` SHALL provide a `custom-npu-profile` Cargo feature mirroring the `dpu-profile` shape from `fpga-dpu-backend-v1`. When enabled, it SHALL record per-dispatch NPU latency, DMA bytes in/out, completion-IRQ wait time, and a counter for inter-op DMA overhead (so the future subgraph-fusion question is data-driven). The feature SHALL be off by default, and builds with it off SHALL pay zero overhead.

#### Scenario: custom-npu-profile off has zero overhead

- **WHEN** `onnx-rt` is built with `custom-npu` on but `custom-npu-profile` off
- **THEN** no profiling counters or timestamp captures SHALL exist in the dispatch hot path

#### Scenario: custom-npu-profile on records inter-op DMA overhead

- **WHEN** `onnx-rt` is built with `custom-npu-profile` on and a session runs at least one inference
- **THEN** the emitted summary SHALL include per-dispatch NPU latency, DMA bytes in/out, and IRQ wait time
- **AND** SHALL include the aggregated inter-op DMA overhead counter

### Requirement: Calibrated estimated_ns

`CustomNpuBackend::estimated_ns` SHALL return per-op values from a compile-time constant table calibrated from per-op latency measurements captured on real silicon during Phase 4 bring-up, not a static placeholder.

#### Scenario: estimated_ns reads the calibrated table

- **WHEN** `estimated_ns` is called for an op in the v1 op set after Phase 4 calibration lands
- **THEN** the returned value SHALL come from the compile-time constant table derived from on-silicon per-op latency measurements
- **AND** the table's provenance SHALL be traceable to the bring-up measurement record

### Requirement: Documentation and Perf Comparison vs DPU Baseline

The change SHALL deliver `docs/zynqmp-custom-npu.md` documenting the NPU micro-architecture, the finalized op coverage, and the pinned toolchain, plus `docs/perf/custom-npu-vs-dpu.md` reporting a head-to-head comparison against the DPU baseline on representative target models. The perf target SHALL be a ≥2× geomean speedup over the DPU on the Phase 0 op set and ≥1× on the rest of the graph. `CLAUDE.md` SHALL be updated with the new `custom-npu` and `custom-npu-profile` feature flags.

#### Scenario: Micro-architecture and op coverage documented

- **WHEN** a reviewer reads `docs/zynqmp-custom-npu.md`
- **THEN** it SHALL describe the NPU micro-architecture (matmul tile, buffers, scratchpad)
- **AND** SHALL list the finalized op coverage and supported numeric formats
- **AND** SHALL document the BOOT.BIN packaging variant for the NPU bitstream

#### Scenario: Head-to-head perf report against targets

- **WHEN** a reviewer reads `docs/perf/custom-npu-vs-dpu.md`
- **THEN** it SHALL report measured geomean speedup of the NPU over the DPU baseline on the Phase 0 op set
- **AND** SHALL state whether the ≥2× op-set and ≥1× whole-graph targets were met
- **AND** the measurements SHALL trace to the same representative models used in `docs/perf/dpu-baseline.md`
