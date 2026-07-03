## Context

`fpga-accelerator-hal-v1` defines the `ExecutionBackend` trait, the AXI/AXI-DMA framework with typed PS-PL coherency, the `arch/aarch64-zynqmp` board crate (UART, GIC-400, generic timer, DDR map), and the QEMU stub backend. It deliberately ships zero real FPGA backends. This change is the first real backend and the most concrete of the three FPGA roadmap follow-ups.

The target hardware is the AMD/Xilinx **Deep Learning Processing Unit (DPU)** — specifically `DPUCZDX8G` on Zynq UltraScale+ (the variant used in the KV260 / KR260 stock bitstream). It is a soft IP block compiled by Vitis AI tooling, AXI-mapped, instruction-stream driven (Vitis AI emits `.xmodel` files containing pre-compiled DPU instruction sequences plus a residual subgraph that runs on the host CPU). AMD ships a stock bitstream for the K26 SOM that includes a DPU instance; we use **that** bitstream as-is. We do not regenerate DPU bitstreams in this change.

Key references:
- AMD PG338 *DPU IP Product Guide* — register map, instruction format, IRQ semantics
- Vitis AI `.xmodel` format (XIR — Xilinx Intermediate Representation, protobuf-encoded)
- Vitis AI compiler (offline, x86 Linux only) — converts a quantized ONNX model to `.xmodel`

**Why DPU first:** The DPU is the shortest path to real FPGA-accelerated inference on a Kria board. It avoids HLS/RTL design loops, has a known-good silicon configuration, and produces honest perf numbers that inform `fpga-custom-npu-v1`. It is also a useful systems-engineering shake-down — it forces the AXI-DMA framework, IRQ flow, and `.xmodel`-to-runtime impedance match into production-quality shape on real hardware.

**Constraints inherited from the project:**
- `#![no_std]`, edition 2021, 4-layer acyclic dependency model
- No new runtime crate dependencies (Vitis AI stays strictly offline / x86 Linux)
- No vendor leak into the HAL trait — `DpuBackend` is one impl among several
- DO-178C DAL A trace target — every requirement testable, every off-target dependency documented
- No PMU IPI / runtime bitstream reconfiguration here (deferred to `fpga-manager-v1`); we run on whatever the FSBL preloaded

**Stakeholders:**
- `onnx-rt` maintainers — own the dispatch surface, CPU fallback semantics
- `arch/aarch64-zynqmp` maintainers — own the AXI/DMA framework `DpuBackend` consumes
- Container/bench owners — DPU enabled via Cargo feature, no API break expected

## Goals / Non-Goals

**Goals:**
- Implement `DpuBackend: ExecutionBackend` in `onnx-rt::backend::dpu`, gated by a non-default `dpu` Cargo feature
- Implement a minimal `.xmodel` parser sufficient to load the subgraph instruction stream emitted by Vitis AI for our representative target models — not the full XIR format
- Implement the DPU register / instruction-stream protocol per AMD PG338 (control regs, descriptor submission, completion IRQ)
- Wire `DpuBackend` to the AXI-DMA framework for input/output tensor transfer with explicit cache-coherency
- Honor existing per-op fallback-to-CPU semantics from the HAL — DPU-unsupported ops return `Err(ExecError::FallbackToCpu)` so the runtime hits `CpuBackend`
- Provide a `gpu-profile`-style perf instrumentation path (per-op DPU latency, DMA bytes in/out, idle stall time) so `fpga-custom-npu-v1` has measured DPU shortfall data to design against
- Document the offline workflow (ONNX → quantized ONNX via Brevitas → Vitis AI compile → `.xmodel`) in `docs/zynqmp-dpu.md`
- Provide a `just run-arm-zynqmp-dpu` recipe that boots SmallAIOS in QEMU with the DPU-bearing bitstream artifact present in the boot image. Under QEMU this validates only software-side packaging and `.xmodel` parsing — DPU instruction execution requires real silicon

**Non-Goals:**
- Custom DPU configurations or DPU bitstream regeneration (we use AMD's stock K26 overlay)
- Dynamic bitstream loading via FPGA Manager / PMU IPI (deferred to `fpga-manager-v1`)
- Custom NPU RTL / HLS designs (deferred to `fpga-custom-npu-v1`)
- Vitis AI as a runtime dependency — compiler stays offline, x86 Linux only
- Full XIR support — we parse only the subset of the `.xmodel` graph needed for stock-DPU subgraph dispatch
- Any DPU-specific public API on `onnx-rt` outside the `dpu` module (vendor stays inside the backend)
- Multi-DPU / multi-tile orchestration (the K26 stock bitstream ships a single B4096 DPU instance)
- Mixed-precision beyond what Vitis AI emits — we accept what the compiler produces (typically INT8 with a small handful of FP32 ops on the residual subgraph)

## Decisions

### Decision 1: Parse only the subset of `.xmodel` needed to dispatch a DPU subgraph; fall back to CPU for everything else

A `.xmodel` is a XIR protobuf. It contains: (a) a graph of ops, (b) per-DPU-subgraph attached attributes including the DPU instruction stream blob, the input/output tensor shape and quant params, and the device-handle metadata. Full XIR has dozens of op types and graph-rewrite history we do not need at runtime.

The parser SHALL extract:
- DPU subgraph boundaries (which ops run on DPU vs which run on the host)
- For each DPU subgraph: the instruction-stream blob bytes, input tensor descriptors, output tensor descriptors, weight/bias blob references, scratch-buffer size hint
- For host-residual ops: the op type, attributes, and tensor shapes — re-emitted as `OpDescriptor` for the existing CPU dispatch path

The parser SHALL **not** attempt to interpret DPU-internal instructions, perform graph optimization, or honor every XIR attribute. Anything we do not understand is either ignored (advisory) or rejected at load time (with a clear error pointing at `docs/zynqmp-dpu.md`).

**Why:** A full XIR implementation is a multi-month project dominated by code we will never use. A small parser sized to "enough to enumerate subgraphs, extract instruction blobs, and re-emit residual ops" is on the order of 1–2 KLOC and stays auditable.

**Alternatives considered:**
- *Vendor a Vitis AI / VART runtime port to `no_std`*: Many KLOC of C++. Rejected.
- *Compile XIR to a SmallAIOS-native IR offline*: Possible later. For v1 we parse on-device; if startup latency is a concern, we add an offline conversion in a follow-up.
- *Reject `.xmodel` entirely; only accept ONNX*: Defeats the point — the DPU instruction stream is what makes the DPU fast. We need the compiler-emitted blob.

### Decision 2: Treat DPU subgraphs as a single `OpDescriptor::DpuSubgraph` op at the runtime boundary

The `ExecutionBackend` trait operates at op granularity. A DPU subgraph is many ONNX ops fused into one DPU instruction stream. We expose this to the runtime as a synthetic op `OpDescriptor::DpuSubgraph { input_tensors, output_tensors, instruction_blob_id }` produced by the `.xmodel` loader. The dispatch table binds these synthetic ops to `DpuBackend`; the residual ONNX ops bind to `CpuBackend` as usual.

**Why:** This keeps the HAL trait contract honest — `DpuBackend::dispatch` runs exactly one instruction stream per call, no internal scheduler. It matches Decision 1 from `fpga-accelerator-hal-v1` (op granularity) without fighting the DPU's natural unit of work.

**Alternatives considered:**
- *Decompose DPU subgraphs into ONNX ops*: Defeats the point of having a pre-compiled instruction stream. Rejected.
- *Add a subgraph-level extension trait*: Proposed in HAL Decision 1 as future work. Premature here — synthetic op suffices.

### Decision 3: IRQ-driven completion, no polling, no busy-wait

The DPU control registers expose a completion-bit-on-IRQ pattern (see PG338 `DPU_INTR`). `DpuBackend::dispatch` SHALL submit the instruction stream, register a completion waker against the GIC SPI line wired to the DPU IRQ, and yield until the IRQ fires. The completion handler SHALL clear the IRQ, read result-status registers, and wake the dispatching task.

A **diagnostic** polling fallback SHALL exist behind a `dpu-polling-debug` feature for bring-up only. It SHALL NOT be enabled in any production or CI build.

**Why:** Polling on a `#![no_std]` cooperative-async kernel wastes the core that should be running CPU residual ops or the next inference's prep work. The HAL's whole story is "yield at op boundaries" — DPU ops are op boundaries.

**Alternatives considered:**
- *Polling-only*: Simpler bring-up, worse production behavior. Use behind feature only.
- *Hybrid (poll for short ops, IRQ for long)*: Premature optimization. Revisit if profiling shows wakeup latency dominates real op runtime.

### Decision 4: `DpuBackend::probe()` reads the DPU MMIO signature; missing DPU returns `BackendUnavailable` cleanly

On construction `DpuBackend` does **not** assume a DPU exists. `probe()` reads the AMD-defined DPU identification register (`DPU_VER` per PG338) at a configured AXI address, validates the signature against a list of supported DPU variants (initially: `DPUCZDX8G` B4096 only), and returns `Err(BackendUnavailable)` on mismatch / unreadable address.

The configured AXI address SHALL come from a board-level constant in `arch/aarch64-zynqmp` (it is fixed on the K26 stock bitstream). It SHALL NOT be hard-coded in `onnx-rt`.

**Why:** Same model as `QemuStubBackend::probe` from the HAL change. Lets the same kernel image run on a non-DPU board and gracefully exclude `DpuBackend` from the active dispatch table.

**Alternatives considered:**
- *Trust construction-time configuration; panic if absent*: Bad for cross-board reuse. Rejected.
- *Probe via a kernel device tree*: We do not consume the device tree at runtime here. Could add later if other boards need it.

### Decision 5: Cache-coherency port choice — HPC0 for activations, HP0 for weights/instructions

DPU traffic splits into three streams:
- Activations (read+write, hot) — placed on **HPC0** (coherent, ACE) so we never flush on the activation-tensor critical path
- Weights (read-only, cold-after-load) — placed on **HP0** (non-coherent), explicit `clean_for_device()` at load time only
- Instruction stream (read-only, cold-after-load) — placed on **HP0**, same as weights

**Why:** Activations dominate dispatch overhead. Coherent paths spare us flush/invalidate per call. Weights and instructions load once per session — the HP path's manual cache management is fine for a one-time cost.

The HAL's typed `DmaBuffer<HpcPort>` / `DmaBuffer<HpPort>` distinction (HAL Decision 5) prevents accidental cross-port misuse: only `DmaBuffer<HpPort>` exposes `clean_for_device()`; the activation buffer's type forbids the call entirely at compile time.

**Alternatives considered:**
- *Everything on HPC0*: Shares HPC0 bandwidth across activations + weights + instructions; can stall the hot path. Rejected.
- *Everything on HP0*: Maximum bandwidth, but every activation transfer pays a flush. Rejected for transformer-y workloads where activations dominate.
- *ACP for activations*: ACP is lower-bandwidth than HPC and intended for small coherent traffic. Wrong tier for full activations.

### Decision 6: `DpuBackend` is constructed by the board crate, not by the runtime

`onnx-rt` does not know about Zynq AXI addresses. The `arch/aarch64-zynqmp` crate exposes a `DpuHandle` that wraps the AXI-mapped DPU peripheral (control registers, IRQ binding, DMA channel handles). `onnx-rt::backend::dpu::DpuBackend::new(handle: DpuHandle) -> DpuBackend` consumes that handle.

**Why:** Keeps the layer model clean — `onnx-rt` is Layer 1, board crate is Layer 2, board hands a pre-bound peripheral *up* to the runtime. This is the same pattern used by the QEMU stub backend (Decision from HAL).

**Alternatives considered:**
- *Runtime opens AXI mappings itself*: Would require Layer 1 to know about AXI addresses — layer violation. Rejected.

### Decision 7: Perf instrumentation behind a `dpu-profile` feature, mirroring the existing `gpu-profile` shape

Per-op DPU dispatch instrumentation: instruction stream submit ts, completion IRQ ts, DMA bytes in, DMA bytes out, residual-CPU op count and total host time. Aggregated to a per-session summary written to stderr at `DpuBackend::drop` (mirrors `CudaRuntime::drop` in the existing CUDA path). Off by default — production builds pay zero overhead.

**Why:** `fpga-custom-npu-v1` explicitly depends on perf data from this change. Building the instrumentation in from the start is much cheaper than retrofitting later.

### Decision 8: Hand-rolled `.xmodel` protobuf parser, not a third-party crate

The XIR `.xmodel` is a protobuf. SmallAIOS already has a hand-rolled `#![no_std]` protobuf parser used by the ONNX runtime for the ONNX model format. We extend it with the small set of XIR-specific message types we need (subgraph, attribute, op-def, tensor-def). No new third-party crate.

**Why:** Same parser, same audit surface, same `no_std` story. Adding `prost` or `rust-protobuf` brings transitive deps and a much larger surface to vet for DO-178C.

**Alternatives considered:**
- *`prost`*: 50+ KLOC of generated code per message set, plus `bytes` and `prost-build` at compile time. Rejected.

## Risks / Trade-offs

- **[Risk] `.xmodel` format drift across Vitis AI versions** → Mitigation: pin a Vitis AI version in `docs/zynqmp-dpu.md`. Parser SHALL warn (not fail) on unknown protobuf field tags so newer minor versions degrade gracefully. Add a CI check that re-runs the parser against a corpus of small `.xmodel` files committed under `tests/fixtures/dpu/`.
- **[Risk] Stock K26 DPU's B4096 config is not optimal for transformer workloads** → Acceptable: this change explicitly does not promise transformer perf. Its goal is first-light + measurement. The data feeds `fpga-custom-npu-v1`.
- **[Risk] `DpuBackend` regresses if a future HAL change moves the trait surface** → Mitigation: keep all DPU vendor knowledge inside `onnx-rt::backend::dpu`; the trait surface is the only contact point. If the HAL changes, `DpuBackend` updates with it as a one-file PR.
- **[Risk] QEMU run recipe creates the impression DPU works under QEMU** → Mitigation: `just run-arm-zynqmp-dpu` SHALL print a banner stating "DPU instructions do not execute under QEMU; this validates packaging only." The CI matrix entry SHALL be labeled accordingly.
- **[Risk] Brevitas-quantized models drift from ONNX-quant ones** → Acceptable: Vitis AI compile is the choke point; the offline pipeline is what it is. Document the calibration recipe in `docs/zynqmp-dpu.md` and pin Brevitas version.
- **[Risk] Cache-coherency type discipline (HAL Decision 5) is wrong for some DPU access pattern** → Mitigation: cover with a unit test that wraps the three buffer roles (activations, weights, instructions) with a debug cache-tracker and verifies each role's expected maintenance pattern.
- **[Trade-off] We pay `.xmodel` parsing cost at session creation** → Acceptable for now. If startup latency becomes a constraint, add an offline `xmodel-to-smallaios` converter in a future change.
- **[Trade-off] `DpuBackend` is a single-instance backend (B4096 only)** → Acceptable for K26. KR260 ships the same. Versal AI Edge would be a separate change with its own backend.

## Migration Plan

Phases land as separate PRs against `develop`:

1. **Phase 1 — `.xmodel` parser + offline corpus.** Hand-rolled protobuf, XIR message types, parse-into-`DpuSubgraph` types, ignore unknowns gracefully. Land a small fixtures corpus in `tests/fixtures/dpu/` (tiny MatMul-only, tiny Conv-only). No driver code; tests run on host.
2. **Phase 2 — DPU register protocol + driver in `arch/aarch64-zynqmp`.** AXI peripheral wrapping per PG338, control registers, IRQ wiring through GIC SPI, completion-IRQ async future. Unit tests via the AXI framework's debug harness; QEMU smoke that verifies no panic on probe-then-fall-through (no real DPU emulation).
3. **Phase 3 — `DpuBackend` wiring, fallback semantics, instrumentation.** `DpuBackend::new(handle)`, `can_run` returns true only for synthetic `DpuSubgraph` ops. Wire into `SessionConfig`. `dpu-profile` feature. End-to-end test: load tiny `.xmodel`, verify dispatch table places `DpuSubgraph` on `DpuBackend` and residuals on `CpuBackend`, simulate stub completion, verify outputs match a CPU reference path within the quant tolerance.
4. **Phase 4 — Documentation + offline workflow.** `docs/zynqmp-dpu.md` (Vitis AI version pin, Brevitas pipeline, BOOT.BIN packaging notes, "QEMU does not execute DPU instructions" banner).

**Rollback:** All phases ship behind the `dpu` feature flag. Default builds are unaffected. Reverting the feature flag from a branch reverts the change cleanly.

## Resolved Decisions

- **Backend module path: `onnx-rt::backend::dpu`.** Mirrors `onnx-rt::backend::qemu_stub` and `onnx-rt::backend::cpu`.
- **Synthetic op naming: `OpDescriptor::DpuSubgraph`.** Considered `OpDescriptor::Subgraph(BackendId)` (more general) — rejected as premature; the moment `fpga-custom-npu-v1` lands we will introduce `OpDescriptor::CustomNpuSubgraph` and possibly refactor to a generic at that time. For now, vendor-keyed is clearer.
- **Probe register: `DPU_VER` at the K26-stock-bitstream-defined AXI offset.** Specific offset constant lives in `arch/aarch64-zynqmp::dpu`, not `onnx-rt`.

## Open Questions

1. Should `DpuBackend` consume `.xmodel` directly, or should the runtime accept ONNX and dispatch some opaque "compiled subgraph" handle the DPU emits? **Default for tasks: `DpuBackend` consumes `.xmodel` at session-build time. The user-facing input is ONNX; the offline pipeline produces a `.xmodel` sidecar that the runtime loads when `dpu` feature is on.**
2. Do we need a "best-effort" path that runs the ONNX graph entirely on CPU when `.xmodel` is missing? **Default for tasks: yes — if `.xmodel` is missing and the model is ONNX, dispatch falls through to `CpuBackend` for every op. The DPU is opt-in per session, not per process.**
3. How do we handle quant scale/zero-point mismatches between Brevitas-quantized ONNX and the DPU's INT8 expectations? **Default for tasks: defer to Vitis AI compiler; if it produces a `.xmodel` we honor what it emits. If real-hardware tests show mismatches, document recovery in `docs/zynqmp-dpu.md`.**
4. Per-session vs per-inference DPU reset on error? **Default for tasks: per-inference reset on instruction-fault IRQ; per-session reset on hard timeout (>10× expected runtime). Both surface as `ExecError::FallbackToCpu` for the offending op so the runtime stays live.**
5. Should `DpuBackend` claim Concat/Transpose/Reshape ops the DPU subgraph absorbs, or leave them as residual? **Default for tasks: leave to Vitis AI's choices — whatever it puts inside the DPU subgraph, the backend runs; whatever it leaves outside is residual. Do not reverse-engineer the compiler's decisions.**
