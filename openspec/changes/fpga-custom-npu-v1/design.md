## Context

`fpga-accelerator-hal-v1` provides the HAL boundary, the `arch/aarch64-zynqmp` board, and the AXI/AXI-DMA framework. `fpga-dpu-backend-v1` lands the AMD stock DPU as a working `ExecutionBackend` and — critically for this change — produces a perf report (`docs/perf/dpu-baseline.md`) describing exactly which ops, shapes, and dtypes the DPU does badly on representative SmallAIOS target models.

This change designs and integrates a **SmallAIOS-native custom NPU** intended to address those measured DPU shortfalls. It is the project's clean-room answer to the FPGA accelerator question: no Vitis AI compiler dependency, no `.xmodel` format, the SmallAIOS runtime is the source of truth for what the accelerator does. It is also philosophically aligned with the existing from-scratch `#![no_std]` ONNX runtime stance.

The custom NPU is intended to **coexist** with the DPU (or replace it for transformer-heavy workloads), not to deprecate it. Both bitstreams may exist; `fpga-manager-v1` (sibling change) provides the runtime swap mechanism. For this change we assume static bitstream load — the custom NPU bitstream is selected at BOOT.BIN packaging time.

**Scope is explicitly contingent on DPU perf data.** The numbered Decisions below describe the **likely shape** of the design and bound the design space; the actual op set, tile sizes, dtype mix, and scratchpad sizing will be finalized after the DPU perf report lands. Tasks in `tasks.md` Phase 0 are gating: until the DPU perf report exists and a board is in hand, downstream tasks remain DEFERRED.

**Target hardware:** Kria K26 SOM (KV260 / KR260), Zynq UltraScale+ MPSoC, Cortex-A53 PS + ~256K LUT / ~144 BRAM / ~1248 DSP UltraScale+ PL. May extend to Versal AI Edge in a future change if K26 fabric proves too small.

**Constraints inherited from the project:**
- `#![no_std]`, edition 2021, 4-layer acyclic dependency model
- No new runtime crate dependencies; UltraScale+ has no open-source bitstream toolchain so Vivado / Vitis HLS stay an offline build dep
- The HAL trait (Decision 1 of `fpga-accelerator-hal-v1`) is op-granularity; the custom NPU exposes itself as a synthetic `OpDescriptor::CustomNpuSubgraph` op or a per-op claim, depending on Decision 1 below
- DO-178C DAL A trace target — every requirement testable, the design-of-the-RTL is also a versioned artifact (`hw/custom-npu/`)

**Stakeholders:**
- `onnx-rt` maintainers — own the `CustomNpuBackend` impl
- New `hw/custom-npu/` artifact tree — versioned hardware design (HLS sources, RTL, simulation testbenches, bitstream provenance)
- `arch/aarch64-zynqmp` maintainers — own the AXI/DMA glue `CustomNpuBackend` consumes
- `fpga-manager-v1` — coexists; runtime swap of DPU vs Custom NPU bitstreams

## Goals / Non-Goals

**Goals:**
- Implement `CustomNpuBackend: ExecutionBackend` in `onnx-rt::backend::custom_npu`, gated behind a non-default `custom-npu` Cargo feature
- Design the NPU itself in HLS (Vitis HLS) **or** hand-written RTL (decision in §Decision 4) — versioned under `hw/custom-npu/` with HLS sources, generated RTL, simulation testbenches, and bitstream provenance metadata
- Cover at minimum the ops the DPU does worst (per the perf report) — likely candidates include: matmul + GEMM-fused-bias, layernorm, RMSNorm, softmax, gather/scatter for KV-cache. Final list set after Phase 0 closes.
- Provide on-die activation and weight buffers, plus a small scratchpad sized for layernorm/softmax statistics
- Achieve a measurable perf win over the DPU for the target ops on representative models (target: ≥2× geomean speedup on the Phase 0 op set, ≥1× on the rest of the graph)
- Provide a co-simulation harness so the runtime's golden output and the RTL's bit-accurate output agree before any board run
- Reuse the AXI/AXI-DMA framework, the typed cache-coherency discipline, and the IRQ-driven completion pattern from the HAL change — no new infrastructure
- Reuse the perf instrumentation hooks introduced by `fpga-dpu-backend-v1` (mirror the `dpu-profile` shape as `custom-npu-profile`)

**Non-Goals:**
- Replacing the DPU backend — both can coexist, picked at BOOT.BIN packaging time or via `fpga-manager-v1`
- Open-source bitstream generation — UltraScale+ has no open toolchain; we pin a Vivado/Vitis HLS version
- Versal AI Edge support — separate change (`versal-aiedge-board-v1`) if/when the K26 fabric is too small
- Multi-NPU / multi-tile within one bitstream (single-instance for v1)
- Quantization-aware training pipelines — we accept what the runtime's existing model loaders produce; if INT8 is needed, the existing CPU INT8 path dictates the format and the NPU matches
- Replacing the existing CPU SIMD kernels — CPU residual remains the lower-priority backend
- Power-aware DVFS, clock gating, or thermal management of the PL — out of scope

## Decisions

### Decision 1: Per-op dispatch claim, not synthetic-subgraph dispatch

Unlike `DpuBackend` which dispatches at synthetic-subgraph granularity (because Vitis AI emits a pre-compiled instruction stream), `CustomNpuBackend::can_run` SHALL claim individual ONNX ops. Each accelerated op is its own dispatch call, decomposed into AXI-DMA + control-register sequences inside the backend.

**Why:** We control the design. There is no upstream compiler emitting opaque blobs. Per-op claims play directly into the existing dispatch table and let the runtime mix-and-match ops between CPU residual and NPU on the same model with no extra parser. This is also what `fpga-accelerator-hal-v1` Decision 1 was already optimized for.

**Alternatives considered:**
- *Subgraph-fusion at session-build time*: Possible later if profiling shows per-op DMA overhead dominates. Current default — per-op — is the simpler contract; revisit when measurement says otherwise.
- *Custom IR layer between ONNX and the backend*: Premature; if it becomes warranted, lift to a separate change.

### Decision 2: Op set is data-driven and locked at end of Phase 0

The set of ops `CustomNpuBackend` claims is finalized after the DPU perf report (`docs/perf/dpu-baseline.md` from `fpga-dpu-backend-v1`) is reviewed in Phase 0. The proposal lists *candidates* — matmul + GEMM-fused-bias, layernorm, RMSNorm, softmax, gather for KV-cache lookup — but the final list MAY add or drop ops based on what the measurement shows.

The **floor** is matmul + GEMM-fused-bias. Anything weaker than that is not worth taping out.

**Why:** Designing an NPU before the perf data is in hand would be Cargo-cult engineering. The whole reason `fpga-dpu-backend-v1` ships first is to pin down what hurts.

**Alternatives considered:**
- *Pick the op set up front based on intuition*: Rejected. Too easy to design for ops the DPU already does fine.
- *Ship a minimal "matmul-only" NPU and grow it*: This is in fact the floor. The question is "how much above the floor." Decided per measurement.

### Decision 3: Numeric format — INT8 floor, BF16 stretch, FP16 only if measurement demands it

The NPU SHALL support INT8 multiply-accumulate at minimum. BF16 SHALL be a stretch goal — included if PL resource budget permits without breaking tile-size targets. FP16 SHALL be considered only if the perf report shows BF16 alone is insufficient for the target models' accuracy.

**Why:** INT8 covers the existing quantized inference path; BF16 is increasingly the format of choice for transformer LLMs and aligns with the project's transformer roadmap. FP16 adds significant DSP-block usage on UltraScale+ and is largely redundant with BF16 for inference.

**Alternatives considered:**
- *INT8-only*: Cheapest, fewest LUTs, but eventually limits transformer accuracy. Acceptable as v1.
- *FP16+INT8 with no BF16*: Worse compute density per DSP. Rejected.
- *INT4*: Promising for LLMs but adds quant-aware-training pipeline work that is out of scope. Possible follow-up.

### Decision 4: HLS first (Vitis HLS), with explicit fallback to hand-RTL on critical kernels if HLS QoR is insufficient

The first cut SHALL be Vitis HLS. The matmul tile, layernorm/softmax pipelines, and AXI-stream interfaces SHALL all be written in C++ HLS. Where post-synthesis Quality-of-Results (QoR) — Fmax, LUT count, DSP utilization, latency — is unacceptable for a critical kernel, that kernel MAY be rewritten in hand-Verilog/SystemVerilog. The matmul tile is the most likely candidate for hand-RTL.

**Why:** HLS dramatically shrinks design loop time, makes the design auditable to a Rust-comfortable team, and produces synthesizable cycles-accurate testbenches we can drive from the SmallAIOS runtime via the AXI framework's debug harness. Hand-RTL is reserved for cases where HLS leaves significant performance on the table.

**Alternatives considered:**
- *Hand-RTL only*: Authentic hardware-engineering practice, but multi-month design loop on a small team. Rejected for v1.
- *MyHDL / SpinalHDL / Chisel*: Interesting, but adds Java/Scala/Python toolchain to the offline build. Rejected — Vivado already requires Linux x86, no need to add more.

### Decision 5: Resource budget — leave 30% PL headroom for AXI/DMA, debug, future overlay coexistence

Of the K26's PL resources (~256K LUT / ~144 BRAM / ~1248 DSP), the custom NPU SHALL target a **70%** ceiling per resource type. The remaining 30% is reserved for: AXI-DMA controllers and AXI plumbing the framework needs, ILA / VIO / chipscope debug instrumentation during bring-up, and headroom for `fpga-manager-v1` to load partial-reconfig overlays alongside the NPU later.

**Why:** Filling the K26 fabric to 100% leaves no room for debug. We will *want* an ILA on the matmul tile during silicon bring-up. 70% is a defensible budget; if the design needs more, that is an explicit conversation, not a creep.

**Alternatives considered:**
- *80–90% target*: Tighter perf, but no debug instrumentation room. Rejected.
- *50% target*: Leaves a third of the fabric idle. Rejected unless the perf goal is met early.

### Decision 6: `CustomNpuBackend` consumes a board-supplied handle — same pattern as DPU and QEMU stub

`arch/aarch64-zynqmp::custom_npu` exposes a `CustomNpuHandle` that wraps the AXI-mapped NPU peripheral (control registers, IRQ subscription, DMA channels). `onnx-rt::backend::custom_npu::CustomNpuBackend::new(handle)` consumes it. Identical pattern to `DpuBackend::new(DpuHandle)` and `QemuStubBackend::new(StubHandle)`.

**Why:** Layer model says runtime cannot know AXI addresses. Established pattern. Trivially consistent.

### Decision 7: Co-simulation as the primary verification — RTL/HLS testbench fed from the same runtime call as the host CPU reference

The `hw/custom-npu/sim/` testbench SHALL be drivable from the same `OpDescriptor` + tensor inputs the runtime would dispatch on real hardware. The harness SHALL run the NPU implementation in a Verilator/QuestaSim co-sim *and* the SmallAIOS CPU reference in parallel and SHALL compare outputs bit-accurately for INT8 ops, within a documented tolerance for BF16/FP16 ops.

**Why:** Without this, every silicon-bring-up debug session is a coin flip between "the runtime is wrong," "the RTL is wrong," and "the integration glue is wrong." Co-sim makes the first two distinguishable before silicon. This is the same discipline the existing CUDA path uses (CPU reference for every kernel).

The host-side runtime tests in this change SHALL exercise the co-sim path so unit tests on a developer Mac validate runtime + RTL together — even before a board is in hand.

**Alternatives considered:**
- *Bring-up direct-on-silicon, debug after*: Wastes silicon-bring-up time. Rejected.
- *Software model only, no RTL co-sim*: Misses RTL bugs entirely. Rejected.

### Decision 8: Bitstream selection is static at BOOT.BIN packaging time; coexistence with DPU is via `fpga-manager-v1`

For this change, only one bitstream is loaded per boot: either DPU or Custom NPU, chosen by which artifact is packed into BOOT.BIN. `fpga-manager-v1` (sibling) adds the runtime swap mechanism. The `CustomNpuBackend` is feature-flag-gated and probes the running PL configuration; if its expected NPU IDs are not present, it returns `Err(BackendUnavailable)` cleanly — same as `DpuBackend`.

**Why:** Adding runtime bitstream swap into this change conflates two large pieces of work. Decoupling them lets each ship on its own merits.

**Alternatives considered:**
- *Land swap mechanism here too*: Out of scope; defer to `fpga-manager-v1`.

## Risks / Trade-offs

- **[Risk] DPU perf report shows the DPU is "good enough" on our target models** → Mitigation: this change is allowed to be cancelled. Phase 0 explicitly closes a go/no-go gate. We do not start tape-out / RTL design until the gate closes "go." The proposal stays as a roadmap reference even if cancelled.
- **[Risk] HLS QoR is so bad we cannot meet the perf goal even with the full PL budget** → Mitigation: Decision 4 already allows hand-RTL fallback for hot kernels. If matmul HLS is the bottleneck, replace the tile with hand-RTL while keeping the AXI plumbing in HLS.
- **[Risk] K26 fabric is too small to fit the design at 70% target while meeting perf** → Mitigation: tile-size scaling is a design parameter; we shrink tiles before we drop ops. If even the smallest viable tile blows the budget, escalate to a Versal AI Edge change in a follow-up. Do not silently bust the 70% budget.
- **[Risk] Co-simulation harness diverges from real silicon behavior** → Mitigation: the same RTL netlist runs in sim and synthesis. Cycle-accurate sim is the gold standard until silicon proves otherwise; once silicon arrives, any sim/silicon mismatch is a recorded bug filed against the harness, not "expected drift."
- **[Risk] Vivado / Vitis HLS version pin breaks against vendor updates** → Mitigation: pin in `docs/zynqmp-custom-npu.md`. CI runs against the pinned version only. Vendor-update churn is a separate, explicit task.
- **[Risk] BF16 support blows resource budget** → Acceptable: BF16 is a stretch. If it does not fit, ship INT8-only v1 and add BF16 in a v2 once we have a Versal board.
- **[Risk] DO-178C trace target is harder for hardware artifacts** → Mitigation: `hw/custom-npu/` carries its own provenance manifests (HLS source git rev, RTL synthesis log hash, bitstream MD5). Trace links from the proposal/spec/tasks to those manifests. Real DAL A acceptance is far in the future; for this change we lay the trace foundation.
- **[Trade-off] Per-op dispatch claim (Decision 1) means more AXI-DMA round trips than subgraph-fused dispatch would** → Acceptable: simpler contract, easier debugging. If profiling shows DMA overhead dominates, lift to subgraph-fused in a follow-up.
- **[Trade-off] Co-existing with DPU under separate bitstreams (Decision 8) means BOOT.BIN flavor proliferation** → Acceptable: the BOOT.BIN packaging guide handles this. If it gets unwieldy, `fpga-manager-v1` handles runtime swap.

## Migration Plan

1. **Phase 0 — Go/no-go gate (gating; downstream phases DEFERRED until this closes "go"):** Read the DPU perf report from `docs/perf/dpu-baseline.md`. Decide which ops are worth accelerating. Decide whether INT8 alone or INT8+BF16. Decide HLS-first or hand-RTL-first. Decide tile sizes that fit the 70% PL budget. Land a `docs/zynqmp-custom-npu-design-doc.md` capturing those decisions with traceability back to specific perf-report numbers.
2. **Phase 1 — HLS / RTL design + co-sim.** `hw/custom-npu/` tree: HLS sources, generated RTL, Verilator/QuestaSim testbenches, runtime-driven co-sim harness. Bit-accurate matches between runtime CPU reference and RTL for the v1 op set.
3. **Phase 2 — `arch/aarch64-zynqmp::custom_npu` driver.** AXI peripheral wrapping, IRQ wiring, DMA channel binding, `CustomNpuHandle` API. Mirrors the DPU driver shape from `fpga-dpu-backend-v1`.
4. **Phase 3 — `CustomNpuBackend` runtime impl.** `onnx-rt::backend::custom_npu`, `custom-npu` feature flag, per-op `can_run` and `dispatch`, `custom-npu-profile` instrumentation mirroring `dpu-profile`.
5. **Phase 4 — Bitstream packaging + bring-up.** Vivado pin, BOOT.BIN packaging variant for the custom NPU bitstream, on-board bring-up using ILA + the runtime's existing diagnostics. Document what works and what does not.
6. **Phase 5 — Documentation, perf comparison, archive prep.** `docs/zynqmp-custom-npu.md`, `docs/perf/custom-npu-vs-dpu.md` (the head-to-head), updates to `CLAUDE.md` for new feature flags.

**Rollback:** Phases 1–4 each ship as separate PRs against `develop`. The runtime side is feature-flag-gated; reverting `custom-npu` from a build returns to DPU-or-CPU. The hardware artifact tree is independent — reverting the runtime PR does not require touching `hw/custom-npu/`.

## Resolved Decisions

- **Backend module path: `onnx-rt::backend::custom_npu`.** Mirrors `onnx-rt::backend::dpu` and `onnx-rt::backend::cpu`.
- **Hardware artifact tree path: `hw/custom-npu/`.** Sibling to existing source trees; not in the Cargo workspace. Provenance manifests committed; build outputs (the bitstream itself) NOT committed (too large; produced offline per pinned Vivado).
- **Naming: "Custom NPU" not "SmallAIOS NPU" or "K26 NPU".** Generic name; the *design* may carry a versioned codename (e.g., `npu-v1-emerald`) recorded in the artifact tree.

## Open Questions

1. Subgraph fusion (Decision 1 alternative) — when to lift? **Default for tasks: not in v1. Add a profile counter that records inter-op DMA overhead so the question is data-driven.**
2. Off-die DDR for weights vs on-die BRAM — does the working set fit BRAM? **Default for tasks: weights via AXI from DDR with on-die BRAM as cache. Pre-fetch via a separate AXI-DMA channel. Actual sizing decided in Phase 0.**
3. Activation function support — fused into matmul tile (ReLU, GELU) or separate pipeline? **Default for tasks: fused for piecewise-linear (ReLU); separate pipeline for transcendentals (GELU, Swish). Final per Phase 0.**
4. KV-cache gather op — accelerated or stays on CPU? **Default for tasks: on the candidate list per the proposal; final decision per Phase 0.**
5. Co-existence of DPU and Custom NPU **in the same bitstream** — supported in v1, or strictly via `fpga-manager-v1` swap? **Default for tasks: strictly via swap. Putting both in one bitstream blows the 70% budget and complicates dispatch policy.**
6. RTL versioning — semver per `hw/custom-npu/`? **Default for tasks: yes. Hardware artifact tree carries its own VERSION file, bumped on any RTL change. Trace from runtime feature-flag value to bitstream provenance MD5.**
7. Should `CustomNpuBackend::estimated_ns` use the perf-report-driven model (per-op calibrated nanoseconds), or a static placeholder? **Default for tasks: calibrated. Hardware-bring-up Phase 4 captures per-op latency on real silicon and `estimated_ns` reads from a compile-time constant table.**
