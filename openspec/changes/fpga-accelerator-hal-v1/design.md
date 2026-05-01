## Context

SmallAIOS is a `#![no_std]` Rust unikernel for AI inference. Its `onnx-rt` crate today dispatches operators directly to CPU implementations (x86 SIMD or ARM NEON/SVE) and to a CUDA execution provider on NVIDIA GPUs. There is no abstraction for "an arbitrary AXI-mapped accelerator," and the CUDA path is special-cased in dispatch logic.

The motivating workload is FPGA-accelerated ONNX inference on AMD/Xilinx Zynq UltraScale+ MPSoC (KV260 / KR260, K26 SOM): SmallAIOS boots on the four Cortex-A53 cores at EL1 via AMD's FSBL+ATF chain, an FPGA bitstream is preloaded into the PL (programmable logic), and ONNX matmul/conv ops are offloaded to PL accelerators over AXI / AXI-DMA.

There are two plausible PL accelerators:
1. **AMD's stock DPU** — a soft IP block compiled by Vitis AI, driven by `.xmodel` instruction streams via XRT/VART. CNN-biased; transformer support is partial.
2. **A custom NPU** designed in HLS — informed by what the DPU does badly for our target models.

Both will eventually exist as backends. The DPU is fastest to first-light; the custom NPU is closer to the project's clean-room ethos and better-suited to LLM workloads. This change deliberately does **not** implement either backend. It establishes the HAL boundary and the Zynq board support that both future backends will sit on top of, plus a QEMU stub backend that lets all of this be exercised on a Mac with no FPGA hardware.

**Constraints inherited from the project:**
- `#![no_std]`, edition 2021, 4-layer acyclic dependency model (kernel/security/compute/sched-types → net/ipc/posix/onnx-rt/usb → arch/* → container/bench)
- Strict acyclicity enforced by `just arch-check` and CI
- DO-178C DAL A compliance target — every requirement testable, traceable
- No new runtime crate dependencies without strong justification

**Stakeholders:**
- `onnx-rt` maintainers — own the HAL trait, dispatch refactor
- New `arch/aarch64-zynqmp` board crate — Zynq-specific code
- Bench / container — consume the HAL via existing entry points; should see no API break

## Goals / Non-Goals

**Goals:**
- Define a `ExecutionBackend` trait in `onnx-rt` with no DPU/`.xmodel`/Vitis AI vocabulary
- Make ARM-only execution one backend among several; fallback semantics explicit and per-op
- Provide a QEMU stub backend that exercises the HAL end-to-end on dev hosts
- Provide a `arch/aarch64-zynqmp` board crate sufficient to boot SmallAIOS on a real KV260/KR260 (UART output, GIC interrupts, generic timer, DDR map) — A53 cores only
- Provide a reusable AXI master + AXI-DMA driver framework with explicit cache-coherency handling
- Boot via FSBL+ATF (SmallAIOS as the EL1 payload in `BOOT.BIN`); static bitstream load only
- Preserve `#![no_std]`, no new runtime dependencies, no layer violations
- Keep Vitis / Vitis AI / `bootgen` strictly offline tools — never required at runtime or for unit tests

**Non-Goals:**
- Any DPU driver code, `.xmodel` parser, or VART runtime (deferred to `fpga-dpu-backend-v1`)
- Custom NPU RTL or HLS designs (deferred to `fpga-custom-npu-v1`)
- Dynamic bitstream reconfiguration via FPGA Manager / PMU IPI
- Cortex-R5F lockstep / safety-island / RPU-side execution
- Mali-400 GPU usage
- Production-quality boot-image signing for Zynq (a hook for `verified-boot` exists; full integration is later)
- Power management (clock gating, frequency scaling) of PL or PS

## Decisions

### Decision 1: Define the HAL with op granularity, not subgraph granularity

The `ExecutionBackend` trait operates at the **single-op** level: `can_run(&self, op: &OpDescriptor) -> bool` and `dispatch(&self, op: &OpDescriptor, tensors: &mut TensorEnv) -> Result<()>`.

**Why:** A subgraph-granularity HAL would mirror the DPU's `.xmodel` model (compile a subgraph, fire it as a unit) and bake AMD's design choices into our boundary. Op-granularity is more general — a backend that wants subgraph-level execution can implement an internal scheduler/JIT and still satisfy an op-level interface (e.g., by lazy-batching). Op granularity also lets us implement per-op fallback to ARM trivially.

**Alternatives considered:**
- *Subgraph granularity*: Faster path for DPU integration, but constrains the custom NPU and `onnx-rt`'s existing per-op dispatcher. Rejected.
- *Whole-graph offload*: Even more constraining; rejected for the same reason.
- *Hybrid (op + optional subgraph batching)*: Possible later. The op-level trait is forward-compatible — a future `BatchedBackend: ExecutionBackend` extension trait can opt in.

### Decision 2: ARM-only execution is a `ExecutionBackend` implementation, not a special case

The existing CPU dispatch path (NEON/SVE on aarch64; AVX/AVX2/AVX-512 on x86) is refactored to live behind a `CpuBackend` struct that implements `ExecutionBackend`. The runtime's dispatcher always selects from a list of registered backends; if no accelerator backend claims an op, `CpuBackend` runs it.

**Why:** Symmetry is the only way to keep the HAL honest. If "CPU" is a special path that bypasses the trait, the trait will accumulate quiet assumptions ("the CPU always runs first, then accelerators get whatever's left"). Making CPU just another backend forces the dispatch policy to be explicit.

**Alternatives considered:**
- *CPU as bypass / non-backend*: Simpler short-term, but makes the trait's ergonomics depend on what CPU happens to need. Rejected.

### Decision 3: Tensor ownership — backends borrow, never allocate from the runtime's heap

Backends receive a `TensorEnv` providing read access to inputs and write access to output buffers that the runtime has already allocated. A backend may *internally* allocate device-side memory (e.g., DMA-coherent buffers for an AXI accelerator) but never returns runtime-owned host buffers it allocated itself.

**Why:** SmallAIOS uses a memory planner that does buffer reuse and lifetime analysis. Letting backends allocate runtime tensors would invalidate the planner's contract. Internal device memory is the backend's problem and stays inside the backend.

**Alternatives considered:**
- *Backend-allocated runtime tensors*: Simpler for some accelerators but breaks memory planning. Rejected.
- *Bring-your-own-allocator API*: Considered for shared device memory; deferred — for now, internal allocation is opaque to the runtime.

### Decision 4: Backend selection is static at session creation, not per-op dynamic

`SessionConfig` gains a `backends: Vec<Box<dyn ExecutionBackend>>` (or const slice in `no_std`). Order = priority. At session-build time, the runtime walks the graph and binds each op to a backend (the first one whose `can_run` returns true) — a precomputed dispatch table. No per-op decisions at inference time.

**Why:** Per-op dynamic dispatch adds latency and makes profiling harder. Static binding plays nicely with the existing memory planner (tensor residency is known up front) and with future hybrid-residency optimizations. Backends can still report soft costs via an `estimated_ns(&self, op: &OpDescriptor) -> u64` hook (estimated nanoseconds of execution time; backends with no real estimate may return a fixed sentinel) so the runtime can pick the cheapest option among multiple capable backends.

**Alternatives considered:**
- *Dynamic dispatch*: Required only if backends have shape/dtype-dependent capability gaps that can't be predicted. Premature for now; revisit if needed.

### Decision 5: PS-PL cache coherency — explicit ports per buffer, not "always coherent"

The Zynq UltraScale+ has multiple PS↔PL ports with different coherency semantics:
- **HPC0/HPC1** (High-Performance Coherent, ACE) — PL master sees PS caches, hardware-coherent
- **HP0–HP3** (High-Performance, AXI3) — non-coherent; PS must flush/invalidate manually
- **HPM0/HPM1** (High-Performance Master, PS as master) — different direction
- **ACP** (Accelerator Coherency Port) — coherent through SCU, low latency, smaller bandwidth

The AXI-DMA framework SHALL expose port choice as an explicit type parameter on each buffer: `DmaBuffer<HpcPort>` vs `DmaBuffer<HpPort>`. Coherent ports require no manual maintenance; non-coherent ports require `clean()` before write-to-PL and `invalidate()` before read-from-PL, and these calls SHALL be enforced by the type system (only `DmaBuffer<HpPort>` exposes them; calling them on `DmaBuffer<HpcPort>` is a compile error).

**Why:** Cache-coherency bugs on Zynq are notoriously silent (you read stale data, ops "work" most of the time). Encoding port semantics in types makes the failure mode "your code doesn't compile" instead of "your model gives wrong answers under load." This is the kind of guarantee Rust's type system is for.

**Alternatives considered:**
- *Always-coherent abstraction (auto-flush everything)*: Wastes performance on coherent ports. Rejected.
- *Document-and-pray (runtime check)*: Hits us in the field. Rejected.

### Decision 6: Static bitstream load only; FPGA Manager deferred

For this change, the bitstream is preloaded by FSBL before SmallAIOS executes. A future change can add a runtime FPGA Manager driver (`fpga-manager-v1`) that talks to the PMU via IPI to load bitstreams.

**Why:** FPGA Manager / PMU IPI integration is its own can of worms (PMU firmware versioning, IPI message formats, error recovery on partial reconfiguration failure). It's not on the critical path for first-light FPGA acceleration on a known-good bitstream.

### Decision 7: QEMU stub backend uses a custom MMIO device, not a real DPU model

The QEMU stub is a simple AXI-mapped device: a few control registers, an interrupt line, a DMA descriptor table, and behavior implemented in QEMU plugin code (or a small `-device` patch). Op dispatch writes a descriptor and waits on the IRQ; the stub reads the input tensor, applies a deterministic transform (e.g., real op semantics for matmul, or a checksum for unsupported ops), and writes the output.

**Why:** The point is exercising the HAL boundary, IRQ flow, and AXI/DMA driver framework — not modeling real DPU latency or behavior. A real DPU emulator would be a multi-month project on its own. The stub is throwaway-grade and replaceable when real hardware arrives.

**Alternatives considered:**
- *No stub; real hardware only*: Defeats the no-hardware milestone goal. Rejected.
- *Cycle-accurate DPU model*: Way out of scope. Rejected.
- *Reuse an existing QEMU device (e.g., VirtIO)*: Doesn't exercise AXI/DMA semantics. Rejected.

### Decision 8: New `arch/aarch64-zynqmp` is its own crate, parallel to existing `arch/aarch64`

Existing `arch/aarch64` may target generic AArch64 / QEMU virt; Zynq UltraScale+ has enough board-specific quirks (Cadence UART vs PL011, GIC-400 vs newer GICv3, IPI to PMU) that a separate crate is cleaner than feature flags.

**Why:** Cleaner DSM (no conditional compilation across unrelated boards). If future Zynq variants ship (Versal AI Edge), they get their own crates too — `arch/aarch64-versal-aiedge`.

## Risks / Trade-offs

- **[Risk] HAL trait churns once a real backend lands** → Mitigation: build the QEMU stub first and treat *it* as the conformance test for the HAL. If the stub plus a sketch of how the DPU/NPU would slot in works, we have higher confidence the trait is right. The DPU change can still propose HAL refinements; that's fine.
- **[Risk] Static dispatch decision (Decision 4) is wrong for some future backend** → Mitigation: cost-based selection still leaves room for the backend to report dynamic costs at session-build time. If we hit a case requiring per-op dynamic dispatch, we add it as an opt-in (`ExecutionBackend::is_dynamic() -> bool`).
- **[Risk] Cache-coherency type discipline (Decision 5) is over-engineered if real workloads only use one port** → Mitigation: it costs little (two phantom-typed buffer types). If we never use HP ports, the `HpPort` type is dead code, and that's fine.
- **[Risk] FSBL+ATF dependency means BOOT.BIN generation requires Vitis on x86 Linux** → Mitigation: this is an offline build step, not a runtime dependency. We document it in `docs/zynqmp-boot.md`. Unit tests run on host CPU and don't need it. CI for the kernel build can still produce an ELF; BOOT.BIN packaging is a downstream step run only when releasing for actual hardware.
- **[Risk] QEMU stub diverges from real DPU/NPU behavior in subtle ways and bakes wrong assumptions into the runtime** → Mitigation: the stub is intentionally minimal — it does not pretend to model DPU latency, error modes, or instruction encoding. Anything stub-specific is gated behind a `qemu-stub` feature, never default.
- **[Trade-off] Op-granularity HAL (Decision 1) leaves performance on the table for backends that benefit from subgraph batching** → Acceptable: a future `BatchedBackend` extension trait can layer on. Op granularity is the lower-bound contract.
- **[Trade-off] Refactoring CPU into a backend (Decision 2) touches a lot of code** → Acceptable: it's a one-time cost. The code surfaces the dispatch policy that's currently implicit, which is a documentation win even ignoring future backends.

## Migration Plan

1. **Phase 1 — Trait + CPU refactor (no behavior change):** Define `ExecutionBackend`, refactor existing CPU dispatch into `CpuBackend`. All existing tests must pass with byte-identical results. Land before any new backend.
2. **Phase 2 — QEMU stub:** Add `qemu-stub` backend behind a feature flag. Add QEMU device (custom `-device` patch or QEMU plugin) and a `just run-arm-zynqmp-stub` recipe. Land alongside Phase 3.
3. **Phase 3 — Zynq board + AXI/DMA:** New `arch/aarch64-zynqmp` crate. New AXI/AXI-DMA driver framework. New `just build-kernel-arm-zynqmp` recipe. Boot test in QEMU. (Real hardware boot is not part of this change; that's a follow-up qualification activity once a board is in hand.)
4. **Phase 4 — Documentation:** `docs/zynqmp-boot.md` (FSBL+ATF chain, BOOT.BIN packaging via `bootgen`), `docs/accelerator-hal.md` (writing a backend), `docs/axi-dma.md` (port semantics, cache coherency).

**Rollback:** Phases 1–3 each land as separate PRs. Trait + CPU refactor can be reverted independently of the Zynq board work. QEMU stub is feature-gated and can be disabled without affecting other targets.

## Resolved Decisions

- **Trait name: `ExecutionBackend`.** Considered `Backend` (too generic — collides with networking/codec backend nomenclature), `OpBackend` (too narrow — implies single-op only). `ExecutionBackend` is unambiguous and used consistently in proposal, design, specs, and tasks.
- **Cost-hook name and units: `estimated_ns(&self, op: &OpDescriptor) -> u64`.** Returns estimated nanoseconds. Backends with no real estimate may return a fixed sentinel. Considered `estimated_cost` (abstract units) — rejected as ambiguous.

## Open Questions

1. Should `ExecutionBackend` be object-safe (`dyn ExecutionBackend`) or generic (`<B: ExecutionBackend>`)? Object-safety simplifies the dispatch table at the cost of a vtable indirection per op. Most backends will be cold-path enough that vtable cost is invisible; lean object-safe unless benchmarks show otherwise. **Default for tasks: object-safe.**
2. Where does the AXI/DMA framework live — inside `arch/aarch64-zynqmp`, or in a sibling crate (`drivers/axi`)? Versal will reuse it. Defer until we see how much code it actually is; if >500 lines, split it. **Default for tasks: start inside `arch/aarch64-zynqmp`, plan to split.**
3. Does the QEMU stub need to model AXI burst behavior, or is a single-beat MMIO transfer enough to exercise the driver? **Default for tasks: single-beat is enough for v1; bursts become a stretch goal.**
4. Should we pin the AMD Vitis version used to generate `BOOT.BIN`? Reproducibility matters for DO-178C. **Default for tasks: yes — pin in `docs/zynqmp-boot.md`, fail soft on mismatch.**
