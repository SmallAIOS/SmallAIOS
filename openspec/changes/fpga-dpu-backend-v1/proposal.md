> **Status:** Roadmap stub. This is the second of three follow-up changes to `fpga-accelerator-hal-v1`.
> Detailed specs/design/tasks deferred until `fpga-accelerator-hal-v1` lands and a KV260/KR260 is available for first-light testing.
> Sibling stubs: `fpga-custom-npu-v1`, `fpga-manager-v1`.

## Why

`fpga-accelerator-hal-v1` defines the `ExecutionBackend` trait and the `arch/aarch64-zynqmp` board support, but ships no real FPGA backend. The shortest path to actually-accelerated ONNX inference on a Kria board is to drive AMD's stock DPU (Deep Learning Processing Unit) — the soft IP block AMD ships pre-built in the KV260's stock bitstream — from SmallAIOS. This is "Option A" of the agreed two-step plan: use the DPU first to learn the platform on a known-good target and to gather perf data that informs the eventual custom NPU.

## What Changes

- New `ExecutionBackend` implementation: `DpuBackend` in `onnx-rt::backend::dpu` (behind a `dpu` Cargo feature), driving the AXI-mapped DPU IP via the framework from `fpga-accelerator-hal-v1`
- Minimal `.xmodel` parser sufficient to load Vitis-AI-compiled subgraph instruction streams (subset of XRT/VART, hand-rolled, `#![no_std]`)
- DPU instruction loader, control-register protocol (per AMD PG338), and IRQ-driven completion handling
- Per-op fallback to `CpuBackend` for ops the DPU does not support (LayerNorm, custom activations, etc.) — leverages the fallback semantics already in the HAL
- Documentation of the offline workflow: ONNX → quantized ONNX (Brevitas) → Vitis AI compile → `.xmodel`, run on x86 Linux box, ship `.xmodel` to SmallAIOS
- New CI matrix entry: build with `dpu` feature enabled
- New `just run-arm-zynqmp-dpu` recipe (bakes a stock DPU bitstream into the QEMU boot image; real-hardware run is documented but not in CI)

Out of scope:
- DPU **bitstream** generation (we use AMD's stock DPU overlay; custom DPU configurations deferred)
- Dynamic bitstream loading (deferred to `fpga-manager-v1`)
- Custom NPU RTL (deferred to `fpga-custom-npu-v1`)
- Vitis AI as a *runtime* dependency (compiler stays offline, Linux x86 only)

## Capabilities

### New Capabilities

- `dpu-backend`: `DpuBackend` `ExecutionBackend` implementation, `.xmodel` runtime, DPU instruction protocol, perf instrumentation hooks. Detailed requirements TBD.

### Modified Capabilities

- *(None expected — the HAL is the contract; this change adds an implementation behind it.)*

## Impact

**Code:** new module in `onnx-rt`, no changes to the HAL trait expected.

**Build:** new `dpu` feature flag, new `just` recipes, new CI matrix entry.

**Dependencies:** offline build dependency on Vitis AI (Linux x86, pinned version). No new runtime crate dependencies.

**Hardware:** assumes a KV260 or KR260 in hand for first-light validation. QEMU runs are documented as software-only (no real DPU instructions execute) — this is a known limitation.

**Architecture:** preserves the 4-layer model; `DpuBackend` is in Layer 1 (`onnx-rt`).

## Predecessors

- `fpga-accelerator-hal-v1` (must land first — defines `ExecutionBackend`, `arch/aarch64-zynqmp`, AXI/DMA framework)

## Followers

- `fpga-custom-npu-v1` — informed by perf measurements collected after this change lands
- `fpga-manager-v1` — adds dynamic bitstream reconfig (e.g., DPU overlay swap)
