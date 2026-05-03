> **Status:** Roadmap stub. This is the third and most ambitious of the FPGA follow-up changes.
> Scope is intentionally TBD — it depends on perf data collected after `fpga-dpu-backend-v1` lands.
> Sibling stubs: `fpga-dpu-backend-v1`, `fpga-manager-v1`.

## Why

The DPU (driven via `fpga-dpu-backend-v1`) is CNN-biased: it does conv/gemm well, but transformer ops (LayerNorm, RoPE, GQA, SwiGLU, custom attention variants) fall back to ARM, and the K26's ~256K LUTs aren't huge. SmallAIOS's ONNX runtime targets transformer workloads (Gemma/Llama/Qwen kernels), so the DPU is unlikely to be the long-term answer. This change introduces a custom NPU — designed in HLS or hand-written RTL — informed by measured DPU shortfalls on our target models.

This change is the project's clean-room answer: no Vitis AI compiler dependency, no `.xmodel` format, runtime stays the source of truth. It is also philosophically aligned with SmallAIOS's "no external C deps, from-scratch" ONNX runtime stance.

## What Changes

(Concrete scope deferred — see "Open Questions" below. The shape of this change will look approximately like:)

- New `CustomNpuBackend` `ExecutionBackend` implementation
- HLS or RTL design for the accelerator (matmul tile + on-chip activation/weight buffers; possibly a tiny scratchpad for layernorm/softmax)
- Bitstream produced by Vivado (still required at offline build time — open-source toolchains do not support UltraScale+)
- Driver in SmallAIOS using the AXI/DMA framework from `fpga-accelerator-hal-v1`
- Documentation: micro-architecture, op coverage, perf comparison vs DPU baseline

Out of scope:
- Replacing the DPU backend (both can coexist — different boards / different bitstreams)
- Open-source bitstream generation (UltraScale+ has no open toolchain; we use Vivado)
- Versal AI Edge support (would be a separate `versal-aiedge-board-v1` + matching backend change)

## Capabilities

### New Capabilities

- `custom-npu-backend`: SmallAIOS-native FPGA accelerator backend. Detailed scope deferred until DPU perf data is in hand.
- `custom-npu-rtl` (tentative): the HLS/RTL design itself, versioned as a hardware artifact.

### Modified Capabilities

- *(None expected — the HAL is the contract; this change adds an implementation behind it.)*

## Impact

**Code:** new `onnx-rt` module + a hardware-design artifact tree (`hw/custom-npu/`).

**Build:** offline Vivado/Vitis HLS dependency for bitstream generation; SmallAIOS runtime stays standalone.

**Dependencies:** no new SmallAIOS runtime deps. New offline tool dep: Vivado (Linux x86, pinned version).

**Hardware:** KV260/KR260 (K26 SOM) initial target. May extend to Versal AI Edge later.

**Architecture:** preserves 4-layer model.

## Open Questions / Pre-conditions

This change cannot start until the following are resolved:

1. **DPU perf data exists** — `fpga-dpu-backend-v1` must have produced honest measurements on representative target models, identifying which ops/shapes/dtype combinations the DPU does badly.
2. **NPU specification** — derived from item 1: which ops to accelerate, what tile sizes, what numeric formats (INT8 only? INT8+BF16? mixed?), what scratchpad sizes.
3. **Resource budget** — how many LUTs/BRAMs/DSPs of the K26's PL we can spend, after reserving headroom for AXI-DMA, debug, etc.
4. **Verification strategy** — golden-vs-RTL co-sim plan; HLS test bench; bit-accuracy targets.
5. **Whether we co-exist with the DPU** (one bitstream with both, or separate bitstreams swapped via `fpga-manager-v1`).

## Predecessors

- `fpga-accelerator-hal-v1` (must land first — provides HAL + board + AXI/DMA framework)
- `fpga-dpu-backend-v1` (must land first — provides perf baseline that determines NPU scope)

## Followers

- Possibly `versal-aiedge-board-v1` if the K26 fabric is too small for the workloads we want to accelerate.
