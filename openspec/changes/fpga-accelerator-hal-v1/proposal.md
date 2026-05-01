## Why

SmallAIOS targets AI inference workloads but currently has no path to FPGA acceleration. AMD/Xilinx Zynq UltraScale+ MPSoC boards (KV260, KR260) are an attractive prototype target — quad Cortex-A53 + ~256K LUTs of FPGA fabric — and we want to run SmallAIOS (not Linux) on the A53 cores while offloading ONNX matmul/conv to the PL. Doing this without a clean accelerator HAL would couple `onnx-rt` to AMD's `.xmodel`/Vitis AI semantics and block a future custom NPU. This change defines the boundary first, on QEMU, before any hardware-specific backend lands.

## What Changes

- Introduce a generic accelerator HAL in `onnx-rt` (a `Backend` trait + runtime dispatch) designed without DPU or `.xmodel` knowledge — backends are pluggable, ARM-only fallback is the default
- Add a QEMU stub backend (a fake AXI-mapped accelerator device with deterministic latency) so the HAL is exercisable end-to-end on Mac/Linux dev hosts with no FPGA hardware
- Add a new `arch/aarch64-zynqmp` board crate for Zynq UltraScale+: Cadence UART (not PL011), GIC-400, generic timer, DDR memory map, EL1 boot via AMD's FSBL+ATF chain (SmallAIOS as the EL1 payload in `BOOT.BIN`)
- Add a reusable AXI master + AXI-DMA driver framework with PS-PL cache-coherency handling (HPC vs HP port semantics on UltraScale+)
- Static bitstream loading via FSBL only — dynamic FPGA Manager / PMU IPI is deferred
- Update `just` recipes and CI to build the new aarch64 board target; QEMU runner gains the stub accelerator device

Not included (future changes):
- `fpga-dpu-backend-v1` — driving AMD's stock DPU (`.xmodel` parser, DPU instructions, VART subgraph runtime)
- `fpga-custom-npu-v1` — HLS-designed matmul/conv accelerator informed by DPU perf measurements
- Dynamic bitstream reconfiguration, Cortex-R5F lockstep / safety-island use

## Capabilities

### New Capabilities

- `accelerator-hal`: Generic ONNX-runtime backend abstraction. Defines the `Backend` trait, op-dispatch contract, tensor/buffer ownership rules, fallback-to-ARM semantics, and the QEMU stub backend used as a reference implementation.
- `zynqmp-board`: Zynq UltraScale+ MPSoC (K26 SOM) board support for `arch/aarch64-zynqmp`. Covers boot via FSBL+ATF, Cadence UART, GIC-400, generic timer, DDR memory map, A53-only execution.
- `axi-dma-framework`: Reusable AXI master + AXI-DMA driver framework. Covers register access, scatter-gather DMA descriptors, IRQ-driven completion, and PS-PL cache-coherency handling for UltraScale+ HPC/HP ports.

### Modified Capabilities

- `onnx-runtime`: Op dispatch gains a backend-selection pre-step. The default ARM-only path is preserved as a backend; new requirements describe the HAL contract that backends must satisfy and how unsupported ops fall back to ARM.

## Impact

**Code:**
- `onnx-rt` (Layer 1): new `backend` module, dispatch refactor, QEMU stub backend behind a feature flag
- New `arch/aarch64-zynqmp` crate (Layer 2)
- New `axi` driver module (Layer 2; co-located with `arch/aarch64-zynqmp` or in a sibling crate — design.md decides)
- `kernel`, `bench`, `container`: no API changes; new build target wired through

**Build / CI:**
- New target triple in `.cargo/config.toml` if needed (still `aarch64-unknown-none`, but with a `--features zynqmp` selector)
- `just build-kernel-arm-zynqmp` recipe
- `just run-arm-zynqmp` QEMU recipe with the stub accelerator device
- CI gate for the new build target

**Dependencies:**
- No new runtime crate dependencies expected (kept `#![no_std]`)
- Vitis / `bootgen` becomes an *optional* offline tool for producing real `BOOT.BIN` — never a runtime or test dependency

**Architecture:**
- 4-layer acyclic dependency model preserved: HAL changes are within Layer 1; new board crate is Layer 2
- DSM should report no new layering violations after this change
