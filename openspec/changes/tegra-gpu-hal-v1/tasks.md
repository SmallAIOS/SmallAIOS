## Phase A: Platform Init (Power, Clocks, GPCPLL, Interrupts)

- [ ] A1: Add `tegra` feature flag to `arch/nvidia/Cargo.toml`; add `arch/aarch64` as optional dependency under `tegra` feature
  Feature enables the Tegra GPU module tree. The aarch64 dep provides platform constants (BAR addresses, IRQs, CAR/PMC bases).

- [ ] A2: Create `arch/nvidia/src/tegra/mod.rs` with `TegraGpu` top-level struct and init orchestration skeleton
  Exports submodules (regs, power, clock, falcon, gr, fifo, gmmu). Defines `TegraGpu::init()` that calls each phase in sequence. Gate entire module behind `#[cfg(feature = "tegra")]`.

- [ ] A3: Create `arch/nvidia/src/tegra/regs.rs` with GM20B register definitions and MIT provenance header
  GPU BAR0/BAR1 offsets, PMC registers, CAR registers, GPCPLL registers, Falcon registers, GR registers, FIFO registers, GMMU registers. Include provenance comment documenting nvgpu MIT source.

- [ ] A4: Add GPU BAR0/BAR1 base addresses and IRQ numbers to `arch/aarch64/src/platform.rs` under `tegra-x1` feature
  `GPU_BAR0_BASE: 0x5700_0000`, `GPU_BAR1_BASE: 0x5800_0000`, `GPU_IRQ_STALL: 189`, `GPU_IRQ_NONSTALL: 190`.

- [ ] A5: Create `arch/nvidia/src/tegra/power.rs` with PMC GPU power partition control
  `power_on()`: read PMC_PWRGATE_STATUS, toggle partition 14, poll for power-up (10 ms timeout), remove clamps. `power_off()`: reverse sequence. Unit tests for bit manipulation and state tracking.

- [ ] A6: Create `arch/nvidia/src/tegra/clock.rs` with CAR clock/reset control and GPCPLL configuration
  `enable_clocks()`: enable GPU clock source, deassert reset. `configure_gpcpll(step)`: 12-step frequency table (76.8-921.6 MHz), bypass/program/lock/switch sequence. Unit tests for PLL coefficient calculations and frequency validation.

- [ ] A7: Add `TegraGpuPlatform` struct combining power and clock init with interrupt setup
  Orchestrates: power_on -> enable_clocks -> configure_gpcpll -> enable_interrupts. Tracks state (powered, clocks_enabled, gpcpll_freq). Unit tests for init sequence state machine.

- [ ] A8: Wire GICv2 SPI 189/190 interrupt enable into Tegra GPU init
  Call `gicv2::enable_irq(189)` and `gicv2::enable_irq(190)`. Configure GPU PMC interrupt tree (`NV_PMC_INTR_EN_0`). Unit tests for interrupt enable register calculations.

## Phase B: Firmware Loading (Falcon, ACR)

- [ ] B1: Create `arch/nvidia/src/tegra/falcon.rs` with Falcon microcontroller DMA loader
  `FalconEngine` struct with base offset, IMEM/DMEM load methods. DMA transfer sequence: halt, set base, block-by-block IMEM load (256B chunks), DMEM load, set bootvec, start CPU, poll idle. Unit tests for DMA descriptor construction and state machine.

- [ ] B2: Create firmware blob directory structure and NVIDIA license file
  Create `arch/nvidia/firmware/gm20b/` directory. Add `LICENSE-NVIDIA` with firmware redistribution license text. Add placeholder `.gitkeep` files (actual blobs are hardware-deferred).

- [ ] B3: Create `LICENSES/MIT-nvgpu.txt` with MIT license text from nvgpu project
  Full MIT license text as referenced by provenance comments in `regs.rs`.

- [ ] B4: Implement ACR secure boot sequence in `falcon.rs`
  Load ACR ucode into PMU Falcon, boot it, wait for ACR to authenticate and load FECS/GPCCS via mailbox protocol. Unit tests for ACR mailbox register interactions.

- [ ] B5: Implement `FirmwareLoader` struct that orchestrates ACR -> FECS -> GPCCS loading
  `boot_all()` method: load ACR, boot ACR, wait for FECS/GPCCS ready. Timeout handling (100 ms per Falcon). Error reporting for each stage. Unit tests for full load sequence.

- [ ] B6: Add `include_bytes!` stubs for firmware embedding with feature-gated compile-time inclusion
  `#[cfg(feature = "tegra")] static FECS_FW: &[u8] = include_bytes!("../../firmware/gm20b/fecs_sig.bin");` (behind a sub-feature or with fallback empty arrays for testing without blobs).

## Phase C: Engine Init (GR, FIFO, GMMU)

- [ ] C1: Create `arch/nvidia/src/tegra/gr.rs` with GR engine initialization
  Reset GR via PMC_ENABLE, query topology (1 GPC, 2 TPC, 4 SM), configure ZCULL and attrib CB, generate golden context image via FECS method. Unit tests for topology parsing and buffer size calculations.

- [ ] C2: Create `arch/nvidia/src/tegra/fifo.rs` with FIFO channel allocation and PBDMA setup
  `FifoChannel::allocate()`: allocate instance block, configure PBDMA registers, set pushbuffer and GPFIFO entry pointers. `bind_to_gr()`: bind channel to GR engine. `submit_work()`: write GPFIFO entry. Unit tests for instance block layout and GPFIFO entry format.

- [ ] C3: Create `arch/nvidia/src/tegra/gmmu.rs` with GPU MMU page table setup
  `GmmuPageTable::new_identity()`: allocate PDB, create small page tables for identity mapping, program PDB base registers. `invalidate_tlb()`: write TLB invalidate register. Unit tests for PDE/PTE entry construction and address mapping correctness.

- [ ] C4: Add `GpuError` variants for Tegra-specific failures
  `FirmwareLoadFailed`, `FalconTimeout`, `GpcpllLockFailed`, `PowerPartitionTimeout`, `FifoError`, `GmmuError`. Add to existing `GpuError` enum in `lib.rs`.

- [ ] C5: Write integration tests for full Phase A-C init sequence with mock MMIO
  Test `TegraGpu::init()` end-to-end: power -> clocks -> GPCPLL -> firmware -> GR -> FIFO -> GMMU. Verify state transitions and error handling at each stage.

## Phase D: Integration (CudaProvider, ONNX Runtime)

- [ ] D1: Add GM20B to `gpu_id.rs` with device ID `0x12B1`, CC 5.3, Maxwell architecture
  `identify_gpu(0x12B1)` returns `GpuInfo` for GM20B (4 SMs, 256 MB, CC 5.3). Unit tests for identification and CC 5.3 compatibility with PTX registry.

- [ ] D2: Add `CudaProvider::new_tegra()` constructor in `cuda_provider.rs` behind `tegra` feature
  Calls `TegraGpu::init()`, creates `VramAllocator` with shared DRAM region, registers PTX kernels, returns Ready provider. Feature-gated behind `#[cfg(feature = "tegra")]`.

- [ ] D3: Update `onnx-rt/Cargo.toml` to support `tegra` feature alongside `cuda`
  Add `tegra` feature that enables both `cuda` and `smallaios-arch-nvidia/tegra`. When both are active, ONNX session uses `new_tegra()`.

- [ ] D4: Wire Tegra GPU init into `arch/aarch64` boot sequence behind `tegra-x1` feature
  After display init, before inference loop: call `TegraGpu::init()` -> `into_provider()`. Print GPU info to UART.

- [ ] D5: Write integration tests for CudaProvider::new_tegra() -> launch_kernel -> sync cycle
  Test full provider lifecycle with mock GPU: init, load weights, allocate workspace, launch MatMul/Relu kernels, synchronize, verify completion. Ensure CC 5.3 PTX kernels are selected.

- [ ] D6: Verify existing PCIe CudaProvider path is unaffected when `tegra` feature is disabled
  Run existing `arch/nvidia` test suite without `tegra` feature. All existing tests must pass unchanged.

## Phase E: Performance (Optional, Hardware-Deferred)

- [ ] E1: Implement DVFS (dynamic voltage/frequency scaling) for GPU clock adjustment
  Runtime GPCPLL frequency changes using the 12-step table. `set_frequency(step)` method on `TegraGpuPlatform`. Requires temperature sensor input (hardware-deferred).

- [ ] E2: Implement GPU power gating via PMC for idle periods
  Gate GPU partition when no work is pending, wake on submit. ~100 us wake latency. Track power state transitions.

- [ ] E3: Implement SMMU integration for GPU address space isolation
  Wire GPU into ARM SMMU for proper virtual address isolation between GPU and CPU. Requires SMMU driver (not yet implemented in SmallAIOS).

## Licensing and Documentation

- [ ] L1: Verify all new `.rs` files have `SPDX-License-Identifier: Apache-2.0` header
  Scan all files in `arch/nvidia/src/tegra/`. Verify `regs.rs` has MIT provenance comment block.

- [ ] L2: Add Tegra GPU section to crate-level documentation in `arch/nvidia/src/lib.rs`
  Document the `tegra` feature flag, module structure, and init sequence in the crate doc comment. Reference licensing strategy.

- [ ] L3: Run `make test`, `make clippy`, and `make fmt-check` with Tegra features enabled
  Verify zero warnings, all tests pass, formatting clean. Run with `--features tegra,cc_53`.
