## Context

SmallAIOS boots on the Jetson Nano (Tegra X1 SoC) with UART, display, GICv2, and PCIe Ethernet working. The integrated GM20B GPU (Maxwell architecture, compute capability 5.3, 1 GPC, 2 TPCs, 128 CUDA cores, 256 MB shared DRAM) has not been initialized. The existing `arch/nvidia` crate provides a PCIe-oriented GPU HAL with compute engine, DMA engine, VRAM allocator, PTX kernel registry, and `CudaProvider` — all currently stubs that track state but never touch hardware registers.

The GM20B is a SoC-integrated GPU: no PCIe enumeration is needed. Instead, it sits at fixed MMIO addresses (BAR0: `0x5700_0000`, 16 MB; BAR1: `0x5800_0000`, 16 MB), is powered through the PMC power partition controller, clocked through the CAR (Clock and Reset) controller, and generates interrupts via GICv2 SPIs 189 (stall) and 190 (nonstall).

The GM20B requires a multi-stage init sequence: power on the GPU partition, enable clocks, configure the GPCPLL (GPU PLL), load firmware into the Falcon microcontrollers (FECS and GPCCS), initialize the GR engine and FIFO channels, and set up GMMU page tables. Only then can compute kernels be dispatched.

## Goals / Non-Goals

**Goals:**
- Initialize the GM20B GPU on Tegra X1 to the point where compute kernels can be dispatched (phases A-D)
- Maintain clean licensing: Apache-2.0 for all SmallAIOS code, MIT attribution for nvgpu-derived register definitions, NVIDIA license for firmware blobs
- Extend the existing `arch/nvidia` crate with a `tegra` module tree alongside the existing `pcie` module
- Wire the initialized GPU into `CudaProvider` so the ONNX runtime can use it
- Add DVFS and power gating for power efficiency (phase E, optional)

**Non-Goals:**
- Display/graphics rendering — the GPU is used for compute only; display is handled by the DC/SOR subsystem (tegra-display-v8)
- Multi-GPU support — the Jetson Nano has exactly one integrated GPU
- PCIe GPU support on Tegra — no discrete GPU slot on the Nano carrier board
- CUDA C/C++ compiler or runtime — we dispatch pre-compiled PTX/SASS kernels
- Full NVIDIA driver compatibility — this is a clean-room HAL, not an NVIDIA Linux driver port
- Any reference to GPL v2 Linux kernel code — only MIT-licensed nvgpu and public documentation

## Decisions

### 1. Licensing strategy: three-tier separation

**Decision:** Maintain strict license separation across three tiers:

| Tier | License | Content | Location |
|------|---------|---------|----------|
| SmallAIOS code | Apache-2.0 | All driver logic, init sequences, abstractions | `arch/nvidia/src/tegra/*.rs` |
| Register definitions | Apache-2.0 (MIT provenance documented) | Register addresses, bit fields, offsets | `arch/nvidia/src/tegra/regs.rs` with provenance comment |
| Firmware blobs | NVIDIA redistributable | FECS/GPCCS ucode (~165 KB) | `arch/nvidia/firmware/` with `LICENSE-NVIDIA` |

**Why this approach?** Register addresses and bit field definitions are factual/functional information (not copyrightable expression), but we document their provenance from the MIT-licensed nvgpu project for transparency. The SmallAIOS implementation code (init sequences, state machines, error handling) is entirely original Apache-2.0 work. Firmware blobs are binary artifacts redistributable under NVIDIA's license.

**What we avoid:** Any reference to `os/linux/` files in the nvgpu tree, which are GPL v2. We only reference the MIT-licensed `drivers/gpu/nvgpu/` source and public NVIDIA documentation (TRM, open-gpu-doc).

### 2. Module structure: `tegra/` subtree in `arch/nvidia`

**Decision:** Add a `tegra` feature flag and `src/tegra/` module tree to the existing `arch/nvidia` crate:

```
arch/nvidia/src/
  tegra/
    mod.rs          -- TegraGpu top-level, init orchestration
    regs.rs         -- Register definitions with MIT provenance
    power.rs        -- PMC power partition control
    clock.rs        -- CAR clock/reset, GPCPLL configuration
    falcon.rs       -- Falcon microcontroller, firmware loading
    gr.rs           -- GR engine init (FECS/GPCCS, context setup)
    fifo.rs         -- FIFO channel allocation, PBDMA
    gmmu.rs         -- GPU MMU page table setup
  pcie.rs           -- Existing PCIe module (unchanged)
  compute.rs        -- Existing compute engine (unchanged)
  ...
```

**Why not a separate crate?** The GM20B shares the same Maxwell architecture as discrete GPUs. The existing `GpuError`, `ComputeEngine`, `DmaEngine`, `VramAllocator`, `PtxRegistry`, and `CudaProvider` types are all reusable. A module within the same crate avoids duplicating these abstractions and lets `CudaProvider` use either PCIe or Tegra init paths based on compile-time features.

### 3. GPCPLL clock configuration

**Decision:** Implement a 12-step frequency table for the GPCPLL, configurable at init time:

| Step | Frequency (MHz) | M | N | PL |
|------|-----------------|---|---|-----|
| 0 | 76.8 | 1 | 1 | 0 |
| 1 | 153.6 | 1 | 2 | 0 |
| 2 | 230.4 | 1 | 3 | 0 |
| 3 | 307.2 | 1 | 4 | 0 |
| 4 | 384.0 | 1 | 5 | 0 |
| 5 | 460.8 | 1 | 6 | 0 |
| 6 | 537.6 | 1 | 7 | 0 |
| 7 | 614.4 | 1 | 8 | 0 |
| 8 | 691.2 | 1 | 9 | 0 |
| 9 | 768.0 | 1 | 10 | 0 |
| 10 | 844.8 | 1 | 11 | 0 |
| 11 | 921.6 | 1 | 12 | 0 |

**Reference oscillator:** 38.4 MHz (Tegra X1 standard). GPCPLL formula: `f = ref_clk * N / (M * 2^PL)`.

The default boot frequency is step 7 (614.4 MHz), a balance between performance and thermal margin. DVFS (phase E) can adjust at runtime.

### 4. Firmware loading via Falcon DMA

**Decision:** Load FECS and GPCCS firmware through the Falcon microcontroller's DMA interface:

1. Write firmware image to a physically contiguous buffer in DRAM
2. Program Falcon DMACTL, DMATRFBASE, DMATRFMOFFS, DMATRFFBOFFS registers
3. Trigger DMA transfer (external-to-IMEM, then external-to-DMEM)
4. Boot Falcon by writing to BOOTVEC and CPUCTL
5. Poll for completion via FALCON_IDLESTATE

**Firmware sources:** The GM20B firmware blobs (`acr_ucode.bin`, `fecs_sig.bin`, `gpccs_sig.bin`, ~165 KB total) are redistributable under NVIDIA's license. They are vendored in `arch/nvidia/firmware/` and included via `include_bytes!()` at compile time.

**ACR (Application Context for Reclocking):** The GM20B requires ACR secure boot for the FECS and GPCCS Falcon engines. The ACR loader authenticates the firmware signatures before allowing execution. This is handled by loading the ACR ucode first, which then loads and verifies FECS/GPCCS.

### 5. GR engine and FIFO channel init

**Decision:** Initialize the GR engine following the standard Maxwell sequence:

1. Reset GR engine via PMC_ENABLE
2. Load golden context image (generated from FECS)
3. Configure GPC, TPC, and SM counts (1 GPC, 2 TPCs, 4 SMs for GM20B)
4. Set up ZCULL and attribute circular buffers
5. Initialize FIFO with one PBDMA channel for compute
6. Configure GMMU page tables (identity-mapped initially, 4 KB pages)

**FIFO approach:** Single channel with one PBDMA engine. The GM20B has 1 PBDMA unit. We allocate a single GPU channel for compute workloads, using a ring-buffer (pushbuffer) of GPU commands. This is sufficient for sequential ONNX operator dispatch.

### 6. CudaProvider integration

**Decision:** Add a `CudaProvider::new_tegra()` constructor that initializes via MMIO registers instead of PCIe BAR mapping:

```rust
#[cfg(feature = "tegra")]
pub fn new_tegra() -> Result<Self, GpuError> {
    // 1. Power on GPU partition
    // 2. Enable clocks, configure GPCPLL
    // 3. Load firmware, init engines
    // 4. Create VramAllocator with shared DRAM region
    // 5. Return Ready provider
}
```

The ONNX runtime's `cuda` feature conditionally uses `new_tegra()` when the `tegra` feature is also active. The `VramAllocator` operates on a reserved region of shared DRAM (not dedicated VRAM), since the GM20B uses unified memory.

### 7. GMMU page table strategy

**Decision:** Start with identity mapping (GPU virtual = physical) using 4 KB small pages. The GM20B GMMU supports two-level page tables (PDE -> PTE). We allocate a single PDB (Page Directory Base) and map the DRAM region that the VramAllocator manages.

**Why identity mapping?** For the initial implementation, identity mapping avoids the complexity of a GPU virtual address allocator. The Jetson Nano has 4 GB of DRAM, and the GPU can address all of it. The VramAllocator already tracks physical offsets, so identity mapping makes GPU addresses == physical addresses, simplifying DMA and compute dispatch.

SMMU integration (phase E) can later add proper isolation between GPU and CPU address spaces.

### 8. Phase E: DVFS and power gating (optional)

**Decision:** DVFS and power gating are optional phase E features, not required for basic compute:

- **DVFS:** Adjust GPCPLL frequency at runtime based on GPU load. Uses the 12-step frequency table. Triggered by the scheduler when GPU utilization changes.
- **Power gating:** Gate the GPU power partition via PMC when no compute work is pending. Wake-up latency is ~100 us. Useful for inference workloads with idle periods between requests.
- **SMMU:** Wire the GPU into the ARM SMMU for address space isolation. Requires SMMU driver (not yet implemented in SmallAIOS).

These are performance optimizations that can be deferred until basic compute is working end-to-end.

## Risks / Trade-offs

**[Firmware blob licensing]** The GM20B firmware blobs are redistributable under NVIDIA's license, but must be distributed alongside a license notice. Mitigation: vendor blobs in `arch/nvidia/firmware/` with a `LICENSE-NVIDIA` file.

**[Falcon init complexity]** The Falcon microcontroller init and ACR secure boot sequence is complex (~500 lines of register programming). If the firmware fails to load, the GR engine cannot be initialized. Mitigation: extensive error checking at each stage, timeout on Falcon boot (100 ms), clear error reporting.

**[Shared memory contention]** The GM20B uses shared DRAM (not dedicated VRAM). GPU DMA transfers compete with CPU memory bandwidth. Mitigation: use physically contiguous buffers for GPU work, minimize CPU access during GPU compute phases.

**[No QEMU testing]** QEMU does not emulate the GM20B GPU. All GPU init code can only be validated on real hardware. Mitigation: maximum unit test coverage for register calculations, PLL math, page table construction, firmware loading state machines; mock MMIO for integration tests; hardware validation on Jetson Nano.

**[Register accuracy]** Register definitions are derived from the MIT-licensed nvgpu source and public NVIDIA documentation. If addresses are wrong, GPU init will hang or crash. Mitigation: cross-reference multiple sources (nvgpu, TRM, open-gpu-doc), add register read-back verification where possible.

**[Thermal limits]** Running the GPU at 921.6 MHz may exceed the Nano's thermal envelope (10W power mode). Mitigation: default to 614.4 MHz; DVFS (phase E) can adjust based on temperature sensor readings.

## Open Questions

1. **Firmware blob distribution:** Should firmware blobs be vendored in the repository (~165 KB), downloaded at build time, or loaded from the SD card at boot? Vendoring is simplest and most reproducible, but adds binary artifacts to the repo.

2. **Shared DRAM region size:** How much DRAM should be reserved for GPU use? The Jetson Nano has 4 GB total. Options: 256 MB (conservative), 512 MB (balanced), 1 GB (aggressive). The ONNX model size and workspace requirements determine the minimum.

3. **Golden context image:** The GR engine requires a "golden context" image generated by FECS. This is typically captured at init time by allocating a context, running a special FECS method, and saving the result. Can this be pre-computed and vendored, or must it be generated at every boot?
