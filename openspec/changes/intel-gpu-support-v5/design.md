# Design: Intel GPU Support (intel-gpu-support-v5)

## Architecture Overview

The Intel GPU crate (`smallaios-arch-intel-gpu`) follows the identical module
decomposition as the NVIDIA crate (`smallaios-arch-nvidia`), adapted for
Intel GPU hardware concepts.

```
arch/intel_gpu/
  src/
    lib.rs                  -- Root module, GpuError enum
    pcie.rs                 -- PCIe enumeration (vendor 0x8086)
    gpu_id.rs               -- GPU identification (Xe-LP/HPG/HPC/LPG/Xe2)
    memory.rs               -- VRAM/GTT allocator (static/dynamic split)
    dma.rs                  -- DMA/Blitter copy engine
    compute.rs              -- EU-based compute engine (SIMD8/16/32)
    gpu_init.rs             -- GPU initialization lifecycle
    spirv_kernels.rs        -- SPIR-V kernel registry (equiv. to ptx.rs)
    level_zero_provider.rs  -- Level Zero provider (equiv. to cuda_provider.rs)
```

## Intel GPU vs NVIDIA GPU: Key Differences

| Concept          | NVIDIA               | Intel                           |
|------------------|----------------------|---------------------------------|
| Compute unit     | Streaming Multiprocessor (SM) | Execution Unit (EU)    |
| Thread grouping  | Warp (32 threads)    | SIMD lane (8/16/32 wide)       |
| Shader IL        | PTX / SASS           | SPIR-V / IGC                    |
| API stack        | CUDA                 | Level Zero / oneAPI             |
| Memory           | VRAM (GDDR/HBM)     | VRAM (GDDR6) or shared (GTT)   |
| Copy engine      | CE (Copy Engine)     | BCS (Blitter Command Streamer)  |
| Firmware         | GSP / PMU            | GuC / HuC                       |

## Module Design

### pcie.rs
- Reuses `PciAddress`, `BarType`, `BaseAddressRegister`, `PciDevice` structs
  from the NVIDIA pattern
- `INTEL_VENDOR_ID = 0x8086`
- `is_intel()` and `is_display_controller()` classification methods
- Mock scan includes Arc A770, A750, Data Center Max 1550, and a non-Intel
  device for filter testing

### gpu_id.rs
- `GpuArchitecture`: XeLP, XeHPG, XeHPC, XeLPG, Xe2, Unknown
- `ComputeCapability` replaced with EU count comparison for minimum
  hardware requirements
- `GpuInfo` fields: device_id, architecture, eu_count, slices, subslices,
  vram_size_mb, max_threads_per_eu (always 8), name
- `identify_gpu()` maps Intel device IDs to GpuInfo

### memory.rs
- `GPU_PAGE_SIZE = 4096` (Intel uses 4 KiB pages, vs NVIDIA's 64 KiB)
- Same `VramAllocator` pattern with 70/30 static/dynamic split
- `MemoryRegion::Static` for weights, `MemoryRegion::Dynamic` for workspace

### dma.rs
- Identical state machine: Pending -> InProgress -> Completed | Failed
- Same `DmaEngine` API: submit, start, complete, fail, cancel
- Constants adapted for Intel BCS capabilities

### compute.rs
- Intel EU model: 8 hardware threads per EU, each thread runs SIMD8/16/32
- `SimdWidth`: Simd8, Simd16, Simd32
- Max workgroup size: 1024 threads (same as NVIDIA block limit)
- Max shared local memory (SLM): 64 KiB per workgroup
- `LaunchConfig` uses `grid` (workgroup count) and `workgroup` (threads per
  workgroup) instead of NVIDIA's grid/block terminology

### gpu_init.rs
- Same state machine: Uninitialized -> BarsMapped -> EnginesReady -> Running
- `GpuRegisters`: MMIO BAR (registers) + Aperture BAR (VRAM/GTT)
- Comments reference Intel GuC (graphics microcontroller) and HuC (HEVC
  micro controller) firmware loading
- Power states: FullPower, LowPower, Suspended

### spirv_kernels.rs (equiv. to ptx.rs)
- `SpirvKernelType`: same operator families as PtxKernelType
- `DataPrecision`: F16, F32, BF16 (Intel Xe-HPG+ has native BF16)
- `SpirvKernel`: name, kernel_type, precision, shared_local_memory,
  min_xe_version (XeLP=1, XeHPG=2, XeHPC=3, XeLPG=4, Xe2=5)
- `SpirvRegistry`: register_defaults with 11 standard kernels

### level_zero_provider.rs (equiv. to cuda_provider.rs)
- `LevelZeroProvider` owns: VramAllocator, DmaEngine, ComputeEngine,
  SpirvRegistry
- Same API surface: load_weights, allocate_workspace, map_operator,
  launch_kernel, synchronize
- Operator mapping identical to CUDA provider

## Feature Flags

```toml
[features]
default = []
xe_lp  = []   # Xe-LP (Arc integrated, DG1)
xe_hpg = []   # Xe-HPG (Arc A770/A750/A580/A380)
xe_hpc = []   # Xe-HPC (Data Center GPU Max / Ponte Vecchio)
xe_lpg = []   # Xe-LPG (Meteor Lake integrated)
xe2    = []   # Xe2 (Battlemage)
```

## Testing Strategy

Each module targets 15-25 unit tests covering:
- Happy path for all public APIs
- Error conditions (invalid state, out of memory, queue full)
- Boundary conditions (max capacity, zero-size, exact limits)
- State machine transitions (valid and invalid)
- Mock hardware data for PCIe enumeration and GPU identification
