# Intel GPU Support - Design

## Architecture

The `arch/intel_gpu` crate follows the same modular pattern as `arch/nvidia`:

```
arch/intel_gpu/
  src/
    lib.rs           - Crate root, GpuError enum, module declarations
    pcie.rs          - PCIe enumeration (vendor 0x8086), BAR mapping
    gpu_id.rs        - GPU identification: Xe-LP/HPG/HPC, EU counts, memory
    gpu_init.rs      - GPU lifecycle: init, BAR mapping, engine setup, suspend/resume
    memory.rs        - Local memory allocator (static weights / dynamic workspace)
    dma.rs           - DMA/copy engine for host<->device transfers
    compute.rs       - EU-based compute dispatch, workgroup config, synchronization
    spirv.rs         - SPIR-V kernel registry (analogous to PTX registry)
    level_zero_provider.rs - Execution provider wiring ONNX ops to GPU kernels
```

## Key Differences from NVIDIA

| Aspect | NVIDIA | Intel |
|--------|--------|-------|
| Compute units | Streaming Multiprocessors (SM) | Execution Units (EU), grouped into Subslices/Dual Subslices (DSS) |
| Thread hierarchy | Warps (32 threads) | SIMD lanes (8/16/32 wide, varies by arch) |
| Kernel format | PTX/SASS | SPIR-V |
| API model | CUDA | Level Zero (oneAPI) |
| Memory | VRAM (dedicated) | Local memory (dedicated or shared with system) |
| Vendor ID | 0x10DE | 0x8086 |

## GPU Families

### Xe-LP (Gen12, Integrated)
- Tiger Lake, Alder Lake, Raptor Lake iGPUs
- 32-96 EUs, shared system memory
- SIMD width: 8

### Xe-HPG (Alchemist/Arc, Discrete)
- Arc A770, A750, A580, A380
- 128-512 EUs, dedicated GDDR6
- SIMD width: 16
- Ray tracing units (not used for compute)

### Xe-HPC (Ponte Vecchio, Data Center)
- Intel Data Center GPU Max series
- 512+ EUs per tile, HBM2e memory
- SIMD width: 16/32
- XMX (matrix) engines for AI

## Memory Model
- Static region: model weights (long-lived, 70% of local memory)
- Dynamic region: intermediate tensors (short-lived, 30%)
- 64 KiB page granularity (matching Intel GPU page size)

## Execution Model
- Workgroups dispatched to Subslices
- Each EU executes SIMD threads
- Barriers synchronize within workgroups
- Level Zero command lists for kernel submission
