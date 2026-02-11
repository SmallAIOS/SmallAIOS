# AMD GPU Support - Design Document

## Architecture Overview

The `arch/amd` crate follows the same modular design as `arch/nvidia`:

```
arch/amd/
  src/
    lib.rs          -- Crate root, GpuError enum, module declarations
    pcie.rs         -- PCIe enumeration (vendor 0x1002)
    gpu_id.rs       -- GPU identification (RDNA/CDNA families)
    memory.rs       -- VRAM allocator (static/dynamic regions)
    dma.rs          -- DMA/SDMA engine for host<->device transfers
    compute.rs      -- Wavefront-based compute engine
    hip_kernels.rs  -- HIP kernel definitions and registry
    rocm_provider.rs -- ROCm execution provider (ONNX operator dispatch)
    gpu_init.rs     -- GPU initialization and lifecycle management
```

## Key Design Decisions

### Wavefront Size
AMD GPUs use wavefronts instead of NVIDIA warps:
- CDNA (MI100/MI200/MI300): 64-wide wavefronts
- RDNA (RX 5000/6000/7000): 32-wide wavefronts (wave32) with wave64 compatibility

The compute engine adapts launch configurations based on detected architecture.

### Architecture Detection
GPU family detection uses PCI device ID ranges:
- RDNA 1 (Navi 10/14): 0x7310-0x73FF
- RDNA 2 (Navi 21/22/23): 0x73A0-0x73EF
- RDNA 3 (Navi 31/32/33): 0x7440-0x74FF
- CDNA 1 (MI100): 0x7380-0x739F
- CDNA 2 (MI200): 0x7400-0x743F
- CDNA 3 (MI300): 0x7500-0x75FF

### Memory Model
Same dual-region approach as NVIDIA: 70% static (weights), 30% dynamic (workspace).
Page size is 64 KiB (matching AMD's GPU page granularity).

### HIP Kernel Registry
Mirrors the PTX kernel registry but uses HIP/GCN ISA terminology.
Kernels specify minimum GFX version (e.g., gfx908 for MI100, gfx942 for MI300X).

### Execution Provider
`RocmProvider` is the AMD equivalent of `CudaProvider`, wiring together:
- VRAM allocator
- DMA engine
- Compute engine
- HIP kernel registry
