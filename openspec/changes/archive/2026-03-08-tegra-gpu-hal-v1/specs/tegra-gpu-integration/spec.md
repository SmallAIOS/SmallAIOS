## Tegra GPU Integration

### Overview

Wire the initialized Tegra GM20B GPU into the existing `CudaProvider` and ONNX runtime, so GPU-accelerated ONNX operators can be dispatched on real hardware. This bridges the Tegra-specific init code (phases A-C) with the architecture-agnostic compute engine and operator mapping.

### CudaProvider Tegra Constructor

Add a `new_tegra()` constructor to `CudaProvider` that initializes via MMIO instead of PCIe:

```rust
#[cfg(feature = "tegra")]
impl CudaProvider {
    pub fn new_tegra(dram_region_base: u64, dram_region_size: u64) -> Result<Self, GpuError> {
        // 1. TegraGpuPlatform::power_on()
        // 2. TegraGpuPlatform::enable_clocks()
        // 3. TegraGpuPlatform::configure_gpcpll(7)  // 614.4 MHz default
        // 4. FirmwareLoader::boot_all()
        // 5. GrEngine::init()
        // 6. FifoChannel::allocate() + bind_to_gr()
        // 7. GmmuPageTable::new_identity(dram_region_base, dram_region_size)
        // 8. VramAllocator::new(dram_region_size, 0.7)
        // 9. Return Ready provider
    }
}
```

### Memory Model

The GM20B uses unified memory (shared DRAM, no dedicated VRAM):

- **VramAllocator** operates on a reserved DRAM region (not separate VRAM)
- **DMA transfers** are CPU-to-GPU-visible-DRAM (may be no-ops with cache flush)
- **Static region:** 70% for model weights
- **Dynamic region:** 30% for workspace buffers

### ONNX Runtime Connection

The `onnx-rt` crate's `cuda` feature currently depends on `smallaios-arch-nvidia`. When both `cuda` and `tegra` features are active:

1. The ONNX session creates a `CudaProvider::new_tegra()` instead of `CudaProvider::new()`
2. Operator dispatch goes through the same `launch_kernel()` path
3. Kernels are submitted as GPFIFO entries to the FIFO channel
4. Synchronization polls the FIFO fence/semaphore

### Tegra-Specific GpuInfo

Add GM20B to the `gpu_id` module:

```rust
GpuInfo {
    device_id: 0x12B1,        // GM20B
    architecture: GpuArchitecture::Maxwell,
    compute_capability: ComputeCapability::new(5, 3),
    sm_count: 4,               // 1 GPC * 2 TPC * 2 SM/TPC
    vram_size_mb: 256,         // Configured DRAM region
    max_threads_per_sm: 2048,
    max_warps_per_sm: 64,
    warp_size: 32,
    max_shared_memory_per_sm: 65536,  // 64 KB
    max_registers_per_sm: 65536,
    name: "NVIDIA GM20B (Tegra X1)",
}
```

### Boot Integration

In the Tegra X1 boot sequence (`arch/aarch64` with `tegra-x1` feature), GPU init is called after display init but before the inference loop:

1. UART init (existing)
2. GICv2 init (existing)
3. PCIe + Ethernet init (existing)
4. Display init (existing, tegra-display-v8)
5. **GPU init** (this change)
6. ONNX model load + inference

### Interface

```rust
// In arch/nvidia/src/tegra/mod.rs
pub struct TegraGpu {
    platform: TegraGpuPlatform,
    firmware: FirmwareLoader,
    gr: GrEngine,
    fifo: FifoChannel,
    gmmu: GmmuPageTable,
}

impl TegraGpu {
    /// Full initialization sequence (phases A-C).
    pub fn init() -> Result<Self, GpuError>;

    /// Create a CudaProvider from an initialized TegraGpu.
    pub fn into_provider(self, dram_size: u64) -> Result<CudaProvider, GpuError>;
}
```

### Verification

- Unit tests for `GpuInfo` construction (GM20B device ID, CC 5.3, 4 SMs)
- Unit tests for `new_tegra()` state machine (power -> clocks -> firmware -> engines -> ready)
- Integration tests for operator mapping with CC 5.3 compatibility
- Integration tests for full init -> launch -> sync cycle (mock MMIO)
- Verify that existing PCIe path is unaffected when `tegra` feature is disabled
