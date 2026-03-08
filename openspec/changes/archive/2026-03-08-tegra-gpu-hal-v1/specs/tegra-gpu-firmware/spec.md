## Tegra GPU Firmware Loading

### Overview

The GM20B GPU requires firmware to be loaded into two Falcon microcontrollers — FECS (Front End Context Switch) and GPCCS (GPC Context Switch) — before the GR engine can operate. The firmware is loaded via DMA through the Falcon's built-in DMA controller, and authenticated via ACR (Application Context for Reclocking) secure boot.

### Firmware Blobs

| Blob | Size (approx) | Purpose |
|------|---------------|---------|
| `acr_ucode.bin` | ~48 KB | ACR loader: authenticates and loads FECS/GPCCS |
| `fecs_sig.bin` | ~56 KB | FECS microcode + signature |
| `gpccs_sig.bin` | ~56 KB | GPCCS microcode + signature |
| **Total** | **~165 KB** | |

Blobs are embedded via `include_bytes!()` at compile time from `arch/nvidia/firmware/`.

### Falcon DMA Loading Sequence

For each Falcon engine (PMU/ACR first, then FECS, then GPCCS):

1. Halt Falcon: write `FALCON_CPUCTL = HALT`
2. Set DMA transfer base: `FALCON_DMATRFBASE = phys_addr >> 8`
3. For each 256-byte block of IMEM:
   - Set `FALCON_DMATRFMOFFS = imem_offset`
   - Set `FALCON_DMATRFFBOFFS = dram_offset`
   - Write `FALCON_DMATRFCMD = IMEM | SIZE_256B`
   - Poll `FALCON_DMATRFCMD` until BUSY bit clears
4. Repeat for DMEM sections
5. Set boot vector: `FALCON_BOOTVEC = entry_point`
6. Start execution: `FALCON_CPUCTL = STARTCPU`
7. Poll `FALCON_IDLESTATE` for idle (timeout: 100 ms)

### ACR Secure Boot

The ACR loader runs first and authenticates FECS/GPCCS:

1. Load ACR ucode into PMU Falcon via DMA
2. Boot ACR loader
3. ACR reads FECS/GPCCS signatures from DRAM
4. ACR verifies signatures against fuse-burned keys
5. ACR loads authenticated FECS/GPCCS into their respective Falcons
6. ACR signals completion via mailbox register

### Firmware Packaging

```
arch/nvidia/firmware/
  gm20b/
    acr_ucode.bin
    fecs_sig.bin
    gpccs_sig.bin
  LICENSE-NVIDIA          # NVIDIA redistributable firmware license
```

### Interface

```rust
pub struct FalconEngine {
    base: usize,     // Falcon MMIO base within BAR0
    loaded: bool,
}

pub struct FirmwareLoader {
    fecs: FalconEngine,
    gpccs: FalconEngine,
}

impl FirmwareLoader {
    pub fn new(bar0_base: usize) -> Self;
    pub fn load_acr() -> Result<(), GpuError>;
    pub fn load_fecs() -> Result<(), GpuError>;
    pub fn load_gpccs() -> Result<(), GpuError>;
    pub fn boot_all() -> Result<(), GpuError>;
}
```

### Verification

- Unit tests for DMA transfer descriptor construction
- Unit tests for Falcon register offset calculations
- Unit tests for firmware loading state machine (halt, load, boot, poll)
- Unit tests for ACR boot sequence with mock Falcon registers
- Verify firmware blob sizes match expected headers
