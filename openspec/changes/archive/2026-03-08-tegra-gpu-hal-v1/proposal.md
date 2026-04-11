## Why

SmallAIOS targets AI inference workloads, and the Jetson Nano's integrated GM20B GPU (Maxwell, CC 5.3, 128 CUDA cores) is the first real hardware capable of GPU-accelerated ONNX inference. The existing `arch/nvidia` crate assumes discrete PCIe-attached GPUs: it uses x86 I/O port PCI scanning, PCIe BAR discovery, and MSI-X interrupts. None of this applies to the Tegra X1 SoC, where the GPU is memory-mapped at fixed addresses (`0x5700_0000` BAR0, `0x5800_0000` BAR1), clocked through the SoC's Clock and Reset controller, and interrupted via GICv2 SPIs 189/190.

Without Tegra-specific GPU init, the `CudaProvider` in `onnx-rt` remains a stub that can queue and dispatch kernel launch descriptors but never actually programs real GPU hardware. This change bridges that gap: it adds a Tegra GPU HAL module alongside the existing PCIe module, implements the full init sequence (power, clocks, GPCPLL PLL, interrupts, Falcon firmware loading, engine init), and wires it into the existing `CudaProvider` so ONNX operators can be dispatched to the real GR engine.

Ref: https://github.com/SmallAIOS/SmallAIOS/issues/20

## What Changes

- **Tegra GPU platform init** in `arch/nvidia`: power domain enable via PMC, clock/reset via CAR, GPCPLL configuration (76.8-921.6 MHz), GICv2 SPI 189/190 interrupt routing
- **Falcon microcontroller firmware loading**: DMA-based firmware upload to FECS/GPCCS Falcon engines, ACR (Application Context for Reclocking) secure boot handshake
- **Firmware packaging**: redistributable NVIDIA firmware blobs (~165 KB), embedded in binary or loaded from storage, with NVIDIA license compliance
- **GPU engine initialization**: GR (graphics/compute) engine context setup, FIFO channel allocation with PBDMA, GMMU page table configuration for GPU virtual address space
- **CudaProvider integration**: Tegra-specific `CudaProvider` constructor that uses MMIO registers instead of PCIe BARs, wired into the ONNX runtime's `cuda` feature
- **Optional performance features**: SMMU/IOMMU integration, DVFS (dynamic voltage/frequency scaling) for power efficiency, GPU power gating for idle periods

## Capabilities

### New Capabilities
- `tegra-gpu-platform`: Tegra T210 GPU power, clock, reset, and GPCPLL PLL initialization
- `tegra-gpu-firmware`: Falcon microcontroller firmware loading and ACR secure boot for FECS/GPCCS
- `tegra-gpu-engines`: GR engine, FIFO/PBDMA channel, and GMMU page table initialization
- `tegra-gpu-integration`: CudaProvider wiring for Tegra MMIO-based GPU, ONNX runtime connection
- `tegra-gpu-licensing`: Licensing strategy for MIT-sourced register definitions and NVIDIA firmware blobs

### Modified Capabilities
- `arch/nvidia` crate: add `tegra` feature flag alongside existing PCIe-oriented architecture
- `onnx-rt` CUDA provider: support both PCIe and Tegra GPU initialization paths

## Impact

- **`arch/nvidia`**: New `tegra/` module tree (~1500-2000 lines) alongside existing `pcie.rs`; new `tegra` feature flag; new error variants in `GpuError`
- **`arch/aarch64`**: Add GPU BAR base addresses and IRQs to `platform.rs` under `tegra-x1` feature
- **`onnx-rt`**: Minor changes to `CudaProvider` to support Tegra init path (feature-gated)
- **Firmware**: ~165 KB of NVIDIA firmware blobs (redistributable, NVIDIA license) vendored or loaded at boot
- **Binary size**: +5-10 KB `.text` for GPU init code, +165 KB `.rodata` for firmware blobs
- **Licensing**: All SmallAIOS code remains Apache-2.0; register definitions document MIT provenance from nvgpu; firmware blobs under NVIDIA redistributable license
- **CI**: Tegra cross-build already in CI; new modules gated behind `tegra` + `cc_53` features
