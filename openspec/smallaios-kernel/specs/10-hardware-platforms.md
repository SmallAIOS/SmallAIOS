# Spec 10: Target Hardware Platforms

## Overview

SmallAIOS targets a range of hardware from edge devices to data center GPU servers.
This spec documents the specific platforms, their hardware characteristics, and any
platform-specific considerations.

## Platform Matrix

| Platform | Architecture | CPU | GPU | RAM | Primary Use |
|---|---|---|---|---|---|
| NVIDIA DGX Spark | ARM64 | Grace (Neoverse V2) | Blackwell B200 | 128 GB unified | Production inference server |
| Intel Core i7 | x86-64 | i7-12xxx+ (Alder Lake+) | Optional discrete NVIDIA | 16-128 GB | Development, CPU inference |
| AMD Ryzen/EPYC | x86-64 | Zen 3+ | Optional discrete NVIDIA | 16-512 GB | Development, CPU inference |
| Snapdragon 845 | ARM64 | Kryo 385 (A75+A55) | Adreno 630 (not supported) | 4-8 GB | Edge inference (CPU only) |
| NVIDIA Jetson Orin Nano | ARM64 | Cortex-A78AE | Ampere (CC 8.7) | 4-8 GB | Edge GPU inference |
| NVIDIA Jetson Nano (orig) | ARM64 | Cortex-A57 | Maxwell (CC 5.3) | 4 GB | Edge testing (limited GPU) |
| Raspberry Pi 4/5 | ARM64 | Cortex-A72/A76 | None (VideoCore, unsupported) | 2-8 GB | Edge CPU inference, testing |
| QEMU Virtual Machine | x86-64/ARM64 | Emulated | None | Configurable | Development, CI testing |
| Cloud VM (AWS, GCP) | x86-64/ARM64 | Various | Optional (T4, A10G, A100) | Varies | Cloud inference |

## Tier 1 Platforms (Primary Support)

### NVIDIA DGX Spark

The DGX Spark is NVIDIA's desktop AI supercomputer:
- **CPU**: NVIDIA Grace (ARM Neoverse V2), 72 cores, SVE2
- **GPU**: NVIDIA Blackwell B200, 192 GB HBM3e
- **Memory**: 128 GB LPDDR5x (CPU) + 192 GB HBM3e (GPU), unified memory architecture
- **Network**: 10GbE, ConnectX-7
- **Storage**: NVMe SSD

SmallAIOS considerations:
- Grace CPU: Full ARM64 support with SVE2 SIMD (up to 128-bit per core, but SVE)
- Blackwell GPU: Compute capability 10.0, FP8/FP4 tensor cores, latest PTX features
- Unified memory: Grace Hopper/Blackwell NVLink-C2C allows CPU and GPU to share
  memory without explicit copies — SmallAIOS should detect and use this
- Large model support: 192 GB GPU memory enables large language models

Platform-specific optimizations:
- SVE2 GEMM kernels optimized for Neoverse V2 pipeline
- NVLink-C2C zero-copy tensor sharing (skip DMA, use unified addressing)
- Grace CPU's large LLC (last-level cache) for CPU inference
- Blackwell FP8 tensor cores for quantized inference

### Intel Core i7 / AMD Ryzen (x86-64)

Standard x86-64 desktop/workstation:
- **CPU**: Intel 12th+ gen (P-cores: Golden Cove/Raptor Cove, E-cores) or AMD Zen 3+
- **GPU**: Optional discrete NVIDIA (GeForce RTX 3060+ or workstation)
- **Memory**: 16-128 GB DDR4/DDR5
- **Features**: AVX2 (baseline), AVX-512 (Intel Xeon, AMD Zen 4+), AMX (Sapphire Rapids+)

SmallAIOS considerations:
- Heterogeneous cores (Intel P/E cores): pin inference to P-cores for consistent latency
- AVX-512 varies: present on some i7 SKUs, all Xeon, AMD Zen 4+; detect at runtime
- AMX (Advanced Matrix Extensions): available on Xeon Sapphire Rapids+ for INT8/BF16 GEMM
- AMD SEV (Secure Encrypted Virtualization): for confidential inference in VM mode

Platform-specific optimizations:
- AVX2 GEMM as baseline, AVX-512/AMX GEMM as accelerated paths
- Cache hierarchy tuning: L1=48KB, L2=1.25MB (per P-core), L3=shared
- Big.LITTLE awareness: use `cpuid` + topology detection to identify P vs E cores

### QEMU Virtual Machine

Primary development and CI platform:
- x86-64: `qemu-system-x86_64 -machine q35 -cpu host -enable-kvm`
- ARM64: `qemu-system-aarch64 -machine virt -cpu cortex-a72`
- Virtio devices: virtio-blk, virtio-net, virtio-console
- GDB stub for kernel debugging

## Tier 2 Platforms (Supported, Best-Effort Optimization)

### NVIDIA Jetson Orin Nano

Recommended Jetson platform (replaces original Jetson Nano for GPU work):
- **CPU**: 6x Cortex-A78AE, NEON + optional SVE
- **GPU**: 1024 CUDA cores, Ampere architecture, CC 8.7
- **Memory**: 4 or 8 GB LPDDR5 (shared CPU/GPU)
- **TDP**: 7-15W

SmallAIOS considerations:
- Shared memory architecture: CPU and GPU share same physical RAM
- SmallAIOS tensor pool should be aware of unified memory
- GPU CC 8.7: full Ampere features, tensor cores, PTX 8.0+
- Low power: idle CPU cores should WFI aggressively
- Boot: U-Boot → UEFI → SmallAIOS (or containerized on JetPack Linux)

### NVIDIA Jetson Nano (Original)

Legacy Jetson for testing only:
- **CPU**: 4x Cortex-A57, NEON
- **GPU**: 128 CUDA cores, Maxwell architecture, CC 5.3
- **Memory**: 4 GB LPDDR4 (shared)
- **TDP**: 5-10W

SmallAIOS considerations:
- **CC 5.3 (Maxwell) is below our primary target (7.0+)**
- No tensor cores, no FP16 hardware support on Maxwell
- Limited GPU memory — small models only (MobileNet-class)
- Useful for testing ARM64 boot and CPU inference
- GPU support: basic CUDA kernels only, no fused attention, no tensor core paths
- Recommend Jetson Orin Nano for serious GPU edge inference

### Raspberry Pi 4 Model B

Edge CPU inference and testing platform:
- **CPU**: 4x Cortex-A72 @ 1.8 GHz, NEON
- **GPU**: VideoCore VI (not supported by SmallAIOS — OpenGL only, no compute)
- **Memory**: 2, 4, or 8 GB LPDDR4
- **Network**: Gigabit Ethernet (Broadcom GENET), WiFi (not supported)
- **Storage**: microSD, USB boot

SmallAIOS considerations:
- CPU inference only (no usable GPU)
- NEON SIMD only (no SVE on A72)
- Limited memory: optimize for small models, aggressive tensor pool management
- Boot: UEFI via Raspberry Pi UEFI firmware (TianoCore EDK2 port), or containerized
- Ethernet: Broadcom GENET driver needed for bare metal networking
- Good test platform for ARM64 code paths

### Raspberry Pi 5

Improved over Pi 4:
- **CPU**: 4x Cortex-A76 @ 2.4 GHz, NEON (no SVE)
- **Memory**: 4 or 8 GB LPDDR4X
- **Network**: Gigabit Ethernet
- Faster CPU makes it viable for lightweight inference (BERT-tiny, MobileNet)

### Qualcomm Snapdragon 845

Mobile/embedded ARM64 platform:
- **CPU**: 4x Kryo 385 Gold (A75) + 4x Kryo 385 Silver (A55), big.LITTLE
- **GPU**: Adreno 630 (not supported — no NVIDIA, no open compute API)
- **Memory**: 4-8 GB LPDDR4X
- **Devices**: Embedded in phones, dev boards (Dragonboard 845c, etc.)

SmallAIOS considerations:
- CPU inference only (Adreno GPU lacks the open compute interface we need)
- Big.LITTLE: schedule inference on A75 (Gold) cores for performance
- NEON SIMD, dotprod extension (INT8 dot product — good for quantized inference)
- Limited memory: small models, quantized INT8 recommended
- Boot: varies by board; likely containerized on Android/Linux base

## Platform Feature Matrix

| Feature | DGX Spark | i7/Ryzen | Jetson Orin | Jetson Nano | RPi 4/5 | SD 845 |
|---|---|---|---|---|---|---|
| NEON | Yes | N/A | Yes | Yes | Yes | Yes |
| SVE/SVE2 | SVE2 | N/A | Maybe | No | No | No |
| AVX2 | N/A | Yes | N/A | N/A | N/A | N/A |
| AVX-512 | N/A | Some | N/A | N/A | N/A | N/A |
| AMX | N/A | Xeon only | N/A | N/A | N/A | N/A |
| NVIDIA GPU | Yes (B200) | Optional | Yes (Ampere) | Yes (Maxwell) | No | No |
| Tensor Cores | Yes (FP8) | If GPU | Yes | No | No | No |
| Unified Memory | NVLink-C2C | No | Shared | Shared | N/A | N/A |
| AES-NI / ARMv8 Crypto | ARMv8 | AES-NI | ARMv8 | ARMv8 | ARMv8 | ARMv8 |
| PAC/BTI | Yes | N/A | Yes | No | No | No |
| MTE | Maybe | N/A | No | No | No | No |
| Bare Metal Boot | Yes | Yes | Yes | Yes | Yes | Board-specific |
| Container Mode | Yes | Yes | Yes | Yes | Yes | Yes |

## Build Targets

```bash
# Tier 1
cargo build --target x86_64-unknown-none       # x86-64 bare metal / VM
cargo build --target aarch64-unknown-none       # ARM64 bare metal / VM
cargo build --target x86_64-unknown-linux-musl  # x86-64 container (static)
cargo build --target aarch64-unknown-linux-musl # ARM64 container (static)

# GPU variants (via Cargo features)
cargo build --features nvidia_gpu               # Enable NVIDIA GPU support
cargo build --features nvidia_gpu,cc_53         # Maxwell (Jetson Nano)
cargo build --features nvidia_gpu,cc_87         # Ampere (Jetson Orin)
cargo build --features nvidia_gpu,cc_100        # Blackwell (DGX Spark)
```

## Testing Matrix

| Test | Where | What |
|---|---|---|
| Unit tests | Host (x86/ARM) | `cargo test` in hosted mode |
| Boot test (x86) | QEMU x86_64 | Boot, print, shutdown |
| Boot test (ARM) | QEMU aarch64 | Boot, print, shutdown |
| CPU inference (x86) | QEMU or native x86 | MobileNetV2, ResNet50 |
| CPU inference (ARM) | QEMU or RPi 4/5 | MobileNetV2 |
| GPU inference | DGX Spark, Jetson Orin | ResNet50, BERT |
| Container mode | Docker (any) | Full stack test |
| Network test | QEMU (virtio-net) | TCP/TLS IPC |
| Stress test | Native x86 or DGX | 24h sustained inference |
| Edge test | Jetson Orin Nano | Thermal throttling, low memory |

## Recommended Development Hardware

For contributors:
1. **Primary dev machine**: Any x86-64 or ARM64 Linux workstation
2. **GPU development**: Machine with NVIDIA GPU (RTX 3060+) or cloud GPU instance
3. **ARM64 testing**: Raspberry Pi 4/5 ($35-80) or QEMU
4. **Edge GPU testing**: Jetson Orin Nano ($200) — recommended over original Jetson Nano
5. **CI**: GitHub Actions (x86-64 runners) + self-hosted ARM64 runner (or QEMU)
