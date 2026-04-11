# SmallAIOS

**A minimal, secure, Rust-based operating system kernel purpose-built for AI inference workloads.**

[![CI](https://github.com/SmallAIOS/SmallAIOS/actions/workflows/ci.yml/badge.svg)](https://github.com/SmallAIOS/SmallAIOS/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/SmallAIOS/SmallAIOS/branch/main/graph/badge.svg)](https://codecov.io/gh/SmallAIOS/SmallAIOS)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

SmallAIOS is a clean-room `#![no_std]` Rust unikernel that boots directly to ONNX inference
with ~46 syscalls (vs Linux ~450). It targets bare-metal, VM, and containerized deployment
on x86-64, ARM64, and RISC-V platforms.

## Features

- **Single purpose** — Boot to ONNX inference. Nothing else.
- **6 architectures** — x86-64, AArch64, RISC-V 64, NVIDIA GPU, AMD GPU, Intel GPU
- **Post-quantum crypto** — ML-KEM-768 + ML-DSA-65 hybrid mode by default
- **Full network stack** — IPv4/IPv6, TCP/UDP, QUIC/HTTP3, TLS 1.3
- **65 ONNX operators** — see the [coverage roadmap](docs/onnx-coverage-roadmap.md) for the path to full standard-spec coverage
- **Formally verified** — 19 TLA+ models, SPIN protocols, Lean 4 proofs
- **4,143 tests** — Zero clippy warnings, MC/DC coverage on safety-critical paths
- **< 15 MB** — Release binary per architecture

## Architecture

![SmallAIOS Architecture](https://kroki.io/plantuml/svg/eNqdVV1v2jAUffevsNhL94DKugJBqlBDSNZoJTA8qmorD1ZyCxGOjRynHVr73-uklHwxrc0LinzOPfee-yEuY0WlSiKGSUQZM90pwab016ECXyUSULwJ-ZZKGmFfRFvBgSuidgyw1ATKV6xICeCeJkw5giuPRoBbBFYC8MJtFUjxmgbiMeQrfE9ZDOhYCvwXYTyi_mYlRcIDSzAhLy40IHfDIf407tmmY6QUIQOQVfhsYHRGg2MKMciH0IdMo-t07a9VjQLhrG_avc4xlQ1IDizlOLYzsPtVkQL-xTrvOMc0qG5yyrC7tmHXrLyhfcewjDF6RmirBehKt9ROXeJrugPZwrnptGG_Ld14GnKQ-FRnlNCegKIMZyHLnFzSI6-O41wy70EmOvW8WzxPuAojWBbhFHRnFj75BVysP9ew2ZS4t9jSM6WqBnqg8MlPa3b6Y-Fa9VgCfiJDtasBCzLS5sh4nlpM4jJe9PU9G0Lu6jCUzNQreseJv4YgYbpjT3gCkdCtfcJkF_v6GGL9eWVeLwvBpQzFO9HVXFEZPOqm5yn3U8wS_jF67d758vCavpmpQvVx7hKrfVN--zZb3HHvxh27pq7JnIz1r8sVsAJNV_af-bf12fH2sDLRd0cVR_3uoNIO6ArLy3SgfWQcqLx0DTXKy9lQpLrEDWXyZW8oUDuKZjroY2dxSLLf7abhb2fQNH5_MU3D_31cCF0CD_SfI3oBCeNYkw==)

<details>
<summary>Diagram source (PlantUML)</summary>

See [`docs/architecture.puml`](docs/architecture.puml)
</details>

## Workspace Crates

| Crate | Description |
|-------|-------------|
| `smallaios-kernel` | Core kernel: scheduler, memory, syscalls, HAL |
| `smallaios-arch-x86_64` | x86-64 HAL: boot, GDT, IDT, APIC, paging |
| `smallaios-arch-aarch64` | AArch64 HAL: boot, GICv3, paging, SVE |
| `smallaios-arch-riscv64` | RISC-V HAL: boot, PLIC, CLINT, SBI |
| `smallaios-arch-nvidia` | NVIDIA GPU HAL: PCIe, compute, PTX |
| `smallaios-arch-amd` | AMD GPU HAL: RDNA/CDNA, wavefront, HIP |
| `smallaios-arch-intel-gpu` | Intel GPU HAL: Xe, EU compute, SPIR-V |
| `smallaios-onnx-rt` | ONNX runtime: protobuf parser, 7 operators |
| `smallaios-security` | Capabilities, PQC crypto, audit, compliance |
| `smallaios-net` | IPv4/IPv6, TCP/UDP, QUIC/HTTP3, TLS 1.3 |
| `smallaios-ipc` | Zenoh-inspired pub/sub with security gate |
| `smallaios-posix` | Minimal POSIX compatibility layer |
| `smallaios-container` | Container interface, health, metrics |
| `smallaios-bus` | CAN, ARINC 429/664, MIL-STD-1553, SpaceWire, DDS, FPGA |
| `smallaios-peripheral` | I2C, SPI, GPIO, UART, CSI camera, I2S audio |
| `smallaios-usb` | xHCI host, descriptors, gadget framework |
| `smallaios-sdr` | HackRF, PlutoSDR, IQ processing pipeline |
| `smallaios-bench` | Benchmarks and performance testing |

## Quick Start

Requires Rust nightly (`nightly-2026-02-01`). The toolchain is pinned in `rust-toolchain.toml`.

```bash
# Run tests
cargo test -p smallaios-kernel -p smallaios-security -p smallaios-net

# Build bare-metal kernel (x86-64)
RUSTFLAGS="-C link-arg=-Tarch/x86_64/linker.ld -C code-model=kernel" \
  cargo build --release --target x86_64-unknown-none -p smallaios-arch-x86_64 \
  -Z build-std=core,compiler_builtins,alloc

# Build bare-metal kernel (AArch64)
RUSTFLAGS="-C link-arg=-Tarch/aarch64/linker.ld" \
  cargo build --release --target aarch64-unknown-none -p smallaios-arch-aarch64 \
  -Z build-std=core,compiler_builtins,alloc

# Build bare-metal kernel (RISC-V)
RUSTFLAGS="-C link-arg=-Tarch/riscv64/linker.ld" \
  cargo build --release --target riscv64gc-unknown-none-elf -p smallaios-arch-riscv64 \
  -Z build-std=core,compiler_builtins,alloc

# Boot in QEMU (RISC-V)
qemu-system-riscv64 -machine virt -cpu rv64 -m 512M -nographic \
  -bios default -kernel target/riscv64gc-unknown-none-elf/release/smallaios-riscv64
```

## CI/CD

The [CI pipeline](.github/workflows/ci.yml) runs on every push and PR:

- Format check (`cargo fmt`)
- Clippy lint (all host-testable crates)
- Unit tests (4,143 tests across 12 crates)
- Bare-metal builds (x86-64, AArch64, RISC-V)
- RISC-V QEMU smoke test
- Binary size validation (< 15 MB)
- TLA+ formal verification (19 models)
- Code coverage (Codecov)
- Static analysis (SonarCloud)

## Formal Verification

19 TLA+ models verify correctness of critical subsystems:

- **Kernel**: BuddyAllocator, Scheduler, SecurityGate, PolicyUpdate
- **Bus protocols**: CanArbitration, Arinc429Scheduler, AfdxVirtualLink, Mil1553Protocol, SpaceWireLink, DdsReliableDelivery, DdsDiscovery
- **Networking**: QuicFlowControl, QuicMigration
- **Peripherals**: I2CArbitration, SPIProtocol, GPIOInterrupt, UARTFlowControl, CSIFrameBuffer, I2SRingBuffer

## License

Apache License 2.0 — see [LICENSE](LICENSE).
