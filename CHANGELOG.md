# Changelog

All notable changes to SmallAIOS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-12

Initial alpha release of SmallAIOS: a minimal, secure, Rust-based OS kernel
purpose-built for AI inference workloads.

### Core Kernel
- Cooperative async scheduler with SYSTEM/IPC/INFERENCE priority classes
- Buddy allocator with slab sub-allocator for kernel memory management
- 46 syscalls with capability-based access control
- Hardware Abstraction Layer (HAL) for multi-architecture support
- Watchdog timer and health monitoring

### Architecture Support
- **x86-64**: Boot (multiboot2), GDT, IDT, APIC, 4-level paging, AVX/SSE
- **AArch64**: Boot, GICv3 interrupt controller, paging, SVE, PSCI
- **RISC-V 64**: Boot, PLIC, CLINT timer, SBI interface, Sv39/Sv48 paging
- **NVIDIA GPU**: PCIe discovery, memory management, DMA, compute dispatch, PTX
- **AMD GPU**: PCIe (0x1002), RDNA/CDNA detection, wavefront compute, HIP kernels
- **Intel GPU**: PCIe (0x8086), Xe-LP/HPG/HPC, EU compute, SPIR-V, Level Zero

### Networking
- IPv4/IPv6 dual-stack with TCP, UDP, ARP, Ethernet
- NDP (Neighbor Discovery Protocol) and SLAAC (RFC 4862)
- QUIC v1 (RFC 9000) with congestion control, connection migration, 0-RTT
- HTTP/3 over QUIC
- TLS 1.3 with ML-KEM-768 hybrid post-quantum key exchange

### Security
- Capability-based access control system
- Post-quantum cryptography: ML-KEM-768, ML-DSA-65, hybrid signatures
- Classical cryptography: SHA-3, AES-256-GCM, Ed25519, X25519, CSPRNG
- NIST SP 800-53 control mapping
- Tamper-evident audit logging with hash chains
- Supply chain security: SBOM generation, vendor attestation
- Incident response framework with severity classification
- OT/safety-critical: WCET analysis, anomaly detection, failsafe shutdown
- Constant-time implementations verified with dudect testing

### ONNX Runtime
- Clean-room `#![no_std]` protobuf parser
- Graph optimizer with operator fusion
- 7 operators: MatMul, Conv, Relu, Softmax, Add, Reshape, GEMM
- CPU and CUDA execution providers

### IPC
- Zenoh-inspired pub/sub messaging
- Capability-enforced message passing
- Type-safe security gate with formal verification (formal-gate feature)
- Bus transport integration

### Bus Protocols
- CAN 2.0/FD with CANaerospace and MCP2515 controller support
- ARINC 429: BNR, BCD, discrete words; scheduler, filter, hardware adapter
- ARINC 664 (AFDX): virtual links, traffic shaping, integrity monitoring
- MIL-STD-1553: bus controller, remote terminal, dual-redundant bus
- SpaceWire: link state machine, RMAP, speed negotiation
- CCSDS: telecommand/telemetry frames, CLTU, packet routing
- DDS: RTPS, CDR serialization, QoS, discovery, reliable delivery, security
- FPGA: AXI-Lite, AXI-Full, DMA, interrupt controller, Zynq/PolarFire

### Peripheral Drivers
- I2C: ARM MMIO, RISC-V MMIO, Xilinx AXI IIC, bit-bang
- SPI: ARM MMIO, RISC-V MMIO, Xilinx AXI Quad SPI
- GPIO: ARM PL061, RISC-V MMIO, Xilinx AXI GPIO
- UART: NS16550A, ARM PL011, SiFive, Xilinx AXI UART Lite, NMEA parser
- CSI-2 Camera: Broadcom Unicam, Tegra VI; IMX219, IMX477, OV5640 sensors
- I2S Audio: ARM, RISC-V, FPGA controllers; WM8960, ES8388, TLV320AIC3x codecs

### USB
- USB core: descriptors, enumeration, device/configuration/interface model
- xHCI host controller driver with transfer ring management
- Hub driver with port power and reset sequencing
- Gadget framework with ONNX inference gadget

### Software-Defined Radio
- HackRF One driver: USB control, sample streaming, frequency/gain tuning
- ADALM-Pluto driver: IIO interface, DMA streaming, calibration
- IQ processing pipeline: FFT (radix-2 Cooley-Tukey), FIR filter, decimator

### Container & Deployment
- Docker/Kubernetes container interface
- Health checks, readiness probes, metrics export
- Go Virtual Kubelet provider
- POSIX compatibility layer for AI runtime operations

### Formal Verification
- 19 TLA+ models covering all major protocols and subsystems
- SPIN protocol models
- Lean 4 type-level proofs
- MC/DC coverage on safety-critical paths

### CI/CD
- GitHub Actions: format, clippy, unit tests, 3 bare-metal builds
- RISC-V QEMU smoke test
- Binary size check (< 15 MB per architecture)
- TLA+ formal verification (19 models)
- Code coverage via cargo-llvm-cov + Codecov
- Static analysis via SonarCloud
- Change gates for PR mergeability

### Metrics
- 4,143 tests passing across 18 crates
- Zero clippy warnings
- All 19 TLA+ models verified
- Release binary < 15 MB per architecture
