## Why

SmallAIOS is positioned as a safety-critical inference platform spanning datacenter, edge, automotive, aviation, defense, and space domains. The current spec covers x86-64, ARM64, and NVIDIA GPU with TCP/Zenoh networking and OCI container deployment. To reach automotive ECUs, avionics LRUs, spacecraft computers, and FPGA SoCs, we need: (1) safety-critical bus protocols as first-class IPC transports, (2) Kubernetes orchestration via Virtual Kubelet so SmallAIOS nodes participate in K8s/K3s clusters without running a Linux kernel, (3) RISC-V architecture support for space-grade and FPGA SoC platforms, (4) SoC FPGA platform support where SmallAIOS runs on embedded ARM/MIPS/RISC-V cores with FPGA fabric providing bus peripherals, and (5) a rigorous benchmark infrastructure comparing SmallAIOS against Linux baselines across three inference modalities (vision, text, audio/signal) on real hardware.

## What Changes

- Add clean-room CAN bus protocol (CAN 2.0A/B, CAN FD) with controller drivers and Zenoh transport adapter
- Add clean-room ARINC 429 word codec, transmit scheduler, and Zenoh transport adapter
- Add clean-room ARINC 664 Part 7 (AFDX) Virtual Link management, redundancy, and Zenoh transport adapter
- Add clean-room MIL-STD-1553 command/response protocol with bus controller and Zenoh transport adapter
- Add clean-room SpaceWire packet codec, link interface, and Zenoh transport adapter
- Add clean-room CCSDS Space Packet Protocol (SPP) codec, telemetry/telecommand framing, and Zenoh transport adapter
- Add clean-room DDS (Data Distribution Service) DCPS API, RTPS wire protocol, QoS policies, DDS-Security, and Zenoh transport adapter
- Add RISC-V (RV64GC) architecture support: boot, paging, interrupts (PLIC/CLINT), SMP (HSM)
- Add SoC FPGA platform support: AXI/AXI-Lite MMIO driver, AXI DMA, DTB-based peripheral discovery
- **BREAKING**: Revise Phase 11 from "Container and Kubernetes Integration" to "Deployment and Provisioning" — separate container packaging from orchestration
- Add Kubernetes integration via Virtual Kubelet provider (Go, Linux-side) with K3s edge and K8s datacenter support
- Add benchmark infrastructure: boot-to-inference timing, Linux baselines (bare metal, Docker, K8s), three models (MobileNetV2, DistilBERT, Whisper-tiny), four hardware targets (DGX Spark, Xeon, Jetson, Raspberry Pi), statistical methodology (N=1000+, p50/p99/p999, jitter)

## Capabilities

### New Capabilities

- `can-bus`: CAN 2.0A/B and CAN FD frame codec, bus state machine, controller drivers (PS-CAN, AXI CAN, MCP2515 SPI), acceptance filtering, bus-off recovery, Zenoh transport adapter mapping CAN IDs to key expressions
- `arinc-429`: ARINC 429 32-bit word encode/decode (BNR, BCD, discrete), label filtering, fixed-rate transmit scheduler, hardware interface abstraction, Zenoh transport adapter mapping labels to key expressions
- `arinc-664`: ARINC 664 Part 7 (AFDX) Virtual Link configuration and BAG traffic shaping, sequence number generation/checking, dual-network redundancy management, sub-VL scheduling, Zenoh transport adapter
- `mil-std-1553`: MIL-STD-1553B command/response protocol, bus controller and remote terminal modes, dual-redundant bus management, message scheduling, Zenoh transport adapter mapping subaddresses to key expressions
- `spacewire`: SpaceWire (ECSS-E-ST-50-12C) link interface, packet codec, time-code distribution, RMAP (Remote Memory Access Protocol) support, Zenoh transport adapter
- `ccsds-spp`: CCSDS Space Packet Protocol (CCSDS 133.0-B) packet encode/decode, APID-based routing, telemetry and telecommand transfer frames, Zenoh transport adapter mapping APIDs to key expressions
- `riscv-arch`: RISC-V RV64GC architecture support — boot (OpenSBI), SV48 4-level page tables, PLIC interrupt controller, CLINT timer, SBI HSM for SMP boot, target: riscv64gc-unknown-none-elf
- `soc-fpga-platform`: SoC FPGA platform support — AXI/AXI-Lite memory-mapped register access, AXI DMA controller driver, DTB-based peripheral discovery for FPGA soft-IP, Zynq UltraScale+ and PolarFire SoC reference bring-up
- `kubernetes-integration`: Kubernetes orchestration via Virtual Kubelet provider — SmallAIOS management API (model deploy, health, metrics, resource reporting), K3s support for edge (Jetson, RPi), K8s support for datacenter (Spark, Xeon), pod spec to model deployment translation
- `benchmark-infrastructure`: Performance comparison framework — boot-to-inference cold start timing, warm inference latency (p50/p99/p999), throughput, jitter/determinism, memory footprint; Linux baselines on bare metal, Docker, K8s/K3s; three models: MobileNetV2 (vision), DistilBERT (text), Whisper-tiny (audio/signal); four hardware targets: DGX Spark, Xeon, Jetson, RPi
- `dds`: OMG Data Distribution Service (DDS) implementation — DCPS (Data-Centric Publish-Subscribe) API with Topic/DataWriter/DataReader, RTPS (Real-Time Publish-Subscribe) wire protocol for interoperability with ROS 2 and AUTOSAR Adaptive, comprehensive QoS policies (reliability, durability, deadline, liveliness, ownership, history, resource limits), DDS-Security plugin for authentication and access control, Zenoh transport adapter mapping DDS domains/topics to key expressions

### Modified Capabilities

- `07-container-interface`: Revise deployment modes to separate container packaging (OCI image, VM image, bare metal provisioning) from Kubernetes orchestration (moved to `kubernetes-integration`). Add UEFI Secure Boot and image signing with ML-DSA-65.
- `05-device-hal`: Extend HAL trait with bus peripheral abstractions (CAN controller, ARINC transceiver, SpaceWire link) and FPGA fabric interface (AXI register access, DMA). Add RISC-V HAL implementation.
- `10-hardware-platforms`: Add RISC-V platforms (PolarFire SoC, SiFive HiFive, QEMU virt) to Tier 2. Add SoC FPGA platforms (Zynq UltraScale+, PolarFire SoC) to Tier 2.
- `04-ipc-messaging`: Add CAN, ARINC 429, ARINC 664, MIL-STD-1553, SpaceWire, CCSDS SPP, and DDS as Zenoh transport types alongside existing TCP, shared memory, and intra-kernel transports.

## Impact

- **Rust workspace**: Add `arch/riscv64` crate, add `bus` crate (CAN, ARINC, 1553, SpaceWire, CCSDS, DDS protocol implementations), add `fpga` crate (AXI drivers, DMA)
- **External tooling**: Virtual Kubelet provider is a separate Go project (outside Rust workspace, outside safety-critical certification boundary)
- **Benchmark harness**: Separate `bench/` directory with scripts, Linux baseline configs, and result analysis tools
- **Build targets**: Add `riscv64gc-unknown-none-elf` bare-metal target
- **Hardware dependencies**: CAN transceiver (MCP2515 or integrated), ARINC 429 transceiver (HI-3593 or FPGA soft-IP), SpaceWire LVDS PHY, MIL-STD-1553 transceiver
- **Formal verification**: New TLA+ models for CAN bus arbitration, ARINC 429 transmit scheduling, MIL-STD-1553 command/response protocol, SpaceWire link state machine, DDS RTPS discovery and reliable delivery
- **Safety certification**: Bus protocol implementations subject to DO-178C DAL A (aviation) and ISO 26262 ASIL D (automotive) coverage requirements
