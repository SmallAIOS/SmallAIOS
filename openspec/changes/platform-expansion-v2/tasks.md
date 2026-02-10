## 14. Kubernetes Integration

- [ ] 14.1 Define SmallAIOS management API Zenoh endpoints (deploy, undeploy, status, config)
- [ ] 14.2 Implement management API handler in kernel IPC module
- [ ] 14.3 Implement node resource reporting (CPU, memory, GPU, loaded models)
- [ ] 14.4 Scaffold Go Virtual Kubelet provider project (go.mod, main, provider interface)
- [ ] 14.5 Implement Virtual Kubelet Zenoh client (connect to SmallAIOS management API)
- [ ] 14.6 Implement pod spec translation (container image → ONNX model URL, env → config)
- [ ] 14.7 Implement pod lifecycle management (Pending → Running → Succeeded/Failed)
- [ ] 14.8 Implement node resource advertisement (CPU, memory, nvidia.com/gpu extended resource)
- [ ] 14.9 Implement health probe integration (liveness/readiness via existing /health endpoint)
- [ ] 14.10 Implement metrics integration (Prometheus /metrics passthrough)
- [ ] 14.11 Test with K3s on Jetson/RPi (edge deployment)
- [ ] 14.12 Test with K8s on Xeon/Spark (datacenter deployment)
- [ ] 14.13 Test multi-node: single provider managing multiple SmallAIOS instances

## 15. Safety-Critical Bus Protocols — CAN Bus

- [ ] 15.1 Create `smallaios-bus` crate with feature flags (can, arinc429, arinc664, mil1553, spacewire, ccsds, dds)
- [ ] 15.2 Define ZenohTransport trait for bus protocol adapters
- [ ] 15.3 Implement CAN 2.0A frame encode/decode (11-bit ID, 0-8 byte payload, CRC-15)
- [ ] 15.4 Implement CAN 2.0B extended frame encode/decode (29-bit ID)
- [ ] 15.5 Implement CAN FD frame support (64-byte payload, BRS flag)
- [ ] 15.6 Implement CAN bus state machine (Error Active, Error Passive, Bus Off, recovery)
- [ ] 15.7 Implement acceptance filtering (hardware mask + software filter)
- [ ] 15.8 Implement CAN controller driver abstraction trait
- [ ] 15.9 Implement MCP2515 SPI CAN controller driver
- [ ] 15.10 Implement AXI CAN controller driver (for FPGA soft-IP)
- [ ] 15.11 Implement CAN-to-Zenoh transport adapter (CAN ID → `can/{bus_id}/{frame_id}`)
- [ ] 15.12 Add CANaerospace profile support for civil aviation applications
- [ ] 15.13 Write TLA+ model for CAN bus arbitration
- [ ] 15.14 Unit tests for CAN frame codec (100% MC/DC on encode/decode)
- [ ] 15.15 Integration test: CAN loopback TX/RX in QEMU (virtio-can or mock)

## 16. Safety-Critical Bus Protocols — ARINC 429

- [ ] 16.1 Implement ARINC 429 32-bit word encode/decode (label, SDI, data, SSM, parity)
- [ ] 16.2 Implement BNR (Binary) data format encoding/decoding
- [ ] 16.3 Implement BCD (Binary Coded Decimal) data format encoding/decoding
- [ ] 16.4 Implement discrete data word support
- [ ] 16.5 Implement label-based filtering and routing
- [ ] 16.6 Implement fixed-rate transmit scheduler (per-label configurable rate)
- [ ] 16.7 Implement hardware interface abstraction (SPI transceiver, FPGA soft-IP)
- [ ] 16.8 Implement ARINC 429-to-Zenoh transport adapter (label → `arinc429/{channel}/{label}`)
- [ ] 16.9 Support both low speed (12.5 kbps) and high speed (100 kbps) configurations
- [ ] 16.10 Unit tests for word codec (100% MC/DC, all data formats)
- [ ] 16.11 Write TLA+ model for ARINC 429 transmit scheduling

## 17. Safety-Critical Bus Protocols — ARINC 664 (AFDX)

- [ ] 17.1 Implement Virtual Link (VL) configuration (VL ID, BAG 1-128ms, Lmax)
- [ ] 17.2 Implement BAG traffic shaping and policing
- [ ] 17.3 Implement sequence number generation and checking (per VL)
- [ ] 17.4 Implement dual-network redundancy management (integrity checking)
- [ ] 17.5 Implement sub-VL scheduling (round-robin within a VL)
- [ ] 17.6 Implement frame filtering (VL ID + destination MAC matching)
- [ ] 17.7 Implement AFDX-to-Zenoh transport adapter (VL → `afdx/{vl_id}`)
- [ ] 17.8 Integrate with existing Ethernet/IP stack
- [ ] 17.9 Unit tests for VL scheduling and redundancy management
- [ ] 17.10 Write TLA+ model for AFDX Virtual Link state machine

## 18. Safety-Critical Bus Protocols — MIL-STD-1553

- [ ] 18.1 Implement command word encode/decode (RT address, subaddress, word count/mode code)
- [ ] 18.2 Implement status word encode/decode (RT address, message error, busy, etc.)
- [ ] 18.3 Implement data word encode/decode (16-bit payload, odd parity)
- [ ] 18.4 Implement Bus Controller (BC) mode: command scheduling, response timeout
- [ ] 18.5 Implement Remote Terminal (RT) mode: command recognition, response generation
- [ ] 18.6 Implement dual-redundant bus management (Bus A / Bus B failover)
- [ ] 18.7 Implement mode codes (transmit status, synchronize, reset, etc.)
- [ ] 18.8 Implement hardware interface abstraction (dedicated 1553 transceiver)
- [ ] 18.9 Implement MIL-1553-to-Zenoh transport adapter (RT/SA → `mil1553/{bus}/{rt}/{sa}`)
- [ ] 18.10 Unit tests for word codec and protocol state machine (100% MC/DC)
- [ ] 18.11 Write TLA+ model for 1553 command/response protocol

## 19. Safety-Critical Bus Protocols — SpaceWire

- [ ] 19.1 Implement SpaceWire packet encode/decode (destination address, cargo, EOP/EEP)
- [ ] 19.2 Implement link interface state machine (ErrorReset → ErrorWait → Ready → Started → Connecting → Run)
- [ ] 19.3 Implement character-level encoding (data characters, control characters, time-codes)
- [ ] 19.4 Implement time-code distribution (6-bit counter broadcast)
- [ ] 19.5 Implement RMAP read/write commands (Remote Memory Access Protocol)
- [ ] 19.6 Implement link speed configuration (2-400 Mbps)
- [ ] 19.7 Implement hardware interface abstraction (LVDS PHY, FPGA codec IP)
- [ ] 19.8 Implement SpaceWire-to-Zenoh transport adapter (dest → `spw/{link}/{dest}`)
- [ ] 19.9 Unit tests for packet codec and link state machine
- [ ] 19.10 Write TLA+ model for SpaceWire link state machine

## 20. Safety-Critical Bus Protocols — CCSDS Space Packet

- [ ] 20.1 Implement CCSDS Space Packet primary header encode/decode (version, type, APID, sequence, length)
- [ ] 20.2 Implement telemetry (TM) packet assembly and parsing
- [ ] 20.3 Implement telecommand (TC) packet assembly and parsing
- [ ] 20.4 Implement APID-based routing and filtering
- [ ] 20.5 Implement TM transfer frame support (CCSDS 132.0-B)
- [ ] 20.6 Implement TC transfer frame support (CCSDS 232.0-B)
- [ ] 20.7 Implement CLTU encoding for uplink (optional, CCSDS 231.0-B)
- [ ] 20.8 Implement CCSDS-to-Zenoh transport adapter (APID → `ccsds/{apid}`)
- [ ] 20.9 Unit tests for packet codec (100% MC/DC)
- [ ] 20.10 Validate against CCSDS Blue Book test vectors (where available)

## 21. DDS (Data Distribution Service)

- [ ] 21.1 Define DDS DCPS core types: DomainParticipant, Topic, Publisher, Subscriber, DataWriter, DataReader
- [ ] 21.2 Implement DomainParticipant creation, configuration, and lifecycle management
- [ ] 21.3 Implement Topic definition and type registration with type consistency enforcement
- [ ] 21.4 Implement DataWriter with write, dispose, unregister operations
- [ ] 21.5 Implement DataReader with read, take, wait-set, and listener notification
- [ ] 21.6 Implement CDR (Common Data Representation) v2 serializer and deserializer
- [ ] 21.7 Implement RTPS 2.3 message format: Header, Submessages (DATA, HEARTBEAT, ACKNACK, GAP, INFO_TS)
- [ ] 21.8 Implement SPDP (Simple Participant Discovery Protocol) with multicast announcements
- [ ] 21.9 Implement SEDP (Simple Endpoint Discovery Protocol) for DataWriter/DataReader matching
- [ ] 21.10 Implement RTPS reliable delivery: heartbeat/acknack protocol, sample retransmission
- [ ] 21.11 Implement RTPS best-effort delivery with sequence number tracking
- [ ] 21.12 Implement QoS policies: RELIABILITY (reliable/best-effort), DURABILITY (volatile/transient-local)
- [ ] 21.13 Implement QoS policies: DEADLINE, LIVELINESS (automatic/manual), OWNERSHIP (shared/exclusive)
- [ ] 21.14 Implement QoS policies: HISTORY (keep-last/keep-all), RESOURCE_LIMITS (max_samples, max_instances)
- [ ] 21.15 Implement QoS compatibility checking between DataWriter offers and DataReader requests
- [ ] 21.16 Implement DDS-Security authentication plugin (mutual auth with ML-DSA-65 certificates)
- [ ] 21.17 Implement DDS-Security access control plugin (governance and permissions documents)
- [ ] 21.18 Implement DDS-to-Zenoh transport adapter (domain/topic → `dds/{domain_id}/{topic}`)
- [ ] 21.19 Implement ROS 2 topic name mangling support (`rt/`, `rq/`, `rr/` prefixes)
- [ ] 21.20 Unit tests for CDR serialization (all primitive types, structs, sequences, strings)
- [ ] 21.21 Unit tests for RTPS message encode/decode (100% MC/DC)
- [ ] 21.22 Unit tests for SPDP/SEDP discovery state machines
- [ ] 21.23 Unit tests for QoS compatibility matrix (all valid/invalid combinations)
- [ ] 21.24 Integration test: DDS pub/sub within SmallAIOS (loopback)
- [ ] 21.25 Interoperability test: SmallAIOS DDS ↔ FastDDS (ROS 2 node) on same network

## 22. RISC-V Architecture Support

- [ ] 22.1 Create `smallaios-arch-riscv64` crate with target riscv64gc-unknown-none-elf
- [ ] 22.2 Create custom target JSON and linker script for RISC-V bare metal
- [ ] 22.3 Implement assembly entry point (set stack, clear BSS, set stvec, call kernel_main)
- [ ] 22.4 Implement SV48 4-level page table management (map, unmap, protect)
- [ ] 22.5 Implement TLB flush via sfence.vma instruction
- [ ] 22.6 Implement PLIC interrupt controller driver (priority, enable, claim, complete)
- [ ] 22.7 Implement CLINT timer driver (mtime/mtimecmp for periodic tick)
- [ ] 22.8 Implement SBI HSM extension calls for SMP boot (hart_start, hart_stop, hart_get_status)
- [ ] 22.9 Implement SBI IPI extension for inter-hart interrupts
- [ ] 22.10 Implement NS16550A UART driver (QEMU virt machine console output)
- [ ] 22.11 Implement CPU feature detection (A, C, D, F extensions)
- [ ] 22.12 Verify boot-to-serial-output in QEMU virt (riscv64)
- [ ] 22.13 Add RISC-V to CI (build + QEMU smoke test)
- [ ] 22.14 Add RISC-V HAL implementation to kernel HAL trait

## 23. SoC FPGA Platform Support

- [ ] 23.1 Create `smallaios-fpga` crate with feature flags (zynq, polarfire)
- [ ] 23.2 Implement AXI4-Lite memory-mapped register read/write driver
- [ ] 23.3 Implement AXI4 full burst transfer driver
- [ ] 23.4 Implement AXI DMA controller driver (simple mode)
- [ ] 23.5 Implement AXI DMA scatter-gather mode
- [ ] 23.6 Implement DTB-based peripheral discovery for FPGA soft-IP blocks
- [ ] 23.7 Implement interrupt routing from FPGA fabric to CPU interrupt controller
- [ ] 23.8 Implement Zynq UltraScale+ platform support package (PS-PL interface, clocks, resets)
- [ ] 23.9 Implement PolarFire SoC platform support package (RISC-V + FPGA fabric)
- [ ] 23.10 Unit tests for AXI register access and DMA transfers (mock MMIO)
- [ ] 23.11 Integration test on Zynq board: read/write FPGA registers from SmallAIOS

## 24. Deployment and Provisioning (Phase 11 Revision)

- [ ] 24.1 Implement UEFI Secure Boot support with ML-DSA-65 signed kernel image
- [ ] 24.2 Implement PXE/iPXE bare metal network boot provisioning
- [ ] 24.3 Implement VM image generation (raw disk, qcow2, VMDK formats)
- [ ] 24.4 Implement image signing with ML-DSA-65 post-quantum signatures
- [ ] 24.5 Update OCI image build for multi-arch (amd64, arm64, riscv64)
- [ ] 24.6 Verify image size target < 15 MB (kernel + runtime, no model)

## 25. IPC Transport Extensions

- [ ] 25.1 Extend ZenohRouter with bus transport registration interface
- [ ] 25.2 Implement transport auto-discovery from DTB/ACPI peripheral enumeration
- [ ] 25.3 Implement cross-transport routing (message on CAN routed to TCP subscriber, DDS to CAN, etc.)
- [ ] 25.4 Unit tests for transport-agnostic pub/sub across mixed transports (including DDS)
- [ ] 25.5 Integration test: inference result published over TCP, delivered over CAN
- [ ] 25.6 Integration test: DDS topic bridged to ARINC 429 via Zenoh router

## 26. HAL Extensions

- [ ] 26.1 Define BusPeripheral HAL trait (init, tx_frame, rx_frame, irq_handler)
- [ ] 26.2 Define FpgaFabric HAL trait (axi_read, axi_write, dma_transfer)
- [ ] 26.3 Implement RISC-V HAL (paging, PLIC, CLINT, SBI wrappers)
- [ ] 26.4 Update hardware platform spec with RISC-V and SoC FPGA tier 2 targets
- [ ] 26.5 Add riscv64gc-unknown-none-elf build target to Makefile and CI

## 27. Benchmark Infrastructure

- [ ] 27.1 Create bench/ directory structure (scripts, configs, models, results)
- [ ] 27.2 Implement cold start measurement harness (power-on to first inference)
- [ ] 27.3 Implement warm inference latency harness (N=1000+ runs, p50/p99/p999)
- [ ] 27.4 Implement throughput benchmark (batch sizes 1, 4, 16, 64)
- [ ] 27.5 Implement jitter measurement (stdev and max deviation)
- [ ] 27.6 Implement memory footprint measurement (peak RSS)
- [ ] 27.7 Download and prepare benchmark models: MobileNetV2 (vision), DistilBERT (text), Whisper-tiny (audio/signal)
- [ ] 27.8 Create Linux bare metal baseline scripts (ONNX Runtime C++ binary)
- [ ] 27.9 Create Docker baseline (Dockerfile + ONNX Runtime)
- [ ] 27.10 Create K8s/K3s baseline (deployment manifests + ONNX Runtime pod)
- [ ] 27.11 Create hardware-specific configs (DGX Spark, Xeon, Jetson, RPi)
- [ ] 27.12 Document BIOS settings, CPU frequency pinning, thermal monitoring
- [ ] 27.13 Implement report generation (markdown tables, CSV export)
- [ ] 27.14 Run benchmark suite on all 4 hardware targets, generate comparison report

## 28. Formal Verification — Bus Protocols and DDS

- [ ] 28.1 Write TLA+ model for CAN bus arbitration (priority-based, no starvation)
- [ ] 28.2 Write TLA+ model for ARINC 429 transmit scheduler (no label starvation)
- [ ] 28.3 Write TLA+ model for AFDX Virtual Link BAG enforcement (no deadline miss)
- [ ] 28.4 Write TLA+ model for MIL-STD-1553 command/response (no unanswered commands)
- [ ] 28.5 Write TLA+ model for SpaceWire link state machine (correct transitions)
- [ ] 28.6 Write TLA+ model for DDS RTPS reliable delivery (no sample loss, no reordering)
- [ ] 28.7 Write TLA+ model for DDS SPDP/SEDP discovery (eventual consistency, no orphan endpoints)
- [ ] 28.8 Create TLC configs for all bus protocol and DDS models
- [ ] 28.9 Add bus protocol and DDS TLA+ verification to CI
