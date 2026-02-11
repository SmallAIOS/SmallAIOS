## 14. Kubernetes Integration

- [x] 14.1 Define SmallAIOS management API Zenoh endpoints (deploy, undeploy, status, config)
- [x] 14.2 Implement management API handler in kernel IPC module
- [x] 14.3 Implement node resource reporting (CPU, memory, GPU, loaded models)
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

- [x] 15.1 Create `smallaios-bus` crate with feature flags (can, arinc429, arinc664, mil1553, spacewire, ccsds, dds)
- [x] 15.2 Define ZenohTransport trait for bus protocol adapters
- [x] 15.3 Implement CAN 2.0A frame encode/decode (11-bit ID, 0-8 byte payload, CRC-15)
- [x] 15.4 Implement CAN 2.0B extended frame encode/decode (29-bit ID)
- [x] 15.5 Implement CAN FD frame support (64-byte payload, BRS flag)
- [x] 15.6 Implement CAN bus state machine (Error Active, Error Passive, Bus Off, recovery)
- [x] 15.7 Implement acceptance filtering (hardware mask + software filter)
- [x] 15.8 Implement CAN controller driver abstraction trait
- [x] 15.9 Implement MCP2515 SPI CAN controller driver
- [x] 15.10 Implement AXI CAN controller driver (for FPGA soft-IP)
- [x] 15.11 Implement CAN-to-Zenoh transport adapter (CAN ID → `can/{bus_id}/{frame_id}`)
- [x] 15.12 Add CANaerospace profile support for civil aviation applications
- [x] 15.13 Write TLA+ model for CAN bus arbitration
- [x] 15.14 Unit tests for CAN frame codec (100% MC/DC on encode/decode)
- [x] 15.15 Integration test: CAN loopback TX/RX in QEMU (virtio-can or mock)

## 16. Safety-Critical Bus Protocols — ARINC 429

- [x] 16.1 Implement ARINC 429 32-bit word encode/decode (label, SDI, data, SSM, parity)
- [x] 16.2 Implement BNR (Binary) data format encoding/decoding
- [x] 16.3 Implement BCD (Binary Coded Decimal) data format encoding/decoding
- [x] 16.4 Implement discrete data word support
- [x] 16.5 Implement label-based filtering and routing
- [x] 16.6 Implement fixed-rate transmit scheduler (per-label configurable rate)
- [x] 16.7 Implement hardware interface abstraction (SPI transceiver, FPGA soft-IP)
- [x] 16.8 Implement ARINC 429-to-Zenoh transport adapter (label → `arinc429/{channel}/{label}`)
- [x] 16.9 Support both low speed (12.5 kbps) and high speed (100 kbps) configurations
- [x] 16.10 Unit tests for word codec (100% MC/DC, all data formats)
- [x] 16.11 Write TLA+ model for ARINC 429 transmit scheduling

## 17. Safety-Critical Bus Protocols — ARINC 664 (AFDX)

- [x] 17.1 Implement Virtual Link (VL) configuration (VL ID, BAG 1-128ms, Lmax)
- [x] 17.2 Implement BAG traffic shaping and policing
- [x] 17.3 Implement sequence number generation and checking (per VL)
- [x] 17.4 Implement dual-network redundancy management (integrity checking)
- [x] 17.5 Implement sub-VL scheduling (round-robin within a VL)
- [x] 17.6 Implement frame filtering (VL ID + destination MAC matching)
- [x] 17.7 Implement AFDX-to-Zenoh transport adapter (VL → `afdx/{vl_id}`)
- [x] 17.8 Integrate with existing Ethernet/IP stack
- [x] 17.9 Unit tests for VL scheduling and redundancy management
- [x] 17.10 Write TLA+ model for AFDX Virtual Link state machine

## 18. Safety-Critical Bus Protocols — MIL-STD-1553

- [x] 18.1 Implement command word encode/decode (RT address, subaddress, word count/mode code)
- [x] 18.2 Implement status word encode/decode (RT address, message error, busy, etc.)
- [x] 18.3 Implement data word encode/decode (16-bit payload, odd parity)
- [x] 18.4 Implement Bus Controller (BC) mode: command scheduling, response timeout
- [x] 18.5 Implement Remote Terminal (RT) mode: command recognition, response generation
- [x] 18.6 Implement dual-redundant bus management (Bus A / Bus B failover)
- [x] 18.7 Implement mode codes (transmit status, synchronize, reset, etc.)
- [x] 18.8 Implement hardware interface abstraction (dedicated 1553 transceiver)
- [x] 18.9 Implement MIL-1553-to-Zenoh transport adapter (RT/SA → `mil1553/{bus}/{rt}/{sa}`)
- [x] 18.10 Unit tests for word codec and protocol state machine (100% MC/DC)
- [x] 18.11 Write TLA+ model for 1553 command/response protocol

## 19. Safety-Critical Bus Protocols — SpaceWire

- [x] 19.1 Implement SpaceWire packet encode/decode (destination address, cargo, EOP/EEP)
- [x] 19.2 Implement link interface state machine (ErrorReset → ErrorWait → Ready → Started → Connecting → Run)
- [x] 19.3 Implement character-level encoding (data characters, control characters, time-codes)
- [x] 19.4 Implement time-code distribution (6-bit counter broadcast)
- [x] 19.5 Implement RMAP read/write commands (Remote Memory Access Protocol)
- [x] 19.6 Implement link speed configuration (2-400 Mbps)
- [x] 19.7 Implement hardware interface abstraction (LVDS PHY, FPGA codec IP)
- [x] 19.8 Implement SpaceWire-to-Zenoh transport adapter (dest → `spw/{link}/{dest}`)
- [x] 19.9 Unit tests for packet codec and link state machine
- [x] 19.10 Write TLA+ model for SpaceWire link state machine

## 20. Safety-Critical Bus Protocols — CCSDS Space Packet

- [x] 20.1 Implement CCSDS Space Packet primary header encode/decode (version, type, APID, sequence, length)
- [x] 20.2 Implement telemetry (TM) packet assembly and parsing
- [x] 20.3 Implement telecommand (TC) packet assembly and parsing
- [x] 20.4 Implement APID-based routing and filtering
- [x] 20.5 Implement TM transfer frame support (CCSDS 132.0-B)
- [x] 20.6 Implement TC transfer frame support (CCSDS 232.0-B)
- [x] 20.7 Implement CLTU encoding for uplink (optional, CCSDS 231.0-B)
- [x] 20.8 Implement CCSDS-to-Zenoh transport adapter (APID → `ccsds/{apid}`)
- [x] 20.9 Unit tests for packet codec (100% MC/DC)
- [x] 20.10 Validate against CCSDS Blue Book test vectors (where available)

## 21. DDS (Data Distribution Service)

- [x] 21.1 Define DDS DCPS core types: DomainParticipant, Topic, Publisher, Subscriber, DataWriter, DataReader
- [x] 21.2 Implement DomainParticipant creation, configuration, and lifecycle management
- [x] 21.3 Implement Topic definition and type registration with type consistency enforcement
- [x] 21.4 Implement DataWriter with write, dispose, unregister operations
- [x] 21.5 Implement DataReader with read, take, wait-set, and listener notification
- [x] 21.6 Implement CDR (Common Data Representation) v2 serializer and deserializer
- [x] 21.7 Implement RTPS 2.3 message format: Header, Submessages (DATA, HEARTBEAT, ACKNACK, GAP, INFO_TS)
- [x] 21.8 Implement SPDP (Simple Participant Discovery Protocol) with multicast announcements
- [x] 21.9 Implement SEDP (Simple Endpoint Discovery Protocol) for DataWriter/DataReader matching
- [x] 21.10 Implement RTPS reliable delivery: heartbeat/acknack protocol, sample retransmission
- [x] 21.11 Implement RTPS best-effort delivery with sequence number tracking
- [x] 21.12 Implement QoS policies: RELIABILITY (reliable/best-effort), DURABILITY (volatile/transient-local)
- [x] 21.13 Implement QoS policies: DEADLINE, LIVELINESS (automatic/manual), OWNERSHIP (shared/exclusive)
- [x] 21.14 Implement QoS policies: HISTORY (keep-last/keep-all), RESOURCE_LIMITS (max_samples, max_instances)
- [x] 21.15 Implement QoS compatibility checking between DataWriter offers and DataReader requests
- [x] 21.16 Implement DDS-Security authentication plugin (mutual auth with ML-DSA-65 certificates)
- [x] 21.17 Implement DDS-Security access control plugin (governance and permissions documents)
- [x] 21.18 Implement DDS-to-Zenoh transport adapter (domain/topic → `dds/{domain_id}/{topic}`)
- [x] 21.19 Implement ROS 2 topic name mangling support (`rt/`, `rq/`, `rr/` prefixes)
- [x] 21.20 Unit tests for CDR serialization (all primitive types, structs, sequences, strings)
- [x] 21.21 Unit tests for RTPS message encode/decode (100% MC/DC)
- [x] 21.22 Unit tests for SPDP/SEDP discovery state machines
- [x] 21.23 Unit tests for QoS compatibility matrix (all valid/invalid combinations)
- [x] 21.24 Integration test: DDS pub/sub within SmallAIOS (loopback)
- [ ] 21.25 Interoperability test: SmallAIOS DDS ↔ FastDDS (ROS 2 node) on same network

## 22. QUIC Transport

- [x] 22.1 Define QUIC connection, stream, and endpoint types in `smallaios-net` crate
- [x] 22.2 Implement QUIC packet encoding/decoding: Initial, Handshake, 0-RTT, 1-RTT packet types
- [x] 22.3 Implement QUIC frame encoding/decoding: STREAM, ACK, CRYPTO, NEW_CONNECTION_ID, PATH_CHALLENGE/RESPONSE, MAX_DATA, MAX_STREAM_DATA, etc.
- [ ] 22.4 Integrate TLS 1.3 handshake with ML-KEM-768 hybrid key exchange (client and server)
- [x] 22.5 Implement QUIC packet protection: AES-128-GCM and ChaCha20-Poly1305 AEAD, header protection
- [x] 22.6 Implement 1-RTT connection establishment (client and server roles)
- [x] 22.7 Implement 0-RTT session resumption with TLS session tickets and replay protection
- [x] 22.8 Implement bidirectional and unidirectional stream management (open, read, write, close, reset)
- [x] 22.9 Implement stream-level and connection-level flow control (MAX_DATA, MAX_STREAM_DATA)
- [x] 22.10 Implement loss detection and congestion control per RFC 9002 (NewReno)
- [x] 22.11 Implement connection migration with PATH_CHALLENGE/PATH_RESPONSE validation
- [x] 22.12 Implement connection ID management (NEW_CONNECTION_ID, RETIRE_CONNECTION_ID)
- [x] 22.13 Implement key update mechanism (RFC 9001 Section 6)
- [x] 22.14 Implement QUIC version negotiation
- [x] 22.15 Implement Zenoh session transport adapter (quic/ locator scheme)
- [x] 22.16 Implement standalone QUIC endpoint API (server and client)
- [x] 22.17 Implement minimal HTTP/3 framing (RFC 9114) for management API (GET/POST /health, /metrics, /deploy)
- [x] 22.18 Unit tests for packet encoding/decoding (RFC 9000 Appendix A test vectors)
- [x] 22.19 Unit tests for frame encoding/decoding (all frame types)
- [x] 22.20 Unit tests for flow control and congestion control state machines
- [x] 22.21 Unit tests for 0-RTT replay protection
- [x] 22.22 Integration test: QUIC connection establishment and data transfer (loopback)
- [x] 22.23 Integration test: Zenoh pub/sub over QUIC transport
- [x] 22.24 Integration test: connection migration (simulated IP address change)
- [ ] 22.25 Interoperability test: SmallAIOS QUIC ↔ external QUIC implementation (e.g., quiche, quinn)

## 23. RISC-V Architecture Support

- [x] 23.1 Create `smallaios-arch-riscv64` crate with target riscv64gc-unknown-none-elf
- [x] 23.2 Create custom target JSON and linker script for RISC-V bare metal
- [x] 23.3 Implement assembly entry point (set stack, clear BSS, set stvec, call kernel_main)
- [x] 23.4 Implement SV48 4-level page table management (map, unmap, protect)
- [x] 23.5 Implement TLB flush via sfence.vma instruction
- [x] 23.6 Implement PLIC interrupt controller driver (priority, enable, claim, complete)
- [x] 23.7 Implement CLINT timer driver (mtime/mtimecmp for periodic tick)
- [x] 23.8 Implement SBI HSM extension calls for SMP boot (hart_start, hart_stop, hart_get_status)
- [x] 23.9 Implement SBI IPI extension for inter-hart interrupts
- [x] 23.10 Implement NS16550A UART driver (QEMU virt machine console output)
- [x] 23.11 Implement CPU feature detection (A, C, D, F extensions)
- [x] 23.12 Verify boot-to-serial-output in QEMU virt (riscv64)
- [x] 23.13 Add RISC-V to CI (build + QEMU smoke test)
- [x] 23.14 Add RISC-V HAL implementation to kernel HAL trait

## 24. SoC FPGA Platform Support

- [x] 24.1 Create `smallaios-fpga` crate with feature flags (zynq, polarfire)
- [x] 24.2 Implement AXI4-Lite memory-mapped register read/write driver
- [x] 24.3 Implement AXI4 full burst transfer driver
- [x] 24.4 Implement AXI DMA controller driver (simple mode)
- [x] 24.5 Implement AXI DMA scatter-gather mode
- [x] 24.6 Implement DTB-based peripheral discovery for FPGA soft-IP blocks
- [x] 24.7 Implement interrupt routing from FPGA fabric to CPU interrupt controller
- [x] 24.8 Implement Zynq UltraScale+ platform support package (PS-PL interface, clocks, resets)
- [x] 24.9 Implement PolarFire SoC platform support package (RISC-V + FPGA fabric)
- [x] 24.10 Unit tests for AXI register access and DMA transfers (mock MMIO)
- [ ] 24.11 Integration test on Zynq board: read/write FPGA registers from SmallAIOS

## 25. Deployment and Provisioning (Phase 11 Revision)

- [x] 25.1 Implement UEFI Secure Boot support with ML-DSA-65 signed kernel image
- [x] 25.2 Implement PXE/iPXE bare metal network boot provisioning
- [x] 25.3 Implement VM image generation (raw disk, qcow2, VMDK formats)
- [x] 25.4 Implement image signing with ML-DSA-65 post-quantum signatures
- [x] 25.5 Update OCI image build for multi-arch (amd64, arm64, riscv64)
- [x] 25.6 Verify image size target < 15 MB (kernel + runtime, no model)

## 26. IPC Transport Extensions

- [x] 26.1 Extend ZenohRouter with bus transport registration interface
- [x] 26.2 Implement transport auto-discovery from DTB/ACPI peripheral enumeration
- [x] 26.3 Implement cross-transport routing (message on CAN routed to TCP subscriber, DDS to CAN, QUIC to ARINC, etc.)
- [x] 26.4 Unit tests for transport-agnostic pub/sub across mixed transports (including DDS and QUIC)
- [x] 26.5 Integration test: inference result published over TCP, delivered over CAN
- [x] 26.6 Integration test: DDS topic bridged to ARINC 429 via Zenoh router
- [x] 26.7 Integration test: Zenoh session over QUIC with connection migration

## 27. HAL Extensions

- [x] 27.1 Define BusPeripheral HAL trait (init, tx_frame, rx_frame, irq_handler)
- [x] 27.2 Define FpgaFabric HAL trait (axi_read, axi_write, dma_transfer)
- [x] 27.3 Implement RISC-V HAL (paging, PLIC, CLINT, SBI wrappers)
- [x] 27.4 Update hardware platform spec with RISC-V and SoC FPGA tier 2 targets
- [x] 27.5 Add riscv64gc-unknown-none-elf build target to Makefile and CI

## 28. Benchmark Infrastructure

- [x] 28.1 Create bench/ directory structure (scripts, configs, models, results)
- [x] 28.2 Implement cold start measurement harness (power-on to first inference)
- [x] 28.3 Implement warm inference latency harness (N=1000+ runs, p50/p99/p999)
- [x] 28.4 Implement throughput benchmark (batch sizes 1, 4, 16, 64)
- [x] 28.5 Implement jitter measurement (stdev and max deviation)
- [x] 28.6 Implement memory footprint measurement (peak RSS)
- [x] 28.7 Download and prepare benchmark models: MobileNetV2 (vision), DistilBERT (text), Whisper-tiny (audio/signal)
- [x] 28.8 Create Linux bare metal baseline scripts (ONNX Runtime C++ binary)
- [x] 28.9 Create Docker baseline (Dockerfile + ONNX Runtime)
- [x] 28.10 Create K8s/K3s baseline (deployment manifests + ONNX Runtime pod)
- [x] 28.11 Create hardware-specific configs (DGX Spark, Xeon, Jetson, RPi)
- [x] 28.12 Document BIOS settings, CPU frequency pinning, thermal monitoring
- [x] 28.13 Implement report generation (markdown tables, CSV export)
- [ ] 28.14 Run benchmark suite on all 4 hardware targets, generate comparison report

## 29. Formal Verification — Bus Protocols, DDS, and QUIC

- [x] 29.1 Write TLA+ model for CAN bus arbitration (priority-based, no starvation)
- [x] 29.2 Write TLA+ model for ARINC 429 transmit scheduler (no label starvation)
- [x] 29.3 Write TLA+ model for AFDX Virtual Link BAG enforcement (no deadline miss)
- [x] 29.4 Write TLA+ model for MIL-STD-1553 command/response (no unanswered commands)
- [x] 29.5 Write TLA+ model for SpaceWire link state machine (correct transitions)
- [x] 29.6 Write TLA+ model for DDS RTPS reliable delivery (no sample loss, no reordering)
- [x] 29.7 Write TLA+ model for DDS SPDP/SEDP discovery (eventual consistency, no orphan endpoints)
- [x] 29.8 Write TLA+ model for QUIC connection migration (no data loss during path switch)
- [x] 29.9 Write TLA+ model for QUIC flow control (no deadlock, no limit violation)
- [x] 29.10 Create TLC configs for all protocol models
- [x] 29.11 Add protocol TLA+ verification to CI
