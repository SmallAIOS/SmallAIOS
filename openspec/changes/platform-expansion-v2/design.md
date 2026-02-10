## Context

SmallAIOS is a clean-room Rust unikernel for ONNX inference, currently targeting x86-64 and ARM64 with TCP/Zenoh networking. Phases 1-3 (bootstrap, memory management, scheduler) are complete with 79 passing tests. The project uses DO-178C DAL A safety processes, formal verification (TLA+, Lean 4, SPIN), and deterministic cooperative scheduling with three priority classes (SYSTEM > IPC > INFERENCE).

This expansion targets automotive, aviation, defense, and space deployment domains. These domains require deterministic bus protocols (CAN, ARINC 429/664, MIL-STD-1553, SpaceWire, CCSDS), SoC FPGA platforms, RISC-V architecture support, Kubernetes orchestration for datacenter/edge, and rigorous performance benchmarking against Linux baselines.

Key constraint: all protocol implementations must be clean-room from public specifications (ISO 11898, ARINC standards, MIL-STD-1553B, ECSS-E-ST-50-12C, CCSDS Blue Books) — no GPL code, no vendor SDK dependencies.

## Goals / Non-Goals

**Goals:**
- Unified transport abstraction: all bus protocols appear as Zenoh transports (pub/sub, request/reply)
- RISC-V as third architecture target (RV64GC, SV48 paging, PLIC/CLINT)
- SoC FPGA support where SmallAIOS runs on embedded CPU cores, FPGA fabric provides bus peripherals
- Virtual Kubelet-based K8s integration keeping orchestration outside safety-critical boundary
- Benchmark infrastructure proving cold-start and jitter advantages over Linux on real hardware
- Three inference modalities benchmarked: vision (MobileNetV2), text (DistilBERT), audio/signal (Whisper-tiny)

**Non-Goals:**
- FPGA-based inference acceleration (ONNX-to-bitstream compilation) — future work
- Full kubelet implementation inside SmallAIOS — complexity belongs on Linux side
- Proprietary bus protocol extensions (CANopen, J1939, DeviceNet) — only base protocols
- RISC-V vector extension (RVV) GEMM kernels — defer until ecosystem matures
- Full HTTP server — keep minimal /health and /metrics only

## Decisions

### 1. Bus Protocol Architecture: Zenoh Transport Adapters

**Decision**: Each bus protocol implements a `ZenohTransport` trait that maps protocol-native addressing to Zenoh key expressions.

**Rationale**: The IPC layer already handles pub/sub and request/reply patterns. By making CAN frames, ARINC 429 words, and MIL-STD-1553 messages appear as Zenoh samples, application code is transport-agnostic. An inference result can be published once and delivered over TCP (datacenter), CAN bus (automotive), or ARINC 429 (avionics) without changing the inference pipeline.

**Alternatives considered**:
- Separate API per bus: rejected — code duplication, forces applications to know transport
- Raw driver-only approach: rejected — loses the Zenoh routing and key expression benefits

**Mapping scheme**:
```
CAN:       can/{bus_id}/{frame_id}      → Zenoh key expression
ARINC 429: arinc429/{channel}/{label}   → Zenoh key expression
ARINC 664: afdx/{vl_id}                → Zenoh key expression
MIL-1553:  mil1553/{bus}/{rt}/{sa}      → Zenoh key expression
SpaceWire: spw/{link}/{dest}            → Zenoh key expression
CCSDS:     ccsds/{apid}                 → Zenoh key expression
```

### 2. Bus Protocol Crate Structure: Single `bus` Crate with Feature Flags

**Decision**: Create a single `smallaios-bus` crate with feature flags per protocol (`can`, `arinc429`, `arinc664`, `mil1553`, `spacewire`, `ccsds`).

**Rationale**: Bus protocols share common infrastructure (Zenoh transport trait, frame buffering, error types, scheduling primitives). A single crate avoids dependency explosion while feature flags keep binary size minimal — an automotive deployment only compiles CAN, an avionics deployment only ARINC 429/664.

**Alternatives considered**:
- One crate per protocol: rejected — too many workspace crates (already 10), shared types would need yet another crate
- Everything in kernel crate: rejected — bus protocols are optional, shouldn't bloat the core

### 3. RISC-V Architecture: Separate `arch/riscv64` Crate

**Decision**: Add `smallaios-arch-riscv64` following the same pattern as x86_64 and aarch64 crates.

**Rationale**: Consistent architecture abstraction. RISC-V uses SBI (Supervisor Binary Interface) for firmware calls analogous to x86 BIOS/UEFI and ARM PSCI. OpenSBI provides M-mode firmware; SmallAIOS runs in S-mode.

**Key differences from ARM64**:
- Interrupt controller: PLIC (external) + CLINT (timer/IPI) vs. GICv3
- Page tables: SV48 (4-level, 4 KiB pages) — structurally identical to ARM64/x86-64
- SMP: SBI HSM extension (hart_start/hart_stop) vs. PSCI CPU_ON
- Timer: mtime/mtimecmp (CLINT) vs. CNTP_TVAL_EL0

### 4. SoC FPGA: Platform Support Package, Not FPGA Programming

**Decision**: SmallAIOS treats FPGA peripherals as memory-mapped devices discovered via device tree. We provide AXI register access and DMA drivers, not FPGA bitstream tooling.

**Rationale**: FPGA bitstreams are created offline by hardware engineers using vendor tools (Vivado, Libero). SmallAIOS only needs to talk to whatever peripherals the FPGA fabric exposes via AXI bus — the same way it talks to any MMIO device. This keeps SmallAIOS vendor-neutral.

**Platform targets**:
- Xilinx/AMD Zynq UltraScale+ (Cortex-A53 + FPGA, common in aerospace/defense)
- Microchip PolarFire SoC (RISC-V + FPGA, radiation-tolerant, gaining space traction)

### 5. Kubernetes Integration: Virtual Kubelet Provider

**Decision**: K8s integration via a Go-based Virtual Kubelet provider that runs on a Linux control node. SmallAIOS exposes a management API over Zenoh/TCP that the provider translates to/from K8s API.

**Rationale**:
- Keeps K8s complexity (client-go, informers, leader election) outside the certified safety boundary
- SmallAIOS stays a minimal unikernel — no Go runtime, no HTTP/2, no etcd client
- The management API is a small Zenoh queryable extension to existing IPC endpoints
- Supports both K8s (datacenter) and K3s (edge) since both speak the same API

**SmallAIOS management API endpoints (Zenoh)**:
```
smallaios/v1/management/deploy    — deploy ONNX model
smallaios/v1/management/undeploy  — remove model
smallaios/v1/management/status    — node status + resources
smallaios/v1/management/config    — runtime configuration
```

### 6. Benchmark Methodology: Cold Start Focus with Statistical Rigor

**Decision**: Benchmark framework measures boot-to-first-inference as primary metric, with warm latency, throughput, and jitter as secondary. All measurements use N=1000+ runs with percentile reporting.

**Rationale**: Cold start is SmallAIOS's killer differentiator. A unikernel booting in milliseconds vs. Linux + container in seconds is the core value proposition for edge/embedded deployment.

**Three models chosen for modality coverage**:
- MobileNetV2 (vision): 14 MB, CPU-bound, tests GEMM/Conv operators
- DistilBERT (text): ~250 MB, attention-heavy, tests operator budgets and memory
- Whisper-tiny (audio): ~150 MB, encoder-decoder, representative of radar/signal processing

**Hardware matrix**:
- DGX Spark: datacenter GPU reference (Blackwell B200)
- Intel Xeon: datacenter CPU reference
- Jetson Orin Nano: edge GPU reference
- Raspberry Pi 4/5: edge CPU-only reference

**Linux baselines (per hardware target)**:
- Bare metal Linux + ONNX Runtime (C++ binary)
- Docker container + ONNX Runtime
- K8s/K3s pod + ONNX Runtime

### 7. DDS Implementation: Core DCPS + RTPS, Zenoh Bridge

**Decision**: Implement core OMG DDS DCPS API and RTPS 2.3 wire protocol from public OMG specifications, bridged to Zenoh via a transport adapter. DDS-Security uses SmallAIOS's existing post-quantum crypto. DDS-XTypes and advanced extensible types are deferred.

**Rationale**: DDS is the mandated middleware for ROS 2 (robotics), AUTOSAR Adaptive Platform (automotive), FACE/GVA/NGVA (defense), and many space/industrial programs. Wire-level RTPS interoperability is essential — SmallAIOS must communicate directly with FastDDS, CycloneDDS, and RTI Connext nodes without protocol translation gateways.

The Zenoh bridge architecture is natural: DDS's data-centric pub/sub model maps directly to Zenoh's key-expression-based pub/sub. Domain/topic pairs become `dds/{domain_id}/{topic}` key expressions. This enables DDS topics to be routed to/from CAN, ARINC, SpaceWire, and other Zenoh transports transparently.

**Alternatives considered**:
- Full DDS stack replacing Zenoh: rejected — DDS is heavyweight (100K+ LOC in production implementations), SmallAIOS IPC is Zenoh-native, DDS is one transport among many
- DDS gateway as external process: rejected — adds latency, another deployment artifact, breaks unikernel model
- ROS 2 rmw_zenoh only (skip DDS): rejected — doesn't cover AUTOSAR, FACE, or non-ROS DDS deployments

**Scope boundaries**:
- IN: DCPS API (DomainParticipant, Topic, DataWriter, DataReader, Publisher, Subscriber), RTPS 2.3 (SPDP, SEDP, reliable/best-effort), CDR v2 serialization, core QoS policies (reliability, durability, deadline, liveliness, ownership, history, resource_limits), DDS-Security (authentication, access control)
- OUT: DDS-XTypes (extensible types), content-filtered topics, multi-topic subscriptions, DDS-Security crypto plugin (SmallAIOS provides its own ML-KEM/ML-DSA), query conditions

**Key expression mapping**:
```
DDS: dds/{domain_id}/{topic}  → Zenoh key expression
ROS 2: dds/{domain_id}/rt/{ros_topic}  → Zenoh key expression (ROS 2 topic mangling)
```

## Risks / Trade-offs

**[Risk] Protocol specification access** → Some standards (ARINC 429, ARINC 664, MIL-STD-1553) are behind paywalls. Mitigation: clean-room implementation from publicly available technical descriptions, cross-reference with open-source implementations for correctness (but no code copying). CAN (ISO 11898) and CCSDS Blue Books are freely available.

**[Risk] RISC-V ecosystem immaturity** → RISC-V toolchain and QEMU support are less mature than x86/ARM. Mitigation: target QEMU `virt` machine first (well-supported), real hardware (PolarFire) as Tier 2. Use stable RV64GC without vector extensions.

**[Risk] Virtual Kubelet maintenance burden** → Go codebase separate from Rust core. Mitigation: Virtual Kubelet provider is a thin translation layer (~2000 LOC), not safety-critical, can lag behind K8s versions without affecting SmallAIOS kernel.

**[Risk] Benchmark reproducibility** → Results depend on BIOS settings, thermal state, background noise. Mitigation: document exact BIOS/firmware settings, use CPU frequency pinning, measure and report thermal throttling, run overnight batches for stability.

**[Risk] Bus protocol hardware availability for testing** → CAN controllers, ARINC transceivers, MIL-STD-1553 couplers are specialized hardware. Mitigation: implement protocol layers in software first, test with loopback and mock devices, QEMU virtio-can for CAN, validate on real hardware as available.

**[Risk] Scope creep into higher-level protocols** → CAN → CANopen → J1939, ARINC 429 → ARINC 615A (data loading), DDS → DDS-XTypes, etc. Mitigation: explicitly scope to base protocol layers only. Application-layer protocols and advanced type systems are future work.

**[Risk] DDS specification complexity** → Full OMG DDS is large (DCPS, RTPS, DDS-Security, DDS-XTypes, IDL). Mitigation: implement core DCPS API and RTPS 2.3 wire protocol only (enough for ROS 2 and AUTOSAR Adaptive interop). DDS-XTypes deferred. DDS-Security scoped to authentication and access control plugins — crypto plugin not needed since SmallAIOS already provides ML-KEM/ML-DSA at the transport layer.

**[Risk] DDS interoperability testing** → RTPS wire-level compatibility with FastDDS, CycloneDDS, RTI Connext requires careful testing. Mitigation: use ROS 2 as primary interop test target (most accessible), validate against OMG RTPS interoperability test suite (publicly available).

**[Trade-off] Single `bus` crate vs. per-protocol crates** → Single crate is simpler but couples protocol release cycles. Acceptable because all protocols share the Zenoh transport trait and are versioned together with SmallAIOS.

**[Trade-off] Virtual Kubelet vs. embedded kubelet** → Virtual Kubelet adds a separate Go component but keeps the safety boundary clean. An embedded kubelet would be more self-contained but would require HTTP/2 + protobuf + complex state management inside the unikernel.

## Open Questions

1. **RISC-V interrupt controller**: PLIC vs. AIA (Advanced Interrupt Architecture)? AIA is newer and better for SMP but less widely supported. Starting with PLIC seems safest.

2. **CAN FD vs. CAN XL**: CAN XL (up to 2048 bytes) is emerging but not yet widely deployed. Include as stretch goal?

3. **SpaceWire RMAP**: How much of RMAP to implement? Full spec or just enough for register read/write to FPGA peripherals?

4. **Benchmark model licensing**: MobileNetV2 (Apache 2.0), DistilBERT (Apache 2.0), Whisper (MIT). All compatible. Confirm model weights can be redistributed with benchmark suite.

5. **PolarFire SoC availability**: Microchip Icicle Kit is the dev board. Confirm hardware access for RISC-V + FPGA testing.

6. **DDS conformance level**: Full OMG DDS compliance requires passing the DDS interoperability test suite. Define which conformance profile to target (Minimum, Content, Complete)?

7. **DDS + ROS 2 topic name mangling**: ROS 2 prefixes topic names with `rt/`, `rq/`, `rr/` for topics, requests, replies. Support this convention natively or via configuration?
