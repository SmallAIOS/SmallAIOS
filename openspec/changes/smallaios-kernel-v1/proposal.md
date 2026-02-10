## Why

General-purpose operating systems carry enormous unnecessary complexity for AI inference workloads — thousands of syscalls, package managers, shells, and display servers create massive attack surfaces, resource overhead, and operational burden. No existing OS is purpose-built to boot and execute ONNX models with minimal overhead, safety-critical assurance, and native hardware acceleration across x86-64, ARM64, and NVIDIA GPU platforms.

## What Changes

- Introduces a clean-room Rust unikernel designed exclusively for ONNX inference
- Provides a minimal POSIX subset (~5% of POSIX) for AI runtime compatibility
- Implements native ONNX model parsing, graph optimization, and multi-target execution
- Builds a Zenoh-inspired pub/sub IPC system as the sole external interface
- Integrates post-quantum cryptography (ML-KEM-768, ML-DSA-65) natively
- Adds native IPv4/IPv6 networking stack for IPC transport
- Supports x86-64 (AVX2/AVX-512/AMX), ARM64 (NEON/SVE), NVIDIA GPU (Volta through Blackwell)
- Targets container-native deployment (Docker/Kubernetes) with < 15 MB images
- **Applies safety-critical development standards (MISRA-C:2023 adapted for Rust, DO-178C DAL A, MC/DC 100% coverage)**
- **Uses formal verification (TLA+ for concurrency, Lean 4 for type-level proofs, SPIN for protocol verification)**
- **Produces Sphinx-needs traceability documentation with PlantUML architecture diagrams**

## Capabilities

### New Capabilities
- `kernel-core`: Unikernel kernel with buddy/slab allocators, cooperative async scheduler, ~46 syscall interface
- `posix-compat`: Minimal POSIX subset — file descriptors, mmap, pthreads, sockets, epoll, clock/time
- `onnx-runtime`: Native ONNX parser, graph optimizer (fusion, constant folding, memory planning), CPU and CUDA execution providers
- `ipc-messaging`: Zenoh-inspired pub/sub with key-expression routing, request/reply queryables, shared memory and TCP transports
- `device-hal`: Hardware abstraction layer for x86-64 (GDT/IDT/APIC/paging), ARM64 (GIC/MMU/PSCI), NVIDIA GPU (PCIe/VRAM/compute/DMA)
- `security-model`: Capability-based access control with no ambient authority, security audit logging
- `pqc-crypto`: Post-quantum cryptography — ML-KEM-768, ML-DSA-65, SLH-DSA, hybrid mode, AES-256-GCM, SHA-3, TLS 1.3 with PQC
- `networking`: Native IPv4/IPv6 TCP/IP stack with ARP, NDP, SLAAC, CUBIC congestion control, built-in firewall
- `container-interface`: OCI image support, Kubernetes health/readiness probes, Prometheus metrics, graceful shutdown
- `safety-critical`: MISRA-Rust coding standards, DO-178C DAL A process compliance, MC/DC 100% structural coverage, requirements traceability
- `formal-verification`: TLA+ models for scheduler and memory allocator concurrency, Lean 4 proofs for type-level invariants, SPIN models for IPC protocol verification
- `documentation`: Sphinx-needs requirements traceability, PlantUML architecture and sequence diagrams, cross-referenced spec-to-code-to-test mapping

### Modified Capabilities

(None — this is a greenfield project)

## Impact

- `kernel/` — Core kernel crate (memory, scheduler, syscall dispatch)
- `arch/x86_64/` — x86-64 hardware abstraction layer
- `arch/aarch64/` — ARM64 hardware abstraction layer
- `arch/nvidia/` — NVIDIA GPU driver and compute engine
- `onnx-rt/` — ONNX runtime (parser, optimizer, execution providers)
- `ipc/` — IPC messaging system
- `net/` — Native TCP/IP network stack
- `posix/` — POSIX compatibility layer
- `security/` — Capability system and PQC cryptography
- `container/` — Container entry point, health, metrics
- `formal/` — TLA+, Lean 4, SPIN formal models
- `docs/` — Sphinx-needs documentation with PlantUML diagrams
- `Cargo.toml` — Workspace configuration
- `Makefile` — Build system
- `docker/` — Container image definitions
