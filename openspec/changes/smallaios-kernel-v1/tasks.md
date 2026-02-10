## 1. Project Bootstrap

- [ ] 1.1 Set up Cargo workspace with all 10 crates and build targets (x86_64-unknown-none, aarch64-unknown-none, x86_64-unknown-linux-musl, aarch64-unknown-linux-musl)
- [ ] 1.2 Create custom target JSON specs and linker scripts for bare metal x86-64 and ARM64
- [ ] 1.3 Implement x86-64 assembly entry point (stack setup, BSS clear, GDT, call kernel_main)
- [ ] 1.4 Implement ARM64 assembly entry point (MMU disable, stack setup, exception vectors, call kernel_main)
- [ ] 1.5 Implement minimal serial console output (COM1 for x86, PL011 UART for ARM64)
- [ ] 1.6 Verify boot-to-serial-output in QEMU for both architectures
- [ ] 1.7 Set up CI (GitHub Actions: build both archs, QEMU smoke test, clippy, rustfmt)
- [ ] 1.8 Initialize Sphinx-needs documentation project with PlantUML integration
- [ ] 1.9 Create initial TLA+ model skeleton for scheduler and allocator
- [ ] 1.10 Establish MISRA-Rust coding standard document and clippy configuration

## 2. Core Kernel — Memory Management

- [ ] 2.1 Parse physical memory map from Multiboot2 (x86) and DTB (ARM64)
- [ ] 2.2 Implement buddy allocator (orders 0-21, 4 KiB to 8 GiB, huge page support)
- [ ] 2.3 Implement slab allocator with per-CPU caches (size classes 16B-2048B)
- [ ] 2.4 Implement global allocator trait for Rust's alloc crate
- [ ] 2.5 Implement tensor memory pool (arena + bump pointer + reference counting)
- [ ] 2.6 Implement x86-64 4-level page table management (map, unmap, protect)
- [ ] 2.7 Implement ARM64 4-level page table management (4 KiB granule)
- [ ] 2.8 Write TLA+ spec for buddy allocator (conservation property, no double-free)
- [ ] 2.9 Run TLC model checker on allocator spec — verify all safety properties
- [ ] 2.10 Achieve 100% MC/DC coverage on allocator with unit tests
- [ ] 2.11 Trace requirements: REQ → SPEC → IMPL → TEST in Sphinx-needs

## 3. Core Kernel — Scheduler and Interrupts

- [ ] 3.1 Implement task struct (state machine, stack, future, priority, affinity)
- [ ] 3.2 Implement per-CPU lock-free run queues (LIFO push, FIFO steal)
- [ ] 3.3 Implement async executor (poll futures, wake on events, work-stealing)
- [ ] 3.4 Implement x86-64 APIC timer and IPI interrupts
- [ ] 3.5 Implement ARM64 GICv3 and Generic Timer interrupts
- [ ] 3.6 Implement SMP boot (SIPI for x86, PSCI for ARM64)
- [ ] 3.7 Write TLA+ spec for scheduler (deadlock freedom, starvation freedom, fairness)
- [ ] 3.8 Run TLC model checker — verify scheduler liveness and safety properties
- [ ] 3.9 Achieve 100% MC/DC coverage on scheduler with unit tests
- [ ] 3.10 Integration test: spawn 1000 tasks in QEMU, verify all complete

## 4. Core Kernel — Syscall Interface

- [ ] 4.1 Implement syscall dispatch table (46 entries, function pointer array)
- [ ] 4.2 Implement x86-64 syscall/sysret handler (for VM mode)
- [ ] 4.3 Implement ARM64 svc handler (for VM mode)
- [ ] 4.4 Implement all memory syscalls (mem_alloc/free, tensor_alloc/free/map_gpu)
- [ ] 4.5 Implement all task syscalls (spawn, yield, exit, join, affinity, priority)
- [ ] 4.6 Implement all system syscalls (info, time, shutdown, log, random, config)
- [ ] 4.7 Implement capability syscalls (create, revoke, delegate, check, list)
- [ ] 4.8 Fuzz syscall interface with invalid inputs — verify no panics
- [ ] 4.9 Achieve 100% MC/DC coverage on syscall dispatch

## 5. Security — Capabilities and Cryptography

- [ ] 5.1 Implement capability token type (id, resource ref, permissions bitmask, expiry)
- [ ] 5.2 Implement capability registry (creation, delegation, revocation, lookup)
- [ ] 5.3 Integrate capability checks into all syscalls
- [ ] 5.4 Implement AES-256-GCM (with AES-NI / ARMv8 Crypto hardware acceleration)
- [ ] 5.5 Implement SHA-3-256 and SHAKE256
- [ ] 5.6 Implement ML-KEM-768 (Kyber) key encapsulation per FIPS 203
- [ ] 5.7 Implement ML-DSA-65 (Dilithium) digital signatures per FIPS 204
- [ ] 5.8 Implement hybrid mode (X25519+ML-KEM, Ed25519+ML-DSA)
- [ ] 5.9 Implement CSPRNG (SHAKE256-based, seeded from RDRAND/RNDR)
- [ ] 5.10 Implement ONNX model signature verification (ML-DSA-65)
- [ ] 5.11 Verify constant-time properties with dudect-style statistical testing
- [ ] 5.12 Write Lean 4 proofs for capability non-forgery invariant
- [ ] 5.13 Validate crypto implementations against NIST test vectors

## 6. POSIX Compatibility Layer

- [ ] 6.1 Implement file descriptor table and lifecycle management
- [ ] 6.2 Implement read-only virtual filesystem (/models/, /config/, /dev/, /proc/self/)
- [ ] 6.3 Implement mmap/munmap/mprotect (MAP_ANONYMOUS, MAP_PRIVATE, MAP_HUGETLB)
- [ ] 6.4 Implement pthreads subset (create, join, mutex, condvar, rwlock)
- [ ] 6.5 Implement epoll (create1, ctl, wait)
- [ ] 6.6 Implement socket API subset (TCP client/server)
- [ ] 6.7 Implement clock_gettime, nanosleep, getrandom
- [ ] 6.8 Implement SIGTERM handler for graceful shutdown
- [ ] 6.9 Return ENOSYS for all unimplemented POSIX calls
- [ ] 6.10 Test Rust std library operations against POSIX layer

## 7. ONNX Runtime

- [ ] 7.1 Implement minimal protobuf decoder (varint, length-delimited, fixed32/64)
- [ ] 7.2 Code-generate Rust structs from onnx.proto3 (build script)
- [ ] 7.3 Implement model validation (opset, operators, shapes, DAG check)
- [ ] 7.4 Implement execution graph builder (topological sort, input/output resolution)
- [ ] 7.5 Implement graph optimizer: constant folding, operator fusion, DCE
- [ ] 7.6 Implement memory planner (tensor lifetimes, buffer reuse, peak computation)
- [ ] 7.7 Implement Tier 1 CPU operators: MatMul, Conv, Relu, Softmax, Add, Reshape, etc.
- [ ] 7.8 Implement AVX2 GEMM micro-kernel (8x8 f32 register tile, cache-blocked)
- [ ] 7.9 Implement NEON GEMM micro-kernel (8x8 f32)
- [ ] 7.10 Implement session API (load_model, create_session, run)
- [ ] 7.11 End-to-end test: MobileNetV2 inference produces correct classification
- [ ] 7.12 Fuzz protobuf parser — verify no panics on random input
- [ ] 7.13 Benchmark GEMM against reference (target: within 2x of OpenBLAS)
- [ ] 7.14 Write SPIN model for inference pipeline (verify no deadlock under concurrent requests)

## 8. Networking

- [ ] 8.1 Implement Ethernet frame handling (send, receive, MAC addressing)
- [ ] 8.2 Implement ARP (table, request/reply, gratuitous ARP)
- [ ] 8.3 Implement IPv4 (header, checksum, routing to default gateway)
- [ ] 8.4 Implement IPv6 (header, link-local address generation)
- [ ] 8.5 Implement NDP (neighbor solicitation/advertisement, router solicitation)
- [ ] 8.6 Implement SLAAC (process RAs, generate global address)
- [ ] 8.7 Implement ICMPv4/v6 (echo request/reply)
- [ ] 8.8 Implement TCP (state machine, 3-way handshake, data transfer, CUBIC, SACK)
- [ ] 8.9 Implement UDP (send/receive, port multiplexing)
- [ ] 8.10 Implement built-in packet filter / firewall
- [ ] 8.11 Implement virtio-net driver (MMIO transport)
- [ ] 8.12 Write SPIN model for TCP state machine — verify correct transitions
- [ ] 8.13 Test IPv4 and IPv6 connectivity in QEMU with virtio-net

## 9. IPC and Messaging

- [ ] 9.1 Implement key expression parser and wildcard matcher (* and **)
- [ ] 9.2 Implement message router (key-expression → subscription dispatch)
- [ ] 9.3 Implement pub/sub pattern (publisher, subscriber, delivery)
- [ ] 9.4 Implement request/reply pattern (queryable, query, reply routing)
- [ ] 9.5 Implement shared memory transport (lock-free SPSC ring buffers)
- [ ] 9.6 Implement TCP transport with Zenoh wire protocol framing
- [ ] 9.7 Implement TLS 1.3 transport with ML-KEM-768 hybrid key exchange
- [ ] 9.8 Implement binary inference request/response protocol
- [ ] 9.9 Implement built-in endpoints (health, metrics, models, inference, logs)
- [ ] 9.10 Implement minimal HTTP handler (GET /health, GET /metrics for K8s probes)
- [ ] 9.11 Write SPIN model for pub/sub routing — verify no message loss
- [ ] 9.12 Test external Zenoh client connecting and running inference

## 10. NVIDIA GPU Support

- [ ] 10.1 Implement PCIe enumeration (scan config space, identify NVIDIA devices, map BARs)
- [ ] 10.2 Implement GPU identification (architecture, SM count, memory size, compute capability)
- [ ] 10.3 Implement GPU initialization (PMU, memory controller, page tables, command FIFO)
- [ ] 10.4 Implement GPU memory allocator (static region for weights, dynamic region for workspace)
- [ ] 10.5 Implement DMA engine (host↔device transfers, async with MSI-X completion)
- [ ] 10.6 Implement compute engine (kernel launch, grid/block config, synchronization)
- [ ] 10.7 Write PTX kernels: gemm_f32, gemm_f16, conv2d, softmax, layernorm, elementwise, reduce
- [ ] 10.8 Build-time PTX → SASS compilation via ptxas (sm_53, sm_70, sm_75, sm_80, sm_87, sm_90, sm_100)
- [ ] 10.9 Implement CUDA execution provider (graph partitioning, memory planning, kernel dispatch)
- [ ] 10.10 Test GPU inference on NVIDIA hardware (ResNet50 correctness and performance)
- [ ] 10.11 Test container GPU passthrough with NVIDIA Container Toolkit

## 11. Container and Kubernetes Integration

- [ ] 11.1 Create multi-stage Dockerfile (build + scratch runtime)
- [ ] 11.2 Build multi-architecture OCI images (amd64, arm64)
- [ ] 11.3 Create Kubernetes Deployment, Service, and HPA manifests
- [ ] 11.4 Implement readiness and liveness probes
- [ ] 11.5 Implement Prometheus metrics endpoint
- [ ] 11.6 Implement graceful shutdown (drain requests, close connections, exit)
- [ ] 11.7 Test 24-hour sustained load in Kubernetes
- [ ] 11.8 Verify container image < 15 MB (CPU-only, no model)

## 12. Safety-Critical Process and Documentation

- [ ] 12.1 Create MISRA-Rust coding standard (adapted from MISRA-C:2023)
- [ ] 12.2 Configure clippy lints to enforce MISRA-Rust rules
- [ ] 12.3 Create DO-178C Plan for Software Aspects of Certification (PSAC)
- [ ] 12.4 Create Software Development Plan (SDP)
- [ ] 12.5 Create Software Verification Plan (SVP) with MC/DC coverage requirements
- [ ] 12.6 Set up Sphinx-needs documentation project with need types (REQ, SPEC, IMPL, TEST, VERIFY)
- [ ] 12.7 Create PlantUML architecture diagrams (component, sequence, state machine, deployment)
- [ ] 12.8 Implement bidirectional traceability matrix (requirements ↔ code ↔ tests ↔ verification)
- [ ] 12.9 Set up MC/DC coverage measurement tooling (cargo-llvm-cov or equivalent)
- [ ] 12.10 Achieve 100% MC/DC coverage on all safety-critical kernel paths
- [ ] 12.11 Run all TLA+ and SPIN models in CI — verify no property violations
- [ ] 12.12 Run all Lean 4 proofs in CI — verify all proofs type-check
- [ ] 12.13 Generate complete Software Accomplishment Summary (SAS)

## 13. Verify

- [ ] 13.1 All unit tests pass on both x86-64 and ARM64
- [ ] 13.2 Integration test: boot → load model → run inference → return result (both archs, QEMU)
- [ ] 13.3 GPU integration test: inference on NVIDIA hardware matches CPU results
- [ ] 13.4 Security audit: all unsafe blocks documented with SAFETY comments
- [ ] 13.5 Fuzz all external interfaces (IPC protocol, ONNX parser, syscalls) for 24 hours with no crashes
- [ ] 13.6 MC/DC coverage report shows 100% on safety-critical paths
- [ ] 13.7 All formal verification models pass (TLA+ TLC, SPIN, Lean 4)
- [ ] 13.8 Sphinx-needs traceability report shows no orphan requirements
- [ ] 13.9 Container image builds and deploys successfully on Kubernetes
- [ ] 13.10 Performance: MobileNetV2 < 50ms CPU, ResNet50 < 5ms GPU (batch=1)
