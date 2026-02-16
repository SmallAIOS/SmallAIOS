Requirements
============

Kernel Core — Memory Management
--------------------------------

.. req:: Physical Memory Map
   :id: REQ_001
   :safety_level: DAL_A
   :tags: kernel, memory

   The kernel shall parse physical memory maps from Multiboot2 (x86-64)
   and Device Tree Blob (ARM64) to enumerate usable RAM regions.

.. req:: Buddy Allocator
   :id: REQ_002
   :safety_level: DAL_A
   :tags: kernel, memory

   The kernel shall implement a buddy allocator for physical page allocation
   supporting orders 0 (4 KiB) through 21 (8 GiB) with coalescing of freed buddy pages.

.. req:: Slab Allocator
   :id: REQ_003
   :safety_level: DAL_A
   :tags: kernel, memory

   The kernel shall implement a slab allocator layered on the buddy allocator
   for sub-page kernel object allocations with size classes 16B through 2048B.

.. req:: Global Allocator
   :id: REQ_004
   :safety_level: DAL_A
   :tags: kernel, memory

   The kernel shall implement the Rust GlobalAlloc trait, routing small
   allocations through the slab allocator and large allocations through
   the buddy allocator.

.. req:: Tensor Memory Pool
   :id: REQ_005
   :safety_level: DAL_A
   :tags: kernel, memory, onnx

   The kernel shall provide a dedicated tensor memory pool with 64-byte
   aligned buffers, reference counting, and arena-based allocation for
   ONNX inference workloads.

.. req:: x86-64 Page Table Management
   :id: REQ_006
   :safety_level: DAL_A
   :tags: kernel, memory, x86

   The kernel shall implement x86-64 4-level page table management with
   map, unmap, and protect operations for identity-mapped memory.

.. req:: ARM64 Page Table Management
   :id: REQ_007
   :safety_level: DAL_A
   :tags: kernel, memory, arm64

   The kernel shall implement ARM64 4-level page table management with
   4 KiB granule for identity-mapped memory.

Kernel Core — Scheduler
-----------------------

.. req:: Soft Real-Time Cooperative Scheduler
   :id: REQ_010
   :safety_level: DAL_A
   :tags: kernel, scheduler

   The kernel shall implement a soft real-time cooperative scheduler with
   three priority classes (SYSTEM, IPC, INFERENCE) and mandatory yield
   points between ONNX operators.

.. req:: Hardware Watchdog Timer
   :id: REQ_011
   :safety_level: DAL_A
   :tags: kernel, scheduler, watchdog

   The kernel shall initialize and service a hardware watchdog timer
   (default 30s timeout) from a SYSTEM class task.

.. req:: Syscall Interface
   :id: REQ_012
   :safety_level: DAL_A
   :tags: kernel, syscall

   The kernel shall expose approximately 49 syscalls organized into 7
   categories (Memory, Task, IPC, ONNX, Device, System, Capability).

ONNX Runtime
------------

.. req:: ONNX Model Loading
   :id: REQ_020
   :safety_level: DAL_A
   :tags: onnx

   The runtime shall load and validate ONNX protobuf files, rejecting
   models with unsupported opset versions or unknown operators.

.. req:: CPU Execution Provider
   :id: REQ_021
   :safety_level: DAL_A
   :tags: onnx, cpu

   The runtime shall provide a CPU execution provider with SIMD
   acceleration (AVX2/AVX-512 on x86-64, NEON/SVE on AArch64).

Security
--------

.. req:: Post-Quantum Key Encapsulation
   :id: REQ_030
   :safety_level: DAL_A
   :tags: security, pqc

   The security subsystem shall implement ML-KEM-768 (FIPS 203) for
   post-quantum key encapsulation with hybrid X25519 fallback.

.. req:: Capability-Based Access Control
   :id: REQ_031
   :safety_level: DAL_A
   :tags: security, capabilities

   All resource access shall be mediated through unforgeable capability
   tokens with explicit permission bitmasks.

Implementation Traceability — Phase 2
--------------------------------------

.. spec:: Multiboot2 Memory Map Parser
   :id: SPEC_001
   :safety_level: DAL_A
   :tags: kernel, memory, x86
   :links: REQ_001

   Parses Multiboot2 boot information structure tag type 6 to extract
   physical memory regions with types (usable, reserved, ACPI reclaimable).

.. spec:: DTB Memory Map Parser
   :id: SPEC_002
   :safety_level: DAL_A
   :tags: kernel, memory, arm64
   :links: REQ_001

   Parses Flattened Device Tree /memory nodes with reg properties to
   extract physical memory regions, supporting variable address-cells
   and size-cells.

.. impl:: kernel/src/mem/phys.rs
   :id: IMPL_001
   :safety_level: DAL_A
   :tags: kernel, memory
   :links: SPEC_001; SPEC_002

   Physical memory map module with PhysMemoryMap, parse_multiboot2(),
   parse_dtb(), and exclude_range() operations.

.. impl:: kernel/src/mem/buddy.rs
   :id: IMPL_002
   :safety_level: DAL_A
   :tags: kernel, memory
   :links: REQ_002

   Buddy allocator with bitmap-per-order design, supporting orders 0-21,
   split/merge operations, and double-free detection via allocation bitmap.

.. impl:: kernel/src/mem/slab.rs
   :id: IMPL_003
   :safety_level: DAL_A
   :tags: kernel, memory
   :links: REQ_003

   Slab allocator with 8 size classes (16-2048B), free-list per slab page,
   O(1) amortized allocation.

.. impl:: kernel/src/mem/global.rs
   :id: IMPL_004
   :safety_level: DAL_A
   :tags: kernel, memory
   :links: REQ_004

   Global allocator implementing core::alloc::GlobalAlloc, routing through
   slab (<=2048B) or buddy (>2048B) allocators.

.. impl:: kernel/src/mem/tensor.rs
   :id: IMPL_005
   :safety_level: DAL_A
   :tags: kernel, memory, onnx
   :links: REQ_005

   Tensor memory pool with 64-byte aligned arena allocation, atomic
   reference counting, and session-level reset.

.. impl:: arch/x86_64/src/paging.rs
   :id: IMPL_006
   :safety_level: DAL_A
   :tags: kernel, memory, x86
   :links: REQ_006

   x86-64 4-level page table management with PTE flag translation,
   walk-or-create hierarchy, and identity mapping support.

.. impl:: arch/aarch64/src/paging.rs
   :id: IMPL_007
   :safety_level: DAL_A
   :tags: kernel, memory, arm64
   :links: REQ_007

   ARM64 4-level page table management with descriptor attribute translation,
   MAIR/TCR configuration, and TLB invalidation.

.. test:: Buddy Allocator Unit Tests
   :id: TEST_001
   :safety_level: DAL_A
   :tags: kernel, memory, test
   :links: IMPL_002

   10 tests covering: allocate, free, double-free detection, exhaustion,
   buddy merge/coalesce, large order allocation, split and recombine,
   order too large, alignment error, invalid address.

.. test:: Physical Memory Map Tests
   :id: TEST_002
   :safety_level: DAL_A
   :tags: kernel, memory, test
   :links: IMPL_001

   5 tests covering: empty map, add regions, exclude range (full, partial,
   split), map full capacity.

.. test:: Slab Allocator Unit Tests
   :id: TEST_003
   :safety_level: DAL_A
   :tags: kernel, memory, test
   :links: IMPL_003

   4 tests covering: size class index lookup, actual size, alloc/free cycle,
   slab exhaustion.

.. test:: Tensor Pool Unit Tests
   :id: TEST_004
   :safety_level: DAL_A
   :tags: kernel, memory, test
   :links: IMPL_005

   5 tests covering: basic allocation, alignment, reference counting,
   pool exhaustion, reset.

.. test:: Page Table Unit Tests
   :id: TEST_005
   :safety_level: DAL_A
   :tags: kernel, memory, test
   :links: IMPL_006; IMPL_007

   5 tests covering: page flags, page table index extraction, page offset,
   PTE construction, empty page table.

Cybersecurity Compliance
------------------------

.. req:: Tamper-Evident Audit Logging
   :id: REQ_040
   :safety_level: DAL_A
   :tags: security, audit, nist-au

   The security subsystem shall implement tamper-evident audit logging with
   SHA-3-256 hash chain, ML-DSA-65 batch signing, and configurable retention
   policy per deployment class (7d edge, 90d datacenter, 365d safety-critical).

.. req:: Continuous Security Monitoring
   :id: REQ_041
   :safety_level: DAL_A
   :tags: security, monitoring, nist-si

   The security subsystem shall implement continuous monitoring including
   capability denial rate tracking, memory allocation failure tracking,
   inference latency anomaly detection (3-sigma), and SYN flood detection.

.. req:: Incident Response Automation
   :id: REQ_042
   :safety_level: DAL_A
   :tags: security, incident, nist-ir

   The security subsystem shall implement automated incident containment
   including capability revocation, task termination, connection reset,
   and evidence preservation with Zenoh IPC event publishing.

.. req:: Supply Chain Security
   :id: REQ_043
   :safety_level: DAL_A
   :tags: security, supply-chain, nist-sr

   The build system shall generate CycloneDX SBOMs, integrate cargo-audit
   for vulnerability scanning, implement reproducible builds, and produce
   ML-DSA-65 build attestation signatures.

.. req:: OT/ICS Security Hardening
   :id: REQ_044
   :safety_level: DAL_A
   :tags: security, ot, iec-61508

   The security subsystem shall implement WCET instrumentation, fail-safe
   state definitions, OT anomaly detection, and safe shutdown procedures
   with bounded time (configurable, default 100ms).

.. req:: Information Flow Enforcement
   :id: REQ_045
   :safety_level: DAL_A
   :tags: security, boundary, nist-ac

   The security subsystem shall enforce information flow rules per task type:
   ONNX runtime shall not access network, IPC router shall not access GPU,
   bus handlers shall not access ONNX runtime.

.. req:: Post-Quantum Key Management
   :id: REQ_046
   :safety_level: DAL_A
   :tags: security, crypto, nist-sc

   The security subsystem shall implement key lifecycle management including
   boot-time generation, memory-only storage, reboot rotation, and volatile-write
   zeroization with verification pass.

.. req:: NIST SP 800-53 Compliance
   :id: REQ_047
   :safety_level: DAL_A
   :tags: security, compliance, nist

   The system shall maintain documented NIST SP 800-53 Rev 5 control mappings
   covering all 20 control families with implementation status per control,
   SSP skeleton, and POA&M for planned controls.

.. spec:: Audit Hash Chain Integrity
   :id: SPEC_040
   :safety_level: DAL_A
   :tags: security, audit
   :links: REQ_040

   SHA-3-256(previous_batch_hash || serialized_entries) per batch, with genesis
   hash of 32 zero bytes, monotonic sequence numbers, and verification function.

.. spec:: Monitoring Alert Thresholds
   :id: SPEC_041
   :safety_level: DAL_A
   :tags: security, monitoring
   :links: REQ_041

   Configurable per-metric alert thresholds with boot-time configuration,
   Prometheus export in OpenMetrics format, and Zenoh IPC export.

.. spec:: Containment Actions
   :id: SPEC_042
   :safety_level: DAL_A
   :tags: security, incident
   :links: REQ_042

   Automated containment: capability revocation for compromised tasks,
   task termination, network connection reset, inference rejection.

.. spec:: WCET Static Bounds
   :id: SPEC_044
   :safety_level: DAL_A
   :tags: security, ot
   :links: REQ_044

   Static WCET bounds for 6 critical paths: syscall dispatch (5us x86),
   capability check (1us), buddy alloc (3us), slab alloc (500ns),
   task schedule (2us), interrupt handle (500ns). No-recursion policy enforced.

.. spec:: Access Matrix
   :id: SPEC_045
   :safety_level: DAL_A
   :tags: security, boundary
   :links: REQ_045

   Task-type to resource-type access matrix: SYSTEM (all 6 resources),
   IPC (memory, network, audit, capability), INFERENCE (memory, gpu).

.. impl:: security/src/audit/
   :id: IMPL_040
   :safety_level: DAL_A
   :tags: security, audit
   :links: SPEC_040

   Audit subsystem: entry.rs (struct), taxonomy.rs (events), accumulator.rs
   (batching), integrity.rs (hash chain), batch_signing.rs (ML-DSA-65),
   ipc_export.rs (Zenoh), retention.rs (policy), ring_buffer.rs, syslog.rs.

.. impl:: security/src/monitoring/
   :id: IMPL_041
   :safety_level: DAL_A
   :tags: security, monitoring
   :links: SPEC_041

   Monitoring subsystem: rate_tracker.rs (denial/allocation counters),
   latency_stats.rs (p50/p99/p999 with 3-sigma anomaly), watchdog_monitor.rs,
   alerts.rs (threshold config), prometheus.rs (OpenMetrics), ipc_export.rs (Zenoh).

.. impl:: security/src/incident/
   :id: IMPL_042
   :safety_level: DAL_A
   :tags: security, incident
   :links: SPEC_042

   Incident subsystem: containment.rs (automated actions), evidence.rs
   (state export), event.rs (Zenoh publishing), severity_classifier.rs,
   communication.rs, post_incident.rs.

.. impl:: security/src/ot/wcet_bounds.rs
   :id: IMPL_044
   :safety_level: DAL_A
   :tags: security, ot
   :links: SPEC_044

   Static WCET bounds for all 6 critical paths with no-recursion policy,
   bounded loop inventory, and per-architecture bound values (x86/ARM64).

.. impl:: security/src/boundary/data_flow_auth.rs
   :id: IMPL_045
   :safety_level: DAL_A
   :tags: security, boundary
   :links: SPEC_045

   Cross-boundary data flow verification with 8 defined flows, authentication
   mechanism requirements, and integrity/confidentiality checks.

.. test:: Audit MC/DC Coverage Tests
   :id: TEST_040
   :safety_level: DAL_A
   :tags: security, audit, mcdc, test
   :links: IMPL_040

   MC/DC tests for audit critical paths: batch accumulator push/tick/flush,
   hash chain verification, retention policy, signing queue, IPC serialization.
   Each condition independently shown to affect decision outcome.

.. test:: Monitoring Unit Tests
   :id: TEST_041
   :safety_level: DAL_A
   :tags: security, monitoring, test
   :links: IMPL_041

   Tests covering: counter accuracy, anomaly detection trigger, threshold
   configuration, metrics export format (Prometheus + Zenoh).

.. test:: Incident Response Unit Tests
   :id: TEST_042
   :safety_level: DAL_A
   :tags: security, incident, test
   :links: IMPL_042

   Tests covering: containment action execution, evidence export format,
   incident event serialization, severity classification.

.. test:: WCET Bounds Tests
   :id: TEST_044
   :safety_level: DAL_A
   :tags: security, ot, test
   :links: IMPL_044

   Tests covering: all paths have bounds, bounds are non-zero, ARM64 >= x86,
   no-recursion verified, loop bounds are finite.

.. verify:: TLA+ Audit Hash Chain Model
   :id: VERIFY_040
   :safety_level: DAL_A
   :tags: security, audit, formal
   :links: SPEC_040; TEST_040

   TLA+ model (formal/tla/AuditHashChain.tla) verifying: batch ordering,
   no sequence gaps, no replay, genesis anchoring, hash chain continuity,
   no empty batches, hash uniqueness.

.. verify:: Lean 4 Information Flow Proof
   :id: VERIFY_045
   :safety_level: DAL_A
   :tags: security, boundary, formal
   :links: SPEC_045; TEST_044

   Lean 4 proof (formal/lean4/InformationFlow.lean) verifying: system
   unrestricted access, inference isolation (no network/capability/bus/audit),
   IPC restrictions (no gpu/bus), privilege monotonicity, access totality.

.. verify:: TLA+ Scheduler Anomaly Detection Model
   :id: VERIFY_041
   :safety_level: DAL_A
   :tags: security, monitoring, formal
   :links: SPEC_041; TEST_041

   TLA+ model (formal/tla/SchedulerAnomaly.tla) verifying: alert boundedness,
   suppression effectiveness, denial count bounds, threshold correctness,
   mutual exclusion, watchdog liveness.

Implementation Traceability — Scheduler, ONNX, Security
---------------------------------------------------------

.. spec:: Cooperative Scheduler Design
   :id: SPEC_010
   :safety_level: DAL_A
   :tags: kernel, scheduler
   :links: REQ_010

   Three-class priority scheduler (SYSTEM > IPC > INFERENCE) with
   cooperative yield at ONNX operator boundaries and round-robin
   within each class.

.. impl:: kernel/src/sched/
   :id: IMPL_010
   :safety_level: DAL_A
   :tags: kernel, scheduler
   :links: SPEC_010

   Scheduler implementation with Task struct, RunQueue, yield_now(),
   spawn(), and priority-based dequeue. Cooperative context switch.

.. test:: Scheduler Unit Tests
   :id: TEST_010
   :safety_level: DAL_A
   :tags: kernel, scheduler, test
   :links: IMPL_010

   Tests covering: task spawn, priority ordering, yield behavior,
   round-robin within class, empty queue handling.

.. impl:: kernel/src/sched/watchdog.rs
   :id: IMPL_011
   :safety_level: DAL_A
   :tags: kernel, scheduler, watchdog
   :links: REQ_011

   Hardware watchdog timer with configurable timeout (default 30s),
   pet/kick interface, and system class task auto-service.

.. spec:: Syscall Dispatch Table
   :id: SPEC_012
   :safety_level: DAL_A
   :tags: kernel, syscall
   :links: REQ_012

   46 syscalls organized into 7 categories with capability-gated
   dispatch, architecture-specific entry points (SYSCALL/SVC/ECALL).

.. impl:: kernel/src/syscall/
   :id: IMPL_012
   :safety_level: DAL_A
   :tags: kernel, syscall
   :links: SPEC_012

   Syscall dispatch table, handler implementations for memory, task,
   IPC, ONNX, device, system, and capability operations.

.. test:: Syscall Unit Tests
   :id: TEST_012
   :safety_level: DAL_A
   :tags: kernel, syscall, test
   :links: IMPL_012

   Tests covering: dispatch table lookup, invalid syscall number,
   capability check enforcement, all 7 categories exercised.

.. spec:: ONNX Protobuf Parser
   :id: SPEC_020
   :safety_level: DAL_A
   :tags: onnx
   :links: REQ_020

   Clean-room no_std protobuf parser for ONNX ModelProto, GraphProto,
   NodeProto, and TensorProto with opset validation.

.. impl:: onnx-rt/src/parser/
   :id: IMPL_020
   :safety_level: DAL_A
   :tags: onnx
   :links: SPEC_020

   Protobuf wire-format parser, ONNX model deserialization, operator
   registry, opset version validation, graph topology check.

.. test:: ONNX Parser Tests
   :id: TEST_020
   :safety_level: DAL_A
   :tags: onnx, test
   :links: IMPL_020

   Tests covering: valid model parse, invalid protobuf rejection,
   unsupported opset, unknown operator, graph cycle detection.

.. spec:: CPU Execution Provider Design
   :id: SPEC_021
   :safety_level: DAL_A
   :tags: onnx, cpu
   :links: REQ_021

   CPU execution provider with 6 operators (Conv, MatMul, Relu,
   Sigmoid, Reshape, Softmax), GEMM, and SIMD-optimized paths.

.. impl:: onnx-rt/src/cpu_provider.rs
   :id: IMPL_021
   :safety_level: DAL_A
   :tags: onnx, cpu
   :links: SPEC_021

   CPU execution provider implementation with operator dispatch,
   tensor workspace allocation, and operator fusion support.

.. test:: CPU Provider Tests
   :id: TEST_021
   :safety_level: DAL_A
   :tags: onnx, cpu, test
   :links: IMPL_021

   Tests covering: all 6 operators produce correct output, GEMM
   accuracy, workspace allocation, operator fusion correctness.

.. impl:: security/src/crypto/ml_kem.rs
   :id: IMPL_030
   :safety_level: DAL_A
   :tags: security, pqc
   :links: REQ_030

   ML-KEM-768 (FIPS 203) implementation with keygen, encapsulate,
   decapsulate, and X25519 hybrid mode.

.. impl:: security/src/capability/
   :id: IMPL_031
   :safety_level: DAL_A
   :tags: security, capabilities
   :links: REQ_031

   Capability-based access control: unforgeable tokens, permission
   bitmasks, revocation, delegation with attenuation.

.. impl:: security/src/supply_chain/
   :id: IMPL_043
   :safety_level: DAL_A
   :tags: security, supply-chain
   :links: REQ_043

   Supply chain security: CycloneDX SBOM generation, vendor
   attestation, build reproducibility checks.

.. impl:: security/src/crypto/key_manager.rs
   :id: IMPL_046
   :safety_level: DAL_A
   :tags: security, crypto
   :links: REQ_046

   Key lifecycle management: boot-time generation, memory-only
   storage, reboot rotation, volatile-write zeroization.

.. impl:: security/src/compliance/nist_controls.rs
   :id: IMPL_047
   :safety_level: DAL_A
   :tags: security, compliance
   :links: REQ_047

   NIST SP 800-53 Rev 5 control mappings for all 20 families,
   implementation status tracking, SSP skeleton, POA&M.
