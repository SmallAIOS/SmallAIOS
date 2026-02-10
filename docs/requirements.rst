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
