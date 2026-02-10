Requirements
============

Kernel Core
-----------

.. req:: Physical Memory Management
   :id: REQ_001
   :safety_level: DAL_A
   :tags: kernel, memory

   The kernel shall provide a buddy allocator for physical page allocation
   supporting orders 0 (4KiB) through 10 (4MiB).

.. req:: Cooperative Task Scheduling
   :id: REQ_002
   :safety_level: DAL_A
   :tags: kernel, scheduler

   The kernel shall provide a cooperative async task scheduler using Rust
   futures with work-stealing across CPU cores.

.. req:: Syscall Interface
   :id: REQ_003
   :safety_level: DAL_A
   :tags: kernel, syscall

   The kernel shall expose no more than 46 syscalls organized into 7
   categories (Memory, Task, IPC, ONNX, Device, System, Capability).

ONNX Runtime
------------

.. req:: ONNX Model Loading
   :id: REQ_010
   :safety_level: DAL_A
   :tags: onnx

   The runtime shall load and validate ONNX protobuf files, rejecting
   models with unsupported opset versions or unknown operators.

.. req:: CPU Execution Provider
   :id: REQ_011
   :safety_level: DAL_A
   :tags: onnx, cpu

   The runtime shall provide a CPU execution provider with SIMD
   acceleration (AVX2/AVX-512 on x86-64, NEON/SVE on AArch64).

Security
--------

.. req:: Post-Quantum Key Encapsulation
   :id: REQ_020
   :safety_level: DAL_A
   :tags: security, pqc

   The security subsystem shall implement ML-KEM-768 (FIPS 203) for
   post-quantum key encapsulation with hybrid X25519 fallback.

.. req:: Capability-Based Access Control
   :id: REQ_021
   :safety_level: DAL_A
   :tags: security, capabilities

   All resource access shall be mediated through unforgeable capability
   tokens with explicit permission bitmasks.
