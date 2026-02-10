# SmallAIOS Kernel — Project Proposal

## Problem Statement

Running AI inference workloads today requires a full general-purpose operating system
(Linux, Windows) that carries enormous unnecessary complexity: package managers, init
systems, shells, network stacks, display servers, and thousands of syscalls. This
creates three fundamental problems:

1. **Attack surface**: Every unnecessary component is a potential exploit vector.
   Container escapes, privilege escalation, and supply chain attacks all exploit
   complexity that has nothing to do with running inference.

2. **Resource overhead**: General-purpose kernels consume memory, CPU cycles, and
   boot time on services irrelevant to AI workloads. In containerized environments,
   this overhead is multiplied across thousands of pods.

3. **Operational complexity**: Patching, configuring, and securing a general-purpose
   OS for a single-purpose workload is ongoing waste.

## Proposed Solution

SmallAIOS is a **clean-room unikernel** written in Rust that does exactly one thing:
boot and execute ONNX model inference. It provides:

- A minimal kernel (~40 syscalls vs. Linux's ~450) with just enough POSIX
  compatibility to support AI runtime operations
- A built-in ONNX runtime with hardware-accelerated execution providers for
  x86-64 (AVX2/AVX-512), ARM64 (NEON/SVE), and NVIDIA GPUs
- A Zenoh-inspired pub/sub IPC system for lightweight function invocation
  — no heavyweight RPC frameworks needed
- First-class container integration (OCI images, Kubernetes CRI compatibility)
- Capability-based security with no ambient authority

## Why a New Kernel

| Concern | Linux + Container | SmallAIOS |
|---|---|---|
| Syscall surface | ~450 syscalls | ~40 syscalls |
| Boot to inference | Seconds | Milliseconds |
| Memory footprint | 50-200 MB base | < 8 MB base |
| Attack surface | Massive (shell, network, fs) | Minimal (no shell, no fs) |
| Language safety | C (manual memory) | Rust (compile-time safety) |
| AI-native | Bolted on | Built in |

### Why Not Use Existing Unikernels

- **Unikraft**: C-based, GPL licensed, general-purpose design
- **Hermit**: Rust but HermitCore license constraints, not AI-focused
- **Nanos**: C-based, focused on general unikernel use cases
- **Redox OS**: Full general-purpose OS, not minimal enough

SmallAIOS is purpose-built: every line of code exists to serve AI inference.

## Scope

### In Scope

- Minimal Rust kernel with POSIX-compatible subset
- ONNX model parser and inference runtime (ONNX opset 21+)
- CPU execution: x86-64 (AVX2/AVX-512/AMX), ARM64 (NEON/SVE/SME)
- GPU execution: NVIDIA (compute capability 7.0+, Volta and newer)
- Zenoh-inspired IPC with key-expression routing
- OCI container image format support
- Kubernetes CRI compatibility layer
- Capability-based security model
- Reproducible, cross-compiled builds

### Out of Scope

- General-purpose computing (no shell, no package manager)
- Training workloads (inference only)
- Non-ONNX model formats (convert to ONNX first)
- Display/GUI support
- Audio subsystem
- Full POSIX compliance (only the subset needed)
- Non-NVIDIA GPU vendors (future work: AMD ROCm, Intel oneAPI)

## Design Principles

1. **If it doesn't serve inference, it doesn't exist.** Every kernel component must
   justify its existence relative to AI workload execution.

2. **Clean room.** All kernel code is original. We reference open specifications
   (POSIX, ONNX, UEFI, PCIe, NVIDIA PTX ISA) but never copy implementation code
   from existing kernels.

3. **Compile-time over runtime.** Prefer static dispatch, compile-time configuration,
   and zero-cost abstractions. The kernel binary should be specialized at build time
   for its target hardware and model.

4. **Zero-copy data paths.** Tensor data should flow from storage through inference
   without unnecessary copies. DMA, shared memory, and memory-mapped I/O are
   first-class citizens.

5. **Defense in depth.** Rust memory safety is the first layer. Capability-based
   security is the second. Minimal syscall surface is the third. No ambient authority
   is the fourth.

## Target Environments

| Environment | Boot Method | Primary Use |
|---|---|---|
| Docker container | Container entry point | Development, CI/CD |
| Kubernetes pod | CRI + container entry | Production inference |
| QEMU/KVM VM | UEFI/virtio | Testing, bare-metal-like |
| Bare metal | UEFI boot | Edge inference appliances |

## License

Apache License 2.0. All contributions must be original work or from
Apache-2.0-compatible sources. See [Clean Room Policy](design/clean-room-policy.md).

## References

- [ONNX Specification](https://onnx.ai/onnx/repo-docs/IR.html)
- [POSIX.1-2024 (IEEE 1003.1)](https://pubs.opengroup.org/onlinepubs/9799919799/)
- [UEFI Specification](https://uefi.org/specifications)
- [Zenoh Protocol](https://zenoh.io/docs/manual/abstractions/)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [OCI Runtime Specification](https://github.com/opencontainers/runtime-spec)
- [Kubernetes CRI](https://kubernetes.io/docs/concepts/architecture/cri/)
