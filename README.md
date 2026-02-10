# SmallAIOS

**A minimal, secure, Rust-based operating system kernel purpose-built for AI inference workloads.**

SmallAIOS is a clean-room unikernel designed to boot and execute ONNX models with
minimal overhead, minimal attack surface, and native hardware acceleration. It targets
containerized deployment (Docker/Kubernetes) on x86-64, ARM64, and NVIDIA GPU platforms.

## Goals

- **Single purpose**: Boot to ONNX inference. Nothing else.
- **Minimal attack surface**: No shell, no unnecessary syscalls, no dynamic linking.
- **Rust from the ground up**: Memory safety without a runtime garbage collector.
- **POSIX-compatible**: Minimal POSIX subset sufficient for AI runtime operations.
- **Hardware accelerated**: Native support for x86-64 (AVX/SSE), ARM64 (NEON/SVE), NVIDIA GPU.
- **Container-native**: First-class Docker and Kubernetes integration.
- **Lightweight IPC**: Zenoh-inspired pub/sub messaging for function invocation.
- **Clean room**: Original implementation; no derived kernel code.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Function Interface                     │
│              (Zenoh-inspired pub/sub IPC)                │
├─────────────────────────────────────────────────────────┤
│                    ONNX Runtime Layer                    │
│          (Parser, Graph Optimizer, Executors)            │
├──────────┬──────────────┬───────────────────────────────┤
│ CPU Exec │  GPU Exec    │     POSIX Compat Layer        │
│ (x86/ARM)│  (NVIDIA)    │     (Minimal Subset)          │
├──────────┴──────────────┴───────────────────────────────┤
│                    Kernel Core                           │
│    (Scheduler, Memory Mgmt, Capability Security)        │
├──────────┬──────────────┬───────────────────────────────┤
│  x86-64  │   ARM64      │   NVIDIA GPU                  │
│  HAL     │   HAL        │   HAL                         │
├──────────┴──────────────┴───────────────────────────────┤
│              Boot Layer (UEFI / Container Entry)         │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
SmallAIOS-Design/
├── openspec/                    # OpenSpec specifications
│   └── smallaios-kernel/
│       ├── proposal.md          # Project proposal and rationale
│       ├── specs/               # Detailed specifications
│       ├── design/              # Architecture and design docs
│       └── tasks/               # Phased implementation plan
├── kernel/                      # Core kernel crate
├── arch/                        # Architecture-specific HALs
│   ├── x86_64/
│   ├── aarch64/
│   └── nvidia/
├── onnx-rt/                     # ONNX runtime crate
├── ipc/                         # IPC/messaging crate
├── posix/                       # POSIX compatibility crate
├── security/                    # Capability-based security crate
├── container/                   # Container interface crate
└── docker/                      # Container build definitions
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Specifications

Full design specifications are maintained using [OpenSpec](https://github.com/Fission-AI/OpenSpec)
in the `openspec/` directory. Start with the [proposal](openspec/smallaios-kernel/proposal.md).
