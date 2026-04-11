# SmallAIOS Architecture

SmallAIOS is a 20-crate `#![no_std]` Rust workspace organized into a strict 4-layer dependency model. Higher layers may depend on same-layer or lower-layer crates only. This document covers the layer model, dependency structure, design rationale, and acyclicity guarantees.

## 4-Layer Model

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Integration                                              │
│  ┌──────────────────┐  ┌──────────────────┐                         │
│  │  container        │  │  bench (dev-only) │                        │
│  │  Entry point,     │  │  Benchmarks       │                        │
│  │  config, health,  │  │                   │                        │
│  │  metrics          │  │                   │                        │
│  └──────────────────┘  └──────────────────┘                         │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — HAL / Drivers                                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │
│  │ arch/x86_64│ │arch/aarch64│ │arch/riscv64│  CPU HALs             │
│  └────────────┘ └────────────┘ └────────────┘                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │
│  │ arch/nvidia│ │  arch/amd  │ │arch/intel  │  GPU HALs (stubs)     │
│  └────────────┘ └────────────┘ └────────────┘                       │
│  ┌────────────┐                                                     │
│  │ arch/apple │  Apple Metal HAL (macOS)                             │
│  └────────────┘                                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │
│  │ peripheral │ │    bus     │ │    sdr     │  Device drivers        │
│  └────────────┘ └────────────┘ └────────────┘                       │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — Core Services                                            │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            │
│  │  net   │ │  ipc   │ │ posix  │ │onnx-rt │ │  usb   │            │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘            │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 0 — Foundation                                               │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐   │
│  │  kernel                   │  │  security                     │   │
│  │  Memory (buddy allocator, │  │  Capability-based access,     │   │
│  │  tensor pool), cooperative│  │  PQC crypto (SHA-3, AES-GCM,  │   │
│  │  scheduler, ~46 syscalls  │  │  ML-KEM, ML-DSA, Ed25519,     │   │
│  │                           │  │  X25519), formal gate          │   │
│  └───────────────────────────┘  └───────────────────────────────┘   │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐   │
│  │  compute                  │  │  sched-types                  │   │
│  │  CPU/GPU/NPU backends,    │  │  Shared scheduler primitives  │   │
│  │  tensor buffer management │  │  (OperatorClass, Budget)      │   │
│  └───────────────────────────┘  └───────────────────────────────┘   │
│           kernel ──depends-on──▶ security                           │
│           kernel ──depends-on──▶ sched-types                        │
└─────────────────────────────────────────────────────────────────────┘

  Dependency direction: Layer 3 → Layer 2 → Layer 1 → Layer 0
  (higher layers depend on lower layers only)
```

### Crate-to-Layer Assignment

| Layer | Crate | Role |
|-------|-------|------|
| 0 | `smallaios-kernel` | Memory management, cooperative scheduler, syscall interface |
| 0 | `smallaios-security` | Capability system, PQC crypto stack, formal verification gate |
| 0 | `smallaios-compute` | Unified compute abstraction: device registry, kernel dispatch, tensor buffers |
| 0 | `smallaios-sched-types` | Shared scheduler primitive types (`OperatorClass`, `OperatorBudget`, `BudgetResult`) |
| 1 | `smallaios-net` | IPv4/IPv6, TCP/UDP, ARP/NDP, QUIC/HTTP3, TLS 1.3 |
| 1 | `smallaios-ipc` | Zenoh-inspired pub/sub messaging |
| 1 | `smallaios-posix` | Minimal POSIX compatibility layer |
| 1 | `smallaios-onnx-rt` | Clean-room ONNX runtime (protobuf parser, optimizer, 6 operators) |
| 1 | `smallaios-usb` | USB core stack, xHCI host controller |
| 2 | `smallaios-arch-x86_64` | x86-64 HAL: boot, GDT, IDT, APIC, paging, syscall |
| 2 | `smallaios-arch-aarch64` | ARM64 HAL: boot, GICv3, paging, SVE, PSCI |
| 2 | `smallaios-arch-riscv64` | RISC-V HAL: boot, SBI, trap handling, paging |
| 2 | `smallaios-arch-nvidia` | NVIDIA GPU HAL stub: PCIe, GPU init, compute, DMA |
| 2 | `smallaios-arch-amd` | AMD RDNA/CDNA GPU HAL stub |
| 2 | `smallaios-arch-intel-gpu` | Intel Xe GPU HAL stub |
| 2 | `smallaios-arch-apple` | Apple Metal GPU HAL (macOS only) |
| 2 | `smallaios-peripheral` | I2C, SPI, GPIO, UART, CSI camera, I2S audio |
| 2 | `smallaios-bus` | CAN, ARINC 429/664, MIL-STD-1553, SpaceWire |
| 2 | `smallaios-sdr` | Software-defined radio: HackRF One, ADALM-Pluto |
| 3 | `smallaios-container` | Entry point, config, health checks, metrics |
| 3 | `smallaios-bench` | Benchmarks (dev-dependency only) |

## DSM Evidence

Design Structure Matrix analysis of the workspace dependency graph. Generated by `tools/dsm/` from `build/analysis/dsm-matrix.json`. Run `just dsm-analyze` to regenerate.

### Propagation Cost

Propagation cost measures the percentage of crates affected when a crate changes (transitively).

| Crate | Propagation Cost | Notes |
|-------|-----------------|-------|
| `security` | 100% | Foundation — all crates transitively affected |
| `kernel` | 94% | 18 of 21 crates depend on it directly |
| `arch/nvidia` | 22% | Used by onnx-rt and aarch64 |
| `net` | 17% | Used by container and posix |
| `onnx-rt` | 17% | Used by container and aarch64 |
| `ipc` | 11% | Used by container |
| `posix` | 11% | Used by container |
| `usb` | 11% | Used by sdr |
| All other Layer 2 | 6% | Leaf or near-leaf crates |
| `container`, `bench` | 6% | Top-level consumers |

### Fan-In / Fan-Out Summary

| Crate | Fan-In | Fan-Out | Layer |
|-------|--------|---------|-------|
| `kernel` | 16 | 1 | 0 |
| `security` | 5 | 0 | 0 |
| `net` | 2 | 1 | 1 |
| `ipc` | 1 | 2 | 1 |
| `posix` | 1 | 2 | 1 |
| `onnx-rt` | 2 | 3 | 1 |
| `usb` | 1 | 1 | 1 |
| `arch/x86_64` | 0 | 1 | 2 |
| `arch/aarch64` | 0 | 3 | 2 |
| `arch/riscv64` | 0 | 1 | 2 |
| `arch/nvidia` | 3 | 1 | 2 |
| `arch/amd` | 0 | 1 | 2 |
| `arch/intel-gpu` | 0 | 1 | 2 |
| `peripheral` | 0 | 2 | 2 |
| `bus` | 0 | 1 | 2 |
| `sdr` | 0 | 2 | 2 |
| `container` | 0 | 7 | 3 |
| `bench` | 0 | 1 | 3 |

Fan-in = number of crates that depend on this crate (production deps only).
Fan-out = number of crates this crate depends on (production deps only).

### Key DSM Observations

- **Hub crate:** `kernel` is the central hub with fan-in=16. This is intentional — it provides the foundation (memory, scheduling, syscalls) that all other crates need.
- **High propagation risk:** `security` has the highest propagation cost (~94%) because `kernel` depends on it and nearly everything depends on `kernel`. Changes to crypto primitives or the capability system ripple through the entire workspace.
- **Clean leaf layer:** All Layer 2 (HAL/driver) crates have fan-in of 0-2, keeping hardware-specific changes isolated.
- **Narrow integration point:** `container` is the sole top-level integrator with fan-out=7, providing a single composition root.

## Design Rationale

### Unikernel Architecture

SmallAIOS runs in a single address space with no kernel/user mode split. This eliminates context switch overhead and IPC marshalling costs that would add latency to inference hot paths. The tradeoff — no process isolation — is acceptable because the system runs a single workload (ONNX inference) and uses capability-based security instead of address space isolation.

### Cooperative Scheduling

The scheduler is cooperative, not preemptive. Tasks yield at ONNX operator boundaries (after each Conv, MatMul, Relu, etc.). This avoids the overhead and complexity of preemption (saving/restoring SIMD/GPU state mid-operator) while providing natural scheduling points that align with the inference workload.

### `#![no_std]` Throughout

All 21 crates are `#![no_std]`. This enables bare-metal deployment on x86-64, ARM64, and RISC-V without a host OS. The same crates also compile for musl targets for container deployment, giving a single codebase for both deployment modes.

### Size Goals

- **<8 MB base kernel:** Fits in embedded flash/SRAM. Achieved via `opt-level = "z"`, LTO, and single codegen unit.
- **<15 MB container image:** Built `FROM scratch` with static musl binary. Current size: ~594 KB.
- **<50 ms container boot:** No init system, no dynamic linking, no filesystem setup. Boot straight to inference.

### Post-Quantum Cryptography by Default

The `pqc-hybrid` feature is on by default, providing ML-KEM-768 (key encapsulation) + ML-DSA-65 (signatures) alongside classical X25519/Ed25519. This future-proofs deployed systems against quantum attacks without waiting for a migration event. The `classical-only` and `pqc-only` feature flags allow operators to choose.

### DO-178C DAL A Compliance Target

Safety-critical aviation certification requires MC/DC 100% coverage on critical paths, formal verification (19 TLA+ models, 6 SPIN/Promela models), and traceability from requirements to tests. The `formal-gate` feature flag enables compile-time formal verification checks. This is a design target, not yet achieved.

## Dependency Rules

### Layer Rules

1. **Downward only:** A crate at Layer N may depend on crates at Layer N or below.
2. **No upward dependencies:** Layer 0 crates never depend on Layer 1+. Layer 1 crates never depend on Layer 2+.
3. **Same-layer allowed:** Crates within the same layer may depend on each other (e.g., `kernel` → `security` within Layer 0).
4. **Dev-dependencies exempt:** Test-only dependencies may cross layers in any direction. These are not compiled into production builds.

### Specific Dependency Edges

```
Layer 0 (internal):
  kernel → security

Layer 1 → Layer 0:
  net → kernel
  ipc → kernel, security
  posix → kernel
  onnx-rt → kernel, security
  usb → kernel

Layer 2 → Layer 0:
  All arch/* crates → kernel
  nvidia → kernel, security
  peripheral → kernel
  bus → kernel
  sdr → kernel

Accepted cross-layer exceptions:
  aarch64 (L2) → nvidia (L2), onnx-rt (L1) — GPU dispatch on ARM64
  onnx-rt (L1) → nvidia (L2) — CUDA execution provider (upward, accepted)
  posix (L1) → net (L1) — POSIX socket layer needs network stack

Layer 3 → Layer 0/1:
  container → kernel, security, net, ipc, posix, onnx-rt, nvidia
```

### Enforcement

- **Cargo workspace:** The `Cargo.toml` dependency graph is the source of truth. Any new dependency edge is visible in PR diffs.
- **`cargo-modules --acyclic`:** Run in CI to verify no production-dependency cycles exist.
- **Code review:** Layer violations are flagged during review. A crate's layer assignment is documented in this file and in its own `Cargo.toml` metadata.

## Acyclicity Guarantee

### Zero Production Cycles

The production dependency graph (normal + build dependencies) contains **zero cycles**. This is verified by `cargo-modules` and is structurally enforced by the layered architecture: since dependencies only flow downward (Layer 3 → 2 → 1 → 0), cycles cannot form across layers. Within Layer 0, the single edge `kernel → security` is unidirectional.

### Benign Dev-Dependency Cycle

One dev-dependency cycle exists and is intentional:

```
security ──[dev-dep]──▶ net ──[normal]──▶ kernel ──[normal]──▶ security
```

`security` has a dev-dependency on `net` for integration testing (verifying that TLS handshakes work with the real network stack). This creates a cycle in the full dependency graph but **not** in the production graph. Cargo handles this correctly — dev-dependencies are only compiled for `cargo test` of that specific crate and are never included in release builds.

### Enforcement Mechanism

- **CI check:** `cargo-modules` with the `--acyclic` flag runs in CI and will fail the build if any production cycle is introduced.
- **Cargo itself:** Cargo forbids cycles in normal dependencies at the workspace level. A PR that introduces a production cycle will fail `cargo check`.
- **Structural guarantee:** The 4-layer model makes cycles structurally unlikely. A cycle would require a lower-layer crate to depend on a higher-layer crate, which violates the documented dependency rules and would be caught in review.
