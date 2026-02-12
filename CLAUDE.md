# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmallAIOS is a minimal, secure, Rust-based OS kernel purpose-built for AI inference workloads. It boots directly to ONNX inference with ~46 syscalls (vs Linux ~450). Targets x86-64, ARM64, and RISC-V. Deploys as either a container (Docker/K8s) or bare-metal/VM via QEMU.

**Current state:** Prototype phase — ~4,100 tests passing. Production-quality networking (IPv4/IPv6/TCP/ARP/NDP), QUIC/HTTP3 with TLS 1.3, protobuf parser, ONNX runtime with 6 real operators, full PQC crypto stack (SHA-3, AES-256-GCM, ML-KEM-768, ML-DSA-65, Ed25519, X25519), capability system. GPU crates (NVIDIA, Intel, AMD) are architectural stubs with HAL interfaces but no hardware interaction.

## Build Commands

Requires Rust nightly (pinned in `rust-toolchain.toml`). Uses `make` as the primary build interface.

```bash
# Container mode (library OS)
make build-container-x86    # x86_64-unknown-linux-musl
make build-container-arm    # aarch64-unknown-linux-musl

# Kernel mode (VM/bare metal)
make build-kernel-x86       # x86_64-unknown-none
make build-kernel-arm       # aarch64-unknown-none

# Testing
make test                   # cargo test --workspace
make clippy                 # cargo clippy -- -D warnings
make fmt                    # cargo fmt
make fmt-check              # cargo fmt -- --check

# QEMU
make run-x86                # Boot in QEMU x86-64
make run-arm                # Boot in QEMU ARM64

# Docker
make docker-build           # Multi-arch container build
```

## Workspace Architecture

18-crate Rust workspace (`#![no_std]`, edition 2021). Dependency flow:

```
kernel (foundation: memory, scheduler, syscall interface)
├── security (capability-based access, PQC crypto, formal gate)
├── arch/x86_64 (HAL: boot, GDT, IDT, APIC, paging, syscall)
├── arch/aarch64 (HAL: boot, GICv3, paging, SVE, PSCI)
├── arch/riscv64 (HAL: boot, SBI, trap handling, paging)
├── arch/nvidia (HAL stub: PCIe, GPU init, compute, DMA)
├── arch/intel_gpu (HAL stub: Xe-LP/HPG/HPC interfaces)
├── arch/amd (HAL stub: RDNA/CDNA interfaces)
├── onnx-rt (parser, optimizer, execution providers)
├── ipc (Zenoh-inspired pub/sub messaging)
├── net (IPv4/IPv6, TCP/UDP native stack)
├── posix (minimal POSIX compat layer)
├── bus (CAN, ARINC 429, MIL-STD-1553, SpaceWire)
├── peripheral (I2C, SPI, GPIO, UART, CSI camera, I2S audio)
├── usb (USB core stack, xHCI host controller)
├── sdr (Software-defined radio drivers)
├── bench (benchmarks)
└── container (entry point, config, health, metrics)
```

## Key Design Decisions

- **Unikernel** — single address space, no microkernel IPC overhead
- **Cooperative async scheduling** — yields at ONNX operator boundaries
- **Clean-room ONNX runtime** — from-scratch `#![no_std]` Rust, no external C deps
- **Post-quantum crypto default** — ML-KEM-768 + ML-DSA-65 hybrid mode
- **DO-178C DAL A compliance target** — MC/DC 100% coverage on safety-critical paths
- **Formal verification** — TLA+ (19 protocol models for concurrency/safety invariants)
- **Size goals** — <8 MB base, <15 MB container, <50ms container boot

## Build Configuration

- **Toolchain:** nightly-2026-02-01, components: rust-src, rustfmt, clippy, llvm-tools
- **Targets:** x86_64-unknown-none, aarch64-unknown-none, riscv64gc-unknown-none-elf, x86_64-unknown-linux-musl, aarch64-unknown-linux-musl
- **Release profile:** `opt-level = "z"`, LTO enabled, single codegen unit (size-optimized)
- **Linker scripts:** Custom per bare-metal target (see `.cargo/config.toml`)

## Git Workflow

**Gitflow branching model:** feature branches -> `develop` -> `main`

- `main` — stable, release-ready code
- `develop` — integration branch for ongoing work
- Feature branches merge into `develop` via PR
- `develop` merges into `main` for releases

### Worktrees + OpenSpec

Each OpenSpec change gets its own branch. Worktrees can be created as needed in `../SmallAIOS-Design-worktrees/`.

**Branch naming:** `change/<openspec-change-name>`

**Workflow:**
1. Create branch from `develop`: `git checkout -b change/<name> develop`
2. Optionally create worktree: `git worktree add ../SmallAIOS-Design-worktrees/<name> change/<name>`
3. Implement tasks, commit, push, create PR against `develop`
4. After merge, clean up: `git worktree remove` / `git branch -d`

## OpenSpec Changes

Active specs in `openspec/changes/`, completed specs in `openspec/archived/`. Reference specs in `openspec/smallaios-kernel/`.

| Change | Status | Tasks | Focus |
|--------|--------|-------|-------|
| `smallaios-kernel-v1` | Active | 142/146 | Core kernel, memory, scheduler, crypto, ONNX, networking |
| `platform-expansion-v2` | Active | 191/205 | RISC-V, CAN/ARINC/MIL-STD buses, K8s, DDS, QUIC |
| `cybersecurity-compliance-v3` | Complete | 110/110 | NIST SP 800-53, audit, supply chain, incident response |
| `amd-gpu-support-v4` | Complete | 14/14 | AMD RDNA/CDNA GPU HAL stub |
| `intel-gpu-support-v5` | Complete | 12/12 | Intel Xe GPU HAL stub |
| `hardware-peripheral-interfaces-v6` | Complete | 74/74 | I2C, SPI, GPIO, UART, CSI, I2S |
| `usb-sdr-support-v1` | Active | 109/117 | USB core stack, xHCI, SDR drivers |
| `formal-type-gate-v1` | Complete | 52/52 | Type-safe security gate with formal verification |

Use OpenSpec skills (e.g. `/opsx:new`, `/opsx:continue`, `/opsx:apply`, `/opsx:verify`, `/opsx:archive`) to manage changes. The workflow is: proposal → design → specs → tasks → implementation → verification → archive.

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on pushes to `main` and `develop`, and on PRs targeting either branch.

**Jobs:**
- **Format Check** — `cargo fmt --check`
- **Clippy Lint** — all host-testable crates
- **Unit Tests** — all host-testable crates
- **Build** — x86-64, AArch64, RISC-V bare-metal kernels
- **RISC-V QEMU Smoke Test** — boots kernel in QEMU
- **Image Size Check** — ensures binaries stay under 15 MB
- **TLA+ Verification** — runs TLC on 19 formal models (5 min timeout per model; timeouts are warnings, not failures)
- **Code Coverage** — `cargo-llvm-cov` with lcov output, uploaded to [Codecov](https://codecov.io)
- **SonarCloud Analysis** — static analysis via [SonarCloud](https://sonarcloud.io)
- **Change Gates** — meta-job that gates PR mergeability

**Required secrets:** `CODECOV_TOKEN`, `SONAR_TOKEN`

## Crate Feature Flags

- `kernel`: `verbose-boot`, `no-global-alloc`, `large-memory` (64 GiB page tracking; default 1 GiB)
- `security`: `pqc-hybrid` (default), `pqc-only`, `classical-only`, `formal-gate`
- `onnx-rt`: `cpu` (default), `cuda`, `formal-gate`
- `net`: `ipv4`, `ipv6` (both default)
- `container`: `nvidia_gpu`, `formal-gate`
- `ipc`: `formal-gate`
- `arch/nvidia`: `cc_53` through `cc_100` (CUDA compute capabilities)
- `peripheral`: `i2c`, `spi`, `gpio`, `uart`, `camera-csi`, `audio-i2s`. Bundles: `sensor-io`, `vision`, `audio`, `full-peripheral`. All default OFF.
