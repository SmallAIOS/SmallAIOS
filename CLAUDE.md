# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmallAIOS is a minimal, secure, Rust-based OS kernel purpose-built for AI inference workloads. It boots directly to ONNX inference with ~46 syscalls (vs Linux ~450). Targets x86-64, ARM64, and NVIDIA GPU. Deploys as either a container (Docker/K8s) or bare-metal/VM via QEMU.

**Current state:** Prototype phase — 5,784+ tests passing. Production-quality networking (IPv4/IPv6/TCP/ARP/NDP), QUIC/HTTP3 with TLS 1.3, protobuf parser, ONNX runtime with 6 real operators, full PQC crypto stack (SHA-3, AES-256-GCM, ML-KEM-768, ML-DSA-65, Ed25519, X25519), capability system, and NVIDIA GPU compute stack.

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

11-crate Rust workspace (`#![no_std]`, edition 2021). Dependency flow:

```
kernel (foundation)
├── security (capability-based access, PQC crypto)
├── arch/x86_64 (HAL: boot, GDT, IDT, APIC, paging)
├── arch/aarch64 (HAL: boot, GICv3, paging, SVE, PSCI)
├── arch/nvidia (HAL: PCIe, GPU init, compute, DMA)
├── onnx-rt (parser, optimizer, execution providers)
│   └── optionally uses arch/nvidia
├── ipc (Zenoh-inspired pub/sub messaging)
│   └── uses security
├── net (IPv4/IPv6, TCP/UDP native stack)
├── posix (minimal POSIX compat layer)
│   └── uses net
├── peripheral (I2C, SPI, GPIO, UART, CSI camera, I2S audio)
│   └── uses kernel, security
└── container (entry point, config, health, metrics)
    └── orchestrates all crates
```

## Key Design Decisions

- **Unikernel** — single address space, no microkernel IPC overhead
- **Cooperative async scheduling** — yields at ONNX operator boundaries
- **Clean-room ONNX runtime** — from-scratch `#![no_std]` Rust, no external C deps
- **Post-quantum crypto default** — ML-KEM-768 + ML-DSA-65 hybrid mode
- **DO-178C DAL A compliance target** — MC/DC 100% coverage on safety-critical paths
- **Formal verification** — TLA+ (concurrency), Lean 4 (type proofs), SPIN (protocols)
- **Size goals** — <8 MB base, <15 MB container, <50ms container boot

## Build Configuration

- **Toolchain:** nightly-2026-02-01, components: rust-src, rustfmt, clippy, llvm-tools
- **Targets:** x86_64-unknown-none, aarch64-unknown-none, x86_64-unknown-linux-musl, aarch64-unknown-linux-musl
- **Release profile:** `opt-level = "z"`, LTO enabled, single codegen unit (size-optimized)
- **Linker scripts:** Custom per bare-metal target (see `.cargo/config.toml`)

## Git Workflow

**Gitflow branching model:** feature branches -> `develop` -> `main`

- `main` — stable, release-ready code
- `develop` — integration branch for ongoing work
- Feature branches merge into `develop` via PR
- `develop` merges into `main` for releases

### Worktrees + OpenSpec

Each OpenSpec change gets its own git worktree and branch. One change = one branch = one PR.

```
Main repo:     /home/e/Development/SmallAIOS-Design              (main)
Worktrees:     /home/e/Development/SmallAIOS-Design-worktrees/
  kernel-v1:     .../smallaios-kernel-v1          (change/smallaios-kernel-v1)
  platform:      .../platform-expansion-v2        (change/platform-expansion-v2)
  cybersec:      .../cybersecurity-compliance-v3   (change/cybersecurity-compliance-v3)
```

**Branch naming:** `change/<openspec-change-name>`

**Workflow:**
1. Work in the worktree directory for your change
2. Use OpenSpec skills (`/opsx:apply`, `/opsx:continue`, etc.) to implement tasks
3. Commit to the change branch, push, create PR against `develop`
4. After merge, update worktree: `git pull origin develop`

## OpenSpec Changes

Specifications live in `openspec/`. Reference specs in `openspec/smallaios-kernel/`.

| Change | Tasks Done | Total | Focus |
|--------|-----------|-------|-------|
| `smallaios-kernel-v1` | 130 | 144 | Core kernel, memory, scheduler, crypto, ONNX, networking |
| `platform-expansion-v2` | 191 | 198 | RISC-V, CAN/ARINC/MIL-STD buses, K8s, DDS, QUIC |
| `cybersecurity-compliance-v3` | 110 | 110 | NIST SP 800-53, audit, supply chain, incident response |

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
- **TLA+ Verification** — runs TLC on all formal protocol models
- **Code Coverage** — `cargo-llvm-cov` with lcov output, uploaded to [Codecov](https://codecov.io)
- **SonarCloud Analysis** — static analysis via [SonarCloud](https://sonarcloud.io)
- **Change Gates** — meta-job that gates PR mergeability

**Required secrets:** `CODECOV_TOKEN`, `SONAR_TOKEN`

## Crate Feature Flags

- `security`: `pqc-hybrid` (default), `pqc-only`, `classical-only`
- `onnx-rt`: `cpu` (default), `cuda`
- `net`: `ipv4`, `ipv6` (both default)
- `container`: `nvidia_gpu`
- `kernel`: `verbose-boot`
- `arch/nvidia`: `cc_53` through `cc_100` (CUDA compute capabilities)
- `peripheral`: `i2c`, `spi`, `gpio`, `uart`, `camera-csi` (requires `i2c`), `audio-i2s` (requires `i2c`). Convenience bundles: `sensor-io`, `vision`, `audio`, `full-peripheral`. All default OFF.
