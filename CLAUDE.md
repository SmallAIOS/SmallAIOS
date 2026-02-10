# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmallAIOS is a minimal, secure, Rust-based OS kernel purpose-built for AI inference workloads. It boots directly to ONNX inference with ~46 syscalls (vs Linux ~450). Targets x86-64, ARM64, and NVIDIA GPU. Deploys as either a container (Docker/K8s) or bare-metal/VM via QEMU.

**Current state:** Early bootstrap — comprehensive OpenSpec specifications exist (183+ tasks across 13 phases) but code is scaffold/stubs only.

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

10-crate Rust workspace (`#![no_std]`, edition 2024). Dependency flow:

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

## OpenSpec Workflow

This project uses OpenSpec for spec-driven development. Specifications live in `openspec/`.

- **Active change:** `openspec/changes/smallaios-kernel-v1/` — contains proposal, design, specs (8 files), and tasks
- **Reference specs:** `openspec/smallaios-kernel/` — canonical specifications and design docs
- **Config:** `.openspec/config.yaml`

Use OpenSpec skills (e.g. `/opsx:new`, `/opsx:continue`, `/opsx:apply`, `/opsx:verify`, `/opsx:archive`) to manage changes. The workflow is: proposal → design → specs → tasks → implementation → verification → archive.

## Crate Feature Flags

- `security`: `pqc-hybrid` (default), `pqc-only`, `classical-only`
- `onnx-rt`: `cpu` (default), `cuda`
- `net`: `ipv4`, `ipv6` (both default)
- `container`: `nvidia_gpu`
- `kernel`: `verbose-boot`
- `arch/nvidia`: `cc_53` through `cc_100` (CUDA compute capabilities)
