# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmallAIOS is a minimal, secure, Rust-based OS kernel purpose-built for AI inference workloads. It boots directly to ONNX inference with ~46 syscalls (vs Linux ~450). Targets x86-64, ARM64, and RISC-V. Deploys as either a container (Docker/K8s) or bare-metal/VM via QEMU.

**Current state:** Prototype phase — ~9,300 test executions passing across the CI matrix (`just test-all`, 2026-07-03; 6,917 in the default group, the rest in feature-gated groups that were dark before `ci-test-gates-v1`). Production-quality networking (IPv4/IPv6/TCP/ARP/NDP), QUIC/HTTP3 with TLS 1.3, protobuf parser, ONNX runtime with 29+ real operators, full PQC crypto stack (SHA-3, AES-256-GCM, ML-KEM-768, ML-DSA-65, Ed25519, X25519), capability system. NVIDIA CUDA container path validated on Jetson Orin NX (compute 8.7) via `Dockerfile.jetson` — cuDNN-backed Conv, cuBLAS-backed GEMM, captured CUDA Graphs, multi-stream overlap. AMD and Intel GPU crates remain architectural stubs. Jetson Orin **unikernel** path: Phase 1 (`unikernel-orin-bringup-v1`) lands a KVM-on-L4T smoke-test workflow — `just run-jetson-kvm SSH_HOST=user@orin` cross-builds the AArch64 kernel from any host and boots it under `qemu-system-aarch64 -accel kvm -cpu host` on the Orin's Cortex-A78AE cores; CI gate `aarch64-qemu-smoke` validates the same path under TCG. Phase 2 (Tegra234 UEFI boot) is in progress — ExitBootServices → `kernel_main` handoff verified on Orin NX hardware, GICv3 driver extracted and feature-gated, OVMF QEMU smoke job in CI; the AArch64 IRQ exception path + Generic Timer tick remain (paired with an on-hardware session).

## Build Commands

Requires Rust nightly (pinned in `rust-toolchain.toml`). Uses `just` as the task runner wrapping Cargo commands. Run `just --list` to see all available recipes.

```bash
# Container mode (library OS)
just build-container-x86    # x86_64-unknown-linux-musl
just build-container-arm    # aarch64-unknown-linux-musl

# Kernel mode (VM/bare metal)
just build-kernel-x86       # x86_64-unknown-none
just build-kernel-arm       # aarch64-unknown-none

# Testing — crate/feature groups are single-sourced in ci/test-matrix.toml
just test                   # default test group (mirrors the CI "Unit Tests" gate)
just test-all               # every host-compatible group (incl. feature-gated suites)
just test-group fs-features # one named group; `just test-matrix` lists groups + floors
just clippy                 # cargo clippy -- -D warnings (matrix-derived crate list)
just fmt                    # cargo fmt
just fmt-check              # cargo fmt -- --check

# QEMU
just run-x86                # Boot in QEMU x86-64
just run-arm                # Boot in QEMU ARM64

# Docker
just docker-build               # Multi-arch container build (CPU, slim, ~1 MB)
just docker-build-gpu           # x86 + discrete GPU (Dockerfile.cuda, ~3 GB)
just docker-build-jetson        # Jetson Orin full JetPack base (Dockerfile.jetson, ~10 GB)
just docker-build-jetson-slim   # Jetson Orin slim (Dockerfile.jetson.slim, ~4 GB)
just docker-local-jetson        # docker compose --profile jetson up --build
just docker-local-jetson-slim   # docker compose --profile jetson-slim up --build
just test-jetson-gpu            # End-to-end Jetson smoke test (full base; must run on Jetson)
just test-jetson-gpu slim       # End-to-end smoke test against the slim variant

# Dependency analysis (requires cargo-depgraph, cargo-modules, graphviz)
just depgraph               # Crate-level DOT/SVG dependency graph
just modgraph               # Module-level graphs for all host crates
just modgraph smallaios-kernel  # Single crate module graph
just arch-check             # Module-level acyclicity check
just dsm                    # DSM adjacency matrix (JSON + CSV)
just dsm-analyze            # DSM + propagation cost, fan-in/out, clusters
just arch                   # All of the above

# Release (requires cargo-release)
just release-dry-run patch  # preview version bump
just release minor          # execute bump + commit + tag
```

### Dev Tool Dependencies

```bash
# Required
rustup toolchain install nightly-2026-02-01
cargo install just --locked                          # task runner

# Pre-commit hooks (run once after clone)
just setup-hooks

# Safety-critical tooling (recommended)
cargo install cargo-audit --locked                   # CVE vulnerability check
cargo install cargo-geiger --locked                  # unsafe code audit
cargo install cargo-deny --locked                    # supply chain security
cargo install cargo-semver-checks --locked           # API breakage detection
cargo install cargo-vet --locked                     # dependency review audit trail
cargo install cargo-careful --locked                 # extra UB detection
cargo install cargo-llvm-cov --locked                # coverage threshold gate

# Optional analysis tools
cargo install cargo-depgraph cargo-modules --locked  # dependency visualization
sudo apt install graphviz                            # SVG graph rendering
```

### Pre-Commit Hooks

Git hooks at `.githooks/pre-commit` run before each commit:
1. `cargo fmt --check` — formatting (blocking)
2. `cargo clippy -D warnings` — lint (blocking)
3. `cargo-geiger` — unsafe code audit (advisory)
4. `cargo-audit` — CVE vulnerability check (advisory)
5. `cargo-semver-checks` — API breakage detection (advisory)
6. Dependency cycle check (blocking, on Cargo.toml changes)
7. Module acyclicity check (advisory)

Install with `just setup-hooks`. Run manually with `just check` (quick) or `just audit` (full safety audit).

## Workspace Architecture

23-crate Rust workspace (`#![no_std]`, edition 2021). Strict 4-layer acyclic dependency model (see `docs/architecture.md` for full details):

```
Layer 3 — Integration:  container, bench
Layer 2 — HAL/Drivers:  arch/{x86_64,aarch64,riscv64,nvidia,intel_gpu,amd,apple}, peripheral, bus, sdr
Layer 1 — Core Services: net, ipc, posix, onnx-rt, usb, auth, mgmt, fs, audit-export, tls-client
Layer 0 — Foundation:    kernel → security, compute, sched-types
```

**Dependency rules:** Higher layers depend on same or lower layers only. Zero production-dependency cycles. The DSM analysis tool (`tools/dsm/`) computes propagation cost, fan-in/out, coupling clusters, and layering violations from `build/analysis/dsm-matrix.json`.

```
just dsm-analyze    # Generate DSM + run analysis
just arch-check     # Module-level acyclicity check
just arch           # Full dependency analysis suite
```

## Key Design Decisions

- **Unikernel** — single address space, no microkernel IPC overhead
- **Cooperative async scheduling** — yields at ONNX operator boundaries (see `docs/scheduling-model.md` for POSIX/RTOS alignment)
- **AMP multi-core** — Core 0 for System/IPC, Cores 1-N for inference data parallelism; no SMP
- **Clean-room ONNX runtime** — from-scratch `#![no_std]` Rust, no external C deps
- **Post-quantum crypto default** — ML-KEM-768 + ML-DSA-65 hybrid mode
- **Clean-room crypto policy** — no C/C++ crypto libraries (cargo-deny enforced); every primitive replays an official vector corpus. See `docs/crypto-validation.md`
- **DO-178C DAL A compliance target** — MC/DC 100% coverage on safety-critical paths
- **Formal verification** — TLA+ (19 protocol models for concurrency/safety invariants)
- **Size goals** — <8 MB base, <15 MB container, <50ms container boot

## Build Configuration

- **Toolchain:** nightly-2026-02-01, components: rust-src, rustfmt, clippy, llvm-tools
- **Targets:** x86_64-unknown-none, aarch64-unknown-none, riscv64gc-unknown-none-elf, x86_64-unknown-linux-musl, aarch64-unknown-linux-musl
- **Release profile:** `opt-level = "z"`, LTO enabled, single codegen unit (size-optimized)
- **Linker scripts:** Custom per bare-metal target (see `.cargo/config.toml`)

## Versioning

**Semantic versioning** with conventional commits for PR titles.

- All workspace crates share a single version in `Cargo.toml` `[workspace.package]`
- PR titles must follow: `<type>[optional scope][!]: <description>`
- CI validates PR titles and auto-labels with `semver:major`, `semver:minor`, `semver:patch`, or `semver:none`

**Allowed types:** `feat`, `fix`, `docs`, `chore`, `ci`, `test`, `refactor`, `style`, `perf`, `build`, `revert`

**Bump rules (pre-1.0):**
| PR Title Pattern | Bump Level |
|-----------------|------------|
| `feat!:` or `fix!:` (breaking) | minor |
| `feat:` or `feat(scope):` | minor |
| `fix:`, `perf:`, `revert:` | patch |
| `docs:`, `chore:`, `ci:`, `test:`, `refactor:`, `style:`, `build:` | none |

**Scripts:**
- `./scripts/check-pr-semver.sh "<title>"` — validate and print bump level
- `./scripts/suggest-release-bump.sh` — scan commits since last tag, suggest aggregate bump level

### Releasing

Releases use [`cargo-release`](https://github.com/crate-ci/cargo-release), configured in `release.toml`. All 21 crates share a single version and are bumped together. See `docs/release-runbook.md` for the full step-by-step process.

```bash
just changelog                   # regenerate CHANGELOG.md via git-cliff
./scripts/suggest-release-bump.sh  # check suggested bump level
just release-dry-run patch       # preview: 0.1.0 → 0.1.1
just release minor               # execute: 0.1.0 → 0.2.0
```

`cargo-release` bumps workspace version, updates `Cargo.lock`, commits, and tags. The pre-release hook runs tests and generates the changelog via `git-cliff`. `push = false` so you review before pushing. Releases are only allowed from `main` (enforced by `allow-branch`).

**Development dependencies for releases:** `git-cliff` (changelog generation), `cargo-release` (version management).

## Git Workflow

**Gitflow branching model:** feature branches -> `develop` -> `main`

- `main` — stable, release-ready code
- `develop` — integration branch for ongoing work
- Feature/fix branches merge into `develop` via PR
- `develop` merges into `main` for releases

### Branch Naming Convention

| Branch Type | Pattern | Example |
|-------------|---------|---------|
| Feature | `feature/<name>` | `feature/onnx-protobuf-parser` |
| Bug fix | `fix/<name>` | `fix/gemm-overflow` |
| OpenSpec change | `change/<openspec-change-name>` | `change/compute-abstraction-v1` |
| Release | `release/<version>` | `release/0.2.0` |
| Hotfix | `hotfix/<name>` | `hotfix/critical-boot-fix` |

Use `change/` for OpenSpec-tracked work. Use `feature/` or `fix/` for ad-hoc work not tracked by OpenSpec. All branches target `develop` except hotfixes (which target `main`).

### Worktrees

Each active branch gets its own worktree for parallel development. Worktrees live in `../SmallAIOS-Design-worktrees/`.

**Workflow:**
1. Create branch from `develop`: `git checkout -b change/<name> develop`
2. Create worktree: `git worktree add ../SmallAIOS-Design-worktrees/<name> change/<name>`
3. Work in the worktree, commit, push, create PR against `develop`
4. After merge, clean up: `git worktree remove ../SmallAIOS-Design-worktrees/<name> --force && git branch -D change/<name>`

## OpenSpec Changes

Active specs in `openspec/changes/`, archived specs in `openspec/changes/archive/` (with `YYYY-MM-DD-` date prefixes). Reference specs in `openspec/smallaios-kernel/`. **`openspec/changes/` is the source of truth — this snapshot (2026-07-16) goes stale.**

Implementation in flight (per each change's `tasks.md`):

| Change | Tasks | Focus |
|--------|-------|-------|
| `verifiable-audit-log-v1` | 59/76 | immudb audit-export bridge — remaining: live-immudb proof fixtures, ALH recomputation, e2e + CI follow-ons |
| `embedded-flash-fs-v1` | 32/71 | littlefs-compatible raw-NAND/NOR flash filesystem |
| `unikernel-orin-bringup-v1` | 30/38 | Jetson Orin bare-metal boot — remaining: AArch64 IRQ exception path + Generic Timer tick (on-hardware session), Phase 2 PR |
| `tls-tcp-client-v1` | 42/66 | TLS 1.3-over-TCP client crate — real-CA ECDSA-P256/RSA-PSS chain + CertificateVerify verification landed (#231); remaining: 5.8 cross-vectors, 7.6 real-endpoint e2e (now unblocked), phases 8–11 |

~33 further changes sit at proposal stage (DO-178C DAL A / confidential-AI-edge roadmap, PRs #200–#204): boot-root-of-trust, op-tee-bridge, remote-attestation, confidential-compute, tegra-smmu-isolation, aarch64-mte-pac-hardening, ecc-scrubbing, deterministic-scheduling, watchdog-lockstep, dynamic-batching, llm-api-translation, the fpga-* series, and more — many gated on hardware access or prerequisite changes. 60 changes are archived (most recently the 2026-07-16 batch: `security-ecdsa-p256-v1`, `security-rsa-pss-v1`, `session-config-eager-validation-v1`, `crypto-validation-strategy-v1`, `ci-test-gates-v1` — main specs now carry `security-ecdsa-p256-verify`, `security-rsa-pss-verify`, `crypto-validation-policy`, `ci-test-matrix`).

Use OpenSpec skills (e.g. `/opsx:new`, `/opsx:continue`, `/opsx:apply`, `/opsx:verify`, `/opsx:archive`) to manage changes. The workflow is: proposal → design → specs → tasks → implementation → verification → archive.

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on pushes to `main` and `develop`, and on PRs targeting either branch.

Whether a job **blocks a merge** is decided by branch protection's required-status-check
list, not by anything in `ci.yml`. Verify with:

```bash
gh api repos/SmallAIOS/SmallAIOS/branches/develop/protection --jq '.required_status_checks.contexts[]'
```

**Enforced gates (required status checks on `develop`) — these block merge:**
- **Format Check** — `cargo fmt --check`
- **Clippy Lint** — matrix-derived crate/feature set (`ci/test-matrix.toml`)
- **Unit Tests** — the `default` group, executed-test counts enforced (zero-test runs fail)
- **Unit Tests (formal-gate)**, **Unit Tests (verified-boot)** — feature-flag variants
- **Build x86-64 / AArch64 / RISC-V / Jetson Nano (Tegra X1) Kernel** — bare-metal targets
- **Docker Build (Local)** — container image build
- **Code Coverage** — `cargo-llvm-cov` uploaded to [Codecov](https://codecov.io)
- **SonarCloud Analysis** — static analysis via [SonarCloud](https://sonarcloud.io)
- **Image Size Check (<15 MB)** — binaries stay under budget
- **RISC-V QEMU Smoke Test** — boots kernel in QEMU
- **TLA+ Formal Verification** — TLC on 19 formal models. **Required but toothless:** the job sets
  `continue-on-error: true`, so it reports success even when TLC fails. Drop that flag to make it real.

Branch protection also sets `strict: true` (branch must be current before merge). It does **not**
require approving reviews, and `enforce_admins` is off.

**Runs on every PR but does NOT block merge** — these fail the workflow run without stopping a merge,
because they are absent from the required-check list:
- **Change Gates** — meta-job depending on 18 jobs. It is *not itself a required check*, so despite
  the name it gates nothing; each of its dependencies blocks only if independently required. Adding
  `Change Gates` to the required list would make the whole set enforcing in one step.
- **Test Matrix Verify** — every workspace member classified in `ci/test-matrix.toml`
- **Unit Tests (\<group\>)** — fs-features, posix-features, tls-client, audit-export, tools,
  gpu-models, arch-apple (macOS runner), arch-x86_64
- **Coverage Threshold (93%)** — `cargo-llvm-cov --fail-under-lines 93` (ratcheted from 80% on
  2026-07-16; observed 93.20%)
- **Supply Chain Security (cargo-deny)** — license/advisory/ban checks
- **Cyclic Dependency Check** — no crate-level cycles
- **Semver PR Title Check** / **API Semver Check** — conventional-commit titles; `cargo-semver-checks`
- **CUDA Feature Check**, **AArch64 QEMU Smoke Test**, **Docker Build (ARM64 / GPU)**
- **Miri UB Detection**, **Fuzz Testing**, **Mutation Testing**, **Benchmark Regression Check**,
  **Dependency Analysis**

**Advisory by design (`continue-on-error: true` — always report green):**
- **Dependency Audit (cargo-vet)** — dependency audit trail (DO-178C traceability). Note the
  exemptions in `supply-chain/config.toml` pin exact versions, so every dependency bump needs a new
  entry or this job goes red.
- **Careful UB Testing** (`cargo-careful`), **Unsafe Usage Report** (`cargo-geiger`)
- **Kani Model Checking**, **SPIN Model Verification**
- **Jetson Image Build**, **Build Jetson Orin UEFI**, **Jetson Orin OVMF Smoke**
- **Spec-Exec Barrier Disasm Audit**, **Spec-Exec Default-On Smoke**

> **Known gap:** the two gating mechanisms disagree. Nine jobs that `change-gates` treats as
> mandatory — including `cargo-deny`, `Cyclic Dependency Check`, `API Semver Check`, and
> `Coverage Threshold (93%)` — are not required checks, so none of them can currently stop a merge.
> Conversely six required checks are absent from `change-gates`' `needs` — `Unit Tests
> (verified-boot)`, `Code Coverage`, `SonarCloud Analysis`, `Image Size Check (<15 MB)`,
> `RISC-V QEMU Smoke Test`, and `TLA+ Formal Verification`. Reconciling the two lists is tracked
> work, not intended behaviour. Re-derive both with `tools/ci/check-gate-parity.sh`.

**Required secrets:** `CODECOV_TOKEN`, `SONAR_TOKEN`

## Container Environment Variables

The container binary (`smallaios-container`) reads runtime configuration
from environment variables:

| Variable | Values | Purpose |
|----------|--------|---------|
| `SMALLAIOS_MODEL_DIR` | path | Directory of ONNX models to load at boot |
| `SMALLAIOS_PORT` | port | HTTP listen port (default `8080`) |
| `SMALLAIOS_GPU_BACKEND` | `cpu`, `cuda`, ... | Inference backend |
| `SMALLAIOS_BUS_BACKEND` | `none`, `zenoh`, `dds`, `can` | Optional bus-backed dataflow runner |
| `SMALLAIOS_CAN_DEVICE` | `loopback`, `mcp2515:<path>`, `axi:<addr>` | CAN controller for `bus_backend=can` |
| `SMALLAIOS_CAN_ROUTING` | path | TOML routing table for CAN inference |

See `docs/can-inference.md` for the CAN bus inference bridge and
`examples/can-routes.toml` for a routing table example.

## Crate Feature Flags

- `kernel`: `verbose-boot`, `no-global-alloc`, `large-memory` (64 GiB page tracking; default 1 GiB), `verified-boot` (boot integrity verification + measurement log)
- `security`: `pqc-hybrid` (default), `pqc-only`, `classical-only`, `formal-gate`, `verified-boot` (boot signature verification APIs)
- `onnx-rt`: `cpu` (default), `cuda`, `formal-gate`, `verified-boot` (model signature verification at load time), `gpu-profile` (per-op timing + memcpy byte counters for the hybrid GPU executor; dumps a summary to stderr at `CudaRuntime::drop`; off by default — production builds pay zero overhead)
- `net`: `ipv4`, `ipv6` (both default), `http2` (off; enables the HTTP/2 + gRPC subset used by `audit-export`)
- `container`: `nvidia_gpu`, `formal-gate`, `bus-zenoh`, `bus-dds` (pub/sub dataflow runner placeholders — see `docs/inference-bus.md`)
- `ipc`: `formal-gate`, `onnx` (opt-in ONNX runtime integration for the dataflow runner)
- `audit-export`: `bearer` (default v1 auth), `mtls` (v2 stub — refused by config validator), `formal-gate`. Compile-time opt-in to immudb verifiable-audit-log export (`verifiable-audit-log-v1`); when off, zero code is linked.
- `arch/nvidia`: `cc_53` through `cc_100` (CUDA compute capabilities)
- `peripheral`: `i2c`, `spi`, `gpio`, `uart`, `camera-csi`, `audio-i2s`. Bundles: `sensor-io`, `vision`, `audio`, `full-peripheral`. All default OFF.
