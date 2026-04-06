# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmallAIOS is a minimal, secure, Rust-based OS kernel purpose-built for AI inference workloads. It boots directly to ONNX inference with ~46 syscalls (vs Linux ~450). Targets x86-64, ARM64, and RISC-V. Deploys as either a container (Docker/K8s) or bare-metal/VM via QEMU.

**Current state:** Prototype phase — 4,143 tests passing. Production-quality networking (IPv4/IPv6/TCP/ARP/NDP), QUIC/HTTP3 with TLS 1.3, protobuf parser, ONNX runtime with 6 real operators, full PQC crypto stack (SHA-3, AES-256-GCM, ML-KEM-768, ML-DSA-65, Ed25519, X25519), capability system. GPU crates (NVIDIA, Intel, AMD) are architectural stubs with HAL interfaces but no hardware interaction.

## Build Commands

Requires Rust nightly (pinned in `rust-toolchain.toml`). Uses `just` as the task runner wrapping Cargo commands. Run `just --list` to see all available recipes.

```bash
# Container mode (library OS)
just build-container-x86    # x86_64-unknown-linux-musl
just build-container-arm    # aarch64-unknown-linux-musl

# Kernel mode (VM/bare metal)
just build-kernel-x86       # x86_64-unknown-none
just build-kernel-arm       # aarch64-unknown-none

# Testing
just test                   # cargo test --workspace
just clippy                 # cargo clippy -- -D warnings
just fmt                    # cargo fmt
just fmt-check              # cargo fmt -- --check

# QEMU
just run-x86                # Boot in QEMU x86-64
just run-arm                # Boot in QEMU ARM64

# Docker
just docker-build           # Multi-arch container build

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

# Optional analysis tools
cargo install cargo-depgraph cargo-modules --locked  # dependency visualization
sudo apt install graphviz                            # SVG graph rendering
```

### Pre-Commit Hooks

Git hooks at `.githooks/pre-commit` run `cargo fmt --check`, `cargo clippy`, and cycle detection before each commit. Install with `just setup-hooks`. Run manually with `just check`. The same checks run in CI.

## Workspace Architecture

18-crate Rust workspace (`#![no_std]`, edition 2021). Strict 4-layer acyclic dependency model (see `docs/architecture.md` for full details):

```
Layer 3 — Integration:  container, bench
Layer 2 — HAL/Drivers:  arch/{x86_64,aarch64,riscv64,nvidia,intel_gpu,amd}, peripheral, bus, sdr
Layer 1 — Core Services: net, ipc, posix, onnx-rt, usb
Layer 0 — Foundation:    kernel → security
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

Releases use [`cargo-release`](https://github.com/crate-ci/cargo-release), configured in `release.toml`. All 18 crates share a single version and are bumped together. See `docs/release-runbook.md` for the full step-by-step process.

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

Active specs in `openspec/changes/`, archived specs in `openspec/changes/archive/` (with `YYYY-MM-DD-` date prefixes). Reference specs in `openspec/smallaios-kernel/`.

| Change | Status | Tasks | Focus |
|--------|--------|-------|-------|
| `architecture-documentation-v1` | Active | In progress | DSM tooling, architecture docs, archive consolidation |
| `smallaios-kernel-v1` | Archived | 143/144 | Core kernel — 1 task DEFERRED (sphinx-needs) |
| `platform-expansion-v2` | Archived | 191/198 | RISC-V, buses — 7 tasks DEFERRED (hardware-dependent) |
| `codeql-remediation-v1` | Archived | 23/25 | CodeQL fixes — 2 tasks DEFERRED (admin gates) |

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
- **Dependency Analysis** — crate/module dependency graphs, DSM matrix, DSM metrics analysis
- **Change Gates** — meta-job that gates PR mergeability

**Required secrets:** `CODECOV_TOKEN`, `SONAR_TOKEN`

## Crate Feature Flags

- `kernel`: `verbose-boot`, `no-global-alloc`, `large-memory` (64 GiB page tracking; default 1 GiB), `verified-boot` (boot integrity verification + measurement log)
- `security`: `pqc-hybrid` (default), `pqc-only`, `classical-only`, `formal-gate`, `verified-boot` (boot signature verification APIs)
- `onnx-rt`: `cpu` (default), `cuda`, `formal-gate`, `verified-boot` (model signature verification at load time)
- `net`: `ipv4`, `ipv6` (both default)
- `container`: `nvidia_gpu`, `formal-gate`
- `ipc`: `formal-gate`
- `arch/nvidia`: `cc_53` through `cc_100` (CUDA compute capabilities)
- `peripheral`: `i2c`, `spi`, `gpio`, `uart`, `camera-csi`, `audio-i2s`. Bundles: `sensor-io`, `vision`, `audio`, `full-peripheral`. All default OFF.
