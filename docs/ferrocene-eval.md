# Ferrocene Compiler Evaluation

**Date:** 2026-03-07
**Status:** Evaluation Only (no migration commitment)

## 1. Nightly Features Audit

The SmallAIOS workspace uses the following nightly/unstable features:

| Feature | Usage Location | Ferrocene Status |
|---------|---------------|-----------------|
| `#[unsafe(naked)]` / `naked_asm!` | `arch/x86_64/src/boot.rs`, `syscall.rs`; `arch/aarch64/src/boot.rs`, `syscall.rs`; `arch/riscv64/src/boot.rs`, `trap.rs` | Rust 2024 edition syntax; Ferrocene tracks upstream stabilization |
| `core::arch::asm!` | All arch crates (inline assembly for MMIO, interrupts, page tables) | Stabilized in Rust 1.59; supported by Ferrocene |
| `-Z build-std=core,compiler_builtins,alloc` | All bare-metal targets (rebuilds core/alloc from source) | Ferrocene ships prebuilt `core`/`alloc` for qualified targets; `-Z build-std` may not be needed |
| `#![no_std]` + `#![no_main]` | All crates (no_std); arch crates (no_main) | Fully supported by Ferrocene |
| `&raw const` / `&raw mut` | `kernel/src/state.rs`, `kernel/src/mem/phys.rs` | Rust 2024 edition syntax; Ferrocene tracks upstream |
| `alloc` crate (no_std) | Kernel, security, onnx-rt, net, ipc | Ferrocene qualifies `core` and `alloc` |
| `compiler_builtins` with `mem` feature | Build-std configuration | Ferrocene provides qualified compiler-builtins |

### Nightly-Only Risk Items

1. **`naked_asm!` macro** — Stabilized in Rust 1.79 (2024). Available in Ferrocene nightly channel but may require specific Ferrocene version.
2. **`-Z build-std`** — Ferrocene provides prebuilt standard library for qualified targets. If our targets are qualified, this flag is unnecessary. If not, we would need Ferrocene's `-Z build-std` support.
3. **Rust 2024 edition** — SmallAIOS uses edition 2024. Ferrocene typically lags upstream by one edition cycle for qualification.

## 2. Target Support

| Target Triple | SmallAIOS Usage | Ferrocene Support |
|--------------|----------------|-------------------|
| `x86_64-unknown-none` | Primary kernel target | **Qualified** (Tier 1 in Ferrocene) |
| `aarch64-unknown-none` | ARM64 kernel + Jetson | **Qualified** (Tier 1 in Ferrocene) |
| `riscv64gc-unknown-none-elf` | RISC-V kernel | **Not yet qualified** — Ferrocene has RISC-V support in preview |
| `x86_64-unknown-linux-musl` | Container mode | **Qualified** (Linux hosted targets are Tier 1) |
| `aarch64-unknown-linux-musl` | Container mode | **Qualified** |

### Gap: RISC-V bare-metal is not yet qualified by Ferrocene. This means RISC-V kernel builds would not carry certification artifacts.

## 3. Build Compatibility Assessment

**Cannot attempt actual build** — Ferrocene prebuilt toolchains require a commercial license. This evaluation is based on documentation review and source code analysis.

### Expected Compatibility Issues

1. **Edition 2024**: Ferrocene may not yet support Rust 2024 edition. SmallAIOS uses `unsafe(...)` attribute syntax and `&raw` operators that are edition 2024 features. Workaround: these features may be available under `#![feature(...)]` on older editions.

2. **Pinned nightly date**: We pin `nightly-2026-02-01`. Ferrocene releases correspond to specific upstream Rust versions (e.g., Ferrocene 24.11.0 ≈ Rust 1.83). We would need to verify API compatibility with the Ferrocene version closest to our pinned nightly.

3. **Build-std**: If Ferrocene provides prebuilt `core`/`alloc` for our targets, we can drop `-Z build-std` and simplify the build. This is actually a benefit.

4. **Workspace size**: 18 crates compile cleanly with upstream nightly. Ferrocene should handle this without issues — it's a fork of rustc, not a reimplementation.

## 4. Qualification Artifacts & Costs

### What Ferrocene Provides
- **Compiler qualification kit**: ISO 26262, IEC 61508, IEC 62304, DO-178C tool qualification evidence
- **TÜV SÜD certification**: Independently certified compiler for safety-critical use
- **Prebuilt toolchains**: Qualified binaries for supported targets
- **Support**: Commercial support for safety-critical deployments

### License Model
- **Source code**: MIT + Apache-2.0 (open source on GitHub, community can build from source)
- **Prebuilt binaries + certification docs**: Commercial license required
- **Pricing**: Contact Ferrous Systems for quotes; typically per-seat annual license
- **Qualification evidence**: Available only with commercial license

### What We Need for Certification
1. Tool Qualification Plan referencing Ferrocene qualification kit
2. Map Ferrocene's Tool Confidence Level (TCL) to our DAL A/ASIL D requirements
3. Configuration management evidence (pinned Ferrocene version)
4. Integration testing evidence (our test suite passing on Ferrocene)

## 5. Go/No-Go Recommendation

### Recommendation: **CONDITIONAL GO** — proceed with evaluation build when budget allows

**Reasons to proceed:**
- x86-64 and AArch64 bare-metal targets are qualified (our primary targets)
- Inline assembly (`asm!`) is stabilized and supported
- Ferrocene is the only TÜV SÜD-qualified Rust compiler available
- Our `#![no_std]` + `alloc` usage pattern is Ferrocene's primary use case
- Open source means we can evaluate source compatibility before purchasing

**Blockers to resolve first:**
1. **RISC-V qualification gap** — acceptable if RISC-V is non-safety-critical
2. **Edition 2024 support** — verify Ferrocene supports `unsafe(...)` syntax or plan migration to feature gates
3. **Budget approval** — commercial license cost needs approval
4. **Nightly features audit** — `naked_asm!` must be available in the Ferrocene version we choose

**Suggested next steps:**
1. Build SmallAIOS from source using Ferrocene's open-source toolchain (free)
2. Document build failures and required code changes
3. Estimate migration effort in developer-hours
4. Request commercial license quote from Ferrous Systems
5. Make final go/no-go with cost/benefit analysis
