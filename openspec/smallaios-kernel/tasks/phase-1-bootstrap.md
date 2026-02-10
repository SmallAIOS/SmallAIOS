# Phase 1: Bootstrap — Minimal Boot to Serial Output

## Objective

Get a minimal Rust `#![no_std]` kernel that boots on x86-64 and ARM64 and prints
"SmallAIOS" to the serial console. This validates the toolchain, build system,
boot process, and basic HAL.

## Prerequisites

- Rust nightly toolchain with `x86_64-unknown-none` and `aarch64-unknown-none` targets
- QEMU for testing (x86_64 and aarch64 system emulators)
- `cargo-make` or equivalent task runner

## Tasks

### 1.1 Cargo Workspace Setup
- [ ] Root `Cargo.toml` with workspace members
- [ ] `kernel/Cargo.toml` with `#![no_std]`, `#![no_main]`
- [ ] `arch/x86_64/Cargo.toml`
- [ ] `arch/aarch64/Cargo.toml`
- [ ] Custom target specs (`.json`) for freestanding targets
- [ ] `.cargo/config.toml` with build targets and linker scripts

### 1.2 x86-64 Boot
- [ ] Linker script (`x86_64.ld`) defining kernel memory layout
- [ ] Assembly entry point: set up stack, clear BSS, call Rust `kernel_main`
- [ ] Minimal GDT (kernel code + data segments)
- [ ] Minimal IDT (exception handlers for debugging: #DE, #PF, #GP, #DF)
- [ ] Serial port output (COM1 at 0x3F8): `outb`-based character output
- [ ] Boot via Multiboot2 header (for QEMU `-kernel` flag)

### 1.3 ARM64 Boot
- [ ] Linker script (`aarch64.ld`)
- [ ] Assembly entry: disable MMU, set up stack, clear BSS, call `kernel_main`
- [ ] Exception vector table (minimal: catch and print sync exceptions)
- [ ] PL011 UART output (at address from QEMU virt machine DTB)
- [ ] Boot via QEMU `-kernel` direct kernel load

### 1.4 Kernel Entry
- [ ] `kernel_main()` function: print "SmallAIOS v0.1.0 booting..."
- [ ] Panic handler that prints panic message to serial and halts
- [ ] Global allocator stub (panic on allocation — no heap yet)

### 1.5 Build System
- [ ] `Makefile` or `cargo-make` tasks:
  - `build-x86_64`: Cross-compile for x86-64
  - `build-aarch64`: Cross-compile for ARM64
  - `run-x86_64`: Build + launch QEMU x86_64
  - `run-aarch64`: Build + launch QEMU aarch64
  - `test`: Run unit tests in hosted mode

### 1.6 CI Setup
- [ ] GitHub Actions workflow: build both architectures, run QEMU smoke test
- [ ] `rustfmt` and `clippy` checks

## Exit Criteria

- `cargo build` succeeds for both targets with zero warnings
- QEMU x86_64: boots and prints "SmallAIOS" to serial output
- QEMU aarch64: boots and prints "SmallAIOS" to serial output
- Panic handler works (intentional panic prints message and halts)
- CI passes on both architectures
