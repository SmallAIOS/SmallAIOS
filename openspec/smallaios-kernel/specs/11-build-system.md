# Spec 11: Build System and Toolchain

## Overview

SmallAIOS uses a Cargo workspace with cross-compilation support for all target
architectures. Builds are reproducible, hermetic, and produce minimal binaries.

## Toolchain Requirements

| Tool | Version | Purpose | License |
|---|---|---|---|
| `rustc` (nightly) | Latest nightly | Compiler (requires unstable features) | MIT/Apache-2.0 |
| `cargo` | Ships with rustc | Build system, dependency management | MIT/Apache-2.0 |
| `llvm` | Via rustc | Code generation, linking | Apache-2.0 w/ LLVM exception |
| `lld` | Via rustc | Linker (fast, cross-platform) | Apache-2.0 w/ LLVM exception |
| `ptxas` | CUDA Toolkit 12+ | PTX → SASS compilation (GPU only) | NVIDIA EULA |
| `qemu` | 8.0+ | Testing (x86_64 and aarch64 system) | GPL-2.0 (tool, not linked) |

## Unstable Rust Features Required

```rust
#![no_std]                    // Freestanding binary
#![no_main]                   // Custom entry point
#![feature(asm_const)]        // Constants in inline asm
#![feature(naked_functions)]  // Naked functions for entry points
#![feature(alloc_error_handler)] // Custom OOM handler
#![feature(lang_items)]       // Panic, eh_personality
```

## Cargo Workspace Layout

```toml
# Root Cargo.toml
[workspace]
resolver = "2"
members = [
    "kernel",
    "arch/x86_64",
    "arch/aarch64",
    "arch/nvidia",
    "onnx-rt",
    "ipc",
    "net",
    "posix",
    "security",
    "container",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
repository = "https://github.com/SmallAIOS/SmallAIOS"

[profile.release]
opt-level = "z"          # Optimize for size
lto = true               # Link-time optimization (whole program)
codegen-units = 1        # Single codegen unit for best optimization
panic = "abort"          # No unwinding (saves binary size)
strip = true             # Strip debug symbols from release
```

## Build Targets

### Custom Target Specifications

SmallAIOS uses custom target JSON specifications for bare metal:

```json
// x86_64-smallaios.json
{
    "llvm-target": "x86_64-unknown-none",
    "data-layout": "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
    "arch": "x86_64",
    "target-endian": "little",
    "target-pointer-width": "64",
    "os": "none",
    "executables": true,
    "linker-flavor": "ld.lld",
    "linker": "rust-lld",
    "panic-strategy": "abort",
    "disable-redzone": true,
    "features": "+sse,+sse2"
}
```

### Cargo Config

```toml
# .cargo/config.toml
[build]
# Default target for development
# target = "x86_64-unknown-linux-musl"  # Container mode

[target.x86_64-unknown-none]
runner = "scripts/run-qemu-x86.sh"
rustflags = ["-C", "link-arg=-Tarch/x86_64/linker.ld"]

[target.aarch64-unknown-none]
runner = "scripts/run-qemu-arm.sh"
rustflags = ["-C", "link-arg=-Tarch/aarch64/linker.ld"]

[target.x86_64-unknown-linux-musl]
rustflags = ["-C", "target-feature=+crt-static"]

[target.aarch64-unknown-linux-musl]
rustflags = ["-C", "target-feature=+crt-static"]
linker = "aarch64-linux-musl-gcc"
```

## Build Commands

```bash
# Container mode (library OS) — primary development target
make build-container-x86     # x86_64-unknown-linux-musl
make build-container-arm     # aarch64-unknown-linux-musl

# VM/bare metal mode
make build-kernel-x86        # x86_64 bare metal kernel
make build-kernel-arm        # aarch64 bare metal kernel

# With GPU support
make build-container-x86 GPU=1
make build-kernel-arm GPU=1

# Docker image
make docker-build            # Multi-arch OCI image
make docker-push             # Push to registry

# Run in QEMU
make run-x86                 # Boot kernel in QEMU x86_64
make run-arm                 # Boot kernel in QEMU aarch64

# Tests
make test                    # Unit tests (hosted)
make test-integration        # Integration tests (QEMU)
make test-fuzz               # Fuzzing (protobuf, syscall, IPC)
make clippy                  # Lint
make fmt-check               # Format check
```

## Makefile

```makefile
CARGO = cargo
DOCKER = docker
QEMU_X86 = qemu-system-x86_64
QEMU_ARM = qemu-system-aarch64

# Feature flags
ifdef GPU
  FEATURES += --features nvidia_gpu
endif

.PHONY: build-container-x86
build-container-x86:
	$(CARGO) build --release --target x86_64-unknown-linux-musl $(FEATURES)

.PHONY: build-container-arm
build-container-arm:
	$(CARGO) build --release --target aarch64-unknown-linux-musl $(FEATURES)

.PHONY: build-kernel-x86
build-kernel-x86:
	$(CARGO) build --release --target x86_64-unknown-none $(FEATURES)

.PHONY: build-kernel-arm
build-kernel-arm:
	$(CARGO) build --release --target aarch64-unknown-none $(FEATURES)

.PHONY: run-x86
run-x86: build-kernel-x86
	$(QEMU_X86) -machine q35 -cpu max -m 512M -nographic \
		-kernel target/x86_64-unknown-none/release/smallaios \
		-serial stdio

.PHONY: run-arm
run-arm: build-kernel-arm
	$(QEMU_ARM) -machine virt -cpu cortex-a72 -m 512M -nographic \
		-kernel target/aarch64-unknown-none/release/smallaios \
		-serial stdio

.PHONY: test
test:
	$(CARGO) test --workspace

.PHONY: docker-build
docker-build:
	$(DOCKER) buildx build --platform linux/amd64,linux/arm64 \
		-t smallaios/runtime:latest .

.PHONY: clean
clean:
	$(CARGO) clean
```

## Reproducible Builds

- **Pinned toolchain**: `rust-toolchain.toml` specifies exact nightly date
- **No external crate deps**: Eliminates crate registry as variable
- **LTO**: Whole-program optimization ensures deterministic output
- **Docker build**: Multi-stage build with pinned base image hashes
- **SBOM**: Generated at build time listing all tools and their versions

```toml
# rust-toolchain.toml
[toolchain]
channel = "nightly-2026-02-01"
components = ["rust-src", "rustfmt", "clippy", "llvm-tools"]
targets = [
    "x86_64-unknown-none",
    "aarch64-unknown-none",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
]
```

## CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  build:
    strategy:
      matrix:
        target:
          - x86_64-unknown-linux-musl
          - aarch64-unknown-linux-musl
          - x86_64-unknown-none
          - aarch64-unknown-none
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - run: cargo build --release --target ${{ matrix.target }}

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - run: cargo test --workspace
      - run: cargo clippy --workspace -- -D warnings
      - run: cargo fmt --all -- --check

  qemu-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - run: sudo apt-get install -y qemu-system-x86 qemu-system-arm
      - run: make run-x86 TIMEOUT=10  # Boot, verify serial output, exit
      - run: make run-arm TIMEOUT=10
```

## Binary Size Budget

| Component | Estimated Size | Notes |
|---|---|---|
| Kernel core | ~200 KB | Memory, scheduler, syscalls |
| x86-64 HAL | ~50 KB | Boot, GDT, IDT, APIC, paging |
| ARM64 HAL | ~50 KB | Boot, GIC, paging, PSCI |
| NVIDIA HAL | ~300 KB | GPU init, memory, compute, DMA |
| ONNX runtime | ~2 MB | Parser, optimizer, all operators |
| CPU SIMD kernels | ~500 KB | GEMM, conv, elementwise (both archs) |
| PTX kernels (GPU) | ~1 MB | Compiled SASS for supported SM versions |
| IPC + networking | ~200 KB | Message router, TCP, TLS |
| POSIX layer | ~100 KB | Subset implementation |
| Security + crypto | ~300 KB | PQC algorithms, capability system |
| **Total (CPU only)** | **~3.5 MB** | |
| **Total (CPU + GPU)** | **~5 MB** | |

With `opt-level=z` and LTO, actual sizes may be 20-40% smaller.
