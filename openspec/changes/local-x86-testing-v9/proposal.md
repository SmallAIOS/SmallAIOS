## Why

SmallAIOS can build for x86_64 in both container mode (`x86_64-unknown-linux-musl`) and bare-metal mode (`x86_64-unknown-none`), and it has a QEMU runner (`make run-x86`). However, there is no documented or tooled workflow for a developer to run SmallAIOS locally on their x86 workstation with GPU acceleration, debug it interactively, or test it in a VM environment other than QEMU. The user has a local x86-64 machine with an NVIDIA GPU accessible via Docker Desktop and wants to validate SmallAIOS before deploying to Jetson Nano hardware.

Three local testing paths need to be built out:

1. **Docker Desktop with NVIDIA GPU** — the container mode binary already targets musl, but there is no Dockerfile, no docker-compose, and no integration with the NVIDIA Container Toolkit for GPU passthrough. The `docker-build` Makefile target uses `buildx` for multi-arch but produces no usable local development image.

2. **VMware** — developers may want to boot the bare-metal kernel in a full VM with persistent disk, snapshot support, and optional PCIe passthrough. There is no VMware-compatible disk image builder.

3. **QEMU x86_64** — `make run-x86` boots the kernel but lacks GDB integration, serial logging to file, network device emulation, and other development amenities that make iterative kernel debugging productive.

## What Changes

- **Dockerfile + docker-compose** for local x86 container mode: multi-stage build (Rust nightly builder stage, minimal runtime stage), NVIDIA Container Toolkit integration (`--gpus all`), volume mounts for ONNX models, health check endpoint, and optional GPU feature flag
- **VMware disk image builder**: script to create a VMDK from the bare-metal x86 kernel binary — BIOS/UEFI-bootable via GRUB, GPT partitioned, with a VMX configuration template for VMware Workstation/Fusion
- **QEMU development workflow improvements**: GDB stub (`-s -S`), serial log to file, virtio-net with TAP/user-mode networking, monitor console, debug vs release targets, and a `make debug-x86` convenience target

## Capabilities

### New Capabilities
- `docker-gpu-runtime`: Dockerfile and docker-compose.yml for local x86 container mode with NVIDIA GPU passthrough via the NVIDIA Container Toolkit; includes multi-stage build, health checks, model volume mounts, and `make docker-local-gpu` target
- `vmware-image`: Shell script and Makefile target to create a VMDK disk image from the bare-metal x86 kernel, with GRUB bootloader, GPT partition table, and a VMX template for VMware Workstation/Fusion; `make vmware-x86` target
- `qemu-dev-workflow`: Enhanced QEMU development workflow with GDB debugging (`-s -S`), serial log capture, virtio-net networking, QEMU monitor, and `make debug-x86` / `make run-x86-net` convenience targets

### Modified Capabilities
<!-- No existing spec-level requirements change; this is all additive -->

## Impact

- **`container/`**: No code changes — the existing container crate compiles to musl and is the binary the Dockerfile packages
- **Build system (`Makefile`)**: New targets: `docker-local-gpu`, `vmware-x86`, `debug-x86`, `run-x86-net`
- **New files**: `Dockerfile`, `docker-compose.yml`, `scripts/make-vmware-x86.sh`, `scripts/vmware-template.vmx`
- **Existing QEMU targets**: `run-x86` unchanged; new `debug-x86` and `run-x86-net` are additive
- **CI**: Add a `docker-build-local` smoke test job that builds the Dockerfile (no GPU required in CI)
- **Documentation**: `docs/local-testing.md` with setup instructions for all three paths
- **No crate code changes** — this change is purely build tooling, scripts, and configuration
