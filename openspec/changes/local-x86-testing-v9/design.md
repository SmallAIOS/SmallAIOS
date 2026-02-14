## Context

SmallAIOS builds for x86_64 in two modes: container (`x86_64-unknown-linux-musl`, a statically-linked Linux binary) and bare-metal (`x86_64-unknown-none`, a freestanding kernel). The container binary runs as a normal Linux process and is the primary deployment target for Docker/K8s. The bare-metal kernel boots via Multiboot2 in QEMU.

The user's local machine is x86-64 with an NVIDIA GPU, running Docker Desktop. Docker Desktop supports GPU passthrough through the NVIDIA Container Toolkit (nvidia-docker2 / nvidia-container-runtime). The existing `make docker-build` target produces a multi-arch image but is designed for CI/registry push, not local development.

QEMU is already functional (`make run-x86`) but the invocation is minimal: no GDB, no network, no serial log file. Kernel debugging requires manually adding flags.

VMware Workstation/Fusion can boot raw disk images if they are packaged as VMDK with a bootloader. This is useful for testing with persistent disk, snapshots, and optional PCIe passthrough to real hardware.

## Goals / Non-Goals

**Goals:**
- Provide a one-command Docker workflow to run SmallAIOS container mode locally with NVIDIA GPU (`make docker-local-gpu`)
- Provide a one-command Docker workflow without GPU for CPU-only testing (`make docker-local`)
- Create a VMware-bootable disk image from the bare-metal x86 kernel (`make vmware-x86`)
- Add GDB debugging support to QEMU (`make debug-x86`) with source-level breakpoints
- Add network-enabled QEMU mode with virtio-net (`make run-x86-net`)
- Add serial output logging to file for post-mortem analysis
- Document all three local testing paths

**Non-Goals:**
- Modifying any Rust crate code — this change is build tooling and scripts only
- KVM/libvirt support — QEMU user-mode is sufficient; KVM can be added later
- Hyper-V support — not in scope for this change
- VirtualBox support — VMware covers the VM use case
- Automated GPU testing in CI — CI runners don't have GPUs; only the Dockerfile build is tested
- UEFI boot for QEMU — Multiboot2/BIOS is already working and simpler
- WSL2 GPU passthrough — Docker Desktop handles this transparently

## Decisions

### 1. Multi-stage Dockerfile with optional GPU

**Decision:** Create a two-stage Dockerfile at the repository root:
- **Builder stage:** `rust:nightly-slim` base, installs musl target, copies workspace, runs `cargo build --release --target x86_64-unknown-linux-musl`
- **Runtime stage:** `scratch` (or `alpine:latest` for debugging), copies only the compiled binary

GPU support is controlled by a build arg `ENABLE_GPU=1` which adds `--features nvidia_gpu` to the cargo build. The docker-compose.yml conditionally adds the NVIDIA runtime.

**Why scratch?** SmallAIOS container mode is a statically-linked musl binary with no libc dependencies. `scratch` produces the smallest image. An `alpine` variant is provided for debugging (shell access, strace).

**Why not a separate GPU Dockerfile?** The only difference is a feature flag. A build arg keeps it DRY.

**Dockerfile structure:**
```dockerfile
FROM rust:nightly-slim AS builder
# Install musl target, copy source, build
ARG ENABLE_GPU=0
RUN if [ "$ENABLE_GPU" = "1" ]; then \
      cargo build --release --target x86_64-unknown-linux-musl --features nvidia_gpu; \
    else \
      cargo build --release --target x86_64-unknown-linux-musl; \
    fi

FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/smallaios-container /smallaios
ENTRYPOINT ["/smallaios"]
```

### 2. docker-compose.yml with NVIDIA Container Toolkit

**Decision:** Provide a `docker-compose.yml` with two service profiles:
- `cpu`: Default profile, no GPU, maps port 8080 (health/metrics) and mounts `./models/` for ONNX models
- `gpu`: Extends `cpu`, adds `runtime: nvidia` and `NVIDIA_VISIBLE_DEVICES=all`; builds with `ENABLE_GPU=1`

Usage:
```bash
make docker-local          # CPU-only
make docker-local-gpu      # With NVIDIA GPU
```

**Why docker-compose over plain docker run?** Compose captures the volume mounts, port mappings, environment variables, and GPU runtime config in a declarative file. It's also easier to extend for multi-container setups later (e.g., adding a model server sidecar).

**NVIDIA Container Toolkit prerequisites:** The host must have:
1. NVIDIA driver installed (host or WSL2)
2. `nvidia-container-toolkit` package installed
3. Docker configured with nvidia runtime (`/etc/docker/daemon.json`)

These prerequisites are documented but not automated — they are one-time host setup.

### 3. VMware VMDK image via GRUB

**Decision:** Create `scripts/make-vmware-x86.sh` that:
1. Builds the bare-metal x86 kernel (`make build-kernel-x86`)
2. Creates a 64 MB raw disk image with GPT partition table + BIOS boot partition + ext4 data partition
3. Installs GRUB2 bootloader configured to chainload the SmallAIOS kernel via Multiboot2
4. Converts the raw image to VMDK format using `qemu-img convert -O vmdk`
5. Generates a `.vmx` file from a template with default settings (1 CPU, 512 MB RAM, VMDK attached)

Output: `build/smallaios-x86.vmdk` and `build/smallaios-x86.vmx`

**Why GRUB and not direct kernel boot?** VMware doesn't support Multiboot2 natively. GRUB is the standard way to boot Multiboot2 kernels in VM environments. GRUB is installed into the BIOS boot partition of the GPT image.

**Why VMDK and not OVA?** VMDK + VMX is the simplest format that works with both VMware Workstation and Fusion. OVA is an archive format that wraps VMDK + OVF; it adds packaging complexity without benefit for local testing.

**VMX template settings:**
- `guestOS = "other-64"`
- `memsize = "512"`
- `numvcpus = "2"`
- `scsi0.virtualDev = "lsilogic"`
- Serial port mapped to a file for UART output capture

**Host tools required:** `grub-install` (or `grub2-install`), `sgdisk`/`gdisk`, `mkfs.ext4`, `qemu-img`, `losetup`. These are standard on most Linux development machines.

### 4. QEMU GDB debugging workflow

**Decision:** Add `make debug-x86` that:
1. Builds the debug (non-release) x86 kernel
2. Launches QEMU with `-s -S` (GDB stub on port 1234, pause at start)
3. Prints instructions: `gdb target/x86_64-unknown-none/debug/smallaios-x86_64 -ex "target remote :1234"`

The debug build uses `build-kernel-x86-debug` (already exists in the Makefile) which compiles without optimization, preserving debug info and symbol names.

**Why pause at start (`-S`)?** Without `-S`, QEMU starts executing immediately. The developer needs time to attach GDB and set breakpoints before `_start` runs. `-S` holds at the reset vector until GDB sends `continue`.

**GDB helper script:** Create `.gdbinit-x86` with:
```
target remote :1234
break kernel_main
continue
```

### 5. QEMU networking with virtio-net

**Decision:** Add `make run-x86-net` that launches QEMU with:
```
-device virtio-net-pci,netdev=net0 -netdev user,id=net0,hostfwd=tcp::8080-:8080
```

This gives the kernel a virtio-net NIC (which SmallAIOS doesn't drive yet, but the MMIO shows up for enumeration testing) and forwards host port 8080 to guest port 8080 for future health check / metrics access.

**Why user-mode networking and not TAP?** TAP requires root and host bridge configuration. User-mode networking works without privileges and provides DHCP + NAT automatically. TAP can be added as an advanced option later.

### 6. Serial log capture

**Decision:** All QEMU invocations gain an optional `-serial file:build/serial.log` alongside `-serial stdio`. QEMU supports multiple serial ports; the first goes to stdio for interactive use, the second to a file for post-mortem analysis.

Updated QEMU flags for `run-x86`:
```
-serial stdio -serial file:build/serial.log
```

The `build/` directory is already in `.gitignore`.

### 7. Documentation

**Decision:** Create `docs/local-testing.md` covering:
1. Prerequisites (Rust nightly, Docker Desktop, NVIDIA Container Toolkit, QEMU, VMware)
2. Docker CPU-only quickstart
3. Docker GPU quickstart
4. QEMU bare-metal boot
5. QEMU GDB debugging
6. VMware image creation and boot
7. Troubleshooting common issues

This is a single document, not split per capability, because developers will want to compare the three approaches.

## Risks / Trade-offs

**[Docker build time]** The multi-stage Dockerfile rebuilds the entire workspace on every source change because Docker invalidates the cache when any file in the COPY context changes. Mitigation: use `cargo-chef` pattern (separate dependency-fetch layer) or mount a cargo cache volume. The initial implementation uses the simple approach; cache optimization is a follow-up.

**[VMware GRUB complexity]** Installing GRUB into a disk image requires `grub-install` with `--target=i386-pc` and a loop device. This is fragile across Linux distributions (GRUB2 package names and paths vary). Mitigation: document required packages per distro (Ubuntu: `grub-pc-bin`, Fedora: `grub2-pc`); test on Ubuntu as the primary dev platform.

**[No virtio-net driver]** SmallAIOS does not have a virtio-net driver, so `run-x86-net` adds a NIC that the kernel cannot use yet. This is intentional — the NIC shows up on the PCI bus and can be used for PCIe enumeration testing. A virtio-net driver is a future change.

**[WSL2 Docker GPU quirks]** GPU passthrough in Docker Desktop on WSL2 requires the NVIDIA WSL driver (not the regular Linux driver). The user's WSL2 kernel (6.6.87.2-microsoft-standard-WSL2) should support this. Mitigation: document the WSL2-specific setup steps.

**[No UEFI for VMware]** Using BIOS/GRUB means the VMware image doesn't test UEFI boot paths. This is acceptable because SmallAIOS uses Multiboot2, which is BIOS-era. UEFI boot support is a separate future capability.

## Open Questions

1. **Cargo chef vs simple COPY:** Should the Dockerfile use the cargo-chef pattern for layer caching from the start, or is the simple approach sufficient for initial local development?

2. **VMware PCIe passthrough:** Should the VMX template include optional PCIe passthrough configuration for the NVIDIA GPU? This would allow bare-metal GPU testing in VMware but requires specific host IOMMU setup.

3. **QEMU KVM acceleration:** Should `run-x86` and `debug-x86` auto-detect KVM availability and add `-enable-kvm` when possible? This would significantly speed up execution but may mask timing-related bugs.
