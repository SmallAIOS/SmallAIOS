# Local Testing Guide

This guide covers building, running, and debugging SmallAIOS locally using Docker, QEMU, and VMware.

## Prerequisites

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  qemu-system-x86 \
  grub-pc-bin \
  gdisk \
  qemu-utils \
  docker.io \
  gdb-multiarch \
  telnet
```

For GPU support (optional):

```bash
# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Fedora

```bash
sudo dnf install -y \
  qemu-system-x86 \
  grub2-pc \
  gdisk \
  qemu-img \
  docker-ce \
  gdb \
  telnet
```

### Rust Toolchain

The project requires Rust nightly. The pinned version is specified in `rust-toolchain.toml`:

```bash
rustup show   # confirms nightly-2026-02-01 is active
```

If not installed:

```bash
rustup toolchain install nightly-2026-02-01
rustup target add x86_64-unknown-linux-musl --toolchain nightly-2026-02-01
```

## Docker CPU-Only

Build and run SmallAIOS as a container image (no GPU required):

```bash
make docker-local
```

This runs `docker compose up --build`, which:
1. Builds the Rust workspace targeting `x86_64-unknown-linux-musl`
2. Creates a scratch-based container with the `smallaios-container` binary
3. Starts the container with port 8080 mapped and `./models` mounted read-only

Expected output:

```
[+] Building ...
 => [builder] cargo build --release --target x86_64-unknown-linux-musl
 => COPY --from=builder /smallaios
[+] Running 1/1
 - Container smallaios  Started
```

Verify the image size:

```bash
docker image inspect smallaios:latest --format='{{.Size}}' | awk '{printf "%.2f MB\n", $1/1024/1024}'
```

The image must be under 15 MB.

To stop and clean up:

```bash
make docker-local-clean
```

## Docker with NVIDIA GPU

Build and run with GPU acceleration:

```bash
make docker-local-gpu
```

This runs `docker compose --profile gpu up --build`, which builds with `ENABLE_GPU=1` (enabling the `nvidia_gpu` feature) and starts with `runtime: nvidia`.

### Requirements

- NVIDIA driver installed on the host
- NVIDIA Container Toolkit configured (see Prerequisites)
- `nvidia-smi` working on the host

### WSL2 Notes

GPU passthrough works on WSL2 with these requirements:
- Windows 11 with WSL2 kernel 5.10.43+
- NVIDIA driver for Windows with WSL2 support (R530+)
- Docker Desktop with WSL2 backend, or Docker CE inside WSL2

Verify GPU is visible inside WSL2:

```bash
nvidia-smi
```

If `nvidia-smi` works on the host but Docker cannot see the GPU, ensure the NVIDIA Container Toolkit is installed inside WSL2 (not just on Windows).

## QEMU Bare-Metal Boot

Boot the x86-64 kernel in QEMU:

```bash
make run-x86
```

This builds the bare-metal kernel (`x86_64-unknown-none`) and launches QEMU with:
- Machine: `q35`
- CPU: `max`
- Memory: 512 MB
- Serial output to stdio and `build/serial.log`
- QEMU monitor on `telnet localhost:4444`

Expected serial output:

```
SmallAIOS v0.1.0 booting...
[kernel] Memory: 512 MB detected
[kernel] Scheduler initialized
[kernel] Ready
```

The kernel binary is at `target/x86_64-unknown-none/release/smallaios-x86_64`.

## QEMU GDB Debugging

Launch QEMU paused, waiting for a GDB connection:

```bash
make debug-x86
```

This builds a debug (unoptimized) kernel and starts QEMU with `-s -S`:
- `-s`: GDB server on port 1234
- `-S`: CPU halted at startup (waits for GDB `continue`)

In a second terminal, attach GDB:

```bash
gdb-multiarch -x .gdbinit-x86
```

The `.gdbinit-x86` file connects to QEMU and sets an initial breakpoint:

```gdb
target remote :1234
break kernel_main
continue
```

### Setting Breakpoints

After GDB connects and hits `kernel_main`:

```gdb
break scheduler_init
break onnx_rt::execute
list                    # show source around current position
info registers          # dump register state
next                    # step over
step                    # step into
```

### Source-Level Debugging

The debug build includes full DWARF symbols. GDB resolves file and line information automatically:

```gdb
(gdb) break kernel_main
Breakpoint 1 at 0xffff800000100000: file kernel/src/lib.rs, line 42.
(gdb) continue
Breakpoint 1, kernel_main () at kernel/src/lib.rs:42
42          init_memory();
```

## QEMU Networking

Boot with a virtio-net NIC and host port forwarding:

```bash
make run-x86-net
```

This adds to the base QEMU invocation:
- `-device virtio-net-pci,netdev=net0`
- `-netdev user,id=net0,hostfwd=tcp::8080-:8080`
- Serial output to `build/serial-net.log`

The virtio-net device appears on the PCI bus. Once the kernel network stack initializes, port 8080 on the host forwards to port 8080 inside the VM.

Verify the PCI device is detected in serial output:

```
[pci] 00:03.0 Ethernet controller: Red Hat Virtio network device
```

## VMware Image

Generate a VMDK disk image and VMX configuration:

```bash
make vmware-x86
```

This runs `scripts/make-vmware-x86.sh`, which:
1. Creates a 64 MB sparse raw disk image
2. Partitions with GPT (BIOS boot + ext4)
3. Installs GRUB with a `multiboot2` entry pointing to the kernel
4. Converts raw image to VMDK format
5. Generates a VMX file from the template

Output files:
- `build/smallaios-x86.vmdk` -- disk image
- `build/smallaios-x86.vmx` -- VMware configuration

### Required Tools

- `grub-install` (from `grub-pc-bin` on Ubuntu, `grub2-pc` on Fedora)
- `sgdisk` (from `gdisk`)
- `mkfs.ext4`
- `qemu-img` (from `qemu-utils`)
- `losetup`

The script checks for all required tools and prints the missing package names.

### Opening in VMware

1. Open VMware Workstation or Fusion
2. File > Open > select `build/smallaios-x86.vmx`
3. Power on the VM
4. Serial output is logged to `build/vmware-serial.log`

The VM is configured with:
- 512 MB RAM, 2 vCPUs
- LSI Logic SCSI controller
- Serial port logging to file

## QEMU Monitor

All QEMU targets start a monitor on port 4444. Connect while the VM is running:

```bash
telnet localhost 4444
```

Useful monitor commands:

| Command | Description |
|---------|-------------|
| `info registers` | Dump CPU registers |
| `info mem` | Show virtual memory mappings |
| `info pci` | List PCI devices |
| `info network` | Show network interfaces |
| `info block` | Show block devices |
| `xp /16xg 0xffff800000000000` | Examine 16 giant words at address |
| `gdbserver` | Start GDB server (if not started with `-s`) |
| `quit` | Terminate QEMU |

## Troubleshooting

### Docker GPU Not Detected

**Symptom:** `docker compose --profile gpu up` fails with "could not select device driver" or similar.

**Fix:** Install and configure NVIDIA Container Toolkit:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify with:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### QEMU Not Found

**Symptom:** `make run-x86` fails with "qemu-system-x86_64: command not found".

**Fix:**

```bash
# Ubuntu
sudo apt-get install -y qemu-system-x86

# Fedora
sudo dnf install -y qemu-system-x86
```

### GRUB Install Fails

**Symptom:** `make vmware-x86` fails with "grub-install: command not found" or "cannot find a BIOS directory".

**Fix:** Install the BIOS version of GRUB (not EFI):

```bash
# Ubuntu
sudo apt-get install -y grub-pc-bin

# Fedora
sudo dnf install -y grub2-pc
```

The script requires the i386-pc GRUB target. If you only have `grub-efi` installed, the BIOS boot partition will not work.

### GDB Connection Refused

**Symptom:** `target remote :1234` fails with "Connection refused".

**Causes and fixes:**
1. QEMU is not running: Start it first with `make debug-x86` in another terminal
2. QEMU was not started with `-s`: The `debug-x86` target includes this flag; do not use `run-x86` for debugging
3. Port conflict: Another process is using port 1234. Check with `ss -tlnp | grep 1234`

### VMware Refuses VMDK

**Symptom:** VMware shows "The file specified is not a virtual disk" or "Unsupported VMDK version".

**Causes and fixes:**
1. Corrupted VMDK: Re-run `make vmware-x86` to regenerate
2. Wrong VMDK subformat: The build uses `qemu-img convert -O vmdk` which produces a monolithic sparse VMDK compatible with VMware Workstation 5+ and Fusion
3. Path mismatch: The VMX file references a relative VMDK path. Ensure both `.vmdk` and `.vmx` are in the same `build/` directory
