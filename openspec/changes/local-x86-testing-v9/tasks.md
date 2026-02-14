## 1. Dockerfile and Container Image

- [x] 1.1 Create `Dockerfile` at repository root: builder stage using `rust:nightly-slim`, install `musl-tools` and `x86_64-unknown-linux-musl` target, copy workspace, build with `cargo build --release --target x86_64-unknown-linux-musl`
- [x] 1.2 Add `ENABLE_GPU` build arg: when `ENABLE_GPU=1`, append `--features nvidia_gpu` to the cargo build command
- [x] 1.3 Add runtime stage: `FROM scratch`, `COPY --from=builder` the `smallaios-container` binary to `/smallaios`, set `ENTRYPOINT ["/smallaios"]`
- [x] 1.4 Add `.dockerignore` at repository root: exclude `target/`, `.git/`, `build/`, `*.vmdk`, `*.vmx`, `*.img`
- [x] 1.5 Build the Dockerfile locally (CPU-only) and verify the image size is under 15 MB (requires Docker runtime)
- [x] 1.6 Build the Dockerfile with `ENABLE_GPU=1` and verify it compiles with the `nvidia_gpu` feature (requires Docker runtime)

## 2. docker-compose and NVIDIA Runtime

- [x] 2.1 Create `docker-compose.yml` at repository root with `smallaios` service: build from `Dockerfile`, map port `8080:8080`, mount `./models:/models:ro`, set restart policy to `unless-stopped`
- [x] 2.2 Add `smallaios-gpu` service with `profiles: [gpu]`: build with `ENABLE_GPU=1` arg, set `runtime: nvidia`, set `NVIDIA_VISIBLE_DEVICES=all` environment variable, same port and volume mounts
- [x] 2.3 Add health check to both services: `test: ["CMD", "/smallaios", "--health-check"]` with `interval: 10s`, `timeout: 5s`, `retries: 3`, `start_period: 5s`
- [x] 2.4 Test `docker compose up --build` (CPU profile) starts successfully (requires Docker runtime)
- [x] 2.5 Test `docker compose --profile gpu up --build` starts with NVIDIA runtime (requires Docker + NVIDIA Container Toolkit)

## 3. Makefile Targets for Docker

- [x] 3.1 Add `docker-local` target: runs `docker compose up --build`
- [x] 3.2 Add `docker-local-gpu` target: runs `docker compose --profile gpu up --build`
- [x] 3.3 Add `docker-local-clean` target: runs `docker compose down --rmi local --volumes`
- [x] 3.4 Verify `make docker-local` builds and runs the container (requires Docker runtime)
- [x] 3.5 Verify `make docker-local-gpu` builds and runs with GPU (requires Docker + NVIDIA Container Toolkit)

## 4. VMware VMDK Image Builder

- [x] 4.1 Create `scripts/make-vmware-x86.sh`: check for required tools (`grub-install`, `sgdisk`, `mkfs.ext4`, `qemu-img`, `losetup`); print missing tool names and package names for Ubuntu/Fedora on failure
- [x] 4.2 Implement disk image creation: `dd` a 64 MB sparse file, create GPT table with `sgdisk`, add 1 MB BIOS boot partition (`ef02`) and ext4 data partition
- [x] 4.3 Implement GRUB installation: `losetup` the image, `mkfs.ext4` the data partition, mount it, `mkdir -p /boot/grub`, copy kernel to `/boot/smallaios-x86_64`, write `grub.cfg` with `multiboot2 /boot/smallaios-x86_64` entry (timeout 3s), run `grub-install --target=i386-pc --boot-directory=<mountpoint>/boot <loop_device>`, unmount and detach loop
- [x] 4.4 Implement VMDK conversion: `qemu-img convert -O vmdk` from raw to `build/smallaios-x86.vmdk`
- [x] 4.5 Create `scripts/vmware-template.vmx` with: `guestOS = "other-64"`, 512 MB RAM, 2 vCPUs, LSI Logic SCSI, VMDK path, serial port to `build/vmware-serial.log`
- [x] 4.6 In the script, copy and patch the VMX template to `build/smallaios-x86.vmx` with the correct VMDK path
- [x] 4.7 Add `vmware-x86` Makefile target: depend on `build-kernel-x86`, invoke `scripts/make-vmware-x86.sh`, print output file paths
- [x] 4.8 Test `make vmware-x86`: verify VMDK is created and is under 64 MB (requires root/losetup)
- [x] 4.9 Test opening `build/smallaios-x86.vmx` in VMware Workstation/Fusion (manual VMware verification)

## 5. QEMU GDB Debugging

- [x] 5.1 Add `debug-x86` Makefile target: depend on `build-kernel-x86-debug`, launch QEMU with `-s -S -machine q35 -cpu max -m 512M` using the debug kernel, serial to both stdio and `build/serial-debug.log`, print GDB connection command
- [x] 5.2 Create `.gdbinit-x86` at repository root: `target remote :1234`, `break kernel_main`, `continue`
- [x] 5.3 Test `make debug-x86`: verify QEMU starts paused, GDB can connect on port 1234, and breakpoint on `kernel_main` is hit (requires QEMU + GDB)
- [x] 5.4 Verify GDB resolves source-level symbols (function names, file/line info) from the debug build (requires QEMU + GDB)

## 6. QEMU Networking

- [x] 6.1 Add `run-x86-net` Makefile target: depend on `build-kernel-x86`, launch QEMU with `-device virtio-net-pci,netdev=net0 -netdev user,id=net0,hostfwd=tcp::8080-:8080`, serial to both stdio and `build/serial-net.log`
- [x] 6.2 Test `make run-x86-net`: verify QEMU boots with the virtio-net device visible on the PCI bus (requires QEMU)
- [x] 6.3 Verify host port 8080 forwarding is established (testable once the kernel has a network stack responding) (requires QEMU + network stack)

## 7. QEMU Serial Logging and Monitor

- [x] 7.1 Update `run-x86` Makefile target: add `-serial file:build/serial.log` as a second serial output (keep `-serial stdio` as primary); ensure `build/` directory exists via `@mkdir -p build`
- [x] 7.2 Add `-monitor telnet:localhost:4444,server,nowait` to all QEMU targets (`run-x86`, `debug-x86`, `run-x86-net`); print monitor connection instruction
- [x] 7.3 Test `make run-x86`: verify `build/serial.log` is created with kernel output (requires QEMU)
- [x] 7.4 Test `telnet localhost 4444` connects to QEMU monitor while kernel is running (requires QEMU)

## 8. CI Integration

- [x] 8.1 Add `docker-build-local` job to `.github/workflows/ci.yml`: build the Dockerfile (CPU-only), verify image size under 15 MB, no GPU runner required
- [x] 8.2 Add `docker-build-local` to the `change-gates` job's `needs` array
- [x] 8.3 Verify CI: Dockerfile builds successfully, existing jobs unaffected (requires GitHub Actions runner)

## 9. Documentation

- [x] 9.1 Create `docs/local-testing.md` with sections: Prerequisites, Docker CPU-Only, Docker with NVIDIA GPU, QEMU Bare-Metal Boot, QEMU GDB Debugging, QEMU Networking, VMware Image, Troubleshooting
- [x] 9.2 Prerequisites section: list required packages for Ubuntu (`qemu-system-x86`, `grub-pc-bin`, `gdisk`, `qemu-utils`, `docker.io`, `nvidia-container-toolkit`) and Fedora equivalents
- [x] 9.3 Each section: include exact `make` commands, expected output, and common failure modes
- [x] 9.4 Troubleshooting section: Docker GPU not detected (missing nvidia-container-toolkit), QEMU not found, GRUB install fails (wrong package), GDB connection refused (QEMU not running), VMware refuses VMDK (wrong format version)

## 10. Integration Testing

- [x] 10.1 Run `make docker-local` end-to-end: Dockerfile builds, container starts, health check passes (or prints expected output) (requires Docker runtime)
- [x] 10.2 Run `make debug-x86` + GDB attach: kernel pauses, breakpoint hits, `continue` resumes (requires QEMU + GDB)
- [x] 10.3 Run `make run-x86-net`: kernel boots with virtio-net PCI device (requires QEMU)
- [x] 10.4 Run `make vmware-x86`: VMDK created, size check passes (requires root/losetup)
- [x] 10.5 Verify `make run-x86` still works unchanged (no regressions) (requires QEMU)
- [x] 10.6 Run `make clippy` and `make fmt-check`: zero warnings, no format issues
