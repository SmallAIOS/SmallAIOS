# Tasks — unikernel-orin-bringup-v1

## 0. Verification on the J4012 (prereq for both phases)

- [ ] 0.1 Capture and record the output of `cat /proc/cmdline | tr ' ' '\n' | grep root=`, `lsblk`, `cat /etc/nv_tegra_release`, `ls /sys/firmware/efi/efivars/ 2>/dev/null && echo UEFI-PRESENT || echo NO-UEFI`, `lsmod | grep kvm`, `cat /proc/device-tree/compatible` — paste in the change PR description so the design assumptions are pinned to ground truth on this specific J4012 SKU
- [ ] 0.2 Confirm UEFI is present (Phase 2 prerequisite) — if not, escalate before starting Phase 2
- [ ] 0.3 Confirm `kvm` module is loaded (Phase 1 prerequisite) — if not, document the `modprobe kvm` workaround in the Phase 1 quickstart

## 1. Phase 1 — KVM-on-L4T smoke test

### 1a. `just` recipe + smoke script

- [ ] 1.1 Add `run-jetson-kvm SSH_HOST [KERNEL_PATH=target/aarch64-unknown-none/release/smallaios-kernel.bin]` recipe to `Justfile`. Recipe: cross-build via `cargo build --target aarch64-unknown-none -p smallaios-kernel --release`, scp to `$SSH_HOST:~`, ssh into `$SSH_HOST` and `qemu-system-aarch64 -M virt,gic-version=3 -cpu host -accel kvm -m 1G -nographic -kernel ~/smallaios-kernel.bin`
- [ ] 1.2 Add `scripts/test-jetson-kvm.sh [SSH_HOST]` — non-interactive wrapper around the recipe, with a serial-output assertion (greps the captured stdout for an expected boot banner). Exit 0 on hit, non-zero with a captured-output dump on miss
- [ ] 1.3 Document expected vs failure exit codes inside the script header (mirroring the convention in `scripts/test-jetson-gpu.sh`)
- [ ] 1.4 Verify the recipe end-to-end on the actual J4012 — capture the full boot output and paste in the PR description as the Phase 1 acceptance evidence

### 1b. CI smoke build

- [ ] 1.5 Add a `kvm-smoke-build` job to `.github/workflows/ci.yml`: `cargo build --target aarch64-unknown-none -p smallaios-kernel --release`, then run the artifact under TCG-emulated `qemu-system-aarch64 -M virt,gic-version=3 -nographic -kernel <bin>` with a 30-second timeout, asserting the same banner the J4012 test asserts
- [ ] 1.6 Wire `kvm-smoke-build` into the `change-gates` meta-job so it blocks merge
- [ ] 1.7 Verify the gate catches a deliberately-broken `boot.rs` (revert the deliberate break before pushing)

### 1c. Docs

- [ ] 1.8 Create `docs/jetson-kvm-quickstart.md` covering: prerequisites (`apt install qemu-system-arm` on the J4012, KVM module check), one-command run, expected boot output snippet, troubleshooting (missing KVM, GICv3 mismatch, virtio panic, "Hello, world but then page fault" — common early-boot symptoms)
- [ ] 1.9 Add a Jetson-KVM row to `README.md`'s deployment matrix linking to the quickstart, distinct from the existing Jetson container row
- [ ] 1.10 Update `CLAUDE.md` "Current state" to note Phase 1 lands a KVM-hosted unikernel test path on Orin

### 1d. Phase 1 close-out

- [ ] 1.11 Tag Phase 1 sub-PR title `feat(arch/aarch64): unikernel-orin-bringup-v1 phase 1 — KVM smoke on Orin`, target `develop`
- [ ] 1.12 PR green + reviewer sign-off + squash-merge

## 2. Phase 2 — Tegra234 BSP scaffolding

### 2a. Cargo + toolchain

- [ ] 2.1 Add `aarch64-unknown-uefi` to `rust-toolchain.toml` `targets` so `rustup target add` resolves cleanly for local devs
- [ ] 2.2 Add `tegra234` feature to `arch/aarch64/Cargo.toml` (`tegra234 = []` initially, gates conditional code), with a doc-comment distinguishing it from `tegra-x1` (X1 / cc 5.3 bare-metal) and from `arch/nvidia`'s `tegra-orin` (Orin userspace CUDA)
- [ ] 2.3 Update `arch/aarch64/Cargo.toml` to gate `tegra-x1`-only files (display, PCIe, GICv2) so `tegra234` builds don't pull them in

### 2b. Linker script + DTS

- [ ] 2.4 Create `arch/aarch64/linker-tegra234.ld` modeled on `linker-tegra.ld`, with Orin DRAM at `0x80000000` and PE/COFF-compatible image base for UEFI loadable
- [ ] 2.5 Create `arch/aarch64/dts/tegra234-smallaios.dts` extracted from the upstream Linux `arch/arm64/boot/dts/nvidia/tegra234.dtsi` (or L4T's), trimmed to just the nodes the unikernel uses: `cpus`, `psci`, `timer`, GIC, the Tegra Combined UART (TCU) at `0x0c280000`, and the memory node. Strip everything else
- [ ] 2.6 Document the DTS extraction provenance (which upstream commit, license boilerplate) in a header comment in the DTS file

### 2c. UEFI entry + boot

- [ ] 2.7 Create `arch/aarch64/src/boot_uefi.rs` — `efi_main(handle, system_table) -> Status` entry. Uses `uefi-rs` (or hand-rolled minimal UEFI bindings) to: walk `system_table.boot_services()` for the memory map, locate the DTB via `EFI_DTB_TABLE_GUID`, call `ExitBootServices`, jump to kernel main with the DTB pointer and memory map
- [ ] 2.8 Modify `arch/aarch64/src/boot.rs` to dispatch on feature: `tegra-x1` keeps the existing X1 boot-ROM hand-off; `tegra234` enters via `boot_uefi.rs`
- [ ] 2.9 Build smoke: `cargo build --target aarch64-unknown-uefi -p smallaios-kernel --features tegra234` produces a `smallaios.efi` PE/COFF artifact

### 2d. Tegra234 UART (first observable signal)

- [ ] 2.10 Create `arch/aarch64/src/tegra234_uart.rs` — Tegra Combined UART (TCU) MMIO driver at `0x0c280000`. NS16550-compatible register layout. Polling write_byte initially; interrupt-driven later
- [ ] 2.11 Wire `tegra234_uart` into the `console` module under the `tegra234` feature, replacing the X1 `uart`
- [ ] 2.12 First milestone observable on the J4012: kernel UEFI-boots from USB, prints "Hello, world from SmallAIOS on Tegra234" over the TCU. Capture the serial output via a USB-to-TTL cable on the J4012's UART header. Paste in the PR description

### 2e. GICv3 + timer + minimal interrupt dispatch

- [ ] 2.13 Create `arch/aarch64/src/gicv3.rs` — GICv3 distributor + redistributor enable, SGI/PPI handling. LPI/ITS not needed for Phase 2 (no MSI-capable peripherals exercised)
- [ ] 2.14 Modify `arch/aarch64/src/interrupts.rs` to feature-gate dispatch: `gicv2` for `tegra-x1`, `gicv3` for `tegra234`
- [ ] 2.15 Wire the ARM Generic Timer to the cooperative scheduler tick under `tegra234`
- [ ] 2.16 Second milestone observable on the J4012: scheduler reaches its idle loop, prints periodic ticks (or yields to a pinned heartbeat task that does), confirming the timer + GICv3 are live

### 2f. USB image packaging

- [ ] 2.17 Add a `build-jetson-usb-image` `just` recipe that takes the built `smallaios.efi`, `mkfs.fat -F32` an image of size N (parameter, default 256 MB), copies `smallaios.efi` to `EFI/BOOT/BOOTAA64.EFI`, emits `build-jetson-usb.img`
- [ ] 2.18 Document the `dd if=build-jetson-usb.img of=/dev/sdX bs=4M status=progress` workflow in `docs/jetson-orin-uefi-boot.md`, with a prominent warning to confirm the target device

### 2g. Boot procedure docs (the primary user surface)

- [ ] 2.19 Create `docs/jetson-orin-uefi-boot.md` covering: prerequisites (USB stick ≥256 MB, USB-to-TTL serial cable connected to the J4012 UART header, terminal emulator configured at the documented baud rate), the `dd` step with target-device confirmation guidance, the J4012 firmware-menu key sequence (likely Esc at boot — confirm against Seeed J4012 docs), the UEFI Secure Boot disable-or-enroll step, the expected serial-console output, and the recovery procedure ("remove stick, power-cycle, reboot to L4T")
- [ ] 2.20 Add a one-paragraph "if you accidentally bricked the box" section linking to NVIDIA's SDK Manager + USB-recovery procedure — even though Phase 2 itself can't brick the box, an `efibootmgr` follow-up could

### 2h. CI advisory build + OVMF QEMU smoke

- [ ] 2.21 Add a `Build Jetson Orin UEFI` job to `.github/workflows/ci.yml` (advisory, `continue-on-error: true`) running `cargo build --target aarch64-unknown-uefi -p smallaios-kernel --features tegra234`
- [ ] 2.22 Extend the same job (or add a sibling `jetson-orin-ovmf-smoke` job) that runs `qemu-system-aarch64 -M virt -cpu cortex-a78 -bios edk2-aarch64-code.fd -drive file=fat:rw:build/efi-root` (OVMF UEFI on QEMU) with `smallaios.efi` as `BOOTAA64.EFI`, asserting the boot banner appears. Catches PE/COFF + UEFI entry regressions even without a self-hosted Jetson runner
- [ ] 2.23 Add a comment block in the workflow file noting "promote to gate when self-hosted Jetson runner is available"

### 2i. Phase 2 close-out

- [ ] 2.24 Tag Phase 2 sub-PR title `feat(arch/aarch64): unikernel-orin-bringup-v1 phase 2 — Tegra234 UEFI USB boot`, target `develop`
- [ ] 2.25 PR green + on-J4012 boot evidence pasted in PR description + reviewer sign-off + squash-merge

## 3. Verify + archive

- [ ] 3.1 Run `openspec validate unikernel-orin-bringup-v1 --strict` after both phases land
- [ ] 3.2 Open archive PR moving the change to `openspec/changes/archive/YYYY-MM-DD-unikernel-orin-bringup-v1` and syncing the spec deltas to main specs

## Phase split escape hatch (if Phase 2 grows)

- [ ] If Phase 2 effort exceeds ~5 weeks of focused work, split out tasks 2.x into a new change `unikernel-orin-bare-metal-v1` and archive this change with only Phase 1 (tasks 0.x, 1.x, 3.x) implemented. Update `proposal.md` "Out of scope" to reference the new change name.
