# Design — unikernel-orin-bringup-v1

## Goal

Two ordered milestones, both verified by serial-console output on a Seeed reComputer J4012 (Jetson Orin NX 16 GB):

1. **Phase 1 — KVM smoke test.** Boot the existing `aarch64-unknown-none` SmallAIOS kernel under `qemu-system-aarch64 -accel kvm -cpu host` running on the J4012's L4T host OS. Success = "SmallAIOS boot banner over the QEMU PL011 UART, observed over `qemu -nographic`, with a recognized GICv3 init line and the cooperative scheduler reaching its idle loop."
2. **Phase 2 — UEFI USB bare-metal boot.** Boot the same kernel (rebuilt for `aarch64-unknown-uefi` with a `tegra234` feature) directly on the J4012 from a USB stick, no L4T involved at runtime. Success = "SmallAIOS boot banner over the J4012's physical UART header, GICv3 init, scheduler idle, observable to a serial cable on a separate machine."

GPU, eMMC writes, NVMe boot, network boot, and Secure-Boot signing are explicitly out of scope.

## Phase 1 — KVM-on-L4T

### Why KVM first, not kexec or UEFI

KVM has the smallest delta to what already works:

- The existing `Build AArch64 Kernel` CI job already produces a kernel that boots under QEMU `virt`. The only delta for KVM is `-accel kvm -cpu host`, which only changes the execution mode (TCG → KVM); the guest sees the same `virt` machine.
- KVM on Orin requires JetPack 6's `kvm` kernel module, which ships enabled on JetPack 6 / L4T R36.4 by default. Verified by `lsmod | grep kvm` + `cat /sys/devices/system/cpu/cpu0/...` for HCR_EL2 access. (The J4012 verification commands the user is collecting include this check.)
- L4T is still in the picture, so a panic during early boot just terminates the QEMU process — no risk to the host. Iteration loop is fast: `cargo build` → `scp` → `qemu-system-aarch64 ...` → console output, sub-30-second cycle.
- KVM gives us "real Cortex-A78AE instructions" plus "real ARMv8 generic timer" plus "real GICv3 (KVM virtualizes a GICv3 by default on aarch64 hosts)". So we are exercising real hardware behavior for the CPU/timer/interrupt paths, even though peripherals are paravirtualized.

`kexec` is rejected for Phase 1 because L4T has already initialized DRAM training, clocks, PMC, and most peripheral controllers; jumping to SmallAIOS via kexec means SmallAIOS inherits a partially-initialized hardware state that doesn't represent a real boot path. UEFI USB boot is rejected for Phase 1 because we don't have a Tegra234 BSP yet — we'd boot off the USB stick and immediately panic on the first MMIO access to a wrong UART base.

### Phase 1 deliverables (no new Rust code)

1. `just run-jetson-kvm SSH_HOST` recipe — cross-builds `smallaios-kernel.bin` for `aarch64-unknown-none`, scp's it to `$SSH_HOST:~`, ssh's into `$SSH_HOST` and runs `qemu-system-aarch64 -M virt,gic-version=3 -cpu host -accel kvm -m 1G -nographic -kernel ~/smallaios-kernel.bin`. The `-nographic` flag pipes the PL011 UART to the user's terminal.
2. `scripts/test-jetson-kvm.sh [SSH_HOST]` — non-interactive smoke test wrapping the recipe, with a serial-output assertion (greps the boot output for an expected banner). Exits 0 on success, non-zero with a captured-output dump on failure.
3. `kvm-smoke-build` CI job — runs `cargo build --target aarch64-unknown-none -p smallaios-kernel --release` and uploads the resulting binary as a workflow artifact, then runs the same kernel under TCG-emulated `virt` (since GitHub runners have no nested KVM). Asserts the same boot banner. Catches regressions in the QEMU-virt boot path that the J4012 can't catch automatically until we have a self-hosted runner.
4. `docs/jetson-kvm-quickstart.md` — prerequisites (`apt install qemu-system-arm`, KVM module check, ssh access), the `just run-jetson-kvm` invocation, expected boot output snippet, troubleshooting (missing KVM module, GICv3 mismatch, virtio panic).

### Why we keep Phase 1 separable

If Phase 2 estimates blow out, Phase 1 still ships a real iteration loop for kernel hackers using the J4012 as a test target. The Phase 1 CI gate (`kvm-smoke-build`) enforces the existing AArch64 kernel boots under QEMU on every PR — useful regardless of what happens in Phase 2.

## Phase 2 — UEFI USB bare-metal

### Why UEFI USB and not other paths

| Boot path | Risk | Iteration cost | Picks up our bring-up code? |
|-----------|------|----------------|------------------------------|
| **UEFI USB stick** | None — fail = remove stick + reboot | Medium (USB write loop) | Yes — we own from `_start` onward |
| `kexec` from L4T | Low — bad jump = reboot | Fast (no flash) | Partial — L4T already initialized clocks/UART/etc. |
| `extlinux.conf` swap | High — bad kernel = brick (recovery via SDK Manager + USB-recovery from x86 host) | Fast | Yes |
| `flash.sh` to NVMe | High — wrong slot = brick | Slow (full flash) | Yes |
| `flash.sh` to eMMC | Highest — wrong layout = brick | Slowest | Yes |

UEFI USB is the only path that combines "we own the hardware from `_start`" with "no risk to the eMMC". JetPack 6 ships UEFI on Orin platforms (Jetson Linux R36.4 release notes confirm `Tegra UEFI` as the top of the boot chain), and the firmware menu lets us select a removable boot device at power-on. This is the standard Linux-on-arm-server boot flow, just on Tegra silicon.

### Build target choice

**`aarch64-unknown-uefi`**, not `aarch64-unknown-none`. UEFI provides a defined entry signature (`efi_main(handle, system_table) -> Status`), gives us the firmware's UART for early debug before our own UART driver is up, and lets us use `ExitBootServices` to take ownership of the hardware exactly when we are ready. The existing `aarch64-unknown-none` kernel can be linked against `uefi-rs` style entry stubs without a wholesale rewrite — the Tegra X1 path uses a similar shape with the X1 boot ROM hand-off.

We do **not** support both targets simultaneously in Phase 2. The `tegra234` feature implies UEFI entry; `tegra-x1` continues to use `aarch64-unknown-none` with the X1 boot-ROM hand-off in `boot.rs`. Two boot entry points, two linker scripts, gated by their feature.

### Tegra234 BSP — fork-from-X1 deltas

Existing Tegra X1 (Tegra210 / cc 5.3) BSP layout:

```
arch/aarch64/
  src/
    boot.rs              ← X1 boot-ROM entry, sets up MMU + stack + jumps to kernel main
    gicv2.rs             ← GICv2 (Tegra X1)
    uart.rs              ← Tegra UART base = 0x70006000 (X1)
    paging.rs, interrupts.rs, image_header.rs, platform.rs
    tegra_dc.rs, tegra_sor.rs, tegra_edid.rs, tegra_pcie.rs   ← display/PCIe (X1-specific, not needed for Orin Phase 2)
  linker-tegra.ld        ← Tegra210 memory map
  dts/tegra210-smallaios.dts
```

Tegra234 BSP additions for Phase 2:

```
arch/aarch64/
  src/
    boot_uefi.rs         ← UEFI efi_main entry — gets RAM map + DTB pointer from system table, exits boot services, jumps into kernel main
    gicv3.rs             ← GICv3 distributor + redistributor + ITS (Orin uses GICv3 LPIs but Phase 2 only needs SGIs/PPIs)
    tegra234_uart.rs     ← Tegra Combined UART (TCU) at 0x0c280000 (Tegra234 base) — different MMIO from X1
    tegra234_platform.rs ← memory-map descriptor, SoC ID, A78AE specific tweaks
  linker-tegra234.ld     ← Tegra234 DRAM at 0x80000000+, image base for UEFI loadable PE/COFF
  dts/tegra234-smallaios.dts  ← extracted from upstream L4T tegra234.dtsi, trimmed to UART + GIC + timer + memory only
```

Files that **stay X1-specific** and are not built when `tegra234` is on:
- `tegra_dc.rs`, `tegra_sor.rs`, `tegra_edid.rs`, `tegra_pcie.rs` — X1 display/PCIe drivers; not needed for Phase 2 (we use UART, not framebuffer, and no PCIe peripherals are touched).
- `gicv2.rs` — X1 only; the `tegra234` build links `gicv3.rs` instead.
- `arch/nvidia/src/tegra/` — Maxwell GM20B GPU HAL; out of scope.

Files that **are shared** between X1 and Orin:
- `paging.rs`, `interrupts.rs` (the dispatch shape, not the controller driver), `image_header.rs`, the kernel main code paths.

### Cargo feature naming

| Crate | Feature | Selects |
|-------|---------|---------|
| `smallaios-arch-aarch64` | `tegra-x1` (existing) | Tegra X1 / cc 5.3 bare-metal HAL via `aarch64-unknown-none` |
| `smallaios-arch-aarch64` | **`tegra234`** (new in Phase 2) | Tegra234 / Orin-family bare-metal HAL via `aarch64-unknown-uefi` |
| `smallaios-arch-nvidia` | `tegra` (existing) | Same as `tegra-x1`, kept for back-compat — see PR #124 |
| `smallaios-arch-nvidia` | `tegra-orin` (existing, from PR #124) | **Userspace CUDA** for cc 8.7 (Orin) — container path, not bare-metal |

The two `tegra-orin`-shaped concepts (userspace CUDA target vs. bare-metal HAL) are deliberately given different names on different crates. `tegra-orin` on `arch/nvidia` means "userspace CUDA for Orin"; `tegra234` on `arch/aarch64` means "bare-metal HAL for the Tegra234 SoC family". A future `unikernel-orin-gpu-v1` change might add a third feature (e.g. `tegra234-gpu` on `arch/nvidia`) for the bare-metal Ampere GA10B HAL.

### Risk: UEFI Secure Boot

JetPack 6 may ship with Secure Boot enabled by default. Our `smallaios.efi` is unsigned in Phase 2; the firmware will refuse to load it. Two mitigations, documented in `docs/jetson-orin-uefi-boot.md`:

1. **Disable Secure Boot in the firmware menu** for development. Restorable.
2. **Enroll our development key** as a UEFI MOK (Machine Owner Key). Slightly more work, doesn't require disabling Secure Boot globally.

Production UEFI signing is a follow-up change.

### Risk: brick recovery

Even though USB-stick boot can't write to eMMC, the user must not be tempted to "fix it permanently" by rewriting the EFI variables to make USB the default boot device. We document the firmware-menu workflow only and warn against `efibootmgr`-style permanent EFI-var changes during Phase 2.

If something does brick the J4012 (e.g. user error during a future `flash.sh` follow-up), recovery is via NVIDIA SDK Manager + USB-recovery from an x86 Linux host. We add a one-paragraph "if you bricked the box" section to `docs/jetson-orin-uefi-boot.md` linking to NVIDIA's recovery procedure. In the Phase 2 critical path itself, no brick is possible.

## Build/CI surface

### Phase 1
- New `just` recipe: `run-jetson-kvm` (parameterized by SSH host).
- New script: `scripts/test-jetson-kvm.sh`.
- New CI job: `kvm-smoke-build` — builds the kernel, runs it under TCG `virt`, asserts banner. Gates the change.
- New docs: `docs/jetson-kvm-quickstart.md`.

### Phase 2
- New cargo feature `tegra234` on `smallaios-arch-aarch64`.
- New build target: `aarch64-unknown-uefi` added to `rust-toolchain.toml` `targets`.
- New CI job: `Build Jetson Orin UEFI` — builds `cargo build --target aarch64-unknown-uefi -p smallaios-kernel --features tegra234`, packages `smallaios.efi`, runs through a CI-side QEMU OVMF-aarch64 boot test (same shape as the OVMF tests Linux distros use to validate kernel images). Advisory initially.
- New artifact: `build-jetson-usb.img` — FAT32 ESP image with `EFI/BOOT/BOOTAA64.EFI` for `dd`-to-USB.
- New docs: `docs/jetson-orin-uefi-boot.md`.

### Promote to gate later
Both new CI jobs start advisory. Promote to gate (`change-gates` blocking) once a self-hosted Jetson runner is wired (separate change).

## What this change explicitly does NOT do

- It does not modify the existing Tegra X1 / Jetson Nano boot path. `tegra-x1` continues to mean what it means today.
- It does not modify the merged Jetson container path. `Dockerfile.jetson*` and the GHCR strategy (still pending) are unaffected.
- It does not add any GPU code. The unikernel boots and prints; everything beyond that is explicitly outside the change.
- It does not change the workspace dependency layering. Both phases live in `arch/aarch64` (Layer 2) and consume only Layer 0/1 services.
