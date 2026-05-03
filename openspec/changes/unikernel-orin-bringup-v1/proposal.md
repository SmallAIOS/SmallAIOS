# unikernel-orin-bringup-v1

## Summary

PR #124 landed the Jetson Orin **container** path: SmallAIOS runs as a Docker workload on top of L4T, using cuDNN/cuBLAS through NVIDIA's CDI runtime. That is the right deployment for Tegra-Orin-as-an-AI-server today. It is not, however, the SmallAIOS unikernel running on the Tegra Orin SoC — it's the SmallAIOS userspace running inside an Ubuntu-derived L4T host kernel. There is currently no path to exercise the native `aarch64-unknown-none` SmallAIOS kernel on Jetson Orin hardware (J4012 / Orin NX 16 GB) at all.

This change opens that path in two phases. **Phase 1** (Goal A — small, fast) hosts the existing `aarch64-unknown-none` SmallAIOS kernel under `qemu-system-aarch64 -accel kvm -cpu host` running on the J4012's L4T host OS. The Orin Cortex-A78AE cores execute SmallAIOS instructions directly via KVM; the unikernel sees a generic ARM `virt` machine (PL011 UART, GICv3, virtio devices). This proves the cross-compile target works on Orin silicon, gives us a serial-console boot we can iterate on, and ships in days — but it does not exercise any Tegra234 hardware specifically. **Phase 2** (Goal B — larger, real) adds a Tegra234 board support package to `smallaios-arch-aarch64` (board file forking the existing Tegra X1 / Tegra210 BSP), then ships a USB-stick UEFI image that the J4012 can boot from without touching its eMMC. After Phase 2, the unikernel runs directly on the Tegra Orin SoC with its own UART, interrupt controller (GICv3), and memory map; L4T is not in the picture at runtime. GPU access (Ampere GA10B host1x) is **out of scope** for both phases — explicitly deferred.

The two phases are released as one change because they share a goal (testing the native unikernel on the J4012), one design (the same cross-build pipeline feeds both), and one verification surface (serial-console boot output). They could split into two changes if Phase 2 grows beyond estimate; the proposal calls out the split point.

## Why

- **Container path is GPU-validated, but the unikernel itself is unproven on Orin silicon.** The existing AArch64 kernel (`smallaios-arch-aarch64`) builds for `aarch64-unknown-none` and is exercised in CI under QEMU `virt` only. There is no CI job and no documented procedure for running it on a real ARMv8 device, let alone a Tegra234 SoC. That is the blocker on every claim about SmallAIOS-as-a-unikernel for Jetson workloads. Phase 1 closes the cheap half (KVM execution) in days; Phase 2 closes the expensive half (real Tegra234 hardware) in weeks.
- **The Tegra X1 (Tegra210 / Jetson Nano original) BSP already exists.** `arch/aarch64/src/{boot,uart,gicv2,paging,interrupts,platform}.rs`, `arch/aarch64/linker-tegra.ld`, `arch/aarch64/dts/tegra210-smallaios.dts`, plus `arch/nvidia/src/tegra/{clock,power,fifo,gmmu,gr,falcon,regs,mod}.rs` form a working bare-metal Tegra board package gated by the `tegra-x1` Cargo feature, built every PR by the existing `Build Jetson Nano (Tegra X1)` CI job. **Tegra234 is a fork-and-modify of that BSP**, not greenfield: same Cargo workspace, same linker-script pattern, same DTS pattern, similar initialization shape. The deltas are bounded (different MMIO bases, GICv3 instead of GICv2, A78AE-specific quirks, JetPack 6 UEFI entry point instead of the X1 boot ROM hand-off). This radically lowers the risk of a multi-month surprise.
- **USB-stick UEFI boot is the right risk profile for a development kernel.** JetPack 6 ships a UEFI firmware on Orin; entering the firmware menu and booting a removable device is a standard supported path. Failure means "remove the USB stick and reboot to L4T". No flashing, no recovery mode, no risk to the eMMC. Compare to overwriting `/boot/extlinux/extlinux.conf` (one bad boot bricks the box; recovery requires SDK Manager + USB-recovery from an x86 host) or `kexec` from L4T (works but L4T has already initialized the peripherals — we wouldn't be exercising our own bring-up code).
- **Tegra234 BSP is the foundation for any future SmallAIOS-as-Tegra story.** Whether we eventually run on Orin Nano Super, Orin NX, or AGX Orin, they all share the same SoC family (Tegra234) and the same UEFI / DRAM / interrupt / clock topology. Investing in this BSP once unlocks the whole Orin family. The container path remains the production-deployment recommendation for Jetson workloads; the unikernel path is for research, RTOS-grade workloads, formal-verification end-to-end coverage, and DAL A claim coverage that needs to span hardware init.

## Phase 1 — KVM-on-L4T smoke test (Goal A, ~3-5 days)

The existing AArch64 kernel build (`cargo build --target aarch64-unknown-none -p smallaios-kernel` for QEMU `virt`, the default no-Tegra build) produces an ELF that QEMU can boot. Phase 1 validates that the same ELF boots under KVM on the J4012's L4T host — Orin's own A78AE cores executing SmallAIOS instructions, virtio devices providing peripherals, PL011 UART providing the serial console.

We add: (a) a CI job `kvm-smoke-build` that re-uses the existing `Build AArch64 Kernel` artifact and verifies a boot under TCG-emulated `virt` (since GitHub runners can't host KVM); (b) a `just run-jetson-kvm [SSH_HOST]` recipe that scp's the kernel to a configurable Jetson host and runs `qemu-system-aarch64 -M virt,gic-version=3 -cpu host -accel kvm -nographic -kernel smallaios-kernel.bin`; (c) `scripts/test-jetson-kvm.sh` that wraps the recipe with a serial-output assertion; (d) `docs/jetson-kvm-quickstart.md` covering prerequisites, the one-command run, and what successful boot output looks like.

No new Rust code is introduced in Phase 1. The deliverable is a documented procedure plus a CI smoke-build plus a `just` recipe — all the kernel changes Phase 1 needs are already in `develop`.

## Phase 2 — UEFI USB bare-metal boot (Goal B, ~3-5 weeks)

Phase 2 adds a `tegra234` Cargo feature to `smallaios-arch-aarch64` (analogous to the existing `tegra-x1`), gated on a Tegra234 board support package. New files: `arch/aarch64/src/gicv3.rs` (GICv3 driver — Orin uses GICv3 LPIs and ITS), `arch/aarch64/src/tegra234_uart.rs` (Tegra234 NS16550-compatible UART at the Tegra234 MMIO base — different from X1), `arch/aarch64/linker-tegra234.ld` (Orin DRAM at `0x80000000`+, image base + reset entry), `arch/aarch64/dts/tegra234-smallaios.dts` (extracted from upstream L4T device tree, trimmed to what the unikernel uses). Modifications: `boot.rs` learns a `tegra234` entry path (same shape as `tegra-x1` but for the Orin DRAM map), `platform.rs` learns the Tegra234 platform descriptor, `interrupts.rs` learns the GICv3 dispatch shape.

The image is built as a UEFI application — `cargo build --target aarch64-unknown-uefi -p smallaios-kernel --features tegra234` produces `smallaios.efi`. We package it into a FAT32 disk image (`build-jetson-usb.img`) with the standard UEFI ESP layout (`/EFI/BOOT/BOOTAA64.EFI`), so any user can `dd` it to a USB stick. CI gains a `Build Jetson Orin UEFI` job (advisory initially, gate later with self-hosted runner) running `cargo build --target aarch64-unknown-uefi -p smallaios-kernel --features tegra234`.

A new `docs/jetson-orin-uefi-boot.md` covers: prerequisites, where to download `build-jetson-usb.img` (CI artifact), how to dd it to a USB stick, the J4012 firmware-menu key sequence to boot from the USB device, the expected serial-console output ("Hello, world from SmallAIOS on Tegra234"), and the recovery procedure (remove stick, reboot to L4T) — emphasizing that nothing is written to the J4012's eMMC.

## Out of scope

- **GPU access (Ampere GA10B / host1x v6).** The existing `arch/nvidia/src/tegra/` HAL targets the Maxwell GM20B GPU on Tegra X1. An Orin GPU port is its own multi-month effort — host1x v6 layout differs from v3 on X1, channel/syncpoint semantics shifted, and Ampere GR class registers diverge from Maxwell. Tracked for a future change (`unikernel-orin-gpu-v1`). Phase 1+2 deliver "the unikernel boots and prints over UART"; that is a meaningful milestone in its own right.
- **eMMC replacement / overwriting L4T.** Out of scope by deliberate choice — USB-stick boot keeps the J4012 a usable L4T machine for the container path while the unikernel is iterated. eMMC replacement would require NVIDIA's `flash.sh` from a host x86 Linux machine and is reversible only via SDK Manager USB-recovery. Defer indefinitely.
- **NVMe boot.** Possible follow-up once UEFI USB boot is solid (UEFI can boot from NVMe with the same `smallaios.efi` payload), but USB is the no-risk first cut. Tracked for a future change.
- **Verified Boot integration.** The existing `verified-boot` Cargo feature on `smallaios-security` validates module signatures inside the kernel; integrating UEFI Secure Boot signing of `smallaios.efi` is a separate concern. Phase 2 documents the requirement to disable UEFI Secure Boot (or enroll our key) in the firmware menu before booting our unsigned EFI image. Production signing is a follow-up.
- **PXE / network boot.** Useful for development iteration speed (no USB removal), but USB stick is the documented path for the first cut. Could be a thin add-on once Phase 2 lands.
- **Reusing the `tegra-orin` feature name.** The existing `tegra-orin` feature on `smallaios-arch-nvidia` selects `cc_87` for the **userspace CUDA** path (PR #124). The new bare-metal feature lives on `smallaios-arch-aarch64` and is named `tegra234` (the SoC family) to avoid overloading the userspace name. Two features, two crates, two distinct meanings — documented in their respective Cargo manifests.

## Sequencing

Phase 1 lands first. It is independent of Phase 2 and provides immediate iteration value (a 1-minute build/scp/run loop for testing kernel changes against real Orin cores). Phase 2 starts after Phase 1 is green and follows the standard intra-phase order: (a) Tegra234 board file scaffolding + UEFI build target + minimal `boot.rs` that reaches `_start`, (b) Tegra234 UART hello-world over the J4012's serial header, (c) GICv3 + timer + minimal interrupt dispatch, (d) USB-image packaging + boot procedure docs + CI advisory job. Each sub-step has a serial-console-observable exit criterion.

If Phase 2 grows beyond ~5 weeks of effort, we split it out as `unikernel-orin-bare-metal-v1` and archive this change with only Phase 1 implemented. The proposal makes the split point easy: Phase 1 deliverables are independent of any Phase 2 Rust code.

## Effort estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | KVM-on-L4T smoke test + `just` recipe + CI build + docs | ~3-5 days |
| 2 | Tegra234 BSP fork + UEFI build target + USB image + bare-metal boot to UART | ~3-5 weeks |
| **Total** | | **~4-6 weeks** |

GPU and any eMMC-write paths add multi-month follow-ups outside this change.
