# Jetson Orin UEFI USB-boot quickstart (Phase 2)

> **Status:** Phase 2 of `unikernel-orin-bringup-v1`. The current Tegra234
> path runs end-to-end on real Orin hardware in **interim mode** —
> `efi_main` skips `ExitBootServices` and the kernel uses UEFI's `con_out`
> for serial output. The full memory map (16 GB on an Orin NX 16 GB) is
> harvested from UEFI and the kernel reaches its idle loop. The remaining
> work to make this a "real" kernel boot — post-EBS UART driver — is
> tracked as a follow-up sub-PR. See `arch/aarch64/src/tegra234_uart.rs`
> for the bring-up notes.

This document covers booting the SmallAIOS unikernel directly from a USB
stick on a Jetson Orin family devkit. The Orin's UEFI firmware loads the
PE32+ `.efi` artifact, hands control to `efi_main`, which then either
hands off to `kernel_main` (interim mode, current default) or stays in
UEFI services until you explicitly exit them (future).

**Distinct from `docs/jetson-kvm-quickstart.md`** — that doc covers the
KVM-on-L4T smoke path (Phase 1), which keeps L4T running and exercises
SmallAIOS as a QEMU guest. Phase 2 boots SmallAIOS directly on the
hardware, no L4T involvement.

---

## Prerequisites

### Hardware

- Jetson Orin family devkit. Tested on Orin NX 16 GB (P3767-0000 + P3768-0000 carrier, JetPack 6.2.1 / L4T R36.4.7).
- USB stick, ≥ 256 MB. The image we produce is fixed at 256 MB by default; the actual `.efi` is ~1.9 MB.
- USB-to-TTL serial cable connected to the J-class carrier's UART header. The Phase 2 kernel's only output channel is the TTL serial — there's no on-screen output until later sub-PRs add framebuffer support.
- A way to read the TTL on your build host: `picocom` or `screen` at **115200 8N1**.

### Build host

- Rust nightly (pinned in `rust-toolchain.toml`) — `rustup target add aarch64-unknown-uefi` is automatic on first build.
- `mtools` (provides `mformat`, `mcopy`):
  - macOS: `brew install mtools`
  - Debian/Ubuntu: `sudo apt install mtools`

### Orin UEFI configuration

Phase 2 boots from USB via the standard UEFI fallback path
(`EFI/BOOT/BOOTAA64.EFI`). Your Orin needs:

- **Secure Boot disabled**, OR our `.efi` enrolled as a trusted image.
  Easiest path is to disable Secure Boot in UEFI Setup before the first
  boot. NVIDIA's UEFI on JetPack 6.x ships with a test auth key by
  default (`I/TC: WARNING: Test UEFI variable auth key is being used !`
  appears in the boot log) — Secure Boot enforcement is off by default
  on dev images.
- **USB in the boot order**, OR pop the UEFI Boot Manager menu manually
  (see below).

You don't need to flash anything to the Orin's eMMC/NVMe — the L4T
install on internal storage is untouched. The USB stick is the boot
media; pull it out and reboot to return to L4T.

---

## One-shot quickstart (assuming all prerequisites)

On the build host (Mac shown; Linux is identical):

```bash
just build-jetson-usb-image
# → produces build/smallaios-jetson-usb.img (256 MB FAT32, contains
#   EFI/BOOT/BOOTAA64.EFI = the SmallAIOS UEFI .efi)

# Find your USB device. CONFIRM TWICE — dd to the wrong device will
# overwrite that device.
diskutil list   # macOS
lsblk           # Linux

# Replace /dev/diskN (macOS) or /dev/sdX (Linux) with your USB stick.
# macOS: unmount the stick first.
diskutil unmountDisk /dev/diskN
sudo dd if=build/smallaios-jetson-usb.img of=/dev/rdiskN bs=4m
# Linux:
sudo dd if=build/smallaios-jetson-usb.img of=/dev/sdX bs=4M status=progress conv=fdatasync
```

> ⚠️ **WARNING — `dd` is destructive.** It overwrites whatever's at the
> target. Confirm the device path *twice* before pressing return. On
> macOS, `/dev/rdiskN` (raw) is faster than `/dev/diskN`. On Linux,
> `bs=4M` and `conv=fdatasync` give a reasonable speed/safety balance.

Then:

1. Plug the stick into a USB-A port on the Orin's carrier.
2. Connect TTL serial to the carrier's UART header. Open a terminal:
   - macOS: `picocom -b 115200 /dev/cu.usbserial-XXX`
   - Linux: `picocom -b 115200 /dev/ttyUSB0`
   - To exit picocom: `Ctrl-A` then `Ctrl-X`.
3. Power-cycle the Orin. The UEFI firmware will walk the USB stick,
   find `EFI/BOOT/BOOTAA64.EFI`, and run it.
4. Watch the TTL terminal for the SmallAIOS banner.

If UEFI auto-boots into L4T before reaching the USB, see "Forcing the
UEFI boot menu" below.

---

## Expected serial output (interim mode)

```
========================================
  Hello, world from SmallAIOS on Tegra234
========================================
[boot] DTB at 0x000000046799D000
[boot] EFI memory map: 12 conventional region(s)
[boot] skipping ExitBootServices (interim mode); calling kernel_main

========================================
  SmallAIOS 0.2.1
  Platform: Tegra234 (Jetson Orin)
========================================

[boot] Stage 1: Early initialization
[boot] BSS cleared, stack initialized
[boot] Running at EL2
[boot] DTB address: 0x46799D000

[boot] Stage 2: Memory detection
[mem]  DTB parsed: 12 region(s), 15915 MiB usable RAM

[boot] Stage 2.5: Heap allocator
[heap] Initialized: 976 MiB

[boot] Stage 3: Interrupt controller

========================================
  SmallAIOS 0.2.1 ready
========================================
```

After the `ready` line the kernel parks in `wfi`. To return to L4T,
power-cycle the Orin with the USB stick removed (UEFI falls through to
the eMMC/NVMe boot entry).

The exact memory size depends on your SKU — Orin Nano 8 GB will report
~7.5 GiB usable, Orin NX 16 GB ~15.5 GiB, AGX Orin 32 GB ~31 GiB.

---

## Forcing the UEFI boot menu

If the Orin auto-boots into L4T before the USB is selected, you need to
either change the UEFI boot order (UEFI Setup) or pop the boot menu
manually each time:

1. With TTL serial connected and `picocom` running, power-cycle.
2. Watch for the firmware banner: `Jetson System firmware version
   36.4.7…` followed by `ESC to enter Setup` / `F11 to enter Boot
   Manager Menu`.
3. Hammer **`F11`** repeatedly during the first few seconds. The Boot
   Manager menu appears.
4. Pick the USB entry (typically named after your stick label, e.g.
   `UEFI USB SanDisk Ultra`).
5. Press Enter. UEFI walks the stick's `EFI/BOOT/BOOTAA64.EFI`.

`Esc` (instead of `F11`) gets you into the Setup menu where you can
re-order boot devices permanently.

NVIDIA's UEFI on JetPack 6.x accepts both keypresses on the serial
console — no USB keyboard required.

---

## Recovery / unbricking

The USB-boot path is non-destructive: it only loads from the USB stick
and runs at EL2. **It does not modify NVRAM, eMMC, NVMe, or any
persistent state.** Pull the USB stick, power-cycle, you're back in
L4T.

The only persistent state Phase 2 *could* affect (in a future sub-PR)
would be UEFI boot variables, e.g. if we ran `efibootmgr` to add a
permanent boot entry. We don't currently do that, but if a future
session does, the unbricking path is:

1. Power-cycle without USB.
2. Boot the L4T install on internal storage as usual.
3. From L4T: `sudo efibootmgr -v` to inspect, `sudo efibootmgr -B
   -b XXXX` to delete the bad entry.

If even that fails (e.g. corrupted UEFI variables), the absolute
fallback is NVIDIA's SDK Manager + USB-recovery procedure: put the
Orin into recovery mode (hold REC, press POWER) and re-flash. See
[NVIDIA's Jetson Orin recovery
docs](https://docs.nvidia.com/jetson/archives/r36.4.7/DeveloperGuide/IN/QuickStart.html)
for the full procedure.

Phase 2 itself can't brick the box. But this section exists so the
recovery procedure is documented in one place if a future sub-PR lands
something more invasive.

---

## Troubleshooting

### UEFI never finds the USB / the stick boots into something else

- Verify the stick's filesystem: mount it on the build host and confirm
  `EFI/BOOT/BOOTAA64.EFI` exists. The path is case-sensitive on FAT32.
- Some sticks ship with vendor recovery partitions that confuse UEFI's
  fallback logic. `wipefs -a` the stick before `dd`-ing if you don't
  trust its prior state.
- On the Orin, enter UEFI Setup (Esc), confirm USB is in the boot
  order, and that the stick appears as a Boot Option.

### "Synchronous Exception at 0x000000045E…" or "Unhandled Exception in EL3"

You're seeing this from UEFI's exception handler. Most commonly means:

1. The kernel post-EBS path is hitting an MMIO firewall (this is the
   known interim-mode issue documented in
   `arch/aarch64/src/tegra234_uart.rs`).
2. The DTB pointer is garbage (UEFI reported it at an unmapped address).
3. ABI mismatch on `efi_main`.

Capture the full register dump from the exception and open an issue —
the `elr_el3` (PC at exception) and `x8` (often the address being
accessed) tell us where the fault is.

### Kernel banner appears but stops after `Stage 2: Memory detection`

If `[mem]  DTB parsed: 0 region(s), 0 MiB usable RAM` shows up on a
firmware variant we haven't tested, the EFI memory-map harvest hasn't
populated regions. Check the `[boot] EFI memory map: N conventional
region(s)` breadcrumb for the count — if it's 0, the firmware's memory
map doesn't expose conventional memory the way ours does and we need a
different harvest strategy.

### Picocom shows control characters / garbled text

Set `picocom -b 115200 /dev/cu.usbserial-XXX` (don't pass `--echo` or
other flow-control flags). NVIDIA's UEFI emits 8N1 + standard ANSI;
picocom defaults handle that.

### macOS: `dd: No such file or directory: /dev/diskN`

Use `diskutil unmountDisk /dev/diskN` first, then `dd`. The `/dev/disk*`
nodes only exist while the disk is recognized; if the stick was just
plugged in and macOS is still mounting it, wait a moment.

---

## CI parity

Phase 2 doesn't yet have a CI gate that boots the .efi under emulation —
that's sub-PR 2h's job (advisory build + OVMF QEMU smoke). The CI build
of `smallaios-uefi.efi` (also sub-PR 2h) catches `aarch64-unknown-uefi`
target regressions. End-to-end "kernel reaches idle loop on real
hardware" is a manual test against this doc — the captured TTL output
above is the acceptance evidence.

---

## What this does NOT do

- No GICv3 driver / interrupt dispatch / scheduler tick (sub-PR 2.13–2.16).
- No post-EBS UART (TTL output goes silent the moment `ExitBootServices`
  is called from EL2 — this is the open follow-up sub-PR after 2e-output).
- No GPU / display / camera / network — the Phase 2 milestone is "kernel
  runs and produces a banner." Hardware enablement past that is a
  separate change.
- No persistent boot-entry modification — the kernel runs entirely from
  the USB stick. To make Phase 2's kernel the default boot, you'd
  manually `efibootmgr` it from L4T (out of scope here).
