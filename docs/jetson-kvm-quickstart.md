# Jetson Orin KVM unikernel quickstart (Phase 1)

This document covers the **fastest** path to running the SmallAIOS
`aarch64-unknown-none` unikernel on Jetson Orin hardware: hosting it under
`qemu-system-aarch64 -accel kvm -cpu host` on the Jetson's L4T host kernel.

The Orin's Cortex-A78AE cores execute SmallAIOS instructions directly via
KVM. Peripherals come from QEMU's `virt` machine (PL011 UART at
`0x09000000`, GICv3, virtio devices). The unikernel sees a generic ARM
`virt` board — Tegra234-specific bring-up is Phase 2 of the
`unikernel-orin-bringup-v1` change.

This is a **container-host smoke test**, not bare-metal. The Orin's eMMC /
NVMe boot is untouched.

---

## Recommended workflow: Mac (or x86 Ubuntu) as build host, Orin as runner

Cross-compiling the kernel from a workstation and `scp`'ing the binary to
the Orin is the supported workflow. The `aarch64-unknown-none` target is
freestanding (`#![no_std]`, no libc), so cross-compile from
`aarch64-apple-darwin` or `x86_64-unknown-linux-gnu` works without a C
cross toolchain — `rust-lld` (bundled with the project's pinned nightly
via the `llvm-tools` component) handles the link.

This keeps the Orin a clean test target rather than a Rust dev environment
(saves ~3 GB of toolchain on the Jetson).

### One-time workstation setup

```bash
# Install rustup (one-time, no sudo, ~500 MB)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain none -y
source "$HOME/.cargo/env"

# Install just (the workspace task runner)
cargo install just --locked

# Clone the repo
git clone git@github.com:SmallAIOS/SmallAIOS.git
cd SmallAIOS
# rust-toolchain.toml auto-installs the pinned nightly + components on first cargo invocation
```

### One-time Jetson setup

```bash
# Install qemu-system-aarch64 on the Orin
sudo apt install qemu-system-arm

# Grant the dev user KVM access (KVM is built into the JetPack 6 kernel,
# so there is no `modprobe kvm` step — `/dev/kvm` exists at boot, but
# access is restricted to the `kvm` group)
sudo usermod -aG kvm $USER
# Log out and back in for the group membership to take effect, or run
# `newgrp kvm` to start a sub-shell with the group active in the current session.

# Verify
[ -c /dev/kvm ] && echo "/dev/kvm OK" || echo "MISSING — JetPack 6 should ship KVM builtin"
id -nG | tr ' ' '\n' | grep -q '^kvm$' && echo "kvm group OK" || echo "kvm group missing — re-login"
```

### Per-iteration loop

From the workstation:

```bash
# `just` wraps build + scp + ssh + qemu in one command
just run-jetson-kvm e@orin-nx.local
# (replace e@orin-nx.local with your Jetson's user@host)
```

What `run-jetson-kvm` does:

1. `just build-kernel-arm` — `cargo build --release --target aarch64-unknown-none -p smallaios-arch-aarch64` (release, with build-std). Produces `target/aarch64-unknown-none/release/smallaios-aarch64`.
2. `scp` the kernel ELF to `~/smallaios-aarch64` on the Jetson.
3. `ssh` into the Jetson and run `qemu-system-aarch64 -M virt,gic-version=3 -cpu host -accel kvm -m 1G -nographic -kernel ~/smallaios-aarch64 -serial mon:stdio`.

Cold-cache build on M-series Mac ≈ 2 min; incremental ≈ 5–30 s. The Orin-side run is sub-second to boot under KVM (the kernel reaches its banner before the SSH session can echo).

---

## Expected serial output (Phase 1 success)

```

========================================
  SmallAIOS 0.1.0
  Platform: AArch64 (QEMU virt)
========================================

[boot] Stage 1: Early initialization
[uart] PL011 @ 0x09000000 initialized
[boot] BSS cleared, stack initialized
[boot] Running at EL1
[boot] DTB address: 0x40000000
…
```

The unikernel keeps booting through memory detection, scheduler init, and
its idle loop. Press `Ctrl-A X` (the QEMU `-nographic` exit) to terminate.

The **Phase 1 acceptance criteria** are:

1. `SmallAIOS` appears in the serial output (proves `kernel_main` ran).
2. `[boot] BSS cleared` appears (proves the assembly `_start` reached
   BSS-clear stage and the EL2→EL1 transition worked).

The `scripts/test-jetson-kvm.sh` smoke script (and `just test-jetson-kvm`
recipe) wraps these assertions and exits non-zero if either is missing.

---

## Local mode (no SSH)

If you've installed Rust on the Jetson itself (not recommended — see the
recommended workflow above), `just run-jetson-kvm` with no `SSH_HOST` arg
runs build + qemu locally:

```bash
# On the Jetson:
just run-jetson-kvm
```

The recipe checks `[ -c /dev/kvm ]` and `[ -r /dev/kvm ] && [ -w /dev/kvm ]`
before invoking QEMU; missing access exits non-zero with a hint to fix the
group membership.

---

## Troubleshooting

### `/dev/kvm: Permission denied` (qemu exits immediately)

You're not in the `kvm` group. Run:

```bash
sudo usermod -aG kvm $USER
# log out + back in (or `newgrp kvm` for current session only)
id -nG | grep kvm
```

KVM is **built into** the JetPack 6 kernel (`modinfo kvm` reports
`filename: (builtin)`). There is no `modprobe kvm` workaround because
there's no module to load; `/dev/kvm` either exists or the running kernel
is not a JetPack 6 kernel — escalate.

### `qemu-system-aarch64: command not found`

```bash
sudo apt install qemu-system-arm
```

The `qemu-system-arm` package on Ubuntu provides `qemu-system-aarch64`
despite the name (32-bit-suffixed, 64-bit-capable).

### `qemu-system-aarch64: gic-version=3 not supported`

You're on an old QEMU (<2.10). Modern QEMU on Ubuntu 22.04 (the L4T R36
base) ships 6.2+, which supports GICv3. Update qemu, or fall back to GICv2
(the kernel doesn't currently support GICv2 on `qemu-virt` — it's a
GICv3-only path).

### Kernel boots but immediately page-faults / crashes

Common early-boot failure patterns and what to inspect:

- **No serial output at all (timeout 30s, log empty)** — `_start` did not
  complete the EL2→EL1 transition, or the assembly hit an exception
  before reaching `kernel_main`. Check `arch/aarch64/src/boot.rs`. KVM
  vs. TCG can differ on the EL the firmware hands off at; QEMU virt + KVM
  hands off at EL2, which the existing `_start` is written to handle.
- **Banner prints but BSS-clear marker missing** — likely the BSS bounds
  symbols (`__bss_start` / `__bss_end`) are wrong in the linker script
  for some build flavor. Check `arch/aarch64/linker.ld`.
- **"Hello, world but then page fault"** — likely an MMU/paging issue in
  `arch/aarch64/src/paging.rs` or memory map in `arch/aarch64/src/platform.rs`.
  Note that on QEMU virt + KVM, memory layout differs from Tegra234 bare
  metal (which is Phase 2 territory).

### Boot is fast but no output appears

Likely a serial-routing issue. With `-nographic -serial mon:stdio`, QEMU
multiplexes monitor + first PL011 UART onto the controlling terminal. If
the kernel is writing to a different UART (say, the Tegra X1 NS16550
instead of QEMU virt's PL011), nothing shows up. Confirm the build used
the default `qemu-virt` feature, not `tegra-x1`:

```bash
# Default features include qemu-virt; if you accidentally built with
# --no-default-features --features tegra-x1, the kernel writes to a UART
# that QEMU virt doesn't expose.
cargo build --release --target aarch64-unknown-none -p smallaios-arch-aarch64
```

---

## What this does NOT exercise

- **Tegra234-specific hardware.** The unikernel sees QEMU virt: PL011
  UART at `0x09000000`, GICv3, virtio devices. Tegra234's Combined UART
  (TCU at `0x0c280000`), GICv3 with LPI/ITS, host1x, GPU (Ampere GA10B),
  PCIe controllers, etc. — all absent. Phase 2 of the
  `unikernel-orin-bringup-v1` change adds those.
- **Bare-metal boot.** The kernel boots from an in-memory ELF loaded by
  QEMU, not from the Orin's UEFI firmware on NVMe / USB. Phase 2 also
  adds the UEFI USB-stick boot path.
- **GPU access.** Out of scope for both Phase 1 and Phase 2 of this
  change. Tracked separately.

---

## CI parity

`.github/workflows/ci.yml` runs an `aarch64-qemu-smoke` job on every PR
that does the equivalent boot under TCG (no KVM on GitHub Linux runners).
Same `qemu-system-aarch64 -M virt,gic-version=3 …` invocation, same
banner assertions. If this job is green, the Phase 1 path is validated for
the build + boot + serial-write loop; the on-Jetson run additionally
validates the `-accel kvm -cpu host` path on real Cortex-A78AE cores.
