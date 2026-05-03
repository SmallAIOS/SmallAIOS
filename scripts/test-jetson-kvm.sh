#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Jetson Orin unikernel KVM smoke test (Phase 1 acceptance).
#
# Boots the SmallAIOS aarch64-unknown-none kernel under qemu-system-aarch64
# with -accel kvm -cpu host on the Jetson Orin's L4T host kernel, and
# asserts the kernel reaches its early-boot banner.
#
# Two execution modes (mirrors `just run-jetson-kvm`):
#   - Pass SSH_HOST as $1: cross-build locally, scp to that host, run via ssh.
#     Use this from a Mac / x86 dev box.
#   - Omit $1: build and run locally. Use this when this script runs on the
#     Jetson itself (Rust toolchain must be installed locally).
#
# Asserts the captured serial output contains both:
#   1. "SmallAIOS"            — the boot banner printed by kernel_main()
#   2. "BSS cleared"          — proves _start reached BSS-clear stage
# Either missing → failure.
#
# Tears the QEMU process down on exit (success or failure).
#
# Exit codes:
#   0   all checks passed (kernel booted to BSS-clear stage)
#   10  build failed (cargo / just exit non-zero)
#   20  qemu-system-aarch64 missing on the runner
#   21  /dev/kvm missing or inaccessible on the runner
#   30  boot timed out (kernel did not print banner within the timeout window)
#   40  banner assertion failed (kernel printed something but not the expected lines)
#   50  prerequisite check failed (missing scp/ssh in SSH_HOST mode, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

SSH_HOST="${1:-}"
KERNEL_PATH="${KERNEL_PATH:-target/aarch64-unknown-none/release/smallaios-aarch64}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-30}"
BANNER_PRIMARY="${BANNER_PRIMARY:-SmallAIOS}"
BANNER_SECONDARY="${BANNER_SECONDARY:-BSS cleared}"

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
info()   { yellow "[jetson-kvm] $*"; }
ok()     { green  "[jetson-kvm] $*"; }
fail()   { red    "[jetson-kvm] $*"; }

LOG=$(mktemp -t jetson-kvm-boot.XXXXXX.log)
trap 'rm -f "$LOG"' EXIT

# ─── Step 1: Prerequisites ────────────────────────────────────────────────────
info "Checking prerequisites"
if [ -n "$SSH_HOST" ]; then
    command -v scp >/dev/null 2>&1 || { fail "scp not found"; exit 50; }
    command -v ssh >/dev/null 2>&1 || { fail "ssh not found"; exit 50; }
    command -v cargo >/dev/null 2>&1 || { fail "cargo not found (build host)"; exit 50; }
    info "Cross-build mode: SSH_HOST=$SSH_HOST"
else
    command -v qemu-system-aarch64 >/dev/null 2>&1 || { fail "qemu-system-aarch64 not found (apt install qemu-system-arm)"; exit 20; }
    [ -c /dev/kvm ] || { fail "/dev/kvm not present (Phase 1 requires KVM)"; exit 21; }
    [ -r /dev/kvm ] && [ -w /dev/kvm ] || { fail "/dev/kvm not accessible (sudo usermod -aG kvm \$USER && re-login)"; exit 21; }
    command -v cargo >/dev/null 2>&1 || { fail "cargo not found (local mode requires Rust toolchain)"; exit 50; }
    info "Local mode (running on the Jetson)"
fi

# ─── Step 2: Build the kernel ─────────────────────────────────────────────────
info "Building AArch64 kernel (release)"
# Set RUSTFLAGS explicitly (matches `Build AArch64 Kernel` CI job and the
# `build-kernel-arm` Justfile recipe). Without it, cargo falls back to
# `[target.aarch64-unknown-none].rustflags` from `.cargo/config.toml`, which
# it then doubles into the bin's rustc invocation when `-Z build-std` is in
# play, causing rust-lld to emit overlapping section file offsets.
if ! RUSTFLAGS="-C link-arg=-Tarch/aarch64/linker.ld" \
    cargo build --release --target aarch64-unknown-none -p smallaios-arch-aarch64 \
    -Z build-std=core,compiler_builtins,alloc \
    -Z build-std-features=compiler-builtins-mem 2>&1; then
    fail "Build failed"
    exit 10
fi
[ -f "$KERNEL_PATH" ] || { fail "Kernel artifact missing after build: $KERNEL_PATH"; exit 10; }
ok "Kernel built: $KERNEL_PATH ($(stat -c%s "$KERNEL_PATH" 2>/dev/null || stat -f%z "$KERNEL_PATH") bytes)"

# ─── Step 3: Boot under QEMU+KVM and capture serial ───────────────────────────
info "Booting under qemu-system-aarch64 (timeout ${BOOT_TIMEOUT}s)"
if [ -n "$SSH_HOST" ]; then
    REMOTE_BIN="$(basename "$KERNEL_PATH")"
    info "Copying kernel to $SSH_HOST:~/$REMOTE_BIN"
    scp -q "$KERNEL_PATH" "$SSH_HOST:~/$REMOTE_BIN"
    info "Running qemu-system-aarch64 over ssh on $SSH_HOST"
    # `timeout` runs on the remote (Linux/coreutils) so the build host
    # doesn't need GNU coreutils — Apple Silicon Mac doesn't ship it.
    ssh "$SSH_HOST" \
        "timeout --foreground $BOOT_TIMEOUT qemu-system-aarch64 \
         -M virt,gic-version=3 -cpu host -accel kvm -m 1G -nographic \
         -kernel ~/$REMOTE_BIN -serial mon:stdio" \
        > "$LOG" 2>&1 || true
else
    # Local mode: this script is running on the Jetson itself.
    timeout --foreground "$BOOT_TIMEOUT" qemu-system-aarch64 \
        -M virt,gic-version=3 -cpu host -accel kvm -m 1G -nographic \
        -kernel "$KERNEL_PATH" -serial mon:stdio \
        > "$LOG" 2>&1 || true
fi

# ─── Step 4: Assert banner ────────────────────────────────────────────────────
echo "=== Captured boot output (${LOG}) ==="
cat "$LOG"
echo "=== End boot output ==="

if [ ! -s "$LOG" ]; then
    fail "QEMU produced no output within ${BOOT_TIMEOUT}s — kernel did not boot or serial misconfigured"
    exit 30
fi

if ! grep -q "$BANNER_PRIMARY" "$LOG"; then
    fail "Boot banner '$BANNER_PRIMARY' missing from serial output"
    exit 40
fi

if ! grep -q "$BANNER_SECONDARY" "$LOG"; then
    fail "Stage marker '$BANNER_SECONDARY' missing — kernel printed banner but did not reach BSS-clear stage"
    exit 40
fi

ok "PASS — kernel booted under KVM and reached BSS-clear stage"
ok "Phase 1 acceptance criteria satisfied"
