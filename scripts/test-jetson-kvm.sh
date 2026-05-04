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
#   30  boot timed out (kernel did not print banner within the timeout window;
#       reported when `timeout` returns 124/137 *and* the BSS-clear marker
#       is absent — distinguishes a true timeout from a banner miss)
#   40  banner assertion failed (kernel printed something but not the
#       expected lines, *not* a timeout)
#   50  prerequisite or transport failed: missing scp/ssh, scp transfer error,
#       BOOT_TIMEOUT not numeric, or ssh exit 255 (host unreachable / auth)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

SSH_HOST="${1:-}"
KERNEL_PATH="${KERNEL_PATH:-target/aarch64-unknown-none/release/smallaios-aarch64}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-30}"
BANNER_PRIMARY="${BANNER_PRIMARY:-SmallAIOS}"
BANNER_SECONDARY="${BANNER_SECONDARY:-BSS cleared}"

# ANSI color helpers — write a single line in the named color to stdout.
# Always emit even if stdout isn't a TTY; CI captures the codes harmlessly.
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }

# Status helpers — prefix the message with "[jetson-kvm]" and color it
# according to severity. info=yellow (in-progress), ok=green (success),
# fail=red (terminal failure; caller should exit immediately after).
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
# RUSTFLAGS matches the CI `build-aarch64` and `aarch64-qemu-smoke` jobs
# byte-for-byte (`-D warnings -C link-arg=-Tarch/aarch64/linker.ld`) so a
# build that's clean here is also clean in CI. Setting it via env (rather
# than relying on `[target.aarch64-unknown-none].rustflags` in
# `.cargo/config.toml`) also dodges a cargo quirk: under `-Z build-std`,
# config-file rustflags get doubled into the bin's rustc invocation, which
# loads `linker.ld` twice and makes rust-lld emit overlapping section
# file offsets.
if ! RUSTFLAGS="-D warnings -C link-arg=-Tarch/aarch64/linker.ld" \
    cargo build --release --target aarch64-unknown-none -p smallaios-arch-aarch64 \
    -Z build-std=core,compiler_builtins,alloc \
    -Z build-std-features=compiler-builtins-mem 2>&1; then
    fail "Build failed"
    exit 10
fi
[ -f "$KERNEL_PATH" ] || { fail "Kernel artifact missing after build: $KERNEL_PATH"; exit 10; }
ok "Kernel built: $KERNEL_PATH ($(stat -c%s "$KERNEL_PATH" 2>/dev/null || stat -f%z "$KERNEL_PATH") bytes)"

# Validate BOOT_TIMEOUT is a positive integer before embedding in remote cmd.
case "$BOOT_TIMEOUT" in
    ''|*[!0-9]*) fail "BOOT_TIMEOUT must be a positive integer (got: '$BOOT_TIMEOUT')"; exit 50 ;;
esac

# ─── Step 3: Boot under QEMU+KVM and capture serial ───────────────────────────
info "Booting under qemu-system-aarch64 (timeout ${BOOT_TIMEOUT}s)"
RUN_RC=0
if [ -n "$SSH_HOST" ]; then
    REMOTE_BIN="$(basename "$KERNEL_PATH")"
    info "Copying kernel to $SSH_HOST:~/$REMOTE_BIN"
    if ! scp -q "$KERNEL_PATH" "$SSH_HOST:~/$REMOTE_BIN" 2>"$LOG"; then
        cat "$LOG" >&2
        fail "scp to $SSH_HOST failed (bad host/key, permission denied, disk full, network?)"
        exit 50
    fi
    info "Running qemu-system-aarch64 over ssh on $SSH_HOST"
    # `timeout` runs on the remote (Linux/coreutils) so the build host
    # doesn't need GNU coreutils — Apple Silicon Mac doesn't ship it.
    # Disable -e briefly so we can capture the ssh exit code without aborting.
    set +e
    ssh "$SSH_HOST" \
        "timeout --foreground $BOOT_TIMEOUT qemu-system-aarch64 \
         -M virt,gic-version=3 -cpu host -accel kvm -m 1G -nographic \
         -kernel ~/$REMOTE_BIN -serial mon:stdio" \
        > "$LOG" 2>&1
    RUN_RC=$?
    set -e
else
    # Local mode: this script is running on the Jetson itself.
    set +e
    timeout --foreground "$BOOT_TIMEOUT" qemu-system-aarch64 \
        -M virt,gic-version=3 -cpu host -accel kvm -m 1G -nographic \
        -kernel "$KERNEL_PATH" -serial mon:stdio \
        > "$LOG" 2>&1
    RUN_RC=$?
    set -e
fi

# ─── Step 4: Classify exit + assert banner ────────────────────────────────────
echo "=== Captured boot output (${LOG}) ==="
cat "$LOG"
echo "=== End boot output ==="

# SSH connection / auth failures (e.g. host unreachable, bad key) typically
# return 255 in the SSH_HOST path. Treat as a prerequisite failure (50) so it
# isn't masquerading as a banner-assertion miss.
if [ -n "$SSH_HOST" ] && [ "$RUN_RC" -eq 255 ]; then
    fail "ssh to $SSH_HOST failed (exit 255) before/while running QEMU"
    exit 50
fi

# Whether or not banners are present, a `timeout` exit (124, or 137 from
# SIGKILL after the grace window) without the early-boot marker means the
# kernel didn't reach BSS-clear inside the budget — that's the documented
# boot-timeout case (30), not a banner-assertion miss (40).
if { [ "$RUN_RC" -eq 124 ] || [ "$RUN_RC" -eq 137 ]; } \
   && ! grep -q "$BANNER_SECONDARY" "$LOG"; then
    fail "Boot timed out after ${BOOT_TIMEOUT}s without reaching '$BANNER_SECONDARY'"
    exit 30
fi

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
