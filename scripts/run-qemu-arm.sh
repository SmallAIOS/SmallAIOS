#!/bin/bash
# SmallAIOS AArch64 QEMU runner
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

KERNEL="${1:-target/aarch64-unknown-none/release/smallaios-aarch64}"

exec qemu-system-aarch64 \
    -machine virt \
    -cpu cortex-a72 \
    -m 512M \
    -nographic \
    -kernel "$KERNEL" \
    -serial stdio \
    -no-reboot \
    -d guest_errors
