#!/bin/bash
# SmallAIOS x86-64 QEMU runner
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

KERNEL="${1:-target/x86_64-unknown-none/release/smallaios-x86_64}"

exec qemu-system-x86_64 \
    -machine q35 \
    -cpu max \
    -m 512M \
    -nographic \
    -kernel "$KERNEL" \
    -serial stdio \
    -no-reboot \
    -d guest_errors
