#!/bin/bash
# SmallAIOS RISC-V QEMU runner
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

KERNEL="${1:-target/riscv64gc-unknown-none-elf/release/smallaios-riscv64}"

exec qemu-system-riscv64 \
    -machine virt \
    -cpu rv64 \
    -m 512M \
    -nographic \
    -bios default \
    -kernel "$KERNEL" \
    -serial stdio \
    -no-reboot \
    -d guest_errors
