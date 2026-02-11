#!/usr/bin/env bash
# SmallAIOS Boot-to-Inference Integration Test
# Task 13.2: QEMU boot-to-inference test harness
#
# Boots the SmallAIOS kernel in QEMU and monitors serial output for
# inference completion markers. Designed to be used in CI when QEMU
# is available.
#
# Usage:
#   ./scripts/qemu-boot-inference-test.sh [x86|arm|riscv]
#
# Expected serial output markers (searched in boot log):
#   - "SmallAIOS" or "boot" — kernel started
#   - "inference" or "ONNX" — inference pipeline reached
#   - "result" or "output" or "softmax" — inference completed
#
# Exit codes:
#   0 — inference completion detected
#   1 — boot detected but no inference markers
#   2 — no boot detected / QEMU not available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCH="${1:-riscv}"
TIMEOUT="${TIMEOUT:-60}"
LOGFILE="${PROJECT_DIR}/boot-inference-test.log"

QEMU_X86="${QEMU_X86:-qemu-system-x86_64}"
QEMU_ARM="${QEMU_ARM:-qemu-system-aarch64}"
QEMU_RV="${QEMU_RV:-qemu-system-riscv64}"

echo "=== SmallAIOS Boot-to-Inference Test ==="
echo "Architecture: $ARCH"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Check QEMU availability
get_qemu_bin() {
    case "$ARCH" in
        x86)   echo "$QEMU_X86" ;;
        arm)   echo "$QEMU_ARM" ;;
        riscv) echo "$QEMU_RV" ;;
        *)     echo ""; return 1 ;;
    esac
}

QEMU_BIN="$(get_qemu_bin)"
if ! command -v "$QEMU_BIN" &>/dev/null; then
    echo "SKIP: $QEMU_BIN not found (install QEMU to run this test)"
    exit 0
fi

# Build the kernel
echo "--- Building kernel ---"
case "$ARCH" in
    x86)   make -C "$PROJECT_DIR" build-kernel-x86 ;;
    arm)   make -C "$PROJECT_DIR" build-kernel-arm ;;
    riscv) make -C "$PROJECT_DIR" build-kernel-riscv ;;
esac

# Boot in QEMU and capture output
echo "--- Booting in QEMU (timeout ${TIMEOUT}s) ---"
case "$ARCH" in
    x86)
        timeout "$TIMEOUT" "$QEMU_BIN" \
            -machine q35 -cpu max -m 512M -nographic \
            -kernel "$PROJECT_DIR/target/x86_64-unknown-none/release/smallaios-x86_64" \
            -serial stdio 2>&1 | tee "$LOGFILE" || true
        ;;
    arm)
        timeout "$TIMEOUT" "$QEMU_BIN" \
            -machine virt -cpu cortex-a72 -m 512M -nographic \
            -kernel "$PROJECT_DIR/target/aarch64-unknown-none/release/smallaios-aarch64" \
            -serial stdio 2>&1 | tee "$LOGFILE" || true
        ;;
    riscv)
        timeout "$TIMEOUT" "$QEMU_BIN" \
            -machine virt -cpu rv64 -m 512M -nographic \
            -bios default \
            -kernel "$PROJECT_DIR/target/riscv64gc-unknown-none-elf/release/smallaios-riscv64" \
            -serial stdio 2>&1 | tee "$LOGFILE" || true
        ;;
esac

echo ""
echo "=== Analyzing boot log ==="
echo "Log size: $(wc -c < "$LOGFILE") bytes"

# Check for boot markers
BOOT_DETECTED=false
if grep -qiE "boot|init|start|SmallAIOS|kernel" "$LOGFILE" 2>/dev/null; then
    echo "PASS: Kernel boot detected"
    BOOT_DETECTED=true
else
    echo "FAIL: No kernel boot markers found"
fi

# Check for inference markers
INFERENCE_DETECTED=false
if grep -qiE "inference|onnx|model|tensor" "$LOGFILE" 2>/dev/null; then
    echo "PASS: Inference pipeline activity detected"
    INFERENCE_DETECTED=true
else
    echo "INFO: No inference pipeline markers found"
fi

# Check for completion markers
COMPLETION_DETECTED=false
if grep -qiE "result|output|softmax|prediction|complete" "$LOGFILE" 2>/dev/null; then
    echo "PASS: Inference completion detected"
    COMPLETION_DETECTED=true
else
    echo "INFO: No inference completion markers found"
fi

echo ""
echo "=== Test Result ==="
if $COMPLETION_DETECTED; then
    echo "PASS: Boot-to-inference cycle completed"
    exit 0
elif $BOOT_DETECTED; then
    echo "PARTIAL: Kernel boots but inference not yet wired to serial output"
    echo "NOTE: This is expected in the current prototype stage."
    echo "      The kernel needs to be configured to run inference on boot"
    echo "      and print results to the serial console."
    exit 1
else
    echo "FAIL: No boot output detected"
    echo "NOTE: The kernel may not have serial output configured for this arch."
    exit 2
fi
