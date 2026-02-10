#!/bin/bash
# SmallAIOS NVIDIA Jetson USB Recovery Flash Script
# SPDX-License-Identifier: Apache-2.0
#
# Flashes SmallAIOS to Jetson dev boards via USB recovery mode.
# Requires NVIDIA L4T Board Support Package installed on the host.
#
# Usage:
#   ./scripts/deploy-jetson-flash.sh <l4t-path> [options]
#
# The Jetson must be in USB recovery mode (hold RECOVERY + press RESET).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNEL_NAME="smallaios-aarch64"
TARGET="aarch64-unknown-none"
PACKAGE="smallaios-arch-aarch64"
BUILD_STD="-Z build-std=core,compiler_builtins,alloc -Z build-std-features=compiler-builtins-mem"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[jetson]${NC} $*"; }
warn()  { echo -e "${YELLOW}[jetson]${NC} $*"; }
error() { echo -e "${RED}[jetson]${NC} $*" >&2; }

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  flash <l4t-path>    Flash SmallAIOS to Jetson via USB recovery
  check               Check if Jetson is in recovery mode
  netboot-setup       Configure U-Boot for TFTP network boot

Options:
  --board BOARD       Board config name (default: jetson-orin-nano-devkit)
  --skip-build        Don't rebuild the kernel
  --server-ip IP      TFTP server IP (for netboot-setup)

Supported boards:
  jetson-orin-nano-devkit   Jetson Orin Nano Developer Kit
  jetson-nano-devkit        Jetson Nano Developer Kit (legacy)

Prerequisites:
  - NVIDIA L4T BSP extracted (download from developer.nvidia.com)
  - Jetson in USB recovery mode (hold RECOVERY + press RESET)
  - USB cable between host and Jetson USB-C/micro-USB port
  - Running as root (for USB device access)

Example:
  # Put Jetson in recovery mode, then:
  sudo ./scripts/deploy-jetson-flash.sh flash ~/nvidia/Linux_for_Tegra
EOF
}

check_recovery_mode() {
    info "Checking for Jetson in USB recovery mode..."

    if ! command -v lsusb &>/dev/null; then
        error "lsusb not found. Install usbutils: sudo apt install usbutils"
        exit 1
    fi

    # NVIDIA USB IDs for recovery mode
    # 0955:7023 = Jetson Orin Nano
    # 0955:7f21 = Jetson Nano
    # 0955:7c18 = Jetson AGX Orin
    local found=false
    local device_info

    if lsusb | grep -q "0955:7023"; then
        device_info="Jetson Orin Nano"
        found=true
    elif lsusb | grep -q "0955:7f21"; then
        device_info="Jetson Nano"
        found=true
    elif lsusb | grep -q "0955:"; then
        device_info="NVIDIA device (unknown variant)"
        found=true
    fi

    if [[ "$found" == "true" ]]; then
        info "Found: $device_info in recovery mode"
        return 0
    else
        warn "No Jetson found in USB recovery mode."
        echo ""
        echo "To enter recovery mode:"
        echo "  1. Power off the Jetson"
        echo "  2. Hold the RECOVERY button (middle button on carrier board)"
        echo "  3. While holding RECOVERY, press RESET (or plug in power)"
        echo "  4. Release RECOVERY after 2 seconds"
        echo "  5. Verify: lsusb | grep -i nvidia"
        echo ""
        return 1
    fi
}

flash_jetson() {
    local l4t_path="$1"
    local board="${2:-jetson-orin-nano-devkit}"
    local skip_build="${3:-false}"

    if [[ $EUID -ne 0 ]]; then
        error "Flashing requires root."
        exit 1
    fi

    # Validate L4T path
    if [[ ! -d "$l4t_path" ]]; then
        error "L4T path not found: $l4t_path"
        error "Download from: https://developer.nvidia.com/embedded/jetson-linux"
        exit 1
    fi

    if [[ ! -f "$l4t_path/flash.sh" ]]; then
        error "flash.sh not found in: $l4t_path"
        error "Ensure this is the Linux_for_Tegra directory from the L4T BSP."
        exit 1
    fi

    # Check recovery mode
    if ! check_recovery_mode; then
        exit 1
    fi

    # Build kernel
    if [[ "$skip_build" == "false" ]]; then
        cd "$PROJECT_DIR"
        info "Building SmallAIOS for $TARGET (release)..."
        # shellcheck disable=SC2086
        cargo build --release --target "$TARGET" -p "$PACKAGE" $BUILD_STD
    fi

    local kernel_path="$PROJECT_DIR/target/$TARGET/release/$KERNEL_NAME"
    if [[ ! -f "$kernel_path" ]]; then
        error "Kernel not found: $kernel_path"
        exit 1
    fi

    local kernel_size
    kernel_size=$(stat -c%s "$kernel_path" 2>/dev/null || stat -f%z "$kernel_path")
    info "Kernel size: $(( kernel_size / 1024 )) KiB"

    # Back up existing kernel in L4T
    local l4t_kernel="$l4t_path/kernel/Image"
    if [[ -f "$l4t_kernel" ]]; then
        info "Backing up original kernel..."
        cp "$l4t_kernel" "${l4t_kernel}.bak"
    fi

    # Copy SmallAIOS as the kernel image
    info "Installing SmallAIOS as kernel image..."
    cp "$kernel_path" "$l4t_kernel"

    # Flash
    info "Flashing to Jetson ($board)..."
    info "This may take several minutes..."
    cd "$l4t_path"
    ./flash.sh "$board" mmcblk0p1

    info "Flash complete!"
    info "The Jetson will reboot into SmallAIOS."
    info "Serial console: make serial DEV=/dev/ttyACM0"
}

setup_netboot() {
    local server_ip="${1:-}"

    if [[ -z "$server_ip" ]]; then
        error "--server-ip is required for netboot-setup"
        exit 1
    fi

    echo ""
    info "Jetson U-Boot Network Boot Setup"
    info "================================"
    echo ""
    echo "Connect to the Jetson serial console and interrupt U-Boot"
    echo "(press any key during the boot countdown)."
    echo ""
    echo "Then paste these commands at the U-Boot prompt:"
    echo ""
    echo "  setenv bootcmd 'dhcp; tftpboot \${kernel_addr_r} $KERNEL_NAME; booti \${kernel_addr_r} - \${fdt_addr}'"
    echo "  setenv serverip $server_ip"
    echo "  saveenv"
    echo "  reset"
    echo ""
    echo "After this, the Jetson will TFTP-boot SmallAIOS on every power-on."
    echo ""
    echo "To revert to normal boot later:"
    echo "  env default -a"
    echo "  saveenv"
    echo "  reset"
}

# Parse arguments
COMMAND="${1:-}"
shift || true

L4T_PATH=""
BOARD="jetson-orin-nano-devkit"
SKIP_BUILD=false
SERVER_IP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --board) BOARD="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --server-ip) SERVER_IP="$2"; shift 2 ;;
        /*|~/*) L4T_PATH="$1"; shift ;;
        *) L4T_PATH="$1"; shift ;;
    esac
done

case "$COMMAND" in
    flash)
        [[ -z "$L4T_PATH" ]] && { error "L4T path required"; usage; exit 1; }
        flash_jetson "$L4T_PATH" "$BOARD" "$SKIP_BUILD"
        ;;
    check)
        check_recovery_mode
        ;;
    netboot-setup)
        setup_netboot "$SERVER_IP"
        ;;
    *)
        usage
        exit 1
        ;;
esac
