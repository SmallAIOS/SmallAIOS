#!/bin/bash
# SmallAIOS Serial Console Helper
# SPDX-License-Identifier: Apache-2.0
#
# Connects to a dev board's serial console with the right settings.
#
# Usage:
#   ./scripts/serial-console.sh                    # Auto-detect device
#   ./scripts/serial-console.sh /dev/ttyUSB0       # Specific device
#   ./scripts/serial-console.sh --baud 921600      # Custom baud rate

set -euo pipefail

BAUD=115200
DEVICE=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[serial]${NC} $*"; }
warn()  { echo -e "${YELLOW}[serial]${NC} $*"; }
error() { echo -e "${RED}[serial]${NC} $*" >&2; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [device] [options]

Arguments:
  device            Serial device path (auto-detected if omitted)

Options:
  --baud RATE       Baud rate (default: 115200)
  --list            List available serial devices

Auto-detection priority:
  1. /dev/ttyACM*  (Jetson built-in FTDI)
  2. /dev/ttyUSB*  (USB-to-UART adapters)

Exit: Ctrl-A then Ctrl-X (picocom) or Ctrl-A then X (minicom)
EOF
}

list_devices() {
    info "Available serial devices:"
    echo ""

    local found=false

    for dev in /dev/ttyACM* /dev/ttyUSB*; do
        if [[ -c "$dev" ]]; then
            found=true
            local desc=""
            # Try to get device info via udevadm
            if command -v udevadm &>/dev/null; then
                desc=$(udevadm info --name="$dev" --query=property 2>/dev/null | \
                    grep -E "^ID_MODEL=" | cut -d= -f2 || echo "")
            fi
            if [[ -n "$desc" ]]; then
                echo "  $dev  ($desc)"
            else
                echo "  $dev"
            fi
        fi
    done

    if [[ "$found" == "false" ]]; then
        warn "No serial devices found."
        echo ""
        echo "Check:"
        echo "  - USB cable is connected"
        echo "  - Board is powered on"
        echo "  - You have permission: sudo usermod -a -G dialout \$USER"
    fi
}

auto_detect() {
    # Prefer ttyACM (Jetson built-in) over ttyUSB (external adapter)
    for pattern in /dev/ttyACM* /dev/ttyUSB*; do
        for dev in $pattern; do
            if [[ -c "$dev" ]]; then
                echo "$dev"
                return 0
            fi
        done
    done

    return 1
}

connect() {
    local dev="$1"
    local baud="$2"

    if [[ ! -c "$dev" ]]; then
        error "Device not found: $dev"
        echo ""
        list_devices
        exit 1
    fi

    # Check permissions
    if [[ ! -r "$dev" || ! -w "$dev" ]]; then
        error "Permission denied: $dev"
        echo "Fix with: sudo usermod -a -G dialout \$USER"
        echo "Then log out and back in."
        exit 1
    fi

    # Use picocom if available, fall back to minicom, then screen
    if command -v picocom &>/dev/null; then
        info "Connecting to $dev @ ${baud}bps (picocom)"
        info "Exit: Ctrl-A then Ctrl-X"
        echo ""
        exec picocom -b "$baud" "$dev"
    elif command -v minicom &>/dev/null; then
        info "Connecting to $dev @ ${baud}bps (minicom)"
        info "Exit: Ctrl-A then X"
        echo ""
        exec minicom -D "$dev" -b "$baud"
    elif command -v screen &>/dev/null; then
        info "Connecting to $dev @ ${baud}bps (screen)"
        info "Exit: Ctrl-A then K, then Y"
        echo ""
        exec screen "$dev" "$baud"
    else
        error "No serial terminal found. Install one:"
        echo "  sudo apt install picocom    # recommended"
        echo "  sudo apt install minicom"
        echo "  sudo apt install screen"
        exit 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baud) BAUD="$2"; shift 2 ;;
        --list) list_devices; exit 0 ;;
        --help|-h) usage; exit 0 ;;
        /dev/*) DEVICE="$1"; shift ;;
        *) error "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

# Auto-detect if no device specified
if [[ -z "$DEVICE" ]]; then
    if DEVICE=$(auto_detect); then
        info "Auto-detected: $DEVICE"
    else
        error "No serial device detected."
        echo ""
        list_devices
        exit 1
    fi
fi

connect "$DEVICE" "$BAUD"
