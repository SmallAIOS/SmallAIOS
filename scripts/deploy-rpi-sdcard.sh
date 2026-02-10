#!/bin/bash
# SmallAIOS Raspberry Pi SD Card Deploy Script
# SPDX-License-Identifier: Apache-2.0
#
# Creates a bootable SD card with RPi UEFI firmware + SmallAIOS kernel.
#
# Usage:
#   ./scripts/deploy-rpi-sdcard.sh full /dev/sdX         # Full image (UEFI + kernel)
#   ./scripts/deploy-rpi-sdcard.sh update /dev/sdX        # Update kernel only
#   ./scripts/deploy-rpi-sdcard.sh uefi-only /dev/sdX     # UEFI firmware only (for network boot)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/smallaios"
KERNEL_NAME="smallaios-aarch64"
TARGET="aarch64-unknown-none"
PACKAGE="smallaios-arch-aarch64"
BUILD_STD="-Z build-std=core,compiler_builtins,alloc -Z build-std-features=compiler-builtins-mem"

# RPi UEFI firmware release — update this when new versions are available
RPI_UEFI_VERSION="v1.39"
RPI_UEFI_URL="https://github.com/pftf/RPi4/releases/download/${RPI_UEFI_VERSION}/RPi4_UEFI_Firmware_${RPI_UEFI_VERSION}.zip"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[rpi-sd]${NC} $*"; }
warn()  { echo -e "${YELLOW}[rpi-sd]${NC} $*"; }
error() { echo -e "${RED}[rpi-sd]${NC} $*" >&2; }

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> <device>

Commands:
  full       Create complete bootable SD card (wipes device)
  update     Update kernel on existing SD card (preserves UEFI)
  uefi-only  Flash UEFI firmware only (for network boot setup)

Arguments:
  device     SD card block device (e.g. /dev/sdX, /dev/mmcblk0)

Options:
  --skip-build    Don't rebuild the kernel (use existing binary)

Examples:
  sudo ./scripts/deploy-rpi-sdcard.sh full /dev/sdb
  sudo ./scripts/deploy-rpi-sdcard.sh update /dev/sdb
  sudo ./scripts/deploy-rpi-sdcard.sh uefi-only /dev/sdb
EOF
}

check_device() {
    local dev="$1"

    if [[ ! -b "$dev" ]]; then
        error "Not a block device: $dev"
        exit 1
    fi

    # Safety check — refuse to write to the system disk
    local root_disk
    root_disk=$(lsblk -ndo PKNAME "$(findmnt -n -o SOURCE /)" 2>/dev/null || echo "")
    if [[ -n "$root_disk" && "$dev" == "/dev/$root_disk" ]]; then
        error "REFUSING to write to system disk: $dev"
        error "This is the disk containing your root filesystem!"
        exit 1
    fi

    # Confirm with user
    echo ""
    warn "This will write to: $dev"
    lsblk "$dev" 2>/dev/null || true
    echo ""
    read -r -p "Are you sure? Type YES to continue: " confirm
    if [[ "$confirm" != "YES" ]]; then
        info "Aborted."
        exit 0
    fi
}

download_uefi_firmware() {
    mkdir -p "$CACHE_DIR"
    local zip_path="$CACHE_DIR/RPi4_UEFI_${RPI_UEFI_VERSION}.zip"
    local extract_dir="$CACHE_DIR/rpi-uefi-${RPI_UEFI_VERSION}"

    if [[ -d "$extract_dir" ]]; then
        info "Using cached UEFI firmware: $extract_dir"
        echo "$extract_dir"
        return
    fi

    if [[ ! -f "$zip_path" ]]; then
        info "Downloading RPi UEFI firmware ${RPI_UEFI_VERSION}..."
        curl -L -o "$zip_path" "$RPI_UEFI_URL"
    fi

    info "Extracting UEFI firmware..."
    mkdir -p "$extract_dir"
    unzip -o "$zip_path" -d "$extract_dir"

    echo "$extract_dir"
}

build_kernel() {
    cd "$PROJECT_DIR"
    info "Building SmallAIOS for $TARGET (release)..."
    # shellcheck disable=SC2086
    cargo build --release --target "$TARGET" -p "$PACKAGE" $BUILD_STD
}

create_full_sdcard() {
    local dev="$1"
    local skip_build="${2:-false}"

    check_device "$dev"

    if [[ $EUID -ne 0 ]]; then
        error "Full SD card creation requires root."
        exit 1
    fi

    # Build kernel
    if [[ "$skip_build" == "false" ]]; then
        build_kernel
    fi

    local kernel_path="$PROJECT_DIR/target/$TARGET/release/$KERNEL_NAME"
    if [[ ! -f "$kernel_path" ]]; then
        error "Kernel not found: $kernel_path"
        exit 1
    fi

    # Download UEFI firmware
    local uefi_dir
    uefi_dir=$(download_uefi_firmware)

    # Unmount any existing partitions
    umount "${dev}"* 2>/dev/null || true

    info "Partitioning $dev ..."
    # Create a single FAT32 partition (RPi boot partition)
    parted -s "$dev" \
        mklabel msdos \
        mkpart primary fat32 1MiB 512MiB \
        set 1 boot on

    # Wait for kernel to re-read partition table
    partprobe "$dev" 2>/dev/null || true
    sleep 2

    # Determine partition device name
    local part
    if [[ "$dev" == *"mmcblk"* || "$dev" == *"nvme"* ]]; then
        part="${dev}p1"
    else
        part="${dev}1"
    fi

    info "Formatting FAT32..."
    mkfs.vfat -F 32 -n SMALLAIOS "$part"

    # Mount and copy files
    local mnt
    mnt=$(mktemp -d)
    mount "$part" "$mnt"

    info "Copying UEFI firmware..."
    cp -r "$uefi_dir"/* "$mnt/"

    info "Copying SmallAIOS kernel..."
    cp "$kernel_path" "$mnt/$KERNEL_NAME"

    # Create UEFI startup script for automatic boot
    info "Creating startup.nsh..."
    cat > "$mnt/startup.nsh" <<'NSH'
@echo SmallAIOS Boot
fs0:\smallaios-aarch64
NSH

    # Enable UART in RPi config
    if [[ ! -f "$mnt/config.txt" ]]; then
        cat > "$mnt/config.txt" <<CFG
# SmallAIOS RPi configuration
enable_uart=1
arm_64bit=1
CFG
    else
        # Ensure UART is enabled in existing config
        if ! grep -q "enable_uart=1" "$mnt/config.txt"; then
            echo "enable_uart=1" >> "$mnt/config.txt"
        fi
    fi

    sync
    umount "$mnt"
    rmdir "$mnt"

    info "SD card ready!"
    info "Insert into RPi and power on."
    info "Serial console: make serial DEV=/dev/ttyUSB0"
}

update_kernel_only() {
    local dev="$1"
    local skip_build="${2:-false}"

    if [[ $EUID -ne 0 ]]; then
        error "Updating SD card requires root."
        exit 1
    fi

    # Build kernel
    if [[ "$skip_build" == "false" ]]; then
        build_kernel
    fi

    local kernel_path="$PROJECT_DIR/target/$TARGET/release/$KERNEL_NAME"
    if [[ ! -f "$kernel_path" ]]; then
        error "Kernel not found: $kernel_path"
        exit 1
    fi

    # Determine partition device name
    local part
    if [[ "$dev" == *"mmcblk"* || "$dev" == *"nvme"* ]]; then
        part="${dev}p1"
    else
        part="${dev}1"
    fi

    if [[ ! -b "$part" ]]; then
        error "Partition not found: $part"
        error "Use 'full' command to create the SD card first."
        exit 1
    fi

    local mnt
    mnt=$(mktemp -d)
    mount "$part" "$mnt"

    info "Updating kernel..."
    cp "$kernel_path" "$mnt/$KERNEL_NAME"

    sync
    umount "$mnt"
    rmdir "$mnt"

    local kernel_size
    kernel_size=$(stat -c%s "$kernel_path" 2>/dev/null || stat -f%z "$kernel_path")
    info "Updated! Kernel size: $(( kernel_size / 1024 )) KiB"
}

create_uefi_only() {
    local dev="$1"

    check_device "$dev"

    if [[ $EUID -ne 0 ]]; then
        error "SD card creation requires root."
        exit 1
    fi

    local uefi_dir
    uefi_dir=$(download_uefi_firmware)

    umount "${dev}"* 2>/dev/null || true

    info "Partitioning $dev (UEFI only)..."
    parted -s "$dev" \
        mklabel msdos \
        mkpart primary fat32 1MiB 256MiB \
        set 1 boot on

    partprobe "$dev" 2>/dev/null || true
    sleep 2

    local part
    if [[ "$dev" == *"mmcblk"* || "$dev" == *"nvme"* ]]; then
        part="${dev}p1"
    else
        part="${dev}1"
    fi

    mkfs.vfat -F 32 -n RPIUEFI "$part"

    local mnt
    mnt=$(mktemp -d)
    mount "$part" "$mnt"

    info "Copying UEFI firmware..."
    cp -r "$uefi_dir"/* "$mnt/"

    # Enable UART
    cat > "$mnt/config.txt" <<CFG
# RPi UEFI + SmallAIOS network boot
enable_uart=1
arm_64bit=1
CFG

    sync
    umount "$mnt"
    rmdir "$mnt"

    info "UEFI SD card ready!"
    info "Boot RPi, press ESC to enter UEFI setup, configure PXE boot."
}

# Parse arguments
COMMAND="${1:-}"
shift || true

SKIP_BUILD=false
DEVICE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        /dev/*) DEVICE="$1"; shift ;;
        *) error "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

case "$COMMAND" in
    full)
        [[ -z "$DEVICE" ]] && { error "Device required"; usage; exit 1; }
        create_full_sdcard "$DEVICE" "$SKIP_BUILD"
        ;;
    update)
        [[ -z "$DEVICE" ]] && { error "Device required"; usage; exit 1; }
        update_kernel_only "$DEVICE" "$SKIP_BUILD"
        ;;
    uefi-only)
        [[ -z "$DEVICE" ]] && { error "Device required"; usage; exit 1; }
        create_uefi_only "$DEVICE"
        ;;
    *)
        usage
        exit 1
        ;;
esac
