#!/bin/bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build a bootable SD card image for Jetson Nano.
#
# The image contains a GPT partition table with a single ext4 partition
# holding /boot/Image (kernel), /boot/tegra210-smallaios.dtb (DTB),
# and /boot/extlinux/extlinux.conf (U-Boot boot config).
#
# Usage:
#   ./scripts/make-sdcard-jetson.sh
#
# Output:
#   build/sdcard-jetson.img (64 MB sparse file)
#
# To flash to a real SD card:
#   sudo dd if=build/sdcard-jetson.img of=/dev/sdX bs=4M status=progress
#   sync

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_FILE="$PROJECT_DIR/build/sdcard-jetson.img"
IMAGE_SIZE_MB=64
KERNEL_IMAGE="$PROJECT_DIR/target/aarch64-unknown-none/release/Image"
DTB_FILE="$PROJECT_DIR/arch/aarch64/dtb/tegra210-smallaios.dtb"
EXTLINUX_CONF="$PROJECT_DIR/arch/aarch64/extlinux.conf"
MOUNT_DIR=""

cleanup() {
    if [ -n "$MOUNT_DIR" ] && mountpoint -q "$MOUNT_DIR" 2>/dev/null; then
        sudo umount "$MOUNT_DIR"
    fi
    if [ -n "$MOUNT_DIR" ] && [ -d "$MOUNT_DIR" ]; then
        rmdir "$MOUNT_DIR"
    fi
    # Detach loop device if set
    if [ -n "${LOOP_DEV:-}" ]; then
        sudo losetup -d "$LOOP_DEV" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Verify kernel image exists
if [ ! -f "$KERNEL_IMAGE" ]; then
    echo "ERROR: Kernel Image not found at $KERNEL_IMAGE"
    echo "Run 'make build-kernel-jetson' first."
    exit 1
fi

# Verify DTB exists
if [ ! -f "$DTB_FILE" ]; then
    echo "WARNING: DTB not found at $DTB_FILE"
    echo "The SD card will boot without a DTB — U-Boot may supply one from its own storage."
    HAS_DTB=0
else
    HAS_DTB=1
fi

# Verify extlinux.conf exists
if [ ! -f "$EXTLINUX_CONF" ]; then
    echo "ERROR: extlinux.conf not found at $EXTLINUX_CONF"
    exit 1
fi

# Create output directory
mkdir -p "$PROJECT_DIR/build"

echo "=== Building Jetson Nano SD card image ==="

# Create sparse image file
echo "Creating ${IMAGE_SIZE_MB} MB sparse image..."
dd if=/dev/zero of="$IMAGE_FILE" bs=1M count=0 seek="$IMAGE_SIZE_MB" 2>/dev/null

# Create GPT partition table with single ext4 partition
echo "Creating GPT partition table..."
sgdisk --clear \
    --new=1:2048:0 --typecode=1:8300 --change-name=1:"SMALLAIOS" \
    "$IMAGE_FILE"

# Set up loop device
LOOP_DEV=$(sudo losetup --find --show --partscan "$IMAGE_FILE")
PART_DEV="${LOOP_DEV}p1"

# Wait for partition device to appear
sleep 1
if [ ! -b "$PART_DEV" ]; then
    sudo partprobe "$LOOP_DEV"
    sleep 1
fi

# Format ext4
echo "Formatting ext4..."
sudo mkfs.ext4 -q -L SMALLAIOS "$PART_DEV"

# Mount and copy files
MOUNT_DIR=$(mktemp -d)
sudo mount "$PART_DEV" "$MOUNT_DIR"

echo "Copying boot files..."
sudo mkdir -p "$MOUNT_DIR/boot/extlinux"
sudo cp "$KERNEL_IMAGE" "$MOUNT_DIR/boot/Image"

if [ "$HAS_DTB" -eq 1 ]; then
    sudo cp "$DTB_FILE" "$MOUNT_DIR/boot/tegra210-smallaios.dtb"
fi

sudo cp "$EXTLINUX_CONF" "$MOUNT_DIR/boot/extlinux/extlinux.conf"

# Show contents
echo ""
echo "Boot partition contents:"
ls -la "$MOUNT_DIR/boot/"
ls -la "$MOUNT_DIR/boot/extlinux/"

# Unmount
sudo umount "$MOUNT_DIR"
rmdir "$MOUNT_DIR"
MOUNT_DIR=""

# Detach loop device
sudo losetup -d "$LOOP_DEV"
LOOP_DEV=""

# Size check
IMAGE_BYTES=$(stat --format=%s "$IMAGE_FILE" 2>/dev/null || stat -f%z "$IMAGE_FILE")
MAX_BYTES=$((IMAGE_SIZE_MB * 1024 * 1024))
echo ""
echo "Image: $IMAGE_FILE"
echo "Size: $IMAGE_BYTES bytes ($((IMAGE_BYTES / 1024 / 1024)) MB)"

if [ "$IMAGE_BYTES" -gt "$MAX_BYTES" ]; then
    echo "ERROR: Image exceeds ${IMAGE_SIZE_MB} MB limit!"
    exit 1
fi

echo ""
echo "=== SD card image ready ==="
echo "Flash with: sudo dd if=$IMAGE_FILE of=/dev/sdX bs=4M status=progress && sync"
