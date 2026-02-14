#!/bin/bash
# SmallAIOS VMware x86-64 Image Builder
# Creates a bootable VMDK from the bare-metal kernel
# Requires: grub-install, sgdisk, mkfs.ext4, qemu-img, losetup
# Usage: ./scripts/make-vmware-x86.sh [kernel-path]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${REPO_DIR}/build"
KERNEL="${1:-${REPO_DIR}/target/x86_64-unknown-none/release/smallaios-x86_64}"
RAW_IMG="${BUILD_DIR}/smallaios-x86.img"
VMDK="${BUILD_DIR}/smallaios-x86.vmdk"
VMX="${BUILD_DIR}/smallaios-x86.vmx"
IMG_SIZE_MB=64

# ─── Check required tools ───────────────────────────────────────────────────

check_tool() {
    local tool="$1"
    local ubuntu_pkg="$2"
    local fedora_pkg="$3"
    if ! command -v "$tool" &>/dev/null; then
        echo "ERROR: '$tool' not found."
        echo "  Ubuntu/Debian: sudo apt-get install $ubuntu_pkg"
        echo "  Fedora/RHEL:   sudo dnf install $fedora_pkg"
        return 1
    fi
}

MISSING=0
check_tool grub-install grub-pc-bin grub2-pc || MISSING=1
check_tool sgdisk gdisk gdisk || MISSING=1
check_tool mkfs.ext4 e2fsprogs e2fsprogs || MISSING=1
check_tool qemu-img qemu-utils qemu-img || MISSING=1
check_tool losetup mount util-linux || MISSING=1

if [ "$MISSING" -ne 0 ]; then
    echo "Install missing tools and retry."
    exit 1
fi

if [ ! -f "$KERNEL" ]; then
    echo "ERROR: Kernel not found at $KERNEL"
    echo "Run 'make build-kernel-x86' first."
    exit 1
fi

mkdir -p "$BUILD_DIR"

# ─── Create raw disk image ──────────────────────────────────────────────────

echo "[vmware] Creating ${IMG_SIZE_MB} MB raw disk image..."
dd if=/dev/zero of="$RAW_IMG" bs=1M count=0 seek="$IMG_SIZE_MB" status=none

echo "[vmware] Creating GPT partition table..."
sgdisk --zap-all "$RAW_IMG" >/dev/null
sgdisk --new=1:2048:4095 --typecode=1:ef02 "$RAW_IMG" >/dev/null  # 1 MB BIOS boot
sgdisk --new=2:4096:0 --typecode=2:8300 "$RAW_IMG" >/dev/null     # ext4 data
sgdisk --print "$RAW_IMG"

# ─── Setup loop device and format ───────────────────────────────────────────

echo "[vmware] Setting up loop device..."
LOOP=$(sudo losetup --find --show --partscan "$RAW_IMG")
echo "[vmware] Loop device: $LOOP"

# Wait for partition devices to appear
sleep 1

PART2="${LOOP}p2"
if [ ! -b "$PART2" ]; then
    sudo partprobe "$LOOP" 2>/dev/null || true
    sleep 1
fi

echo "[vmware] Formatting ext4 partition..."
sudo mkfs.ext4 -q -L SmallAIOS "$PART2"

# ─── Mount and install GRUB + kernel ────────────────────────────────────────

MOUNT_DIR=$(mktemp -d)
echo "[vmware] Mounting at $MOUNT_DIR..."
sudo mount "$PART2" "$MOUNT_DIR"

sudo mkdir -p "$MOUNT_DIR/boot/grub"
sudo cp "$KERNEL" "$MOUNT_DIR/boot/smallaios-x86_64"

echo "[vmware] Writing GRUB configuration..."
sudo tee "$MOUNT_DIR/boot/grub/grub.cfg" >/dev/null <<'GRUBEOF'
set timeout=3
set default=0

menuentry "SmallAIOS" {
    multiboot2 /boot/smallaios-x86_64
    boot
}
GRUBEOF

echo "[vmware] Installing GRUB bootloader..."
# Detect grub-install command name (Ubuntu vs Fedora)
GRUB_CMD="grub-install"
if ! command -v grub-install &>/dev/null; then
    GRUB_CMD="grub2-install"
fi
sudo "$GRUB_CMD" --target=i386-pc --boot-directory="$MOUNT_DIR/boot" "$LOOP"

# ─── Cleanup mount ──────────────────────────────────────────────────────────

echo "[vmware] Unmounting..."
sudo umount "$MOUNT_DIR"
rmdir "$MOUNT_DIR"
sudo losetup --detach "$LOOP"

# ─── Convert to VMDK ────────────────────────────────────────────────────────

echo "[vmware] Converting to VMDK..."
qemu-img convert -O vmdk "$RAW_IMG" "$VMDK"
rm -f "$RAW_IMG"

# ─── Generate VMX ────────────────────────────────────────────────────────────

echo "[vmware] Generating VMX configuration..."
VMDK_BASENAME=$(basename "$VMDK")
sed "s|__VMDK_PATH__|${VMDK_BASENAME}|g" "${SCRIPT_DIR}/vmware-template.vmx" > "$VMX"

# ─── Summary ────────────────────────────────────────────────────────────────

VMDK_SIZE=$(stat --format=%s "$VMDK" 2>/dev/null || stat -f%z "$VMDK")
echo ""
echo "=== VMware image created ==="
echo "  VMDK: $VMDK ($(( VMDK_SIZE / 1024 )) KB)"
echo "  VMX:  $VMX"
echo ""
echo "Open in VMware: vmrun start $VMX"
echo "Or: File → Open → $VMX"
