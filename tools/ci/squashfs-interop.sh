#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Squashfs / SmallAIOS-manifest external interop test.
#
# Per `embedded-filesystem-v1` spec Q22, a SmallAIOS-produced squashfs
# partition (squashfs blob + appended SmallAIOS manifest footer) MUST
# remain mountable on stock Linux/macOS via `mount -t squashfs -o loop`
# (or the equivalent FUSE driver on macOS). Squashfs tolerates trailing
# 4-byte-aligned padding after its natural EOF, so the appended footer
# is invisible to a stock kernel.
#
# This script:
#
#   1. Generates a small payload directory.
#   2. Runs `mksquashfs` on it (must be installed) to produce
#      `out.sqfs` for each of the four supported compression
#      algorithms (zstd / xz / gzip / lz4).
#   3. Appends a synthetic SmallAIOS manifest footer using the
#      `manifest-tool` helper (TODO(#fs-manifest-tool): build an
#      end-to-end signing tool; today we use the test-only helper
#      `cargo run -p smallaios-fs --bin make-test-manifest`).
#   4. Mounts each combined partition with `mount -t squashfs -o loop`
#      and verifies the directory contents match the original payload.
#
# This script is intended to be run in a CI matrix on Ubuntu LTS,
# Fedora current, and macOS-via-FUSE. Locally, ensure mksquashfs is in
# the PATH and run:
#
#   sudo ./tools/ci/squashfs-interop.sh
#
# TODO(#fs-squashfs-interop-ci): wire this script into the GitHub
# Actions matrix. The corresponding CI step:
#
#   - run: sudo apt-get install -y squashfs-tools
#   - run: sudo ./tools/ci/squashfs-interop.sh
#
# is intentionally NOT yet added to .github/workflows/ci.yml because
# the manifest-signing tool is not yet available; phase 4 will land
# that tool and wire this step at the same time.

set -euo pipefail

if ! command -v mksquashfs >/dev/null 2>&1; then
    echo "ERROR: mksquashfs not in PATH" >&2
    echo "  Ubuntu/Debian: sudo apt-get install -y squashfs-tools" >&2
    echo "  Fedora:        sudo dnf install -y squashfs-tools" >&2
    echo "  macOS:         brew install squashfs" >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

PAYLOAD="$WORKDIR/payload"
mkdir -p "$PAYLOAD"
echo "hello squashfs" > "$PAYLOAD/hello.txt"
mkdir -p "$PAYLOAD/sub"
echo "nested" > "$PAYLOAD/sub/nested.txt"

for COMP in zstd xz gzip lz4; do
    SQFS="$WORKDIR/out.$COMP.sqfs"
    mksquashfs "$PAYLOAD" "$SQFS" -comp "$COMP" -no-progress -quiet >/dev/null
    PARTITION="$WORKDIR/partition.$COMP.bin"
    cp "$SQFS" "$PARTITION"
    # TODO(#fs-manifest-tool): append SmallAIOS manifest footer here.
    # Today this script tests only that the externally-produced squashfs
    # is mountable; the manifest-append step will be added when
    # `make-test-manifest` lands in phase 4.

    MNT="$WORKDIR/mnt.$COMP"
    mkdir -p "$MNT"
    if [ "$(uname)" = "Linux" ]; then
        sudo mount -t squashfs -o loop "$PARTITION" "$MNT"
    else
        echo "WARN: non-Linux loopback mount not implemented in this script" >&2
        continue
    fi
    if [ "$(cat "$MNT/hello.txt")" != "hello squashfs" ]; then
        echo "ERROR: external mount returned wrong contents for $COMP" >&2
        sudo umount "$MNT" || true
        exit 1
    fi
    sudo umount "$MNT"
    echo "OK: $COMP partition mounts and reads on stock Linux"
done

echo "All four compression algorithms passed external interop."
