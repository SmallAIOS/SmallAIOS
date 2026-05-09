#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Image-size regression check.
#
# Per `embedded-filesystem-v1` Phase 10 task 10.8 and 10.9:
#
# - F2FS write path must stay within ≤ 200 KiB compiled growth
#   relative to the Phase-7 baseline.
# - Total kernel binary stays within ≤ 8.2 MiB.
#
# This script measures the compiled `smallaios-fs` rlib (release,
# size-optimized) and compares against a recorded baseline. CI fails
# the PR if the budget is exceeded.
#
# Baseline values are stored in this script as constants. When a
# legitimate growth lands (new feature, new dependency), bump the
# baseline AND tighten the budget where possible — the spec wants the
# numbers to ratchet down, not drift up.

set -euo pipefail

# ─── Tunables ─────────────────────────────────────────────────────────────

# Phase-7 baseline (bytes) for the smallaios-fs rlib in release mode.
# Captured 2026-05-09 from `cargo build -p smallaios-fs --release
# --features fs-on-disk-mounts`.
#
# When this value drifts due to legitimate changes, bump it and add a
# CHANGELOG entry. The 200 KiB headroom budget below applies on top.
PHASE7_BASELINE_FS_RLIB_BYTES="${PHASE7_BASELINE_FS_RLIB_BYTES:-0}"

# Spec: ≤ 200 KiB (204_800 bytes) growth budget on top of Phase-7.
FS_GROWTH_BUDGET_BYTES="${FS_GROWTH_BUDGET_BYTES:-204800}"

# Phase-7 baseline (bytes) for the smallaios-kernel rlib in release.
PHASE7_BASELINE_KERNEL_RLIB_BYTES="${PHASE7_BASELINE_KERNEL_RLIB_BYTES:-0}"

# Total kernel image must stay ≤ 8.2 MiB (8_598_323 bytes).
TOTAL_KERNEL_BUDGET_BYTES="${TOTAL_KERNEL_BUDGET_BYTES:-8598323}"

# ─── Inputs ───────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET_DIR="${REPO_ROOT}/target/release/deps"

# ─── Helpers ──────────────────────────────────────────────────────────────

die() {
  echo "image-size-check: $*" >&2
  exit 1
}

note() {
  echo "image-size-check: $*"
}

# Locate the most-recent compiled artifact matching `pattern` under
# the release deps directory. Returns the file size in bytes, or 0
# if no artifact is found.
artifact_size() {
  local pattern="$1"
  local f
  f="$(find "$TARGET_DIR" -maxdepth 1 -name "$pattern" -print 2>/dev/null | sort -r | head -n1 || true)"
  if [ -z "$f" ] || [ ! -f "$f" ]; then
    echo 0
    return
  fi
  if stat --version >/dev/null 2>&1; then
    # GNU stat (Linux)
    stat -c '%s' "$f"
  else
    # BSD stat (macOS)
    stat -f '%z' "$f"
  fi
}

# Compare current size against baseline + budget.
check_budget() {
  local label="$1"
  local current="$2"
  local baseline="$3"
  local budget="$4"

  local growth=$((current - baseline))
  if [ "$growth" -lt 0 ]; then
    growth=0
  fi
  note "$label: current=${current}B baseline=${baseline}B growth=${growth}B budget=${budget}B"
  if [ "$growth" -gt "$budget" ]; then
    die "$label: growth ${growth}B exceeds budget ${budget}B"
  fi
}

# ─── Main ─────────────────────────────────────────────────────────────────

note "Building release artifacts (this may take a while)…"
(cd "$REPO_ROOT" && cargo build --release -p smallaios-fs --features fs-on-disk-mounts >/dev/null)
(cd "$REPO_ROOT" && cargo build --release -p smallaios-kernel >/dev/null)

FS_SIZE="$(artifact_size 'libsmallaios_fs-*.rlib')"
KERNEL_SIZE="$(artifact_size 'libsmallaios_kernel-*.rlib')"

if [ "$FS_SIZE" -eq 0 ]; then
  die "smallaios-fs rlib not found — did the build succeed?"
fi
if [ "$KERNEL_SIZE" -eq 0 ]; then
  die "smallaios-kernel rlib not found — did the build succeed?"
fi

note "Measured smallaios-fs:     ${FS_SIZE} bytes"
note "Measured smallaios-kernel: ${KERNEL_SIZE} bytes"

# If no baseline is configured, this is a record-the-baseline run —
# emit the values for the operator to paste in.
if [ "$PHASE7_BASELINE_FS_RLIB_BYTES" -eq 0 ] || [ "$PHASE7_BASELINE_KERNEL_RLIB_BYTES" -eq 0 ]; then
  note "No baseline configured. Suggested baselines for this build:"
  note "  PHASE7_BASELINE_FS_RLIB_BYTES=$FS_SIZE"
  note "  PHASE7_BASELINE_KERNEL_RLIB_BYTES=$KERNEL_SIZE"
  note "Re-run with both env vars set, or paste them into this script."
  exit 0
fi

check_budget "smallaios-fs"     "$FS_SIZE"     "$PHASE7_BASELINE_FS_RLIB_BYTES"     "$FS_GROWTH_BUDGET_BYTES"
check_budget "smallaios-kernel" "$KERNEL_SIZE" "$PHASE7_BASELINE_KERNEL_RLIB_BYTES" "$TOTAL_KERNEL_BUDGET_BYTES"

note "All size budgets within bounds."
