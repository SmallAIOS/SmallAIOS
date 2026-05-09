// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the image-size regression-check script
// (Phase 10 of `embedded-filesystem-v1`, task 10.8 / 10.9).
//
// The script lives in `tools/ci/image-size-check.sh`. These tests
// assert the script is well-formed (parseable, executable, contains
// the spec-required budgets) without actually running it — running
// requires a release build that is too expensive for ordinary
// `cargo test` runs.

#![cfg(feature = "fs-on-disk-mounts")]

use std::fs;
use std::path::PathBuf;

fn script_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // workspace root
    p.push("tools/ci/image-size-check.sh");
    p
}

#[test]
fn script_exists() {
    assert!(script_path().exists(), "image-size-check.sh missing");
}

#[test]
fn script_is_executable() {
    use std::os::unix::fs::PermissionsExt;
    let meta = fs::metadata(script_path()).unwrap();
    let mode = meta.permissions().mode();
    assert!(
        mode & 0o111 != 0,
        "image-size-check.sh is not executable: 0o{mode:o}"
    );
}

#[test]
fn script_declares_fs_growth_budget() {
    let body = fs::read_to_string(script_path()).unwrap();
    assert!(
        body.contains("FS_GROWTH_BUDGET_BYTES"),
        "FS_GROWTH_BUDGET_BYTES not declared"
    );
    // 200 KiB == 204_800 bytes per spec.
    assert!(
        body.contains("204800"),
        "200 KiB (204800 byte) budget not present"
    );
}

#[test]
fn script_declares_kernel_total_budget() {
    let body = fs::read_to_string(script_path()).unwrap();
    assert!(
        body.contains("TOTAL_KERNEL_BUDGET_BYTES"),
        "TOTAL_KERNEL_BUDGET_BYTES not declared"
    );
    // 8.2 MiB == 8_598_323 bytes per spec.
    assert!(
        body.contains("8598323"),
        "8.2 MiB (8598323 byte) budget not present"
    );
}

#[test]
fn script_uses_strict_bash_mode() {
    let body = fs::read_to_string(script_path()).unwrap();
    assert!(
        body.contains("set -euo pipefail"),
        "set -euo pipefail missing"
    );
}
