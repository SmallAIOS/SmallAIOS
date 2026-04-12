// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Fixture-based smoke test for `smallaios-coverage-probe`.
//!
//! This test is `#[ignore]`'d by default because the ONNX fixtures
//! (~2 GB total) are not committed to the repository. To run it
//! locally, first download the fixtures per the instructions in
//! `tools/coverage-probe/REPORT.md`, then invoke:
//!
//! ```bash
//! cargo test -p smallaios-coverage-probe --test fixture_smoke -- --ignored
//! ```
//!
//! The test walks `tests/fixtures/onnx-models/` (relative to the
//! repo root), runs the probe against every `*.onnx` file found,
//! and asserts that the verdict is not `HasUnrecognizedOps`. Models
//! with known out-of-roadmap ops (currently only `llama-3.2-1b.onnx`
//! pending Phase 2 fused-op amendments) are listed in `KNOWN_GAPS`
//! and checked for a *matching* gap rather than failing the run.

use smallaios_coverage_probe::{walk_bytes, Inventory, Verdict};
use std::path::{Path, PathBuf};

/// Fixtures that are expected to surface unrecognized ops because
/// the Phase 2 plan does not yet cover them. See
/// `tools/coverage-probe/REPORT.md` for the full gap analysis.
const KNOWN_GAPS: &[&str] = &["llama-3.2-1b.onnx"];

fn repo_root() -> PathBuf {
    // Walk up from CARGO_MANIFEST_DIR (tools/coverage-probe) to the
    // workspace root that contains `tests/fixtures/onnx-models/`.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or(manifest)
}

fn fixtures_dir() -> PathBuf {
    repo_root()
        .join("tests")
        .join("fixtures")
        .join("onnx-models")
}

fn inventory_path() -> PathBuf {
    repo_root().join("docs").join("onnx-coverage-roadmap.md")
}

#[test]
#[ignore = "requires tests/fixtures/onnx-models/ to be populated; \
            see tools/coverage-probe/REPORT.md"]
fn every_fixture_loads_at_least_after_known_phase() {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!("fixtures dir {} does not exist; skipping", dir.display());
        return;
    }

    let inventory = Inventory::from_roadmap_file(&inventory_path())
        .unwrap_or_else(|_| Inventory::from_roadmap_str(""));

    let mut checked = 0usize;
    for entry in std::fs::read_dir(&dir).expect("read fixtures dir") {
        let entry = entry.expect("read dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("onnx") {
            continue;
        }
        checked += 1;

        let label = path.display().to_string();
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {label}: {e}"));
        let report =
            walk_bytes(&bytes, &label, &inventory).unwrap_or_else(|e| panic!("walk {label}: {e}"));

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let expect_gap = KNOWN_GAPS.iter().any(|k| *k == filename);

        match report.verdict {
            Verdict::LoadableToday
            | Verdict::LoadableAfterPhase1
            | Verdict::LoadableAfterPhase2
            | Verdict::LoadableAfterPhase3
            | Verdict::LoadableAfterPhase4 => {
                assert!(
                    !expect_gap,
                    "{filename}: expected a known-gap (unrecognized) \
                     verdict but got {:?}; please update KNOWN_GAPS \
                     in tests/fixture_smoke.rs",
                    report.verdict
                );
            }
            Verdict::NotLoadable | Verdict::HasUnrecognizedOps => {
                assert!(
                    expect_gap,
                    "{filename}: verdict {:?} is unexpected — this \
                     fixture introduces a new gap not listed in \
                     tools/coverage-probe/REPORT.md. Counts: {:?}",
                    report.verdict, report.counts
                );
            }
        }
    }

    assert!(
        checked > 0,
        "no .onnx fixtures found in {}; please populate per \
         tools/coverage-probe/REPORT.md",
        dir.display()
    );
}

#[test]
fn fixtures_dir_path_resolves_even_when_missing() {
    // Guard test — ensure the path computation is sane even when
    // the directory itself is absent (CI default).
    let p: &Path = &fixtures_dir();
    assert!(p.ends_with("tests/fixtures/onnx-models"));
}
