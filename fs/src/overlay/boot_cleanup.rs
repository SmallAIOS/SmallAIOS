// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Boot-time cleanup of orphan staging files in the overlay upper.
//!
//! Spec: `openspec/changes/embedded-overlay-v1/specs/fs-overlay-write/spec.md`
//! Requirement "Boot-cleanup pass: on mount, remove any orphan
//! `<name>.tmp` from `/data/models-upper/`".
//!
//! ## Why
//!
//! `OverlayWriter::add` runs stage-and-rename: it writes
//! `<name>.tmp`, fsyncs, then renames to `<name>`. A crash or power
//! cut between the open-tmp and the rename leaves a `.tmp` file
//! behind. The boot path SHALL sweep these so the upper layer's cap
//! accounting and `model list` views never see ghosts.
//!
//! ## How
//!
//! [`run_boot_cleanup`] iterates the upper-layer root, finds every
//! regular file ending in `.tmp`, and unlinks it. It also issues an
//! `fsync` if anything was unlinked so the cleanup is durable.
//!
//! Designed to be called from `fs::mount::mount_all` (or its
//! kernel-side caller) after the F2FS mount succeeds and before the
//! VFS exposes `/models/` to user-space syscalls.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::upper_layer::{UpperEntryKind, UpperError};
use super::write::{WritableUpperLayer, STAGING_SUFFIX};

/// Outcome of [`run_boot_cleanup`]. The caller (kernel boot path)
/// translates this into one audit record per orphan removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootCleanupReport {
    /// Names of every `.tmp` file that was unlinked. Suitable for
    /// "model_add_aborted_recovered" audit emission.
    pub removed: Vec<String>,
    /// Best-effort errors observed mid-walk. The cleanup keeps going
    /// when one entry fails so a single bad file can't strand others.
    pub errors: Vec<UpperError>,
}

impl BootCleanupReport {
    /// `true` iff at least one `.tmp` file was removed.
    pub fn any_removed(&self) -> bool {
        !self.removed.is_empty()
    }

    /// Number of `.tmp` files removed.
    pub fn count(&self) -> usize {
        self.removed.len()
    }
}

/// Sweep `<name>.tmp` orphan staging files from the upper layer's
/// root directory. Returns a [`BootCleanupReport`] of names removed +
/// any per-entry errors observed.
///
/// Called once at boot from `fs::mount::mount_all` after the F2FS
/// mount succeeds, BEFORE the overlay is exposed via VFS.
pub fn run_boot_cleanup<U: WritableUpperLayer>(upper: &mut U) -> BootCleanupReport {
    let mut report = BootCleanupReport {
        removed: Vec::new(),
        errors: Vec::new(),
    };
    let entries = match upper.iter_dir("") {
        Ok(e) => e,
        Err(UpperError::NotFound) => return report,
        Err(e) => {
            report.errors.push(e);
            return report;
        }
    };
    let names: Vec<String> = entries
        .into_iter()
        .filter(|e| matches!(e.kind, UpperEntryKind::RegularFile))
        .map(|e| e.name)
        .filter(|n| n.ends_with(STAGING_SUFFIX))
        .collect();
    for name in names {
        match upper.unlink(&name) {
            Ok(()) => report.removed.push(name),
            Err(e) => report.errors.push(e),
        }
    }
    if !report.removed.is_empty() {
        if let Err(e) = upper.fsync() {
            report.errors.push(e);
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::write::MockWritableUpperLayer;

    #[test]
    fn empty_upper_runs_clean() {
        let mut u = MockWritableUpperLayer::new();
        let r = run_boot_cleanup(&mut u);
        assert!(!r.any_removed());
        assert!(r.errors.is_empty());
    }

    #[test]
    fn single_orphan_tmp_removed() {
        let mut u = MockWritableUpperLayer::new();
        u.add_file("foo.onnx.tmp", b"junk");
        u.add_file("kept.onnx", b"keep");
        let r = run_boot_cleanup(&mut u);
        assert_eq!(r.count(), 1);
        assert_eq!(r.removed[0], "foo.onnx.tmp");
        assert!(!u.exists("foo.onnx.tmp"));
        assert!(u.exists("kept.onnx"));
    }

    #[test]
    fn multiple_orphans_removed() {
        let mut u = MockWritableUpperLayer::new();
        u.add_file("a.tmp", b"x");
        u.add_file("b.tmp", b"y");
        u.add_file("c.onnx", b"z");
        let r = run_boot_cleanup(&mut u);
        assert_eq!(r.count(), 2);
        assert!(!u.exists("a.tmp"));
        assert!(!u.exists("b.tmp"));
        assert!(u.exists("c.onnx"));
    }

    #[test]
    fn nontmp_files_preserved() {
        let mut u = MockWritableUpperLayer::new();
        u.add_file("foo.onnx", b"keep");
        u.add_file("foo.onnx.sha3", b"keep");
        u.add_file("foo.onnx.sig", b"keep");
        u.add_file("bar.whiteout", b"");
        let r = run_boot_cleanup(&mut u);
        assert!(!r.any_removed());
        assert!(u.exists("foo.onnx"));
        assert!(u.exists("foo.onnx.sha3"));
        assert!(u.exists("foo.onnx.sig"));
    }

    #[test]
    fn report_count_matches_removed_len() {
        let mut u = MockWritableUpperLayer::new();
        u.add_file("a.tmp", b"x");
        u.add_file("b.tmp", b"y");
        u.add_file("c.tmp", b"z");
        let r = run_boot_cleanup(&mut u);
        assert_eq!(r.count(), r.removed.len());
        assert_eq!(r.count(), 3);
    }

    #[test]
    fn idempotent_double_run() {
        let mut u = MockWritableUpperLayer::new();
        u.add_file("a.tmp", b"x");
        let r1 = run_boot_cleanup(&mut u);
        assert_eq!(r1.count(), 1);
        let r2 = run_boot_cleanup(&mut u);
        assert_eq!(r2.count(), 0);
    }
}
