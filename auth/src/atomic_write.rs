// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Atomic shadow-file rewrite — `stage → fsync → rename`.
//!
//! Spec: `openspec/changes/management-login-v1/specs/auth-shadow/spec.md`
//! Requirement "Atomic shadow rewrite".
//!
//! Every write to `/data/auth/shadow` SHALL stage to `<path>.tmp`, `fsync`
//! the staged file, then `rename` atomically over the target. The rename is
//! the only operation visible to other readers; partial writes SHALL never
//! appear at the canonical path. On crash mid-rename, the next boot SHALL
//! see the original `<path>` intact and the orphan `<path>.tmp` cleared
//! before any login is permitted.
//!
//! ## Status
//!
//! This module defines the [`AtomicWriter`] trait and ships two
//! implementations:
//!
//! - [`TestAtomicWriter`] — in-memory, used by unit tests in this crate
//!   and downstream callers (kernel session-table tests, console-login).
//!   It models the staging/rename machinery so the contract is exercised.
//! - [`KernelAtomicWriter`] — production stub. Returns
//!   [`AtomicWriteError::NotImplemented`] until `embedded-filesystem-v1`
//!   Phase 7 lands the F2FS read-write path.
//!
//! When F2FS RW arrives, the kernel impl will be wired through to a
//! `vfs::write_atomic` syscall. The trait surface is stable now so the
//! shadow-file persistence path (Phases 3-4) can ship against the test
//! impl and be re-targeted by feature-flag in Phase 7.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

/// Errors raised by [`AtomicWriter`] implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomicWriteError {
    /// The production [`KernelAtomicWriter`] is not yet wired through to a
    /// real F2FS read-write path. Returned until `embedded-filesystem-v1`
    /// Phase 7 ships.
    NotImplemented,
    /// Backing storage I/O error (test impl uses this for crash-fault
    /// injection).
    Io(&'static str),
    /// The path is empty or otherwise unusable.
    InvalidPath,
}

impl fmt::Display for AtomicWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "auth: KernelAtomicWriter requires embedded-filesystem-v1 Phase 7"
            ),
            Self::Io(s) => write!(f, "auth: atomic-write I/O error: {s}"),
            Self::InvalidPath => write!(f, "auth: atomic-write invalid path"),
        }
    }
}

/// Atomic file rewrite contract for the shadow file (and other
/// crash-safety-critical management state).
///
/// Implementations SHALL:
///
/// 1. Stage the new contents to a sibling path derived from `path` — by
///    convention, `<path>.tmp`.
/// 2. Flush staged contents to durable storage (`fsync` semantics).
/// 3. Atomically rename the staged file over `path`.
///
/// Readers observing `path` SHALL never see a partially-written record.
pub trait AtomicWriter {
    /// Atomically replace the file at `path` with `contents`. On error the
    /// file at `path` SHALL be left untouched.
    fn write_atomic(&self, path: &str, contents: &[u8]) -> Result<(), AtomicWriteError>;

    /// Remove any orphan `<path>.tmp` siblings under `parent_dir` left over
    /// from an interrupted rewrite. Called once at boot before any login is
    /// permitted, per the "Crash mid-rename leaves shadow intact" scenario.
    fn cleanup_orphan_tmp(&self, parent_dir: &str) -> Result<(), AtomicWriteError>;
}

// ─── Test implementation ─────────────────────────────────────────────────────

/// In-memory [`AtomicWriter`] that simulates the `stage → fsync → rename`
/// machinery. Used by unit tests in this crate and by downstream tests in
/// `kernel/`, `peripheral/`, and `mgmt/`.
///
/// The simulator separately tracks `path` (the canonical file) and
/// `<path>.tmp` (the staging file). The rename step swaps them in a single
/// atomic step that other readers cannot interrupt.
///
/// Crash-injection helpers ([`TestAtomicWriter::inject_crash_after_stage`])
/// model a power-fail between fsync and rename, leaving an orphan tmp file
/// to be cleaned up by [`AtomicWriter::cleanup_orphan_tmp`].
pub struct TestAtomicWriter {
    inner: RefCell<TestAtomicWriterInner>,
}

struct TestAtomicWriterInner {
    files: BTreeMap<String, Vec<u8>>,
    crash_after_stage: bool,
    crash_during_io: bool,
}

impl Default for TestAtomicWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TestAtomicWriter {
    /// Construct an empty in-memory writer.
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(TestAtomicWriterInner {
                files: BTreeMap::new(),
                crash_after_stage: false,
                crash_during_io: false,
            }),
        }
    }

    /// Install a pre-existing file (e.g., a "previous-boot" shadow) at
    /// `path` for tests.
    pub fn seed(&self, path: &str, contents: &[u8]) {
        let mut g = self.inner.borrow_mut();
        g.files.insert(path.to_string(), contents.to_vec());
    }

    /// Read the current contents at `path` — for assertion.
    pub fn read(&self, path: &str) -> Option<Vec<u8>> {
        self.inner.borrow().files.get(path).cloned()
    }

    /// Return `true` iff a staged `<path>.tmp` sibling exists. Used by
    /// crash-recovery tests.
    pub fn has_staged(&self, path: &str) -> bool {
        let tmp = staging_path(path);
        self.inner.borrow().files.contains_key(&tmp)
    }

    /// Inject a "crash" after the stage+fsync but before the rename. The
    /// next [`AtomicWriter::write_atomic`] call SHALL leave `<path>.tmp`
    /// populated and `<path>` untouched, then return
    /// [`AtomicWriteError::Io`].
    pub fn inject_crash_after_stage(&self) {
        self.inner.borrow_mut().crash_after_stage = true;
    }

    /// Inject a "crash" during the staging write. The next write SHALL
    /// fail before any rename; the staged tmp may be partially written
    /// (test simulates this as removing both staged and target).
    pub fn inject_crash_during_io(&self) {
        self.inner.borrow_mut().crash_during_io = true;
    }
}

impl AtomicWriter for TestAtomicWriter {
    fn write_atomic(&self, path: &str, contents: &[u8]) -> Result<(), AtomicWriteError> {
        if path.is_empty() {
            return Err(AtomicWriteError::InvalidPath);
        }

        let mut g = self.inner.borrow_mut();

        if g.crash_during_io {
            // Simulate a crash *during* the stage write — the staged tmp
            // is half-formed, target is untouched.
            g.crash_during_io = false;
            return Err(AtomicWriteError::Io("crash during stage"));
        }

        // Step 1: stage to `<path>.tmp`.
        let tmp = staging_path(path);
        g.files.insert(tmp.clone(), contents.to_vec());

        if g.crash_after_stage {
            // Simulate a power-fail between fsync and rename.
            g.crash_after_stage = false;
            return Err(AtomicWriteError::Io("crash after stage"));
        }

        // Step 2 (implicit): fsync — modeled by the in-memory commit above.
        // Step 3: atomic rename.
        if let Some(staged) = g.files.remove(&tmp) {
            g.files.insert(path.to_string(), staged);
        }
        Ok(())
    }

    fn cleanup_orphan_tmp(&self, parent_dir: &str) -> Result<(), AtomicWriteError> {
        let mut g = self.inner.borrow_mut();
        let prefix = if parent_dir.is_empty() {
            String::new()
        } else if parent_dir.ends_with('/') {
            parent_dir.to_string()
        } else {
            // dir + '/'
            let mut p = String::from(parent_dir);
            p.push('/');
            p
        };
        let to_remove: Vec<String> = g
            .files
            .keys()
            .filter(|k| k.starts_with(&prefix) && k.ends_with(".tmp"))
            .cloned()
            .collect();
        for k in to_remove {
            g.files.remove(&k);
        }
        Ok(())
    }
}

/// Compute the staging path for `<path>` per the rewrite convention.
fn staging_path(path: &str) -> String {
    let mut tmp = String::from(path);
    tmp.push_str(".tmp");
    tmp
}

// ─── Production stub ─────────────────────────────────────────────────────────

/// Production [`AtomicWriter`] — currently a stub. Returns
/// [`AtomicWriteError::NotImplemented`] until `embedded-filesystem-v1`
/// Phase 7 lands the F2FS read-write VFS path.
///
/// The constructor is `const fn` so the type can be embedded in static
/// contexts without paying allocation up front.
pub struct KernelAtomicWriter {
    _private: (),
}

impl Default for KernelAtomicWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelAtomicWriter {
    /// Construct the production writer. Wire-up to F2FS lands in
    /// `embedded-filesystem-v1` Phase 7.
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

impl AtomicWriter for KernelAtomicWriter {
    fn write_atomic(&self, _path: &str, _contents: &[u8]) -> Result<(), AtomicWriteError> {
        Err(AtomicWriteError::NotImplemented)
    }

    fn cleanup_orphan_tmp(&self, _parent_dir: &str) -> Result<(), AtomicWriteError> {
        Err(AtomicWriteError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHADOW_PATH: &str = "/data/auth/shadow";

    #[test]
    fn round_trip_replaces_target() {
        let w = TestAtomicWriter::new();
        w.write_atomic(SHADOW_PATH, b"hello").unwrap();
        assert_eq!(w.read(SHADOW_PATH).as_deref(), Some(&b"hello"[..]));

        w.write_atomic(SHADOW_PATH, b"world").unwrap();
        assert_eq!(w.read(SHADOW_PATH).as_deref(), Some(&b"world"[..]));

        // No leftover staged tmp visible on success.
        assert!(!w.has_staged(SHADOW_PATH));
    }

    #[test]
    fn rejects_empty_path() {
        let w = TestAtomicWriter::new();
        assert_eq!(w.write_atomic("", b"x"), Err(AtomicWriteError::InvalidPath));
    }

    #[test]
    fn crash_after_stage_leaves_target_intact_and_staged_visible() {
        // Spec: "Crash mid-rename leaves shadow intact"
        let w = TestAtomicWriter::new();
        w.seed(SHADOW_PATH, b"original");

        w.inject_crash_after_stage();
        let result = w.write_atomic(SHADOW_PATH, b"new-contents");
        assert!(matches!(result, Err(AtomicWriteError::Io(_))));

        // Target untouched.
        assert_eq!(
            w.read(SHADOW_PATH).as_deref(),
            Some(&b"original"[..]),
            "target file SHALL be untouched after crash mid-rename"
        );

        // Orphan tmp visible until cleanup.
        assert!(
            w.has_staged(SHADOW_PATH),
            "staged tmp SHALL remain after crash"
        );

        // Boot-time cleanup removes it.
        w.cleanup_orphan_tmp("/data/auth").unwrap();
        assert!(
            !w.has_staged(SHADOW_PATH),
            "cleanup_orphan_tmp SHALL remove orphan staged file"
        );

        // Target is still the original.
        assert_eq!(w.read(SHADOW_PATH).as_deref(), Some(&b"original"[..]));
    }

    #[test]
    fn crash_during_io_leaves_target_and_staged_clean() {
        let w = TestAtomicWriter::new();
        w.seed(SHADOW_PATH, b"original");

        w.inject_crash_during_io();
        let result = w.write_atomic(SHADOW_PATH, b"new-contents");
        assert!(matches!(result, Err(AtomicWriteError::Io(_))));

        // Original is intact, no staged file was created.
        assert_eq!(w.read(SHADOW_PATH).as_deref(), Some(&b"original"[..]));
        assert!(!w.has_staged(SHADOW_PATH));
    }

    #[test]
    fn cleanup_orphan_tmp_is_idempotent() {
        let w = TestAtomicWriter::new();
        w.cleanup_orphan_tmp("/data/auth").unwrap();
        w.cleanup_orphan_tmp("/data/auth").unwrap();
    }

    #[test]
    fn cleanup_orphan_tmp_only_removes_tmp_siblings() {
        let w = TestAtomicWriter::new();
        w.seed("/data/auth/shadow", b"keep me");
        w.seed("/data/auth/shadow.tmp", b"orphan");
        w.seed("/data/auth/policy.toml", b"keep me too");

        w.cleanup_orphan_tmp("/data/auth").unwrap();

        assert_eq!(
            w.read("/data/auth/shadow").as_deref(),
            Some(&b"keep me"[..])
        );
        assert_eq!(
            w.read("/data/auth/policy.toml").as_deref(),
            Some(&b"keep me too"[..])
        );
        assert!(w.read("/data/auth/shadow.tmp").is_none());
    }

    #[test]
    fn kernel_writer_is_not_implemented() {
        let w = KernelAtomicWriter::new();
        assert_eq!(
            w.write_atomic(SHADOW_PATH, b"x"),
            Err(AtomicWriteError::NotImplemented)
        );
        assert_eq!(
            w.cleanup_orphan_tmp("/data/auth"),
            Err(AtomicWriteError::NotImplemented)
        );
    }

    #[test]
    fn staging_path_appends_dot_tmp() {
        assert_eq!(staging_path("/data/auth/shadow"), "/data/auth/shadow.tmp");
        assert_eq!(staging_path(""), ".tmp");
    }
}
