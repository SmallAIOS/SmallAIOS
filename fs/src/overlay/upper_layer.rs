// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `UpperLayer` trait — abstracts the read-write side of the overlay.
//!
//! Phase 1 ships:
//!
//! - [`UpperLayer`] trait (lookup / read / iter_dir).
//! - [`UpperEntry`] / [`UpperEntryKind`] descriptors.
//! - [`MockUpperLayer`] in-memory test fixture (used by Phase 1 tests
//!   and any later phase wanting to exercise the merge in unit tests).
//!
//! `F2fsUpperLayer` is the production implementation that wires the
//! trait to `embedded-filesystem-v1`'s F2FS RW path. Phase 2 of
//! `embedded-overlay-v1` fills it in once F2FS RW lands.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Errors produced by the upper layer.
///
/// Mirrors the error vocabulary the future F2FS RW path will surface.
/// Phase 1's [`MockUpperLayer`] returns a strict subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpperError {
    /// Path does not exist in the upper layer.
    NotFound,
    /// Caller asked for a directory operation on a non-directory entry
    /// (or vice versa).
    NotADirectory,
    /// Caller asked to read a file but the entry is not a regular file
    /// (e.g. opaque marker, whiteout marker, or a directory).
    NotARegularFile,
    /// Read offset exceeds file size.
    OutOfRange,
    /// I/O error on the backing store (Phase 2 — F2FS error chain).
    Io,
}

impl fmt::Display for UpperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "upper: not found"),
            Self::NotADirectory => write!(f, "upper: not a directory"),
            Self::NotARegularFile => write!(f, "upper: not a regular file"),
            Self::OutOfRange => write!(f, "upper: read out of range"),
            Self::Io => write!(f, "upper: I/O error"),
        }
    }
}

/// Kind of an entry surfaced by the upper layer.
///
/// `Whiteout` and `Opaque` are reported by the upper layer rather than
/// the raw filesystem — when the layer detects a `<name>.whiteout`
/// regular file it surfaces a `Whiteout` entry under the bare name,
/// and when a directory contains `.opaque` it surfaces `Opaque` for
/// that directory.
///
/// The `MockUpperLayer` in this module follows the same convention so
/// the merge code path is exercised identically in tests and at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpperEntryKind {
    /// Ordinary regular file.
    RegularFile,
    /// Ordinary directory.
    Directory,
    /// Whiteout marker (the upper has `<name>.whiteout`).
    Whiteout,
    /// Opaque-directory marker (the upper has `<dir>/.opaque`).
    Opaque,
}

/// One entry surfaced by the upper layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpperEntry {
    /// File or directory name (final component, never a path).
    pub name: String,
    /// Kind of entry. See [`UpperEntryKind`].
    pub kind: UpperEntryKind,
    /// Size in bytes (0 for non-regular-file entries).
    pub size: u64,
    /// POSIX-style mode bits. Phase 1 doesn't enforce permissions; this
    /// is plumbing for the Phase 5 integrity layer.
    pub mode: u16,
}

/// Read-write upper-layer abstraction.
///
/// The merge layer ([`super::lookup`], [`super::merge`]) consumes only
/// these methods. Any backing store (the test fixture, the production
/// F2FS path) implements them and gets the merge logic for free.
///
/// All paths passed in are **slash-relative to the upper-layer root**,
/// i.e. they do NOT start with `/` and do NOT include a `/data/models-upper/`
/// prefix. The merge layer normalizes operator-facing paths before
/// calling these methods.
pub trait UpperLayer {
    /// Resolve a path to an entry.
    ///
    /// Returns `Ok(Some(entry))` when present, `Ok(None)` when absent,
    /// or `Err(...)` on I/O failure. **Whiteouts are surfaced as
    /// `Some(UpperEntry { kind: Whiteout, .. })`** under the bare name
    /// (i.e. `lookup("foo")` returns `Whiteout` if `foo.whiteout`
    /// exists), so the merge layer doesn't need to do the suffix
    /// dance itself.
    fn lookup(&self, path: &str) -> Result<Option<UpperEntry>, UpperError>;

    /// Read up to `buf.len()` bytes from `path` starting at `offset`.
    ///
    /// Returns the number of bytes actually read (≤ `buf.len()`). Reading
    /// at exactly `file_size` returns `Ok(0)`. Reading past `file_size`
    /// returns [`UpperError::OutOfRange`].
    fn read_file(&self, path: &str, offset: u64, buf: &mut [u8]) -> Result<usize, UpperError>;

    /// List entries in directory `path`.
    ///
    /// **Whiteout files are surfaced as `Whiteout` entries under the
    /// bare name** (the `.whiteout` suffix is hidden by the layer
    /// itself). Likewise an `.opaque` marker surfaces as a single
    /// `Opaque` entry on the parent directory's listing. The merge
    /// layer relies on this convention — see [`super::merge`].
    fn iter_dir(&self, path: &str) -> Result<Vec<UpperEntry>, UpperError>;
}

// ─── MockUpperLayer ────────────────────────────────────────────────────────────

/// Internal mock representation of one entry.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MockEntry {
    /// Regular file with raw content.
    File(Vec<u8>),
    /// Directory (entries enumerated by traversing the BTreeMap).
    Directory,
    /// Whiteout — internally stored as a regular file with the
    /// `.whiteout` suffix. The Mock layer hides the suffix when
    /// surfacing entries.
    Whiteout(Vec<u8>),
    /// Opaque marker — internally stored as a regular file named
    /// `.opaque` inside a directory.
    Opaque,
}

/// In-memory `UpperLayer` for unit tests.
///
/// Stores entries in a `BTreeMap<String, MockEntry>` keyed by full
/// path (slash-relative to the upper root, no leading or trailing
/// slashes). The test layer presents the SAME contract as the
/// future F2FS-backed upper: whiteouts surface as `Whiteout` entries
/// under the bare name, opaque markers surface as `Opaque` entries
/// for the containing directory.
///
/// **Not** for production use; gated by `#[cfg(...)]` only via the
/// surrounding cargo feature `fs-overlay-mounts`.
#[derive(Debug, Default, Clone)]
pub struct MockUpperLayer {
    entries: BTreeMap<String, MockEntry>,
}

impl MockUpperLayer {
    /// Construct an empty mock upper layer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a regular file at the given upper-relative path with the
    /// given content. Auto-creates parent directories.
    pub fn add_file(&mut self, path: &str, content: &[u8]) {
        self.ensure_parents(path);
        self.entries
            .insert(normalize(path), MockEntry::File(content.to_vec()));
    }

    /// Add an explicit directory entry. Most callers don't need this —
    /// `add_file` auto-creates parents — but tests for empty
    /// directories use it.
    pub fn add_dir(&mut self, path: &str) {
        self.ensure_parents(path);
        let key = normalize(path);
        self.entries.entry(key).or_insert(MockEntry::Directory);
    }

    /// Add a whiteout marker for `name` at the given parent.
    ///
    /// Internally creates a regular file at `parent/name.whiteout`.
    /// Test code calls `add_whiteout("foo", "")` to put a whiteout
    /// for `foo` at the upper-root, or `add_whiteout("custom/foo", "")`
    /// to put a whiteout for `foo` inside `custom/`.
    pub fn add_whiteout(&mut self, full_name: &str) {
        self.ensure_parents(full_name);
        let key = alloc::format!("{}.whiteout", normalize(full_name));
        self.entries.insert(key, MockEntry::Whiteout(Vec::new()));
    }

    /// Add a non-empty whiteout marker (still hides the lower entry,
    /// but stores arbitrary bytes per the spec scenario "Non-empty
    /// whiteout still functional").
    pub fn add_whiteout_with_content(&mut self, full_name: &str, content: &[u8]) {
        self.ensure_parents(full_name);
        let key = alloc::format!("{}.whiteout", normalize(full_name));
        self.entries
            .insert(key, MockEntry::Whiteout(content.to_vec()));
    }

    /// Add an opaque-directory marker on `dir`. Internally creates the
    /// directory (if absent) and a regular file `<dir>/.opaque`.
    pub fn add_opaque(&mut self, dir: &str) {
        self.add_dir(dir);
        let dir_key = normalize(dir);
        let key = if dir_key.is_empty() {
            ".opaque".into()
        } else {
            alloc::format!("{}/.opaque", dir_key)
        };
        self.entries.insert(key, MockEntry::Opaque);
    }

    fn ensure_parents(&mut self, path: &str) {
        let n = normalize(path);
        let mut walked = String::new();
        for component in n.split('/').filter(|s| !s.is_empty()) {
            // Don't auto-promote the leaf into a directory; only
            // create entries for each non-leaf segment.
            if !walked.is_empty() {
                walked.push('/');
            }
            let candidate = alloc::format!("{}{}", walked, component);
            // Skip the leaf — that's the entry being created.
            if candidate == n {
                break;
            }
            self.entries
                .entry(candidate.clone())
                .or_insert(MockEntry::Directory);
            walked = candidate;
        }
    }
}

fn normalize(p: &str) -> String {
    p.trim_start_matches('/').trim_end_matches('/').into()
}

impl UpperLayer for MockUpperLayer {
    fn lookup(&self, path: &str) -> Result<Option<UpperEntry>, UpperError> {
        let key = normalize(path);

        // Empty key = upper root directory.
        if key.is_empty() {
            return Ok(Some(UpperEntry {
                name: String::new(),
                kind: UpperEntryKind::Directory,
                size: 0,
                mode: 0o755,
            }));
        }

        let bare_name = key.rsplit('/').next().unwrap_or("").to_string();

        // Case 1 — caller asks for `foo`: check `foo.whiteout` first,
        // then `foo` itself.
        let whiteout_key = alloc::format!("{}.whiteout", key);
        if matches!(
            self.entries.get(&whiteout_key),
            Some(MockEntry::Whiteout(_))
        ) {
            return Ok(Some(UpperEntry {
                name: bare_name.clone(),
                kind: UpperEntryKind::Whiteout,
                size: 0,
                mode: 0o644,
            }));
        }

        match self.entries.get(&key) {
            Some(MockEntry::File(bytes)) => Ok(Some(UpperEntry {
                name: bare_name,
                kind: UpperEntryKind::RegularFile,
                size: bytes.len() as u64,
                mode: 0o644,
            })),
            Some(MockEntry::Directory) => {
                // Surface `Opaque` if `<dir>/.opaque` exists.
                let opaque_key = alloc::format!("{}/.opaque", key);
                if self.entries.contains_key(&opaque_key) {
                    Ok(Some(UpperEntry {
                        name: bare_name,
                        kind: UpperEntryKind::Opaque,
                        size: 0,
                        mode: 0o755,
                    }))
                } else {
                    Ok(Some(UpperEntry {
                        name: bare_name,
                        kind: UpperEntryKind::Directory,
                        size: 0,
                        mode: 0o755,
                    }))
                }
            }
            Some(MockEntry::Whiteout(_)) => {
                // The caller asked for the *literal* whiteout-suffixed
                // name. Not exposed at the lookup layer.
                Ok(None)
            }
            Some(MockEntry::Opaque) => {
                // The caller asked for the literal `.opaque` filename.
                // Not exposed.
                Ok(None)
            }
            None => Ok(None),
        }
    }

    fn read_file(&self, path: &str, offset: u64, buf: &mut [u8]) -> Result<usize, UpperError> {
        let key = normalize(path);
        match self.entries.get(&key) {
            Some(MockEntry::File(bytes)) => {
                let len = bytes.len() as u64;
                if offset > len {
                    return Err(UpperError::OutOfRange);
                }
                let start = offset as usize;
                let available = bytes.len() - start;
                let n = core::cmp::min(buf.len(), available);
                buf[..n].copy_from_slice(&bytes[start..start + n]);
                Ok(n)
            }
            Some(MockEntry::Directory) | Some(MockEntry::Opaque) => {
                Err(UpperError::NotARegularFile)
            }
            Some(MockEntry::Whiteout(_)) => Err(UpperError::NotARegularFile),
            None => Err(UpperError::NotFound),
        }
    }

    fn iter_dir(&self, path: &str) -> Result<Vec<UpperEntry>, UpperError> {
        let key = normalize(path);

        // The upper-root listing scans for direct children in the map.
        // For non-root, verify the path resolves to a directory first.
        if !key.is_empty() {
            match self.entries.get(&key) {
                Some(MockEntry::Directory) => {}
                Some(_) => return Err(UpperError::NotADirectory),
                None => return Err(UpperError::NotFound),
            }
        }

        let prefix = if key.is_empty() {
            String::new()
        } else {
            alloc::format!("{}/", key)
        };

        let mut out = Vec::new();
        for (k, v) in &self.entries {
            // Direct child only — `prefix` <= k <= no further `/`.
            if !k.starts_with(&prefix) {
                continue;
            }
            let suffix = &k[prefix.len()..];
            if suffix.is_empty() || suffix.contains('/') {
                continue;
            }

            // Hide the `.opaque` marker file from listings; the
            // upper layer surfaces opaqueness via `Opaque` on the
            // PARENT directory entry, not as a separate listing.
            if suffix == ".opaque" {
                continue;
            }

            // Whiteout file: surface under the bare name with kind
            // Whiteout (suffix stripped).
            if let Some(stripped) = suffix.strip_suffix(".whiteout") {
                if let MockEntry::Whiteout(_) = v {
                    out.push(UpperEntry {
                        name: stripped.into(),
                        kind: UpperEntryKind::Whiteout,
                        size: 0,
                        mode: 0o644,
                    });
                    continue;
                }
            }

            match v {
                MockEntry::File(bytes) => out.push(UpperEntry {
                    name: suffix.into(),
                    kind: UpperEntryKind::RegularFile,
                    size: bytes.len() as u64,
                    mode: 0o644,
                }),
                MockEntry::Directory => {
                    let opaque_key = alloc::format!("{}/.opaque", k);
                    let kind = if self.entries.contains_key(&opaque_key) {
                        UpperEntryKind::Opaque
                    } else {
                        UpperEntryKind::Directory
                    };
                    out.push(UpperEntry {
                        name: suffix.into(),
                        kind,
                        size: 0,
                        mode: 0o755,
                    });
                }
                // Whiteout files that for any reason didn't strip
                // (e.g. the suffix didn't match): suppress.
                MockEntry::Whiteout(_) => {}
                MockEntry::Opaque => {}
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_upper_lookup_file() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo.onnx", b"hello");
        let e = u.lookup("foo.onnx").unwrap().unwrap();
        assert_eq!(e.name, "foo.onnx");
        assert_eq!(e.kind, UpperEntryKind::RegularFile);
        assert_eq!(e.size, 5);
    }

    #[test]
    fn mock_upper_lookup_absent() {
        let u = MockUpperLayer::new();
        assert_eq!(u.lookup("foo").unwrap(), None);
    }

    #[test]
    fn mock_upper_lookup_whiteout_surfaces_under_bare_name() {
        let mut u = MockUpperLayer::new();
        u.add_whiteout("foo");
        let e = u.lookup("foo").unwrap().unwrap();
        assert_eq!(e.kind, UpperEntryKind::Whiteout);
        assert_eq!(e.name, "foo");
    }

    #[test]
    fn mock_upper_lookup_nonempty_whiteout_still_surfaces() {
        let mut u = MockUpperLayer::new();
        u.add_whiteout_with_content("foo", b"# comment");
        let e = u.lookup("foo").unwrap().unwrap();
        assert_eq!(e.kind, UpperEntryKind::Whiteout);
    }

    #[test]
    fn mock_upper_lookup_opaque_dir_surfaces_as_opaque() {
        let mut u = MockUpperLayer::new();
        u.add_opaque("custom");
        let e = u.lookup("custom").unwrap().unwrap();
        assert_eq!(e.kind, UpperEntryKind::Opaque);
    }

    #[test]
    fn mock_upper_read_file_bytes() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo", b"abcdef");
        let mut buf = [0u8; 4];
        let n = u.read_file("foo", 0, &mut buf).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&buf, b"abcd");
        let mut buf2 = [0u8; 8];
        let n = u.read_file("foo", 2, &mut buf2).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&buf2[..4], b"cdef");
    }

    #[test]
    fn mock_upper_read_file_at_eof_returns_zero() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo", b"abc");
        let mut buf = [0u8; 4];
        let n = u.read_file("foo", 3, &mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn mock_upper_read_past_eof_errors() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo", b"abc");
        let mut buf = [0u8; 4];
        assert_eq!(
            u.read_file("foo", 100, &mut buf),
            Err(UpperError::OutOfRange)
        );
    }

    #[test]
    fn mock_upper_iter_dir_root() {
        let mut u = MockUpperLayer::new();
        u.add_file("a", b"a");
        u.add_file("b", b"b");
        u.add_file("nested/c", b"c");
        let entries = u.iter_dir("/").unwrap();
        let names: alloc::vec::Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
        assert!(names.contains(&"nested"));
        // `nested/c` is NOT a direct child.
        assert!(!names.contains(&"c"));
    }

    #[test]
    fn mock_upper_iter_dir_hides_dot_opaque() {
        let mut u = MockUpperLayer::new();
        u.add_opaque("custom");
        u.add_file("custom/foo", b"x");
        let entries = u.iter_dir("custom").unwrap();
        let names: alloc::vec::Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(!names.contains(&".opaque"));
    }

    #[test]
    fn mock_upper_iter_dir_surfaces_whiteout_under_bare_name() {
        let mut u = MockUpperLayer::new();
        u.add_whiteout("foo");
        let entries = u.iter_dir("/").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "foo");
        assert_eq!(entries[0].kind, UpperEntryKind::Whiteout);
    }

    #[test]
    fn mock_upper_iter_dir_not_found() {
        let u = MockUpperLayer::new();
        assert_eq!(u.iter_dir("missing"), Err(UpperError::NotFound));
    }

    #[test]
    fn mock_upper_iter_dir_not_a_directory() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo", b"x");
        assert_eq!(u.iter_dir("foo"), Err(UpperError::NotADirectory));
    }
}
