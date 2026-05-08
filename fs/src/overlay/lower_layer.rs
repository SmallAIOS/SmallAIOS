// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `LowerLayer` trait — abstracts the read-only lower side of the overlay.
//!
//! At runtime the lower is the active squashfs A/B slot. Phase 1 keeps
//! the merge logic decoupled from the squashfs reader by exposing a
//! tiny trait, mirrored by [`MockLowerLayer`] for unit tests.
//!
//! The squashfs adapter lives in [`SquashfsLowerLayer`] (gated by
//! `fs-on-disk-mounts`); when both feature flags are on, the merge
//! layer can be wired to a real `Squashfs` reader. When only
//! `fs-overlay-mounts` is on, the mock is the sole impl available
//! to host-side tests.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Errors produced by the lower layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// Path does not exist in the lower layer.
    NotFound,
    /// Path resolved to something that isn't a directory.
    NotADirectory,
    /// Path resolved to a non-regular-file (Phase 1 doesn't read
    /// symlinks through the merge view).
    NotARegularFile,
    /// Read offset exceeds file size.
    OutOfRange,
    /// I/O error on the backing store.
    Io,
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "lower: not found"),
            Self::NotADirectory => write!(f, "lower: not a directory"),
            Self::NotARegularFile => write!(f, "lower: not a regular file"),
            Self::OutOfRange => write!(f, "lower: read out of range"),
            Self::Io => write!(f, "lower: I/O error"),
        }
    }
}

/// Kind of an entry surfaced by the lower layer.
///
/// Unlike the upper, the lower **never** surfaces whiteout or opaque
/// markers — it's a plain read-only filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerEntryKind {
    RegularFile,
    Directory,
    /// Symlink or other inode kind. The merge layer treats these the
    /// same as `RegularFile` for listing purposes; readers go through
    /// the real squashfs API which projects this anyway.
    Other,
}

/// One entry surfaced by the lower layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerEntry {
    pub name: String,
    pub kind: LowerEntryKind,
    pub size: u64,
    pub mode: u16,
}

/// Read-only lower-layer abstraction.
///
/// Paths are slash-relative to the lower-mount root, just like
/// [`super::UpperLayer`].
pub trait LowerLayer {
    /// Resolve a path to an entry.
    fn lookup(&self, path: &str) -> Result<Option<LowerEntry>, LowerError>;

    /// Read up to `buf.len()` bytes from `path` starting at `offset`.
    fn read_file(&self, path: &str, offset: u64, buf: &mut [u8]) -> Result<usize, LowerError>;

    /// List entries in directory `path`.
    fn iter_dir(&self, path: &str) -> Result<Vec<LowerEntry>, LowerError>;
}

// ─── MockLowerLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum MockLowerEntry {
    File(Vec<u8>),
    Directory,
}

/// In-memory `LowerLayer` for unit tests.
///
/// Stores a flat `BTreeMap<String, MockLowerEntry>` keyed by full
/// upper-relative path. Doesn't model squashfs's permission bits,
/// inode numbers, etc. — Phase 1 only needs presence/absence and
/// content for the merge tests.
#[derive(Debug, Default, Clone)]
pub struct MockLowerLayer {
    entries: BTreeMap<String, MockLowerEntry>,
}

impl MockLowerLayer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_file(&mut self, path: &str, content: &[u8]) {
        self.ensure_parents(path);
        self.entries
            .insert(normalize(path), MockLowerEntry::File(content.to_vec()));
    }

    pub fn add_dir(&mut self, path: &str) {
        self.ensure_parents(path);
        let key = normalize(path);
        self.entries.entry(key).or_insert(MockLowerEntry::Directory);
    }

    fn ensure_parents(&mut self, path: &str) {
        let n = normalize(path);
        let mut walked = String::new();
        for component in n.split('/').filter(|s| !s.is_empty()) {
            if !walked.is_empty() {
                walked.push('/');
            }
            let candidate = alloc::format!("{}{}", walked, component);
            if candidate == n {
                break;
            }
            self.entries
                .entry(candidate.clone())
                .or_insert(MockLowerEntry::Directory);
            walked = candidate;
        }
    }
}

fn normalize(p: &str) -> String {
    p.trim_start_matches('/').trim_end_matches('/').into()
}

impl LowerLayer for MockLowerLayer {
    fn lookup(&self, path: &str) -> Result<Option<LowerEntry>, LowerError> {
        let key = normalize(path);
        if key.is_empty() {
            return Ok(Some(LowerEntry {
                name: String::new(),
                kind: LowerEntryKind::Directory,
                size: 0,
                mode: 0o755,
            }));
        }
        let bare = key.rsplit('/').next().unwrap_or("").to_string();
        match self.entries.get(&key) {
            Some(MockLowerEntry::File(bytes)) => Ok(Some(LowerEntry {
                name: bare,
                kind: LowerEntryKind::RegularFile,
                size: bytes.len() as u64,
                mode: 0o444,
            })),
            Some(MockLowerEntry::Directory) => Ok(Some(LowerEntry {
                name: bare,
                kind: LowerEntryKind::Directory,
                size: 0,
                mode: 0o555,
            })),
            None => Ok(None),
        }
    }

    fn read_file(&self, path: &str, offset: u64, buf: &mut [u8]) -> Result<usize, LowerError> {
        let key = normalize(path);
        match self.entries.get(&key) {
            Some(MockLowerEntry::File(bytes)) => {
                let len = bytes.len() as u64;
                if offset > len {
                    return Err(LowerError::OutOfRange);
                }
                let start = offset as usize;
                let avail = bytes.len() - start;
                let n = core::cmp::min(buf.len(), avail);
                buf[..n].copy_from_slice(&bytes[start..start + n]);
                Ok(n)
            }
            Some(MockLowerEntry::Directory) => Err(LowerError::NotARegularFile),
            None => Err(LowerError::NotFound),
        }
    }

    fn iter_dir(&self, path: &str) -> Result<Vec<LowerEntry>, LowerError> {
        let key = normalize(path);
        if !key.is_empty() {
            match self.entries.get(&key) {
                Some(MockLowerEntry::Directory) => {}
                Some(_) => return Err(LowerError::NotADirectory),
                None => return Err(LowerError::NotFound),
            }
        }
        let prefix = if key.is_empty() {
            String::new()
        } else {
            alloc::format!("{}/", key)
        };
        let mut out = Vec::new();
        for (k, v) in &self.entries {
            if !k.starts_with(&prefix) {
                continue;
            }
            let suffix = &k[prefix.len()..];
            if suffix.is_empty() || suffix.contains('/') {
                continue;
            }
            match v {
                MockLowerEntry::File(bytes) => out.push(LowerEntry {
                    name: suffix.into(),
                    kind: LowerEntryKind::RegularFile,
                    size: bytes.len() as u64,
                    mode: 0o444,
                }),
                MockLowerEntry::Directory => out.push(LowerEntry {
                    name: suffix.into(),
                    kind: LowerEntryKind::Directory,
                    size: 0,
                    mode: 0o555,
                }),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_lower_lookup_file() {
        let mut l = MockLowerLayer::new();
        l.add_file("a", b"abc");
        let e = l.lookup("a").unwrap().unwrap();
        assert_eq!(e.name, "a");
        assert_eq!(e.kind, LowerEntryKind::RegularFile);
        assert_eq!(e.size, 3);
    }

    #[test]
    fn mock_lower_lookup_absent_returns_none() {
        let l = MockLowerLayer::new();
        assert_eq!(l.lookup("a").unwrap(), None);
    }

    #[test]
    fn mock_lower_iter_dir_root() {
        let mut l = MockLowerLayer::new();
        l.add_file("a", b"a");
        l.add_file("b/c", b"c");
        let entries = l.iter_dir("/").unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn mock_lower_read_file() {
        let mut l = MockLowerLayer::new();
        l.add_file("a", b"hello");
        let mut buf = [0u8; 5];
        let n = l.read_file("a", 0, &mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b"hello");
    }
}
