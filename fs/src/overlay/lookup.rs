// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Upper-wins path lookup resolver.
//!
//! Implements the merge precedence rule from `fs-overlay-mount` spec
//! "Upper-wins lookup precedence":
//!
//! > Path lookup under `/models/` SHALL consult the upper layer first.
//! > If the upper has a regular file at the requested path, that file
//! > SHALL be returned. If the upper has a whiteout marker
//! > `<name>.whiteout` at the parent path, the lookup SHALL return
//! > `-ENOENT`. Otherwise the lookup SHALL fall through to the lower
//! > layer.
//!
//! Opaque-directory markers add a third precedence wrinkle: when the
//! caller looks up something inside an opaque upper dir, the
//! lower-side subtree is hidden — even if the upper has no entry for
//! that specific path. See [`LookupResult::OpaqueHidesLowerSubtree`].

use super::lower_layer::{LowerEntry, LowerError, LowerLayer};
use super::upper_layer::{UpperEntry, UpperEntryKind, UpperError, UpperLayer};
use super::OverlayError;

/// One of five possible outcomes of a merge-aware path lookup.
///
/// Generic over the lower-entry type so the merge layer can be
/// monomorphized against either [`super::MockLowerLayer`] or the
/// future squashfs-backed lower without a `dyn` indirection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupResult<L> {
    /// The upper has the entry; this is what the caller sees.
    Upper(UpperEntry),
    /// The upper has a `<name>.whiteout` for this path — the lower's
    /// entry (if any) SHALL NOT be visible. POSIX-level surface:
    /// `-ENOENT`.
    WhiteoutHidesLower,
    /// The lookup is for a path inside a directory that the upper
    /// marks as opaque. The lower-side subtree at this path is hidden
    /// and the upper has no entry for the specific path. POSIX-level
    /// surface: `-ENOENT`.
    OpaqueHidesLowerSubtree,
    /// The upper has no entry, no whiteout, no opaque hide. The
    /// lookup falls through to the lower's entry.
    Lower(L),
    /// Neither layer has the entry.
    NotFound,
}

/// Look up `path` against the merged view.
///
/// `path` is slash-relative to the merge root (i.e. for `/models/foo.onnx`
/// callers pass `"foo.onnx"`; for `/models/custom/x` they pass
/// `"custom/x"`). An empty `path` resolves to the merge-root directory
/// itself.
///
/// Returns [`OverlayError::Upper`] / [`OverlayError::Lower`] on backing-store
/// failures. [`UpperError::NotFound`] from the upper is treated as
/// "not in upper" (Ok(None)), not an error — this lets the merge
/// keep working when only one side has the path.
pub fn lookup<U: UpperLayer, L: LowerLayer>(
    upper: &U,
    lower: &L,
    path: &str,
) -> Result<LookupResult<LowerEntry>, OverlayError> {
    // Step 1 — upper lookup at the exact path.
    let upper_entry = upper.lookup(path)?;
    if let Some(entry) = upper_entry {
        return Ok(match entry.kind {
            UpperEntryKind::Whiteout => LookupResult::WhiteoutHidesLower,
            // RegularFile, Directory, Opaque all "win" — the caller
            // sees the upper's entry directly. Opaque is a directory
            // that hides the lower-side subtree but is itself a
            // valid upper-side directory; the caller still sees it.
            _ => LookupResult::Upper(entry),
        });
    }

    // Step 2 — no exact upper hit. Check whether some ancestor
    // directory of `path` is opaque. If yes, lower-side fall-through
    // is suppressed.
    if any_ancestor_opaque(upper, path)? {
        return Ok(LookupResult::OpaqueHidesLowerSubtree);
    }

    // Step 3 — fall through to the lower layer.
    match lower.lookup(path) {
        Ok(Some(entry)) => Ok(LookupResult::Lower(entry)),
        Ok(None) => Ok(LookupResult::NotFound),
        Err(LowerError::NotFound) => Ok(LookupResult::NotFound),
        Err(e) => Err(OverlayError::Lower(e)),
    }
}

/// Walk every strict ancestor directory of `path` on the upper. Return
/// `true` if any of them is opaque.
///
/// Pure ancestor walk — does NOT consult `path` itself. The caller
/// already did that as Step 1 of [`lookup`].
fn any_ancestor_opaque<U: UpperLayer>(upper: &U, path: &str) -> Result<bool, UpperError> {
    let trimmed = path.trim_start_matches('/').trim_end_matches('/');
    if trimmed.is_empty() {
        return Ok(false);
    }
    let mut walked = alloc::string::String::new();
    let parts: alloc::vec::Vec<&str> = trimmed.split('/').collect();
    // Walk every prefix EXCEPT the full path.
    let ancestor_count = parts.len().saturating_sub(1);
    for part in parts.iter().take(ancestor_count) {
        if !walked.is_empty() {
            walked.push('/');
        }
        walked.push_str(part);
        if let Some(entry) = upper.lookup(&walked)? {
            if entry.kind == UpperEntryKind::Opaque {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::super::lower_layer::{LowerEntryKind, MockLowerLayer};
    use super::super::upper_layer::MockUpperLayer;
    use super::*;

    #[test]
    fn upper_file_shadows_lower_file() {
        let mut u = MockUpperLayer::new();
        u.add_file("a.onnx", b"upper");
        let mut l = MockLowerLayer::new();
        l.add_file("a.onnx", b"lower");
        match lookup(&u, &l, "a.onnx").unwrap() {
            LookupResult::Upper(e) => {
                assert_eq!(e.name, "a.onnx");
                assert_eq!(e.kind, UpperEntryKind::RegularFile);
            }
            other => panic!("expected Upper, got {:?}", other),
        }
    }

    #[test]
    fn lower_visible_when_upper_absent() {
        let u = MockUpperLayer::new();
        let mut l = MockLowerLayer::new();
        l.add_file("a.onnx", b"lower");
        match lookup(&u, &l, "a.onnx").unwrap() {
            LookupResult::Lower(e) => {
                assert_eq!(e.name, "a.onnx");
                assert_eq!(e.kind, LowerEntryKind::RegularFile);
            }
            other => panic!("expected Lower, got {:?}", other),
        }
    }

    #[test]
    fn whiteout_hides_lower_returns_whiteoutresult() {
        let mut u = MockUpperLayer::new();
        u.add_whiteout("a.onnx");
        let mut l = MockLowerLayer::new();
        l.add_file("a.onnx", b"lower");
        assert_eq!(
            lookup(&u, &l, "a.onnx").unwrap(),
            LookupResult::WhiteoutHidesLower
        );
    }

    #[test]
    fn opaque_dir_hides_lower_subtree() {
        let mut u = MockUpperLayer::new();
        u.add_opaque("custom");
        let mut l = MockLowerLayer::new();
        l.add_file("custom/leaf", b"lower");
        // `custom/leaf`: upper has no entry, but `custom/` is opaque.
        assert_eq!(
            lookup(&u, &l, "custom/leaf").unwrap(),
            LookupResult::OpaqueHidesLowerSubtree
        );
    }

    #[test]
    fn opaque_dir_itself_returns_upper() {
        let mut u = MockUpperLayer::new();
        u.add_opaque("custom");
        let mut l = MockLowerLayer::new();
        l.add_file("custom/leaf", b"lower");
        // The opaque dir itself is an upper entry.
        match lookup(&u, &l, "custom").unwrap() {
            LookupResult::Upper(e) => {
                assert_eq!(e.kind, UpperEntryKind::Opaque);
            }
            other => panic!("expected Upper(Opaque), got {:?}", other),
        }
    }

    #[test]
    fn not_found_when_neither_layer_has_entry() {
        let u = MockUpperLayer::new();
        let l = MockLowerLayer::new();
        assert_eq!(lookup(&u, &l, "missing").unwrap(), LookupResult::NotFound);
    }

    #[test]
    fn upper_directory_wins_over_lower_file() {
        // Edge: upper has `foo` as a directory, lower has `foo` as a
        // regular file. Upper-wins applies — caller sees the directory.
        let mut u = MockUpperLayer::new();
        u.add_dir("foo");
        let mut l = MockLowerLayer::new();
        l.add_file("foo", b"lower");
        match lookup(&u, &l, "foo").unwrap() {
            LookupResult::Upper(e) => assert_eq!(e.kind, UpperEntryKind::Directory),
            other => panic!("expected Upper(Directory), got {:?}", other),
        }
    }

    #[test]
    fn nested_path_falls_through_when_upper_empty() {
        let u = MockUpperLayer::new();
        let mut l = MockLowerLayer::new();
        l.add_file("a/b/c", b"lower");
        match lookup(&u, &l, "a/b/c").unwrap() {
            LookupResult::Lower(e) => assert_eq!(e.size, 5),
            other => panic!("expected Lower, got {:?}", other),
        }
    }

    #[test]
    fn opaque_only_hides_at_or_below_marker() {
        let mut u = MockUpperLayer::new();
        u.add_opaque("custom");
        let mut l = MockLowerLayer::new();
        l.add_file("other.onnx", b"lower");
        // `other.onnx` is not under `custom/` → lower visible.
        match lookup(&u, &l, "other.onnx").unwrap() {
            LookupResult::Lower(_) => {}
            other => panic!("expected Lower, got {:?}", other),
        }
    }
}
