// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Whiteout-marker detection.
//!
//! A whiteout is a regular file named `<name>.whiteout` placed at the
//! parent path in the upper layer. Its mere presence hides the lower
//! layer's `<name>` entry; the file's contents (zero bytes or
//! arbitrary comment text) are never inspected.
//!
//! Per `embedded-overlay-v1` spec scenario "Empty whiteout file enough
//! to hide" + "Non-empty whiteout still functional".

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::upper_layer::{UpperEntryKind, UpperError, UpperLayer};

/// The reserved suffix for whiteout markers.
pub const WHITEOUT_SUFFIX: &str = ".whiteout";

/// Returns `true` if the upper layer has an active whiteout for
/// `name` at `parent`.
///
/// The check is "does `parent/name` resolve to an `Whiteout` entry on
/// the upper layer?". The `MockUpperLayer` (and the future F2FS
/// upper) surface whiteouts under the bare name with kind
/// [`UpperEntryKind::Whiteout`], so the merge layer doesn't need to
/// dance with the suffix itself.
pub fn is_whiteout<U: UpperLayer>(upper: &U, parent: &str, name: &str) -> bool {
    let path = if parent.is_empty() {
        alloc::string::ToString::to_string(name)
    } else {
        alloc::format!("{}/{}", parent.trim_end_matches('/'), name)
    };
    matches!(
        upper.lookup(&path),
        Ok(Some(e)) if e.kind == UpperEntryKind::Whiteout
    )
}

/// List all active whiteouts in `parent` (the bare names of entries
/// that the upper hides via a `<name>.whiteout` marker).
///
/// Returns an empty Vec if `parent` doesn't exist on the upper.
pub fn list_whiteouts_in<U: UpperLayer>(upper: &U, parent: &str) -> Vec<String> {
    match upper.iter_dir(parent) {
        Ok(entries) => entries
            .into_iter()
            .filter(|e| e.kind == UpperEntryKind::Whiteout)
            .map(|e| e.name)
            .collect(),
        Err(UpperError::NotFound) | Err(UpperError::NotADirectory) => Vec::new(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::upper_layer::MockUpperLayer;
    use super::*;

    #[test]
    fn whiteout_suffix_value() {
        assert_eq!(WHITEOUT_SUFFIX, ".whiteout");
    }

    #[test]
    fn is_whiteout_detects_marker() {
        let mut u = MockUpperLayer::new();
        u.add_whiteout("foo");
        assert!(is_whiteout(&u, "", "foo"));
    }

    #[test]
    fn is_whiteout_false_when_absent() {
        let u = MockUpperLayer::new();
        assert!(!is_whiteout(&u, "", "foo"));
    }

    #[test]
    fn is_whiteout_false_for_regular_file_with_same_name() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo", b"bar");
        assert!(!is_whiteout(&u, "", "foo"));
    }

    #[test]
    fn is_whiteout_works_for_nested_path() {
        let mut u = MockUpperLayer::new();
        u.add_whiteout("custom/foo");
        assert!(is_whiteout(&u, "custom", "foo"));
        assert!(!is_whiteout(&u, "", "foo"));
    }

    #[test]
    fn is_whiteout_nonempty_marker_still_hides() {
        let mut u = MockUpperLayer::new();
        u.add_whiteout_with_content("foo", b"# operator note");
        assert!(is_whiteout(&u, "", "foo"));
    }

    #[test]
    fn list_whiteouts_in_root() {
        use alloc::string::ToString;
        let mut u = MockUpperLayer::new();
        u.add_whiteout("a");
        u.add_whiteout("b");
        u.add_file("c", b"x");
        let mut wh = list_whiteouts_in(&u, "/");
        wh.sort();
        assert_eq!(wh, alloc::vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn list_whiteouts_in_nonexistent_dir_is_empty() {
        let u = MockUpperLayer::new();
        let wh = list_whiteouts_in(&u, "/missing");
        assert!(wh.is_empty());
    }
}
