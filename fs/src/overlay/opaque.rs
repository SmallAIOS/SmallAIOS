// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Opaque-directory detection.
//!
//! An opaque-directory marker is a regular file named `.opaque` placed
//! inside an upper-layer subdirectory. Its presence hides the entire
//! lower-side subtree at that path; only upper entries are listed.
//!
//! Per `embedded-overlay-v1` spec scenario "Opaque dir hides whole
//! subtree".

use super::upper_layer::{UpperEntryKind, UpperLayer};

/// The opaque-directory marker filename.
pub const OPAQUE_MARKER_NAME: &str = ".opaque";

/// Returns `true` if the upper layer's `dir` is opaque (i.e. an
/// `.opaque` regular file lives inside it).
///
/// Implemented via `UpperLayer::lookup` of the directory itself: the
/// upper-layer impl is responsible for surfacing
/// [`UpperEntryKind::Opaque`] when the `.opaque` marker is present.
/// The `MockUpperLayer` does this; the future F2FS upper will do the
/// same.
pub fn is_opaque_dir<U: UpperLayer>(upper: &U, dir: &str) -> bool {
    matches!(
        upper.lookup(dir),
        Ok(Some(e)) if e.kind == UpperEntryKind::Opaque
    )
}

#[cfg(test)]
mod tests {
    use super::super::upper_layer::MockUpperLayer;
    use super::*;

    #[test]
    fn opaque_marker_name_value() {
        assert_eq!(OPAQUE_MARKER_NAME, ".opaque");
    }

    #[test]
    fn detects_opaque_directory() {
        let mut u = MockUpperLayer::new();
        u.add_opaque("custom");
        assert!(is_opaque_dir(&u, "custom"));
    }

    #[test]
    fn returns_false_for_normal_directory() {
        let mut u = MockUpperLayer::new();
        u.add_dir("custom");
        assert!(!is_opaque_dir(&u, "custom"));
    }

    #[test]
    fn returns_false_for_regular_file() {
        let mut u = MockUpperLayer::new();
        u.add_file("custom", b"data");
        assert!(!is_opaque_dir(&u, "custom"));
    }

    #[test]
    fn returns_false_for_missing_directory() {
        let u = MockUpperLayer::new();
        assert!(!is_opaque_dir(&u, "missing"));
    }

    #[test]
    fn nested_opaque_directory() {
        let mut u = MockUpperLayer::new();
        u.add_opaque("models/custom");
        assert!(is_opaque_dir(&u, "models/custom"));
        assert!(!is_opaque_dir(&u, "models"));
    }
}
