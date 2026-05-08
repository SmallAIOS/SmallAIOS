// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Reserved-suffix list.
//!
//! Names ending in any of these suffixes are reserved for overlay-internal
//! markers and SHALL NOT be operator-creatable. The merge layer filters
//! them from `readdir` listings; the future `model_add` syscall (Phase 3)
//! rejects them with `-EINVAL`.
//!
//! Per `embedded-overlay-v1` spec Q9 / `fs-overlay-mount` Requirement
//! "Reserved-suffix names rejected".

extern crate alloc;

use alloc::string::ToString;

use super::OverlayError;

/// Suffixes reserved for overlay-internal sidecars and markers.
///
/// - `.whiteout` — file-based whiteout marker (per `fs-overlay-mount`).
/// - `.opaque`   — opaque-directory marker.
/// - `.sha3`     — SHA-3-256 fingerprint sidecar (Phase 5 integrity).
/// - `.sig`      — ML-DSA-65 signature sidecar (Phase 5 integrity).
pub const RESERVED_SUFFIXES: &[&str] = &[".whiteout", ".opaque", ".sha3", ".sig"];

/// Returns `true` if `name` ends in any reserved suffix.
///
/// Matching is case-sensitive — operator-supplied names like
/// `Foo.WHITEOUT` are NOT considered reserved (the operating system
/// is case-sensitive throughout, and reserved markers are always
/// emitted lowercase).
pub fn is_reserved_name(name: &str) -> bool {
    matched_reserved_suffix(name).is_some()
}

/// Returns the matched reserved suffix when one applies, else `None`.
///
/// Useful for emitting a clear error message naming the conflicting
/// suffix (per the spec scenario "Reserved suffix on add").
pub fn matched_reserved_suffix(name: &str) -> Option<&'static str> {
    RESERVED_SUFFIXES
        .iter()
        .find(|&&suffix| name.ends_with(suffix))
        .copied()
}

/// Reject `name` with [`OverlayError::ReservedSuffix`] if it ends in
/// any reserved suffix. Used by `model_add` (Phase 3) and the merge
/// layer's listing filter.
pub fn reject_if_reserved(name: &str) -> Result<(), OverlayError> {
    if let Some(suffix) = matched_reserved_suffix(name) {
        Err(OverlayError::ReservedSuffix(suffix.to_string()))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_suffix_list_is_complete() {
        // Spec Q9 names exactly these four suffixes.
        assert_eq!(RESERVED_SUFFIXES.len(), 4);
        assert!(RESERVED_SUFFIXES.contains(&".whiteout"));
        assert!(RESERVED_SUFFIXES.contains(&".opaque"));
        assert!(RESERVED_SUFFIXES.contains(&".sha3"));
        assert!(RESERVED_SUFFIXES.contains(&".sig"));
    }

    #[test]
    fn is_reserved_name_matches_each_suffix() {
        assert!(is_reserved_name("foo.whiteout"));
        assert!(is_reserved_name("foo.opaque"));
        assert!(is_reserved_name("foo.sha3"));
        assert!(is_reserved_name("foo.sig"));
        // The opaque-marker file itself is reserved.
        assert!(is_reserved_name(".opaque"));
    }

    #[test]
    fn is_reserved_name_rejects_non_reserved() {
        assert!(!is_reserved_name("foo.onnx"));
        assert!(!is_reserved_name("model.bin"));
        assert!(!is_reserved_name("README"));
        assert!(!is_reserved_name(""));
    }

    #[test]
    fn is_reserved_name_is_case_sensitive() {
        assert!(!is_reserved_name("foo.WHITEOUT"));
        assert!(!is_reserved_name("foo.Sha3"));
    }

    #[test]
    fn is_reserved_name_only_matches_real_suffix() {
        // ".whiteout" inside the name (not at the end) is not a match.
        assert!(!is_reserved_name("whiteout.foo"));
        assert!(!is_reserved_name("foo.whiteout.tmp"));
    }

    #[test]
    fn matched_reserved_suffix_returns_first_hit() {
        assert_eq!(matched_reserved_suffix("foo.whiteout"), Some(".whiteout"));
        assert_eq!(matched_reserved_suffix("foo.opaque"), Some(".opaque"));
        assert_eq!(matched_reserved_suffix("foo.sha3"), Some(".sha3"));
        assert_eq!(matched_reserved_suffix("foo.sig"), Some(".sig"));
        assert_eq!(matched_reserved_suffix("foo.onnx"), None);
    }

    #[test]
    fn reject_if_reserved_returns_error_with_suffix() {
        let e = reject_if_reserved("foo.whiteout").unwrap_err();
        if let OverlayError::ReservedSuffix(s) = e {
            assert_eq!(s, ".whiteout");
        } else {
            panic!("expected ReservedSuffix error, got {:?}", e);
        }
    }

    #[test]
    fn reject_if_reserved_passes_normal_name() {
        assert!(reject_if_reserved("model.onnx").is_ok());
        assert!(reject_if_reserved("foo").is_ok());
    }
}
