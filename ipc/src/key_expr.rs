// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Key expression parser and wildcard matcher.
//!
//! Zenoh-style hierarchical key expressions with `/` separators.
//! Supports `*` (single-level wildcard) and `**` (multi-level wildcard).

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};

use crate::IpcError;

/// A validated Zenoh-style key expression.
///
/// Key expressions are `/`-separated hierarchical paths where each segment
/// is either a literal identifier (alphanumeric, `_`, `-`), a single-level
/// wildcard (`*`), or a multi-level wildcard (`**`).
///
/// # Examples
///
/// - `"sensor/temp/room1"` — concrete key
/// - `"sensor/*/room1"` — matches any single segment in the middle
/// - `"sensor/**"` — matches everything under `sensor/`
#[derive(Clone, Debug)]
pub struct KeyExpr {
    expr: String,
}

impl PartialEq for KeyExpr {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

impl Eq for KeyExpr {}

impl Hash for KeyExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.hash(state);
    }
}

impl fmt::Display for KeyExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

/// Check whether a single segment is valid.
///
/// A segment must be one of:
/// - `*` (single-level wildcard)
/// - `**` (multi-level wildcard)
/// - A non-empty string of alphanumeric characters, `_`, or `-`
fn is_valid_segment(seg: &str) -> bool {
    if seg.is_empty() {
        return false;
    }
    if seg == "*" || seg == "**" {
        return true;
    }
    seg.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

impl KeyExpr {
    /// Create a new `KeyExpr` after validating the expression string.
    ///
    /// # Validation rules
    ///
    /// - Must be non-empty.
    /// - No double-slash (`//`).
    /// - No trailing slash (except the root `"/"`).
    /// - Each segment must be `*`, `**`, or consist of alphanumeric/`_`/`-` characters.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::InvalidKeyExpression`] if any rule is violated.
    pub fn new(expr: &str) -> Result<KeyExpr, IpcError> {
        // Non-empty check
        if expr.is_empty() {
            return Err(IpcError::InvalidKeyExpression);
        }

        // Special-case: root "/"
        if expr == "/" {
            return Ok(KeyExpr {
                expr: String::from(expr),
            });
        }

        // No double-slash
        if expr.contains("//") {
            return Err(IpcError::InvalidKeyExpression);
        }

        // No trailing slash
        if expr.ends_with('/') {
            return Err(IpcError::InvalidKeyExpression);
        }

        // No leading slash (we already handled "/")
        let to_validate = if let Some(stripped) = expr.strip_prefix('/') {
            stripped
        } else {
            expr
        };

        // Validate each segment
        for seg in to_validate.split('/') {
            if !is_valid_segment(seg) {
                return Err(IpcError::InvalidKeyExpression);
            }
        }

        Ok(KeyExpr {
            expr: String::from(expr),
        })
    }

    /// Split the key expression into its `/`-separated segments.
    pub fn segments(&self) -> Vec<&str> {
        if self.expr == "/" {
            return Vec::new();
        }
        let s = if self.expr.starts_with('/') {
            &self.expr[1..]
        } else {
            &self.expr
        };
        s.split('/').collect()
    }

    /// Returns `true` if this key expression contains any wildcard (`*` or `**`).
    pub fn is_wild(&self) -> bool {
        self.segments().iter().any(|s| *s == "*" || *s == "**")
    }

    /// Return the raw string form of the key expression.
    pub fn as_str(&self) -> &str {
        &self.expr
    }

    /// Test whether this key expression matches another.
    ///
    /// Wildcard semantics:
    /// - `*` matches exactly one segment.
    /// - `**` matches zero or more segments.
    /// - Literal segments must match exactly.
    ///
    /// Matching is symmetric: `a.matches(b)` iff `b.matches(a)` for concrete
    /// keys against patterns. When both sides contain wildcards, we check
    /// whether any concrete key could satisfy both.
    pub fn matches(&self, other: &KeyExpr) -> bool {
        let a = self.segments();
        let b = other.segments();
        segments_match(&a, &b)
    }

    /// Test whether two key expressions could match the same concrete key.
    ///
    /// This is the "intersection" check: returns `true` if there exists some
    /// concrete (wildcard-free) key expression that would match both `self`
    /// and `other`.
    pub fn intersects(&self, other: &KeyExpr) -> bool {
        let a = self.segments();
        let b = other.segments();
        segments_intersect(&a, &b)
    }
}

/// Recursive wildcard matching between two segment lists.
///
/// `*` on either side matches exactly one segment on the other side.
/// `**` on either side matches zero or more segments on the other side.
fn segments_match(a: &[&str], b: &[&str]) -> bool {
    match (a.first(), b.first()) {
        // Both exhausted: match!
        (None, None) => true,
        // One side has `**`, which can match zero remaining segments on the other.
        (Some(&"**"), None) => segments_match(&a[1..], b),
        (None, Some(&"**")) => segments_match(a, &b[1..]),
        // One side exhausted, other is not `**`: no match.
        (None, Some(_)) | (Some(_), None) => false,
        // `**` on left: try matching zero segments (skip `**`) or one+ (skip one from b).
        (Some(&"**"), Some(_)) => segments_match(&a[1..], b) || segments_match(a, &b[1..]),
        // `**` on right: symmetric.
        (Some(_), Some(&"**")) => segments_match(a, &b[1..]) || segments_match(&a[1..], b),
        // `*` matches any single segment.
        (Some(&"*"), Some(_)) | (Some(_), Some(&"*")) => segments_match(&a[1..], &b[1..]),
        // Literal comparison.
        (Some(sa), Some(sb)) => {
            if sa == sb {
                segments_match(&a[1..], &b[1..])
            } else {
                false
            }
        }
    }
}

/// Check intersection: could any concrete key match both segment lists?
///
/// This is very similar to `segments_match` but treats wildcards on both
/// sides as flexible placeholders.
fn segments_intersect(a: &[&str], b: &[&str]) -> bool {
    // Intersection uses the same logic as matching for our purposes,
    // since both `*` and `**` can be "filled in" with concrete values.
    segments_match(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // === Valid expressions ===

    #[test]
    fn valid_simple_path() {
        let ke = KeyExpr::new("a/b/c").unwrap();
        assert_eq!(ke.as_str(), "a/b/c");
    }

    #[test]
    fn valid_single_segment() {
        let ke = KeyExpr::new("hello").unwrap();
        assert_eq!(ke.as_str(), "hello");
    }

    #[test]
    fn valid_single_wildcard() {
        let ke = KeyExpr::new("sensor/temp/*").unwrap();
        assert!(ke.is_wild());
    }

    #[test]
    fn valid_double_wildcard() {
        let ke = KeyExpr::new("data/**").unwrap();
        assert!(ke.is_wild());
    }

    #[test]
    fn valid_star_only() {
        let ke = KeyExpr::new("*").unwrap();
        assert!(ke.is_wild());
    }

    #[test]
    fn valid_double_star_only() {
        let ke = KeyExpr::new("**").unwrap();
        assert!(ke.is_wild());
    }

    #[test]
    fn valid_with_dashes_and_underscores() {
        let ke = KeyExpr::new("my-sensor/temp_1/data").unwrap();
        assert!(!ke.is_wild());
    }

    // === Invalid expressions ===

    #[test]
    fn invalid_empty() {
        assert_eq!(KeyExpr::new(""), Err(IpcError::InvalidKeyExpression));
    }

    #[test]
    fn invalid_double_slash() {
        assert_eq!(KeyExpr::new("a//b"), Err(IpcError::InvalidKeyExpression));
    }

    #[test]
    fn invalid_trailing_slash() {
        assert_eq!(KeyExpr::new("a/b/"), Err(IpcError::InvalidKeyExpression));
    }

    #[test]
    fn invalid_space_in_segment() {
        assert_eq!(KeyExpr::new("a/b c/d"), Err(IpcError::InvalidKeyExpression));
    }

    #[test]
    fn invalid_special_characters() {
        assert_eq!(KeyExpr::new("a/b@c/d"), Err(IpcError::InvalidKeyExpression));
    }

    // === Segments parsing ===

    #[test]
    fn segments_simple() {
        let ke = KeyExpr::new("a/b/c").unwrap();
        assert_eq!(ke.segments(), vec!["a", "b", "c"]);
    }

    #[test]
    fn segments_with_wildcards() {
        let ke = KeyExpr::new("a/*/c/**").unwrap();
        assert_eq!(ke.segments(), vec!["a", "*", "c", "**"]);
    }

    #[test]
    fn segments_single() {
        let ke = KeyExpr::new("hello").unwrap();
        assert_eq!(ke.segments(), vec!["hello"]);
    }

    // === Matching ===

    #[test]
    fn match_exact() {
        let a = KeyExpr::new("a/b/c").unwrap();
        let b = KeyExpr::new("a/b/c").unwrap();
        assert!(a.matches(&b));
    }

    #[test]
    fn match_exact_no_match() {
        let a = KeyExpr::new("a/b/c").unwrap();
        let b = KeyExpr::new("a/b/d").unwrap();
        assert!(!a.matches(&b));
    }

    #[test]
    fn match_single_wildcard() {
        let pattern = KeyExpr::new("sensor/*/temp").unwrap();
        let concrete = KeyExpr::new("sensor/room1/temp").unwrap();
        assert!(pattern.matches(&concrete));
    }

    #[test]
    fn match_single_wildcard_no_match_extra_depth() {
        let pattern = KeyExpr::new("sensor/*/temp").unwrap();
        let concrete = KeyExpr::new("sensor/room1/sub/temp").unwrap();
        assert!(!pattern.matches(&concrete));
    }

    #[test]
    fn match_double_wildcard_tail() {
        let pattern = KeyExpr::new("sensor/**").unwrap();
        let concrete = KeyExpr::new("sensor/room1/temp").unwrap();
        assert!(pattern.matches(&concrete));
    }

    #[test]
    fn match_double_wildcard_matches_zero() {
        let pattern = KeyExpr::new("sensor/**").unwrap();
        let concrete = KeyExpr::new("sensor").unwrap();
        assert!(pattern.matches(&concrete));
    }

    #[test]
    fn match_double_wildcard_middle() {
        let pattern = KeyExpr::new("a/**/z").unwrap();
        let concrete = KeyExpr::new("a/b/c/z").unwrap();
        assert!(pattern.matches(&concrete));
    }

    #[test]
    fn match_double_wildcard_middle_zero() {
        let pattern = KeyExpr::new("a/**/z").unwrap();
        let concrete = KeyExpr::new("a/z").unwrap();
        assert!(pattern.matches(&concrete));
    }

    #[test]
    fn no_match_different_length() {
        let a = KeyExpr::new("a/b").unwrap();
        let b = KeyExpr::new("a/b/c").unwrap();
        assert!(!a.matches(&b));
    }

    // === Intersection ===

    #[test]
    fn intersect_exact_same() {
        let a = KeyExpr::new("a/b/c").unwrap();
        let b = KeyExpr::new("a/b/c").unwrap();
        assert!(a.intersects(&b));
    }

    #[test]
    fn intersect_wildcard_overlap() {
        let a = KeyExpr::new("sensor/*/temp").unwrap();
        let b = KeyExpr::new("sensor/room1/*").unwrap();
        assert!(a.intersects(&b));
    }

    #[test]
    fn intersect_double_wildcard() {
        let a = KeyExpr::new("sensor/**").unwrap();
        let b = KeyExpr::new("sensor/room1/temp").unwrap();
        assert!(a.intersects(&b));
    }

    #[test]
    fn no_intersect_disjoint() {
        let a = KeyExpr::new("sensor/temp").unwrap();
        let b = KeyExpr::new("actuator/motor").unwrap();
        assert!(!a.intersects(&b));
    }

    // === Display, Clone, Eq, Hash ===

    #[test]
    fn display_impl() {
        let ke = KeyExpr::new("a/b/c").unwrap();
        let s = alloc::format!("{}", ke);
        assert_eq!(s, "a/b/c");
    }

    #[test]
    fn clone_and_eq() {
        let a = KeyExpr::new("x/y/z").unwrap();
        let b = a.clone();
        assert_eq!(a, b);
    }
}
