// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Clean-room `#![no_std]` TOML 1.0 subset parser and serialiser.
//!
//! Spec context: the SmallAIOS clean-room rule (no production
//! third-party deps in the kernel) precludes the upstream `toml` crate
//! at runtime. The `mgmt` config surface needs only a subset of the
//! TOML 1.0 grammar, so this module implements the pieces we use and
//! rejects everything else with a stable error.
//!
//! ## Subset (v1)
//!
//! Supported:
//!
//! - Top-level key/value pairs.
//! - `[table]` and `[sub.table]` headers (dotted paths).
//! - String values: basic (`"..."`) and literal (`'...'`).
//! - Integer values: `123`, `-1`, `0`, `+0`. Underscores allowed.
//! - Boolean values: `true` / `false`.
//! - Inline arrays of primitives: `[1, 2, 3]`, `["a", "b"]`.
//! - Dotted keys on the LHS of a value (`a.b.c = 1`).
//! - `# line comments` and trailing whitespace.
//!
//! Out of v1 scope (rejected with [`Error`]):
//!
//! - Multi-line strings (`"""..."""`, `'''...'''`).
//! - Date/time values.
//! - Floating-point values.
//! - Inline tables (`{ a = 1, b = 2 }`).
//! - Arrays of tables (`[[fruits]]`).
//! - Hexadecimal / binary / octal integers.
//!
//! Production callers SHALL author config files within the subset.
//! Files exceeding the subset SHALL produce a parse error and the
//! file SHALL be treated as corrupt by the loader.
//!
//! ## API
//!
//! - [`parse`] turns a `&str` into a tree of [`TomlValue`].
//! - [`serialize`] turns a [`TomlTable`] back into a deterministic
//!   `String`. Round-trip is exact for any value in-subset.
//!
//! ## Determinism
//!
//! [`serialize`] visits keys in BTree order (alphabetical within a
//! table) so re-serialising a parsed file produces a stable byte
//! sequence — required for diff-able config commits.

#![allow(clippy::result_large_err)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

pub mod lex;
pub mod parse;
pub mod serialize;

pub use lex::Token;
pub use parse::parse;
pub use serialize::serialize;

// ─── AST ─────────────────────────────────────────────────────────────────────

/// Typed TOML value. Variants are restricted to the subset declared in
/// the module-level docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TomlValue {
    /// `true` or `false`.
    Bool(bool),
    /// Decimal integer; underscores stripped during parse.
    Integer(i64),
    /// String — basic or literal; both decode to the same variant.
    String(String),
    /// Inline array of primitive [`TomlValue`]s. The parser rejects
    /// nested arrays-of-tables.
    Array(Vec<TomlValue>),
    /// Sub-table (a `[header]` block or a value that expands a dotted
    /// LHS into a table).
    Table(TomlTable),
}

/// A TOML table — a `BTreeMap<key, value>` so keys serialise in stable
/// order and round-trips are byte-identical for in-subset input.
pub type TomlTable = BTreeMap<String, TomlValue>;

impl TomlValue {
    /// Variant tag for diagnostics (`"bool"`, `"integer"`, `"string"`,
    /// `"array"`, `"table"`).
    pub fn type_tag(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::Integer(_) => "integer",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::Table(_) => "table",
        }
    }

    /// Try to extract a `bool`.
    pub fn as_bool(&self) -> Result<bool, Error> {
        match self {
            Self::Bool(b) => Ok(*b),
            _ => Err(Error::Type {
                expected: "bool",
                got: self.type_tag(),
            }),
        }
    }

    /// Try to extract an `i64`.
    pub fn as_integer(&self) -> Result<i64, Error> {
        match self {
            Self::Integer(n) => Ok(*n),
            _ => Err(Error::Type {
                expected: "integer",
                got: self.type_tag(),
            }),
        }
    }

    /// Try to extract a string.
    pub fn as_str(&self) -> Result<&str, Error> {
        match self {
            Self::String(s) => Ok(s.as_str()),
            _ => Err(Error::Type {
                expected: "string",
                got: self.type_tag(),
            }),
        }
    }

    /// Try to borrow a `&Table`.
    pub fn as_table(&self) -> Result<&TomlTable, Error> {
        match self {
            Self::Table(t) => Ok(t),
            _ => Err(Error::Type {
                expected: "table",
                got: self.type_tag(),
            }),
        }
    }
}

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Parse / serialise error.
///
/// Errors carry a `&'static str` reason rather than a heap-allocated
/// `String` so the error type is `Copy`-friendly for downstream
/// callers and stable across allocation regimes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A token was rejected by the lexer (e.g. invalid escape, EOF in
    /// string).
    Lex {
        /// 1-based line number where the lexer gave up.
        line: u32,
        /// Stable static reason.
        reason: &'static str,
    },
    /// The parser encountered a token sequence the v1 subset doesn't
    /// support.
    Parse {
        /// 1-based line number where the parser gave up.
        line: u32,
        /// Stable static reason.
        reason: &'static str,
    },
    /// The proposed key collides with an existing one in the same
    /// table — TOML 1.0 forbids redefinition of a value.
    DuplicateKey {
        /// Dotted path of the duplicate.
        path: String,
    },
    /// The supplied value's type doesn't match what the caller asked
    /// for (used by [`TomlValue::as_bool`] etc.).
    Type {
        /// Expected variant tag.
        expected: &'static str,
        /// Actual variant tag.
        got: &'static str,
    },
    /// A feature outside the v1 subset was used (e.g. multi-line
    /// strings, dates). The reason names the rejected feature.
    Unsupported(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lex { line, reason } => write!(f, "toml: lex error at line {line}: {reason}"),
            Self::Parse { line, reason } => write!(f, "toml: parse error at line {line}: {reason}"),
            Self::DuplicateKey { path } => write!(f, "toml: duplicate key: {path}"),
            Self::Type { expected, got } => {
                write!(f, "toml: expected {expected}, got {got}")
            }
            Self::Unsupported(reason) => write!(f, "toml: unsupported: {reason}"),
        }
    }
}

/// Convenience: stitch two dotted segments into one dotted path with
/// proper escape so duplicate-key errors are unambiguous.
pub(crate) fn join_path(prefix: &str, segment: &str) -> String {
    if prefix.is_empty() {
        segment.to_string()
    } else {
        let mut s = String::with_capacity(prefix.len() + 1 + segment.len());
        s.push_str(prefix);
        s.push('.');
        s.push_str(segment);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_tags() {
        assert_eq!(TomlValue::Bool(true).type_tag(), "bool");
        assert_eq!(TomlValue::Integer(0).type_tag(), "integer");
        assert_eq!(TomlValue::String(String::new()).type_tag(), "string");
        assert_eq!(TomlValue::Array(Vec::new()).type_tag(), "array");
        assert_eq!(TomlValue::Table(TomlTable::new()).type_tag(), "table");
    }

    #[test]
    fn extractors_match_variant() {
        assert_eq!(TomlValue::Bool(true).as_bool(), Ok(true));
        assert_eq!(TomlValue::Integer(42).as_integer(), Ok(42));
        assert_eq!(TomlValue::String("x".to_string()).as_str(), Ok("x"));
    }

    #[test]
    fn extractors_diagnose_mismatch() {
        match TomlValue::Bool(true).as_integer() {
            Err(Error::Type { expected, got }) => {
                assert_eq!(expected, "integer");
                assert_eq!(got, "bool");
            }
            _ => panic!("expected Error::Type"),
        }
    }

    #[test]
    fn join_path_empty_prefix() {
        assert_eq!(join_path("", "a"), "a");
        assert_eq!(join_path("a", "b"), "a.b");
        assert_eq!(join_path("a.b", "c"), "a.b.c");
    }

    #[test]
    fn error_display_includes_line() {
        let e = Error::Parse {
            line: 3,
            reason: "expected '='",
        };
        let s = alloc::format!("{}", e);
        assert!(s.contains("line 3"));
        assert!(s.contains("expected '='"));
    }
}
