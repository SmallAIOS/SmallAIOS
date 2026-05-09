// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic TOML serialiser for the v1 subset.
//!
//! Produces a `String` that re-parses (via [`super::parse`]) to a
//! byte-identical [`super::TomlTable`]. Within each table, keys are
//! emitted in [`alloc::collections::BTreeMap`] iteration order
//! (alphabetical) so output is stable across runs.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use super::{Error, TomlTable, TomlValue};

/// Serialise `table` into a TOML string.
///
/// Layout:
///
/// 1. Top-level scalars first, in alphabetical order.
/// 2. One blank line then `[header]` per nested table, recursively.
///
/// This nests deterministically so a 30-field config diffs cleanly
/// in `git`.
pub fn serialize(table: &TomlTable) -> Result<String, Error> {
    let mut out = String::new();
    serialise_table(&mut out, "", table)?;
    Ok(out)
}

fn serialise_table(out: &mut String, header: &str, table: &TomlTable) -> Result<(), Error> {
    // Pass 1: scalars (and arrays) at this header. Tables get their
    // own header lines later.
    let mut sub_tables: Vec<(&String, &TomlTable)> = Vec::new();
    for (k, v) in table.iter() {
        match v {
            TomlValue::Table(t) => sub_tables.push((k, t)),
            _ => {
                serialise_kv(out, k, v)?;
            }
        }
    }
    // Pass 2: sub-tables.
    for (name, sub) in sub_tables {
        let new_header = if header.is_empty() {
            name.clone()
        } else {
            super::join_path(header, name)
        };
        if !out.is_empty() && !out.ends_with("\n\n") {
            // Ensure exactly one blank line before each header.
            if !out.ends_with('\n') {
                out.push('\n');
            }
            if !out.is_empty() {
                out.push('\n');
            }
        }
        let _ = writeln!(out, "[{}]", new_header);
        serialise_table(out, &new_header, sub)?;
    }
    Ok(())
}

fn serialise_kv(out: &mut String, key: &str, value: &TomlValue) -> Result<(), Error> {
    let kbuf = serialise_key(key);
    let vbuf = serialise_value(value)?;
    let _ = writeln!(out, "{} = {}", kbuf, vbuf);
    Ok(())
}

fn serialise_key(key: &str) -> String {
    // TOML 1.0: bare keys are `[A-Za-z0-9_-]+`. If the key is bare we
    // emit it bare; otherwise quote.
    if !key.is_empty() && key.chars().all(is_bare_key_char) {
        key.into()
    } else {
        let mut s = String::with_capacity(key.len() + 2);
        s.push('"');
        // Conservatively escape the same set the lexer accepts.
        for c in key.chars() {
            match c {
                '\\' => s.push_str("\\\\"),
                '"' => s.push_str("\\\""),
                '\n' => s.push_str("\\n"),
                '\t' => s.push_str("\\t"),
                '\r' => s.push_str("\\r"),
                _ => s.push(c),
            }
        }
        s.push('"');
        s
    }
}

fn is_bare_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn serialise_value(value: &TomlValue) -> Result<String, Error> {
    match value {
        TomlValue::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        TomlValue::Integer(n) => Ok(format!("{n}")),
        TomlValue::String(s) => Ok(serialise_string(s)),
        TomlValue::Array(items) => {
            let mut out = String::from("[");
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                if matches!(v, TomlValue::Array(_) | TomlValue::Table(_)) {
                    return Err(Error::Unsupported("nested array or inline table"));
                }
                out.push_str(&serialise_value(v)?);
            }
            out.push(']');
            Ok(out)
        }
        TomlValue::Table(_) => Err(Error::Unsupported("inline table value")),
    }
}

fn serialise_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toml::{parse, TomlTable, TomlValue};
    use alloc::string::ToString;

    #[test]
    fn empty_table_serialises_to_empty_string() {
        let t = TomlTable::new();
        assert_eq!(serialize(&t).unwrap(), "");
    }

    #[test]
    fn flat_pairs_round_trip() {
        let mut t = TomlTable::new();
        t.insert("name".into(), TomlValue::String("alice".to_string()));
        t.insert("age".into(), TomlValue::Integer(30));
        let s = serialize(&t).unwrap();
        let back = parse(&s).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn keys_sorted_alphabetically() {
        let mut t = TomlTable::new();
        t.insert("zebra".into(), TomlValue::Integer(1));
        t.insert("alpha".into(), TomlValue::Integer(2));
        t.insert("middle".into(), TomlValue::Integer(3));
        let s = serialize(&t).unwrap();
        let lines: alloc::vec::Vec<&str> = s.lines().collect();
        // First three non-empty lines are alpha/middle/zebra.
        assert!(lines[0].starts_with("alpha"));
        assert!(lines[1].starts_with("middle"));
        assert!(lines[2].starts_with("zebra"));
    }

    #[test]
    fn header_table_serialises_with_brackets() {
        let mut sub = TomlTable::new();
        sub.insert("k".into(), TomlValue::Integer(1));
        let mut t = TomlTable::new();
        t.insert("section".into(), TomlValue::Table(sub));
        let s = serialize(&t).unwrap();
        assert!(s.contains("[section]"));
        assert!(s.contains("k = 1"));
    }

    #[test]
    fn nested_header_serialises_with_dotted_path() {
        let mut deep = TomlTable::new();
        deep.insert("x".into(), TomlValue::Integer(7));
        let mut mid = TomlTable::new();
        mid.insert("c".into(), TomlValue::Table(deep));
        let mut top = TomlTable::new();
        top.insert(
            "a".into(),
            TomlValue::Table({
                let mut t = TomlTable::new();
                t.insert(
                    "b".into(),
                    TomlValue::Table({
                        let mut q = TomlTable::new();
                        q.insert("x".into(), TomlValue::Integer(7));
                        q
                    }),
                );
                t
            }),
        );
        let _ = mid;
        let s = serialize(&top).unwrap();
        assert!(s.contains("[a.b]"));
        assert!(s.contains("x = 7"));
    }

    #[test]
    fn array_round_trips() {
        let mut t = TomlTable::new();
        t.insert(
            "xs".into(),
            TomlValue::Array(alloc::vec![
                TomlValue::Integer(1),
                TomlValue::Integer(2),
                TomlValue::Integer(3),
            ]),
        );
        let s = serialize(&t).unwrap();
        assert_eq!(s, "xs = [1, 2, 3]\n");
        let back = parse(&s).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn string_escapes_round_trip() {
        let mut t = TomlTable::new();
        t.insert("s".into(), TomlValue::String("a\nb\t\"c".to_string()));
        let s = serialize(&t).unwrap();
        let back = parse(&s).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn quoted_key_when_non_bare() {
        let mut t = TomlTable::new();
        t.insert("dotted.key".into(), TomlValue::Integer(1));
        let s = serialize(&t).unwrap();
        // The serialiser must quote the key so a re-parse doesn't
        // treat the `.` as a path separator.
        assert!(s.starts_with("\"dotted.key\""));
        let back = parse(&s).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn deterministic_output_across_invocations() {
        let mut t = TomlTable::new();
        t.insert("b".into(), TomlValue::Integer(2));
        t.insert("a".into(), TomlValue::Integer(1));
        let s1 = serialize(&t).unwrap();
        let s2 = serialize(&t).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn negative_integer_round_trips() {
        let mut t = TomlTable::new();
        t.insert("n".into(), TomlValue::Integer(-42));
        let s = serialize(&t).unwrap();
        let back = parse(&s).unwrap();
        assert_eq!(back, t);
    }
}
