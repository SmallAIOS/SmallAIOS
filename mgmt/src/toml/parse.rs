// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TOML 1.0 (v1 subset) parser.
//!
//! Consumes the [`super::lex::Spanned`] token stream produced by
//! [`super::lex::lex`] and assembles a [`super::TomlTable`].

use alloc::string::String;
use alloc::vec::Vec;

use super::lex::{lex, Spanned, Token};
use super::{join_path, Error, TomlTable, TomlValue};

/// Parse `src` (a `&str` slice of in-subset TOML) into a top-level
/// [`TomlTable`].
pub fn parse(src: &str) -> Result<TomlTable, Error> {
    let tokens = lex(src)?;
    let mut p = Parser {
        tokens,
        pos: 0,
        out: TomlTable::new(),
        current_header: Vec::new(),
    };
    p.run()?;
    Ok(p.out)
}

struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
    out: TomlTable,
    /// Current `[a.b.c]` header, as a stack of segments. Empty means
    /// "top-level".
    current_header: Vec<String>,
}

impl Parser {
    fn run(&mut self) -> Result<(), Error> {
        while !self.is_eof() {
            // Skip blank lines.
            if matches!(self.peek_token(), Some(Token::Newline)) {
                self.advance();
                continue;
            }
            if matches!(self.peek_token(), Some(Token::LBracket)) {
                self.parse_header()?;
            } else {
                self.parse_kv()?;
            }
        }
        Ok(())
    }

    fn parse_header(&mut self) -> Result<(), Error> {
        let line = self.peek_line();
        self.expect(&Token::LBracket)?;
        // Reject `[[arrays.of.tables]]` — out of v1 scope. The lexer
        // emits two LBracket tokens for `[[`, so detecting a second
        // LBracket here is the trigger.
        if matches!(self.peek_token(), Some(Token::LBracket)) {
            return Err(Error::Unsupported("array of tables ([[ ]])"));
        }
        let mut segments = Vec::new();
        loop {
            let key = self.expect_bare_key_or_string()?;
            segments.push(key);
            if matches!(self.peek_token(), Some(Token::Dot)) {
                self.advance();
                continue;
            }
            break;
        }
        self.expect(&Token::RBracket)?;
        self.expect_newline_or_eof(line)?;
        self.current_header = segments.clone();

        // Materialise the empty sub-table (so subsequent kv parsing has
        // a place to put values). If the path collides with a non-table
        // existing value, reject with a `Parse` error.
        ensure_table(&mut self.out, &segments)?;
        Ok(())
    }

    fn parse_kv(&mut self) -> Result<(), Error> {
        let line = self.peek_line();
        let mut key_segments = Vec::new();
        loop {
            let k = self.expect_bare_key_or_string()?;
            key_segments.push(k);
            if matches!(self.peek_token(), Some(Token::Dot)) {
                self.advance();
                continue;
            }
            break;
        }
        self.expect(&Token::Equals)?;
        let value = self.parse_value(line)?;
        self.expect_newline_or_eof(line)?;

        // Resolve the absolute path = current_header / key_segments.
        let mut full = self.current_header.clone();
        full.extend(key_segments);
        insert_value(&mut self.out, &full, value)?;
        Ok(())
    }

    fn parse_value(&mut self, line: u32) -> Result<TomlValue, Error> {
        match self.peek_token() {
            Some(Token::Bool(b)) => {
                let v = TomlValue::Bool(*b);
                self.advance();
                Ok(v)
            }
            Some(Token::Integer(n)) => {
                let v = TomlValue::Integer(*n);
                self.advance();
                Ok(v)
            }
            Some(Token::String(s)) => {
                let v = TomlValue::String(s.clone());
                self.advance();
                Ok(v)
            }
            Some(Token::LBracket) => self.parse_array(line),
            _ => Err(Error::Parse {
                line,
                reason: "expected value",
            }),
        }
    }

    fn parse_array(&mut self, _line: u32) -> Result<TomlValue, Error> {
        let line = self.peek_line();
        self.expect(&Token::LBracket)?;
        let mut items = Vec::new();
        loop {
            // Allow newlines inside the array brackets so files can
            // wrap long lists.
            while matches!(self.peek_token(), Some(Token::Newline)) {
                self.advance();
            }
            if matches!(self.peek_token(), Some(Token::RBracket)) {
                self.advance();
                return Ok(TomlValue::Array(items));
            }
            let v = self.parse_value(line)?;
            // v1 subset rejects nested arrays/tables inside arrays.
            if matches!(v, TomlValue::Array(_) | TomlValue::Table(_)) {
                return Err(Error::Unsupported("nested array or inline table"));
            }
            items.push(v);
            while matches!(self.peek_token(), Some(Token::Newline)) {
                self.advance();
            }
            match self.peek_token() {
                Some(Token::Comma) => {
                    self.advance();
                }
                Some(Token::RBracket) => {
                    self.advance();
                    return Ok(TomlValue::Array(items));
                }
                _ => {
                    return Err(Error::Parse {
                        line,
                        reason: "expected ',' or ']' in array",
                    });
                }
            }
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────

    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|s| &s.token)
    }

    fn peek_line(&self) -> u32 {
        self.tokens.get(self.pos).map(|s| s.line).unwrap_or(0)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn expect(&mut self, want: &Token) -> Result<(), Error> {
        let line = self.peek_line();
        match self.peek_token() {
            Some(t) if t == want => {
                self.advance();
                Ok(())
            }
            _ => Err(Error::Parse {
                line,
                reason: "unexpected token",
            }),
        }
    }

    fn expect_bare_key_or_string(&mut self) -> Result<String, Error> {
        let line = self.peek_line();
        match self.peek_token().cloned() {
            Some(Token::BareKey(k)) => {
                self.advance();
                Ok(k)
            }
            Some(Token::String(k)) => {
                self.advance();
                Ok(k)
            }
            _ => Err(Error::Parse {
                line,
                reason: "expected key",
            }),
        }
    }

    fn expect_newline_or_eof(&mut self, line: u32) -> Result<(), Error> {
        match self.peek_token() {
            Some(Token::Newline) => {
                self.advance();
                Ok(())
            }
            None => Ok(()),
            _ => Err(Error::Parse {
                line,
                reason: "expected newline",
            }),
        }
    }
}

/// Walk the path inside `root`, materialising sub-tables as needed.
/// Returns an [`Error::DuplicateKey`] if any path segment collides
/// with an existing non-table value.
fn ensure_table(root: &mut TomlTable, path: &[String]) -> Result<(), Error> {
    if path.is_empty() {
        return Ok(());
    }
    let mut cursor = root;
    let mut walked = String::new();
    for seg in path {
        walked = join_path(&walked, seg);
        // Promote the cursor through the entry API so we never hold
        // two mutable borrows.
        let entry = cursor
            .entry(seg.clone())
            .or_insert_with(|| TomlValue::Table(TomlTable::new()));
        match entry {
            TomlValue::Table(t) => cursor = t,
            _ => {
                return Err(Error::DuplicateKey {
                    path: walked.clone(),
                });
            }
        }
    }
    Ok(())
}

/// Insert `value` at `path` in `root`. The leaf segment must not
/// already exist (TOML 1.0 forbids redefining a value).
fn insert_value(root: &mut TomlTable, path: &[String], value: TomlValue) -> Result<(), Error> {
    if path.is_empty() {
        return Err(Error::Parse {
            line: 0,
            reason: "empty key path",
        });
    }
    let (leaf, prefix) = path.split_last().expect("non-empty path");
    ensure_table(root, prefix)?;
    let mut cursor = root;
    let mut walked = String::new();
    for seg in prefix {
        walked = join_path(&walked, seg);
        match cursor.get_mut(seg) {
            Some(TomlValue::Table(t)) => cursor = t,
            _ => {
                return Err(Error::DuplicateKey {
                    path: walked.clone(),
                });
            }
        }
    }
    let dotted = join_path(&walked, leaf);
    if cursor.contains_key(leaf) {
        return Err(Error::DuplicateKey { path: dotted });
    }
    cursor.insert(leaf.clone(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn empty_input_yields_empty_table() {
        let t = parse("").unwrap();
        assert!(t.is_empty());
    }

    #[test]
    fn comment_only_input_yields_empty_table() {
        let t = parse("# nothing\n# really\n").unwrap();
        assert!(t.is_empty());
    }

    #[test]
    fn flat_kv_pair() {
        let t = parse("name = \"alice\"\nage = 30\n").unwrap();
        assert_eq!(t.get("name"), Some(&TomlValue::String("alice".to_string())));
        assert_eq!(t.get("age"), Some(&TomlValue::Integer(30)));
    }

    #[test]
    fn header_table() {
        let t = parse("[idle]\nroot_minutes = 15\n").unwrap();
        let idle = t.get("idle").unwrap().as_table().unwrap();
        assert_eq!(idle.get("root_minutes"), Some(&TomlValue::Integer(15)));
    }

    #[test]
    fn nested_header_table() {
        let t = parse("[a.b.c]\nx = 1\n").unwrap();
        let a = t.get("a").unwrap().as_table().unwrap();
        let b = a.get("b").unwrap().as_table().unwrap();
        let c = b.get("c").unwrap().as_table().unwrap();
        assert_eq!(c.get("x"), Some(&TomlValue::Integer(1)));
    }

    #[test]
    fn dotted_key_under_header() {
        let t = parse("[idle]\nlevels.root = 15\n").unwrap();
        let idle = t.get("idle").unwrap().as_table().unwrap();
        let levels = idle.get("levels").unwrap().as_table().unwrap();
        assert_eq!(levels.get("root"), Some(&TomlValue::Integer(15)));
    }

    #[test]
    fn multiple_headers_isolated() {
        let t = parse("[idle]\nroot_minutes = 15\n\n[password]\nmin_length = 12\n").unwrap();
        assert!(t.contains_key("idle"));
        assert!(t.contains_key("password"));
    }

    #[test]
    fn array_of_primitives() {
        let t = parse("xs = [1, 2, 3]\n").unwrap();
        match t.get("xs") {
            Some(TomlValue::Array(items)) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], TomlValue::Integer(1));
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn empty_array() {
        let t = parse("xs = []\n").unwrap();
        match t.get("xs") {
            Some(TomlValue::Array(items)) => assert!(items.is_empty()),
            _ => panic!("expected empty array"),
        }
    }

    #[test]
    fn duplicate_key_rejected() {
        let err = parse("a = 1\na = 2\n").unwrap_err();
        assert!(matches!(err, Error::DuplicateKey { .. }));
    }

    #[test]
    fn duplicate_dotted_key_rejected() {
        let err = parse("a.b = 1\na.b = 2\n").unwrap_err();
        assert!(matches!(err, Error::DuplicateKey { .. }));
    }

    #[test]
    fn redefining_table_as_value_rejected() {
        // a is a table from `a.b = 1`, then `a = 2` tries to overwrite.
        let err = parse("a.b = 1\na = 2\n").unwrap_err();
        assert!(matches!(err, Error::DuplicateKey { .. }));
    }

    #[test]
    fn array_of_tables_rejected() {
        let err = parse("[[fruits]]\nname = 'apple'\n").unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn missing_equals_diagnoses() {
        let err = parse("a 1\n").unwrap_err();
        assert!(matches!(err, Error::Parse { .. }));
    }

    #[test]
    fn line_number_in_error_points_to_offender() {
        let err = parse("a = 1\nb 2\n").unwrap_err();
        match err {
            Error::Parse { line, .. } => assert_eq!(line, 2),
            other => panic!("expected Parse error, got {:?}", other),
        }
    }

    #[test]
    fn quoted_key_accepted_as_segment() {
        let t = parse("\"hyphen-key\" = 1\n").unwrap();
        assert_eq!(t.get("hyphen-key"), Some(&TomlValue::Integer(1)));
    }

    #[test]
    fn array_with_mixed_string_and_int_accepted() {
        // TOML 1.0 rejects mixed-type arrays, but our v1 subset is
        // permissive; the loader on top enforces typed keys.
        let t = parse("xs = [1, \"a\"]\n").unwrap();
        match t.get("xs") {
            Some(TomlValue::Array(_)) => {}
            _ => panic!("expected array"),
        }
    }
}
