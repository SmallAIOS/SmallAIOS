// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TOML 1.0 (v1 subset) lexer.
//!
//! Converts a `&str` source into a [`Vec<Token>`]. The token stream
//! preserves the line number of every token so the parser can name
//! the source line in its error messages.
//!
//! See [`crate::toml`] for the v1 subset boundary.

use alloc::string::String;
use alloc::vec::Vec;

use super::Error;

/// One lexical token. Span info is reduced to a 1-based `line` count;
/// the parser only needs that for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// Bare key — `[a-zA-Z0-9_-]+` segment.
    BareKey(String),
    /// Quoted string value (basic or literal — both decode to a
    /// runtime `String`).
    String(String),
    /// Decimal integer; the lexer parses sign and underscores and
    /// hands the parser a `i64`.
    Integer(i64),
    /// Boolean literal.
    Bool(bool),
    /// `=` between a key and its value.
    Equals,
    /// `.` between dotted-key segments.
    Dot,
    /// `,` between array elements.
    Comma,
    /// `[` — opens a header or an inline array. The parser
    /// disambiguates from context.
    LBracket,
    /// `]` — closes a header or an inline array.
    RBracket,
    /// `\n` — meaningful in TOML; key/value pairs SHALL terminate at a
    /// newline. The lexer emits one [`Token::Newline`] per line break
    /// so the parser can tell adjacent kv-pairs apart.
    Newline,
}

/// One token plus its source line number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned {
    /// The token.
    pub token: Token,
    /// 1-based source line.
    pub line: u32,
}

/// Lex `src` into a sequence of [`Spanned`] tokens. Comments and pure
/// whitespace runs are dropped. Newlines are preserved as
/// [`Token::Newline`] markers.
pub fn lex(src: &str) -> Result<Vec<Spanned>, Error> {
    let mut out = Vec::new();
    let mut chars = src.chars().peekable();
    let mut line: u32 = 1;

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\r' => {
                chars.next();
            }
            '\n' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Newline,
                    line,
                });
                line += 1;
            }
            '#' => {
                // Comment to EOL.
                while let Some(&c) = chars.peek() {
                    if c == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            '=' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Equals,
                    line,
                });
            }
            '.' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Dot,
                    line,
                });
            }
            ',' => {
                chars.next();
                out.push(Spanned {
                    token: Token::Comma,
                    line,
                });
            }
            '[' => {
                chars.next();
                out.push(Spanned {
                    token: Token::LBracket,
                    line,
                });
            }
            ']' => {
                chars.next();
                out.push(Spanned {
                    token: Token::RBracket,
                    line,
                });
            }
            '"' => {
                let s = lex_basic_string(&mut chars, line)?;
                out.push(Spanned {
                    token: Token::String(s),
                    line,
                });
            }
            '\'' => {
                let s = lex_literal_string(&mut chars, line)?;
                out.push(Spanned {
                    token: Token::String(s),
                    line,
                });
            }
            't' | 'f' => {
                let word = read_word(&mut chars);
                match word.as_str() {
                    "true" => out.push(Spanned {
                        token: Token::Bool(true),
                        line,
                    }),
                    "false" => out.push(Spanned {
                        token: Token::Bool(false),
                        line,
                    }),
                    _ => out.push(Spanned {
                        token: Token::BareKey(word),
                        line,
                    }),
                }
            }
            '-' | '+' | '0'..='9' => {
                let n = lex_integer(&mut chars, line)?;
                out.push(Spanned {
                    token: Token::Integer(n),
                    line,
                });
            }
            _ if is_bare_key_char(c) => {
                let word = read_word(&mut chars);
                out.push(Spanned {
                    token: Token::BareKey(word),
                    line,
                });
            }
            _ => {
                return Err(Error::Lex {
                    line,
                    reason: "unexpected character",
                });
            }
        }
    }

    Ok(out)
}

/// Read a bare-key word — `[a-zA-Z0-9_-]+`. Caller has already
/// confirmed the leading char is a bare-key char.
fn read_word(chars: &mut core::iter::Peekable<core::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&c) = chars.peek() {
        if is_bare_key_char(c) {
            s.push(c);
            chars.next();
        } else {
            break;
        }
    }
    s
}

/// Bare-key alphabet per TOML 1.0: ASCII alphanum, `_`, `-`.
fn is_bare_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Parse a basic (`"..."`) string. v1 subset rejects multi-line strings
/// and only handles the standard escape sequences `\\`, `\"`, `\n`,
/// `\t`, `\r`, `\0`.
fn lex_basic_string(
    chars: &mut core::iter::Peekable<core::str::Chars<'_>>,
    line: u32,
) -> Result<String, Error> {
    // Consume the leading `"`. Reject multi-line strings (`"""..."""`) —
    // v1 out of scope.
    chars.next();
    if matches!(chars.peek(), Some('"')) {
        // Could be `""` (empty) or `"""` (multi-line); look ahead.
        chars.next();
        if matches!(chars.peek(), Some('"')) {
            return Err(Error::Lex {
                line,
                reason: "multi-line basic strings not supported",
            });
        }
        return Ok(String::new());
    }
    let mut out = String::new();
    loop {
        match chars.next() {
            None => {
                return Err(Error::Lex {
                    line,
                    reason: "unterminated basic string",
                });
            }
            Some('"') => return Ok(out),
            Some('\n') => {
                return Err(Error::Lex {
                    line,
                    reason: "newline in basic string",
                });
            }
            Some('\\') => match chars.next() {
                None => {
                    return Err(Error::Lex {
                        line,
                        reason: "unterminated escape",
                    });
                }
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('0') => out.push('\0'),
                Some(_) => {
                    return Err(Error::Lex {
                        line,
                        reason: "unsupported escape sequence",
                    });
                }
            },
            Some(c) => out.push(c),
        }
    }
}

/// Parse a literal (`'...'`) string. No escapes.
fn lex_literal_string(
    chars: &mut core::iter::Peekable<core::str::Chars<'_>>,
    line: u32,
) -> Result<String, Error> {
    // Consume the leading `'`.
    chars.next();
    if matches!(chars.peek(), Some('\'')) {
        chars.next();
        if matches!(chars.peek(), Some('\'')) {
            return Err(Error::Lex {
                line,
                reason: "multi-line literal strings not supported",
            });
        }
        return Ok(String::new());
    }
    let mut out = String::new();
    loop {
        match chars.next() {
            None => {
                return Err(Error::Lex {
                    line,
                    reason: "unterminated literal string",
                });
            }
            Some('\'') => return Ok(out),
            Some('\n') => {
                return Err(Error::Lex {
                    line,
                    reason: "newline in literal string",
                });
            }
            Some(c) => out.push(c),
        }
    }
}

/// Parse a decimal integer with optional leading `+`/`-` and embedded
/// `_` digit separators. Rejects hex/oct/bin (v1 subset) and
/// floating-point.
fn lex_integer(
    chars: &mut core::iter::Peekable<core::str::Chars<'_>>,
    line: u32,
) -> Result<i64, Error> {
    let mut raw = String::new();
    let mut neg = false;
    match chars.peek() {
        Some('-') => {
            chars.next();
            neg = true;
        }
        Some('+') => {
            chars.next();
        }
        _ => {}
    }
    let mut have_digit = false;
    while let Some(&c) = chars.peek() {
        match c {
            '0'..='9' => {
                raw.push(c);
                chars.next();
                have_digit = true;
            }
            '_' => {
                if !have_digit {
                    return Err(Error::Lex {
                        line,
                        reason: "leading underscore in integer",
                    });
                }
                chars.next();
                // The next char SHALL be a digit (TOML disallows
                // trailing or paired underscores).
                match chars.peek() {
                    Some(d) if d.is_ascii_digit() => {}
                    _ => {
                        return Err(Error::Lex {
                            line,
                            reason: "underscore must separate digits",
                        });
                    }
                }
            }
            'x' | 'X' | 'o' | 'O' | 'b' | 'B' => {
                return Err(Error::Lex {
                    line,
                    reason: "non-decimal integer not supported",
                });
            }
            '.' | 'e' | 'E' => {
                return Err(Error::Lex {
                    line,
                    reason: "floating-point not supported",
                });
            }
            _ => break,
        }
    }
    if !have_digit {
        return Err(Error::Lex {
            line,
            reason: "missing digits in integer",
        });
    }
    // Reject leading zero as per TOML 1.0 unless the value is exactly
    // "0".
    if raw.len() > 1 && raw.starts_with('0') {
        return Err(Error::Lex {
            line,
            reason: "leading zero in integer",
        });
    }
    let n: i64 = raw.parse().map_err(|_| Error::Lex {
        line,
        reason: "integer out of range",
    })?;
    Ok(if neg { -n } else { n })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn tokens(src: &str) -> Vec<Token> {
        lex(src).unwrap().into_iter().map(|s| s.token).collect()
    }

    #[test]
    fn key_value_basic() {
        let t = tokens("hello = 7\n");
        assert_eq!(
            t,
            alloc::vec![
                Token::BareKey("hello".to_string()),
                Token::Equals,
                Token::Integer(7),
                Token::Newline,
            ]
        );
    }

    #[test]
    fn comment_dropped() {
        let t = tokens("# top comment\nfoo = 1 # trailing\n");
        assert_eq!(
            t,
            alloc::vec![
                Token::Newline,
                Token::BareKey("foo".to_string()),
                Token::Equals,
                Token::Integer(1),
                Token::Newline,
            ]
        );
    }

    #[test]
    fn header_and_dotted_key() {
        let t = tokens("[a.b]\nx = true\n");
        assert_eq!(
            t,
            alloc::vec![
                Token::LBracket,
                Token::BareKey("a".to_string()),
                Token::Dot,
                Token::BareKey("b".to_string()),
                Token::RBracket,
                Token::Newline,
                Token::BareKey("x".to_string()),
                Token::Equals,
                Token::Bool(true),
                Token::Newline,
            ]
        );
    }

    #[test]
    fn basic_string_with_escapes() {
        let t = tokens("name = \"a\\nb\\t\\\"c\"\n");
        assert_eq!(t[2], Token::String("a\nb\t\"c".to_string()));
    }

    #[test]
    fn literal_string_no_escapes() {
        let t = tokens("name = 'a\\nb'\n");
        assert_eq!(t[2], Token::String("a\\nb".to_string()));
    }

    #[test]
    fn integer_with_underscore() {
        let t = tokens("n = 1_000_000\n");
        assert_eq!(t[2], Token::Integer(1_000_000));
    }

    #[test]
    fn integer_signed() {
        assert_eq!(tokens("n = -5\n")[2], Token::Integer(-5));
        assert_eq!(tokens("n = +5\n")[2], Token::Integer(5));
        assert_eq!(tokens("n = 0\n")[2], Token::Integer(0));
    }

    #[test]
    fn rejects_leading_zero() {
        assert!(matches!(lex("n = 01\n"), Err(Error::Lex { .. })));
    }

    #[test]
    fn rejects_hex_integer() {
        assert!(matches!(lex("n = 0xFF\n"), Err(Error::Lex { .. })));
    }

    #[test]
    fn rejects_floating_point() {
        assert!(matches!(lex("n = 1.5\n"), Err(Error::Lex { .. })));
    }

    #[test]
    fn rejects_unterminated_string() {
        assert!(matches!(lex("k = \"unterm\n"), Err(Error::Lex { .. })));
    }

    #[test]
    fn rejects_multiline_string() {
        assert!(matches!(lex("k = \"\"\""), Err(Error::Lex { .. })));
    }

    #[test]
    fn arrays_emit_brackets_and_commas() {
        let t = tokens("xs = [1, 2, 3]\n");
        assert_eq!(
            t,
            alloc::vec![
                Token::BareKey("xs".to_string()),
                Token::Equals,
                Token::LBracket,
                Token::Integer(1),
                Token::Comma,
                Token::Integer(2),
                Token::Comma,
                Token::Integer(3),
                Token::RBracket,
                Token::Newline,
            ]
        );
    }

    #[test]
    fn line_numbering_advances() {
        let spans = lex("a = 1\nb = 2\n").unwrap();
        // a = 1 \n b = 2 \n
        // tokens: BareKey(a)@1 = @1 1 @1 \n@1 BareKey(b)@2 ...
        assert_eq!(spans[0].line, 1);
        assert_eq!(spans[3].line, 1); // the '\n' after the first line
        assert_eq!(spans[4].line, 2);
    }

    #[test]
    fn handles_crlf() {
        let t = tokens("a = 1\r\n");
        assert_eq!(
            t,
            alloc::vec![
                Token::BareKey("a".to_string()),
                Token::Equals,
                Token::Integer(1),
                Token::Newline,
            ]
        );
    }

    #[test]
    fn empty_string_decodes() {
        assert_eq!(tokens("a = \"\"\n")[2], Token::String("".to_string()));
        assert_eq!(tokens("a = ''\n")[2], Token::String("".to_string()));
    }

    #[test]
    fn rejects_invalid_escape() {
        assert!(matches!(lex("a = \"\\q\"\n"), Err(Error::Lex { .. })));
    }
}
