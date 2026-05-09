// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal `#![no_std]` JSON encoder/decoder targeted at the admin
//! keyspace request/response shapes.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-zenoh-admin/spec.md`
//! Requirement "Admin keyspace contract" — every payload is a JSON
//! object of the form
//!
//! ```text
//! { "token": "<opaque-id-or-empty-on-login>", "args": { ... } }
//! { "ok": true, "payload": { ... } }
//! { "ok": false, "code": <negative-errno>, "reason": "<text>" }
//! ```
//!
//! ## Why a clean-room codec instead of `serde_json` + `serde`?
//!
//! - `serde_json` is `std`-only by default and pulls a substantial
//!   transitive tree (`itoa`, `ryu`, `serde`, `serde_derive`).
//! - The wire shapes here are tightly bounded — three top-level fields
//!   per request, four per response, every nested `args` object is a
//!   flat string→value map.
//! - Phase 5+6 already brought a clean-room `mgmt::toml` parser; this
//!   module follows the same pattern for JSON.
//!
//! ## Scope
//!
//! - Decoder: parses RFC 8259 JSON limited to **objects**, **strings**,
//!   **numbers (signed integers, fits in `i64`)**, **booleans**, and
//!   **null**. Arrays are accepted but only as `Vec<JsonValue>` —
//!   admin payloads do not currently use them.
//! - Encoder: emits canonical lowercase booleans, `null`, decimal
//!   integers, double-quoted strings with the standard 8-character
//!   escapes (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`) plus
//!   `\u00XX` for any byte `< 0x20` not in the eight specials. UTF-8
//!   pass-through above `0x7F` is preserved verbatim — the wire is
//!   already UTF-8 and `\uXXXX` escapes for the BMP are unnecessary.
//!
//! ## Hard limits
//!
//! - Maximum payload bytes: [`MAX_PAYLOAD_BYTES`] (256 KiB). Larger
//!   inputs reject with [`JsonError::PayloadTooLarge`] before any
//!   parsing work runs — this caps memory exposure when the wire codec
//!   is fed adversarial input from an authenticated peer.
//! - Maximum nesting depth: [`MAX_DEPTH`] (32). Crossing the limit
//!   returns [`JsonError::TooDeep`].

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Hard cap on JSON payload bytes accepted by [`decode`]. The wire
/// transport (Zenoh) enforces its own framing limits; this is a
/// belt-and-braces cap so the decoder never allocates more than ~ 256
/// KiB even under adversarial input.
pub const MAX_PAYLOAD_BYTES: usize = 256 * 1024;

/// Hard cap on JSON nesting depth.
pub const MAX_DEPTH: usize = 32;

/// Sum type for decoded JSON values. The encoder accepts the same
/// type, plus convenience constructors via the `From` impls below.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON `null`.
    Null,
    /// JSON `true`/`false`.
    Bool(bool),
    /// JSON number — limited to signed integers fitting in `i64`. The
    /// decoder rejects fractional or exponent forms with
    /// [`JsonError::Unsupported`] so the codec never silently truncates.
    Int(i64),
    /// JSON string. Owned so the value outlives the source slice.
    String(String),
    /// JSON array. Permitted but not currently used by admin verbs.
    Array(Vec<JsonValue>),
    /// JSON object. Stored as a `BTreeMap` so encoded output is
    /// deterministic — important for tests and audit traces.
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// Construct a string value from any `&str`.
    pub fn string(s: &str) -> Self {
        Self::String(s.into())
    }

    /// Construct an empty object.
    pub fn empty_object() -> Self {
        Self::Object(BTreeMap::new())
    }

    /// Try to extract a string slice.
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Try to extract an integer.
    pub fn as_int(&self) -> Option<i64> {
        if let Self::Int(n) = self {
            Some(*n)
        } else {
            None
        }
    }

    /// Try to extract a bool.
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    /// Try to extract an object.
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        if let Self::Object(m) = self {
            Some(m)
        } else {
            None
        }
    }

    /// Convenience: lookup a field on an object value. Returns `None`
    /// if `self` is not an object or the key is absent.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object().and_then(|m| m.get(key))
    }
}

impl From<bool> for JsonValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<i64> for JsonValue {
    fn from(n: i64) -> Self {
        Self::Int(n)
    }
}

impl From<u32> for JsonValue {
    fn from(n: u32) -> Self {
        Self::Int(n as i64)
    }
}

impl From<u64> for JsonValue {
    fn from(n: u64) -> Self {
        // u64 → i64 lossless for ≤ i64::MAX; on overflow we saturate
        // because the admin payloads never carry values that high
        // (timestamps, byte counts in our schema fit in i63).
        Self::Int(if n > i64::MAX as u64 {
            i64::MAX
        } else {
            n as i64
        })
    }
}

impl From<&str> for JsonValue {
    fn from(s: &str) -> Self {
        Self::String(s.into())
    }
}

impl From<String> for JsonValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

/// JSON codec error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonError {
    /// Input exceeded [`MAX_PAYLOAD_BYTES`] — rejected before parsing.
    PayloadTooLarge,
    /// Nesting depth exceeded [`MAX_DEPTH`].
    TooDeep,
    /// Unexpected end of input.
    Eof,
    /// Invalid character at byte offset `at`. The diagnostic carries
    /// the expected token for the caller to log if useful.
    Unexpected {
        /// Byte offset within the input where parsing failed.
        at: usize,
        /// Static descriptor of what the parser expected.
        wanted: &'static str,
    },
    /// Numeric form not supported (fractions, exponents, NaN, Infinity).
    Unsupported(&'static str),
    /// Numeric overflow — value did not fit in `i64`.
    Overflow,
    /// Trailing non-whitespace data after a complete value.
    TrailingData {
        /// Byte offset of the first trailing character.
        at: usize,
    },
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PayloadTooLarge => write!(f, "json: payload too large"),
            Self::TooDeep => write!(f, "json: nesting too deep"),
            Self::Eof => write!(f, "json: unexpected end of input"),
            Self::Unexpected { at, wanted } => {
                write!(
                    f,
                    "json: unexpected character at byte {at}, wanted {wanted}"
                )
            }
            Self::Unsupported(reason) => write!(f, "json: unsupported: {reason}"),
            Self::Overflow => write!(f, "json: numeric overflow"),
            Self::TrailingData { at } => write!(f, "json: trailing data at byte {at}"),
        }
    }
}

// ─── Decode ──────────────────────────────────────────────────────────────────

/// Parse `bytes` as a single JSON value. Returns
/// [`JsonError::TrailingData`] if any non-whitespace remains after the
/// value.
pub fn decode(bytes: &[u8]) -> Result<JsonValue, JsonError> {
    if bytes.len() > MAX_PAYLOAD_BYTES {
        return Err(JsonError::PayloadTooLarge);
    }
    let mut p = Parser::new(bytes);
    p.skip_whitespace();
    let v = p.parse_value(0)?;
    p.skip_whitespace();
    if p.pos != bytes.len() {
        return Err(JsonError::TrailingData { at: p.pos });
    }
    Ok(v)
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self { src, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, c: u8, wanted: &'static str) -> Result<(), JsonError> {
        match self.peek() {
            Some(g) if g == c => {
                self.pos += 1;
                Ok(())
            }
            _ => Err(JsonError::Unexpected {
                at: self.pos,
                wanted,
            }),
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        if depth > MAX_DEPTH {
            return Err(JsonError::TooDeep);
        }
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.parse_object(depth),
            Some(b'[') => self.parse_array(depth),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(_) => Err(JsonError::Unexpected {
                at: self.pos,
                wanted: "value",
            }),
            None => Err(JsonError::Eof),
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        self.expect(b'{', "'{'")?;
        let mut map = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect(b':', "':'")?;
            let value = self.parse_value(depth + 1)?;
            map.insert(key, value);
            self.skip_whitespace();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(JsonValue::Object(map)),
                Some(_) => {
                    return Err(JsonError::Unexpected {
                        at: self.pos - 1,
                        wanted: "',' or '}'",
                    })
                }
                None => return Err(JsonError::Eof),
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        self.expect(b'[', "'['")?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let v = self.parse_value(depth + 1)?;
            items.push(v);
            self.skip_whitespace();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => return Ok(JsonValue::Array(items)),
                Some(_) => {
                    return Err(JsonError::Unexpected {
                        at: self.pos - 1,
                        wanted: "',' or ']'",
                    })
                }
                None => return Err(JsonError::Eof),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"', "'\"'")?;
        let mut out = String::new();
        loop {
            match self.bump() {
                Some(b'"') => return Ok(out),
                Some(b'\\') => match self.bump() {
                    Some(b'"') => out.push('"'),
                    Some(b'\\') => out.push('\\'),
                    Some(b'/') => out.push('/'),
                    Some(b'b') => out.push('\u{0008}'),
                    Some(b'f') => out.push('\u{000C}'),
                    Some(b'n') => out.push('\n'),
                    Some(b'r') => out.push('\r'),
                    Some(b't') => out.push('\t'),
                    Some(b'u') => {
                        // \uXXXX — accept ≤ 0xFFFF and append as UTF-8.
                        let mut cp: u32 = 0;
                        for _ in 0..4 {
                            let h = self.bump().ok_or(JsonError::Eof)?;
                            let d = match h {
                                b'0'..=b'9' => h - b'0',
                                b'a'..=b'f' => h - b'a' + 10,
                                b'A'..=b'F' => h - b'A' + 10,
                                _ => {
                                    return Err(JsonError::Unexpected {
                                        at: self.pos - 1,
                                        wanted: "hex digit",
                                    })
                                }
                            };
                            cp = (cp << 4) | u32::from(d);
                        }
                        // Refuse surrogate halves — admin payloads carry
                        // ASCII-token / UTF-8 strings only.
                        if (0xD800..=0xDFFF).contains(&cp) {
                            return Err(JsonError::Unsupported(
                                "surrogate \\uXXXX escapes not supported",
                            ));
                        }
                        if let Some(c) = char::from_u32(cp) {
                            out.push(c);
                        } else {
                            return Err(JsonError::Unsupported("invalid \\uXXXX scalar"));
                        }
                    }
                    Some(_) => {
                        return Err(JsonError::Unexpected {
                            at: self.pos - 1,
                            wanted: "escape character",
                        })
                    }
                    None => return Err(JsonError::Eof),
                },
                Some(c) if c < 0x20 => {
                    return Err(JsonError::Unexpected {
                        at: self.pos - 1,
                        wanted: "string character",
                    })
                }
                Some(c) => {
                    if c < 0x80 {
                        out.push(c as char);
                    } else {
                        // Multi-byte UTF-8 — collect the whole sequence
                        // and validate via core::str::from_utf8 on the
                        // assembled bytes.
                        let start = self.pos - 1;
                        let extra = match c {
                            0xC2..=0xDF => 1,
                            0xE0..=0xEF => 2,
                            0xF0..=0xF4 => 3,
                            _ => {
                                return Err(JsonError::Unexpected {
                                    at: start,
                                    wanted: "valid utf-8 leading byte",
                                })
                            }
                        };
                        if self.pos + extra > self.src.len() {
                            return Err(JsonError::Eof);
                        }
                        let bytes = &self.src[start..self.pos + extra];
                        let s = core::str::from_utf8(bytes).map_err(|_| JsonError::Unexpected {
                            at: start,
                            wanted: "valid utf-8 sequence",
                        })?;
                        out.push_str(s);
                        self.pos += extra;
                    }
                }
                None => return Err(JsonError::Eof),
            }
        }
    }

    fn parse_bool(&mut self) -> Result<JsonValue, JsonError> {
        if self.src[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.src[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(JsonError::Unexpected {
                at: self.pos,
                wanted: "'true' or 'false'",
            })
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, JsonError> {
        if self.src[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(JsonError::Unexpected {
                at: self.pos,
                wanted: "'null'",
            })
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        if matches!(self.peek(), Some(b'.') | Some(b'e') | Some(b'E')) {
            return Err(JsonError::Unsupported(
                "fractional / exponent numbers not accepted",
            ));
        }
        let s = core::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| JsonError::Unsupported("non-utf8 number"))?;
        s.parse::<i64>().map(JsonValue::Int).map_err(|e| {
            if matches!(
                e.kind(),
                core::num::IntErrorKind::PosOverflow | core::num::IntErrorKind::NegOverflow
            ) {
                JsonError::Overflow
            } else {
                JsonError::Unexpected {
                    at: start,
                    wanted: "integer",
                }
            }
        })
    }
}

// ─── Encode ──────────────────────────────────────────────────────────────────

/// Encode `value` into a deterministic UTF-8 JSON string.
pub fn encode(value: &JsonValue) -> String {
    let mut out = String::new();
    encode_into(value, &mut out);
    out
}

/// Append the encoded form of `value` to `out`. Useful when building
/// up a wire frame in-place without allocating a fresh `String`.
pub fn encode_into(value: &JsonValue, out: &mut String) {
    match value {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(true) => out.push_str("true"),
        JsonValue::Bool(false) => out.push_str("false"),
        JsonValue::Int(n) => write_int(*n, out),
        JsonValue::String(s) => encode_string(s, out),
        JsonValue::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode_into(v, out);
            }
            out.push(']');
        }
        JsonValue::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode_string(k, out);
                out.push(':');
                encode_into(v, out);
            }
            out.push('}');
        }
    }
}

fn encode_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str("\\u");
                let v = c as u32;
                for shift in (0..4).rev() {
                    let nibble = ((v >> (shift * 4)) & 0xF) as u8;
                    out.push(if nibble < 10 {
                        (b'0' + nibble) as char
                    } else {
                        (b'a' + nibble - 10) as char
                    });
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_int(n: i64, out: &mut String) {
    // Manual i64 → decimal so we stay no_std-clean and skip the
    // `itoa`/`ryu` deps. Worst case is "-9223372036854775808" (20 chars).
    let mut buf = [0u8; 20];
    let mut len = 0;
    let (neg, mut abs) = if n < 0 {
        // Use unsigned absolute to avoid -i64::MIN overflow.
        (true, (n as i128).unsigned_abs())
    } else {
        (false, n as u128)
    };
    if abs == 0 {
        out.push('0');
        return;
    }
    while abs > 0 {
        buf[len] = b'0' + (abs % 10) as u8;
        abs /= 10;
        len += 1;
    }
    if neg {
        out.push('-');
    }
    for i in (0..len).rev() {
        out.push(buf[i] as char);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn obj(pairs: &[(&str, JsonValue)]) -> JsonValue {
        let mut m = BTreeMap::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        JsonValue::Object(m)
    }

    #[test]
    fn encode_decode_round_trip_object() {
        let v = obj(&[
            ("ok", JsonValue::Bool(true)),
            ("count", JsonValue::Int(7)),
            ("name", JsonValue::string("alice")),
        ]);
        let s = encode(&v);
        // Deterministic: keys sorted alphabetically.
        assert_eq!(s, r#"{"count":7,"name":"alice","ok":true}"#);
        let back = decode(s.as_bytes()).unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn decode_login_request() {
        let raw = br#"{"args":{"pass":"hunter2","user":"root"}}"#;
        let v = decode(raw).unwrap();
        assert_eq!(
            v.get("args")
                .and_then(|a| a.get("user"))
                .and_then(|n| n.as_str()),
            Some("root")
        );
    }

    #[test]
    fn decode_response_error_shape() {
        let raw = br#"{"code":-13,"ok":false,"reason":"Authentication failed"}"#;
        let v = decode(raw).unwrap();
        assert_eq!(v.get("ok").and_then(|b| b.as_bool()), Some(false));
        assert_eq!(v.get("code").and_then(|n| n.as_int()), Some(-13));
    }

    #[test]
    fn decode_rejects_floats() {
        let err = decode(b"3.14").unwrap_err();
        assert!(matches!(err, JsonError::Unsupported(_)));
    }

    #[test]
    fn decode_rejects_exponent() {
        let err = decode(b"1e10").unwrap_err();
        assert!(matches!(err, JsonError::Unsupported(_)));
    }

    #[test]
    fn decode_rejects_overflow() {
        // i64::MAX + 1
        let err = decode(b"9223372036854775808").unwrap_err();
        assert!(matches!(err, JsonError::Overflow));
    }

    #[test]
    fn decode_rejects_payload_too_large() {
        let bytes = alloc::vec![b'"'; MAX_PAYLOAD_BYTES + 1];
        assert_eq!(decode(&bytes).unwrap_err(), JsonError::PayloadTooLarge);
    }

    #[test]
    fn decode_rejects_too_deep() {
        let mut s = String::new();
        for _ in 0..(MAX_DEPTH + 2) {
            s.push('[');
        }
        assert_eq!(decode(s.as_bytes()).unwrap_err(), JsonError::TooDeep);
    }

    #[test]
    fn decode_rejects_trailing_data() {
        let err = decode(b"true false").unwrap_err();
        assert!(matches!(err, JsonError::TrailingData { .. }));
    }

    #[test]
    fn decode_rejects_unterminated_string() {
        let err = decode(b"\"abc").unwrap_err();
        assert!(matches!(err, JsonError::Eof));
    }

    #[test]
    fn decode_rejects_bare_control_char_in_string() {
        // Newline (0x0A) inside a string must use \n escape.
        let err = decode(b"\"a\nb\"").unwrap_err();
        assert!(matches!(err, JsonError::Unexpected { .. }));
    }

    #[test]
    fn decode_handles_escape_sequences() {
        let raw = br#""a\"b\\c\nd""#;
        let v = decode(raw).unwrap();
        assert_eq!(v.as_str(), Some("a\"b\\c\nd"));
    }

    #[test]
    fn decode_handles_unicode_escape_bmp() {
        // é = é (Latin small letter e with acute) — use the
        // JSON \u escape rather than raw UTF-8 in the test source.
        let raw = b"\"\\u00e9\"";
        let v = decode(raw).unwrap();
        assert_eq!(v.as_str(), Some("\u{00e9}"));
    }

    #[test]
    fn decode_rejects_surrogate_escape() {
        let err = decode(br#""\ud83d""#).unwrap_err();
        assert!(matches!(err, JsonError::Unsupported(_)));
    }

    #[test]
    fn decode_passes_through_utf8_bytes() {
        // "café" UTF-8: 63 61 66 c3 a9
        let raw = b"\"caf\xc3\xa9\"";
        let v = decode(raw).unwrap();
        assert_eq!(v.as_str(), Some("café"));
    }

    #[test]
    fn encode_int_signed_min() {
        let s = encode(&JsonValue::Int(i64::MIN));
        assert_eq!(s, "-9223372036854775808");
    }

    #[test]
    fn encode_int_zero() {
        assert_eq!(encode(&JsonValue::Int(0)), "0");
    }

    #[test]
    fn encode_string_escapes_control_bytes() {
        let s = encode(&JsonValue::string("\x01"));
        assert_eq!(s, "\"\\u0001\"");
    }

    #[test]
    fn encode_empty_object_and_array() {
        assert_eq!(encode(&JsonValue::empty_object()), "{}");
        assert_eq!(encode(&JsonValue::Array(Vec::new())), "[]");
    }

    #[test]
    fn decode_nested_object_and_array() {
        let raw = br#"{"a":[1,2,3],"b":{"c":true}}"#;
        let v = decode(raw).unwrap();
        let arr = v.get("a").unwrap();
        if let JsonValue::Array(items) = arr {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].as_int(), Some(1));
        } else {
            panic!("expected array");
        }
        assert_eq!(
            v.get("b")
                .and_then(|x| x.get("c"))
                .and_then(|x| x.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn decode_empty_object() {
        let v = decode(b"{}").unwrap();
        assert_eq!(v, JsonValue::empty_object());
    }

    #[test]
    fn fuzz_smoke_random_bytes_dont_panic() {
        // Exhaustively try every single-byte input — none should panic.
        for b in 0u8..=255u8 {
            let _ = decode(&[b]);
        }
    }
}
