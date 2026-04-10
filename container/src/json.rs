// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal hand-written JSON parser and serializer.
//!
//! Avoids pulling in `serde_json` to keep the container binary small.
//! Supports the subset of JSON needed for the inference API:
//! objects, arrays, strings, numbers, booleans, and null.

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    /// Preserves insertion order.
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Parse a JSON value from a string.
    pub fn parse(input: &str) -> Result<JsonValue, String> {
        let trimmed = input.trim();
        let (value, rest) = parse_value(trimmed)?;
        let rest = rest.trim();
        if !rest.is_empty() {
            return Err(format!(
                "trailing characters: {:?}",
                &rest[..rest.len().min(20)]
            ));
        }
        Ok(value)
    }

    /// Serialize to a JSON string.
    pub fn serialize(&self) -> String {
        let mut buf = String::new();
        serialize_value(self, &mut buf);
        buf
    }

    /// Look up a key in an object.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Extract as string slice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Extract as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Extract as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Extract as array slice.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    /// Extract as object entries slice.
    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(o) => Some(o.as_slice()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Serializer
// ---------------------------------------------------------------------------

fn serialize_value(value: &JsonValue, buf: &mut String) {
    match value {
        JsonValue::Null => buf.push_str("null"),
        JsonValue::Bool(true) => buf.push_str("true"),
        JsonValue::Bool(false) => buf.push_str("false"),
        JsonValue::Number(n) => {
            // Use integer formatting when the value is a whole number for cleaner output.
            if n.fract() == 0.0 && n.is_finite() && n.abs() < (i64::MAX as f64) {
                buf.push_str(&format!("{}", *n as i64));
            } else {
                buf.push_str(&format!("{}", n));
            }
        }
        JsonValue::String(s) => {
            buf.push('"');
            for ch in s.chars() {
                match ch {
                    '"' => buf.push_str("\\\""),
                    '\\' => buf.push_str("\\\\"),
                    '\n' => buf.push_str("\\n"),
                    '\r' => buf.push_str("\\r"),
                    '\t' => buf.push_str("\\t"),
                    c if c < '\x20' => buf.push_str(&format!("\\u{:04x}", c as u32)),
                    c => buf.push(c),
                }
            }
            buf.push('"');
        }
        JsonValue::Array(items) => {
            buf.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                serialize_value(item, buf);
            }
            buf.push(']');
        }
        JsonValue::Object(entries) => {
            buf.push('{');
            for (i, (key, val)) in entries.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                serialize_value(&JsonValue::String(key.clone()), buf);
                buf.push(':');
                serialize_value(val, buf);
            }
            buf.push('}');
        }
    }
}

// ---------------------------------------------------------------------------
// Parser — recursive descent
// ---------------------------------------------------------------------------

fn parse_value(input: &str) -> Result<(JsonValue, &str), String> {
    let input = skip_ws(input);
    if input.is_empty() {
        return Err("unexpected end of input".into());
    }
    match input.as_bytes()[0] {
        b'"' => parse_string(input).map(|(s, r)| (JsonValue::String(s), r)),
        b'{' => parse_object(input),
        b'[' => parse_array(input),
        b't' | b'f' => parse_bool(input),
        b'n' => parse_null(input),
        b'-' | b'0'..=b'9' => parse_number(input),
        ch => Err(format!("unexpected character: {:?}", ch as char)),
    }
}

fn skip_ws(input: &str) -> &str {
    input.trim_start()
}

fn parse_null(input: &str) -> Result<(JsonValue, &str), String> {
    if let Some(rest) = input.strip_prefix("null") {
        Ok((JsonValue::Null, rest))
    } else {
        Err(format!(
            "expected 'null', got {:?}",
            &input[..input.len().min(4)]
        ))
    }
}

fn parse_bool(input: &str) -> Result<(JsonValue, &str), String> {
    if let Some(rest) = input.strip_prefix("true") {
        Ok((JsonValue::Bool(true), rest))
    } else if let Some(rest) = input.strip_prefix("false") {
        Ok((JsonValue::Bool(false), rest))
    } else {
        Err(format!(
            "expected 'true' or 'false', got {:?}",
            &input[..input.len().min(5)]
        ))
    }
}

fn parse_number(input: &str) -> Result<(JsonValue, &str), String> {
    let mut end = 0;
    let bytes = input.as_bytes();

    // Optional leading minus
    if end < bytes.len() && bytes[end] == b'-' {
        end += 1;
    }

    // Integer part
    if end >= bytes.len() || !bytes[end].is_ascii_digit() {
        return Err("invalid number".into());
    }
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }

    // Fractional part
    if end < bytes.len() && bytes[end] == b'.' {
        end += 1;
        if end >= bytes.len() || !bytes[end].is_ascii_digit() {
            return Err("invalid number: no digits after decimal point".into());
        }
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }

    // Exponent
    if end < bytes.len() && (bytes[end] == b'e' || bytes[end] == b'E') {
        end += 1;
        if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
            end += 1;
        }
        if end >= bytes.len() || !bytes[end].is_ascii_digit() {
            return Err("invalid number: no digits in exponent".into());
        }
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }

    let num_str = &input[..end];
    let n: f64 = num_str
        .parse()
        .map_err(|e| format!("invalid number {:?}: {}", num_str, e))?;
    Ok((JsonValue::Number(n), &input[end..]))
}

fn parse_string(input: &str) -> Result<(String, &str), String> {
    if !input.starts_with('"') {
        return Err("expected '\"'".into());
    }
    let bytes = input.as_bytes();
    let mut result = String::new();
    let mut i = 1;

    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Ok((result, &input[i + 1..])),
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return Err("unexpected end of string escape".into());
                }
                match bytes[i] {
                    b'"' => result.push('"'),
                    b'\\' => result.push('\\'),
                    b'/' => result.push('/'),
                    b'n' => result.push('\n'),
                    b'r' => result.push('\r'),
                    b't' => result.push('\t'),
                    b'b' => result.push('\x08'),
                    b'f' => result.push('\x0c'),
                    b'u' => {
                        if i + 4 >= bytes.len() {
                            return Err("incomplete unicode escape".into());
                        }
                        let hex = &input[i + 1..i + 5];
                        let cp = u16::from_str_radix(hex, 16)
                            .map_err(|_| format!("invalid unicode escape: \\u{}", hex))?;
                        if let Some(ch) = char::from_u32(cp as u32) {
                            result.push(ch);
                        } else {
                            return Err(format!("invalid unicode codepoint: {}", cp));
                        }
                        i += 4;
                    }
                    other => return Err(format!("invalid escape character: {:?}", other as char)),
                }
            }
            _ => {
                // Multi-byte UTF-8: find the char boundary
                let rest = &input[i..];
                let ch = rest.chars().next().unwrap();
                result.push(ch);
                i += ch.len_utf8() - 1;
            }
        }
        i += 1;
    }
    Err("unterminated string".into())
}

fn parse_array(input: &str) -> Result<(JsonValue, &str), String> {
    let mut rest = skip_ws(&input[1..]); // skip '['
    let mut items = Vec::new();

    if let Some(r) = rest.strip_prefix(']') {
        return Ok((JsonValue::Array(items), r));
    }

    loop {
        let (val, r) = parse_value(rest)?;
        items.push(val);
        rest = skip_ws(r);
        if let Some(r) = rest.strip_prefix(']') {
            return Ok((JsonValue::Array(items), r));
        }
        if let Some(r) = rest.strip_prefix(',') {
            rest = skip_ws(r);
        } else {
            return Err(format!(
                "expected ',' or ']' in array, got {:?}",
                rest.chars().next()
            ));
        }
    }
}

fn parse_object(input: &str) -> Result<(JsonValue, &str), String> {
    let mut rest = skip_ws(&input[1..]); // skip '{'
    let mut entries = Vec::new();

    if let Some(r) = rest.strip_prefix('}') {
        return Ok((JsonValue::Object(entries), r));
    }

    loop {
        // Key
        if !rest.starts_with('"') {
            return Err(format!(
                "expected string key in object, got {:?}",
                rest.chars().next()
            ));
        }
        let (key, r) = parse_string(rest)?;
        rest = skip_ws(r);

        // Colon
        if let Some(r) = rest.strip_prefix(':') {
            rest = skip_ws(r);
        } else {
            return Err("expected ':' after object key".into());
        }

        // Value
        let (val, r) = parse_value(rest)?;
        entries.push((key, val));
        rest = skip_ws(r);

        if let Some(r) = rest.strip_prefix('}') {
            return Ok((JsonValue::Object(entries), r));
        }
        if let Some(r) = rest.strip_prefix(',') {
            rest = skip_ws(r);
        } else {
            return Err(format!(
                "expected ',' or '}}' in object, got {:?}",
                rest.chars().next()
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Primitive parsing ---

    #[test]
    fn parse_null() {
        assert_eq!(JsonValue::parse("null").unwrap(), JsonValue::Null);
    }

    #[test]
    fn parse_true() {
        assert_eq!(JsonValue::parse("true").unwrap(), JsonValue::Bool(true));
    }

    #[test]
    fn parse_false() {
        assert_eq!(JsonValue::parse("false").unwrap(), JsonValue::Bool(false));
    }

    #[test]
    fn parse_integer() {
        assert_eq!(JsonValue::parse("42").unwrap(), JsonValue::Number(42.0));
    }

    #[test]
    fn parse_negative_number() {
        assert_eq!(JsonValue::parse("-3.14").unwrap(), JsonValue::Number(-3.14));
    }

    #[test]
    fn parse_exponent() {
        assert_eq!(JsonValue::parse("1e3").unwrap(), JsonValue::Number(1000.0));
    }

    #[test]
    fn parse_string_simple() {
        assert_eq!(
            JsonValue::parse(r#""hello""#).unwrap(),
            JsonValue::String("hello".into())
        );
    }

    #[test]
    fn parse_string_escapes() {
        assert_eq!(
            JsonValue::parse(r#""a\nb\\c""#).unwrap(),
            JsonValue::String("a\nb\\c".into())
        );
    }

    #[test]
    fn parse_string_unicode_escape() {
        assert_eq!(
            JsonValue::parse(r#""\u0041""#).unwrap(),
            JsonValue::String("A".into())
        );
    }

    // --- Compound types ---

    #[test]
    fn parse_empty_array() {
        assert_eq!(JsonValue::parse("[]").unwrap(), JsonValue::Array(vec![]));
    }

    #[test]
    fn parse_number_array() {
        let val = JsonValue::parse("[1, 2, 3]").unwrap();
        let arr = val.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_f64(), Some(1.0));
        assert_eq!(arr[2].as_f64(), Some(3.0));
    }

    #[test]
    fn parse_empty_object() {
        assert_eq!(JsonValue::parse("{}").unwrap(), JsonValue::Object(vec![]));
    }

    #[test]
    fn parse_simple_object() {
        let val = JsonValue::parse(r#"{"name": "test", "count": 5}"#).unwrap();
        assert_eq!(val.get("name").unwrap().as_str(), Some("test"));
        assert_eq!(val.get("count").unwrap().as_f64(), Some(5.0));
    }

    #[test]
    fn parse_nested_object() {
        let input = r#"{"model": "resnet", "inputs": {"x": {"shape": [1,3], "data": [1.0, 2.0, 3.0], "dtype": "float32"}}}"#;
        let val = JsonValue::parse(input).unwrap();
        assert_eq!(val.get("model").unwrap().as_str(), Some("resnet"));
        let inputs = val.get("inputs").unwrap();
        let x = inputs.get("x").unwrap();
        assert_eq!(x.get("dtype").unwrap().as_str(), Some("float32"));
        let shape = x.get("shape").unwrap().as_array().unwrap();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0].as_f64(), Some(1.0));
        let data = x.get("data").unwrap().as_array().unwrap();
        assert_eq!(data.len(), 3);
        assert!((data[1].as_f64().unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_mixed_array() {
        let val = JsonValue::parse(r#"[1, "two", true, null, [3]]"#).unwrap();
        let arr = val.as_array().unwrap();
        assert_eq!(arr.len(), 5);
        assert_eq!(arr[0].as_f64(), Some(1.0));
        assert_eq!(arr[1].as_str(), Some("two"));
        assert_eq!(arr[2].as_bool(), Some(true));
        assert_eq!(arr[3], JsonValue::Null);
        assert_eq!(arr[4].as_array().unwrap().len(), 1);
    }

    // --- Round-trip serialization ---

    #[test]
    fn roundtrip_null() {
        assert_eq!(JsonValue::Null.serialize(), "null");
        assert_eq!(JsonValue::parse("null").unwrap().serialize(), "null");
    }

    #[test]
    fn roundtrip_bool() {
        assert_eq!(JsonValue::Bool(true).serialize(), "true");
        assert_eq!(JsonValue::Bool(false).serialize(), "false");
    }

    #[test]
    fn roundtrip_number() {
        assert_eq!(JsonValue::Number(42.0).serialize(), "42");
        assert_eq!(JsonValue::Number(3.14).serialize(), "3.14");
    }

    #[test]
    fn roundtrip_string() {
        let val = JsonValue::String("hello\nworld".into());
        let s = val.serialize();
        assert_eq!(s, r#""hello\nworld""#);
        assert_eq!(JsonValue::parse(&s).unwrap(), val);
    }

    #[test]
    fn roundtrip_complex() {
        let input =
            r#"{"model":"resnet","inputs":{"x":{"shape":[1,3],"data":[1,2,3],"dtype":"float32"}}}"#;
        let val = JsonValue::parse(input).unwrap();
        let output = val.serialize();
        let val2 = JsonValue::parse(&output).unwrap();
        assert_eq!(val, val2);
    }

    // --- Error cases ---

    #[test]
    fn error_empty_input() {
        assert!(JsonValue::parse("").is_err());
    }

    #[test]
    fn error_trailing_chars() {
        assert!(JsonValue::parse("true false").is_err());
    }

    #[test]
    fn error_unterminated_string() {
        assert!(JsonValue::parse(r#""hello"#).is_err());
    }

    #[test]
    fn error_unterminated_array() {
        assert!(JsonValue::parse("[1, 2").is_err());
    }

    #[test]
    fn error_unterminated_object() {
        assert!(JsonValue::parse(r#"{"key": 1"#).is_err());
    }

    #[test]
    fn error_missing_colon() {
        assert!(JsonValue::parse(r#"{"key" 1}"#).is_err());
    }

    #[test]
    fn error_invalid_number() {
        assert!(JsonValue::parse("1.").is_err());
    }

    // --- Accessor coverage ---

    #[test]
    fn accessor_returns_none_on_wrong_type() {
        let n = JsonValue::Number(1.0);
        assert!(n.as_str().is_none());
        assert!(n.as_bool().is_none());
        assert!(n.as_array().is_none());
        assert!(n.as_object().is_none());
        assert!(n.get("x").is_none());

        let s = JsonValue::String("hi".into());
        assert!(s.as_f64().is_none());
    }

    #[test]
    fn whitespace_tolerance() {
        let val = JsonValue::parse("  {  \"a\" :  1  }  ").unwrap();
        assert_eq!(val.get("a").unwrap().as_f64(), Some(1.0));
    }
}
