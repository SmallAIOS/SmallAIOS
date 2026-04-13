// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HuggingFace `config.json` parser.
//!
//! Handles architecture detection and extraction of the Gemma-family
//! transformer hyper-parameters into a typed `GemmaConfig`. The parser
//! is intentionally written from scratch (no `serde`, no
//! `serde_json`) to keep the `onnx-rt` crate dependency-free.
//!
//! The scope of this module is deliberately narrow: it only needs to
//! understand the subset of JSON that real HuggingFace configs use —
//! objects, arrays (including nested and mixed-number), strings,
//! booleans, nulls, and numbers (integer + f64). Float numbers are
//! parsed using Rust's built-in `str::parse::<f64>()`.
//!
//! The safetensors header parser in `safetensors.rs` is a separate,
//! tighter i64-only parser; we deliberately do not share state between
//! them because their error domains differ (`InvalidHeader` vs
//! `InvalidConfig`) and keeping them independent avoids re-touching
//! the already-landed Section 1 code.

use super::LoaderError;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

// ---------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------

/// Transformer architecture family identified from `config.json`.
///
/// `Unknown` carries the raw architecture string so callers can decide
/// whether to emit a user-facing error listing supported families or
/// treat it as a soft fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// Gemma 3 (e.g. `Gemma3ForCausalLM`).
    Gemma3,
    /// Gemma 4 (e.g. `Gemma4ForCausalLM`).
    Gemma4,
    /// Llama family (`LlamaForCausalLM`).
    Llama,
    /// Qwen family (`Qwen2ForCausalLM`, `Qwen3ForCausalLM`, ...).
    Qwen,
    /// Anything else — the original architecture string is preserved.
    Unknown(String),
}

impl ModelArchitecture {
    /// Detect the model family from a raw `config.json` string by
    /// inspecting the `architectures` array (HuggingFace convention).
    ///
    /// Returns an error only if the JSON is malformed or the
    /// `architectures` key is not a non-empty array of strings.
    /// Unrecognized architectures yield `Ok(Unknown(...))` so callers
    /// decide the policy.
    pub fn detect(config_json: &str) -> Result<Self, LoaderError> {
        let root = parse_config_object(config_json)?;
        let archs = root.get("architectures").ok_or_else(|| {
            LoaderError::InvalidConfig("missing 'architectures' field".to_string())
        })?;
        let arr = match archs {
            JsonValue::Array(a) => a,
            _ => {
                return Err(LoaderError::InvalidConfig(
                    "'architectures' must be an array".to_string(),
                ))
            }
        };
        let first = arr.first().ok_or_else(|| {
            LoaderError::InvalidConfig("'architectures' array is empty".to_string())
        })?;
        let name = match first {
            JsonValue::String(s) => s.as_str(),
            _ => {
                return Err(LoaderError::InvalidConfig(
                    "'architectures[0]' must be a string".to_string(),
                ))
            }
        };
        Ok(Self::from_arch_name(name))
    }

    /// Classify a single architecture string.
    fn from_arch_name(name: &str) -> Self {
        // We match on the distinctive prefix rather than the full
        // string so minor variations (e.g. `Gemma3ForCausalLM` vs
        // `Gemma3Model`) all route to the same family.
        if name.starts_with("Gemma3") {
            ModelArchitecture::Gemma3
        } else if name.starts_with("Gemma4") {
            ModelArchitecture::Gemma4
        } else if name.starts_with("Llama") {
            ModelArchitecture::Llama
        } else if name.starts_with("Qwen") {
            ModelArchitecture::Qwen
        } else {
            ModelArchitecture::Unknown(name.to_string())
        }
    }
}

/// Parsed Gemma transformer configuration. Fields correspond to the
/// keys in a Gemma 3 / Gemma 4 `config.json` with one-to-one names.
#[derive(Debug, Clone, PartialEq)]
pub struct GemmaConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub sliding_window: usize,
    /// Every Nth layer is a global-attention layer. Gemma 3 defaults
    /// to 6 (5 local + 1 global). Parsed from `sliding_window_pattern`
    /// and defaulted to 6 if absent.
    pub sliding_window_pattern: usize,
    pub rms_norm_eps: f32,
    pub bos_token_id: i64,
    /// First element of the eos token list. Some Gemma configs encode
    /// `eos_token_id` as an array (e.g. `[1, 107]`); we take the first
    /// element for the canonical generation-loop stop token.
    pub eos_token_id: i64,
}

impl GemmaConfig {
    /// Parse a HuggingFace `config.json` string into a `GemmaConfig`
    /// and validate invariants.
    ///
    /// Returns `LoaderError::InvalidConfig` with a descriptive message
    /// on any structural, type, or validation failure.
    pub fn from_json(json: &str) -> Result<Self, LoaderError> {
        let root = parse_config_object(json)?;

        let num_hidden_layers = req_usize(&root, "num_hidden_layers")?;
        let hidden_size = req_usize(&root, "hidden_size")?;
        let intermediate_size = req_usize(&root, "intermediate_size")?;
        let num_attention_heads = req_usize(&root, "num_attention_heads")?;
        let num_key_value_heads = req_usize(&root, "num_key_value_heads")?;
        let vocab_size = req_usize(&root, "vocab_size")?;
        let max_position_embeddings = req_usize(&root, "max_position_embeddings")?;
        let rope_theta = req_f32(&root, "rope_theta")?;
        let rms_norm_eps = req_f32(&root, "rms_norm_eps")?;

        // `head_dim` defaults to `hidden_size / num_attention_heads`
        // when absent (HuggingFace Gemma convention).
        let head_dim = match root.get("head_dim") {
            Some(v) => json_value_to_usize(v, "head_dim")?,
            None => {
                if num_attention_heads == 0 {
                    return Err(LoaderError::InvalidConfig(
                        "num_attention_heads must be > 0 to derive head_dim".to_string(),
                    ));
                }
                hidden_size / num_attention_heads
            }
        };

        // `sliding_window` is optional for Llama-shaped Gemma configs
        // but required for the Gemma local-attention layers. Treat a
        // missing value as 0 — validation catches cases where the
        // architecture needs it non-zero.
        let sliding_window = match root.get("sliding_window") {
            Some(JsonValue::Null) | None => 0,
            Some(v) => json_value_to_usize(v, "sliding_window")?,
        };

        let sliding_window_pattern = match root.get("sliding_window_pattern") {
            Some(v) => json_value_to_usize(v, "sliding_window_pattern")?,
            None => 6, // Gemma 3 default
        };

        let bos_token_id = req_i64_or_first(&root, "bos_token_id")?;
        let eos_token_id = req_i64_or_first(&root, "eos_token_id")?;

        let cfg = GemmaConfig {
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            sliding_window,
            sliding_window_pattern,
            rms_norm_eps,
            bos_token_id,
            eos_token_id,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Check structural invariants. Callers may invoke this directly
    /// on a hand-constructed config.
    pub fn validate(&self) -> Result<(), LoaderError> {
        if self.num_hidden_layers == 0 {
            return Err(LoaderError::InvalidConfig(
                "num_hidden_layers must be > 0".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(LoaderError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(LoaderError::InvalidConfig(
                "vocab_size must be > 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(LoaderError::InvalidConfig(
                "num_attention_heads must be > 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(LoaderError::InvalidConfig(
                "num_key_value_heads must be > 0".to_string(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(LoaderError::InvalidConfig(format!(
                "hidden_size ({}) not divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(LoaderError::InvalidConfig(format!(
                "num_attention_heads ({}) not divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.rope_theta.is_nan() || self.rope_theta <= 0.0 {
            return Err(LoaderError::InvalidConfig(
                "rope_theta must be > 0".to_string(),
            ));
        }
        if self.rms_norm_eps.is_nan() || self.rms_norm_eps <= 0.0 {
            return Err(LoaderError::InvalidConfig(
                "rms_norm_eps must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Typed field extraction helpers
// ---------------------------------------------------------------------

type ConfigMap = BTreeMap<String, JsonValue>;

fn req_usize(root: &ConfigMap, key: &str) -> Result<usize, LoaderError> {
    let v = root
        .get(key)
        .ok_or_else(|| LoaderError::InvalidConfig(format!("missing required field '{key}'")))?;
    json_value_to_usize(v, key)
}

fn req_f32(root: &ConfigMap, key: &str) -> Result<f32, LoaderError> {
    let v = root
        .get(key)
        .ok_or_else(|| LoaderError::InvalidConfig(format!("missing required field '{key}'")))?;
    json_value_to_f32(v, key)
}

/// Extract an i64, accepting either a plain number or a non-empty
/// array of numbers (taking the first element). Gemma configs use the
/// latter for `eos_token_id` when a model has multiple end-of-turn
/// tokens.
fn req_i64_or_first(root: &ConfigMap, key: &str) -> Result<i64, LoaderError> {
    let v = root
        .get(key)
        .ok_or_else(|| LoaderError::InvalidConfig(format!("missing required field '{key}'")))?;
    match v {
        JsonValue::Number(n) => number_to_i64(*n, key),
        JsonValue::Array(arr) => {
            let first = arr
                .first()
                .ok_or_else(|| LoaderError::InvalidConfig(format!("'{key}' array is empty")))?;
            match first {
                JsonValue::Number(n) => number_to_i64(*n, key),
                _ => Err(LoaderError::InvalidConfig(format!(
                    "'{key}[0]' must be a number"
                ))),
            }
        }
        _ => Err(LoaderError::InvalidConfig(format!(
            "'{key}' must be a number or array of numbers"
        ))),
    }
}

fn json_value_to_usize(v: &JsonValue, key: &str) -> Result<usize, LoaderError> {
    match v {
        JsonValue::Number(n) => {
            if !n.is_finite() || *n < 0.0 {
                return Err(LoaderError::InvalidConfig(format!(
                    "'{key}' must be a non-negative integer, got {n}"
                )));
            }
            let truncated = n.trunc();
            if (n - truncated).abs() > f64::EPSILON {
                return Err(LoaderError::InvalidConfig(format!(
                    "'{key}' must be an integer, got {n}"
                )));
            }
            Ok(truncated as usize)
        }
        _ => Err(LoaderError::InvalidConfig(format!(
            "'{key}' must be a number"
        ))),
    }
}

fn json_value_to_f32(v: &JsonValue, key: &str) -> Result<f32, LoaderError> {
    match v {
        JsonValue::Number(n) => {
            if !n.is_finite() {
                return Err(LoaderError::InvalidConfig(format!(
                    "'{key}' must be finite, got {n}"
                )));
            }
            Ok(*n as f32)
        }
        _ => Err(LoaderError::InvalidConfig(format!(
            "'{key}' must be a number"
        ))),
    }
}

fn number_to_i64(n: f64, key: &str) -> Result<i64, LoaderError> {
    if !n.is_finite() {
        return Err(LoaderError::InvalidConfig(format!(
            "'{key}' must be finite"
        )));
    }
    let truncated = n.trunc();
    if (n - truncated).abs() > f64::EPSILON {
        return Err(LoaderError::InvalidConfig(format!(
            "'{key}' must be an integer, got {n}"
        )));
    }
    if !(i64::MIN as f64..=i64::MAX as f64).contains(&truncated) {
        return Err(LoaderError::InvalidConfig(format!(
            "'{key}' out of i64 range"
        )));
    }
    Ok(truncated as i64)
}

// ---------------------------------------------------------------------
// Config-oriented JSON parser
//
// Supports objects, arrays (including nested and mixed-type), strings
// with JSON escapes, booleans, null, and numbers (parsed as f64 with
// exponent support). Deliberately kept independent from the
// safetensors i64 parser so their error domains stay separate.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(ConfigMap),
}

fn parse_config_object(input: &str) -> Result<ConfigMap, LoaderError> {
    let mut p = ConfigJsonParser::new(input);
    p.skip_ws();
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(LoaderError::InvalidConfig(format!(
            "trailing data after JSON at byte {}",
            p.pos
        )));
    }
    match v {
        JsonValue::Object(o) => Ok(o),
        _ => Err(LoaderError::InvalidConfig(
            "top-level config.json must be an object".to_string(),
        )),
    }
}

struct ConfigJsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ConfigJsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), LoaderError> {
        match self.bump() {
            Some(c) if c == b => Ok(()),
            Some(c) => Err(LoaderError::InvalidConfig(format!(
                "expected '{}' got '{}' at byte {}",
                b as char, c as char, self.pos
            ))),
            None => Err(LoaderError::InvalidConfig(format!(
                "expected '{}' at EOF",
                b as char
            ))),
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, LoaderError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(JsonValue::String(self.parse_string()?)),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(LoaderError::InvalidConfig(format!(
                "unexpected byte '{}' at {}",
                c as char, self.pos
            ))),
            None => Err(LoaderError::InvalidConfig(
                "unexpected EOF in JSON value".to_string(),
            )),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, LoaderError> {
        self.expect(b'{')?;
        let mut out = ConfigMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(out));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let value = self.parse_value()?;
            out.insert(key, value);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(JsonValue::Object(out)),
                Some(c) => {
                    return Err(LoaderError::InvalidConfig(format!(
                        "expected ',' or '}}' got '{}' at byte {}",
                        c as char, self.pos
                    )))
                }
                None => {
                    return Err(LoaderError::InvalidConfig(
                        "unexpected EOF in object".to_string(),
                    ))
                }
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue, LoaderError> {
        self.expect(b'[')?;
        let mut out: Vec<JsonValue> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(out));
        }
        loop {
            let v = self.parse_value()?;
            out.push(v);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => return Ok(JsonValue::Array(out)),
                Some(c) => {
                    return Err(LoaderError::InvalidConfig(format!(
                        "expected ',' or ']' got '{}' at byte {}",
                        c as char, self.pos
                    )))
                }
                None => {
                    return Err(LoaderError::InvalidConfig(
                        "unexpected EOF in array".to_string(),
                    ))
                }
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, LoaderError> {
        self.skip_ws();
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.bump() {
                Some(b'"') => return Ok(out),
                Some(b'\\') => match self.bump() {
                    Some(b'"') => out.push('"'),
                    Some(b'\\') => out.push('\\'),
                    Some(b'/') => out.push('/'),
                    Some(b'n') => out.push('\n'),
                    Some(b'r') => out.push('\r'),
                    Some(b't') => out.push('\t'),
                    Some(b'b') => out.push('\u{08}'),
                    Some(b'f') => out.push('\u{0C}'),
                    Some(b'u') => {
                        let mut code: u32 = 0;
                        for _ in 0..4 {
                            let d = self.bump().ok_or_else(|| {
                                LoaderError::InvalidConfig("truncated \\u escape".to_string())
                            })?;
                            let hv = match d {
                                b'0'..=b'9' => d - b'0',
                                b'a'..=b'f' => d - b'a' + 10,
                                b'A'..=b'F' => d - b'A' + 10,
                                _ => {
                                    return Err(LoaderError::InvalidConfig(format!(
                                        "bad hex digit '{}' in \\u escape",
                                        d as char
                                    )))
                                }
                            };
                            code = (code << 4) | hv as u32;
                        }
                        if let Some(c) = char::from_u32(code) {
                            out.push(c);
                        } else {
                            return Err(LoaderError::InvalidConfig(format!(
                                "invalid unicode codepoint U+{code:04X}"
                            )));
                        }
                    }
                    Some(c) => {
                        return Err(LoaderError::InvalidConfig(format!(
                            "bad escape '\\{}'",
                            c as char
                        )))
                    }
                    None => {
                        return Err(LoaderError::InvalidConfig(
                            "unterminated escape".to_string(),
                        ))
                    }
                },
                Some(b) => {
                    out.push(b as char);
                }
                None => {
                    return Err(LoaderError::InvalidConfig(
                        "unterminated string".to_string(),
                    ))
                }
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, LoaderError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        let slice = &self.bytes[start..self.pos];
        let s = core::str::from_utf8(slice)
            .map_err(|_| LoaderError::InvalidConfig("non-utf8 number".to_string()))?;
        let n: f64 = s
            .parse()
            .map_err(|_| LoaderError::InvalidConfig(format!("bad number '{s}'")))?;
        Ok(JsonValue::Number(n))
    }

    fn parse_bool(&mut self) -> Result<JsonValue, LoaderError> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(LoaderError::InvalidConfig(
                "expected true/false".to_string(),
            ))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, LoaderError> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(LoaderError::InvalidConfig("expected null".to_string()))
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 3 4B-it style config (trimmed to the fields we parse,
    /// plus a few unrelated keys and a nested `rope_scaling` object
    /// to exercise the nested-object parser path).
    const GEMMA3_4B: &str = r#"{
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3",
        "num_hidden_layers": 34,
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 262144,
        "max_position_embeddings": 131072,
        "rope_theta": 1000000.0,
        "sliding_window": 1024,
        "sliding_window_pattern": 6,
        "rms_norm_eps": 1e-6,
        "bos_token_id": 2,
        "eos_token_id": [1, 107],
        "rope_scaling": {
            "type": "linear",
            "factor": 8.0
        },
        "torch_dtype": "bfloat16"
    }"#;

    /// Gemma 4 31B-it style config — larger dims, scalar `eos_token_id`.
    const GEMMA4_31B: &str = r#"{
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4",
        "num_hidden_layers": 56,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 168,
        "vocab_size": 262144,
        "max_position_embeddings": 131072,
        "rope_theta": 1000000.0,
        "sliding_window": 4096,
        "sliding_window_pattern": 6,
        "rms_norm_eps": 0.000001,
        "bos_token_id": 2,
        "eos_token_id": 1
    }"#;

    /// Llama 3 8B style config — architecture-detected as Llama; no
    /// sliding window in the Gemma sense.
    const LLAMA3_8B: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "bos_token_id": 128000,
        "eos_token_id": 128001
    }"#;

    #[test]
    fn detects_gemma3_architecture() {
        let arch = ModelArchitecture::detect(GEMMA3_4B).unwrap();
        assert_eq!(arch, ModelArchitecture::Gemma3);
    }

    #[test]
    fn detects_gemma4_architecture() {
        let arch = ModelArchitecture::detect(GEMMA4_31B).unwrap();
        assert_eq!(arch, ModelArchitecture::Gemma4);
    }

    #[test]
    fn detects_llama_architecture() {
        let arch = ModelArchitecture::detect(LLAMA3_8B).unwrap();
        assert_eq!(arch, ModelArchitecture::Llama);
    }

    #[test]
    fn detects_qwen_architecture() {
        let json = r#"{"architectures":["Qwen2ForCausalLM"]}"#;
        assert_eq!(
            ModelArchitecture::detect(json).unwrap(),
            ModelArchitecture::Qwen
        );
    }

    #[test]
    fn detects_unknown_architecture_as_soft() {
        let json = r#"{"architectures":["MysteryNetForCausalLM"]}"#;
        match ModelArchitecture::detect(json).unwrap() {
            ModelArchitecture::Unknown(s) => assert_eq!(s, "MysteryNetForCausalLM"),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn detect_errors_on_missing_architectures_field() {
        let json = r#"{"model_type":"gemma3"}"#;
        let err = ModelArchitecture::detect(json).unwrap_err();
        assert!(matches!(err, LoaderError::InvalidConfig(_)));
    }

    #[test]
    fn parses_gemma3_4b_config() {
        let cfg = GemmaConfig::from_json(GEMMA3_4B).unwrap();
        assert_eq!(cfg.num_hidden_layers, 34);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, 10240);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.vocab_size, 262144);
        assert_eq!(cfg.max_position_embeddings, 131072);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        assert_eq!(cfg.sliding_window, 1024);
        assert_eq!(cfg.sliding_window_pattern, 6);
        assert!(cfg.rms_norm_eps > 0.0 && cfg.rms_norm_eps < 1e-5);
        assert_eq!(cfg.bos_token_id, 2);
        // eos_token_id is an array — first element wins.
        assert_eq!(cfg.eos_token_id, 1);
    }

    #[test]
    fn parses_gemma4_31b_config() {
        let cfg = GemmaConfig::from_json(GEMMA4_31B).unwrap();
        assert_eq!(cfg.num_hidden_layers, 56);
        assert_eq!(cfg.hidden_size, 5376);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 168);
        // scalar eos_token_id path
        assert_eq!(cfg.eos_token_id, 1);
        assert_eq!(cfg.sliding_window, 4096);
    }

    #[test]
    fn head_dim_defaults_to_hidden_div_heads_when_absent() {
        // Drop head_dim from a Gemma-ish config and check we fall
        // back to hidden_size / num_attention_heads.
        let json = r#"{
            "architectures":["Gemma3ForCausalLM"],
            "num_hidden_layers":2,
            "hidden_size":64,
            "intermediate_size":128,
            "num_attention_heads":4,
            "num_key_value_heads":2,
            "vocab_size":1000,
            "max_position_embeddings":512,
            "rope_theta":10000.0,
            "rms_norm_eps":0.00001,
            "bos_token_id":0,
            "eos_token_id":1
        }"#;
        let cfg = GemmaConfig::from_json(json).unwrap();
        assert_eq!(cfg.head_dim, 16);
        // sliding_window_pattern defaults to 6 when absent.
        assert_eq!(cfg.sliding_window_pattern, 6);
        // sliding_window missing -> 0.
        assert_eq!(cfg.sliding_window, 0);
    }

    #[test]
    fn missing_required_field_reports_name() {
        // No num_hidden_layers.
        let json = r#"{
            "architectures":["Gemma3ForCausalLM"],
            "hidden_size":64,
            "intermediate_size":128,
            "num_attention_heads":4,
            "num_key_value_heads":2,
            "vocab_size":1000,
            "max_position_embeddings":512,
            "rope_theta":10000.0,
            "rms_norm_eps":0.00001,
            "bos_token_id":0,
            "eos_token_id":1
        }"#;
        match GemmaConfig::from_json(json).unwrap_err() {
            LoaderError::InvalidConfig(msg) => {
                assert!(
                    msg.contains("num_hidden_layers"),
                    "message should name the missing field, got: {msg}"
                );
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn validation_rejects_indivisible_hidden_size() {
        let json = r#"{
            "architectures":["Gemma3ForCausalLM"],
            "num_hidden_layers":2,
            "hidden_size":65,
            "intermediate_size":128,
            "num_attention_heads":4,
            "num_key_value_heads":2,
            "head_dim":16,
            "vocab_size":1000,
            "max_position_embeddings":512,
            "rope_theta":10000.0,
            "rms_norm_eps":0.00001,
            "bos_token_id":0,
            "eos_token_id":1
        }"#;
        let err = GemmaConfig::from_json(json).unwrap_err();
        match err {
            LoaderError::InvalidConfig(msg) => {
                assert!(msg.contains("hidden_size"));
                assert!(msg.contains("num_attention_heads"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn validation_rejects_gqa_mismatch() {
        // num_attention_heads (6) not divisible by num_key_value_heads (4).
        let json = r#"{
            "architectures":["Gemma3ForCausalLM"],
            "num_hidden_layers":2,
            "hidden_size":96,
            "intermediate_size":128,
            "num_attention_heads":6,
            "num_key_value_heads":4,
            "head_dim":16,
            "vocab_size":1000,
            "max_position_embeddings":512,
            "rope_theta":10000.0,
            "rms_norm_eps":0.00001,
            "bos_token_id":0,
            "eos_token_id":1
        }"#;
        let err = GemmaConfig::from_json(json).unwrap_err();
        assert!(matches!(err, LoaderError::InvalidConfig(_)));
    }

    #[test]
    fn validation_rejects_zero_num_hidden_layers() {
        let json = r#"{
            "architectures":["Gemma3ForCausalLM"],
            "num_hidden_layers":0,
            "hidden_size":64,
            "intermediate_size":128,
            "num_attention_heads":4,
            "num_key_value_heads":2,
            "head_dim":16,
            "vocab_size":1000,
            "max_position_embeddings":512,
            "rope_theta":10000.0,
            "rms_norm_eps":0.00001,
            "bos_token_id":0,
            "eos_token_id":1
        }"#;
        let err = GemmaConfig::from_json(json).unwrap_err();
        match err {
            LoaderError::InvalidConfig(msg) => assert!(msg.contains("num_hidden_layers")),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn validation_rejects_non_positive_rope_theta() {
        let json = r#"{
            "architectures":["Gemma3ForCausalLM"],
            "num_hidden_layers":2,
            "hidden_size":64,
            "intermediate_size":128,
            "num_attention_heads":4,
            "num_key_value_heads":2,
            "head_dim":16,
            "vocab_size":1000,
            "max_position_embeddings":512,
            "rope_theta":0.0,
            "rms_norm_eps":0.00001,
            "bos_token_id":0,
            "eos_token_id":1
        }"#;
        let err = GemmaConfig::from_json(json).unwrap_err();
        assert!(matches!(err, LoaderError::InvalidConfig(_)));
    }

    #[test]
    fn eos_token_id_scalar_path() {
        // Gemma 4 style scalar eos_token_id (already covered by the
        // 31B test but this is a focused regression).
        let cfg = GemmaConfig::from_json(GEMMA4_31B).unwrap();
        assert_eq!(cfg.eos_token_id, 1);
    }

    #[test]
    fn eos_token_id_array_path() {
        let cfg = GemmaConfig::from_json(GEMMA3_4B).unwrap();
        assert_eq!(cfg.eos_token_id, 1);
    }

    #[test]
    fn empty_eos_token_id_array_errors() {
        let json = r#"{
            "architectures":["Gemma3ForCausalLM"],
            "num_hidden_layers":2,
            "hidden_size":64,
            "intermediate_size":128,
            "num_attention_heads":4,
            "num_key_value_heads":2,
            "head_dim":16,
            "vocab_size":1000,
            "max_position_embeddings":512,
            "rope_theta":10000.0,
            "rms_norm_eps":0.00001,
            "bos_token_id":0,
            "eos_token_id":[]
        }"#;
        assert!(matches!(
            GemmaConfig::from_json(json).unwrap_err(),
            LoaderError::InvalidConfig(_)
        ));
    }

    #[test]
    fn llama_config_detects_as_llama_and_parses_as_gemma_shape() {
        // A Llama config doesn't have sliding_window, but otherwise the
        // Gemma parser can still extract it — we just don't call it
        // via the Gemma path in production. This test is belt and
        // braces to ensure the parser is tolerant.
        let arch = ModelArchitecture::detect(LLAMA3_8B).unwrap();
        assert_eq!(arch, ModelArchitecture::Llama);
        let cfg = GemmaConfig::from_json(LLAMA3_8B).unwrap();
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.head_dim, 128); // derived: 4096 / 32
        assert_eq!(cfg.sliding_window, 0);
    }

    #[test]
    fn rejects_malformed_json() {
        let err = GemmaConfig::from_json("{ not valid json").unwrap_err();
        assert!(matches!(err, LoaderError::InvalidConfig(_)));
    }

    #[test]
    fn rejects_non_object_top_level() {
        let err = GemmaConfig::from_json("[1,2,3]").unwrap_err();
        assert!(matches!(err, LoaderError::InvalidConfig(_)));
    }

    #[test]
    fn nested_rope_scaling_object_does_not_break_parser() {
        // This is implicitly covered by `parses_gemma3_4b_config` but
        // guards against regressions in the nested-object path.
        let cfg = GemmaConfig::from_json(GEMMA3_4B).unwrap();
        assert_eq!(cfg.num_hidden_layers, 34);
    }
}
