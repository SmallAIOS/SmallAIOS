// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Single-file safetensors reader.
//!
//! A safetensors file has the layout:
//!
//! ```text
//! [ 8 bytes: LE u64 header_len ][ header_len bytes: UTF-8 JSON ][ payload ]
//! ```
//!
//! The JSON header is a flat object mapping tensor name to an object
//! with `dtype` (string), `shape` (array of ints), and `data_offsets`
//! (2-element array of ints — relative to the start of the payload).
//! An optional `__metadata__` key holds free-form metadata and is
//! ignored here.
//!
//! This reader memory-maps the file and exposes `tensor_view`, which
//! returns a borrowed slice into the payload region along with the
//! decoded dtype and shape. No tensor bytes are ever copied.

use super::LoaderError;
use crate::tensor::DataType;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Parsed header entry describing a single tensor inside a safetensors
/// file. Offsets are **relative to the start of the file** (already
/// adjusted from the raw header offsets which are relative to the end
/// of the header).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// Decoded ONNX-runtime data type.
    pub dtype: DataType,
    /// Shape as i64 dimensions (matches onnx-rt `TensorShape` convention).
    pub shape: Vec<i64>,
    /// Absolute start offset of the tensor bytes in the mmap region.
    pub data_offset_start: usize,
    /// Absolute end offset (exclusive) of the tensor bytes in the mmap region.
    pub data_offset_end: usize,
}

/// Borrowed view into a tensor inside a memory-mapped safetensors file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorView<'a> {
    /// Decoded data type.
    pub dtype: DataType,
    /// Shape (i64 per dimension).
    pub shape: &'a [i64],
    /// Raw bytes of the tensor, borrowed from the mmap region.
    pub data: &'a [u8],
}

/// A memory-mapped safetensors file with a parsed header index.
pub struct SafetensorsFile {
    mmap: Mmap,
    entries: BTreeMap<String, TensorEntry>,
}

impl core::fmt::Debug for SafetensorsFile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SafetensorsFile")
            .field("num_tensors", &self.entries.len())
            .field("mmap_len", &self.mmap.len())
            .finish()
    }
}

impl SafetensorsFile {
    /// Open and map a safetensors file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let file = File::open(path.as_ref()).map_err(|e| LoaderError::Io(format!("open: {e}")))?;
        // SAFETY: we hold the `File` implicitly via the returned `Mmap`
        // (memmap2 keeps the file descriptor alive internally). The
        // usual mmap UB caveat — concurrent modification by another
        // process — is accepted at the API boundary: callers are
        // expected to point at immutable model weight files.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::Io(format!("mmap: {e}")))?;
        Self::from_mmap(mmap)
    }

    /// Build a `SafetensorsFile` from an already-constructed `Mmap`.
    /// Useful for tests that fabricate payloads in memory.
    pub fn from_mmap(mmap: Mmap) -> Result<Self, LoaderError> {
        let entries = parse_header(&mmap)?;
        Ok(Self { mmap, entries })
    }

    /// Iterate tensor names in lexicographic order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    /// Number of tensors in the file.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the file contains any tensors.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a tensor entry by name without borrowing the payload.
    pub fn entry(&self, name: &str) -> Option<&TensorEntry> {
        self.entries.get(name)
    }

    /// Return a zero-copy view into a tensor's bytes in the mmap region.
    pub fn tensor_view(&self, name: &str) -> Option<TensorView<'_>> {
        let entry = self.entries.get(name)?;
        let data = &self.mmap[entry.data_offset_start..entry.data_offset_end];
        Some(TensorView {
            dtype: entry.dtype,
            shape: &entry.shape,
            data,
        })
    }
}

/// Parse the 8-byte length prefix + JSON header region of a mmap'd file.
fn parse_header(bytes: &[u8]) -> Result<BTreeMap<String, TensorEntry>, LoaderError> {
    let (header_json, payload_start) = split_header_region(bytes)?;
    let raw = parse_json_object(header_json)?;

    let mut out = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry = build_tensor_entry(&name, value, payload_start, bytes.len())?;
        out.insert(name, entry);
    }
    Ok(out)
}

/// Split a safetensors file into (header JSON, payload start offset).
fn split_header_region(bytes: &[u8]) -> Result<(&str, usize), LoaderError> {
    if bytes.len() < 8 {
        return Err(LoaderError::InvalidHeader(
            "file smaller than 8-byte length prefix".to_string(),
        ));
    }
    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) as usize;
    let header_start = 8usize;
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| LoaderError::InvalidHeader("header length overflows usize".to_string()))?;
    if header_end > bytes.len() {
        return Err(LoaderError::InvalidHeader(format!(
            "header length {header_len} exceeds file size {}",
            bytes.len()
        )));
    }
    let header_json = core::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| LoaderError::InvalidHeader(format!("header not valid UTF-8: {e}")))?;
    Ok((header_json, header_end))
}

/// Build a single [`TensorEntry`] from the parsed JSON object value.
fn build_tensor_entry(
    name: &str,
    value: JsonValue,
    payload_start: usize,
    file_len: usize,
) -> Result<TensorEntry, LoaderError> {
    let obj = match value {
        JsonValue::Object(o) => o,
        _ => {
            return Err(LoaderError::InvalidHeader(format!(
                "tensor entry {name} is not a JSON object"
            )))
        }
    };
    let dtype = decode_entry_dtype(name, &obj)?;
    let shape = decode_entry_shape(name, &obj)?;
    let (rel_start, rel_end) = decode_entry_offsets(name, &obj)?;
    let (abs_start, abs_end) =
        resolve_absolute_offsets(name, payload_start, file_len, rel_start, rel_end)?;
    Ok(TensorEntry {
        dtype,
        shape,
        data_offset_start: abs_start,
        data_offset_end: abs_end,
    })
}

/// Find a key in a JSON object payload, or fail with a helpful error.
fn find_obj_field<'a>(
    obj: &'a [(String, JsonValue)],
    key: &str,
    name: &str,
) -> Result<&'a JsonValue, LoaderError> {
    obj.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .ok_or_else(|| LoaderError::InvalidHeader(format!("{name}: missing {key}")))
}

fn decode_entry_dtype(name: &str, obj: &[(String, JsonValue)]) -> Result<DataType, LoaderError> {
    let dtype_val = find_obj_field(obj, "dtype", name)?;
    let dtype_str = match dtype_val {
        JsonValue::String(s) => s.as_str(),
        _ => {
            return Err(LoaderError::InvalidHeader(format!(
                "{name}: dtype must be a string"
            )))
        }
    };
    dtype_from_str(dtype_str).ok_or_else(|| LoaderError::UnsupportedDtype(dtype_str.to_string()))
}

fn decode_entry_shape(name: &str, obj: &[(String, JsonValue)]) -> Result<Vec<i64>, LoaderError> {
    let shape_val = find_obj_field(obj, "shape", name)?;
    let arr = match shape_val {
        JsonValue::Array(arr) => arr,
        _ => {
            return Err(LoaderError::InvalidHeader(format!(
                "{name}: shape must be an array"
            )))
        }
    };
    let mut dims = Vec::with_capacity(arr.len());
    for dim in arr {
        match dim {
            JsonValue::Number(n) => dims.push(*n),
            _ => {
                return Err(LoaderError::InvalidHeader(format!(
                    "{name}: shape element not a number"
                )))
            }
        }
    }
    Ok(dims)
}

fn decode_entry_offsets(
    name: &str,
    obj: &[(String, JsonValue)],
) -> Result<(i64, i64), LoaderError> {
    let offsets_val = find_obj_field(obj, "data_offsets", name)?;
    match offsets_val {
        JsonValue::Array(arr) if arr.len() == 2 => {
            let a = offset_element(name, &arr[0], 0)?;
            let b = offset_element(name, &arr[1], 1)?;
            Ok((a, b))
        }
        _ => Err(LoaderError::InvalidHeader(format!(
            "{name}: data_offsets must be a 2-element array"
        ))),
    }
}

fn offset_element(name: &str, value: &JsonValue, idx: usize) -> Result<i64, LoaderError> {
    match value {
        JsonValue::Number(n) => Ok(*n),
        _ => Err(LoaderError::InvalidHeader(format!(
            "{name}: data_offsets[{idx}] not a number"
        ))),
    }
}

fn resolve_absolute_offsets(
    name: &str,
    payload_start: usize,
    file_len: usize,
    rel_start: i64,
    rel_end: i64,
) -> Result<(usize, usize), LoaderError> {
    if rel_start < 0 || rel_end < rel_start {
        return Err(LoaderError::InvalidHeader(format!(
            "{name}: invalid data_offsets [{rel_start}, {rel_end}]"
        )));
    }
    let abs_start = payload_start
        .checked_add(rel_start as usize)
        .ok_or_else(|| LoaderError::InvalidHeader(format!("{name}: offset overflow")))?;
    let abs_end = payload_start
        .checked_add(rel_end as usize)
        .ok_or_else(|| LoaderError::InvalidHeader(format!("{name}: offset overflow")))?;
    if abs_end > file_len {
        return Err(LoaderError::InvalidHeader(format!(
            "{name}: data_offsets end {abs_end} exceeds file size {file_len}"
        )));
    }
    Ok((abs_start, abs_end))
}

/// Convert a single ASCII hex digit to its numeric value.
fn hex_digit_value(d: u8) -> Option<u8> {
    match d {
        b'0'..=b'9' => Some(d - b'0'),
        b'a'..=b'f' => Some(d - b'a' + 10),
        b'A'..=b'F' => Some(d - b'A' + 10),
        _ => None,
    }
}

/// Map safetensors dtype strings to ONNX-runtime `DataType`.
fn dtype_from_str(s: &str) -> Option<DataType> {
    match s {
        "BF16" => Some(DataType::BFloat16),
        "F16" => Some(DataType::Float16),
        "F32" => Some(DataType::Float),
        "I8" => Some(DataType::Int8),
        "I32" => Some(DataType::Int32),
        "I64" => Some(DataType::Int64),
        "U8" => Some(DataType::Uint8),
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Minimal JSON parser (object / array / string / number).
//
// Scope: enough for safetensors headers and the two-level
// `model.safetensors.index.json` manifest consumed by `sharded.rs`.
// No floats, no escapes beyond `\"`, `\\`, `\/`, `\n`, `\r`, `\t`,
// and `\u00XX` (ASCII-range unicode escapes). Numbers are parsed as
// i64.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum JsonValue {
    Null,
    Bool(bool),
    Number(i64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

pub(super) fn parse_json_object(input: &str) -> Result<Vec<(String, JsonValue)>, LoaderError> {
    let mut p = JsonParser::new(input);
    p.skip_ws();
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(LoaderError::InvalidHeader(format!(
            "trailing data after JSON at byte {}",
            p.pos
        )));
    }
    match v {
        JsonValue::Object(o) => Ok(o),
        _ => Err(LoaderError::InvalidHeader(
            "top-level JSON is not an object".to_string(),
        )),
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
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
            Some(c) => Err(LoaderError::InvalidHeader(format!(
                "expected '{}' got '{}' at {}",
                b as char, c as char, self.pos
            ))),
            None => Err(LoaderError::InvalidHeader(format!(
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
            Some(c) => Err(LoaderError::InvalidHeader(format!(
                "unexpected byte '{}' at {}",
                c as char, self.pos
            ))),
            None => Err(LoaderError::InvalidHeader(
                "unexpected EOF in JSON value".to_string(),
            )),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, LoaderError> {
        self.expect(b'{')?;
        let mut out: Vec<(String, JsonValue)> = Vec::new();
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
            out.push((key, value));
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(JsonValue::Object(out)),
                Some(c) => {
                    return Err(LoaderError::InvalidHeader(format!(
                        "expected ',' or '}}' got '{}' at {}",
                        c as char, self.pos
                    )))
                }
                None => {
                    return Err(LoaderError::InvalidHeader(
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
                    return Err(LoaderError::InvalidHeader(format!(
                        "expected ',' or ']' got '{}' at {}",
                        c as char, self.pos
                    )))
                }
                None => {
                    return Err(LoaderError::InvalidHeader(
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
                Some(b'\\') => self.consume_escape(&mut out)?,
                Some(b) => {
                    // Accept UTF-8 continuation bytes verbatim.
                    out.push(b as char);
                }
                None => {
                    return Err(LoaderError::InvalidHeader(
                        "unterminated string".to_string(),
                    ))
                }
            }
        }
    }

    /// Consume one `\<x>` escape sequence (already past the backslash).
    fn consume_escape(&mut self, out: &mut String) -> Result<(), LoaderError> {
        match self.bump() {
            Some(b'"') => out.push('"'),
            Some(b'\\') => out.push('\\'),
            Some(b'/') => out.push('/'),
            Some(b'n') => out.push('\n'),
            Some(b'r') => out.push('\r'),
            Some(b't') => out.push('\t'),
            Some(b'b') => out.push('\u{08}'),
            Some(b'f') => out.push('\u{0C}'),
            Some(b'u') => self.consume_unicode_escape(out)?,
            Some(c) => {
                return Err(LoaderError::InvalidHeader(format!(
                    "bad escape '\\{}'",
                    c as char
                )))
            }
            None => {
                return Err(LoaderError::InvalidHeader(
                    "unterminated escape".to_string(),
                ))
            }
        }
        Ok(())
    }

    /// Consume the 4 hex digits of a `\uXXXX` escape and push the
    /// resulting code point onto `out`.
    fn consume_unicode_escape(&mut self, out: &mut String) -> Result<(), LoaderError> {
        let mut code: u32 = 0;
        for _ in 0..4 {
            let d = self
                .bump()
                .ok_or_else(|| LoaderError::InvalidHeader("truncated \\u escape".to_string()))?;
            let hv = hex_digit_value(d).ok_or_else(|| {
                LoaderError::InvalidHeader(format!("bad hex digit '{}' in \\u escape", d as char))
            })?;
            code = (code << 4) | hv as u32;
        }
        let c = char::from_u32(code).ok_or_else(|| {
            LoaderError::InvalidHeader(format!("invalid unicode codepoint U+{code:04X}"))
        })?;
        out.push(c);
        Ok(())
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
        let slice = &self.bytes[start..self.pos];
        let s = core::str::from_utf8(slice)
            .map_err(|_| LoaderError::InvalidHeader("non-utf8 number".to_string()))?;
        let n: i64 = s
            .parse()
            .map_err(|_| LoaderError::InvalidHeader(format!("bad number '{s}'")))?;
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
            Err(LoaderError::InvalidHeader(
                "expected true/false".to_string(),
            ))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, LoaderError> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(LoaderError::InvalidHeader("expected null".to_string()))
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a minimal safetensors byte vector with the given header
    /// JSON and raw payload. Returns the assembled file bytes.
    pub(crate) fn build_safetensors(header_json: &str, payload: &[u8]) -> Vec<u8> {
        let header_bytes = header_json.as_bytes();
        let mut out = Vec::with_capacity(8 + header_bytes.len() + payload.len());
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(payload);
        out
    }

    /// Wrap a byte vector in a mmap by round-tripping through a tempfile.
    /// `memmap2::Mmap` cannot map an anonymous memory region directly,
    /// so we write to a temp file under `std::env::temp_dir()`.
    pub(crate) fn mmap_from_bytes(tag: &str, bytes: &[u8]) -> Mmap {
        use std::io::Write;
        let mut path = std::env::temp_dir();
        // Include nanos and a tag to avoid collisions across parallel tests.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("smallaios-safetensors-{tag}-{nanos}.bin"));
        {
            let mut f = File::create(&path).expect("create tmp");
            f.write_all(bytes).expect("write tmp");
            f.sync_all().ok();
        }
        let f = File::open(&path).expect("open tmp");
        let mm = unsafe { Mmap::map(&f) }.expect("mmap tmp");
        // Best-effort cleanup; file handle is held by the mmap on Linux.
        let _ = std::fs::remove_file(&path);
        mm
    }

    #[test]
    fn parses_single_tensor() {
        // Two F32 values = 8 bytes.
        let payload: [u8; 8] = [0, 0, 0x80, 0x3F, 0, 0, 0, 0x40]; // [1.0, 2.0]
        let header = r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let bytes = build_safetensors(header, &payload);
        let mm = mmap_from_bytes("single", &bytes);
        let st = SafetensorsFile::from_mmap(mm).expect("parse");
        assert_eq!(st.len(), 1);
        let v = st.tensor_view("w").expect("lookup");
        assert_eq!(v.dtype, DataType::Float);
        assert_eq!(v.shape, &[2i64]);
        assert_eq!(v.data, &payload[..]);
    }

    #[test]
    fn parses_multiple_tensors_and_dtypes() {
        // 4 BF16 values = 8 bytes, then 2 I64 values = 16 bytes.
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0x00, 0x3F, 0x80, 0x3F, 0x00, 0x40, 0x40, 0x40]);
        payload.extend_from_slice(&7i64.to_le_bytes());
        payload.extend_from_slice(&(-3i64).to_le_bytes());
        let header = r#"{"__metadata__":{"framework":"pt"},"bf":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"ids":{"dtype":"I64","shape":[2],"data_offsets":[8,24]}}"#;
        let bytes = build_safetensors(header, &payload);
        let mm = mmap_from_bytes("multi", &bytes);
        let st = SafetensorsFile::from_mmap(mm).expect("parse");
        assert_eq!(st.len(), 2);
        assert!(st.tensor_view("__metadata__").is_none());

        let bf = st.tensor_view("bf").expect("bf");
        assert_eq!(bf.dtype, DataType::BFloat16);
        assert_eq!(bf.shape, &[2i64, 2]);
        assert_eq!(bf.data.len(), 8);

        let ids = st.tensor_view("ids").expect("ids");
        assert_eq!(ids.dtype, DataType::Int64);
        assert_eq!(ids.shape, &[2i64]);
        let a = i64::from_le_bytes(ids.data[..8].try_into().unwrap());
        let b = i64::from_le_bytes(ids.data[8..].try_into().unwrap());
        assert_eq!(a, 7);
        assert_eq!(b, -3);
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let header = r#"{"w":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#;
        let bytes = build_safetensors(header, &[0u8; 8]);
        let mm = mmap_from_bytes("f64", &bytes);
        let err = SafetensorsFile::from_mmap(mm).unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(_)));
    }

    #[test]
    fn rejects_truncated_payload() {
        // Declares 16 bytes but only supplies 4.
        let header = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let bytes = build_safetensors(header, &[0u8; 4]);
        let mm = mmap_from_bytes("trunc", &bytes);
        let err = SafetensorsFile::from_mmap(mm).unwrap_err();
        assert!(matches!(err, LoaderError::InvalidHeader(_)));
    }

    #[test]
    fn rejects_short_prefix() {
        let mm = mmap_from_bytes("tiny", &[1u8, 2, 3]);
        let err = SafetensorsFile::from_mmap(mm).unwrap_err();
        assert!(matches!(err, LoaderError::InvalidHeader(_)));
    }

    #[test]
    fn tensor_not_found_returns_none() {
        let header = r#"{"w":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let bytes = build_safetensors(header, &[7u8, 8]);
        let mm = mmap_from_bytes("nf", &bytes);
        let st = SafetensorsFile::from_mmap(mm).unwrap();
        assert!(st.tensor_view("missing").is_none());
    }
}
