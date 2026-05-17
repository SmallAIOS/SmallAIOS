// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Protobuf wire-format primitives.
//!
//! Reference: [Protobuf encoding](https://protobuf.dev/programming-guides/encoding/).
//!
//! Wire types we accept:
//!
//! | code | name              | used for                      |
//! |-----:|-------------------|-------------------------------|
//! | 0    | VARINT            | int32, int64, uint32, uint64, sint32, sint64, bool, enum |
//! | 1    | I64               | fixed64, sfixed64, double     |
//! | 2    | LEN               | string, bytes, embedded msg, packed repeated |
//! | 5    | I32               | fixed32, sfixed32, float      |
//!
//! Group wire types (3, 4) are obsolete and are rejected with
//! [`WireError::UnknownWireType`].

use super::{Result, WireError, DEFAULT_FIELD_CAP};
use alloc::string::String;
use alloc::vec::Vec;

/// Wire-type codes per the protobuf encoding spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WireType {
    Varint = 0,
    I64 = 1,
    Len = 2,
    I32 = 5,
}

impl WireType {
    /// Decode the low 3 bits of a field key into a [`WireType`].
    pub fn from_u8(b: u8) -> Result<Self> {
        match b & 0x7 {
            0 => Ok(Self::Varint),
            1 => Ok(Self::I64),
            2 => Ok(Self::Len),
            5 => Ok(Self::I32),
            _ => Err(WireError::UnknownWireType),
        }
    }
}

/// Maximum number of bytes a varint can occupy when representing a
/// `u64` — 10 bytes (RFC: ceil(64 / 7) = 10).
pub const VARINT_MAX_BYTES: usize = 10;

/// Decode an unsigned varint starting at `cursor` in `input`.
///
/// On success, advances `cursor` past the varint and returns the
/// decoded `u64`. Rejects truncated input, varints longer than
/// 10 bytes, and the special `u64::MAX + 1` overflow case.
pub fn decode_varint(input: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let start = *cursor;
    while *cursor < input.len() {
        let b = input[*cursor];
        *cursor += 1;
        if shift >= 64 {
            return Err(WireError::VarintOverflow);
        }
        let chunk = (b & 0x7f) as u64;
        // Reject bits that would not fit; the high 7-bit chunk is
        // capped to 1 valid bit on a `u64`.
        if shift == 63 && chunk > 1 {
            return Err(WireError::VarintOverflow);
        }
        value |= chunk << shift;
        if b & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if *cursor - start > VARINT_MAX_BYTES {
            return Err(WireError::VarintOverflow);
        }
    }
    Err(WireError::Truncated)
}

/// Encode an unsigned varint into `out`.
pub fn encode_varint(value: u64, out: &mut Vec<u8>) {
    let mut v = value;
    while v >= 0x80 {
        out.push(((v as u8) & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Decode a zig-zag-encoded signed varint (`sint32` / `sint64`).
pub fn decode_sint(input: &[u8], cursor: &mut usize) -> Result<i64> {
    let n = decode_varint(input, cursor)?;
    Ok(((n >> 1) as i64) ^ -((n & 1) as i64))
}

/// Encode a signed value with zig-zag encoding into `out`.
pub fn encode_sint(value: i64, out: &mut Vec<u8>) {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint(zigzag, out);
}

/// Decode a fixed 4-byte little-endian unsigned integer.
pub fn decode_fixed32(input: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > input.len() {
        return Err(WireError::Truncated);
    }
    let v = u32::from_le_bytes([
        input[*cursor],
        input[*cursor + 1],
        input[*cursor + 2],
        input[*cursor + 3],
    ]);
    *cursor += 4;
    Ok(v)
}

/// Decode a fixed 8-byte little-endian unsigned integer.
pub fn decode_fixed64(input: &[u8], cursor: &mut usize) -> Result<u64> {
    if *cursor + 8 > input.len() {
        return Err(WireError::Truncated);
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&input[*cursor..*cursor + 8]);
    *cursor += 8;
    Ok(u64::from_le_bytes(bytes))
}

/// Decode a length-delimited byte slice. Returns the borrowed
/// slice and advances `cursor` past it.
///
/// `cap` bounds the maximum length we accept. Pass
/// [`DEFAULT_FIELD_CAP`] when there is no specific upper bound.
pub fn decode_len_delim<'a>(input: &'a [u8], cursor: &mut usize, cap: usize) -> Result<&'a [u8]> {
    let len = decode_varint(input, cursor)? as usize;
    if len > cap {
        return Err(WireError::OversizedField);
    }
    if *cursor + len > input.len() {
        return Err(WireError::OversizedField);
    }
    let slice = &input[*cursor..*cursor + len];
    *cursor += len;
    Ok(slice)
}

/// Encode a length-delimited body (length varint + raw bytes).
pub fn encode_len_delim(body: &[u8], out: &mut Vec<u8>) {
    encode_varint(body.len() as u64, out);
    out.extend_from_slice(body);
}

/// Decode a UTF-8 string from a length-delimited field.
pub fn decode_string(input: &[u8], cursor: &mut usize, cap: usize) -> Result<String> {
    let bytes = decode_len_delim(input, cursor, cap)?;
    let s = core::str::from_utf8(bytes).map_err(|_| WireError::InvalidUtf8)?;
    Ok(s.into())
}

/// Encode the field key (`tag << 3 | wire_type`) into `out`.
pub fn encode_key(tag: u32, wt: WireType, out: &mut Vec<u8>) {
    encode_varint(((tag as u64) << 3) | (wt as u64), out);
}

/// Decode a field key, returning `(tag, wire_type)`.
pub fn decode_key(input: &[u8], cursor: &mut usize) -> Result<(u32, WireType)> {
    let key = decode_varint(input, cursor)?;
    let wt = WireType::from_u8(key as u8)?;
    let tag = (key >> 3) as u32;
    Ok((tag, wt))
}

/// Skip a single field whose wire type was already decoded.
/// Used by message decoders to discard unknown / future fields
/// per the protobuf forward-compat contract.
pub fn skip_field(input: &[u8], cursor: &mut usize, wt: WireType) -> Result<()> {
    match wt {
        WireType::Varint => {
            let _ = decode_varint(input, cursor)?;
        }
        WireType::I64 => {
            let _ = decode_fixed64(input, cursor)?;
        }
        WireType::Len => {
            let _ = decode_len_delim(input, cursor, DEFAULT_FIELD_CAP)?;
        }
        WireType::I32 => {
            let _ = decode_fixed32(input, cursor)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip_small() {
        let mut out = Vec::new();
        encode_varint(150, &mut out);
        assert_eq!(out, alloc::vec![0x96, 0x01]);
        let mut cur = 0;
        assert_eq!(decode_varint(&out, &mut cur).unwrap(), 150);
        assert_eq!(cur, 2);
    }

    #[test]
    fn varint_roundtrip_zero() {
        let mut out = Vec::new();
        encode_varint(0, &mut out);
        assert_eq!(out, alloc::vec![0]);
        let mut cur = 0;
        assert_eq!(decode_varint(&out, &mut cur).unwrap(), 0);
    }

    #[test]
    fn varint_roundtrip_u64_max() {
        let mut out = Vec::new();
        encode_varint(u64::MAX, &mut out);
        let mut cur = 0;
        assert_eq!(decode_varint(&out, &mut cur).unwrap(), u64::MAX);
    }

    #[test]
    fn varint_truncated_rejected() {
        let bad = [0x80, 0x80]; // continuation bits set, no terminator.
        let mut cur = 0;
        assert_eq!(
            decode_varint(&bad, &mut cur).unwrap_err(),
            WireError::Truncated
        );
    }

    #[test]
    fn varint_oversized_rejected() {
        // 11 bytes with continuation bit always set ⇒ overflow.
        let bad = [0x80; 11];
        let mut cur = 0;
        assert_eq!(
            decode_varint(&bad, &mut cur).unwrap_err(),
            WireError::VarintOverflow
        );
    }

    #[test]
    fn varint_overflow_high_bit_rejected() {
        // 10 bytes encoding a value > u64::MAX.
        let bad = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02];
        let mut cur = 0;
        assert_eq!(
            decode_varint(&bad, &mut cur).unwrap_err(),
            WireError::VarintOverflow
        );
    }

    #[test]
    fn sint_zigzag_roundtrip() {
        for &v in &[
            0i64,
            -1,
            1,
            -2,
            2,
            i32::MIN as i64,
            i32::MAX as i64,
            i64::MIN,
            i64::MAX,
        ] {
            let mut out = Vec::new();
            encode_sint(v, &mut out);
            let mut cur = 0;
            assert_eq!(decode_sint(&out, &mut cur).unwrap(), v);
        }
    }

    #[test]
    fn fixed32_roundtrip() {
        let buf = [0x78, 0x56, 0x34, 0x12]; // 0x12345678 little-endian
        let mut cur = 0;
        assert_eq!(decode_fixed32(&buf, &mut cur).unwrap(), 0x1234_5678);
    }

    #[test]
    fn fixed64_truncated_rejected() {
        let buf = [1, 2, 3];
        let mut cur = 0;
        assert_eq!(
            decode_fixed64(&buf, &mut cur).unwrap_err(),
            WireError::Truncated
        );
    }

    #[test]
    fn len_delim_roundtrip() {
        let mut out = Vec::new();
        encode_len_delim(b"hello", &mut out);
        let mut cur = 0;
        let body = decode_len_delim(&out, &mut cur, DEFAULT_FIELD_CAP).unwrap();
        assert_eq!(body, b"hello");
    }

    #[test]
    fn len_delim_cap_enforced() {
        let mut out = Vec::new();
        encode_len_delim(b"hello", &mut out);
        let mut cur = 0;
        assert_eq!(
            decode_len_delim(&out, &mut cur, 4).unwrap_err(),
            WireError::OversizedField
        );
    }

    #[test]
    fn len_delim_truncated_rejected() {
        // Length 10 declared, only 3 bytes follow.
        let buf = [10, 1, 2, 3];
        let mut cur = 0;
        assert_eq!(
            decode_len_delim(&buf, &mut cur, DEFAULT_FIELD_CAP).unwrap_err(),
            WireError::OversizedField
        );
    }

    #[test]
    fn string_invalid_utf8_rejected() {
        let mut out = Vec::new();
        encode_len_delim(&[0xff, 0xfe], &mut out);
        let mut cur = 0;
        assert_eq!(
            decode_string(&out, &mut cur, DEFAULT_FIELD_CAP).unwrap_err(),
            WireError::InvalidUtf8
        );
    }

    #[test]
    fn key_roundtrip() {
        let mut out = Vec::new();
        encode_key(15, WireType::Len, &mut out);
        let mut cur = 0;
        let (tag, wt) = decode_key(&out, &mut cur).unwrap();
        assert_eq!(tag, 15);
        assert_eq!(wt, WireType::Len);
    }

    #[test]
    fn group_wire_type_rejected() {
        // Wire type 3 = SGROUP (obsolete).
        let buf = [0x0b]; // tag 1, wire type 3
        let mut cur = 0;
        assert_eq!(
            decode_key(&buf, &mut cur).unwrap_err(),
            WireError::UnknownWireType
        );
    }

    #[test]
    fn skip_unknown_field() {
        let mut buf = Vec::new();
        encode_key(99, WireType::Len, &mut buf);
        encode_len_delim(b"unknown", &mut buf);
        let mut cur = 0;
        let (_tag, wt) = decode_key(&buf, &mut cur).unwrap();
        skip_field(&buf, &mut cur, wt).unwrap();
        assert_eq!(cur, buf.len());
    }
}
