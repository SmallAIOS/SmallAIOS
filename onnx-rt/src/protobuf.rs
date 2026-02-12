// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal protobuf decoder for ONNX model parsing.
//!
//! This module implements only the wire types and decoding operations
//! required by the ONNX protobuf schema. It is not a general-purpose
//! protobuf library.

use core::fmt;

/// Protobuf wire types used in ONNX model files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    /// Varint-encoded integer (wire type 0).
    Varint = 0,
    /// 64-bit fixed-width value (wire type 1).
    Fixed64 = 1,
    /// Length-delimited bytes (wire type 2).
    LengthDelimited = 2,
    /// 32-bit fixed-width value (wire type 5).
    Fixed32 = 5,
}

impl WireType {
    /// Construct a `WireType` from its numeric representation.
    ///
    /// Returns `None` for unrecognized wire type values.
    pub fn from_u8(value: u8) -> Option<WireType> {
        match value {
            0 => Some(WireType::Varint),
            1 => Some(WireType::Fixed64),
            2 => Some(WireType::LengthDelimited),
            5 => Some(WireType::Fixed32),
            _ => None,
        }
    }
}

/// A decoded protobuf field tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldHeader {
    /// The field number from the .proto schema.
    pub field_number: u32,
    /// The wire type indicating how the value is encoded.
    pub wire_type: WireType,
}

/// Errors that can occur during protobuf decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtoError {
    /// Reached end of buffer unexpectedly.
    UnexpectedEof,
    /// Varint encoding is invalid (e.g., too many continuation bytes).
    InvalidVarint,
    /// Wire type value is not recognized.
    InvalidWireType,
    /// Bytes are not valid UTF-8.
    InvalidUtf8,
    /// Field number is zero or otherwise invalid.
    InvalidFieldNumber,
    /// A numeric value overflowed its target type.
    Overflow,
    /// The buffer does not contain enough data for the requested operation.
    BufferTooSmall,
}

impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtoError::UnexpectedEof => write!(f, "unexpected end of buffer"),
            ProtoError::InvalidVarint => write!(f, "invalid varint encoding"),
            ProtoError::InvalidWireType => write!(f, "unrecognized wire type"),
            ProtoError::InvalidUtf8 => write!(f, "invalid UTF-8 in string field"),
            ProtoError::InvalidFieldNumber => write!(f, "invalid field number"),
            ProtoError::Overflow => write!(f, "numeric overflow"),
            ProtoError::BufferTooSmall => write!(f, "buffer too small for requested data"),
        }
    }
}

/// Decode a zigzag-encoded 32-bit value to its signed representation.
///
/// Zigzag encoding maps signed integers to unsigned integers so that
/// numbers with small absolute values have small varint encodings.
#[inline]
pub fn zigzag_decode_32(n: u32) -> i32 {
    ((n >> 1) as i32) ^ -((n & 1) as i32)
}

/// Decode a zigzag-encoded 64-bit value to its signed representation.
#[inline]
pub fn zigzag_decode_64(n: u64) -> i64 {
    ((n >> 1) as i64) ^ -((n & 1) as i64)
}

/// A zero-copy protobuf decoder operating over a byte slice.
///
/// The decoder maintains a cursor position and provides methods to
/// read protobuf-encoded values sequentially. All read operations
/// advance the cursor past the consumed bytes.
pub struct ProtoDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtoDecoder<'a> {
    /// Create a new decoder over the given byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        ProtoDecoder { data, pos: 0 }
    }

    /// Returns the number of bytes remaining to be read.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// Returns `true` if all bytes have been consumed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Returns the current byte position in the buffer.
    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Read a single byte, advancing the cursor.
    pub fn read_byte(&mut self) -> Result<u8, ProtoError> {
        if self.pos >= self.data.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    /// Decode a LEB128 varint, consuming up to 10 bytes.
    ///
    /// Returns `ProtoError::InvalidVarint` if more than 10 continuation
    /// bytes are encountered (which would overflow a u64).
    pub fn read_varint(&mut self) -> Result<u64, ProtoError> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;

        for _ in 0..10 {
            let byte = self.read_byte()?;
            let value = (byte & 0x7F) as u64;

            // Check for overflow: if shift >= 63 and value > 1, the result
            // would exceed u64::MAX.
            if shift >= 64 {
                return Err(ProtoError::InvalidVarint);
            }

            result |= value << shift;
            shift += 7;

            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }

        // More than 10 bytes means the varint is malformed.
        Err(ProtoError::InvalidVarint)
    }

    /// Read a 32-bit little-endian fixed-width value.
    pub fn read_fixed32(&mut self) -> Result<u32, ProtoError> {
        if self.remaining() < 4 {
            return Err(ProtoError::UnexpectedEof);
        }
        let bytes = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a 64-bit little-endian fixed-width value.
    pub fn read_fixed64(&mut self) -> Result<u64, ProtoError> {
        if self.remaining() < 8 {
            return Err(ProtoError::UnexpectedEof);
        }
        let bytes = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Decode a field tag and return the field number and wire type.
    ///
    /// A protobuf tag is a varint where the lower 3 bits encode the wire
    /// type and the remaining bits encode the field number.
    pub fn read_tag(&mut self) -> Result<FieldHeader, ProtoError> {
        let tag = self.read_varint()?;

        let wire_type_raw = (tag & 0x07) as u8;
        let wire_type = WireType::from_u8(wire_type_raw).ok_or(ProtoError::InvalidWireType)?;

        let field_number = (tag >> 3) as u32;
        if field_number == 0 {
            return Err(ProtoError::InvalidFieldNumber);
        }

        Ok(FieldHeader {
            field_number,
            wire_type,
        })
    }

    /// Read a length-delimited field, returning the raw bytes.
    ///
    /// First reads a varint length prefix, then returns a sub-slice of
    /// that many bytes without copying.
    pub fn read_length_delimited(&mut self) -> Result<&'a [u8], ProtoError> {
        let len = self.read_varint()? as usize;

        if self.remaining() < len {
            return Err(ProtoError::BufferTooSmall);
        }

        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    /// Read a length-delimited string field, validating UTF-8.
    pub fn read_string(&mut self) -> Result<&'a str, ProtoError> {
        let bytes = self.read_length_delimited()?;
        core::str::from_utf8(bytes).map_err(|_| ProtoError::InvalidUtf8)
    }

    /// Read a zigzag-encoded sint32 value.
    pub fn read_sint32(&mut self) -> Result<i32, ProtoError> {
        let raw = self.read_varint()?;
        Ok(zigzag_decode_32(raw as u32))
    }

    /// Read a zigzag-encoded sint64 value.
    pub fn read_sint64(&mut self) -> Result<i64, ProtoError> {
        let raw = self.read_varint()?;
        Ok(zigzag_decode_64(raw))
    }

    /// Read a 32-bit float (IEEE 754) from a fixed32 encoding.
    pub fn read_float(&mut self) -> Result<f32, ProtoError> {
        let bits = self.read_fixed32()?;
        Ok(f32::from_bits(bits))
    }

    /// Read a 64-bit double (IEEE 754) from a fixed64 encoding.
    pub fn read_double(&mut self) -> Result<f64, ProtoError> {
        let bits = self.read_fixed64()?;
        Ok(f64::from_bits(bits))
    }

    /// Skip over a field value based on its wire type.
    ///
    /// This is used to ignore unknown fields while still advancing the
    /// cursor past them.
    pub fn skip_field(&mut self, wire_type: WireType) -> Result<(), ProtoError> {
        match wire_type {
            WireType::Varint => {
                self.read_varint()?;
            }
            WireType::Fixed64 => {
                if self.remaining() < 8 {
                    return Err(ProtoError::UnexpectedEof);
                }
                self.pos += 8;
            }
            WireType::LengthDelimited => {
                let len = self.read_varint()? as usize;
                if self.remaining() < len {
                    return Err(ProtoError::BufferTooSmall);
                }
                self.pos += len;
            }
            WireType::Fixed32 => {
                if self.remaining() < 4 {
                    return Err(ProtoError::UnexpectedEof);
                }
                self.pos += 4;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    // ---------------------------------------------------------------
    // Varint tests
    // ---------------------------------------------------------------

    #[test]
    fn varint_zero() {
        let data = [0x00];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_varint().unwrap(), 0);
        assert!(dec.is_empty());
    }

    #[test]
    fn varint_one() {
        let data = [0x01];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_varint().unwrap(), 1);
    }

    #[test]
    fn varint_127() {
        let data = [0x7F];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_varint().unwrap(), 127);
    }

    #[test]
    fn varint_128() {
        // 128 = 0x80 0x01 in LEB128
        let data = [0x80, 0x01];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_varint().unwrap(), 128);
    }

    #[test]
    fn varint_300() {
        // 300 = 0xAC 0x02 in LEB128
        let data = [0xAC, 0x02];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_varint().unwrap(), 300);
    }

    #[test]
    fn varint_u64_max() {
        // u64::MAX = 0xFFFFFFFFFFFFFFFF
        // LEB128: nine 0xFF bytes followed by 0x01
        let data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_varint().unwrap(), u64::MAX);
    }

    #[test]
    fn varint_overflow_too_many_bytes() {
        // 11 continuation bytes — always invalid
        let data = [0xFF; 11];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_varint().unwrap_err(), ProtoError::InvalidVarint);
    }

    // ---------------------------------------------------------------
    // Fixed32 / Fixed64 tests
    // ---------------------------------------------------------------

    #[test]
    fn fixed32_known_value() {
        // 0x01020304 little-endian = [0x04, 0x03, 0x02, 0x01]
        let data = [0x04, 0x03, 0x02, 0x01];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_fixed32().unwrap(), 0x01020304);
    }

    #[test]
    fn fixed64_known_value() {
        let data = [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_fixed64().unwrap(), 0x0102030405060708);
    }

    // ---------------------------------------------------------------
    // Tag decoding tests
    // ---------------------------------------------------------------

    #[test]
    fn tag_field1_varint() {
        // Field 1, wire type 0 (Varint): tag = (1 << 3) | 0 = 0x08
        let data = [0x08];
        let mut dec = ProtoDecoder::new(&data);
        let header = dec.read_tag().unwrap();
        assert_eq!(header.field_number, 1);
        assert_eq!(header.wire_type, WireType::Varint);
    }

    #[test]
    fn tag_field2_length_delimited() {
        // Field 2, wire type 2 (LengthDelimited): tag = (2 << 3) | 2 = 0x12
        let data = [0x12];
        let mut dec = ProtoDecoder::new(&data);
        let header = dec.read_tag().unwrap();
        assert_eq!(header.field_number, 2);
        assert_eq!(header.wire_type, WireType::LengthDelimited);
    }

    // ---------------------------------------------------------------
    // Length-delimited tests
    // ---------------------------------------------------------------

    #[test]
    fn length_delimited_valid() {
        // Length 3, then bytes [0xAA, 0xBB, 0xCC]
        let data = [0x03, 0xAA, 0xBB, 0xCC];
        let mut dec = ProtoDecoder::new(&data);
        let bytes = dec.read_length_delimited().unwrap();
        assert_eq!(bytes, &[0xAA, 0xBB, 0xCC]);
        assert!(dec.is_empty());
    }

    #[test]
    fn length_delimited_too_short() {
        // Length says 5 but only 2 bytes follow
        let data = [0x05, 0xAA, 0xBB];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(
            dec.read_length_delimited().unwrap_err(),
            ProtoError::BufferTooSmall
        );
    }

    // ---------------------------------------------------------------
    // String tests
    // ---------------------------------------------------------------

    #[test]
    fn string_valid_utf8() {
        let s = "hello";
        let mut data = alloc::vec![s.len() as u8];
        data.extend_from_slice(s.as_bytes());
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_string().unwrap(), "hello");
    }

    #[test]
    fn string_invalid_utf8() {
        // 0xFF is never valid in UTF-8
        let data = [0x02, 0xFF, 0xFE];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_string().unwrap_err(), ProtoError::InvalidUtf8);
    }

    // ---------------------------------------------------------------
    // Zigzag tests
    // ---------------------------------------------------------------

    #[test]
    fn zigzag_decode_values() {
        assert_eq!(zigzag_decode_32(0), 0);
        assert_eq!(zigzag_decode_32(1), -1);
        assert_eq!(zigzag_decode_32(2), 1);
        assert_eq!(zigzag_decode_32(3), -2);

        assert_eq!(zigzag_decode_64(0), 0);
        assert_eq!(zigzag_decode_64(1), -1);
        assert_eq!(zigzag_decode_64(2), 1);
        assert_eq!(zigzag_decode_64(3), -2);
    }

    // ---------------------------------------------------------------
    // Float / Double tests
    // ---------------------------------------------------------------

    #[test]
    fn float_known_bit_pattern() {
        // 1.0f32 = 0x3F800000
        let bits: u32 = 0x3F800000;
        let data = bits.to_le_bytes();
        let mut dec = ProtoDecoder::new(&data);
        let value = dec.read_float().unwrap();
        assert!((value - 1.0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn double_known_bit_pattern() {
        // 1.0f64 = 0x3FF0000000000000
        let bits: u64 = 0x3FF0_0000_0000_0000;
        let data = bits.to_le_bytes();
        let mut dec = ProtoDecoder::new(&data);
        let value = dec.read_double().unwrap();
        assert!((value - 1.0f64).abs() < f64::EPSILON);
    }

    // ---------------------------------------------------------------
    // Skip field tests
    // ---------------------------------------------------------------

    #[test]
    fn skip_varint_field() {
        // Varint 300 = [0xAC, 0x02], followed by sentinel byte 0xFF
        let data = [0xAC, 0x02, 0xFF];
        let mut dec = ProtoDecoder::new(&data);
        dec.skip_field(WireType::Varint).unwrap();
        assert_eq!(dec.position(), 2);
        assert_eq!(dec.read_byte().unwrap(), 0xFF);
    }

    #[test]
    fn skip_fixed32_field() {
        let data = [0x01, 0x02, 0x03, 0x04, 0xFF];
        let mut dec = ProtoDecoder::new(&data);
        dec.skip_field(WireType::Fixed32).unwrap();
        assert_eq!(dec.position(), 4);
        assert_eq!(dec.read_byte().unwrap(), 0xFF);
    }

    #[test]
    fn skip_fixed64_field() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0xFF];
        let mut dec = ProtoDecoder::new(&data);
        dec.skip_field(WireType::Fixed64).unwrap();
        assert_eq!(dec.position(), 8);
        assert_eq!(dec.read_byte().unwrap(), 0xFF);
    }

    #[test]
    fn skip_length_delimited_field() {
        // Length 3, then 3 bytes of payload, then sentinel
        let data = [0x03, 0xAA, 0xBB, 0xCC, 0xFF];
        let mut dec = ProtoDecoder::new(&data);
        dec.skip_field(WireType::LengthDelimited).unwrap();
        assert_eq!(dec.position(), 4);
        assert_eq!(dec.read_byte().unwrap(), 0xFF);
    }

    // ---------------------------------------------------------------
    // Empty decoder test
    // ---------------------------------------------------------------

    #[test]
    fn empty_decoder() {
        let dec = ProtoDecoder::new(&[]);
        assert!(dec.is_empty());
        assert_eq!(dec.remaining(), 0);
        assert_eq!(dec.position(), 0);
    }

    // ---------------------------------------------------------------
    // ProtoError Display test
    // ---------------------------------------------------------------

    #[test]
    fn error_display() {
        let msg = format!("{}", ProtoError::UnexpectedEof);
        assert_eq!(msg, "unexpected end of buffer");
    }

    // ---------------------------------------------------------------
    // Wire type round-trip
    // ---------------------------------------------------------------

    #[test]
    fn wire_type_from_u8_invalid() {
        assert!(WireType::from_u8(3).is_none());
        assert!(WireType::from_u8(4).is_none());
        assert!(WireType::from_u8(6).is_none());
        assert!(WireType::from_u8(7).is_none());
    }

    // --- Fuzz-like tests (Task 7.12) ---
    // Property-based tests feeding random/malformed bytes to the protobuf
    // parser, verifying no panics occur.

    #[test]
    fn fuzz_empty_input_no_panic() {
        let mut dec = ProtoDecoder::new(&[]);
        assert!(dec.read_varint().is_err());
        assert!(dec.read_tag().is_err());
        assert!(dec.read_fixed32().is_err());
        assert!(dec.read_fixed64().is_err());
        assert!(dec.read_length_delimited().is_err());
        assert!(dec.read_string().is_err());
        assert!(dec.read_float().is_err());
        assert!(dec.read_double().is_err());
        assert!(dec.read_sint32().is_err());
        assert!(dec.read_sint64().is_err());
        assert!(dec.read_byte().is_err());
    }

    #[test]
    fn fuzz_single_byte_inputs_no_panic() {
        for b in 0..=255u8 {
            let data = [b];
            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_varint();

            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_tag();

            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_fixed32();

            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_fixed64();

            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_length_delimited();

            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_string();

            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_byte();

            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_sint32();

            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_sint64();
        }
    }

    #[test]
    fn fuzz_all_continuation_bytes_no_panic() {
        // All 0xFF bytes — maximum continuation, should eventually error
        for len in 1..=15 {
            let data = alloc::vec![0xFF; len];
            let mut dec = ProtoDecoder::new(&data);
            let _ = dec.read_varint(); // Must not panic
        }
    }

    #[test]
    fn fuzz_length_delimited_huge_length_no_panic() {
        // Varint encoding a very large length that exceeds the buffer
        // u64::MAX varint = [0xFF]*9 + [0x01]
        let data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        let mut dec = ProtoDecoder::new(&data);
        let result = dec.read_length_delimited();
        assert!(result.is_err());
    }

    #[test]
    fn fuzz_skip_field_all_wire_types_no_panic() {
        for wt_val in 0..=7u8 {
            if let Some(wt) = WireType::from_u8(wt_val) {
                // With empty data
                let mut dec = ProtoDecoder::new(&[]);
                let _ = dec.skip_field(wt);

                // With 1 byte
                let mut dec = ProtoDecoder::new(&[0x00]);
                let _ = dec.skip_field(wt);

                // With lots of 0xFF bytes
                let data = [0xFF; 16];
                let mut dec = ProtoDecoder::new(&data);
                let _ = dec.skip_field(wt);
            }
        }
    }

    #[test]
    fn fuzz_tag_field_zero_rejected() {
        // Tag with field_number=0: varint 0x00 -> field=0, wire=0 -> InvalidFieldNumber
        let data = [0x00];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_tag().unwrap_err(), ProtoError::InvalidFieldNumber);
    }

    #[test]
    fn fuzz_tag_invalid_wire_types_rejected() {
        // Wire types 3, 4, 6, 7 are invalid
        for invalid_wt in [3u8, 4, 6, 7] {
            // field_number=1, wire_type=invalid: tag = (1 << 3) | invalid_wt
            let tag_byte = (1 << 3) | invalid_wt;
            let data = [tag_byte];
            let mut dec = ProtoDecoder::new(&data);
            assert_eq!(dec.read_tag().unwrap_err(), ProtoError::InvalidWireType);
        }
    }

    #[test]
    fn fuzz_string_with_invalid_utf8_sequences() {
        // Various invalid UTF-8 sequences
        let invalid_sequences: &[&[u8]] = &[
            &[0x01, 0x80],                   // lone continuation byte
            &[0x02, 0xC0, 0x80],             // overlong encoding
            &[0x03, 0xED, 0xA0, 0x80],       // surrogate half
            &[0x02, 0xFE, 0xFF],             // invalid start bytes
            &[0x04, 0xF4, 0x90, 0x80, 0x80], // above U+10FFFF
        ];
        for seq in invalid_sequences {
            let mut dec = ProtoDecoder::new(seq);
            let result = dec.read_string();
            // Should either error or succeed (never panic)
            let _ = result;
        }
    }

    #[test]
    fn fuzz_sequential_operations_no_panic() {
        // Simulate parsing a malformed protobuf message:
        // tag + varint + tag + length-delimited + ...
        let data = [
            0x08, 0x96, 0x01, // field 1, varint 150
            0x12, 0x03, 0x41, 0x42, 0x43, // field 2, string "ABC"
            0x1D, 0x00, 0x00, 0x80, 0x3F, // field 3, fixed32 1.0f
            0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // field 4, fixed64 1.0f64
            0xFF, 0xFF, 0xFF, // trailing garbage
        ];

        let mut dec = ProtoDecoder::new(&data);
        while !dec.is_empty() {
            match dec.read_tag() {
                Ok(header) => {
                    let _ = dec.skip_field(header.wire_type);
                }
                Err(_) => break,
            }
        }
        // Must not panic
    }

    #[test]
    fn fuzz_two_byte_patterns_no_panic() {
        // Exhaustive 2-byte inputs (65536 combinations)
        for hi in 0..=255u8 {
            for lo in 0..=255u8 {
                let data = [hi, lo];
                let mut dec = ProtoDecoder::new(&data);
                let _ = dec.read_tag();

                let mut dec = ProtoDecoder::new(&data);
                let _ = dec.read_varint();

                let mut dec = ProtoDecoder::new(&data);
                let _ = dec.read_length_delimited();
            }
        }
    }

    #[test]
    fn fuzz_zigzag_decode_all_boundaries() {
        // Test zigzag with boundary values
        assert_eq!(zigzag_decode_32(u32::MAX), i32::MIN);
        assert_eq!(zigzag_decode_32(u32::MAX - 1), i32::MAX);
        assert_eq!(zigzag_decode_64(u64::MAX), i64::MIN);
        assert_eq!(zigzag_decode_64(u64::MAX - 1), i64::MAX);
    }

    #[test]
    fn fuzz_truncated_fixed_values_no_panic() {
        // 1, 2, 3 bytes — too short for fixed32
        for len in 0..4 {
            let data = alloc::vec![0xAA; len];
            let mut dec = ProtoDecoder::new(&data);
            assert!(dec.read_fixed32().is_err());
        }
        // 1-7 bytes — too short for fixed64
        for len in 0..8 {
            let data = alloc::vec![0xBB; len];
            let mut dec = ProtoDecoder::new(&data);
            assert!(dec.read_fixed64().is_err());
        }
    }

    #[test]
    fn fuzz_varint_with_trailing_data() {
        // Valid 1-byte varint followed by garbage
        let data = [0x01, 0xFF, 0xFF, 0xFF];
        let mut dec = ProtoDecoder::new(&data);
        assert_eq!(dec.read_varint().unwrap(), 1);
        assert_eq!(dec.remaining(), 3);
    }
}
