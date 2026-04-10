// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CDR (Common Data Representation) v2 serializer and deserializer.
//!
//! Implements XCDR2 (OMG CDR encoding) for DDS data exchange. CDR defines
//! a wire format for all primitive types, strings, sequences, and structs
//! with configurable endianness and natural alignment.

use crate::BusError;
use alloc::string::String;
use alloc::vec::Vec;

/// CDR byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// Big-endian (network byte order).
    BigEndian,
    /// Little-endian.
    LittleEndian,
}

/// CDR serializer — writes typed values into a byte buffer.
#[derive(Debug)]
pub struct CdrSerializer {
    buf: Vec<u8>,
    endian: Endianness,
}

impl CdrSerializer {
    /// Create a new CDR serializer with the given endianness.
    pub fn new(endian: Endianness) -> Self {
        Self {
            buf: Vec::new(),
            endian,
        }
    }

    /// Returns the serialized bytes, consuming the serializer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Returns the current byte length.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Align the buffer position to the given alignment.
    fn align(&mut self, alignment: usize) {
        let rem = self.buf.len() % alignment;
        if rem != 0 {
            let padding = alignment - rem;
            self.buf.extend(core::iter::repeat_n(0u8, padding));
        }
    }

    /// Serialize a `bool` (1 byte).
    pub fn serialize_bool(&mut self, val: bool) {
        self.buf.push(if val { 1 } else { 0 });
    }

    /// Serialize a `u8`.
    pub fn serialize_u8(&mut self, val: u8) {
        self.buf.push(val);
    }

    /// Serialize an `i8`.
    pub fn serialize_i8(&mut self, val: i8) {
        self.buf.push(val as u8);
    }

    /// Serialize a `u16` with alignment.
    pub fn serialize_u16(&mut self, val: u16) {
        self.align(2);
        let bytes = match self.endian {
            Endianness::BigEndian => val.to_be_bytes(),
            Endianness::LittleEndian => val.to_le_bytes(),
        };
        self.buf.extend_from_slice(&bytes);
    }

    /// Serialize an `i16` with alignment.
    pub fn serialize_i16(&mut self, val: i16) {
        self.align(2);
        let bytes = match self.endian {
            Endianness::BigEndian => val.to_be_bytes(),
            Endianness::LittleEndian => val.to_le_bytes(),
        };
        self.buf.extend_from_slice(&bytes);
    }

    /// Serialize a `u32` with alignment.
    pub fn serialize_u32(&mut self, val: u32) {
        self.align(4);
        let bytes = match self.endian {
            Endianness::BigEndian => val.to_be_bytes(),
            Endianness::LittleEndian => val.to_le_bytes(),
        };
        self.buf.extend_from_slice(&bytes);
    }

    /// Serialize an `i32` with alignment.
    pub fn serialize_i32(&mut self, val: i32) {
        self.align(4);
        let bytes = match self.endian {
            Endianness::BigEndian => val.to_be_bytes(),
            Endianness::LittleEndian => val.to_le_bytes(),
        };
        self.buf.extend_from_slice(&bytes);
    }

    /// Serialize a `u64` with alignment.
    pub fn serialize_u64(&mut self, val: u64) {
        self.align(8);
        let bytes = match self.endian {
            Endianness::BigEndian => val.to_be_bytes(),
            Endianness::LittleEndian => val.to_le_bytes(),
        };
        self.buf.extend_from_slice(&bytes);
    }

    /// Serialize an `i64` with alignment.
    pub fn serialize_i64(&mut self, val: i64) {
        self.align(8);
        let bytes = match self.endian {
            Endianness::BigEndian => val.to_be_bytes(),
            Endianness::LittleEndian => val.to_le_bytes(),
        };
        self.buf.extend_from_slice(&bytes);
    }

    /// Serialize an `f32` with alignment.
    pub fn serialize_f32(&mut self, val: f32) {
        self.serialize_u32(val.to_bits());
    }

    /// Serialize an `f64` with alignment.
    pub fn serialize_f64(&mut self, val: f64) {
        self.serialize_u64(val.to_bits());
    }

    /// Serialize a CDR string (length-prefixed, NUL-terminated).
    ///
    /// CDR strings are encoded as: u32 length (including NUL) + bytes + NUL.
    pub fn serialize_string(&mut self, val: &str) {
        let len = val.len() as u32 + 1; // +1 for NUL terminator
        self.serialize_u32(len);
        self.buf.extend_from_slice(val.as_bytes());
        self.buf.push(0); // NUL terminator
    }

    /// Serialize a byte sequence (length-prefixed).
    pub fn serialize_byte_seq(&mut self, val: &[u8]) {
        self.serialize_u32(val.len() as u32);
        self.buf.extend_from_slice(val);
    }

    /// Serialize a sequence of `u32` values (length-prefixed).
    pub fn serialize_u32_seq(&mut self, vals: &[u32]) {
        self.serialize_u32(vals.len() as u32);
        for &v in vals {
            self.serialize_u32(v);
        }
    }
}

/// CDR deserializer — reads typed values from a byte buffer.
#[derive(Debug)]
pub struct CdrDeserializer<'a> {
    data: &'a [u8],
    pos: usize,
    endian: Endianness,
}

impl<'a> CdrDeserializer<'a> {
    /// Create a new CDR deserializer.
    pub fn new(data: &'a [u8], endian: Endianness) -> Self {
        Self {
            data,
            pos: 0,
            endian,
        }
    }

    /// Returns the current byte position.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Returns the number of bytes remaining.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Align the read position to the given alignment.
    fn align(&mut self, alignment: usize) {
        let rem = self.pos % alignment;
        if rem != 0 {
            self.pos += alignment - rem;
        }
    }

    /// Read N bytes from the current position.
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], BusError> {
        if self.pos + n > self.data.len() {
            return Err(BusError::FrameTooShort);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Deserialize a `bool`.
    pub fn deserialize_bool(&mut self) -> Result<bool, BusError> {
        let b = self.read_bytes(1)?;
        Ok(b[0] != 0)
    }

    /// Deserialize a `u8`.
    pub fn deserialize_u8(&mut self) -> Result<u8, BusError> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    /// Deserialize an `i8`.
    pub fn deserialize_i8(&mut self) -> Result<i8, BusError> {
        let b = self.read_bytes(1)?;
        Ok(b[0] as i8)
    }

    /// Deserialize a `u16`.
    pub fn deserialize_u16(&mut self) -> Result<u16, BusError> {
        self.align(2);
        let b = self.read_bytes(2)?;
        Ok(match self.endian {
            Endianness::BigEndian => u16::from_be_bytes([b[0], b[1]]),
            Endianness::LittleEndian => u16::from_le_bytes([b[0], b[1]]),
        })
    }

    /// Deserialize an `i16`.
    pub fn deserialize_i16(&mut self) -> Result<i16, BusError> {
        self.align(2);
        let b = self.read_bytes(2)?;
        Ok(match self.endian {
            Endianness::BigEndian => i16::from_be_bytes([b[0], b[1]]),
            Endianness::LittleEndian => i16::from_le_bytes([b[0], b[1]]),
        })
    }

    /// Deserialize a `u32`.
    pub fn deserialize_u32(&mut self) -> Result<u32, BusError> {
        self.align(4);
        let b = self.read_bytes(4)?;
        Ok(match self.endian {
            Endianness::BigEndian => u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
            Endianness::LittleEndian => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        })
    }

    /// Deserialize an `i32`.
    pub fn deserialize_i32(&mut self) -> Result<i32, BusError> {
        self.align(4);
        let b = self.read_bytes(4)?;
        Ok(match self.endian {
            Endianness::BigEndian => i32::from_be_bytes([b[0], b[1], b[2], b[3]]),
            Endianness::LittleEndian => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        })
    }

    /// Deserialize a `u64`.
    pub fn deserialize_u64(&mut self) -> Result<u64, BusError> {
        self.align(8);
        let b = self.read_bytes(8)?;
        Ok(match self.endian {
            Endianness::BigEndian => u64::from_be_bytes(b.try_into().unwrap()),
            Endianness::LittleEndian => u64::from_le_bytes(b.try_into().unwrap()),
        })
    }

    /// Deserialize an `i64`.
    pub fn deserialize_i64(&mut self) -> Result<i64, BusError> {
        self.align(8);
        let b = self.read_bytes(8)?;
        Ok(match self.endian {
            Endianness::BigEndian => i64::from_be_bytes(b.try_into().unwrap()),
            Endianness::LittleEndian => i64::from_le_bytes(b.try_into().unwrap()),
        })
    }

    /// Deserialize an `f32`.
    pub fn deserialize_f32(&mut self) -> Result<f32, BusError> {
        let bits = self.deserialize_u32()?;
        Ok(f32::from_bits(bits))
    }

    /// Deserialize an `f64`.
    pub fn deserialize_f64(&mut self) -> Result<f64, BusError> {
        let bits = self.deserialize_u64()?;
        Ok(f64::from_bits(bits))
    }

    /// Deserialize a CDR string (length-prefixed, NUL-terminated).
    pub fn deserialize_string(&mut self) -> Result<String, BusError> {
        let len = self.deserialize_u32()? as usize;
        if len == 0 {
            return Ok(String::new());
        }
        let bytes = self.read_bytes(len)?;
        // Strip NUL terminator
        let str_len = if bytes[len - 1] == 0 { len - 1 } else { len };
        let s = core::str::from_utf8(&bytes[..str_len]).map_err(|_| BusError::InvalidHeader)?;
        Ok(String::from(s))
    }

    /// Deserialize a byte sequence (length-prefixed).
    pub fn deserialize_byte_seq(&mut self) -> Result<Vec<u8>, BusError> {
        let len = self.deserialize_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        Ok(Vec::from(bytes))
    }

    /// Deserialize a sequence of `u32` values (length-prefixed).
    pub fn deserialize_u32_seq(&mut self) -> Result<Vec<u32>, BusError> {
        let len = self.deserialize_u32()? as usize;
        let mut vals = Vec::with_capacity(len);
        for _ in 0..len {
            vals.push(self.deserialize_u32()?);
        }
        Ok(vals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // === Primitive roundtrip tests ===

    #[test]
    fn test_bool_roundtrip() {
        for endian in [Endianness::BigEndian, Endianness::LittleEndian] {
            let mut ser = CdrSerializer::new(endian);
            ser.serialize_bool(true);
            ser.serialize_bool(false);
            let data = ser.into_bytes();
            let mut de = CdrDeserializer::new(&data, endian);
            assert!(de.deserialize_bool().unwrap());
            assert!(!de.deserialize_bool().unwrap());
        }
    }

    #[test]
    fn test_u8_roundtrip() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_u8(0);
        ser.serialize_u8(127);
        ser.serialize_u8(255);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_u8().unwrap(), 0);
        assert_eq!(de.deserialize_u8().unwrap(), 127);
        assert_eq!(de.deserialize_u8().unwrap(), 255);
    }

    #[test]
    fn test_i8_roundtrip() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_i8(-128);
        ser.serialize_i8(0);
        ser.serialize_i8(127);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_i8().unwrap(), -128);
        assert_eq!(de.deserialize_i8().unwrap(), 0);
        assert_eq!(de.deserialize_i8().unwrap(), 127);
    }

    #[test]
    fn test_u16_roundtrip_le() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_u16(0);
        ser.serialize_u16(0x1234);
        ser.serialize_u16(0xFFFF);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_u16().unwrap(), 0);
        assert_eq!(de.deserialize_u16().unwrap(), 0x1234);
        assert_eq!(de.deserialize_u16().unwrap(), 0xFFFF);
    }

    #[test]
    fn test_u16_roundtrip_be() {
        let mut ser = CdrSerializer::new(Endianness::BigEndian);
        ser.serialize_u16(0xABCD);
        let data = ser.into_bytes();
        assert_eq!(&data, &[0xAB, 0xCD]);
        let mut de = CdrDeserializer::new(&data, Endianness::BigEndian);
        assert_eq!(de.deserialize_u16().unwrap(), 0xABCD);
    }

    #[test]
    fn test_i16_roundtrip() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_i16(-32768);
        ser.serialize_i16(0);
        ser.serialize_i16(32767);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_i16().unwrap(), -32768);
        assert_eq!(de.deserialize_i16().unwrap(), 0);
        assert_eq!(de.deserialize_i16().unwrap(), 32767);
    }

    #[test]
    fn test_u32_roundtrip() {
        for endian in [Endianness::BigEndian, Endianness::LittleEndian] {
            let mut ser = CdrSerializer::new(endian);
            ser.serialize_u32(0);
            ser.serialize_u32(0x12345678);
            ser.serialize_u32(0xFFFFFFFF);
            let data = ser.into_bytes();
            let mut de = CdrDeserializer::new(&data, endian);
            assert_eq!(de.deserialize_u32().unwrap(), 0);
            assert_eq!(de.deserialize_u32().unwrap(), 0x12345678);
            assert_eq!(de.deserialize_u32().unwrap(), 0xFFFFFFFF);
        }
    }

    #[test]
    fn test_i32_roundtrip() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_i32(i32::MIN);
        ser.serialize_i32(0);
        ser.serialize_i32(i32::MAX);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_i32().unwrap(), i32::MIN);
        assert_eq!(de.deserialize_i32().unwrap(), 0);
        assert_eq!(de.deserialize_i32().unwrap(), i32::MAX);
    }

    #[test]
    fn test_u64_roundtrip() {
        for endian in [Endianness::BigEndian, Endianness::LittleEndian] {
            let mut ser = CdrSerializer::new(endian);
            ser.serialize_u64(0);
            ser.serialize_u64(0x0102030405060708);
            ser.serialize_u64(u64::MAX);
            let data = ser.into_bytes();
            let mut de = CdrDeserializer::new(&data, endian);
            assert_eq!(de.deserialize_u64().unwrap(), 0);
            assert_eq!(de.deserialize_u64().unwrap(), 0x0102030405060708);
            assert_eq!(de.deserialize_u64().unwrap(), u64::MAX);
        }
    }

    #[test]
    fn test_i64_roundtrip() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_i64(i64::MIN);
        ser.serialize_i64(0);
        ser.serialize_i64(i64::MAX);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_i64().unwrap(), i64::MIN);
        assert_eq!(de.deserialize_i64().unwrap(), 0);
        assert_eq!(de.deserialize_i64().unwrap(), i64::MAX);
    }

    #[test]
    fn test_f32_roundtrip() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_f32(0.0);
        ser.serialize_f32(2.5);
        ser.serialize_f32(-1.0e10);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_f32().unwrap(), 0.0);
        assert!((de.deserialize_f32().unwrap() - 2.5).abs() < 1e-6);
        assert_eq!(de.deserialize_f32().unwrap(), -1.0e10);
    }

    #[test]
    fn test_f64_roundtrip() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_f64(core::f64::consts::PI);
        ser.serialize_f64(-0.0);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_f64().unwrap(), core::f64::consts::PI);
        assert_eq!(de.deserialize_f64().unwrap(), -0.0);
    }

    // === String tests ===

    #[test]
    fn test_string_roundtrip() {
        for endian in [Endianness::BigEndian, Endianness::LittleEndian] {
            let mut ser = CdrSerializer::new(endian);
            ser.serialize_string("Hello, DDS!");
            let data = ser.into_bytes();
            let mut de = CdrDeserializer::new(&data, endian);
            assert_eq!(de.deserialize_string().unwrap(), "Hello, DDS!");
        }
    }

    #[test]
    fn test_empty_string() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_string("");
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_string().unwrap(), "");
    }

    #[test]
    fn test_string_nul_terminated() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_string("abc");
        let data = ser.into_bytes();
        // length prefix (4 bytes) + "abc" (3) + NUL (1) = 8
        assert_eq!(data.len(), 8);
        assert_eq!(data[7], 0); // NUL terminator
    }

    // === Sequence tests ===

    #[test]
    fn test_byte_seq_roundtrip() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_byte_seq(&[0x01, 0x02, 0x03, 0xFF]);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(
            de.deserialize_byte_seq().unwrap(),
            &[0x01, 0x02, 0x03, 0xFF]
        );
    }

    #[test]
    fn test_empty_byte_seq() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_byte_seq(&[]);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert!(de.deserialize_byte_seq().unwrap().is_empty());
    }

    #[test]
    fn test_u32_seq_roundtrip() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_u32_seq(&[100, 200, 300]);
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_u32_seq().unwrap(), &[100, 200, 300]);
    }

    // === Alignment tests ===

    #[test]
    fn test_alignment_u8_then_u16() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_u8(0xAA);
        ser.serialize_u16(0x1234);
        let data = ser.into_bytes();
        // u8(1) + pad(1) + u16(2) = 4
        assert_eq!(data.len(), 4);
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_u8().unwrap(), 0xAA);
        assert_eq!(de.deserialize_u16().unwrap(), 0x1234);
    }

    #[test]
    fn test_alignment_u8_then_u32() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_u8(0xBB);
        ser.serialize_u32(0xDEADBEEF);
        let data = ser.into_bytes();
        // u8(1) + pad(3) + u32(4) = 8
        assert_eq!(data.len(), 8);
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_u8().unwrap(), 0xBB);
        assert_eq!(de.deserialize_u32().unwrap(), 0xDEADBEEF);
    }

    #[test]
    fn test_alignment_u8_then_u64() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_u8(0xCC);
        ser.serialize_u64(0x0102030405060708);
        let data = ser.into_bytes();
        // u8(1) + pad(7) + u64(8) = 16
        assert_eq!(data.len(), 16);
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_u8().unwrap(), 0xCC);
        assert_eq!(de.deserialize_u64().unwrap(), 0x0102030405060708);
    }

    // === Struct-like serialization test ===

    #[test]
    fn test_struct_serialization() {
        // Simulate a struct: { u8 kind; u32 id; string name; f64 value; }
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_u8(1); // kind
        ser.serialize_u32(42); // id (aligned to 4)
        ser.serialize_string("sensor"); // name
        ser.serialize_f64(98.6); // value (aligned to 8)

        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_u8().unwrap(), 1);
        assert_eq!(de.deserialize_u32().unwrap(), 42);
        assert_eq!(de.deserialize_string().unwrap(), "sensor");
        assert_eq!(de.deserialize_f64().unwrap(), 98.6);
    }

    // === Error tests ===

    #[test]
    fn test_deserialize_past_end() {
        let data = [0u8; 2];
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        de.deserialize_u16().unwrap();
        assert_eq!(de.deserialize_u8().err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_deserialize_u32_insufficient() {
        let data = [0u8; 3];
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_u32().err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_serializer_len_empty() {
        let ser = CdrSerializer::new(Endianness::LittleEndian);
        assert!(ser.is_empty());
        assert_eq!(ser.len(), 0);
    }

    #[test]
    fn test_deserializer_remaining() {
        let data = [0u8; 10];
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.remaining(), 10);
        de.deserialize_u8().unwrap();
        assert_eq!(de.remaining(), 9);
    }

    #[test]
    fn test_multiple_strings() {
        let mut ser = CdrSerializer::new(Endianness::LittleEndian);
        ser.serialize_string("hello");
        ser.serialize_string("world");
        let data = ser.into_bytes();
        let mut de = CdrDeserializer::new(&data, Endianness::LittleEndian);
        assert_eq!(de.deserialize_string().unwrap(), "hello");
        assert_eq!(de.deserialize_string().unwrap(), "world");
    }
}
