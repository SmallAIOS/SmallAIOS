// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal protobuf decoder for ONNX model parsing.
//!
//! This module implements only the wire types and decoding operations
//! required by the ONNX protobuf schema. It is not a general-purpose
//! protobuf library.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::onnx_types::{
    AttributeProto, AttributeType, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    TensorProto, ValueInfoProto,
};

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
    /// Nested `GraphProto` attributes exceeded [`MAX_GRAPH_NESTING_DEPTH`].
    ///
    /// Emitted when a model's inner-graph attributes (`If.then_branch`,
    /// `Loop.body`, `Scan.body`, etc.) nest more deeply than the parser's
    /// compile-time safety bound. The limit prevents a maliciously-crafted
    /// model from blowing the host stack via unbounded recursion.
    NestingTooDeep,
}

/// Maximum depth of nested `GraphProto` attributes the parser will decode.
///
/// Typical hand-written models nest at most 1–2 levels (a `Loop` body, or a
/// `Loop` containing an `If`). 16 is a generous safety bound that keeps
/// parser stack usage bounded while accommodating any realistic model.
pub const MAX_GRAPH_NESTING_DEPTH: usize = 16;

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
            ProtoError::NestingTooDeep => {
                write!(f, "nested GraphProto attributes exceed maximum depth")
            }
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

    /// Read packed f32 values (fixed32 encoding).
    pub fn read_packed_f32(&mut self, byte_len: usize) -> Result<Vec<f32>, ProtoError> {
        let end = self.pos + byte_len;
        if end > self.data.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        if !byte_len.is_multiple_of(4) {
            return Err(ProtoError::BufferTooSmall);
        }
        let count = byte_len / 4;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_float()?);
        }
        Ok(values)
    }

    /// Read packed i64 values (fixed64 encoding).
    pub fn read_packed_i64_fixed(&mut self, byte_len: usize) -> Result<Vec<i64>, ProtoError> {
        let end = self.pos + byte_len;
        if end > self.data.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        if !byte_len.is_multiple_of(8) {
            return Err(ProtoError::BufferTooSmall);
        }
        let count = byte_len / 8;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_fixed64()? as i64);
        }
        Ok(values)
    }

    /// Read packed varint-encoded i64 values.
    pub fn read_packed_varint_i64(&mut self, byte_len: usize) -> Result<Vec<i64>, ProtoError> {
        let end = self.pos + byte_len;
        if end > self.data.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        let mut values = Vec::new();
        while self.pos < end {
            values.push(self.read_varint()? as i64);
        }
        Ok(values)
    }

    /// Read packed i32 values (fixed32 encoding).
    pub fn read_packed_i32(&mut self, byte_len: usize) -> Result<Vec<i32>, ProtoError> {
        let end = self.pos + byte_len;
        if end > self.data.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        if !byte_len.is_multiple_of(4) {
            return Err(ProtoError::BufferTooSmall);
        }
        let count = byte_len / 4;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_fixed32()? as i32);
        }
        Ok(values)
    }

    /// Read packed varint-encoded i32 values.
    pub fn read_packed_varint_i32(&mut self, byte_len: usize) -> Result<Vec<i32>, ProtoError> {
        let end = self.pos + byte_len;
        if end > self.data.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        let mut values = Vec::new();
        while self.pos < end {
            values.push(self.read_varint()? as i32);
        }
        Ok(values)
    }

    /// Read packed f64 values (fixed64 encoding).
    pub fn read_packed_f64(&mut self, byte_len: usize) -> Result<Vec<f64>, ProtoError> {
        let end = self.pos + byte_len;
        if end > self.data.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        if !byte_len.is_multiple_of(8) {
            return Err(ProtoError::BufferTooSmall);
        }
        let count = byte_len / 8;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_double()?);
        }
        Ok(values)
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

// ---------------------------------------------------------------------------
// ONNX message decoders
// ---------------------------------------------------------------------------

/// Decode an ONNX `OperatorSetIdProto` from raw protobuf bytes.
pub fn decode_opset_import(data: &[u8]) -> Result<OperatorSetIdProto, ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    let mut opset = OperatorSetIdProto::default();
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => opset.domain = String::from(decoder.read_string()?),
            2 => opset.version = decoder.read_varint()? as i64,
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(opset)
}

/// Decode an ONNX `AttributeProto` from raw protobuf bytes.
///
/// Public entry point — starts decoding at nesting depth 0. Inner
/// sub-graph attributes (field 6, `g: GraphProto`) recurse via
/// [`decode_graph_with_depth`] and increment the depth counter.
pub fn decode_attribute(data: &[u8]) -> Result<AttributeProto, ProtoError> {
    decode_attribute_with_depth(data, 0)
}

/// Internal `AttributeProto` decoder that threads a nesting depth counter.
///
/// See [`decode_attribute`] for the public entry point.
fn decode_attribute_with_depth(data: &[u8], depth: usize) -> Result<AttributeProto, ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    let mut attr = AttributeProto::default();
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => attr.name = String::from(decoder.read_string()?),
            2 => attr.f = decoder.read_float()?,
            3 => attr.i = decoder.read_varint()? as i64,
            4 => {
                let bytes = decoder.read_length_delimited()?;
                attr.s = bytes.to_vec();
            }
            5 => {
                // t (TensorProto) — length-delimited nested message.
                // Decode and store the tensor; also flip `attr_type` to
                // Tensor so downstream consumers don't see an Undefined
                // attribute when field 20 is absent.
                let tensor_data = decoder.read_length_delimited()?;
                let tensor = decode_tensor(tensor_data)?;
                attr.t = Some(alloc::boxed::Box::new(tensor));
                if attr.attr_type == AttributeType::Undefined {
                    attr.attr_type = AttributeType::Tensor;
                }
            }
            6 => {
                // g (GraphProto) — length-delimited nested sub-graph body
                // for `If`/`Loop`/`Scan`. Recursively decode the inner
                // graph with a bumped depth counter so malicious models
                // can't blow the host stack.
                if depth + 1 > MAX_GRAPH_NESTING_DEPTH {
                    return Err(ProtoError::NestingTooDeep);
                }
                let graph_data = decoder.read_length_delimited()?;
                let inner = decode_graph_with_depth(graph_data, depth + 1)?;
                attr.g = Some(alloc::boxed::Box::new(inner));
                if attr.attr_type == AttributeType::Undefined {
                    attr.attr_type = AttributeType::Graph;
                }
            }
            7 => {
                // packed floats
                let bytes = decoder.read_length_delimited()?;
                let mut sub = ProtoDecoder::new(bytes);
                attr.floats = sub.read_packed_f32(bytes.len())?;
            }
            8 => {
                // packed varint ints
                let bytes = decoder.read_length_delimited()?;
                let mut sub = ProtoDecoder::new(bytes);
                attr.ints = sub.read_packed_varint_i64(bytes.len())?;
            }
            20 => {
                let type_val = decoder.read_varint()? as i32;
                if let Some(at) = AttributeType::from_i32(type_val) {
                    attr.attr_type = at;
                }
            }
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(attr)
}

/// Decode an ONNX `TensorProto` from raw protobuf bytes.
pub fn decode_tensor(data: &[u8]) -> Result<TensorProto, ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    let mut tensor = TensorProto::default();
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => {
                // dims: packed varint
                let bytes = decoder.read_length_delimited()?;
                let mut sub = ProtoDecoder::new(bytes);
                tensor.dims = sub.read_packed_varint_i64(bytes.len())?;
            }
            2 => tensor.data_type = decoder.read_varint()? as i32,
            4 => {
                // float_data: packed f32
                let bytes = decoder.read_length_delimited()?;
                let mut sub = ProtoDecoder::new(bytes);
                tensor.float_data = sub.read_packed_f32(bytes.len())?;
            }
            5 => {
                // int32_data: packed varint i32
                let bytes = decoder.read_length_delimited()?;
                let mut sub = ProtoDecoder::new(bytes);
                tensor.int32_data = sub.read_packed_varint_i32(bytes.len())?;
            }
            7 => {
                // int64_data: packed varint i64
                let bytes = decoder.read_length_delimited()?;
                let mut sub = ProtoDecoder::new(bytes);
                tensor.int64_data = sub.read_packed_varint_i64(bytes.len())?;
            }
            8 => tensor.name = String::from(decoder.read_string()?),
            9 => {
                // raw_data
                let bytes = decoder.read_length_delimited()?;
                tensor.raw_data = bytes.to_vec();
            }
            10 => {
                // double_data: packed f64
                let bytes = decoder.read_length_delimited()?;
                let mut sub = ProtoDecoder::new(bytes);
                tensor.double_data = sub.read_packed_f64(bytes.len())?;
            }
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(tensor)
}

/// Decode an ONNX `ValueInfoProto` from raw protobuf bytes.
///
/// Flattens the nested TypeProto -> TensorTypeProto -> elem_type + shape.
pub fn decode_value_info(data: &[u8]) -> Result<ValueInfoProto, ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    let mut vi = ValueInfoProto::default();
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => vi.name = String::from(decoder.read_string()?),
            2 => {
                // TypeProto (nested message)
                let type_data = decoder.read_length_delimited()?;
                decode_type_proto(type_data, &mut vi)?;
            }
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(vi)
}

/// Decode a TypeProto, extracting tensor type info into the ValueInfoProto.
fn decode_type_proto(data: &[u8], vi: &mut ValueInfoProto) -> Result<(), ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => {
                // TensorTypeProto (nested message)
                let tensor_type_data = decoder.read_length_delimited()?;
                decode_tensor_type_proto(tensor_type_data, vi)?;
            }
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(())
}

/// Decode a TensorTypeProto, extracting elem_type and shape dims.
fn decode_tensor_type_proto(data: &[u8], vi: &mut ValueInfoProto) -> Result<(), ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => vi.elem_type = decoder.read_varint()? as i32,
            2 => {
                // TensorShapeProto (nested message)
                let shape_data = decoder.read_length_delimited()?;
                decode_tensor_shape_proto(shape_data, vi)?;
            }
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(())
}

/// Decode a TensorShapeProto, extracting dimension values.
fn decode_tensor_shape_proto(data: &[u8], vi: &mut ValueInfoProto) -> Result<(), ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => {
                // Dimension (nested message)
                let dim_data = decoder.read_length_delimited()?;
                let dim = decode_dimension(dim_data)?;
                vi.shape.push(dim);
            }
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(())
}

/// Decode a Dimension, returning the dim_value (or -1 for symbolic).
fn decode_dimension(data: &[u8]) -> Result<i64, ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    let mut dim_value: i64 = -1; // default to symbolic
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => dim_value = decoder.read_varint()? as i64,
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(dim_value)
}

/// Decode an ONNX `NodeProto` from raw protobuf bytes.
///
/// Public entry point — decodes at nesting depth 0. Inner attributes
/// carrying `GraphProto` bodies recurse via [`decode_attribute_with_depth`].
pub fn decode_node(data: &[u8]) -> Result<NodeProto, ProtoError> {
    decode_node_with_depth(data, 0)
}

/// Internal `NodeProto` decoder that threads a nesting depth counter.
fn decode_node_with_depth(data: &[u8], depth: usize) -> Result<NodeProto, ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    let mut node = NodeProto::default();
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => {
                let s = decoder.read_string()?;
                node.input.push(String::from(s));
            }
            2 => {
                let s = decoder.read_string()?;
                node.output.push(String::from(s));
            }
            3 => node.name = String::from(decoder.read_string()?),
            4 => node.op_type = String::from(decoder.read_string()?),
            5 => {
                let sub_data = decoder.read_length_delimited()?;
                node.attribute.push(decode_attribute_with_depth(sub_data, depth)?);
            }
            6 => {
                // doc_string — skip
                decoder.skip_field(header.wire_type)?;
            }
            7 => node.domain = String::from(decoder.read_string()?),
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(node)
}

/// Decode an ONNX `GraphProto` from raw protobuf bytes.
///
/// Public entry point — starts decoding at nesting depth 0.
pub fn decode_graph(data: &[u8]) -> Result<GraphProto, ProtoError> {
    decode_graph_with_depth(data, 0)
}

/// Internal `GraphProto` decoder that threads a nesting depth counter.
///
/// Inner sub-graphs reached via `AttributeProto.g` recurse through this
/// helper with `depth + 1`. [`MAX_GRAPH_NESTING_DEPTH`] caps recursion to
/// keep parser stack usage bounded.
fn decode_graph_with_depth(data: &[u8], depth: usize) -> Result<GraphProto, ProtoError> {
    if depth > MAX_GRAPH_NESTING_DEPTH {
        return Err(ProtoError::NestingTooDeep);
    }
    let mut decoder = ProtoDecoder::new(data);
    let mut graph = GraphProto::default();
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => {
                let sub_data = decoder.read_length_delimited()?;
                graph.node.push(decode_node_with_depth(sub_data, depth)?);
            }
            2 => graph.name = String::from(decoder.read_string()?),
            5 => {
                let sub_data = decoder.read_length_delimited()?;
                graph.initializer.push(decode_tensor(sub_data)?);
            }
            10 => {
                // doc_string — skip
                decoder.skip_field(header.wire_type)?;
            }
            11 => {
                let sub_data = decoder.read_length_delimited()?;
                graph.input.push(decode_value_info(sub_data)?);
            }
            12 => {
                let sub_data = decoder.read_length_delimited()?;
                graph.output.push(decode_value_info(sub_data)?);
            }
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(graph)
}

/// Decode an ONNX `ModelProto` from raw protobuf bytes.
pub fn decode_model(data: &[u8]) -> Result<ModelProto, ProtoError> {
    let mut decoder = ProtoDecoder::new(data);
    let mut model = ModelProto::default();
    while decoder.remaining() > 0 {
        let header = decoder.read_tag()?;
        match header.field_number {
            1 => model.ir_version = decoder.read_varint()? as i64,
            2 => model.producer_name = String::from(decoder.read_string()?),
            3 => model.producer_version = String::from(decoder.read_string()?),
            4 => model.domain = String::from(decoder.read_string()?),
            5 => model.model_version = decoder.read_varint()? as i64,
            6 => model.doc_string = String::from(decoder.read_string()?),
            7 => {
                let sub_data = decoder.read_length_delimited()?;
                model.graph = Some(decode_graph(sub_data)?);
            }
            8 => {
                let sub_data = decoder.read_length_delimited()?;
                model.opset_import.push(decode_opset_import(sub_data)?);
            }
            _ => decoder.skip_field(header.wire_type)?,
        }
    }
    Ok(model)
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

    // ---------------------------------------------------------------
    // Protobuf encoding helpers for tests
    // ---------------------------------------------------------------

    fn encode_varint(value: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut v = value;
        loop {
            let mut byte = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            buf.push(byte);
            if v == 0 {
                break;
            }
        }
        buf
    }

    fn encode_tag(field: u32, wire_type: u8) -> Vec<u8> {
        encode_varint(((field as u64) << 3) | wire_type as u64)
    }

    fn encode_length_delimited(field: u32, data: &[u8]) -> Vec<u8> {
        let mut buf = encode_tag(field, 2);
        buf.extend(encode_varint(data.len() as u64));
        buf.extend(data);
        buf
    }

    fn encode_varint_field(field: u32, value: u64) -> Vec<u8> {
        let mut buf = encode_tag(field, 0);
        buf.extend(encode_varint(value));
        buf
    }

    fn encode_fixed32_field(field: u32, value: u32) -> Vec<u8> {
        let mut buf = encode_tag(field, 5);
        buf.extend(value.to_le_bytes());
        buf
    }

    // ---------------------------------------------------------------
    // Packed reader tests
    // ---------------------------------------------------------------

    #[test]
    fn packed_f32_empty() {
        let mut dec = ProtoDecoder::new(&[]);
        let result = dec.read_packed_f32(0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn packed_f32_single() {
        let data = 1.5f32.to_le_bytes();
        let mut dec = ProtoDecoder::new(&data);
        let result = dec.read_packed_f32(4).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn packed_f32_multiple() {
        let mut data = Vec::new();
        data.extend(1.0f32.to_le_bytes());
        data.extend(2.0f32.to_le_bytes());
        data.extend(3.0f32.to_le_bytes());
        let mut dec = ProtoDecoder::new(&data);
        let result = dec.read_packed_f32(12).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < f32::EPSILON);
        assert!((result[1] - 2.0).abs() < f32::EPSILON);
        assert!((result[2] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn packed_f32_truncated() {
        let data = [0u8; 3]; // not enough for 4 bytes
        let mut dec = ProtoDecoder::new(&data);
        assert!(dec.read_packed_f32(6).is_err());
    }

    #[test]
    fn packed_i64_fixed_empty() {
        let mut dec = ProtoDecoder::new(&[]);
        let result = dec.read_packed_i64_fixed(0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn packed_i64_fixed_single() {
        let data = 42u64.to_le_bytes();
        let mut dec = ProtoDecoder::new(&data);
        let result = dec.read_packed_i64_fixed(8).unwrap();
        assert_eq!(result, alloc::vec![42i64]);
    }

    #[test]
    fn packed_i64_fixed_truncated() {
        let data = [0u8; 5];
        let mut dec = ProtoDecoder::new(&data);
        assert!(dec.read_packed_i64_fixed(10).is_err());
    }

    #[test]
    fn packed_varint_i64_empty() {
        let mut dec = ProtoDecoder::new(&[]);
        let result = dec.read_packed_varint_i64(0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn packed_varint_i64_multiple() {
        let mut data = Vec::new();
        data.extend(encode_varint(1));
        data.extend(encode_varint(300));
        data.extend(encode_varint(0));
        let len = data.len();
        let mut dec = ProtoDecoder::new(&data);
        let result = dec.read_packed_varint_i64(len).unwrap();
        assert_eq!(result, alloc::vec![1i64, 300, 0]);
    }

    #[test]
    fn packed_varint_i64_truncated() {
        let data = [0x80]; // continuation byte with no follow-up
        let mut dec = ProtoDecoder::new(&data);
        // byte_len check passes (1 <= 1), but varint read will fail
        // because 0x80 has continuation bit set but there's no next byte
        // Actually the end check passes since pos + byte_len <= data.len(),
        // but the while loop tries to read past end
        assert!(dec.read_packed_varint_i64(2).is_err()); // end > data.len()
    }

    #[test]
    fn packed_i32_empty() {
        let mut dec = ProtoDecoder::new(&[]);
        let result = dec.read_packed_i32(0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn packed_i32_single() {
        let data = 99u32.to_le_bytes();
        let mut dec = ProtoDecoder::new(&data);
        let result = dec.read_packed_i32(4).unwrap();
        assert_eq!(result, alloc::vec![99i32]);
    }

    #[test]
    fn packed_i32_truncated() {
        let data = [0u8; 2];
        let mut dec = ProtoDecoder::new(&data);
        assert!(dec.read_packed_i32(6).is_err());
    }

    #[test]
    fn packed_f64_empty() {
        let mut dec = ProtoDecoder::new(&[]);
        let result = dec.read_packed_f64(0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn packed_f64_single() {
        let data = 1.25f64.to_le_bytes();
        let mut dec = ProtoDecoder::new(&data);
        let result = dec.read_packed_f64(8).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.25).abs() < 1e-10);
    }

    #[test]
    fn packed_f64_truncated() {
        let data = [0u8; 5];
        let mut dec = ProtoDecoder::new(&data);
        assert!(dec.read_packed_f64(10).is_err());
    }

    // ---------------------------------------------------------------
    // Message decoder tests
    // ---------------------------------------------------------------

    #[test]
    fn decode_opset_import_basic() {
        let mut data = Vec::new();
        // field 1 (domain): string ""
        data.extend(encode_length_delimited(1, b""));
        // field 2 (version): varint 17
        data.extend(encode_varint_field(2, 17));
        let opset = decode_opset_import(&data).unwrap();
        assert_eq!(opset.domain, "");
        assert_eq!(opset.version, 17);
    }

    #[test]
    fn decode_opset_import_with_domain() {
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"ai.onnx.ml"));
        data.extend(encode_varint_field(2, 3));
        let opset = decode_opset_import(&data).unwrap();
        assert_eq!(opset.domain, "ai.onnx.ml");
        assert_eq!(opset.version, 3);
    }

    #[test]
    fn decode_attribute_float() {
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"alpha"));
        // field 2 (f): fixed32 float
        data.extend(encode_fixed32_field(2, 0.5f32.to_bits()));
        data.extend(encode_varint_field(20, 1)); // type = Float
        let attr = decode_attribute(&data).unwrap();
        assert_eq!(attr.name, "alpha");
        assert!((attr.f - 0.5).abs() < f32::EPSILON);
        assert_eq!(attr.attr_type, AttributeType::Float);
    }

    #[test]
    fn decode_attribute_int() {
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"axis"));
        data.extend(encode_varint_field(3, 2));
        data.extend(encode_varint_field(20, 2)); // type = Int
        let attr = decode_attribute(&data).unwrap();
        assert_eq!(attr.name, "axis");
        assert_eq!(attr.i, 2);
        assert_eq!(attr.attr_type, AttributeType::Int);
    }

    #[test]
    fn decode_attribute_bytes() {
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"mode"));
        data.extend(encode_length_delimited(4, b"linear"));
        data.extend(encode_varint_field(20, 3)); // type = String
        let attr = decode_attribute(&data).unwrap();
        assert_eq!(attr.name, "mode");
        assert_eq!(attr.s, b"linear");
        assert_eq!(attr.attr_type, AttributeType::String);
    }

    #[test]
    fn decode_attribute_tensor_field_without_type() {
        // Field 5 (t) present but field 20 (type) absent. The decoder
        // should still set attr_type = Tensor so downstream code can
        // recognize the attribute category.
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"value"));
        // Empty length-delimited TensorProto bytes for field 5.
        data.extend(encode_length_delimited(5, &[]));
        let attr = decode_attribute(&data).unwrap();
        assert_eq!(attr.name, "value");
        assert_eq!(attr.attr_type, AttributeType::Tensor);
    }

    #[test]
    fn decode_attribute_packed_floats() {
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"scales"));
        // field 7: packed f32 values
        let mut float_bytes = Vec::new();
        float_bytes.extend(1.0f32.to_le_bytes());
        float_bytes.extend(2.0f32.to_le_bytes());
        data.extend(encode_length_delimited(7, &float_bytes));
        data.extend(encode_varint_field(20, 6)); // type = Floats
        let attr = decode_attribute(&data).unwrap();
        assert_eq!(attr.floats.len(), 2);
        assert!((attr.floats[0] - 1.0).abs() < f32::EPSILON);
        assert!((attr.floats[1] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn decode_attribute_packed_ints() {
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"kernel_shape"));
        // field 8: packed varint ints
        let mut int_bytes = Vec::new();
        int_bytes.extend(encode_varint(3));
        int_bytes.extend(encode_varint(3));
        data.extend(encode_length_delimited(8, &int_bytes));
        data.extend(encode_varint_field(20, 7)); // type = Ints
        let attr = decode_attribute(&data).unwrap();
        assert_eq!(attr.ints, alloc::vec![3i64, 3]);
        assert_eq!(attr.attr_type, AttributeType::Ints);
    }

    #[test]
    fn decode_attribute_tensor_value() {
        // Build an inner TensorProto: dims=[3], data_type=FLOAT,
        // float_data=[1.0, 2.0, 3.0].
        let mut tensor_bytes = Vec::new();
        let mut dims_bytes = Vec::new();
        dims_bytes.extend(encode_varint(3));
        tensor_bytes.extend(encode_length_delimited(1, &dims_bytes));
        tensor_bytes.extend(encode_varint_field(2, 1));
        let mut float_bytes = Vec::new();
        for v in [1.0f32, 2.0, 3.0] {
            float_bytes.extend(v.to_le_bytes());
        }
        tensor_bytes.extend(encode_length_delimited(4, &float_bytes));

        // Wrap in an AttributeProto: name="value", t=<tensor>, type=Tensor(4).
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"value"));
        data.extend(encode_length_delimited(5, &tensor_bytes));
        data.extend(encode_varint_field(20, 4));

        let attr = decode_attribute(&data).unwrap();
        assert_eq!(attr.name, "value");
        assert_eq!(attr.attr_type, AttributeType::Tensor);
        let t = attr.t.as_ref().expect("tensor should be populated");
        assert_eq!(t.dims, alloc::vec![3i64]);
        assert_eq!(t.data_type, 1);
        assert_eq!(t.float_data.len(), 3);
        assert!((t.float_data[2] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn decode_attribute_tensor_with_mixed_fields() {
        // An AttributeProto that sets t, f, and i simultaneously, but
        // declares attr_type = Tensor. The parser should populate every
        // raw field it decoded, but downstream lookups via
        // `get_attr_tensor` should only match the declared type.
        let mut tensor_bytes = Vec::new();
        let mut dims_bytes = Vec::new();
        dims_bytes.extend(encode_varint(1));
        tensor_bytes.extend(encode_length_delimited(1, &dims_bytes));
        tensor_bytes.extend(encode_varint_field(2, 1));
        let mut float_bytes = Vec::new();
        float_bytes.extend(7.5f32.to_le_bytes());
        tensor_bytes.extend(encode_length_delimited(4, &float_bytes));

        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"value"));
        // f = 0.5 (irrelevant — attr_type is Tensor)
        data.extend(encode_fixed32_field(2, 0.5f32.to_bits()));
        // i = 42 (irrelevant — attr_type is Tensor)
        data.extend(encode_varint_field(3, 42));
        // t = <tensor>
        data.extend(encode_length_delimited(5, &tensor_bytes));
        data.extend(encode_varint_field(20, 4));

        let attr = decode_attribute(&data).unwrap();
        assert_eq!(attr.attr_type, AttributeType::Tensor);
        assert!((attr.f - 0.5).abs() < f32::EPSILON);
        assert_eq!(attr.i, 42);
        let t = attr.t.as_ref().expect("tensor should be populated");
        assert_eq!(t.dims, alloc::vec![1i64]);
        assert_eq!(t.float_data.len(), 1);
        assert!((t.float_data[0] - 7.5).abs() < f32::EPSILON);
    }

    #[test]
    fn decode_tensor_with_float_data() {
        let mut data = Vec::new();
        // dims: [2, 3]
        let mut dims_bytes = Vec::new();
        dims_bytes.extend(encode_varint(2));
        dims_bytes.extend(encode_varint(3));
        data.extend(encode_length_delimited(1, &dims_bytes));
        // data_type = 1 (FLOAT)
        data.extend(encode_varint_field(2, 1));
        // name
        data.extend(encode_length_delimited(8, b"weight"));
        // float_data: packed f32
        let mut float_bytes = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            float_bytes.extend(v.to_le_bytes());
        }
        data.extend(encode_length_delimited(4, &float_bytes));
        let tensor = decode_tensor(&data).unwrap();
        assert_eq!(tensor.dims, alloc::vec![2i64, 3]);
        assert_eq!(tensor.data_type, 1);
        assert_eq!(tensor.name, "weight");
        assert_eq!(tensor.float_data.len(), 6);
        assert!((tensor.float_data[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn decode_tensor_with_raw_data() {
        let mut data = Vec::new();
        let mut dims_bytes = Vec::new();
        dims_bytes.extend(encode_varint(4));
        data.extend(encode_length_delimited(1, &dims_bytes));
        data.extend(encode_varint_field(2, 2)); // UINT8
        data.extend(encode_length_delimited(9, &[10, 20, 30, 40]));
        let tensor = decode_tensor(&data).unwrap();
        assert_eq!(tensor.dims, alloc::vec![4i64]);
        assert_eq!(tensor.raw_data, alloc::vec![10u8, 20, 30, 40]);
    }

    #[test]
    fn decode_node_relu() {
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"x")); // input
        data.extend(encode_length_delimited(2, b"y")); // output
        data.extend(encode_length_delimited(3, b"relu0")); // name
        data.extend(encode_length_delimited(4, b"Relu")); // op_type
        let node = decode_node(&data).unwrap();
        assert_eq!(node.input, alloc::vec!["x"]);
        assert_eq!(node.output, alloc::vec!["y"]);
        assert_eq!(node.name, "relu0");
        assert_eq!(node.op_type, "Relu");
        assert!(node.domain.is_empty());
    }

    #[test]
    fn decode_node_with_attribute() {
        let mut attr_bytes = Vec::new();
        attr_bytes.extend(encode_length_delimited(1, b"axis"));
        attr_bytes.extend(encode_varint_field(3, 1));
        attr_bytes.extend(encode_varint_field(20, 2)); // Int

        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"x"));
        data.extend(encode_length_delimited(2, b"y"));
        data.extend(encode_length_delimited(4, b"Softmax"));
        data.extend(encode_length_delimited(5, &attr_bytes));
        let node = decode_node(&data).unwrap();
        assert_eq!(node.op_type, "Softmax");
        assert_eq!(node.attribute.len(), 1);
        assert_eq!(node.attribute[0].name, "axis");
        assert_eq!(node.attribute[0].i, 1);
    }

    #[test]
    fn decode_value_info_with_shape() {
        // Build Dimension messages
        fn encode_dim(val: u64) -> Vec<u8> {
            encode_varint_field(1, val)
        }

        // TensorShapeProto with dims [1, 3]
        let mut shape_data = Vec::new();
        shape_data.extend(encode_length_delimited(1, &encode_dim(1)));
        shape_data.extend(encode_length_delimited(1, &encode_dim(3)));

        // TensorTypeProto: elem_type=1 (FLOAT), shape
        let mut tensor_type_data = Vec::new();
        tensor_type_data.extend(encode_varint_field(1, 1));
        tensor_type_data.extend(encode_length_delimited(2, &shape_data));

        // TypeProto: field 1 = TensorTypeProto
        let type_data = encode_length_delimited(1, &tensor_type_data);

        // ValueInfoProto
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"input_0"));
        data.extend(encode_length_delimited(2, &type_data));

        let vi = decode_value_info(&data).unwrap();
        assert_eq!(vi.name, "input_0");
        assert_eq!(vi.elem_type, 1);
        assert_eq!(vi.shape, alloc::vec![1i64, 3]);
    }

    #[test]
    fn decode_graph_with_node() {
        // Build a simple node
        let mut node_data = Vec::new();
        node_data.extend(encode_length_delimited(1, b"x"));
        node_data.extend(encode_length_delimited(2, b"y"));
        node_data.extend(encode_length_delimited(4, b"Relu"));

        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, &node_data)); // node
        data.extend(encode_length_delimited(2, b"test_graph")); // name

        let graph = decode_graph(&data).unwrap();
        assert_eq!(graph.name, "test_graph");
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Relu");
    }

    // ---------------------------------------------------------------
    // Round-trip: build ONNX bytes -> decode -> verify
    // ---------------------------------------------------------------

    /// Construct a minimal valid ONNX model (Relu graph) as protobuf bytes.
    fn build_test_onnx_bytes() -> Vec<u8> {
        // Dimension helper
        fn encode_dim(val: u64) -> Vec<u8> {
            encode_varint_field(1, val)
        }

        // Build ValueInfoProto for "x" with shape [1, 3], elem_type=1
        fn build_value_info(name: &[u8], dims: &[u64]) -> Vec<u8> {
            let mut shape_data = Vec::new();
            for &d in dims {
                shape_data.extend(encode_length_delimited(1, &encode_dim(d)));
            }
            let mut tensor_type = Vec::new();
            tensor_type.extend(encode_varint_field(1, 1)); // elem_type = FLOAT
            tensor_type.extend(encode_length_delimited(2, &shape_data));
            let type_proto = encode_length_delimited(1, &tensor_type);

            let mut vi = Vec::new();
            vi.extend(encode_length_delimited(1, name));
            vi.extend(encode_length_delimited(2, &type_proto));
            vi
        }

        // Node: Relu with input "x", output "y"
        let mut node_data = Vec::new();
        node_data.extend(encode_length_delimited(1, b"x"));
        node_data.extend(encode_length_delimited(2, b"y"));
        node_data.extend(encode_length_delimited(3, b"relu0"));
        node_data.extend(encode_length_delimited(4, b"Relu"));

        let input_vi = build_value_info(b"x", &[1, 3]);
        let output_vi = build_value_info(b"y", &[1, 3]);

        // Graph
        let mut graph_data = Vec::new();
        graph_data.extend(encode_length_delimited(1, &node_data));
        graph_data.extend(encode_length_delimited(2, b"relu_graph"));
        graph_data.extend(encode_length_delimited(11, &input_vi));
        graph_data.extend(encode_length_delimited(12, &output_vi));

        // Opset import: default domain, version 17
        let mut opset_data = Vec::new();
        opset_data.extend(encode_length_delimited(1, b""));
        opset_data.extend(encode_varint_field(2, 17));

        // Model (field numbers per ONNX spec: 1=ir_version, 2=producer_name,
        // 3=producer_version, 7=graph, 8=opset_import)
        let mut model_data = Vec::new();
        model_data.extend(encode_varint_field(1, 9)); // ir_version
        model_data.extend(encode_length_delimited(2, b"test-producer"));
        model_data.extend(encode_length_delimited(3, b"1.0"));
        model_data.extend(encode_length_delimited(7, &graph_data));
        model_data.extend(encode_length_delimited(8, &opset_data));

        model_data
    }

    #[test]
    fn decode_model_full_roundtrip() {
        let bytes = build_test_onnx_bytes();
        let model = decode_model(&bytes).unwrap();

        assert_eq!(model.ir_version, 9);
        assert_eq!(model.producer_name, "test-producer");
        assert_eq!(model.producer_version, "1.0");
        assert_eq!(model.opset_import.len(), 1);
        assert_eq!(model.opset_import[0].domain, "");
        assert_eq!(model.opset_import[0].version, 17);

        let graph = model.graph.as_ref().unwrap();
        assert_eq!(graph.name, "relu_graph");
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Relu");
        assert_eq!(graph.node[0].name, "relu0");
        assert_eq!(graph.node[0].input, alloc::vec!["x"]);
        assert_eq!(graph.node[0].output, alloc::vec!["y"]);

        assert_eq!(graph.input.len(), 1);
        assert_eq!(graph.input[0].name, "x");
        assert_eq!(graph.input[0].elem_type, 1);
        assert_eq!(graph.input[0].shape, alloc::vec![1i64, 3]);

        assert_eq!(graph.output.len(), 1);
        assert_eq!(graph.output[0].name, "y");
        assert_eq!(graph.output[0].elem_type, 1);
        assert_eq!(graph.output[0].shape, alloc::vec![1i64, 3]);
    }

    #[test]
    fn decode_model_empty_returns_default() {
        assert!(decode_model(&[]).is_ok()); // empty data = default model
    }

    #[test]
    fn decode_model_unknown_fields_skipped() {
        let mut data = Vec::new();
        data.extend(encode_varint_field(1, 9)); // ir_version
                                                // Unknown field 99, varint
        data.extend(encode_varint_field(99, 42));
        // Unknown field 100, length-delimited
        data.extend(encode_length_delimited(100, b"unknown"));
        let model = decode_model(&data).unwrap();
        assert_eq!(model.ir_version, 9);
    }

    #[test]
    fn decode_tensor_int64_data() {
        let mut data = Vec::new();
        let mut dims_bytes = Vec::new();
        dims_bytes.extend(encode_varint(3));
        data.extend(encode_length_delimited(1, &dims_bytes));
        data.extend(encode_varint_field(2, 7)); // INT64

        let mut int64_bytes = Vec::new();
        int64_bytes.extend(encode_varint(100));
        int64_bytes.extend(encode_varint(200));
        int64_bytes.extend(encode_varint(300));
        data.extend(encode_length_delimited(7, &int64_bytes));

        let tensor = decode_tensor(&data).unwrap();
        assert_eq!(tensor.dims, alloc::vec![3i64]);
        assert_eq!(tensor.data_type, 7);
        assert_eq!(tensor.int64_data, alloc::vec![100i64, 200, 300]);
    }

    #[test]
    fn decode_tensor_double_data() {
        let mut data = Vec::new();
        data.extend(encode_varint_field(2, 11)); // DOUBLE
        let mut double_bytes = Vec::new();
        double_bytes.extend(2.5f64.to_le_bytes());
        data.extend(encode_length_delimited(10, &double_bytes));
        let tensor = decode_tensor(&data).unwrap();
        assert_eq!(tensor.data_type, 11);
        assert_eq!(tensor.double_data.len(), 1);
        assert!((tensor.double_data[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn decode_node_with_domain() {
        let mut data = Vec::new();
        data.extend(encode_length_delimited(1, b"x"));
        data.extend(encode_length_delimited(2, b"y"));
        data.extend(encode_length_delimited(4, b"CustomOp"));
        data.extend(encode_length_delimited(7, b"ai.custom"));
        let node = decode_node(&data).unwrap();
        assert_eq!(node.domain, "ai.custom");
    }

    #[test]
    fn decode_graph_with_initializer() {
        let mut tensor_data = Vec::new();
        let mut dims_bytes = Vec::new();
        dims_bytes.extend(encode_varint(2));
        tensor_data.extend(encode_length_delimited(1, &dims_bytes));
        tensor_data.extend(encode_varint_field(2, 1)); // FLOAT
        tensor_data.extend(encode_length_delimited(8, b"bias"));
        let mut float_bytes = Vec::new();
        float_bytes.extend(0.1f32.to_le_bytes());
        float_bytes.extend(0.2f32.to_le_bytes());
        tensor_data.extend(encode_length_delimited(4, &float_bytes));

        let mut data = Vec::new();
        data.extend(encode_length_delimited(2, b"g"));
        data.extend(encode_length_delimited(5, &tensor_data));

        let graph = decode_graph(&data).unwrap();
        assert_eq!(graph.name, "g");
        assert_eq!(graph.initializer.len(), 1);
        assert_eq!(graph.initializer[0].name, "bias");
        assert_eq!(graph.initializer[0].float_data.len(), 2);
    }

    // ---- AttributeProto.g (field 6) — graph-attr-parser-v1 ----

    /// Builds the protobuf bytes for a minimal `NodeProto` with no attributes.
    fn encode_simple_node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
        let mut node_bytes = Vec::new();
        for inp in inputs {
            node_bytes.extend(encode_length_delimited(1, inp.as_bytes()));
        }
        for out in outputs {
            node_bytes.extend(encode_length_delimited(2, out.as_bytes()));
        }
        node_bytes.extend(encode_length_delimited(4, op_type.as_bytes()));
        node_bytes
    }

    /// Builds a `GraphProto` containing the given pre-encoded nodes and
    /// `input`/`output` names.
    fn encode_simple_graph(node_bodies: &[Vec<u8>], inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
        let mut graph_bytes = Vec::new();
        for nb in node_bodies {
            graph_bytes.extend(encode_length_delimited(1, nb));
        }
        // Field 11: input ValueInfoProto. Each entry is a length-delimited
        // ValueInfoProto containing only a name (field 1).
        for name in inputs {
            let inner = encode_length_delimited(1, name.as_bytes());
            graph_bytes.extend(encode_length_delimited(11, &inner));
        }
        for name in outputs {
            let inner = encode_length_delimited(1, name.as_bytes());
            graph_bytes.extend(encode_length_delimited(12, &inner));
        }
        graph_bytes
    }

    #[test]
    fn decode_attribute_graph_loop_body_roundtrip() {
        // Build an inner GraphProto: one MatMul node and one Add node,
        // matching the `body` attribute of a `Loop` op.
        let matmul = encode_simple_node("MatMul", &["a", "b"], &["mm"]);
        let add = encode_simple_node("Add", &["mm", "c"], &["y"]);
        let graph_bytes = encode_simple_graph(&[matmul, add], &["a", "b", "c"], &["y"]);

        // Wrap it in an AttributeProto { name: "body", g: <graph> }.
        let mut attr_bytes = Vec::new();
        attr_bytes.extend(encode_length_delimited(1, b"body"));
        attr_bytes.extend(encode_length_delimited(6, &graph_bytes));

        let attr = decode_attribute(&attr_bytes).unwrap();
        assert_eq!(attr.name, "body");
        // attr_type should auto-flip to Graph when field 20 is absent.
        assert_eq!(attr.attr_type, AttributeType::Graph);
        let g = attr.g.as_ref().expect("g should be populated");
        assert_eq!(g.node.len(), 2);
        assert_eq!(g.node[0].op_type, "MatMul");
        assert_eq!(g.node[1].op_type, "Add");
        assert_eq!(g.input.len(), 3);
        assert_eq!(g.output.len(), 1);
        assert_eq!(g.output[0].name, "y");
    }

    #[test]
    fn decode_attribute_graph_scalar_unaffected() {
        // A Float attribute with no graph field should still round-trip
        // with g == None and attr_type == Float.
        let mut attr_bytes = Vec::new();
        attr_bytes.extend(encode_length_delimited(1, b"alpha"));
        attr_bytes.extend(encode_fixed32_field(2, 0.5f32.to_bits()));
        attr_bytes.extend(encode_varint_field(20, 1)); // Float

        let attr = decode_attribute(&attr_bytes).unwrap();
        assert_eq!(attr.attr_type, AttributeType::Float);
        assert!((attr.f - 0.5).abs() < f32::EPSILON);
        assert!(attr.g.is_none());
    }

    #[test]
    fn decode_attribute_graph_nesting_rejects_overflow() {
        // Build a chain of nested GraphProto attributes 17 levels deep.
        // The parser must return NestingTooDeep at the 17th level before
        // recursing further.
        //
        // Innermost (level 17): empty graph bytes.
        let mut current_graph: Vec<u8> = Vec::new();
        // Wrap it in a node with an AttributeProto.g over and over.
        for _ in 0..17 {
            // AttributeProto { name: "body", g: <current_graph> }
            let mut attr_bytes = Vec::new();
            attr_bytes.extend(encode_length_delimited(1, b"body"));
            attr_bytes.extend(encode_length_delimited(6, &current_graph));
            // NodeProto { op_type: "Loop", attribute: [attr] }
            let mut node_bytes = Vec::new();
            node_bytes.extend(encode_length_delimited(4, b"Loop"));
            node_bytes.extend(encode_length_delimited(5, &attr_bytes));
            // GraphProto containing that node.
            let mut graph_bytes = Vec::new();
            graph_bytes.extend(encode_length_delimited(1, &node_bytes));
            current_graph = graph_bytes;
        }

        // The outermost bytes represent the top-level graph. Decoding it
        // must fail at depth 17 (when the 17th nested AttributeProto.g is
        // encountered).
        let err = decode_graph(&current_graph).expect_err("should reject depth 17");
        assert_eq!(err, ProtoError::NestingTooDeep);
    }

    #[test]
    fn decode_attribute_graph_nesting_depth_16_ok() {
        // A 16-deep chain is the maximum permissible and must decode
        // successfully. This verifies the boundary condition.
        let mut current_graph: Vec<u8> = Vec::new();
        for _ in 0..16 {
            let mut attr_bytes = Vec::new();
            attr_bytes.extend(encode_length_delimited(1, b"body"));
            attr_bytes.extend(encode_length_delimited(6, &current_graph));
            let mut node_bytes = Vec::new();
            node_bytes.extend(encode_length_delimited(4, b"Loop"));
            node_bytes.extend(encode_length_delimited(5, &attr_bytes));
            let mut graph_bytes = Vec::new();
            graph_bytes.extend(encode_length_delimited(1, &node_bytes));
            current_graph = graph_bytes;
        }
        let decoded = decode_graph(&current_graph).expect("depth 16 should decode");
        assert_eq!(decoded.node.len(), 1);
    }

    #[test]
    fn decode_node_with_graph_attribute() {
        // A NodeProto containing an AttributeProto whose g field carries
        // a single-node body graph. Exercises decode_node -> decode_attribute
        // -> decode_graph_with_depth recursion.
        let inner_node = encode_simple_node("Relu", &["x"], &["y"]);
        let inner_graph = encode_simple_graph(&[inner_node], &["x"], &["y"]);

        let mut attr_bytes = Vec::new();
        attr_bytes.extend(encode_length_delimited(1, b"body"));
        attr_bytes.extend(encode_length_delimited(6, &inner_graph));

        let mut node_bytes = Vec::new();
        node_bytes.extend(encode_length_delimited(4, b"Loop"));
        node_bytes.extend(encode_length_delimited(5, &attr_bytes));

        let node = decode_node(&node_bytes).unwrap();
        assert_eq!(node.op_type, "Loop");
        assert_eq!(node.attribute.len(), 1);
        let attr = &node.attribute[0];
        assert_eq!(attr.name, "body");
        assert_eq!(attr.attr_type, AttributeType::Graph);
        let g = attr.g.as_ref().unwrap();
        assert_eq!(g.node.len(), 1);
        assert_eq!(g.node[0].op_type, "Relu");
    }

    #[test]
    fn proto_error_nesting_too_deep_display() {
        use alloc::format;
        assert_eq!(
            format!("{}", ProtoError::NestingTooDeep),
            "nested GraphProto attributes exceed maximum depth"
        );
    }
}
