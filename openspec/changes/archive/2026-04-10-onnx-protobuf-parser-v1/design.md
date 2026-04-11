## Context

The `ProtoDecoder<'a>` in `protobuf.rs` provides wire-format primitives: `read_varint()`, `read_fixed32/64()`, `read_string()`, `read_length_delimited()`, `skip_field()`, and tag parsing. All 38 tests pass including fuzz tests.

The ONNX type structs in `onnx_types.rs` (`ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, `AttributeProto`, `ValueInfoProto`, `OperatorSetIdProto`) are defined with all fields matching the ONNX protobuf schema. They have `Default` impls.

The gap: no code bridges `ProtoDecoder` → `onnx_types`. `load_model()` validates the magic byte then returns `NotImplemented`.

## Goals / Non-Goals

**Goals:**
- Parse any valid ONNX protobuf binary into `ModelProto` with all nested messages
- Handle packed repeated fields (dims, float_data, int64_data, etc.)
- Skip unknown fields gracefully (forward compatibility)
- Reject malformed input with descriptive errors (no panics)
- Wire into `load_model()` so the full pipeline works end-to-end

**Non-Goals:**
- Protobuf encoding (write direction) — only decoding needed
- ONNX opset version negotiation — validation already exists in session.rs
- Supporting ONNX external data (tensors stored in separate files)
- Supporting ONNX-ML operators (only standard ONNX domain)

## Decisions

### D1: Message Decoders as Free Functions in protobuf.rs

Each ONNX message type gets a `decode_*` function in `protobuf.rs`:
```rust
pub fn decode_model(data: &[u8]) -> Result<ModelProto, ProtoError>
pub fn decode_graph(data: &[u8]) -> Result<GraphProto, ProtoError>
pub fn decode_node(data: &[u8]) -> Result<NodeProto, ProtoError>
// etc.
```

Each function creates a `ProtoDecoder`, loops over fields with `read_tag()`, matches field numbers to struct fields, and calls nested decoders for sub-messages. Unknown fields are skipped via `skip_field()`.

**Why free functions over methods:** Keeps `ProtoDecoder` as a generic wire-format tool. The ONNX-specific field mappings are separate concerns.

### D2: Packed Repeated Fields via Slice Iteration

Add `read_packed_f32()`, `read_packed_i64()`, `read_packed_i32()` to `ProtoDecoder`:
```rust
pub fn read_packed_f32(&mut self, len: usize) -> Result<Vec<f32>, ProtoError> {
    let count = len / 4;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count { values.push(self.read_f32()?); }
    Ok(values)
}
```

Packed fields are encoded as: tag (varint) + length (varint) + raw element bytes. The length gives the byte count, elements are tightly packed with no per-element tags.

### D3: ValueInfoProto Shape from Nested TypeProto

ONNX `ValueInfoProto` has a nested `TypeProto` → `TensorTypeProto` → `TensorShapeProto` → `Dimension[]`. Rather than modeling this full hierarchy, flatten during decode: extract `elem_type` and `shape` dims directly. This matches the existing `ValueInfoProto` struct which already has `elem_type: i32` and `shape: Vec<i64>`.

### D4: TensorProto Data Priority

TensorProto can store data in `raw_data` (preferred) or typed arrays (`float_data`, `int32_data`, etc.). During decode, populate whichever fields are present. The executor's `tensor_from_proto()` already handles the priority: raw_data first, then typed arrays.

## Risks / Trade-offs

**[Risk] Large model files** — A 500 MB model will allocate significant memory during parsing. Mitigation: The `no_std` allocator handles this; the protobuf format is streaming (field-by-field) so peak memory is model size + parsed struct size.

**[Risk] Malformed protobuf** — Invalid wire types, truncated messages, circular nesting. Mitigation: `ProtoDecoder` already validates wire types and returns errors on truncation. Sub-message decoding uses length-delimited bounds to prevent overread.
