## ADDED Requirements

### Requirement: Decode ONNX ModelProto from Protobuf Binary
The runtime SHALL decode a complete ONNX model from protobuf-serialized bytes into the ModelProto type hierarchy.

#### Scenario: Decode a valid ONNX model
- **WHEN** valid protobuf bytes containing a ModelProto are provided
- **THEN** the decoder MUST return a ModelProto with ir_version, opset_import, producer metadata, and graph populated
- **AND** the graph MUST contain decoded nodes, inputs, outputs, and initializers

#### Scenario: Decode nested NodeProto with attributes
- **WHEN** a graph contains nodes with operator attributes (ints, floats, strings)
- **THEN** each NodeProto MUST have its attribute list populated with correct types and values

#### Scenario: Decode TensorProto initializers
- **WHEN** a graph contains initializer tensors with float_data or raw_data
- **THEN** each TensorProto MUST have dims, data_type, name, and the correct data field populated

#### Scenario: Reject malformed protobuf
- **WHEN** truncated, corrupted, or invalid protobuf bytes are provided
- **THEN** the decoder MUST return a ProtoError without panicking
- **AND** MUST NOT allocate unbounded memory

#### Scenario: Skip unknown fields
- **WHEN** the protobuf contains fields not recognized by the decoder
- **THEN** the decoder MUST skip them gracefully and continue parsing known fields

### Requirement: Packed Repeated Field Decoding
The ProtoDecoder SHALL support packed repeated fields for arrays of primitives.

#### Scenario: Decode packed float array
- **WHEN** a TensorProto contains float_data as a packed repeated field
- **THEN** the decoder MUST read the length prefix and extract all f32 values

#### Scenario: Decode packed int64 array
- **WHEN** a TensorProto contains dims or int64_data as packed repeated fields
- **THEN** the decoder MUST read the length prefix and extract all i64 values
