## 1. Packed Repeated Field Support

- [ ] 1.1 Add `read_packed_f32(len)` to `ProtoDecoder`: read `len/4` f32 values from the byte stream
- [ ] 1.2 Add `read_packed_i64(len)` to `ProtoDecoder`: read `len/8` i64 values (little-endian fixed64)
- [ ] 1.3 Add `read_packed_i32(len)` to `ProtoDecoder`: read `len/4` i32 values (little-endian fixed32)
- [ ] 1.4 Add `read_packed_varint_i64(len)` to `ProtoDecoder`: read varint-encoded i64 values until `len` bytes consumed
- [ ] 1.5 Add `read_packed_double(len)` to `ProtoDecoder`: read `len/8` f64 values
- [ ] 1.6 Unit tests for all packed readers: empty, single element, multiple elements, truncated input

## 2. Message Decoders

- [ ] 2.1 Implement `decode_opset_import(data) -> Result<OperatorSetIdProto, ProtoError>`: field 1 = domain (string), field 2 = version (varint)
- [ ] 2.2 Implement `decode_attribute(data) -> Result<AttributeProto, ProtoError>`: fields 1-10 mapping to name, type, f, i, s, floats, ints, strings
- [ ] 2.3 Implement `decode_tensor(data) -> Result<TensorProto, ProtoError>`: fields 1-8 mapping to dims, data_type, name, raw_data, float_data, int32_data, int64_data, double_data
- [ ] 2.4 Implement `decode_value_info(data) -> Result<ValueInfoProto, ProtoError>`: field 1 = name, field 2 = type_proto (nested: extract elem_type and shape dims)
- [ ] 2.5 Implement `decode_node(data) -> Result<NodeProto, ProtoError>`: fields 1-6 mapping to input, output, name, op_type, attribute, domain
- [ ] 2.6 Implement `decode_graph(data) -> Result<GraphProto, ProtoError>`: fields 1,2,5,6,10 mapping to node, name, input, output, initializer
- [ ] 2.7 Implement `decode_model(data) -> Result<ModelProto, ProtoError>`: fields 1,3-8,14 mapping to ir_version, producer_name/version, domain, model_version, doc_string, graph, opset_import
- [ ] 2.8 Unit tests for each decoder: construct valid protobuf bytes manually, decode, verify all fields

## 3. Wire into Session

- [ ] 3.1 Replace `Err(SessionError::NotImplemented)` in `load_model()` with `decode_model(data).map_err(...)` 
- [ ] 3.2 Ensure `validate_model()` still runs after successful decode
- [ ] 3.3 Update `test_load_model_valid_header_returns_not_implemented` test to expect success with valid protobuf

## 4. End-to-End Testing

- [ ] 4.1 Create helper function `build_test_onnx_bytes()` that constructs a minimal valid ONNX protobuf binary (ModelProto with 1-node Relu graph, 1 input, 1 output)
- [ ] 4.2 Integration test: load_model(bytes) → Session::initialize → Session::run → verify Relu output
- [ ] 4.3 Integration test: load a model with initializers (weights as TensorProto), run MatMul inference
- [ ] 4.4 Fuzz test: random mutations of valid protobuf — verify no panics
- [ ] 4.5 Verify `just test` passes; run `just clippy` and `just fmt-check`
