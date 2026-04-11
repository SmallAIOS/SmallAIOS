## Why

The ONNX runtime has a complete execution pipeline — 29 operators, graph executor, parallel compute, GPU dispatch, HTTP server — but `Session::load_model()` returns `NotImplemented`. The protobuf wire-format decoder (`ProtoDecoder` in `protobuf.rs`) is fully implemented with 38 tests, but there are no message-level decoders to parse `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, or `AttributeProto` from raw bytes. This is the last gap preventing real ONNX model inference.

## What Changes

- Add packed repeated field decoder to `ProtoDecoder` (for float_data, int64_data, dims arrays)
- Implement recursive message decoders: `decode_model`, `decode_graph`, `decode_node`, `decode_tensor`, `decode_attribute`, `decode_value_info`, `decode_opset_import`
- Wire `load_model()` in `session.rs` to use `ProtoDecoder` → `decode_model()` instead of returning `NotImplemented`
- Add end-to-end test: construct a valid ONNX protobuf binary, load it, run inference, verify output

## Capabilities

### New Capabilities
- `onnx-protobuf-decoding`: Decode ONNX protobuf binary format into runtime type hierarchy (ModelProto → GraphProto → NodeProto/TensorProto)

### Modified Capabilities
- `onnx-runtime`: Session::load_model() decodes real protobuf bytes instead of returning NotImplemented

## Impact

- **Code:** `onnx-rt/src/protobuf.rs` (packed arrays + message decoders ~400 lines), `onnx-rt/src/session.rs` (wire load_model)
- **APIs:** `load_model()` returns `Ok(ModelProto)` for valid ONNX files instead of `Err(NotImplemented)`
- **Testing:** End-to-end: raw protobuf bytes → load → inference → verify output
- **Dependencies:** None — pure `#![no_std]` with `alloc`
