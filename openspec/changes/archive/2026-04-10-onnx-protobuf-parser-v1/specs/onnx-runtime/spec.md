## MODIFIED Requirements

### Requirement: Model Loading via Protobuf Parsing
The ONNX runtime SHALL parse ONNX model files using a minimal protobuf parser code-generated from the onnx.proto3 schema.

#### Scenario: Load a valid ONNX model
- **WHEN** the runtime receives a valid .onnx protobuf-serialized model file
- **THEN** it MUST parse the ModelProto structure including graph, nodes, initializers, and metadata
- **AND** MUST support ONNX IR version 10 (ONNX 1.16+) and opset version 21

#### Scenario: Reject a malformed protobuf
- **WHEN** the runtime receives a corrupted or truncated protobuf file
- **THEN** it MUST return a descriptive OnnxError without panicking
- **AND** MUST NOT allocate unbounded memory during parsing

#### Scenario: Load model and run inference end-to-end
- **WHEN** load_model() is called with valid ONNX protobuf bytes
- **THEN** it MUST return Ok(ModelProto) with a fully populated graph
- **AND** initializing a Session with this model and calling run() MUST produce correct inference output
