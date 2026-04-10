## ADDED Requirements

### Requirement: Inference Endpoint Wired to ONNX Runtime
The IPC inference endpoint SHALL execute real inference using the ONNX runtime instead of returning a stub response.

#### Scenario: Process RunInference request
- **WHEN** the inference endpoint receives a binary `RunInference` request
- **AND** the requested model is loaded
- **THEN** the endpoint MUST decode input tensors from the request
- **AND** call `Session::run()` with the decoded inputs
- **AND** encode the output tensors in the response

#### Scenario: Handle unknown model
- **WHEN** a `RunInference` request specifies a model that is not loaded
- **THEN** the endpoint MUST return an error response with `IpcError::NotFound`

#### Scenario: Optional ONNX feature flag
- **WHEN** the IPC crate is built without the `onnx` feature
- **THEN** the inference endpoint MUST return a not-implemented response
- **AND** the rest of the IPC functionality MUST remain available
