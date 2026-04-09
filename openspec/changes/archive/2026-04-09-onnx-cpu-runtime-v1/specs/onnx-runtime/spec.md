## MODIFIED Requirements

### Requirement: Session API
The runtime SHALL expose a Session API with load, create_session, run, and metadata operations.

#### Scenario: Create and run an inference session
- **WHEN** a client calls load_model with valid ONNX bytes followed by create_session
- **THEN** the runtime MUST return a ready Session handle
- **AND** calling run with correctly shaped input tensors MUST return output tensors matching the model's output specification

#### Scenario: Query model metadata
- **WHEN** a client calls get_metadata on a loaded model
- **THEN** the runtime MUST return the model's input names/shapes, output names/shapes, opset version, and producer name

#### Scenario: Run returns computed output tensors
- **WHEN** `Session::run()` is called with valid input tensors on a model containing only supported Tier 1 operators
- **THEN** the session MUST execute the full inference graph and return output tensors with correct values
- **AND** MUST NOT return `NotImplemented`

#### Scenario: Run with missing input tensor
- **WHEN** `Session::run()` is called with an input set that does not include all required model inputs
- **THEN** the session MUST return `SessionError::InvalidInput` identifying the missing tensor name

#### Scenario: Run with shape-mismatched input
- **WHEN** `Session::run()` is called with an input tensor whose shape does not match the model's expected input shape
- **THEN** the session MUST return `SessionError::InvalidInput` with shape details
