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

#### Scenario: Load model from file bytes
- **WHEN** a caller provides raw bytes read from an `.onnx` file to `Session::load_model()`
- **THEN** the session MUST parse the protobuf, build the execution graph, and prepare for inference
- **AND** MUST return the list of input and output tensor names and shapes for validation
