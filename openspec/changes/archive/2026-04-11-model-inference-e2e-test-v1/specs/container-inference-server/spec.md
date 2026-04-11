## ADDED Requirements

### Requirement: Full Inference Pipeline E2E Test
The container test suite SHALL include an end-to-end test that exercises the complete inference pipeline from .onnx file on disk to JSON inference response.

#### Scenario: Real Relu model produces correct output
- **WHEN** the test writes a real ONNX Relu model file to a temp directory
- **AND** starts the container binary as a subprocess pointing at that directory
- **AND** waits for `/healthz` to return 200
- **AND** POSTs a `/v1/inference` request with input tensor [-30.0, ..., 29.0]
- **THEN** the response MUST contain output tensor data
- **AND** all negative input values MUST map to 0.0
- **AND** all non-negative input values MUST map to themselves

#### Scenario: Model registry lists the loaded model
- **WHEN** the test queries `/v1/models` after server startup
- **THEN** the response MUST list the relu model with its metadata
- **AND** MUST indicate the model is loaded

#### Scenario: Container shutdown is clean
- **WHEN** the test ends and the subprocess guard's Drop runs
- **THEN** the container subprocess MUST be terminated
- **AND** no zombie processes MUST remain
