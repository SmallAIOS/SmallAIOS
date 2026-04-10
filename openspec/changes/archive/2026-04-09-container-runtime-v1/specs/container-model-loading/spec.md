## ADDED Requirements

### Requirement: Load Models from Filesystem at Startup
The container SHALL scan a configured directory for ONNX model files and load them into inference sessions at boot time.

#### Scenario: Load all models from model directory
- **WHEN** the container starts and `SMALLAIOS_MODEL_DIR` contains `.onnx` files
- **THEN** the model manager MUST load each `.onnx` file into a separate `Session`
- **AND** MUST register each model by its filename (without extension) as the model name
- **AND** MUST log the name, size, and load time for each model

#### Scenario: Empty model directory
- **WHEN** the container starts and the model directory contains no `.onnx` files
- **THEN** the model manager MUST log a warning
- **AND** the readiness probe MUST return not-ready
- **AND** the server MUST still start and accept health/metrics requests

#### Scenario: Corrupt model file
- **WHEN** the container encounters a `.onnx` file that fails to parse or validate
- **THEN** the model manager MUST log an error with the filename and failure reason
- **AND** MUST continue loading remaining models
- **AND** MUST NOT crash the server

### Requirement: Model Registry API
The container SHALL expose model metadata for operational visibility.

#### Scenario: List loaded models
- **WHEN** a client sends `GET /v1/models`
- **THEN** the server MUST return a JSON array of loaded models with name, input shapes, output shapes, and load status

#### Scenario: Query single model metadata
- **WHEN** a client sends `GET /v1/models/{name}`
- **THEN** the server MUST return the model's input names/shapes, output names/shapes, opset version, and operator count
- **AND** MUST return HTTP 404 if the model name is not found
