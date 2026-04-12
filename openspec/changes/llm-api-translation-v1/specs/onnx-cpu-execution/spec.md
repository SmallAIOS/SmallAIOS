## MODIFIED Requirements

### Requirement: Container HTTP API Surface
The container's HTTP server SHALL expose two additional endpoints (`/v1/chat/completions` and `/v1/messages`) alongside the existing `/v1/inference`, `/v1/models`, `/healthz`, and `/readyz` endpoints. The existing endpoints MUST remain unchanged in behavior.

#### Scenario: All endpoints coexist
- **WHEN** the container starts with at least one loaded model
- **THEN** `/v1/chat/completions` MUST accept OpenAI-format requests
- **AND** `/v1/messages` MUST accept Anthropic-format requests
- **AND** `/v1/inference` MUST continue to accept raw tensor I/O requests
- **AND** all three endpoints MUST use the same underlying ONNX inference engine
