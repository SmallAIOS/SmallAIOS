## ADDED Requirements

### Requirement: HTTP Inference Endpoint
The container SHALL expose a `POST /v1/inference` endpoint that accepts input tensors and returns model predictions.

#### Scenario: Successful inference request
- **WHEN** a client sends `POST /v1/inference` with a valid JSON body containing model name, input tensor shapes, data, and dtypes
- **THEN** the server MUST execute inference on the specified model
- **AND** MUST return a JSON response with output tensor shapes, data, and dtypes
- **AND** MUST include `timing_ms` in the response

#### Scenario: Unknown model name
- **WHEN** a client sends an inference request with a model name that is not loaded
- **THEN** the server MUST return HTTP 404 with an error message identifying the unknown model

#### Scenario: Invalid input tensor shape
- **WHEN** a client sends an inference request with input tensor shapes that do not match the model's expected inputs
- **THEN** the server MUST return HTTP 400 with an error message describing the shape mismatch

#### Scenario: Inference execution failure
- **WHEN** an inference request triggers an operator error or timeout
- **THEN** the server MUST return HTTP 500 with the error details
- **AND** MUST NOT crash or leave the server in an unrecoverable state

### Requirement: Kubernetes Health Probes
The container SHALL expose liveness and readiness probe endpoints compatible with Kubernetes.

#### Scenario: Liveness probe
- **WHEN** a client sends `GET /health`
- **THEN** the server MUST return HTTP 200 with `{"status": "healthy"}` when the process is alive
- **AND** MUST return HTTP 503 with `{"status": "unhealthy"}` if any critical component has failed

#### Scenario: Readiness probe
- **WHEN** a client sends `GET /ready`
- **THEN** the server MUST return HTTP 200 with `{"status": "ready"}` only after the boot sequence has reached the Ready phase and at least one model is loaded
- **AND** MUST return HTTP 503 with `{"status": "not_ready"}` during boot or if no models are loaded

### Requirement: Prometheus Metrics Endpoint
The container SHALL expose a `/metrics` endpoint in Prometheus text exposition format.

#### Scenario: Scrape metrics
- **WHEN** a Prometheus scraper sends `GET /metrics`
- **THEN** the server MUST return HTTP 200 with `Content-Type: text/plain`
- **AND** MUST include counters for: `inference_requests_total`, `inference_errors_total`
- **AND** MUST include histograms for: `inference_duration_seconds`
- **AND** MUST include gauges for: `models_loaded`, `gpu_available` (0 or 1)

### Requirement: Graceful Shutdown
The container SHALL handle SIGTERM and SIGINT signals for graceful shutdown.

#### Scenario: SIGTERM during idle
- **WHEN** the container receives SIGTERM with no in-flight requests
- **THEN** the server MUST stop accepting new connections
- **AND** MUST exit with code 0 within 5 seconds

#### Scenario: SIGTERM during inference
- **WHEN** the container receives SIGTERM while an inference request is in progress
- **THEN** the server MUST stop accepting new connections
- **AND** MUST allow the in-flight request to complete
- **AND** MUST exit after the request completes or after a configurable timeout (default 30 seconds)

### Requirement: Configuration via Environment Variables
The container SHALL be configurable via environment variables following Kubernetes conventions.

#### Scenario: Configure model directory
- **WHEN** `SMALLAIOS_MODEL_DIR` is set
- **THEN** the server MUST load models from that directory
- **AND** if unset, MUST default to `/models`

#### Scenario: Configure listen port
- **WHEN** `SMALLAIOS_PORT` is set
- **THEN** the server MUST bind to that port
- **AND** if unset, MUST default to `8080`

#### Scenario: Configure GPU backend
- **WHEN** `SMALLAIOS_GPU_BACKEND` is set to `metal`, `cuda`, `rocm`, or `cpu`
- **THEN** the server MUST attempt to initialize the specified GPU backend
- **AND** if initialization fails, MUST fall back to CPU with a warning log
- **AND** if unset, MUST default to `cpu`
