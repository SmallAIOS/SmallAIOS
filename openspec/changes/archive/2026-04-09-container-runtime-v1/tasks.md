## 1. HTTP Server Core

- [ ] 1.1 Create `container/src/server.rs` with `HttpServer` struct: holds `TcpListener`, model manager ref, shutdown flag
- [ ] 1.2 Implement minimal HTTP/1.1 request parser: read request line (method, path, version), headers (Content-Length, Content-Type), body
- [ ] 1.3 Implement HTTP response writer: status line, headers (Content-Type, Content-Length), body
- [ ] 1.4 Implement request router: match method+path to handler functions for `/v1/inference`, `/v1/models`, `/health`, `/ready`, `/metrics`
- [ ] 1.5 Implement accept loop in `HttpServer::run()`: accept connection, parse request, route, write response, close
- [ ] 1.6 Unit tests for HTTP parser: valid GET, valid POST with body, malformed request, missing Content-Length

## 2. JSON Serialization

- [ ] 2.1 Create `container/src/json.rs` with minimal JSON parser: parse objects, arrays, strings, numbers, booleans, null
- [ ] 2.2 Implement JSON serializer: serialize objects, arrays, strings, numbers to String
- [ ] 2.3 Define inference request type: `InferenceRequest { model: String, inputs: BTreeMap<String, TensorInput> }` where `TensorInput { shape: Vec<i64>, data: Vec<f64>, dtype: String }`
- [ ] 2.4 Define inference response type: `InferenceResponse { outputs: BTreeMap<String, TensorOutput>, timing_ms: f64 }`
- [ ] 2.5 Implement parse/serialize for request and response types
- [ ] 2.6 Unit tests for JSON round-trip: parse request → serialize response

## 3. Model Manager

- [ ] 3.1 Create `container/src/model_manager.rs` with `ModelManager` struct: holds `BTreeMap<String, Session>`
- [ ] 3.2 Implement `load_directory(path)`: scan dir for `.onnx` files, load each via `Session::load_model()`, register by filename
- [ ] 3.3 Implement `get_session(name)`: return reference to named session
- [ ] 3.4 Implement `list_models()`: return Vec of model name + metadata (input/output shapes)
- [ ] 3.5 Handle load errors: log and skip corrupt files, continue loading remaining models
- [ ] 3.6 Unit test: load a test `.onnx` file (create a minimal valid protobuf in test fixtures), verify session is registered

## 4. Inference Endpoint

- [ ] 4.1 Implement `POST /v1/inference` handler: parse JSON request, look up model, convert input JSON to Tensor objects, call `Session::run()`, convert output Tensors to JSON response
- [ ] 4.2 Implement error responses: 404 for unknown model, 400 for invalid inputs, 500 for execution failure
- [ ] 4.3 Measure inference time and include `timing_ms` in response
- [ ] 4.4 Register inference counters with metrics: `inference_requests_total`, `inference_errors_total`, `inference_duration_seconds`

## 5. Model Registry Endpoints

- [ ] 5.1 Implement `GET /v1/models` handler: return JSON array of loaded model names and metadata
- [ ] 5.2 Implement `GET /v1/models/{name}` handler: return detailed model metadata or 404

## 6. Health and Readiness Probes

- [ ] 6.1 Wire existing `health.rs` module into HTTP server: register components (model_manager, server) as health check sources
- [ ] 6.2 Implement `GET /health` handler: query `HealthChecker`, return 200/503 with JSON status
- [ ] 6.3 Implement `GET /ready` handler: return 200 only when boot phase is Ready and models are loaded; 503 otherwise
- [ ] 6.4 Implement `GET /metrics` handler: export registered metrics in Prometheus text format via existing `metrics.rs`

## 7. Boot Sequence Integration

- [ ] 7.1 Wire `main.rs` to execute boot phases: parse config from env vars (ConfigLoaded), init GPU backend (RuntimeReady), load models (ModelsLoaded), bind server (Ready)
- [ ] 7.2 Set readiness to false during boot, true only after ModelsLoaded completes
- [ ] 7.3 Log each boot phase transition with timing
- [ ] 7.4 Read `SMALLAIOS_MODEL_DIR`, `SMALLAIOS_PORT`, `SMALLAIOS_GPU_BACKEND` from environment with defaults

## 8. Signal Handling and Shutdown

- [ ] 8.1 Register SIGTERM and SIGINT handlers using `std::sync::atomic::AtomicBool` shutdown flag
- [ ] 8.2 Check shutdown flag in accept loop; stop accepting when set
- [ ] 8.3 Wait for in-flight request to complete (or timeout after configurable grace period)
- [ ] 8.4 Log shutdown sequence and exit with code 0

## 9. Docker Integration

- [ ] 9.1 Update `Dockerfile`: add `EXPOSE 8080`, add `VOLUME /models`, set `ENTRYPOINT` to container binary
- [ ] 9.2 Update `docker-compose.yml`: add model volume mount, add environment variables section, update health check to use `/health` endpoint
- [ ] 9.3 Add example model file and `README` section for running inference via Docker

## 10. End-to-End Testing

- [ ] 10.1 Integration test: start server in background thread, send health check request, verify 200 response
- [ ] 10.2 Integration test: load a test model, send inference request, verify output values match expected
- [ ] 10.3 Integration test: send request for unknown model, verify 404
- [ ] 10.4 Integration test: send malformed JSON, verify 400
- [ ] 10.5 Verify `just test` passes; run `just clippy` and `just fmt-check`
