## 1. Test Helper Setup

- [x] 1.1 Create `container/tests/test_full_inference_e2e.rs`
- [x] 1.2 Copy `RELU_MODEL` bytes from `onnx-rt/tests/test_real_model.rs`
- [x] 1.3 Add `ChildGuard` RAII struct (or copy from `e2e_bus.rs`)
- [x] 1.4 Add `pick_free_port()` helper using `TcpListener::bind("127.0.0.1:0")`
- [x] 1.5 Add `start_container(model_dir, port)` helper
- [x] 1.6 Add `wait_for_healthz(addr, timeout)` helper
- [x] 1.7 Add `http_get(addr, path)` and `http_post(addr, path, body)` helpers

## 2. JSON Helpers

- [x] 2.1 Add `build_inference_request_json(model, input_name, shape, data)` that constructs the JSON request body
- [x] 2.2 Add `extract_output_data(response_body, output_name)` that parses the JSON response and returns Vec<f32>
- [x] 2.3 Use simple string/regex parsing to avoid serde_json dependency
- [x] 2.4 Unit tests for the JSON helpers

## 3. End-to-End Test

- [x] 3.1 Implement `test_full_inference_pipeline_relu` test
- [x] 3.2 Test flow: write model → start container → wait healthz → list models → POST inference → verify Relu output
- [x] 3.3 Test should NOT be `#[ignore]` — runs in CI by default
- [x] 3.4 Verify all 60 output values match expected Relu output

## 4. Negative Tests

- [x] 4.1 Add `test_inference_unknown_model_returns_404`
- [x] 4.2 Add `test_inference_invalid_input_shape_returns_400`
- [x] 4.3 Add `test_inference_malformed_json_returns_400`

## 5. Documentation

- [x] 5.1 Update `docs/inference-bus.md` with a section on the E2E test pattern
- [x] 5.2 Document the JSON request/response format with the working test as canonical example

## 6. Validation

- [x] 6.1 `just fmt` clean
- [x] 6.2 `just clippy --all-targets` clean
- [x] 6.3 `cargo test -p smallaios-container --test test_full_inference_e2e` passes
- [x] 6.4 Full `just test` workspace clean (container package — all 84 unit + 7 e2e + 4 bus + 5 integration_boot tests pass)

## 7. Handler Wiring (discovered during implementation)

The handler at `container/src/handlers.rs::handle_inference` was a
metadata-only stub — it looked up the model by name and returned a
placeholder `"inference endpoint ready, model execution pending"`
message without ever calling `Session::run`. The E2E test exposed this
gap immediately, so as part of landing the test we:

- [x] 7.1 Add `handle_inference_exec(req, &BTreeMap<String, Session>)` that actually runs the executor
- [x] 7.2 Parse `inputs` JSON objects into typed `Tensor` with `float32` support
- [x] 7.3 Serialise `InferenceOutput` tensors back into the documented response format
- [x] 7.4 Build a shared `Arc<BTreeMap<String, Session>>` in `main.rs` at startup and wire it into the `/v1/inference` route
- [x] 7.5 Fall back to the metadata handler when no sessions loaded so unit-test coverage for the stub path is preserved

Additional dtypes beyond `float32`, multi-input graphs, and richer
error taxonomies remain future work.
