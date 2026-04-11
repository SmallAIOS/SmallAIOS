## 1. Test Helper Setup

- [ ] 1.1 Create `container/tests/test_full_inference_e2e.rs`
- [ ] 1.2 Copy `RELU_MODEL` bytes from `onnx-rt/tests/test_real_model.rs`
- [ ] 1.3 Add `ChildGuard` RAII struct (or copy from `e2e_bus.rs`)
- [ ] 1.4 Add `pick_free_port()` helper using `TcpListener::bind("127.0.0.1:0")`
- [ ] 1.5 Add `start_container(model_dir, port)` helper
- [ ] 1.6 Add `wait_for_healthz(addr, timeout)` helper
- [ ] 1.7 Add `http_get(addr, path)` and `http_post(addr, path, body)` helpers

## 2. JSON Helpers

- [ ] 2.1 Add `build_inference_request_json(model, input_name, shape, data)` that constructs the JSON request body
- [ ] 2.2 Add `extract_output_data(response_body, output_name)` that parses the JSON response and returns Vec<f32>
- [ ] 2.3 Use simple string/regex parsing to avoid serde_json dependency
- [ ] 2.4 Unit tests for the JSON helpers

## 3. End-to-End Test

- [ ] 3.1 Implement `test_full_inference_pipeline_relu` test
- [ ] 3.2 Test flow: write model → start container → wait healthz → list models → POST inference → verify Relu output
- [ ] 3.3 Test should NOT be `#[ignore]` — runs in CI by default
- [ ] 3.4 Verify all 60 output values match expected Relu output

## 4. Negative Tests

- [ ] 4.1 Add `test_inference_unknown_model_returns_404`
- [ ] 4.2 Add `test_inference_invalid_input_shape_returns_400`
- [ ] 4.3 Add `test_inference_malformed_json_returns_400`

## 5. Documentation

- [ ] 5.1 Update `docs/inference-bus.md` with a section on the E2E test pattern
- [ ] 5.2 Document the JSON request/response format with the working test as canonical example

## 6. Validation

- [ ] 6.1 `just fmt` clean
- [ ] 6.2 `just clippy --all-targets` clean
- [ ] 6.3 `cargo test -p smallaios-container --test test_full_inference_e2e` passes
- [ ] 6.4 Full `just test` workspace clean
