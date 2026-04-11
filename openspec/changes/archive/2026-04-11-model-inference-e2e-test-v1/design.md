## Context

The full inference pipeline now works in isolation. We have these tested pieces:

| Component | Test file | What it covers |
|-----------|-----------|----------------|
| Protobuf decode | `onnx-rt/src/protobuf.rs` (in-file tests) | Parses individual ONNX messages |
| Real model decode | `onnx-rt/tests/test_real_model.rs` | Decodes a real Relu .onnx + runs Session::run |
| Operator execution | `onnx-rt/src/operators.rs` (in-file tests) | Each of 29 operators independently |
| Graph executor | `onnx-rt/src/executor.rs` (in-file tests) | Operator dispatch with mock graphs |
| HTTP routes | `container/tests/e2e_server.rs` | HTTP endpoints with empty model manager |
| Bus runner | `container/tests/e2e_bus.rs` | Runner spawn with empty manager |

What's missing: the end-to-end path that combines them all in one test. The container's `inference_handler` parses the JSON request, looks up the session in the model manager, calls `Session::run()`, and serializes the response — but no test exercises this path with a real model loaded from the filesystem.

## Goals / Non-Goals

**Goals:**
- One test that goes: write .onnx file → start container → POST JSON → verify response
- Use the same minimal Relu model from `test_real_model.rs` for known-good behavior
- Subprocess-based (not in-process) to validate the actual binary
- Catches regressions in any layer (parser, executor, HTTP, JSON, model manager)
- Runs in CI without needing real hardware

**Non-Goals:**
- Multiple models — one is enough to prove the pipeline
- GPU backend testing — that's a separate concern
- Bus backend testing — already covered by `e2e_bus.rs`
- Performance benchmarking — separate work
- Loading from actual ONNX zoo URLs — keep tests offline

## Decisions

### D1: Subprocess-based Test, Not In-process

The test starts the actual `smallaios-container` binary as a subprocess via `Command::new(env!("CARGO_BIN_EXE_smallaios-container"))`. This catches:
- Binary startup issues
- Env var parsing
- HTTP server binding
- Real model loading from disk
- Real signal handling for shutdown

Trade-off: subprocess tests are slower than in-process. Acceptable for one comprehensive test.

### D2: Reuse Existing Relu Model Bytes

The bytes for a minimal valid Relu ONNX file are already in `onnx-rt/tests/test_real_model.rs::RELU_MODEL`. Copy these into the new test (or extract to a shared module). The model expects a tensor of shape `[3, 4, 5]` (60 elements).

### D3: Hand-rolled JSON for Test Requests

The container's JSON parser is hand-written and the request format is documented. The test can construct JSON requests and parse responses without depending on `serde_json`. Format:

```json
{
  "model": "relu",
  "inputs": {
    "x": {
      "shape": [3, 4, 5],
      "dtype": "float32",
      "data": [-30.0, -29.0, ..., 29.0]
    }
  }
}
```

Response:
```json
{
  "outputs": {
    "y": {
      "shape": [3, 4, 5],
      "dtype": "float32",
      "data": [0.0, 0.0, ..., 29.0]
    }
  },
  "timing_ms": 0.5
}
```

### D4: Test Structure

```rust
#[test]
fn test_full_inference_pipeline() {
    // 1. Setup
    let tmp_dir = tempfile::tempdir().unwrap();
    let model_path = tmp_dir.path().join("relu.onnx");
    std::fs::write(&model_path, RELU_MODEL).unwrap();
    
    // 2. Start container
    let port = pick_free_port();
    let mut child = start_container(&tmp_dir.path(), port);
    let _guard = ChildGuard(&mut child);  // RAII cleanup
    
    let addr = format!("127.0.0.1:{}", port);
    wait_for_healthz(&addr, Duration::from_secs(5)).expect("server didn't start");
    
    // 3. Verify model is listed
    let models_response = http_get(&addr, "/v1/models");
    assert!(models_response.contains("\"relu\""));
    
    // 4. POST inference request
    let request_body = build_inference_json("relu", "x", &[3,4,5], &input_data);
    let response = http_post(&addr, "/v1/inference", &request_body);
    
    // 5. Parse response and verify
    assert!(response.contains("\"y\""));
    let output_data = extract_output_data(&response);
    // Relu: negatives become 0, positives stay
    for (i, val) in output_data.iter().enumerate() {
        let expected = (i as f32 - 30.0).max(0.0);
        assert!((val - expected).abs() < 1e-6);
    }
}
```

The test should be marked **NOT `#[ignore]`** so it runs in CI by default.

## Risks / Trade-offs

**[Risk] Race condition between server startup and request** — Mitigation: poll `/healthz` until 200 with timeout, then proceed.

**[Risk] Port collision** — Mitigation: bind to port 0 first to get an OS-assigned port, OR use a high random port (50000+) and retry on bind failure.

**[Risk] Subprocess leakage on test panic** — Mitigation: RAII guard struct that calls `child.kill()` in Drop. Existing `e2e_bus.rs` and `e2e_server.rs` already have this pattern — copy it.

**[Trade-off] JSON parsing in tests is verbose** — Acceptable since the test is one-off and the verbose parsing doubles as documentation of the wire format.
