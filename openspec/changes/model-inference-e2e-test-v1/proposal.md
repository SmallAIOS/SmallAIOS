## Why

We have all the pieces for end-to-end ONNX inference:
- Protobuf parser that loads real .onnx files (PR #60, #61)
- 29 CPU operators (PR #55)
- Graph executor (PR #55)
- HTTP inference server (PR #59)
- Inference handler that calls Session::run (PR #59)

But there's no test that exercises the **whole stack together**: write a real .onnx file to a model directory, start the container, POST a JSON inference request via HTTP, parse the JSON response, verify the output values are correct. The closest we have is `test_real_model.rs` (decoder + executor) and `e2e_server.rs` (HTTP routes with stub model manager). Neither runs the entire pipeline.

This change adds the missing test that proves SmallAIOS works as advertised: `docker run → POST inference → correct result`.

## What Changes

- Add a new integration test `container/tests/test_full_inference_e2e.rs` that:
  - Creates a temp directory with a real Relu .onnx model (using the bytes from `onnx-rt/tests/test_real_model.rs`)
  - Starts the container binary as a subprocess with `SMALLAIOS_MODEL_DIR` pointing at the temp dir
  - Waits for `/healthz` to return 200 (server ready)
  - Verifies `/v1/models` lists the loaded model
  - POSTs a JSON inference request to `/v1/inference` with input tensor data
  - Parses the JSON response and verifies the Relu output values
  - Cleans up the subprocess on test exit
- Add helper functions for JSON encoding/decoding of inference requests in tests
- Document the E2E test pattern in `docs/inference-bus.md` for future test additions

## Capabilities

### Modified Capabilities
- `container-inference-server`: Add E2E validation that the full HTTP → load_model → Session::run → output → JSON pipeline works

## Impact

- **Code:** New `container/tests/test_full_inference_e2e.rs` (~200 lines), no source changes
- **Behavior:** No new functionality — pure validation of existing capability
- **CI:** New e2e test runs as part of `cargo test -p smallaios-container`
- **Documentation:** Updates to inference-bus.md showing the test pattern
