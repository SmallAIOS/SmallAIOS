// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for the full inference pipeline.
//!
//! This test exercises every layer of the stack:
//! 1. Write a real ONNX model to disk
//! 2. Start the container binary as a subprocess
//! 3. Wait for the HTTP server to become healthy
//! 4. Query the model registry to verify the model loaded
//! 5. POST a JSON inference request
//! 6. Verify the Relu output values are correct
//!
//! If this test fails, something is broken in: protobuf parser,
//! graph executor, operator dispatch, model manager, HTTP server,
//! JSON parser, or the inference handler.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// Real ONNX Relu model bytes — copied verbatim from
/// `onnx-rt/tests/test_real_model.rs::RELU_MODEL`.
///
/// Shape: `[3, 4, 5]` (60 elements). Input name `"x"`, output `"y"`.
const RELU_MODEL: &[u8] = &[
    0x08, 0x07, 0x12, 0x0c, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74,
    0x3a, 0x4b, 0x0a, 0x0c, 0x0a, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x04, 0x52, 0x65, 0x6c, 0x75,
    0x12, 0x09, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x72, 0x65, 0x6c, 0x75, 0x5a, 0x17, 0x0a, 0x01, 0x78,
    0x12, 0x12, 0x0a, 0x10, 0x08, 0x01, 0x12, 0x0c, 0x0a, 0x02, 0x08, 0x03, 0x0a, 0x02, 0x08, 0x04,
    0x0a, 0x02, 0x08, 0x05, 0x62, 0x17, 0x0a, 0x01, 0x79, 0x12, 0x12, 0x0a, 0x10, 0x08, 0x01, 0x12,
    0x0c, 0x0a, 0x02, 0x08, 0x03, 0x0a, 0x02, 0x08, 0x04, 0x0a, 0x02, 0x08, 0x05, 0x42, 0x04, 0x0a,
    0x00, 0x10, 0x09,
];

// ---------------------------------------------------------------------------
// Test helpers — subprocess + HTTP
// ---------------------------------------------------------------------------

/// RAII guard that kills the child process on drop so a panicking test
/// never leaks a background container.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Pick an OS-assigned free port by binding to port 0, reading the
/// assigned port, and dropping the listener.
fn pick_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind 127.0.0.1:0");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

/// Spawn the `smallaios-container` binary with the given model directory
/// and port. Uses the CARGO_BIN_EXE env var so the path is always
/// correct under `cargo test`, `cargo llvm-cov`, and friends.
fn start_container(model_dir: &Path, port: u16) -> Child {
    Command::new(env!("CARGO_BIN_EXE_smallaios-container"))
        .env("SMALLAIOS_PORT", port.to_string())
        .env(
            "SMALLAIOS_MODEL_DIR",
            model_dir.to_string_lossy().to_string(),
        )
        .env("SMALLAIOS_GPU_BACKEND", "cpu")
        .env("SMALLAIOS_BUS_BACKEND", "none")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to start smallaios-container")
}

/// Poll `/healthz` until it returns 200 OK or the timeout expires.
fn wait_for_healthz(addr: &str, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Ok(mut stream) = TcpStream::connect(addr) {
            let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
            let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
            let req = "GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
            if stream.write_all(req.as_bytes()).is_ok() {
                let _ = stream.shutdown(std::net::Shutdown::Write);
                let mut buf = String::new();
                if stream.read_to_string(&mut buf).is_ok() && buf.contains("200 OK") {
                    return true;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// Send a raw HTTP request string and return the full response.
fn http_request(addr: &str, request: &str) -> String {
    let mut stream = TcpStream::connect(addr).expect("connect failed");
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(5))).ok();
    stream.write_all(request.as_bytes()).expect("write failed");
    stream.flush().ok();
    stream.shutdown(std::net::Shutdown::Write).ok();
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    response
}

fn http_get(addr: &str, path: &str) -> String {
    let req = format!(
        "GET {} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        path
    );
    http_request(addr, &req)
}

fn http_post(addr: &str, path: &str, body: &str) -> String {
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path,
        body.len(),
        body
    );
    http_request(addr, &req)
}

// ---------------------------------------------------------------------------
// JSON helpers — hand-built to avoid a serde_json dep in the tests.
// ---------------------------------------------------------------------------

/// Build an inference request body matching the wire format documented
/// in `docs/inference-bus.md`:
///
/// ```json
/// {"model":"relu","inputs":{"x":{"shape":[3,4,5],"dtype":"float32","data":[...]}}}
/// ```
fn build_inference_request(model: &str, input_name: &str, shape: &[i64], data: &[f32]) -> String {
    let shape_str = shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let data_str = data
        .iter()
        .map(|v| format!("{}", v))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"model":"{}","inputs":{{"{}":{{"shape":[{}],"dtype":"float32","data":[{}]}}}}}}"#,
        model, input_name, shape_str, data_str
    )
}

/// Extract the body (after the blank line) from a raw HTTP response.
fn response_body(response: &str) -> &str {
    response
        .find("\r\n\r\n")
        .map(|i| &response[i + 4..])
        .unwrap_or(response)
}

/// Pull the first `"data":[...]` float array that appears after the
/// named output key in the JSON response body. Good enough for the
/// Relu single-output case — we don't need a full JSON parser here.
fn extract_output_data(response: &str, output_name: &str) -> Vec<f32> {
    let body = response_body(response);
    let needle = format!("\"{}\"", output_name);
    let start_idx = body.find(&needle).unwrap_or_else(|| {
        panic!(
            "output name {:?} not found in response body: {}",
            output_name, body
        )
    });
    let after_name = &body[start_idx..];
    let data_idx = after_name
        .find("\"data\"")
        .expect("data field not found after output name");
    let after_data = &after_name[data_idx..];
    let bracket_start = after_data
        .find('[')
        .expect("opening bracket not found after data field");
    let bracket_end = after_data[bracket_start..]
        .find(']')
        .expect("closing bracket not found after data field");
    let nums = &after_data[bracket_start + 1..bracket_start + bracket_end];
    if nums.trim().is_empty() {
        return Vec::new();
    }
    nums.split(',')
        .map(|s| {
            s.trim()
                .parse::<f32>()
                .expect("non-numeric element in data array")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests for the JSON helpers
// ---------------------------------------------------------------------------

#[test]
fn build_inference_request_shape_matches_wire_format() {
    let body = build_inference_request("relu", "x", &[1, 2], &[1.0, 2.0, 3.0, 4.0]);
    assert!(body.contains(r#""model":"relu""#));
    assert!(body.contains(r#""inputs":{"x":"#));
    assert!(body.contains(r#""shape":[1,2]"#));
    assert!(body.contains(r#""dtype":"float32""#));
    assert!(body.contains(r#""data":[1,2,3,4]"#));
}

#[test]
fn extract_output_data_parses_simple_response() {
    let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n\
                {\"outputs\":{\"y\":{\"shape\":[3],\"dtype\":\"float32\",\"data\":[0,1.5,2]}}}";
    let out = extract_output_data(resp, "y");
    assert_eq!(out, vec![0.0, 1.5, 2.0]);
}

#[test]
fn extract_output_data_handles_negative_and_exponent() {
    let resp = "HTTP/1.1 200 OK\r\n\r\n{\"y\":{\"data\":[-1,2.5,1e2]}}";
    let out = extract_output_data(resp, "y");
    assert_eq!(out, vec![-1.0, 2.5, 100.0]);
}

// ---------------------------------------------------------------------------
// Full end-to-end inference pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_full_inference_pipeline_relu() {
    // 1. Setup: write model file
    let tmp = tempfile::tempdir().expect("tempdir failed");
    let model_path = tmp.path().join("relu.onnx");
    std::fs::write(&model_path, RELU_MODEL).expect("write model failed");

    // 2. Start container
    let port = pick_free_port();
    let child = start_container(tmp.path(), port);
    let _guard = ChildGuard(child);

    let addr = format!("127.0.0.1:{}", port);

    // 3. Wait for server to become healthy
    assert!(
        wait_for_healthz(&addr, Duration::from_secs(30)),
        "container did not become healthy within 30s"
    );

    // 4. Verify model is listed
    let models_response = http_get(&addr, "/v1/models");
    assert!(
        models_response.contains("200 OK"),
        "models endpoint did not return 200: {}",
        models_response
    );
    assert!(
        models_response.contains("\"relu\""),
        "model 'relu' not in /v1/models response: {}",
        models_response
    );

    // 5. Build inference request. The Relu model expects shape [3, 4, 5]
    //    (60 elements). Fill with -30..29 so every element exercises
    //    either the negative or positive branch of Relu.
    let input_data: Vec<f32> = (0..60).map(|i| (i as f32) - 30.0).collect();
    let request_body = build_inference_request("relu", "x", &[3, 4, 5], &input_data);

    // 6. POST inference
    let response = http_post(&addr, "/v1/inference", &request_body);
    assert!(
        response.contains("200 OK"),
        "inference did not return 200: {}",
        response
    );

    // 7. Verify output: Relu of [-30..29] should be max(0, x)
    let output_data = extract_output_data(&response, "y");
    assert_eq!(
        output_data.len(),
        60,
        "expected 60 output values, got {}: {}",
        output_data.len(),
        response
    );
    for (i, &val) in output_data.iter().enumerate() {
        let input_val = (i as f32) - 30.0;
        let expected = input_val.max(0.0);
        assert!(
            (val - expected).abs() < 1e-5,
            "mismatch at index {}: got {} expected {}",
            i,
            val,
            expected
        );
    }
}

// ---------------------------------------------------------------------------
// Negative tests
// ---------------------------------------------------------------------------

#[test]
fn test_inference_unknown_model_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("relu.onnx"), RELU_MODEL).unwrap();

    let port = pick_free_port();
    let child = start_container(tmp.path(), port);
    let _guard = ChildGuard(child);
    let addr = format!("127.0.0.1:{}", port);
    assert!(
        wait_for_healthz(&addr, Duration::from_secs(30)),
        "container did not become healthy"
    );

    let body = build_inference_request("nonexistent", "x", &[3, 4, 5], &[0.0; 60]);
    let response = http_post(&addr, "/v1/inference", &body);
    assert!(response.contains("404"), "expected 404, got: {}", response);
}

#[test]
fn test_inference_malformed_json_returns_400() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("relu.onnx"), RELU_MODEL).unwrap();

    let port = pick_free_port();
    let child = start_container(tmp.path(), port);
    let _guard = ChildGuard(child);
    let addr = format!("127.0.0.1:{}", port);
    assert!(
        wait_for_healthz(&addr, Duration::from_secs(30)),
        "container did not become healthy"
    );

    let response = http_post(&addr, "/v1/inference", "not json {{{");
    assert!(response.contains("400"), "expected 400, got: {}", response);
}

#[test]
fn test_inference_invalid_input_shape_returns_400() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("relu.onnx"), RELU_MODEL).unwrap();

    let port = pick_free_port();
    let child = start_container(tmp.path(), port);
    let _guard = ChildGuard(child);
    let addr = format!("127.0.0.1:{}", port);
    assert!(
        wait_for_healthz(&addr, Duration::from_secs(30)),
        "container did not become healthy"
    );

    // Shape [2, 2] = 4 elements but we send 10 — should be rejected as 400.
    let body = build_inference_request("relu", "x", &[2, 2], &[0.0; 10]);
    let response = http_post(&addr, "/v1/inference", &body);
    assert!(
        response.contains("400"),
        "expected 400 for shape/data mismatch, got: {}",
        response
    );
}
