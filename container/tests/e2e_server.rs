// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the SmallAIOS container HTTP server.
//!
//! These tests start the container binary as a subprocess and send real HTTP
//! requests over TCP to exercise the full request path.
//!
//! NOTE: These tests require the `smallaios-container` binary to be built
//! first. They also depend on the handler wiring (handlers.rs) which another
//! agent is implementing. Tests are marked `#[ignore]` until the server
//! endpoints are fully wired in main.rs.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command};
use std::time::Duration;

/// Find the built binary path.
fn binary_path() -> String {
    // cargo test sets OUT_DIR but we need the binary in target/debug
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let target_dir = format!("{}/../target/debug/smallaios-container", manifest_dir);
    // Normalize path
    std::fs::canonicalize(&target_dir)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or(target_dir)
}

/// Start the server binary on the given port, with a temporary model dir.
fn start_server(port: u16, model_dir: &str) -> Child {
    Command::new(binary_path())
        .env("SMALLAIOS_PORT", port.to_string())
        .env("SMALLAIOS_MODEL_DIR", model_dir)
        .env("SMALLAIOS_GPU_BACKEND", "cpu")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to start smallaios-container binary")
}

/// Wait for the server to accept connections, with a timeout.
fn wait_for_server(addr: &str, timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect(addr).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// Send an HTTP request string and return the raw response.
fn send_request(addr: &str, request: &str) -> String {
    let mut stream = TcpStream::connect(addr).expect("failed to connect");
    stream.set_write_timeout(Some(Duration::from_secs(2))).ok();
    stream.write_all(request.as_bytes()).unwrap();
    stream.flush().unwrap();
    // Signal end of writing so the server knows the request is complete.
    stream.shutdown(std::net::Shutdown::Write).ok();
    let mut response = String::new();
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let _ = stream.read_to_string(&mut response);
    response
}

/// Extract the HTTP status code from a raw response.
fn status_code(response: &str) -> Option<u16> {
    // e.g. "HTTP/1.1 200 OK\r\n..."
    response
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
}

/// Extract the body from a raw HTTP response (everything after \r\n\r\n).
fn response_body(response: &str) -> &str {
    response
        .find("\r\n\r\n")
        .map(|i| &response[i + 4..])
        .unwrap_or("")
}

/// Guard that kills the child process on drop.
struct ServerGuard {
    child: Child,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// End-to-end tests
//
// These are ignored by default because they require:
// 1. The binary to be built (`cargo build -p smallaios-container`)
// 2. The handler wiring to be complete (handlers.rs + main.rs updates)
//
// Run with: cargo test -p smallaios-container --test e2e_server -- --ignored
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_health_endpoint() {
    let port = 18080;
    let model_dir = std::env::temp_dir().join("smallaios_e2e_health");
    let _ = std::fs::create_dir_all(&model_dir);

    let child = start_server(port, model_dir.to_str().unwrap());
    let _guard = ServerGuard { child };
    let addr = format!("127.0.0.1:{}", port);

    assert!(
        wait_for_server(&addr, Duration::from_secs(5)),
        "server did not start in time"
    );

    let resp = send_request(&addr, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert_eq!(status_code(&resp), Some(200), "expected 200, got: {}", resp);

    let body = response_body(&resp);
    assert!(
        body.contains("healthy"),
        "expected body to contain 'healthy', got: {}",
        body
    );
}

#[test]
#[ignore]
fn test_ready_no_models() {
    let port = 18081;
    // Use an empty temporary directory so no models are loaded.
    let model_dir = std::env::temp_dir().join("smallaios_e2e_ready_empty");
    let _ = std::fs::create_dir_all(&model_dir);

    let child = start_server(port, model_dir.to_str().unwrap());
    let _guard = ServerGuard { child };
    let addr = format!("127.0.0.1:{}", port);

    assert!(
        wait_for_server(&addr, Duration::from_secs(5)),
        "server did not start in time"
    );

    let resp = send_request(&addr, "GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert_eq!(status_code(&resp), Some(503), "expected 503, got: {}", resp);

    let body = response_body(&resp);
    assert!(
        body.contains("not_ready") || body.contains("not ready"),
        "expected body to indicate not ready, got: {}",
        body
    );
}

#[test]
#[ignore]
fn test_unknown_model_404() {
    let port = 18082;
    let model_dir = std::env::temp_dir().join("smallaios_e2e_unknown_model");
    let _ = std::fs::create_dir_all(&model_dir);

    let child = start_server(port, model_dir.to_str().unwrap());
    let _guard = ServerGuard { child };
    let addr = format!("127.0.0.1:{}", port);

    assert!(
        wait_for_server(&addr, Duration::from_secs(5)),
        "server did not start in time"
    );

    let body = r#"{"model":"nonexistent","inputs":[]}"#;
    let req = format!(
        "POST /v1/inference HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let resp = send_request(&addr, &req);
    assert_eq!(status_code(&resp), Some(404), "expected 404, got: {}", resp);
}

#[test]
#[ignore]
fn test_malformed_json_400() {
    let port = 18083;
    let model_dir = std::env::temp_dir().join("smallaios_e2e_malformed_json");
    let _ = std::fs::create_dir_all(&model_dir);

    let child = start_server(port, model_dir.to_str().unwrap());
    let _guard = ServerGuard { child };
    let addr = format!("127.0.0.1:{}", port);

    assert!(
        wait_for_server(&addr, Duration::from_secs(5)),
        "server did not start in time"
    );

    let body = r#"{this is not valid json"#;
    let req = format!(
        "POST /v1/inference HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let resp = send_request(&addr, &req);
    assert_eq!(status_code(&resp), Some(400), "expected 400, got: {}", resp);
}

#[test]
#[ignore]
fn test_metrics_endpoint() {
    let port = 18084;
    let model_dir = std::env::temp_dir().join("smallaios_e2e_metrics");
    let _ = std::fs::create_dir_all(&model_dir);

    let child = start_server(port, model_dir.to_str().unwrap());
    let _guard = ServerGuard { child };
    let addr = format!("127.0.0.1:{}", port);

    assert!(
        wait_for_server(&addr, Duration::from_secs(5)),
        "server did not start in time"
    );

    let resp = send_request(&addr, "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert_eq!(status_code(&resp), Some(200), "expected 200, got: {}", resp);

    let body = response_body(&resp);
    // Prometheus exposition format uses "# HELP" and "# TYPE" lines
    assert!(
        body.contains("# HELP") || body.contains("# TYPE") || body.contains("smallaios_"),
        "expected prometheus-format metrics, got: {}",
        body
    );
}
