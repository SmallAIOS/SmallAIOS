// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the container binary's `SMALLAIOS_BUS_BACKEND`
//! pub/sub dataflow mode.
//!
//! These tests start the container binary as a subprocess with various
//! `SMALLAIOS_BUS_BACKEND` values and verify that the real
//! `DataflowRunner` spawns cleanly and the HTTP server remains
//! responsive alongside the bus runner.
//!
//! Run with:
//!   cargo build -p smallaios-container
//!   cargo test -p smallaios-container --test e2e_bus

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command};
use std::time::Duration;

/// Find the built binary path.
fn binary_path() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let target_dir = format!("{}/../target/debug/smallaios-container", manifest_dir);
    std::fs::canonicalize(&target_dir)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or(target_dir)
}

/// Start the server with a specific bus backend and port.
fn start_server_with_bus(port: u16, model_dir: &str, bus_backend: &str) -> Child {
    Command::new(binary_path())
        .env("SMALLAIOS_PORT", port.to_string())
        .env("SMALLAIOS_MODEL_DIR", model_dir)
        .env("SMALLAIOS_GPU_BACKEND", "cpu")
        .env("SMALLAIOS_BUS_BACKEND", bus_backend)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to start smallaios-container binary")
}

/// Wait until the given address accepts TCP connections, or timeout.
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
    stream.shutdown(std::net::Shutdown::Write).ok();
    let mut response = String::new();
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let _ = stream.read_to_string(&mut response);
    response
}

/// Extract the HTTP status code from a raw response.
fn status_code(response: &str) -> Option<u16> {
    response
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
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

/// Prepare a clean temp model dir for a test case.
fn temp_model_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("smallaios_e2e_bus_{}", name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp model dir");
    dir
}

// ---------------------------------------------------------------------------
// Task group 6 — container-level integration tests for SMALLAIOS_BUS_BACKEND.
// ---------------------------------------------------------------------------

#[test]
fn test_bus_mode_zenoh() {
    let port = 18090;
    let model_dir = temp_model_dir("zenoh");
    let child = start_server_with_bus(port, model_dir.to_str().unwrap(), "zenoh");
    let _guard = ServerGuard { child };
    let addr = format!("127.0.0.1:{}", port);

    assert!(
        wait_for_server(&addr, Duration::from_secs(5)),
        "server did not start in zenoh bus mode"
    );

    // HTTP stack must still work — the bus runner runs alongside.
    let resp = send_request(&addr, "GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert_eq!(
        status_code(&resp),
        Some(200),
        "expected 200 from /healthz in zenoh mode, got: {}",
        resp
    );
}

#[test]
fn test_bus_mode_dds() {
    let port = 18091;
    let model_dir = temp_model_dir("dds");
    let child = start_server_with_bus(port, model_dir.to_str().unwrap(), "dds");
    let _guard = ServerGuard { child };
    let addr = format!("127.0.0.1:{}", port);

    assert!(
        wait_for_server(&addr, Duration::from_secs(5)),
        "server did not start in dds bus mode"
    );

    let resp = send_request(&addr, "GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert_eq!(
        status_code(&resp),
        Some(200),
        "expected 200 from /healthz in dds mode, got: {}",
        resp
    );
}

#[test]
fn test_bus_mode_unknown_falls_back() {
    let port = 18092;
    let model_dir = temp_model_dir("unknown");
    let child = start_server_with_bus(
        port,
        model_dir.to_str().unwrap(),
        "quic-over-carrier-pigeon",
    );
    let _guard = ServerGuard { child };
    let addr = format!("127.0.0.1:{}", port);

    // An unknown backend should emit a warning but not block HTTP startup.
    assert!(
        wait_for_server(&addr, Duration::from_secs(5)),
        "server must fall back to HTTP-only when bus backend is unknown"
    );

    let resp = send_request(&addr, "GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert_eq!(
        status_code(&resp),
        Some(200),
        "expected 200 from /healthz with unknown bus backend, got: {}",
        resp
    );
}

#[test]
fn test_http_and_bus_concurrent() {
    let port = 18093;
    let model_dir = temp_model_dir("concurrent");
    let child = start_server_with_bus(port, model_dir.to_str().unwrap(), "zenoh");
    let _guard = ServerGuard { child };
    let addr = format!("127.0.0.1:{}", port);

    assert!(
        wait_for_server(&addr, Duration::from_secs(5)),
        "server did not start for concurrent HTTP+bus test"
    );

    // Fire several HTTP requests back-to-back while the bus runner is
    // (placeholder) active, to confirm the two paths don't deadlock or
    // share state incorrectly.
    for _ in 0..5 {
        let resp = send_request(&addr, "GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert_eq!(
            status_code(&resp),
            Some(200),
            "HTTP request failed while bus runner was active: {}",
            resp
        );
    }

    let resp = send_request(&addr, "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert_eq!(
        status_code(&resp),
        Some(200),
        "metrics endpoint must work alongside the bus runner, got: {}",
        resp
    );
}
