// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal HTTP/1.1 server for the SmallAIOS container runtime.
//!
//! Uses `std::net::TcpListener` with a single-threaded blocking accept loop.
//! Designed for inference serving and Kubernetes probe endpoints.
//! No external dependencies — hand-written request parser and response writer.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A parsed HTTP request.
#[derive(Debug)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// An HTTP response to send.
pub struct HttpResponse {
    pub status: u16,
    pub status_text: String,
    pub content_type: String,
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Create a JSON response.
    pub fn json(status: u16, body: &str) -> Self {
        Self {
            status,
            status_text: status_text(status).into(),
            content_type: "application/json".into(),
            body: body.as_bytes().to_vec(),
        }
    }

    /// Create a plain text response.
    pub fn text(status: u16, body: &str) -> Self {
        Self {
            status,
            status_text: status_text(status).into(),
            content_type: "text/plain".into(),
            body: body.as_bytes().to_vec(),
        }
    }

    /// Write the response to a TCP stream.
    pub fn write_to(&self, stream: &mut TcpStream) -> std::io::Result<()> {
        write!(
            stream,
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.status,
            self.status_text,
            self.content_type,
            self.body.len()
        )?;
        stream.write_all(&self.body)?;
        stream.flush()
    }
}

/// Map status codes to reason phrases.
fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

/// Parse an HTTP request from a TCP stream.
///
/// Reads the request line, headers, and body (based on Content-Length).
/// Returns an error string on malformed requests.
pub fn parse_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| format!("set_read_timeout: {}", e))?;

    let mut reader = BufReader::new(stream);

    // --- Request line ---
    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .map_err(|e| format!("read request line: {}", e))?;
    let request_line = request_line.trim_end();
    if request_line.is_empty() {
        return Err("empty request line".into());
    }

    let parts: Vec<&str> = request_line.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err(format!("malformed request line: {:?}", request_line));
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    // --- Headers ---
    let mut headers = Vec::new();
    let mut content_length: usize = 0;

    loop {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|e| format!("read header: {}", e))?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            let name = name.trim().to_lowercase();
            let value = value.trim().to_string();
            if name == "content-length" {
                content_length = value
                    .parse()
                    .map_err(|_| format!("invalid Content-Length: {:?}", value))?;
            }
            headers.push((name, value));
        }
    }

    // --- Body ---
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("read body: {}", e))?;
    }

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

/// A route handler function.
pub type RouteHandler = fn(&HttpRequest) -> HttpResponse;

/// Minimal HTTP/1.1 server.
///
/// Routes are matched by exact method and path prefix.
/// The server runs a blocking accept loop on a single thread.
pub struct HttpServer {
    listener: TcpListener,
    routes: Vec<(String, String, RouteHandler)>,
    shutdown: Arc<AtomicBool>,
}

impl HttpServer {
    /// Bind to the given address (e.g. `"0.0.0.0:8080"`).
    pub fn bind(addr: &str, shutdown: Arc<AtomicBool>) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        // Non-blocking so we can poll the shutdown flag.
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            routes: Vec::new(),
            shutdown,
        })
    }

    /// Register a route handler for a (method, path_prefix) pair.
    pub fn route(&mut self, method: &str, path: &str, handler: RouteHandler) {
        self.routes
            .push((method.to_uppercase(), path.to_string(), handler));
    }

    /// Return the local address the server is bound to.
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    /// Run the accept loop until shutdown is signalled.
    pub fn run(&self) {
        while !self.shutdown.load(Ordering::Relaxed) {
            match self.listener.accept() {
                Ok((mut stream, _addr)) => {
                    // Set blocking for request/response handling.
                    if stream.set_nonblocking(false).is_err() {
                        continue;
                    }
                    self.handle_connection(&mut stream);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No connection ready — sleep briefly then retry.
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => {
                    // Accept error — sleep briefly to avoid tight loop.
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }

    fn handle_connection(&self, stream: &mut TcpStream) {
        let request = match parse_request(stream) {
            Ok(req) => req,
            Err(_) => {
                let _ = HttpResponse::text(400, "Bad Request").write_to(stream);
                return;
            }
        };

        let response = self.dispatch(&request);
        let _ = response.write_to(stream);
    }

    fn dispatch(&self, request: &HttpRequest) -> HttpResponse {
        for (method, path_prefix, handler) in &self.routes {
            if request.method == *method && request.path.starts_with(path_prefix) {
                return handler(request);
            }
        }
        HttpResponse::json(404, r#"{"error":"not found"}"#)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpStream;

    // Helper: spin up a server on a random port and return the address.
    fn start_test_server(
        routes: Vec<(&str, &str, RouteHandler)>,
    ) -> (std::net::SocketAddr, Arc<AtomicBool>) {
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut server = HttpServer::bind("127.0.0.1:0", shutdown.clone()).unwrap();
        for (method, path, handler) in routes {
            server.route(method, path, handler);
        }
        let addr = server.local_addr().unwrap();
        let shutdown_clone = shutdown.clone();
        std::thread::spawn(move || {
            server.run();
            drop(shutdown_clone);
        });
        // Give the server thread a moment to start.
        std::thread::sleep(Duration::from_millis(20));
        (addr, shutdown)
    }

    fn send_raw(addr: std::net::SocketAddr, request: &str) -> String {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        stream.flush().unwrap();
        // Shutdown the write side so the server sees EOF.
        stream.shutdown(std::net::Shutdown::Write).ok();
        let mut response = String::new();
        std::io::Read::read_to_string(&mut stream, &mut response).unwrap();
        response
    }

    // --- parse_request tests using real TCP ---

    #[test]
    fn parse_get_request() {
        fn handler(_req: &HttpRequest) -> HttpResponse {
            HttpResponse::text(200, "ok")
        }
        let (addr, shutdown) = start_test_server(vec![("GET", "/health", handler)]);

        let resp = send_raw(addr, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(resp.contains("200 OK"));
        assert!(resp.contains("ok"));

        shutdown.store(true, Ordering::Relaxed);
    }

    #[test]
    fn parse_post_with_body() {
        fn handler(req: &HttpRequest) -> HttpResponse {
            let body_str = String::from_utf8_lossy(&req.body);
            HttpResponse::json(200, &format!(r#"{{"echo":"{}"}}"#, body_str))
        }
        let (addr, shutdown) = start_test_server(vec![("POST", "/echo", handler)]);

        let body = r#"{"msg":"hello"}"#;
        let req = format!(
            "POST /echo HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let resp = send_raw(addr, &req);
        assert!(resp.contains("200 OK"));
        assert!(resp.contains(r#"{"msg":"hello"}"#));

        shutdown.store(true, Ordering::Relaxed);
    }

    #[test]
    fn malformed_request_returns_400() {
        fn handler(_req: &HttpRequest) -> HttpResponse {
            HttpResponse::text(200, "should not reach")
        }
        let (addr, shutdown) = start_test_server(vec![("GET", "/", handler)]);

        let resp = send_raw(addr, "GARBAGE\r\n\r\n");
        assert!(resp.contains("400"));

        shutdown.store(true, Ordering::Relaxed);
    }

    #[test]
    fn unknown_route_returns_404() {
        fn handler(_req: &HttpRequest) -> HttpResponse {
            HttpResponse::text(200, "ok")
        }
        let (addr, shutdown) = start_test_server(vec![("GET", "/known", handler)]);

        let resp = send_raw(addr, "GET /unknown HTTP/1.1\r\n\r\n");
        assert!(resp.contains("404"));
        assert!(resp.contains("not found"));

        shutdown.store(true, Ordering::Relaxed);
    }

    // --- HttpResponse unit tests ---

    #[test]
    fn json_response_sets_content_type() {
        let resp = HttpResponse::json(200, r#"{"ok":true}"#);
        assert_eq!(resp.content_type, "application/json");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.status_text, "OK");
    }

    #[test]
    fn text_response_sets_content_type() {
        let resp = HttpResponse::text(404, "not found");
        assert_eq!(resp.content_type, "text/plain");
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn write_response_format() {
        // Use a real TCP pair to test write_to.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut stream, &mut buf).unwrap();
            buf
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        let resp = HttpResponse::json(200, r#"{"status":"ok"}"#);
        resp.write_to(&mut server_stream).unwrap();
        drop(server_stream);

        let output = handle.join().unwrap();
        assert!(output.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(output.contains("Content-Type: application/json"));
        assert!(output.contains("Content-Length: 15"));
        assert!(output.contains(r#"{"status":"ok"}"#));
    }

    #[test]
    fn status_text_coverage() {
        assert_eq!(status_text(200), "OK");
        assert_eq!(status_text(201), "Created");
        assert_eq!(status_text(204), "No Content");
        assert_eq!(status_text(400), "Bad Request");
        assert_eq!(status_text(404), "Not Found");
        assert_eq!(status_text(405), "Method Not Allowed");
        assert_eq!(status_text(500), "Internal Server Error");
        assert_eq!(status_text(503), "Service Unavailable");
        assert_eq!(status_text(999), "Unknown");
    }

    #[test]
    fn server_shutdown() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let server = HttpServer::bind("127.0.0.1:0", shutdown.clone()).unwrap();
        let _ = server.local_addr().unwrap();

        // Signal shutdown before running — run() should return immediately.
        shutdown.store(true, Ordering::Relaxed);
        let handle = std::thread::spawn(move || {
            server.run();
        });
        handle.join().unwrap(); // Should complete quickly.
    }
}
