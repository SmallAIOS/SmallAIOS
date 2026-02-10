// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal HTTP/1.1 handler for health and metrics endpoints.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::IpcError;

/// HTTP request methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpMethod {
    /// GET: Retrieve a resource.
    Get,
    /// POST: Submit data.
    Post,
    /// PUT: Replace a resource.
    Put,
    /// DELETE: Remove a resource.
    Delete,
    /// HEAD: Retrieve headers only.
    Head,
    /// OPTIONS: Query supported methods.
    Options,
}

impl HttpMethod {
    /// Parse an HTTP method from its string representation.
    ///
    /// Returns `None` if the method string is not recognized.
    pub fn from_str(s: &str) -> Option<HttpMethod> {
        match s {
            "GET" => Some(HttpMethod::Get),
            "POST" => Some(HttpMethod::Post),
            "PUT" => Some(HttpMethod::Put),
            "DELETE" => Some(HttpMethod::Delete),
            "HEAD" => Some(HttpMethod::Head),
            "OPTIONS" => Some(HttpMethod::Options),
            _ => None,
        }
    }
}

/// HTTP response status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpStatus {
    /// 200 OK
    Ok,
    /// 400 Bad Request
    BadRequest,
    /// 404 Not Found
    NotFound,
    /// 405 Method Not Allowed
    MethodNotAllowed,
    /// 500 Internal Server Error
    InternalError,
}

impl HttpStatus {
    /// Return the numeric HTTP status code.
    pub fn code(&self) -> u16 {
        match self {
            HttpStatus::Ok => 200,
            HttpStatus::BadRequest => 400,
            HttpStatus::NotFound => 404,
            HttpStatus::MethodNotAllowed => 405,
            HttpStatus::InternalError => 500,
        }
    }

    /// Return the standard HTTP reason phrase.
    pub fn reason(&self) -> &'static str {
        match self {
            HttpStatus::Ok => "OK",
            HttpStatus::BadRequest => "Bad Request",
            HttpStatus::NotFound => "Not Found",
            HttpStatus::MethodNotAllowed => "Method Not Allowed",
            HttpStatus::InternalError => "Internal Server Error",
        }
    }
}

/// A parsed HTTP/1.1 request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    /// The HTTP method (GET, POST, etc.).
    pub method: HttpMethod,
    /// The request path (e.g., "/health").
    pub path: String,
    /// Request headers as (name, value) pairs.
    pub headers: Vec<(String, String)>,
    /// Request body bytes.
    pub body: Vec<u8>,
}

impl HttpRequest {
    /// Parse an HTTP/1.1 request from raw bytes.
    ///
    /// Expected format:
    /// ```text
    /// METHOD SP PATH SP HTTP/1.1\r\n
    /// Header-Name: Header-Value\r\n
    /// ...
    /// \r\n
    /// [body]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::InvalidHeader`] if the request is malformed.
    pub fn parse(data: &[u8]) -> Result<HttpRequest, IpcError> {
        let text = core::str::from_utf8(data).map_err(|_| IpcError::InvalidHeader)?;

        // Split at the header/body boundary (\r\n\r\n)
        let (header_section, body_section) = if let Some(pos) = text.find("\r\n\r\n") {
            (&text[..pos], &data[pos + 4..])
        } else {
            // No body separator found -- treat entire input as headers
            (text, &[] as &[u8])
        };

        // Split header section into lines
        let mut lines = header_section.split("\r\n");

        // Parse request line: METHOD SP PATH SP HTTP/1.1
        let request_line = lines.next().ok_or(IpcError::InvalidHeader)?;
        let mut parts = request_line.splitn(3, ' ');

        let method_str = parts.next().ok_or(IpcError::InvalidHeader)?;
        let method = HttpMethod::from_str(method_str).ok_or(IpcError::InvalidHeader)?;

        let path = parts.next().ok_or(IpcError::InvalidHeader)?;
        if path.is_empty() {
            return Err(IpcError::InvalidHeader);
        }

        let version = parts.next().ok_or(IpcError::InvalidHeader)?;
        if version != "HTTP/1.1" {
            return Err(IpcError::InvalidHeader);
        }

        // Parse headers
        let mut headers = Vec::new();
        for line in lines {
            if line.is_empty() {
                break;
            }
            let colon_pos = line.find(':').ok_or(IpcError::InvalidHeader)?;
            let name = line[..colon_pos].trim();
            let value = line[colon_pos + 1..].trim();
            if name.is_empty() {
                return Err(IpcError::InvalidHeader);
            }
            headers.push((String::from(name), String::from(value)));
        }

        Ok(HttpRequest {
            method,
            path: String::from(path),
            headers,
            body: body_section.to_vec(),
        })
    }
}

/// An HTTP/1.1 response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// Response status.
    pub status: HttpStatus,
    /// Response headers as (name, value) pairs.
    pub headers: Vec<(String, String)>,
    /// Response body bytes.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Create a new response with just a status code and no body.
    pub fn new(status: HttpStatus) -> Self {
        HttpResponse {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    /// Create a response with a body, automatically adding
    /// `Content-Type` and `Content-Length` headers.
    pub fn with_body(status: HttpStatus, content_type: &str, body: Vec<u8>) -> Self {
        let mut headers = Vec::new();
        headers.push((String::from("Content-Type"), String::from(content_type)));
        headers.push((
            String::from("Content-Length"),
            alloc::format!("{}", body.len()),
        ));
        HttpResponse {
            status,
            headers,
            body,
        }
    }

    /// Serialize the response to the HTTP/1.1 wire format.
    ///
    /// Format:
    /// ```text
    /// HTTP/1.1 {code} {reason}\r\n
    /// {headers}\r\n
    /// \r\n
    /// {body}
    /// ```
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Status line
        let status_line = alloc::format!(
            "HTTP/1.1 {} {}\r\n",
            self.status.code(),
            self.status.reason()
        );
        buf.extend_from_slice(status_line.as_bytes());

        // Headers
        for (name, value) in &self.headers {
            let header_line = alloc::format!("{}: {}\r\n", name, value);
            buf.extend_from_slice(header_line.as_bytes());
        }

        // Empty line separating headers from body
        buf.extend_from_slice(b"\r\n");

        // Body
        buf.extend_from_slice(&self.body);

        buf
    }
}

/// Handle an HTTP request and produce the appropriate response.
///
/// Supported routes:
/// - `GET /health` -> 200, `{"status":"healthy"}`
/// - `GET /metrics` -> 200, `{"inference_count":0,"uptime_secs":0}`
/// - `GET /ready` -> 200, `{"ready":true}`
/// - `GET /live` -> 200, `{"alive":true}`
/// - `POST|PUT|DELETE *` -> 405 Method Not Allowed
/// - Everything else -> 404 Not Found
pub fn handle_request(request: &HttpRequest) -> HttpResponse {
    match &request.method {
        HttpMethod::Get => match request.path.as_str() {
            "/health" => HttpResponse::with_body(
                HttpStatus::Ok,
                "application/json",
                Vec::from(b"{\"status\":\"healthy\"}" as &[u8]),
            ),
            "/metrics" => HttpResponse::with_body(
                HttpStatus::Ok,
                "application/json",
                Vec::from(b"{\"inference_count\":0,\"uptime_secs\":0}" as &[u8]),
            ),
            "/ready" => HttpResponse::with_body(
                HttpStatus::Ok,
                "application/json",
                Vec::from(b"{\"ready\":true}" as &[u8]),
            ),
            "/live" => HttpResponse::with_body(
                HttpStatus::Ok,
                "application/json",
                Vec::from(b"{\"alive\":true}" as &[u8]),
            ),
            _ => HttpResponse::new(HttpStatus::NotFound),
        },
        HttpMethod::Post | HttpMethod::Put | HttpMethod::Delete => {
            HttpResponse::new(HttpStatus::MethodNotAllowed)
        }
        _ => HttpResponse::new(HttpStatus::NotFound),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    // === HttpMethod ===

    #[test]
    fn method_from_str_all() {
        assert_eq!(HttpMethod::from_str("GET"), Some(HttpMethod::Get));
        assert_eq!(HttpMethod::from_str("POST"), Some(HttpMethod::Post));
        assert_eq!(HttpMethod::from_str("PUT"), Some(HttpMethod::Put));
        assert_eq!(HttpMethod::from_str("DELETE"), Some(HttpMethod::Delete));
        assert_eq!(HttpMethod::from_str("HEAD"), Some(HttpMethod::Head));
        assert_eq!(HttpMethod::from_str("OPTIONS"), Some(HttpMethod::Options));
    }

    #[test]
    fn method_from_str_unknown() {
        assert_eq!(HttpMethod::from_str("PATCH"), None);
        assert_eq!(HttpMethod::from_str("get"), None);
        assert_eq!(HttpMethod::from_str(""), None);
    }

    // === HttpStatus ===

    #[test]
    fn status_codes() {
        assert_eq!(HttpStatus::Ok.code(), 200);
        assert_eq!(HttpStatus::BadRequest.code(), 400);
        assert_eq!(HttpStatus::NotFound.code(), 404);
        assert_eq!(HttpStatus::MethodNotAllowed.code(), 405);
        assert_eq!(HttpStatus::InternalError.code(), 500);
    }

    #[test]
    fn status_reasons() {
        assert_eq!(HttpStatus::Ok.reason(), "OK");
        assert_eq!(HttpStatus::BadRequest.reason(), "Bad Request");
        assert_eq!(HttpStatus::NotFound.reason(), "Not Found");
        assert_eq!(HttpStatus::MethodNotAllowed.reason(), "Method Not Allowed");
        assert_eq!(HttpStatus::InternalError.reason(), "Internal Server Error");
    }

    // === Request parsing ===

    #[test]
    fn parse_valid_get_request() {
        let raw = b"GET /health HTTP/1.1\r\n\r\n";
        let req = HttpRequest::parse(raw).unwrap();
        assert_eq!(req.method, HttpMethod::Get);
        assert_eq!(req.path, "/health");
        assert!(req.headers.is_empty());
        assert!(req.body.is_empty());
    }

    #[test]
    fn parse_request_with_headers() {
        let raw = b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\n\r\n";
        let req = HttpRequest::parse(raw).unwrap();
        assert_eq!(req.method, HttpMethod::Get);
        assert_eq!(req.path, "/metrics");
        assert_eq!(req.headers.len(), 2);
        assert_eq!(req.headers[0].0, "Host");
        assert_eq!(req.headers[0].1, "localhost");
        assert_eq!(req.headers[1].0, "Accept");
        assert_eq!(req.headers[1].1, "application/json");
    }

    #[test]
    fn parse_request_with_body() {
        let raw = b"POST /inference HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello";
        let req = HttpRequest::parse(raw).unwrap();
        assert_eq!(req.method, HttpMethod::Post);
        assert_eq!(req.path, "/inference");
        assert_eq!(req.headers.len(), 1);
        assert_eq!(req.body, b"hello");
    }

    #[test]
    fn parse_invalid_request_no_method() {
        let raw = b"\r\n\r\n";
        assert_eq!(HttpRequest::parse(raw), Err(IpcError::InvalidHeader));
    }

    #[test]
    fn parse_invalid_request_bad_version() {
        let raw = b"GET /test HTTP/2.0\r\n\r\n";
        assert_eq!(HttpRequest::parse(raw), Err(IpcError::InvalidHeader));
    }

    #[test]
    fn parse_invalid_request_unknown_method() {
        let raw = b"PATCH /test HTTP/1.1\r\n\r\n";
        assert_eq!(HttpRequest::parse(raw), Err(IpcError::InvalidHeader));
    }

    // === Request handling ===

    #[test]
    fn handle_health_endpoint() {
        let req = HttpRequest {
            method: HttpMethod::Get,
            path: String::from("/health"),
            headers: Vec::new(),
            body: Vec::new(),
        };
        let resp = handle_request(&req);
        assert_eq!(resp.status, HttpStatus::Ok);
        let body = core::str::from_utf8(&resp.body).unwrap();
        assert_eq!(body, "{\"status\":\"healthy\"}");
    }

    #[test]
    fn handle_metrics_endpoint() {
        let req = HttpRequest {
            method: HttpMethod::Get,
            path: String::from("/metrics"),
            headers: Vec::new(),
            body: Vec::new(),
        };
        let resp = handle_request(&req);
        assert_eq!(resp.status, HttpStatus::Ok);
        let body = core::str::from_utf8(&resp.body).unwrap();
        assert_eq!(body, "{\"inference_count\":0,\"uptime_secs\":0}");
    }

    #[test]
    fn handle_ready_endpoint() {
        let req = HttpRequest {
            method: HttpMethod::Get,
            path: String::from("/ready"),
            headers: Vec::new(),
            body: Vec::new(),
        };
        let resp = handle_request(&req);
        assert_eq!(resp.status, HttpStatus::Ok);
        let body = core::str::from_utf8(&resp.body).unwrap();
        assert_eq!(body, "{\"ready\":true}");
    }

    #[test]
    fn handle_live_endpoint() {
        let req = HttpRequest {
            method: HttpMethod::Get,
            path: String::from("/live"),
            headers: Vec::new(),
            body: Vec::new(),
        };
        let resp = handle_request(&req);
        assert_eq!(resp.status, HttpStatus::Ok);
        let body = core::str::from_utf8(&resp.body).unwrap();
        assert_eq!(body, "{\"alive\":true}");
    }

    #[test]
    fn handle_404_unknown_path() {
        let req = HttpRequest {
            method: HttpMethod::Get,
            path: String::from("/nonexistent"),
            headers: Vec::new(),
            body: Vec::new(),
        };
        let resp = handle_request(&req);
        assert_eq!(resp.status, HttpStatus::NotFound);
    }

    #[test]
    fn handle_405_post_method() {
        let req = HttpRequest {
            method: HttpMethod::Post,
            path: String::from("/health"),
            headers: Vec::new(),
            body: Vec::new(),
        };
        let resp = handle_request(&req);
        assert_eq!(resp.status, HttpStatus::MethodNotAllowed);
    }

    #[test]
    fn handle_405_put_method() {
        let req = HttpRequest {
            method: HttpMethod::Put,
            path: String::from("/metrics"),
            headers: Vec::new(),
            body: Vec::new(),
        };
        let resp = handle_request(&req);
        assert_eq!(resp.status, HttpStatus::MethodNotAllowed);
    }

    #[test]
    fn handle_405_delete_method() {
        let req = HttpRequest {
            method: HttpMethod::Delete,
            path: String::from("/live"),
            headers: Vec::new(),
            body: Vec::new(),
        };
        let resp = handle_request(&req);
        assert_eq!(resp.status, HttpStatus::MethodNotAllowed);
    }

    // === Response serialization ===

    #[test]
    fn serialize_simple_response() {
        let resp = HttpResponse::new(HttpStatus::Ok);
        let bytes = resp.serialize();
        let text = core::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("\r\n\r\n"));
    }

    #[test]
    fn serialize_response_with_body() {
        let resp = HttpResponse::with_body(
            HttpStatus::Ok,
            "application/json",
            Vec::from(b"{\"ok\":true}" as &[u8]),
        );
        let bytes = resp.serialize();
        let text = core::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Content-Type: application/json\r\n"));
        assert!(text.contains("Content-Length: 11\r\n"));
        assert!(text.ends_with("{\"ok\":true}"));
    }

    #[test]
    fn serialize_not_found_response() {
        let resp = HttpResponse::new(HttpStatus::NotFound);
        let bytes = resp.serialize();
        let text = core::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("HTTP/1.1 404 Not Found\r\n"));
    }

    #[test]
    fn response_with_body_has_correct_headers() {
        let body = Vec::from(b"test body data" as &[u8]);
        let resp = HttpResponse::with_body(HttpStatus::Ok, "text/plain", body.clone());
        assert_eq!(resp.headers.len(), 2);
        assert_eq!(resp.headers[0], (String::from("Content-Type"), String::from("text/plain")));
        assert_eq!(
            resp.headers[1],
            (String::from("Content-Length"), alloc::format!("{}", body.len()))
        );
    }

    // === Full roundtrip ===

    #[test]
    fn parse_and_handle_health_check() {
        let raw = b"GET /health HTTP/1.1\r\nHost: localhost:8080\r\n\r\n";
        let req = HttpRequest::parse(raw).unwrap();
        let resp = handle_request(&req);
        assert_eq!(resp.status, HttpStatus::Ok);
        let serialized = resp.serialize();
        let text = core::str::from_utf8(&serialized).unwrap();
        assert!(text.contains("200 OK"));
        assert!(text.contains("{\"status\":\"healthy\"}"));
    }
}
