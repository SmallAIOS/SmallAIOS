// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! gRPC transport abstraction.
//!
//! The actual TCP + TLS 1.3 + HTTP/2 multiplexer lives in
//! `container/` (Layer 3) because the `audit-export` crate is
//! `#![no_std]` and has no I/O. This trait is the contract
//! between the two layers: pass request headers + a
//! length-prefixed protobuf body, get back response headers +
//! length-prefixed protobuf body + the gRPC status code from
//! the trailing HEADERS frame.

use alloc::string::String;
use alloc::vec::Vec;

/// One HTTP/2 header value as it crosses the gRPC boundary.
/// Names are case-sensitive ASCII; values are unconstrained
/// bytes. Pseudo-headers (`:method`, `:scheme`, `:path`,
/// `:authority`) are added by the transport implementation,
/// not by the crate's callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub name: String,
    pub value: String,
}

impl Header {
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// One full unary RPC reply: response headers + body + trailers.
#[derive(Debug, Clone, Default)]
pub struct UnaryResponse {
    pub initial_headers: Vec<Header>,
    pub body: Vec<u8>,
    pub trailers: Vec<Header>,
}

impl UnaryResponse {
    /// Return the gRPC status code from the `grpc-status` trailer,
    /// or `0` (OK) if none is present (per the gRPC HTTP/2 spec,
    /// absence is treated as OK by lenient implementations — we
    /// only invoke this when the transport already considered the
    /// response well-formed).
    pub fn grpc_status(&self) -> u32 {
        for h in &self.trailers {
            if h.name.eq_ignore_ascii_case("grpc-status") {
                return h.value.parse().unwrap_or(2 /* UNKNOWN */);
            }
        }
        0
    }

    /// Optional `grpc-message` trailer (a UTF-8 status description).
    pub fn grpc_message(&self) -> Option<&str> {
        for h in &self.trailers {
            if h.name.eq_ignore_ascii_case("grpc-message") {
                return Some(&h.value);
            }
        }
        None
    }
}

/// Errors a transport may surface back to the client. The audit
/// pipeline maps these to retry / halt classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportError {
    /// TCP / TLS connect failed.
    ConnectFailed,
    /// TLS handshake rejected (e.g. peer advertised < TLS 1.3).
    TlsHandshake,
    /// Network read or write returned EOF mid-frame.
    Io,
    /// HTTP/2 wire-level error (frame parse, flow control, etc.).
    Http2,
    /// Inactivity or response timer exceeded.
    Timeout,
}

/// Contract the integration layer (`container/`) implements.
///
/// Implementations MUST:
///
/// - Enforce TLS 1.3 minimum on the wire and offer the
///   PQC-hybrid `X25519+ML-KEM-768` ciphersuite when configured
///   to do so (task 2.7).
/// - Refuse HTTP/2 server push and dynamic-table HPACK on the
///   inbound side (already enforced by `net::http2`).
/// - Send the HTTP/2 connection preface + an initial empty
///   `SETTINGS` frame.
/// - Open exactly one client-initiated stream per call and
///   close it via END_STREAM on send + receipt of END_STREAM
///   on receive.
///
/// The trait is intentionally minimal so that a host-side mock
/// can implement it for unit tests.
pub trait GrpcTransport {
    /// Perform one unary gRPC call against the given full method
    /// path (e.g. `/immudb.schema.ImmuService/VerifiableSet`).
    /// `body` is the already-gRPC-framed protobuf request
    /// (5-byte prefix + protobuf bytes). The callee is
    /// responsible for transmitting it as a single HTTP/2 DATA
    /// frame and collecting the response analogously.
    fn unary_call(
        &mut self,
        full_method: &str,
        headers: &[Header],
        body: &[u8],
    ) -> Result<UnaryResponse, TransportError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grpc_status_present() {
        let r = UnaryResponse {
            trailers: alloc::vec![Header::new("grpc-status", "14")],
            ..UnaryResponse::default()
        };
        assert_eq!(r.grpc_status(), 14);
    }

    #[test]
    fn grpc_status_absent_treated_as_ok() {
        let r = UnaryResponse::default();
        assert_eq!(r.grpc_status(), 0);
    }

    #[test]
    fn grpc_status_case_insensitive() {
        let r = UnaryResponse {
            trailers: alloc::vec![Header::new("Grpc-Status", "7")],
            ..UnaryResponse::default()
        };
        assert_eq!(r.grpc_status(), 7);
    }

    #[test]
    fn grpc_message_optional() {
        let r = UnaryResponse {
            trailers: alloc::vec![
                Header::new("grpc-status", "16"),
                Header::new("grpc-message", "bearer token rejected"),
            ],
            ..UnaryResponse::default()
        };
        assert_eq!(r.grpc_message(), Some("bearer token rejected"));
    }
}
