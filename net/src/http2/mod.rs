// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal HTTP/2 client subset for gRPC unary calls.
//!
//! This module implements **only** what the `audit-export` crate
//! needs to talk to an immudb server over gRPC: the wire format
//! from RFC 7540 (frame layout, flow control), the static-only
//! subset of HPACK from RFC 7541, a client-only stream state
//! machine, and the gRPC length-prefixed-message framing layer
//! (5 bytes: `[compressed: u8, length: u32 be]`).
//!
//! ## Deliberate omissions
//!
//! - **Server push** (`PUSH_PROMISE`). Refused — see
//!   [`Connection::on_push_promise`].
//! - **Stream priority** (`PRIORITY`, `HEADERS` priority bits).
//!   Parsed but ignored.
//! - **Dynamic-table HPACK**. The decoder rejects any indexed
//!   reference into the dynamic table; the encoder never emits
//!   one. This eliminates a large attacker-controlled state
//!   surface (the "HPACK Bomb" class of CVEs cannot apply).
//! - **Server-initiated streams**. The state machine only
//!   transitions client-initiated odd-numbered streams.
//! - **Trailers-only responses**. We accept trailers but always
//!   require a HEADERS+DATA pair before them.
//!
//! ## Status
//!
//! Phase 2 of `verifiable-audit-log-v1`. Frame, HPACK, stream,
//! flow, and gRPC framing modules are wire-level only — there
//! is no async transport here. The `audit-export` crate's
//! `client.rs` (Phase 4) drives this layer over a TLS 1.3
//! connection from `security/`.

pub mod flow;
pub mod frame;
pub mod grpc;
pub mod hpack;
pub mod stream;

/// HTTP/2 connection preface bytes (RFC 7540 § 3.5).
pub const CONNECTION_PREFACE: &[u8; 24] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// Default per-stream initial flow-control window (RFC 7540 § 6.5.2).
pub const DEFAULT_INITIAL_WINDOW_SIZE: u32 = 65_535;

/// Maximum frame payload size we will ever accept on an inbound
/// frame, regardless of any negotiated `SETTINGS_MAX_FRAME_SIZE`.
/// 64 KiB is a deliberately tight cap suited to gRPC unary calls.
pub const MAX_FRAME_PAYLOAD: usize = 65_536;

/// HTTP/2-level error reasons we may surface to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Http2Error {
    /// Frame header was shorter than 9 bytes.
    ShortFrameHeader,
    /// Frame payload length exceeded [`MAX_FRAME_PAYLOAD`].
    OversizedPayload,
    /// Unknown / refused frame type (PUSH_PROMISE, PRIORITY).
    RefusedFrameType,
    /// HPACK dynamic-table reference (refused per design).
    HpackDynamicReference,
    /// HPACK Huffman input was malformed or oversized.
    HpackBadEncoding,
    /// Reserved bit / invalid flag combination per RFC 7540.
    Protocol,
    /// Stream id was invalid for the operation (e.g. server-initiated even id).
    BadStreamId,
    /// Stream state machine refused the requested transition.
    BadStreamState,
    /// Flow-control window would underflow.
    FlowControl,
    /// gRPC frame length exceeded the configured cap.
    GrpcOversized,
    /// gRPC compression flag set (we never advertise compression).
    GrpcCompressed,
}

/// Result alias for HTTP/2 module functions.
pub type Result<T> = core::result::Result<T, Http2Error>;
