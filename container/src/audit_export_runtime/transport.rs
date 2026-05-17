// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `TlsGrpcTransport` — implements
//! `audit_export::immudb::transport::GrpcTransport` over
//! `smallaios-net::http2` framing and an abstract
//! [`TlsStreamLike`] stream.
//!
//! This module is the integration seam between the audit-export
//! pipeline and a TLS 1.3 client. The actual TLS 1.3
//! over TCP client lands in a future change
//! (`tls-tcp-client-v1`) that implements `TlsStreamLike`. Until
//! that lands, this module is exercised against a mock stream
//! in tests.
//!
//! ## Wire protocol
//!
//! For each `unary_call`:
//!
//! 1. If this is the first call on the connection, write the
//!    HTTP/2 connection preface and an empty SETTINGS frame.
//! 2. Build the request HEADERS block via
//!    `net::http2::hpack::encode_block` with the four
//!    pseudo-headers (`:method POST`, `:scheme https`,
//!    `:authority`, `:path`) followed by the regular headers
//!    the caller provided.
//! 3. Send HEADERS (END_HEADERS, no END_STREAM) + DATA (with
//!    END_STREAM) on the next odd stream id.
//! 4. Read response frames: HEADERS (initial response,
//!    END_HEADERS), one or more DATA frames concatenated,
//!    HEADERS (trailers, END_STREAM).
//! 5. Decode response HEADERS blocks; build a `UnaryResponse`.
//!
//! Anything that arrives outside this dance — server push,
//! PRIORITY, unknown frame types, dynamic-table HPACK — is
//! refused per the `net::http2` design.

use audit_export::immudb::transport::{
    GrpcTransport, Header as GrpcHeader, TransportError, UnaryResponse,
};
use smallaios_audit_export as audit_export;
use smallaios_net::http2::{
    frame::{
        build_data, build_headers, build_settings, build_window_update, flags, FrameHeader,
        FrameType,
    },
    grpc::{DEFAULT_MAX_MESSAGE_LEN, PREFIX_LEN},
    hpack::{decode_block, encode_block, Header as HpackHeader},
    stream::Stream,
    CONNECTION_PREFACE, DEFAULT_INITIAL_WINDOW_SIZE, MAX_FRAME_PAYLOAD,
};
use std::io::{self, Read, Write};

/// Post-handshake TLS read/write contract.
///
/// Implementors guarantee:
/// - TLS 1.3 minimum on the wire (TLS 1.2 and below MUST be
///   rejected at handshake).
/// - End-of-stream on the underlying TCP connection surfaces
///   as `Ok(0)` on `read`.
/// - A `close` call sends the close-notify alert.
///
/// The post-handshake stream is `Read + Write`; any state
/// machine (handshake, key rotation, close-notify) lives below
/// the trait implementation.
pub trait TlsStreamLike: Read + Write {
    /// Close the stream — sends close-notify and shuts the
    /// underlying transport. After `close`, no further reads
    /// or writes are valid.
    fn close(&mut self) -> io::Result<()>;
}

/// One client-side gRPC connection backed by a single
/// `TlsStreamLike` stream.
pub struct TlsGrpcTransport<S: TlsStreamLike> {
    stream: S,
    authority: String,
    next_stream_id: u32,
    preface_sent: bool,
    inbound_max_message_len: u32,
}

impl<S: TlsStreamLike> TlsGrpcTransport<S> {
    /// Construct over an already-handshaken TLS stream.
    /// `authority` is the `:authority` pseudo-header value
    /// (`host:port` from the operator's `endpoint`).
    pub fn new(stream: S, authority: impl Into<String>) -> Self {
        Self {
            stream,
            authority: authority.into(),
            next_stream_id: 1,
            preface_sent: false,
            inbound_max_message_len: DEFAULT_MAX_MESSAGE_LEN,
        }
    }

    /// Configure the maximum size we'll accept on an inbound
    /// gRPC message. Default matches immudb's `MaxRecvMsgSize`.
    pub fn with_max_message_len(mut self, max: u32) -> Self {
        self.inbound_max_message_len = max;
        self
    }

    fn allocate_stream_id(&mut self) -> u32 {
        let id = self.next_stream_id;
        // RFC 7540 § 5.1.1: client-initiated streams use odd ids.
        self.next_stream_id = self.next_stream_id.saturating_add(2);
        id
    }

    fn send_preface(&mut self) -> io::Result<()> {
        if self.preface_sent {
            return Ok(());
        }
        self.stream.write_all(CONNECTION_PREFACE)?;
        // Empty SETTINGS — accept the peer's defaults.
        let settings = build_settings(false, &[]);
        self.stream.write_all(&settings)?;
        self.preface_sent = true;
        Ok(())
    }

    /// Read exactly one full frame (header + payload) from the
    /// stream into `out`. Returns the parsed header.
    fn read_frame(&mut self, out: &mut Vec<u8>) -> Result<FrameHeader, TransportError> {
        let mut header_buf = [0u8; 9];
        read_exact(&mut self.stream, &mut header_buf).map_err(map_io)?;
        let header = FrameHeader::parse(&header_buf).map_err(|_| TransportError::Http2)?;
        if header.length as usize > MAX_FRAME_PAYLOAD {
            return Err(TransportError::Http2);
        }
        out.resize(header.length as usize, 0);
        if !out.is_empty() {
            read_exact(&mut self.stream, out).map_err(map_io)?;
        }
        Ok(header)
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<(), TransportError> {
        self.stream.write_all(bytes).map_err(map_io)
    }
}

/// Outcome of dispatching a single inbound response frame.
enum FrameOutcome {
    /// Keep reading frames.
    Continue,
    /// Response is complete; carries the resolved trailers.
    Done(Vec<GrpcHeader>),
}

/// Mutable response-accumulation state shared across the
/// frame-dispatch loop in `unary_call`.
struct ResponseState {
    stream_id: u32,
    stream_state: Stream,
    initial_headers: Option<Vec<GrpcHeader>>,
    response_body: Vec<u8>,
}

impl<S: TlsStreamLike> TlsGrpcTransport<S> {
    /// Build the HEADERS block: pseudo-headers in fixed order
    /// (per RFC 7540 § 8.1.2.3) followed by user headers.
    fn build_request_headers(&self, full_method: &str, headers: &[GrpcHeader]) -> Vec<HpackHeader> {
        let mut all_headers: Vec<HpackHeader> = Vec::with_capacity(4 + headers.len());
        all_headers.push(HpackHeader {
            name: ":method".into(),
            value: "POST".into(),
        });
        all_headers.push(HpackHeader {
            name: ":scheme".into(),
            value: "https".into(),
        });
        all_headers.push(HpackHeader {
            name: ":authority".into(),
            value: self.authority.clone(),
        });
        all_headers.push(HpackHeader {
            name: ":path".into(),
            value: full_method.into(),
        });
        for h in headers {
            all_headers.push(HpackHeader {
                name: h.name.clone(),
                value: h.value.clone(),
            });
        }
        all_headers
    }

    /// ACK any incoming SETTINGS (no fields applied; we accept
    /// defaults).
    fn handle_settings_frame(&mut self, header: &FrameHeader) -> Result<(), TransportError> {
        if header.flags & flags::ACK == 0 {
            let ack = build_settings(true, &[]);
            self.write_all(&ack)?;
        }
        Ok(())
    }

    /// Echo PING with ACK if it isn't already an ACK.
    fn handle_ping_frame(
        &mut self,
        header: &FrameHeader,
        payload: &[u8],
    ) -> Result<(), TransportError> {
        if header.flags & flags::ACK == 0 && header.length == 8 {
            let mut data = [0u8; 8];
            data.copy_from_slice(payload);
            self.write_all(&smallaios_net::http2::frame::build_ping(true, data))?;
        }
        Ok(())
    }

    fn handle_headers_frame(
        &mut self,
        header: &FrameHeader,
        payload: &[u8],
        state: &mut ResponseState,
    ) -> Result<FrameOutcome, TransportError> {
        if header.stream_id != state.stream_id {
            return Err(TransportError::Http2);
        }
        let end_stream = header.flags & flags::END_STREAM != 0;
        let decoded = decode_block(payload).map_err(|_| TransportError::Http2)?;
        let mapped: Vec<GrpcHeader> = decoded
            .into_iter()
            .map(|h| GrpcHeader::new(h.name, h.value))
            .collect();
        if state.initial_headers.is_none() {
            state
                .stream_state
                .recv_headers(end_stream)
                .map_err(|_| TransportError::Http2)?;
            state.initial_headers = Some(mapped);
            if end_stream {
                return Ok(FrameOutcome::Done(Vec::new()));
            }
            Ok(FrameOutcome::Continue)
        } else {
            // Trailing HEADERS MUST carry END_STREAM.
            if !end_stream {
                return Err(TransportError::Http2);
            }
            state
                .stream_state
                .recv_headers(true)
                .map_err(|_| TransportError::Http2)?;
            Ok(FrameOutcome::Done(mapped))
        }
    }

    fn handle_data_frame(
        &mut self,
        header: &FrameHeader,
        payload: &[u8],
        state: &mut ResponseState,
    ) -> Result<FrameOutcome, TransportError> {
        if header.stream_id != state.stream_id {
            return Err(TransportError::Http2);
        }
        let end_stream = header.flags & flags::END_STREAM != 0;
        state
            .stream_state
            .recv_data(end_stream)
            .map_err(|_| TransportError::Http2)?;
        state.response_body.extend_from_slice(payload);
        if state.response_body.len() > (self.inbound_max_message_len as usize) + PREFIX_LEN {
            return Err(TransportError::Http2);
        }
        if end_stream {
            return Ok(FrameOutcome::Done(Vec::new()));
        }
        // Replenish connection-level window if we just consumed
        // a non-trivial DATA frame.
        if header.length > 0 {
            self.write_all(&build_window_update(0, header.length))?;
        }
        Ok(FrameOutcome::Continue)
    }

    /// Dispatch a single inbound frame, mutating `state` and
    /// emitting any required protocol responses.
    fn dispatch_response_frame(
        &mut self,
        header: &FrameHeader,
        payload: &[u8],
        state: &mut ResponseState,
    ) -> Result<FrameOutcome, TransportError> {
        match header.frame_type {
            FrameType::Settings => {
                self.handle_settings_frame(header)?;
                Ok(FrameOutcome::Continue)
            }
            FrameType::Ping => {
                self.handle_ping_frame(header, payload)?;
                Ok(FrameOutcome::Continue)
            }
            FrameType::WindowUpdate => {
                // Ignore — flow-control credits are tracked but
                // we never request more than the default window.
                Ok(FrameOutcome::Continue)
            }
            FrameType::GoAway => Err(TransportError::Http2),
            FrameType::RstStream => {
                if header.stream_id == state.stream_id {
                    state.stream_state.remote_reset();
                    return Err(TransportError::Http2);
                }
                Ok(FrameOutcome::Continue)
            }
            FrameType::Headers => self.handle_headers_frame(header, payload, state),
            FrameType::Data => self.handle_data_frame(header, payload, state),
            FrameType::Continuation => {
                // We require END_HEADERS on the initial HEADERS
                // frame, so CONTINUATION is unexpected.
                Err(TransportError::Http2)
            }
        }
    }

    /// Read response frames until the stream completes,
    /// accumulating body DATA frames between the initial HEADERS
    /// and the trailing HEADERS.
    fn read_unary_response(
        &mut self,
        mut state: ResponseState,
    ) -> Result<UnaryResponse, TransportError> {
        let mut payload = Vec::new();
        let trailers = loop {
            payload.clear();
            let header = self.read_frame(&mut payload)?;
            match self.dispatch_response_frame(&header, &payload, &mut state)? {
                FrameOutcome::Continue => {}
                FrameOutcome::Done(t) => break t,
            }
        };

        Ok(UnaryResponse {
            initial_headers: state.initial_headers.unwrap_or_default(),
            body: state.response_body,
            trailers,
        })
    }
}

impl<S: TlsStreamLike> GrpcTransport for TlsGrpcTransport<S> {
    fn unary_call(
        &mut self,
        full_method: &str,
        headers: &[GrpcHeader],
        body: &[u8],
    ) -> Result<UnaryResponse, TransportError> {
        self.send_preface().map_err(map_io)?;

        let stream_id = self.allocate_stream_id();
        let mut stream_state = Stream::new(stream_id).map_err(|_| TransportError::Http2)?;

        let all_headers = self.build_request_headers(full_method, headers);
        let block = encode_block(&all_headers);

        stream_state
            .send_headers(false)
            .map_err(|_| TransportError::Http2)?;
        self.write_all(&build_headers(stream_id, &block, false, true))?;

        // DATA with END_STREAM.
        if body.len() > MAX_FRAME_PAYLOAD {
            return Err(TransportError::Http2);
        }
        stream_state
            .send_data(true)
            .map_err(|_| TransportError::Http2)?;
        self.write_all(&build_data(stream_id, body, true))?;

        self.read_unary_response(ResponseState {
            stream_id,
            stream_state,
            initial_headers: None,
            response_body: Vec::new(),
        })
    }
}

fn map_io(e: io::Error) -> TransportError {
    match e.kind() {
        io::ErrorKind::TimedOut => TransportError::Timeout,
        io::ErrorKind::ConnectionRefused
        | io::ErrorKind::ConnectionReset
        | io::ErrorKind::NotConnected
        | io::ErrorKind::HostUnreachable
        | io::ErrorKind::NetworkUnreachable => TransportError::ConnectFailed,
        _ => TransportError::Io,
    }
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        let n = r.read(&mut buf[filled..])?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
        }
        filled += n;
    }
    Ok(())
}

// Silence the unused-import lint when DEFAULT_INITIAL_WINDOW_SIZE
// is only indirectly referenced for documentation purposes.
#[allow(dead_code)]
const _USE_INITIAL_WINDOW: u32 = DEFAULT_INITIAL_WINDOW_SIZE;

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_net::http2::frame::FrameType as FT;
    use std::io::Cursor;

    /// Mock stream: a reader from a Vec<u8> + a writer into another
    /// Vec<u8>. Lets us assert wire bytes the transport emits and
    /// feed it canned response frames.
    struct MockStream {
        read_buf: Cursor<Vec<u8>>,
        write_buf: Vec<u8>,
        closed: bool,
    }

    impl MockStream {
        fn new(inbound: Vec<u8>) -> Self {
            Self {
                read_buf: Cursor::new(inbound),
                write_buf: Vec::new(),
                closed: false,
            }
        }
    }

    impl Read for MockStream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.read_buf.read(buf)
        }
    }

    impl Write for MockStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.write_buf.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl TlsStreamLike for MockStream {
        fn close(&mut self) -> io::Result<()> {
            self.closed = true;
            Ok(())
        }
    }

    /// Build a response frame sequence: SETTINGS ACK + initial
    /// HEADERS + DATA + trailing HEADERS.
    fn canned_unary_response(stream_id: u32, body: &[u8], grpc_status: &str) -> Vec<u8> {
        let mut out = Vec::new();
        // Server SETTINGS (empty).
        out.extend_from_slice(&build_settings(false, &[]));
        // Initial HEADERS.
        let initial = encode_block(&[
            HpackHeader {
                name: ":status".into(),
                value: "200".into(),
            },
            HpackHeader {
                name: "content-type".into(),
                value: "application/grpc+proto".into(),
            },
        ]);
        out.extend_from_slice(&build_headers(stream_id, &initial, false, true));
        // DATA.
        out.extend_from_slice(&build_data(stream_id, body, false));
        // Trailing HEADERS (grpc-status).
        let trailers = encode_block(&[HpackHeader {
            name: "grpc-status".into(),
            value: grpc_status.into(),
        }]);
        out.extend_from_slice(&build_headers(stream_id, &trailers, true, true));
        out
    }

    #[test]
    fn transport_sends_preface_then_request() {
        let response = canned_unary_response(1, b"hello", "0");
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "immudb.example.com:3322");
        let resp = t
            .unary_call("/svc/method", &[], b"\x00\x00\x00\x00\x05hello")
            .unwrap();
        // Connection preface MUST be the first 24 bytes sent.
        assert!(t.stream.write_buf.starts_with(CONNECTION_PREFACE));
        assert_eq!(resp.grpc_status(), 0);
        assert_eq!(resp.body, b"hello");
    }

    #[test]
    fn transport_attaches_pseudo_and_user_headers() {
        let response = canned_unary_response(1, b"", "0");
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "immudb.example.com:3322");
        t.unary_call(
            "/immudb.schema.ImmuService/CurrentState",
            &[GrpcHeader::new("content-type", "application/grpc+proto")],
            b"",
        )
        .unwrap();
        // Find the first HEADERS frame in the write buffer and
        // decode its block to confirm the pseudo-headers are
        // present in the right order.
        let written = t.stream.write_buf.clone();
        let preface_len = CONNECTION_PREFACE.len();
        let mut cur = preface_len;
        // Skip our SETTINGS frame.
        let hdr = FrameHeader::parse(&written[cur..cur + 9]).unwrap();
        assert_eq!(hdr.frame_type, FT::Settings);
        cur += 9 + hdr.length as usize;
        // Now the HEADERS frame.
        let hdr = FrameHeader::parse(&written[cur..cur + 9]).unwrap();
        assert_eq!(hdr.frame_type, FT::Headers);
        let block_start = cur + 9;
        let block_end = block_start + hdr.length as usize;
        let block = &written[block_start..block_end];
        let decoded = decode_block(block).unwrap();
        let names: Vec<&str> = decoded.iter().map(|h| h.name.as_str()).collect();
        assert_eq!(names[0], ":method");
        assert_eq!(names[1], ":scheme");
        assert_eq!(names[2], ":authority");
        assert_eq!(names[3], ":path");
        let path = &decoded[3].value;
        assert_eq!(path, "/immudb.schema.ImmuService/CurrentState");
        let authority = &decoded[2].value;
        assert_eq!(authority, "immudb.example.com:3322");
    }

    #[test]
    fn transport_only_sends_preface_once() {
        let response1 = canned_unary_response(1, b"", "0");
        let response2 = canned_unary_response(3, b"", "0");
        let mut combined = response1;
        combined.extend(response2);
        let stream = MockStream::new(combined);
        let mut t = TlsGrpcTransport::new(stream, "x:1");
        t.unary_call("/a/b", &[], b"").unwrap();
        let after_first = t.stream.write_buf.len();
        t.unary_call("/a/c", &[], b"").unwrap();
        let _ = after_first;
        // Count occurrences of the preface bytes in the buffer.
        let count = t
            .stream
            .write_buf
            .windows(CONNECTION_PREFACE.len())
            .filter(|w| *w == CONNECTION_PREFACE)
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn transport_propagates_grpc_status() {
        let response = canned_unary_response(1, b"x", "14");
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "x:1");
        let resp = t.unary_call("/a/b", &[], b"").unwrap();
        assert_eq!(resp.grpc_status(), 14);
    }

    #[test]
    fn transport_handles_inline_settings_ack() {
        // Server sends two SETTINGS frames before the HEADERS — one
        // with fields (we ACK), one which is itself an ACK (we
        // ignore).
        let mut response = Vec::new();
        response.extend(build_settings(false, &[(0x3, 100)]));
        response.extend(build_settings(true, &[]));
        response.extend(canned_unary_response(1, b"", "0"));
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "x:1");
        t.unary_call("/a/b", &[], b"").unwrap();
        // We should have written: preface + our SETTINGS + HEADERS
        // + DATA + SETTINGS ACK (for the inbound non-ACK SETTINGS).
        let ack = build_settings(true, &[]);
        let count = t
            .stream
            .write_buf
            .windows(ack.len())
            .filter(|w| *w == ack)
            .count();
        // First write is our SETTINGS (non-ACK). Then later we
        // ACK the inbound. So at least one ACK in the buffer.
        assert!(count >= 1);
    }

    #[test]
    fn transport_responds_to_ping_with_ack() {
        let mut response = Vec::new();
        response.extend(build_settings(false, &[]));
        // PING with non-ACK.
        response.extend(smallaios_net::http2::frame::build_ping(false, [1u8; 8]));
        response.extend(canned_unary_response(1, b"", "0"));
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "x:1");
        t.unary_call("/a/b", &[], b"").unwrap();
        // PING ACK must have been written.
        let ack = smallaios_net::http2::frame::build_ping(true, [1u8; 8]);
        assert!(t.stream.write_buf.windows(ack.len()).any(|w| w == ack));
    }

    #[test]
    fn transport_rejects_goaway() {
        let mut response = Vec::new();
        response.extend(build_settings(false, &[]));
        response.extend(smallaios_net::http2::frame::build_goaway(0, 1));
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "x:1");
        let err = t.unary_call("/a/b", &[], b"").unwrap_err();
        assert_eq!(err, TransportError::Http2);
    }

    #[test]
    fn transport_rejects_rst_stream_on_our_stream() {
        let mut response = Vec::new();
        response.extend(build_settings(false, &[]));
        // RST_STREAM on stream id 1.
        let mut rst = vec![0u8; 9];
        FrameHeader {
            length: 4,
            frame_type: FT::RstStream,
            flags: 0,
            stream_id: 1,
        }
        .encode(&mut rst)
        .unwrap();
        rst.extend_from_slice(&8u32.to_be_bytes()); // CANCEL
        response.extend(rst);
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "x:1");
        assert!(t.unary_call("/a/b", &[], b"").is_err());
    }

    #[test]
    fn transport_concatenates_split_data_frames() {
        let mut response = Vec::new();
        response.extend(build_settings(false, &[]));
        let initial = encode_block(&[HpackHeader {
            name: ":status".into(),
            value: "200".into(),
        }]);
        response.extend(build_headers(1, &initial, false, true));
        response.extend(build_data(1, b"hello ", false));
        response.extend(build_data(1, b"world", false));
        let trailers = encode_block(&[HpackHeader {
            name: "grpc-status".into(),
            value: "0".into(),
        }]);
        response.extend(build_headers(1, &trailers, true, true));
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "x:1");
        let resp = t.unary_call("/a/b", &[], b"").unwrap();
        assert_eq!(resp.body, b"hello world");
    }

    #[test]
    fn transport_oversized_message_rejected() {
        let mut response = Vec::new();
        response.extend(build_settings(false, &[]));
        let initial = encode_block(&[HpackHeader {
            name: ":status".into(),
            value: "200".into(),
        }]);
        response.extend(build_headers(1, &initial, false, true));
        // 32 KiB DATA frame (fits in one HTTP/2 frame) on a 4 KiB
        // inbound gRPC-message cap.
        let big = vec![0xab; 32 * 1024];
        response.extend(build_data(1, &big, true));
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "x:1").with_max_message_len(4 * 1024);
        let err = t.unary_call("/a/b", &[], b"").unwrap_err();
        assert_eq!(err, TransportError::Http2);
    }

    #[test]
    fn transport_truncated_response_propagates_io() {
        // Server SETTINGS only — no HEADERS, no DATA, EOF.
        let response = build_settings(false, &[]);
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "x:1");
        let err = t.unary_call("/a/b", &[], b"").unwrap_err();
        // EOF maps through io::Error to TransportError::Io.
        assert_eq!(err, TransportError::Io);
    }

    #[test]
    fn transport_stream_ids_are_odd_and_increasing() {
        let mut response = Vec::new();
        response.extend(canned_unary_response(1, b"", "0"));
        response.extend(canned_unary_response(3, b"", "0"));
        response.extend(canned_unary_response(5, b"", "0"));
        let stream = MockStream::new(response);
        let mut t = TlsGrpcTransport::new(stream, "x:1");
        t.unary_call("/a/b", &[], b"").unwrap();
        t.unary_call("/a/c", &[], b"").unwrap();
        t.unary_call("/a/d", &[], b"").unwrap();
        // The next stream id we'd allocate would be 7.
        assert_eq!(t.next_stream_id, 7);
    }
}
