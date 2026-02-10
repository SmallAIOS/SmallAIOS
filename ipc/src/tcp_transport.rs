// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TCP transport with Zenoh-compatible wire protocol framing.
//!
//! Implements a simplified Zenoh-inspired wire protocol over TCP for
//! communication with external clients. The transport manages a
//! connection state machine that models the Zenoh session lifecycle:
//! Init, Open, Data exchange, and Close.
//!
//! Frame format (simplified Zenoh):
//! ```text
//! +--------+--------+--------+--------+--------+--------+--------+
//! |      length (u32 BE)              | type   | flags  | payload...
//! +--------+--------+--------+--------+--------+--------+--------+
//! ```
//!
//! The `length` field covers everything after it: type (1) + flags (1) + payload.

#![allow(dead_code)]

use alloc::vec::Vec;

use crate::IpcError;

/// Minimum frame size: 4 (length) + 1 (type) + 1 (flags) = 6 bytes.
const MIN_FRAME_SIZE: usize = 6;

/// Frame types in the Zenoh-inspired wire protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    /// Session initialisation request.
    Init = 0x01,
    /// Acknowledgement of Init.
    InitAck = 0x02,
    /// Open session (after Init/InitAck).
    Open = 0x03,
    /// Acknowledgement of Open.
    OpenAck = 0x04,
    /// Close the session.
    Close = 0x05,
    /// Keep-alive ping/pong.
    KeepAlive = 0x06,
    /// Data message.
    Data = 0x10,
    /// Query message.
    Query = 0x11,
    /// Reply message.
    Reply = 0x12,
}

impl FrameType {
    /// Try to convert a raw byte into a `FrameType`.
    ///
    /// Returns `None` if the byte does not correspond to a known type.
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(FrameType::Init),
            0x02 => Some(FrameType::InitAck),
            0x03 => Some(FrameType::Open),
            0x04 => Some(FrameType::OpenAck),
            0x05 => Some(FrameType::Close),
            0x06 => Some(FrameType::KeepAlive),
            0x10 => Some(FrameType::Data),
            0x11 => Some(FrameType::Query),
            0x12 => Some(FrameType::Reply),
            _ => None,
        }
    }
}

/// A single wire-protocol frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Type of this frame.
    pub frame_type: FrameType,
    /// Flags byte (interpretation depends on frame type).
    pub flags: u8,
    /// Payload data.
    pub payload: Vec<u8>,
}

impl Frame {
    /// Create a new frame.
    pub fn new(frame_type: FrameType, flags: u8, payload: Vec<u8>) -> Self {
        Self {
            frame_type,
            flags,
            payload,
        }
    }

    /// Encode this frame into a byte vector.
    ///
    /// Wire format: length(u32 BE) + frame_type(u8) + flags(u8) + payload.
    /// The `length` field is the number of bytes following the length field
    /// itself (i.e., `2 + payload.len()`).
    pub fn encode(&self) -> Vec<u8> {
        let body_len = 2 + self.payload.len(); // type(1) + flags(1) + payload
        let total_len = 4 + body_len; // length(4) + body
        let mut buf = Vec::with_capacity(total_len);

        // Length (u32 big-endian): covers type + flags + payload.
        let len_u32 = body_len as u32;
        buf.push((len_u32 >> 24) as u8);
        buf.push((len_u32 >> 16) as u8);
        buf.push((len_u32 >> 8) as u8);
        buf.push(len_u32 as u8);

        // Frame type.
        buf.push(self.frame_type as u8);

        // Flags.
        buf.push(self.flags);

        // Payload.
        buf.extend_from_slice(&self.payload);

        buf
    }

    /// Decode a frame from a byte slice.
    ///
    /// Returns the decoded frame and the total number of bytes consumed
    /// (including the 4-byte length prefix).
    ///
    /// # Errors
    ///
    /// - [`IpcError::PacketTooShort`] if `data` has fewer than 6 bytes or
    ///   fewer bytes than the length field indicates.
    /// - [`IpcError::InvalidState`] if the frame type byte is unknown.
    pub fn decode(data: &[u8]) -> Result<(Frame, usize), IpcError> {
        if data.len() < MIN_FRAME_SIZE {
            return Err(IpcError::PacketTooShort);
        }

        // Read the length prefix (u32 big-endian).
        let body_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;

        // Total frame size = 4 (length) + body_len.
        let total = 4 + body_len;
        if data.len() < total {
            return Err(IpcError::PacketTooShort);
        }

        // Body must have at least type(1) + flags(1).
        if body_len < 2 {
            return Err(IpcError::PacketTooShort);
        }

        let frame_type_byte = data[4];
        let flags = data[5];
        let payload = data[6..total].to_vec();

        let frame_type = FrameType::from_u8(frame_type_byte).ok_or(IpcError::InvalidState)?;

        Ok((
            Frame {
                frame_type,
                flags,
                payload,
            },
            total,
        ))
    }
}

/// Connection state machine for the TCP transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// No session active.
    Idle,
    /// We sent Init, waiting for InitAck.
    InitSent,
    /// We received Init from peer, sent InitAck, waiting for Open.
    InitReceived,
    /// Session is open and data can flow.
    Open,
    /// We sent Close, waiting for acknowledgement.
    Closing,
    /// Session is closed.
    Closed,
}

/// Configuration for a TCP transport session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportConfig {
    /// Maximum frame size in bytes (including header).
    pub max_frame_size: usize,
    /// Keep-alive interval in milliseconds.
    pub keepalive_interval_ms: u64,
}

impl TransportConfig {
    /// Create a `TransportConfig` with sensible defaults.
    pub fn default_config() -> Self {
        Self {
            max_frame_size: 65536,
            keepalive_interval_ms: 10000,
        }
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// TCP transport implementing a Zenoh-inspired session state machine.
///
/// The transport manages the connection lifecycle:
/// 1. **Initiation**: One side sends Init, the other replies with InitAck.
/// 2. **Open**: The initiator sends Open, the responder replies with OpenAck,
///    and the session enters the Open state.
/// 3. **Data exchange**: Data, Query, and Reply frames can be sent.
/// 4. **Close**: Either side can close the session.
#[derive(Debug)]
pub struct TcpTransport {
    /// Current connection state.
    state: ConnectionState,
    /// Transport configuration.
    config: TransportConfig,
    /// Remote peer identifier (set during handshake).
    peer_id: Option<u64>,
    /// Outgoing frame buffer.
    tx_buffer: Vec<u8>,
    /// Incoming data buffer (accumulates raw bytes before framing).
    rx_buffer: Vec<u8>,
}

impl TcpTransport {
    /// Create a new TCP transport in the Idle state.
    pub fn new(config: TransportConfig) -> Self {
        Self {
            state: ConnectionState::Idle,
            config,
            peer_id: None,
            tx_buffer: Vec::new(),
            rx_buffer: Vec::new(),
        }
    }

    /// Return a reference to the current connection state.
    pub fn state(&self) -> &ConnectionState {
        &self.state
    }

    /// Return a reference to the transport configuration.
    pub fn config(&self) -> &TransportConfig {
        &self.config
    }

    /// Return the remote peer ID if the session has been established.
    pub fn peer_id(&self) -> Option<u64> {
        self.peer_id
    }

    /// Initiate a session by creating an Init frame.
    ///
    /// Transitions the transport from `Idle` to `InitSent`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the current state is `Idle`.
    pub fn initiate(&mut self) -> Frame {
        debug_assert_eq!(self.state, ConnectionState::Idle);
        self.state = ConnectionState::InitSent;
        Frame::new(FrameType::Init, 0x00, Vec::new())
    }

    /// Handle an incoming frame according to the connection state machine.
    ///
    /// Returns `Ok(Some(response_frame))` if a response should be sent,
    /// `Ok(None)` if no response is needed (e.g., data frames in Open state),
    /// or `Err(IpcError::InvalidState)` for invalid state transitions.
    pub fn handle_frame(&mut self, frame: &Frame) -> Result<Option<Frame>, IpcError> {
        match (&self.state, frame.frame_type) {
            // --- Handshake: initiator side ---
            // We sent Init, received InitAck -> send Open, transition to Open.
            (ConnectionState::InitSent, FrameType::InitAck) => {
                // Extract peer_id from InitAck payload if present (first 8 bytes).
                if frame.payload.len() >= 8 {
                    let peer_id_bytes: [u8; 8] = frame.payload[..8]
                        .try_into()
                        .map_err(|_| IpcError::InvalidState)?;
                    self.peer_id = Some(u64::from_be_bytes(peer_id_bytes));
                }
                self.state = ConnectionState::Open;
                Ok(Some(Frame::new(FrameType::Open, 0x00, Vec::new())))
            }

            // --- Handshake: responder side ---
            // We are Idle, received Init -> send InitAck, transition to InitReceived.
            (ConnectionState::Idle, FrameType::Init) => {
                self.state = ConnectionState::InitReceived;
                Ok(Some(Frame::new(FrameType::InitAck, 0x00, Vec::new())))
            }

            // We sent InitAck (InitReceived), received Open -> send OpenAck, transition to Open.
            (ConnectionState::InitReceived, FrameType::Open) => {
                self.state = ConnectionState::Open;
                Ok(Some(Frame::new(FrameType::OpenAck, 0x00, Vec::new())))
            }

            // --- Open state ---
            // Close request while open.
            (ConnectionState::Open, FrameType::Close) => {
                self.state = ConnectionState::Closed;
                Ok(None)
            }

            // KeepAlive while open -> respond with KeepAlive.
            (ConnectionState::Open, FrameType::KeepAlive) => {
                Ok(Some(Frame::new(FrameType::KeepAlive, 0x00, Vec::new())))
            }

            // Data, Query, Reply while open -> accept silently.
            (ConnectionState::Open, FrameType::Data)
            | (ConnectionState::Open, FrameType::Query)
            | (ConnectionState::Open, FrameType::Reply) => Ok(None),

            // --- Closing ---
            // We sent Close (Closing), received Close -> transition to Closed.
            (ConnectionState::Closing, FrameType::Close) => {
                self.state = ConnectionState::Closed;
                Ok(None)
            }

            // --- Everything else is an invalid transition ---
            _ => Err(IpcError::InvalidState),
        }
    }

    /// Create a Close frame and transition to the Closing state.
    ///
    /// If the transport is already in Open state, transitions to Closing.
    /// Returns the Close frame to be sent to the peer.
    pub fn close(&mut self) -> Frame {
        self.state = ConnectionState::Closing;
        Frame::new(FrameType::Close, 0x00, Vec::new())
    }

    /// Append raw bytes to the internal receive buffer.
    ///
    /// The caller feeds TCP stream data here; complete frames can then be
    /// extracted with [`try_decode_frame`](TcpTransport::try_decode_frame).
    pub fn feed_data(&mut self, data: &[u8]) {
        self.rx_buffer.extend_from_slice(data);
    }

    /// Attempt to decode one complete frame from the receive buffer.
    ///
    /// If enough data is available, decodes the frame, removes the consumed
    /// bytes from the buffer, and returns the frame. Otherwise returns
    /// `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns an error if frame decoding fails for reasons other than
    /// insufficient data (e.g., unknown frame type).
    pub fn try_decode_frame(&mut self) -> Result<Option<Frame>, IpcError> {
        if self.rx_buffer.len() < MIN_FRAME_SIZE {
            return Ok(None);
        }

        // Peek at the length to see if we have enough data.
        let body_len = u32::from_be_bytes([
            self.rx_buffer[0],
            self.rx_buffer[1],
            self.rx_buffer[2],
            self.rx_buffer[3],
        ]) as usize;

        let total = 4 + body_len;
        if self.rx_buffer.len() < total {
            return Ok(None);
        }

        // We have enough data; decode and drain.
        let (frame, consumed) = Frame::decode(&self.rx_buffer)?;
        self.rx_buffer.drain(..consumed);
        Ok(Some(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // === Frame encode/decode ===

    #[test]
    fn frame_encode_decode_roundtrip_empty_payload() {
        let frame = Frame::new(FrameType::Init, 0x00, Vec::new());
        let encoded = frame.encode();
        let (decoded, consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded, frame);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn frame_encode_decode_roundtrip_with_payload() {
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let frame = Frame::new(FrameType::Data, 0x42, payload.clone());
        let encoded = frame.encode();

        let (decoded, consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.frame_type, FrameType::Data);
        assert_eq!(decoded.flags, 0x42);
        assert_eq!(decoded.payload, payload);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn frame_encode_length_field() {
        let frame = Frame::new(FrameType::KeepAlive, 0x00, vec![1, 2, 3]);
        let encoded = frame.encode();

        // Length field: type(1) + flags(1) + payload(3) = 5
        let len = u32::from_be_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        assert_eq!(len, 5);
        // Total size: 4 + 5 = 9
        assert_eq!(encoded.len(), 9);
    }

    #[test]
    fn frame_decode_too_short() {
        let short_data = [0x00, 0x00, 0x00]; // Only 3 bytes
        let result = Frame::decode(&short_data);
        assert_eq!(result, Err(IpcError::PacketTooShort));
    }

    #[test]
    fn frame_decode_insufficient_body() {
        // Length says 10 bytes of body, but we only provide 6 total bytes.
        let data = [0x00, 0x00, 0x00, 0x0A, 0x01, 0x00]; // len=10, but only 2 body bytes
        let result = Frame::decode(&data);
        assert_eq!(result, Err(IpcError::PacketTooShort));
    }

    #[test]
    fn frame_decode_unknown_type() {
        // Valid length but unknown frame type (0xFF).
        let data = [0x00, 0x00, 0x00, 0x02, 0xFF, 0x00];
        let result = Frame::decode(&data);
        assert_eq!(result, Err(IpcError::InvalidState));
    }

    #[test]
    fn frame_decode_all_types() {
        let types = [
            (0x01u8, FrameType::Init),
            (0x02, FrameType::InitAck),
            (0x03, FrameType::Open),
            (0x04, FrameType::OpenAck),
            (0x05, FrameType::Close),
            (0x06, FrameType::KeepAlive),
            (0x10, FrameType::Data),
            (0x11, FrameType::Query),
            (0x12, FrameType::Reply),
        ];
        for (byte, expected) in types.iter() {
            let data = [0x00, 0x00, 0x00, 0x02, *byte, 0x00];
            let (frame, _) = Frame::decode(&data).unwrap();
            assert_eq!(frame.frame_type, *expected);
        }
    }

    // === Connection state machine ===

    #[test]
    fn init_open_handshake_initiator() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());
        assert_eq!(*transport.state(), ConnectionState::Idle);

        // Initiator sends Init.
        let init_frame = transport.initiate();
        assert_eq!(init_frame.frame_type, FrameType::Init);
        assert_eq!(*transport.state(), ConnectionState::InitSent);

        // Initiator receives InitAck.
        let init_ack = Frame::new(FrameType::InitAck, 0x00, Vec::new());
        let response = transport.handle_frame(&init_ack).unwrap();
        assert!(response.is_some());
        let open_frame = response.unwrap();
        assert_eq!(open_frame.frame_type, FrameType::Open);
        assert_eq!(*transport.state(), ConnectionState::Open);
    }

    #[test]
    fn init_open_handshake_responder() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());
        assert_eq!(*transport.state(), ConnectionState::Idle);

        // Responder receives Init.
        let init = Frame::new(FrameType::Init, 0x00, Vec::new());
        let response = transport.handle_frame(&init).unwrap();
        assert!(response.is_some());
        let init_ack = response.unwrap();
        assert_eq!(init_ack.frame_type, FrameType::InitAck);
        assert_eq!(*transport.state(), ConnectionState::InitReceived);

        // Responder receives Open.
        let open = Frame::new(FrameType::Open, 0x00, Vec::new());
        let response = transport.handle_frame(&open).unwrap();
        assert!(response.is_some());
        let open_ack = response.unwrap();
        assert_eq!(open_ack.frame_type, FrameType::OpenAck);
        assert_eq!(*transport.state(), ConnectionState::Open);
    }

    #[test]
    fn close_sequence() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());

        // Fast-forward to Open state.
        let _ = transport.initiate();
        transport
            .handle_frame(&Frame::new(FrameType::InitAck, 0x00, Vec::new()))
            .unwrap();
        assert_eq!(*transport.state(), ConnectionState::Open);

        // Receive Close from peer.
        let close = Frame::new(FrameType::Close, 0x00, Vec::new());
        let response = transport.handle_frame(&close).unwrap();
        assert!(response.is_none());
        assert_eq!(*transport.state(), ConnectionState::Closed);
    }

    #[test]
    fn close_initiated_locally() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());

        // Fast-forward to Open state.
        let _ = transport.initiate();
        transport
            .handle_frame(&Frame::new(FrameType::InitAck, 0x00, Vec::new()))
            .unwrap();
        assert_eq!(*transport.state(), ConnectionState::Open);

        // Locally initiate close.
        let close_frame = transport.close();
        assert_eq!(close_frame.frame_type, FrameType::Close);
        assert_eq!(*transport.state(), ConnectionState::Closing);

        // Receive Close acknowledgement from peer.
        let response = transport
            .handle_frame(&Frame::new(FrameType::Close, 0x00, Vec::new()))
            .unwrap();
        assert!(response.is_none());
        assert_eq!(*transport.state(), ConnectionState::Closed);
    }

    #[test]
    fn keepalive_in_open_state() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());

        // Fast-forward to Open.
        let _ = transport.initiate();
        transport
            .handle_frame(&Frame::new(FrameType::InitAck, 0x00, Vec::new()))
            .unwrap();

        let ka = Frame::new(FrameType::KeepAlive, 0x00, Vec::new());
        let response = transport.handle_frame(&ka).unwrap();
        assert!(response.is_some());
        let ka_reply = response.unwrap();
        assert_eq!(ka_reply.frame_type, FrameType::KeepAlive);
    }

    #[test]
    fn data_frame_in_open_state() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());

        // Fast-forward to Open.
        let _ = transport.initiate();
        transport
            .handle_frame(&Frame::new(FrameType::InitAck, 0x00, Vec::new()))
            .unwrap();

        let data = Frame::new(FrameType::Data, 0x00, vec![1, 2, 3, 4]);
        let response = transport.handle_frame(&data).unwrap();
        assert!(response.is_none()); // Data accepted silently.
    }

    #[test]
    fn query_and_reply_frames_in_open_state() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());

        // Fast-forward to Open.
        let _ = transport.initiate();
        transport
            .handle_frame(&Frame::new(FrameType::InitAck, 0x00, Vec::new()))
            .unwrap();

        // Query frame.
        let query = Frame::new(FrameType::Query, 0x00, vec![0xAA]);
        let response = transport.handle_frame(&query).unwrap();
        assert!(response.is_none());

        // Reply frame.
        let reply = Frame::new(FrameType::Reply, 0x00, vec![0xBB]);
        let response = transport.handle_frame(&reply).unwrap();
        assert!(response.is_none());
    }

    #[test]
    fn invalid_state_transition_data_in_idle() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());
        let data = Frame::new(FrameType::Data, 0x00, vec![1]);
        let result = transport.handle_frame(&data);
        assert_eq!(result, Err(IpcError::InvalidState));
    }

    #[test]
    fn invalid_state_transition_open_in_idle() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());
        let open = Frame::new(FrameType::Open, 0x00, Vec::new());
        let result = transport.handle_frame(&open);
        assert_eq!(result, Err(IpcError::InvalidState));
    }

    // === feed_data / try_decode_frame ===

    #[test]
    fn feed_data_and_decode_frame() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());

        let frame = Frame::new(FrameType::Init, 0x00, vec![0x01, 0x02]);
        let encoded = frame.encode();

        transport.feed_data(&encoded);
        let decoded = transport.try_decode_frame().unwrap();
        assert!(decoded.is_some());
        assert_eq!(decoded.unwrap(), frame);

        // Buffer should be drained.
        let next = transport.try_decode_frame().unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn feed_data_partial_then_complete() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());

        let frame = Frame::new(FrameType::Data, 0x00, vec![0xAA, 0xBB, 0xCC]);
        let encoded = frame.encode();

        // Feed first half.
        let mid = encoded.len() / 2;
        transport.feed_data(&encoded[..mid]);
        let result = transport.try_decode_frame().unwrap();
        assert!(result.is_none()); // Not enough data yet.

        // Feed second half.
        transport.feed_data(&encoded[mid..]);
        let decoded = transport.try_decode_frame().unwrap();
        assert!(decoded.is_some());
        assert_eq!(decoded.unwrap(), frame);
    }

    #[test]
    fn feed_data_two_frames() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());

        let f1 = Frame::new(FrameType::Init, 0x00, Vec::new());
        let f2 = Frame::new(FrameType::Data, 0x01, vec![42]);

        let mut combined = f1.encode();
        combined.extend_from_slice(&f2.encode());

        transport.feed_data(&combined);

        let d1 = transport.try_decode_frame().unwrap().unwrap();
        assert_eq!(d1, f1);

        let d2 = transport.try_decode_frame().unwrap().unwrap();
        assert_eq!(d2, f2);

        // No more frames.
        assert!(transport.try_decode_frame().unwrap().is_none());
    }

    // === Config defaults ===

    #[test]
    fn config_defaults() {
        let config = TransportConfig::default_config();
        assert_eq!(config.max_frame_size, 65536);
        assert_eq!(config.keepalive_interval_ms, 10000);
    }

    #[test]
    fn config_default_trait() {
        let config = TransportConfig::default();
        assert_eq!(config, TransportConfig::default_config());
    }

    // === Peer ID extraction ===

    #[test]
    fn peer_id_extracted_from_init_ack() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());
        let _ = transport.initiate();

        let peer_id: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut payload = Vec::new();
        payload.extend_from_slice(&peer_id.to_be_bytes());

        let init_ack = Frame::new(FrameType::InitAck, 0x00, payload);
        transport.handle_frame(&init_ack).unwrap();

        assert_eq!(transport.peer_id(), Some(0xCAFE_BABE_DEAD_BEEF));
    }

    #[test]
    fn peer_id_none_when_no_payload() {
        let mut transport = TcpTransport::new(TransportConfig::default_config());
        let _ = transport.initiate();

        let init_ack = Frame::new(FrameType::InitAck, 0x00, Vec::new());
        transport.handle_frame(&init_ack).unwrap();

        assert_eq!(transport.peer_id(), None);
    }
}
