// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Zenoh session transport adapter over QUIC.
//!
//! Provides a `quic://` locator scheme for Zenoh session transport,
//! mapping Zenoh sessions to QUIC connections and Zenoh key expressions
//! to QUIC streams.
//!
//! Architecture:
//! - Each Zenoh session maps to one QUIC connection
//! - Control messages (scout, open, close, keepalive) use stream 0
//! - Each pub/sub key expression gets its own bidirectional stream
//! - QoS maps to QUIC priority (not yet implemented in base QUIC)

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use crate::NetError;
use super::endpoint::{QuicEndpoint, ConnectionHandle};
use super::connection::TransportParameters;

/// Maximum key expression length.
const MAX_KEY_EXPR_LEN: usize = 256;

/// Zenoh-over-QUIC message types (control channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ZenohMsgType {
    /// Scout: discover routers/peers.
    Scout = 0x01,
    /// Hello: response to scout.
    Hello = 0x02,
    /// Open session.
    Open = 0x03,
    /// Accept session.
    Accept = 0x04,
    /// Close session.
    Close = 0x05,
    /// Keepalive ping.
    KeepAlive = 0x06,
    /// Declare (subscribe, publisher, queryable).
    Declare = 0x07,
    /// Data message.
    Data = 0x08,
    /// Query.
    Query = 0x09,
    /// Reply.
    Reply = 0x0A,
}

impl ZenohMsgType {
    /// Parse from byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Scout),
            0x02 => Some(Self::Hello),
            0x03 => Some(Self::Open),
            0x04 => Some(Self::Accept),
            0x05 => Some(Self::Close),
            0x06 => Some(Self::KeepAlive),
            0x07 => Some(Self::Declare),
            0x08 => Some(Self::Data),
            0x09 => Some(Self::Query),
            0x0A => Some(Self::Reply),
            _ => None,
        }
    }
}

/// Zenoh locator parsed from `quic://<host>:<port>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuicLocator {
    /// Host address (IP or hostname).
    pub host: Vec<u8>,
    /// Port number.
    pub port: u16,
}

impl QuicLocator {
    /// Parse a `quic://<host>:<port>` locator.
    pub fn parse(locator: &[u8]) -> Result<Self, NetError> {
        // Expect: "quic://<host>:<port>"
        let prefix = b"quic://";
        if locator.len() < prefix.len() + 3 {
            return Err(NetError::InvalidAddress);
        }
        if &locator[..prefix.len()] != prefix {
            return Err(NetError::InvalidAddress);
        }

        let rest = &locator[prefix.len()..];

        // Find the last ':' for port separator
        let colon_pos = rest.iter().rposition(|&b| b == b':')
            .ok_or(NetError::InvalidAddress)?;

        if colon_pos == 0 || colon_pos >= rest.len() - 1 {
            return Err(NetError::InvalidAddress);
        }

        let host = Vec::from(&rest[..colon_pos]);

        // Parse port
        let port_bytes = &rest[colon_pos + 1..];
        let mut port: u16 = 0;
        for &b in port_bytes {
            if !b.is_ascii_digit() {
                return Err(NetError::InvalidAddress);
            }
            port = port.checked_mul(10)
                .and_then(|p| p.checked_add((b - b'0') as u16))
                .ok_or(NetError::InvalidAddress)?;
        }

        if port == 0 {
            return Err(NetError::InvalidAddress);
        }

        Ok(QuicLocator { host, port })
    }
}

/// Stream assignment for a key expression.
#[derive(Debug, Clone)]
struct KeyExprStream {
    /// The key expression.
    key_expr: Vec<u8>,
    /// QUIC stream ID assigned to this key expression.
    stream_id: u64,
}

/// Zenoh session state over a QUIC connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session not yet established.
    Init,
    /// Scout sent, waiting for Hello.
    Scouting,
    /// Open sent, waiting for Accept.
    Opening,
    /// Session established.
    Active,
    /// Session closing.
    Closing,
    /// Session closed.
    Closed,
}

/// Zenoh-over-QUIC transport adapter.
///
/// Maps Zenoh sessions to QUIC connections and provides the session
/// transport interface for SmallAIOS Zenoh IPC.
#[derive(Debug)]
pub struct ZenohQuicTransport {
    /// Underlying QUIC endpoint.
    endpoint: QuicEndpoint,
    /// Session states indexed by connection handle.
    sessions: BTreeMap<ConnectionHandle, SessionState>,
    /// Key expression to stream ID mapping per connection.
    key_streams: BTreeMap<ConnectionHandle, Vec<KeyExprStream>>,
}

impl ZenohQuicTransport {
    /// Create a new Zenoh-QUIC transport in client mode.
    pub fn new_client(params: TransportParameters) -> Self {
        Self {
            endpoint: QuicEndpoint::client(params),
            sessions: BTreeMap::new(),
            key_streams: BTreeMap::new(),
        }
    }

    /// Create a new Zenoh-QUIC transport in server (router) mode.
    pub fn new_server(params: TransportParameters) -> Self {
        Self {
            endpoint: QuicEndpoint::server(params),
            sessions: BTreeMap::new(),
            key_streams: BTreeMap::new(),
        }
    }

    /// Open a session to a peer via a `quic://` locator.
    pub fn open_session(
        &mut self,
        locator: &[u8],
        now_us: u64,
    ) -> Result<ConnectionHandle, NetError> {
        let _loc = QuicLocator::parse(locator)?;

        // Use the parsed locator as the remote address
        let remote_addr = locator;
        let local_addr = b"0.0.0.0:0";

        let handle = self.endpoint.connect(local_addr, remote_addr, now_us)?;
        self.sessions.insert(handle, SessionState::Init);
        self.key_streams.insert(handle, Vec::new());

        Ok(handle)
    }

    /// Get the session state for a connection.
    pub fn session_state(&self, handle: ConnectionHandle) -> Option<SessionState> {
        self.sessions.get(&handle).copied()
    }

    /// Transition session state after handshake completes.
    pub fn on_handshake_complete(&mut self, handle: ConnectionHandle) {
        if let Some(state) = self.sessions.get_mut(&handle) {
            *state = SessionState::Opening;
        }
    }

    /// Mark session as active (after Open/Accept exchange).
    pub fn on_session_established(&mut self, handle: ConnectionHandle) {
        if let Some(state) = self.sessions.get_mut(&handle) {
            *state = SessionState::Active;
        }
    }

    /// Assign a QUIC stream to a key expression.
    pub fn assign_stream(
        &mut self,
        handle: ConnectionHandle,
        key_expr: &[u8],
        stream_id: u64,
    ) -> Result<(), NetError> {
        if key_expr.len() > MAX_KEY_EXPR_LEN {
            return Err(NetError::InvalidAddress);
        }

        let streams = self.key_streams.get_mut(&handle)
            .ok_or(NetError::NotFound)?;

        // Check for duplicate
        if streams.iter().any(|s| s.key_expr == key_expr) {
            return Ok(()); // Already assigned
        }

        streams.push(KeyExprStream {
            key_expr: Vec::from(key_expr),
            stream_id,
        });
        Ok(())
    }

    /// Get the stream ID for a key expression.
    pub fn stream_for_key(
        &self,
        handle: ConnectionHandle,
        key_expr: &[u8],
    ) -> Option<u64> {
        self.key_streams.get(&handle)?
            .iter()
            .find(|s| s.key_expr == key_expr)
            .map(|s| s.stream_id)
    }

    /// Encode a control message (type + payload).
    pub fn encode_control_msg(
        msg_type: ZenohMsgType,
        payload: &[u8],
        buf: &mut [u8],
    ) -> Result<usize, NetError> {
        let total = 1 + 2 + payload.len(); // type(1) + length(2) + payload
        if buf.len() < total {
            return Err(NetError::BufferTooSmall);
        }
        buf[0] = msg_type as u8;
        let len = payload.len() as u16;
        buf[1..3].copy_from_slice(&len.to_be_bytes());
        buf[3..3 + payload.len()].copy_from_slice(payload);
        Ok(total)
    }

    /// Decode a control message.
    pub fn decode_control_msg(data: &[u8]) -> Result<(ZenohMsgType, &[u8]), NetError> {
        if data.len() < 3 {
            return Err(NetError::PacketTooShort);
        }
        let msg_type = ZenohMsgType::from_byte(data[0])
            .ok_or(NetError::InvalidProtocol)?;
        let len = u16::from_be_bytes([data[1], data[2]]) as usize;
        if data.len() < 3 + len {
            return Err(NetError::PacketTooShort);
        }
        Ok((msg_type, &data[3..3 + len]))
    }

    /// Close a session.
    pub fn close_session(
        &mut self,
        handle: ConnectionHandle,
        now_us: u64,
    ) -> Result<(), NetError> {
        if let Some(state) = self.sessions.get_mut(&handle) {
            *state = SessionState::Closing;
        }
        self.endpoint.close(handle, 0, b"zenoh session close", now_us)?;
        if let Some(state) = self.sessions.get_mut(&handle) {
            *state = SessionState::Closed;
        }
        Ok(())
    }

    /// Number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.iter()
            .filter(|(_, s)| **s == SessionState::Active || **s == SessionState::Opening)
            .count()
    }

    /// Total sessions (all states).
    pub fn total_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Access the underlying endpoint.
    pub fn endpoint(&self) -> &QuicEndpoint {
        &self.endpoint
    }

    /// Access the underlying endpoint mutably.
    pub fn endpoint_mut(&mut self) -> &mut QuicEndpoint {
        &mut self.endpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn default_params() -> TransportParameters {
        TransportParameters::default()
    }

    #[test]
    fn test_parse_locator_valid() {
        let loc = QuicLocator::parse(b"quic://192.168.1.1:4433").unwrap();
        assert_eq!(loc.host, b"192.168.1.1");
        assert_eq!(loc.port, 4433);
    }

    #[test]
    fn test_parse_locator_ipv6() {
        let loc = QuicLocator::parse(b"quic://[::1]:443").unwrap();
        assert_eq!(loc.host, b"[::1]");
        assert_eq!(loc.port, 443);
    }

    #[test]
    fn test_parse_locator_hostname() {
        let loc = QuicLocator::parse(b"quic://example.com:8443").unwrap();
        assert_eq!(loc.host, b"example.com");
        assert_eq!(loc.port, 8443);
    }

    #[test]
    fn test_parse_locator_missing_prefix() {
        assert_eq!(
            QuicLocator::parse(b"tcp://host:80").err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_parse_locator_missing_port() {
        assert_eq!(
            QuicLocator::parse(b"quic://host").err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_parse_locator_zero_port() {
        assert_eq!(
            QuicLocator::parse(b"quic://host:0").err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_parse_locator_too_short() {
        assert_eq!(
            QuicLocator::parse(b"quic://").err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_zenoh_msg_type_roundtrip() {
        for b in 0x01..=0x0Au8 {
            let mt = ZenohMsgType::from_byte(b).unwrap();
            assert_eq!(mt as u8, b);
        }
        assert!(ZenohMsgType::from_byte(0x00).is_none());
        assert!(ZenohMsgType::from_byte(0xFF).is_none());
    }

    #[test]
    fn test_control_msg_encode_decode() {
        let payload = b"hello zenoh";
        let mut buf = [0u8; 32];
        let len = ZenohQuicTransport::encode_control_msg(
            ZenohMsgType::Open,
            payload,
            &mut buf,
        ).unwrap();
        assert_eq!(len, 3 + payload.len());

        let (msg_type, decoded_payload) = ZenohQuicTransport::decode_control_msg(&buf[..len]).unwrap();
        assert_eq!(msg_type, ZenohMsgType::Open);
        assert_eq!(decoded_payload, payload);
    }

    #[test]
    fn test_control_msg_encode_buffer_too_small() {
        let mut buf = [0u8; 2];
        assert_eq!(
            ZenohQuicTransport::encode_control_msg(ZenohMsgType::KeepAlive, b"x", &mut buf).err(),
            Some(NetError::BufferTooSmall)
        );
    }

    #[test]
    fn test_control_msg_decode_too_short() {
        assert_eq!(
            ZenohQuicTransport::decode_control_msg(&[0x01, 0x00]).err(),
            Some(NetError::PacketTooShort)
        );
    }

    #[test]
    fn test_control_msg_decode_unknown_type() {
        assert_eq!(
            ZenohQuicTransport::decode_control_msg(&[0xFF, 0x00, 0x00]).err(),
            Some(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn test_new_client_transport() {
        let t = ZenohQuicTransport::new_client(default_params());
        assert!(!t.endpoint().is_server());
        assert_eq!(t.total_sessions(), 0);
        assert_eq!(t.session_count(), 0);
    }

    #[test]
    fn test_new_server_transport() {
        let t = ZenohQuicTransport::new_server(default_params());
        assert!(t.endpoint().is_server());
    }

    #[test]
    fn test_open_session() {
        let mut t = ZenohQuicTransport::new_client(default_params());
        let handle = t.open_session(b"quic://10.0.0.1:4433", 1000).unwrap();
        assert_eq!(t.total_sessions(), 1);
        assert_eq!(t.session_state(handle), Some(SessionState::Init));
    }

    #[test]
    fn test_open_session_invalid_locator() {
        let mut t = ZenohQuicTransport::new_client(default_params());
        assert_eq!(
            t.open_session(b"invalid", 1000).err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_session_state_transitions() {
        let mut t = ZenohQuicTransport::new_client(default_params());
        let handle = t.open_session(b"quic://10.0.0.1:4433", 1000).unwrap();

        t.on_handshake_complete(handle);
        assert_eq!(t.session_state(handle), Some(SessionState::Opening));

        t.on_session_established(handle);
        assert_eq!(t.session_state(handle), Some(SessionState::Active));
        assert_eq!(t.session_count(), 1);
    }

    #[test]
    fn test_close_session() {
        let mut t = ZenohQuicTransport::new_client(default_params());
        let handle = t.open_session(b"quic://10.0.0.1:4433", 1000).unwrap();
        t.close_session(handle, 2000).unwrap();
        assert_eq!(t.session_state(handle), Some(SessionState::Closed));
    }

    #[test]
    fn test_assign_and_lookup_stream() {
        let mut t = ZenohQuicTransport::new_client(default_params());
        let handle = t.open_session(b"quic://10.0.0.1:4433", 1000).unwrap();

        t.assign_stream(handle, b"demo/sensor/temp", 4).unwrap();
        assert_eq!(t.stream_for_key(handle, b"demo/sensor/temp"), Some(4));
        assert_eq!(t.stream_for_key(handle, b"demo/sensor/other"), None);
    }

    #[test]
    fn test_assign_duplicate_stream() {
        let mut t = ZenohQuicTransport::new_client(default_params());
        let handle = t.open_session(b"quic://10.0.0.1:4433", 1000).unwrap();

        t.assign_stream(handle, b"key", 4).unwrap();
        // Assigning again should be a no-op
        t.assign_stream(handle, b"key", 8).unwrap();
        assert_eq!(t.stream_for_key(handle, b"key"), Some(4));
    }

    #[test]
    fn test_assign_stream_unknown_handle() {
        let mut t = ZenohQuicTransport::new_client(default_params());
        assert_eq!(
            t.assign_stream(999, b"key", 4).err(),
            Some(NetError::NotFound)
        );
    }

    #[test]
    fn test_assign_stream_key_too_long() {
        let mut t = ZenohQuicTransport::new_client(default_params());
        let handle = t.open_session(b"quic://10.0.0.1:4433", 1000).unwrap();
        let long_key = vec![b'a'; MAX_KEY_EXPR_LEN + 1];
        assert_eq!(
            t.assign_stream(handle, &long_key, 4).err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_control_msg_keepalive() {
        let mut buf = [0u8; 8];
        let len = ZenohQuicTransport::encode_control_msg(
            ZenohMsgType::KeepAlive,
            &[],
            &mut buf,
        ).unwrap();
        assert_eq!(len, 3); // type + 2-byte length + 0 payload

        let (mt, payload) = ZenohQuicTransport::decode_control_msg(&buf[..len]).unwrap();
        assert_eq!(mt, ZenohMsgType::KeepAlive);
        assert!(payload.is_empty());
    }

    #[test]
    fn test_session_state_variants() {
        assert_ne!(SessionState::Init, SessionState::Active);
        assert_ne!(SessionState::Opening, SessionState::Closing);
        assert_ne!(SessionState::Scouting, SessionState::Closed);
    }
}
