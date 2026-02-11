// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC connection state machine (RFC 9000 Section 10).
//!
//! Manages connection lifecycle: Idle -> Handshaking -> Connected -> Draining -> Closed.
//! Tracks per-space state, connection IDs, transport parameters, and migration.

use alloc::vec::Vec;
use crate::NetError;
use super::connid::{ConnectionIdManager, DEFAULT_CID_LEN};
use super::spaces::{PacketNumberSpace, PacketNumberSpaceState};

/// QUIC connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// No connection established yet.
    Idle,
    /// TLS handshake in progress.
    Handshaking,
    /// Connection established, ready for data.
    Connected,
    /// Connection is being closed, draining period.
    Draining,
    /// Connection is fully closed.
    Closed,
}

/// QUIC connection role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionRole {
    /// Client initiated the connection.
    Client,
    /// Server accepted the connection.
    Server,
}

/// Transport parameters (RFC 9000 Section 18).
#[derive(Debug, Clone)]
pub struct TransportParameters {
    /// Maximum idle timeout in milliseconds (0 = disabled).
    pub max_idle_timeout_ms: u64,
    /// Maximum UDP payload size the peer is willing to receive.
    pub max_udp_payload_size: u64,
    /// Initial maximum data the peer can send (connection level).
    pub initial_max_data: u64,
    /// Initial maximum data on a locally-initiated bidi stream.
    pub initial_max_stream_data_bidi_local: u64,
    /// Initial maximum data on a remotely-initiated bidi stream.
    pub initial_max_stream_data_bidi_remote: u64,
    /// Initial maximum data on a unidirectional stream.
    pub initial_max_stream_data_uni: u64,
    /// Initial maximum bidirectional streams the peer may initiate.
    pub initial_max_streams_bidi: u64,
    /// Initial maximum unidirectional streams the peer may initiate.
    pub initial_max_streams_uni: u64,
    /// ACK delay exponent.
    pub ack_delay_exponent: u64,
    /// Maximum ACK delay in milliseconds.
    pub max_ack_delay_ms: u64,
    /// Active connection ID limit.
    pub active_connection_id_limit: u64,
}

impl Default for TransportParameters {
    fn default() -> Self {
        Self {
            max_idle_timeout_ms: 30_000,
            max_udp_payload_size: 1472,
            initial_max_data: 1_048_576,            // 1 MB
            initial_max_stream_data_bidi_local: 262_144,  // 256 KB
            initial_max_stream_data_bidi_remote: 262_144,
            initial_max_stream_data_uni: 262_144,
            initial_max_streams_bidi: 100,
            initial_max_streams_uni: 100,
            ack_delay_exponent: 3,
            max_ack_delay_ms: 25,
            active_connection_id_limit: 4,
        }
    }
}

/// Close reason for a QUIC connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloseReason {
    /// Application-level close with error code and reason.
    Application { error_code: u64, reason: Vec<u8> },
    /// Transport-level close with error code.
    Transport { error_code: u64, frame_type: u64, reason: Vec<u8> },
    /// Idle timeout.
    IdleTimeout,
    /// Stateless reset received.
    StatelessReset,
}

/// QUIC connection.
#[derive(Debug)]
pub struct Connection {
    /// Connection state.
    state: ConnectionState,
    /// Our role (client or server).
    role: ConnectionRole,
    /// Connection ID manager.
    cid_manager: ConnectionIdManager,
    /// Per-space packet number state.
    initial_space: PacketNumberSpaceState,
    handshake_space: PacketNumberSpaceState,
    app_space: PacketNumberSpaceState,
    /// Local transport parameters.
    local_params: TransportParameters,
    /// Remote transport parameters (received during handshake).
    remote_params: Option<TransportParameters>,
    /// Close reason (set when entering Draining/Closed).
    close_reason: Option<CloseReason>,
    /// Last activity time (microseconds).
    last_activity_us: u64,
    /// Handshake confirmed (both sides have acknowledged handshake).
    handshake_confirmed: bool,
    /// Whether we have sent the handshake done signal.
    handshake_done_sent: bool,
}

impl Connection {
    /// Create a new QUIC connection.
    pub fn new(role: ConnectionRole, params: TransportParameters) -> Result<Self, NetError> {
        Ok(Self {
            state: ConnectionState::Idle,
            role,
            cid_manager: ConnectionIdManager::new(DEFAULT_CID_LEN)?,
            initial_space: PacketNumberSpaceState::new(PacketNumberSpace::Initial),
            handshake_space: PacketNumberSpaceState::new(PacketNumberSpace::Handshake),
            app_space: PacketNumberSpaceState::new(PacketNumberSpace::ApplicationData),
            local_params: params,
            remote_params: None,
            close_reason: None,
            last_activity_us: 0,
            handshake_confirmed: false,
            handshake_done_sent: false,
        })
    }

    /// Get the current connection state.
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Get the connection role.
    pub fn role(&self) -> ConnectionRole {
        self.role
    }

    /// Get local transport parameters.
    pub fn local_params(&self) -> &TransportParameters {
        &self.local_params
    }

    /// Get remote transport parameters (if received).
    pub fn remote_params(&self) -> Option<&TransportParameters> {
        self.remote_params.as_ref()
    }

    /// Set remote transport parameters (during handshake).
    pub fn set_remote_params(&mut self, params: TransportParameters) {
        self.remote_params = Some(params);
    }

    /// Get the CID manager.
    pub fn cid_manager(&self) -> &ConnectionIdManager {
        &self.cid_manager
    }

    /// Get mutable CID manager.
    pub fn cid_manager_mut(&mut self) -> &mut ConnectionIdManager {
        &mut self.cid_manager
    }

    /// Start the handshake (Idle -> Handshaking).
    pub fn start_handshake(&mut self, now_us: u64) -> Result<(), NetError> {
        if self.state != ConnectionState::Idle {
            return Err(NetError::InvalidProtocol);
        }
        self.state = ConnectionState::Handshaking;
        self.last_activity_us = now_us;
        Ok(())
    }

    /// Complete the handshake (Handshaking -> Connected).
    pub fn complete_handshake(&mut self, now_us: u64) -> Result<(), NetError> {
        if self.state != ConnectionState::Handshaking {
            return Err(NetError::InvalidProtocol);
        }
        self.state = ConnectionState::Connected;
        self.handshake_confirmed = true;
        self.last_activity_us = now_us;

        // Discard Initial and Handshake keys per RFC 9001 Section 4.9
        self.initial_space.discard();
        self.handshake_space.discard();

        Ok(())
    }

    /// Check if the handshake is confirmed.
    pub fn is_handshake_confirmed(&self) -> bool {
        self.handshake_confirmed
    }

    /// Mark handshake done as sent (server side).
    pub fn mark_handshake_done_sent(&mut self) {
        self.handshake_done_sent = true;
    }

    /// Whether handshake done has been sent.
    pub fn handshake_done_sent(&self) -> bool {
        self.handshake_done_sent
    }

    /// Initiate connection close (Connected -> Draining).
    pub fn close(&mut self, reason: CloseReason, now_us: u64) -> Result<(), NetError> {
        match self.state {
            ConnectionState::Connected | ConnectionState::Handshaking => {
                self.state = ConnectionState::Draining;
                self.close_reason = Some(reason);
                self.last_activity_us = now_us;
                Ok(())
            }
            ConnectionState::Draining | ConnectionState::Closed => Ok(()),
            ConnectionState::Idle => Err(NetError::InvalidProtocol),
        }
    }

    /// Complete the draining period (Draining -> Closed).
    pub fn finish_draining(&mut self) -> Result<(), NetError> {
        if self.state != ConnectionState::Draining {
            return Err(NetError::InvalidProtocol);
        }
        self.state = ConnectionState::Closed;
        Ok(())
    }

    /// Get the close reason.
    pub fn close_reason(&self) -> Option<&CloseReason> {
        self.close_reason.as_ref()
    }

    /// Get the packet number space state for a given space.
    pub fn space(&self, space: PacketNumberSpace) -> &PacketNumberSpaceState {
        match space {
            PacketNumberSpace::Initial => &self.initial_space,
            PacketNumberSpace::Handshake => &self.handshake_space,
            PacketNumberSpace::ApplicationData => &self.app_space,
        }
    }

    /// Get mutable packet number space state.
    pub fn space_mut(&mut self, space: PacketNumberSpace) -> &mut PacketNumberSpaceState {
        match space {
            PacketNumberSpace::Initial => &mut self.initial_space,
            PacketNumberSpace::Handshake => &mut self.handshake_space,
            PacketNumberSpace::ApplicationData => &mut self.app_space,
        }
    }

    /// Record activity (resets idle timeout).
    pub fn record_activity(&mut self, now_us: u64) {
        self.last_activity_us = now_us;
    }

    /// Check if the connection has timed out due to idle.
    pub fn is_idle_timed_out(&self, now_us: u64) -> bool {
        if self.local_params.max_idle_timeout_ms == 0 {
            return false;
        }
        let timeout_us = self.local_params.max_idle_timeout_ms * 1000;
        now_us.saturating_sub(self.last_activity_us) >= timeout_us
    }

    /// Get the last activity timestamp.
    pub fn last_activity_us(&self) -> u64 {
        self.last_activity_us
    }

    /// Check if connection is usable for sending data.
    pub fn is_established(&self) -> bool {
        self.state == ConnectionState::Connected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client_conn() -> Connection {
        Connection::new(ConnectionRole::Client, TransportParameters::default()).unwrap()
    }

    fn server_conn() -> Connection {
        Connection::new(ConnectionRole::Server, TransportParameters::default()).unwrap()
    }

    #[test]
    fn test_new_connection() {
        let conn = client_conn();
        assert_eq!(conn.state(), ConnectionState::Idle);
        assert_eq!(conn.role(), ConnectionRole::Client);
        assert!(!conn.is_established());
    }

    #[test]
    fn test_handshake_lifecycle() {
        let mut conn = client_conn();
        conn.start_handshake(1000).unwrap();
        assert_eq!(conn.state(), ConnectionState::Handshaking);
        assert!(!conn.is_established());

        conn.complete_handshake(2000).unwrap();
        assert_eq!(conn.state(), ConnectionState::Connected);
        assert!(conn.is_established());
        assert!(conn.is_handshake_confirmed());
    }

    #[test]
    fn test_close_lifecycle() {
        let mut conn = client_conn();
        conn.start_handshake(1000).unwrap();
        conn.complete_handshake(2000).unwrap();

        let reason = CloseReason::Application {
            error_code: 0,
            reason: Vec::new(),
        };
        conn.close(reason.clone(), 3000).unwrap();
        assert_eq!(conn.state(), ConnectionState::Draining);
        assert_eq!(conn.close_reason(), Some(&reason));

        conn.finish_draining().unwrap();
        assert_eq!(conn.state(), ConnectionState::Closed);
    }

    #[test]
    fn test_invalid_state_transitions() {
        let mut conn = client_conn();
        // Can't complete handshake from Idle
        assert_eq!(
            conn.complete_handshake(1000).err(),
            Some(NetError::InvalidProtocol)
        );
        // Can't close from Idle
        assert_eq!(
            conn.close(CloseReason::IdleTimeout, 1000).err(),
            Some(NetError::InvalidProtocol)
        );
        // Can't finish draining from Idle
        assert_eq!(
            conn.finish_draining().err(),
            Some(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn test_double_start_handshake() {
        let mut conn = client_conn();
        conn.start_handshake(1000).unwrap();
        assert_eq!(
            conn.start_handshake(2000).err(),
            Some(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn test_idle_timeout() {
        let conn = client_conn();
        // Default timeout is 30,000 ms = 30,000,000 us
        assert!(!conn.is_idle_timed_out(29_999_999));
        assert!(conn.is_idle_timed_out(30_000_000));
    }

    #[test]
    fn test_idle_timeout_disabled() {
        let params = TransportParameters {
            max_idle_timeout_ms: 0,
            ..Default::default()
        };
        let conn = Connection::new(ConnectionRole::Client, params).unwrap();
        assert!(!conn.is_idle_timed_out(u64::MAX));
    }

    #[test]
    fn test_record_activity_resets_timeout() {
        let mut conn = client_conn();
        conn.record_activity(10_000_000);
        assert!(!conn.is_idle_timed_out(39_999_999));
        assert!(conn.is_idle_timed_out(40_000_000));
    }

    #[test]
    fn test_transport_parameters_default() {
        let params = TransportParameters::default();
        assert_eq!(params.max_idle_timeout_ms, 30_000);
        assert_eq!(params.initial_max_streams_bidi, 100);
        assert_eq!(params.initial_max_streams_uni, 100);
        assert_eq!(params.initial_max_data, 1_048_576);
    }

    #[test]
    fn test_remote_params() {
        let mut conn = client_conn();
        assert!(conn.remote_params().is_none());
        conn.set_remote_params(TransportParameters::default());
        assert!(conn.remote_params().is_some());
    }

    #[test]
    fn test_space_access() {
        let mut conn = client_conn();
        let pn = conn.space_mut(PacketNumberSpace::Initial).allocate_pn();
        assert_eq!(pn, 0);
        assert_eq!(conn.space(PacketNumberSpace::Initial).next_pn(), 1);
    }

    #[test]
    fn test_handshake_discards_spaces() {
        let mut conn = client_conn();
        conn.start_handshake(1000).unwrap();

        // Send in initial space
        conn.space_mut(PacketNumberSpace::Initial).allocate_pn();
        conn.space_mut(PacketNumberSpace::Initial).on_packet_received(0, 1000);

        conn.complete_handshake(2000).unwrap();

        // Initial space should be discarded
        assert!(!conn.space(PacketNumberSpace::Initial).ack_needed());
    }

    #[test]
    fn test_server_connection() {
        let conn = server_conn();
        assert_eq!(conn.role(), ConnectionRole::Server);
    }

    #[test]
    fn test_close_from_handshaking() {
        let mut conn = client_conn();
        conn.start_handshake(1000).unwrap();
        conn.close(
            CloseReason::Transport {
                error_code: 0x01,
                frame_type: 0,
                reason: Vec::new(),
            },
            2000,
        )
        .unwrap();
        assert_eq!(conn.state(), ConnectionState::Draining);
    }

    #[test]
    fn test_close_idempotent_in_draining() {
        let mut conn = client_conn();
        conn.start_handshake(1000).unwrap();
        conn.complete_handshake(2000).unwrap();
        conn.close(CloseReason::IdleTimeout, 3000).unwrap();
        // Closing again in draining should be ok
        conn.close(CloseReason::IdleTimeout, 4000).unwrap();
    }

    #[test]
    fn test_handshake_done() {
        let mut conn = server_conn();
        assert!(!conn.handshake_done_sent());
        conn.mark_handshake_done_sent();
        assert!(conn.handshake_done_sent());
    }
}
