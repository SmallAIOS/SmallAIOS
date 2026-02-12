// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC endpoint API (server and client).
//!
//! Provides a standalone endpoint that manages multiple connections,
//! dispatches incoming packets to the correct connection, and handles
//! connection setup including version negotiation and retry.

use super::congestion::CongestionController;
use super::connection::{Connection, ConnectionRole, ConnectionState, TransportParameters};
use super::flow::ConnectionFlowControl;
use super::key_update::KeyUpdateManager;
use super::migration::MigrationManager;
use super::packet::{is_long_header, LongHeader, PacketType};
use super::retry::{ReplayFilter, SessionTicketStore};
use super::stream::StreamManager;
use crate::NetError;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Maximum number of concurrent connections per endpoint.
const MAX_CONNECTIONS: usize = 256;

/// Connection handle — opaque identifier used by the API consumer.
pub type ConnectionHandle = u64;

/// Events produced by the endpoint for the application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointEvent {
    /// A new connection was accepted (server) or established (client).
    Connected { handle: ConnectionHandle },
    /// Data is available to read on a stream.
    StreamReadable {
        handle: ConnectionHandle,
        stream_id: u64,
    },
    /// A stream is writable (flow control window opened).
    StreamWritable {
        handle: ConnectionHandle,
        stream_id: u64,
    },
    /// A stream was finished by the peer (FIN received).
    StreamFinished {
        handle: ConnectionHandle,
        stream_id: u64,
    },
    /// Connection was closed.
    ConnectionClosed { handle: ConnectionHandle },
    /// Handshake completed.
    HandshakeComplete { handle: ConnectionHandle },
}

/// Per-connection state managed by the endpoint.
#[derive(Debug)]
struct EndpointConnection {
    /// QUIC connection state machine.
    conn: Connection,
    /// Stream manager.
    streams: StreamManager,
    /// Connection-level flow control (used during packet processing).
    #[allow(dead_code)]
    flow: ConnectionFlowControl,
    /// Congestion controller (used during packet processing).
    #[allow(dead_code)]
    congestion: CongestionController,
    /// Migration manager.
    migration: MigrationManager,
    /// Key update manager (used during key rotation).
    #[allow(dead_code)]
    key_update: KeyUpdateManager,
    /// Remote address (opaque bytes: IP + port).
    remote_addr: Vec<u8>,
    /// Local address (opaque bytes).
    local_addr: Vec<u8>,
}

/// QUIC endpoint configuration.
#[derive(Debug, Clone)]
pub struct EndpointConfig {
    /// Transport parameters for new connections.
    pub transport_params: TransportParameters,
    /// Whether this endpoint acts as a server.
    pub is_server: bool,
    /// Maximum concurrent connections.
    pub max_connections: usize,
    /// Enable retry tokens (server only).
    pub retry_required: bool,
    /// Enable 0-RTT (requires session ticket store).
    pub enable_0rtt: bool,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            transport_params: TransportParameters::default(),
            is_server: false,
            max_connections: MAX_CONNECTIONS,
            retry_required: false,
            enable_0rtt: false,
        }
    }
}

/// QUIC endpoint — manages multiple connections.
///
/// The endpoint is the top-level API for QUIC. It:
/// - Accepts incoming connections (server mode)
/// - Initiates outgoing connections (client mode)
/// - Dispatches packets to the correct connection
/// - Produces events for the application layer
#[derive(Debug)]
pub struct QuicEndpoint {
    /// Configuration.
    config: EndpointConfig,
    /// Active connections indexed by handle.
    connections: BTreeMap<ConnectionHandle, EndpointConnection>,
    /// Map from DCID bytes to connection handle (for packet dispatch).
    cid_to_handle: BTreeMap<Vec<u8>, ConnectionHandle>,
    /// Next connection handle to assign.
    next_handle: ConnectionHandle,
    /// Pending events for the application.
    events: Vec<EndpointEvent>,
    /// Session ticket store (for 0-RTT).
    ticket_store: SessionTicketStore,
    /// Replay filter (for 0-RTT server-side).
    #[allow(dead_code)]
    replay_filter: ReplayFilter,
}

impl QuicEndpoint {
    /// Create a new QUIC endpoint.
    pub fn new(config: EndpointConfig) -> Self {
        Self {
            config,
            connections: BTreeMap::new(),
            cid_to_handle: BTreeMap::new(),
            next_handle: 1,
            events: Vec::new(),
            ticket_store: SessionTicketStore::new(),
            replay_filter: ReplayFilter::new(60_000_000), // 60s window
        }
    }

    /// Create a server endpoint with default config.
    pub fn server(params: TransportParameters) -> Self {
        Self::new(EndpointConfig {
            transport_params: params,
            is_server: true,
            ..Default::default()
        })
    }

    /// Create a client endpoint with default config.
    pub fn client(params: TransportParameters) -> Self {
        Self::new(EndpointConfig {
            transport_params: params,
            is_server: false,
            ..Default::default()
        })
    }

    /// Initiate a connection to a remote address (client mode).
    ///
    /// Returns a connection handle that can be used to interact with
    /// the connection once established.
    pub fn connect(
        &mut self,
        local_addr: &[u8],
        remote_addr: &[u8],
        now_us: u64,
    ) -> Result<ConnectionHandle, NetError> {
        if self.config.is_server {
            return Err(NetError::InvalidProtocol);
        }
        if self.connections.len() >= self.config.max_connections {
            return Err(NetError::TableFull);
        }

        let mut conn =
            Connection::new(ConnectionRole::Client, self.config.transport_params.clone())?;
        conn.start_handshake(now_us)?;

        let handle = self.next_handle;
        self.next_handle += 1;

        // Generate a local CID and register it for packet dispatch
        if let Ok(cid) = conn.cid_manager_mut().generate_local_cid() {
            self.cid_to_handle.insert(cid.id.clone(), handle);
        }

        let is_client = true;
        let params = &self.config.transport_params;

        let mut migration = MigrationManager::new();
        migration.set_initial_path(local_addr, remote_addr);

        let ec = EndpointConnection {
            conn,
            streams: StreamManager::new(
                is_client,
                params.initial_max_streams_bidi,
                params.initial_max_streams_uni,
            ),
            flow: ConnectionFlowControl::new(params.initial_max_data, params.initial_max_data),
            congestion: CongestionController::new(),
            migration,
            key_update: KeyUpdateManager::new(),
            remote_addr: Vec::from(remote_addr),
            local_addr: Vec::from(local_addr),
        };

        self.connections.insert(handle, ec);
        Ok(handle)
    }

    /// Process an incoming packet from the network.
    ///
    /// Dispatches the packet to the correct connection based on DCID,
    /// or creates a new connection for Initial packets (server mode).
    pub fn receive(
        &mut self,
        data: &[u8],
        local_addr: &[u8],
        remote_addr: &[u8],
        now_us: u64,
    ) -> Result<(), NetError> {
        if data.is_empty() {
            return Err(NetError::PacketTooShort);
        }

        if is_long_header(data[0]) {
            self.receive_long_header(data, local_addr, remote_addr, now_us)
        } else {
            self.receive_short_header(data, local_addr, remote_addr, now_us)
        }
    }

    /// Handle a long header packet (Initial, Handshake, 0-RTT, Retry).
    fn receive_long_header(
        &mut self,
        data: &[u8],
        local_addr: &[u8],
        remote_addr: &[u8],
        now_us: u64,
    ) -> Result<(), NetError> {
        let (header, _header_len) = LongHeader::decode(data)?;

        // Version negotiation
        if !header.version.is_supported() && header.version.0 != 0 {
            // We don't support this version — would send VN in a real impl
            return Err(NetError::InvalidProtocol);
        }

        // Try to find an existing connection by DCID
        if let Some(&handle) = self.cid_to_handle.get(&header.dcid) {
            if let Some(ec) = self.connections.get_mut(&handle) {
                ec.conn.record_activity(now_us);
                if header.packet_type == PacketType::Handshake {
                    // Process handshake packet
                    if ec.conn.state() == ConnectionState::Handshaking {
                        ec.conn.complete_handshake(now_us)?;
                        self.events
                            .push(EndpointEvent::HandshakeComplete { handle });
                        self.events.push(EndpointEvent::Connected { handle });
                    }
                }
            }
            return Ok(());
        }

        // No existing connection — if server and Initial, create a new one
        if self.config.is_server && header.packet_type == PacketType::Initial {
            return self.accept_new_connection(&header, local_addr, remote_addr, now_us);
        }

        // Unknown connection
        Err(NetError::NotFound)
    }

    /// Handle a short header (1-RTT) packet.
    fn receive_short_header(
        &mut self,
        data: &[u8],
        local_addr: &[u8],
        remote_addr: &[u8],
        now_us: u64,
    ) -> Result<(), NetError> {
        // We need to try matching against known DCID lengths.
        // In practice, the endpoint knows the DCID length from connection state.
        // For simplicity, try the default length.
        let dcid_len = super::connid::DEFAULT_CID_LEN;
        if data.len() < 1 + dcid_len {
            return Err(NetError::PacketTooShort);
        }

        let dcid = &data[1..1 + dcid_len];

        if let Some(&handle) = self.cid_to_handle.get(dcid) {
            if let Some(ec) = self.connections.get_mut(&handle) {
                ec.conn.record_activity(now_us);

                // Check for connection migration
                if ec.remote_addr != remote_addr || ec.local_addr != local_addr {
                    ec.migration
                        .on_packet_from_new_address(local_addr, remote_addr);
                }
            }
            return Ok(());
        }

        Err(NetError::NotFound)
    }

    /// Accept a new incoming connection (server mode).
    fn accept_new_connection(
        &mut self,
        header: &LongHeader,
        local_addr: &[u8],
        remote_addr: &[u8],
        now_us: u64,
    ) -> Result<(), NetError> {
        if self.connections.len() >= self.config.max_connections {
            return Err(NetError::TableFull);
        }

        let mut conn =
            Connection::new(ConnectionRole::Server, self.config.transport_params.clone())?;
        conn.start_handshake(now_us)?;

        let handle = self.next_handle;
        self.next_handle += 1;

        // Generate our local CID and register it
        if let Ok(cid) = conn.cid_manager_mut().generate_local_cid() {
            self.cid_to_handle.insert(cid.id.clone(), handle);
        }
        // Also register the client's chosen DCID (from the header)
        self.cid_to_handle.insert(header.dcid.clone(), handle);

        let params = &self.config.transport_params;
        let mut migration = MigrationManager::new();
        migration.set_initial_path(local_addr, remote_addr);

        let ec = EndpointConnection {
            conn,
            streams: StreamManager::new(
                false,
                params.initial_max_streams_bidi,
                params.initial_max_streams_uni,
            ),
            flow: ConnectionFlowControl::new(params.initial_max_data, params.initial_max_data),
            congestion: CongestionController::new(),
            migration,
            key_update: KeyUpdateManager::new(),
            remote_addr: Vec::from(remote_addr),
            local_addr: Vec::from(local_addr),
        };

        self.connections.insert(handle, ec);
        Ok(())
    }

    /// Open a bidirectional stream on an established connection.
    pub fn open_stream_bidi(&mut self, handle: ConnectionHandle) -> Result<u64, NetError> {
        let ec = self
            .connections
            .get_mut(&handle)
            .ok_or(NetError::NotFound)?;
        if !ec.conn.is_established() {
            return Err(NetError::InvalidProtocol);
        }
        ec.streams.open_bidi()
    }

    /// Open a unidirectional stream on an established connection.
    pub fn open_stream_uni(&mut self, handle: ConnectionHandle) -> Result<u64, NetError> {
        let ec = self
            .connections
            .get_mut(&handle)
            .ok_or(NetError::NotFound)?;
        if !ec.conn.is_established() {
            return Err(NetError::InvalidProtocol);
        }
        ec.streams.open_uni()
    }

    /// Write data to a stream.
    pub fn stream_write(
        &mut self,
        handle: ConnectionHandle,
        stream_id: u64,
        data: &[u8],
    ) -> Result<usize, NetError> {
        let ec = self
            .connections
            .get_mut(&handle)
            .ok_or(NetError::NotFound)?;
        let stream = ec.streams.get_mut(stream_id).ok_or(NetError::NotFound)?;
        stream.write(data)
    }

    /// Read data from a stream.
    pub fn stream_read(
        &mut self,
        handle: ConnectionHandle,
        stream_id: u64,
        buf: &mut [u8],
    ) -> Result<usize, NetError> {
        let ec = self
            .connections
            .get_mut(&handle)
            .ok_or(NetError::NotFound)?;
        let stream = ec.streams.get_mut(stream_id).ok_or(NetError::NotFound)?;
        stream.read(buf)
    }

    /// Finish (FIN) a stream.
    pub fn stream_finish(
        &mut self,
        handle: ConnectionHandle,
        stream_id: u64,
    ) -> Result<(), NetError> {
        let ec = self
            .connections
            .get_mut(&handle)
            .ok_or(NetError::NotFound)?;
        let stream = ec.streams.get_mut(stream_id).ok_or(NetError::NotFound)?;
        stream.finish()
    }

    /// Close a connection.
    pub fn close(
        &mut self,
        handle: ConnectionHandle,
        error_code: u64,
        reason: &[u8],
        now_us: u64,
    ) -> Result<(), NetError> {
        let ec = self
            .connections
            .get_mut(&handle)
            .ok_or(NetError::NotFound)?;
        ec.conn.close(
            super::connection::CloseReason::Application {
                error_code,
                reason: Vec::from(reason),
            },
            now_us,
        )?;
        self.events.push(EndpointEvent::ConnectionClosed { handle });
        Ok(())
    }

    /// Drain pending events for the application.
    pub fn poll_events(&mut self) -> Vec<EndpointEvent> {
        core::mem::take(&mut self.events)
    }

    /// Get the connection state for a handle.
    pub fn connection_state(&self, handle: ConnectionHandle) -> Option<ConnectionState> {
        self.connections.get(&handle).map(|ec| ec.conn.state())
    }

    /// Number of active connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Check for idle timeouts and clean up closed connections.
    pub fn handle_timeouts(&mut self, now_us: u64) {
        let mut to_close = Vec::new();
        for (&handle, ec) in &self.connections {
            if ec.conn.is_idle_timed_out(now_us) {
                to_close.push(handle);
            }
        }

        for handle in to_close {
            if let Some(ec) = self.connections.get_mut(&handle) {
                let _ = ec
                    .conn
                    .close(super::connection::CloseReason::IdleTimeout, now_us);
                self.events.push(EndpointEvent::ConnectionClosed { handle });
            }
        }
    }

    /// Remove fully closed connections.
    pub fn garbage_collect(&mut self) {
        let closed: Vec<ConnectionHandle> = self
            .connections
            .iter()
            .filter(|(_, ec)| ec.conn.state() == ConnectionState::Closed)
            .map(|(&h, _)| h)
            .collect();

        for handle in closed {
            self.connections.remove(&handle);
            // Remove all CID mappings that point to this handle
            self.cid_to_handle.retain(|_, &mut h| h != handle);
        }
    }

    /// Whether this endpoint is a server.
    pub fn is_server(&self) -> bool {
        self.config.is_server
    }

    /// Get the session ticket store (for 0-RTT).
    pub fn ticket_store(&self) -> &SessionTicketStore {
        &self.ticket_store
    }

    /// Get mutable session ticket store.
    pub fn ticket_store_mut(&mut self) -> &mut SessionTicketStore {
        &mut self.ticket_store
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
    fn test_new_endpoint() {
        let ep = QuicEndpoint::new(EndpointConfig::default());
        assert_eq!(ep.connection_count(), 0);
        assert!(!ep.is_server());
    }

    #[test]
    fn test_server_endpoint() {
        let ep = QuicEndpoint::server(default_params());
        assert!(ep.is_server());
        assert_eq!(ep.connection_count(), 0);
    }

    #[test]
    fn test_client_endpoint() {
        let ep = QuicEndpoint::client(default_params());
        assert!(!ep.is_server());
    }

    #[test]
    fn test_client_connect() {
        let mut ep = QuicEndpoint::client(default_params());
        let handle = ep.connect(b"10.0.0.1:4433", b"10.0.0.2:443", 1000).unwrap();
        assert_eq!(ep.connection_count(), 1);
        assert_eq!(
            ep.connection_state(handle),
            Some(ConnectionState::Handshaking)
        );
    }

    #[test]
    fn test_server_cannot_connect() {
        let mut ep = QuicEndpoint::server(default_params());
        assert_eq!(
            ep.connect(b"local", b"remote", 1000).err(),
            Some(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn test_connection_limit() {
        let mut ep = QuicEndpoint::new(EndpointConfig {
            max_connections: 2,
            ..Default::default()
        });
        ep.connect(b"l", b"r1", 1000).unwrap();
        ep.connect(b"l", b"r2", 1000).unwrap();
        assert_eq!(
            ep.connect(b"l", b"r3", 1000).err(),
            Some(NetError::TableFull)
        );
    }

    #[test]
    fn test_poll_events_empty() {
        let mut ep = QuicEndpoint::client(default_params());
        assert!(ep.poll_events().is_empty());
    }

    #[test]
    fn test_receive_empty_packet() {
        let mut ep = QuicEndpoint::server(default_params());
        assert_eq!(
            ep.receive(&[], b"local", b"remote", 1000).err(),
            Some(NetError::PacketTooShort)
        );
    }

    #[test]
    fn test_receive_unknown_short_header() {
        let mut ep = QuicEndpoint::server(default_params());
        // Short header with unknown DCID
        let mut pkt = vec![0x40]; // short header
        pkt.extend_from_slice(&[0xFF; 8]); // unknown DCID
        pkt.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // PN
        assert_eq!(
            ep.receive(&pkt, b"local", b"remote", 1000).err(),
            Some(NetError::NotFound)
        );
    }

    #[test]
    fn test_server_accept_initial() {
        let mut ep = QuicEndpoint::server(default_params());
        // Construct a minimal Initial packet
        let mut hdr = LongHeader::new(PacketType::Initial, &[0x01; 8], &[0x02; 8]).unwrap();
        hdr.payload_length = 20;
        hdr.packet_number = 0;
        let mut buf = [0u8; 128];
        let hdr_len = hdr.encode(&mut buf).unwrap();
        // Add minimal payload
        for i in hdr_len..hdr_len + 20 {
            buf[i] = 0;
        }
        let total = hdr_len + 20;
        ep.receive(&buf[..total], b"local", b"remote", 1000)
            .unwrap();
        assert_eq!(ep.connection_count(), 1);
    }

    #[test]
    fn test_close_connection() {
        let mut ep = QuicEndpoint::client(default_params());
        let handle = ep.connect(b"l", b"r", 1000).unwrap();
        // Connection is in Handshaking state, which can be closed
        ep.close(handle, 0, b"bye", 2000).unwrap();
        let events = ep.poll_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, EndpointEvent::ConnectionClosed { .. })));
    }

    #[test]
    fn test_close_nonexistent() {
        let mut ep = QuicEndpoint::client(default_params());
        assert_eq!(ep.close(999, 0, b"", 1000).err(), Some(NetError::NotFound));
    }

    #[test]
    fn test_stream_on_non_established() {
        let mut ep = QuicEndpoint::client(default_params());
        let handle = ep.connect(b"l", b"r", 1000).unwrap();
        // Connection not yet established (still handshaking)
        assert_eq!(
            ep.open_stream_bidi(handle).err(),
            Some(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn test_garbage_collect() {
        let mut ep = QuicEndpoint::client(default_params());
        let handle = ep.connect(b"l", b"r", 1000).unwrap();
        ep.close(handle, 0, b"", 2000).unwrap();

        // Transition to Closed
        if let Some(ec) = ep.connections.get_mut(&handle) {
            let _ = ec.conn.finish_draining();
        }

        ep.garbage_collect();
        assert_eq!(ep.connection_count(), 0);
    }

    #[test]
    fn test_handle_idle_timeout() {
        let mut ep = QuicEndpoint::client(default_params());
        let _handle = ep.connect(b"l", b"r", 0).unwrap();
        // Default idle timeout is 30s = 30,000,000 us
        ep.handle_timeouts(30_000_001);
        let events = ep.poll_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, EndpointEvent::ConnectionClosed { .. })));
    }

    #[test]
    fn test_multiple_connections() {
        let mut ep = QuicEndpoint::client(default_params());
        let h1 = ep.connect(b"l", b"r1", 1000).unwrap();
        let h2 = ep.connect(b"l", b"r2", 1000).unwrap();
        assert_ne!(h1, h2);
        assert_eq!(ep.connection_count(), 2);
    }

    #[test]
    fn test_ticket_store_access() {
        let mut ep = QuicEndpoint::client(default_params());
        assert_eq!(ep.ticket_store().count(), 0);
        ep.ticket_store_mut()
            .store(super::super::retry::SessionTicket {
                data: Vec::from([1u8]),
                server_name: Vec::from(b"test.com" as &[u8]),
                issued_at_us: 0,
                lifetime_us: 1_000_000,
                max_early_data: 0,
            });
        assert_eq!(ep.ticket_store().count(), 1);
    }

    #[test]
    fn test_endpoint_event_variants() {
        let e1 = EndpointEvent::Connected { handle: 1 };
        let e2 = EndpointEvent::StreamReadable {
            handle: 1,
            stream_id: 0,
        };
        let e3 = EndpointEvent::StreamWritable {
            handle: 1,
            stream_id: 0,
        };
        let e4 = EndpointEvent::StreamFinished {
            handle: 1,
            stream_id: 0,
        };
        let e5 = EndpointEvent::ConnectionClosed { handle: 1 };
        let e6 = EndpointEvent::HandshakeComplete { handle: 1 };
        assert_ne!(e1, e2);
        assert_ne!(e3, e4);
        assert_ne!(e5, e6);
    }
}
