// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC v1 transport protocol (RFC 9000/9001/9002).
//!
//! Implements:
//! - Packet header encoding/decoding (Initial, Handshake, 0-RTT, 1-RTT)
//! - Frame encoding/decoding (all QUIC frame types)
//! - Connection ID management (generation, retirement, rotation)
//! - Packet number encoding/decoding (RFC 9000 Appendix A)
//! - Packet protection (AEAD stub)
//! - Packet number spaces (Initial, Handshake, Application)
//! - Connection state machine (Idle -> Handshaking -> Connected -> Draining -> Closed)
//! - Stream management (bidirectional/unidirectional, flow control)
//! - Connection-level flow control
//! - Congestion control (NewReno) and RTT estimation
//! - Connection migration and path validation
//! - Retry tokens, version negotiation, 0-RTT session tickets
//! - Key update mechanism (RFC 9001 Section 6)
//! - Standalone endpoint API (server + client)
//! - Zenoh session transport adapter (quic:// locator)
//! - Minimal HTTP/3 framing (RFC 9114)

pub mod packet;
pub mod frame;
pub mod connid;
pub mod pktnum;
pub mod protection;
pub mod spaces;
pub mod connection;
pub mod stream;
pub mod flow;
pub mod congestion;
pub mod migration;
pub mod retry;
pub mod key_update;
pub mod endpoint;
pub mod zenoh;
pub mod h3;
pub mod tls;

pub use packet::{PacketType, LongHeader, ShortHeader, QuicVersion};
pub use frame::{FrameType, QuicFrame};
pub use connid::ConnectionIdManager;
pub use pktnum::{encode_packet_number, decode_packet_number};
pub use spaces::{PacketNumberSpace, PacketNumberSpaceState};
pub use connection::{Connection, ConnectionState, ConnectionRole, TransportParameters};
pub use stream::{StreamManager, StreamType, StreamInitiator};
pub use flow::ConnectionFlowControl;
pub use congestion::{CongestionController, CongestionState, RttEstimator};
pub use migration::MigrationManager;
pub use retry::{SessionTicketStore, ReplayFilter, RetryToken};
pub use key_update::{KeyUpdateManager, KeyUpdateState};
pub use endpoint::{QuicEndpoint, EndpointConfig, EndpointEvent, ConnectionHandle};
pub use zenoh::{ZenohQuicTransport, QuicLocator, ZenohMsgType, SessionState};
pub use h3::{H3Frame, H3FrameType, H3Settings, HttpMethod, HttpStatus, ManagementPath};
pub use tls::{TlsHandshake, TlsHandshakeState, TlsRole, TlsKeySchedule, HybridKeyShare, HybridServerShare, NAMED_GROUP_X25519_MLKEM768};
