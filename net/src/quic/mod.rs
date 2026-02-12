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

pub mod congestion;
pub mod connection;
pub mod connid;
pub mod endpoint;
pub mod flow;
pub mod frame;
pub mod h3;
pub mod key_update;
pub mod migration;
pub mod packet;
pub mod pktnum;
pub mod protection;
pub mod retry;
pub mod spaces;
pub mod stream;
pub mod tls;
pub mod zenoh;

pub use congestion::{CongestionController, CongestionState, RttEstimator};
pub use connection::{Connection, ConnectionRole, ConnectionState, TransportParameters};
pub use connid::ConnectionIdManager;
pub use endpoint::{ConnectionHandle, EndpointConfig, EndpointEvent, QuicEndpoint};
pub use flow::ConnectionFlowControl;
pub use frame::{FrameType, QuicFrame};
pub use h3::{H3Frame, H3FrameType, H3Settings, HttpMethod, HttpStatus, ManagementPath};
pub use key_update::{KeyUpdateManager, KeyUpdateState};
pub use migration::MigrationManager;
pub use packet::{LongHeader, PacketType, QuicVersion, ShortHeader};
pub use pktnum::{decode_packet_number, encode_packet_number};
pub use retry::{ReplayFilter, RetryToken, SessionTicketStore};
pub use spaces::{PacketNumberSpace, PacketNumberSpaceState};
pub use stream::{StreamInitiator, StreamManager, StreamType};
pub use tls::{
    HybridKeyShare, HybridServerShare, TlsHandshake, TlsHandshakeState, TlsKeySchedule, TlsRole,
    NAMED_GROUP_X25519_MLKEM768,
};
pub use zenoh::{QuicLocator, SessionState, ZenohMsgType, ZenohQuicTransport};
