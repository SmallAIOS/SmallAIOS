// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DDS (Data Distribution Service) — OMG DCPS/RTPS.
//!
//! Implements DDS core types (DomainParticipant, Topic, DataWriter, DataReader),
//! QoS policies, RTPS 2.3 wire protocol, reliable delivery, SPDP/SEDP
//! discovery, and DDS-Security authentication/access control.

pub mod adapter;
pub mod best_effort;
pub mod cdr;
pub mod discovery;
pub mod endpoint;
pub mod loopback;
pub mod qos;
pub mod reliable;
pub mod rtps;
pub mod security;
pub mod topic;
pub mod types;

pub use adapter::DdsZenohAdapter;
pub use endpoint::{DataReader, DataWriter, MatchStatus};
pub use qos::{
    DeadlineQos, DurabilityKind, HistoryKind, HistoryQos, LivelinessKind,
    OwnershipKind, QosPolicy, ReliabilityKind,
};
pub use rtps::{RtpsHeader, RtpsMessage, SubmessageKind};
pub use topic::Topic;
pub use types::{DomainId, DomainParticipant, Guid, GuidPrefix};
pub use reliable::{WriterHeartbeat, ReaderAckNack};
pub use best_effort::{BestEffortWriter, BestEffortReader, BestEffortReceiveResult};
pub use cdr::{CdrSerializer, CdrDeserializer, Endianness};
pub use discovery::{SpdpDatabase, SedpDatabase, DiscoveredEndpoint, EndpointKind};
pub use security::{AuthStatus, AuthenticationPlugin, AccessControlPlugin};
