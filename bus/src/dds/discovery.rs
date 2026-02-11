// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DDS Discovery Protocols: SPDP and SEDP.
//!
//! SPDP (Simple Participant Discovery Protocol) — periodic multicast
//! announcements for participant presence and lease management.
//!
//! SEDP (Simple Endpoint Discovery Protocol) — exchange of DataWriter
//! and DataReader endpoint information for automatic matching.

use alloc::string::String;
use alloc::vec::Vec;
use crate::BusError;
use crate::dds::endpoint::MatchStatus;
use crate::dds::qos::QosPolicy;
use crate::dds::types::{DomainId, Guid, GuidPrefix};

/// Default SPDP announcement interval in microseconds (30 seconds / 3 = 10s).
pub const SPDP_DEFAULT_INTERVAL_US: u64 = 10_000_000;

/// Default participant lease duration in microseconds (30 seconds).
pub const SPDP_DEFAULT_LEASE_US: u64 = 30_000_000;

/// Maximum number of discovered participants.
pub const MAX_DISCOVERED_PARTICIPANTS: usize = 64;

/// Maximum number of discovered endpoints per type (writers + readers).
pub const MAX_DISCOVERED_ENDPOINTS: usize = 128;

/// SPDP multicast address: 239.255.0.1.
pub const SPDP_MULTICAST_ADDR: [u8; 4] = [239, 255, 0, 1];

/// Discovered participant information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredParticipant {
    /// Participant GUID prefix.
    pub guid_prefix: GuidPrefix,
    /// Domain ID.
    pub domain_id: DomainId,
    /// Lease duration in microseconds.
    pub lease_duration_us: u64,
    /// Timestamp of last announcement (microseconds since boot).
    pub last_seen_us: u64,
    /// Whether this participant is still alive.
    pub alive: bool,
}

impl DiscoveredParticipant {
    /// Check if the participant's lease has expired.
    pub fn is_expired(&self, now_us: u64) -> bool {
        now_us.saturating_sub(self.last_seen_us) > self.lease_duration_us
    }

    /// Refresh the participant's last-seen timestamp.
    pub fn refresh(&mut self, now_us: u64) {
        self.last_seen_us = now_us;
        self.alive = true;
    }
}

/// SPDP announcement data — what goes into an SPDP DATA submessage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdpData {
    /// Participant GUID prefix.
    pub guid_prefix: GuidPrefix,
    /// Domain ID.
    pub domain_id: DomainId,
    /// Lease duration in microseconds.
    pub lease_duration_us: u64,
    /// SPDP multicast port.
    pub multicast_port: u16,
    /// SPDP unicast port.
    pub unicast_port: u16,
}

impl SpdpData {
    /// Encode SPDP announcement data.
    ///
    /// Layout (20 bytes):
    /// - guid_prefix (12 bytes)
    /// - domain_id (4 bytes, LE)
    /// - lease_duration_us high 16 bits as seconds (2 bytes, LE)
    /// - multicast_port (2 bytes, LE) — not encoded, derived from domain
    ///
    /// Simplified encoding for SmallAIOS (not full PL_CDR).
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        const LEN: usize = 20;
        if buf.len() < LEN {
            return Err(BusError::BufferTooSmall);
        }
        buf[0..12].copy_from_slice(&self.guid_prefix.0);
        buf[12..16].copy_from_slice(&self.domain_id.to_le_bytes());
        // Lease in seconds (truncated from microseconds)
        let lease_sec = (self.lease_duration_us / 1_000_000) as u16;
        buf[16..18].copy_from_slice(&lease_sec.to_le_bytes());
        buf[18..20].copy_from_slice(&self.unicast_port.to_le_bytes());
        Ok(LEN)
    }

    /// Decode SPDP announcement data.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < 20 {
            return Err(BusError::FrameTooShort);
        }
        let mut prefix = [0u8; 12];
        prefix.copy_from_slice(&data[0..12]);
        let domain_id = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let lease_sec = u16::from_le_bytes(data[16..18].try_into().unwrap());
        let unicast_port = u16::from_le_bytes(data[18..20].try_into().unwrap());
        // Compute multicast port from domain
        let multicast_port = (7400 + 250 * domain_id) as u16;
        Ok(Self {
            guid_prefix: GuidPrefix(prefix),
            domain_id,
            lease_duration_us: lease_sec as u64 * 1_000_000,
            multicast_port,
            unicast_port,
        })
    }
}

/// SPDP participant database.
#[derive(Debug)]
pub struct SpdpDatabase {
    /// Local participant GUID prefix.
    local_prefix: GuidPrefix,
    /// Local domain ID.
    domain_id: DomainId,
    /// Discovered remote participants.
    participants: Vec<DiscoveredParticipant>,
    /// Lease duration for the local participant.
    lease_duration_us: u64,
    /// Timestamp of last SPDP announcement sent.
    last_announce_us: u64,
    /// Announcement interval.
    announce_interval_us: u64,
    /// Whether we have ever announced.
    has_announced: bool,
}

impl SpdpDatabase {
    /// Create a new SPDP database.
    pub fn new(local_prefix: GuidPrefix, domain_id: DomainId) -> Self {
        Self {
            local_prefix,
            domain_id,
            participants: Vec::new(),
            lease_duration_us: SPDP_DEFAULT_LEASE_US,
            last_announce_us: 0,
            announce_interval_us: SPDP_DEFAULT_INTERVAL_US,
            has_announced: false,
        }
    }

    /// Returns the local GUID prefix.
    pub fn local_prefix(&self) -> &GuidPrefix {
        &self.local_prefix
    }

    /// Returns the domain ID.
    pub fn domain_id(&self) -> DomainId {
        self.domain_id
    }

    /// Returns discovered participants.
    pub fn participants(&self) -> &[DiscoveredParticipant] {
        &self.participants
    }

    /// Returns the number of discovered participants.
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// Check if it's time to send an SPDP announcement.
    pub fn should_announce(&self, now_us: u64) -> bool {
        !self.has_announced
            || now_us.saturating_sub(self.last_announce_us) >= self.announce_interval_us
    }

    /// Record that an announcement was sent.
    pub fn mark_announced(&mut self, now_us: u64) {
        self.last_announce_us = now_us;
        self.has_announced = true;
    }

    /// Generate SPDP announcement data for the local participant.
    pub fn generate_announcement(&self) -> SpdpData {
        let multicast_port = (7400 + 250 * self.domain_id) as u16;
        let unicast_port = (7400 + 250 * self.domain_id + 10) as u16;
        SpdpData {
            guid_prefix: self.local_prefix,
            domain_id: self.domain_id,
            lease_duration_us: self.lease_duration_us,
            multicast_port,
            unicast_port,
        }
    }

    /// Process a received SPDP announcement.
    pub fn process_announcement(&mut self, data: &SpdpData, now_us: u64) -> Result<bool, BusError> {
        // Ignore our own announcements
        if data.guid_prefix == self.local_prefix {
            return Ok(false);
        }
        // Ignore wrong domain
        if data.domain_id != self.domain_id {
            return Ok(false);
        }

        // Check if already known
        if let Some(p) = self.participants.iter_mut().find(|p| p.guid_prefix == data.guid_prefix) {
            p.refresh(now_us);
            p.lease_duration_us = data.lease_duration_us;
            return Ok(false); // Already known, just refreshed
        }

        // New participant
        if self.participants.len() >= MAX_DISCOVERED_PARTICIPANTS {
            return Err(BusError::OutOfRange);
        }
        self.participants.push(DiscoveredParticipant {
            guid_prefix: data.guid_prefix,
            domain_id: data.domain_id,
            lease_duration_us: data.lease_duration_us,
            last_seen_us: now_us,
            alive: true,
        });
        Ok(true) // New participant discovered
    }

    /// Expire participants whose lease has elapsed.
    pub fn expire_participants(&mut self, now_us: u64) -> usize {
        let mut expired_count = 0;
        for p in &mut self.participants {
            if p.alive && p.is_expired(now_us) {
                p.alive = false;
                expired_count += 1;
            }
        }
        // Remove dead entries
        self.participants.retain(|p| p.alive);
        expired_count
    }
}

/// Endpoint kind for SEDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointKind {
    /// DataWriter endpoint.
    Writer,
    /// DataReader endpoint.
    Reader,
}

/// Discovered endpoint information for SEDP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredEndpoint {
    /// Endpoint GUID.
    pub guid: Guid,
    /// Participant GUID prefix.
    pub participant_prefix: GuidPrefix,
    /// Endpoint kind (writer or reader).
    pub kind: EndpointKind,
    /// Topic name.
    pub topic_name: String,
    /// Type name.
    pub type_name: String,
    /// QoS policy.
    pub qos: QosPolicy,
}

/// SEDP endpoint database for automatic writer/reader matching.
#[derive(Debug)]
pub struct SedpDatabase {
    /// Domain ID.
    domain_id: DomainId,
    /// Discovered remote endpoints.
    endpoints: Vec<DiscoveredEndpoint>,
    /// Number of matches made.
    match_count: u32,
}

impl SedpDatabase {
    /// Create a new SEDP database.
    pub fn new(domain_id: DomainId) -> Self {
        Self {
            domain_id,
            endpoints: Vec::new(),
            match_count: 0,
        }
    }

    /// Returns the domain ID.
    pub fn domain_id(&self) -> DomainId {
        self.domain_id
    }

    /// Returns all discovered endpoints.
    pub fn endpoints(&self) -> &[DiscoveredEndpoint] {
        &self.endpoints
    }

    /// Returns discovered writers.
    pub fn writers(&self) -> Vec<&DiscoveredEndpoint> {
        self.endpoints.iter().filter(|e| e.kind == EndpointKind::Writer).collect()
    }

    /// Returns discovered readers.
    pub fn readers(&self) -> Vec<&DiscoveredEndpoint> {
        self.endpoints.iter().filter(|e| e.kind == EndpointKind::Reader).collect()
    }

    /// Returns the number of discovered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Returns the number of matches made.
    pub fn match_count(&self) -> u32 {
        self.match_count
    }

    /// Register a discovered endpoint. Returns true if new.
    pub fn add_endpoint(&mut self, ep: DiscoveredEndpoint) -> Result<bool, BusError> {
        // Check for duplicates
        if self.endpoints.iter().any(|e| e.guid == ep.guid) {
            return Ok(false);
        }
        if self.endpoints.len() >= MAX_DISCOVERED_ENDPOINTS {
            return Err(BusError::OutOfRange);
        }
        self.endpoints.push(ep);
        Ok(true)
    }

    /// Remove all endpoints belonging to a participant.
    pub fn remove_participant_endpoints(&mut self, prefix: &GuidPrefix) -> usize {
        let before = self.endpoints.len();
        self.endpoints.retain(|e| e.participant_prefix != *prefix);
        before - self.endpoints.len()
    }

    /// Find matching writer/reader pairs based on topic name, type name, and QoS.
    ///
    /// Returns a list of (writer_guid, reader_guid) pairs that should be matched.
    pub fn find_matches(&mut self) -> Vec<(Guid, Guid)> {
        let mut matches = Vec::new();
        let writers: Vec<&DiscoveredEndpoint> = self.endpoints
            .iter()
            .filter(|e| e.kind == EndpointKind::Writer)
            .collect();
        let readers: Vec<&DiscoveredEndpoint> = self.endpoints
            .iter()
            .filter(|e| e.kind == EndpointKind::Reader)
            .collect();

        for w in &writers {
            for r in &readers {
                // Same participant — skip self-matching
                if w.participant_prefix == r.participant_prefix {
                    continue;
                }
                // Topic name match
                if w.topic_name != r.topic_name {
                    continue;
                }
                // Type name match
                if w.type_name != r.type_name {
                    continue;
                }
                // QoS compatibility
                if QosPolicy::is_compatible(&w.qos, &r.qos) {
                    matches.push((w.guid, r.guid));
                    self.match_count += 1;
                }
            }
        }
        matches
    }

    /// Check if a specific writer-reader pair would match.
    pub fn check_pair(&self, writer_guid: &Guid, reader_guid: &Guid) -> MatchStatus {
        let writer = self.endpoints.iter().find(|e| e.guid == *writer_guid && e.kind == EndpointKind::Writer);
        let reader = self.endpoints.iter().find(|e| e.guid == *reader_guid && e.kind == EndpointKind::Reader);
        match (writer, reader) {
            (Some(w), Some(r)) => {
                if w.topic_name != r.topic_name {
                    MatchStatus::NotMatched
                } else if w.type_name != r.type_name {
                    MatchStatus::TypeMismatch
                } else if !QosPolicy::is_compatible(&w.qos, &r.qos) {
                    MatchStatus::QosIncompatible
                } else {
                    MatchStatus::Matched
                }
            }
            _ => MatchStatus::NotMatched,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dds::qos::*;
    use crate::dds::types::*;

    fn prefix(id: u8) -> GuidPrefix {
        GuidPrefix::new([id; 12])
    }

    fn guid(id: u8) -> Guid {
        Guid::new(prefix(id), EntityId::user([0, 0, id], 0x02))
    }

    // --- SPDP Tests ---

    #[test]
    fn test_spdp_database_new() {
        let db = SpdpDatabase::new(prefix(1), 0);
        assert_eq!(db.participant_count(), 0);
        assert_eq!(db.domain_id(), 0);
    }

    #[test]
    fn test_spdp_data_encode_decode() {
        let data = SpdpData {
            guid_prefix: prefix(1),
            domain_id: 0,
            lease_duration_us: 30_000_000,
            multicast_port: 7400,
            unicast_port: 7410,
        };
        let mut buf = [0u8; 32];
        let len = data.encode(&mut buf).unwrap();
        assert_eq!(len, 20);
        let decoded = SpdpData::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.guid_prefix, prefix(1));
        assert_eq!(decoded.domain_id, 0);
        assert_eq!(decoded.lease_duration_us, 30_000_000);
        assert_eq!(decoded.unicast_port, 7410);
    }

    #[test]
    fn test_spdp_data_encode_too_small() {
        let data = SpdpData {
            guid_prefix: prefix(1),
            domain_id: 0,
            lease_duration_us: 30_000_000,
            multicast_port: 7400,
            unicast_port: 7410,
        };
        let mut buf = [0u8; 5];
        assert_eq!(data.encode(&mut buf).err(), Some(BusError::BufferTooSmall));
    }

    #[test]
    fn test_spdp_data_decode_too_short() {
        assert_eq!(SpdpData::decode(&[0u8; 5]).err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_spdp_process_announcement() {
        let mut db = SpdpDatabase::new(prefix(1), 0);
        let data = SpdpData {
            guid_prefix: prefix(2),
            domain_id: 0,
            lease_duration_us: 30_000_000,
            multicast_port: 7400,
            unicast_port: 7410,
        };
        let is_new = db.process_announcement(&data, 1000).unwrap();
        assert!(is_new);
        assert_eq!(db.participant_count(), 1);
        assert_eq!(db.participants()[0].guid_prefix, prefix(2));
    }

    #[test]
    fn test_spdp_ignore_own_announcement() {
        let mut db = SpdpDatabase::new(prefix(1), 0);
        let data = SpdpData {
            guid_prefix: prefix(1),
            domain_id: 0,
            lease_duration_us: 30_000_000,
            multicast_port: 7400,
            unicast_port: 7410,
        };
        let is_new = db.process_announcement(&data, 1000).unwrap();
        assert!(!is_new);
        assert_eq!(db.participant_count(), 0);
    }

    #[test]
    fn test_spdp_ignore_wrong_domain() {
        let mut db = SpdpDatabase::new(prefix(1), 0);
        let data = SpdpData {
            guid_prefix: prefix(2),
            domain_id: 1, // Different domain
            lease_duration_us: 30_000_000,
            multicast_port: 7650,
            unicast_port: 7660,
        };
        let is_new = db.process_announcement(&data, 1000).unwrap();
        assert!(!is_new);
    }

    #[test]
    fn test_spdp_refresh_existing() {
        let mut db = SpdpDatabase::new(prefix(1), 0);
        let data = SpdpData {
            guid_prefix: prefix(2),
            domain_id: 0,
            lease_duration_us: 30_000_000,
            multicast_port: 7400,
            unicast_port: 7410,
        };
        db.process_announcement(&data, 1000).unwrap();
        let is_new = db.process_announcement(&data, 5000).unwrap();
        assert!(!is_new);
        assert_eq!(db.participants()[0].last_seen_us, 5000);
    }

    #[test]
    fn test_spdp_expire_participants() {
        let mut db = SpdpDatabase::new(prefix(1), 0);
        let data = SpdpData {
            guid_prefix: prefix(2),
            domain_id: 0,
            lease_duration_us: 10_000_000, // 10 seconds
            multicast_port: 7400,
            unicast_port: 7410,
        };
        db.process_announcement(&data, 1_000_000).unwrap();
        assert_eq!(db.participant_count(), 1);

        // After 11 seconds, should expire
        let expired = db.expire_participants(12_000_000);
        assert_eq!(expired, 1);
        assert_eq!(db.participant_count(), 0);
    }

    #[test]
    fn test_spdp_should_announce() {
        let db = SpdpDatabase::new(prefix(1), 0);
        assert!(db.should_announce(0));
        assert!(db.should_announce(SPDP_DEFAULT_INTERVAL_US));
    }

    #[test]
    fn test_spdp_mark_announced() {
        let mut db = SpdpDatabase::new(prefix(1), 0);
        db.mark_announced(5_000_000);
        assert!(!db.should_announce(5_000_000));
        assert!(db.should_announce(5_000_000 + SPDP_DEFAULT_INTERVAL_US));
    }

    #[test]
    fn test_spdp_generate_announcement() {
        let db = SpdpDatabase::new(prefix(1), 0);
        let data = db.generate_announcement();
        assert_eq!(data.guid_prefix, prefix(1));
        assert_eq!(data.domain_id, 0);
        assert_eq!(data.multicast_port, 7400);
    }

    #[test]
    fn test_discovered_participant_expired() {
        let p = DiscoveredParticipant {
            guid_prefix: prefix(2),
            domain_id: 0,
            lease_duration_us: 10_000_000,
            last_seen_us: 1_000_000,
            alive: true,
        };
        assert!(!p.is_expired(5_000_000));
        assert!(p.is_expired(12_000_000));
    }

    // --- SEDP Tests ---

    #[test]
    fn test_sedp_database_new() {
        let db = SedpDatabase::new(0);
        assert_eq!(db.endpoint_count(), 0);
        assert_eq!(db.match_count(), 0);
    }

    #[test]
    fn test_sedp_add_endpoint() {
        let mut db = SedpDatabase::new(0);
        let ep = DiscoveredEndpoint {
            guid: guid(1),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        assert!(db.add_endpoint(ep).unwrap());
        assert_eq!(db.endpoint_count(), 1);
    }

    #[test]
    fn test_sedp_add_duplicate() {
        let mut db = SedpDatabase::new(0);
        let ep = DiscoveredEndpoint {
            guid: guid(1),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        db.add_endpoint(ep.clone()).unwrap();
        assert!(!db.add_endpoint(ep).unwrap());
        assert_eq!(db.endpoint_count(), 1);
    }

    #[test]
    fn test_sedp_find_matches() {
        let mut db = SedpDatabase::new(0);
        let writer = DiscoveredEndpoint {
            guid: guid(1),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        let reader = DiscoveredEndpoint {
            guid: guid(2),
            participant_prefix: prefix(2),
            kind: EndpointKind::Reader,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        db.add_endpoint(writer).unwrap();
        db.add_endpoint(reader).unwrap();
        let matches = db.find_matches();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, guid(1));
        assert_eq!(matches[0].1, guid(2));
    }

    #[test]
    fn test_sedp_no_self_match() {
        let mut db = SedpDatabase::new(0);
        // Writer and reader from the same participant
        let writer = DiscoveredEndpoint {
            guid: Guid::new(prefix(1), EntityId::user([0, 0, 1], 0x02)),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        let reader = DiscoveredEndpoint {
            guid: Guid::new(prefix(1), EntityId::user([0, 0, 2], 0x07)),
            participant_prefix: prefix(1),
            kind: EndpointKind::Reader,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        db.add_endpoint(writer).unwrap();
        db.add_endpoint(reader).unwrap();
        let matches = db.find_matches();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_sedp_no_match_different_topic() {
        let mut db = SedpDatabase::new(0);
        let writer = DiscoveredEndpoint {
            guid: guid(1),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("TopicA"),
            type_name: String::from("Msg"),
            qos: QosPolicy::default(),
        };
        let reader = DiscoveredEndpoint {
            guid: guid(2),
            participant_prefix: prefix(2),
            kind: EndpointKind::Reader,
            topic_name: String::from("TopicB"),
            type_name: String::from("Msg"),
            qos: QosPolicy::default(),
        };
        db.add_endpoint(writer).unwrap();
        db.add_endpoint(reader).unwrap();
        let matches = db.find_matches();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_sedp_no_match_type_mismatch() {
        let mut db = SedpDatabase::new(0);
        let writer = DiscoveredEndpoint {
            guid: guid(1),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        let reader = DiscoveredEndpoint {
            guid: guid(2),
            participant_prefix: prefix(2),
            kind: EndpointKind::Reader,
            topic_name: String::from("Sensor"),
            type_name: String::from("OtherMsg"),
            qos: QosPolicy::default(),
        };
        db.add_endpoint(writer).unwrap();
        db.add_endpoint(reader).unwrap();
        let matches = db.find_matches();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_sedp_no_match_qos_incompatible() {
        let mut db = SedpDatabase::new(0);
        let writer = DiscoveredEndpoint {
            guid: guid(1),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy {
                reliability: ReliabilityKind::BestEffort,
                ..Default::default()
            },
        };
        let reader = DiscoveredEndpoint {
            guid: guid(2),
            participant_prefix: prefix(2),
            kind: EndpointKind::Reader,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy {
                reliability: ReliabilityKind::Reliable,
                ..Default::default()
            },
        };
        db.add_endpoint(writer).unwrap();
        db.add_endpoint(reader).unwrap();
        let matches = db.find_matches();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_sedp_remove_participant_endpoints() {
        let mut db = SedpDatabase::new(0);
        let ep1 = DiscoveredEndpoint {
            guid: guid(1),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("A"),
            type_name: String::from("M"),
            qos: QosPolicy::default(),
        };
        let ep2 = DiscoveredEndpoint {
            guid: guid(2),
            participant_prefix: prefix(2),
            kind: EndpointKind::Reader,
            topic_name: String::from("A"),
            type_name: String::from("M"),
            qos: QosPolicy::default(),
        };
        db.add_endpoint(ep1).unwrap();
        db.add_endpoint(ep2).unwrap();
        let removed = db.remove_participant_endpoints(&prefix(1));
        assert_eq!(removed, 1);
        assert_eq!(db.endpoint_count(), 1);
    }

    #[test]
    fn test_sedp_check_pair() {
        let mut db = SedpDatabase::new(0);
        let writer = DiscoveredEndpoint {
            guid: guid(1),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        let reader = DiscoveredEndpoint {
            guid: guid(2),
            participant_prefix: prefix(2),
            kind: EndpointKind::Reader,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        db.add_endpoint(writer).unwrap();
        db.add_endpoint(reader).unwrap();
        assert_eq!(db.check_pair(&guid(1), &guid(2)), MatchStatus::Matched);
    }

    #[test]
    fn test_sedp_check_pair_not_found() {
        let db = SedpDatabase::new(0);
        assert_eq!(db.check_pair(&guid(1), &guid(2)), MatchStatus::NotMatched);
    }

    #[test]
    fn test_sedp_writers_readers_accessors() {
        let mut db = SedpDatabase::new(0);
        let ep1 = DiscoveredEndpoint {
            guid: guid(1),
            participant_prefix: prefix(1),
            kind: EndpointKind::Writer,
            topic_name: String::from("A"),
            type_name: String::from("M"),
            qos: QosPolicy::default(),
        };
        let ep2 = DiscoveredEndpoint {
            guid: guid(2),
            participant_prefix: prefix(2),
            kind: EndpointKind::Reader,
            topic_name: String::from("A"),
            type_name: String::from("M"),
            qos: QosPolicy::default(),
        };
        db.add_endpoint(ep1).unwrap();
        db.add_endpoint(ep2).unwrap();
        assert_eq!(db.writers().len(), 1);
        assert_eq!(db.readers().len(), 1);
    }

    #[test]
    fn test_sedp_multiple_matches() {
        let mut db = SedpDatabase::new(0);
        // Two writers, one reader on same topic
        for i in 1..=2u8 {
            let w = DiscoveredEndpoint {
                guid: Guid::new(prefix(i), EntityId::user([0, 0, i], 0x02)),
                participant_prefix: prefix(i),
                kind: EndpointKind::Writer,
                topic_name: String::from("Sensor"),
                type_name: String::from("SensorMsg"),
                qos: QosPolicy::default(),
            };
            db.add_endpoint(w).unwrap();
        }
        let r = DiscoveredEndpoint {
            guid: Guid::new(prefix(3), EntityId::user([0, 0, 3], 0x07)),
            participant_prefix: prefix(3),
            kind: EndpointKind::Reader,
            topic_name: String::from("Sensor"),
            type_name: String::from("SensorMsg"),
            qos: QosPolicy::default(),
        };
        db.add_endpoint(r).unwrap();
        let matches = db.find_matches();
        assert_eq!(matches.len(), 2);
    }
}
