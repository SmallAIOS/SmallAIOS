// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DDS core types: DomainId, GUID, DomainParticipant.

use crate::BusError;

/// DDS Domain ID (0-232 per OMG spec).
pub type DomainId = u32;

/// Maximum valid domain ID.
pub const MAX_DOMAIN_ID: DomainId = 232;

/// GUID prefix (12 bytes) — uniquely identifies a participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuidPrefix(pub [u8; 12]);

impl GuidPrefix {
    /// Create a GUID prefix from bytes.
    pub fn new(bytes: [u8; 12]) -> Self {
        Self(bytes)
    }

    /// Unknown/unset GUID prefix.
    pub fn unknown() -> Self {
        Self([0; 12])
    }
}

/// Entity ID (4 bytes) — identifies an entity within a participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityId(pub [u8; 4]);

impl EntityId {
    /// Participant built-in entity.
    pub fn participant() -> Self {
        Self([0x00, 0x00, 0x01, 0xC1])
    }

    /// SPDP built-in writer.
    pub fn spdp_writer() -> Self {
        Self([0x00, 0x01, 0x00, 0xC2])
    }

    /// SPDP built-in reader.
    pub fn spdp_reader() -> Self {
        Self([0x00, 0x01, 0x00, 0xC7])
    }

    /// SEDP publications writer.
    pub fn sedp_pub_writer() -> Self {
        Self([0x00, 0x00, 0x03, 0xC2])
    }

    /// SEDP subscriptions writer.
    pub fn sedp_sub_writer() -> Self {
        Self([0x00, 0x00, 0x04, 0xC2])
    }

    /// User-defined entity.
    pub fn user(key: [u8; 3], kind: u8) -> Self {
        Self([key[0], key[1], key[2], kind])
    }
}

/// Globally Unique Identifier (16 bytes) = GuidPrefix(12) + EntityId(4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Guid {
    pub prefix: GuidPrefix,
    pub entity_id: EntityId,
}

impl Guid {
    /// Create a new GUID.
    pub fn new(prefix: GuidPrefix, entity_id: EntityId) -> Self {
        Self { prefix, entity_id }
    }

    /// Encode GUID to 16 bytes.
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..12].copy_from_slice(&self.prefix.0);
        bytes[12..16].copy_from_slice(&self.entity_id.0);
        bytes
    }

    /// Decode GUID from 16 bytes.
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let mut prefix = [0u8; 12];
        let mut entity = [0u8; 4];
        prefix.copy_from_slice(&bytes[0..12]);
        entity.copy_from_slice(&bytes[12..16]);
        Self {
            prefix: GuidPrefix(prefix),
            entity_id: EntityId(entity),
        }
    }
}

/// DDS DomainParticipant.
///
/// Represents a node's participation in a DDS domain.
#[derive(Debug)]
pub struct DomainParticipant {
    /// Domain ID.
    domain_id: DomainId,
    /// Participant GUID.
    guid: Guid,
    /// Whether the participant is enabled.
    enabled: bool,
}

impl DomainParticipant {
    /// Create a new DomainParticipant.
    pub fn new(domain_id: DomainId, guid_prefix: GuidPrefix) -> Result<Self, BusError> {
        if domain_id > MAX_DOMAIN_ID {
            return Err(BusError::InvalidDomain);
        }
        Ok(Self {
            domain_id,
            guid: Guid::new(guid_prefix, EntityId::participant()),
            enabled: true,
        })
    }

    /// Returns the domain ID.
    pub fn domain_id(&self) -> DomainId {
        self.domain_id
    }

    /// Returns the participant GUID.
    pub fn guid(&self) -> &Guid {
        &self.guid
    }

    /// Returns the GUID prefix.
    pub fn guid_prefix(&self) -> &GuidPrefix {
        &self.guid.prefix
    }

    /// Returns whether the participant is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Disable the participant.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Compute the SPDP multicast port for this domain.
    ///
    /// Per RTPS spec: port = PB + DG * domain_id + d0
    /// PB=7400, DG=250, d0=0 (multicast)
    pub fn spdp_multicast_port(&self) -> u16 {
        (7400 + 250 * self.domain_id) as u16
    }

    /// Compute the SPDP unicast port for a participant index.
    ///
    /// Per RTPS spec: port = PB + DG * domain_id + d1 + PG * participant_id
    /// PB=7400, DG=250, d1=10, PG=2
    pub fn spdp_unicast_port(&self, participant_id: u32) -> u16 {
        (7400 + 250 * self.domain_id + 10 + 2 * participant_id) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_prefix() -> GuidPrefix {
        GuidPrefix::new([0x01; 12])
    }

    #[test]
    fn test_domain_participant_create() {
        let dp = DomainParticipant::new(0, test_prefix()).unwrap();
        assert_eq!(dp.domain_id(), 0);
        assert!(dp.is_enabled());
    }

    #[test]
    fn test_domain_participant_invalid_domain() {
        assert_eq!(
            DomainParticipant::new(233, test_prefix()).err(),
            Some(BusError::InvalidDomain)
        );
    }

    #[test]
    fn test_domain_participant_max_domain() {
        let dp = DomainParticipant::new(MAX_DOMAIN_ID, test_prefix()).unwrap();
        assert_eq!(dp.domain_id(), 232);
    }

    #[test]
    fn test_domain_participant_guid() {
        let dp = DomainParticipant::new(0, test_prefix()).unwrap();
        assert_eq!(dp.guid_prefix(), &test_prefix());
        assert_eq!(dp.guid().entity_id, EntityId::participant());
    }

    #[test]
    fn test_domain_participant_disable() {
        let mut dp = DomainParticipant::new(0, test_prefix()).unwrap();
        dp.disable();
        assert!(!dp.is_enabled());
    }

    #[test]
    fn test_spdp_multicast_port() {
        let dp = DomainParticipant::new(0, test_prefix()).unwrap();
        assert_eq!(dp.spdp_multicast_port(), 7400);
        let dp1 = DomainParticipant::new(1, test_prefix()).unwrap();
        assert_eq!(dp1.spdp_multicast_port(), 7650);
    }

    #[test]
    fn test_spdp_unicast_port() {
        let dp = DomainParticipant::new(0, test_prefix()).unwrap();
        assert_eq!(dp.spdp_unicast_port(0), 7410);
        assert_eq!(dp.spdp_unicast_port(1), 7412);
    }

    #[test]
    fn test_guid_encode_decode() {
        let guid = Guid::new(test_prefix(), EntityId::participant());
        let bytes = guid.to_bytes();
        let decoded = Guid::from_bytes(&bytes);
        assert_eq!(decoded, guid);
    }

    #[test]
    fn test_guid_prefix_unknown() {
        let unknown = GuidPrefix::unknown();
        assert_eq!(unknown.0, [0; 12]);
    }

    #[test]
    fn test_entity_id_builtin() {
        assert_eq!(EntityId::participant().0, [0x00, 0x00, 0x01, 0xC1]);
        assert_eq!(EntityId::spdp_writer().0, [0x00, 0x01, 0x00, 0xC2]);
        assert_eq!(EntityId::spdp_reader().0, [0x00, 0x01, 0x00, 0xC7]);
    }

    #[test]
    fn test_entity_id_user() {
        let eid = EntityId::user([0x01, 0x02, 0x03], 0x04);
        assert_eq!(eid.0, [0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_multiple_participants_different_domains() {
        let dp0 = DomainParticipant::new(0, GuidPrefix::new([0x01; 12])).unwrap();
        let dp1 = DomainParticipant::new(1, GuidPrefix::new([0x02; 12])).unwrap();
        assert_ne!(dp0.domain_id(), dp1.domain_id());
        assert_ne!(dp0.guid_prefix(), dp1.guid_prefix());
    }
}
