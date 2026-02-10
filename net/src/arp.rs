// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARP (Address Resolution Protocol) for SmallAIOS.
//!
//! Provides [`ArpPacket`] for parsing/serializing ARP requests and replies,
//! and [`ArpTable`] for maintaining an IP-to-MAC address mapping cache.

use alloc::vec::Vec;

use crate::ethernet::MacAddress;
use crate::NetError;

// ---------------------------------------------------------------------------
// ArpOperation
// ---------------------------------------------------------------------------

/// ARP operation codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArpOperation {
    /// ARP request (who-has).
    Request,
    /// ARP reply (is-at).
    Reply,
}

impl ArpOperation {
    /// Convert a raw `u16` operation code to an [`ArpOperation`].
    ///
    /// Returns [`NetError::InvalidProtocol`] for unknown operation codes.
    pub fn from_u16(value: u16) -> Result<Self, NetError> {
        match value {
            1 => Ok(ArpOperation::Request),
            2 => Ok(ArpOperation::Reply),
            _ => Err(NetError::InvalidProtocol),
        }
    }

    /// Return the raw `u16` value.
    pub fn as_u16(self) -> u16 {
        match self {
            ArpOperation::Request => 1,
            ArpOperation::Reply => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// ArpPacket
// ---------------------------------------------------------------------------

/// An ARP packet for IPv4 over Ethernet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArpPacket {
    /// Hardware type (1 = Ethernet).
    pub hardware_type: u16,
    /// Protocol type (0x0800 = IPv4).
    pub protocol_type: u16,
    /// Hardware address length (6 for Ethernet).
    pub hw_len: u8,
    /// Protocol address length (4 for IPv4).
    pub proto_len: u8,
    /// Operation code (request / reply).
    pub operation: ArpOperation,
    /// Sender hardware (MAC) address.
    pub sender_hw: MacAddress,
    /// Sender protocol (IPv4) address.
    pub sender_ip: [u8; 4],
    /// Target hardware (MAC) address.
    pub target_hw: MacAddress,
    /// Target protocol (IPv4) address.
    pub target_ip: [u8; 4],
}

impl ArpPacket {
    /// Standard ARP packet length for IPv4-over-Ethernet (28 bytes).
    pub const PACKET_LEN: usize = 28;

    /// Parse an ARP packet from raw bytes.
    ///
    /// `data` must be at least [`Self::PACKET_LEN`] (28) bytes.
    pub fn parse(data: &[u8]) -> Result<Self, NetError> {
        if data.len() < Self::PACKET_LEN {
            return Err(NetError::PacketTooShort);
        }

        let hardware_type = u16::from_be_bytes([data[0], data[1]]);
        let protocol_type = u16::from_be_bytes([data[2], data[3]]);
        let hw_len = data[4];
        let proto_len = data[5];
        let operation = ArpOperation::from_u16(u16::from_be_bytes([data[6], data[7]]))?;

        let sender_hw = MacAddress::from_bytes(&data[8..14])?;
        let mut sender_ip = [0u8; 4];
        sender_ip.copy_from_slice(&data[14..18]);

        let target_hw = MacAddress::from_bytes(&data[18..24])?;
        let mut target_ip = [0u8; 4];
        target_ip.copy_from_slice(&data[24..28]);

        Ok(ArpPacket {
            hardware_type,
            protocol_type,
            hw_len,
            proto_len,
            operation,
            sender_hw,
            sender_ip,
            target_hw,
            target_ip,
        })
    }

    /// Serialize the ARP packet into a byte vector.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::PACKET_LEN);
        buf.extend_from_slice(&self.hardware_type.to_be_bytes());
        buf.extend_from_slice(&self.protocol_type.to_be_bytes());
        buf.push(self.hw_len);
        buf.push(self.proto_len);
        buf.extend_from_slice(&self.operation.as_u16().to_be_bytes());
        buf.extend_from_slice(self.sender_hw.as_bytes());
        buf.extend_from_slice(&self.sender_ip);
        buf.extend_from_slice(self.target_hw.as_bytes());
        buf.extend_from_slice(&self.target_ip);
        buf
    }
}

/// Create an ARP request packet.
///
/// The target hardware address is set to all zeros (unknown).
pub fn create_request(sender_mac: MacAddress, sender_ip: [u8; 4], target_ip: [u8; 4]) -> ArpPacket {
    ArpPacket {
        hardware_type: 1,      // Ethernet
        protocol_type: 0x0800, // IPv4
        hw_len: 6,
        proto_len: 4,
        operation: ArpOperation::Request,
        sender_hw: sender_mac,
        sender_ip,
        target_hw: MacAddress::new(0, 0, 0, 0, 0, 0),
        target_ip,
    }
}

/// Create an ARP reply packet.
pub fn create_reply(
    sender_mac: MacAddress,
    sender_ip: [u8; 4],
    target_mac: MacAddress,
    target_ip: [u8; 4],
) -> ArpPacket {
    ArpPacket {
        hardware_type: 1,      // Ethernet
        protocol_type: 0x0800, // IPv4
        hw_len: 6,
        proto_len: 4,
        operation: ArpOperation::Reply,
        sender_hw: sender_mac,
        sender_ip,
        target_hw: target_mac,
        target_ip,
    }
}

// ---------------------------------------------------------------------------
// ArpEntry / ArpTable
// ---------------------------------------------------------------------------

/// A single entry in the ARP table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArpEntry {
    /// IPv4 address.
    pub ip: [u8; 4],
    /// Resolved MAC address.
    pub mac: MacAddress,
    /// Timestamp when the entry was created or last updated (monotonic ticks).
    pub timestamp: u64,
}

/// An ARP table mapping IPv4 addresses to MAC addresses.
#[derive(Debug, Clone)]
pub struct ArpTable {
    entries: Vec<ArpEntry>,
}

impl Default for ArpTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ArpTable {
    /// Maximum number of entries in the ARP table.
    pub const MAX_ENTRIES: usize = 256;

    /// Create a new, empty ARP table.
    pub fn new() -> Self {
        ArpTable {
            entries: Vec::new(),
        }
    }

    /// Look up a MAC address by IPv4 address.
    pub fn lookup(&self, ip: &[u8; 4]) -> Option<&MacAddress> {
        self.entries.iter().find(|e| &e.ip == ip).map(|e| &e.mac)
    }

    /// Insert or update an entry. If the IP already exists its MAC and
    /// timestamp are updated. Returns [`NetError::TableFull`] if the table is
    /// at capacity and the IP is not already present.
    pub fn insert(&mut self, ip: [u8; 4], mac: MacAddress, timestamp: u64) -> Result<(), NetError> {
        // Update existing entry if present.
        for entry in self.entries.iter_mut() {
            if entry.ip == ip {
                entry.mac = mac;
                entry.timestamp = timestamp;
                return Ok(());
            }
        }

        if self.entries.len() >= Self::MAX_ENTRIES {
            return Err(NetError::TableFull);
        }

        self.entries.push(ArpEntry { ip, mac, timestamp });
        Ok(())
    }

    /// Remove the entry for the given IP. Returns `true` if an entry was
    /// removed, `false` if the IP was not found.
    pub fn remove(&mut self, ip: &[u8; 4]) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| &e.ip != ip);
        self.entries.len() < before
    }

    /// Return the number of entries in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the table contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` if the table has reached [`Self::MAX_ENTRIES`].
    pub fn is_full(&self) -> bool {
        self.entries.len() >= Self::MAX_ENTRIES
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample_request_bytes() -> Vec<u8> {
        let pkt = create_request(
            MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF),
            [192, 168, 1, 10],
            [192, 168, 1, 1],
        );
        pkt.serialize()
    }

    #[test]
    fn test_arp_operation_from_u16() {
        assert_eq!(ArpOperation::from_u16(1), Ok(ArpOperation::Request));
        assert_eq!(ArpOperation::from_u16(2), Ok(ArpOperation::Reply));
        assert_eq!(ArpOperation::from_u16(0), Err(NetError::InvalidProtocol));
        assert_eq!(ArpOperation::from_u16(3), Err(NetError::InvalidProtocol));
    }

    #[test]
    fn test_parse_request() {
        let data = sample_request_bytes();
        let pkt = ArpPacket::parse(&data).unwrap();
        assert_eq!(pkt.hardware_type, 1);
        assert_eq!(pkt.protocol_type, 0x0800);
        assert_eq!(pkt.hw_len, 6);
        assert_eq!(pkt.proto_len, 4);
        assert_eq!(pkt.operation, ArpOperation::Request);
        assert_eq!(
            pkt.sender_hw,
            MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF)
        );
        assert_eq!(pkt.sender_ip, [192, 168, 1, 10]);
        assert_eq!(pkt.target_hw, MacAddress::new(0, 0, 0, 0, 0, 0));
        assert_eq!(pkt.target_ip, [192, 168, 1, 1]);
    }

    #[test]
    fn test_parse_reply() {
        let pkt = create_reply(
            MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55),
            [10, 0, 0, 1],
            MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF),
            [10, 0, 0, 2],
        );
        let data = pkt.serialize();
        let parsed = ArpPacket::parse(&data).unwrap();
        assert_eq!(parsed.operation, ArpOperation::Reply);
        assert_eq!(parsed, pkt);
    }

    #[test]
    fn test_parse_too_short() {
        let data = vec![0u8; 27]; // One byte short
        assert_eq!(ArpPacket::parse(&data), Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let pkt = create_request(
            MacAddress::new(0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01),
            [172, 16, 0, 100],
            [172, 16, 0, 1],
        );
        let bytes = pkt.serialize();
        assert_eq!(bytes.len(), ArpPacket::PACKET_LEN);
        let parsed = ArpPacket::parse(&bytes).unwrap();
        assert_eq!(parsed, pkt);
    }

    #[test]
    fn test_create_request_fields() {
        let pkt = create_request(
            MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55),
            [192, 168, 0, 1],
            [192, 168, 0, 2],
        );
        assert_eq!(pkt.operation, ArpOperation::Request);
        assert_eq!(pkt.target_hw, MacAddress::new(0, 0, 0, 0, 0, 0));
        assert_eq!(pkt.hardware_type, 1);
        assert_eq!(pkt.protocol_type, 0x0800);
    }

    #[test]
    fn test_create_reply_fields() {
        let pkt = create_reply(
            MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55),
            [10, 0, 0, 1],
            MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF),
            [10, 0, 0, 2],
        );
        assert_eq!(pkt.operation, ArpOperation::Reply);
        assert_eq!(
            pkt.target_hw,
            MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF)
        );
    }

    #[test]
    fn test_table_insert_and_lookup() {
        let mut table = ArpTable::new();
        let mac = MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55);
        let ip = [192, 168, 1, 1];
        table.insert(ip, mac, 1000).unwrap();
        assert_eq!(table.lookup(&ip), Some(&mac));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_table_update_existing() {
        let mut table = ArpTable::new();
        let mac1 = MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55);
        let mac2 = MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF);
        let ip = [10, 0, 0, 1];
        table.insert(ip, mac1, 100).unwrap();
        table.insert(ip, mac2, 200).unwrap();
        assert_eq!(table.lookup(&ip), Some(&mac2));
        assert_eq!(table.len(), 1); // Should not have grown
    }

    #[test]
    fn test_table_remove() {
        let mut table = ArpTable::new();
        let mac = MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55);
        let ip = [192, 168, 1, 1];
        table.insert(ip, mac, 1000).unwrap();
        assert!(table.remove(&ip));
        assert_eq!(table.lookup(&ip), None);
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_table_remove_nonexistent() {
        let mut table = ArpTable::new();
        assert!(!table.remove(&[1, 2, 3, 4]));
    }

    #[test]
    fn test_table_lookup_missing() {
        let table = ArpTable::new();
        assert_eq!(table.lookup(&[10, 0, 0, 1]), None);
    }

    #[test]
    fn test_table_full() {
        let mut table = ArpTable::new();
        for i in 0..ArpTable::MAX_ENTRIES {
            let ip = [((i >> 8) & 0xFF) as u8, (i & 0xFF) as u8, 0, 1];
            let mac = MacAddress::new(0, 0, 0, 0, (i >> 8) as u8, (i & 0xFF) as u8);
            table.insert(ip, mac, i as u64).unwrap();
        }
        assert!(table.is_full());
        assert_eq!(table.len(), ArpTable::MAX_ENTRIES);

        // One more should fail
        let result = table.insert([255, 255, 255, 255], MacAddress::BROADCAST, 9999);
        assert_eq!(result, Err(NetError::TableFull));
    }

    #[test]
    fn test_packet_len_constant() {
        assert_eq!(ArpPacket::PACKET_LEN, 28);
    }
}
