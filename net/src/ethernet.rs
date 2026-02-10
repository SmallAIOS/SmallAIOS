// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Ethernet frame handling for SmallAIOS.
//!
//! Provides [`MacAddress`] for IEEE 802.3 MAC addresses and [`EthernetFrame`]
//! for parsing and serializing Ethernet II frames.

use alloc::vec::Vec;
use core::fmt;

use crate::NetError;

// ---------------------------------------------------------------------------
// EtherType constants
// ---------------------------------------------------------------------------

/// EtherType for IPv4 (0x0800).
pub const ETHERTYPE_IPV4: u16 = 0x0800;

/// EtherType for IPv6 (0x86DD).
pub const ETHERTYPE_IPV6: u16 = 0x86DD;

/// EtherType for ARP (0x0806).
pub const ETHERTYPE_ARP: u16 = 0x0806;

// ---------------------------------------------------------------------------
// MacAddress
// ---------------------------------------------------------------------------

/// IEEE 802.3 MAC address (6 octets).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacAddress([u8; 6]);

impl MacAddress {
    /// The broadcast MAC address (ff:ff:ff:ff:ff:ff).
    pub const BROADCAST: MacAddress = MacAddress([0xFF; 6]);

    /// Create a new [`MacAddress`] from six individual octets.
    pub const fn new(a: u8, b: u8, c: u8, d: u8, e: u8, f: u8) -> Self {
        MacAddress([a, b, c, d, e, f])
    }

    /// Create a [`MacAddress`] from a 6-byte slice.
    ///
    /// Returns [`NetError::InvalidAddress`] if `bytes` is not exactly 6 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NetError> {
        if bytes.len() != 6 {
            return Err(NetError::InvalidAddress);
        }
        let mut addr = [0u8; 6];
        addr.copy_from_slice(bytes);
        Ok(MacAddress(addr))
    }

    /// Return the address as a byte slice.
    pub fn as_bytes(&self) -> &[u8; 6] {
        &self.0
    }

    /// Returns `true` if this is the broadcast address (ff:ff:ff:ff:ff:ff).
    pub fn is_broadcast(&self) -> bool {
        self.0 == [0xFF; 6]
    }

    /// Returns `true` if this is a multicast address (bit 0 of first octet set,
    /// but not broadcast).
    pub fn is_multicast(&self) -> bool {
        (self.0[0] & 0x01) != 0 && !self.is_broadcast()
    }

    /// Returns `true` if this is a unicast address (bit 0 of first octet clear).
    pub fn is_unicast(&self) -> bool {
        (self.0[0] & 0x01) == 0
    }
}

impl fmt::Display for MacAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5]
        )
    }
}

impl fmt::Debug for MacAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MacAddress({:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x})",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5]
        )
    }
}

// ---------------------------------------------------------------------------
// EthernetFrame
// ---------------------------------------------------------------------------

/// An Ethernet II frame (IEEE 802.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EthernetFrame {
    /// Destination MAC address.
    pub dst_mac: MacAddress,
    /// Source MAC address.
    pub src_mac: MacAddress,
    /// EtherType field (e.g., [`ETHERTYPE_IPV4`]).
    pub ether_type: u16,
    /// Payload data (up to [`Self::MAX_PAYLOAD_LEN`] bytes).
    pub payload: Vec<u8>,
}

impl EthernetFrame {
    /// Size of the Ethernet header (dst + src + ethertype) in bytes.
    pub const HEADER_LEN: usize = 14;

    /// Minimum Ethernet frame length (including header, excluding FCS).
    pub const MIN_FRAME_LEN: usize = 60;

    /// Maximum Ethernet frame length (including header, excluding FCS).
    pub const MAX_FRAME_LEN: usize = 1514;

    /// Maximum payload length (MTU).
    pub const MAX_PAYLOAD_LEN: usize = 1500;

    /// Parse an Ethernet frame from raw bytes.
    ///
    /// `data` must be at least [`Self::HEADER_LEN`] (14) bytes. The payload is
    /// extracted from everything after the header.
    pub fn parse(data: &[u8]) -> Result<Self, NetError> {
        if data.len() < Self::HEADER_LEN {
            return Err(NetError::PacketTooShort);
        }

        let dst_mac = MacAddress::from_bytes(&data[0..6])?;
        let src_mac = MacAddress::from_bytes(&data[6..12])?;
        let ether_type = u16::from_be_bytes([data[12], data[13]]);
        let payload = Vec::from(&data[Self::HEADER_LEN..]);

        Ok(EthernetFrame {
            dst_mac,
            src_mac,
            ether_type,
            payload,
        })
    }

    /// Serialize the frame into a byte vector.
    ///
    /// The resulting vector contains the 14-byte header followed by the payload.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::HEADER_LEN + self.payload.len());
        buf.extend_from_slice(self.dst_mac.as_bytes());
        buf.extend_from_slice(self.src_mac.as_bytes());
        buf.extend_from_slice(&self.ether_type.to_be_bytes());
        buf.extend_from_slice(&self.payload);
        buf
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::vec;

    #[test]
    fn test_mac_address_new() {
        let mac = MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF);
        assert_eq!(mac.as_bytes(), &[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_mac_address_from_bytes_valid() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mac = MacAddress::from_bytes(&bytes).unwrap();
        assert_eq!(mac.as_bytes(), &bytes);
    }

    #[test]
    fn test_mac_address_from_bytes_invalid() {
        let short = [0x01, 0x02, 0x03];
        assert_eq!(
            MacAddress::from_bytes(&short),
            Err(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_mac_address_broadcast() {
        assert!(MacAddress::BROADCAST.is_broadcast());
        assert!(!MacAddress::BROADCAST.is_unicast());
        // Broadcast has bit 0 set but is_multicast() excludes broadcast
        assert!(!MacAddress::BROADCAST.is_multicast());
    }

    #[test]
    fn test_mac_address_multicast() {
        // Bit 0 of first octet set, not broadcast
        let mac = MacAddress::new(0x01, 0x00, 0x5E, 0x00, 0x00, 0x01);
        assert!(mac.is_multicast());
        assert!(!mac.is_broadcast());
        assert!(!mac.is_unicast());
    }

    #[test]
    fn test_mac_address_unicast() {
        let mac = MacAddress::new(0x00, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E);
        assert!(mac.is_unicast());
        assert!(!mac.is_multicast());
        assert!(!mac.is_broadcast());
    }

    #[test]
    fn test_mac_address_display() {
        let mac = MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF);
        assert_eq!(format!("{}", mac), "aa:bb:cc:dd:ee:ff");
    }

    #[test]
    fn test_mac_address_debug() {
        let mac = MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55);
        assert_eq!(format!("{:?}", mac), "MacAddress(00:11:22:33:44:55)");
    }

    #[test]
    fn test_ethertype_constants() {
        assert_eq!(ETHERTYPE_IPV4, 0x0800);
        assert_eq!(ETHERTYPE_IPV6, 0x86DD);
        assert_eq!(ETHERTYPE_ARP, 0x0806);
    }

    #[test]
    fn test_parse_valid_frame() {
        let mut data = vec![0u8; 20];
        // dst: 00:11:22:33:44:55
        data[0..6].copy_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        // src: AA:BB:CC:DD:EE:FF
        data[6..12].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        // ethertype: IPv4
        data[12..14].copy_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
        // payload: 6 bytes
        data[14..20].copy_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);

        let frame = EthernetFrame::parse(&data).unwrap();
        assert_eq!(
            frame.dst_mac,
            MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55)
        );
        assert_eq!(
            frame.src_mac,
            MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF)
        );
        assert_eq!(frame.ether_type, ETHERTYPE_IPV4);
        assert_eq!(frame.payload, vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
    }

    #[test]
    fn test_parse_too_short() {
        let data = [0u8; 13]; // Less than HEADER_LEN
        assert_eq!(EthernetFrame::parse(&data), Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let frame = EthernetFrame {
            dst_mac: MacAddress::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF),
            src_mac: MacAddress::new(0x00, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E),
            ether_type: ETHERTYPE_ARP,
            payload: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };

        let bytes = frame.serialize();
        let parsed = EthernetFrame::parse(&bytes).unwrap();
        assert_eq!(parsed, frame);
    }

    #[test]
    fn test_frame_constants() {
        assert_eq!(EthernetFrame::HEADER_LEN, 14);
        assert_eq!(EthernetFrame::MIN_FRAME_LEN, 60);
        assert_eq!(EthernetFrame::MAX_FRAME_LEN, 1514);
        assert_eq!(EthernetFrame::MAX_PAYLOAD_LEN, 1500);
    }

    #[test]
    fn test_parse_header_only() {
        // Exactly 14 bytes — valid frame with empty payload
        let mut data = [0u8; 14];
        data[12..14].copy_from_slice(&ETHERTYPE_IPV6.to_be_bytes());
        let frame = EthernetFrame::parse(&data).unwrap();
        assert_eq!(frame.ether_type, ETHERTYPE_IPV6);
        assert!(frame.payload.is_empty());
    }
}
