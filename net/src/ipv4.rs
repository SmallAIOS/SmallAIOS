// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! IPv4 protocol handling for SmallAIOS.
//!
//! Provides [`Ipv4Addr`] for IPv4 addresses, [`Ipv4Header`] for parsing and
//! serializing IPv4 packet headers, and Internet checksum utilities.

use core::fmt;

use crate::NetError;

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

/// Minimum IPv4 header length in bytes (no options).
pub const HEADER_MIN_LEN: usize = 20;

/// Protocol number for TCP.
pub const PROTOCOL_TCP: u8 = 6;

/// Protocol number for UDP.
pub const PROTOCOL_UDP: u8 = 17;

/// Protocol number for ICMP.
pub const PROTOCOL_ICMP: u8 = 1;

// ---------------------------------------------------------------------------
// Ipv4Addr
// ---------------------------------------------------------------------------

/// An IPv4 address (4 octets).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ipv4Addr([u8; 4]);

impl Ipv4Addr {
    /// The loopback address (127.0.0.1).
    pub const LOOPBACK: Ipv4Addr = Ipv4Addr([127, 0, 0, 1]);

    /// The broadcast address (255.255.255.255).
    pub const BROADCAST: Ipv4Addr = Ipv4Addr([255, 255, 255, 255]);

    /// The unspecified / any address (0.0.0.0).
    pub const ANY: Ipv4Addr = Ipv4Addr([0, 0, 0, 0]);

    /// Create a new [`Ipv4Addr`] from four octets.
    pub const fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        Ipv4Addr([a, b, c, d])
    }

    /// Create an [`Ipv4Addr`] from a 4-byte slice.
    ///
    /// Returns [`NetError::InvalidAddress`] if `bytes` is not exactly 4 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NetError> {
        if bytes.len() != 4 {
            return Err(NetError::InvalidAddress);
        }
        let mut addr = [0u8; 4];
        addr.copy_from_slice(bytes);
        Ok(Ipv4Addr(addr))
    }

    /// Return the address as a byte slice.
    pub fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }

    /// Returns `true` if this is a loopback address (127.0.0.0/8).
    pub fn is_loopback(&self) -> bool {
        self.0[0] == 127
    }

    /// Returns `true` if this is the limited broadcast address
    /// (255.255.255.255).
    pub fn is_broadcast(&self) -> bool {
        self.0 == [255, 255, 255, 255]
    }

    /// Returns `true` if this is a multicast address (224.0.0.0/4).
    pub fn is_multicast(&self) -> bool {
        self.0[0] >= 224 && self.0[0] <= 239
    }

    /// Returns `true` if this is a private address (RFC 1918):
    /// - 10.0.0.0/8
    /// - 172.16.0.0/12
    /// - 192.168.0.0/16
    pub fn is_private(&self) -> bool {
        match self.0[0] {
            10 => true,
            172 => self.0[1] >= 16 && self.0[1] <= 31,
            192 => self.0[1] == 168,
            _ => false,
        }
    }
}

impl fmt::Display for Ipv4Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}.{}", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

impl fmt::Debug for Ipv4Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ipv4Addr({}.{}.{}.{})",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

// ---------------------------------------------------------------------------
// Internet Checksum
// ---------------------------------------------------------------------------

/// Compute the Internet checksum (RFC 1071) over `data`.
///
/// Returns the ones-complement of the ones-complement 16-bit sum.
pub fn checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let mut i = 0;

    // Sum 16-bit words
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }

    // If odd number of bytes, pad the last byte with zero
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }

    // Fold 32-bit sum into 16 bits
    while (sum >> 16) != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    !(sum as u16)
}

/// Verify the Internet checksum of a header.
///
/// A valid header (with checksum included) sums to zero.
pub fn verify_checksum(header_bytes: &[u8]) -> bool {
    checksum(header_bytes) == 0
}

// ---------------------------------------------------------------------------
// Ipv4Header
// ---------------------------------------------------------------------------

/// An IPv4 packet header (RFC 791).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ipv4Header {
    /// IP version (must be 4).
    pub version: u8,
    /// Internet Header Length in 32-bit words (minimum 5).
    pub ihl: u8,
    /// Differentiated Services Code Point.
    pub dscp: u8,
    /// Total length of the IP packet (header + payload) in bytes.
    pub total_length: u16,
    /// Identification field for fragmentation reassembly.
    pub identification: u16,
    /// Flags (3 bits: Reserved, DF, MF).
    pub flags: u8,
    /// Fragment offset (13 bits, in units of 8 bytes).
    pub fragment_offset: u16,
    /// Time to Live.
    pub ttl: u8,
    /// Upper-layer protocol number (e.g., [`PROTOCOL_TCP`]).
    pub protocol: u8,
    /// Header checksum.
    pub checksum: u16,
    /// Source IPv4 address.
    pub src_addr: Ipv4Addr,
    /// Destination IPv4 address.
    pub dst_addr: Ipv4Addr,
}

impl Ipv4Header {
    /// Parse an IPv4 header from raw bytes.
    ///
    /// Returns the parsed header and a slice referencing the payload (data
    /// after the header). Validates minimum length, version == 4, and the
    /// header checksum.
    pub fn parse(data: &[u8]) -> Result<(Self, &[u8]), NetError> {
        if data.len() < HEADER_MIN_LEN {
            return Err(NetError::PacketTooShort);
        }

        let version = data[0] >> 4;
        if version != 4 {
            return Err(NetError::InvalidHeader);
        }

        let ihl = data[0] & 0x0F;
        let header_len = (ihl as usize) * 4;
        if header_len < HEADER_MIN_LEN || data.len() < header_len {
            return Err(NetError::InvalidHeader);
        }

        // Verify checksum over the full header (including options)
        if !verify_checksum(&data[..header_len]) {
            return Err(NetError::ChecksumMismatch);
        }

        let dscp = data[1];
        let total_length = u16::from_be_bytes([data[2], data[3]]);
        let identification = u16::from_be_bytes([data[4], data[5]]);
        let flags = data[6] >> 5;
        let fragment_offset = u16::from_be_bytes([data[6] & 0x1F, data[7]]);
        let ttl = data[8];
        let protocol = data[9];
        let hdr_checksum = u16::from_be_bytes([data[10], data[11]]);
        let src_addr = Ipv4Addr::from_bytes(&data[12..16])?;
        let dst_addr = Ipv4Addr::from_bytes(&data[16..20])?;

        let total_len = total_length as usize;
        let payload_end = if total_len > data.len() {
            data.len()
        } else {
            total_len
        };
        let payload_start = header_len;
        let payload = if payload_start <= payload_end {
            &data[payload_start..payload_end]
        } else {
            &data[payload_start..payload_start]
        };

        Ok((
            Ipv4Header {
                version,
                ihl,
                dscp,
                total_length,
                identification,
                flags,
                fragment_offset,
                ttl,
                protocol,
                checksum: hdr_checksum,
                src_addr,
                dst_addr,
            },
            payload,
        ))
    }

    /// Serialize the header into a 20-byte array.
    ///
    /// The checksum field is recomputed from the serialized bytes.
    pub fn serialize(&self) -> [u8; 20] {
        let mut buf = [0u8; 20];

        buf[0] = (self.version << 4) | (self.ihl & 0x0F);
        buf[1] = self.dscp;
        buf[2..4].copy_from_slice(&self.total_length.to_be_bytes());
        buf[4..6].copy_from_slice(&self.identification.to_be_bytes());
        buf[6] = (self.flags << 5) | ((self.fragment_offset >> 8) as u8 & 0x1F);
        buf[7] = (self.fragment_offset & 0xFF) as u8;
        buf[8] = self.ttl;
        buf[9] = self.protocol;
        // Checksum field initially zero for computation
        buf[10] = 0;
        buf[11] = 0;
        buf[12..16].copy_from_slice(self.src_addr.as_bytes());
        buf[16..20].copy_from_slice(self.dst_addr.as_bytes());

        // Compute and insert checksum
        let cksum = checksum(&buf);
        buf[10..12].copy_from_slice(&cksum.to_be_bytes());

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

    /// Build a minimal valid IPv4 header for testing.
    fn make_test_header() -> Ipv4Header {
        Ipv4Header {
            version: 4,
            ihl: 5,
            dscp: 0,
            total_length: 40,
            identification: 0x1234,
            flags: 0b010, // Don't Fragment
            fragment_offset: 0,
            ttl: 64,
            protocol: PROTOCOL_TCP,
            checksum: 0, // Will be computed during serialize
            src_addr: Ipv4Addr::new(192, 168, 1, 10),
            dst_addr: Ipv4Addr::new(192, 168, 1, 1),
        }
    }

    #[test]
    fn test_ipv4_addr_new() {
        let addr = Ipv4Addr::new(10, 0, 0, 1);
        assert_eq!(addr.as_bytes(), &[10, 0, 0, 1]);
    }

    #[test]
    fn test_ipv4_addr_from_bytes_valid() {
        let addr = Ipv4Addr::from_bytes(&[172, 16, 0, 1]).unwrap();
        assert_eq!(addr, Ipv4Addr::new(172, 16, 0, 1));
    }

    #[test]
    fn test_ipv4_addr_from_bytes_invalid() {
        assert_eq!(
            Ipv4Addr::from_bytes(&[1, 2, 3]),
            Err(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_ipv4_addr_display() {
        assert_eq!(format!("{}", Ipv4Addr::new(192, 168, 1, 1)), "192.168.1.1");
        assert_eq!(format!("{}", Ipv4Addr::LOOPBACK), "127.0.0.1");
    }

    #[test]
    fn test_ipv4_addr_loopback() {
        assert!(Ipv4Addr::LOOPBACK.is_loopback());
        assert!(Ipv4Addr::new(127, 0, 0, 1).is_loopback());
        assert!(Ipv4Addr::new(127, 255, 255, 255).is_loopback());
        assert!(!Ipv4Addr::new(128, 0, 0, 1).is_loopback());
    }

    #[test]
    fn test_ipv4_addr_broadcast() {
        assert!(Ipv4Addr::BROADCAST.is_broadcast());
        assert!(Ipv4Addr::new(255, 255, 255, 255).is_broadcast());
        assert!(!Ipv4Addr::new(255, 255, 255, 0).is_broadcast());
    }

    #[test]
    fn test_ipv4_addr_multicast() {
        assert!(Ipv4Addr::new(224, 0, 0, 1).is_multicast());
        assert!(Ipv4Addr::new(239, 255, 255, 255).is_multicast());
        assert!(!Ipv4Addr::new(223, 255, 255, 255).is_multicast());
        assert!(!Ipv4Addr::new(240, 0, 0, 0).is_multicast());
    }

    #[test]
    fn test_ipv4_addr_private() {
        // 10.0.0.0/8
        assert!(Ipv4Addr::new(10, 0, 0, 1).is_private());
        assert!(Ipv4Addr::new(10, 255, 255, 255).is_private());
        // 172.16.0.0/12
        assert!(Ipv4Addr::new(172, 16, 0, 1).is_private());
        assert!(Ipv4Addr::new(172, 31, 255, 255).is_private());
        assert!(!Ipv4Addr::new(172, 15, 0, 1).is_private());
        assert!(!Ipv4Addr::new(172, 32, 0, 1).is_private());
        // 192.168.0.0/16
        assert!(Ipv4Addr::new(192, 168, 0, 1).is_private());
        assert!(Ipv4Addr::new(192, 168, 255, 255).is_private());
        assert!(!Ipv4Addr::new(192, 167, 0, 1).is_private());
        // Public
        assert!(!Ipv4Addr::new(8, 8, 8, 8).is_private());
    }

    #[test]
    fn test_ipv4_addr_constants() {
        assert_eq!(Ipv4Addr::LOOPBACK, Ipv4Addr::new(127, 0, 0, 1));
        assert_eq!(Ipv4Addr::BROADCAST, Ipv4Addr::new(255, 255, 255, 255));
        assert_eq!(Ipv4Addr::ANY, Ipv4Addr::new(0, 0, 0, 0));
    }

    #[test]
    fn test_checksum_rfc_example() {
        // RFC 1071 example: verify that a correct header sums to zero
        let header = make_test_header();
        let bytes = header.serialize();
        assert!(verify_checksum(&bytes));
    }

    #[test]
    fn test_checksum_corruption() {
        let header = make_test_header();
        let mut bytes = header.serialize();
        // Corrupt one byte
        bytes[5] ^= 0xFF;
        assert!(!verify_checksum(&bytes));
    }

    #[test]
    fn test_parse_valid_packet() {
        let header = make_test_header();
        let hdr_bytes = header.serialize();
        // Append 20 bytes of "payload" to match total_length = 40
        let mut packet = hdr_bytes.to_vec();
        packet.extend_from_slice(&[0xABu8; 20]);

        let (parsed, payload) = Ipv4Header::parse(&packet).unwrap();
        assert_eq!(parsed.version, 4);
        assert_eq!(parsed.ihl, 5);
        assert_eq!(parsed.ttl, 64);
        assert_eq!(parsed.protocol, PROTOCOL_TCP);
        assert_eq!(parsed.src_addr, Ipv4Addr::new(192, 168, 1, 10));
        assert_eq!(parsed.dst_addr, Ipv4Addr::new(192, 168, 1, 1));
        assert_eq!(payload.len(), 20);
        assert!(payload.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn test_parse_too_short() {
        let data = [0u8; 19];
        assert_eq!(Ipv4Header::parse(&data), Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_parse_wrong_version() {
        let mut data = [0u8; 20];
        data[0] = 0x60; // version 6
        assert_eq!(Ipv4Header::parse(&data), Err(NetError::InvalidHeader));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let header = make_test_header();
        let bytes = header.serialize();
        // Append payload to match total_length
        let mut packet = bytes.to_vec();
        packet.extend_from_slice(&[0u8; 20]);

        let (parsed, _payload) = Ipv4Header::parse(&packet).unwrap();
        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.ihl, header.ihl);
        assert_eq!(parsed.total_length, header.total_length);
        assert_eq!(parsed.identification, header.identification);
        assert_eq!(parsed.ttl, header.ttl);
        assert_eq!(parsed.protocol, header.protocol);
        assert_eq!(parsed.src_addr, header.src_addr);
        assert_eq!(parsed.dst_addr, header.dst_addr);
    }

    #[test]
    fn test_protocol_constants() {
        assert_eq!(PROTOCOL_TCP, 6);
        assert_eq!(PROTOCOL_UDP, 17);
        assert_eq!(PROTOCOL_ICMP, 1);
        assert_eq!(HEADER_MIN_LEN, 20);
    }
}
