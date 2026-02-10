// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ICMPv4 and ICMPv6 packet handling.
//!
//! Provides enums for message types, an [`IcmpPacket`] structure for
//! parsing and serializing ICMP messages, and helpers for constructing
//! Echo Request / Echo Reply packets.

use alloc::vec::Vec;

use crate::NetError;

// ---------------------------------------------------------------------------
// ICMPv4 types
// ---------------------------------------------------------------------------

/// ICMPv4 message types (subset used by SmallAIOS).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IcmpType {
    /// Echo Reply (type 0).
    EchoReply = 0,
    /// Destination Unreachable (type 3).
    DestinationUnreachable = 3,
    /// Echo Request (type 8).
    EchoRequest = 8,
    /// Time Exceeded (type 11).
    TimeExceeded = 11,
}

impl IcmpType {
    /// Try to convert a raw `u8` into an [`IcmpType`].
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::EchoReply),
            3 => Some(Self::DestinationUnreachable),
            8 => Some(Self::EchoRequest),
            11 => Some(Self::TimeExceeded),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ICMPv6 types
// ---------------------------------------------------------------------------

/// ICMPv6 message types (subset used by SmallAIOS).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Icmpv6Type {
    /// Destination Unreachable (type 1).
    DestUnreachable = 1,
    /// Packet Too Big (type 2).
    PacketTooBig = 2,
    /// Time Exceeded (type 3).
    TimeExceeded = 3,
    /// Echo Request (type 128).
    EchoRequest = 128,
    /// Echo Reply (type 129).
    EchoReply = 129,
    /// Router Solicitation (type 133).
    RouterSolicitation = 133,
    /// Router Advertisement (type 134).
    RouterAdvertisement = 134,
    /// Neighbor Solicitation (type 135).
    NeighborSolicitation = 135,
    /// Neighbor Advertisement (type 136).
    NeighborAdvertisement = 136,
}

impl Icmpv6Type {
    /// Try to convert a raw `u8` into an [`Icmpv6Type`].
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::DestUnreachable),
            2 => Some(Self::PacketTooBig),
            3 => Some(Self::TimeExceeded),
            128 => Some(Self::EchoRequest),
            129 => Some(Self::EchoReply),
            133 => Some(Self::RouterSolicitation),
            134 => Some(Self::RouterAdvertisement),
            135 => Some(Self::NeighborSolicitation),
            136 => Some(Self::NeighborAdvertisement),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ICMP packet
// ---------------------------------------------------------------------------

/// A parsed ICMP (v4 or v6) packet.
///
/// The fixed header is 4 bytes: type (1), code (1), checksum (2).
/// Everything after the fixed header is stored in `body`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IcmpPacket {
    /// Message type (interpretation depends on v4 vs v6 context).
    pub msg_type: u8,
    /// Sub-type code.
    pub code: u8,
    /// Internet checksum over the entire ICMP message.
    pub checksum: u16,
    /// Payload bytes following the 4-byte fixed header.
    pub body: Vec<u8>,
}

impl IcmpPacket {
    /// Size of the fixed ICMP header (type + code + checksum).
    pub const HEADER_LEN: usize = 4;

    /// Parse an ICMP packet from raw bytes.
    ///
    /// The input must contain at least [`HEADER_LEN`](Self::HEADER_LEN)
    /// bytes.
    pub fn parse(data: &[u8]) -> Result<IcmpPacket, NetError> {
        if data.len() < Self::HEADER_LEN {
            return Err(NetError::PacketTooShort);
        }

        Ok(IcmpPacket {
            msg_type: data[0],
            code: data[1],
            checksum: u16::from_be_bytes([data[2], data[3]]),
            body: data[Self::HEADER_LEN..].to_vec(),
        })
    }

    /// Serialize the packet into a byte vector.
    ///
    /// The checksum field is written as-is; call [`icmp_checksum`] first if
    /// you need to recompute it.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::HEADER_LEN + self.body.len());
        buf.push(self.msg_type);
        buf.push(self.code);
        buf.extend_from_slice(&self.checksum.to_be_bytes());
        buf.extend_from_slice(&self.body);
        buf
    }
}

// ---------------------------------------------------------------------------
// Echo helpers
// ---------------------------------------------------------------------------

/// Create an ICMPv4 Echo Request packet.
///
/// The body is laid out as: identifier (2 bytes, BE), sequence number
/// (2 bytes, BE), followed by the supplied `data`. The checksum is
/// computed automatically.
pub fn create_echo_request(id: u16, seq: u16, data: &[u8]) -> IcmpPacket {
    let mut body = Vec::with_capacity(4 + data.len());
    body.extend_from_slice(&id.to_be_bytes());
    body.extend_from_slice(&seq.to_be_bytes());
    body.extend_from_slice(data);

    let mut pkt = IcmpPacket {
        msg_type: IcmpType::EchoRequest as u8,
        code: 0,
        checksum: 0,
        body,
    };

    // Compute checksum over the entire serialized message.
    let raw = pkt.serialize();
    pkt.checksum = icmp_checksum(&raw);
    pkt
}

/// Create an ICMPv4 Echo Reply packet.
///
/// Layout and checksum handling are identical to [`create_echo_request`].
pub fn create_echo_reply(id: u16, seq: u16, data: &[u8]) -> IcmpPacket {
    let mut body = Vec::with_capacity(4 + data.len());
    body.extend_from_slice(&id.to_be_bytes());
    body.extend_from_slice(&seq.to_be_bytes());
    body.extend_from_slice(data);

    let mut pkt = IcmpPacket {
        msg_type: IcmpType::EchoReply as u8,
        code: 0,
        checksum: 0,
        body,
    };

    let raw = pkt.serialize();
    pkt.checksum = icmp_checksum(&raw);
    pkt
}

// ---------------------------------------------------------------------------
// Checksum
// ---------------------------------------------------------------------------

/// Compute the Internet checksum (RFC 1071) over `data`.
///
/// The input should be the entire ICMP message (header + body) with the
/// checksum field set to zero.
pub fn icmp_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let mut i = 0;

    // Sum 16-bit words.
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }

    // If odd length, pad the last byte with a zero byte.
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }

    // Fold 32-bit accumulator into 16 bits.
    while (sum >> 16) != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }

    !(sum as u16)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // -- IcmpType --

    #[test]
    fn test_icmpv4_type_values() {
        assert_eq!(IcmpType::EchoReply as u8, 0);
        assert_eq!(IcmpType::DestinationUnreachable as u8, 3);
        assert_eq!(IcmpType::EchoRequest as u8, 8);
        assert_eq!(IcmpType::TimeExceeded as u8, 11);
    }

    #[test]
    fn test_icmpv4_type_from_u8() {
        assert_eq!(IcmpType::from_u8(8), Some(IcmpType::EchoRequest));
        assert_eq!(IcmpType::from_u8(0), Some(IcmpType::EchoReply));
        assert_eq!(IcmpType::from_u8(99), None);
    }

    // -- Icmpv6Type --

    #[test]
    fn test_icmpv6_type_values() {
        assert_eq!(Icmpv6Type::DestUnreachable as u8, 1);
        assert_eq!(Icmpv6Type::PacketTooBig as u8, 2);
        assert_eq!(Icmpv6Type::TimeExceeded as u8, 3);
        assert_eq!(Icmpv6Type::EchoRequest as u8, 128);
        assert_eq!(Icmpv6Type::EchoReply as u8, 129);
        assert_eq!(Icmpv6Type::RouterSolicitation as u8, 133);
        assert_eq!(Icmpv6Type::RouterAdvertisement as u8, 134);
        assert_eq!(Icmpv6Type::NeighborSolicitation as u8, 135);
        assert_eq!(Icmpv6Type::NeighborAdvertisement as u8, 136);
    }

    #[test]
    fn test_icmpv6_type_from_u8() {
        assert_eq!(Icmpv6Type::from_u8(128), Some(Icmpv6Type::EchoRequest));
        assert_eq!(Icmpv6Type::from_u8(136), Some(Icmpv6Type::NeighborAdvertisement));
        assert_eq!(Icmpv6Type::from_u8(200), None);
    }

    // -- IcmpPacket --

    #[test]
    fn test_parse_valid_packet() {
        // Echo request: type=8, code=0, checksum=0x1234, body = [0x00, 0x01, 0x00, 0x01]
        let data = [8, 0, 0x12, 0x34, 0x00, 0x01, 0x00, 0x01];
        let pkt = IcmpPacket::parse(&data).unwrap();
        assert_eq!(pkt.msg_type, 8);
        assert_eq!(pkt.code, 0);
        assert_eq!(pkt.checksum, 0x1234);
        assert_eq!(pkt.body, vec![0x00, 0x01, 0x00, 0x01]);
    }

    #[test]
    fn test_parse_too_short() {
        let data = [8, 0]; // only 2 bytes
        assert!(IcmpPacket::parse(&data).is_err());
    }

    #[test]
    fn test_parse_header_only() {
        let data = [0, 0, 0x00, 0x00];
        let pkt = IcmpPacket::parse(&data).unwrap();
        assert_eq!(pkt.msg_type, 0);
        assert!(pkt.body.is_empty());
    }

    #[test]
    fn test_serialize_roundtrip() {
        let original = IcmpPacket {
            msg_type: 8,
            code: 0,
            checksum: 0xabcd,
            body: vec![1, 2, 3, 4],
        };
        let bytes = original.serialize();
        let parsed = IcmpPacket::parse(&bytes).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_create_echo_request() {
        let pkt = create_echo_request(0x1234, 1, &[0xaa, 0xbb]);
        assert_eq!(pkt.msg_type, IcmpType::EchoRequest as u8);
        assert_eq!(pkt.code, 0);
        // id=0x1234 seq=0x0001 data=0xaa,0xbb
        assert_eq!(pkt.body.len(), 6);
        assert_eq!(&pkt.body[0..2], &0x1234u16.to_be_bytes());
        assert_eq!(&pkt.body[2..4], &1u16.to_be_bytes());
        assert_eq!(&pkt.body[4..6], &[0xaa, 0xbb]);
        // Verify checksum is valid
        let raw = pkt.serialize();
        assert_eq!(icmp_checksum(&raw), 0); // complementary checksum should yield 0 on valid data
    }

    #[test]
    fn test_create_echo_reply() {
        let pkt = create_echo_reply(0x5678, 2, &[0xcc]);
        assert_eq!(pkt.msg_type, IcmpType::EchoReply as u8);
        assert_eq!(pkt.code, 0);
        assert_eq!(&pkt.body[0..2], &0x5678u16.to_be_bytes());
        assert_eq!(&pkt.body[2..4], &2u16.to_be_bytes());
        assert_eq!(&pkt.body[4..5], &[0xcc]);
        // Verify checksum
        let raw = pkt.serialize();
        assert_eq!(icmp_checksum(&raw), 0);
    }

    #[test]
    fn test_checksum_zero_data() {
        // All-zero input should produce 0xffff.
        let data = [0u8; 8];
        assert_eq!(icmp_checksum(&data), 0xffff);
    }

    #[test]
    fn test_checksum_odd_length() {
        // Odd-length input should still compute correctly.
        let data = [0x00, 0x01, 0x00, 0x02, 0xaa];
        let cksum = icmp_checksum(&data);
        // Should produce a valid non-zero checksum
        assert_ne!(cksum, 0);
        // Deterministic: same input produces same checksum
        assert_eq!(cksum, icmp_checksum(&data));
    }

    #[test]
    fn test_checksum_known_value() {
        // ICMPv4 Echo Request: type=8, code=0, cksum=0, id=0x0001, seq=0x0001
        let data = [0x08, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01];
        let cksum = icmp_checksum(&data);
        // Re-verify: replace checksum in packet and confirm overall checksum is zero
        let mut pkt = data;
        pkt[2] = (cksum >> 8) as u8;
        pkt[3] = cksum as u8;
        assert_eq!(icmp_checksum(&pkt), 0);
    }

    #[test]
    fn test_header_len_constant() {
        assert_eq!(IcmpPacket::HEADER_LEN, 4);
    }
}
