// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! IPv6 address handling and header parsing/serialization.
//!
//! Provides [`Ipv6Addr`] for address representation with EUI-64 link-local
//! derivation and solicited-node multicast, plus [`Ipv6Header`] for
//! parsing and serializing the fixed 40-byte IPv6 header.

use core::fmt;

use crate::ethernet::MacAddress;
use crate::NetError;

// ---------------------------------------------------------------------------
// Ipv6Addr
// ---------------------------------------------------------------------------

/// 128-bit IPv6 address.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ipv6Addr {
    octets: [u8; 16],
}

impl Ipv6Addr {
    /// The unspecified address `::`.
    pub const UNSPECIFIED: Self = Self { octets: [0; 16] };

    /// The loopback address `::1`.
    pub const LOOPBACK: Self = Self {
        octets: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    };

    /// Create an address from eight 16-bit groups.
    pub const fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Self {
        Self {
            octets: [
                (a >> 8) as u8,
                a as u8,
                (b >> 8) as u8,
                b as u8,
                (c >> 8) as u8,
                c as u8,
                (d >> 8) as u8,
                d as u8,
                (e >> 8) as u8,
                e as u8,
                (f >> 8) as u8,
                f as u8,
                (g >> 8) as u8,
                g as u8,
                (h >> 8) as u8,
                h as u8,
            ],
        }
    }

    /// Create an address from a 16-byte array.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self { octets: bytes }
    }

    /// Return the raw 16-byte representation.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.octets
    }

    /// Returns `true` for the loopback address `::1`.
    pub const fn is_loopback(&self) -> bool {
        let o = &self.octets;
        o[0] == 0
            && o[1] == 0
            && o[2] == 0
            && o[3] == 0
            && o[4] == 0
            && o[5] == 0
            && o[6] == 0
            && o[7] == 0
            && o[8] == 0
            && o[9] == 0
            && o[10] == 0
            && o[11] == 0
            && o[12] == 0
            && o[13] == 0
            && o[14] == 0
            && o[15] == 1
    }

    /// Returns `true` for multicast addresses (`ff00::/8`).
    pub const fn is_multicast(&self) -> bool {
        self.octets[0] == 0xff
    }

    /// Returns `true` for link-local unicast addresses (`fe80::/10`).
    pub const fn is_link_local(&self) -> bool {
        self.octets[0] == 0xfe && (self.octets[1] & 0xc0) == 0x80
    }

    /// Returns `true` for the unspecified address `::`.
    pub const fn is_unspecified(&self) -> bool {
        let o = &self.octets;
        o[0] == 0
            && o[1] == 0
            && o[2] == 0
            && o[3] == 0
            && o[4] == 0
            && o[5] == 0
            && o[6] == 0
            && o[7] == 0
            && o[8] == 0
            && o[9] == 0
            && o[10] == 0
            && o[11] == 0
            && o[12] == 0
            && o[13] == 0
            && o[14] == 0
            && o[15] == 0
    }

    /// Compute the solicited-node multicast address for this unicast address.
    ///
    /// Result: `ff02::1:ffXX:XXXX` where `XX:XXXX` are the low 24 bits.
    pub const fn solicited_node_multicast(addr: &Ipv6Addr) -> Self {
        Self {
            octets: [
                0xff, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0xff,
                addr.octets[13],
                addr.octets[14],
                addr.octets[15],
            ],
        }
    }

    /// Derive a link-local address from a MAC via Modified EUI-64.
    ///
    /// `fe80::` + modified-EUI-64 (insert `ff:fe` in the middle of the MAC
    /// and flip the Universal/Local bit in the first octet).
    pub fn link_local_from_mac(mac: &MacAddress) -> Self {
        let m = mac.as_bytes();
        // Flip the U/L bit (bit 1 of the first octet).
        let eui0 = m[0] ^ 0x02;
        Self {
            octets: [
                0xfe, 0x80, 0, 0, 0, 0, 0, 0, eui0, m[1], m[2], 0xff, 0xfe, m[3], m[4], m[5],
            ],
        }
    }
}

impl fmt::Display for Ipv6Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let o = &self.octets;
        for i in 0..8 {
            if i > 0 {
                write!(f, ":")?;
            }
            let group = u16::from_be_bytes([o[i * 2], o[i * 2 + 1]]);
            write!(f, "{:x}", group)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Ipv6Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ipv6Addr({})", self)
    }
}

// ---------------------------------------------------------------------------
// Ipv6Header
// ---------------------------------------------------------------------------

/// Fixed 40-byte IPv6 header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ipv6Header {
    /// IP version (must be 6).
    pub version: u8,
    /// Traffic class (DSCP + ECN).
    pub traffic_class: u8,
    /// Flow label (20 bits).
    pub flow_label: u32,
    /// Length of the payload following this header, in bytes.
    pub payload_length: u16,
    /// Identifies the type of the next header (e.g. TCP, UDP, ICMPv6).
    pub next_header: u8,
    /// Decremented by 1 at each forwarding node.
    pub hop_limit: u8,
    /// Source address.
    pub src_addr: Ipv6Addr,
    /// Destination address.
    pub dst_addr: Ipv6Addr,
}

impl Ipv6Header {
    /// Fixed header length in bytes.
    pub const HEADER_LEN: usize = 40;

    /// Next-header value for TCP.
    pub const NEXT_HEADER_TCP: u8 = 6;
    /// Next-header value for UDP.
    pub const NEXT_HEADER_UDP: u8 = 17;
    /// Next-header value for ICMPv6.
    pub const NEXT_HEADER_ICMPV6: u8 = 58;

    /// Parse an IPv6 header from a byte slice.
    ///
    /// Returns the parsed header and a sub-slice pointing to the payload.
    pub fn parse(data: &[u8]) -> Result<(Ipv6Header, &[u8]), NetError> {
        if data.len() < Self::HEADER_LEN {
            return Err(NetError::PacketTooShort);
        }

        let version = data[0] >> 4;
        if version != 6 {
            return Err(NetError::InvalidHeader);
        }

        let traffic_class = ((data[0] & 0x0f) << 4) | (data[1] >> 4);
        let flow_label =
            ((data[1] as u32 & 0x0f) << 16) | ((data[2] as u32) << 8) | (data[3] as u32);
        let payload_length = u16::from_be_bytes([data[4], data[5]]);
        let next_header = data[6];
        let hop_limit = data[7];

        let mut src = [0u8; 16];
        src.copy_from_slice(&data[8..24]);
        let mut dst = [0u8; 16];
        dst.copy_from_slice(&data[24..40]);

        let header = Ipv6Header {
            version,
            traffic_class,
            flow_label,
            payload_length,
            next_header,
            hop_limit,
            src_addr: Ipv6Addr::from_bytes(src),
            dst_addr: Ipv6Addr::from_bytes(dst),
        };

        let payload_end = Self::HEADER_LEN + payload_length as usize;
        let payload = if payload_end <= data.len() {
            &data[Self::HEADER_LEN..payload_end]
        } else {
            &data[Self::HEADER_LEN..]
        };

        Ok((header, payload))
    }

    /// Serialize this header into a 40-byte array.
    pub fn serialize(&self) -> [u8; 40] {
        let mut buf = [0u8; 40];

        // Version (4 bits) | Traffic class high (4 bits)
        buf[0] = (self.version << 4) | (self.traffic_class >> 4);
        // Traffic class low (4 bits) | Flow label high (4 bits)
        buf[1] = (self.traffic_class << 4) | ((self.flow_label >> 16) as u8 & 0x0f);
        // Flow label mid
        buf[2] = (self.flow_label >> 8) as u8;
        // Flow label low
        buf[3] = self.flow_label as u8;

        let pl = self.payload_length.to_be_bytes();
        buf[4] = pl[0];
        buf[5] = pl[1];
        buf[6] = self.next_header;
        buf[7] = self.hop_limit;

        buf[8..24].copy_from_slice(self.src_addr.as_bytes());
        buf[24..40].copy_from_slice(self.dst_addr.as_bytes());

        buf
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a minimal valid 40-byte IPv6 header.
    fn make_header_bytes(
        next_header: u8,
        payload_len: u16,
        src: [u8; 16],
        dst: [u8; 16],
    ) -> [u8; 40] {
        let mut buf = [0u8; 40];
        // Version 6, traffic_class 0, flow_label 0
        buf[0] = 0x60;
        let pl = payload_len.to_be_bytes();
        buf[4] = pl[0];
        buf[5] = pl[1];
        buf[6] = next_header;
        buf[7] = 64; // hop limit
        buf[8..24].copy_from_slice(&src);
        buf[24..40].copy_from_slice(&dst);
        buf
    }

    #[test]
    fn test_ipv6_addr_new() {
        let addr = Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0, 1);
        assert_eq!(addr.as_bytes()[0], 0x20);
        assert_eq!(addr.as_bytes()[1], 0x01);
        assert_eq!(addr.as_bytes()[15], 1);
    }

    #[test]
    fn test_ipv6_addr_from_bytes() {
        let bytes = [0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let addr = Ipv6Addr::from_bytes(bytes);
        assert_eq!(*addr.as_bytes(), bytes);
    }

    #[test]
    fn test_loopback() {
        assert!(Ipv6Addr::LOOPBACK.is_loopback());
        assert!(!Ipv6Addr::UNSPECIFIED.is_loopback());
    }

    #[test]
    fn test_unspecified() {
        assert!(Ipv6Addr::UNSPECIFIED.is_unspecified());
        assert!(!Ipv6Addr::LOOPBACK.is_unspecified());
    }

    #[test]
    fn test_multicast() {
        let mcast = Ipv6Addr::new(0xff02, 0, 0, 0, 0, 0, 0, 1);
        assert!(mcast.is_multicast());
        assert!(!Ipv6Addr::LOOPBACK.is_multicast());
    }

    #[test]
    fn test_link_local() {
        let ll = Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1);
        assert!(ll.is_link_local());
        assert!(!Ipv6Addr::LOOPBACK.is_link_local());
        // fe80 but not link-local prefix (fec0 is site-local, not link-local)
        let not_ll = Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 1);
        assert!(!not_ll.is_link_local());
    }

    #[test]
    fn test_solicited_node_multicast() {
        let addr = Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0x0001, 0x1234);
        let snm = Ipv6Addr::solicited_node_multicast(&addr);
        assert!(snm.is_multicast());
        let b = snm.as_bytes();
        assert_eq!(b[0], 0xff);
        assert_eq!(b[1], 0x02);
        assert_eq!(b[11], 0x01);
        assert_eq!(b[12], 0xff);
        // Last 3 bytes should be 00:12:34 (from 0x0001:0x1234 -> octets[13..16] = 01,12,34)
        assert_eq!(b[13], 0x01);
        assert_eq!(b[14], 0x12);
        assert_eq!(b[15], 0x34);
    }

    #[test]
    fn test_link_local_from_mac() {
        let mac = MacAddress::new(0x02, 0x42, 0xac, 0x11, 0x00, 0x02);
        let ll = Ipv6Addr::link_local_from_mac(&mac);
        assert!(ll.is_link_local());
        let b = ll.as_bytes();
        // First octet: 0x02 ^ 0x02 = 0x00
        assert_eq!(b[8], 0x00);
        assert_eq!(b[9], 0x42);
        assert_eq!(b[10], 0xac);
        assert_eq!(b[11], 0xff);
        assert_eq!(b[12], 0xfe);
        assert_eq!(b[13], 0x11);
        assert_eq!(b[14], 0x00);
        assert_eq!(b[15], 0x02);
    }

    #[test]
    fn test_display() {
        let addr = Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0, 1);
        let s = alloc::format!("{}", addr);
        assert_eq!(s, "2001:db8:0:0:0:0:0:1");
    }

    #[test]
    fn test_parse_valid_header() {
        let src = [0; 16];
        let dst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let buf = make_header_bytes(Ipv6Header::NEXT_HEADER_TCP, 0, src, dst);
        let (hdr, payload) = Ipv6Header::parse(&buf).unwrap();
        assert_eq!(hdr.version, 6);
        assert_eq!(hdr.next_header, 6);
        assert_eq!(hdr.hop_limit, 64);
        assert_eq!(hdr.src_addr, Ipv6Addr::UNSPECIFIED);
        assert_eq!(hdr.dst_addr, Ipv6Addr::LOOPBACK);
        assert!(payload.is_empty());
    }

    #[test]
    fn test_parse_too_short() {
        let buf = [0u8; 20];
        assert!(Ipv6Header::parse(&buf).is_err());
    }

    #[test]
    fn test_parse_bad_version() {
        let mut buf = [0u8; 40];
        buf[0] = 0x40; // version 4, not 6
        assert!(Ipv6Header::parse(&buf).is_err());
    }

    #[test]
    fn test_serialize_roundtrip() {
        let hdr = Ipv6Header {
            version: 6,
            traffic_class: 0xe0,
            flow_label: 0xabcde,
            payload_length: 100,
            next_header: Ipv6Header::NEXT_HEADER_UDP,
            hop_limit: 128,
            src_addr: Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1),
            dst_addr: Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 2),
        };
        let bytes = hdr.serialize();
        // Append enough payload bytes so parse can slice properly
        let mut data = [0u8; 140];
        data[..40].copy_from_slice(&bytes);
        let (parsed, payload) = Ipv6Header::parse(&data).unwrap();
        assert_eq!(parsed.version, 6);
        assert_eq!(parsed.traffic_class, 0xe0);
        assert_eq!(parsed.flow_label, 0xabcde);
        assert_eq!(parsed.payload_length, 100);
        assert_eq!(parsed.next_header, Ipv6Header::NEXT_HEADER_UDP);
        assert_eq!(parsed.hop_limit, 128);
        assert_eq!(parsed.src_addr, hdr.src_addr);
        assert_eq!(parsed.dst_addr, hdr.dst_addr);
        assert_eq!(payload.len(), 100);
    }

    #[test]
    fn test_header_constants() {
        assert_eq!(Ipv6Header::HEADER_LEN, 40);
        assert_eq!(Ipv6Header::NEXT_HEADER_TCP, 6);
        assert_eq!(Ipv6Header::NEXT_HEADER_UDP, 17);
        assert_eq!(Ipv6Header::NEXT_HEADER_ICMPV6, 58);
    }

    #[test]
    fn test_link_local_from_mac_eui64_flip() {
        // MAC with U/L bit already set: 0xaa -> 0xa8 after flip
        let mac = MacAddress::new(0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff);
        let ll = Ipv6Addr::link_local_from_mac(&mac);
        let b = ll.as_bytes();
        assert_eq!(b[8], 0xa8); // 0xaa ^ 0x02
        assert_eq!(b[9], 0xbb);
        assert_eq!(b[10], 0xcc);
        assert_eq!(b[11], 0xff);
        assert_eq!(b[12], 0xfe);
        assert_eq!(b[13], 0xdd);
        assert_eq!(b[14], 0xee);
        assert_eq!(b[15], 0xff);
    }
}
