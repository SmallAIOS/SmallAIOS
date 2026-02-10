// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! UDP datagram handling for SmallAIOS.
//!
//! Provides [`UdpHeader`] for parsing and serializing UDP headers and
//! [`UdpSocket`] for basic socket lifecycle management.

use alloc::vec::Vec;

use crate::NetError;

// ---------------------------------------------------------------------------
// UdpHeader
// ---------------------------------------------------------------------------

/// Parsed UDP datagram header (8 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdpHeader {
    /// Source port number.
    pub src_port: u16,
    /// Destination port number.
    pub dst_port: u16,
    /// Total datagram length (header + payload) in bytes.
    pub length: u16,
    /// Header checksum (0 means unused in IPv4).
    pub checksum: u16,
}

/// UDP header length in bytes.
pub const HEADER_LEN: usize = 8;

impl UdpHeader {
    /// Parse a UDP header from raw bytes.
    ///
    /// Returns the parsed header and a slice over the payload (everything past
    /// the 8-byte header). The payload length is determined by the `length`
    /// field minus [`HEADER_LEN`]; if the buffer is shorter than the declared
    /// length, only the available bytes are returned.
    ///
    /// Returns [`NetError::PacketTooShort`] if `data` is shorter than 8 bytes
    /// and [`NetError::InvalidHeader`] if the length field is less than 8.
    pub fn parse(data: &[u8]) -> Result<(UdpHeader, &[u8]), NetError> {
        if data.len() < HEADER_LEN {
            return Err(NetError::PacketTooShort);
        }

        let src_port = u16::from_be_bytes([data[0], data[1]]);
        let dst_port = u16::from_be_bytes([data[2], data[3]]);
        let length = u16::from_be_bytes([data[4], data[5]]);
        let checksum = u16::from_be_bytes([data[6], data[7]]);

        if (length as usize) < HEADER_LEN {
            return Err(NetError::InvalidHeader);
        }

        let payload_len = (length as usize) - HEADER_LEN;
        let available = data.len() - HEADER_LEN;
        let actual_payload_len = core::cmp::min(payload_len, available);

        let header = UdpHeader {
            src_port,
            dst_port,
            length,
            checksum,
        };

        Ok((header, &data[HEADER_LEN..HEADER_LEN + actual_payload_len]))
    }

    /// Serialize the 8-byte UDP header to a byte array.
    pub fn serialize(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0..2].copy_from_slice(&self.src_port.to_be_bytes());
        buf[2..4].copy_from_slice(&self.dst_port.to_be_bytes());
        buf[4..6].copy_from_slice(&self.length.to_be_bytes());
        buf[6..8].copy_from_slice(&self.checksum.to_be_bytes());
        buf
    }

    /// Build a complete UDP datagram (header + payload) as a [`Vec<u8>`].
    ///
    /// The `length` field is set to `HEADER_LEN + payload.len()`.
    pub fn build_datagram(src_port: u16, dst_port: u16, payload: &[u8]) -> Vec<u8> {
        let total_len = HEADER_LEN + payload.len();
        let header = UdpHeader {
            src_port,
            dst_port,
            length: total_len as u16,
            checksum: 0,
        };
        let mut buf = Vec::with_capacity(total_len);
        buf.extend_from_slice(&header.serialize());
        buf.extend_from_slice(payload);
        buf
    }
}

// ---------------------------------------------------------------------------
// UdpSocket
// ---------------------------------------------------------------------------

/// A minimal UDP socket for sending and receiving datagrams.
pub struct UdpSocket {
    /// Local port this socket is associated with.
    pub local_port: u16,
    /// Whether the socket has been bound to its port.
    bound: bool,
}

impl UdpSocket {
    /// Create a new [`UdpSocket`] associated with `port`.
    ///
    /// The socket is not bound until [`UdpSocket::bind`] is called.
    pub fn new(port: u16) -> Self {
        UdpSocket {
            local_port: port,
            bound: false,
        }
    }

    /// Returns `true` if the socket has been bound.
    pub fn is_bound(&self) -> bool {
        self.bound
    }

    /// Bind the socket to its local port.
    ///
    /// Returns [`NetError::InvalidAddress`] if the socket is already bound.
    pub fn bind(&mut self) -> Result<(), NetError> {
        if self.bound {
            return Err(NetError::InvalidAddress);
        }
        self.bound = true;
        Ok(())
    }

    /// Send `data` to `(addr, port)`.
    ///
    /// Returns [`NetError::InvalidProtocol`] if the socket is not yet bound.
    /// The actual send path is not yet implemented.
    pub fn send_to(&self, _data: &[u8], _addr: [u8; 4], _port: u16) -> Result<usize, NetError> {
        if !self.bound {
            return Err(NetError::InvalidProtocol);
        }
        Err(NetError::NotImplemented)
    }

    /// Receive a datagram into `buf`.
    ///
    /// On success returns `(bytes_read, src_port)`.
    /// Returns [`NetError::InvalidProtocol`] if the socket is not yet bound.
    /// The actual receive path is not yet implemented.
    pub fn recv_from(&self, _buf: &mut [u8]) -> Result<(usize, u16), NetError> {
        if !self.bound {
            return Err(NetError::InvalidProtocol);
        }
        Err(NetError::NotImplemented)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- UdpHeader parse / serialize ----------------------------------------

    #[test]
    fn test_parse_valid_header() {
        let mut data = [0u8; 12];
        // src_port = 5353
        data[0..2].copy_from_slice(&5353u16.to_be_bytes());
        // dst_port = 53
        data[2..4].copy_from_slice(&53u16.to_be_bytes());
        // length = 12 (8 header + 4 payload)
        data[4..6].copy_from_slice(&12u16.to_be_bytes());
        // checksum = 0
        data[6..8].copy_from_slice(&0u16.to_be_bytes());
        // payload
        data[8..12].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let (hdr, payload) = UdpHeader::parse(&data).unwrap();
        assert_eq!(hdr.src_port, 5353);
        assert_eq!(hdr.dst_port, 53);
        assert_eq!(hdr.length, 12);
        assert_eq!(hdr.checksum, 0);
        assert_eq!(payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_parse_too_short() {
        let data = [0u8; 7];
        assert_eq!(UdpHeader::parse(&data), Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_parse_invalid_length_field() {
        let mut data = [0u8; 8];
        // length = 4 (less than HEADER_LEN)
        data[4..6].copy_from_slice(&4u16.to_be_bytes());
        assert_eq!(UdpHeader::parse(&data), Err(NetError::InvalidHeader));
    }

    #[test]
    fn test_parse_header_only() {
        let mut data = [0u8; 8];
        // length = 8 (header only, no payload)
        data[4..6].copy_from_slice(&8u16.to_be_bytes());
        let (hdr, payload) = UdpHeader::parse(&data).unwrap();
        assert_eq!(hdr.length, 8);
        assert!(payload.is_empty());
    }

    #[test]
    fn test_serialize_roundtrip() {
        let original = UdpHeader {
            src_port: 12345,
            dst_port: 443,
            length: 28,
            checksum: 0xABCD,
        };

        let bytes = original.serialize();
        assert_eq!(bytes.len(), 8);

        // Append enough dummy payload so parse succeeds for the declared length.
        let mut full = Vec::from(bytes.as_slice());
        full.extend_from_slice(&[0u8; 20]);

        let (parsed, _) = UdpHeader::parse(&full).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_header_len_constant() {
        assert_eq!(HEADER_LEN, 8);
    }

    #[test]
    fn test_build_datagram() {
        let payload = b"hello";
        let datagram = UdpHeader::build_datagram(5000, 80, payload);
        assert_eq!(datagram.len(), HEADER_LEN + 5);

        let (hdr, p) = UdpHeader::parse(&datagram).unwrap();
        assert_eq!(hdr.src_port, 5000);
        assert_eq!(hdr.dst_port, 80);
        assert_eq!(hdr.length, 13);
        assert_eq!(p, b"hello");
    }

    // -- UdpSocket ----------------------------------------------------------

    #[test]
    fn test_socket_creation() {
        let sock = UdpSocket::new(8080);
        assert_eq!(sock.local_port, 8080);
        assert!(!sock.is_bound());
    }

    #[test]
    fn test_socket_bind() {
        let mut sock = UdpSocket::new(53);
        assert!(sock.bind().is_ok());
        assert!(sock.is_bound());
    }

    #[test]
    fn test_socket_double_bind() {
        let mut sock = UdpSocket::new(53);
        sock.bind().unwrap();
        assert_eq!(sock.bind(), Err(NetError::InvalidAddress));
    }

    #[test]
    fn test_socket_send_unbound() {
        let sock = UdpSocket::new(5000);
        assert_eq!(
            sock.send_to(b"data", [10, 0, 0, 1], 80),
            Err(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn test_socket_recv_unbound() {
        let sock = UdpSocket::new(5000);
        let mut buf = [0u8; 64];
        assert_eq!(sock.recv_from(&mut buf), Err(NetError::InvalidProtocol));
    }

    #[test]
    fn test_socket_send_stub() {
        let mut sock = UdpSocket::new(5000);
        sock.bind().unwrap();
        assert_eq!(
            sock.send_to(b"data", [10, 0, 0, 1], 80),
            Err(NetError::NotImplemented)
        );
    }

    #[test]
    fn test_socket_recv_stub() {
        let mut sock = UdpSocket::new(5000);
        sock.bind().unwrap();
        let mut buf = [0u8; 64];
        assert_eq!(sock.recv_from(&mut buf), Err(NetError::NotImplemented));
    }
}
