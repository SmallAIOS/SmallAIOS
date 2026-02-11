// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SpaceWire packet encode/decode.
//!
//! A SpaceWire packet consists of a destination address (one or more bytes),
//! cargo (payload data), and a termination marker (EOP or EEP).
//! Path addressing: address bytes are consumed at each router hop.
//! Logical addressing: single byte >= 32, not consumed.

use alloc::vec::Vec;
use crate::BusError;

/// End-of-packet marker type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EopMarker {
    /// Normal end of packet.
    Eop,
    /// Error end of packet (partial data).
    Eep,
}

/// SpaceWire packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceWirePacket {
    /// Destination address byte(s).
    /// For logical addressing: single byte >= 32.
    /// For path addressing: sequence of port numbers (0-31).
    pub dest_addr: Vec<u8>,
    /// Payload cargo data.
    pub cargo: Vec<u8>,
    /// Packet termination marker.
    pub marker: EopMarker,
}

/// Encoded packet layout:
/// dest_addr_len(1) + dest_addr(N) + cargo(M) + marker(1)
/// marker byte: 0x00 = EOP, 0x01 = EEP
const MARKER_EOP: u8 = 0x00;
const MARKER_EEP: u8 = 0x01;

impl SpaceWirePacket {
    /// Create a new SpaceWire packet with normal EOP.
    pub fn new(dest_addr: &[u8], cargo: &[u8]) -> Result<Self, BusError> {
        if dest_addr.is_empty() {
            return Err(BusError::InvalidHeader);
        }
        Ok(Self {
            dest_addr: Vec::from(dest_addr),
            cargo: Vec::from(cargo),
            marker: EopMarker::Eop,
        })
    }

    /// Create a new SpaceWire packet with logical addressing.
    pub fn new_logical(logical_addr: u8, cargo: &[u8]) -> Result<Self, BusError> {
        if logical_addr < 32 {
            return Err(BusError::InvalidId);
        }
        Ok(Self {
            dest_addr: alloc::vec![logical_addr],
            cargo: Vec::from(cargo),
            marker: EopMarker::Eop,
        })
    }

    /// Create a packet with EEP (error) marker.
    pub fn new_error(dest_addr: &[u8], partial_cargo: &[u8]) -> Result<Self, BusError> {
        if dest_addr.is_empty() {
            return Err(BusError::InvalidHeader);
        }
        Ok(Self {
            dest_addr: Vec::from(dest_addr),
            cargo: Vec::from(partial_cargo),
            marker: EopMarker::Eep,
        })
    }

    /// Returns true if this uses logical addressing (dest >= 32).
    pub fn is_logical_addr(&self) -> bool {
        self.dest_addr.len() == 1 && self.dest_addr[0] >= 32
    }

    /// Returns true if this packet terminated with an error.
    pub fn is_error(&self) -> bool {
        self.marker == EopMarker::Eep
    }

    /// Encode the packet into a byte buffer.
    ///
    /// Layout: addr_len(1) + dest_addr(N) + cargo(M) + marker(1)
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        let total = 1 + self.dest_addr.len() + self.cargo.len() + 1;
        if buf.len() < total {
            return Err(BusError::BufferTooSmall);
        }
        buf[0] = self.dest_addr.len() as u8;
        buf[1..1 + self.dest_addr.len()].copy_from_slice(&self.dest_addr);
        let cargo_start = 1 + self.dest_addr.len();
        buf[cargo_start..cargo_start + self.cargo.len()].copy_from_slice(&self.cargo);
        buf[total - 1] = match self.marker {
            EopMarker::Eop => MARKER_EOP,
            EopMarker::Eep => MARKER_EEP,
        };
        Ok(total)
    }

    /// Decode a SpaceWire packet from a byte buffer.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < 3 {
            return Err(BusError::FrameTooShort);
        }
        let addr_len = data[0] as usize;
        if addr_len == 0 {
            return Err(BusError::InvalidHeader);
        }
        if data.len() < 1 + addr_len + 1 {
            return Err(BusError::FrameTooShort);
        }
        let dest_addr = Vec::from(&data[1..1 + addr_len]);
        let cargo_end = data.len() - 1;
        let cargo = Vec::from(&data[1 + addr_len..cargo_end]);
        let marker = match data[cargo_end] {
            MARKER_EOP => EopMarker::Eop,
            MARKER_EEP => EopMarker::Eep,
            _ => return Err(BusError::InvalidHeader),
        };
        Ok(Self {
            dest_addr,
            cargo,
            marker,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_new() {
        let pkt = SpaceWirePacket::new(&[42], &[0xAA, 0xBB]).unwrap();
        assert_eq!(pkt.dest_addr, &[42]);
        assert_eq!(pkt.cargo, &[0xAA, 0xBB]);
        assert_eq!(pkt.marker, EopMarker::Eop);
    }

    #[test]
    fn test_packet_new_empty_addr() {
        assert_eq!(SpaceWirePacket::new(&[], &[0xAA]).err(), Some(BusError::InvalidHeader));
    }

    #[test]
    fn test_packet_logical_addressing() {
        let pkt = SpaceWirePacket::new_logical(42, &[0x11]).unwrap();
        assert!(pkt.is_logical_addr());
        assert_eq!(pkt.dest_addr, &[42]);
    }

    #[test]
    fn test_packet_logical_addr_too_low() {
        assert_eq!(SpaceWirePacket::new_logical(31, &[]).err(), Some(BusError::InvalidId));
    }

    #[test]
    fn test_packet_path_addressing() {
        let pkt = SpaceWirePacket::new(&[1, 3, 5], &[0xCC]).unwrap();
        assert!(!pkt.is_logical_addr());
        assert_eq!(pkt.dest_addr, &[1, 3, 5]);
    }

    #[test]
    fn test_packet_error() {
        let pkt = SpaceWirePacket::new_error(&[42], &[0xAA]).unwrap();
        assert!(pkt.is_error());
        assert_eq!(pkt.marker, EopMarker::Eep);
    }

    #[test]
    fn test_packet_encode_decode_roundtrip() {
        let pkt = SpaceWirePacket::new(&[42], &[0x01, 0x02, 0x03]).unwrap();
        let mut buf = [0u8; 64];
        let len = pkt.encode(&mut buf).unwrap();
        let decoded = SpaceWirePacket::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, pkt);
    }

    #[test]
    fn test_packet_encode_decode_path_addr() {
        let pkt = SpaceWirePacket::new(&[1, 3, 5], &[0xAA, 0xBB]).unwrap();
        let mut buf = [0u8; 64];
        let len = pkt.encode(&mut buf).unwrap();
        let decoded = SpaceWirePacket::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.dest_addr, &[1, 3, 5]);
        assert_eq!(decoded.cargo, &[0xAA, 0xBB]);
    }

    #[test]
    fn test_packet_encode_decode_eep() {
        let pkt = SpaceWirePacket::new_error(&[42], &[0xFF]).unwrap();
        let mut buf = [0u8; 64];
        let len = pkt.encode(&mut buf).unwrap();
        let decoded = SpaceWirePacket::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.marker, EopMarker::Eep);
    }

    #[test]
    fn test_packet_encode_buffer_too_small() {
        let pkt = SpaceWirePacket::new(&[42], &[0xAA; 100]).unwrap();
        let mut buf = [0u8; 10];
        assert_eq!(pkt.encode(&mut buf).err(), Some(BusError::BufferTooSmall));
    }

    #[test]
    fn test_packet_decode_too_short() {
        assert_eq!(SpaceWirePacket::decode(&[0x01, 0x42]).err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_packet_decode_invalid_marker() {
        let data = [0x01, 42, 0xAA, 0xFF]; // marker = 0xFF is invalid
        assert_eq!(SpaceWirePacket::decode(&data).err(), Some(BusError::InvalidHeader));
    }

    #[test]
    fn test_packet_empty_cargo() {
        let pkt = SpaceWirePacket::new(&[42], &[]).unwrap();
        let mut buf = [0u8; 64];
        let len = pkt.encode(&mut buf).unwrap();
        let decoded = SpaceWirePacket::decode(&buf[..len]).unwrap();
        assert!(decoded.cargo.is_empty());
    }
}
