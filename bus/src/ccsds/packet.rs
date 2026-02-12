// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CCSDS Space Packet encode/decode.
//!
//! A Space Packet has a 6-byte primary header:
//! - Packet Version Number (3 bits) = 000
//! - Packet Type (1 bit): 0 = TM, 1 = TC
//! - Secondary Header Flag (1 bit)
//! - APID (11 bits)
//! - Sequence Flags (2 bits): 01=first, 00=cont, 10=last, 11=unsegmented
//! - Sequence Count (14 bits): 0-16383
//! - Packet Data Length (16 bits): number of octets in data field minus 1

use crate::BusError;
use alloc::vec::Vec;

/// Primary header size in bytes.
pub const PRIMARY_HEADER_LEN: usize = 6;

/// Maximum APID value (11 bits).
pub const MAX_APID: u16 = 0x7FF;

/// Idle packet APID.
pub const IDLE_APID: u16 = 0x7FF;

/// Maximum sequence count (14 bits).
pub const MAX_SEQ_COUNT: u16 = 0x3FFF;

/// Packet type: telemetry or telecommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketType {
    /// Telemetry (downlink).
    Telemetry,
    /// Telecommand (uplink).
    Telecommand,
}

/// Sequence flags indicating segmentation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqFlags {
    /// Continuation segment.
    Continuation,
    /// First segment.
    First,
    /// Last segment.
    Last,
    /// Unsegmented (standalone packet).
    Unsegmented,
}

impl SeqFlags {
    fn to_bits(self) -> u8 {
        match self {
            SeqFlags::Continuation => 0b00,
            SeqFlags::First => 0b01,
            SeqFlags::Last => 0b10,
            SeqFlags::Unsegmented => 0b11,
        }
    }

    fn from_bits(bits: u8) -> Result<Self, BusError> {
        match bits & 0x03 {
            0b00 => Ok(SeqFlags::Continuation),
            0b01 => Ok(SeqFlags::First),
            0b10 => Ok(SeqFlags::Last),
            0b11 => Ok(SeqFlags::Unsegmented),
            _ => Err(BusError::InvalidHeader),
        }
    }
}

/// CCSDS Space Packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpacePacket {
    /// Packet type (TM or TC).
    pub packet_type: PacketType,
    /// Secondary header flag.
    pub secondary_header: bool,
    /// Application Process Identifier (0-2046; 2047 = idle).
    pub apid: u16,
    /// Sequence flags.
    pub seq_flags: SeqFlags,
    /// Sequence count (0-16383).
    pub seq_count: u16,
    /// User data payload.
    pub data: Vec<u8>,
}

impl SpacePacket {
    /// Create a new Space Packet.
    pub fn new(
        packet_type: PacketType,
        secondary_header: bool,
        apid: u16,
        seq_flags: SeqFlags,
        seq_count: u16,
        data: &[u8],
    ) -> Result<Self, BusError> {
        if apid > MAX_APID {
            return Err(BusError::InvalidApid);
        }
        if seq_count > MAX_SEQ_COUNT {
            return Err(BusError::OutOfRange);
        }
        if data.is_empty() {
            return Err(BusError::FrameTooShort);
        }
        Ok(Self {
            packet_type,
            secondary_header,
            apid,
            seq_flags,
            seq_count,
            data: Vec::from(data),
        })
    }

    /// Create an idle packet.
    pub fn idle() -> Self {
        Self {
            packet_type: PacketType::Telemetry,
            secondary_header: false,
            apid: IDLE_APID,
            seq_flags: SeqFlags::Unsegmented,
            seq_count: 0,
            data: alloc::vec![0x00],
        }
    }

    /// Returns true if this is an idle packet.
    pub fn is_idle(&self) -> bool {
        self.apid == IDLE_APID
    }

    /// Total packet length in bytes (header + data).
    pub fn total_length(&self) -> usize {
        PRIMARY_HEADER_LEN + self.data.len()
    }

    /// Encode the packet into a byte buffer.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        let total = self.total_length();
        if buf.len() < total {
            return Err(BusError::BufferTooSmall);
        }

        // Word 1: version(3) + type(1) + sec_hdr(1) + apid(11)
        let type_bit = match self.packet_type {
            PacketType::Telemetry => 0u16,
            PacketType::Telecommand => 1u16,
        };
        let sec_bit = if self.secondary_header { 1u16 } else { 0 };
        let word1: u16 = (type_bit << 12) | (sec_bit << 11) | (self.apid & MAX_APID);
        buf[0] = (word1 >> 8) as u8;
        buf[1] = word1 as u8;

        // Word 2: seq_flags(2) + seq_count(14)
        let word2: u16 =
            ((self.seq_flags.to_bits() as u16) << 14) | (self.seq_count & MAX_SEQ_COUNT);
        buf[2] = (word2 >> 8) as u8;
        buf[3] = word2 as u8;

        // Word 3: packet data length = data_len - 1
        let pdl = (self.data.len() - 1) as u16;
        buf[4] = (pdl >> 8) as u8;
        buf[5] = pdl as u8;

        // Data field
        buf[PRIMARY_HEADER_LEN..total].copy_from_slice(&self.data);

        Ok(total)
    }

    /// Decode a Space Packet from a byte buffer.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < PRIMARY_HEADER_LEN + 1 {
            return Err(BusError::FrameTooShort);
        }

        let word1 = ((data[0] as u16) << 8) | data[1] as u16;
        let version = (word1 >> 13) & 0x07;
        if version != 0 {
            return Err(BusError::InvalidHeader);
        }
        let type_bit = (word1 >> 12) & 1;
        let packet_type = if type_bit == 0 {
            PacketType::Telemetry
        } else {
            PacketType::Telecommand
        };
        let secondary_header = (word1 >> 11) & 1 == 1;
        let apid = word1 & MAX_APID;

        let word2 = ((data[2] as u16) << 8) | data[3] as u16;
        let seq_flags = SeqFlags::from_bits(((word2 >> 14) & 0x03) as u8)?;
        let seq_count = word2 & MAX_SEQ_COUNT;

        let pdl = ((data[4] as u16) << 8) | data[5] as u16;
        let data_len = pdl as usize + 1;
        let total = PRIMARY_HEADER_LEN + data_len;
        if data.len() < total {
            return Err(BusError::FrameTooShort);
        }

        Ok(Self {
            packet_type,
            secondary_header,
            apid,
            seq_flags,
            seq_count,
            data: Vec::from(&data[PRIMARY_HEADER_LEN..total]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_new_valid() {
        let pkt = SpacePacket::new(
            PacketType::Telemetry,
            true,
            0x1A3,
            SeqFlags::Unsegmented,
            1042,
            &[0xAA; 256],
        )
        .unwrap();
        assert_eq!(pkt.apid, 0x1A3);
        assert!(pkt.secondary_header);
        assert_eq!(pkt.seq_count, 1042);
        assert_eq!(pkt.data.len(), 256);
        assert_eq!(pkt.total_length(), 262);
    }

    #[test]
    fn test_packet_invalid_apid() {
        assert_eq!(
            SpacePacket::new(
                PacketType::Telemetry,
                false,
                0x800,
                SeqFlags::Unsegmented,
                0,
                &[0x00]
            )
            .err(),
            Some(BusError::InvalidApid)
        );
    }

    #[test]
    fn test_packet_invalid_seq_count() {
        assert_eq!(
            SpacePacket::new(
                PacketType::Telemetry,
                false,
                0,
                SeqFlags::Unsegmented,
                0x4000,
                &[0x00]
            )
            .err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_packet_empty_data() {
        assert_eq!(
            SpacePacket::new(
                PacketType::Telemetry,
                false,
                0,
                SeqFlags::Unsegmented,
                0,
                &[]
            )
            .err(),
            Some(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_encode_decode_roundtrip_tm() {
        let pkt = SpacePacket::new(
            PacketType::Telemetry,
            true,
            0x1A3,
            SeqFlags::Unsegmented,
            1042,
            &[0x01, 0x02, 0x03],
        )
        .unwrap();
        let mut buf = [0u8; 256];
        let len = pkt.encode(&mut buf).unwrap();
        let decoded = SpacePacket::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, pkt);
    }

    #[test]
    fn test_encode_decode_roundtrip_tc() {
        let pkt = SpacePacket::new(
            PacketType::Telecommand,
            false,
            0x100,
            SeqFlags::First,
            0,
            &[0xFF],
        )
        .unwrap();
        let mut buf = [0u8; 64];
        let len = pkt.encode(&mut buf).unwrap();
        let decoded = SpacePacket::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.packet_type, PacketType::Telecommand);
        assert_eq!(decoded.seq_flags, SeqFlags::First);
    }

    #[test]
    fn test_encode_packet_data_length() {
        let pkt = SpacePacket::new(
            PacketType::Telemetry,
            false,
            0,
            SeqFlags::Unsegmented,
            0,
            &[0xAA; 256],
        )
        .unwrap();
        let mut buf = [0u8; 300];
        pkt.encode(&mut buf).unwrap();
        // PDL = 256 - 1 = 255
        let pdl = ((buf[4] as u16) << 8) | buf[5] as u16;
        assert_eq!(pdl, 255);
    }

    #[test]
    fn test_decode_invalid_version() {
        let mut buf = [0u8; 10];
        // Set version to 001 (invalid)
        buf[0] = 0x20;
        buf[4] = 0;
        buf[5] = 0; // PDL = 0 means 1 byte data
        assert_eq!(
            SpacePacket::decode(&buf).err(),
            Some(BusError::InvalidHeader)
        );
    }

    #[test]
    fn test_decode_too_short() {
        assert_eq!(
            SpacePacket::decode(&[0u8; 6]).err(),
            Some(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_decode_data_length_mismatch() {
        let mut buf = [0u8; 10];
        // PDL = 99 means 100 bytes data, but we only have 4 bytes after header
        buf[4] = 0;
        buf[5] = 99;
        assert_eq!(
            SpacePacket::decode(&buf).err(),
            Some(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_idle_packet() {
        let idle = SpacePacket::idle();
        assert!(idle.is_idle());
        assert_eq!(idle.apid, IDLE_APID);
    }

    #[test]
    fn test_non_idle_packet() {
        let pkt = SpacePacket::new(
            PacketType::Telemetry,
            false,
            0x100,
            SeqFlags::Unsegmented,
            0,
            &[0x00],
        )
        .unwrap();
        assert!(!pkt.is_idle());
    }

    #[test]
    fn test_all_seq_flags_roundtrip() {
        for flags in [
            SeqFlags::Continuation,
            SeqFlags::First,
            SeqFlags::Last,
            SeqFlags::Unsegmented,
        ] {
            let pkt = SpacePacket::new(PacketType::Telemetry, false, 0, flags, 0, &[0x00]).unwrap();
            let mut buf = [0u8; 64];
            let len = pkt.encode(&mut buf).unwrap();
            let decoded = SpacePacket::decode(&buf[..len]).unwrap();
            assert_eq!(decoded.seq_flags, flags);
        }
    }

    #[test]
    fn test_encode_buffer_too_small() {
        let pkt = SpacePacket::new(
            PacketType::Telemetry,
            false,
            0,
            SeqFlags::Unsegmented,
            0,
            &[0xAA; 100],
        )
        .unwrap();
        let mut buf = [0u8; 10];
        assert_eq!(pkt.encode(&mut buf).err(), Some(BusError::BufferTooSmall));
    }

    #[test]
    fn test_seq_count_wraps_modulo() {
        // Sequence count should be mod 16384
        let pkt = SpacePacket::new(
            PacketType::Telemetry,
            false,
            0,
            SeqFlags::Unsegmented,
            MAX_SEQ_COUNT,
            &[0x00],
        )
        .unwrap();
        let mut buf = [0u8; 64];
        let len = pkt.encode(&mut buf).unwrap();
        let decoded = SpacePacket::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.seq_count, MAX_SEQ_COUNT);
    }

    #[test]
    fn test_max_apid() {
        // APID 0x7FE is valid (max non-idle)
        let pkt = SpacePacket::new(
            PacketType::Telemetry,
            false,
            0x7FE,
            SeqFlags::Unsegmented,
            0,
            &[0x00],
        )
        .unwrap();
        assert_eq!(pkt.apid, 0x7FE);
    }
}
