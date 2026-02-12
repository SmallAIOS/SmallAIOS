// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC v1 packet header encode/decode (RFC 9000 Section 17).
//!
//! Supports Initial, Handshake, 0-RTT, 1-RTT (short header),
//! Version Negotiation, and Retry packet types.

use crate::NetError;
use alloc::vec::Vec;

/// QUIC v1 version number.
pub const QUIC_VERSION_1: u32 = 0x00000001;

/// Version Negotiation uses version 0.
pub const QUIC_VERSION_NEGOTIATION: u32 = 0x00000000;

/// Maximum connection ID length (20 bytes per RFC 9000).
pub const MAX_CID_LEN: usize = 20;

/// Minimum QUIC packet size.
pub const MIN_PACKET_LEN: usize = 1;

/// Long header fixed bit mask.
const LONG_HEADER_FORM: u8 = 0x80;
/// Fixed bit (must be 1).
const FIXED_BIT: u8 = 0x40;

/// QUIC version wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuicVersion(pub u32);

impl QuicVersion {
    /// QUIC v1.
    pub fn v1() -> Self {
        Self(QUIC_VERSION_1)
    }

    /// Version negotiation (version 0).
    pub fn negotiation() -> Self {
        Self(QUIC_VERSION_NEGOTIATION)
    }

    /// Check if this is a supported version.
    pub fn is_supported(&self) -> bool {
        self.0 == QUIC_VERSION_1
    }
}

/// QUIC packet type (from long header type bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketType {
    /// Initial packet (type 0x00).
    Initial,
    /// 0-RTT packet (type 0x01).
    ZeroRtt,
    /// Handshake packet (type 0x02).
    Handshake,
    /// Retry packet (type 0x03).
    Retry,
    /// Short header (1-RTT) packet.
    OneRtt,
    /// Version Negotiation packet.
    VersionNegotiation,
}

impl PacketType {
    /// Get the 2-bit type value for long headers.
    fn to_type_bits(self) -> u8 {
        match self {
            PacketType::Initial => 0x00,
            PacketType::ZeroRtt => 0x01,
            PacketType::Handshake => 0x02,
            PacketType::Retry => 0x03,
            _ => 0x00, // Short/VN don't use type bits
        }
    }

    /// Parse from 2-bit type field.
    fn from_type_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0x00 => PacketType::Initial,
            0x01 => PacketType::ZeroRtt,
            0x02 => PacketType::Handshake,
            0x03 => PacketType::Retry,
            _ => unreachable!(),
        }
    }
}

/// QUIC long header (Initial, Handshake, 0-RTT, Retry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongHeader {
    /// Packet type.
    pub packet_type: PacketType,
    /// QUIC version.
    pub version: QuicVersion,
    /// Destination connection ID.
    pub dcid: Vec<u8>,
    /// Source connection ID.
    pub scid: Vec<u8>,
    /// Packet number (not present in Retry/VN).
    pub packet_number: u32,
    /// Packet number length (1-4 bytes).
    pub pn_length: u8,
    /// Token (Initial packets only).
    pub token: Vec<u8>,
    /// Payload length (including packet number).
    pub payload_length: usize,
}

impl LongHeader {
    /// Create a new long header.
    pub fn new(packet_type: PacketType, dcid: &[u8], scid: &[u8]) -> Result<Self, NetError> {
        if dcid.len() > MAX_CID_LEN || scid.len() > MAX_CID_LEN {
            return Err(NetError::InvalidAddress);
        }
        Ok(Self {
            packet_type,
            version: QuicVersion::v1(),
            dcid: Vec::from(dcid),
            scid: Vec::from(scid),
            packet_number: 0,
            pn_length: 4,
            token: Vec::new(),
            payload_length: 0,
        })
    }

    /// Encode the long header into a buffer. Returns bytes written.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, NetError> {
        // First byte: form(1) + fixed(1) + type(2) + reserved(2) + pn_len(2)
        let first_byte = LONG_HEADER_FORM
            | FIXED_BIT
            | (self.packet_type.to_type_bits() << 4)
            | ((self.pn_length - 1) & 0x03);

        // Calculate total header size
        let token_len_size = if self.packet_type == PacketType::Initial {
            var_int_len(self.token.len() as u64)
        } else {
            0
        };
        let payload_len_size = if self.packet_type != PacketType::Retry {
            var_int_len(self.payload_length as u64)
        } else {
            0
        };
        let total = 1 // first byte
            + 4 // version
            + 1 + self.dcid.len() // DCID length + DCID
            + 1 + self.scid.len() // SCID length + SCID
            + token_len_size + self.token.len()
            + payload_len_size
            + self.pn_length as usize;

        if buf.len() < total {
            return Err(NetError::BufferTooSmall);
        }

        let mut off = 0;
        buf[off] = first_byte;
        off += 1;

        // Version
        buf[off..off + 4].copy_from_slice(&self.version.0.to_be_bytes());
        off += 4;

        // DCID
        buf[off] = self.dcid.len() as u8;
        off += 1;
        buf[off..off + self.dcid.len()].copy_from_slice(&self.dcid);
        off += self.dcid.len();

        // SCID
        buf[off] = self.scid.len() as u8;
        off += 1;
        buf[off..off + self.scid.len()].copy_from_slice(&self.scid);
        off += self.scid.len();

        // Token (Initial only)
        if self.packet_type == PacketType::Initial {
            off += encode_var_int(self.token.len() as u64, &mut buf[off..]);
            buf[off..off + self.token.len()].copy_from_slice(&self.token);
            off += self.token.len();
        }

        // Payload length (not for Retry)
        if self.packet_type != PacketType::Retry {
            off += encode_var_int(self.payload_length as u64, &mut buf[off..]);
        }

        // Packet number (not for Retry/VN)
        if self.packet_type != PacketType::Retry
            && self.packet_type != PacketType::VersionNegotiation
        {
            let pn_bytes = self.packet_number.to_be_bytes();
            let start = 4 - self.pn_length as usize;
            buf[off..off + self.pn_length as usize].copy_from_slice(&pn_bytes[start..]);
            off += self.pn_length as usize;
        }

        Ok(off)
    }

    /// Decode a long header from a buffer.
    pub fn decode(data: &[u8]) -> Result<(Self, usize), NetError> {
        if data.is_empty() {
            return Err(NetError::PacketTooShort);
        }
        let first_byte = data[0];
        if first_byte & LONG_HEADER_FORM == 0 {
            return Err(NetError::InvalidHeader);
        }

        if data.len() < 6 {
            return Err(NetError::PacketTooShort);
        }

        let mut off = 1;

        // Version
        let version = u32::from_be_bytes(data[off..off + 4].try_into().unwrap());
        off += 4;

        // Check for Version Negotiation (version == 0)
        if version == QUIC_VERSION_NEGOTIATION {
            // DCID
            if off >= data.len() {
                return Err(NetError::PacketTooShort);
            }
            let dcid_len = data[off] as usize;
            off += 1;
            if off + dcid_len > data.len() || dcid_len > MAX_CID_LEN {
                return Err(NetError::PacketTooShort);
            }
            let dcid = Vec::from(&data[off..off + dcid_len]);
            off += dcid_len;

            // SCID
            if off >= data.len() {
                return Err(NetError::PacketTooShort);
            }
            let scid_len = data[off] as usize;
            off += 1;
            if off + scid_len > data.len() || scid_len > MAX_CID_LEN {
                return Err(NetError::PacketTooShort);
            }
            let scid = Vec::from(&data[off..off + scid_len]);
            off += scid_len;

            return Ok((
                LongHeader {
                    packet_type: PacketType::VersionNegotiation,
                    version: QuicVersion(version),
                    dcid,
                    scid,
                    packet_number: 0,
                    pn_length: 0,
                    token: Vec::new(),
                    payload_length: data.len() - off,
                },
                off,
            ));
        }

        let type_bits = (first_byte >> 4) & 0x03;
        let pn_length = (first_byte & 0x03) + 1;
        let packet_type = PacketType::from_type_bits(type_bits);

        // DCID
        if off >= data.len() {
            return Err(NetError::PacketTooShort);
        }
        let dcid_len = data[off] as usize;
        off += 1;
        if dcid_len > MAX_CID_LEN || off + dcid_len > data.len() {
            return Err(NetError::PacketTooShort);
        }
        let dcid = Vec::from(&data[off..off + dcid_len]);
        off += dcid_len;

        // SCID
        if off >= data.len() {
            return Err(NetError::PacketTooShort);
        }
        let scid_len = data[off] as usize;
        off += 1;
        if scid_len > MAX_CID_LEN || off + scid_len > data.len() {
            return Err(NetError::PacketTooShort);
        }
        let scid = Vec::from(&data[off..off + scid_len]);
        off += scid_len;

        // Token (Initial only)
        let mut token = Vec::new();
        if packet_type == PacketType::Initial {
            let (token_len, consumed) = decode_var_int(&data[off..])?;
            off += consumed;
            if off + token_len as usize > data.len() {
                return Err(NetError::PacketTooShort);
            }
            token = Vec::from(&data[off..off + token_len as usize]);
            off += token_len as usize;
        }

        // Payload length (not Retry)
        let mut payload_length = 0;
        if packet_type != PacketType::Retry {
            let (pl, consumed) = decode_var_int(&data[off..])?;
            payload_length = pl as usize;
            off += consumed;
        }

        // Packet number
        let mut packet_number = 0u32;
        if packet_type != PacketType::Retry {
            if off + pn_length as usize > data.len() {
                return Err(NetError::PacketTooShort);
            }
            for i in 0..pn_length as usize {
                packet_number = (packet_number << 8) | data[off + i] as u32;
            }
            off += pn_length as usize;
        }

        Ok((
            LongHeader {
                packet_type,
                version: QuicVersion(version),
                dcid,
                scid,
                packet_number,
                pn_length,
                token,
                payload_length,
            },
            off,
        ))
    }
}

/// QUIC short header (1-RTT).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortHeader {
    /// Spin bit.
    pub spin_bit: bool,
    /// Key phase bit.
    pub key_phase: bool,
    /// Destination connection ID.
    pub dcid: Vec<u8>,
    /// Packet number.
    pub packet_number: u32,
    /// Packet number length (1-4 bytes).
    pub pn_length: u8,
}

impl ShortHeader {
    /// Create a new short header.
    pub fn new(dcid: &[u8]) -> Result<Self, NetError> {
        if dcid.len() > MAX_CID_LEN {
            return Err(NetError::InvalidAddress);
        }
        Ok(Self {
            spin_bit: false,
            key_phase: false,
            dcid: Vec::from(dcid),
            packet_number: 0,
            pn_length: 4,
        })
    }

    /// Encode the short header. Caller must know DCID length.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, NetError> {
        let total = 1 + self.dcid.len() + self.pn_length as usize;
        if buf.len() < total {
            return Err(NetError::BufferTooSmall);
        }

        // First byte: form=0, fixed=1, spin, reserved(2), key_phase, pn_len(2)
        let mut first_byte = FIXED_BIT; // form=0
        if self.spin_bit {
            first_byte |= 0x20;
        }
        if self.key_phase {
            first_byte |= 0x04;
        }
        first_byte |= (self.pn_length - 1) & 0x03;

        let mut off = 0;
        buf[off] = first_byte;
        off += 1;

        // DCID (length is implicit — receiver knows it)
        buf[off..off + self.dcid.len()].copy_from_slice(&self.dcid);
        off += self.dcid.len();

        // Packet number
        let pn_bytes = self.packet_number.to_be_bytes();
        let start = 4 - self.pn_length as usize;
        buf[off..off + self.pn_length as usize].copy_from_slice(&pn_bytes[start..]);
        off += self.pn_length as usize;

        Ok(off)
    }

    /// Decode a short header. `dcid_len` must be known from connection state.
    pub fn decode(data: &[u8], dcid_len: usize) -> Result<(Self, usize), NetError> {
        if data.is_empty() {
            return Err(NetError::PacketTooShort);
        }
        let first_byte = data[0];
        // Must be short header (form bit = 0)
        if first_byte & LONG_HEADER_FORM != 0 {
            return Err(NetError::InvalidHeader);
        }

        let spin_bit = first_byte & 0x20 != 0;
        let key_phase = first_byte & 0x04 != 0;
        let pn_length = (first_byte & 0x03) + 1;

        let mut off = 1;
        if off + dcid_len > data.len() {
            return Err(NetError::PacketTooShort);
        }
        let dcid = Vec::from(&data[off..off + dcid_len]);
        off += dcid_len;

        if off + pn_length as usize > data.len() {
            return Err(NetError::PacketTooShort);
        }
        let mut packet_number = 0u32;
        for i in 0..pn_length as usize {
            packet_number = (packet_number << 8) | data[off + i] as u32;
        }
        off += pn_length as usize;

        Ok((
            ShortHeader {
                spin_bit,
                key_phase,
                dcid,
                packet_number,
                pn_length,
            },
            off,
        ))
    }
}

/// Determine if a packet has a long header (first bit = 1) or short (first bit = 0).
pub fn is_long_header(first_byte: u8) -> bool {
    first_byte & LONG_HEADER_FORM != 0
}

// --- Variable-length integer encoding (RFC 9000 Section 16) ---

/// Encode a variable-length integer. Returns bytes written.
pub fn encode_var_int(value: u64, buf: &mut [u8]) -> usize {
    if value < 64 {
        buf[0] = value as u8;
        1
    } else if value < 16384 {
        let v = (value as u16) | 0x4000;
        buf[0..2].copy_from_slice(&v.to_be_bytes());
        2
    } else if value < 1_073_741_824 {
        let v = (value as u32) | 0x80000000;
        buf[0..4].copy_from_slice(&v.to_be_bytes());
        4
    } else {
        let v = value | 0xC000000000000000;
        buf[0..8].copy_from_slice(&v.to_be_bytes());
        8
    }
}

/// Decode a variable-length integer. Returns (value, bytes_consumed).
pub fn decode_var_int(data: &[u8]) -> Result<(u64, usize), NetError> {
    if data.is_empty() {
        return Err(NetError::PacketTooShort);
    }
    let prefix = data[0] >> 6;
    let length = 1 << prefix;
    if data.len() < length {
        return Err(NetError::PacketTooShort);
    }
    let value = match length {
        1 => data[0] as u64 & 0x3F,
        2 => {
            let v = u16::from_be_bytes([data[0], data[1]]);
            (v & 0x3FFF) as u64
        }
        4 => {
            let v = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
            (v & 0x3FFFFFFF) as u64
        }
        8 => {
            let v = u64::from_be_bytes([
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ]);
            v & 0x3FFFFFFFFFFFFFFF
        }
        _ => return Err(NetError::InvalidHeader),
    };
    Ok((value, length))
}

/// Returns the encoded size of a variable-length integer.
fn var_int_len(value: u64) -> usize {
    if value < 64 {
        1
    } else if value < 16384 {
        2
    } else if value < 1_073_741_824 {
        4
    } else {
        8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_quic_version() {
        let v = QuicVersion::v1();
        assert_eq!(v.0, 0x00000001);
        assert!(v.is_supported());
        assert!(!QuicVersion::negotiation().is_supported());
    }

    #[test]
    fn test_var_int_encode_decode_1byte() {
        let mut buf = [0u8; 8];
        let len = encode_var_int(37, &mut buf);
        assert_eq!(len, 1);
        let (val, consumed) = decode_var_int(&buf[..len]).unwrap();
        assert_eq!(val, 37);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_var_int_encode_decode_2byte() {
        let mut buf = [0u8; 8];
        let len = encode_var_int(15293, &mut buf);
        assert_eq!(len, 2);
        let (val, consumed) = decode_var_int(&buf[..len]).unwrap();
        assert_eq!(val, 15293);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_var_int_encode_decode_4byte() {
        let mut buf = [0u8; 8];
        let len = encode_var_int(494_878, &mut buf);
        assert_eq!(len, 4);
        let (val, consumed) = decode_var_int(&buf[..len]).unwrap();
        assert_eq!(val, 494_878);
        assert_eq!(consumed, 4);
    }

    #[test]
    fn test_var_int_encode_decode_8byte() {
        let mut buf = [0u8; 8];
        let val_in = 2_000_000_000u64;
        let len = encode_var_int(val_in, &mut buf);
        assert_eq!(len, 8);
        let (val, consumed) = decode_var_int(&buf[..len]).unwrap();
        assert_eq!(val, val_in);
        assert_eq!(consumed, 8);
    }

    #[test]
    fn test_var_int_boundary_values() {
        // 63 = max 1-byte
        let mut buf = [0u8; 8];
        assert_eq!(encode_var_int(63, &mut buf), 1);
        // 64 = min 2-byte
        assert_eq!(encode_var_int(64, &mut buf), 2);
        // 16383 = max 2-byte
        assert_eq!(encode_var_int(16383, &mut buf), 2);
        // 16384 = min 4-byte
        assert_eq!(encode_var_int(16384, &mut buf), 4);
    }

    #[test]
    fn test_var_int_decode_empty() {
        assert_eq!(decode_var_int(&[]).err(), Some(NetError::PacketTooShort));
    }

    #[test]
    fn test_long_header_initial_encode_decode() {
        let mut hdr = LongHeader::new(
            PacketType::Initial,
            &[0x01, 0x02, 0x03, 0x04],
            &[0xAA, 0xBB],
        )
        .unwrap();
        hdr.packet_number = 42;
        hdr.payload_length = 100;

        let mut buf = [0u8; 128];
        let len = hdr.encode(&mut buf).unwrap();
        let (decoded, consumed) = LongHeader::decode(&buf[..len]).unwrap();
        assert_eq!(consumed, len);
        assert_eq!(decoded.packet_type, PacketType::Initial);
        assert_eq!(decoded.version, QuicVersion::v1());
        assert_eq!(decoded.dcid, vec![0x01, 0x02, 0x03, 0x04]);
        assert_eq!(decoded.scid, vec![0xAA, 0xBB]);
        assert_eq!(decoded.packet_number, 42);
        assert_eq!(decoded.payload_length, 100);
    }

    #[test]
    fn test_long_header_handshake_encode_decode() {
        let mut hdr = LongHeader::new(PacketType::Handshake, &[0x01], &[0x02]).unwrap();
        hdr.packet_number = 1;
        hdr.payload_length = 50;

        let mut buf = [0u8; 64];
        let len = hdr.encode(&mut buf).unwrap();
        let (decoded, _) = LongHeader::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.packet_type, PacketType::Handshake);
    }

    #[test]
    fn test_long_header_zerortt_encode_decode() {
        let mut hdr = LongHeader::new(PacketType::ZeroRtt, &[0x01], &[0x02]).unwrap();
        hdr.packet_number = 0;
        hdr.payload_length = 200;

        let mut buf = [0u8; 64];
        let len = hdr.encode(&mut buf).unwrap();
        let (decoded, _) = LongHeader::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.packet_type, PacketType::ZeroRtt);
        assert_eq!(decoded.payload_length, 200);
    }

    #[test]
    fn test_long_header_initial_with_token() {
        let mut hdr = LongHeader::new(PacketType::Initial, &[0x01], &[0x02]).unwrap();
        hdr.token = vec![0xDE, 0xAD, 0xBE, 0xEF];
        hdr.packet_number = 0;
        hdr.payload_length = 500;

        let mut buf = [0u8; 128];
        let len = hdr.encode(&mut buf).unwrap();
        let (decoded, _) = LongHeader::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.token, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_long_header_cid_too_long() {
        let long_cid = vec![0u8; 21];
        assert_eq!(
            LongHeader::new(PacketType::Initial, &long_cid, &[]).err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_long_header_buffer_too_small() {
        let hdr = LongHeader::new(PacketType::Initial, &[0x01; 8], &[0x02; 8]).unwrap();
        let mut buf = [0u8; 5];
        assert_eq!(hdr.encode(&mut buf).err(), Some(NetError::BufferTooSmall));
    }

    #[test]
    fn test_long_header_decode_too_short() {
        assert_eq!(
            LongHeader::decode(&[0x80]).err(),
            Some(NetError::PacketTooShort)
        );
    }

    #[test]
    fn test_long_header_decode_not_long() {
        assert_eq!(
            LongHeader::decode(&[0x40, 0, 0, 0, 0, 0]).err(),
            Some(NetError::InvalidHeader)
        );
    }

    #[test]
    fn test_version_negotiation_decode() {
        // Build a version negotiation packet manually
        let mut buf = [0u8; 32];
        buf[0] = 0x80 | 0x40; // Long header form + fixed bit
                              // Version = 0
        buf[1..5].copy_from_slice(&0u32.to_be_bytes());
        // DCID len=2
        buf[5] = 2;
        buf[6] = 0x01;
        buf[7] = 0x02;
        // SCID len=1
        buf[8] = 1;
        buf[9] = 0xAA;
        // Supported versions follow
        buf[10..14].copy_from_slice(&QUIC_VERSION_1.to_be_bytes());

        let (decoded, _off) = LongHeader::decode(&buf[..14]).unwrap();
        assert_eq!(decoded.packet_type, PacketType::VersionNegotiation);
        assert_eq!(decoded.version.0, 0);
        assert_eq!(decoded.dcid, vec![0x01, 0x02]);
        assert_eq!(decoded.scid, vec![0xAA]);
    }

    #[test]
    fn test_short_header_encode_decode() {
        let mut hdr = ShortHeader::new(&[0x01, 0x02, 0x03, 0x04]).unwrap();
        hdr.packet_number = 99;
        hdr.spin_bit = true;
        hdr.key_phase = true;

        let mut buf = [0u8; 32];
        let len = hdr.encode(&mut buf).unwrap();
        let (decoded, consumed) = ShortHeader::decode(&buf[..len], 4).unwrap();
        assert_eq!(consumed, len);
        assert!(decoded.spin_bit);
        assert!(decoded.key_phase);
        assert_eq!(decoded.dcid, vec![0x01, 0x02, 0x03, 0x04]);
        assert_eq!(decoded.packet_number, 99);
    }

    #[test]
    fn test_short_header_cid_too_long() {
        let long_cid = vec![0u8; 21];
        assert_eq!(
            ShortHeader::new(&long_cid).err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_short_header_buffer_too_small() {
        let hdr = ShortHeader::new(&[0x01; 8]).unwrap();
        let mut buf = [0u8; 3];
        assert_eq!(hdr.encode(&mut buf).err(), Some(NetError::BufferTooSmall));
    }

    #[test]
    fn test_short_header_decode_too_short() {
        assert_eq!(
            ShortHeader::decode(&[], 4).err(),
            Some(NetError::PacketTooShort)
        );
    }

    #[test]
    fn test_short_header_decode_not_short() {
        assert_eq!(
            ShortHeader::decode(&[0xC0], 0).err(),
            Some(NetError::InvalidHeader)
        );
    }

    #[test]
    fn test_is_long_header() {
        assert!(is_long_header(0x80));
        assert!(is_long_header(0xC0));
        assert!(!is_long_header(0x40));
        assert!(!is_long_header(0x00));
    }

    #[test]
    fn test_short_header_no_spin_no_key_phase() {
        let mut hdr = ShortHeader::new(&[0x01]).unwrap();
        hdr.packet_number = 0;
        hdr.spin_bit = false;
        hdr.key_phase = false;

        let mut buf = [0u8; 32];
        let len = hdr.encode(&mut buf).unwrap();
        let (decoded, _) = ShortHeader::decode(&buf[..len], 1).unwrap();
        assert!(!decoded.spin_bit);
        assert!(!decoded.key_phase);
    }

    #[test]
    fn test_packet_type_roundtrip() {
        for pt in [
            PacketType::Initial,
            PacketType::ZeroRtt,
            PacketType::Handshake,
            PacketType::Retry,
        ] {
            let bits = pt.to_type_bits();
            assert_eq!(PacketType::from_type_bits(bits), pt);
        }
    }
}
