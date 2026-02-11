// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RTPS 2.3 wire protocol message encoding.
//!
//! Implements the RTPS message header and submessage encode/decode for
//! DATA, HEARTBEAT, ACKNACK, and GAP submessages.

use alloc::vec::Vec;
use crate::BusError;
use crate::dds::types::GuidPrefix;

/// RTPS protocol identifier: "RTPS".
pub const RTPS_MAGIC: [u8; 4] = [0x52, 0x54, 0x50, 0x53];

/// RTPS protocol version 2.3.
pub const RTPS_VERSION_MAJOR: u8 = 2;
pub const RTPS_VERSION_MINOR: u8 = 3;

/// RTPS vendor ID for SmallAIOS (custom).
pub const VENDOR_ID: [u8; 2] = [0x01, 0x42];

/// RTPS message header size.
pub const RTPS_HEADER_LEN: usize = 20;

/// Submessage header size.
pub const SUBMSG_HEADER_LEN: usize = 4;

/// RTPS submessage kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmessageKind {
    /// DATA submessage (carries a sample).
    Data,
    /// HEARTBEAT submessage (writer announces available samples).
    Heartbeat,
    /// ACKNACK submessage (reader acknowledges/requests samples).
    AckNack,
    /// GAP submessage (writer declares irrelevant samples).
    Gap,
    /// INFO_TS submessage (timestamp).
    InfoTimestamp,
    /// Unknown/unsupported.
    Unknown(u8),
}

impl SubmessageKind {
    fn to_id(self) -> u8 {
        match self {
            SubmessageKind::Data => 0x15,
            SubmessageKind::Heartbeat => 0x07,
            SubmessageKind::AckNack => 0x06,
            SubmessageKind::Gap => 0x08,
            SubmessageKind::InfoTimestamp => 0x09,
            SubmessageKind::Unknown(id) => id,
        }
    }

    fn from_id(id: u8) -> Self {
        match id {
            0x15 => SubmessageKind::Data,
            0x07 => SubmessageKind::Heartbeat,
            0x06 => SubmessageKind::AckNack,
            0x08 => SubmessageKind::Gap,
            0x09 => SubmessageKind::InfoTimestamp,
            other => SubmessageKind::Unknown(other),
        }
    }
}

/// RTPS message header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpsHeader {
    /// Protocol version (major, minor).
    pub version: (u8, u8),
    /// Vendor ID.
    pub vendor_id: [u8; 2],
    /// GUID prefix of the sender.
    pub guid_prefix: GuidPrefix,
}

impl RtpsHeader {
    /// Create a new RTPS header with SmallAIOS vendor ID.
    pub fn new(guid_prefix: GuidPrefix) -> Self {
        Self {
            version: (RTPS_VERSION_MAJOR, RTPS_VERSION_MINOR),
            vendor_id: VENDOR_ID,
            guid_prefix,
        }
    }

    /// Encode the header into a byte buffer.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        if buf.len() < RTPS_HEADER_LEN {
            return Err(BusError::BufferTooSmall);
        }
        buf[0..4].copy_from_slice(&RTPS_MAGIC);
        buf[4] = self.version.0;
        buf[5] = self.version.1;
        buf[6..8].copy_from_slice(&self.vendor_id);
        buf[8..20].copy_from_slice(&self.guid_prefix.0);
        Ok(RTPS_HEADER_LEN)
    }

    /// Decode a header from a byte buffer.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < RTPS_HEADER_LEN {
            return Err(BusError::FrameTooShort);
        }
        if data[0..4] != RTPS_MAGIC {
            return Err(BusError::InvalidHeader);
        }
        let version = (data[4], data[5]);
        let mut vendor_id = [0u8; 2];
        vendor_id.copy_from_slice(&data[6..8]);
        let mut prefix = [0u8; 12];
        prefix.copy_from_slice(&data[8..20]);
        Ok(Self {
            version,
            vendor_id,
            guid_prefix: GuidPrefix(prefix),
        })
    }
}

/// RTPS submessage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submessage {
    /// Submessage kind.
    pub kind: SubmessageKind,
    /// Flags byte.
    pub flags: u8,
    /// Submessage payload.
    pub payload: Vec<u8>,
}

impl Submessage {
    /// Create a new submessage.
    pub fn new(kind: SubmessageKind, flags: u8, payload: &[u8]) -> Self {
        Self {
            kind,
            flags,
            payload: Vec::from(payload),
        }
    }

    /// Encode the submessage into a byte buffer.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        let total = SUBMSG_HEADER_LEN + self.payload.len();
        if buf.len() < total {
            return Err(BusError::BufferTooSmall);
        }
        buf[0] = self.kind.to_id();
        buf[1] = self.flags;
        let len = self.payload.len() as u16;
        // Little-endian length (RTPS default for submessage length)
        buf[2] = len as u8;
        buf[3] = (len >> 8) as u8;
        buf[SUBMSG_HEADER_LEN..total].copy_from_slice(&self.payload);
        Ok(total)
    }

    /// Decode a submessage from a byte buffer.
    pub fn decode(data: &[u8]) -> Result<(Self, usize), BusError> {
        if data.len() < SUBMSG_HEADER_LEN {
            return Err(BusError::FrameTooShort);
        }
        let kind = SubmessageKind::from_id(data[0]);
        let flags = data[1];
        let len = (data[2] as u16) | ((data[3] as u16) << 8);
        let total = SUBMSG_HEADER_LEN + len as usize;
        if data.len() < total {
            return Err(BusError::FrameTooShort);
        }
        let payload = Vec::from(&data[SUBMSG_HEADER_LEN..total]);
        Ok((Self { kind, flags, payload }, total))
    }
}

/// Complete RTPS message (header + submessages).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpsMessage {
    /// Message header.
    pub header: RtpsHeader,
    /// Submessages.
    pub submessages: Vec<Submessage>,
}

impl RtpsMessage {
    /// Create a new RTPS message.
    pub fn new(header: RtpsHeader) -> Self {
        Self {
            header,
            submessages: Vec::new(),
        }
    }

    /// Add a submessage.
    pub fn add_submessage(&mut self, submsg: Submessage) {
        self.submessages.push(submsg);
    }

    /// Encode the complete message into a byte buffer.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        let mut offset = self.header.encode(buf)?;
        for submsg in &self.submessages {
            offset += submsg.encode(&mut buf[offset..])?;
        }
        Ok(offset)
    }

    /// Decode a complete RTPS message from a byte buffer.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        let header = RtpsHeader::decode(data)?;
        let mut offset = RTPS_HEADER_LEN;
        let mut submessages = Vec::new();
        while offset < data.len() {
            if data.len() - offset < SUBMSG_HEADER_LEN {
                break; // Not enough for another submessage header
            }
            let (submsg, consumed) = Submessage::decode(&data[offset..])?;
            submessages.push(submsg);
            offset += consumed;
        }
        Ok(Self { header, submessages })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_prefix() -> GuidPrefix {
        GuidPrefix::new([0xAA; 12])
    }

    #[test]
    fn test_rtps_header_encode_decode() {
        let header = RtpsHeader::new(test_prefix());
        let mut buf = [0u8; 64];
        let len = header.encode(&mut buf).unwrap();
        assert_eq!(len, RTPS_HEADER_LEN);
        let decoded = RtpsHeader::decode(&buf[..len]).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_rtps_header_magic() {
        let header = RtpsHeader::new(test_prefix());
        let mut buf = [0u8; 64];
        header.encode(&mut buf).unwrap();
        assert_eq!(&buf[0..4], b"RTPS");
    }

    #[test]
    fn test_rtps_header_invalid_magic() {
        let mut buf = [0u8; 20];
        buf[0..4].copy_from_slice(b"NOPE");
        assert_eq!(RtpsHeader::decode(&buf).err(), Some(BusError::InvalidHeader));
    }

    #[test]
    fn test_rtps_header_too_short() {
        assert_eq!(RtpsHeader::decode(&[0u8; 10]).err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_rtps_header_version() {
        let header = RtpsHeader::new(test_prefix());
        assert_eq!(header.version, (2, 3));
    }

    #[test]
    fn test_submessage_encode_decode() {
        let submsg = Submessage::new(SubmessageKind::Data, 0x05, &[0x01, 0x02, 0x03]);
        let mut buf = [0u8; 64];
        let len = submsg.encode(&mut buf).unwrap();
        let (decoded, consumed) = Submessage::decode(&buf[..len]).unwrap();
        assert_eq!(consumed, len);
        assert_eq!(decoded, submsg);
    }

    #[test]
    fn test_submessage_kind_roundtrip() {
        for kind in [
            SubmessageKind::Data,
            SubmessageKind::Heartbeat,
            SubmessageKind::AckNack,
            SubmessageKind::Gap,
            SubmessageKind::InfoTimestamp,
        ] {
            let submsg = Submessage::new(kind, 0, &[0xAA]);
            let mut buf = [0u8; 32];
            let len = submsg.encode(&mut buf).unwrap();
            let (decoded, _) = Submessage::decode(&buf[..len]).unwrap();
            assert_eq!(decoded.kind, kind);
        }
    }

    #[test]
    fn test_submessage_unknown_kind() {
        let submsg = Submessage::new(SubmessageKind::Unknown(0xFE), 0, &[]);
        let mut buf = [0u8; 32];
        let len = submsg.encode(&mut buf).unwrap();
        let (decoded, _) = Submessage::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.kind, SubmessageKind::Unknown(0xFE));
    }

    #[test]
    fn test_submessage_too_short() {
        assert_eq!(Submessage::decode(&[0u8; 2]).err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_rtps_message_encode_decode() {
        let header = RtpsHeader::new(test_prefix());
        let mut msg = RtpsMessage::new(header);
        msg.add_submessage(Submessage::new(SubmessageKind::Data, 0x05, &[0x01, 0x02]));
        msg.add_submessage(Submessage::new(SubmessageKind::Heartbeat, 0x00, &[0xAA; 8]));

        let mut buf = [0u8; 256];
        let len = msg.encode(&mut buf).unwrap();
        let decoded = RtpsMessage::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.header, msg.header);
        assert_eq!(decoded.submessages.len(), 2);
        assert_eq!(decoded.submessages[0].kind, SubmessageKind::Data);
        assert_eq!(decoded.submessages[1].kind, SubmessageKind::Heartbeat);
    }

    #[test]
    fn test_rtps_message_empty() {
        let header = RtpsHeader::new(test_prefix());
        let msg = RtpsMessage::new(header);
        let mut buf = [0u8; 64];
        let len = msg.encode(&mut buf).unwrap();
        assert_eq!(len, RTPS_HEADER_LEN);
        let decoded = RtpsMessage::decode(&buf[..len]).unwrap();
        assert!(decoded.submessages.is_empty());
    }

    #[test]
    fn test_rtps_message_encode_buffer_too_small() {
        let header = RtpsHeader::new(test_prefix());
        let mut msg = RtpsMessage::new(header);
        msg.add_submessage(Submessage::new(SubmessageKind::Data, 0, &[0xAA; 100]));
        let mut buf = [0u8; 10];
        assert_eq!(msg.encode(&mut buf).err(), Some(BusError::BufferTooSmall));
    }

    #[test]
    fn test_submessage_length_little_endian() {
        let submsg = Submessage::new(SubmessageKind::Data, 0, &[0xAA; 300]);
        let mut buf = [0u8; 512];
        submsg.encode(&mut buf).unwrap();
        // Length 300 = 0x012C, little-endian: [0x2C, 0x01]
        assert_eq!(buf[2], 0x2C);
        assert_eq!(buf[3], 0x01);
    }
}
