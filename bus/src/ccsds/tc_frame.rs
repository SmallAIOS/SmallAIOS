// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CCSDS Telecommand (TC) Transfer Frame (CCSDS 232.0-B).
//!
//! A TC frame encapsulates Space Packets for uplink with CRC-16 integrity.

use alloc::vec::Vec;
use crate::BusError;

/// TC frame header size in bytes.
pub const TC_HEADER_LEN: usize = 5;

/// CRC-16 size in bytes (FECF).
pub const FECF_LEN: usize = 2;

/// CRC-16 polynomial for TC FECF (CCITT).
const CRC16_POLY: u16 = 0x1021;

/// Compute CRC-16/CCITT over a byte slice.
pub fn crc16_ccitt(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ CRC16_POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// TC Transfer Frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcFrame {
    /// Transfer Frame Version (2 bits) — always 00.
    pub version: u8,
    /// Bypass flag (Type-B if set).
    pub bypass: bool,
    /// Control Command flag.
    pub control_cmd: bool,
    /// Spacecraft ID (10 bits).
    pub scid: u16,
    /// Virtual Channel ID (6 bits).
    pub vcid: u8,
    /// Frame Sequence Number (8 bits).
    pub frame_seq: u8,
    /// Frame data (Space Packets).
    pub data: Vec<u8>,
}

impl TcFrame {
    /// Create a new TC frame.
    pub fn new(
        bypass: bool,
        control_cmd: bool,
        scid: u16,
        vcid: u8,
        frame_seq: u8,
        data: &[u8],
    ) -> Result<Self, BusError> {
        if scid > 0x3FF {
            return Err(BusError::OutOfRange);
        }
        if vcid > 63 {
            return Err(BusError::OutOfRange);
        }
        Ok(Self {
            version: 0,
            bypass,
            control_cmd,
            scid,
            vcid,
            frame_seq,
            data: Vec::from(data),
        })
    }

    /// Encode the TC frame into a byte buffer (including CRC-16 FECF).
    ///
    /// Header layout:
    /// - Byte 0-1: version(2) + bypass(1) + ctrl_cmd(1) + spare(2) + scid(10)
    /// - Byte 2: vcid(6) + spare(2)
    /// - Byte 3: frame_length(8) — not full spec, simplified
    /// - Byte 4: frame_seq(8)
    /// - Data + FECF(2)
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        let total = TC_HEADER_LEN + self.data.len() + FECF_LEN;
        if buf.len() < total {
            return Err(BusError::BufferTooSmall);
        }

        // Bytes 0-1: version(2) + bypass(1) + ctrl(1) + spare(2) + scid(10)
        let bypass_bit = if self.bypass { 1u16 } else { 0 };
        let ctrl_bit = if self.control_cmd { 1u16 } else { 0 };
        let w0: u16 = ((self.version as u16 & 0x03) << 14)
            | (bypass_bit << 13)
            | (ctrl_bit << 12)
            | (self.scid & 0x3FF);
        buf[0] = (w0 >> 8) as u8;
        buf[1] = w0 as u8;

        // Byte 2: vcid(6) + spare(2)
        buf[2] = (self.vcid & 0x3F) << 2;

        // Byte 3: frame data length (simplified: data.len() as u8)
        let frame_len = (self.data.len() + FECF_LEN) as u8;
        buf[3] = frame_len;

        // Byte 4: frame sequence number
        buf[4] = self.frame_seq;

        // Data
        buf[TC_HEADER_LEN..TC_HEADER_LEN + self.data.len()].copy_from_slice(&self.data);

        // CRC-16 over header + data
        let crc = crc16_ccitt(&buf[..TC_HEADER_LEN + self.data.len()]);
        let crc_offset = TC_HEADER_LEN + self.data.len();
        buf[crc_offset] = (crc >> 8) as u8;
        buf[crc_offset + 1] = crc as u8;

        Ok(total)
    }

    /// Decode a TC frame from a byte buffer (verifying CRC-16 FECF).
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < TC_HEADER_LEN + FECF_LEN {
            return Err(BusError::FrameTooShort);
        }

        // Verify CRC
        let payload_len = data.len() - FECF_LEN;
        let expected_crc = crc16_ccitt(&data[..payload_len]);
        let received_crc = ((data[payload_len] as u16) << 8) | data[payload_len + 1] as u16;
        if expected_crc != received_crc {
            return Err(BusError::CrcMismatch);
        }

        let w0 = ((data[0] as u16) << 8) | data[1] as u16;
        let version = ((w0 >> 14) & 0x03) as u8;
        let bypass = (w0 >> 13) & 1 == 1;
        let control_cmd = (w0 >> 12) & 1 == 1;
        let scid = w0 & 0x3FF;

        let vcid = (data[2] >> 2) & 0x3F;
        let frame_seq = data[4];

        let frame_data = Vec::from(&data[TC_HEADER_LEN..payload_len]);

        Ok(Self {
            version,
            bypass,
            control_cmd,
            scid,
            vcid,
            frame_seq,
            data: frame_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc16_deterministic() {
        let data = [0x01, 0x02, 0x03];
        let crc = crc16_ccitt(&data);
        assert_eq!(crc16_ccitt(&data), crc);
    }

    #[test]
    fn test_crc16_changes_with_data() {
        let crc1 = crc16_ccitt(&[0x00]);
        let crc2 = crc16_ccitt(&[0x01]);
        assert_ne!(crc1, crc2);
    }

    #[test]
    fn test_tc_frame_new() {
        let frame = TcFrame::new(false, false, 0x100, 3, 42, &[0xAA; 20]).unwrap();
        assert_eq!(frame.scid, 0x100);
        assert_eq!(frame.vcid, 3);
        assert_eq!(frame.frame_seq, 42);
        assert!(!frame.bypass);
    }

    #[test]
    fn test_tc_frame_invalid_scid() {
        assert_eq!(TcFrame::new(false, false, 0x400, 0, 0, &[0x00]).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_tc_frame_invalid_vcid() {
        assert_eq!(TcFrame::new(false, false, 0, 64, 0, &[0x00]).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_tc_frame_encode_decode_roundtrip() {
        let frame = TcFrame::new(true, false, 0x1A3, 5, 99, &[0x01, 0x02, 0x03]).unwrap();
        let mut buf = [0u8; 256];
        let len = frame.encode(&mut buf).unwrap();
        let decoded = TcFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.scid, 0x1A3);
        assert_eq!(decoded.vcid, 5);
        assert_eq!(decoded.frame_seq, 99);
        assert!(decoded.bypass);
        assert_eq!(decoded.data, &[0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_tc_frame_crc_mismatch() {
        let frame = TcFrame::new(false, false, 0, 0, 0, &[0xAA]).unwrap();
        let mut buf = [0u8; 64];
        let len = frame.encode(&mut buf).unwrap();
        // Corrupt CRC
        buf[len - 1] ^= 0xFF;
        assert_eq!(TcFrame::decode(&buf[..len]).err(), Some(BusError::CrcMismatch));
    }

    #[test]
    fn test_tc_frame_decode_too_short() {
        assert_eq!(TcFrame::decode(&[0u8; 6]).err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_tc_frame_encode_buffer_too_small() {
        let frame = TcFrame::new(false, false, 0, 0, 0, &[0xAA; 100]).unwrap();
        let mut buf = [0u8; 10];
        assert_eq!(frame.encode(&mut buf).err(), Some(BusError::BufferTooSmall));
    }

    #[test]
    fn test_tc_frame_bypass_flag() {
        let frame = TcFrame::new(true, false, 0, 0, 0, &[0x00]).unwrap();
        let mut buf = [0u8; 64];
        let len = frame.encode(&mut buf).unwrap();
        let decoded = TcFrame::decode(&buf[..len]).unwrap();
        assert!(decoded.bypass);
    }

    #[test]
    fn test_tc_frame_control_cmd_flag() {
        let frame = TcFrame::new(false, true, 0, 0, 0, &[0x00]).unwrap();
        let mut buf = [0u8; 64];
        let len = frame.encode(&mut buf).unwrap();
        let decoded = TcFrame::decode(&buf[..len]).unwrap();
        assert!(decoded.control_cmd);
    }
}
