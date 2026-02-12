// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CAN 2.0A and CAN 2.0B frame encode/decode.
//!
//! Clean-room implementation from ISO 11898 and BOSCH CAN 2.0 specification.
//!
//! # Wire format (logical, pre-bit-stuffing)
//!
//! ## CAN 2.0A (standard frame)
//! ```text
//! | SOF(1) | ID[10:0](11) | RTR(1) | IDE(0)(1) | r0(1) | DLC(4) | Data(0-64) | CRC(15) | CRC_DEL(1) | ACK(1) | ACK_DEL(1) | EOF(7) |
//! ```
//!
//! ## CAN 2.0B (extended frame)
//! ```text
//! | SOF(1) | ID[28:18](11) | SRR(1) | IDE(1)(1) | ID[17:0](18) | RTR(1) | r1(1) | r0(1) | DLC(4) | Data(0-64) | CRC(15) | ... |
//! ```
//!
//! # Serialized byte format
//!
//! This module uses a compact byte serialization for software transport:
//! ```text
//! Byte 0:       Flags (bit 0 = IDE, bit 1 = RTR)
//! Bytes 1-4:    Identifier (big-endian, 11-bit or 29-bit depending on IDE)
//! Byte 5:       DLC (0-8)
//! Bytes 6..6+N: Data payload (N = DLC bytes)
//! Bytes -2..-1: CRC-15 (big-endian, 15-bit value in u16)
//! ```

use alloc::vec::Vec;

use super::crc::crc15;
use crate::BusError;

/// Maximum payload size for CAN 2.0A/B frames.
pub const MAX_CLASSIC_PAYLOAD: usize = 8;

/// Maximum 11-bit identifier value.
pub const MAX_STANDARD_ID: u32 = 0x7FF;

/// Maximum 29-bit identifier value.
pub const MAX_EXTENDED_ID: u32 = 0x1FFF_FFFF;

/// Flag byte: IDE bit (extended frame).
const FLAG_IDE: u8 = 0x01;

/// Flag byte: RTR bit (remote transmission request).
const FLAG_RTR: u8 = 0x02;

/// Minimum serialized frame size: flags(1) + id(4) + dlc(1) + crc(2) = 8.
const MIN_FRAME_BYTES: usize = 8;

/// CAN frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// CAN 2.0A standard frame (11-bit identifier).
    Standard,
    /// CAN 2.0B extended frame (29-bit identifier).
    Extended,
}

/// A CAN 2.0A or CAN 2.0B frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanFrame {
    /// Frame type (standard 11-bit or extended 29-bit).
    pub frame_type: FrameType,
    /// CAN identifier (11-bit for standard, 29-bit for extended).
    pub id: u32,
    /// Remote Transmission Request flag.
    pub rtr: bool,
    /// Data payload (0-8 bytes).
    pub data: Vec<u8>,
    /// CRC-15 value (computed on encode, verified on decode).
    pub crc: u16,
}

impl CanFrame {
    /// Create a new CAN 2.0A standard frame.
    ///
    /// Returns [`BusError::InvalidId`] if `id` exceeds 11 bits (0x7FF).
    /// Returns [`BusError::PayloadTooLarge`] if `data` exceeds 8 bytes.
    pub fn new_standard(id: u32, data: &[u8]) -> Result<Self, BusError> {
        if id > MAX_STANDARD_ID {
            return Err(BusError::InvalidId);
        }
        if data.len() > MAX_CLASSIC_PAYLOAD {
            return Err(BusError::PayloadTooLarge);
        }
        let mut frame = CanFrame {
            frame_type: FrameType::Standard,
            id,
            rtr: false,
            data: Vec::from(data),
            crc: 0,
        };
        frame.crc = frame.compute_crc();
        Ok(frame)
    }

    /// Create a new CAN 2.0B extended frame.
    ///
    /// Returns [`BusError::InvalidId`] if `id` exceeds 29 bits (0x1FFFFFFF).
    /// Returns [`BusError::PayloadTooLarge`] if `data` exceeds 8 bytes.
    pub fn new_extended(id: u32, data: &[u8]) -> Result<Self, BusError> {
        if id > MAX_EXTENDED_ID {
            return Err(BusError::InvalidId);
        }
        if data.len() > MAX_CLASSIC_PAYLOAD {
            return Err(BusError::PayloadTooLarge);
        }
        let mut frame = CanFrame {
            frame_type: FrameType::Extended,
            id,
            rtr: false,
            data: Vec::from(data),
            crc: 0,
        };
        frame.crc = frame.compute_crc();
        Ok(frame)
    }

    /// Create a new RTR (Remote Transmission Request) frame.
    ///
    /// RTR frames have no data payload; `dlc` indicates the expected response length.
    pub fn new_rtr(frame_type: FrameType, id: u32, dlc: u8) -> Result<Self, BusError> {
        let max_id = match frame_type {
            FrameType::Standard => MAX_STANDARD_ID,
            FrameType::Extended => MAX_EXTENDED_ID,
        };
        if id > max_id {
            return Err(BusError::InvalidId);
        }
        if dlc > 8 {
            return Err(BusError::InvalidDlc);
        }
        // RTR frames carry the DLC but no actual data. We store empty data
        // and encode the DLC separately to preserve the requested length.
        let mut frame = CanFrame {
            frame_type,
            id,
            rtr: true,
            data: Vec::new(),
            crc: 0,
        };
        // Store DLC in a temporary way: we use an empty vec but will encode
        // the requested DLC. For simplicity, we fill data with zeros to DLC
        // length so the DLC field serializes correctly. RTR receivers know
        // to ignore the data bytes.
        frame.data.resize(dlc as usize, 0);
        frame.crc = frame.compute_crc();
        Ok(frame)
    }

    /// Returns the Data Length Code (number of data bytes).
    pub fn dlc(&self) -> u8 {
        self.data.len() as u8
    }

    /// Returns the base identifier (11-bit, bits [28:18] for extended frames).
    pub fn base_id(&self) -> u16 {
        match self.frame_type {
            FrameType::Standard => self.id as u16,
            FrameType::Extended => ((self.id >> 18) & 0x7FF) as u16,
        }
    }

    /// Returns the extension identifier (18-bit, bits [17:0]) for extended frames.
    /// Returns 0 for standard frames.
    pub fn extension_id(&self) -> u32 {
        match self.frame_type {
            FrameType::Standard => 0,
            FrameType::Extended => self.id & 0x3FFFF,
        }
    }

    /// Compute CRC-15 over the frame fields that are covered by the CRC.
    ///
    /// Per ISO 11898, the CRC covers: SOF, arbitration field, control field,
    /// and data field. We serialize these fields into a byte buffer and compute
    /// CRC-15 over the result.
    fn compute_crc(&self) -> u16 {
        let mut buf = Vec::with_capacity(6 + self.data.len());
        // Flags byte (IDE, RTR)
        let mut flags = 0u8;
        if self.frame_type == FrameType::Extended {
            flags |= FLAG_IDE;
        }
        if self.rtr {
            flags |= FLAG_RTR;
        }
        buf.push(flags);
        // Identifier (4 bytes, big-endian)
        buf.extend_from_slice(&self.id.to_be_bytes());
        // DLC
        buf.push(self.dlc());
        // Data
        buf.extend_from_slice(&self.data);
        crc15(&buf)
    }

    /// Verify the CRC-15 of this frame.
    pub fn verify_crc(&self) -> bool {
        self.crc == self.compute_crc()
    }

    /// Encode the frame into a byte vector.
    ///
    /// Format: flags(1) + id(4) + dlc(1) + data(0-8) + crc(2)
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(MIN_FRAME_BYTES + self.data.len());
        // Flags byte
        let mut flags = 0u8;
        if self.frame_type == FrameType::Extended {
            flags |= FLAG_IDE;
        }
        if self.rtr {
            flags |= FLAG_RTR;
        }
        buf.push(flags);
        // Identifier (4 bytes, big-endian)
        buf.extend_from_slice(&self.id.to_be_bytes());
        // DLC
        buf.push(self.dlc());
        // Data payload
        buf.extend_from_slice(&self.data);
        // CRC-15 (2 bytes, big-endian)
        buf.extend_from_slice(&self.crc.to_be_bytes());
        buf
    }

    /// Decode a CAN frame from a byte slice.
    ///
    /// Returns [`BusError::FrameTooShort`] if `data` is too short.
    /// Returns [`BusError::CrcMismatch`] if the CRC does not match.
    /// Returns [`BusError::InvalidId`] if the identifier exceeds the valid range.
    /// Returns [`BusError::InvalidDlc`] if the DLC exceeds 8.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < MIN_FRAME_BYTES {
            return Err(BusError::FrameTooShort);
        }

        let flags = data[0];
        let ide = (flags & FLAG_IDE) != 0;
        let rtr = (flags & FLAG_RTR) != 0;

        let id = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
        let dlc = data[5];

        if dlc > 8 {
            return Err(BusError::InvalidDlc);
        }

        let frame_type = if ide {
            FrameType::Extended
        } else {
            FrameType::Standard
        };

        // Validate ID range
        let max_id = match frame_type {
            FrameType::Standard => MAX_STANDARD_ID,
            FrameType::Extended => MAX_EXTENDED_ID,
        };
        if id > max_id {
            return Err(BusError::InvalidId);
        }

        let payload_len = dlc as usize;
        let expected_len = 6 + payload_len + 2; // header + payload + crc
        if data.len() < expected_len {
            return Err(BusError::FrameTooShort);
        }

        let payload = Vec::from(&data[6..6 + payload_len]);
        let crc = u16::from_be_bytes([data[6 + payload_len], data[6 + payload_len + 1]]);

        let frame = CanFrame {
            frame_type,
            id,
            rtr,
            data: payload,
            crc,
        };

        if !frame.verify_crc() {
            return Err(BusError::CrcMismatch);
        }

        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // -----------------------------------------------------------------------
    // CAN 2.0A standard frame tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_standard_valid() {
        let frame = CanFrame::new_standard(0x123, &[0x01, 0x02, 0x03]).unwrap();
        assert_eq!(frame.frame_type, FrameType::Standard);
        assert_eq!(frame.id, 0x123);
        assert_eq!(frame.dlc(), 3);
        assert_eq!(frame.data, vec![0x01, 0x02, 0x03]);
        assert!(!frame.rtr);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_standard_empty_payload() {
        let frame = CanFrame::new_standard(0x000, &[]).unwrap();
        assert_eq!(frame.dlc(), 0);
        assert!(frame.data.is_empty());
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_standard_max_payload() {
        let data = [0xAA; 8];
        let frame = CanFrame::new_standard(0x7FF, &data).unwrap();
        assert_eq!(frame.dlc(), 8);
        assert_eq!(frame.id, MAX_STANDARD_ID);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_standard_id_too_large() {
        let err = CanFrame::new_standard(0x800, &[0x01]).unwrap_err();
        assert_eq!(err, BusError::InvalidId);
    }

    #[test]
    fn test_new_standard_payload_too_large() {
        let data = [0x00; 9];
        let err = CanFrame::new_standard(0x100, &data).unwrap_err();
        assert_eq!(err, BusError::PayloadTooLarge);
    }

    // -----------------------------------------------------------------------
    // CAN 2.0B extended frame tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_extended_valid() {
        let frame = CanFrame::new_extended(0x1234_5678, &[0xDE, 0xAD]).unwrap();
        assert_eq!(frame.frame_type, FrameType::Extended);
        assert_eq!(frame.id, 0x1234_5678);
        assert_eq!(frame.dlc(), 2);
        assert!(!frame.rtr);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_extended_max_id() {
        let frame = CanFrame::new_extended(MAX_EXTENDED_ID, &[0xFF]).unwrap();
        assert_eq!(frame.id, 0x1FFF_FFFF);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_extended_id_too_large() {
        let err = CanFrame::new_extended(0x2000_0000, &[0x01]).unwrap_err();
        assert_eq!(err, BusError::InvalidId);
    }

    #[test]
    fn test_new_extended_payload_too_large() {
        let data = [0x00; 9];
        let err = CanFrame::new_extended(0x100, &data).unwrap_err();
        assert_eq!(err, BusError::PayloadTooLarge);
    }

    #[test]
    fn test_extended_base_and_extension_id() {
        // ID = 0x1234_5678
        // base_id = bits [28:18] = 0x1234_5678 >> 18 = 0x48D = 1165
        // extension_id = bits [17:0] = 0x1234_5678 & 0x3FFFF = 0x15678
        let frame = CanFrame::new_extended(0x1234_5678, &[]).unwrap();
        assert_eq!(frame.base_id(), ((0x1234_5678u32 >> 18) & 0x7FF) as u16);
        assert_eq!(frame.extension_id(), 0x1234_5678u32 & 0x3FFFF);
    }

    #[test]
    fn test_standard_base_id() {
        let frame = CanFrame::new_standard(0x3AB, &[]).unwrap();
        assert_eq!(frame.base_id(), 0x3AB);
        assert_eq!(frame.extension_id(), 0);
    }

    // -----------------------------------------------------------------------
    // RTR frame tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_rtr_standard() {
        let frame = CanFrame::new_rtr(FrameType::Standard, 0x100, 4).unwrap();
        assert!(frame.rtr);
        assert_eq!(frame.dlc(), 4);
        assert_eq!(frame.frame_type, FrameType::Standard);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_rtr_extended() {
        let frame = CanFrame::new_rtr(FrameType::Extended, 0x1000, 8).unwrap();
        assert!(frame.rtr);
        assert_eq!(frame.dlc(), 8);
        assert_eq!(frame.frame_type, FrameType::Extended);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_rtr_invalid_dlc() {
        let err = CanFrame::new_rtr(FrameType::Standard, 0x100, 9).unwrap_err();
        assert_eq!(err, BusError::InvalidDlc);
    }

    #[test]
    fn test_new_rtr_invalid_id() {
        let err = CanFrame::new_rtr(FrameType::Standard, 0x800, 4).unwrap_err();
        assert_eq!(err, BusError::InvalidId);
    }

    // -----------------------------------------------------------------------
    // Encode/decode roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_decode_standard_roundtrip() {
        let original = CanFrame::new_standard(0x123, &[0x01, 0x02, 0x03, 0x04]).unwrap();
        let encoded = original.encode();
        let decoded = CanFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_decode_extended_roundtrip() {
        let original = CanFrame::new_extended(0x1234_5678, &[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        let encoded = original.encode();
        let decoded = CanFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_decode_empty_payload_roundtrip() {
        let original = CanFrame::new_standard(0x000, &[]).unwrap();
        let encoded = original.encode();
        let decoded = CanFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_decode_max_payload_roundtrip() {
        let data = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let original = CanFrame::new_standard(0x7FF, &data).unwrap();
        let encoded = original.encode();
        let decoded = CanFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_decode_rtr_roundtrip() {
        let original = CanFrame::new_rtr(FrameType::Extended, 0xABCDE, 6).unwrap();
        let encoded = original.encode();
        let decoded = CanFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    // -----------------------------------------------------------------------
    // Decode error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_too_short() {
        let data = [0u8; 7]; // less than MIN_FRAME_BYTES
        assert_eq!(CanFrame::decode(&data), Err(BusError::FrameTooShort));
    }

    #[test]
    fn test_decode_invalid_dlc() {
        // Construct a frame with DLC = 9
        let mut data = [0u8; 10];
        data[5] = 9; // invalid DLC
        assert_eq!(CanFrame::decode(&data), Err(BusError::InvalidDlc));
    }

    #[test]
    fn test_decode_truncated_payload() {
        // Valid header with DLC=4 but only 1 byte of payload + no CRC
        let data = [0x00, 0x00, 0x00, 0x01, 0x23, 0x04, 0xAA, 0x00];
        // Total needed: 6 + 4 + 2 = 12, but we only have 8
        assert_eq!(CanFrame::decode(&data), Err(BusError::FrameTooShort));
    }

    #[test]
    fn test_decode_corrupted_crc() {
        let frame = CanFrame::new_standard(0x100, &[0x01, 0x02]).unwrap();
        let mut encoded = frame.encode();
        // Corrupt the last byte (CRC)
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF;
        assert_eq!(CanFrame::decode(&encoded), Err(BusError::CrcMismatch));
    }

    #[test]
    fn test_decode_corrupted_payload() {
        let frame = CanFrame::new_standard(0x200, &[0xAA, 0xBB, 0xCC]).unwrap();
        let mut encoded = frame.encode();
        // Corrupt a payload byte
        encoded[7] ^= 0x01;
        assert_eq!(CanFrame::decode(&encoded), Err(BusError::CrcMismatch));
    }

    #[test]
    fn test_decode_invalid_standard_id() {
        // Manually construct a frame with ID > 0x7FF and IDE=0 (standard)
        let mut encoded = vec![0x00]; // flags: standard, no RTR
        encoded.extend_from_slice(&0x0000_0800u32.to_be_bytes()); // ID = 0x800
        encoded.push(0); // DLC = 0
        encoded.extend_from_slice(&0u16.to_be_bytes()); // CRC (doesn't matter)
        assert_eq!(CanFrame::decode(&encoded), Err(BusError::InvalidId));
    }

    // -----------------------------------------------------------------------
    // Frame type discrimination tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_frame_type_from_ide_bit() {
        let std_frame = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        let ext_frame = CanFrame::new_extended(0x100, &[0x01]).unwrap();

        let std_encoded = std_frame.encode();
        let ext_encoded = ext_frame.encode();

        // IDE bit is bit 0 of flags byte
        assert_eq!(std_encoded[0] & FLAG_IDE, 0);
        assert_eq!(ext_encoded[0] & FLAG_IDE, FLAG_IDE);

        let std_decoded = CanFrame::decode(&std_encoded).unwrap();
        let ext_decoded = CanFrame::decode(&ext_encoded).unwrap();

        assert_eq!(std_decoded.frame_type, FrameType::Standard);
        assert_eq!(ext_decoded.frame_type, FrameType::Extended);
    }

    // -----------------------------------------------------------------------
    // Encode size tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_size_empty_payload() {
        let frame = CanFrame::new_standard(0x100, &[]).unwrap();
        let encoded = frame.encode();
        // flags(1) + id(4) + dlc(1) + data(0) + crc(2) = 8
        assert_eq!(encoded.len(), 8);
    }

    #[test]
    fn test_encode_size_full_payload() {
        let frame = CanFrame::new_standard(0x100, &[0; 8]).unwrap();
        let encoded = frame.encode();
        // flags(1) + id(4) + dlc(1) + data(8) + crc(2) = 16
        assert_eq!(encoded.len(), 16);
    }

    // -----------------------------------------------------------------------
    // CRC computation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_crc_changes_with_id() {
        let f1 = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        let f2 = CanFrame::new_standard(0x200, &[0x01]).unwrap();
        assert_ne!(f1.crc, f2.crc);
    }

    #[test]
    fn test_crc_changes_with_data() {
        let f1 = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        let f2 = CanFrame::new_standard(0x100, &[0x02]).unwrap();
        assert_ne!(f1.crc, f2.crc);
    }

    #[test]
    fn test_crc_changes_with_frame_type() {
        let f1 = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        let f2 = CanFrame::new_extended(0x100, &[0x01]).unwrap();
        assert_ne!(f1.crc, f2.crc);
    }

    // -----------------------------------------------------------------------
    // All zero tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_zeros_standard() {
        let frame = CanFrame::new_standard(0x000, &[0x00; 8]).unwrap();
        let encoded = frame.encode();
        let decoded = CanFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn test_all_ones_extended() {
        let frame = CanFrame::new_extended(MAX_EXTENDED_ID, &[0xFF; 8]).unwrap();
        let encoded = frame.encode();
        let decoded = CanFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, frame);
    }

    // -----------------------------------------------------------------------
    // Extra bytes after frame should not cause errors
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_with_trailing_bytes() {
        let frame = CanFrame::new_standard(0x100, &[0x01, 0x02]).unwrap();
        let mut encoded = frame.encode();
        encoded.extend_from_slice(&[0xFF, 0xFF, 0xFF]); // trailing garbage
        let decoded = CanFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, frame);
    }
}
