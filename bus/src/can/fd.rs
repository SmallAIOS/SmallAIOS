// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CAN FD (Flexible Data-rate) frame encode/decode.
//!
//! Clean-room implementation from ISO 11898-1:2015.
//!
//! CAN FD extends classic CAN with:
//! - Payloads up to 64 bytes (vs 8 for classic CAN)
//! - Bit Rate Switching (BRS) for faster data phase
//! - FDF (FD Format) bit distinguishing FD from classic frames
//! - ESI (Error State Indicator) bit
//! - DLC mapping for sizes > 8: 12, 16, 20, 24, 32, 48, 64 bytes
//! - CRC-17 for payloads <= 16 bytes, CRC-21 for payloads > 16 bytes
//!
//! # Serialized byte format
//!
//! ```text
//! Byte 0:       Flags (bit 0 = IDE, bit 1 = BRS, bit 2 = ESI)
//! Bytes 1-4:    Identifier (big-endian, 11-bit or 29-bit depending on IDE)
//! Byte 5:       DLC (0-15, maps to payload length via FD DLC table)
//! Bytes 6..6+N: Data payload (N = decoded payload length)
//! Bytes -4..-1: CRC (big-endian u32, CRC-17 or CRC-21)
//! ```

use alloc::vec::Vec;

use crate::BusError;
use super::crc::{crc17, crc21};
use super::frame::{MAX_EXTENDED_ID, MAX_STANDARD_ID, FrameType};

/// Maximum payload size for CAN FD frames.
pub const MAX_FD_PAYLOAD: usize = 64;

/// Flag byte: IDE bit (extended frame).
const FLAG_IDE: u8 = 0x01;

/// Flag byte: BRS bit (bit rate switching).
const FLAG_BRS: u8 = 0x02;

/// Flag byte: ESI bit (error state indicator).
const FLAG_ESI: u8 = 0x04;

/// Minimum serialized FD frame size: flags(1) + id(4) + dlc(1) + crc(4) = 10.
const MIN_FD_FRAME_BYTES: usize = 10;

/// CAN FD DLC (Data Length Code) with mapping to actual payload lengths.
///
/// DLC values 0-8 map directly to 0-8 bytes. Values 9-15 map to
/// 12, 16, 20, 24, 32, 48, 64 bytes respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FdDlc(u8);

impl FdDlc {
    /// DLC-to-payload-length mapping table for CAN FD.
    const DLC_TO_LEN: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64];

    /// Create an FdDlc from a raw DLC value (0-15).
    pub fn from_raw(dlc: u8) -> Result<Self, BusError> {
        if dlc > 15 {
            return Err(BusError::InvalidDlc);
        }
        Ok(FdDlc(dlc))
    }

    /// Find the smallest DLC that can hold `payload_len` bytes.
    ///
    /// Returns [`BusError::PayloadTooLarge`] if `payload_len` > 64.
    pub fn from_payload_len(payload_len: usize) -> Result<Self, BusError> {
        for (dlc, &len) in Self::DLC_TO_LEN.iter().enumerate() {
            if payload_len <= len {
                return Ok(FdDlc(dlc as u8));
            }
        }
        Err(BusError::PayloadTooLarge)
    }

    /// Returns the raw DLC value (0-15).
    pub fn raw(self) -> u8 {
        self.0
    }

    /// Returns the actual payload length for this DLC.
    pub fn payload_len(self) -> usize {
        Self::DLC_TO_LEN[self.0 as usize]
    }
}

/// A CAN FD frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanFdFrame {
    /// Frame type (standard 11-bit or extended 29-bit).
    pub frame_type: FrameType,
    /// CAN identifier (11-bit for standard, 29-bit for extended).
    pub id: u32,
    /// Bit Rate Switching flag.
    pub brs: bool,
    /// Error State Indicator flag.
    pub esi: bool,
    /// Data Length Code.
    pub dlc: FdDlc,
    /// Data payload (0-64 bytes, padded to DLC length with zeros).
    pub data: Vec<u8>,
    /// CRC value (CRC-17 for payloads <= 16 bytes, CRC-21 for > 16 bytes).
    pub crc: u32,
}

impl CanFdFrame {
    /// Create a new CAN FD frame with standard (11-bit) identifier.
    ///
    /// The payload is padded with zeros to the next valid CAN FD DLC length.
    /// Returns [`BusError::InvalidId`] if `id` exceeds 11 bits.
    /// Returns [`BusError::PayloadTooLarge`] if `data` exceeds 64 bytes.
    pub fn new_standard(id: u32, data: &[u8], brs: bool) -> Result<Self, BusError> {
        if id > MAX_STANDARD_ID {
            return Err(BusError::InvalidId);
        }
        Self::build(FrameType::Standard, id, data, brs, false)
    }

    /// Create a new CAN FD frame with extended (29-bit) identifier.
    ///
    /// The payload is padded with zeros to the next valid CAN FD DLC length.
    /// Returns [`BusError::InvalidId`] if `id` exceeds 29 bits.
    /// Returns [`BusError::PayloadTooLarge`] if `data` exceeds 64 bytes.
    pub fn new_extended(id: u32, data: &[u8], brs: bool) -> Result<Self, BusError> {
        if id > MAX_EXTENDED_ID {
            return Err(BusError::InvalidId);
        }
        Self::build(FrameType::Extended, id, data, brs, false)
    }

    fn build(
        frame_type: FrameType,
        id: u32,
        data: &[u8],
        brs: bool,
        esi: bool,
    ) -> Result<Self, BusError> {
        if data.len() > MAX_FD_PAYLOAD {
            return Err(BusError::PayloadTooLarge);
        }
        let dlc = FdDlc::from_payload_len(data.len())?;
        // Pad data to the DLC-mandated length
        let mut padded = Vec::from(data);
        padded.resize(dlc.payload_len(), 0);

        let mut frame = CanFdFrame {
            frame_type,
            id,
            brs,
            esi,
            dlc,
            data: padded,
            crc: 0,
        };
        frame.crc = frame.compute_crc();
        Ok(frame)
    }

    /// Compute the appropriate CRC for this frame.
    ///
    /// Per ISO 11898-1:2015:
    /// - CRC-17 for payloads <= 16 bytes
    /// - CRC-21 for payloads > 16 bytes
    fn compute_crc(&self) -> u32 {
        let mut buf = Vec::with_capacity(6 + self.data.len());
        // Flags byte
        let mut flags = 0u8;
        if self.frame_type == FrameType::Extended {
            flags |= FLAG_IDE;
        }
        if self.brs {
            flags |= FLAG_BRS;
        }
        if self.esi {
            flags |= FLAG_ESI;
        }
        buf.push(flags);
        // Identifier
        buf.extend_from_slice(&self.id.to_be_bytes());
        // DLC
        buf.push(self.dlc.raw());
        // Data
        buf.extend_from_slice(&self.data);

        if self.dlc.payload_len() <= 16 {
            crc17(&buf)
        } else {
            crc21(&buf)
        }
    }

    /// Verify the CRC of this frame.
    pub fn verify_crc(&self) -> bool {
        self.crc == self.compute_crc()
    }

    /// Returns the actual payload length (from DLC mapping).
    pub fn payload_len(&self) -> usize {
        self.dlc.payload_len()
    }

    /// Encode the frame into a byte vector.
    ///
    /// Format: flags(1) + id(4) + dlc(1) + data(0-64) + crc(4)
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(MIN_FD_FRAME_BYTES + self.data.len());
        // Flags byte
        let mut flags = 0u8;
        if self.frame_type == FrameType::Extended {
            flags |= FLAG_IDE;
        }
        if self.brs {
            flags |= FLAG_BRS;
        }
        if self.esi {
            flags |= FLAG_ESI;
        }
        buf.push(flags);
        // Identifier
        buf.extend_from_slice(&self.id.to_be_bytes());
        // DLC (raw)
        buf.push(self.dlc.raw());
        // Data payload
        buf.extend_from_slice(&self.data);
        // CRC (4 bytes, big-endian)
        buf.extend_from_slice(&self.crc.to_be_bytes());
        buf
    }

    /// Decode a CAN FD frame from a byte slice.
    ///
    /// Returns [`BusError::FrameTooShort`] if `data` is too short.
    /// Returns [`BusError::CrcMismatch`] if the CRC does not match.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < MIN_FD_FRAME_BYTES {
            return Err(BusError::FrameTooShort);
        }

        let flags = data[0];
        let ide = (flags & FLAG_IDE) != 0;
        let brs = (flags & FLAG_BRS) != 0;
        let esi = (flags & FLAG_ESI) != 0;

        let id = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
        let raw_dlc = data[5];

        let dlc = FdDlc::from_raw(raw_dlc)?;
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

        let payload_len = dlc.payload_len();
        let expected_len = 6 + payload_len + 4; // header + payload + crc(4 bytes)
        if data.len() < expected_len {
            return Err(BusError::FrameTooShort);
        }

        let payload = Vec::from(&data[6..6 + payload_len]);
        let crc = u32::from_be_bytes([
            data[6 + payload_len],
            data[6 + payload_len + 1],
            data[6 + payload_len + 2],
            data[6 + payload_len + 3],
        ]);

        let frame = CanFdFrame {
            frame_type,
            id,
            brs,
            esi,
            dlc,
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
    // FdDlc tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fd_dlc_direct_mapping() {
        for dlc in 0..=8u8 {
            let fd_dlc = FdDlc::from_raw(dlc).unwrap();
            assert_eq!(fd_dlc.payload_len(), dlc as usize);
        }
    }

    #[test]
    fn test_fd_dlc_extended_mapping() {
        let expected = [12, 16, 20, 24, 32, 48, 64];
        for (i, &len) in expected.iter().enumerate() {
            let dlc = FdDlc::from_raw(9 + i as u8).unwrap();
            assert_eq!(dlc.payload_len(), len);
        }
    }

    #[test]
    fn test_fd_dlc_invalid() {
        assert_eq!(FdDlc::from_raw(16), Err(BusError::InvalidDlc));
        assert_eq!(FdDlc::from_raw(255), Err(BusError::InvalidDlc));
    }

    #[test]
    fn test_fd_dlc_from_payload_len() {
        // 0 bytes -> DLC 0
        assert_eq!(FdDlc::from_payload_len(0).unwrap().raw(), 0);
        // 5 bytes -> DLC 5
        assert_eq!(FdDlc::from_payload_len(5).unwrap().raw(), 5);
        // 8 bytes -> DLC 8
        assert_eq!(FdDlc::from_payload_len(8).unwrap().raw(), 8);
        // 9 bytes -> DLC 9 (12 bytes)
        assert_eq!(FdDlc::from_payload_len(9).unwrap().raw(), 9);
        assert_eq!(FdDlc::from_payload_len(9).unwrap().payload_len(), 12);
        // 13 bytes -> DLC 10 (16 bytes)
        assert_eq!(FdDlc::from_payload_len(13).unwrap().raw(), 10);
        // 32 bytes -> DLC 13 (32 bytes)
        assert_eq!(FdDlc::from_payload_len(32).unwrap().raw(), 13);
        // 48 bytes -> DLC 14 (48 bytes)
        assert_eq!(FdDlc::from_payload_len(48).unwrap().raw(), 14);
        // 64 bytes -> DLC 15 (64 bytes)
        assert_eq!(FdDlc::from_payload_len(64).unwrap().raw(), 15);
    }

    #[test]
    fn test_fd_dlc_from_payload_len_too_large() {
        assert_eq!(FdDlc::from_payload_len(65), Err(BusError::PayloadTooLarge));
        assert_eq!(FdDlc::from_payload_len(128), Err(BusError::PayloadTooLarge));
    }

    // -----------------------------------------------------------------------
    // CAN FD standard frame tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_fd_standard_small_payload() {
        let frame = CanFdFrame::new_standard(0x100, &[0x01, 0x02, 0x03], true).unwrap();
        assert_eq!(frame.frame_type, FrameType::Standard);
        assert_eq!(frame.id, 0x100);
        assert!(frame.brs);
        assert!(!frame.esi);
        assert_eq!(frame.dlc.raw(), 3);
        assert_eq!(frame.data.len(), 3);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_fd_standard_12_byte_payload() {
        let data = [0xAA; 10]; // 10 bytes -> rounds up to DLC 9 (12 bytes)
        let frame = CanFdFrame::new_standard(0x200, &data, false).unwrap();
        assert_eq!(frame.dlc.raw(), 9);
        assert_eq!(frame.data.len(), 12);
        // First 10 bytes match input, last 2 are zero-padded
        assert_eq!(&frame.data[..10], &data);
        assert_eq!(&frame.data[10..], &[0x00, 0x00]);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_fd_standard_64_byte_payload() {
        let data = [0x55; 64];
        let frame = CanFdFrame::new_standard(0x7FF, &data, true).unwrap();
        assert_eq!(frame.dlc.raw(), 15);
        assert_eq!(frame.data.len(), 64);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_fd_standard_payload_too_large() {
        let data = [0x00; 65];
        assert_eq!(
            CanFdFrame::new_standard(0x100, &data, false),
            Err(BusError::PayloadTooLarge)
        );
    }

    #[test]
    fn test_new_fd_standard_id_too_large() {
        assert_eq!(
            CanFdFrame::new_standard(0x800, &[0x01], false),
            Err(BusError::InvalidId)
        );
    }

    // -----------------------------------------------------------------------
    // CAN FD extended frame tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_fd_extended() {
        let frame = CanFdFrame::new_extended(0x1234_5678, &[0xDE, 0xAD], true).unwrap();
        assert_eq!(frame.frame_type, FrameType::Extended);
        assert_eq!(frame.id, 0x1234_5678);
        assert!(frame.brs);
        assert!(frame.verify_crc());
    }

    #[test]
    fn test_new_fd_extended_id_too_large() {
        assert_eq!(
            CanFdFrame::new_extended(0x2000_0000, &[0x01], false),
            Err(BusError::InvalidId)
        );
    }

    // -----------------------------------------------------------------------
    // Encode/decode roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fd_encode_decode_small_payload() {
        let original = CanFdFrame::new_standard(0x100, &[0x01, 0x02], false).unwrap();
        let encoded = original.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_fd_encode_decode_12_bytes() {
        let data = [0xBB; 12];
        let original = CanFdFrame::new_standard(0x300, &data, true).unwrap();
        let encoded = original.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_fd_encode_decode_16_bytes() {
        let data = [0xCC; 16];
        let original = CanFdFrame::new_extended(0xABCDE, &data, true).unwrap();
        let encoded = original.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
        // 16 bytes should use CRC-17
    }

    #[test]
    fn test_fd_encode_decode_20_bytes() {
        let data = [0xDD; 20];
        let original = CanFdFrame::new_standard(0x400, &data, false).unwrap();
        let encoded = original.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
        // 20 bytes should use CRC-21
    }

    #[test]
    fn test_fd_encode_decode_32_bytes() {
        let data: Vec<u8> = (0..32).collect();
        let original = CanFdFrame::new_standard(0x500, &data, true).unwrap();
        let encoded = original.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_fd_encode_decode_48_bytes() {
        let data = [0xEE; 48];
        let original = CanFdFrame::new_extended(0x1FFF_FFFF, &data, false).unwrap();
        let encoded = original.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_fd_encode_decode_64_bytes() {
        let data = [0xFF; 64];
        let original = CanFdFrame::new_standard(0x7FF, &data, true).unwrap();
        let encoded = original.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_fd_encode_decode_empty() {
        let original = CanFdFrame::new_standard(0x000, &[], false).unwrap();
        let encoded = original.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_fd_encode_decode_extended_roundtrip() {
        let original = CanFdFrame::new_extended(0x1234_5678, &[0xAA; 24], true).unwrap();
        let encoded = original.encode();
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    // -----------------------------------------------------------------------
    // BRS and ESI flag tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fd_brs_flag_roundtrip() {
        let with_brs = CanFdFrame::new_standard(0x100, &[0x01], true).unwrap();
        let without_brs = CanFdFrame::new_standard(0x100, &[0x01], false).unwrap();

        let decoded_with = CanFdFrame::decode(&with_brs.encode()).unwrap();
        let decoded_without = CanFdFrame::decode(&without_brs.encode()).unwrap();

        assert!(decoded_with.brs);
        assert!(!decoded_without.brs);
    }

    #[test]
    fn test_fd_brs_affects_crc() {
        let with_brs = CanFdFrame::new_standard(0x100, &[0x01], true).unwrap();
        let without_brs = CanFdFrame::new_standard(0x100, &[0x01], false).unwrap();
        assert_ne!(with_brs.crc, without_brs.crc);
    }

    // -----------------------------------------------------------------------
    // CRC selection tests (CRC-17 vs CRC-21)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fd_crc17_for_small_payload() {
        // Payloads <= 16 bytes use CRC-17
        let frame = CanFdFrame::new_standard(0x100, &[0x01; 16], false).unwrap();
        assert!(frame.verify_crc());
        assert_eq!(frame.dlc.payload_len(), 16);
    }

    #[test]
    fn test_fd_crc21_for_large_payload() {
        // Payloads > 16 bytes use CRC-21
        let frame = CanFdFrame::new_standard(0x100, &[0x01; 20], false).unwrap();
        assert!(frame.verify_crc());
        assert_eq!(frame.dlc.payload_len(), 20);
    }

    // -----------------------------------------------------------------------
    // Decode error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fd_decode_too_short() {
        let data = [0u8; 9]; // less than MIN_FD_FRAME_BYTES
        assert_eq!(CanFdFrame::decode(&data), Err(BusError::FrameTooShort));
    }

    #[test]
    fn test_fd_decode_invalid_dlc() {
        // DLC > 15 is invalid for FD but FdDlc::from_raw handles 0-15
        // Since DLC is u8, we can manually set byte 5 to 16
        let mut data = vec![0u8; 14];
        data[5] = 16; // invalid DLC
        assert_eq!(CanFdFrame::decode(&data), Err(BusError::InvalidDlc));
    }

    #[test]
    fn test_fd_decode_corrupted_crc() {
        let frame = CanFdFrame::new_standard(0x100, &[0x01, 0x02, 0x03], false).unwrap();
        let mut encoded = frame.encode();
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF;
        assert_eq!(CanFdFrame::decode(&encoded), Err(BusError::CrcMismatch));
    }

    #[test]
    fn test_fd_decode_corrupted_payload() {
        let frame = CanFdFrame::new_standard(0x200, &[0xAA; 32], true).unwrap();
        let mut encoded = frame.encode();
        encoded[10] ^= 0x01; // flip a bit in the payload
        assert_eq!(CanFdFrame::decode(&encoded), Err(BusError::CrcMismatch));
    }

    #[test]
    fn test_fd_decode_truncated_payload() {
        let frame = CanFdFrame::new_standard(0x100, &[0x01; 32], false).unwrap();
        let encoded = frame.encode();
        // Truncate the encoded data
        let truncated = &encoded[..15];
        assert_eq!(CanFdFrame::decode(truncated), Err(BusError::FrameTooShort));
    }

    #[test]
    fn test_fd_decode_invalid_standard_id() {
        let mut encoded = vec![0x00u8]; // flags: standard
        encoded.extend_from_slice(&0x0000_0800u32.to_be_bytes()); // ID = 0x800 (too large)
        encoded.push(0); // DLC = 0
        encoded.extend_from_slice(&0u32.to_be_bytes()); // CRC placeholder
        assert_eq!(CanFdFrame::decode(&encoded), Err(BusError::InvalidId));
    }

    // -----------------------------------------------------------------------
    // Payload padding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fd_payload_padding() {
        // 10 bytes input -> DLC 9 -> 12 bytes padded
        let data = [0xAA; 10];
        let frame = CanFdFrame::new_standard(0x100, &data, false).unwrap();
        assert_eq!(frame.data.len(), 12);
        assert_eq!(&frame.data[..10], &[0xAA; 10]);
        assert_eq!(&frame.data[10..], &[0x00, 0x00]);
    }

    #[test]
    fn test_fd_exact_dlc_no_padding() {
        // Exact 12 bytes -> DLC 9 -> no padding needed
        let data = [0xBB; 12];
        let frame = CanFdFrame::new_standard(0x100, &data, false).unwrap();
        assert_eq!(frame.data.len(), 12);
        assert_eq!(&frame.data, &data);
    }

    // -----------------------------------------------------------------------
    // Encode size tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fd_encode_size_empty() {
        let frame = CanFdFrame::new_standard(0x100, &[], false).unwrap();
        let encoded = frame.encode();
        // flags(1) + id(4) + dlc(1) + data(0) + crc(4) = 10
        assert_eq!(encoded.len(), 10);
    }

    #[test]
    fn test_fd_encode_size_64_bytes() {
        let frame = CanFdFrame::new_standard(0x100, &[0; 64], false).unwrap();
        let encoded = frame.encode();
        // flags(1) + id(4) + dlc(1) + data(64) + crc(4) = 74
        assert_eq!(encoded.len(), 74);
    }

    // -----------------------------------------------------------------------
    // Extra bytes after frame
    // -----------------------------------------------------------------------

    #[test]
    fn test_fd_decode_with_trailing_bytes() {
        let frame = CanFdFrame::new_standard(0x100, &[0x01, 0x02], false).unwrap();
        let mut encoded = frame.encode();
        encoded.extend_from_slice(&[0xFF; 10]); // trailing garbage
        let decoded = CanFdFrame::decode(&encoded).unwrap();
        assert_eq!(decoded, frame);
    }
}
