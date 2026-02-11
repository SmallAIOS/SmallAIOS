// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CCSDS Telemetry (TM) Transfer Frame (CCSDS 132.0-B).
//!
//! A TM frame encapsulates Space Packets for downlink with a fixed-size
//! frame header containing spacecraft ID, virtual channel, frame counter,
//! and First Header Pointer.

use alloc::vec::Vec;
use crate::BusError;

/// TM frame primary header size in bytes.
pub const TM_HEADER_LEN: usize = 6;

/// First Header Pointer indicating no packet starts in this frame.
pub const FHP_NO_PACKET: u16 = 0x7FF;

/// TM Transfer Frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TmFrame {
    /// Transfer Frame Version (2 bits) — always 00.
    pub version: u8,
    /// Spacecraft ID (10 bits).
    pub scid: u16,
    /// Virtual Channel ID (3 bits).
    pub vcid: u8,
    /// Master Channel Frame Counter (8 bits).
    pub frame_counter: u8,
    /// First Header Pointer (11 bits): offset to first packet start, or 0x7FF for idle.
    pub first_header_pointer: u16,
    /// Frame data zone (fixed-size, filled with packet data or idle pattern).
    pub data: Vec<u8>,
}

impl TmFrame {
    /// Create a new TM frame.
    pub fn new(
        scid: u16,
        vcid: u8,
        frame_counter: u8,
        first_header_pointer: u16,
        data: &[u8],
    ) -> Result<Self, BusError> {
        if scid > 0x3FF {
            return Err(BusError::OutOfRange);
        }
        if vcid > 7 {
            return Err(BusError::OutOfRange);
        }
        if first_header_pointer > FHP_NO_PACKET {
            return Err(BusError::OutOfRange);
        }
        Ok(Self {
            version: 0,
            scid,
            vcid,
            frame_counter,
            first_header_pointer,
            data: Vec::from(data),
        })
    }

    /// Create an idle frame (no packets).
    pub fn idle(scid: u16, vcid: u8, frame_counter: u8, frame_size: usize) -> Result<Self, BusError> {
        if scid > 0x3FF || vcid > 7 {
            return Err(BusError::OutOfRange);
        }
        Ok(Self {
            version: 0,
            scid,
            vcid,
            frame_counter,
            first_header_pointer: FHP_NO_PACKET,
            data: alloc::vec![0xFE; frame_size], // Idle fill pattern
        })
    }

    /// Returns true if this is an idle frame.
    pub fn is_idle(&self) -> bool {
        self.first_header_pointer == FHP_NO_PACKET
    }

    /// Encode the TM frame into a byte buffer.
    ///
    /// Header layout:
    /// - Word 0: version(2) + scid(10) + vcid(3) + ocf_flag(1) = 16 bits
    /// - Word 1: frame_counter(8) + spare(5) + first_header_pointer(11) = 24 bits (3 bytes)
    ///
    /// Total header: 5 bytes, but we use 6 for alignment with spare byte.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        let total = TM_HEADER_LEN + self.data.len();
        if buf.len() < total {
            return Err(BusError::BufferTooSmall);
        }

        // Bytes 0-1: version(2) + scid(10) + vcid(3) + ocf_flag(1)
        let w0: u16 = ((self.version as u16 & 0x03) << 14)
            | ((self.scid & 0x3FF) << 4)
            | ((self.vcid as u16 & 0x07) << 1);
        buf[0] = (w0 >> 8) as u8;
        buf[1] = w0 as u8;

        // Byte 2: frame counter
        buf[2] = self.frame_counter;

        // Bytes 3-4: spare(5) + first_header_pointer(11)
        let fhp_word: u16 = self.first_header_pointer & 0x7FF;
        buf[3] = (fhp_word >> 8) as u8;
        buf[4] = fhp_word as u8;

        // Byte 5: spare
        buf[5] = 0;

        // Data zone
        buf[TM_HEADER_LEN..total].copy_from_slice(&self.data);

        Ok(total)
    }

    /// Decode a TM frame from a byte buffer.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < TM_HEADER_LEN + 1 {
            return Err(BusError::FrameTooShort);
        }

        let w0 = ((data[0] as u16) << 8) | data[1] as u16;
        let version = ((w0 >> 14) & 0x03) as u8;
        let scid = (w0 >> 4) & 0x3FF;
        let vcid = ((w0 >> 1) & 0x07) as u8;

        let frame_counter = data[2];

        let fhp_word = ((data[3] as u16) << 8) | data[4] as u16;
        let first_header_pointer = fhp_word & 0x7FF;

        let frame_data = Vec::from(&data[TM_HEADER_LEN..]);

        Ok(Self {
            version,
            scid,
            vcid,
            frame_counter,
            first_header_pointer,
            data: frame_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tm_frame_new() {
        let frame = TmFrame::new(0x100, 2, 42, 0, &[0xAA; 100]).unwrap();
        assert_eq!(frame.scid, 0x100);
        assert_eq!(frame.vcid, 2);
        assert_eq!(frame.frame_counter, 42);
        assert_eq!(frame.first_header_pointer, 0);
        assert!(!frame.is_idle());
    }

    #[test]
    fn test_tm_frame_invalid_scid() {
        assert_eq!(TmFrame::new(0x400, 0, 0, 0, &[0x00]).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_tm_frame_invalid_vcid() {
        assert_eq!(TmFrame::new(0, 8, 0, 0, &[0x00]).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_tm_frame_idle() {
        let frame = TmFrame::idle(0x100, 0, 0, 256).unwrap();
        assert!(frame.is_idle());
        assert_eq!(frame.first_header_pointer, FHP_NO_PACKET);
        assert_eq!(frame.data.len(), 256);
    }

    #[test]
    fn test_tm_frame_encode_decode_roundtrip() {
        let frame = TmFrame::new(0x1A3, 5, 99, 16, &[0x01, 0x02, 0x03]).unwrap();
        let mut buf = [0u8; 256];
        let len = frame.encode(&mut buf).unwrap();
        let decoded = TmFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.scid, 0x1A3);
        assert_eq!(decoded.vcid, 5);
        assert_eq!(decoded.frame_counter, 99);
        assert_eq!(decoded.first_header_pointer, 16);
        assert_eq!(decoded.data, &[0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_tm_frame_decode_too_short() {
        assert_eq!(TmFrame::decode(&[0u8; 6]).err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_tm_frame_encode_buffer_too_small() {
        let frame = TmFrame::new(0, 0, 0, 0, &[0xAA; 100]).unwrap();
        let mut buf = [0u8; 10];
        assert_eq!(frame.encode(&mut buf).err(), Some(BusError::BufferTooSmall));
    }

    #[test]
    fn test_tm_frame_fhp_no_packet() {
        let frame = TmFrame::new(0, 0, 0, FHP_NO_PACKET, &[0x00]).unwrap();
        assert!(frame.is_idle());
    }
}
