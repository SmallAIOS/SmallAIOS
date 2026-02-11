// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AFDX frame encode/decode.
//!
//! An AFDX frame is an Ethernet frame with an AFDX-specific trailer containing
//! a per-VL sequence number. The Ethernet header uses multicast MAC addressing
//! derived from the VL ID.

use crate::BusError;

/// AFDX trailer: 1-byte sequence number appended to the Ethernet payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AfdxTrailer {
    /// Per-VL sequence number (0-255, wrapping).
    pub sequence_number: u8,
}

/// An AFDX frame with Ethernet header, payload, and trailer.
#[derive(Debug, Clone)]
pub struct AfdxFrame {
    /// Destination MAC (AFDX multicast, derived from VL ID).
    pub dest_mac: [u8; 6],
    /// Source MAC.
    pub src_mac: [u8; 6],
    /// EtherType (typically 0x0800 for IP).
    pub ether_type: u16,
    /// Application payload (not including trailer).
    pub payload: alloc::vec::Vec<u8>,
    /// AFDX trailer with sequence number.
    pub trailer: AfdxTrailer,
}

/// Minimum encoded frame: 6 + 6 + 2 + 0 + 1 = 15 bytes (header + trailer, no payload).
const MIN_FRAME_LEN: usize = 15;

impl AfdxFrame {
    /// Create a new AFDX frame.
    ///
    /// Returns `PayloadTooLarge` if payload exceeds `lmax`.
    pub fn new(
        dest_mac: [u8; 6],
        src_mac: [u8; 6],
        ether_type: u16,
        payload: &[u8],
        sequence_number: u8,
        lmax: u16,
    ) -> Result<Self, BusError> {
        // lmax covers the full Ethernet payload including trailer
        if payload.len() + 1 > lmax as usize {
            return Err(BusError::PayloadTooLarge);
        }
        Ok(Self {
            dest_mac,
            src_mac,
            ether_type,
            payload: alloc::vec::Vec::from(payload),
            trailer: AfdxTrailer { sequence_number },
        })
    }

    /// Encode the frame into a byte buffer.
    ///
    /// Layout: dest_mac(6) + src_mac(6) + ether_type(2) + payload(N) + seq_num(1)
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        let total = 6 + 6 + 2 + self.payload.len() + 1;
        if buf.len() < total {
            return Err(BusError::BufferTooSmall);
        }
        buf[0..6].copy_from_slice(&self.dest_mac);
        buf[6..12].copy_from_slice(&self.src_mac);
        buf[12] = (self.ether_type >> 8) as u8;
        buf[13] = self.ether_type as u8;
        buf[14..14 + self.payload.len()].copy_from_slice(&self.payload);
        buf[14 + self.payload.len()] = self.trailer.sequence_number;
        Ok(total)
    }

    /// Decode an AFDX frame from a byte buffer.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < MIN_FRAME_LEN {
            return Err(BusError::FrameTooShort);
        }
        let mut dest_mac = [0u8; 6];
        let mut src_mac = [0u8; 6];
        dest_mac.copy_from_slice(&data[0..6]);
        src_mac.copy_from_slice(&data[6..12]);
        let ether_type = ((data[12] as u16) << 8) | data[13] as u16;
        let payload_end = data.len() - 1;
        let payload = alloc::vec::Vec::from(&data[14..payload_end]);
        let sequence_number = data[payload_end];
        Ok(Self {
            dest_mac,
            src_mac,
            ether_type,
            payload,
            trailer: AfdxTrailer { sequence_number },
        })
    }

    /// Returns the VL ID extracted from the AFDX multicast MAC.
    pub fn vl_id(&self) -> u16 {
        ((self.dest_mac[4] as u16) << 8) | self.dest_mac[5] as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mac_dest() -> [u8; 6] {
        [0x03, 0x00, 0x00, 0x00, 0x03, 0xE9]
    }

    fn test_mac_src() -> [u8; 6] {
        [0x02, 0x00, 0x00, 0x00, 0x00, 0x01]
    }

    #[test]
    fn test_frame_new_valid() {
        let payload = [0xAA; 100];
        let frame = AfdxFrame::new(test_mac_dest(), test_mac_src(), 0x0800, &payload, 42, 512).unwrap();
        assert_eq!(frame.payload.len(), 100);
        assert_eq!(frame.trailer.sequence_number, 42);
        assert_eq!(frame.vl_id(), 0x03E9);
    }

    #[test]
    fn test_frame_payload_too_large() {
        let payload = [0xAA; 512];
        let result = AfdxFrame::new(test_mac_dest(), test_mac_src(), 0x0800, &payload, 0, 512);
        assert_eq!(result.err(), Some(BusError::PayloadTooLarge));
    }

    #[test]
    fn test_frame_encode_decode_roundtrip() {
        let payload = [0x01, 0x02, 0x03, 0x04];
        let frame = AfdxFrame::new(test_mac_dest(), test_mac_src(), 0x0800, &payload, 99, 512).unwrap();
        let mut buf = [0u8; 256];
        let len = frame.encode(&mut buf).unwrap();
        assert_eq!(len, 6 + 6 + 2 + 4 + 1);

        let decoded = AfdxFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.dest_mac, test_mac_dest());
        assert_eq!(decoded.src_mac, test_mac_src());
        assert_eq!(decoded.ether_type, 0x0800);
        assert_eq!(decoded.payload.as_slice(), &payload);
        assert_eq!(decoded.trailer.sequence_number, 99);
    }

    #[test]
    fn test_frame_decode_too_short() {
        let data = [0u8; 14]; // less than MIN_FRAME_LEN
        assert_eq!(AfdxFrame::decode(&data).err(), Some(BusError::FrameTooShort));
    }

    #[test]
    fn test_frame_encode_buffer_too_small() {
        let payload = [0xAA; 100];
        let frame = AfdxFrame::new(test_mac_dest(), test_mac_src(), 0x0800, &payload, 0, 512).unwrap();
        let mut buf = [0u8; 10];
        assert_eq!(frame.encode(&mut buf).err(), Some(BusError::BufferTooSmall));
    }

    #[test]
    fn test_frame_empty_payload() {
        let frame = AfdxFrame::new(test_mac_dest(), test_mac_src(), 0x0800, &[], 0, 512).unwrap();
        let mut buf = [0u8; 64];
        let len = frame.encode(&mut buf).unwrap();
        assert_eq!(len, 15); // MIN_FRAME_LEN
        let decoded = AfdxFrame::decode(&buf[..len]).unwrap();
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn test_vl_id_extraction() {
        let frame = AfdxFrame::new(
            [0x03, 0x00, 0x00, 0x00, 0x07, 0xD2],
            test_mac_src(),
            0x0800,
            &[],
            0,
            512,
        )
        .unwrap();
        assert_eq!(frame.vl_id(), 2002);
    }

    #[test]
    fn test_sequence_number_wraps() {
        let frame = AfdxFrame::new(test_mac_dest(), test_mac_src(), 0x0800, &[], 255, 512).unwrap();
        assert_eq!(frame.trailer.sequence_number, 255);
        // Next would be 0, handled by the shaper
    }
}
