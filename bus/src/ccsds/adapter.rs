// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CCSDS-to-Zenoh transport adapter.
//!
//! Maps CCSDS Space Packets to Zenoh key expressions: `ccsds/{apid}`
//!
//! The APID (Application Process Identifier) determines the routing key.
//! The payload is the full encoded CCSDS Space Packet (header + data).

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::transport::{BusSample, ZenohTransport};
use crate::BusError;
use super::packet::{SpacePacket, MAX_APID};

/// Maximum number of queued receive packets.
const MAX_RX_QUEUE: usize = 64;

/// CCSDS-to-Zenoh transport adapter.
pub struct CcsdsZenohAdapter {
    /// Key expression prefix (always "ccsds").
    prefix: String,
    /// Receive buffer for encoded packets.
    rx_queue: VecDeque<(String, Vec<u8>)>,
}

impl Default for CcsdsZenohAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl CcsdsZenohAdapter {
    /// Create a new CCSDS-to-Zenoh adapter.
    pub fn new() -> Self {
        Self {
            prefix: String::from("ccsds"),
            rx_queue: VecDeque::new(),
        }
    }

    /// Parse an APID from a key expression suffix.
    fn parse_apid(suffix: &str) -> Result<u16, BusError> {
        let apid = suffix.parse::<u16>().map_err(|_| BusError::InvalidApid)?;
        if apid > MAX_APID {
            return Err(BusError::InvalidApid);
        }
        Ok(apid)
    }

    /// Build a key expression for a CCSDS packet.
    fn key_for_apid(apid: u16) -> String {
        format!("ccsds/{}", apid)
    }

    /// Inject a received CCSDS packet into the adapter's RX queue.
    pub fn inject_rx_packet(&mut self, packet: &SpacePacket) {
        if self.rx_queue.len() >= MAX_RX_QUEUE {
            return;
        }
        let key = Self::key_for_apid(packet.apid);
        let mut buf = alloc::vec![0u8; packet.total_length()];
        if packet.encode(&mut buf).is_ok() {
            self.rx_queue.push_back((key, buf));
        }
    }

    /// Inject raw encoded bytes into the adapter's RX queue.
    pub fn inject_rx_raw(&mut self, apid: u16, encoded: &[u8]) {
        if self.rx_queue.len() >= MAX_RX_QUEUE {
            return;
        }
        let key = Self::key_for_apid(apid);
        self.rx_queue.push_back((key, Vec::from(encoded)));
    }

    /// Return the number of queued receive packets.
    pub fn rx_count(&self) -> usize {
        self.rx_queue.len()
    }
}

impl ZenohTransport for CcsdsZenohAdapter {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn transmit(&mut self, key_expr: &str, payload: &[u8]) -> Result<(), BusError> {
        let suffix = key_expr
            .strip_prefix("ccsds/")
            .ok_or(BusError::InvalidApid)?;
        let _apid = Self::parse_apid(suffix)?;

        if payload.len() < 7 {
            // Minimum: 6 byte header + 1 byte data
            return Err(BusError::FrameTooShort);
        }

        Ok(())
    }

    fn receive<'a>(
        &mut self,
        buf: &'a mut [u8],
    ) -> Result<Option<BusSample<'a>>, BusError> {
        if let Some((key, data)) = self.rx_queue.pop_front() {
            let key_bytes = key.as_bytes();
            let total = key_bytes.len() + 1 + data.len();
            if buf.len() < total {
                return Err(BusError::BufferTooSmall);
            }
            buf[..key_bytes.len()].copy_from_slice(key_bytes);
            buf[key_bytes.len()] = 0;
            let payload_start = key_bytes.len() + 1;
            buf[payload_start..payload_start + data.len()].copy_from_slice(&data);

            Ok(Some(BusSample {
                key_expr: core::str::from_utf8(&buf[..key_bytes.len()])
                    .map_err(|_| BusError::InvalidHeader)?,
                payload: &buf[payload_start..payload_start + data.len()],
                timestamp_us: 0,
            }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccsds::packet::{PacketType, SeqFlags, SpacePacket};

    #[test]
    fn test_adapter_prefix() {
        let adapter = CcsdsZenohAdapter::new();
        assert_eq!(adapter.prefix(), "ccsds");
    }

    #[test]
    fn test_parse_apid() {
        assert_eq!(CcsdsZenohAdapter::parse_apid("100").unwrap(), 100);
        assert_eq!(CcsdsZenohAdapter::parse_apid("2047").unwrap(), 2047);
        assert!(CcsdsZenohAdapter::parse_apid("2048").is_err());
        assert!(CcsdsZenohAdapter::parse_apid("abc").is_err());
    }

    #[test]
    fn test_key_for_apid() {
        assert_eq!(CcsdsZenohAdapter::key_for_apid(100), "ccsds/100");
        assert_eq!(CcsdsZenohAdapter::key_for_apid(0), "ccsds/0");
    }

    #[test]
    fn test_inject_packet_and_receive() {
        let mut adapter = CcsdsZenohAdapter::new();
        let packet = SpacePacket::new(
            PacketType::Telemetry,
            false,
            100,
            SeqFlags::Unsegmented,
            0,
            &[0x01, 0x02, 0x03],
        )
        .unwrap();
        adapter.inject_rx_packet(&packet);
        assert_eq!(adapter.rx_count(), 1);

        let mut buf = [0u8; 256];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(sample.key_expr, "ccsds/100");
        // Payload should be the full encoded CCSDS packet.
        assert_eq!(sample.payload.len(), 6 + 3); // header + data
    }

    #[test]
    fn test_inject_raw_and_receive() {
        let mut adapter = CcsdsZenohAdapter::new();
        let raw = [0x00, 0x64, 0xC0, 0x00, 0x00, 0x02, 0x01, 0x02, 0x03];
        adapter.inject_rx_raw(100, &raw);
        assert_eq!(adapter.rx_count(), 1);

        let mut buf = [0u8; 256];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(sample.key_expr, "ccsds/100");
    }

    #[test]
    fn test_receive_empty() {
        let mut adapter = CcsdsZenohAdapter::new();
        let mut buf = [0u8; 256];
        assert!(adapter.receive(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_transmit_valid() {
        let mut adapter = CcsdsZenohAdapter::new();
        let mut packet_buf = [0u8; 32];
        let packet = SpacePacket::new(
            PacketType::Telecommand,
            false,
            50,
            SeqFlags::Unsegmented,
            0,
            &[0xFF],
        )
        .unwrap();
        let len = packet.encode(&mut packet_buf).unwrap();
        adapter.transmit("ccsds/50", &packet_buf[..len]).unwrap();
    }

    #[test]
    fn test_transmit_too_short() {
        let mut adapter = CcsdsZenohAdapter::new();
        assert_eq!(
            adapter.transmit("ccsds/50", &[0x01, 0x02]),
            Err(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_transmit_invalid_key() {
        let mut adapter = CcsdsZenohAdapter::new();
        assert!(adapter.transmit("can/0/0x100", &[0; 7]).is_err());
    }

    #[test]
    fn test_transmit_invalid_apid() {
        let mut adapter = CcsdsZenohAdapter::new();
        assert!(adapter.transmit("ccsds/9999", &[0; 7]).is_err());
    }

    #[test]
    fn test_multiple_apids() {
        let mut adapter = CcsdsZenohAdapter::new();
        for apid in [10, 20, 100, 500] {
            let packet = SpacePacket::new(
                PacketType::Telemetry,
                false,
                apid,
                SeqFlags::Unsegmented,
                0,
                &[0x01],
            )
            .unwrap();
            adapter.inject_rx_packet(&packet);
        }
        assert_eq!(adapter.rx_count(), 4);
    }
}
