// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MIL-1553-to-Zenoh transport adapter.
//!
//! Maps MIL-STD-1553 messages to Zenoh key expressions: `mil1553/{bus}/{rt}/{sa}`
//!
//! - `bus`: "A" or "B" (dual-redundant bus)
//! - `rt`: Remote Terminal address (0-30, or "bc" for broadcast)
//! - `sa`: Subaddress (0-31)
//!
//! The payload is an array of 16-bit data words encoded as big-endian bytes.

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::bc::Bus;
use crate::transport::{BusSample, ZenohTransport};
use crate::BusError;

/// Maximum number of queued receive messages.
const MAX_RX_QUEUE: usize = 64;

/// MIL-1553-to-Zenoh transport adapter.
pub struct Mil1553ZenohAdapter {
    /// Bus identifier (A or B).
    bus: Bus,
    /// Key expression prefix (e.g., "mil1553/A").
    prefix: String,
    /// Receive buffer for messages.
    rx_queue: VecDeque<(String, Vec<u8>)>,
}

impl Mil1553ZenohAdapter {
    /// Create a new MIL-1553-to-Zenoh adapter.
    pub fn new(bus: Bus) -> Self {
        let bus_str = match bus {
            Bus::A => "A",
            Bus::B => "B",
        };
        Self {
            bus,
            prefix: format!("mil1553/{}", bus_str),
            rx_queue: VecDeque::new(),
        }
    }

    /// Returns the bus identifier.
    pub fn bus(&self) -> Bus {
        self.bus
    }

    /// Parse RT address and subaddress from key expression suffix.
    ///
    /// Expected format: `{rt}/{sa}`
    fn parse_rt_sa(suffix: &str) -> Result<(u8, u8), BusError> {
        let parts: Vec<&str> = suffix.splitn(2, '/').collect();
        if parts.len() != 2 {
            return Err(BusError::InvalidId);
        }
        let rt: u8 = parts[0].parse().map_err(|_| BusError::InvalidId)?;
        let sa: u8 = parts[1].parse().map_err(|_| BusError::InvalidId)?;
        if rt > 31 || sa > 31 {
            return Err(BusError::InvalidId);
        }
        Ok((rt, sa))
    }

    /// Build a key expression for a MIL-1553 message.
    fn key_for_message(&self, rt: u8, sa: u8) -> String {
        format!("{}/{}/{}", self.prefix, rt, sa)
    }

    /// Inject a received message into the adapter's RX queue.
    ///
    /// `data_words` is a slice of 16-bit data words from the RT response.
    pub fn inject_rx(&mut self, rt: u8, sa: u8, data_words: &[u16]) {
        if self.rx_queue.len() >= MAX_RX_QUEUE {
            return;
        }
        let key = self.key_for_message(rt, sa);
        let mut payload = Vec::with_capacity(data_words.len() * 2);
        for &w in data_words {
            payload.extend_from_slice(&w.to_be_bytes());
        }
        self.rx_queue.push_back((key, payload));
    }

    /// Return the number of queued receive messages.
    pub fn rx_count(&self) -> usize {
        self.rx_queue.len()
    }
}

impl ZenohTransport for Mil1553ZenohAdapter {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn transmit(&mut self, key_expr: &str, payload: &[u8]) -> Result<(), BusError> {
        let suffix = key_expr
            .strip_prefix(&self.prefix)
            .and_then(|s| s.strip_prefix('/'))
            .ok_or(BusError::InvalidId)?;

        let (_rt, _sa) = Self::parse_rt_sa(suffix)?;

        // Payload must be an even number of bytes (16-bit data words).
        if !payload.len().is_multiple_of(2) {
            return Err(BusError::InvalidDlc);
        }
        // Max 32 data words.
        if payload.len() > 64 {
            return Err(BusError::PayloadTooLarge);
        }

        Ok(())
    }

    fn receive<'a>(&mut self, buf: &'a mut [u8]) -> Result<Option<BusSample<'a>>, BusError> {
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

    #[test]
    fn test_adapter_prefix_bus_a() {
        let adapter = Mil1553ZenohAdapter::new(Bus::A);
        assert_eq!(adapter.prefix(), "mil1553/A");
    }

    #[test]
    fn test_adapter_prefix_bus_b() {
        let adapter = Mil1553ZenohAdapter::new(Bus::B);
        assert_eq!(adapter.prefix(), "mil1553/B");
    }

    #[test]
    fn test_parse_rt_sa_valid() {
        let (rt, sa) = Mil1553ZenohAdapter::parse_rt_sa("5/3").unwrap();
        assert_eq!(rt, 5);
        assert_eq!(sa, 3);
    }

    #[test]
    fn test_parse_rt_sa_broadcast() {
        let (rt, sa) = Mil1553ZenohAdapter::parse_rt_sa("31/0").unwrap();
        assert_eq!(rt, 31);
        assert_eq!(sa, 0);
    }

    #[test]
    fn test_parse_rt_sa_invalid_format() {
        assert!(Mil1553ZenohAdapter::parse_rt_sa("5").is_err());
    }

    #[test]
    fn test_parse_rt_sa_out_of_range() {
        assert!(Mil1553ZenohAdapter::parse_rt_sa("32/0").is_err());
        assert!(Mil1553ZenohAdapter::parse_rt_sa("5/32").is_err());
    }

    #[test]
    fn test_key_for_message() {
        let adapter = Mil1553ZenohAdapter::new(Bus::A);
        assert_eq!(adapter.key_for_message(5, 3), "mil1553/A/5/3");
    }

    #[test]
    fn test_inject_and_receive() {
        let mut adapter = Mil1553ZenohAdapter::new(Bus::A);
        adapter.inject_rx(5, 3, &[0x1234, 0x5678]);
        assert_eq!(adapter.rx_count(), 1);

        let mut buf = [0u8; 256];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(sample.key_expr, "mil1553/A/5/3");
        assert_eq!(sample.payload.len(), 4); // 2 words * 2 bytes
        assert_eq!(sample.payload[0], 0x12);
        assert_eq!(sample.payload[1], 0x34);
        assert_eq!(sample.payload[2], 0x56);
        assert_eq!(sample.payload[3], 0x78);
    }

    #[test]
    fn test_receive_empty() {
        let mut adapter = Mil1553ZenohAdapter::new(Bus::A);
        let mut buf = [0u8; 256];
        assert!(adapter.receive(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_transmit_valid() {
        let mut adapter = Mil1553ZenohAdapter::new(Bus::A);
        let data = [0x12, 0x34, 0x56, 0x78]; // 2 data words
        adapter.transmit("mil1553/A/5/3", &data).unwrap();
    }

    #[test]
    fn test_transmit_odd_byte_count() {
        let mut adapter = Mil1553ZenohAdapter::new(Bus::A);
        assert_eq!(
            adapter.transmit("mil1553/A/5/3", &[0x01, 0x02, 0x03]),
            Err(BusError::InvalidDlc)
        );
    }

    #[test]
    fn test_transmit_too_large() {
        let mut adapter = Mil1553ZenohAdapter::new(Bus::A);
        let big = [0u8; 66]; // 33 words
        assert_eq!(
            adapter.transmit("mil1553/A/5/3", &big),
            Err(BusError::PayloadTooLarge)
        );
    }

    #[test]
    fn test_transmit_invalid_key() {
        let mut adapter = Mil1553ZenohAdapter::new(Bus::A);
        assert!(adapter.transmit("can/0/0x100", &[0x01, 0x02]).is_err());
    }
}
