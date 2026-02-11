// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SpaceWire-to-Zenoh transport adapter.
//!
//! Maps SpaceWire packets to Zenoh key expressions: `spw/{link}/{dest}`
//!
//! - `link`: SpaceWire link index (0, 1, 2...)
//! - `dest`: Destination address (logical address or first path byte)
//!
//! The payload is the SpaceWire packet cargo (without address/EOP header).

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::transport::{BusSample, ZenohTransport};
use crate::BusError;

/// Maximum number of queued receive packets.
const MAX_RX_QUEUE: usize = 64;

/// SpaceWire-to-Zenoh transport adapter.
pub struct SpwZenohAdapter {
    /// Link index.
    link: u8,
    /// Key expression prefix (e.g., "spw/0").
    prefix: String,
    /// Receive buffer for packets.
    rx_queue: VecDeque<(String, Vec<u8>)>,
}

impl SpwZenohAdapter {
    /// Create a new SpaceWire-to-Zenoh adapter.
    pub fn new(link: u8) -> Self {
        Self {
            link,
            prefix: format!("spw/{}", link),
            rx_queue: VecDeque::new(),
        }
    }

    /// Returns the link index.
    pub fn link(&self) -> u8 {
        self.link
    }

    /// Parse a destination address from a key expression suffix.
    fn parse_dest(suffix: &str) -> Result<u8, BusError> {
        suffix.parse::<u8>().map_err(|_| BusError::InvalidId)
    }

    /// Build a key expression for a SpaceWire packet.
    fn key_for_packet(&self, dest: u8) -> String {
        format!("{}/{}", self.prefix, dest)
    }

    /// Inject a received packet into the adapter's RX queue.
    pub fn inject_rx(&mut self, dest: u8, cargo: &[u8]) {
        if self.rx_queue.len() >= MAX_RX_QUEUE {
            return;
        }
        let key = self.key_for_packet(dest);
        self.rx_queue.push_back((key, Vec::from(cargo)));
    }

    /// Return the number of queued receive packets.
    pub fn rx_count(&self) -> usize {
        self.rx_queue.len()
    }
}

impl ZenohTransport for SpwZenohAdapter {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn transmit(&mut self, key_expr: &str, payload: &[u8]) -> Result<(), BusError> {
        let suffix = key_expr
            .strip_prefix(&self.prefix)
            .and_then(|s| s.strip_prefix('/'))
            .ok_or(BusError::InvalidId)?;

        let _dest = Self::parse_dest(suffix)?;

        if payload.is_empty() {
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

    #[test]
    fn test_adapter_prefix() {
        let adapter = SpwZenohAdapter::new(0);
        assert_eq!(adapter.prefix(), "spw/0");
    }

    #[test]
    fn test_adapter_prefix_link1() {
        let adapter = SpwZenohAdapter::new(1);
        assert_eq!(adapter.prefix(), "spw/1");
    }

    #[test]
    fn test_parse_dest() {
        assert_eq!(SpwZenohAdapter::parse_dest("42").unwrap(), 42);
        assert_eq!(SpwZenohAdapter::parse_dest("255").unwrap(), 255);
        assert!(SpwZenohAdapter::parse_dest("abc").is_err());
    }

    #[test]
    fn test_key_for_packet() {
        let adapter = SpwZenohAdapter::new(0);
        assert_eq!(adapter.key_for_packet(42), "spw/0/42");
    }

    #[test]
    fn test_inject_and_receive() {
        let mut adapter = SpwZenohAdapter::new(0);
        adapter.inject_rx(42, &[0xAA, 0xBB, 0xCC]);
        assert_eq!(adapter.rx_count(), 1);

        let mut buf = [0u8; 256];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(sample.key_expr, "spw/0/42");
        assert_eq!(sample.payload, &[0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_receive_empty() {
        let mut adapter = SpwZenohAdapter::new(0);
        let mut buf = [0u8; 256];
        assert!(adapter.receive(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_transmit_valid() {
        let mut adapter = SpwZenohAdapter::new(0);
        adapter.transmit("spw/0/42", &[0x01, 0x02]).unwrap();
    }

    #[test]
    fn test_transmit_empty_payload() {
        let mut adapter = SpwZenohAdapter::new(0);
        assert_eq!(
            adapter.transmit("spw/0/42", &[]),
            Err(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_transmit_invalid_key() {
        let mut adapter = SpwZenohAdapter::new(0);
        assert!(adapter.transmit("can/0/0x100", &[0x01]).is_err());
    }

    #[test]
    fn test_multiple_destinations() {
        let mut adapter = SpwZenohAdapter::new(0);
        adapter.inject_rx(32, &[0x01]);
        adapter.inject_rx(42, &[0x02]);
        adapter.inject_rx(100, &[0x03]);
        assert_eq!(adapter.rx_count(), 3);
    }
}
