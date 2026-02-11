// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AFDX-to-Zenoh transport adapter.
//!
//! Maps AFDX Virtual Links to Zenoh key expressions: `afdx/{vl_id}`
//!
//! Each Virtual Link maps to a single key expression. The payload is the
//! AFDX frame (Ethernet header + payload + trailer) in its encoded form.

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::transport::{BusSample, ZenohTransport};
use crate::BusError;

/// Maximum number of queued receive frames.
const MAX_RX_QUEUE: usize = 64;

/// AFDX-to-Zenoh transport adapter.
///
/// Maps AFDX Virtual Links to Zenoh key expressions: `afdx/{vl_id}`.
pub struct AfdxZenohAdapter {
    /// Key expression prefix (always "afdx").
    prefix: String,
    /// Receive buffer for encoded AFDX frames.
    rx_queue: VecDeque<(String, Vec<u8>)>,
}

impl Default for AfdxZenohAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl AfdxZenohAdapter {
    /// Create a new AFDX-to-Zenoh adapter.
    pub fn new() -> Self {
        Self {
            prefix: String::from("afdx"),
            rx_queue: VecDeque::new(),
        }
    }

    /// Parse a VL ID from a key expression suffix.
    fn parse_vl_id(suffix: &str) -> Result<u16, BusError> {
        suffix.parse::<u16>().map_err(|_| BusError::InvalidId)
    }

    /// Inject a received AFDX frame into the adapter's RX queue.
    pub fn inject_rx(&mut self, vl_id: u16, encoded_frame: &[u8]) {
        if self.rx_queue.len() >= MAX_RX_QUEUE {
            return;
        }
        let key = format!("afdx/{}", vl_id);
        self.rx_queue.push_back((key, Vec::from(encoded_frame)));
    }

    /// Return the number of queued receive frames.
    pub fn rx_count(&self) -> usize {
        self.rx_queue.len()
    }
}

impl ZenohTransport for AfdxZenohAdapter {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn transmit(&mut self, key_expr: &str, payload: &[u8]) -> Result<(), BusError> {
        let suffix = key_expr
            .strip_prefix("afdx/")
            .ok_or(BusError::InvalidId)?;
        let _vl_id = Self::parse_vl_id(suffix)?;

        if payload.is_empty() {
            return Err(BusError::FrameTooShort);
        }
        // In a real implementation, this would forward to the AFDX hardware.
        // For now, we validate the key expression and payload.
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
        let adapter = AfdxZenohAdapter::new();
        assert_eq!(adapter.prefix(), "afdx");
    }

    #[test]
    fn test_parse_vl_id() {
        assert_eq!(AfdxZenohAdapter::parse_vl_id("100").unwrap(), 100);
        assert_eq!(AfdxZenohAdapter::parse_vl_id("0").unwrap(), 0);
        assert!(AfdxZenohAdapter::parse_vl_id("abc").is_err());
    }

    #[test]
    fn test_inject_and_receive() {
        let mut adapter = AfdxZenohAdapter::new();
        let frame_data = [0x01, 0x02, 0x03, 0x04];
        adapter.inject_rx(100, &frame_data);
        assert_eq!(adapter.rx_count(), 1);

        let mut buf = [0u8; 256];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(sample.key_expr, "afdx/100");
        assert_eq!(sample.payload, &frame_data);
    }

    #[test]
    fn test_receive_empty() {
        let mut adapter = AfdxZenohAdapter::new();
        let mut buf = [0u8; 256];
        assert!(adapter.receive(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_transmit_valid() {
        let mut adapter = AfdxZenohAdapter::new();
        adapter.transmit("afdx/100", &[0x01, 0x02]).unwrap();
    }

    #[test]
    fn test_transmit_empty_payload() {
        let mut adapter = AfdxZenohAdapter::new();
        assert_eq!(
            adapter.transmit("afdx/100", &[]),
            Err(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_transmit_invalid_key() {
        let mut adapter = AfdxZenohAdapter::new();
        assert!(adapter.transmit("can/0/0x100", &[0x01]).is_err());
    }

    #[test]
    fn test_multiple_vls() {
        let mut adapter = AfdxZenohAdapter::new();
        adapter.inject_rx(10, &[0x01]);
        adapter.inject_rx(20, &[0x02]);
        adapter.inject_rx(30, &[0x03]);
        assert_eq!(adapter.rx_count(), 3);

        let mut buf = [0u8; 256];
        let s1 = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(s1.key_expr, "afdx/10");

        let s2 = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(s2.key_expr, "afdx/20");

        let s3 = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(s3.key_expr, "afdx/30");
    }
}
