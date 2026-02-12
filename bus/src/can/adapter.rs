// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CAN-to-Zenoh transport adapter.
//!
//! Maps CAN frames to Zenoh key expressions: `can/{bus_id}/{frame_id}`
//!
//! The frame ID is encoded as a hex string (e.g., `0x1A3` for standard frames,
//! `0x12345678` for extended frames). The adapter serializes/deserializes CAN
//! frames to their compact byte encoding for transmission over Zenoh.

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::controller::CanController;
use super::frame::CanFrame;
use crate::transport::{BusSample, ZenohTransport};
use crate::BusError;

/// Maximum number of queued receive frames.
const MAX_RX_QUEUE: usize = 64;

/// CAN-to-Zenoh transport adapter.
///
/// Wraps a [`CanController`] and translates between CAN frames and Zenoh
/// key expressions of the form `can/{bus_id}/{frame_id}`.
pub struct CanZenohAdapter<C: CanController> {
    /// Underlying CAN controller.
    controller: C,
    /// Key expression prefix (e.g., "can/0").
    prefix: String,
    /// Receive buffer for decoded frames.
    rx_queue: VecDeque<(String, Vec<u8>)>,
}

impl<C: CanController> CanZenohAdapter<C> {
    /// Create a new CAN-to-Zenoh adapter.
    pub fn new(controller: C, bus_id: u8) -> Self {
        Self {
            controller,
            prefix: format!("can/{}", bus_id),
            rx_queue: VecDeque::new(),
        }
    }

    /// Returns a reference to the underlying controller.
    pub fn controller(&self) -> &C {
        &self.controller
    }

    /// Returns a mutable reference to the underlying controller.
    pub fn controller_mut(&mut self) -> &mut C {
        &mut self.controller
    }

    /// Parse a CAN frame ID from a key expression suffix.
    ///
    /// Accepts hex format: `0x1A3` or `0x12345678`
    fn parse_frame_id(suffix: &str) -> Result<u32, BusError> {
        let hex = suffix
            .strip_prefix("0x")
            .or_else(|| suffix.strip_prefix("0X"));
        match hex {
            Some(h) => u32::from_str_radix(h, 16).map_err(|_| BusError::InvalidId),
            None => suffix.parse::<u32>().map_err(|_| BusError::InvalidId),
        }
    }

    /// Build a key expression for a CAN frame.
    fn key_for_frame(&self, frame: &CanFrame) -> String {
        format!("{}/0x{:X}", self.prefix, frame.id)
    }

    /// Poll the controller for received frames and enqueue them.
    pub fn poll_rx(&mut self) -> Result<usize, BusError> {
        let mut count = 0;
        while self.rx_queue.len() < MAX_RX_QUEUE {
            match self.controller.receive() {
                Ok(Some(frame)) => {
                    let key = self.key_for_frame(&frame);
                    let encoded = frame.encode();
                    self.rx_queue.push_back((key, encoded));
                    count += 1;
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(count)
    }
}

impl<C: CanController> ZenohTransport for CanZenohAdapter<C> {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn transmit(&mut self, key_expr: &str, payload: &[u8]) -> Result<(), BusError> {
        // Extract frame ID from key expression.
        // Expected format: can/{bus_id}/{frame_id}
        let suffix = key_expr
            .strip_prefix(&self.prefix)
            .and_then(|s| s.strip_prefix('/'))
            .ok_or(BusError::InvalidId)?;

        let frame_id = Self::parse_frame_id(suffix)?;

        // If payload is an encoded CAN frame whose ID matches, decode and transmit it.
        // Otherwise, wrap the raw payload as a standard frame.
        let frame = if let Ok(decoded) = CanFrame::decode(payload) {
            if decoded.id == frame_id {
                decoded
            } else {
                // Decoded ID doesn't match key — treat as raw payload.
                if payload.len() > 8 {
                    return Err(BusError::PayloadTooLarge);
                }
                if frame_id > 0x7FF {
                    CanFrame::new_extended(frame_id, payload)?
                } else {
                    CanFrame::new_standard(frame_id, payload)?
                }
            }
        } else {
            // Construct a new frame from raw payload.
            if payload.len() > 8 {
                return Err(BusError::PayloadTooLarge);
            }
            if frame_id > 0x7FF {
                CanFrame::new_extended(frame_id, payload)?
            } else {
                CanFrame::new_standard(frame_id, payload)?
            }
        };

        self.controller.transmit(&frame)
    }

    fn receive<'a>(&mut self, buf: &'a mut [u8]) -> Result<Option<BusSample<'a>>, BusError> {
        // Try polling the controller first.
        let _ = self.poll_rx();

        if let Some((key, data)) = self.rx_queue.pop_front() {
            if buf.len() < key.len() + 1 + data.len() {
                return Err(BusError::BufferTooSmall);
            }
            // Pack key + null + data into buffer.
            let key_bytes = key.as_bytes();
            buf[..key_bytes.len()].copy_from_slice(key_bytes);
            buf[key_bytes.len()] = 0; // null separator
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
    use crate::can::controller::{CanMode, MockCanController};
    use crate::can::frame::{CanFrame, FrameType};
    use alloc::vec;

    /// Create a mock controller in Normal mode (ready for TX/RX).
    fn ready_ctrl() -> MockCanController {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Normal).unwrap();
        ctrl
    }

    #[test]
    fn test_adapter_prefix() {
        let adapter = CanZenohAdapter::new(ready_ctrl(), 0);
        assert_eq!(adapter.prefix(), "can/0");
    }

    #[test]
    fn test_adapter_prefix_bus1() {
        let adapter = CanZenohAdapter::new(ready_ctrl(), 1);
        assert_eq!(adapter.prefix(), "can/1");
    }

    #[test]
    fn test_parse_frame_id_hex() {
        assert_eq!(
            CanZenohAdapter::<MockCanController>::parse_frame_id("0x1A3").unwrap(),
            0x1A3
        );
        assert_eq!(
            CanZenohAdapter::<MockCanController>::parse_frame_id("0x7FF").unwrap(),
            0x7FF
        );
        assert_eq!(
            CanZenohAdapter::<MockCanController>::parse_frame_id("0x12345678").unwrap(),
            0x12345678
        );
    }

    #[test]
    fn test_parse_frame_id_decimal() {
        assert_eq!(
            CanZenohAdapter::<MockCanController>::parse_frame_id("256").unwrap(),
            256
        );
    }

    #[test]
    fn test_parse_frame_id_invalid() {
        assert!(CanZenohAdapter::<MockCanController>::parse_frame_id("xyz").is_err());
    }

    #[test]
    fn test_key_for_frame_standard() {
        let adapter = CanZenohAdapter::new(ready_ctrl(), 0);
        let frame = CanFrame::new_standard(0x1A3, &[0x01]).unwrap();
        assert_eq!(adapter.key_for_frame(&frame), "can/0/0x1A3");
    }

    #[test]
    fn test_key_for_frame_extended() {
        let adapter = CanZenohAdapter::new(ready_ctrl(), 0);
        let frame = CanFrame::new_extended(0x12345678, &[0x01]).unwrap();
        assert_eq!(adapter.key_for_frame(&frame), "can/0/0x12345678");
    }

    #[test]
    fn test_transmit_raw_payload() {
        let mut adapter = CanZenohAdapter::new(ready_ctrl(), 0);
        adapter
            .transmit("can/0/0x100", &[0x01, 0x02, 0x03])
            .unwrap();
        let sent = adapter.controller_mut().drain_tx();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].id, 0x100);
        assert_eq!(sent[0].data, vec![0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_transmit_encoded_frame() {
        let mut adapter = CanZenohAdapter::new(ready_ctrl(), 0);
        let frame = CanFrame::new_standard(0x200, &[0xDE, 0xAD]).unwrap();
        let encoded = frame.encode();
        adapter.transmit("can/0/0x200", &encoded).unwrap();
        let sent = adapter.controller_mut().drain_tx();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].id, 0x200);
    }

    #[test]
    fn test_transmit_extended_frame() {
        let mut adapter = CanZenohAdapter::new(ready_ctrl(), 0);
        adapter.transmit("can/0/0x1ABCDEF0", &[0x01]).unwrap();
        let sent = adapter.controller_mut().drain_tx();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].id, 0x1ABCDEF0);
        assert_eq!(sent[0].frame_type, FrameType::Extended);
    }

    #[test]
    fn test_transmit_payload_too_large() {
        let mut adapter = CanZenohAdapter::new(ready_ctrl(), 0);
        let big = [0u8; 9];
        assert_eq!(
            adapter.transmit("can/0/0x100", &big),
            Err(BusError::PayloadTooLarge)
        );
    }

    #[test]
    fn test_transmit_invalid_key() {
        let mut adapter = CanZenohAdapter::new(ready_ctrl(), 0);
        assert!(adapter.transmit("arinc429/0/0o200", &[0x01]).is_err());
    }

    #[test]
    fn test_receive_from_controller() {
        let mut ctrl = ready_ctrl();
        let frame = CanFrame::new_standard(0x150, &[0xAA, 0xBB]).unwrap();
        ctrl.inject_rx(frame);

        let mut adapter = CanZenohAdapter::new(ctrl, 0);
        let mut buf = [0u8; 256];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(sample.key_expr, "can/0/0x150");
        let decoded = CanFrame::decode(sample.payload).unwrap();
        assert_eq!(decoded.id, 0x150);
        assert_eq!(decoded.data, vec![0xAA, 0xBB]);
    }

    #[test]
    fn test_receive_empty() {
        let mut adapter = CanZenohAdapter::new(ready_ctrl(), 0);
        let mut buf = [0u8; 256];
        assert!(adapter.receive(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_poll_multiple_frames() {
        let mut ctrl = ready_ctrl();
        ctrl.inject_rx(CanFrame::new_standard(0x100, &[0x01]).unwrap());
        ctrl.inject_rx(CanFrame::new_standard(0x200, &[0x02]).unwrap());
        ctrl.inject_rx(CanFrame::new_standard(0x300, &[0x03]).unwrap());

        let mut adapter = CanZenohAdapter::new(ctrl, 0);
        let count = adapter.poll_rx().unwrap();
        assert_eq!(count, 3);
        assert_eq!(adapter.rx_queue.len(), 3);
    }
}
