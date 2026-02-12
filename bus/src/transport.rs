// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Zenoh transport adapter trait for bus protocols.
//!
//! Each bus protocol implements [`ZenohTransport`] to map protocol-native
//! addressing to Zenoh key expressions, enabling transport-agnostic messaging.
//!
//! Mapping scheme:
//! - CAN: `can/{bus_id}/{frame_id}`
//! - ARINC 429: `arinc429/{channel}/{label}`
//! - ARINC 664: `afdx/{vl_id}`
//! - MIL-1553: `mil1553/{bus}/{rt}/{sa}`
//! - SpaceWire: `spw/{link}/{dest}`
//! - CCSDS: `ccsds/{apid}`
//! - DDS: `dds/{domain_id}/{topic}`

use crate::BusError;

/// A raw bus frame with its Zenoh key expression and payload.
pub struct BusSample<'a> {
    /// Zenoh key expression (e.g., `can/0/0x1A3`).
    pub key_expr: &'a str,
    /// Raw payload bytes.
    pub payload: &'a [u8],
    /// Timestamp in microseconds (system monotonic).
    pub timestamp_us: u64,
}

/// Transport adapter trait mapping bus protocols to Zenoh key expressions.
///
/// Implementors convert between protocol-native frames and Zenoh-compatible
/// samples. The IPC router calls [`receive`] to poll for incoming frames and
/// [`transmit`] to send outgoing frames on the physical bus.
pub trait ZenohTransport {
    /// Returns the key expression prefix for this transport (e.g., `"can"`).
    fn prefix(&self) -> &str;

    /// Transmit a payload to the bus, addressed by Zenoh key expression.
    ///
    /// The implementation parses the key expression to extract protocol-native
    /// addressing (e.g., bus ID and frame ID for CAN).
    fn transmit(&mut self, key_expr: &str, payload: &[u8]) -> Result<(), BusError>;

    /// Poll for a received frame from the bus.
    ///
    /// Returns `Ok(Some(sample))` if a frame is available, `Ok(None)` if the
    /// receive queue is empty, or `Err` on bus error.
    fn receive<'a>(&mut self, buf: &'a mut [u8]) -> Result<Option<BusSample<'a>>, BusError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_sample_fields() {
        let payload = [0x01, 0x02, 0x03];
        let sample = BusSample {
            key_expr: "can/0/0x1A3",
            payload: &payload,
            timestamp_us: 12345,
        };
        assert_eq!(sample.key_expr, "can/0/0x1A3");
        assert_eq!(sample.payload, &[0x01, 0x02, 0x03]);
        assert_eq!(sample.timestamp_us, 12345);
    }
}
