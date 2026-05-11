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

/// Pack a key expression and payload into the caller-supplied scratch buffer
/// and return a borrowed [`BusSample`].
///
/// This is the shared layout used by every Zenoh adapter:
/// `[key bytes][NUL separator][payload bytes]`. Centralising the layout keeps
/// the `receive` implementations identical across protocols and avoids the
/// `rust:S4144` duplicated-block findings flagged by SonarCloud.
///
/// # Errors
///
/// - [`BusError::BufferTooSmall`] if `buf` cannot hold `key.len() + 1 + data.len()` bytes.
/// - [`BusError::InvalidHeader`] if the freshly copied key bytes do not round-trip as UTF-8
///   (only possible if `key` itself was not valid UTF-8, which Rust `&str` guarantees against).
pub fn pack_sample<'a>(
    buf: &'a mut [u8],
    key: &str,
    data: &[u8],
) -> Result<BusSample<'a>, BusError> {
    let key_bytes = key.as_bytes();
    let key_len = key_bytes.len();
    let total = key_len + 1 + data.len();
    if buf.len() < total {
        return Err(BusError::BufferTooSmall);
    }
    buf[..key_len].copy_from_slice(key_bytes);
    buf[key_len] = 0;
    let payload_start = key_len + 1;
    let payload_end = payload_start + data.len();
    buf[payload_start..payload_end].copy_from_slice(data);

    let (key_slice, rest) = buf.split_at(key_len);
    let key_expr = core::str::from_utf8(key_slice).map_err(|_| BusError::InvalidHeader)?;
    let payload = &rest[1..1 + data.len()];

    Ok(BusSample {
        key_expr,
        payload,
        timestamp_us: 0,
    })
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

    #[test]
    fn test_pack_sample_basic() {
        let mut buf = [0u8; 64];
        let sample = pack_sample(&mut buf, "afdx/100", &[0xAA, 0xBB]).unwrap();
        assert_eq!(sample.key_expr, "afdx/100");
        assert_eq!(sample.payload, &[0xAA, 0xBB]);
        assert_eq!(sample.timestamp_us, 0);
    }

    #[test]
    fn test_pack_sample_empty_payload() {
        let mut buf = [0u8; 16];
        let sample = pack_sample(&mut buf, "ccsds/7", &[]).unwrap();
        assert_eq!(sample.key_expr, "ccsds/7");
        assert!(sample.payload.is_empty());
    }

    #[test]
    fn test_pack_sample_buffer_too_small() {
        let mut buf = [0u8; 4];
        assert_eq!(
            pack_sample(&mut buf, "afdx/100", &[0xAA, 0xBB]).err(),
            Some(BusError::BufferTooSmall)
        );
    }

    #[test]
    fn test_pack_sample_exact_size() {
        // key (8) + nul (1) + payload (2) == 11
        let mut buf = [0u8; 11];
        let sample = pack_sample(&mut buf, "afdx/100", &[0xAA, 0xBB]).unwrap();
        assert_eq!(sample.key_expr, "afdx/100");
        assert_eq!(sample.payload, &[0xAA, 0xBB]);
    }
}
