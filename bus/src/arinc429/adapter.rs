// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429-to-Zenoh transport adapter.
//!
//! Maps ARINC 429 words to Zenoh key expressions: `arinc429/{channel}/{label}`
//!
//! The label is encoded in octal (e.g., `0o310`). The payload is the raw 32-bit
//! ARINC 429 word encoded as 4 bytes big-endian.

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;

use super::word::{Arinc429Word, Label};
use crate::transport::{BusSample, ZenohTransport};
use crate::BusError;

/// Maximum number of queued receive words.
const MAX_RX_QUEUE: usize = 128;

/// ARINC 429-to-Zenoh transport adapter.
///
/// Wraps an ARINC 429 channel and translates between ARINC 429 words and Zenoh
/// key expressions of the form `arinc429/{channel}/{label}`.
pub struct Arinc429ZenohAdapter {
    /// Channel identifier (0, 1, 2...).
    channel: u8,
    /// Key expression prefix (e.g., "arinc429/0").
    prefix: String,
    /// Receive buffer for words.
    rx_queue: VecDeque<(String, [u8; 4])>,
}

impl Arinc429ZenohAdapter {
    /// Create a new ARINC 429-to-Zenoh adapter.
    pub fn new(channel: u8) -> Self {
        Self {
            channel,
            prefix: format!("arinc429/{}", channel),
            rx_queue: VecDeque::new(),
        }
    }

    /// Returns the channel identifier.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// Parse a label from a key expression suffix.
    ///
    /// Accepts octal format: `0o310` or decimal: `200`
    pub fn parse_label(suffix: &str) -> Result<Label, BusError> {
        if let Some(oct) = suffix
            .strip_prefix("0o")
            .or_else(|| suffix.strip_prefix("0O"))
        {
            let val = u16::from_str_radix(oct, 8).map_err(|_| BusError::InvalidLabel)?;
            Label::from_octal(val)
        } else {
            let val: u8 = suffix.parse().map_err(|_| BusError::InvalidLabel)?;
            Ok(Label::new(val))
        }
    }

    /// Build a key expression for an ARINC 429 word.
    fn key_for_word(prefix: &str, label: Label) -> String {
        format!("{}/0o{:o}", prefix, label.value())
    }

    /// Inject a received word into the adapter's RX queue.
    pub fn inject_rx(&mut self, word: u32) {
        if self.rx_queue.len() >= MAX_RX_QUEUE {
            return;
        }
        let label = Arinc429Word::raw_label(word);
        let key = Self::key_for_word(&self.prefix, label);
        self.rx_queue.push_back((key, word.to_be_bytes()));
    }

    /// Return the number of queued receive words.
    pub fn rx_count(&self) -> usize {
        self.rx_queue.len()
    }
}

impl ZenohTransport for Arinc429ZenohAdapter {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn transmit(&mut self, key_expr: &str, payload: &[u8]) -> Result<(), BusError> {
        // Validate the key expression format.
        let _suffix = key_expr
            .strip_prefix(&self.prefix)
            .and_then(|s| s.strip_prefix('/'))
            .ok_or(BusError::InvalidLabel)?;

        // Validate payload is a 32-bit word.
        if payload.len() != 4 {
            return Err(BusError::FrameTooShort);
        }

        let word = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
        // Verify parity.
        let _ = Arinc429Word::decode(word)?;

        Ok(())
    }

    fn receive<'a>(&mut self, buf: &'a mut [u8]) -> Result<Option<BusSample<'a>>, BusError> {
        if let Some((key, word_bytes)) = self.rx_queue.pop_front() {
            let key_bytes = key.as_bytes();
            let total = key_bytes.len() + 1 + 4;
            if buf.len() < total {
                return Err(BusError::BufferTooSmall);
            }
            buf[..key_bytes.len()].copy_from_slice(key_bytes);
            buf[key_bytes.len()] = 0;
            let payload_start = key_bytes.len() + 1;
            buf[payload_start..payload_start + 4].copy_from_slice(&word_bytes);

            Ok(Some(BusSample {
                key_expr: core::str::from_utf8(&buf[..key_bytes.len()])
                    .map_err(|_| BusError::InvalidHeader)?,
                payload: &buf[payload_start..payload_start + 4],
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
    use crate::arinc429::word::{Arinc429Word, Label, Sdi, Ssm};

    #[test]
    fn test_adapter_prefix() {
        let adapter = Arinc429ZenohAdapter::new(0);
        assert_eq!(adapter.prefix(), "arinc429/0");
    }

    #[test]
    fn test_adapter_prefix_channel1() {
        let adapter = Arinc429ZenohAdapter::new(1);
        assert_eq!(adapter.prefix(), "arinc429/1");
    }

    #[test]
    fn test_parse_label_octal() {
        let label = Arinc429ZenohAdapter::parse_label("0o310").unwrap();
        assert_eq!(label.value(), 0o310);
    }

    #[test]
    fn test_parse_label_decimal() {
        let label = Arinc429ZenohAdapter::parse_label("200").unwrap();
        assert_eq!(label.value(), 200);
    }

    #[test]
    fn test_parse_label_invalid() {
        assert!(Arinc429ZenohAdapter::parse_label("xyz").is_err());
    }

    #[test]
    fn test_key_for_word() {
        let key = Arinc429ZenohAdapter::key_for_word("arinc429/0", Label::new(0o310));
        assert_eq!(key, "arinc429/0/0o310");
    }

    #[test]
    fn test_inject_and_receive() {
        let mut adapter = Arinc429ZenohAdapter::new(0);
        let word = Arinc429Word {
            label: Label::new(0o200),
            sdi: Sdi::Zero,
            data: 0x12345,
            ssm: Ssm::NormalOp,
        };
        let encoded = word.encode();
        adapter.inject_rx(encoded);
        assert_eq!(adapter.rx_count(), 1);

        let mut buf = [0u8; 256];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(sample.key_expr, "arinc429/0/0o200");
        assert_eq!(sample.payload.len(), 4);
        let received_word = u32::from_be_bytes([
            sample.payload[0],
            sample.payload[1],
            sample.payload[2],
            sample.payload[3],
        ]);
        assert_eq!(received_word, encoded);
    }

    #[test]
    fn test_receive_empty() {
        let mut adapter = Arinc429ZenohAdapter::new(0);
        let mut buf = [0u8; 256];
        assert!(adapter.receive(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_transmit_valid_word() {
        let mut adapter = Arinc429ZenohAdapter::new(0);
        let word = Arinc429Word {
            label: Label::new(0o310),
            sdi: Sdi::Zero,
            data: 0,
            ssm: Ssm::NormalOp,
        };
        let encoded = word.encode();
        let payload = encoded.to_be_bytes();
        adapter.transmit("arinc429/0/0o310", &payload).unwrap();
    }

    #[test]
    fn test_transmit_wrong_size() {
        let mut adapter = Arinc429ZenohAdapter::new(0);
        assert_eq!(
            adapter.transmit("arinc429/0/0o310", &[0x01, 0x02]),
            Err(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_transmit_invalid_key() {
        let mut adapter = Arinc429ZenohAdapter::new(0);
        assert!(adapter.transmit("can/0/0x100", &[0; 4]).is_err());
    }

    #[test]
    fn test_multiple_labels() {
        let mut adapter = Arinc429ZenohAdapter::new(0);
        for label_val in [0o100, 0o200, 0o310] {
            let word = Arinc429Word {
                label: Label::new(label_val),
                sdi: Sdi::Zero,
                data: 0,
                ssm: Ssm::NormalOp,
            };
            adapter.inject_rx(word.encode());
        }
        assert_eq!(adapter.rx_count(), 3);
    }
}
