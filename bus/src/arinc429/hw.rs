// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429 hardware interface abstraction.
//!
//! Defines traits for SPI transceiver ICs (e.g., Holt HI-3585, DDC)
//! and FPGA soft-IP ARINC 429 controllers. Platform HALs provide
//! concrete implementations.

use crate::BusError;

/// ARINC 429 channel speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arinc429Speed {
    /// Low speed: 12.5 kbps.
    Low,
    /// High speed: 100 kbps.
    High,
}

/// ARINC 429 channel direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelDirection {
    /// Transmit channel.
    Tx,
    /// Receive channel.
    Rx,
}

/// Hardware-independent ARINC 429 transceiver trait.
///
/// Implementations target specific hardware:
/// - SPI transceivers (Holt HI-3585, HI-3593, DDC BU-67118)
/// - FPGA soft-IP (AXI-mapped ARINC 429 controllers on Zynq/PolarFire)
/// - Memory-mapped ASIC controllers
pub trait Arinc429Transceiver {
    /// Initialize the transceiver hardware.
    fn init(&mut self) -> Result<(), BusError>;

    /// Configure the channel speed (12.5 kbps or 100 kbps).
    fn set_speed(&mut self, channel: u8, speed: Arinc429Speed) -> Result<(), BusError>;

    /// Transmit a 32-bit ARINC 429 word on the specified TX channel.
    fn transmit(&mut self, channel: u8, word: u32) -> Result<(), BusError>;

    /// Receive a 32-bit ARINC 429 word from the specified RX channel.
    ///
    /// Returns `Ok(Some(word))` if data is available, `Ok(None)` if empty.
    fn receive(&mut self, channel: u8) -> Result<Option<u32>, BusError>;

    /// Get the number of available TX channels.
    fn tx_channel_count(&self) -> u8;

    /// Get the number of available RX channels.
    fn rx_channel_count(&self) -> u8;

    /// Enable or disable a label filter on an RX channel.
    fn set_label_filter(&mut self, channel: u8, label: u8, enabled: bool) -> Result<(), BusError>;

    /// Reset the transceiver to its initial state.
    fn reset(&mut self) -> Result<(), BusError>;
}

/// Mock ARINC 429 transceiver for testing.
pub struct MockArinc429Transceiver {
    tx_channels: u8,
    rx_channels: u8,
    speed: [Arinc429Speed; 8],
    rx_queue: [alloc::vec::Vec<u32>; 8],
    initialized: bool,
}

impl MockArinc429Transceiver {
    /// Create a mock transceiver with the given number of TX and RX channels.
    pub fn new(tx_channels: u8, rx_channels: u8) -> Self {
        Self {
            tx_channels,
            rx_channels,
            speed: [Arinc429Speed::High; 8],
            rx_queue: Default::default(),
            initialized: false,
        }
    }

    /// Inject a word into an RX channel queue.
    pub fn inject_rx(&mut self, channel: u8, word: u32) {
        if (channel as usize) < self.rx_queue.len() {
            self.rx_queue[channel as usize].push(word);
        }
    }
}

impl Arinc429Transceiver for MockArinc429Transceiver {
    fn init(&mut self) -> Result<(), BusError> {
        self.initialized = true;
        Ok(())
    }

    fn set_speed(&mut self, channel: u8, speed: Arinc429Speed) -> Result<(), BusError> {
        if (channel as usize) >= self.speed.len() {
            return Err(BusError::OutOfRange);
        }
        self.speed[channel as usize] = speed;
        Ok(())
    }

    fn transmit(&mut self, channel: u8, _word: u32) -> Result<(), BusError> {
        if channel >= self.tx_channels {
            return Err(BusError::OutOfRange);
        }
        if !self.initialized {
            return Err(BusError::NotSupported);
        }
        Ok(())
    }

    fn receive(&mut self, channel: u8) -> Result<Option<u32>, BusError> {
        if channel >= self.rx_channels {
            return Err(BusError::OutOfRange);
        }
        let q = &mut self.rx_queue[channel as usize];
        if q.is_empty() {
            Ok(None)
        } else {
            Ok(Some(q.remove(0)))
        }
    }

    fn tx_channel_count(&self) -> u8 {
        self.tx_channels
    }

    fn rx_channel_count(&self) -> u8 {
        self.rx_channels
    }

    fn set_label_filter(
        &mut self,
        channel: u8,
        _label: u8,
        _enabled: bool,
    ) -> Result<(), BusError> {
        if channel >= self.rx_channels {
            return Err(BusError::OutOfRange);
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<(), BusError> {
        self.initialized = false;
        for q in &mut self.rx_queue {
            q.clear();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_init() {
        let mut hw = MockArinc429Transceiver::new(2, 4);
        assert!(!hw.initialized);
        hw.init().unwrap();
        assert!(hw.initialized);
    }

    #[test]
    fn test_mock_channel_counts() {
        let hw = MockArinc429Transceiver::new(2, 4);
        assert_eq!(hw.tx_channel_count(), 2);
        assert_eq!(hw.rx_channel_count(), 4);
    }

    #[test]
    fn test_mock_speed() {
        let mut hw = MockArinc429Transceiver::new(1, 1);
        hw.init().unwrap();
        hw.set_speed(0, Arinc429Speed::Low).unwrap();
        assert_eq!(hw.speed[0], Arinc429Speed::Low);
    }

    #[test]
    fn test_mock_transmit() {
        let mut hw = MockArinc429Transceiver::new(1, 1);
        hw.init().unwrap();
        hw.transmit(0, 0x12345678).unwrap();
    }

    #[test]
    fn test_mock_transmit_uninit() {
        let mut hw = MockArinc429Transceiver::new(1, 1);
        assert_eq!(hw.transmit(0, 0).err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_mock_receive_empty() {
        let mut hw = MockArinc429Transceiver::new(1, 1);
        assert!(hw.receive(0).unwrap().is_none());
    }

    #[test]
    fn test_mock_receive_inject() {
        let mut hw = MockArinc429Transceiver::new(1, 1);
        hw.inject_rx(0, 0xAABBCCDD);
        let word = hw.receive(0).unwrap().unwrap();
        assert_eq!(word, 0xAABBCCDD);
    }

    #[test]
    fn test_mock_reset() {
        let mut hw = MockArinc429Transceiver::new(1, 1);
        hw.init().unwrap();
        hw.inject_rx(0, 0x12345678);
        hw.reset().unwrap();
        assert!(!hw.initialized);
        assert!(hw.receive(0).unwrap().is_none());
    }

    #[test]
    fn test_mock_label_filter() {
        let mut hw = MockArinc429Transceiver::new(1, 1);
        hw.set_label_filter(0, 0o200, true).unwrap();
    }

    #[test]
    fn test_channel_out_of_range() {
        let mut hw = MockArinc429Transceiver::new(1, 1);
        hw.init().unwrap();
        assert_eq!(hw.transmit(5, 0).err(), Some(BusError::OutOfRange));
        assert_eq!(hw.receive(5).err(), Some(BusError::OutOfRange));
    }
}
