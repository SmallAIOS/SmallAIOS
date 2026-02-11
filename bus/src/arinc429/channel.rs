// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429 channel configuration for low speed and high speed operation.
//!
//! ARINC 429 defines two bus speeds:
//! - **High speed**: 100 kbps, 10 us per bit, 320 us per word (32 bits)
//! - **Low speed**: 12.5 kbps, 80 us per bit, 2560 us per word (32 bits)
//!
//! Each channel is simplex (unidirectional): a single transmitter drives
//! one or more receivers on a twisted-pair wire. This module provides
//! timing configuration for both speeds.

use crate::BusError;
use super::scheduler::TxScheduler;

/// ARINC 429 bus speed selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusSpeed {
    /// Low speed: 12.5 kbps (12,500 bits per second).
    Low,
    /// High speed: 100 kbps (100,000 bits per second).
    High,
}

impl BusSpeed {
    /// Bit rate in bits per second.
    pub fn bit_rate_bps(self) -> u32 {
        match self {
            BusSpeed::Low => 12_500,
            BusSpeed::High => 100_000,
        }
    }

    /// Duration of a single bit in microseconds.
    pub fn bit_time_us(self) -> u64 {
        match self {
            BusSpeed::Low => 80,
            BusSpeed::High => 10,
        }
    }

    /// Duration of a complete 32-bit word on the wire (microseconds).
    ///
    /// A word is exactly 32 bit-times.
    pub fn word_time_us(self) -> u64 {
        self.bit_time_us() * 32
    }

    /// Minimum inter-word gap in microseconds (4 bit-times per ARINC 429).
    pub fn min_gap_us(self) -> u64 {
        self.bit_time_us() * 4
    }

    /// Maximum number of words per second at this speed (including gaps).
    ///
    /// Each word occupies 32 + 4 = 36 bit-times minimum.
    pub fn max_words_per_second(self) -> u32 {
        let bits_per_word_with_gap = 36u64;
        (self.bit_rate_bps() as u64 / bits_per_word_with_gap) as u32
    }
}

/// ARINC 429 channel configuration.
///
/// Captures the bus speed, direction, and channel identifier for a physical
/// ARINC 429 connection. Used to configure adapters and schedulers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelConfig {
    /// Channel identifier (0, 1, 2, ...).
    pub channel_id: u8,
    /// Bus speed selection.
    pub speed: BusSpeed,
    /// Whether this channel is a transmitter (true) or receiver (false).
    pub is_transmitter: bool,
    /// Maximum number of scheduled labels (for transmit channels).
    pub max_labels: usize,
}

impl ChannelConfig {
    /// Create a high-speed transmit channel configuration.
    pub fn high_speed_tx(channel_id: u8) -> Self {
        Self {
            channel_id,
            speed: BusSpeed::High,
            is_transmitter: true,
            max_labels: 128,
        }
    }

    /// Create a high-speed receive channel configuration.
    pub fn high_speed_rx(channel_id: u8) -> Self {
        Self {
            channel_id,
            speed: BusSpeed::High,
            is_transmitter: false,
            max_labels: 0,
        }
    }

    /// Create a low-speed transmit channel configuration.
    pub fn low_speed_tx(channel_id: u8) -> Self {
        Self {
            channel_id,
            speed: BusSpeed::Low,
            is_transmitter: true,
            max_labels: 128,
        }
    }

    /// Create a low-speed receive channel configuration.
    pub fn low_speed_rx(channel_id: u8) -> Self {
        Self {
            channel_id,
            speed: BusSpeed::Low,
            is_transmitter: false,
            max_labels: 0,
        }
    }

    /// Create a transmit scheduler configured for this channel's speed.
    ///
    /// Returns [`BusError::NotSupported`] if this is a receive-only channel.
    pub fn create_scheduler(&self) -> Result<TxScheduler, BusError> {
        if !self.is_transmitter {
            return Err(BusError::NotSupported);
        }
        Ok(TxScheduler::new(self.speed.min_gap_us()))
    }

    /// Calculate the maximum transmit rate (in Hz) for a single label
    /// at this channel's speed, given that `total_labels` labels share
    /// the bus.
    ///
    /// Assumes round-robin scheduling with equal rate per label.
    pub fn max_rate_per_label(&self, total_labels: u32) -> u32 {
        if total_labels == 0 {
            return 0;
        }
        self.speed.max_words_per_second() / total_labels
    }

    /// Validate that a requested transmission rate (Hz) is achievable
    /// given the current number of active labels on the channel.
    pub fn validate_rate(
        &self,
        rate_hz: u32,
        active_labels: u32,
    ) -> Result<(), BusError> {
        if rate_hz == 0 {
            return Err(BusError::OutOfRange);
        }
        // Check if total bandwidth fits
        let required_words_per_sec = rate_hz as u64 * active_labels as u64;
        if required_words_per_sec > self.speed.max_words_per_second() as u64 {
            return Err(BusError::OutOfRange);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BusSpeed tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_high_speed_bit_rate() {
        assert_eq!(BusSpeed::High.bit_rate_bps(), 100_000);
    }

    #[test]
    fn test_low_speed_bit_rate() {
        assert_eq!(BusSpeed::Low.bit_rate_bps(), 12_500);
    }

    #[test]
    fn test_high_speed_bit_time() {
        assert_eq!(BusSpeed::High.bit_time_us(), 10);
    }

    #[test]
    fn test_low_speed_bit_time() {
        assert_eq!(BusSpeed::Low.bit_time_us(), 80);
    }

    #[test]
    fn test_high_speed_word_time() {
        // 32 bits * 10us = 320us
        assert_eq!(BusSpeed::High.word_time_us(), 320);
    }

    #[test]
    fn test_low_speed_word_time() {
        // 32 bits * 80us = 2560us
        assert_eq!(BusSpeed::Low.word_time_us(), 2560);
    }

    #[test]
    fn test_high_speed_min_gap() {
        // 4 bits * 10us = 40us
        assert_eq!(BusSpeed::High.min_gap_us(), 40);
    }

    #[test]
    fn test_low_speed_min_gap() {
        // 4 bits * 80us = 320us
        assert_eq!(BusSpeed::Low.min_gap_us(), 320);
    }

    #[test]
    fn test_high_speed_max_words_per_second() {
        // 100000 bps / 36 bits per word = 2777
        assert_eq!(BusSpeed::High.max_words_per_second(), 2777);
    }

    #[test]
    fn test_low_speed_max_words_per_second() {
        // 12500 bps / 36 bits per word = 347
        assert_eq!(BusSpeed::Low.max_words_per_second(), 347);
    }

    // -----------------------------------------------------------------------
    // ChannelConfig constructor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_high_speed_tx() {
        let cfg = ChannelConfig::high_speed_tx(0);
        assert_eq!(cfg.channel_id, 0);
        assert_eq!(cfg.speed, BusSpeed::High);
        assert!(cfg.is_transmitter);
        assert_eq!(cfg.max_labels, 128);
    }

    #[test]
    fn test_high_speed_rx() {
        let cfg = ChannelConfig::high_speed_rx(1);
        assert_eq!(cfg.channel_id, 1);
        assert_eq!(cfg.speed, BusSpeed::High);
        assert!(!cfg.is_transmitter);
    }

    #[test]
    fn test_low_speed_tx() {
        let cfg = ChannelConfig::low_speed_tx(2);
        assert_eq!(cfg.channel_id, 2);
        assert_eq!(cfg.speed, BusSpeed::Low);
        assert!(cfg.is_transmitter);
    }

    #[test]
    fn test_low_speed_rx() {
        let cfg = ChannelConfig::low_speed_rx(3);
        assert_eq!(cfg.channel_id, 3);
        assert_eq!(cfg.speed, BusSpeed::Low);
        assert!(!cfg.is_transmitter);
    }

    // -----------------------------------------------------------------------
    // Scheduler creation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_scheduler_tx() {
        let cfg = ChannelConfig::high_speed_tx(0);
        let sched = cfg.create_scheduler().unwrap();
        assert_eq!(sched.min_gap_us(), 40);
    }

    #[test]
    fn test_create_scheduler_low_speed() {
        let cfg = ChannelConfig::low_speed_tx(0);
        let sched = cfg.create_scheduler().unwrap();
        assert_eq!(sched.min_gap_us(), 320);
    }

    #[test]
    fn test_create_scheduler_rx_fails() {
        let cfg = ChannelConfig::high_speed_rx(0);
        assert_eq!(cfg.create_scheduler().err(), Some(BusError::NotSupported));
    }

    // -----------------------------------------------------------------------
    // Rate calculation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_rate_per_label_single() {
        let cfg = ChannelConfig::high_speed_tx(0);
        // One label gets full bandwidth: 2777 Hz
        assert_eq!(cfg.max_rate_per_label(1), 2777);
    }

    #[test]
    fn test_max_rate_per_label_multiple() {
        let cfg = ChannelConfig::high_speed_tx(0);
        // 10 labels share: 2777/10 = 277 Hz each
        assert_eq!(cfg.max_rate_per_label(10), 277);
    }

    #[test]
    fn test_max_rate_per_label_low_speed() {
        let cfg = ChannelConfig::low_speed_tx(0);
        // One label: 347 Hz
        assert_eq!(cfg.max_rate_per_label(1), 347);
    }

    #[test]
    fn test_max_rate_per_label_zero() {
        let cfg = ChannelConfig::high_speed_tx(0);
        assert_eq!(cfg.max_rate_per_label(0), 0);
    }

    // -----------------------------------------------------------------------
    // Rate validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_rate_ok() {
        let cfg = ChannelConfig::high_speed_tx(0);
        // 50 Hz * 10 labels = 500 words/s << 2777
        cfg.validate_rate(50, 10).unwrap();
    }

    #[test]
    fn test_validate_rate_exceeds_bandwidth() {
        let cfg = ChannelConfig::high_speed_tx(0);
        // 500 Hz * 10 labels = 5000 words/s > 2777
        assert_eq!(
            cfg.validate_rate(500, 10).err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_validate_rate_zero() {
        let cfg = ChannelConfig::high_speed_tx(0);
        assert_eq!(cfg.validate_rate(0, 1).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_validate_rate_low_speed_boundary() {
        let cfg = ChannelConfig::low_speed_tx(0);
        // 347 words/s max, 1 label at 347 Hz should fit
        cfg.validate_rate(347, 1).unwrap();
        // But 348 should not
        assert_eq!(
            cfg.validate_rate(348, 1).err(),
            Some(BusError::OutOfRange)
        );
    }

    // -----------------------------------------------------------------------
    // BusSpeed equality
    // -----------------------------------------------------------------------

    #[test]
    fn test_bus_speed_eq() {
        assert_eq!(BusSpeed::High, BusSpeed::High);
        assert_ne!(BusSpeed::High, BusSpeed::Low);
    }

    #[test]
    fn test_bus_speed_clone() {
        let s = BusSpeed::Low;
        let c = s;
        assert_eq!(s, c);
    }
}
