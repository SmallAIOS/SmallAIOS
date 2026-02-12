// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! BAG (Bandwidth Allocation Gap) traffic shaping.
//!
//! Enforces minimum inter-frame spacing per Virtual Link and implements
//! round-robin sub-VL scheduling.

use crate::arinc664::vl::VirtualLink;
use crate::BusError;

/// Per-VL transmit state for BAG enforcement and sequence numbering.
#[derive(Debug)]
pub struct BagShaper {
    /// Virtual Link configuration.
    vl: VirtualLink,
    /// Timestamp (us) of the last transmitted frame.
    last_tx_us: u64,
    /// Whether any frame has been transmitted yet.
    has_transmitted: bool,
    /// Current sequence number (0-255, wrapping).
    sequence_number: u8,
    /// Round-robin index for sub-VL scheduling.
    rr_index: u8,
}

impl BagShaper {
    /// Create a new BAG shaper for the given Virtual Link.
    pub fn new(vl: VirtualLink) -> Self {
        Self {
            vl,
            last_tx_us: 0,
            has_transmitted: false,
            sequence_number: 0,
            rr_index: 0,
        }
    }

    /// Returns a reference to the Virtual Link configuration.
    pub fn vl(&self) -> &VirtualLink {
        &self.vl
    }

    /// Returns the current sequence number (before next increment).
    pub fn sequence_number(&self) -> u8 {
        self.sequence_number
    }

    /// Check whether a frame can be transmitted at time `now_us`.
    ///
    /// Returns `true` if the BAG interval has elapsed since the last transmission.
    pub fn can_transmit(&self, now_us: u64) -> bool {
        if !self.has_transmitted {
            return true;
        }
        now_us >= self.last_tx_us + self.vl.bag_us()
    }

    /// Record a frame transmission at time `now_us`.
    ///
    /// Returns the sequence number assigned to this frame.
    /// Returns `Err(OutOfRange)` if the BAG interval hasn't elapsed.
    pub fn record_tx(&mut self, now_us: u64) -> Result<u8, BusError> {
        if !self.can_transmit(now_us) {
            return Err(BusError::OutOfRange);
        }
        let seq = self.sequence_number;
        self.sequence_number = self.sequence_number.wrapping_add(1);
        self.last_tx_us = now_us;
        self.has_transmitted = true;
        Ok(seq)
    }

    /// Get the next sub-VL index to service using round-robin.
    ///
    /// `has_data` is a closure that returns `true` if the sub-VL at the
    /// given index has pending data. Returns `None` if no sub-VL has data.
    pub fn next_sub_vl<F>(&mut self, has_data: F) -> Option<u8>
    where
        F: Fn(u8) -> bool,
    {
        let n = self.vl.num_sub_vls();
        for _ in 0..n {
            let idx = self.rr_index % n;
            self.rr_index = self.rr_index.wrapping_add(1);
            if has_data(idx) {
                return Some(idx);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vl(bag_ms: u16, sub_vls: u8) -> VirtualLink {
        VirtualLink::new(1001, bag_ms, 512, sub_vls, false).unwrap()
    }

    #[test]
    fn test_shaper_first_tx_always_allowed() {
        let shaper = BagShaper::new(make_vl(16, 1));
        assert!(shaper.can_transmit(0));
    }

    #[test]
    fn test_shaper_bag_enforcement() {
        let mut shaper = BagShaper::new(make_vl(16, 1));
        let seq = shaper.record_tx(0).unwrap();
        assert_eq!(seq, 0);
        assert!(!shaper.can_transmit(15_999));
        assert!(shaper.can_transmit(16_000));
    }

    #[test]
    fn test_shaper_rejects_early_tx() {
        let mut shaper = BagShaper::new(make_vl(16, 1));
        shaper.record_tx(0).unwrap();
        assert_eq!(shaper.record_tx(15_000).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_sequence_number_wraps() {
        let mut shaper = BagShaper::new(make_vl(1, 1));
        for i in 0..=255u16 {
            let seq = shaper.record_tx(i as u64 * 1000).unwrap();
            assert_eq!(seq, i as u8);
        }
        // 256th transmission wraps to 0
        let seq = shaper.record_tx(256_000).unwrap();
        assert_eq!(seq, 0);
    }

    #[test]
    fn test_round_robin_all_have_data() {
        let mut shaper = BagShaper::new(make_vl(16, 3));
        assert_eq!(shaper.next_sub_vl(|_| true), Some(0));
        assert_eq!(shaper.next_sub_vl(|_| true), Some(1));
        assert_eq!(shaper.next_sub_vl(|_| true), Some(2));
        assert_eq!(shaper.next_sub_vl(|_| true), Some(0));
    }

    #[test]
    fn test_round_robin_skip_empty() {
        let mut shaper = BagShaper::new(make_vl(16, 3));
        // Only sub-VL 0 and 2 have data
        assert_eq!(shaper.next_sub_vl(|i| i != 1), Some(0));
        assert_eq!(shaper.next_sub_vl(|i| i != 1), Some(2));
        assert_eq!(shaper.next_sub_vl(|i| i != 1), Some(0));
    }

    #[test]
    fn test_round_robin_no_data() {
        let mut shaper = BagShaper::new(make_vl(16, 3));
        assert_eq!(shaper.next_sub_vl(|_| false), None);
    }

    #[test]
    fn test_round_robin_single_sub_vl() {
        let mut shaper = BagShaper::new(make_vl(16, 1));
        assert_eq!(shaper.next_sub_vl(|_| true), Some(0));
        assert_eq!(shaper.next_sub_vl(|_| true), Some(0));
    }
}
