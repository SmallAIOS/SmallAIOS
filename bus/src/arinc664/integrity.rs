// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AFDX Virtual Link integrity checking.
//!
//! Validates received frames against the VL configuration including
//! BAG policing (receive-side rate enforcement), Lmax checking,
//! and VL ID / MAC address filtering.

use crate::BusError;
use crate::arinc664::vl::VirtualLink;

/// Jitter tolerance for BAG policing in microseconds.
const JITTER_TOLERANCE_US: u64 = 500;

/// Per-VL receive-side integrity checker.
#[derive(Debug)]
pub struct IntegrityChecker {
    /// Virtual Link configuration.
    vl: VirtualLink,
    /// Timestamp (us) of the last accepted frame.
    last_rx_us: u64,
    /// Whether any frame has been received.
    has_received: bool,
    /// Count of frames rejected by BAG policing.
    policing_violations: u64,
    /// Count of frames rejected for exceeding Lmax.
    lmax_violations: u64,
    /// Count of frames rejected for VL ID / MAC mismatch.
    filter_rejects: u64,
}

impl IntegrityChecker {
    /// Create a new integrity checker for the given VL.
    pub fn new(vl: VirtualLink) -> Self {
        Self {
            vl,
            last_rx_us: 0,
            has_received: false,
            policing_violations: 0,
            lmax_violations: 0,
            filter_rejects: 0,
        }
    }

    /// Check whether a frame passes VL ID / MAC filtering.
    pub fn check_filter(&mut self, dest_mac: &[u8; 6]) -> Result<(), BusError> {
        if *dest_mac != self.vl.dest_mac() {
            self.filter_rejects += 1;
            return Err(BusError::InvalidHeader);
        }
        Ok(())
    }

    /// Check whether a frame's payload size is within Lmax.
    ///
    /// `payload_len` is the Ethernet payload length (including AFDX trailer).
    pub fn check_lmax(&mut self, payload_len: usize) -> Result<(), BusError> {
        if payload_len > self.vl.lmax() as usize {
            self.lmax_violations += 1;
            return Err(BusError::PayloadTooLarge);
        }
        Ok(())
    }

    /// Check BAG policing: reject frames arriving faster than BAG - jitter tolerance.
    ///
    /// Returns `Ok(())` if the frame is within the allowed rate.
    pub fn check_bag(&mut self, now_us: u64) -> Result<(), BusError> {
        if !self.has_received {
            self.has_received = true;
            self.last_rx_us = now_us;
            return Ok(());
        }
        let min_interval = self.vl.bag_us().saturating_sub(JITTER_TOLERANCE_US);
        let elapsed = now_us.saturating_sub(self.last_rx_us);
        if elapsed < min_interval {
            self.policing_violations += 1;
            return Err(BusError::OutOfRange);
        }
        self.last_rx_us = now_us;
        Ok(())
    }

    /// Returns the policing violation count.
    pub fn policing_violations(&self) -> u64 {
        self.policing_violations
    }

    /// Returns the Lmax violation count.
    pub fn lmax_violations(&self) -> u64 {
        self.lmax_violations
    }

    /// Returns the filter reject count.
    pub fn filter_rejects(&self) -> u64 {
        self.filter_rejects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vl() -> VirtualLink {
        VirtualLink::new(1001, 16, 512, 1, false).unwrap()
    }

    #[test]
    fn test_filter_accepts_correct_mac() {
        let mut checker = IntegrityChecker::new(make_vl());
        let mac = [0x03, 0x00, 0x00, 0x00, 0x03, 0xE9]; // VL 1001
        assert!(checker.check_filter(&mac).is_ok());
    }

    #[test]
    fn test_filter_rejects_wrong_mac() {
        let mut checker = IntegrityChecker::new(make_vl());
        let mac = [0x03, 0x00, 0x00, 0x00, 0x00, 0x01]; // wrong VL
        assert_eq!(checker.check_filter(&mac).err(), Some(BusError::InvalidHeader));
        assert_eq!(checker.filter_rejects(), 1);
    }

    #[test]
    fn test_lmax_accepts_valid_size() {
        let mut checker = IntegrityChecker::new(make_vl());
        assert!(checker.check_lmax(512).is_ok());
        assert!(checker.check_lmax(100).is_ok());
    }

    #[test]
    fn test_lmax_rejects_oversized() {
        let mut checker = IntegrityChecker::new(make_vl());
        assert_eq!(checker.check_lmax(513).err(), Some(BusError::PayloadTooLarge));
        assert_eq!(checker.lmax_violations(), 1);
    }

    #[test]
    fn test_bag_first_frame_always_accepted() {
        let mut checker = IntegrityChecker::new(make_vl());
        assert!(checker.check_bag(0).is_ok());
    }

    #[test]
    fn test_bag_accepts_at_interval() {
        let mut checker = IntegrityChecker::new(make_vl());
        checker.check_bag(0).unwrap();
        assert!(checker.check_bag(16_000).is_ok()); // exactly BAG
    }

    #[test]
    fn test_bag_accepts_with_jitter() {
        let mut checker = IntegrityChecker::new(make_vl());
        checker.check_bag(0).unwrap();
        // 15_500 is BAG - 500us jitter tolerance
        assert!(checker.check_bag(15_500).is_ok());
    }

    #[test]
    fn test_bag_rejects_too_fast() {
        let mut checker = IntegrityChecker::new(make_vl());
        checker.check_bag(0).unwrap();
        // 15_499 is below BAG - jitter tolerance
        assert_eq!(checker.check_bag(15_499).err(), Some(BusError::OutOfRange));
        assert_eq!(checker.policing_violations(), 1);
    }

    #[test]
    fn test_bag_multiple_violations() {
        let mut checker = IntegrityChecker::new(make_vl());
        checker.check_bag(0).unwrap();
        checker.check_bag(1000).unwrap_err();
        checker.check_bag(2000).unwrap_err();
        assert_eq!(checker.policing_violations(), 2);
        // Timestamp not updated on violation, so BAG is still relative to 0
        assert!(checker.check_bag(16_000).is_ok());
    }
}
