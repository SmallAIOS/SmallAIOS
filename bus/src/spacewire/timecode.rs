// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SpaceWire time-code distribution.
//!
//! Implements 6-bit time-code broadcast for time synchronization across
//! the SpaceWire network. Time-codes have the highest priority and
//! pre-empt any packet in progress.

/// Time-code counter modulus (6-bit: 0-63).
pub const TIME_CODE_MODULUS: u8 = 64;

/// Time-code manager for a SpaceWire node.
#[derive(Debug)]
pub struct TimeCodeManager {
    /// Local 6-bit time counter (0-63).
    local_time: u8,
    /// Whether this node is the time master.
    is_master: bool,
    /// Count of accepted time-codes.
    accepted_count: u64,
    /// Count of rejected time-codes (wrong value).
    rejected_count: u64,
}

impl TimeCodeManager {
    /// Create a new time-code manager.
    ///
    /// If `is_master` is true, this node generates time-codes.
    pub fn new(is_master: bool) -> Self {
        Self {
            local_time: 0,
            is_master,
            accepted_count: 0,
            rejected_count: 0,
        }
    }

    /// Returns the current local time counter.
    pub fn local_time(&self) -> u8 {
        self.local_time
    }

    /// Returns whether this node is the time master.
    pub fn is_master(&self) -> bool {
        self.is_master
    }

    /// Generate a time tick (master only).
    ///
    /// Increments the local counter and returns the time-code value to transmit.
    /// Returns `None` if this node is not the time master.
    pub fn generate_tick(&mut self) -> Option<u8> {
        if !self.is_master {
            return None;
        }
        self.local_time = (self.local_time + 1) % TIME_CODE_MODULUS;
        Some(self.local_time)
    }

    /// Receive a time-code from the network.
    ///
    /// Accepts the time-code if it equals (local_time + 1) mod 64.
    /// Returns `true` if accepted, `false` if rejected.
    pub fn receive_time_code(&mut self, value: u8) -> bool {
        let expected = (self.local_time + 1) % TIME_CODE_MODULUS;
        if value % TIME_CODE_MODULUS == expected {
            self.local_time = expected;
            self.accepted_count += 1;
            true
        } else {
            self.rejected_count += 1;
            false
        }
    }

    /// Returns the count of accepted time-codes.
    pub fn accepted_count(&self) -> u64 {
        self.accepted_count
    }

    /// Returns the count of rejected time-codes.
    pub fn rejected_count(&self) -> u64 {
        self.rejected_count
    }

    /// Encode a time-code for transmission.
    ///
    /// Returns an 8-bit value: 2-bit control (0b00) + 6-bit time counter.
    pub fn encode_time_code(value: u8) -> u8 {
        value & 0x3F // Control bits = 00 in upper 2 bits (implicitly zero)
    }

    /// Decode a received time-code byte.
    ///
    /// Returns the 6-bit time counter value.
    pub fn decode_time_code(byte: u8) -> u8 {
        byte & 0x3F
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_master_generate_tick() {
        let mut mgr = TimeCodeManager::new(true);
        assert_eq!(mgr.local_time(), 0);
        assert_eq!(mgr.generate_tick(), Some(1));
        assert_eq!(mgr.local_time(), 1);
        assert_eq!(mgr.generate_tick(), Some(2));
    }

    #[test]
    fn test_non_master_cannot_generate() {
        let mut mgr = TimeCodeManager::new(false);
        assert_eq!(mgr.generate_tick(), None);
    }

    #[test]
    fn test_master_wraps_at_63() {
        let mut mgr = TimeCodeManager::new(true);
        for _ in 0..63 {
            mgr.generate_tick();
        }
        assert_eq!(mgr.local_time(), 63);
        assert_eq!(mgr.generate_tick(), Some(0)); // wraps
        assert_eq!(mgr.local_time(), 0);
    }

    #[test]
    fn test_receive_valid_time_code() {
        let mut mgr = TimeCodeManager::new(false);
        assert!(mgr.receive_time_code(1)); // expects (0+1) % 64 = 1
        assert_eq!(mgr.local_time(), 1);
        assert_eq!(mgr.accepted_count(), 1);
    }

    #[test]
    fn test_receive_sequential_time_codes() {
        let mut mgr = TimeCodeManager::new(false);
        for i in 1..=10 {
            assert!(mgr.receive_time_code(i));
        }
        assert_eq!(mgr.local_time(), 10);
        assert_eq!(mgr.accepted_count(), 10);
    }

    #[test]
    fn test_receive_invalid_time_code() {
        let mut mgr = TimeCodeManager::new(false);
        assert!(!mgr.receive_time_code(5)); // expects 1, got 5
        assert_eq!(mgr.local_time(), 0); // unchanged
        assert_eq!(mgr.rejected_count(), 1);
    }

    #[test]
    fn test_receive_wrapping_time_code() {
        let mut mgr = TimeCodeManager::new(false);
        // Advance to 63
        for i in 1..=63 {
            assert!(mgr.receive_time_code(i));
        }
        assert_eq!(mgr.local_time(), 63);
        // Wrap to 0
        assert!(mgr.receive_time_code(0));
        assert_eq!(mgr.local_time(), 0);
    }

    #[test]
    fn test_encode_decode_time_code() {
        for v in 0..64 {
            let encoded = TimeCodeManager::encode_time_code(v);
            let decoded = TimeCodeManager::decode_time_code(encoded);
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn test_encode_masks_upper_bits() {
        let encoded = TimeCodeManager::encode_time_code(0xFF);
        assert_eq!(encoded, 0x3F); // only lower 6 bits
    }
}
