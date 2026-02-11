// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Dual-network redundancy management for AFDX.
//!
//! Handles duplicate elimination based on sequence numbers across Network A
//! and Network B. The first valid frame is delivered; the duplicate is discarded.

/// Network identifier for dual-redundant AFDX.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Network {
    /// Network A.
    A,
    /// Network B.
    B,
}

/// Per-VL redundancy state.
#[derive(Debug)]
pub struct RedundancyManager {
    /// Last accepted sequence number, or `None` if no frame accepted yet.
    last_accepted_seq: Option<u8>,
    /// Network from which the last accepted frame came.
    last_accepted_network: Option<Network>,
    /// Count of duplicate frames discarded.
    duplicate_count: u64,
    /// Count of lost frames detected.
    lost_count: u64,
    /// Network A integrity failure counter.
    net_a_failures: u64,
    /// Network B integrity failure counter.
    net_b_failures: u64,
}

impl Default for RedundancyManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RedundancyManager {
    /// Create a new redundancy manager for a VL.
    pub fn new() -> Self {
        Self {
            last_accepted_seq: None,
            last_accepted_network: None,
            duplicate_count: 0,
            lost_count: 0,
            net_a_failures: 0,
            net_b_failures: 0,
        }
    }

    /// Process a received frame and determine whether to accept or discard it.
    ///
    /// Returns `true` if the frame should be delivered to the application.
    /// Returns `false` if it's a duplicate.
    pub fn accept(&mut self, seq: u8, network: Network) -> bool {
        match self.last_accepted_seq {
            None => {
                self.last_accepted_seq = Some(seq);
                self.last_accepted_network = Some(network);
                true
            }
            Some(last_seq) => {
                if seq == last_seq {
                    // Duplicate from the other network
                    self.duplicate_count += 1;
                    false
                } else {
                    // New sequence number — check for gaps
                    let expected = last_seq.wrapping_add(1);
                    if seq != expected {
                        // Calculate lost frames (accounting for wrapping)
                        let gap = seq.wrapping_sub(expected);
                        self.lost_count += gap as u64;
                    }
                    self.last_accepted_seq = Some(seq);
                    self.last_accepted_network = Some(network);
                    true
                }
            }
        }
    }

    /// Record an integrity failure on a specific network.
    pub fn record_integrity_failure(&mut self, network: Network) {
        match network {
            Network::A => self.net_a_failures += 1,
            Network::B => self.net_b_failures += 1,
        }
    }

    /// Returns the duplicate frame count.
    pub fn duplicate_count(&self) -> u64 {
        self.duplicate_count
    }

    /// Returns the lost frame count.
    pub fn lost_count(&self) -> u64 {
        self.lost_count
    }

    /// Returns the last accepted sequence number.
    pub fn last_accepted_seq(&self) -> Option<u8> {
        self.last_accepted_seq
    }

    /// Returns the network of the last accepted frame.
    pub fn last_accepted_network(&self) -> Option<Network> {
        self.last_accepted_network
    }

    /// Returns integrity failure count for a network.
    pub fn integrity_failures(&self, network: Network) -> u64 {
        match network {
            Network::A => self.net_a_failures,
            Network::B => self.net_b_failures,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_frame_always_accepted() {
        let mut rm = RedundancyManager::new();
        assert!(rm.accept(0, Network::A));
        assert_eq!(rm.last_accepted_seq(), Some(0));
        assert_eq!(rm.last_accepted_network(), Some(Network::A));
    }

    #[test]
    fn test_duplicate_discarded() {
        let mut rm = RedundancyManager::new();
        assert!(rm.accept(0, Network::A));
        assert!(!rm.accept(0, Network::B)); // duplicate
        assert_eq!(rm.duplicate_count(), 1);
    }

    #[test]
    fn test_sequential_frames() {
        let mut rm = RedundancyManager::new();
        assert!(rm.accept(0, Network::A));
        assert!(rm.accept(1, Network::A));
        assert!(rm.accept(2, Network::B));
        assert_eq!(rm.lost_count(), 0);
        assert_eq!(rm.duplicate_count(), 0);
    }

    #[test]
    fn test_lost_frame_detection() {
        let mut rm = RedundancyManager::new();
        assert!(rm.accept(0, Network::A));
        assert!(rm.accept(3, Network::A)); // seq 1 and 2 lost
        assert_eq!(rm.lost_count(), 2);
    }

    #[test]
    fn test_sequence_wrapping() {
        let mut rm = RedundancyManager::new();
        assert!(rm.accept(254, Network::A));
        assert!(rm.accept(255, Network::A));
        assert!(rm.accept(0, Network::B)); // wraps
        assert_eq!(rm.lost_count(), 0);
    }

    #[test]
    fn test_dual_network_failover() {
        let mut rm = RedundancyManager::new();
        // Normal dual
        assert!(rm.accept(0, Network::A));
        assert!(!rm.accept(0, Network::B));
        // Network B only
        assert!(rm.accept(1, Network::B));
        assert_eq!(rm.last_accepted_network(), Some(Network::B));
    }

    #[test]
    fn test_integrity_failure_counters() {
        let mut rm = RedundancyManager::new();
        rm.record_integrity_failure(Network::A);
        rm.record_integrity_failure(Network::A);
        rm.record_integrity_failure(Network::B);
        assert_eq!(rm.integrity_failures(Network::A), 2);
        assert_eq!(rm.integrity_failures(Network::B), 1);
    }

    #[test]
    fn test_duplicate_same_network() {
        let mut rm = RedundancyManager::new();
        assert!(rm.accept(5, Network::A));
        // Same seq from same network is still a duplicate
        assert!(!rm.accept(5, Network::A));
        assert_eq!(rm.duplicate_count(), 1);
    }
}
