// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC packet number spaces (RFC 9000 Section 17.2.2).
//!
//! QUIC uses three distinct packet number spaces:
//! - Initial: for Initial packets during handshake
//! - Handshake: for Handshake packets during handshake
//! - Application Data: for 0-RTT and 1-RTT packets
//!
//! Each space has independent packet number tracking and ACK state.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// Packet number space identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PacketNumberSpace {
    /// Initial encryption level.
    Initial,
    /// Handshake encryption level.
    Handshake,
    /// Application data (0-RTT and 1-RTT).
    ApplicationData,
}

/// Sent packet metadata for loss detection.
#[derive(Debug, Clone)]
pub struct SentPacket {
    /// Packet number.
    pub pn: u64,
    /// Time sent (microseconds since connection start).
    pub time_sent_us: u64,
    /// Whether this packet is ACK-eliciting.
    pub ack_eliciting: bool,
    /// Whether this packet is in-flight (counts toward bytes in flight).
    pub in_flight: bool,
    /// Size in bytes (for congestion control).
    pub sent_bytes: usize,
}

/// Per-space packet number tracking and ACK state.
#[derive(Debug)]
pub struct PacketNumberSpaceState {
    /// Which space this is.
    pub space: PacketNumberSpace,
    /// Next packet number to send.
    next_pn: u64,
    /// Largest packet number received (for ACK generation).
    largest_received_pn: Option<u64>,
    /// Time the largest PN was received (microseconds).
    largest_received_time_us: u64,
    /// Set of received packet numbers (for ACK ranges).
    received_pns: BTreeSet<u64>,
    /// Whether we need to send an ACK.
    ack_needed: bool,
    /// Sent but not yet acknowledged packets.
    sent_packets: Vec<SentPacket>,
    /// Largest acknowledged packet number.
    largest_acked_pn: Option<u64>,
    /// Loss time (earliest time a packet is considered lost).
    loss_time_us: Option<u64>,
}

/// Maximum number of received PNs to track for ACK generation.
const MAX_RECEIVED_PNS: usize = 256;

impl PacketNumberSpaceState {
    /// Create a new packet number space.
    pub fn new(space: PacketNumberSpace) -> Self {
        Self {
            space,
            next_pn: 0,
            largest_received_pn: None,
            largest_received_time_us: 0,
            received_pns: BTreeSet::new(),
            ack_needed: false,
            sent_packets: Vec::new(),
            largest_acked_pn: None,
            loss_time_us: None,
        }
    }

    /// Allocate the next packet number for sending.
    pub fn allocate_pn(&mut self) -> u64 {
        let pn = self.next_pn;
        self.next_pn += 1;
        pn
    }

    /// Get the next packet number without allocating.
    pub fn next_pn(&self) -> u64 {
        self.next_pn
    }

    /// Get the largest acknowledged packet number for encoding.
    pub fn largest_acked(&self) -> u64 {
        self.largest_acked_pn.unwrap_or(0)
    }

    /// Record a sent packet for loss detection.
    pub fn on_packet_sent(&mut self, packet: SentPacket) {
        self.sent_packets.push(packet);
    }

    /// Record a received packet number.
    pub fn on_packet_received(&mut self, pn: u64, time_us: u64) {
        if self.largest_received_pn.is_none_or(|lrp| pn > lrp) {
            self.largest_received_pn = Some(pn);
            self.largest_received_time_us = time_us;
        }
        self.received_pns.insert(pn);
        // Trim old entries to bound memory
        while self.received_pns.len() > MAX_RECEIVED_PNS {
            if let Some(&oldest) = self.received_pns.iter().next() {
                self.received_pns.remove(&oldest);
            }
        }
        self.ack_needed = true;
    }

    /// Check if an ACK needs to be sent.
    pub fn ack_needed(&self) -> bool {
        self.ack_needed
    }

    /// Clear the ACK-needed flag after sending an ACK.
    pub fn clear_ack_needed(&mut self) {
        self.ack_needed = false;
    }

    /// Get the largest received packet number.
    pub fn largest_received(&self) -> Option<u64> {
        self.largest_received_pn
    }

    /// Generate ACK ranges from received packet numbers.
    ///
    /// Returns (largest_acked, ranges) where ranges are (gap, ack_range) pairs.
    /// The first range is the count of consecutive PNs from largest_acked down.
    pub fn generate_ack_ranges(&self) -> Option<(u64, Vec<(u64, u64)>)> {
        let largest = self.largest_received_pn?;
        if self.received_pns.is_empty() {
            return None;
        }

        let mut ranges = Vec::new();
        let mut current = largest;
        let mut first_range = true;

        // Walk backwards through received PNs to build ranges
        let pns: Vec<u64> = self.received_pns.iter().copied().rev().collect();
        let mut i = 0;

        while i < pns.len() {
            let range_start = pns[i];
            let mut range_end = range_start;

            // Find consecutive run
            while i + 1 < pns.len() && pns[i + 1] + 1 == pns[i] {
                range_end = pns[i + 1];
                i += 1;
            }

            if first_range {
                // First ACK range: count of packets from largest
                ranges.push((0, range_start - range_end));
                current = range_end;
                first_range = false;
            } else {
                // Gap then range
                let gap = current - range_start - 1;
                ranges.push((gap, range_start - range_end));
                current = range_end;
            }
            i += 1;
        }

        Some((largest, ranges))
    }

    /// Process an ACK for packets we sent. Returns newly acked packet numbers.
    pub fn on_ack_received(&mut self, largest_acked: u64, acked_ranges: &[(u64, u64)]) -> Vec<SentPacket> {
        if self.largest_acked_pn.is_none_or(|la| largest_acked > la) {
            self.largest_acked_pn = Some(largest_acked);
        }

        // Build set of acked PNs from ranges
        let mut acked_set = BTreeSet::new();
        let mut current = largest_acked;
        for (i, &(gap, range_len)) in acked_ranges.iter().enumerate() {
            if i == 0 {
                // First range: largest_acked down by range_len
                for pn in (current - range_len)..=current {
                    acked_set.insert(pn);
                }
                current = current.saturating_sub(range_len + 1);
            } else {
                // Skip gap, then ack range
                current = current.saturating_sub(gap);
                let range_start = current;
                for pn in (range_start - range_len)..=range_start {
                    acked_set.insert(pn);
                }
                current = range_start.saturating_sub(range_len + 1);
            }
        }

        // Remove acked packets from sent list
        let mut acked = Vec::new();
        self.sent_packets.retain(|p| {
            if acked_set.contains(&p.pn) {
                acked.push(p.clone());
                false
            } else {
                true
            }
        });

        acked
    }

    /// Get sent packets that may be lost (older than threshold).
    pub fn detect_lost_packets(&mut self, loss_delay_us: u64, now_us: u64) -> Vec<SentPacket> {
        let largest_acked = match self.largest_acked_pn {
            Some(la) => la,
            None => return Vec::new(),
        };

        let mut lost = Vec::new();
        let mut earliest_loss_time = None;

        self.sent_packets.retain(|p| {
            if p.pn <= largest_acked {
                // Packet is old enough — check time-based loss
                if p.time_sent_us + loss_delay_us <= now_us {
                    lost.push(p.clone());
                    return false;
                }
                // Or packet-number-based loss (3 packets ahead)
                if largest_acked >= p.pn + 3 {
                    lost.push(p.clone());
                    return false;
                }
                // Track earliest potential loss time
                let loss_time = p.time_sent_us + loss_delay_us;
                if earliest_loss_time.is_none_or(|t| loss_time < t) {
                    earliest_loss_time = Some(loss_time);
                }
            }
            true
        });

        self.loss_time_us = earliest_loss_time;
        lost
    }

    /// Get the earliest loss time for this space.
    pub fn loss_time(&self) -> Option<u64> {
        self.loss_time_us
    }

    /// Number of unacknowledged sent packets.
    pub fn unacked_count(&self) -> usize {
        self.sent_packets.len()
    }

    /// Discard all state (used when keys are discarded).
    pub fn discard(&mut self) {
        self.sent_packets.clear();
        self.received_pns.clear();
        self.ack_needed = false;
        self.loss_time_us = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_variants() {
        assert_ne!(PacketNumberSpace::Initial, PacketNumberSpace::Handshake);
        assert_ne!(PacketNumberSpace::Handshake, PacketNumberSpace::ApplicationData);
    }

    #[test]
    fn test_allocate_pn() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::Initial);
        assert_eq!(state.allocate_pn(), 0);
        assert_eq!(state.allocate_pn(), 1);
        assert_eq!(state.allocate_pn(), 2);
        assert_eq!(state.next_pn(), 3);
    }

    #[test]
    fn test_on_packet_received() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::ApplicationData);
        state.on_packet_received(5, 1000);
        assert_eq!(state.largest_received(), Some(5));
        assert!(state.ack_needed());
    }

    #[test]
    fn test_on_packet_received_ordering() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::ApplicationData);
        state.on_packet_received(5, 1000);
        state.on_packet_received(3, 2000);
        // Largest should still be 5
        assert_eq!(state.largest_received(), Some(5));
    }

    #[test]
    fn test_clear_ack_needed() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::Initial);
        state.on_packet_received(0, 100);
        assert!(state.ack_needed());
        state.clear_ack_needed();
        assert!(!state.ack_needed());
    }

    #[test]
    fn test_on_packet_sent_and_ack() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::ApplicationData);
        state.on_packet_sent(SentPacket {
            pn: 0,
            time_sent_us: 100,
            ack_eliciting: true,
            in_flight: true,
            sent_bytes: 1200,
        });
        state.on_packet_sent(SentPacket {
            pn: 1,
            time_sent_us: 200,
            ack_eliciting: true,
            in_flight: true,
            sent_bytes: 1200,
        });
        assert_eq!(state.unacked_count(), 2);

        // ACK packet 1
        let acked = state.on_ack_received(1, &[(0, 0)]);
        assert_eq!(acked.len(), 1);
        assert_eq!(acked[0].pn, 1);
        assert_eq!(state.unacked_count(), 1);
    }

    #[test]
    fn test_on_ack_range() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::ApplicationData);
        for pn in 0..5 {
            state.on_packet_sent(SentPacket {
                pn,
                time_sent_us: pn * 100,
                ack_eliciting: true,
                in_flight: true,
                sent_bytes: 100,
            });
        }

        // ACK packets 3-4 (largest=4, first_range=1 meaning 2 packets: 4,3)
        let acked = state.on_ack_received(4, &[(0, 1)]);
        assert_eq!(acked.len(), 2);
        assert_eq!(state.unacked_count(), 3);
    }

    #[test]
    fn test_detect_lost_packets_by_pn() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::ApplicationData);
        for pn in 0..5 {
            state.on_packet_sent(SentPacket {
                pn,
                time_sent_us: pn * 100,
                ack_eliciting: true,
                in_flight: true,
                sent_bytes: 100,
            });
        }

        // ACK packet 4
        state.on_ack_received(4, &[(0, 0)]);

        // Packet 0 and 1 are more than 3 packets behind largest_acked=4
        let lost = state.detect_lost_packets(100_000, 500);
        assert_eq!(lost.len(), 2); // pn 0 and 1 (4-0>=3, 4-1>=3)
        assert_eq!(state.unacked_count(), 2); // pn 2 and 3 remain
    }

    #[test]
    fn test_detect_lost_packets_by_time() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::ApplicationData);
        state.on_packet_sent(SentPacket {
            pn: 0,
            time_sent_us: 100,
            ack_eliciting: true,
            in_flight: true,
            sent_bytes: 100,
        });
        state.on_packet_sent(SentPacket {
            pn: 1,
            time_sent_us: 500,
            ack_eliciting: true,
            in_flight: true,
            sent_bytes: 100,
        });

        state.on_ack_received(1, &[(0, 0)]);

        // With loss_delay=300, packet 0 (sent at 100) is lost at time 500
        let lost = state.detect_lost_packets(300, 500);
        assert_eq!(lost.len(), 1);
        assert_eq!(lost[0].pn, 0);
    }

    #[test]
    fn test_generate_ack_ranges_single() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::Initial);
        state.on_packet_received(0, 100);
        state.on_packet_received(1, 200);
        state.on_packet_received(2, 300);

        let (largest, ranges) = state.generate_ack_ranges().unwrap();
        assert_eq!(largest, 2);
        assert_eq!(ranges.len(), 1);
        // Single range: 0 gap, 2 length (packets 0,1,2)
        assert_eq!(ranges[0], (0, 2));
    }

    #[test]
    fn test_generate_ack_ranges_with_gap() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::ApplicationData);
        state.on_packet_received(0, 100);
        state.on_packet_received(1, 200);
        // Gap: 2 missing
        state.on_packet_received(3, 400);
        state.on_packet_received(4, 500);

        let (largest, ranges) = state.generate_ack_ranges().unwrap();
        assert_eq!(largest, 4);
        assert!(ranges.len() >= 2);
    }

    #[test]
    fn test_discard() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::Handshake);
        state.on_packet_received(0, 100);
        state.on_packet_sent(SentPacket {
            pn: 0,
            time_sent_us: 100,
            ack_eliciting: true,
            in_flight: true,
            sent_bytes: 100,
        });
        state.discard();
        assert!(!state.ack_needed());
        assert_eq!(state.unacked_count(), 0);
    }

    #[test]
    fn test_largest_acked_default() {
        let state = PacketNumberSpaceState::new(PacketNumberSpace::Initial);
        assert_eq!(state.largest_acked(), 0);
    }

    #[test]
    fn test_received_pn_trimming() {
        let mut state = PacketNumberSpaceState::new(PacketNumberSpace::ApplicationData);
        for pn in 0..300 {
            state.on_packet_received(pn, pn * 10);
        }
        // Should be trimmed to MAX_RECEIVED_PNS
        assert!(state.received_pns.len() <= MAX_RECEIVED_PNS);
        // Largest should still be tracked
        assert_eq!(state.largest_received(), Some(299));
    }
}
