// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AFDX frame filtering (VL ID + destination MAC matching).
//!
//! Implements receive-side filtering for AFDX end systems. Each end system
//! is configured with a set of expected Virtual Links; frames with VL IDs
//! not in the filter table are dropped. Destination MAC address is also
//! validated against the AFDX multicast convention.

use alloc::vec::Vec;
use crate::BusError;
use crate::arinc664::frame::AfdxFrame;

/// Maximum number of VL filter entries.
pub const MAX_FILTER_ENTRIES: usize = 256;

/// Result of filtering an AFDX frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterResult {
    /// Frame accepted (VL ID and MAC match).
    Accept,
    /// Frame rejected — VL ID not in filter table.
    RejectUnknownVl,
    /// Frame rejected — destination MAC does not match AFDX convention.
    RejectBadMac,
    /// Frame rejected — frame exceeds Lmax for this VL.
    RejectOversized,
}

/// A single VL filter entry.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VlFilterEntry {
    /// Expected VL ID.
    vl_id: u16,
    /// Expected destination MAC (derived from VL ID).
    dest_mac: [u8; 6],
    /// Maximum frame payload size (Lmax) for this VL.
    lmax: u16,
}

/// AFDX frame filter.
///
/// Filters incoming frames by VL ID and destination MAC address.
/// Only frames matching a registered VL are accepted.
#[derive(Debug)]
pub struct AfdxFilter {
    /// Registered VL filter entries.
    entries: Vec<VlFilterEntry>,
    /// Statistics: accepted frames.
    accepted: u64,
    /// Statistics: rejected frames.
    rejected: u64,
}

impl Default for AfdxFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl AfdxFilter {
    /// Create a new empty AFDX frame filter.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            accepted: 0,
            rejected: 0,
        }
    }

    /// Register a VL for acceptance.
    ///
    /// The destination MAC is derived from the VL ID per AFDX convention.
    pub fn add_vl(&mut self, vl_id: u16, lmax: u16) -> Result<(), BusError> {
        if self.entries.len() >= MAX_FILTER_ENTRIES {
            return Err(BusError::FilterTableFull);
        }
        // Check for duplicate
        if self.entries.iter().any(|e| e.vl_id == vl_id) {
            return Ok(()); // Already registered
        }
        let dest_mac = [
            0x03,
            0x00,
            0x00,
            0x00,
            (vl_id >> 8) as u8,
            vl_id as u8,
        ];
        self.entries.push(VlFilterEntry {
            vl_id,
            dest_mac,
            lmax,
        });
        Ok(())
    }

    /// Remove a VL from the filter table.
    pub fn remove_vl(&mut self, vl_id: u16) {
        self.entries.retain(|e| e.vl_id != vl_id);
    }

    /// Returns the number of registered VLs.
    pub fn vl_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the number of accepted frames.
    pub fn accepted(&self) -> u64 {
        self.accepted
    }

    /// Returns the number of rejected frames.
    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    /// Filter an incoming AFDX frame.
    ///
    /// Returns [`FilterResult::Accept`] if the frame matches a registered VL
    /// and the destination MAC is correct. Otherwise returns the rejection reason.
    pub fn filter(&mut self, frame: &AfdxFrame) -> FilterResult {
        // Check destination MAC prefix (must be AFDX multicast)
        if frame.dest_mac[0] != 0x03
            || frame.dest_mac[1] != 0x00
            || frame.dest_mac[2] != 0x00
            || frame.dest_mac[3] != 0x00
        {
            self.rejected += 1;
            return FilterResult::RejectBadMac;
        }

        let vl_id = frame.vl_id();

        // Look up VL in filter table
        let entry = self.entries.iter().find(|e| e.vl_id == vl_id);
        match entry {
            None => {
                self.rejected += 1;
                FilterResult::RejectUnknownVl
            }
            Some(e) => {
                // Verify destination MAC matches exactly
                if frame.dest_mac != e.dest_mac {
                    self.rejected += 1;
                    return FilterResult::RejectBadMac;
                }
                // Check Lmax (payload + trailer must fit)
                if frame.payload.len() + 1 > e.lmax as usize {
                    self.rejected += 1;
                    return FilterResult::RejectOversized;
                }
                self.accepted += 1;
                FilterResult::Accept
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arinc664::frame::AfdxFrame;

    fn make_frame(vl_id: u16, payload_len: usize) -> AfdxFrame {
        let dest_mac = [0x03, 0x00, 0x00, 0x00, (vl_id >> 8) as u8, vl_id as u8];
        let src_mac = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
        let payload = alloc::vec![0xAA; payload_len];
        AfdxFrame::new(dest_mac, src_mac, 0x0800, &payload, 0, 1518).unwrap()
    }

    #[test]
    fn test_filter_new() {
        let filter = AfdxFilter::new();
        assert_eq!(filter.vl_count(), 0);
        assert_eq!(filter.accepted(), 0);
        assert_eq!(filter.rejected(), 0);
    }

    #[test]
    fn test_filter_add_vl() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(1001, 512).unwrap();
        assert_eq!(filter.vl_count(), 1);
    }

    #[test]
    fn test_filter_add_duplicate_vl() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(1001, 512).unwrap();
        filter.add_vl(1001, 512).unwrap(); // Should not duplicate
        assert_eq!(filter.vl_count(), 1);
    }

    #[test]
    fn test_filter_remove_vl() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(1001, 512).unwrap();
        filter.add_vl(1002, 512).unwrap();
        filter.remove_vl(1001);
        assert_eq!(filter.vl_count(), 1);
    }

    #[test]
    fn test_filter_accept() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(1001, 512).unwrap();
        let frame = make_frame(1001, 100);
        assert_eq!(filter.filter(&frame), FilterResult::Accept);
        assert_eq!(filter.accepted(), 1);
        assert_eq!(filter.rejected(), 0);
    }

    #[test]
    fn test_filter_reject_unknown_vl() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(1001, 512).unwrap();
        let frame = make_frame(9999, 100);
        assert_eq!(filter.filter(&frame), FilterResult::RejectUnknownVl);
        assert_eq!(filter.rejected(), 1);
    }

    #[test]
    fn test_filter_reject_bad_mac() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(1001, 512).unwrap();
        // Create a frame with non-AFDX MAC prefix
        let frame = AfdxFrame {
            dest_mac: [0x01, 0x00, 0x00, 0x00, 0x03, 0xE9],
            src_mac: [0x02, 0x00, 0x00, 0x00, 0x00, 0x01],
            ether_type: 0x0800,
            payload: alloc::vec![0xAA; 10],
            trailer: crate::arinc664::frame::AfdxTrailer { sequence_number: 0 },
        };
        assert_eq!(filter.filter(&frame), FilterResult::RejectBadMac);
    }

    #[test]
    fn test_filter_reject_oversized() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(1001, 64).unwrap(); // Small Lmax
        let frame = make_frame(1001, 100); // Payload too large
        assert_eq!(filter.filter(&frame), FilterResult::RejectOversized);
    }

    #[test]
    fn test_filter_multiple_vls() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(100, 512).unwrap();
        filter.add_vl(200, 512).unwrap();
        filter.add_vl(300, 512).unwrap();

        assert_eq!(filter.filter(&make_frame(100, 10)), FilterResult::Accept);
        assert_eq!(filter.filter(&make_frame(200, 10)), FilterResult::Accept);
        assert_eq!(filter.filter(&make_frame(300, 10)), FilterResult::Accept);
        assert_eq!(filter.filter(&make_frame(400, 10)), FilterResult::RejectUnknownVl);
        assert_eq!(filter.accepted(), 3);
        assert_eq!(filter.rejected(), 1);
    }

    #[test]
    fn test_filter_table_full() {
        let mut filter = AfdxFilter::new();
        for i in 0..MAX_FILTER_ENTRIES as u16 {
            filter.add_vl(i, 512).unwrap();
        }
        assert_eq!(
            filter.add_vl(MAX_FILTER_ENTRIES as u16, 512).err(),
            Some(BusError::FilterTableFull)
        );
    }

    #[test]
    fn test_filter_exact_lmax() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(1001, 101).unwrap(); // Lmax = 101 (100 payload + 1 trailer)
        let frame = make_frame(1001, 100); // Exactly at limit
        assert_eq!(filter.filter(&frame), FilterResult::Accept);
    }

    #[test]
    fn test_filter_stats_accumulate() {
        let mut filter = AfdxFilter::new();
        filter.add_vl(1001, 512).unwrap();
        for _ in 0..5 {
            filter.filter(&make_frame(1001, 10));
        }
        for _ in 0..3 {
            filter.filter(&make_frame(9999, 10));
        }
        assert_eq!(filter.accepted(), 5);
        assert_eq!(filter.rejected(), 3);
    }
}
