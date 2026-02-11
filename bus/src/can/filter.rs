// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CAN acceptance filtering per ISO 11898.
//!
//! Two-stage filtering:
//! 1. **Hardware mask filters**: `(frame_id & mask) == match_value`
//! 2. **Software filters**: exact ID match with optional callback routing
//!
//! A frame must pass at least one hardware filter AND at least one software
//! filter to be accepted. If no software filters are configured, all
//! hardware-accepted frames are delivered.

use alloc::vec::Vec;

use crate::BusError;

/// Maximum number of hardware mask filters (typical CAN controller limit).
pub const MAX_HW_FILTERS: usize = 16;

/// Maximum number of software filters.
pub const MAX_SW_FILTERS: usize = 64;

/// A hardware mask-based acceptance filter.
///
/// Matches when `(frame_id & mask) == match_value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HwFilter {
    /// Mask applied to the incoming frame ID.
    pub mask: u32,
    /// Expected value after masking.
    pub match_value: u32,
}

impl HwFilter {
    /// Create a new hardware filter.
    pub fn new(mask: u32, match_value: u32) -> Self {
        HwFilter { mask, match_value }
    }

    /// Check if a frame ID matches this filter.
    pub fn matches(&self, frame_id: u32) -> bool {
        (frame_id & self.mask) == self.match_value
    }

    /// Create a filter that matches a single exact ID.
    pub fn exact(id: u32) -> Self {
        HwFilter {
            mask: u32::MAX,
            match_value: id,
        }
    }

    /// Create a filter that accepts all frames.
    pub fn accept_all() -> Self {
        HwFilter {
            mask: 0,
            match_value: 0,
        }
    }
}

/// A software acceptance filter matching exact frame IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwFilter {
    /// The exact frame ID to match.
    pub id: u32,
}

impl SwFilter {
    /// Create a new software filter for the given ID.
    pub fn new(id: u32) -> Self {
        SwFilter { id }
    }

    /// Check if a frame ID matches this filter.
    pub fn matches(&self, frame_id: u32) -> bool {
        self.id == frame_id
    }
}

/// Two-stage CAN acceptance filter chain.
///
/// A frame is accepted if it matches at least one hardware filter AND
/// (either no software filters are configured OR it matches at least one
/// software filter).
#[derive(Debug, Clone)]
pub struct AcceptanceFilter {
    hw_filters: Vec<HwFilter>,
    sw_filters: Vec<SwFilter>,
}

impl AcceptanceFilter {
    /// Create a new acceptance filter with no filters configured.
    ///
    /// With no filters, all frames are rejected.
    pub fn new() -> Self {
        AcceptanceFilter {
            hw_filters: Vec::new(),
            sw_filters: Vec::new(),
        }
    }

    /// Add a hardware mask filter.
    ///
    /// Returns [`BusError::FilterTableFull`] if the hardware filter table is full.
    pub fn add_hw_filter(&mut self, filter: HwFilter) -> Result<(), BusError> {
        if self.hw_filters.len() >= MAX_HW_FILTERS {
            return Err(BusError::FilterTableFull);
        }
        self.hw_filters.push(filter);
        Ok(())
    }

    /// Remove a hardware filter by index.
    ///
    /// Returns `false` if the index is out of bounds.
    pub fn remove_hw_filter(&mut self, index: usize) -> bool {
        if index < self.hw_filters.len() {
            self.hw_filters.remove(index);
            true
        } else {
            false
        }
    }

    /// Clear all hardware filters.
    pub fn clear_hw_filters(&mut self) {
        self.hw_filters.clear();
    }

    /// Add a software filter for an exact ID.
    ///
    /// Returns [`BusError::FilterTableFull`] if the software filter table is full.
    pub fn add_sw_filter(&mut self, filter: SwFilter) -> Result<(), BusError> {
        if self.sw_filters.len() >= MAX_SW_FILTERS {
            return Err(BusError::FilterTableFull);
        }
        self.sw_filters.push(filter);
        Ok(())
    }

    /// Remove a software filter by index.
    ///
    /// Returns `false` if the index is out of bounds.
    pub fn remove_sw_filter(&mut self, index: usize) -> bool {
        if index < self.sw_filters.len() {
            self.sw_filters.remove(index);
            true
        } else {
            false
        }
    }

    /// Clear all software filters.
    pub fn clear_sw_filters(&mut self) {
        self.sw_filters.clear();
    }

    /// Returns the number of hardware filters currently configured.
    pub fn hw_filter_count(&self) -> usize {
        self.hw_filters.len()
    }

    /// Returns the number of software filters currently configured.
    pub fn sw_filter_count(&self) -> usize {
        self.sw_filters.len()
    }

    /// Test whether a frame ID is accepted by this filter chain.
    ///
    /// A frame is accepted if:
    /// 1. It matches at least one hardware filter, AND
    /// 2. Either no software filters are configured, or it matches at
    ///    least one software filter.
    pub fn accepts(&self, frame_id: u32) -> bool {
        // Stage 1: Hardware filter
        let hw_pass = self.hw_filters.iter().any(|f| f.matches(frame_id));
        if !hw_pass {
            return false;
        }

        // Stage 2: Software filter (pass-through if none configured)
        if self.sw_filters.is_empty() {
            return true;
        }
        self.sw_filters.iter().any(|f| f.matches(frame_id))
    }
}

impl Default for AcceptanceFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // HwFilter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hw_filter_exact_match() {
        let filter = HwFilter::exact(0x123);
        assert!(filter.matches(0x123));
        assert!(!filter.matches(0x124));
        assert!(!filter.matches(0x000));
    }

    #[test]
    fn test_hw_filter_mask_match() {
        // Match IDs 0x100-0x1FF (mask top bits)
        let filter = HwFilter::new(0x700, 0x100);
        assert!(filter.matches(0x100));
        assert!(filter.matches(0x1FF));
        assert!(filter.matches(0x150));
        assert!(!filter.matches(0x200));
        assert!(!filter.matches(0x000));
    }

    #[test]
    fn test_hw_filter_accept_all() {
        let filter = HwFilter::accept_all();
        assert!(filter.matches(0x000));
        assert!(filter.matches(0x7FF));
        assert!(filter.matches(0x1FFF_FFFF));
    }

    // -----------------------------------------------------------------------
    // SwFilter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sw_filter_match() {
        let filter = SwFilter::new(0x100);
        assert!(filter.matches(0x100));
        assert!(!filter.matches(0x101));
    }

    // -----------------------------------------------------------------------
    // AcceptanceFilter chain tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_filters_rejects_all() {
        let af = AcceptanceFilter::new();
        assert!(!af.accepts(0x100));
        assert!(!af.accepts(0x000));
    }

    #[test]
    fn test_hw_only_accepts_matching() {
        let mut af = AcceptanceFilter::new();
        af.add_hw_filter(HwFilter::exact(0x100)).unwrap();
        assert!(af.accepts(0x100));
        assert!(!af.accepts(0x200));
    }

    #[test]
    fn test_hw_and_sw_both_must_match() {
        let mut af = AcceptanceFilter::new();
        af.add_hw_filter(HwFilter::new(0x700, 0x100)).unwrap();
        af.add_sw_filter(SwFilter::new(0x123)).unwrap();

        // Passes HW filter (0x123 & 0x700 == 0x100) AND SW filter (exact 0x123)
        assert!(af.accepts(0x123));
        // Passes HW filter but not SW filter
        assert!(!af.accepts(0x150));
        // Fails HW filter
        assert!(!af.accepts(0x200));
    }

    #[test]
    fn test_multiple_hw_filters_or() {
        let mut af = AcceptanceFilter::new();
        af.add_hw_filter(HwFilter::exact(0x100)).unwrap();
        af.add_hw_filter(HwFilter::exact(0x200)).unwrap();
        assert!(af.accepts(0x100));
        assert!(af.accepts(0x200));
        assert!(!af.accepts(0x300));
    }

    #[test]
    fn test_multiple_sw_filters_or() {
        let mut af = AcceptanceFilter::new();
        af.add_hw_filter(HwFilter::accept_all()).unwrap();
        af.add_sw_filter(SwFilter::new(0x100)).unwrap();
        af.add_sw_filter(SwFilter::new(0x200)).unwrap();
        assert!(af.accepts(0x100));
        assert!(af.accepts(0x200));
        assert!(!af.accepts(0x300));
    }

    #[test]
    fn test_hw_filter_table_full() {
        let mut af = AcceptanceFilter::new();
        for i in 0..MAX_HW_FILTERS {
            af.add_hw_filter(HwFilter::exact(i as u32)).unwrap();
        }
        assert_eq!(
            af.add_hw_filter(HwFilter::exact(0xFFF)),
            Err(BusError::FilterTableFull)
        );
    }

    #[test]
    fn test_sw_filter_table_full() {
        let mut af = AcceptanceFilter::new();
        for i in 0..MAX_SW_FILTERS {
            af.add_sw_filter(SwFilter::new(i as u32)).unwrap();
        }
        assert_eq!(
            af.add_sw_filter(SwFilter::new(0xFFF)),
            Err(BusError::FilterTableFull)
        );
    }

    #[test]
    fn test_remove_hw_filter() {
        let mut af = AcceptanceFilter::new();
        af.add_hw_filter(HwFilter::exact(0x100)).unwrap();
        af.add_hw_filter(HwFilter::exact(0x200)).unwrap();
        assert_eq!(af.hw_filter_count(), 2);

        assert!(af.remove_hw_filter(0));
        assert_eq!(af.hw_filter_count(), 1);
        // 0x100 was removed, 0x200 remains
        assert!(!af.accepts(0x100));
        assert!(af.accepts(0x200));
    }

    #[test]
    fn test_remove_hw_filter_out_of_bounds() {
        let mut af = AcceptanceFilter::new();
        assert!(!af.remove_hw_filter(0));
    }

    #[test]
    fn test_remove_sw_filter() {
        let mut af = AcceptanceFilter::new();
        af.add_hw_filter(HwFilter::accept_all()).unwrap();
        af.add_sw_filter(SwFilter::new(0x100)).unwrap();
        af.add_sw_filter(SwFilter::new(0x200)).unwrap();

        assert!(af.remove_sw_filter(0));
        assert_eq!(af.sw_filter_count(), 1);
        assert!(!af.accepts(0x100));
        assert!(af.accepts(0x200));
    }

    #[test]
    fn test_remove_sw_filter_out_of_bounds() {
        let mut af = AcceptanceFilter::new();
        assert!(!af.remove_sw_filter(0));
    }

    #[test]
    fn test_clear_hw_filters() {
        let mut af = AcceptanceFilter::new();
        af.add_hw_filter(HwFilter::exact(0x100)).unwrap();
        af.clear_hw_filters();
        assert_eq!(af.hw_filter_count(), 0);
        assert!(!af.accepts(0x100));
    }

    #[test]
    fn test_clear_sw_filters() {
        let mut af = AcceptanceFilter::new();
        af.add_hw_filter(HwFilter::accept_all()).unwrap();
        af.add_sw_filter(SwFilter::new(0x100)).unwrap();
        af.clear_sw_filters();
        assert_eq!(af.sw_filter_count(), 0);
        // With no SW filters, all HW-matched frames pass
        assert!(af.accepts(0x100));
        assert!(af.accepts(0x200));
    }

    #[test]
    fn test_runtime_reconfiguration() {
        let mut af = AcceptanceFilter::new();
        af.add_hw_filter(HwFilter::accept_all()).unwrap();
        af.add_sw_filter(SwFilter::new(0x100)).unwrap();

        assert!(af.accepts(0x100));
        assert!(!af.accepts(0x200));

        // Add a new filter at runtime
        af.add_sw_filter(SwFilter::new(0x200)).unwrap();
        assert!(af.accepts(0x200));

        // Remove old filter
        af.remove_sw_filter(0);
        assert!(!af.accepts(0x100));
        assert!(af.accepts(0x200));
    }

    #[test]
    fn test_default() {
        let af = AcceptanceFilter::default();
        assert_eq!(af.hw_filter_count(), 0);
        assert_eq!(af.sw_filter_count(), 0);
    }
}
