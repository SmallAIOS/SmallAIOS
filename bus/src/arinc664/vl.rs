// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AFDX Virtual Link configuration.
//!
//! A Virtual Link (VL) defines a unidirectional logical connection with
//! guaranteed bandwidth (BAG interval) and maximum frame size (Lmax).

use crate::BusError;

/// Maximum number of sub-VLs per Virtual Link.
pub const MAX_SUB_VLS: usize = 4;

/// Valid BAG intervals in milliseconds (powers of 2, 1..=128).
const VALID_BAG_MS: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// AFDX Virtual Link configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualLink {
    /// Virtual Link identifier (used for MAC derivation).
    vl_id: u16,
    /// Bandwidth Allocation Gap in microseconds.
    bag_us: u64,
    /// Maximum frame size in bytes (Ethernet payload, excluding FCS).
    lmax: u16,
    /// Number of sub-VLs (1..=MAX_SUB_VLS).
    num_sub_vls: u8,
    /// Whether dual-network redundancy is enabled.
    redundant: bool,
}

impl VirtualLink {
    /// Create a new Virtual Link configuration.
    ///
    /// `bag_ms` must be a power of 2 in range 1..=128.
    /// `lmax` must be in range 64..=1518 (Ethernet frame limits).
    pub fn new(
        vl_id: u16,
        bag_ms: u16,
        lmax: u16,
        num_sub_vls: u8,
        redundant: bool,
    ) -> Result<Self, BusError> {
        if !VALID_BAG_MS.contains(&bag_ms) {
            return Err(BusError::OutOfRange);
        }
        if !(64..=1518).contains(&lmax) {
            return Err(BusError::OutOfRange);
        }
        if num_sub_vls == 0 || num_sub_vls as usize > MAX_SUB_VLS {
            return Err(BusError::OutOfRange);
        }
        Ok(Self {
            vl_id,
            bag_us: bag_ms as u64 * 1000,
            lmax,
            num_sub_vls,
            redundant,
        })
    }

    /// Returns the VL identifier.
    pub fn vl_id(&self) -> u16 {
        self.vl_id
    }

    /// Returns the BAG interval in microseconds.
    pub fn bag_us(&self) -> u64 {
        self.bag_us
    }

    /// Returns the maximum frame size.
    pub fn lmax(&self) -> u16 {
        self.lmax
    }

    /// Returns the number of sub-VLs.
    pub fn num_sub_vls(&self) -> u8 {
        self.num_sub_vls
    }

    /// Returns whether dual-network redundancy is enabled.
    pub fn is_redundant(&self) -> bool {
        self.redundant
    }

    /// Derive the AFDX multicast destination MAC address from the VL ID.
    ///
    /// Per AFDX convention: `03:00:00:00:xx:xx` where xx:xx is the VL ID.
    pub fn dest_mac(&self) -> [u8; 6] {
        [
            0x03,
            0x00,
            0x00,
            0x00,
            (self.vl_id >> 8) as u8,
            self.vl_id as u8,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vl_create_valid() {
        let vl = VirtualLink::new(1001, 32, 512, 1, true).unwrap();
        assert_eq!(vl.vl_id(), 1001);
        assert_eq!(vl.bag_us(), 32_000);
        assert_eq!(vl.lmax(), 512);
        assert_eq!(vl.num_sub_vls(), 1);
        assert!(vl.is_redundant());
    }

    #[test]
    fn test_vl_invalid_bag() {
        assert_eq!(VirtualLink::new(1, 3, 512, 1, false), Err(BusError::OutOfRange));
        assert_eq!(VirtualLink::new(1, 0, 512, 1, false), Err(BusError::OutOfRange));
        assert_eq!(VirtualLink::new(1, 256, 512, 1, false), Err(BusError::OutOfRange));
    }

    #[test]
    fn test_vl_invalid_lmax() {
        assert_eq!(VirtualLink::new(1, 32, 63, 1, false), Err(BusError::OutOfRange));
        assert_eq!(VirtualLink::new(1, 32, 1519, 1, false), Err(BusError::OutOfRange));
    }

    #[test]
    fn test_vl_invalid_sub_vls() {
        assert_eq!(VirtualLink::new(1, 32, 512, 0, false), Err(BusError::OutOfRange));
        assert_eq!(VirtualLink::new(1, 32, 512, 5, false), Err(BusError::OutOfRange));
    }

    #[test]
    fn test_vl_all_valid_bags() {
        for &bag in &[1, 2, 4, 8, 16, 32, 64, 128] {
            assert!(VirtualLink::new(1, bag, 512, 1, false).is_ok());
        }
    }

    #[test]
    fn test_dest_mac_derivation() {
        let vl = VirtualLink::new(0x03E9, 32, 512, 1, false).unwrap(); // 1001 = 0x03E9
        assert_eq!(vl.dest_mac(), [0x03, 0x00, 0x00, 0x00, 0x03, 0xE9]);
    }

    #[test]
    fn test_dest_mac_zero_vl() {
        let vl = VirtualLink::new(0, 1, 64, 1, false).unwrap();
        assert_eq!(vl.dest_mac(), [0x03, 0x00, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_vl_multiple_sub_vls() {
        let vl = VirtualLink::new(100, 16, 256, 4, true).unwrap();
        assert_eq!(vl.num_sub_vls(), 4);
    }
}
