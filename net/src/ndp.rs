// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Neighbor Discovery Protocol (NDP) for IPv6.
//!
//! Implements RFC 4861 message types, NDP options, and a neighbor cache
//! (`NeighborTable`) for mapping IPv6 addresses to link-layer (MAC)
//! addresses.

use alloc::vec::Vec;

use crate::ethernet::MacAddress;
use crate::ipv6::Ipv6Addr;
use crate::NetError;

// ---------------------------------------------------------------------------
// NDP message types (ICMPv6 type codes)
// ---------------------------------------------------------------------------

/// ICMPv6 type values used by NDP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NdpType {
    /// Router Solicitation.
    RouterSolicitation = 133,
    /// Router Advertisement.
    RouterAdvertisement = 134,
    /// Neighbor Solicitation.
    NeighborSolicitation = 135,
    /// Neighbor Advertisement.
    NeighborAdvertisement = 136,
    /// Redirect.
    Redirect = 137,
}

impl NdpType {
    /// Try to convert a raw `u8` into an [`NdpType`].
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            133 => Some(Self::RouterSolicitation),
            134 => Some(Self::RouterAdvertisement),
            135 => Some(Self::NeighborSolicitation),
            136 => Some(Self::NeighborAdvertisement),
            137 => Some(Self::Redirect),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// NDP options
// ---------------------------------------------------------------------------

/// NDP option carried inside an NDP message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NdpOption {
    /// Source Link-Layer Address (option type 1).
    SourceLinkLayerAddress(MacAddress),
    /// Target Link-Layer Address (option type 2).
    TargetLinkLayerAddress(MacAddress),
    /// Prefix Information (option type 3).
    PrefixInformation {
        /// Number of leading bits in the prefix that are valid.
        prefix_len: u8,
        /// 128-bit prefix (only `prefix_len` bits are significant).
        prefix: [u8; 16],
        /// Time (seconds) the prefix is valid for on-link determination.
        valid_lifetime: u32,
        /// Time (seconds) an address generated from this prefix remains
        /// preferred.
        preferred_lifetime: u32,
    },
    /// MTU (option type 5).
    Mtu(u32),
}

// ---------------------------------------------------------------------------
// Neighbor cache
// ---------------------------------------------------------------------------

/// Reachability state of a neighbor cache entry (RFC 4861 section 7.3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighborState {
    /// Address resolution is in progress; no link-layer address yet.
    Incomplete,
    /// Positive confirmation of reachability was received recently.
    Reachable,
    /// Reachability timer expired; traffic may still flow but we need
    /// reconfirmation on the next send.
    Stale,
    /// A packet was sent recently and we are waiting a short time before
    /// probing.
    Delay,
    /// Actively sending Neighbor Solicitation probes.
    Probe,
}

/// A single entry in the neighbor cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeighborEntry {
    /// IPv6 address of the neighbor.
    pub ip: Ipv6Addr,
    /// Link-layer (MAC) address of the neighbor.
    pub mac: MacAddress,
    /// Current reachability state.
    pub state: NeighborState,
    /// Monotonic timestamp (tick count) of the last state transition.
    pub timestamp: u64,
}

/// Bounded neighbor cache mapping IPv6 addresses to MAC addresses.
///
/// When the table is full, the oldest entry is evicted to make room.
pub struct NeighborTable {
    entries: Vec<NeighborEntry>,
}

impl Default for NeighborTable {
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborTable {
    /// Maximum number of entries the table will hold.
    pub const MAX_ENTRIES: usize = 256;

    /// Create an empty neighbor table.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Look up the MAC address for a given IPv6 address.
    pub fn lookup(&self, ip: &Ipv6Addr) -> Option<&MacAddress> {
        self.entries.iter().find(|e| e.ip == *ip).map(|e| &e.mac)
    }

    /// Insert or update a neighbor entry.
    ///
    /// If an entry with the same `ip` already exists it is updated in place.
    /// If the table is full (`MAX_ENTRIES`), the oldest entry (by timestamp)
    /// is evicted first.
    pub fn insert(&mut self, ip: Ipv6Addr, mac: MacAddress, state: NeighborState, timestamp: u64) {
        // Update existing entry if present.
        if let Some(entry) = self.entries.iter_mut().find(|e| e.ip == ip) {
            entry.mac = mac;
            entry.state = state;
            entry.timestamp = timestamp;
            return;
        }

        // Evict oldest when at capacity.
        if self.entries.len() >= Self::MAX_ENTRIES {
            if let Some((oldest_idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.timestamp)
            {
                self.entries.swap_remove(oldest_idx);
            }
        }

        self.entries.push(NeighborEntry {
            ip,
            mac,
            state,
            timestamp,
        });
    }

    /// Remove the entry for `ip`, returning `true` if it existed.
    pub fn remove(&mut self, ip: &Ipv6Addr) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.ip != *ip);
        self.entries.len() != before
    }

    /// Number of entries currently in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// NDP message construction stubs
// ---------------------------------------------------------------------------

/// Build a Neighbor Solicitation message (ICMPv6 type 135).
///
/// # Stub
/// This function is not yet implemented and always returns
/// `Err(NetError::NotImplemented)`.
pub fn create_neighbor_solicitation(
    _src_ip: &Ipv6Addr,
    _target_ip: &Ipv6Addr,
    _src_mac: &MacAddress,
) -> Result<Vec<u8>, NetError> {
    Err(NetError::NotImplemented)
}

/// Build a Neighbor Advertisement message (ICMPv6 type 136).
///
/// # Stub
/// This function is not yet implemented and always returns
/// `Err(NetError::NotImplemented)`.
pub fn create_neighbor_advertisement(
    _src_ip: &Ipv6Addr,
    _target_ip: &Ipv6Addr,
    _src_mac: &MacAddress,
    _solicited: bool,
) -> Result<Vec<u8>, NetError> {
    Err(NetError::NotImplemented)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mac(b: u8) -> MacAddress {
        MacAddress::new(b, b, b, b, b, b)
    }

    fn ip(last: u8) -> Ipv6Addr {
        Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, last as u16)
    }

    // -- NdpType --

    #[test]
    fn test_ndp_type_values() {
        assert_eq!(NdpType::RouterSolicitation as u8, 133);
        assert_eq!(NdpType::RouterAdvertisement as u8, 134);
        assert_eq!(NdpType::NeighborSolicitation as u8, 135);
        assert_eq!(NdpType::NeighborAdvertisement as u8, 136);
        assert_eq!(NdpType::Redirect as u8, 137);
    }

    #[test]
    fn test_ndp_type_from_u8() {
        assert_eq!(NdpType::from_u8(135), Some(NdpType::NeighborSolicitation));
        assert_eq!(NdpType::from_u8(0), None);
    }

    // -- NeighborTable --

    #[test]
    fn test_table_new_empty() {
        let tbl = NeighborTable::new();
        assert!(tbl.is_empty());
        assert_eq!(tbl.len(), 0);
    }

    #[test]
    fn test_table_insert_and_lookup() {
        let mut tbl = NeighborTable::new();
        tbl.insert(ip(1), mac(0xaa), NeighborState::Reachable, 100);
        assert_eq!(tbl.len(), 1);
        let found = tbl.lookup(&ip(1));
        assert!(found.is_some());
        assert_eq!(*found.unwrap(), mac(0xaa));
    }

    #[test]
    fn test_table_update_existing() {
        let mut tbl = NeighborTable::new();
        tbl.insert(ip(1), mac(0xaa), NeighborState::Reachable, 100);
        tbl.insert(ip(1), mac(0xbb), NeighborState::Stale, 200);
        assert_eq!(tbl.len(), 1);
        assert_eq!(*tbl.lookup(&ip(1)).unwrap(), mac(0xbb));
    }

    #[test]
    fn test_table_remove() {
        let mut tbl = NeighborTable::new();
        tbl.insert(ip(1), mac(0xaa), NeighborState::Reachable, 100);
        assert!(tbl.remove(&ip(1)));
        assert!(tbl.is_empty());
        assert!(!tbl.remove(&ip(1))); // already gone
    }

    #[test]
    fn test_table_lookup_miss() {
        let tbl = NeighborTable::new();
        assert!(tbl.lookup(&ip(99)).is_none());
    }

    #[test]
    fn test_table_multiple_entries() {
        let mut tbl = NeighborTable::new();
        for i in 1..=5 {
            tbl.insert(ip(i), mac(i), NeighborState::Reachable, i as u64);
        }
        assert_eq!(tbl.len(), 5);
        for i in 1..=5 {
            assert!(tbl.lookup(&ip(i)).is_some());
        }
    }

    #[test]
    fn test_table_eviction_at_max() {
        let mut tbl = NeighborTable::new();
        // Fill to MAX_ENTRIES
        for i in 0..NeighborTable::MAX_ENTRIES {
            let addr = Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, i as u16);
            tbl.insert(addr, mac(i as u8), NeighborState::Reachable, i as u64);
        }
        assert_eq!(tbl.len(), NeighborTable::MAX_ENTRIES);

        // Insert one more — oldest (timestamp 0) should be evicted
        let new_addr = Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 0xffff);
        tbl.insert(new_addr, mac(0xff), NeighborState::Reachable, 999);
        assert_eq!(tbl.len(), NeighborTable::MAX_ENTRIES);
        assert!(tbl.lookup(&new_addr).is_some());
    }

    #[test]
    fn test_neighbor_state_variants() {
        let states = [
            NeighborState::Incomplete,
            NeighborState::Reachable,
            NeighborState::Stale,
            NeighborState::Delay,
            NeighborState::Probe,
        ];
        // All five variants should be distinct.
        for (i, a) in states.iter().enumerate() {
            for (j, b) in states.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_create_neighbor_solicitation_stub() {
        let result = create_neighbor_solicitation(&ip(1), &ip(2), &mac(0xaa));
        assert!(result.is_err());
    }

    #[test]
    fn test_create_neighbor_advertisement_stub() {
        let result = create_neighbor_advertisement(&ip(1), &ip(2), &mac(0xaa), true);
        assert!(result.is_err());
    }
}
