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

/// Compute the ICMPv6 checksum per RFC 2460 / RFC 4443.
///
/// The pseudo-header includes source/destination IPv6 addresses, the ICMPv6
/// payload length, and next-header value 58 (ICMPv6).
fn icmpv6_checksum(src_ip: &Ipv6Addr, dst_ip: &Ipv6Addr, icmpv6_payload: &[u8]) -> u16 {
    let mut sum: u32 = 0;

    // Pseudo-header: source address (16 bytes as 8 u16 words)
    let src = src_ip.as_bytes();
    for i in (0..16).step_by(2) {
        sum += u16::from_be_bytes([src[i], src[i + 1]]) as u32;
    }

    // Pseudo-header: destination address
    let dst = dst_ip.as_bytes();
    for i in (0..16).step_by(2) {
        sum += u16::from_be_bytes([dst[i], dst[i + 1]]) as u32;
    }

    // Pseudo-header: upper-layer packet length (u32 BE)
    let len = icmpv6_payload.len() as u32;
    sum += len >> 16;
    sum += len & 0xFFFF;

    // Pseudo-header: next header = 58 (ICMPv6)
    sum += 58u32;

    // ICMPv6 payload
    let mut i = 0;
    while i + 1 < icmpv6_payload.len() {
        sum += u16::from_be_bytes([icmpv6_payload[i], icmpv6_payload[i + 1]]) as u32;
        i += 2;
    }
    if i < icmpv6_payload.len() {
        sum += (icmpv6_payload[i] as u32) << 8;
    }

    // Fold carries
    while (sum >> 16) != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    !(sum as u16)
}

/// Build a Neighbor Solicitation message (ICMPv6 type 135).
///
/// The message layout (RFC 4861 section 4.3):
/// ```text
/// Type(1) = 135 | Code(1) = 0 | Checksum(2)
/// Reserved(4) = 0
/// Target Address(16)
/// [Option: Source Link-Layer Address (type=1, len=1, mac=6)]
/// ```
///
/// Returns the raw ICMPv6 message bytes with a valid checksum.
pub fn create_neighbor_solicitation(
    src_ip: &Ipv6Addr,
    target_ip: &Ipv6Addr,
    src_mac: &MacAddress,
) -> Result<Vec<u8>, NetError> {
    // Total: 4 (header) + 4 (reserved) + 16 (target) + 8 (option) = 32 bytes
    let mut msg = Vec::with_capacity(32);

    // Type = 135 (Neighbor Solicitation)
    msg.push(NdpType::NeighborSolicitation as u8);
    // Code = 0
    msg.push(0);
    // Checksum placeholder (filled in below)
    msg.push(0);
    msg.push(0);
    // Reserved (4 bytes)
    msg.extend_from_slice(&[0u8; 4]);
    // Target Address (16 bytes)
    msg.extend_from_slice(target_ip.as_bytes());
    // Option: Source Link-Layer Address
    // Type = 1, Length = 1 (in units of 8 octets), followed by 6-byte MAC
    msg.push(1); // option type
    msg.push(1); // option length (8 bytes total)
    msg.extend_from_slice(src_mac.as_bytes());

    // Compute ICMPv6 checksum
    let cksum = icmpv6_checksum(src_ip, &Ipv6Addr::solicited_node_multicast(target_ip), &msg);
    msg[2] = (cksum >> 8) as u8;
    msg[3] = cksum as u8;

    Ok(msg)
}

/// Build a Neighbor Advertisement message (ICMPv6 type 136).
///
/// The message layout (RFC 4861 section 4.4):
/// ```text
/// Type(1) = 136 | Code(1) = 0 | Checksum(2)
/// R(1 bit) | S(1 bit) | O(1 bit) | Reserved(29 bits)
/// Target Address(16)
/// [Option: Target Link-Layer Address (type=2, len=1, mac=6)]
/// ```
///
/// Flags set:
/// - `solicited`: sets the S (Solicited) and O (Override) flags
/// - unsolicited: sets only the O (Override) flag
pub fn create_neighbor_advertisement(
    src_ip: &Ipv6Addr,
    target_ip: &Ipv6Addr,
    src_mac: &MacAddress,
    solicited: bool,
) -> Result<Vec<u8>, NetError> {
    // Total: 4 (header) + 4 (flags+reserved) + 16 (target) + 8 (option) = 32 bytes
    let mut msg = Vec::with_capacity(32);

    // Type = 136 (Neighbor Advertisement)
    msg.push(NdpType::NeighborAdvertisement as u8);
    // Code = 0
    msg.push(0);
    // Checksum placeholder
    msg.push(0);
    msg.push(0);
    // Flags: R=0, S=solicited, O=1 (override)
    // Bits: R(7) S(6) O(5) in the first byte of the 4-byte flags field
    let flags_byte = if solicited {
        0x60 // S=1, O=1
    } else {
        0x20 // S=0, O=1
    };
    msg.push(flags_byte);
    msg.extend_from_slice(&[0u8; 3]); // remaining 3 bytes of flags+reserved
                                      // Target Address (16 bytes)
    msg.extend_from_slice(src_ip.as_bytes());
    // Option: Target Link-Layer Address
    // Type = 2, Length = 1 (in units of 8 octets)
    msg.push(2); // option type
    msg.push(1); // option length
    msg.extend_from_slice(src_mac.as_bytes());

    // Compute ICMPv6 checksum (src -> dst = target_ip for solicited, or multicast)
    let cksum = icmpv6_checksum(src_ip, target_ip, &msg);
    msg[2] = (cksum >> 8) as u8;
    msg[3] = cksum as u8;

    Ok(msg)
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
    fn test_create_neighbor_solicitation() {
        let src = ip(1);
        let target = ip(2);
        let m = mac(0xaa);
        let msg = create_neighbor_solicitation(&src, &target, &m).unwrap();

        // Total length: 4 (header) + 4 (reserved) + 16 (target) + 8 (option) = 32
        assert_eq!(msg.len(), 32);

        // Type = 135
        assert_eq!(msg[0], NdpType::NeighborSolicitation as u8);
        // Code = 0
        assert_eq!(msg[1], 0);

        // Reserved bytes (4..8) should be zero
        assert_eq!(&msg[4..8], &[0, 0, 0, 0]);

        // Target address at offset 8..24
        assert_eq!(&msg[8..24], target.as_bytes());

        // Source Link-Layer Address option: type=1, len=1
        assert_eq!(msg[24], 1);
        assert_eq!(msg[25], 1);
        assert_eq!(&msg[26..32], m.as_bytes());
    }

    #[test]
    fn test_create_neighbor_advertisement_solicited() {
        let src = ip(1);
        let target = ip(2);
        let m = mac(0xbb);
        let msg = create_neighbor_advertisement(&src, &target, &m, true).unwrap();

        assert_eq!(msg.len(), 32);

        // Type = 136
        assert_eq!(msg[0], NdpType::NeighborAdvertisement as u8);
        // Code = 0
        assert_eq!(msg[1], 0);

        // Flags byte: S=1, O=1 -> 0x60
        assert_eq!(msg[4], 0x60);

        // Target address at offset 8..24 is the sender's own IP
        assert_eq!(&msg[8..24], src.as_bytes());

        // Target Link-Layer Address option: type=2, len=1
        assert_eq!(msg[24], 2);
        assert_eq!(msg[25], 1);
        assert_eq!(&msg[26..32], m.as_bytes());
    }

    #[test]
    fn test_create_neighbor_advertisement_unsolicited() {
        let src = ip(1);
        let target = ip(2);
        let m = mac(0xcc);
        let msg = create_neighbor_advertisement(&src, &target, &m, false).unwrap();

        assert_eq!(msg.len(), 32);
        assert_eq!(msg[0], NdpType::NeighborAdvertisement as u8);

        // Flags byte: S=0, O=1 -> 0x20
        assert_eq!(msg[4], 0x20);
    }

    #[test]
    fn test_icmpv6_checksum_nonzero() {
        // Verify checksum bytes are filled in (not left as zeros)
        let src = ip(1);
        let target = ip(2);
        let m = mac(0xdd);
        let msg = create_neighbor_solicitation(&src, &target, &m).unwrap();
        let cksum = u16::from_be_bytes([msg[2], msg[3]]);
        // Checksum should be non-zero for non-trivial data
        assert_ne!(cksum, 0);
    }
}
