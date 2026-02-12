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
// Router Advertisement parsing
// ---------------------------------------------------------------------------

/// Parsed Router Advertisement message fields (RFC 4861 section 4.2).
///
/// ```text
/// Type(1) = 134 | Code(1) = 0 | Checksum(2)
/// Cur Hop Limit(1) | M|O|Rsvd(1) | Router Lifetime(2)
/// Reachable Time(4)
/// Retrans Timer(4)
/// [Options: Prefix Info, MTU, Source LLA, ...]
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouterAdvertisement {
    /// Default hop limit for outgoing packets (0 = unspecified).
    pub cur_hop_limit: u8,
    /// Managed address configuration flag (M).
    pub managed: bool,
    /// Other configuration flag (O).
    pub other_config: bool,
    /// Router lifetime in seconds (0 = not a default router).
    pub router_lifetime: u16,
    /// Reachable time in milliseconds (0 = unspecified).
    pub reachable_time: u32,
    /// Retransmission timer in milliseconds (0 = unspecified).
    pub retrans_timer: u32,
    /// NDP options parsed from the RA.
    pub options: Vec<NdpOption>,
}

/// Parse a Router Advertisement ICMPv6 message body.
///
/// The `data` slice should start at the ICMPv6 type byte (i.e., data[0] == 134).
/// Minimum RA length is 16 bytes (4 header + 12 body).
pub fn parse_router_advertisement(data: &[u8]) -> Result<RouterAdvertisement, NetError> {
    // Minimum: type(1) + code(1) + checksum(2) + cur_hop(1) + flags(1) + lifetime(2)
    //        + reachable(4) + retrans(4) = 16 bytes
    if data.len() < 16 {
        return Err(NetError::PacketTooShort);
    }
    if data[0] != NdpType::RouterAdvertisement as u8 {
        return Err(NetError::InvalidHeader);
    }

    let cur_hop_limit = data[4];
    let flags = data[5];
    let managed = (flags & 0x80) != 0;
    let other_config = (flags & 0x40) != 0;
    let router_lifetime = u16::from_be_bytes([data[6], data[7]]);
    let reachable_time = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
    let retrans_timer = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);

    let options = parse_ndp_options(&data[16..])?;

    Ok(RouterAdvertisement {
        cur_hop_limit,
        managed,
        other_config,
        router_lifetime,
        reachable_time,
        retrans_timer,
        options,
    })
}

/// Parse NDP options from a byte slice (TLV format, type-length-value).
///
/// Each option: type(1) + length(1, in units of 8 octets) + value.
fn parse_ndp_options(mut data: &[u8]) -> Result<Vec<NdpOption>, NetError> {
    let mut options = Vec::new();

    while data.len() >= 2 {
        let opt_type = data[0];
        let opt_len_units = data[1] as usize;
        if opt_len_units == 0 {
            return Err(NetError::InvalidHeader);
        }
        let opt_len_bytes = opt_len_units * 8;
        if data.len() < opt_len_bytes {
            return Err(NetError::PacketTooShort);
        }

        match opt_type {
            1 => {
                // Source Link-Layer Address
                if opt_len_bytes >= 8 {
                    let mac = MacAddress::new(data[2], data[3], data[4], data[5], data[6], data[7]);
                    options.push(NdpOption::SourceLinkLayerAddress(mac));
                }
            }
            2 => {
                // Target Link-Layer Address
                if opt_len_bytes >= 8 {
                    let mac = MacAddress::new(data[2], data[3], data[4], data[5], data[6], data[7]);
                    options.push(NdpOption::TargetLinkLayerAddress(mac));
                }
            }
            3 => {
                // Prefix Information (32 bytes)
                if opt_len_bytes >= 32 {
                    let prefix_len = data[2];
                    let valid_lifetime = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                    let preferred_lifetime =
                        u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
                    let mut prefix = [0u8; 16];
                    prefix.copy_from_slice(&data[16..32]);
                    options.push(NdpOption::PrefixInformation {
                        prefix_len,
                        prefix,
                        valid_lifetime,
                        preferred_lifetime,
                    });
                }
            }
            5 => {
                // MTU
                if opt_len_bytes >= 8 {
                    let mtu = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                    options.push(NdpOption::Mtu(mtu));
                }
            }
            _ => {
                // Skip unknown options
            }
        }

        data = &data[opt_len_bytes..];
    }

    Ok(options)
}

// ---------------------------------------------------------------------------
// SLAAC — Stateless Address Autoconfiguration (RFC 4862)
// ---------------------------------------------------------------------------

/// Result of processing a Router Advertisement for SLAAC.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlaacAddress {
    /// The generated global IPv6 address.
    pub address: Ipv6Addr,
    /// Prefix length (typically 64).
    pub prefix_len: u8,
    /// Valid lifetime in seconds.
    pub valid_lifetime: u32,
    /// Preferred lifetime in seconds.
    pub preferred_lifetime: u32,
}

/// Generate a global IPv6 address via SLAAC from a Router Advertisement.
///
/// For each Prefix Information option with a /64 prefix, generates an address
/// by combining the prefix with a Modified EUI-64 interface identifier
/// derived from the given MAC address (RFC 4862 section 5.5.3).
///
/// Returns all generated addresses (one per eligible prefix).
pub fn slaac_process_ra(ra: &RouterAdvertisement, mac: &MacAddress) -> Vec<SlaacAddress> {
    let mut addresses = Vec::new();

    for opt in &ra.options {
        if let NdpOption::PrefixInformation {
            prefix_len,
            prefix,
            valid_lifetime,
            preferred_lifetime,
        } = opt
        {
            // SLAAC only works with /64 prefixes (interface ID is 64 bits)
            if *prefix_len != 64 {
                continue;
            }
            // Skip if valid lifetime is 0
            if *valid_lifetime == 0 {
                continue;
            }

            // Generate interface ID via Modified EUI-64
            let m = mac.as_bytes();
            let eui0 = m[0] ^ 0x02; // Flip U/L bit

            let mut addr_bytes = [0u8; 16];
            // Copy the first 8 bytes from the prefix
            addr_bytes[..8].copy_from_slice(&prefix[..8]);
            // Fill interface ID (Modified EUI-64)
            addr_bytes[8] = eui0;
            addr_bytes[9] = m[1];
            addr_bytes[10] = m[2];
            addr_bytes[11] = 0xff;
            addr_bytes[12] = 0xfe;
            addr_bytes[13] = m[3];
            addr_bytes[14] = m[4];
            addr_bytes[15] = m[5];

            addresses.push(SlaacAddress {
                address: Ipv6Addr::from_bytes(addr_bytes),
                prefix_len: *prefix_len,
                valid_lifetime: *valid_lifetime,
                preferred_lifetime: *preferred_lifetime,
            });
        }
    }

    addresses
}

/// Build a Router Solicitation message (ICMPv6 type 133).
///
/// Layout (RFC 4861 section 4.1):
/// ```text
/// Type(1) = 133 | Code(1) = 0 | Checksum(2)
/// Reserved(4)
/// [Option: Source Link-Layer Address (type=1, len=1, mac=6)]
/// ```
pub fn create_router_solicitation(
    src_ip: &Ipv6Addr,
    src_mac: &MacAddress,
) -> Result<Vec<u8>, NetError> {
    // Total: 4 (header) + 4 (reserved) + 8 (option) = 16 bytes
    let mut msg = Vec::with_capacity(16);

    // Type = 133 (Router Solicitation)
    msg.push(NdpType::RouterSolicitation as u8);
    // Code = 0
    msg.push(0);
    // Checksum placeholder
    msg.push(0);
    msg.push(0);
    // Reserved (4 bytes)
    msg.extend_from_slice(&[0u8; 4]);
    // Option: Source Link-Layer Address
    msg.push(1); // option type
    msg.push(1); // option length (8 bytes total)
    msg.extend_from_slice(src_mac.as_bytes());

    // Compute checksum (dst = all-routers multicast ff02::2)
    let all_routers = Ipv6Addr::new(0xff02, 0, 0, 0, 0, 0, 0, 2);
    let cksum = icmpv6_checksum(src_ip, &all_routers, &msg);
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
    use alloc::vec;

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

    // -- SLAAC tests --

    #[test]
    fn test_parse_router_advertisement_minimal() {
        // Build a minimal RA: type=134, code=0, checksum=0, cur_hop=64,
        // flags=0, lifetime=1800, reachable=0, retrans=0
        let mut data = vec![0u8; 16];
        data[0] = 134; // RA type
        data[4] = 64; // cur_hop_limit
        data[5] = 0; // flags
        data[6..8].copy_from_slice(&1800u16.to_be_bytes()); // router lifetime
        let ra = parse_router_advertisement(&data).unwrap();
        assert_eq!(ra.cur_hop_limit, 64);
        assert!(!ra.managed);
        assert!(!ra.other_config);
        assert_eq!(ra.router_lifetime, 1800);
        assert_eq!(ra.reachable_time, 0);
        assert_eq!(ra.retrans_timer, 0);
        assert!(ra.options.is_empty());
    }

    #[test]
    fn test_parse_router_advertisement_with_flags() {
        let mut data = vec![0u8; 16];
        data[0] = 134;
        data[5] = 0xC0; // M=1, O=1
        let ra = parse_router_advertisement(&data).unwrap();
        assert!(ra.managed);
        assert!(ra.other_config);
    }

    #[test]
    fn test_parse_router_advertisement_too_short() {
        let data = vec![0u8; 10];
        assert_eq!(
            parse_router_advertisement(&data),
            Err(NetError::PacketTooShort)
        );
    }

    #[test]
    fn test_parse_router_advertisement_wrong_type() {
        let mut data = vec![0u8; 16];
        data[0] = 135; // NS, not RA
        assert_eq!(
            parse_router_advertisement(&data),
            Err(NetError::InvalidHeader)
        );
    }

    #[test]
    fn test_parse_ra_with_prefix_info() {
        // RA header (16 bytes) + Prefix Info option (32 bytes)
        let mut data = vec![0u8; 48];
        data[0] = 134; // RA type
        data[4] = 64; // cur_hop_limit
                      // Prefix Information option at offset 16
        data[16] = 3; // type = Prefix Information
        data[17] = 4; // length = 4 (4*8 = 32 bytes)
        data[18] = 64; // prefix length = /64
                       // valid lifetime at offset 20..24
        data[20..24].copy_from_slice(&3600u32.to_be_bytes());
        // preferred lifetime at offset 24..28
        data[24..28].copy_from_slice(&1800u32.to_be_bytes());
        // prefix at offset 32..48
        let prefix_bytes = Ipv6Addr::new(0x2001, 0xdb8, 0, 1, 0, 0, 0, 0);
        data[32..48].copy_from_slice(prefix_bytes.as_bytes());

        let ra = parse_router_advertisement(&data).unwrap();
        assert_eq!(ra.options.len(), 1);
        match &ra.options[0] {
            NdpOption::PrefixInformation {
                prefix_len,
                prefix,
                valid_lifetime,
                preferred_lifetime,
            } => {
                assert_eq!(*prefix_len, 64);
                assert_eq!(*valid_lifetime, 3600);
                assert_eq!(*preferred_lifetime, 1800);
                assert_eq!(&prefix[..2], &[0x20, 0x01]);
            }
            _ => panic!("expected PrefixInformation"),
        }
    }

    #[test]
    fn test_slaac_process_ra_generates_address() {
        let ra = RouterAdvertisement {
            cur_hop_limit: 64,
            managed: false,
            other_config: false,
            router_lifetime: 1800,
            reachable_time: 0,
            retrans_timer: 0,
            options: alloc::vec![NdpOption::PrefixInformation {
                prefix_len: 64,
                prefix: {
                    let addr = Ipv6Addr::new(0x2001, 0xdb8, 0, 1, 0, 0, 0, 0);
                    let mut p = [0u8; 16];
                    p.copy_from_slice(addr.as_bytes());
                    p
                },
                valid_lifetime: 3600,
                preferred_lifetime: 1800,
            }],
        };

        let m = MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55);
        let addresses = slaac_process_ra(&ra, &m);
        assert_eq!(addresses.len(), 1);

        let slaac = &addresses[0];
        assert_eq!(slaac.prefix_len, 64);
        assert_eq!(slaac.valid_lifetime, 3600);
        assert_eq!(slaac.preferred_lifetime, 1800);

        // Verify the address: prefix + modified EUI-64
        let b = slaac.address.as_bytes();
        // Prefix bytes
        assert_eq!(b[0], 0x20);
        assert_eq!(b[1], 0x01);
        assert_eq!(b[2], 0x0d);
        assert_eq!(b[3], 0xb8);
        assert_eq!(b[4], 0x00);
        assert_eq!(b[5], 0x00);
        assert_eq!(b[6], 0x00);
        assert_eq!(b[7], 0x01);
        // Interface ID: modified EUI-64
        assert_eq!(b[8], 0x02); // 0x00 ^ 0x02
        assert_eq!(b[9], 0x11);
        assert_eq!(b[10], 0x22);
        assert_eq!(b[11], 0xff);
        assert_eq!(b[12], 0xfe);
        assert_eq!(b[13], 0x33);
        assert_eq!(b[14], 0x44);
        assert_eq!(b[15], 0x55);
    }

    #[test]
    fn test_slaac_skips_non_64_prefix() {
        let ra = RouterAdvertisement {
            cur_hop_limit: 64,
            managed: false,
            other_config: false,
            router_lifetime: 1800,
            reachable_time: 0,
            retrans_timer: 0,
            options: alloc::vec![NdpOption::PrefixInformation {
                prefix_len: 48, // Not /64
                prefix: [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                valid_lifetime: 3600,
                preferred_lifetime: 1800,
            }],
        };

        let m = MacAddress::new(0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff);
        let addresses = slaac_process_ra(&ra, &m);
        assert!(addresses.is_empty());
    }

    #[test]
    fn test_slaac_skips_zero_valid_lifetime() {
        let ra = RouterAdvertisement {
            cur_hop_limit: 64,
            managed: false,
            other_config: false,
            router_lifetime: 1800,
            reachable_time: 0,
            retrans_timer: 0,
            options: alloc::vec![NdpOption::PrefixInformation {
                prefix_len: 64,
                prefix: [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                valid_lifetime: 0,
                preferred_lifetime: 0,
            }],
        };

        let m = MacAddress::new(0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff);
        let addresses = slaac_process_ra(&ra, &m);
        assert!(addresses.is_empty());
    }

    #[test]
    fn test_slaac_multiple_prefixes() {
        let ra = RouterAdvertisement {
            cur_hop_limit: 64,
            managed: false,
            other_config: false,
            router_lifetime: 1800,
            reachable_time: 0,
            retrans_timer: 0,
            options: alloc::vec![
                NdpOption::PrefixInformation {
                    prefix_len: 64,
                    prefix: {
                        let addr = Ipv6Addr::new(0x2001, 0xdb8, 0, 1, 0, 0, 0, 0);
                        let mut p = [0u8; 16];
                        p.copy_from_slice(addr.as_bytes());
                        p
                    },
                    valid_lifetime: 3600,
                    preferred_lifetime: 1800,
                },
                NdpOption::PrefixInformation {
                    prefix_len: 64,
                    prefix: {
                        let addr = Ipv6Addr::new(0x2001, 0xdb8, 0, 2, 0, 0, 0, 0);
                        let mut p = [0u8; 16];
                        p.copy_from_slice(addr.as_bytes());
                        p
                    },
                    valid_lifetime: 7200,
                    preferred_lifetime: 3600,
                },
            ],
        };

        let m = MacAddress::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55);
        let addresses = slaac_process_ra(&ra, &m);
        assert_eq!(addresses.len(), 2);
        // Different prefixes produce different addresses
        assert_ne!(addresses[0].address, addresses[1].address);
    }

    #[test]
    fn test_create_router_solicitation() {
        let src = ip(1);
        let m = mac(0xaa);
        let msg = create_router_solicitation(&src, &m).unwrap();
        assert_eq!(msg.len(), 16);
        assert_eq!(msg[0], NdpType::RouterSolicitation as u8);
        assert_eq!(msg[1], 0);
        // Reserved bytes
        assert_eq!(&msg[4..8], &[0, 0, 0, 0]);
        // Source LLA option
        assert_eq!(msg[8], 1); // type
        assert_eq!(msg[9], 1); // length
    }

    #[test]
    fn test_parse_ndp_options_zero_length_rejected() {
        // Option with length=0 should be rejected
        let data = [3u8, 0]; // type=3, length=0
        assert_eq!(parse_ndp_options(&data), Err(NetError::InvalidHeader));
    }

    #[test]
    fn test_parse_ndp_options_mtu() {
        // MTU option: type=5, length=1 (8 bytes), reserved(2), mtu(4)
        let mut data = [0u8; 8];
        data[0] = 5; // type = MTU
        data[1] = 1; // length = 1 (8 bytes)
                     // MTU at offset 4..8
        data[4..8].copy_from_slice(&1500u32.to_be_bytes());
        let options = parse_ndp_options(&data).unwrap();
        assert_eq!(options.len(), 1);
        assert_eq!(options[0], NdpOption::Mtu(1500));
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
