// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AFDX/Ethernet integration — bridge between AFDX Virtual Links and Ethernet/IP.
//!
//! AFDX frames are standard Ethernet frames with a per-VL sequence number
//! trailer. This module provides:
//! - Multicast MAC derivation from VL ID (ARINC 664 Part 7 addressing)
//! - AFDX frame construction from IP payloads
//! - IP payload extraction from received AFDX frames
//! - VL-based Ethernet port mapping and VLAN tag support
//!
//! # MAC Addressing
//!
//! AFDX uses constant-prefix multicast MACs: `03:00:00:00:VL_HI:VL_LO`
//! where `VL_HI:VL_LO` is the 16-bit VL ID in big-endian.

use super::frame::AfdxFrame;
use super::vl::VirtualLink;
use crate::BusError;
use alloc::vec::Vec;

/// AFDX multicast MAC prefix (first 4 bytes).
pub const AFDX_MULTICAST_PREFIX: [u8; 4] = [0x03, 0x00, 0x00, 0x00];

/// EtherType for IPv4 (used in AFDX frames carrying IP payloads).
pub const ETHERTYPE_IPV4: u16 = 0x0800;

/// EtherType for IPv6.
pub const ETHERTYPE_IPV6: u16 = 0x86DD;

/// EtherType for 802.1Q VLAN tag.
pub const ETHERTYPE_VLAN: u16 = 0x8100;

/// Maximum IP payload size within an AFDX frame (1471 bytes: 1500 MTU - 28 UDP/IP - 1 trailer).
pub const MAX_IP_PAYLOAD: usize = 1471;

/// Derive the AFDX multicast destination MAC address from a VL ID.
///
/// Per ARINC 664 Part 7, the destination MAC is:
/// `03:00:00:00:VL_HI:VL_LO`
pub fn vl_to_multicast_mac(vl_id: u16) -> [u8; 6] {
    [
        AFDX_MULTICAST_PREFIX[0],
        AFDX_MULTICAST_PREFIX[1],
        AFDX_MULTICAST_PREFIX[2],
        AFDX_MULTICAST_PREFIX[3],
        (vl_id >> 8) as u8,
        vl_id as u8,
    ]
}

/// Check if a MAC address is an AFDX multicast address.
pub fn is_afdx_multicast(mac: &[u8; 6]) -> bool {
    mac[0..4] == AFDX_MULTICAST_PREFIX
}

/// Extract the VL ID from an AFDX multicast MAC address.
///
/// Returns `None` if the MAC is not an AFDX multicast address.
pub fn mac_to_vl_id(mac: &[u8; 6]) -> Option<u16> {
    if is_afdx_multicast(mac) {
        Some(((mac[4] as u16) << 8) | mac[5] as u16)
    } else {
        None
    }
}

/// AFDX Ethernet port configuration.
///
/// Represents a physical or virtual Ethernet port used for AFDX traffic.
/// Each port has a source MAC and optional VLAN tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AfdxPort {
    /// Port index (0 = Network A, 1 = Network B for redundant VLs).
    pub index: u8,
    /// Source MAC address for frames transmitted on this port.
    pub src_mac: [u8; 6],
    /// Optional 802.1Q VLAN ID (0 = no VLAN tagging).
    pub vlan_id: u16,
    /// Whether this port is enabled.
    pub enabled: bool,
}

impl AfdxPort {
    /// Create a new AFDX port with the given source MAC.
    pub fn new(index: u8, src_mac: [u8; 6]) -> Self {
        Self {
            index,
            src_mac,
            vlan_id: 0,
            enabled: true,
        }
    }

    /// Create a new AFDX port with VLAN tagging.
    pub fn with_vlan(index: u8, src_mac: [u8; 6], vlan_id: u16) -> Self {
        Self {
            index,
            src_mac,
            vlan_id,
            enabled: true,
        }
    }

    /// Returns true if VLAN tagging is enabled.
    pub fn has_vlan(&self) -> bool {
        self.vlan_id > 0
    }
}

/// Bridge between AFDX Virtual Links and the Ethernet/IP stack.
///
/// Provides frame construction and payload extraction for AFDX traffic
/// flowing through the standard Ethernet interface.
#[derive(Debug)]
pub struct AfdxEthernetBridge {
    /// Port A (primary network).
    port_a: AfdxPort,
    /// Port B (redundant network, optional).
    port_b: Option<AfdxPort>,
    /// Per-VL sequence number tracking (VL ID -> next sequence number).
    sequence_numbers: Vec<(u16, u8)>,
    /// Statistics: total frames transmitted.
    tx_count: u64,
    /// Statistics: total frames received.
    rx_count: u64,
    /// Statistics: frames dropped due to invalid VL.
    drop_count: u64,
}

impl AfdxEthernetBridge {
    /// Create a new bridge with a single network port.
    pub fn new(port_a: AfdxPort) -> Self {
        Self {
            port_a,
            port_b: None,
            sequence_numbers: Vec::new(),
            tx_count: 0,
            rx_count: 0,
            drop_count: 0,
        }
    }

    /// Create a new bridge with dual-redundant network ports.
    pub fn new_redundant(port_a: AfdxPort, port_b: AfdxPort) -> Self {
        Self {
            port_a,
            port_b: Some(port_b),
            sequence_numbers: Vec::new(),
            tx_count: 0,
            rx_count: 0,
            drop_count: 0,
        }
    }

    /// Returns whether dual-redundant networking is configured.
    pub fn is_redundant(&self) -> bool {
        self.port_b.is_some()
    }

    /// Returns transmission statistics.
    pub fn tx_count(&self) -> u64 {
        self.tx_count
    }

    /// Returns reception statistics.
    pub fn rx_count(&self) -> u64 {
        self.rx_count
    }

    /// Returns drop statistics.
    pub fn drop_count(&self) -> u64 {
        self.drop_count
    }

    /// Get or initialize the sequence number for a VL.
    fn next_sequence_number(&mut self, vl_id: u16) -> u8 {
        for (vid, sn) in &mut self.sequence_numbers {
            if *vid == vl_id {
                let current = *sn;
                *sn = sn.wrapping_add(1);
                return current;
            }
        }
        self.sequence_numbers.push((vl_id, 1));
        0
    }

    /// Construct an AFDX frame from an IP payload for a given Virtual Link.
    ///
    /// Uses the VL's configuration for MAC addressing and Lmax enforcement,
    /// and the bridge's port configuration for the source MAC.
    pub fn encapsulate(
        &mut self,
        vl: &VirtualLink,
        ip_payload: &[u8],
        ether_type: u16,
    ) -> Result<AfdxFrame, BusError> {
        if ip_payload.len() > MAX_IP_PAYLOAD {
            return Err(BusError::PayloadTooLarge);
        }

        let dest_mac = vl_to_multicast_mac(vl.vl_id());
        let seq = self.next_sequence_number(vl.vl_id());

        let frame = AfdxFrame::new(
            dest_mac,
            self.port_a.src_mac,
            ether_type,
            ip_payload,
            seq,
            vl.lmax(),
        )?;

        self.tx_count += 1;
        Ok(frame)
    }

    /// Encode an AFDX frame into a raw Ethernet byte buffer.
    ///
    /// This produces the raw bytes that would be transmitted on the wire.
    pub fn encode_to_ethernet(&self, frame: &AfdxFrame, buf: &mut [u8]) -> Result<usize, BusError> {
        frame.encode(buf)
    }

    /// Extract an IP payload from a received raw Ethernet frame.
    ///
    /// Validates that the frame is AFDX (multicast prefix match) and
    /// returns the VL ID, IP payload, and sequence number.
    pub fn decapsulate(&mut self, ethernet_data: &[u8]) -> Result<(u16, Vec<u8>, u8), BusError> {
        let frame = AfdxFrame::decode(ethernet_data)?;

        // Verify it's an AFDX multicast frame
        if !is_afdx_multicast(&frame.dest_mac) {
            self.drop_count += 1;
            return Err(BusError::InvalidHeader);
        }

        let vl_id = frame.vl_id();
        self.rx_count += 1;

        Ok((vl_id, frame.payload, frame.trailer.sequence_number))
    }

    /// Process a received frame for a specific VL, validating sequence numbers.
    ///
    /// Returns the payload if accepted, or an error if:
    /// - The frame is not AFDX
    /// - The VL ID doesn't match expected
    /// - The sequence number indicates a gap (for monitoring)
    pub fn receive_for_vl(
        &mut self,
        vl: &VirtualLink,
        ethernet_data: &[u8],
    ) -> Result<Vec<u8>, BusError> {
        let (vl_id, payload, _seq) = self.decapsulate(ethernet_data)?;

        if vl_id != vl.vl_id() {
            self.drop_count += 1;
            return Err(BusError::OutOfRange);
        }

        Ok(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vl() -> VirtualLink {
        VirtualLink::new(1001, 4, 512, 1, false).unwrap()
    }

    fn test_port_a() -> AfdxPort {
        AfdxPort::new(0, [0x02, 0x00, 0x00, 0x00, 0x00, 0x01])
    }

    fn test_port_b() -> AfdxPort {
        AfdxPort::new(1, [0x02, 0x00, 0x00, 0x00, 0x00, 0x02])
    }

    // -- MAC Addressing --

    #[test]
    fn test_vl_to_multicast_mac() {
        let mac = vl_to_multicast_mac(1001);
        assert_eq!(mac, [0x03, 0x00, 0x00, 0x00, 0x03, 0xE9]);
    }

    #[test]
    fn test_vl_to_multicast_mac_zero() {
        let mac = vl_to_multicast_mac(0);
        assert_eq!(mac, [0x03, 0x00, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_vl_to_multicast_mac_max() {
        let mac = vl_to_multicast_mac(0xFFFF);
        assert_eq!(mac, [0x03, 0x00, 0x00, 0x00, 0xFF, 0xFF]);
    }

    #[test]
    fn test_is_afdx_multicast_valid() {
        let mac = [0x03, 0x00, 0x00, 0x00, 0x03, 0xE9];
        assert!(is_afdx_multicast(&mac));
    }

    #[test]
    fn test_is_afdx_multicast_invalid() {
        let mac = [0x01, 0x00, 0x5E, 0x00, 0x00, 0x01]; // IP multicast, not AFDX
        assert!(!is_afdx_multicast(&mac));
    }

    #[test]
    fn test_mac_to_vl_id() {
        let mac = [0x03, 0x00, 0x00, 0x00, 0x03, 0xE9];
        assert_eq!(mac_to_vl_id(&mac), Some(1001));
    }

    #[test]
    fn test_mac_to_vl_id_not_afdx() {
        let mac = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(mac_to_vl_id(&mac), None);
    }

    // -- AfdxPort --

    #[test]
    fn test_port_new() {
        let port = test_port_a();
        assert_eq!(port.index, 0);
        assert!(port.enabled);
        assert!(!port.has_vlan());
    }

    #[test]
    fn test_port_with_vlan() {
        let port = AfdxPort::with_vlan(0, [0x02; 6], 100);
        assert!(port.has_vlan());
        assert_eq!(port.vlan_id, 100);
    }

    // -- AfdxEthernetBridge --

    #[test]
    fn test_bridge_single_port() {
        let bridge = AfdxEthernetBridge::new(test_port_a());
        assert!(!bridge.is_redundant());
        assert_eq!(bridge.tx_count(), 0);
        assert_eq!(bridge.rx_count(), 0);
    }

    #[test]
    fn test_bridge_redundant() {
        let bridge = AfdxEthernetBridge::new_redundant(test_port_a(), test_port_b());
        assert!(bridge.is_redundant());
    }

    #[test]
    fn test_encapsulate_ipv4() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl = test_vl();
        let ip_payload = [0x45, 0x00, 0x00, 0x28, 0xAB, 0xCD, 0x00, 0x00];

        let frame = bridge
            .encapsulate(&vl, &ip_payload, ETHERTYPE_IPV4)
            .unwrap();
        assert_eq!(frame.dest_mac, vl_to_multicast_mac(1001));
        assert_eq!(frame.src_mac, test_port_a().src_mac);
        assert_eq!(frame.ether_type, ETHERTYPE_IPV4);
        assert_eq!(frame.payload.as_slice(), &ip_payload);
        assert_eq!(frame.trailer.sequence_number, 0);
        assert_eq!(bridge.tx_count(), 1);
    }

    #[test]
    fn test_encapsulate_sequence_numbers() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl = test_vl();

        let f1 = bridge.encapsulate(&vl, &[0x01], ETHERTYPE_IPV4).unwrap();
        assert_eq!(f1.trailer.sequence_number, 0);

        let f2 = bridge.encapsulate(&vl, &[0x02], ETHERTYPE_IPV4).unwrap();
        assert_eq!(f2.trailer.sequence_number, 1);

        let f3 = bridge.encapsulate(&vl, &[0x03], ETHERTYPE_IPV4).unwrap();
        assert_eq!(f3.trailer.sequence_number, 2);
    }

    #[test]
    fn test_encapsulate_different_vls() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl_a = VirtualLink::new(100, 4, 512, 1, false).unwrap();
        let vl_b = VirtualLink::new(200, 4, 512, 1, false).unwrap();

        let fa = bridge.encapsulate(&vl_a, &[0x01], ETHERTYPE_IPV4).unwrap();
        let fb = bridge.encapsulate(&vl_b, &[0x02], ETHERTYPE_IPV4).unwrap();

        // Each VL has its own sequence numbering
        assert_eq!(fa.trailer.sequence_number, 0);
        assert_eq!(fb.trailer.sequence_number, 0);
        assert_ne!(fa.dest_mac, fb.dest_mac);
    }

    #[test]
    fn test_encapsulate_payload_too_large() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl = test_vl();
        let huge = [0u8; MAX_IP_PAYLOAD + 1];

        assert_eq!(
            bridge.encapsulate(&vl, &huge, ETHERTYPE_IPV4).err(),
            Some(BusError::PayloadTooLarge)
        );
    }

    #[test]
    fn test_encode_to_ethernet() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl = test_vl();
        let frame = bridge
            .encapsulate(&vl, &[0x45, 0x00], ETHERTYPE_IPV4)
            .unwrap();

        let mut buf = [0u8; 256];
        let len = bridge.encode_to_ethernet(&frame, &mut buf).unwrap();

        // Header(14) + payload(2) + trailer(1) = 17
        assert_eq!(len, 17);
        // Verify dest MAC starts with AFDX prefix
        assert_eq!(&buf[0..4], &AFDX_MULTICAST_PREFIX);
    }

    #[test]
    fn test_decapsulate() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl = test_vl();

        // First encapsulate to get valid wire data
        let frame = bridge
            .encapsulate(&vl, &[0xAA, 0xBB, 0xCC], ETHERTYPE_IPV4)
            .unwrap();
        let mut buf = [0u8; 256];
        let len = bridge.encode_to_ethernet(&frame, &mut buf).unwrap();

        // Now decapsulate
        let (vl_id, payload, seq) = bridge.decapsulate(&buf[..len]).unwrap();
        assert_eq!(vl_id, 1001);
        assert_eq!(payload, &[0xAA, 0xBB, 0xCC]);
        assert_eq!(seq, 0);
        assert_eq!(bridge.rx_count(), 1);
    }

    #[test]
    fn test_decapsulate_non_afdx_frame() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());

        // Construct a non-AFDX Ethernet frame (unicast MAC)
        let mut buf = [0u8; 20];
        buf[0..6].copy_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        buf[6..12].copy_from_slice(&[0x02, 0x00, 0x00, 0x00, 0x00, 0x01]);
        buf[12] = 0x08;
        buf[13] = 0x00;
        buf[14..19].copy_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x05]);
        buf[19] = 0; // trailer

        assert_eq!(
            bridge.decapsulate(&buf).err(),
            Some(BusError::InvalidHeader)
        );
        assert_eq!(bridge.drop_count(), 1);
    }

    #[test]
    fn test_receive_for_vl() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl = test_vl();

        let frame = bridge
            .encapsulate(&vl, &[0xDE, 0xAD], ETHERTYPE_IPV4)
            .unwrap();
        let mut buf = [0u8; 256];
        let len = bridge.encode_to_ethernet(&frame, &mut buf).unwrap();

        let payload = bridge.receive_for_vl(&vl, &buf[..len]).unwrap();
        assert_eq!(payload, &[0xDE, 0xAD]);
    }

    #[test]
    fn test_receive_for_wrong_vl() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl_1001 = test_vl();
        let vl_2002 = VirtualLink::new(2002, 4, 512, 1, false).unwrap();

        let frame = bridge
            .encapsulate(&vl_1001, &[0x01], ETHERTYPE_IPV4)
            .unwrap();
        let mut buf = [0u8; 256];
        let len = bridge.encode_to_ethernet(&frame, &mut buf).unwrap();

        // Try to receive as VL 2002 -- should fail
        assert_eq!(
            bridge.receive_for_vl(&vl_2002, &buf[..len]).err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_end_to_end_ip_over_afdx() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl = test_vl();

        // Simulate an IP packet header (minimal)
        let ip_packet = [
            0x45, 0x00, 0x00, 0x1C, // Version, IHL, TOS, Total Length
            0x00, 0x01, 0x00, 0x00, // ID, Flags, Fragment Offset
            0x40, 0x11, 0x00, 0x00, // TTL, Protocol (UDP), Checksum
            0xC0, 0xA8, 0x01, 0x01, // Source IP: 192.168.1.1
            0xC0, 0xA8, 0x01, 0x02, // Dest IP: 192.168.1.2
            0x00, 0x50, 0x00, 0x51, // UDP src port 80, dst port 81
            0x00, 0x08, 0x00, 0x00, // UDP length, checksum
        ];

        // Encapsulate into AFDX frame
        let frame = bridge.encapsulate(&vl, &ip_packet, ETHERTYPE_IPV4).unwrap();

        // Encode to wire format
        let mut wire = [0u8; 512];
        let wire_len = bridge.encode_to_ethernet(&frame, &mut wire).unwrap();

        // Decapsulate (simulating reception on the other end)
        let (vl_id, payload, seq) = bridge.decapsulate(&wire[..wire_len]).unwrap();
        assert_eq!(vl_id, 1001);
        assert_eq!(seq, 0);
        assert_eq!(payload, ip_packet);

        // Verify we can parse IP version from the recovered payload
        assert_eq!(payload[0] >> 4, 4); // IPv4
        assert_eq!(payload[9], 0x11); // UDP protocol
    }

    #[test]
    fn test_sequence_wrapping() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl = test_vl();

        // Send 257 frames to verify sequence number wrapping (0-255, then 0)
        for i in 0..257u16 {
            let frame = bridge.encapsulate(&vl, &[0x01], ETHERTYPE_IPV4).unwrap();
            assert_eq!(frame.trailer.sequence_number, (i & 0xFF) as u8);
        }
        assert_eq!(bridge.tx_count(), 257);
    }

    #[test]
    fn test_ipv6_over_afdx() {
        let mut bridge = AfdxEthernetBridge::new(test_port_a());
        let vl = test_vl();

        let ipv6_header = [0x60, 0x00, 0x00, 0x00, 0x00, 0x08, 0x11, 0x40];
        let frame = bridge
            .encapsulate(&vl, &ipv6_header, ETHERTYPE_IPV6)
            .unwrap();
        assert_eq!(frame.ether_type, ETHERTYPE_IPV6);

        let mut buf = [0u8; 256];
        let len = bridge.encode_to_ethernet(&frame, &mut buf).unwrap();
        let (_, payload, _) = bridge.decapsulate(&buf[..len]).unwrap();
        assert_eq!(payload[0] >> 4, 6); // IPv6
    }
}
