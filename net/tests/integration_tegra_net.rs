// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Tegra networking stack.
//!
//! Tests the wiring between DHCP, static IP, and the existing
//! Ethernet/ARP/IPv4 stack. Uses test constructors and mock data —
//! no real hardware MMIO.

#![cfg(all(feature = "rtl8169", feature = "dhcp", feature = "static-ip"))]

extern crate alloc;

use alloc::vec::Vec;
use smallaios_net::dhcp::{DhcpClient, DhcpMessage, MSG_TYPE_ACK, MSG_TYPE_OFFER};
use smallaios_net::ethernet::{EthernetFrame, MacAddress, ETHERTYPE_ARP, ETHERTYPE_IPV4};
use smallaios_net::static_ip::{InterfaceConfig, StaticIpConfig};

// ─── 12.8: Full DHCP flow ───────────────────────────────────────────────────

#[test]
fn test_dhcp_full_handshake() {
    let mac = [0x00, 0x04, 0x4B, 0xDE, 0xAD, 0x01];
    let mut client = DhcpClient::new(mac);

    // 1. DISCOVER
    let discover = client.start_discover();
    assert_eq!(discover.op, 1);
    assert_eq!(discover.htype, 1);
    assert_eq!(discover.hlen, 6);

    // 2. Simulate OFFER
    let mut offer = DhcpMessage::default();
    offer.op = 2;
    offer.xid = discover.xid;
    offer.yiaddr = [192, 168, 1, 100];
    offer.add_option(53, &[MSG_TYPE_OFFER]).unwrap();
    offer.add_option(54, &[192, 168, 1, 1]).unwrap();
    offer.add_option(1, &[255, 255, 255, 0]).unwrap();
    offer.add_option(3, &[192, 168, 1, 1]).unwrap();
    offer.add_option(6, &[8, 8, 8, 8]).unwrap();
    offer.add_option(51, &[0, 0, 0x0E, 0x10]).unwrap();
    offer.add_end_option().unwrap();

    let request = client.handle_offer(&offer);
    assert!(request.is_some());

    // 3. Simulate ACK
    let mut ack = DhcpMessage::default();
    ack.op = 2;
    ack.xid = discover.xid;
    ack.yiaddr = [192, 168, 1, 100];
    ack.add_option(53, &[MSG_TYPE_ACK]).unwrap();
    ack.add_option(54, &[192, 168, 1, 1]).unwrap();
    ack.add_option(1, &[255, 255, 255, 0]).unwrap();
    ack.add_option(3, &[192, 168, 1, 1]).unwrap();
    ack.add_option(6, &[8, 8, 8, 8]).unwrap();
    ack.add_option(51, &[0, 0, 0x0E, 0x10]).unwrap();
    ack.add_end_option().unwrap();

    let lease = client.handle_ack(&ack);
    assert!(lease.is_some());
    let lease = lease.unwrap();
    assert_eq!(lease.ip_addr, [192, 168, 1, 100]);
    assert_eq!(lease.subnet_mask, [255, 255, 255, 0]);
    assert_eq!(lease.gateway, [192, 168, 1, 1]);
    assert_eq!(lease.dns, [8, 8, 8, 8]);
    assert_eq!(lease.lease_time_secs, 3600);
}

#[test]
fn test_dhcp_message_serialize_roundtrip() {
    let mac = [0x00, 0x04, 0x4B, 0xDE, 0xAD, 0x02];
    let mut client = DhcpClient::new(mac);
    let discover = client.start_discover();

    let mut buf = [0u8; 576];
    let len = discover.serialize(&mut buf).unwrap();
    assert!(len >= 240);

    let parsed = DhcpMessage::parse(&buf[..len]).unwrap();
    assert_eq!(parsed.xid, discover.xid);
    assert_eq!(parsed.op, 1);
}

// ─── 12.9: Static IP config ────────────────────────────────────────────────

#[test]
fn test_static_ip_config_applied() {
    smallaios_net::static_ip::clear_config();

    let cfg = StaticIpConfig::new(
        [10, 0, 0, 50],
        [255, 255, 255, 0],
        [10, 0, 0, 1],
        [1, 1, 1, 1],
    );

    smallaios_net::static_ip::apply_static_config(&cfg);
    let applied = smallaios_net::static_ip::get_applied_config();
    assert!(applied.is_some());
    let applied = applied.unwrap();
    assert_eq!(applied.ipv4_addr, [10, 0, 0, 50]);
    assert_eq!(applied.gateway, [10, 0, 0, 1]);

    smallaios_net::static_ip::clear_config();
}

#[test]
fn test_static_ip_with_ipv6_dual_stack() {
    smallaios_net::static_ip::clear_config();

    let cfg = StaticIpConfig::new(
        [10, 0, 0, 50],
        [255, 255, 255, 0],
        [10, 0, 0, 1],
        [1, 1, 1, 1],
    );

    smallaios_net::static_ip::apply_static_config(&cfg);
    let v6_addr = [0xFD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
    smallaios_net::static_ip::apply_static_ipv6(v6_addr);

    let v6 = smallaios_net::static_ip::get_applied_ipv6();
    assert!(v6.is_some());
    assert_eq!(v6.unwrap(), v6_addr);

    smallaios_net::static_ip::clear_config();
}

// ─── 12.10: DHCP fallback behavior ─────────────────────────────────────────

#[test]
fn test_static_overrides_dhcp() {
    let cfg = StaticIpConfig::new(
        [172, 16, 0, 10],
        [255, 255, 0, 0],
        [172, 16, 0, 1],
        [8, 8, 4, 4],
    );

    let result = smallaios_net::static_ip::resolve_config(Some(cfg));
    match result {
        InterfaceConfig::Static(c) => {
            assert_eq!(c.ipv4_addr, [172, 16, 0, 10]);
        }
        InterfaceConfig::Dhcp => panic!("Expected static config"),
    }
}

#[test]
fn test_dhcp_fallback_when_no_static() {
    let result = smallaios_net::static_ip::resolve_config(None);
    match result {
        InterfaceConfig::Dhcp => {}
        InterfaceConfig::Static(_) => panic!("Expected DHCP fallback"),
    }
}

// ─── 12.2/12.3: TX/RX path through Ethernet framing ────────────────────────

#[test]
fn test_ethernet_frame_for_arp() {
    let src = MacAddress::new(0x00, 0x04, 0x4B, 0xDE, 0xAD, 0x01);
    let dst = MacAddress::BROADCAST;

    let arp_payload: Vec<u8> = alloc::vec![
        0x00, 0x01, // HW type: Ethernet
        0x08, 0x00, // Proto type: IPv4
        0x06, 0x04, // HW/proto addr len
        0x00, 0x01, // Opcode: Request
        0x00, 0x04, 0x4B, 0xDE, 0xAD, 0x01, // Sender MAC
        192, 168, 1, 100, // Sender IP
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Target MAC
        192, 168, 1, 1, // Target IP
    ];

    let frame = EthernetFrame {
        dst_mac: dst,
        src_mac: src,
        ether_type: ETHERTYPE_ARP,
        payload: arp_payload.clone(),
    };
    let serialized = frame.serialize();

    assert_eq!(serialized.len(), 14 + arp_payload.len());

    let parsed = EthernetFrame::parse(&serialized).unwrap();
    assert_eq!(parsed.ether_type, ETHERTYPE_ARP);
    assert_eq!(parsed.src_mac, src);
    assert_eq!(parsed.dst_mac, dst);
    assert_eq!(parsed.payload, arp_payload);
}

#[test]
fn test_ethernet_frame_for_ipv4() {
    let src = MacAddress::new(0x00, 0x04, 0x4B, 0xDE, 0xAD, 0x01);
    let dst = MacAddress::new(0x00, 0x04, 0x4B, 0xDE, 0xAD, 0x02);

    let ipv4_payload: Vec<u8> = alloc::vec![
        0x45, 0x00, 0x00, 0x28, // Version, IHL, DSCP, total length
        0x00, 0x01, 0x00, 0x00, // ID, flags, offset
        0x40, 0x06, 0x00, 0x00, // TTL, protocol=TCP, checksum
        192, 168, 1, 100, // Src IP
        192, 168, 1, 1, // Dst IP
    ];

    let frame = EthernetFrame {
        dst_mac: dst,
        src_mac: src,
        ether_type: ETHERTYPE_IPV4,
        payload: ipv4_payload,
    };
    let serialized = frame.serialize();
    assert!(serialized.len() <= 1536);

    let parsed = EthernetFrame::parse(&serialized).unwrap();
    assert_eq!(parsed.ether_type, ETHERTYPE_IPV4);
}

// ─── 12.5: Network config resolution at boot ───────────────────────────────

#[test]
fn test_boot_config_resolution_static() {
    let static_cfg = Some(StaticIpConfig::new(
        [10, 0, 0, 50],
        [255, 255, 255, 0],
        [10, 0, 0, 1],
        [8, 8, 8, 8],
    ));

    match smallaios_net::static_ip::resolve_config(static_cfg) {
        InterfaceConfig::Static(cfg) => {
            assert_eq!(cfg.ipv4_addr, [10, 0, 0, 50]);
            assert_eq!(cfg.subnet_mask, [255, 255, 255, 0]);
        }
        InterfaceConfig::Dhcp => panic!("Should use static config"),
    }
}

#[test]
fn test_boot_config_resolution_dhcp_fallback() {
    match smallaios_net::static_ip::resolve_config(None) {
        InterfaceConfig::Dhcp => {
            let mac = [0x00, 0x04, 0x4B, 0xDE, 0xAD, 0x03];
            let mut client = DhcpClient::new(mac);
            let discover = client.start_discover();
            assert_eq!(discover.op, 1);
        }
        InterfaceConfig::Static(_) => panic!("Should fall back to DHCP"),
    }
}
