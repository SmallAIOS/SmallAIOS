// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS Native Network Stack
//!
//! Minimal TCP/IP stack for AI inference IPC transport:
//! - Ethernet frame handling
//! - IPv4 with ARP, static routing
//! - IPv6 with NDP, SLAAC, static routing
//! - TCP with CUBIC congestion control, SACK, window scaling
//! - UDP for DNS/NTP
//! - ICMPv4/ICMPv6 (echo, neighbor discovery)
//! - Built-in packet filter / firewall
//! - Network device drivers: virtio-net, Broadcom GENET (RPi), Intel I210

#![no_std]
