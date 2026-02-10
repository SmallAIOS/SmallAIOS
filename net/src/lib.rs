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

extern crate alloc;

pub mod arp;
pub mod ethernet;
pub mod firewall;
pub mod icmp;
pub mod ipv4;
pub mod ipv6;
pub mod ndp;
pub mod tcp;
pub mod udp;
pub mod virtio_net;

use core::fmt;

/// Network stack error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetError {
    /// Packet data is shorter than the minimum required length.
    PacketTooShort,
    /// Header fields are invalid or malformed.
    InvalidHeader,
    /// Computed checksum does not match the header checksum.
    ChecksumMismatch,
    /// Provided buffer is too small for the requested operation.
    BufferTooSmall,
    /// Protocol field contains an unsupported or unknown value.
    InvalidProtocol,
    /// Address is malformed or invalid.
    InvalidAddress,
    /// Table has reached its maximum capacity.
    TableFull,
    /// Requested entry was not found.
    NotFound,
    /// Feature or protocol is not yet implemented.
    NotImplemented,
    /// Operation timed out.
    Timeout,
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetError::PacketTooShort => write!(f, "packet too short"),
            NetError::InvalidHeader => write!(f, "invalid header"),
            NetError::ChecksumMismatch => write!(f, "checksum mismatch"),
            NetError::BufferTooSmall => write!(f, "buffer too small"),
            NetError::InvalidProtocol => write!(f, "invalid protocol"),
            NetError::InvalidAddress => write!(f, "invalid address"),
            NetError::TableFull => write!(f, "table full"),
            NetError::NotFound => write!(f, "not found"),
            NetError::NotImplemented => write!(f, "not implemented"),
            NetError::Timeout => write!(f, "timeout"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_net_error_display() {
        assert_eq!(format!("{}", NetError::PacketTooShort), "packet too short");
        assert_eq!(
            format!("{}", NetError::ChecksumMismatch),
            "checksum mismatch"
        );
        assert_eq!(format!("{}", NetError::TableFull), "table full");
        assert_eq!(format!("{}", NetError::Timeout), "timeout");
    }

    #[test]
    fn test_net_error_debug() {
        assert_eq!(format!("{:?}", NetError::PacketTooShort), "PacketTooShort");
        assert_eq!(format!("{:?}", NetError::InvalidHeader), "InvalidHeader");
    }

    #[test]
    fn test_net_error_eq() {
        assert_eq!(NetError::PacketTooShort, NetError::PacketTooShort);
        assert_ne!(NetError::PacketTooShort, NetError::InvalidHeader);
    }

    #[test]
    fn test_net_error_clone() {
        let err = NetError::Timeout;
        let cloned = err;
        assert_eq!(err, cloned);
    }
}
