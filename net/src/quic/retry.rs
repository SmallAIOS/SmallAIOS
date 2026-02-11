// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC Retry mechanism and version negotiation (RFC 9000 Sections 7.3, 17.2.5).
//!
//! Implements:
//! - Retry token generation and validation
//! - Version negotiation packet handling
//! - 0-RTT session ticket storage and replay protection

use alloc::vec::Vec;

/// Supported QUIC versions.
pub const SUPPORTED_VERSIONS: &[u32] = &[0x00000001]; // QUIC v1

/// Retry token validity duration (default: 30 seconds).
pub const RETRY_TOKEN_LIFETIME_US: u64 = 30_000_000;

/// Maximum number of stored session tickets.
const MAX_SESSION_TICKETS: usize = 16;

/// Maximum number of entries in the replay cache.
const MAX_REPLAY_ENTRIES: usize = 256;

/// Session ticket for 0-RTT resumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionTicket {
    /// Ticket data (opaque, from TLS).
    pub data: Vec<u8>,
    /// Server name indication (hostname).
    pub server_name: Vec<u8>,
    /// Time the ticket was issued (microseconds).
    pub issued_at_us: u64,
    /// Ticket lifetime (microseconds).
    pub lifetime_us: u64,
    /// Maximum early data size.
    pub max_early_data: u64,
}

impl SessionTicket {
    /// Check if the ticket is still valid.
    pub fn is_valid(&self, now_us: u64) -> bool {
        now_us.saturating_sub(self.issued_at_us) < self.lifetime_us
    }
}

/// Session ticket store for 0-RTT.
#[derive(Debug)]
pub struct SessionTicketStore {
    tickets: Vec<SessionTicket>,
}

impl Default for SessionTicketStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionTicketStore {
    /// Create an empty ticket store.
    pub fn new() -> Self {
        Self {
            tickets: Vec::new(),
        }
    }

    /// Store a session ticket.
    pub fn store(&mut self, ticket: SessionTicket) {
        // Remove existing ticket for same server
        self.tickets.retain(|t| t.server_name != ticket.server_name);

        // Evict oldest if full
        if self.tickets.len() >= MAX_SESSION_TICKETS {
            self.tickets.remove(0);
        }

        self.tickets.push(ticket);
    }

    /// Lookup a valid ticket for a server.
    pub fn lookup(&self, server_name: &[u8], now_us: u64) -> Option<&SessionTicket> {
        self.tickets
            .iter()
            .rev()
            .find(|t| t.server_name == server_name && t.is_valid(now_us))
    }

    /// Remove expired tickets.
    pub fn expire(&mut self, now_us: u64) {
        self.tickets.retain(|t| t.is_valid(now_us));
    }

    /// Number of stored tickets.
    pub fn count(&self) -> usize {
        self.tickets.len()
    }
}

/// Replay protection for 0-RTT data.
///
/// Uses a timestamp-based cache to detect replayed 0-RTT data.
#[derive(Debug)]
pub struct ReplayFilter {
    /// (client_hello_hash, timestamp) pairs.
    entries: Vec<(u64, u64)>,
    /// Window start (reject anything before this).
    window_start_us: u64,
    /// Window size (microseconds).
    window_size_us: u64,
}

impl ReplayFilter {
    /// Create a new replay filter.
    pub fn new(window_size_us: u64) -> Self {
        Self {
            entries: Vec::new(),
            window_start_us: 0,
            window_size_us,
        }
    }

    /// Check if the given data has been seen before.
    /// Returns true if this is a NEW (non-replayed) entry.
    pub fn check_and_record(&mut self, hash: u64, now_us: u64) -> bool {
        // Reject if outside window
        if now_us < self.window_start_us {
            return false;
        }

        // Check for duplicate
        if self.entries.iter().any(|&(h, _)| h == hash) {
            return false;
        }

        // Record entry
        if self.entries.len() >= MAX_REPLAY_ENTRIES {
            // Evict oldest
            self.entries.remove(0);
        }
        self.entries.push((hash, now_us));

        true
    }

    /// Advance the window, removing old entries.
    pub fn advance_window(&mut self, now_us: u64) {
        if now_us > self.window_size_us {
            self.window_start_us = now_us - self.window_size_us;
        }
        self.entries
            .retain(|&(_, ts)| ts >= self.window_start_us);
    }

    /// Number of entries in the filter.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// Retry token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryToken {
    /// Original destination connection ID.
    pub odcid: Vec<u8>,
    /// Client IP address (opaque).
    pub client_addr: Vec<u8>,
    /// Issue time (microseconds).
    pub issued_at_us: u64,
}

impl RetryToken {
    /// Encode a retry token into bytes.
    ///
    /// Format: [odcid_len(1)] [odcid] [addr_len(1)] [addr] [timestamp(8)]
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, crate::NetError> {
        let total = 1 + self.odcid.len() + 1 + self.client_addr.len() + 8;
        if buf.len() < total {
            return Err(crate::NetError::BufferTooSmall);
        }

        let mut offset = 0;
        buf[offset] = self.odcid.len() as u8;
        offset += 1;
        buf[offset..offset + self.odcid.len()].copy_from_slice(&self.odcid);
        offset += self.odcid.len();
        buf[offset] = self.client_addr.len() as u8;
        offset += 1;
        buf[offset..offset + self.client_addr.len()].copy_from_slice(&self.client_addr);
        offset += self.client_addr.len();
        buf[offset..offset + 8].copy_from_slice(&self.issued_at_us.to_be_bytes());
        offset += 8;

        Ok(offset)
    }

    /// Decode a retry token from bytes.
    pub fn decode(data: &[u8]) -> Result<Self, crate::NetError> {
        if data.len() < 2 {
            return Err(crate::NetError::PacketTooShort);
        }

        let mut offset = 0;
        let odcid_len = data[offset] as usize;
        offset += 1;
        if offset + odcid_len > data.len() {
            return Err(crate::NetError::PacketTooShort);
        }
        let odcid = Vec::from(&data[offset..offset + odcid_len]);
        offset += odcid_len;

        if offset >= data.len() {
            return Err(crate::NetError::PacketTooShort);
        }
        let addr_len = data[offset] as usize;
        offset += 1;
        if offset + addr_len + 8 > data.len() {
            return Err(crate::NetError::PacketTooShort);
        }
        let client_addr = Vec::from(&data[offset..offset + addr_len]);
        offset += addr_len;

        let issued_at_us = u64::from_be_bytes(
            data[offset..offset + 8].try_into().map_err(|_| crate::NetError::InvalidHeader)?,
        );

        Ok(Self {
            odcid,
            client_addr,
            issued_at_us,
        })
    }

    /// Check if the token is still valid.
    pub fn is_valid(&self, now_us: u64, client_addr: &[u8]) -> bool {
        if self.client_addr != client_addr {
            return false;
        }
        now_us.saturating_sub(self.issued_at_us) < RETRY_TOKEN_LIFETIME_US
    }
}

/// Check if a QUIC version is supported.
pub fn is_version_supported(version: u32) -> bool {
    SUPPORTED_VERSIONS.contains(&version)
}

/// Encode a Version Negotiation packet.
///
/// Format: [first_byte(1)] [version=0(4)] [dcid_len(1)] [dcid] [scid_len(1)] [scid] [versions(4*N)]
pub fn encode_version_negotiation(
    dcid: &[u8],
    scid: &[u8],
    buf: &mut [u8],
) -> Result<usize, crate::NetError> {
    let total = 1 + 4 + 1 + dcid.len() + 1 + scid.len() + SUPPORTED_VERSIONS.len() * 4;
    if buf.len() < total {
        return Err(crate::NetError::BufferTooSmall);
    }

    let mut offset = 0;
    // First byte: long header form, random bits
    buf[offset] = 0x80 | 0x40; // Form=1, Fixed=1
    offset += 1;

    // Version = 0 (signals version negotiation)
    buf[offset..offset + 4].copy_from_slice(&0u32.to_be_bytes());
    offset += 4;

    // DCID (note: in VN, DCID is the client's SCID)
    buf[offset] = dcid.len() as u8;
    offset += 1;
    buf[offset..offset + dcid.len()].copy_from_slice(dcid);
    offset += dcid.len();

    // SCID (note: in VN, SCID is the client's DCID)
    buf[offset] = scid.len() as u8;
    offset += 1;
    buf[offset..offset + scid.len()].copy_from_slice(scid);
    offset += scid.len();

    // Supported versions
    for &ver in SUPPORTED_VERSIONS {
        buf[offset..offset + 4].copy_from_slice(&ver.to_be_bytes());
        offset += 4;
    }

    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // ---- Session Ticket ----

    #[test]
    fn test_session_ticket_validity() {
        let ticket = SessionTicket {
            data: vec![0x01; 32],
            server_name: Vec::from(b"example.com" as &[u8]),
            issued_at_us: 1_000_000,
            lifetime_us: 300_000_000, // 5 minutes
            max_early_data: 16384,
        };
        assert!(ticket.is_valid(100_000_000));
        assert!(!ticket.is_valid(500_000_000));
    }

    #[test]
    fn test_ticket_store() {
        let mut store = SessionTicketStore::new();
        assert_eq!(store.count(), 0);

        store.store(SessionTicket {
            data: vec![1],
            server_name: Vec::from(b"a.com" as &[u8]),
            issued_at_us: 0,
            lifetime_us: 1_000_000,
            max_early_data: 0,
        });
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_ticket_store_lookup() {
        let mut store = SessionTicketStore::new();
        store.store(SessionTicket {
            data: vec![1],
            server_name: Vec::from(b"a.com" as &[u8]),
            issued_at_us: 0,
            lifetime_us: 1_000_000,
            max_early_data: 0,
        });

        assert!(store.lookup(b"a.com", 500_000).is_some());
        assert!(store.lookup(b"b.com", 500_000).is_none());
        assert!(store.lookup(b"a.com", 2_000_000).is_none());
    }

    #[test]
    fn test_ticket_store_replaces_same_server() {
        let mut store = SessionTicketStore::new();
        store.store(SessionTicket {
            data: vec![1],
            server_name: Vec::from(b"a.com" as &[u8]),
            issued_at_us: 0,
            lifetime_us: 1_000_000,
            max_early_data: 0,
        });
        store.store(SessionTicket {
            data: vec![2],
            server_name: Vec::from(b"a.com" as &[u8]),
            issued_at_us: 100,
            lifetime_us: 1_000_000,
            max_early_data: 0,
        });
        assert_eq!(store.count(), 1);
        assert_eq!(store.lookup(b"a.com", 200).unwrap().data, vec![2]);
    }

    #[test]
    fn test_ticket_store_expire() {
        let mut store = SessionTicketStore::new();
        store.store(SessionTicket {
            data: vec![1],
            server_name: Vec::from(b"a.com" as &[u8]),
            issued_at_us: 0,
            lifetime_us: 100,
            max_early_data: 0,
        });
        store.expire(200);
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_ticket_store_capacity() {
        let mut store = SessionTicketStore::new();
        for i in 0..20u8 {
            store.store(SessionTicket {
                data: vec![i],
                server_name: Vec::from([i]),
                issued_at_us: 0,
                lifetime_us: u64::MAX,
                max_early_data: 0,
            });
        }
        assert!(store.count() <= MAX_SESSION_TICKETS);
    }

    // ---- Replay Filter ----

    #[test]
    fn test_replay_filter_new_entry() {
        let mut filter = ReplayFilter::new(60_000_000);
        assert!(filter.check_and_record(0x1234, 1000));
        assert_eq!(filter.entry_count(), 1);
    }

    #[test]
    fn test_replay_filter_duplicate() {
        let mut filter = ReplayFilter::new(60_000_000);
        assert!(filter.check_and_record(0x1234, 1000));
        assert!(!filter.check_and_record(0x1234, 2000));
    }

    #[test]
    fn test_replay_filter_different_hash() {
        let mut filter = ReplayFilter::new(60_000_000);
        assert!(filter.check_and_record(0x1234, 1000));
        assert!(filter.check_and_record(0x5678, 2000));
    }

    #[test]
    fn test_replay_filter_advance_window() {
        let mut filter = ReplayFilter::new(10_000);
        filter.check_and_record(0x1, 1000);
        filter.check_and_record(0x2, 11000);
        filter.check_and_record(0x3, 15000);

        filter.advance_window(20000);
        // Entry at 1000 evicted (1000 < window_start=10000), 11000 and 15000 remain
        assert_eq!(filter.entry_count(), 2);
    }

    #[test]
    fn test_replay_filter_capacity() {
        let mut filter = ReplayFilter::new(u64::MAX);
        for i in 0..300u64 {
            filter.check_and_record(i, i);
        }
        assert!(filter.entry_count() <= MAX_REPLAY_ENTRIES);
    }

    // ---- Retry Token ----

    #[test]
    fn test_retry_token_encode_decode() {
        let token = RetryToken {
            odcid: vec![0x01, 0x02, 0x03, 0x04],
            client_addr: vec![10, 0, 0, 1],
            issued_at_us: 1_000_000,
        };
        let mut buf = [0u8; 64];
        let len = token.encode(&mut buf).unwrap();
        let decoded = RetryToken::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.odcid, token.odcid);
        assert_eq!(decoded.client_addr, token.client_addr);
        assert_eq!(decoded.issued_at_us, token.issued_at_us);
    }

    #[test]
    fn test_retry_token_validity() {
        let token = RetryToken {
            odcid: vec![0x01],
            client_addr: vec![10, 0, 0, 1],
            issued_at_us: 1_000_000,
        };
        assert!(token.is_valid(10_000_000, &[10, 0, 0, 1]));
        assert!(!token.is_valid(50_000_000, &[10, 0, 0, 1])); // Expired
        assert!(!token.is_valid(10_000_000, &[10, 0, 0, 2])); // Wrong addr
    }

    #[test]
    fn test_retry_token_encode_too_small() {
        let token = RetryToken {
            odcid: vec![0x01; 20],
            client_addr: vec![10, 0, 0, 1],
            issued_at_us: 0,
        };
        let mut buf = [0u8; 5];
        assert_eq!(
            token.encode(&mut buf).err(),
            Some(crate::NetError::BufferTooSmall)
        );
    }

    #[test]
    fn test_retry_token_decode_too_short() {
        assert_eq!(
            RetryToken::decode(&[0x01]).err(),
            Some(crate::NetError::PacketTooShort)
        );
    }

    // ---- Version Negotiation ----

    #[test]
    fn test_is_version_supported() {
        assert!(is_version_supported(0x00000001));
        assert!(!is_version_supported(0x00000002));
        assert!(!is_version_supported(0));
    }

    #[test]
    fn test_encode_version_negotiation() {
        let dcid = [0x01, 0x02, 0x03, 0x04];
        let scid = [0xAA, 0xBB];
        let mut buf = [0u8; 64];
        let len = encode_version_negotiation(&dcid, &scid, &mut buf).unwrap();

        // Parse: first byte, version=0, dcid, scid, supported versions
        assert_eq!(buf[0] & 0x80, 0x80); // Long header form
        assert_eq!(&buf[1..5], &[0, 0, 0, 0]); // Version = 0
        assert_eq!(buf[5], 4); // DCID len
        assert_eq!(&buf[6..10], &dcid);
        assert_eq!(buf[10], 2); // SCID len
        assert_eq!(&buf[11..13], &scid);
        // Supported version
        assert_eq!(&buf[13..17], &0x00000001u32.to_be_bytes());
        assert_eq!(len, 17);
    }

    #[test]
    fn test_encode_vn_buffer_too_small() {
        let mut buf = [0u8; 5];
        assert_eq!(
            encode_version_negotiation(&[0x01], &[0x02], &mut buf).err(),
            Some(crate::NetError::BufferTooSmall)
        );
    }
}
