// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! APID-based Space Packet routing and filtering.
//!
//! Routes received packets to registered handlers based on their APID.
//! Idle packets (APID 0x7FF) are silently discarded.

use crate::BusError;
use crate::ccsds::packet::{IDLE_APID, MAX_APID};

/// Maximum number of registered APID handlers.
pub const MAX_APID_HANDLERS: usize = 64;

/// APID registration entry.
#[derive(Debug, Clone, Copy)]
struct ApidEntry {
    apid: u16,
    active: bool,
}

/// APID-based packet router.
#[derive(Debug)]
pub struct ApidRouter {
    /// Registered APID entries.
    entries: [ApidEntry; MAX_APID_HANDLERS],
    /// Number of active entries.
    count: usize,
    /// Per-APID discard counter (unregistered packets).
    discard_count: u64,
    /// Idle packet counter.
    idle_count: u64,
}

impl Default for ApidRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ApidRouter {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            entries: [ApidEntry { apid: 0, active: false }; MAX_APID_HANDLERS],
            count: 0,
            discard_count: 0,
            idle_count: 0,
        }
    }

    /// Register a handler for the given APID.
    pub fn register(&mut self, apid: u16) -> Result<(), BusError> {
        if apid > MAX_APID {
            return Err(BusError::InvalidApid);
        }
        if apid == IDLE_APID {
            return Err(BusError::InvalidApid);
        }
        // Check for duplicate
        for entry in &self.entries[..self.count] {
            if entry.active && entry.apid == apid {
                return Ok(()); // Already registered
            }
        }
        if self.count >= MAX_APID_HANDLERS {
            return Err(BusError::FilterTableFull);
        }
        self.entries[self.count] = ApidEntry { apid, active: true };
        self.count += 1;
        Ok(())
    }

    /// Unregister a handler for the given APID.
    pub fn unregister(&mut self, apid: u16) -> bool {
        for entry in &mut self.entries[..self.count] {
            if entry.active && entry.apid == apid {
                entry.active = false;
                return true;
            }
        }
        false
    }

    /// Route a packet by APID. Returns `true` if the packet is accepted
    /// (has a registered handler), `false` if discarded.
    pub fn route(&mut self, apid: u16) -> bool {
        if apid == IDLE_APID {
            self.idle_count += 1;
            return false;
        }
        for entry in &self.entries[..self.count] {
            if entry.active && entry.apid == apid {
                return true;
            }
        }
        self.discard_count += 1;
        false
    }

    /// Returns the count of discarded packets (unregistered APIDs).
    pub fn discard_count(&self) -> u64 {
        self.discard_count
    }

    /// Returns the count of idle packets filtered.
    pub fn idle_count(&self) -> u64 {
        self.idle_count
    }

    /// Returns the number of registered APIDs.
    pub fn registered_count(&self) -> usize {
        self.entries[..self.count].iter().filter(|e| e.active).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_route() {
        let mut router = ApidRouter::new();
        router.register(0x1A3).unwrap();
        assert!(router.route(0x1A3));
    }

    #[test]
    fn test_unregistered_apid_discarded() {
        let mut router = ApidRouter::new();
        router.register(0x100).unwrap();
        assert!(!router.route(0x200));
        assert_eq!(router.discard_count(), 1);
    }

    #[test]
    fn test_idle_packet_filtered() {
        let mut router = ApidRouter::new();
        assert!(!router.route(IDLE_APID));
        assert_eq!(router.idle_count(), 1);
        assert_eq!(router.discard_count(), 0); // Idle doesn't count as discard
    }

    #[test]
    fn test_register_invalid_apid() {
        let mut router = ApidRouter::new();
        assert_eq!(router.register(0x800).err(), Some(BusError::InvalidApid));
    }

    #[test]
    fn test_register_idle_apid() {
        let mut router = ApidRouter::new();
        assert_eq!(router.register(IDLE_APID).err(), Some(BusError::InvalidApid));
    }

    #[test]
    fn test_duplicate_registration() {
        let mut router = ApidRouter::new();
        router.register(0x100).unwrap();
        router.register(0x100).unwrap(); // no error, idempotent
        assert_eq!(router.registered_count(), 1);
    }

    #[test]
    fn test_unregister() {
        let mut router = ApidRouter::new();
        router.register(0x100).unwrap();
        assert!(router.route(0x100));
        assert!(router.unregister(0x100));
        assert!(!router.route(0x100));
    }

    #[test]
    fn test_unregister_nonexistent() {
        let mut router = ApidRouter::new();
        assert!(!router.unregister(0x100));
    }

    #[test]
    fn test_multiple_apids() {
        let mut router = ApidRouter::new();
        router.register(0x100).unwrap();
        router.register(0x1A3).unwrap();
        router.register(0x200).unwrap();
        assert!(router.route(0x100));
        assert!(router.route(0x1A3));
        assert!(router.route(0x200));
        assert!(!router.route(0x300));
        assert_eq!(router.registered_count(), 3);
    }

    #[test]
    fn test_router_table_full() {
        let mut router = ApidRouter::new();
        for i in 0..MAX_APID_HANDLERS as u16 {
            router.register(i).unwrap();
        }
        assert_eq!(router.register(0x7FE).err(), Some(BusError::FilterTableFull));
    }

    #[test]
    fn test_default_trait() {
        let router = ApidRouter::default();
        assert_eq!(router.registered_count(), 0);
    }
}
