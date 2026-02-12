// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC Connection ID management (RFC 9000 Section 5.1).
//!
//! Handles CID generation, retirement, rotation, and lookup.

use super::packet::MAX_CID_LEN;
use crate::NetError;
use alloc::vec::Vec;

/// Maximum number of active connection IDs per connection.
pub const MAX_ACTIVE_CIDS: usize = 8;

/// Default connection ID length.
pub const DEFAULT_CID_LEN: usize = 8;

/// A connection ID with associated metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionId {
    /// Connection ID bytes.
    pub id: Vec<u8>,
    /// Sequence number.
    pub sequence: u64,
    /// Stateless reset token (16 bytes).
    pub reset_token: [u8; 16],
    /// Whether this CID has been retired.
    pub retired: bool,
}

impl ConnectionId {
    /// Create a new connection ID.
    pub fn new(id: &[u8], sequence: u64, reset_token: [u8; 16]) -> Result<Self, NetError> {
        if id.len() > MAX_CID_LEN {
            return Err(NetError::InvalidAddress);
        }
        Ok(Self {
            id: Vec::from(id),
            sequence,
            reset_token,
            retired: false,
        })
    }
}

/// Connection ID manager — tracks local and remote CIDs.
#[derive(Debug)]
pub struct ConnectionIdManager {
    /// Local CIDs (we generated these, remote sends to us using them).
    local_cids: Vec<ConnectionId>,
    /// Remote CIDs (remote generated these, we send to remote using them).
    remote_cids: Vec<ConnectionId>,
    /// Next local sequence number.
    next_local_seq: u64,
    /// Retire-prior-to threshold for remote CIDs.
    retire_prior_to: u64,
    /// CID length (fixed per connection).
    cid_length: usize,
    /// Seed for deterministic CID generation (in lieu of real CSPRNG).
    gen_seed: u64,
}

impl ConnectionIdManager {
    /// Create a new CID manager.
    pub fn new(cid_length: usize) -> Result<Self, NetError> {
        if cid_length > MAX_CID_LEN {
            return Err(NetError::InvalidAddress);
        }
        Ok(Self {
            local_cids: Vec::new(),
            remote_cids: Vec::new(),
            next_local_seq: 0,
            retire_prior_to: 0,
            cid_length,
            gen_seed: 0x5A11_A105,
        })
    }

    /// Returns the CID length for this connection.
    pub fn cid_length(&self) -> usize {
        self.cid_length
    }

    /// Returns the number of active local CIDs.
    pub fn local_cid_count(&self) -> usize {
        self.local_cids.iter().filter(|c| !c.retired).count()
    }

    /// Returns the number of active remote CIDs.
    pub fn remote_cid_count(&self) -> usize {
        self.remote_cids.iter().filter(|c| !c.retired).count()
    }

    /// Generate a new local CID. Returns the CID and its sequence number.
    pub fn generate_local_cid(&mut self) -> Result<&ConnectionId, NetError> {
        if self.local_cid_count() >= MAX_ACTIVE_CIDS {
            return Err(NetError::TableFull);
        }
        let seq = self.next_local_seq;
        self.next_local_seq += 1;

        // Deterministic pseudo-random CID generation
        let mut id = Vec::with_capacity(self.cid_length);
        let mut state = self
            .gen_seed
            .wrapping_add(seq)
            .wrapping_mul(0x5851_F42D_4C95_7F2D);
        for _ in 0..self.cid_length {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            id.push((state >> 33) as u8);
        }
        self.gen_seed = state;

        // Deterministic reset token
        let mut token = [0u8; 16];
        for byte in token.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (state >> 33) as u8;
        }

        let cid = ConnectionId {
            id,
            sequence: seq,
            reset_token: token,
            retired: false,
        };
        self.local_cids.push(cid);
        Ok(self.local_cids.last().unwrap())
    }

    /// Register a remote CID (received via NEW_CONNECTION_ID frame).
    pub fn add_remote_cid(&mut self, cid: ConnectionId) -> Result<(), NetError> {
        // Check for duplicates
        if self.remote_cids.iter().any(|c| c.sequence == cid.sequence) {
            return Ok(()); // Already known
        }
        if self.remote_cid_count() >= MAX_ACTIVE_CIDS {
            return Err(NetError::TableFull);
        }
        self.remote_cids.push(cid);
        Ok(())
    }

    /// Get the current active remote CID (lowest non-retired sequence).
    pub fn active_remote_cid(&self) -> Option<&ConnectionId> {
        self.remote_cids
            .iter()
            .filter(|c| !c.retired)
            .min_by_key(|c| c.sequence)
    }

    /// Retire a local CID by sequence number.
    pub fn retire_local_cid(&mut self, sequence: u64) -> bool {
        if let Some(cid) = self.local_cids.iter_mut().find(|c| c.sequence == sequence) {
            cid.retired = true;
            true
        } else {
            false
        }
    }

    /// Retire remote CIDs with sequence < retire_prior_to.
    pub fn retire_remote_cids_prior_to(&mut self, retire_prior_to: u64) -> usize {
        let mut count = 0;
        for cid in &mut self.remote_cids {
            if cid.sequence < retire_prior_to && !cid.retired {
                cid.retired = true;
                count += 1;
            }
        }
        self.retire_prior_to = retire_prior_to;
        count
    }

    /// Lookup a local CID by its bytes. Returns the sequence number if found.
    pub fn lookup_local_cid(&self, id: &[u8]) -> Option<u64> {
        self.local_cids
            .iter()
            .find(|c| !c.retired && c.id == id)
            .map(|c| c.sequence)
    }

    /// Rotate to the next remote CID (use after the current one is "used up").
    pub fn rotate_remote_cid(&mut self) -> Option<&ConnectionId> {
        let current_seq = self.active_remote_cid().map(|c| c.sequence)?;
        // Retire the current one
        if let Some(cid) = self
            .remote_cids
            .iter_mut()
            .find(|c| c.sequence == current_seq)
        {
            cid.retired = true;
        }
        // Return the next active one
        self.active_remote_cid()
    }

    /// Clean up retired CIDs from the internal lists.
    pub fn garbage_collect(&mut self) {
        self.local_cids.retain(|c| !c.retired);
        self.remote_cids.retain(|c| !c.retired);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_connection_id_new() {
        let cid = ConnectionId::new(&[0x01, 0x02, 0x03, 0x04], 0, [0; 16]).unwrap();
        assert_eq!(cid.id, vec![0x01, 0x02, 0x03, 0x04]);
        assert_eq!(cid.sequence, 0);
        assert!(!cid.retired);
    }

    #[test]
    fn test_connection_id_too_long() {
        let long = vec![0u8; 21];
        assert_eq!(
            ConnectionId::new(&long, 0, [0; 16]).err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_manager_new() {
        let mgr = ConnectionIdManager::new(8).unwrap();
        assert_eq!(mgr.cid_length(), 8);
        assert_eq!(mgr.local_cid_count(), 0);
        assert_eq!(mgr.remote_cid_count(), 0);
    }

    #[test]
    fn test_manager_cid_too_long() {
        assert_eq!(
            ConnectionIdManager::new(21).err(),
            Some(NetError::InvalidAddress)
        );
    }

    #[test]
    fn test_generate_local_cid() {
        let mut mgr = ConnectionIdManager::new(8).unwrap();
        let cid = mgr.generate_local_cid().unwrap();
        assert_eq!(cid.id.len(), 8);
        assert_eq!(cid.sequence, 0);
        assert_eq!(mgr.local_cid_count(), 1);
    }

    #[test]
    fn test_generate_multiple_local_cids() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        for i in 0..MAX_ACTIVE_CIDS {
            let cid = mgr.generate_local_cid().unwrap();
            assert_eq!(cid.sequence, i as u64);
        }
        assert_eq!(mgr.local_cid_count(), MAX_ACTIVE_CIDS);
        // Next should fail
        assert_eq!(mgr.generate_local_cid().err(), Some(NetError::TableFull));
    }

    #[test]
    fn test_generate_unique_cids() {
        let mut mgr = ConnectionIdManager::new(8).unwrap();
        let cid1 = mgr.generate_local_cid().unwrap().id.clone();
        let cid2 = mgr.generate_local_cid().unwrap().id.clone();
        assert_ne!(cid1, cid2);
    }

    #[test]
    fn test_add_remote_cid() {
        let mut mgr = ConnectionIdManager::new(8).unwrap();
        let cid = ConnectionId::new(&[0x01; 8], 0, [0xAA; 16]).unwrap();
        mgr.add_remote_cid(cid).unwrap();
        assert_eq!(mgr.remote_cid_count(), 1);
    }

    #[test]
    fn test_add_remote_duplicate() {
        let mut mgr = ConnectionIdManager::new(8).unwrap();
        let cid = ConnectionId::new(&[0x01; 8], 0, [0xAA; 16]).unwrap();
        mgr.add_remote_cid(cid.clone()).unwrap();
        mgr.add_remote_cid(cid).unwrap(); // duplicate — should be OK
        assert_eq!(mgr.remote_cid_count(), 1);
    }

    #[test]
    fn test_active_remote_cid() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        let cid0 = ConnectionId::new(&[0x01; 4], 0, [0; 16]).unwrap();
        let cid1 = ConnectionId::new(&[0x02; 4], 1, [0; 16]).unwrap();
        mgr.add_remote_cid(cid1).unwrap();
        mgr.add_remote_cid(cid0).unwrap();
        // Should return the one with lowest sequence
        let active = mgr.active_remote_cid().unwrap();
        assert_eq!(active.sequence, 0);
    }

    #[test]
    fn test_retire_local_cid() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        mgr.generate_local_cid().unwrap();
        mgr.generate_local_cid().unwrap();
        assert_eq!(mgr.local_cid_count(), 2);
        assert!(mgr.retire_local_cid(0));
        assert_eq!(mgr.local_cid_count(), 1);
    }

    #[test]
    fn test_retire_nonexistent_cid() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        assert!(!mgr.retire_local_cid(99));
    }

    #[test]
    fn test_retire_remote_cids_prior_to() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        for i in 0..4u64 {
            let cid = ConnectionId::new(&[i as u8; 4], i, [0; 16]).unwrap();
            mgr.add_remote_cid(cid).unwrap();
        }
        let retired = mgr.retire_remote_cids_prior_to(2);
        assert_eq!(retired, 2);
        assert_eq!(mgr.remote_cid_count(), 2);
    }

    #[test]
    fn test_lookup_local_cid() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        let id = mgr.generate_local_cid().unwrap().id.clone();
        assert_eq!(mgr.lookup_local_cid(&id), Some(0));
        assert_eq!(mgr.lookup_local_cid(&[0xFF; 4]), None);
    }

    #[test]
    fn test_lookup_retired_cid_not_found() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        let id = mgr.generate_local_cid().unwrap().id.clone();
        mgr.retire_local_cid(0);
        assert_eq!(mgr.lookup_local_cid(&id), None);
    }

    #[test]
    fn test_rotate_remote_cid() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        let cid0 = ConnectionId::new(&[0x01; 4], 0, [0; 16]).unwrap();
        let cid1 = ConnectionId::new(&[0x02; 4], 1, [0; 16]).unwrap();
        mgr.add_remote_cid(cid0).unwrap();
        mgr.add_remote_cid(cid1).unwrap();

        let next = mgr.rotate_remote_cid().unwrap();
        assert_eq!(next.sequence, 1);
        assert_eq!(mgr.remote_cid_count(), 1);
    }

    #[test]
    fn test_rotate_no_next() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        let cid0 = ConnectionId::new(&[0x01; 4], 0, [0; 16]).unwrap();
        mgr.add_remote_cid(cid0).unwrap();

        assert!(mgr.rotate_remote_cid().is_none());
    }

    #[test]
    fn test_garbage_collect() {
        let mut mgr = ConnectionIdManager::new(4).unwrap();
        mgr.generate_local_cid().unwrap();
        mgr.generate_local_cid().unwrap();
        mgr.retire_local_cid(0);
        let cid = ConnectionId::new(&[0x01; 4], 0, [0; 16]).unwrap();
        mgr.add_remote_cid(cid).unwrap();
        mgr.retire_remote_cids_prior_to(1);
        mgr.garbage_collect();
        assert_eq!(mgr.local_cid_count(), 1);
        assert_eq!(mgr.remote_cid_count(), 0);
    }
}
