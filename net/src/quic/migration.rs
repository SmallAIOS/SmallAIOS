// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC connection migration and path validation (RFC 9000 Section 9).
//!
//! Implements:
//! - Path state tracking (active, validating, failed)
//! - PATH_CHALLENGE / PATH_RESPONSE exchange
//! - Migration to new network paths
//! - NAT rebinding detection

use alloc::vec::Vec;

/// Path validation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathState {
    /// Path is active and validated.
    Active,
    /// Path validation in progress (challenge sent, awaiting response).
    Validating,
    /// Path validation failed.
    Failed,
    /// Path unused (previous path after migration).
    Unused,
}

/// Network path (IP:port pair).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkPath {
    /// Local address (IP + port as opaque bytes).
    pub local_addr: Vec<u8>,
    /// Remote address (IP + port as opaque bytes).
    pub remote_addr: Vec<u8>,
    /// Path validation state.
    pub state: PathState,
    /// Challenge data sent (8 bytes per RFC 9000).
    pub challenge_data: Option<[u8; 8]>,
    /// Time the challenge was sent (microseconds).
    pub challenge_sent_us: u64,
    /// Number of challenge attempts.
    pub challenge_attempts: u32,
    /// Maximum challenge attempts before declaring failure.
    pub max_challenge_attempts: u32,
    /// Challenge timeout (microseconds).
    pub challenge_timeout_us: u64,
}

/// Default maximum challenge retries.
const DEFAULT_MAX_CHALLENGE_ATTEMPTS: u32 = 3;

/// Default challenge timeout (3 * initial RTT estimate = ~1 second).
const DEFAULT_CHALLENGE_TIMEOUT_US: u64 = 1_000_000;

impl NetworkPath {
    /// Create a new network path.
    pub fn new(local_addr: &[u8], remote_addr: &[u8]) -> Self {
        Self {
            local_addr: Vec::from(local_addr),
            remote_addr: Vec::from(remote_addr),
            state: PathState::Validating,
            challenge_data: None,
            challenge_sent_us: 0,
            challenge_attempts: 0,
            max_challenge_attempts: DEFAULT_MAX_CHALLENGE_ATTEMPTS,
            challenge_timeout_us: DEFAULT_CHALLENGE_TIMEOUT_US,
        }
    }

    /// Create an already-validated path (e.g., the initial path).
    pub fn new_active(local_addr: &[u8], remote_addr: &[u8]) -> Self {
        Self {
            local_addr: Vec::from(local_addr),
            remote_addr: Vec::from(remote_addr),
            state: PathState::Active,
            challenge_data: None,
            challenge_sent_us: 0,
            challenge_attempts: 0,
            max_challenge_attempts: DEFAULT_MAX_CHALLENGE_ATTEMPTS,
            challenge_timeout_us: DEFAULT_CHALLENGE_TIMEOUT_US,
        }
    }

    /// Whether this path matches a given address pair.
    pub fn matches(&self, local_addr: &[u8], remote_addr: &[u8]) -> bool {
        self.local_addr == local_addr && self.remote_addr == remote_addr
    }
}

/// Path migration manager.
#[derive(Debug)]
pub struct MigrationManager {
    /// Current active path.
    active_path: Option<NetworkPath>,
    /// Path being validated (new candidate).
    pending_path: Option<NetworkPath>,
    /// Previous path (kept for fallback).
    previous_path: Option<NetworkPath>,
    /// Pending PATH_RESPONSE data to send.
    pending_response: Option<[u8; 8]>,
    /// Deterministic seed for challenge data.
    challenge_seed: u64,
}

impl Default for MigrationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MigrationManager {
    /// Create a new migration manager.
    pub fn new() -> Self {
        Self {
            active_path: None,
            pending_path: None,
            previous_path: None,
            pending_response: None,
            challenge_seed: 0x4D49_4752,
        }
    }

    /// Set the initial path (already validated by handshake).
    pub fn set_initial_path(&mut self, local_addr: &[u8], remote_addr: &[u8]) {
        self.active_path = Some(NetworkPath::new_active(local_addr, remote_addr));
    }

    /// Get the active path.
    pub fn active_path(&self) -> Option<&NetworkPath> {
        self.active_path.as_ref()
    }

    /// Detect that a packet arrived from a new address.
    /// Returns true if migration was detected.
    pub fn on_packet_from_new_address(
        &mut self,
        local_addr: &[u8],
        remote_addr: &[u8],
    ) -> bool {
        if let Some(ref active) = self.active_path {
            if active.matches(local_addr, remote_addr) {
                return false; // Same path
            }
        }

        // New address detected — start path validation
        let path = NetworkPath::new(local_addr, remote_addr);
        self.pending_path = Some(path);
        true
    }

    /// Initiate migration to a new path (client-initiated).
    pub fn initiate_migration(
        &mut self,
        local_addr: &[u8],
        remote_addr: &[u8],
        now_us: u64,
    ) -> [u8; 8] {
        let challenge = self.generate_challenge();
        let mut path = NetworkPath::new(local_addr, remote_addr);
        path.challenge_data = Some(challenge);
        path.challenge_sent_us = now_us;
        path.challenge_attempts = 1;
        self.pending_path = Some(path);
        challenge
    }

    /// Generate a PATH_CHALLENGE for the pending path.
    pub fn generate_challenge_for_pending(&mut self, now_us: u64) -> Option<[u8; 8]> {
        // Check if there's already an active challenge
        if let Some(ref path) = self.pending_path {
            if path.challenge_data.is_some() && path.challenge_attempts > 0 {
                return path.challenge_data;
            }
        } else {
            return None;
        }
        let challenge = self.generate_challenge();
        let path = self.pending_path.as_mut().unwrap();
        path.challenge_data = Some(challenge);
        path.challenge_sent_us = now_us;
        path.challenge_attempts += 1;
        Some(challenge)
    }

    /// Handle a received PATH_CHALLENGE (we must respond with PATH_RESPONSE).
    pub fn on_path_challenge(&mut self, data: [u8; 8]) {
        self.pending_response = Some(data);
    }

    /// Get a pending PATH_RESPONSE to send (if any).
    pub fn take_pending_response(&mut self) -> Option<[u8; 8]> {
        self.pending_response.take()
    }

    /// Handle a received PATH_RESPONSE.
    /// Returns true if it matches our pending challenge and validates the path.
    pub fn on_path_response(&mut self, data: [u8; 8]) -> bool {
        if let Some(ref mut path) = self.pending_path {
            if path.challenge_data == Some(data) {
                path.state = PathState::Active;
                path.challenge_data = None;

                // Promote pending to active, demote active to previous
                let old_active = self.active_path.take();
                self.active_path = self.pending_path.take();
                if let Some(mut prev) = old_active {
                    prev.state = PathState::Unused;
                    self.previous_path = Some(prev);
                }
                return true;
            }
        }
        false
    }

    /// Check if the pending path validation has timed out.
    pub fn check_challenge_timeout(&mut self, now_us: u64) -> bool {
        // Early checks with immutable borrow
        let (should_fail, should_retry) = match self.pending_path.as_ref() {
            None => return false,
            Some(path) => {
                if path.state != PathState::Validating {
                    return false;
                }
                if now_us.saturating_sub(path.challenge_sent_us) < path.challenge_timeout_us {
                    return false;
                }
                if path.challenge_attempts >= path.max_challenge_attempts {
                    (true, false)
                } else {
                    (false, true)
                }
            }
        };

        if should_fail {
            self.pending_path.as_mut().unwrap().state = PathState::Failed;
            return true;
        }

        if should_retry {
            let challenge = self.generate_challenge();
            let path = self.pending_path.as_mut().unwrap();
            path.challenge_data = Some(challenge);
            path.challenge_sent_us = now_us;
            path.challenge_attempts += 1;
        }
        false
    }

    /// Revert to the previous path if migration failed.
    pub fn revert_to_previous(&mut self) -> bool {
        if let Some(mut prev) = self.previous_path.take() {
            prev.state = PathState::Active;
            self.active_path = Some(prev);
            self.pending_path = None;
            return true;
        }
        false
    }

    /// Whether we have a pending migration.
    pub fn has_pending_migration(&self) -> bool {
        self.pending_path
            .as_ref()
            .is_some_and(|p| p.state == PathState::Validating)
    }

    fn generate_challenge(&mut self) -> [u8; 8] {
        let mut data = [0u8; 8];
        let mut state = self.challenge_seed;
        for byte in &mut data {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }
        self.challenge_seed = state;
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_new() {
        let path = NetworkPath::new(b"local", b"remote");
        assert_eq!(path.state, PathState::Validating);
        assert!(path.challenge_data.is_none());
    }

    #[test]
    fn test_path_new_active() {
        let path = NetworkPath::new_active(b"local", b"remote");
        assert_eq!(path.state, PathState::Active);
    }

    #[test]
    fn test_path_matches() {
        let path = NetworkPath::new(b"local", b"remote");
        assert!(path.matches(b"local", b"remote"));
        assert!(!path.matches(b"local", b"other"));
        assert!(!path.matches(b"other", b"remote"));
    }

    #[test]
    fn test_migration_manager_new() {
        let mgr = MigrationManager::new();
        assert!(mgr.active_path().is_none());
        assert!(!mgr.has_pending_migration());
    }

    #[test]
    fn test_set_initial_path() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"10.0.0.1:4433", b"10.0.0.2:443");
        let path = mgr.active_path().unwrap();
        assert_eq!(path.state, PathState::Active);
    }

    #[test]
    fn test_detect_new_address() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"10.0.0.1:4433", b"10.0.0.2:443");

        // Same address — no migration
        assert!(!mgr.on_packet_from_new_address(b"10.0.0.1:4433", b"10.0.0.2:443"));

        // New address — migration detected
        assert!(mgr.on_packet_from_new_address(b"10.0.0.1:4433", b"10.0.0.3:443"));
        assert!(mgr.has_pending_migration());
    }

    #[test]
    fn test_initiate_migration() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"local1", b"remote1");
        let challenge = mgr.initiate_migration(b"local2", b"remote2", 1000);
        assert!(mgr.has_pending_migration());
        assert_eq!(challenge.len(), 8);
    }

    #[test]
    fn test_path_challenge_response() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"local1", b"remote1");

        // Receive a PATH_CHALLENGE
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        mgr.on_path_challenge(data);
        assert_eq!(mgr.take_pending_response(), Some(data));
        assert_eq!(mgr.take_pending_response(), None);
    }

    #[test]
    fn test_successful_migration() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"local1", b"remote1");

        let challenge = mgr.initiate_migration(b"local2", b"remote2", 1000);

        // Receive matching response
        assert!(mgr.on_path_response(challenge));
        assert!(!mgr.has_pending_migration());

        // Active path should be the new one
        let path = mgr.active_path().unwrap();
        assert!(path.matches(b"local2", b"remote2"));
        assert_eq!(path.state, PathState::Active);
    }

    #[test]
    fn test_wrong_response() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"local1", b"remote1");
        mgr.initiate_migration(b"local2", b"remote2", 1000);

        // Wrong response data
        assert!(!mgr.on_path_response([0xFF; 8]));
        assert!(mgr.has_pending_migration());
    }

    #[test]
    fn test_challenge_timeout() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"local1", b"remote1");
        mgr.initiate_migration(b"local2", b"remote2", 1000);

        // Not timed out yet
        assert!(!mgr.check_challenge_timeout(500_000));

        // After timeout — should retry (attempt 2)
        assert!(!mgr.check_challenge_timeout(1_100_000));

        // After timeout again — retry (attempt 3 = max)
        assert!(!mgr.check_challenge_timeout(2_200_000));

        // After max attempts — should fail
        assert!(mgr.check_challenge_timeout(3_300_000));
    }

    #[test]
    fn test_revert_to_previous() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"local1", b"remote1");

        let challenge = mgr.initiate_migration(b"local2", b"remote2", 1000);
        mgr.on_path_response(challenge);

        // Now revert
        assert!(mgr.revert_to_previous());
        let path = mgr.active_path().unwrap();
        assert!(path.matches(b"local1", b"remote1"));
    }

    #[test]
    fn test_revert_no_previous() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"local1", b"remote1");
        assert!(!mgr.revert_to_previous());
    }

    #[test]
    fn test_unique_challenges() {
        let mut mgr = MigrationManager::new();
        mgr.set_initial_path(b"local1", b"remote1");
        let c1 = mgr.initiate_migration(b"local2", b"remote2", 1000);
        // Reset pending for second challenge
        mgr.pending_path = None;
        let c2 = mgr.initiate_migration(b"local3", b"remote3", 2000);
        assert_ne!(c1, c2);
    }
}
