// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC key update mechanism (RFC 9001 Section 6).
//!
//! Implements key phase toggling for 1-RTT traffic key rotation.
//! Both peers can initiate a key update; the transition period
//! allows packets encrypted with either old or new keys.

use super::protection::{CipherSuite, PacketProtectionKeys};

/// Key update state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyUpdateState {
    /// Normal operation, no key update in progress.
    Stable,
    /// Key update initiated locally, awaiting peer acknowledgment.
    UpdateSent,
    /// Received a packet with new key phase, transition in progress.
    UpdateReceived,
}

/// Key generation number (incremented on each key update).
pub type KeyGeneration = u32;

/// Traffic keys for one encryption direction (send or receive).
#[derive(Debug, Clone)]
pub struct DirectionalKeys {
    /// Current keys.
    pub current: PacketProtectionKeys,
    /// Previous keys (retained during transition, discarded after).
    pub previous: Option<PacketProtectionKeys>,
    /// Key generation counter.
    pub generation: KeyGeneration,
}

impl DirectionalKeys {
    /// Create initial directional keys.
    pub fn new(keys: PacketProtectionKeys) -> Self {
        Self {
            current: keys,
            previous: None,
            generation: 0,
        }
    }

    /// Rotate to new keys, saving the current keys as previous.
    pub fn rotate(&mut self, new_keys: PacketProtectionKeys) {
        let old = core::mem::replace(&mut self.current, new_keys);
        self.previous = Some(old);
        self.generation += 1;
    }

    /// Discard previous keys (after key update is confirmed).
    pub fn discard_previous(&mut self) {
        self.previous = None;
    }

    /// Whether previous keys are still available.
    pub fn has_previous(&self) -> bool {
        self.previous.is_some()
    }
}

/// Key update manager for a QUIC 1-RTT connection.
///
/// Tracks the current key phase, manages key rotation, and
/// handles the transition period where both old and new keys
/// are accepted.
#[derive(Debug)]
pub struct KeyUpdateManager {
    /// Current key update state.
    state: KeyUpdateState,
    /// Current key phase bit (toggled on each key update).
    key_phase: bool,
    /// Send-direction keys.
    send_keys: Option<DirectionalKeys>,
    /// Receive-direction keys.
    recv_keys: Option<DirectionalKeys>,
    /// Packet number of the first packet sent with the new key phase.
    update_pn: Option<u64>,
    /// Whether key update is allowed (not during handshake).
    enabled: bool,
    /// Minimum packets between key updates (to prevent rapid cycling).
    min_packets_before_update: u64,
    /// Packets sent since last key update.
    packets_since_update: u64,
}

impl Default for KeyUpdateManager {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyUpdateManager {
    /// Default minimum packets between key updates.
    const DEFAULT_MIN_PACKETS: u64 = 100;

    /// Create a new key update manager.
    pub fn new() -> Self {
        Self {
            state: KeyUpdateState::Stable,
            key_phase: false,
            send_keys: None,
            recv_keys: None,
            update_pn: None,
            enabled: false,
            min_packets_before_update: Self::DEFAULT_MIN_PACKETS,
            packets_since_update: 0,
        }
    }

    /// Initialize with the first set of 1-RTT keys.
    pub fn init(&mut self, send_keys: PacketProtectionKeys, recv_keys: PacketProtectionKeys) {
        self.send_keys = Some(DirectionalKeys::new(send_keys));
        self.recv_keys = Some(DirectionalKeys::new(recv_keys));
        self.enabled = true;
        self.key_phase = false;
        self.packets_since_update = 0;
    }

    /// Current key update state.
    pub fn state(&self) -> KeyUpdateState {
        self.state
    }

    /// Current key phase bit.
    pub fn key_phase(&self) -> bool {
        self.key_phase
    }

    /// Whether key update is currently possible.
    pub fn can_update(&self) -> bool {
        self.enabled
            && self.state == KeyUpdateState::Stable
            && self.packets_since_update >= self.min_packets_before_update
    }

    /// Current key generation (send side).
    pub fn send_generation(&self) -> KeyGeneration {
        self.send_keys.as_ref().map_or(0, |k| k.generation)
    }

    /// Current key generation (receive side).
    pub fn recv_generation(&self) -> KeyGeneration {
        self.recv_keys.as_ref().map_or(0, |k| k.generation)
    }

    /// Get the current send keys.
    pub fn send_keys(&self) -> Option<&PacketProtectionKeys> {
        self.send_keys.as_ref().map(|k| &k.current)
    }

    /// Get the current receive keys.
    pub fn recv_keys(&self) -> Option<&PacketProtectionKeys> {
        self.recv_keys.as_ref().map(|k| &k.current)
    }

    /// Get previous receive keys (for transition period).
    pub fn prev_recv_keys(&self) -> Option<&PacketProtectionKeys> {
        self.recv_keys.as_ref().and_then(|k| k.previous.as_ref())
    }

    /// Initiate a key update (local side).
    ///
    /// Derives new send keys and toggles the key phase.
    /// Returns the new key phase bit, or None if update is not possible.
    pub fn initiate_update(
        &mut self,
        new_send_keys: PacketProtectionKeys,
        new_recv_keys: PacketProtectionKeys,
        first_pn: u64,
    ) -> Option<bool> {
        if !self.can_update() {
            return None;
        }

        if let Some(ref mut send) = self.send_keys {
            send.rotate(new_send_keys);
        }
        if let Some(ref mut recv) = self.recv_keys {
            recv.rotate(new_recv_keys);
        }

        self.key_phase = !self.key_phase;
        self.state = KeyUpdateState::UpdateSent;
        self.update_pn = Some(first_pn);
        self.packets_since_update = 0;

        Some(self.key_phase)
    }

    /// Handle receiving a packet with a different key phase.
    ///
    /// Called when a received 1-RTT packet has a key phase bit that
    /// differs from our current expected receive key phase.
    pub fn on_peer_key_update(
        &mut self,
        new_recv_keys: PacketProtectionKeys,
        new_send_keys: PacketProtectionKeys,
    ) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(ref mut recv) = self.recv_keys {
            recv.rotate(new_recv_keys);
        }
        if let Some(ref mut send) = self.send_keys {
            send.rotate(new_send_keys);
        }

        self.key_phase = !self.key_phase;
        self.state = KeyUpdateState::UpdateReceived;
        self.packets_since_update = 0;
        true
    }

    /// Confirm that the key update is complete.
    ///
    /// Called when we receive an ACK for a packet sent with the new keys
    /// (for locally-initiated updates), or when we send a packet with
    /// the new keys (for peer-initiated updates).
    pub fn confirm_update(&mut self) {
        if let Some(ref mut send) = self.send_keys {
            send.discard_previous();
        }
        if let Some(ref mut recv) = self.recv_keys {
            recv.discard_previous();
        }
        self.state = KeyUpdateState::Stable;
        self.update_pn = None;
    }

    /// Record that a packet was sent (for rate-limiting key updates).
    pub fn on_packet_sent(&mut self) {
        self.packets_since_update += 1;
    }

    /// Check if a received packet number confirms our key update.
    pub fn is_update_confirmed_by_ack(&self, acked_pn: u64) -> bool {
        if self.state != KeyUpdateState::UpdateSent {
            return false;
        }
        self.update_pn.is_some_and(|upn| acked_pn >= upn)
    }

    /// Whether we're in a key transition period.
    pub fn in_transition(&self) -> bool {
        self.state != KeyUpdateState::Stable
    }

    /// Derive next-generation keys using HKDF-Expand-Label (stub).
    ///
    /// In a real implementation this would use HKDF-Expand-Label with
    /// the "quic ku" label as specified in RFC 9001 Section 6.
    pub fn derive_next_keys(
        current: &PacketProtectionKeys,
        cipher: CipherSuite,
    ) -> PacketProtectionKeys {
        // Stub: rotate each byte of the key material forward
        let mut new_key = current.key.clone();
        for (i, byte) in new_key.iter_mut().enumerate() {
            *byte = byte.wrapping_add((i as u8).wrapping_add(1));
        }
        let mut new_iv = current.iv.clone();
        for (i, byte) in new_iv.iter_mut().enumerate() {
            *byte = byte.wrapping_add((i as u8).wrapping_add(1));
        }
        let mut keys = PacketProtectionKeys::new(&new_key, &new_iv, &current.hp_key, cipher);
        keys.key_phase = !current.key_phase;
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keys(seed: u8) -> PacketProtectionKeys {
        PacketProtectionKeys::new(
            &[seed; 32],
            &[seed.wrapping_add(1); 12],
            &[seed.wrapping_add(2); 16],
            CipherSuite::Aes256Gcm,
        )
    }

    #[test]
    fn test_new() {
        let mgr = KeyUpdateManager::new();
        assert_eq!(mgr.state(), KeyUpdateState::Stable);
        assert!(!mgr.key_phase());
        assert!(!mgr.can_update());
        assert!(!mgr.in_transition());
    }

    #[test]
    fn test_init() {
        let mut mgr = KeyUpdateManager::new();
        mgr.init(test_keys(0xAA), test_keys(0xBB));
        assert!(mgr.send_keys().is_some());
        assert!(mgr.recv_keys().is_some());
        assert!(!mgr.can_update()); // need min_packets first
    }

    #[test]
    fn test_can_update_after_min_packets() {
        let mut mgr = KeyUpdateManager::new();
        mgr.init(test_keys(0xAA), test_keys(0xBB));
        for _ in 0..100 {
            mgr.on_packet_sent();
        }
        assert!(mgr.can_update());
    }

    #[test]
    fn test_initiate_update() {
        let mut mgr = KeyUpdateManager::new();
        mgr.init(test_keys(0xAA), test_keys(0xBB));
        for _ in 0..100 {
            mgr.on_packet_sent();
        }

        let new_send = test_keys(0xCC);
        let new_recv = test_keys(0xDD);
        let phase = mgr.initiate_update(new_send, new_recv, 42).unwrap();
        assert!(phase); // toggled from false to true
        assert_eq!(mgr.state(), KeyUpdateState::UpdateSent);
        assert!(mgr.in_transition());
        assert_eq!(mgr.send_generation(), 1);
        assert_eq!(mgr.recv_generation(), 1);
    }

    #[test]
    fn test_cannot_update_during_transition() {
        let mut mgr = KeyUpdateManager::new();
        mgr.init(test_keys(0xAA), test_keys(0xBB));
        for _ in 0..100 {
            mgr.on_packet_sent();
        }
        mgr.initiate_update(test_keys(0xCC), test_keys(0xDD), 42);
        assert!(!mgr.can_update());
    }

    #[test]
    fn test_confirm_update() {
        let mut mgr = KeyUpdateManager::new();
        mgr.init(test_keys(0xAA), test_keys(0xBB));
        for _ in 0..100 {
            mgr.on_packet_sent();
        }
        mgr.initiate_update(test_keys(0xCC), test_keys(0xDD), 42);
        assert!(mgr.is_update_confirmed_by_ack(42));
        mgr.confirm_update();
        assert_eq!(mgr.state(), KeyUpdateState::Stable);
        assert!(!mgr.in_transition());
        assert!(mgr.prev_recv_keys().is_none());
    }

    #[test]
    fn test_ack_before_update_pn() {
        let mut mgr = KeyUpdateManager::new();
        mgr.init(test_keys(0xAA), test_keys(0xBB));
        for _ in 0..100 {
            mgr.on_packet_sent();
        }
        mgr.initiate_update(test_keys(0xCC), test_keys(0xDD), 42);
        assert!(!mgr.is_update_confirmed_by_ack(41)); // too early
    }

    #[test]
    fn test_peer_key_update() {
        let mut mgr = KeyUpdateManager::new();
        mgr.init(test_keys(0xAA), test_keys(0xBB));
        assert!(mgr.on_peer_key_update(test_keys(0xCC), test_keys(0xDD)));
        assert_eq!(mgr.state(), KeyUpdateState::UpdateReceived);
        assert!(mgr.key_phase()); // toggled
        assert!(mgr.prev_recv_keys().is_some());
    }

    #[test]
    fn test_peer_update_not_enabled() {
        let mut mgr = KeyUpdateManager::new();
        assert!(!mgr.on_peer_key_update(test_keys(0xCC), test_keys(0xDD)));
    }

    #[test]
    fn test_directional_keys() {
        let mut dk = DirectionalKeys::new(test_keys(0xAA));
        assert_eq!(dk.generation, 0);
        assert!(!dk.has_previous());

        dk.rotate(test_keys(0xBB));
        assert_eq!(dk.generation, 1);
        assert!(dk.has_previous());

        dk.discard_previous();
        assert!(!dk.has_previous());
    }

    #[test]
    fn test_derive_next_keys() {
        let current = test_keys(0xAA);
        let next = KeyUpdateManager::derive_next_keys(&current, CipherSuite::Aes256Gcm);
        // Keys should differ
        assert_ne!(next.key, current.key);
        assert_ne!(next.iv, current.iv);
        // HP key should be the same (not rotated in key update)
        assert_eq!(next.hp_key, current.hp_key);
        // Key phase should be toggled
        assert_ne!(next.key_phase, current.key_phase);
    }

    #[test]
    fn test_multiple_key_updates() {
        let mut mgr = KeyUpdateManager::new();
        mgr.init(test_keys(0xAA), test_keys(0xBB));

        // First update
        for _ in 0..100 {
            mgr.on_packet_sent();
        }
        mgr.initiate_update(test_keys(0xCC), test_keys(0xDD), 100);
        mgr.confirm_update();
        assert_eq!(mgr.send_generation(), 1);

        // Second update
        for _ in 0..100 {
            mgr.on_packet_sent();
        }
        mgr.initiate_update(test_keys(0xEE), test_keys(0xFF), 200);
        mgr.confirm_update();
        assert_eq!(mgr.send_generation(), 2);
        assert!(!mgr.key_phase()); // toggled back: false -> true -> false
    }

    #[test]
    fn test_key_update_state_variants() {
        assert_ne!(KeyUpdateState::Stable, KeyUpdateState::UpdateSent);
        assert_ne!(KeyUpdateState::UpdateSent, KeyUpdateState::UpdateReceived);
    }
}
