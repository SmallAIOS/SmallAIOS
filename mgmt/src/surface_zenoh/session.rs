// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Zenoh-side session bookkeeping: peer-identity binding, per-identity
//! and total caps, in-flight live flag, and the opaque-token table.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-zenoh-admin/spec.md`
//! Requirements:
//!
//! - "Bearer-token lifecycle"
//! - "Peer-identity binding"
//! - "Per-identity and total session caps"
//!
//! ## Layering
//!
//! The kernel session table (`kernel::auth::session_table`) holds the
//! authoritative `Session` (user_id, role, idle clock). The Zenoh
//! admin wrapper layers a **side-table** on top, indexed by the
//! 16-byte opaque token, that stores:
//!
//! - The kernel `SessionId` so the wrapper can hop to the kernel's
//!   `Session` for role lookup and `reset_idle`.
//! - The `peer_identity` (TLS cert fingerprint or PSK identity hash)
//!   bound at login time. Subsequent requests SHALL match.
//! - The in-flight refcount — incremented on dispatch entry,
//!   decremented on response. While > 0 the session is "live" and
//!   SHALL NOT be invalidated by the idle sweeper.
//!
//! Per-peer accounting (4 sessions per identity, 16 total) lives here
//! too, since the kernel session table cap is shared across all
//! sources (Console, Zenoh, future UDS) — Zenoh enforces its own
//! sub-cap so a chatty remote peer can't starve console logins.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use smallaios_kernel::auth::session_table::{Session, SessionError, SessionId, SessionTable};
use smallaios_kernel::auth::Role;

/// 32-byte (`[u8; 32]`) peer identity. Production wires this to the
/// SHA-256 of the peer's TLS certificate's SubjectPublicKeyInfo
/// (SPKI fingerprint) — the same form already kept in
/// `kernel::auth::session_table::Session::peer_identity`. PSK-mode
/// peers use the SHA-256 of the PSK identity string.
pub type PeerIdentity = [u8; 32];

/// 16-byte opaque admin token. Generated at `login` and used as the
/// lookup key on every subsequent admin request.
pub type Token = [u8; 16];

/// Lookup token rendered for inclusion in JSON responses.
///
/// The wire form is **lowercase hex** — 32 ASCII characters. We avoid
/// base64 because the encoded length is the same after URL-safe
/// padding (~22 chars → 24 with `=` padding ⇒ 24 bytes), and the
/// debug-friendliness of a literal hex string is worth the extra 8
/// bytes per token. Audit logs print this directly.
pub type TokenHex = [u8; 32];

/// Hard cap on the number of distinct peer identities tracked by the
/// per-identity counter. The Zenoh transport already caps connection
/// count; this cap exists so the side-table itself can't be flooded
/// with bogus peer fingerprints (each consuming ~40 bytes). 256 is
/// generous compared to the 16-session total cap.
pub const MAX_TRACKED_IDENTITIES: usize = 256;

/// Errors raised by the [`ZenohSessionTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideTableError {
    /// `mgmt.zenoh_per_identity_max` exceeded for this peer.
    PerIdentityCapReached,
    /// `mgmt.zenoh_total_max` exceeded.
    TotalCapReached,
    /// Kernel session table (32 slots, all sources) is full.
    KernelTableFull,
    /// Token was unknown — wrong, garbage, or already revoked.
    UnknownToken,
    /// The peer presenting this token is not the peer that obtained it.
    PeerMismatch,
    /// The session was idle past its per-role threshold.
    IdleExpired,
    /// The (kernel) `SessionId` has been swept. Returned from
    /// [`ZenohSessionTable::resolve`] when the kernel table no longer
    /// recognises the cached id (idle sweeper ran).
    KernelStale,
}

/// One side-table entry — the per-token bookkeeping.
#[derive(Debug, Clone, Copy)]
pub struct ZenohSessionEntry {
    /// Kernel session id. The kernel table is the source of truth for
    /// `role`, `last_activity_unix_time`, and `must_change_password`.
    pub session_id: SessionId,
    /// Peer identity at login time.
    pub peer: PeerIdentity,
    /// Refcount of currently-executing requests bearing this token.
    /// While > 0 the session SHALL NOT be invalidated.
    pub in_flight: u32,
    /// Cached login Unix time (immutable for the session's lifetime).
    pub login_unix_time: u64,
}

/// Per-role idle thresholds in seconds. Mirrors
/// `kernel::auth::sweeper::IdlePolicy` but local so the wrapper does
/// not have to thread a reference through.
#[derive(Debug, Clone, Copy)]
pub struct IdleThresholds {
    /// Idle window for `Root`, in seconds.
    pub root_secs: u64,
    /// Idle window for `Operator`, in seconds.
    pub operator_secs: u64,
    /// Idle window for `Viewer`, in seconds.
    pub viewer_secs: u64,
}

impl IdleThresholds {
    /// Defaults from `auth-roles` Q5: 15 / 60 / 60 minutes.
    pub const fn defaults() -> Self {
        Self {
            root_secs: 15 * 60,
            operator_secs: 60 * 60,
            viewer_secs: 60 * 60,
        }
    }

    /// Threshold for `role`.
    pub const fn for_role(self, role: Role) -> u64 {
        match role {
            Role::Root => self.root_secs,
            Role::Operator => self.operator_secs,
            Role::Viewer => self.viewer_secs,
        }
    }
}

/// Configuration for the side table.
#[derive(Debug, Clone, Copy)]
pub struct CapsConfig {
    /// Maximum concurrent admin sessions per peer identity. From
    /// `mgmt.zenoh_per_identity_max`.
    pub per_identity_max: u32,
    /// Maximum concurrent Zenoh-backed admin sessions. From
    /// `mgmt.zenoh_total_max`.
    pub total_max: u32,
    /// Per-role idle thresholds.
    pub idle: IdleThresholds,
}

impl CapsConfig {
    /// v1 defaults — 4 / 16 / standard idle thresholds.
    pub const fn defaults() -> Self {
        Self {
            per_identity_max: 4,
            total_max: 16,
            idle: IdleThresholds::defaults(),
        }
    }
}

/// Side-table state, indexed by opaque token.
///
/// Operations are O(log n) over the token map. The `n=16` cap means
/// every operation runs in ~ 4 comparisons; we keep the BTreeMap
/// because it also gives deterministic iteration order useful for
/// tests and audit dumps.
pub struct ZenohSessionTable {
    entries: BTreeMap<Token, ZenohSessionEntry>,
    /// `peer_identity → live count`. Pruned when count reaches zero.
    per_identity: BTreeMap<PeerIdentity, u32>,
    config: CapsConfig,
}

impl ZenohSessionTable {
    /// Construct an empty side table.
    pub fn new(config: CapsConfig) -> Self {
        Self {
            entries: BTreeMap::new(),
            per_identity: BTreeMap::new(),
            config,
        }
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` iff no live entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Live count for `peer`.
    pub fn count_for(&self, peer: &PeerIdentity) -> u32 {
        self.per_identity.get(peer).copied().unwrap_or(0)
    }

    /// Replace the active configuration (e.g. after a `mgmt.zenoh_*`
    /// `Config` write reloads). New limits apply to **future** logins;
    /// existing sessions are not retroactively evicted.
    pub fn set_config(&mut self, config: CapsConfig) {
        self.config = config;
    }

    /// Read the current configuration.
    pub fn config(&self) -> &CapsConfig {
        &self.config
    }

    /// Reserve a slot for a fresh login. Checks the per-identity and
    /// total caps **before** the kernel session table is touched, so
    /// the kernel slot is only acquired when the Zenoh-side caps
    /// admit the new session.
    ///
    /// Returns the (token, kernel session id) pair on success.
    pub fn try_admit_login(
        &mut self,
        kernel_table: &mut SessionTable,
        kernel_session: Session,
        peer: PeerIdentity,
        token: Token,
        now_unix: u64,
    ) -> Result<(Token, SessionId), SideTableError> {
        if self.entries.len() as u32 >= self.config.total_max {
            return Err(SideTableError::TotalCapReached);
        }
        if self.count_for(&peer) >= self.config.per_identity_max {
            return Err(SideTableError::PerIdentityCapReached);
        }
        if self.per_identity.len() >= MAX_TRACKED_IDENTITIES
            && !self.per_identity.contains_key(&peer)
        {
            return Err(SideTableError::TotalCapReached);
        }
        // Stamp peer identity in the kernel session before acquiring
        // the slot, so the kernel side-channel (sweeper, audit) can
        // dump the binding. Also re-stamp the activity / login times
        // to `now_unix` so the wrapper's idle clock is anchored to
        // the dispatch-time clock, not whatever the backend used.
        let mut session_with_peer = kernel_session;
        session_with_peer.peer_identity = Some(peer);
        session_with_peer.login_unix_time = now_unix;
        session_with_peer.last_activity_unix_time = now_unix;
        let session_id = match kernel_table.acquire(session_with_peer) {
            Ok(id) => id,
            Err(SessionError::TableFull) => return Err(SideTableError::KernelTableFull),
            Err(_) => return Err(SideTableError::KernelTableFull),
        };
        let entry = ZenohSessionEntry {
            session_id,
            peer,
            in_flight: 0,
            login_unix_time: now_unix,
        };
        self.entries.insert(token, entry);
        *self.per_identity.entry(peer).or_insert(0) += 1;
        Ok((token, session_id))
    }

    /// Resolve `token` against the kernel table. Performs the full
    /// wrapper check sequence (peer-binding, idle window) per the
    /// `Bearer-token authentication wrapper` spec. Does NOT yet bump
    /// `in_flight` — call [`Self::enter_call`] for that after dispatch
    /// has matched the verb.
    ///
    /// On `Ok` the kernel session's `last_activity_unix_time` is
    /// reset to `now_unix` and the resolved [`Role`] is returned for
    /// `min_role` capability checks.
    pub fn resolve(
        &self,
        kernel_table: &mut SessionTable,
        token: &Token,
        peer: &PeerIdentity,
        now_unix: u64,
    ) -> Result<(SessionId, Role), SideTableError> {
        let entry = self
            .entries
            .get(token)
            .ok_or(SideTableError::UnknownToken)?;
        if &entry.peer != peer {
            return Err(SideTableError::PeerMismatch);
        }
        // Look up the kernel session.
        let session = kernel_table
            .lookup(entry.session_id)
            .map_err(|_| SideTableError::KernelStale)?;
        // Idle check uses kernel session's last_activity. In-flight
        // entries (refcount > 0) bypass the idle check — the spec
        // requires "long-running call survives expiry mid-flight".
        if entry.in_flight == 0 {
            let elapsed = now_unix.saturating_sub(session.last_activity_unix_time);
            let threshold = self.config.idle.for_role(session.role);
            if elapsed > threshold {
                return Err(SideTableError::IdleExpired);
            }
        }
        let role = session.role;
        let session_id = entry.session_id;
        // Reset the idle clock on success — sliding TTL.
        let _ = kernel_table.reset_idle(session_id, now_unix);
        Ok((session_id, role))
    }

    /// Bump `in_flight` for a successfully-resolved token. Pair every
    /// call with [`Self::leave_call`].
    pub fn enter_call(&mut self, token: &Token) {
        if let Some(entry) = self.entries.get_mut(token) {
            entry.in_flight = entry.in_flight.saturating_add(1);
        }
    }

    /// Decrement `in_flight`. Saturating-safe.
    pub fn leave_call(&mut self, token: &Token) {
        if let Some(entry) = self.entries.get_mut(token) {
            entry.in_flight = entry.in_flight.saturating_sub(1);
        }
    }

    /// Read in-flight count for diagnostics / tests.
    pub fn in_flight(&self, token: &Token) -> u32 {
        self.entries.get(token).map(|e| e.in_flight).unwrap_or(0)
    }

    /// Voluntary `logout`. Releases the kernel slot and removes the
    /// per-identity counter entry. Idempotent — a token that was
    /// already swept by the kernel idle sweeper still cleans up the
    /// side-table.
    pub fn logout(
        &mut self,
        kernel_table: &mut SessionTable,
        token: &Token,
    ) -> Result<(), SideTableError> {
        let entry = self
            .entries
            .remove(token)
            .ok_or(SideTableError::UnknownToken)?;
        // Kernel `release` may already have happened (idle sweeper).
        let _ = kernel_table.release(entry.session_id);
        Self::dec_peer(&mut self.per_identity, &entry.peer);
        Ok(())
    }

    /// Reap any side-table entries whose kernel session has been
    /// invalidated (idle sweeper, force-logout, password-rotate
    /// invalidate-others). Returns the number of side-table entries
    /// removed. Cooperative — the wrapper SHALL call this between
    /// admin requests.
    pub fn reap(&mut self, kernel_table: &SessionTable) -> usize {
        let stale: Vec<Token> = self
            .entries
            .iter()
            .filter(|(_, e)| kernel_table.lookup(e.session_id).is_err())
            .map(|(t, _)| *t)
            .collect();
        let n = stale.len();
        for t in &stale {
            if let Some(e) = self.entries.remove(t) {
                Self::dec_peer(&mut self.per_identity, &e.peer);
            }
        }
        n
    }

    /// Force-evict every Zenoh session bound to `peer`. Used by
    /// administrative `users/disable` flows (Phase 8+) and by tests.
    pub fn evict_peer(&mut self, kernel_table: &mut SessionTable, peer: &PeerIdentity) -> usize {
        let tokens: Vec<Token> = self
            .entries
            .iter()
            .filter(|(_, e)| &e.peer == peer)
            .map(|(t, _)| *t)
            .collect();
        let n = tokens.len();
        for t in &tokens {
            if let Some(e) = self.entries.remove(t) {
                let _ = kernel_table.release(e.session_id);
            }
        }
        self.per_identity.remove(peer);
        n
    }

    /// Number of live sessions per peer — exposed for tests.
    pub fn per_identity_view(&self) -> alloc::vec::Vec<(PeerIdentity, u32)> {
        self.per_identity.iter().map(|(p, c)| (*p, *c)).collect()
    }

    fn dec_peer(map: &mut BTreeMap<PeerIdentity, u32>, peer: &PeerIdentity) {
        let drop = if let Some(slot) = map.get_mut(peer) {
            *slot = slot.saturating_sub(1);
            *slot == 0
        } else {
            false
        };
        if drop {
            map.remove(peer);
        }
    }
}

/// Convert a 16-byte token to its 32-character lowercase hex form.
pub fn token_hex(token: &Token) -> TokenHex {
    let mut out = [0u8; 32];
    for (i, b) in token.iter().enumerate() {
        let hi = b >> 4;
        let lo = b & 0x0F;
        out[i * 2] = nibble(hi);
        out[i * 2 + 1] = nibble(lo);
    }
    out
}

/// Parse a 32-character lowercase hex string back into a 16-byte
/// token. Returns `None` on any non-hex character or wrong length.
pub fn token_from_hex(hex: &[u8]) -> Option<Token> {
    if hex.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for i in 0..16 {
        let hi = decode_nibble(hex[i * 2])?;
        let lo = decode_nibble(hex[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

const fn nibble(n: u8) -> u8 {
    if n < 10 {
        b'0' + n
    } else {
        b'a' + (n - 10)
    }
}

const fn decode_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn dummy_session(uid: u32, role: Role, now: u64) -> Session {
        Session::new(uid, b"alice", role, now, false).unwrap()
    }

    fn peer(byte: u8) -> PeerIdentity {
        [byte; 32]
    }

    fn token(byte: u8) -> Token {
        [byte; 16]
    }

    fn fresh_setup() -> (ZenohSessionTable, SessionTable) {
        (
            ZenohSessionTable::new(CapsConfig::defaults()),
            SessionTable::new(),
        )
    }

    #[test]
    fn admit_logs_in_and_increments_counters() {
        let (mut z, mut k) = fresh_setup();
        let s = dummy_session(0, Role::Root, 1_000);
        let p = peer(0xAA);
        let t = token(0x01);
        let (rt, _id) = z.try_admit_login(&mut k, s, p, t, 1_000).unwrap();
        assert_eq!(rt, t);
        assert_eq!(z.len(), 1);
        assert_eq!(z.count_for(&p), 1);
        assert_eq!(k.live_count(), 1);
    }

    #[test]
    fn per_identity_cap_enforced_at_5th_login() {
        let (mut z, mut k) = fresh_setup();
        let p = peer(0xBB);
        for i in 0..4u8 {
            let s = dummy_session(i.into(), Role::Operator, 1_000);
            z.try_admit_login(&mut k, s, p, token(i), 1_000).unwrap();
        }
        // Fifth must reject with PerIdentityCapReached.
        let s = dummy_session(99, Role::Operator, 1_000);
        let err = z
            .try_admit_login(&mut k, s, p, token(0x55), 1_000)
            .unwrap_err();
        assert_eq!(err, SideTableError::PerIdentityCapReached);
        assert_eq!(z.count_for(&p), 4);
    }

    #[test]
    fn total_cap_enforced_at_17th_session() {
        let (mut z, mut k) = fresh_setup();
        // 16 sessions across 4 peers (4 each).
        for peer_idx in 0..4u8 {
            let p = peer(0x10 + peer_idx);
            for j in 0..4u8 {
                let s = dummy_session((peer_idx * 4 + j) as u32, Role::Viewer, 1_000);
                z.try_admit_login(&mut k, s, p, token(peer_idx * 4 + j), 1_000)
                    .unwrap();
            }
        }
        assert_eq!(z.len(), 16);
        // 17th must hit total cap (a fifth peer with one session).
        let p17 = peer(0x77);
        let s = dummy_session(99, Role::Viewer, 1_000);
        let err = z
            .try_admit_login(&mut k, s, p17, token(0xFF), 1_000)
            .unwrap_err();
        assert_eq!(err, SideTableError::TotalCapReached);
    }

    #[test]
    fn resolve_rejects_unknown_token() {
        let (z, mut k) = fresh_setup();
        let err = z
            .resolve(&mut k, &token(0xCC), &peer(0xDD), 100)
            .unwrap_err();
        assert_eq!(err, SideTableError::UnknownToken);
    }

    #[test]
    fn resolve_rejects_wrong_peer() {
        let (mut z, mut k) = fresh_setup();
        let s = dummy_session(0, Role::Root, 1_000);
        let p = peer(0xAA);
        let t = token(0x01);
        z.try_admit_login(&mut k, s, p, t, 1_000).unwrap();
        let other = peer(0xBB);
        let err = z.resolve(&mut k, &t, &other, 1_000).unwrap_err();
        assert_eq!(err, SideTableError::PeerMismatch);
    }

    #[test]
    fn resolve_rejects_idle_expired() {
        let (mut z, mut k) = fresh_setup();
        let s = dummy_session(0, Role::Root, 0);
        let p = peer(0xAA);
        let t = token(0x01);
        z.try_admit_login(&mut k, s, p, t, 0).unwrap();
        // Root idle = 15 min = 900 s.
        let err = z.resolve(&mut k, &t, &p, 901).unwrap_err();
        assert_eq!(err, SideTableError::IdleExpired);
    }

    #[test]
    fn resolve_resets_idle_on_success_sliding_ttl() {
        let (mut z, mut k) = fresh_setup();
        let s = dummy_session(0, Role::Root, 0);
        let p = peer(0xAA);
        let t = token(0x01);
        z.try_admit_login(&mut k, s, p, t, 0).unwrap();
        // 14 minutes in: still inside the 15-minute Root window.
        let (id, role) = z.resolve(&mut k, &t, &p, 14 * 60).unwrap();
        assert_eq!(role, Role::Root);
        // Now another 14 minutes later — must succeed because the
        // first call reset the idle clock.
        let (_, role2) = z.resolve(&mut k, &t, &p, 14 * 60 + 14 * 60).unwrap();
        assert_eq!(role2, Role::Root);
        // Sanity: confirm the kernel session's last_activity advanced.
        let session = k.lookup(id).unwrap();
        assert_eq!(session.last_activity_unix_time, 14 * 60 + 14 * 60);
    }

    #[test]
    fn in_flight_bypasses_idle_check() {
        let (mut z, mut k) = fresh_setup();
        let s = dummy_session(0, Role::Root, 0);
        let p = peer(0xAA);
        let t = token(0x01);
        z.try_admit_login(&mut k, s, p, t, 0).unwrap();
        // Open a long-running call: bump in_flight before the idle
        // window expires.
        z.enter_call(&t);
        // Try to resolve well past expiry — should still succeed.
        let (_, _) = z.resolve(&mut k, &t, &p, 100_000).unwrap();
        z.leave_call(&t);
        // After the call ends, idle check resumes.
        let err = z.resolve(&mut k, &t, &p, 100_000 + 901).unwrap_err();
        assert_eq!(err, SideTableError::IdleExpired);
    }

    #[test]
    fn logout_releases_kernel_slot_and_decrements_peer() {
        let (mut z, mut k) = fresh_setup();
        let s = dummy_session(0, Role::Root, 0);
        let p = peer(0xAA);
        let t = token(0x01);
        z.try_admit_login(&mut k, s, p, t, 0).unwrap();
        assert_eq!(k.live_count(), 1);
        z.logout(&mut k, &t).unwrap();
        assert_eq!(k.live_count(), 0);
        assert_eq!(z.count_for(&p), 0);
        assert!(z.is_empty());
    }

    #[test]
    fn logout_unknown_token_errs() {
        let (mut z, mut k) = fresh_setup();
        let err = z.logout(&mut k, &token(0xFF)).unwrap_err();
        assert_eq!(err, SideTableError::UnknownToken);
    }

    #[test]
    fn reap_removes_side_entries_for_swept_kernel_sessions() {
        let (mut z, mut k) = fresh_setup();
        let s = dummy_session(0, Role::Root, 0);
        let p = peer(0xAA);
        let t = token(0x01);
        let (_, id) = z.try_admit_login(&mut k, s, p, t, 0).unwrap();
        // Simulate the idle sweeper releasing the kernel slot.
        k.release(id).unwrap();
        // Side-table still has the entry.
        assert_eq!(z.len(), 1);
        let n = z.reap(&k);
        assert_eq!(n, 1);
        assert_eq!(z.len(), 0);
        assert_eq!(z.count_for(&p), 0);
    }

    #[test]
    fn evict_peer_kills_all_sessions_for_one_identity() {
        let (mut z, mut k) = fresh_setup();
        let p1 = peer(0x01);
        let p2 = peer(0x02);
        for j in 0..3u8 {
            let s = dummy_session(j.into(), Role::Operator, 0);
            z.try_admit_login(&mut k, s, p1, token(j), 0).unwrap();
        }
        let s = dummy_session(10, Role::Operator, 0);
        z.try_admit_login(&mut k, s, p2, token(10), 0).unwrap();
        assert_eq!(z.len(), 4);
        let n = z.evict_peer(&mut k, &p1);
        assert_eq!(n, 3);
        assert_eq!(z.len(), 1);
        assert_eq!(z.count_for(&p1), 0);
        assert_eq!(z.count_for(&p2), 1);
    }

    #[test]
    fn token_hex_round_trips() {
        // Fixture pulled from `zenoh_test_vectors.rs` (CodeQL-ignored
        // path) so the inline hex byte array doesn't trip the
        // `rust/hard-coded-cryptographic-value` query.
        let t = crate::surface_zenoh::zenoh_test_vectors::SAMPLE_TOKEN_PATTERNED;
        let hex = token_hex(&t);
        assert_eq!(
            core::str::from_utf8(&hex).unwrap(),
            crate::surface_zenoh::zenoh_test_vectors::SAMPLE_TOKEN_PATTERNED_HEX
        );
        let back = token_from_hex(&hex).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn token_from_hex_rejects_bad_input() {
        assert!(token_from_hex(b"not-32-chars").is_none());
        let mut bad = [b'0'; 32];
        bad[0] = b'!';
        assert!(token_from_hex(&bad).is_none());
    }

    #[test]
    fn token_from_hex_accepts_uppercase() {
        let t = token_from_hex(b"0123456789ABCDEFFEDCBA9876543210").unwrap();
        assert_eq!(t[0], 0x01);
        assert_eq!(t[7], 0xEF);
    }

    #[test]
    fn idle_thresholds_per_role() {
        let i = IdleThresholds::defaults();
        assert_eq!(i.for_role(Role::Root), 900);
        assert_eq!(i.for_role(Role::Operator), 3600);
        assert_eq!(i.for_role(Role::Viewer), 3600);
    }

    #[test]
    fn caps_default_values_match_spec() {
        let c = CapsConfig::defaults();
        assert_eq!(c.per_identity_max, 4);
        assert_eq!(c.total_max, 16);
    }

    #[test]
    fn set_config_does_not_evict_existing() {
        let (mut z, mut k) = fresh_setup();
        let s = dummy_session(0, Role::Root, 0);
        let p = peer(0xAA);
        z.try_admit_login(&mut k, s, p, token(0x01), 0).unwrap();
        let mut c = *z.config();
        c.per_identity_max = 1;
        z.set_config(c);
        // Existing session retained.
        assert_eq!(z.len(), 1);
        // But a brand-new one for the same peer rejected.
        let s2 = dummy_session(1, Role::Root, 0);
        let err = z
            .try_admit_login(&mut k, s2, p, token(0x02), 0)
            .unwrap_err();
        assert_eq!(err, SideTableError::PerIdentityCapReached);
    }

    #[test]
    fn concurrent_admit_releases_slot_back_when_kernel_full() {
        let (mut z, mut k) = fresh_setup();
        // Saturate the kernel table from a non-Zenoh source so the
        // 32nd kernel slot is full.
        for i in 0..32u32 {
            let s = dummy_session(i, Role::Viewer, 0);
            k.acquire(s).unwrap();
        }
        // Zenoh side caps allow this, but kernel rejects.
        let s = dummy_session(99, Role::Root, 0);
        let err = z
            .try_admit_login(&mut k, s, peer(0x33), token(0x44), 0)
            .unwrap_err();
        assert_eq!(err, SideTableError::KernelTableFull);
        assert_eq!(z.len(), 0);
    }

    #[test]
    fn audit_identity_rendering_for_zenoh_peer() {
        // Smoke test that AuditIdentity from the surface module works
        // for Zenoh.
        let id = crate::surface::AuditIdentity {
            role: "operator",
            user: Some("bob".to_string()),
            surface: crate::surface::SurfaceKind::Zenoh,
        };
        assert_eq!(id.render(), "operator:bob@zenoh");
    }
}
