// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Fixed-capacity in-kernel session table.
//!
//! Spec: `openspec/changes/management-login-v1/specs/kernel-syscalls/spec.md`
//! Requirement "Session table capacity + lifecycle".
//!
//! The table holds at most [`SESSION_TABLE_CAP`] live sessions in a flat
//! `[Option<Session>; N]` array — no `alloc` is touched on the hot login
//! path. A 24-bit per-slot generation counter is bundled with each slot
//! index into a [`SessionId`] so that a stale id from a prior incarnation
//! cannot accidentally name a freshly-allocated session (ABA-safety).
//!
//! ## Why fixed-size
//!
//! - Kernel code runs `#![no_std]`; the boot path allocates the table as
//!   one slab and never grows it.
//! - The hot login/logout/whoami paths must be O(1) wrt the number of
//!   live sessions.
//! - A bounded table makes the formal model in
//!   `formal/tla/SessionState.tla` tractable.
//!
//! ## Username storage
//!
//! Usernames are stored inline as `[u8; MAX_USERNAME_LEN]` plus a length
//! byte rather than `String` / `heapless::Vec` to keep the kernel free of
//! transitive third-party deps and the `alloc` allocator.

use super::Role;

/// Maximum number of concurrently active sessions across the box.
pub const SESSION_TABLE_CAP: usize = 32;

/// Maximum username length stored inline in a [`Session`].
///
/// Matches the spec's "Account name length" cap (`auth-shadow` Q4) and
/// keeps each `Session` slot trivially `Copy` for in-place updates.
pub const MAX_USERNAME_LEN: usize = 64;

/// 24-bit generation counter mask. The high byte of a [`SessionId`] is
/// the slot index, the low 24 bits are the per-slot generation.
const GEN_MASK: u32 = 0x00FF_FFFF;

/// A handle returned to userspace identifying an installed session.
///
/// Encoding: `(slot_index << 24) | (generation & 0x00FF_FFFF)`.
///
/// Generations roll over after `2^24` lifetimes — the table releases
/// slots only on logout / sweep, so this affords ~16M reuse cycles
/// before a roll-over loses ABA-safety. At one login/logout per second
/// the roll-over horizon is over half a year.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SessionId(u32);

impl SessionId {
    /// Sentinel for "no session installed". The high byte
    /// (slot-index) is `0xFF`, which is one past the table capacity
    /// (`SESSION_TABLE_CAP = 32`), so a valid SessionId can never
    /// equal this value.
    pub const NONE: SessionId = SessionId(0xFFFF_FFFF);

    /// Encode a `(slot, generation)` pair.
    fn encode(slot: usize, generation: u32) -> Self {
        debug_assert!(slot < SESSION_TABLE_CAP);
        Self(((slot as u32) << 24) | (generation & GEN_MASK))
    }

    /// Slot index in the table (0..[`SESSION_TABLE_CAP`]). Returns
    /// `None` for [`SessionId::NONE`].
    pub fn slot(self) -> Option<usize> {
        let s = (self.0 >> 24) as usize;
        if s >= SESSION_TABLE_CAP {
            None
        } else {
            Some(s)
        }
    }

    /// Generation counter (low 24 bits).
    pub fn generation(self) -> u32 {
        self.0 & GEN_MASK
    }

    /// Raw `u32` form for the syscall ABI.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Decode a raw u32 from userspace into a SessionId. No validation
    /// is performed; callers SHALL pair this with a [`SessionTable`]
    /// `lookup` that re-checks the (slot, generation) tuple.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Whether this id is the [`Self::NONE`] sentinel.
    pub const fn is_none(self) -> bool {
        self.0 == Self::NONE.0
    }
}

/// One authenticated session.
///
/// Held inline in the table — no heap pointers — so the whole struct is
/// trivially memcpy-able. The 256-bit `peer_identity` slot is reserved
/// for the TLS certificate fingerprint that Phase 7 (Zenoh admin) will
/// stamp; until then it is `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Session {
    /// Stable user identifier — the index into the loaded shadow table.
    pub user_id: u32,
    /// Inline username buffer; only `username[..username_len]` is valid.
    pub username: [u8; MAX_USERNAME_LEN],
    /// Number of valid bytes in `username`.
    pub username_len: u8,
    /// Cached role from the shadow record at login time.
    pub role: Role,
    /// Login time in Unix seconds.
    pub login_unix_time: u64,
    /// Last activity time in Unix seconds, used by the idle sweeper.
    pub last_activity_unix_time: u64,
    /// TLS peer fingerprint placeholder. Filled in Phase 7 for remote
    /// (Zenoh) sessions; left `None` for console / unikernel-local
    /// sessions.
    pub peer_identity: Option<[u8; 32]>,
    /// Bit flags reserved for future use (e.g. `WAS_AUDITED`).
    pub flags: u32,
    /// When the shadow record carried `flags &
    /// FLAG_MUST_CHANGE_PASSWORD`, the kernel SHALL refuse all syscalls
    /// other than `auth_change_password` / `auth_whoami` /
    /// `auth_logout` until this flag is cleared by a successful
    /// password rotation.
    pub must_change_password: bool,
}

impl Session {
    /// Borrow the username as a byte slice (no UTF-8 validation).
    pub fn username_bytes(&self) -> &[u8] {
        let n = core::cmp::min(self.username_len as usize, MAX_USERNAME_LEN);
        &self.username[..n]
    }

    /// Builder that copies `name` into the inline buffer. Returns
    /// `None` if `name.len() > MAX_USERNAME_LEN`.
    pub fn new(
        user_id: u32,
        name: &[u8],
        role: Role,
        now_unix: u64,
        must_change_password: bool,
    ) -> Option<Self> {
        if name.len() > MAX_USERNAME_LEN {
            return None;
        }
        let mut buf = [0u8; MAX_USERNAME_LEN];
        buf[..name.len()].copy_from_slice(name);
        Some(Self {
            user_id,
            username: buf,
            username_len: name.len() as u8,
            role,
            login_unix_time: now_unix,
            last_activity_unix_time: now_unix,
            peer_identity: None,
            flags: 0,
            must_change_password,
        })
    }
}

/// Errors raised by [`SessionTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionError {
    /// The table is at capacity ([`SESSION_TABLE_CAP`] live sessions).
    TableFull,
    /// The supplied [`SessionId`] does not name a live session, either
    /// because the slot is empty or the generation no longer matches.
    Stale,
    /// Username exceeded the inline buffer cap ([`MAX_USERNAME_LEN`]).
    UsernameTooLong,
}

/// Per-slot record. The generation counter is bumped on every release
/// so that holders of a freed [`SessionId`] cannot accidentally name a
/// new session.
#[derive(Debug, Clone, Copy)]
struct Slot {
    session: Option<Session>,
    generation: u32,
}

impl Slot {
    const fn empty() -> Self {
        Self {
            session: None,
            generation: 0,
        }
    }
}

/// Fixed-capacity session table.
///
/// Owned by `kernel::auth::CURRENT_AUTH_STATE` (one global instance).
/// Tests construct their own freestanding tables.
#[derive(Debug)]
pub struct SessionTable {
    slots: [Slot; SESSION_TABLE_CAP],
    live_count: usize,
}

impl Default for SessionTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionTable {
    /// Construct an empty session table. `const fn` so the kernel can
    /// embed it in a static.
    pub const fn new() -> Self {
        Self {
            slots: [Slot::empty(); SESSION_TABLE_CAP],
            live_count: 0,
        }
    }

    /// Number of live sessions.
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Try to allocate a slot for a fresh session. Returns the
    /// [`SessionId`] handle on success.
    pub fn acquire(&mut self, session: Session) -> Result<SessionId, SessionError> {
        for (idx, slot) in self.slots.iter_mut().enumerate() {
            if slot.session.is_none() {
                slot.session = Some(session);
                self.live_count += 1;
                return Ok(SessionId::encode(idx, slot.generation));
            }
        }
        Err(SessionError::TableFull)
    }

    /// Release the session named by `id`. Returns `Ok(())` if the slot
    /// was alive and is now free, [`SessionError::Stale`] otherwise.
    /// Bumps the slot's generation counter so old `id`s cannot rebind.
    pub fn release(&mut self, id: SessionId) -> Result<Session, SessionError> {
        let slot = id.slot().ok_or(SessionError::Stale)?;
        let slot_ref = &mut self.slots[slot];
        if slot_ref.generation != id.generation() {
            return Err(SessionError::Stale);
        }
        let session = slot_ref.session.take().ok_or(SessionError::Stale)?;
        slot_ref.generation = slot_ref.generation.wrapping_add(1) & GEN_MASK;
        self.live_count -= 1;
        Ok(session)
    }

    /// Fetch a `&Session` by id without mutating it. Returns
    /// [`SessionError::Stale`] if the (slot, generation) no longer
    /// matches.
    pub fn lookup(&self, id: SessionId) -> Result<&Session, SessionError> {
        let slot = id.slot().ok_or(SessionError::Stale)?;
        let slot_ref = &self.slots[slot];
        if slot_ref.generation != id.generation() {
            return Err(SessionError::Stale);
        }
        slot_ref.session.as_ref().ok_or(SessionError::Stale)
    }

    /// Fetch a `&mut Session` by id, used by [`Self::reset_idle`] and
    /// the password-rotation handler that clears
    /// `must_change_password`.
    pub fn lookup_mut(&mut self, id: SessionId) -> Result<&mut Session, SessionError> {
        let slot = id.slot().ok_or(SessionError::Stale)?;
        let slot_ref = &mut self.slots[slot];
        if slot_ref.generation != id.generation() {
            return Err(SessionError::Stale);
        }
        slot_ref.session.as_mut().ok_or(SessionError::Stale)
    }

    /// Convenience: stamp `last_activity_unix_time = now`. Used by the
    /// dispatch path on every successful syscall.
    pub fn reset_idle(&mut self, id: SessionId, now_unix: u64) -> Result<(), SessionError> {
        let session = self.lookup_mut(id)?;
        session.last_activity_unix_time = now_unix;
        Ok(())
    }

    /// Convenience: return the session role for capability checks.
    pub fn current_role(&self, id: SessionId) -> Option<Role> {
        self.lookup(id).ok().map(|s| s.role)
    }

    /// Iterate over all live `(SessionId, &Session)` pairs. Used by the
    /// idle sweeper.
    pub fn iter(&self) -> SessionTableIter<'_> {
        SessionTableIter {
            table: self,
            next: 0,
        }
    }

    /// Force-release every session belonging to `user_id`. Returns the
    /// number of sessions invalidated. Used by the cross-rotate
    /// password change handler ("force logout other sessions for the
    /// same user").
    pub fn invalidate_user(&mut self, user_id: u32) -> usize {
        let mut count = 0;
        for slot_ref in self.slots.iter_mut() {
            if let Some(s) = slot_ref.session.as_ref() {
                if s.user_id == user_id {
                    slot_ref.session = None;
                    slot_ref.generation = slot_ref.generation.wrapping_add(1) & GEN_MASK;
                    count += 1;
                }
            }
        }
        self.live_count -= count;
        count
    }
}

/// Iterator over live `(SessionId, &Session)` pairs.
pub struct SessionTableIter<'a> {
    table: &'a SessionTable,
    next: usize,
}

impl<'a> Iterator for SessionTableIter<'a> {
    type Item = (SessionId, &'a Session);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < SESSION_TABLE_CAP {
            let slot_idx = self.next;
            self.next += 1;
            let slot = &self.table.slots[slot_idx];
            if let Some(s) = slot.session.as_ref() {
                return Some((SessionId::encode(slot_idx, slot.generation), s));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_session(uid: u32, role: Role) -> Session {
        Session::new(uid, b"alice", role, 1_700_000_000, false).unwrap()
    }

    #[test]
    fn acquire_and_release_round_trip() {
        let mut tbl = SessionTable::new();
        assert_eq!(tbl.live_count(), 0);

        let id = tbl.acquire(mk_session(1, Role::Operator)).unwrap();
        assert_eq!(tbl.live_count(), 1);
        assert_eq!(tbl.lookup(id).unwrap().user_id, 1);

        tbl.release(id).unwrap();
        assert_eq!(tbl.live_count(), 0);

        // Stale id rejected.
        assert_eq!(tbl.lookup(id), Err(SessionError::Stale));
        assert_eq!(tbl.release(id), Err(SessionError::Stale));
    }

    #[test]
    fn aba_safety_after_release() {
        let mut tbl = SessionTable::new();
        let id1 = tbl.acquire(mk_session(1, Role::Root)).unwrap();
        let slot1 = id1.slot().unwrap();
        tbl.release(id1).unwrap();

        let id2 = tbl.acquire(mk_session(2, Role::Viewer)).unwrap();
        assert_eq!(id2.slot(), Some(slot1)); // reused slot
        assert_ne!(id2.generation(), id1.generation()); // different generation

        // Old id MUST NOT name the freshly-allocated session.
        assert_eq!(tbl.lookup(id1), Err(SessionError::Stale));
        // New id works.
        assert_eq!(tbl.lookup(id2).unwrap().user_id, 2);
    }

    #[test]
    fn table_full_returns_table_full() {
        let mut tbl = SessionTable::new();
        let mut ids = alloc::vec::Vec::new();
        for i in 0..SESSION_TABLE_CAP {
            ids.push(tbl.acquire(mk_session(i as u32, Role::Viewer)).unwrap());
        }
        assert_eq!(tbl.live_count(), SESSION_TABLE_CAP);
        assert_eq!(
            tbl.acquire(mk_session(99, Role::Viewer)),
            Err(SessionError::TableFull)
        );
    }

    #[test]
    fn reset_idle_updates_last_activity() {
        let mut tbl = SessionTable::new();
        let id = tbl.acquire(mk_session(1, Role::Operator)).unwrap();
        let before = tbl.lookup(id).unwrap().last_activity_unix_time;

        tbl.reset_idle(id, before + 60).unwrap();
        assert_eq!(tbl.lookup(id).unwrap().last_activity_unix_time, before + 60);
        // login time SHOULD NOT change.
        assert_eq!(tbl.lookup(id).unwrap().login_unix_time, 1_700_000_000);
    }

    #[test]
    fn current_role_returns_session_role() {
        let mut tbl = SessionTable::new();
        let id = tbl.acquire(mk_session(1, Role::Operator)).unwrap();
        assert_eq!(tbl.current_role(id), Some(Role::Operator));
    }

    #[test]
    fn current_role_returns_none_for_stale() {
        let mut tbl = SessionTable::new();
        let id = tbl.acquire(mk_session(1, Role::Root)).unwrap();
        tbl.release(id).unwrap();
        assert_eq!(tbl.current_role(id), None);
    }

    #[test]
    fn iter_visits_only_live_slots() {
        let mut tbl = SessionTable::new();
        let id_a = tbl.acquire(mk_session(1, Role::Operator)).unwrap();
        let _id_b = tbl.acquire(mk_session(2, Role::Viewer)).unwrap();
        let id_c = tbl.acquire(mk_session(3, Role::Root)).unwrap();
        tbl.release(id_a).unwrap();

        let live: alloc::vec::Vec<u32> = tbl.iter().map(|(_, s)| s.user_id).collect();
        assert_eq!(live.len(), 2);
        assert!(live.contains(&2));
        assert!(live.contains(&3));

        // SessionId from iter is usable for lookup.
        let mut found_c = false;
        for (sid, s) in tbl.iter() {
            if s.user_id == 3 {
                assert_eq!(sid, id_c);
                found_c = true;
            }
        }
        assert!(found_c);
    }

    #[test]
    fn invalidate_user_drops_all_their_sessions_only() {
        let mut tbl = SessionTable::new();
        // Three sessions for user 7, two for user 8.
        let _ = tbl.acquire(mk_session(7, Role::Operator)).unwrap();
        let _ = tbl.acquire(mk_session(7, Role::Operator)).unwrap();
        let _ = tbl.acquire(mk_session(8, Role::Viewer)).unwrap();
        let _ = tbl.acquire(mk_session(7, Role::Operator)).unwrap();
        let _ = tbl.acquire(mk_session(8, Role::Viewer)).unwrap();
        assert_eq!(tbl.live_count(), 5);

        let killed = tbl.invalidate_user(7);
        assert_eq!(killed, 3);
        assert_eq!(tbl.live_count(), 2);

        for (_id, s) in tbl.iter() {
            assert_ne!(s.user_id, 7, "user 7 sessions must be gone");
        }
    }

    #[test]
    fn session_id_none_is_unrepresentable() {
        // NONE.slot() should return None — out of range.
        assert!(SessionId::NONE.is_none());
        assert_eq!(SessionId::NONE.slot(), None);
    }

    #[test]
    fn session_id_round_trips_via_raw() {
        let mut tbl = SessionTable::new();
        let id = tbl.acquire(mk_session(42, Role::Root)).unwrap();
        let raw = id.as_u32();
        let id2 = SessionId::from_raw(raw);
        assert_eq!(id, id2);
        assert_eq!(tbl.lookup(id2).unwrap().user_id, 42);
    }

    #[test]
    fn username_too_long_rejected() {
        let too_long = [b'a'; MAX_USERNAME_LEN + 1];
        assert!(Session::new(0, &too_long, Role::Viewer, 0, false).is_none());
    }

    #[test]
    fn username_at_cap_accepted_and_round_trips() {
        let exact = [b'a'; MAX_USERNAME_LEN];
        let s = Session::new(0, &exact, Role::Viewer, 0, false).unwrap();
        assert_eq!(s.username_bytes().len(), MAX_USERNAME_LEN);
        assert_eq!(s.username_bytes(), &exact[..]);
    }
}
