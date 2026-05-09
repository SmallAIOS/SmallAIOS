// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-name advisory lock table for the overlay write path.
//!
//! Spec: `openspec/changes/embedded-overlay-v1/specs/fs-overlay-syscalls/spec.md`
//! Requirement "model_add syscall" — "acquires the per-name advisory
//! lock". Also `fs-overlay-write` Requirement "Per-name advisory lock"
//! and "Aborted-add cleanup".
//!
//! ## Design
//!
//! Each in-flight `model_add` holds an advisory lock on the model
//! `name` for the duration of the upload. A second concurrent caller
//! on the same name immediately fails with `-EBUSY` (no blocking, no
//! retry inside the kernel — the operator-supplied client is expected
//! to back off and retry).
//!
//! Locks are held by `SessionId`. When a session is released (logout,
//! idle sweep, table flush) every lock the session was holding SHALL
//! be released and the matching `<name>.tmp` staging file SHALL be
//! unlinked from the upper layer (the "aborted-add cleanup" path).
//!
//! ## Storage
//!
//! Fixed-size `[Option<NameLock>; OVERLAY_LOCK_TABLE_CAP]`. No `alloc`
//! on the hot path. The default capacity (16 slots) is well above the
//! design-time concurrent-upload ceiling (the unikernel's session
//! table is capped at 32, and per-session concurrency is one — Zenoh
//! admin runs sequentially per peer); raise the cap if the cap is
//! ever observed contended.
//!
//! ## Why `[u8; 256]` for the name?
//!
//! Names are validated to `len() in 1..=255` at the syscall boundary
//! (NUL-terminated for the 256th byte). Inline storage avoids `alloc`
//! and matches the F2FS upper-layer's filename limit.

use super::SessionId;

/// Maximum simultaneous in-flight model uploads.
pub const OVERLAY_LOCK_TABLE_CAP: usize = 16;

/// Maximum bytes of a name (matches POSIX `NAME_MAX = 255` plus one
/// byte for the kernel's internal NUL guard).
pub const OVERLAY_NAME_MAX: usize = 256;

/// One held lock — the name being uploaded plus the session that
/// holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameLock {
    /// Inline name buffer; `name[..name_len]` is valid.
    pub name: [u8; OVERLAY_NAME_MAX],
    /// Number of bytes valid in `name`.
    pub name_len: u8,
    /// Session that holds the lock.
    pub holder: SessionId,
}

impl NameLock {
    /// Borrow the active name slice.
    pub fn name_bytes(&self) -> &[u8] {
        let n = core::cmp::min(self.name_len as usize, OVERLAY_NAME_MAX);
        &self.name[..n]
    }
}

/// Errors raised by [`OverlayLockTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayLockError {
    /// Name exceeds [`OVERLAY_NAME_MAX`].
    NameTooLong,
    /// Lock for `name` already held (-EBUSY at the syscall boundary).
    AlreadyHeld,
    /// All slots in the table are full.
    TableFull,
    /// Caller asked to release a lock it does NOT hold (or that
    /// doesn't exist).
    NotHeld,
}

/// Per-name advisory lock table. Owned globally by the kernel's
/// overlay-syscall layer (one instance), constructed inline by tests.
#[derive(Debug)]
pub struct OverlayLockTable {
    slots: [Option<NameLock>; OVERLAY_LOCK_TABLE_CAP],
}

impl Default for OverlayLockTable {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlayLockTable {
    /// Construct an empty lock table.
    pub const fn new() -> Self {
        Self {
            slots: [None; OVERLAY_LOCK_TABLE_CAP],
        }
    }

    /// Number of currently-held locks.
    pub fn live_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// `true` iff a lock for `name` is currently held by anyone.
    pub fn is_held(&self, name: &[u8]) -> bool {
        self.find(name).is_some()
    }

    /// Lookup the slot index of an existing lock for `name`, if any.
    fn find(&self, name: &[u8]) -> Option<usize> {
        self.slots.iter().position(|s| match s {
            Some(lock) => lock.name_bytes() == name,
            None => false,
        })
    }

    /// Try to acquire `name` for `holder`. Returns
    /// [`OverlayLockError::AlreadyHeld`] if any session already holds
    /// it (including `holder` itself — a session SHALL NOT recursively
    /// re-acquire).
    pub fn try_acquire(&mut self, name: &[u8], holder: SessionId) -> Result<(), OverlayLockError> {
        if name.len() > OVERLAY_NAME_MAX {
            return Err(OverlayLockError::NameTooLong);
        }
        if self.find(name).is_some() {
            return Err(OverlayLockError::AlreadyHeld);
        }
        for slot in self.slots.iter_mut() {
            if slot.is_none() {
                let mut buf = [0u8; OVERLAY_NAME_MAX];
                buf[..name.len()].copy_from_slice(name);
                *slot = Some(NameLock {
                    name: buf,
                    name_len: name.len() as u8,
                    holder,
                });
                return Ok(());
            }
        }
        Err(OverlayLockError::TableFull)
    }

    /// Release a lock for `name` held by `holder`. The holder check
    /// prevents one session from clobbering another's lock.
    pub fn release(&mut self, name: &[u8], holder: SessionId) -> Result<(), OverlayLockError> {
        let idx = self.find(name).ok_or(OverlayLockError::NotHeld)?;
        let slot = &mut self.slots[idx];
        match slot {
            Some(lock) if lock.holder == holder => {
                *slot = None;
                Ok(())
            }
            _ => Err(OverlayLockError::NotHeld),
        }
    }

    /// Release every lock held by `holder`. Called by the session
    /// release path so an aborted upload (logout / idle sweep / crash)
    /// frees its lock automatically.
    ///
    /// Returns the inline names of every released lock so the caller
    /// can run the aborted-add cleanup (unlink each `<name>.tmp` from
    /// the upper layer). The slice writes the first
    /// [`OVERLAY_LOCK_TABLE_CAP`] entries; capacity-exceeded is
    /// architecturally impossible because the table caps at this size.
    pub fn release_on_session_drop(
        &mut self,
        holder: SessionId,
        out: &mut [NameLock; OVERLAY_LOCK_TABLE_CAP],
    ) -> usize {
        let mut count = 0usize;
        for slot in self.slots.iter_mut() {
            if let Some(lock) = slot {
                if lock.holder == holder {
                    out[count] = *lock;
                    count += 1;
                    *slot = None;
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid(raw: u32) -> SessionId {
        SessionId::from_raw(raw)
    }

    #[test]
    fn empty_table_starts_with_zero_holds() {
        let t = OverlayLockTable::new();
        assert_eq!(t.live_count(), 0);
        assert!(!t.is_held(b"foo.onnx"));
    }

    #[test]
    fn try_acquire_then_release_round_trip() {
        let mut t = OverlayLockTable::new();
        assert!(t.try_acquire(b"foo.onnx", sid(0x10)).is_ok());
        assert!(t.is_held(b"foo.onnx"));
        assert_eq!(t.live_count(), 1);

        assert!(t.release(b"foo.onnx", sid(0x10)).is_ok());
        assert!(!t.is_held(b"foo.onnx"));
        assert_eq!(t.live_count(), 0);
    }

    #[test]
    fn try_acquire_same_name_twice_returns_already_held() {
        let mut t = OverlayLockTable::new();
        assert!(t.try_acquire(b"foo", sid(1)).is_ok());
        assert_eq!(
            t.try_acquire(b"foo", sid(2)),
            Err(OverlayLockError::AlreadyHeld)
        );
        // Same holder also rejected (no recursion).
        assert_eq!(
            t.try_acquire(b"foo", sid(1)),
            Err(OverlayLockError::AlreadyHeld)
        );
    }

    #[test]
    fn different_names_can_be_held_concurrently() {
        let mut t = OverlayLockTable::new();
        assert!(t.try_acquire(b"foo", sid(1)).is_ok());
        assert!(t.try_acquire(b"bar", sid(2)).is_ok());
        assert!(t.try_acquire(b"baz", sid(3)).is_ok());
        assert_eq!(t.live_count(), 3);
    }

    #[test]
    fn release_with_wrong_holder_is_rejected() {
        let mut t = OverlayLockTable::new();
        assert!(t.try_acquire(b"foo", sid(1)).is_ok());
        assert_eq!(t.release(b"foo", sid(2)), Err(OverlayLockError::NotHeld));
        // Original holder can still release.
        assert!(t.release(b"foo", sid(1)).is_ok());
    }

    #[test]
    fn release_unknown_name_is_not_held() {
        let mut t = OverlayLockTable::new();
        assert_eq!(
            t.release(b"missing", sid(1)),
            Err(OverlayLockError::NotHeld)
        );
    }

    #[test]
    fn name_too_long_is_rejected() {
        let mut t = OverlayLockTable::new();
        let big = [b'A'; OVERLAY_NAME_MAX + 1];
        assert_eq!(
            t.try_acquire(&big, sid(1)),
            Err(OverlayLockError::NameTooLong)
        );
    }

    #[test]
    fn table_full_returns_table_full() {
        let mut t = OverlayLockTable::new();
        for i in 0..OVERLAY_LOCK_TABLE_CAP {
            let name = [b'x', i as u8];
            assert!(t.try_acquire(&name, sid(i as u32)).is_ok());
        }
        let extra = [b'y', 0];
        assert_eq!(
            t.try_acquire(&extra, sid(99)),
            Err(OverlayLockError::TableFull)
        );
    }

    #[test]
    fn release_on_session_drop_frees_all_locks_for_holder() {
        let mut t = OverlayLockTable::new();
        let s1 = sid(1);
        let s2 = sid(2);
        assert!(t.try_acquire(b"a", s1).is_ok());
        assert!(t.try_acquire(b"b", s1).is_ok());
        assert!(t.try_acquire(b"c", s2).is_ok());

        let mut buf = [NameLock {
            name: [0; OVERLAY_NAME_MAX],
            name_len: 0,
            holder: SessionId::NONE,
        }; OVERLAY_LOCK_TABLE_CAP];
        let n = t.release_on_session_drop(s1, &mut buf);
        assert_eq!(n, 2);
        assert!(!t.is_held(b"a"));
        assert!(!t.is_held(b"b"));
        // Other holder unaffected.
        assert!(t.is_held(b"c"));

        // The reported names cover the freed locks.
        let mut released_names = alloc::vec::Vec::new();
        for entry in buf.iter().take(n) {
            released_names.push(entry.name_bytes().to_vec());
        }
        released_names.sort();
        assert_eq!(released_names, alloc::vec![b"a".to_vec(), b"b".to_vec()]);
    }

    #[test]
    fn release_on_session_drop_no_holder_match_is_zero() {
        let mut t = OverlayLockTable::new();
        assert!(t.try_acquire(b"a", sid(1)).is_ok());
        let mut buf = [NameLock {
            name: [0; OVERLAY_NAME_MAX],
            name_len: 0,
            holder: SessionId::NONE,
        }; OVERLAY_LOCK_TABLE_CAP];
        let n = t.release_on_session_drop(sid(99), &mut buf);
        assert_eq!(n, 0);
        assert!(t.is_held(b"a"));
    }

    #[test]
    fn name_lock_inline_storage_round_trip() {
        let mut t = OverlayLockTable::new();
        assert!(t.try_acquire(b"long-name.onnx", sid(7)).is_ok());
        // The lock survives intact.
        let mut buf = [NameLock {
            name: [0; OVERLAY_NAME_MAX],
            name_len: 0,
            holder: SessionId::NONE,
        }; OVERLAY_LOCK_TABLE_CAP];
        let n = t.release_on_session_drop(sid(7), &mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf[0].name_bytes(), b"long-name.onnx");
        assert_eq!(buf[0].holder, sid(7));
    }
}
