// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Epoll-style I/O event multiplexing.
//!
//! Provides a Linux-compatible `epoll` interface for monitoring multiple file
//! descriptors for readiness events. This is the primary async I/O mechanism
//! used by AI inference runtimes (via libuv, tokio, etc.).
//!
//! Supported operations:
//! - `epoll_create1` — create an epoll instance (with optional `O_CLOEXEC`)
//! - `epoll_ctl`     — add, modify, or delete monitored file descriptors
//! - `epoll_wait`    — wait for events (currently a stub returning 0 events)
//!
//! Maximum of [`MAX_EPOLL_ENTRIES`] file descriptors per epoll instance.

use crate::errno::Errno;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum number of file descriptors monitored by a single epoll instance.
pub const MAX_EPOLL_ENTRIES: usize = 64;

/// `O_CLOEXEC` flag value, accepted by `epoll_create1`.
const O_CLOEXEC: i32 = 0o2000000;

// ─── Epoll Operations ───────────────────────────────────────────────────────

/// Operation to perform on an epoll instance via `epoll_ctl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum EpollOp {
    /// Register a new file descriptor.
    Add = 1,
    /// Modify the events for an already-registered file descriptor.
    Mod = 2,
    /// Remove a file descriptor from the interest list.
    Del = 3,
}

// ─── Epoll Flags ────────────────────────────────────────────────────────────

/// Epoll event flag constants (bitfield values OR'd into `EpollEvent::events`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpollFlags(u32);

impl EpollFlags {
    /// The associated file descriptor is available for read operations.
    pub const EPOLLIN: u32 = 0x001;
    /// The associated file descriptor is available for write operations.
    pub const EPOLLOUT: u32 = 0x004;
    /// Error condition on the associated file descriptor.
    pub const EPOLLERR: u32 = 0x008;
    /// Hang-up on the associated file descriptor.
    pub const EPOLLHUP: u32 = 0x010;
    /// Edge-triggered notification.
    pub const EPOLLET: u32 = 0x8000_0000;
    /// One-shot notification — automatically removed after one event.
    pub const EPOLLONESHOT: u32 = 0x4000_0000;

    /// Create a new `EpollFlags` from raw bits.
    pub const fn new(bits: u32) -> Self {
        Self(bits)
    }

    /// Return the raw bits.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Check whether a specific flag is set.
    pub const fn contains(self, flag: u32) -> bool {
        self.0 & flag != 0
    }
}

// ─── Epoll Event ────────────────────────────────────────────────────────────

/// An epoll event returned by `epoll_wait` or passed to `epoll_ctl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpollEvent {
    /// Bitmask of events (combination of `EpollFlags` constants).
    pub events: u32,
    /// Opaque user data associated with this event.
    pub data: u64,
}

impl EpollEvent {
    /// Create a new epoll event.
    pub const fn new(events: u32, data: u64) -> Self {
        Self { events, data }
    }
}

// ─── Epoll Entry (internal) ─────────────────────────────────────────────────

/// Internal bookkeeping entry for a monitored file descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpollEntry {
    /// The file descriptor being monitored.
    pub fd: i32,
    /// The event mask for this descriptor.
    pub events: u32,
    /// Opaque user data.
    pub data: u64,
}

// ─── Epoll Instance ─────────────────────────────────────────────────────────

/// An epoll instance, analogous to the kernel object returned by `epoll_create1`.
#[derive(Debug, PartialEq)]
pub struct EpollInstance {
    /// Monitored file descriptors. `None` indicates a free slot.
    entries: [Option<EpollEntry>; MAX_EPOLL_ENTRIES],
    /// Number of active entries.
    count: usize,
}

impl EpollInstance {
    /// Return the number of actively monitored file descriptors.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Find the slot index for a given file descriptor, if registered.
    fn find_fd(&self, fd: i32) -> Option<usize> {
        for (i, entry) in self.entries.iter().enumerate() {
            if let Some(e) = entry {
                if e.fd == fd {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Find the first free slot index.
    fn find_free_slot(&self) -> Option<usize> {
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_none() {
                return Some(i);
            }
        }
        None
    }
}

// ─── Operations ─────────────────────────────────────────────────────────────

/// Create a new epoll instance.
///
/// # Arguments
/// * `flags` — Must be `0` or `O_CLOEXEC` (0o2000000). Any other value
///   returns `EINVAL`.
///
/// # Errors
/// * `Errno::EINVAL` — Invalid flags.
pub fn epoll_create1(flags: i32) -> Result<EpollInstance, Errno> {
    if flags != 0 && flags != O_CLOEXEC {
        return Err(Errno::EINVAL);
    }

    Ok(EpollInstance {
        entries: core::array::from_fn(|_| None),
        count: 0,
    })
}

/// Control an epoll instance: add, modify, or delete a file descriptor.
///
/// # Arguments
/// * `instance` — The epoll instance to modify.
/// * `op`       — `EpollOp::Add`, `EpollOp::Mod`, or `EpollOp::Del`.
/// * `fd`       — The file descriptor to operate on.
/// * `event`    — The event specification (ignored for `Del`).
///
/// # Errors
/// * `Errno::EEXIST` — `Add` for a file descriptor already registered.
/// * `Errno::ENOENT` — `Mod` or `Del` for a file descriptor not registered.
/// * `Errno::ENOSPC` — No free slots available for `Add`.
pub fn epoll_ctl(
    instance: &mut EpollInstance,
    op: EpollOp,
    fd: i32,
    event: &EpollEvent,
) -> Result<(), Errno> {
    match op {
        EpollOp::Add => {
            // Reject duplicates.
            if instance.find_fd(fd).is_some() {
                return Err(Errno::EEXIST);
            }
            // Find a free slot.
            let slot = instance.find_free_slot().ok_or(Errno::ENOSPC)?;
            instance.entries[slot] = Some(EpollEntry {
                fd,
                events: event.events,
                data: event.data,
            });
            instance.count += 1;
            Ok(())
        }
        EpollOp::Mod => {
            let idx = instance.find_fd(fd).ok_or(Errno::ENOENT)?;
            if let Some(entry) = &mut instance.entries[idx] {
                entry.events = event.events;
                entry.data = event.data;
            }
            Ok(())
        }
        EpollOp::Del => {
            let idx = instance.find_fd(fd).ok_or(Errno::ENOENT)?;
            instance.entries[idx] = None;
            instance.count -= 1;
            Ok(())
        }
    }
}

/// Wait for events on an epoll instance (stub — always returns 0 events).
///
/// # Arguments
/// * `_instance`   — The epoll instance to poll.
/// * `_events`     — Buffer to fill with ready events.
/// * `_max_events` — Maximum number of events to return.
/// * `_timeout_ms` — Timeout in milliseconds (-1 = block, 0 = non-blocking).
///
/// # Errors
/// * `Errno::EINVAL` — `max_events` is less than or equal to zero.
pub fn epoll_wait(
    _instance: &EpollInstance,
    _events: &mut [EpollEvent],
    _max_events: i32,
    _timeout_ms: i32,
) -> Result<usize, Errno> {
    if _max_events <= 0 {
        return Err(Errno::EINVAL);
    }
    // Stub: no events are ever ready yet.
    Ok(0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── epoll_create1 ───────────────────────────────────────────────────

    #[test]
    fn create_with_zero_flags() {
        let inst = epoll_create1(0).unwrap();
        assert_eq!(inst.count(), 0);
    }

    #[test]
    fn create_with_cloexec() {
        let inst = epoll_create1(O_CLOEXEC).unwrap();
        assert_eq!(inst.count(), 0);
    }

    #[test]
    fn create_with_invalid_flags() {
        assert_eq!(epoll_create1(0x42), Err(Errno::EINVAL));
        assert_eq!(epoll_create1(-1), Err(Errno::EINVAL));
    }

    // ── epoll_ctl: Add ──────────────────────────────────────────────────

    #[test]
    fn ctl_add_single() {
        let mut inst = epoll_create1(0).unwrap();
        let ev = EpollEvent::new(EpollFlags::EPOLLIN, 100);
        epoll_ctl(&mut inst, EpollOp::Add, 5, &ev).unwrap();
        assert_eq!(inst.count(), 1);
    }

    #[test]
    fn ctl_add_duplicate_returns_eexist() {
        let mut inst = epoll_create1(0).unwrap();
        let ev = EpollEvent::new(EpollFlags::EPOLLIN, 100);
        epoll_ctl(&mut inst, EpollOp::Add, 5, &ev).unwrap();
        assert_eq!(
            epoll_ctl(&mut inst, EpollOp::Add, 5, &ev),
            Err(Errno::EEXIST)
        );
    }

    // ── epoll_ctl: Mod ──────────────────────────────────────────────────

    #[test]
    fn ctl_mod_existing() {
        let mut inst = epoll_create1(0).unwrap();
        let ev = EpollEvent::new(EpollFlags::EPOLLIN, 100);
        epoll_ctl(&mut inst, EpollOp::Add, 5, &ev).unwrap();

        let ev2 = EpollEvent::new(EpollFlags::EPOLLOUT, 200);
        epoll_ctl(&mut inst, EpollOp::Mod, 5, &ev2).unwrap();

        // Verify the entry was updated.
        let idx = inst.find_fd(5).unwrap();
        let entry = inst.entries[idx].unwrap();
        assert_eq!(entry.events, EpollFlags::EPOLLOUT);
        assert_eq!(entry.data, 200);
    }

    #[test]
    fn ctl_mod_nonexistent_returns_enoent() {
        let mut inst = epoll_create1(0).unwrap();
        let ev = EpollEvent::new(EpollFlags::EPOLLIN, 100);
        assert_eq!(
            epoll_ctl(&mut inst, EpollOp::Mod, 99, &ev),
            Err(Errno::ENOENT)
        );
    }

    // ── epoll_ctl: Del ──────────────────────────────────────────────────

    #[test]
    fn ctl_del_existing() {
        let mut inst = epoll_create1(0).unwrap();
        let ev = EpollEvent::new(EpollFlags::EPOLLIN, 100);
        epoll_ctl(&mut inst, EpollOp::Add, 5, &ev).unwrap();
        assert_eq!(inst.count(), 1);

        // The event arg is ignored for Del, but we still pass one.
        let dummy = EpollEvent::new(0, 0);
        epoll_ctl(&mut inst, EpollOp::Del, 5, &dummy).unwrap();
        assert_eq!(inst.count(), 0);
    }

    #[test]
    fn ctl_del_nonexistent_returns_enoent() {
        let mut inst = epoll_create1(0).unwrap();
        let dummy = EpollEvent::new(0, 0);
        assert_eq!(
            epoll_ctl(&mut inst, EpollOp::Del, 42, &dummy),
            Err(Errno::ENOENT)
        );
    }

    // ── epoll_wait ──────────────────────────────────────────────────────

    #[test]
    fn wait_returns_zero_events() {
        let inst = epoll_create1(0).unwrap();
        let mut buf = [EpollEvent::new(0, 0); 8];
        let n = epoll_wait(&inst, &mut buf, 8, 0).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn wait_invalid_max_events() {
        let inst = epoll_create1(0).unwrap();
        let mut buf = [EpollEvent::new(0, 0); 8];
        assert_eq!(epoll_wait(&inst, &mut buf, 0, 0), Err(Errno::EINVAL));
        assert_eq!(epoll_wait(&inst, &mut buf, -1, 0), Err(Errno::EINVAL));
    }

    // ── EpollFlags ──────────────────────────────────────────────────────

    #[test]
    fn epoll_flags_constants() {
        assert_eq!(EpollFlags::EPOLLIN, 0x001);
        assert_eq!(EpollFlags::EPOLLOUT, 0x004);
        assert_eq!(EpollFlags::EPOLLERR, 0x008);
        assert_eq!(EpollFlags::EPOLLHUP, 0x010);
        assert_eq!(EpollFlags::EPOLLET, 0x8000_0000);
        assert_eq!(EpollFlags::EPOLLONESHOT, 0x4000_0000);
    }

    #[test]
    fn epoll_flags_contains() {
        let flags = EpollFlags::new(EpollFlags::EPOLLIN | EpollFlags::EPOLLET);
        assert!(flags.contains(EpollFlags::EPOLLIN));
        assert!(flags.contains(EpollFlags::EPOLLET));
        assert!(!flags.contains(EpollFlags::EPOLLOUT));
    }
}
