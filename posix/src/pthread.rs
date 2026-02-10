// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! POSIX threads compatibility subset.
//!
//! Provides the threading primitives that AI inference runtimes commonly use:
//! - Thread creation and joining (`pthread_create`, `pthread_join`)
//! - Mutexes with Normal, Recursive, and ErrorCheck semantics
//! - Condition variables (`condvar_wait`, `condvar_signal`, `condvar_broadcast`)
//! - Read-write locks (`rwlock_rdlock`, `rwlock_wrlock`, `rwlock_unlock`)
//!
//! All operations are currently stubs returning `ENOSYS`. They will be wired
//! to the SmallAIOS scheduler and synchronization primitives in a later phase.

use crate::errno::Errno;

// ─── Thread Identity ────────────────────────────────────────────────────────

/// Opaque POSIX thread identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ThreadId(u64);

impl ThreadId {
    /// Create a `ThreadId` from a raw u64 value.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw u64 value.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

// ─── Thread Attributes ──────────────────────────────────────────────────────

/// Default thread stack size: 64 KiB.
pub const DEFAULT_STACK_SIZE: usize = 64 * 1024;

/// Thread creation attributes (analogous to `pthread_attr_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadAttr {
    /// Stack size in bytes.
    pub stack_size: usize,
    /// Whether the thread is created in detached state.
    pub detached: bool,
}

impl ThreadAttr {
    /// Create a new `ThreadAttr` with default values.
    pub const fn new() -> Self {
        Self {
            stack_size: DEFAULT_STACK_SIZE,
            detached: false,
        }
    }

    /// Set the stack size.
    pub const fn with_stack_size(mut self, size: usize) -> Self {
        self.stack_size = size;
        self
    }

    /// Set the detached state.
    pub const fn with_detached(mut self, detached: bool) -> Self {
        self.detached = detached;
        self
    }
}

impl Default for ThreadAttr {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Mutex ──────────────────────────────────────────────────────────────────

/// Mutex kind (analogous to `PTHREAD_MUTEX_*` type constants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutexKind {
    /// Normal (fast) mutex — undefined behaviour on double-lock.
    Normal,
    /// Recursive mutex — the same thread may lock multiple times.
    Recursive,
    /// Error-checking mutex — returns an error on double-lock.
    ErrorCheck,
}

/// POSIX-style mutual exclusion lock.
#[derive(Debug)]
pub struct Mutex {
    /// Whether the mutex is currently locked.
    pub locked: bool,
    /// The thread that currently holds the lock, if any.
    pub owner: Option<ThreadId>,
    /// The kind of mutex semantics.
    pub kind: MutexKind,
    /// Recursive lock count (only meaningful for `MutexKind::Recursive`).
    pub lock_count: u32,
}

impl Mutex {
    /// Create a new unlocked mutex with `Normal` semantics.
    pub const fn new() -> Self {
        Self {
            locked: false,
            owner: None,
            kind: MutexKind::Normal,
            lock_count: 0,
        }
    }
}

// ─── Condition Variable ─────────────────────────────────────────────────────

/// POSIX-style condition variable.
#[derive(Debug)]
pub struct Condvar {
    /// Number of threads currently waiting on this condition variable.
    pub waiters_count: u32,
}

impl Condvar {
    /// Create a new condition variable with zero waiters.
    pub const fn new() -> Self {
        Self { waiters_count: 0 }
    }
}

// ─── Read-Write Lock ────────────────────────────────────────────────────────

/// POSIX-style read-write lock.
#[derive(Debug)]
pub struct RwLock {
    /// Number of active readers.
    pub readers: u32,
    /// The thread holding the write lock, if any.
    pub writer: Option<ThreadId>,
}

impl RwLock {
    /// Create a new unlocked read-write lock.
    pub const fn new() -> Self {
        Self {
            readers: 0,
            writer: None,
        }
    }
}

// ─── Thread Operations (stubs) ──────────────────────────────────────────────

/// Create a new thread (stub — returns `ENOSYS`).
///
/// # Arguments
/// * `_attr`     — Thread attributes (stack size, detach state).
/// * `_entry_fn` — Entry-point function pointer (`extern "C" fn(*mut u8) -> *mut u8`).
/// * `_arg`      — Opaque argument passed to the entry function.
pub fn pthread_create(
    _attr: &ThreadAttr,
    _entry_fn: usize,
    _arg: usize,
) -> Result<ThreadId, Errno> {
    Err(Errno::ENOSYS)
}

/// Join a thread, blocking until it terminates (stub — returns `ENOSYS`).
pub fn pthread_join(_tid: ThreadId) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

// ─── Mutex Operations (stubs) ───────────────────────────────────────────────

/// Initialise a mutex with the given kind.
pub fn mutex_init(kind: MutexKind) -> Mutex {
    Mutex {
        locked: false,
        owner: None,
        kind,
        lock_count: 0,
    }
}

/// Lock a mutex (stub — returns `ENOSYS`).
pub fn mutex_lock(_mutex: &mut Mutex) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

/// Try to lock a mutex without blocking (stub — returns `ENOSYS`).
pub fn mutex_trylock(_mutex: &mut Mutex) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

/// Unlock a mutex (stub — returns `ENOSYS`).
pub fn mutex_unlock(_mutex: &mut Mutex) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

// ─── Condition Variable Operations (stubs) ──────────────────────────────────

/// Wait on a condition variable, atomically releasing the mutex
/// (stub — returns `ENOSYS`).
pub fn condvar_wait(_cv: &mut Condvar, _mutex: &mut Mutex) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

/// Wake one thread waiting on a condition variable (stub — returns `ENOSYS`).
pub fn condvar_signal(_cv: &mut Condvar) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

/// Wake all threads waiting on a condition variable (stub — returns `ENOSYS`).
pub fn condvar_broadcast(_cv: &mut Condvar) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

// ─── Read-Write Lock Operations (stubs) ─────────────────────────────────────

/// Acquire a read lock (stub — returns `ENOSYS`).
pub fn rwlock_rdlock(_rw: &mut RwLock) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

/// Acquire a write lock (stub — returns `ENOSYS`).
pub fn rwlock_wrlock(_rw: &mut RwLock) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

/// Release a read or write lock (stub — returns `ENOSYS`).
pub fn rwlock_unlock(_rw: &mut RwLock) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ThreadId ────────────────────────────────────────────────────────

    #[test]
    fn thread_id_equality() {
        let a = ThreadId::new(42);
        let b = ThreadId::new(42);
        let c = ThreadId::new(99);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn thread_id_roundtrip() {
        let id = ThreadId::new(0xDEAD_BEEF);
        assert_eq!(id.as_u64(), 0xDEAD_BEEF);
    }

    // ── ThreadAttr ──────────────────────────────────────────────────────

    #[test]
    fn thread_attr_defaults() {
        let attr = ThreadAttr::new();
        assert_eq!(attr.stack_size, DEFAULT_STACK_SIZE);
        assert_eq!(attr.stack_size, 64 * 1024);
        assert!(!attr.detached);
    }

    #[test]
    fn thread_attr_builder() {
        let attr = ThreadAttr::new()
            .with_stack_size(128 * 1024)
            .with_detached(true);
        assert_eq!(attr.stack_size, 128 * 1024);
        assert!(attr.detached);
    }

    #[test]
    fn thread_attr_default_trait() {
        let attr: ThreadAttr = Default::default();
        assert_eq!(attr, ThreadAttr::new());
    }

    // ── Thread creation / join stubs ────────────────────────────────────

    #[test]
    fn pthread_create_returns_enosys() {
        let attr = ThreadAttr::new();
        let result = pthread_create(&attr, 0x1000, 0);
        assert_eq!(result, Err(Errno::ENOSYS));
    }

    #[test]
    fn pthread_join_returns_enosys() {
        let tid = ThreadId::new(1);
        assert_eq!(pthread_join(tid), Err(Errno::ENOSYS));
    }

    // ── Mutex ───────────────────────────────────────────────────────────

    #[test]
    fn mutex_new_is_unlocked() {
        let m = Mutex::new();
        assert!(!m.locked);
        assert_eq!(m.owner, None);
        assert_eq!(m.kind, MutexKind::Normal);
        assert_eq!(m.lock_count, 0);
    }

    #[test]
    fn mutex_init_sets_kind() {
        let m = mutex_init(MutexKind::Recursive);
        assert_eq!(m.kind, MutexKind::Recursive);
        assert!(!m.locked);

        let m2 = mutex_init(MutexKind::ErrorCheck);
        assert_eq!(m2.kind, MutexKind::ErrorCheck);
    }

    #[test]
    fn mutex_lock_unlock_trylock_return_enosys() {
        let mut m = Mutex::new();
        assert_eq!(mutex_lock(&mut m), Err(Errno::ENOSYS));
        assert_eq!(mutex_trylock(&mut m), Err(Errno::ENOSYS));
        assert_eq!(mutex_unlock(&mut m), Err(Errno::ENOSYS));
    }

    // ── Condvar ─────────────────────────────────────────────────────────

    #[test]
    fn condvar_new_zero_waiters() {
        let cv = Condvar::new();
        assert_eq!(cv.waiters_count, 0);
    }

    #[test]
    fn condvar_stubs_return_enosys() {
        let mut cv = Condvar::new();
        let mut m = Mutex::new();
        assert_eq!(condvar_wait(&mut cv, &mut m), Err(Errno::ENOSYS));
        assert_eq!(condvar_signal(&mut cv), Err(Errno::ENOSYS));
        assert_eq!(condvar_broadcast(&mut cv), Err(Errno::ENOSYS));
    }

    // ── RwLock ──────────────────────────────────────────────────────────

    #[test]
    fn rwlock_new_is_unlocked() {
        let rw = RwLock::new();
        assert_eq!(rw.readers, 0);
        assert_eq!(rw.writer, None);
    }

    #[test]
    fn rwlock_stubs_return_enosys() {
        let mut rw = RwLock::new();
        assert_eq!(rwlock_rdlock(&mut rw), Err(Errno::ENOSYS));
        assert_eq!(rwlock_wrlock(&mut rw), Err(Errno::ENOSYS));
        assert_eq!(rwlock_unlock(&mut rw), Err(Errno::ENOSYS));
    }
}
