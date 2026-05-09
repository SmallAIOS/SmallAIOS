// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Dependency-inverted boot-success commit hook.
//!
//! Spec: `openspec/changes/embedded-filesystem-v1/specs/fs-ab-boot/spec.md`
//!
//! The kernel's `SYS_BOOT_SUCCESS = 0x57` syscall must call into the
//! `fs::boot_config::boot_success(...)` helper to flip
//! `tentative=0, boot_success=1` on the active A/B boot record.
//! Because the kernel sits at Layer 0 (below `fs/`), it cannot
//! directly import the helper.
//!
//! The split mirrors [`crate::auth::ShadowProvider`]: the kernel
//! defines a [`BootSuccessProvider`] trait, the `fs/` crate
//! implements it from outside, and the boot path installs the
//! provider via [`install_provider`] before flipping the auth gate
//! on. Tests plug a [`MockBootSuccessProvider`].
//!
//! Owned by `embedded-filesystem-v1` Phase 10.

extern crate alloc;

use core::cell::UnsafeCell;
use core::fmt;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors surfaced by [`BootSuccessProvider`] implementations.
///
/// The variants are deliberately coarse — every fatal storage error
/// collapses into [`Self::Storage`], every transient retryable error
/// into [`Self::Retry`]. The syscall handler maps the variants to
/// POSIX-flavoured errnos for the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootSuccessError {
    /// No production provider is plugged in. Only [`KernelBootSuccessProvider`]
    /// returns this; tests use [`MockBootSuccessProvider`] which never
    /// returns it. Maps to `-ENOSYS`.
    NotImplemented,
    /// Underlying boot-config storage failed (block I/O, CRC mismatch,
    /// etc.). Maps to `-EIO`.
    Storage(&'static str),
    /// Transient watchdog or block-device error; caller may retry.
    /// Maps to `-EAGAIN`.
    Retry(&'static str),
    /// `boot_success` was already committed for the active record.
    /// The handler treats this as success (idempotent re-entry) and
    /// does NOT surface this variant to userspace; it is captured in
    /// the audit record's `idempotent=true` field.
    AlreadyCommitted,
}

impl fmt::Display for BootSuccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("boot_success: provider not installed"),
            Self::Storage(s) => write!(f, "boot_success: storage error: {s}"),
            Self::Retry(s) => write!(f, "boot_success: retry: {s}"),
            Self::AlreadyCommitted => f.write_str("boot_success: already committed"),
        }
    }
}

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Outcome reported by a successful [`BootSuccessProvider::commit`].
///
/// Carries the data needed for the kernel to emit the
/// `boot_success_committed` audit record without re-reading the boot
/// config record itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootSuccessOutcome {
    /// Boot-record generation that the kernel committed (the new,
    /// post-`+1` value). On idempotent re-entry, this is the existing
    /// generation.
    pub generation: u64,
    /// `true` if the commit was a no-op (record already had
    /// `boot_success=1`).
    pub idempotent: bool,
}

/// Producer of the on-disk boot-success transition.
///
/// Production wiring: `fs::boot_config::KernelBootSuccessProvider`
/// (in `fs/src/boot_config/`) holds an `&'static mut dyn BlockDevice`
/// and an `&'static mut dyn Watchdog` and calls
/// `fs::boot_config::boot_success(...)` from inside [`Self::commit`].
///
/// Tests use [`MockBootSuccessProvider`].
pub trait BootSuccessProvider: Send + Sync {
    /// Atomically commit `tentative=0, boot_success=1` on the active
    /// A/B boot record. Idempotent: a second call after a successful
    /// first SHALL return `Ok(BootSuccessOutcome { idempotent: true,
    /// .. })` without advancing the generation.
    fn commit(&self) -> Result<BootSuccessOutcome, BootSuccessError>;
}

// ─── Production stub ─────────────────────────────────────────────────────────

/// Production [`BootSuccessProvider`] — returns
/// [`BootSuccessError::NotImplemented`] until `fs/` plugs in its
/// `KernelBootSuccessProvider`. Carries no state. Used as the boot-
/// time default so the syscall has a sane behaviour before the fs
/// crate registers its provider.
#[derive(Debug, Default)]
pub struct KernelBootSuccessProvider;

impl KernelBootSuccessProvider {
    /// Construct a stub provider.
    pub const fn new() -> Self {
        Self
    }
}

impl BootSuccessProvider for KernelBootSuccessProvider {
    fn commit(&self) -> Result<BootSuccessOutcome, BootSuccessError> {
        Err(BootSuccessError::NotImplemented)
    }
}

// ─── Mock for tests ──────────────────────────────────────────────────────────

/// In-memory [`BootSuccessProvider`] used by kernel + fs tests.
///
/// Tracks a `generation`, an `idempotent` flag, and an injectable
/// failure mode so syscall tests can exercise every error path
/// without dragging the on-disk machinery into the kernel test-bench.
#[cfg(any(test, feature = "test-mocks"))]
pub struct MockBootSuccessProvider {
    inner: core::cell::RefCell<MockInner>,
}

#[cfg(any(test, feature = "test-mocks"))]
#[derive(Debug)]
struct MockInner {
    generation: u64,
    committed: bool,
    inject_error: Option<BootSuccessError>,
    /// How many times [`commit`](Self::commit) has been called.
    commit_count: u32,
}

#[cfg(any(test, feature = "test-mocks"))]
impl Default for MockBootSuccessProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-mocks"))]
impl MockBootSuccessProvider {
    /// Construct a fresh mock. Initial generation is 1, `committed`
    /// false (first call advances to 2 with `committed=true`).
    pub fn new() -> Self {
        Self {
            inner: core::cell::RefCell::new(MockInner {
                generation: 1,
                committed: false,
                inject_error: None,
                commit_count: 0,
            }),
        }
    }

    /// Force the next [`commit`](BootSuccessProvider::commit) to
    /// return `err`. Cleared after the call returns.
    pub fn fail_next(&self, err: BootSuccessError) {
        self.inner.borrow_mut().inject_error = Some(err);
    }

    /// Number of commit calls observed so far.
    pub fn commit_count(&self) -> u32 {
        self.inner.borrow().commit_count
    }

    /// Current generation (post-commit).
    pub fn generation(&self) -> u64 {
        self.inner.borrow().generation
    }

    /// `true` iff the active record has been committed.
    pub fn is_committed(&self) -> bool {
        self.inner.borrow().committed
    }
}

// SAFETY: MockBootSuccessProvider's RefCell is only entered from
// tests, where each call is single-threaded. The Send + Sync impls
// are needed because the `BootSuccessProvider` trait requires them
// for the dispatch path.
#[cfg(any(test, feature = "test-mocks"))]
unsafe impl Send for MockBootSuccessProvider {}
#[cfg(any(test, feature = "test-mocks"))]
unsafe impl Sync for MockBootSuccessProvider {}

#[cfg(any(test, feature = "test-mocks"))]
impl BootSuccessProvider for MockBootSuccessProvider {
    fn commit(&self) -> Result<BootSuccessOutcome, BootSuccessError> {
        let mut g = self.inner.borrow_mut();
        g.commit_count = g.commit_count.saturating_add(1);
        if let Some(err) = g.inject_error.take() {
            return Err(err);
        }
        if g.committed {
            // Idempotent re-entry — generation unchanged.
            return Ok(BootSuccessOutcome {
                generation: g.generation,
                idempotent: true,
            });
        }
        g.generation = g.generation.saturating_add(1);
        g.committed = true;
        Ok(BootSuccessOutcome {
            generation: g.generation,
            idempotent: false,
        })
    }
}

// ─── Global slot ─────────────────────────────────────────────────────────────

/// Wrapper around a `&'static dyn BootSuccessProvider`.
///
/// The kernel boot path installs an instance via [`install_provider`]
/// after the `fs/` mount completes; the syscall handler reads the
/// installed instance via [`with_provider`].
///
/// In `#![no_std]` we cannot use `OnceLock`, so we store the pointer
/// in an [`UnsafeCell`] that is mutated only at boot (single-threaded)
/// and read with `&` thereafter. The `installed` atomic gates reads.
struct ProviderSlot {
    ptr: UnsafeCell<*const dyn BootSuccessProvider>,
    installed: core::sync::atomic::AtomicBool,
}

// SAFETY: the slot is only mutated during boot (before any other CPU
// is unparked); after `installed.store(Release)`, the pointer is read
// with Acquire ordering and cannot change. The kernel's MinRole gate
// further serializes invocations of `with_provider`.
unsafe impl Sync for ProviderSlot {}

const STUB: &KernelBootSuccessProvider = &KernelBootSuccessProvider::new();

static SLOT: ProviderSlot = ProviderSlot {
    ptr: UnsafeCell::new(STUB),
    installed: core::sync::atomic::AtomicBool::new(false),
};

/// Install a production [`BootSuccessProvider`]. Idempotent — a
/// second call replaces the previous pointer (the kernel's boot path
/// only calls this once, but the API tolerates re-installation so
/// integration test harnesses can swap fixtures).
///
/// # Safety
///
/// Caller MUST ensure `provider` outlives the kernel — i.e. either
/// it is `&'static` (production) or it is leaked via
/// `Box::leak` for tests.
pub fn install_provider(provider: &'static dyn BootSuccessProvider) {
    // SAFETY: writes happen only during boot or test setup, and the
    // installed flag's Release ordering pairs with the Acquire load
    // in `with_provider`.
    unsafe {
        *SLOT.ptr.get() = provider as *const dyn BootSuccessProvider;
    }
    SLOT.installed
        .store(true, core::sync::atomic::Ordering::Release);
}

/// Run `f` against the currently-installed provider. If no provider
/// was installed, runs against [`KernelBootSuccessProvider`] (which
/// returns [`BootSuccessError::NotImplemented`]).
pub fn with_provider<R, F>(f: F) -> R
where
    F: FnOnce(&dyn BootSuccessProvider) -> R,
{
    if !SLOT.installed.load(core::sync::atomic::Ordering::Acquire) {
        return f(STUB);
    }
    // SAFETY: `installed` is true ⇒ a call to `install_provider`
    // happened-before this load, so `*SLOT.ptr.get()` is a valid
    // pointer to a `dyn BootSuccessProvider` whose lifetime is
    // `'static`.
    let provider: &dyn BootSuccessProvider = unsafe { &**SLOT.ptr.get() };
    f(provider)
}

/// `true` iff a non-stub provider has been installed. Tests use this
/// to assert their setup ran.
pub fn provider_installed() -> bool {
    SLOT.installed.load(core::sync::atomic::Ordering::Acquire)
}

// ─── Audit hook ──────────────────────────────────────────────────────────────

/// In-RAM mirror of the most-recent `boot_success_committed` audit
/// record.
///
/// Once the kernel has a wired audit ring (Phase N of the audit
/// roadmap), this module moves into `audit_ring/` and the syscall
/// handler emits via that path. For now we stash the most-recent
/// event so tests can assert on it. In `#![no_std]` we use an
/// `UnsafeCell` guarded by an atomic flag — kernel syscalls run with
/// interrupts masked so single-threaded access is safe.
#[cfg(any(test, feature = "test-mocks"))]
pub mod audit_capture {
    use super::BootSuccessOutcome;
    use core::cell::UnsafeCell;
    use core::sync::atomic::{AtomicBool, Ordering};

    struct LastRecord {
        cell: UnsafeCell<Option<(BootSuccessOutcome, alloc::string::String)>>,
        present: AtomicBool,
    }

    // SAFETY: writes happen only from the syscall handler under
    // interrupt-mask serialization; readers (tests) use the atomic
    // flag to gate access.
    unsafe impl Sync for LastRecord {}

    static LAST: LastRecord = LastRecord {
        cell: UnsafeCell::new(None),
        present: AtomicBool::new(false),
    };

    /// Record an audit event for tests.
    pub fn record(outcome: BootSuccessOutcome, tag: &str) {
        // SAFETY: serialized by the kernel's syscall dispatch (single
        // caller per core with interrupts masked).
        unsafe {
            *LAST.cell.get() = Some((outcome, tag.into()));
        }
        LAST.present.store(true, Ordering::Release);
    }

    /// Read the most recent audit event.
    pub fn last() -> Option<(BootSuccessOutcome, alloc::string::String)> {
        if !LAST.present.load(Ordering::Acquire) {
            return None;
        }
        // SAFETY: the present flag's Release ordering pairs with the
        // Acquire load above; the cell's contents are stable for the
        // duration of this read because writes are serialized.
        unsafe { (*LAST.cell.get()).clone() }
    }

    /// Clear the captured audit state.
    pub fn clear() {
        // SAFETY: same reasoning as `record`.
        unsafe {
            *LAST.cell.get() = None;
        }
        LAST.present.store(false, Ordering::Release);
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_provider_returns_not_implemented() {
        let p = KernelBootSuccessProvider::new();
        assert!(matches!(p.commit(), Err(BootSuccessError::NotImplemented)));
    }

    #[test]
    fn mock_first_commit_advances_generation() {
        let m = MockBootSuccessProvider::new();
        let r = m.commit().unwrap();
        assert!(!r.idempotent);
        assert_eq!(r.generation, 2);
        assert!(m.is_committed());
        assert_eq!(m.commit_count(), 1);
    }

    #[test]
    fn mock_second_commit_is_idempotent() {
        let m = MockBootSuccessProvider::new();
        m.commit().unwrap();
        let r = m.commit().unwrap();
        assert!(r.idempotent);
        assert_eq!(r.generation, 2);
        assert_eq!(m.commit_count(), 2);
    }

    #[test]
    fn mock_inject_error_round_trip() {
        let m = MockBootSuccessProvider::new();
        m.fail_next(BootSuccessError::Storage("media"));
        assert!(matches!(
            m.commit(),
            Err(BootSuccessError::Storage("media"))
        ));
        // Cleared after the call: subsequent commit succeeds.
        let r = m.commit().unwrap();
        assert!(!r.idempotent);
    }

    #[test]
    fn boot_success_error_display() {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        write!(s, "{}", BootSuccessError::NotImplemented).unwrap();
        assert!(s.contains("not installed"));
        s.clear();
        write!(s, "{}", BootSuccessError::Storage("x")).unwrap();
        assert!(s.contains("storage"));
        s.clear();
        write!(s, "{}", BootSuccessError::Retry("y")).unwrap();
        assert!(s.contains("retry"));
        s.clear();
        write!(s, "{}", BootSuccessError::AlreadyCommitted).unwrap();
        assert!(s.contains("already"));
    }

    #[test]
    fn outcome_round_trip() {
        let o = BootSuccessOutcome {
            generation: 7,
            idempotent: false,
        };
        assert_eq!(o.generation, 7);
        assert!(!o.idempotent);
    }
}
