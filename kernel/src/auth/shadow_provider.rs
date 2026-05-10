// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Dependency-inverted shadow-file accessor.
//!
//! Spec: `openspec/changes/management-login-v1/specs/auth-shadow/spec.md`
//!
//! The kernel's auth syscall handlers must look up users by name, but the
//! kernel sits *below* `smallaios-auth` in the layer graph and therefore
//! cannot directly import [`smallaios_auth::shadow::ShadowEntry`]. We
//! solve this by defining a kernel-local mirror struct
//! ([`ShadowProviderEntry`]) plus a [`ShadowProvider`] trait that the
//! `auth` crate (and the future `fs` crate) can implement.
//!
//! ## Wiring
//!
//! - **Tests** plug a [`MockShadowProvider`] populated in-memory.
//! - **Production** uses [`KernelShadowProvider`], which currently
//!   returns [`ShadowProviderError::NotImplemented`] until
//!   `embedded-filesystem-v1` Phase 6 lands the F2FS read path. The
//!   wire-up surface is stable now so syscall handlers can be tested
//!   end-to-end against the mock.
//!
//! Round-trip with [`smallaios_auth::shadow::ShadowEntry`] is performed
//! at the integration boundary (the to-be-written `auth` console-login
//! and Zenoh-admin handlers in Phase 4 / Phase 7), not here. The kernel
//! deliberately avoids cycling back into `auth` to keep the layer
//! invariant of `kernel ⊂ Layer 0` intact.

use super::Role;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors raised by [`ShadowProvider`] implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShadowProviderError {
    /// The production [`KernelShadowProvider`] is not yet wired to
    /// `embedded-filesystem-v1`'s F2FS read path. Returned until
    /// Phase 6 of that change ships.
    NotImplemented,
    /// I/O error reading the underlying file. The string is a
    /// short, machine-stable token (caller may match on it).
    Io(&'static str),
    /// The shadow file's mode bits are laxer than 0600.
    PermissionTooLax(u32),
}

impl fmt::Display for ShadowProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "auth: KernelShadowProvider awaits embedded-filesystem-v1 Phase 6"
            ),
            Self::Io(s) => write!(f, "auth: shadow provider I/O error: {s}"),
            Self::PermissionTooLax(m) => {
                write!(f, "auth: shadow file mode 0o{m:o} laxer than 0600")
            }
        }
    }
}

// ─── Mirror entry ────────────────────────────────────────────────────────────

/// Kernel-local mirror of [`smallaios_auth::shadow::ShadowEntry`].
///
/// The fields required by the syscall handlers are mirrored: PHC, role,
/// flags, lockout, and (since Phase 9) the TOTP enrolment fields. The
/// `last_changed_unix_day` and `trailing_unknown_fields` slots are
/// omitted; the `auth` crate is the source of truth for those and they
/// are not observed by the kernel.
///
/// The [`Self::stable_user_id`] field carries the slot index assigned by
/// the loader at boot — the kernel uses it as a stable handle that
/// outlives username changes (Phase 10).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowProviderEntry {
    /// Stable user identifier — the loader assigns this when the
    /// shadow file is parsed. The kernel session table refers to
    /// users by this id, not by username, so a future "rename"
    /// operation does not invalidate live sessions.
    pub stable_user_id: u32,
    /// Username — the lookup key.
    pub username: String,
    /// Argon2id PHC string.
    pub phc: String,
    /// Cached role from the shadow record.
    pub role: Role,
    /// Bit-flag set; bit 0 is `FLAG_MUST_CHANGE_PASSWORD`.
    pub flags: u32,
    /// Lockout deadline (Unix seconds); 0 means not locked out.
    pub lockout_until_unix: u64,
    /// Optional RFC 6238 TOTP shared secret (decoded bytes). `None`
    /// means the user has not enrolled. Populated by the loader from
    /// `smallaios_auth::shadow::ShadowEntry::totp_secret`.
    pub totp_secret: Option<Vec<u8>>,
    /// Whether `auth_login` SHALL require a `factor2` TOTP code for
    /// this user. Mirrors `smallaios_auth::shadow::ShadowEntry::totp_required`.
    pub totp_required: bool,
}

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Read-side accessor over the shadow file.
///
/// The kernel calls into a [`ShadowProvider`] for every login attempt.
/// The trait is dependency-inverted so the kernel can stay independent
/// of `fs/`'s and `auth/`'s implementations and so tests can plug a
/// mock.
pub trait ShadowProvider {
    /// Look up a user by name. Returns `Ok(None)` if no user matches —
    /// the caller (login handler) SHALL still run a constant-time
    /// dummy verify to avoid leaking enumeration via timing.
    fn read_user(&self, name: &str) -> Result<Option<ShadowProviderEntry>, ShadowProviderError>;

    /// Persist a freshly-rotated PHC string for `name`. Phase 3
    /// implementation MAY return [`ShadowProviderError::NotImplemented`]
    /// — the kernel handler then audits the request and returns
    /// `-ENOSYS` to the caller.
    fn write_password(
        &self,
        name: &str,
        new_phc: &str,
        clear_must_change: bool,
    ) -> Result<(), ShadowProviderError>;

    /// Append a brand-new account. Used by `auth_create_user`.
    fn create_user(
        &self,
        name: &str,
        initial_phc: &str,
        role: Role,
    ) -> Result<u32, ShadowProviderError>;

    /// Persist a freshly-generated TOTP shared secret for `name`.
    /// Used by `auth_totp_setup` (kernel syscall 0x95). The default
    /// implementation returns [`ShadowProviderError::NotImplemented`]
    /// so providers that don't support TOTP enrolment (e.g., the
    /// `KernelShadowProvider` stub) compile without modification.
    ///
    /// Implementations SHALL NOT clear `totp_required`; that bit is
    /// owned by `mgmt::Config::totp.enforced_for_roles` and toggled
    /// independently. The intent is "store the secret; the policy
    /// gate decides when to require it".
    fn write_totp_secret(
        &self,
        _name: &str,
        _new_secret: &[u8],
    ) -> Result<(), ShadowProviderError> {
        Err(ShadowProviderError::NotImplemented)
    }
}

// ─── Mock for tests ──────────────────────────────────────────────────────────

/// In-memory [`ShadowProvider`] used by kernel auth tests.
///
/// Holds the shadow as a `Vec<ShadowProviderEntry>` in a `RefCell` so
/// tests can mutate it through `&self`. `stable_user_id` values are
/// assigned monotonically.
#[cfg(any(test, feature = "test-mocks"))]
pub struct MockShadowProvider {
    inner: core::cell::RefCell<MockInner>,
}

#[cfg(any(test, feature = "test-mocks"))]
struct MockInner {
    entries: alloc::vec::Vec<ShadowProviderEntry>,
    next_id: u32,
    write_should_fail: bool,
    create_should_fail: bool,
}

#[cfg(any(test, feature = "test-mocks"))]
impl Default for MockShadowProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-mocks"))]
impl MockShadowProvider {
    /// Construct an empty mock provider.
    pub fn new() -> Self {
        Self {
            inner: core::cell::RefCell::new(MockInner {
                entries: alloc::vec::Vec::new(),
                next_id: 0,
                write_should_fail: false,
                create_should_fail: false,
            }),
        }
    }

    /// Pre-seed the mock with an entry. Returns the assigned
    /// `stable_user_id`.
    pub fn seed(&self, mut entry: ShadowProviderEntry) -> u32 {
        let mut g = self.inner.borrow_mut();
        let id = g.next_id;
        g.next_id += 1;
        entry.stable_user_id = id;
        g.entries.push(entry);
        id
    }

    /// Inject a failure for the next [`ShadowProvider::write_password`]
    /// call. Used by audit-on-failure tests.
    pub fn fail_next_write(&self) {
        self.inner.borrow_mut().write_should_fail = true;
    }

    /// Inject a failure for the next [`ShadowProvider::create_user`]
    /// call.
    pub fn fail_next_create(&self) {
        self.inner.borrow_mut().create_should_fail = true;
    }

    /// Borrow the current PHC stored for `name`, if any.
    pub fn current_phc(&self, name: &str) -> Option<String> {
        self.inner
            .borrow()
            .entries
            .iter()
            .find(|e| e.username == name)
            .map(|e| e.phc.clone())
    }

    /// Borrow the current `flags` for `name`, if any.
    pub fn current_flags(&self, name: &str) -> Option<u32> {
        self.inner
            .borrow()
            .entries
            .iter()
            .find(|e| e.username == name)
            .map(|e| e.flags)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.inner.borrow().entries.len()
    }

    /// Returns true if the mock is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(any(test, feature = "test-mocks"))]
impl ShadowProvider for MockShadowProvider {
    fn read_user(&self, name: &str) -> Result<Option<ShadowProviderEntry>, ShadowProviderError> {
        Ok(self
            .inner
            .borrow()
            .entries
            .iter()
            .find(|e| e.username == name)
            .cloned())
    }

    fn write_password(
        &self,
        name: &str,
        new_phc: &str,
        clear_must_change: bool,
    ) -> Result<(), ShadowProviderError> {
        let mut g = self.inner.borrow_mut();
        if g.write_should_fail {
            g.write_should_fail = false;
            return Err(ShadowProviderError::Io("injected write failure"));
        }
        for e in g.entries.iter_mut() {
            if e.username == name {
                e.phc = new_phc.into();
                if clear_must_change {
                    e.flags &= !crate::auth::FLAG_MUST_CHANGE_PASSWORD;
                }
                return Ok(());
            }
        }
        Err(ShadowProviderError::Io("user not found"))
    }

    fn create_user(
        &self,
        name: &str,
        initial_phc: &str,
        role: Role,
    ) -> Result<u32, ShadowProviderError> {
        let mut g = self.inner.borrow_mut();
        if g.create_should_fail {
            g.create_should_fail = false;
            return Err(ShadowProviderError::Io("injected create failure"));
        }
        if g.entries.iter().any(|e| e.username == name) {
            return Err(ShadowProviderError::Io("user already exists"));
        }
        let id = g.next_id;
        g.next_id += 1;
        g.entries.push(ShadowProviderEntry {
            stable_user_id: id,
            username: name.into(),
            phc: initial_phc.into(),
            role,
            flags: crate::auth::FLAG_MUST_CHANGE_PASSWORD,
            lockout_until_unix: 0,
            totp_secret: None,
            totp_required: false,
        });
        Ok(id)
    }

    fn write_totp_secret(&self, name: &str, new_secret: &[u8]) -> Result<(), ShadowProviderError> {
        let mut g = self.inner.borrow_mut();
        for e in g.entries.iter_mut() {
            if e.username == name {
                e.totp_secret = Some(new_secret.to_vec());
                return Ok(());
            }
        }
        Err(ShadowProviderError::Io("user not found"))
    }
}

// ─── Production stub ─────────────────────────────────────────────────────────

/// Production [`ShadowProvider`] — currently a stub.
///
/// Returns [`ShadowProviderError::NotImplemented`] until
/// `embedded-filesystem-v1` Phase 6 lands the F2FS read path. The
/// constructor is `const fn` so the type can live in a static.
pub struct KernelShadowProvider {
    _private: (),
}

impl Default for KernelShadowProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelShadowProvider {
    /// Construct the production provider.
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

impl ShadowProvider for KernelShadowProvider {
    fn read_user(&self, _name: &str) -> Result<Option<ShadowProviderEntry>, ShadowProviderError> {
        Err(ShadowProviderError::NotImplemented)
    }

    fn write_password(
        &self,
        _name: &str,
        _new_phc: &str,
        _clear_must_change: bool,
    ) -> Result<(), ShadowProviderError> {
        Err(ShadowProviderError::NotImplemented)
    }

    fn create_user(
        &self,
        _name: &str,
        _initial_phc: &str,
        _role: Role,
    ) -> Result<u32, ShadowProviderError> {
        Err(ShadowProviderError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn fixture_entry(name: &str) -> ShadowProviderEntry {
        ShadowProviderEntry {
            stable_user_id: 0, // overwritten by seed
            username: name.to_string(),
            // lgtm[rust/hard-coded-cryptographic-value] — synthetic test fixture, not a real credential
            phc: "$argon2id$v=19$m=8192,t=3,p=1$YWFhYWFhYWFhYWFhYWFhYQ$YWJjZGVmZ2hpamtsbW5vcA"
                .to_string(),
            role: Role::Operator,
            flags: 0,
            lockout_until_unix: 0,
            totp_secret: None,
            totp_required: false,
        }
    }

    #[test]
    fn mock_read_user_returns_seeded_entry() {
        let p = MockShadowProvider::new();
        let id = p.seed(fixture_entry("alice"));
        let got = p.read_user("alice").unwrap().unwrap();
        assert_eq!(got.username, "alice");
        assert_eq!(got.stable_user_id, id);
    }

    #[test]
    fn mock_read_user_missing_returns_none() {
        let p = MockShadowProvider::new();
        assert!(p.read_user("nobody").unwrap().is_none());
    }

    #[test]
    fn mock_write_password_replaces_phc_and_clears_must_change() {
        let p = MockShadowProvider::new();
        let mut entry = fixture_entry("bob");
        entry.flags = crate::auth::FLAG_MUST_CHANGE_PASSWORD;
        p.seed(entry);

        p.write_password(
            "bob",
            "$argon2id$v=19$m=8192,t=3,p=1$c2FsdHNhbHRzYWx0$dGFndGFn",
            true,
        )
        .unwrap();
        let got = p.read_user("bob").unwrap().unwrap();
        assert!(got.phc.contains("dGFndGFn"));
        assert_eq!(got.flags & crate::auth::FLAG_MUST_CHANGE_PASSWORD, 0);
    }

    #[test]
    fn mock_create_user_assigns_must_change_flag_and_unique_id() {
        let p = MockShadowProvider::new();
        let id_a = p
            .create_user(
                "op-a",
                "$argon2id$v=19$m=8192,t=3,p=1$c2FsdHNhbHQ$dGFndGFn",
                Role::Operator,
            )
            .unwrap();
        let id_b = p
            .create_user(
                "op-b",
                "$argon2id$v=19$m=8192,t=3,p=1$c2FsdHNhbHQ$dGFndGFn",
                Role::Viewer,
            )
            .unwrap();
        assert_ne!(id_a, id_b);

        let entry_a = p.read_user("op-a").unwrap().unwrap();
        assert_eq!(
            entry_a.flags & crate::auth::FLAG_MUST_CHANGE_PASSWORD,
            crate::auth::FLAG_MUST_CHANGE_PASSWORD
        );
        assert_eq!(entry_a.role, Role::Operator);
    }

    #[test]
    fn mock_create_user_rejects_duplicate() {
        let p = MockShadowProvider::new();
        p.create_user(
            "dup",
            "$argon2id$v=19$m=8192,t=3,p=1$c2FsdHNhbHQ$dGFndGFn",
            Role::Root,
        )
        .unwrap();
        let e = p
            .create_user(
                "dup",
                "$argon2id$v=19$m=8192,t=3,p=1$c2FsdHNhbHQ$dGFndGFn",
                Role::Root,
            )
            .unwrap_err();
        assert!(matches!(e, ShadowProviderError::Io(_)));
    }

    #[test]
    fn mock_injected_failure_propagates_once() {
        let p = MockShadowProvider::new();
        p.seed(fixture_entry("alice"));
        p.fail_next_write();
        let e = p
            .write_password(
                "alice",
                "$argon2id$v=19$m=8192,t=3,p=1$c2FsdHNhbHQ$dGFndGFn",
                false,
            )
            .unwrap_err();
        assert!(matches!(e, ShadowProviderError::Io(_)));
        // Failure is one-shot.
        p.write_password(
            "alice",
            "$argon2id$v=19$m=8192,t=3,p=1$c2FsdHNhbHQ$dGFndGFn",
            false,
        )
        .unwrap();
    }

    #[test]
    fn kernel_provider_returns_not_implemented() {
        let p = KernelShadowProvider::new();
        assert_eq!(
            p.read_user("anyone"),
            Err(ShadowProviderError::NotImplemented)
        );
        assert_eq!(
            p.write_password("anyone", "$argon2id$...", false),
            Err(ShadowProviderError::NotImplemented)
        );
        assert_eq!(
            p.create_user("anyone", "$argon2id$...", Role::Viewer),
            Err(ShadowProviderError::NotImplemented)
        );
    }
}
