// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Overlay model-add / model-remove syscall handlers (0x36 / 0x37).
//!
//! Spec: `openspec/changes/embedded-overlay-v1/specs/fs-overlay-syscalls/spec.md`
//!
//! These two handlers gate, validate, and dispatch operator-driven
//! mutations of the overlay's writable upper layer. The actual byte
//! shuffling is owned by the FS crate's `OverlayWriter` (or any other
//! impl of [`OverlayBackend`]); the kernel's job is:
//!
//! 1. RBAC — `model_add` requires `MinRole::Operator`; `model_remove`
//!    requires `MinRole::Root` for modes 0/1, and Operator iff
//!    `fs.overlay.allow_operator_unhide = true` for mode 2.
//! 2. Name validation — UTF-8, 1..=255 bytes, no `..`, no `/`,
//!    no reserved suffix.
//! 3. Per-name advisory lock acquisition (-EBUSY if held).
//! 4. Delegation to the [`OverlayBackend`] for the actual stage-and-
//!    rename / unlink / whiteout.
//! 5. Audit emission via [`OverlayAuditSink`].
//!
//! ## Pattern note
//!
//! Mirrors `kernel::syscall::auth` — handlers are functions parametric
//! over the backend / audit sink so unit tests can inject mocks
//! without touching globals. The dispatcher stubs (`sys_onnx_model_*`
//! in [`super::onnx`]) keep returning `-ENOSYS` until a real backend
//! is installed at boot — then the dispatcher swaps to the real
//! handler. This file does not own the dispatcher entry point.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use smallaios_security::crypto::sha3::Sha3_256Digest;

use crate::auth::{
    NameLock, OverlayLockError, OverlayLockTable, Role, SessionId, OVERLAY_LOCK_TABLE_CAP,
    OVERLAY_NAME_MAX,
};

use super::SyscallResult;

// ─── errno constants (kernel-local) ──────────────────────────────────────────

const ERRNO_EPERM: i64 = -1;
const ERRNO_EIO: i64 = -5;
const ERRNO_EBUSY: i64 = -16;
const ERRNO_EINVAL: i64 = -22;
const ERRNO_ENOSPC: i64 = -28;
const ERRNO_EROFS: i64 = -30;
const ERRNO_ENOSYS: i64 = -38;
const ERRNO_EFAULT: i64 = -14;
const ERRNO_EAUTH: i64 = -13;

/// Reserved overlay suffixes — copied from
/// `smallaios_fs::overlay::reserved::RESERVED_SUFFIXES`. The kernel
/// can't depend on `fs` (cycle) so the list is duplicated here. The
/// fs-side reserved-suffix CI lint (Phase 7.3) enforces parity.
pub const RESERVED_SUFFIXES: &[&str] = &[".whiteout", ".opaque", ".sha3", ".sig"];

/// Maximum operator-supplied model name length.
pub const MODEL_NAME_MAX: usize = 255;

// ─── Backend trait ───────────────────────────────────────────────────────────

/// Result of a successful [`OverlayBackend::add`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddOutcome {
    /// Final upper-relative name.
    pub name: String,
    /// SHA-3-256 of the written bytes.
    pub sha3: Sha3_256Digest,
    /// Bytes streamed in.
    pub size: u64,
    /// `true` iff a `<name>.sig` sidecar was written.
    pub signature_written: bool,
}

/// Backend errors mirrored from `OverlayWriteError` in the FS crate.
/// Defined here so the kernel does not depend on `smallaios-fs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// `-EINVAL` — reserved suffix.
    ReservedSuffix(String),
    /// `-EBUSY` — concurrent add detected by the backend.
    Busy(String),
    /// `-ENOSPC` — capacity cap exceeded.
    NoSpace,
    /// `-EROFS` — write into the read-only lower.
    ReadOnlyLower(String),
    /// `-EAUTH` — `require_signed=true` but no signature supplied.
    SignatureRequired,
    /// `-EIO` — backend I/O failure.
    Io,
    /// `-EAUTH` — signature malformed / invalid.
    SignatureInvalid,
    /// `-EPERM` — backend rejected operation (RBAC).
    PermissionDenied,
}

impl BackendError {
    /// Map to the negative errno value the kernel returns.
    pub const fn errno(&self) -> i64 {
        match self {
            Self::ReservedSuffix(_) => ERRNO_EINVAL,
            Self::Busy(_) => ERRNO_EBUSY,
            Self::NoSpace => ERRNO_ENOSPC,
            Self::ReadOnlyLower(_) => ERRNO_EROFS,
            Self::SignatureRequired | Self::SignatureInvalid => ERRNO_EAUTH,
            Self::Io => ERRNO_EIO,
            Self::PermissionDenied => ERRNO_EPERM,
        }
    }
}

/// Source-side reader (e.g. an FD into Zenoh / posix file).
pub trait ContentSource {
    /// Read up to `buf.len()` bytes. `Ok(0)` is EOF; `Err(())` is I/O
    /// error (mapped to `-EIO`).
    #[allow(clippy::result_unit_err)]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ()>;
}

/// In-memory [`ContentSource`] backed by a byte slice (for tests +
/// the small-payload syscall path).
pub struct SliceSource<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> SliceSource<'a> {
    /// Construct a slice source.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
}

impl<'a> ContentSource for SliceSource<'a> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ()> {
        let remaining = self.bytes.len() - self.pos;
        let n = core::cmp::min(remaining, buf.len());
        if n == 0 {
            return Ok(0);
        }
        buf[..n].copy_from_slice(&self.bytes[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Backend trait — implemented in production by
/// `smallaios_fs::overlay::OverlayWriter`. Defined here so the kernel
/// doesn't pull in the FS crate.
pub trait OverlayBackend {
    /// Streaming add of `name` from `source`. The backend handles
    /// stage-and-rename + sidecars + cap enforcement.
    fn add(
        &mut self,
        name: &str,
        source: &mut dyn ContentSource,
        expected_size: u64,
        signature: Option<&[u8]>,
    ) -> Result<AddOutcome, BackendError>;

    /// Delete the upper-layer entry at `name` (and its `.sha3`/`.sig`
    /// sidecars). Idempotent.
    fn remove_upper(&mut self, name: &str) -> Result<(), BackendError>;

    /// Write a whiteout `<name>.whiteout` to hide the lower's `<name>`.
    fn write_whiteout(&mut self, name: &str) -> Result<(), BackendError>;

    /// Remove an existing `<name>.whiteout` (un-hide).
    fn remove_whiteout(&mut self, name: &str) -> Result<(), BackendError>;

    /// Cleanup the orphan `<name>.tmp` left behind by an aborted add
    /// for `name`. Best-effort — failure is suppressed so the session-
    /// release path always returns `Ok`.
    fn cleanup_orphan_tmp_for(&mut self, name: &str);
}

// ─── Audit sink ──────────────────────────────────────────────────────────────

/// One audit record emitted by the model-add / model-remove handlers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayAuditEvent {
    /// `model_added{ who, name, sha3, size }` per the spec scenario.
    ModelAdded {
        /// User id from the holder's session.
        who: u32,
        /// Final overlay name.
        name: String,
        /// SHA-3-256 of the streamed bytes.
        sha3: Sha3_256Digest,
        /// Size of the streamed bytes.
        size: u64,
    },
    /// `model_removed{ who, name, mode }` (mode = 0).
    ModelRemoved {
        /// User id.
        who: u32,
        /// Removed name.
        name: String,
    },
    /// `model_hidden{ who, name }` (mode = 1).
    ModelHidden {
        /// User id.
        who: u32,
        /// Lower name now hidden.
        name: String,
    },
    /// `model_unhidden{ who, name, role }` (mode = 2).
    ModelUnhidden {
        /// User id.
        who: u32,
        /// Lower name no longer hidden.
        name: String,
        /// Role byte at unhide time (Operator-unhide observability per
        /// spec scenario "Operator may unhide if policy permits").
        role: u8,
    },
    /// `DENY:onnx_model_add` — RBAC denial during `model_add`.
    DenyModelAdd {
        /// Calling user id (or `0xFFFF_FFFF` for unauthenticated).
        who: u32,
        /// Role byte (or `0xFF` for unauthenticated).
        role: u8,
    },
    /// `DENY:onnx_model_remove` — RBAC denial during `model_remove`.
    DenyModelRemove {
        /// Calling user id.
        who: u32,
        /// Role byte.
        role: u8,
        /// Mode byte (0/1/2).
        mode: u8,
    },
    /// `model_add_capacity_exceeded` — declared/actual size violated cap.
    ModelAddCapacityExceeded {
        /// User id.
        who: u32,
        /// Name attempted.
        name: String,
        /// `expected_size` sent to the syscall.
        declared: u64,
    },
    /// `model_load_unsigned` — strict signature policy
    /// (`fs.overlay.require_signed = true`) was in force at load time
    /// but the model in question had no `<name>.sig` sidecar. The
    /// load fails closed with `-EAUTH`. Spec:
    /// `embedded-overlay-v1` `fs-overlay-integrity` Phase 5.
    ModelLoadUnsigned {
        /// User id (or `0xFFFF_FFFF` for unauthenticated).
        who: u32,
        /// Model name attempted.
        name: String,
    },
    /// `model_signature_invalid` — a `<name>.sig` sidecar exists but
    /// failed ML-DSA-65 verification (defense in depth: emitted in
    /// permissive mode too whenever a sidecar fails to verify, since
    /// presenting an invalid signature should never silently succeed).
    /// Surfaces as `-EAUTH` to the syscall ABI.
    ModelSignatureInvalid {
        /// User id.
        who: u32,
        /// Model name attempted.
        name: String,
    },
    /// `model_signature_verified` — strict signature policy passed:
    /// `<name>.sig` was present and ML-DSA-65 verified against the
    /// configured trust anchor. INFO-level forensic record so
    /// auditors can confirm positive coverage; emitted only when
    /// strict policy is active to keep the permissive path quiet.
    ModelSignatureVerified {
        /// User id.
        who: u32,
        /// Model name loaded.
        name: String,
    },
}

/// Audit-sink trait — production wires this to the global audit ring,
/// tests use [`CapturingOverlayAuditSink`].
pub trait OverlayAuditSink {
    /// Append one audit event. Backpressure is the sink's problem;
    /// this trait drops nothing.
    fn append(&mut self, event: OverlayAuditEvent);
}

/// In-memory audit sink for tests — collects every event in a `Vec`.
#[derive(Debug, Default)]
pub struct CapturingOverlayAuditSink {
    /// Events captured in append order.
    pub events: Vec<OverlayAuditEvent>,
}

impl CapturingOverlayAuditSink {
    /// Construct an empty capturing sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of events captured.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// `true` iff no events have been captured.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl OverlayAuditSink for CapturingOverlayAuditSink {
    fn append(&mut self, event: OverlayAuditEvent) {
        self.events.push(event);
    }
}

// ─── Name validation ─────────────────────────────────────────────────────────

/// Errors raised by [`validate_name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameValidationError {
    /// Empty name supplied.
    Empty,
    /// Name longer than [`MODEL_NAME_MAX`].
    TooLong,
    /// Name was not valid UTF-8.
    NotUtf8,
    /// Name contained a `/` separator.
    ContainsSlash,
    /// Name contained the `..` parent-traversal sequence.
    ParentTraversal,
    /// Name has a reserved overlay suffix.
    ReservedSuffix,
}

/// Validate a raw byte slice as a model name. Returns the decoded
/// `&str` on success.
///
/// Rules per `fs-overlay-mount` spec + `fs-overlay-syscalls` spec:
///
/// - 1..=255 bytes
/// - valid UTF-8
/// - no `/`
/// - no `..` (defense-in-depth path-traversal block)
/// - no reserved suffix (`.whiteout`, `.opaque`, `.sha3`, `.sig`)
pub fn validate_name(bytes: &[u8]) -> Result<&str, NameValidationError> {
    if bytes.is_empty() {
        return Err(NameValidationError::Empty);
    }
    if bytes.len() > MODEL_NAME_MAX {
        return Err(NameValidationError::TooLong);
    }
    let s = core::str::from_utf8(bytes).map_err(|_| NameValidationError::NotUtf8)?;
    if s.contains('/') {
        return Err(NameValidationError::ContainsSlash);
    }
    if s == ".." || s.split('/').any(|seg| seg == "..") {
        return Err(NameValidationError::ParentTraversal);
    }
    for suf in RESERVED_SUFFIXES {
        if s.ends_with(suf) {
            return Err(NameValidationError::ReservedSuffix);
        }
    }
    Ok(s)
}

/// Map a [`NameValidationError`] to its kernel-ABI errno. Used by the
/// syscall handlers.
pub const fn name_validation_errno(_e: NameValidationError) -> i64 {
    ERRNO_EINVAL
}

// ─── Handler context ─────────────────────────────────────────────────────────

/// Bundle of references threaded through every overlay model syscall.
/// Mirrors `AuthCtx` — production constructs one on the fly from
/// global state, tests build one with mocks.
pub struct ModelCtx<'a, B, A>
where
    B: OverlayBackend + ?Sized,
    A: OverlayAuditSink + ?Sized,
{
    /// Per-name advisory-lock table.
    pub locks: &'a mut OverlayLockTable,
    /// FS-side overlay writer (or any [`OverlayBackend`] mock).
    pub backend: &'a mut B,
    /// Audit sink.
    pub audit: &'a mut A,
    /// Caller's session id (for lock holder + audit `who` field).
    pub holder: SessionId,
    /// Caller's role.
    pub role: Option<Role>,
    /// User id from the session (`0xFFFF_FFFF` for unauthenticated).
    pub user_id: u32,
    /// Snapshot of `fs.overlay.allow_operator_unhide`.
    pub allow_operator_unhide: bool,
    /// Snapshot of `fs.overlay.require_signed`. Forwarded to the
    /// backend and surfaced here for the deny-record audit.
    pub require_signed: bool,
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// `onnx_model_add(name_ptr, name_len, contents_fd, expected_size,
/// signature_ptr, signature_len) -> 0 | -errno`
///
/// Spec: `fs-overlay-syscalls` Requirement "model_add syscall".
///
/// Returns:
/// - `0` on success (matches the spec scenario "Operator successfully
///   adds a model"; the human-visible result is the audit record).
/// - negative errno on failure per the spec's per-error rows.
pub fn handle_onnx_model_add<B, A>(
    ctx: &mut ModelCtx<'_, B, A>,
    name: &[u8],
    source: &mut dyn ContentSource,
    expected_size: u64,
    signature: Option<&[u8]>,
) -> SyscallResult
where
    B: OverlayBackend + ?Sized,
    A: OverlayAuditSink + ?Sized,
{
    // 1. RBAC (Operator+).
    let permitted = matches!(ctx.role, Some(Role::Operator) | Some(Role::Root));
    if !permitted {
        ctx.audit.append(OverlayAuditEvent::DenyModelAdd {
            who: ctx.user_id,
            role: ctx.role.map(|r| r.as_u8()).unwrap_or(0xFF),
        });
        return ERRNO_EPERM;
    }

    // 2. Validate name + reject reserved suffix (BEFORE acquiring the
    //    lock, per spec scenario "Reserved suffix rejected" — "SHALL
    //    NOT acquire the per-name lock").
    let name_str = match validate_name(name) {
        Ok(s) => s,
        Err(e) => return name_validation_errno(e),
    };

    // 3. Acquire the per-name advisory lock.
    match ctx.locks.try_acquire(name_str.as_bytes(), ctx.holder) {
        Ok(()) => {}
        Err(OverlayLockError::AlreadyHeld) => return ERRNO_EBUSY,
        Err(OverlayLockError::NameTooLong) => return ERRNO_EINVAL,
        Err(OverlayLockError::TableFull) => return ERRNO_EBUSY,
        Err(OverlayLockError::NotHeld) => unreachable!(),
    }

    // 4. Delegate to the backend (it does the cap-pre-flight + stage-
    //    and-rename internally).
    let backend_result = ctx.backend.add(name_str, source, expected_size, signature);

    // 5. Always release the lock — even on failure paths.
    let _ = ctx.locks.release(name_str.as_bytes(), ctx.holder);

    match backend_result {
        Ok(outcome) => {
            ctx.audit.append(OverlayAuditEvent::ModelAdded {
                who: ctx.user_id,
                name: outcome.name.clone(),
                sha3: outcome.sha3,
                size: outcome.size,
            });
            0
        }
        Err(BackendError::NoSpace) => {
            ctx.audit
                .append(OverlayAuditEvent::ModelAddCapacityExceeded {
                    who: ctx.user_id,
                    name: alloc::string::ToString::to_string(name_str),
                    declared: expected_size,
                });
            ERRNO_ENOSPC
        }
        Err(e) => e.errno(),
    }
}

/// `onnx_model_remove(name_ptr, name_len, mode) -> 0 | -errno`
///
/// Spec: `fs-overlay-syscalls` Requirement "model_remove syscall".
///
/// `mode`:
/// - `0` — delete-upper (Root only).
/// - `1` — hide-lower whiteout (Root only).
/// - `2` — unhide whiteout (Root, or Operator iff
///   `allow_operator_unhide=true`).
pub fn handle_onnx_model_remove<B, A>(
    ctx: &mut ModelCtx<'_, B, A>,
    name: &[u8],
    mode: u8,
) -> SyscallResult
where
    B: OverlayBackend + ?Sized,
    A: OverlayAuditSink + ?Sized,
{
    // 1. RBAC by mode.
    let permitted = match mode {
        0 | 1 => matches!(ctx.role, Some(Role::Root)),
        2 => match ctx.role {
            Some(Role::Root) => true,
            Some(Role::Operator) => ctx.allow_operator_unhide,
            _ => false,
        },
        _ => return ERRNO_EINVAL,
    };

    if !permitted {
        ctx.audit.append(OverlayAuditEvent::DenyModelRemove {
            who: ctx.user_id,
            role: ctx.role.map(|r| r.as_u8()).unwrap_or(0xFF),
            mode,
        });
        return ERRNO_EPERM;
    }

    // 2. Validate name.
    let name_str = match validate_name(name) {
        Ok(s) => s,
        Err(e) => return name_validation_errno(e),
    };

    // 3. Dispatch.
    let result = match mode {
        0 => ctx.backend.remove_upper(name_str),
        1 => ctx.backend.write_whiteout(name_str),
        2 => ctx.backend.remove_whiteout(name_str),
        _ => unreachable!(),
    };

    match result {
        Ok(()) => {
            let event = match mode {
                0 => OverlayAuditEvent::ModelRemoved {
                    who: ctx.user_id,
                    name: alloc::string::ToString::to_string(name_str),
                },
                1 => OverlayAuditEvent::ModelHidden {
                    who: ctx.user_id,
                    name: alloc::string::ToString::to_string(name_str),
                },
                2 => OverlayAuditEvent::ModelUnhidden {
                    who: ctx.user_id,
                    name: alloc::string::ToString::to_string(name_str),
                    role: ctx.role.map(|r| r.as_u8()).unwrap_or(0xFF),
                },
                _ => unreachable!(),
            };
            ctx.audit.append(event);
            0
        }
        Err(e) => e.errno(),
    }
}

// ─── Session-release hook ────────────────────────────────────────────────────

/// Release every overlay lock held by `holder` and run aborted-add
/// cleanup on the staging files left behind. Called by the auth
/// session-release path so a logout / idle-sweep tear-down does NOT
/// leak `<name>.tmp` files.
///
/// Returns the number of locks released.
pub fn release_locks_on_session_drop<B>(
    locks: &mut OverlayLockTable,
    backend: &mut B,
    holder: SessionId,
) -> usize
where
    B: OverlayBackend + ?Sized,
{
    let mut buf = [NameLock {
        name: [0; OVERLAY_NAME_MAX],
        name_len: 0,
        holder: SessionId::NONE,
    }; OVERLAY_LOCK_TABLE_CAP];
    let n = locks.release_on_session_drop(holder, &mut buf);
    for entry in buf.iter().take(n) {
        let bytes = entry.name_bytes();
        if let Ok(name) = core::str::from_utf8(bytes) {
            backend.cleanup_orphan_tmp_for(name);
        }
    }
    n
}

// ─── Wave-0 dispatcher hook (still ENOSYS) ───────────────────────────────────

/// `0x36` dispatcher placeholder — the dispatch table in
/// `super::onnx::sys_onnx_model_add` returns `-ENOSYS` until the boot
/// path installs a real [`ModelCtx`]. This function is exposed so the
/// integration test in `kernel/tests/model_syscall_conformance.rs`
/// can call directly into the gate-checked handler.
pub fn dispatch_model_add_unwired() -> SyscallResult {
    ERRNO_ENOSYS
}

/// `0x37` dispatcher placeholder. See [`dispatch_model_add_unwired`].
pub fn dispatch_model_remove_unwired() -> SyscallResult {
    ERRNO_ENOSYS
}

/// Helper exported for the unauthenticated/EFAULT path: returns true
/// iff the supplied user-pointer / length pair is plausibly safe to
/// slice from in the unikernel's same-address-space dispatch.
pub fn check_user_buffer(ptr: usize, len: usize) -> Result<(), i64> {
    if len == 0 && ptr == 0 {
        // Empty optional buffer.
        return Ok(());
    }
    if ptr == 0 {
        return Err(ERRNO_EFAULT);
    }
    if len > MODEL_NAME_MAX * 1024 * 1024 {
        // 256 MiB sanity bound.
        return Err(ERRNO_EINVAL);
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_test_vectors.rs"]
mod test_vectors;

#[cfg(test)]
mod tests {
    use super::test_vectors::*;
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::string::ToString;
    use alloc::vec;
    use smallaios_security::crypto::sha3::Sha3_256;

    // ─── Mock backend ────────────────────────────────────────────────────────

    #[derive(Debug, Default)]
    struct MockBackend {
        files: BTreeMap<String, Vec<u8>>,
        sha3: BTreeMap<String, Sha3_256Digest>,
        sig: BTreeMap<String, Vec<u8>>,
        whiteouts: BTreeMap<String, ()>,
        cleanup_calls: Vec<String>,
        force_no_space: bool,
        force_io: bool,
        cap_remaining: u64,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                cap_remaining: u64::MAX,
                ..Self::default()
            }
        }

        fn with_cap(cap: u64) -> Self {
            Self {
                cap_remaining: cap,
                ..Self::default()
            }
        }

        #[allow(dead_code)]
        fn force_no_space(mut self) -> Self {
            self.force_no_space = true;
            self
        }

        fn force_io(mut self) -> Self {
            self.force_io = true;
            self
        }
    }

    impl OverlayBackend for MockBackend {
        fn add(
            &mut self,
            name: &str,
            source: &mut dyn ContentSource,
            expected_size: u64,
            signature: Option<&[u8]>,
        ) -> Result<AddOutcome, BackendError> {
            if self.force_no_space {
                return Err(BackendError::NoSpace);
            }
            if self.force_io {
                return Err(BackendError::Io);
            }
            if expected_size > self.cap_remaining {
                return Err(BackendError::NoSpace);
            }
            let mut hasher = Sha3_256::new();
            let mut buf = [0u8; 8192];
            let mut bytes = Vec::new();
            loop {
                match source.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if (bytes.len() as u64) + (n as u64) > self.cap_remaining {
                            return Err(BackendError::NoSpace);
                        }
                        hasher.update(&buf[..n]).unwrap();
                        bytes.extend_from_slice(&buf[..n]);
                    }
                    Err(()) => return Err(BackendError::Io),
                }
            }
            let digest = hasher.finalize().unwrap();
            let size = bytes.len() as u64;
            self.files.insert(name.to_string(), bytes);
            self.sha3.insert(name.to_string(), digest);
            let signature_written = if let Some(sig) = signature {
                self.sig.insert(name.to_string(), sig.to_vec());
                true
            } else {
                self.sig.remove(name);
                false
            };
            Ok(AddOutcome {
                name: name.to_string(),
                sha3: digest,
                size,
                signature_written,
            })
        }

        fn remove_upper(&mut self, name: &str) -> Result<(), BackendError> {
            self.files.remove(name);
            self.sha3.remove(name);
            self.sig.remove(name);
            Ok(())
        }

        fn write_whiteout(&mut self, name: &str) -> Result<(), BackendError> {
            self.whiteouts.insert(name.to_string(), ());
            Ok(())
        }

        fn remove_whiteout(&mut self, name: &str) -> Result<(), BackendError> {
            self.whiteouts.remove(name);
            Ok(())
        }

        fn cleanup_orphan_tmp_for(&mut self, name: &str) {
            self.cleanup_calls.push(name.to_string());
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    fn ctx<'a>(
        locks: &'a mut OverlayLockTable,
        backend: &'a mut MockBackend,
        audit: &'a mut CapturingOverlayAuditSink,
        role: Option<Role>,
        allow_operator_unhide: bool,
    ) -> ModelCtx<'a, MockBackend, CapturingOverlayAuditSink> {
        ModelCtx {
            locks,
            backend,
            audit,
            holder: SessionId::from_raw(0x0100_0001),
            role,
            user_id: 7,
            allow_operator_unhide,
            require_signed: false,
        }
    }

    // ─── name validation ─────────────────────────────────────────────────────

    #[test]
    fn validate_name_accepts_normal_model_name() {
        assert_eq!(validate_name(b"resnet50.onnx"), Ok("resnet50.onnx"));
    }

    #[test]
    fn validate_name_rejects_empty() {
        assert_eq!(validate_name(b""), Err(NameValidationError::Empty));
    }

    #[test]
    fn validate_name_rejects_too_long() {
        let big = ALL_A_256;
        assert_eq!(validate_name(big), Err(NameValidationError::TooLong));
    }

    #[test]
    fn validate_name_rejects_not_utf8() {
        assert_eq!(validate_name(NOT_UTF8), Err(NameValidationError::NotUtf8));
    }

    #[test]
    fn validate_name_rejects_slash() {
        assert_eq!(
            validate_name(b"foo/bar"),
            Err(NameValidationError::ContainsSlash)
        );
    }

    #[test]
    fn validate_name_rejects_parent_traversal() {
        assert_eq!(
            validate_name(b".."),
            Err(NameValidationError::ParentTraversal)
        );
    }

    #[test]
    fn validate_name_rejects_each_reserved_suffix() {
        for suf in RESERVED_SUFFIXES {
            let mut name = String::from("foo");
            name.push_str(suf);
            assert_eq!(
                validate_name(name.as_bytes()),
                Err(NameValidationError::ReservedSuffix),
                "should reject reserved suffix {suf}"
            );
        }
    }

    // ─── model_add: happy path + RBAC matrix ─────────────────────────────────

    #[test]
    fn model_add_operator_succeeds() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 5, None);
        assert_eq!(r, 0);
        assert!(matches!(
            audit.events[0],
            OverlayAuditEvent::ModelAdded { ref name, size, .. } if name == "foo.onnx" && size == 5
        ));
    }

    #[test]
    fn model_add_root_succeeds() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(&mut locks, &mut backend, &mut audit, Some(Role::Root), true);
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 5, None);
        assert_eq!(r, 0);
    }

    #[test]
    fn model_add_viewer_denied_eperm_with_audit() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Viewer),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 5, None);
        assert_eq!(r, ERRNO_EPERM);
        assert!(matches!(
            audit.events[0],
            OverlayAuditEvent::DenyModelAdd { .. }
        ));
    }

    #[test]
    fn model_add_unauth_denied_eperm() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(&mut locks, &mut backend, &mut audit, None, true);
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 5, None);
        assert_eq!(r, ERRNO_EPERM);
    }

    // ─── model_add: name validation ──────────────────────────────────────────

    #[test]
    fn model_add_reserved_suffix_rejected_no_lock_acquired() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"foo.whiteout", &mut src, 5, None);
        assert_eq!(r, ERRNO_EINVAL);
        // Spec: "SHALL NOT acquire the per-name lock"
        assert!(!locks.is_held(b"foo.whiteout"));
        // Backend NEVER consulted.
        assert!(backend.files.is_empty());
    }

    #[test]
    fn model_add_path_traversal_rejected() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"..", &mut src, 5, None);
        assert_eq!(r, ERRNO_EINVAL);
    }

    #[test]
    fn model_add_slash_rejected() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"foo/bar", &mut src, 5, None);
        assert_eq!(r, ERRNO_EINVAL);
    }

    #[test]
    fn model_add_too_long_rejected() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, ALL_A_256, &mut src, 5, None);
        assert_eq!(r, ERRNO_EINVAL);
    }

    #[test]
    fn model_add_empty_name_rejected() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"", &mut src, 0, None);
        assert_eq!(r, ERRNO_EINVAL);
    }

    // ─── model_add: lock contention + cap ────────────────────────────────────

    #[test]
    fn model_add_busy_when_lock_already_held() {
        let mut locks = OverlayLockTable::new();
        // Pre-acquire the lock for a different session.
        let other = SessionId::from_raw(0x0200_0002);
        assert!(locks.try_acquire(b"foo.onnx", other).is_ok());

        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 5, None);
        assert_eq!(r, ERRNO_EBUSY);
    }

    #[test]
    fn model_add_capacity_cap_audits_capacity_exceeded() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::with_cap(10);
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BIG);
        let r = handle_onnx_model_add(&mut c, b"big.onnx", &mut src, 200, None);
        assert_eq!(r, ERRNO_ENOSPC);
        assert!(matches!(
            audit.events[0],
            OverlayAuditEvent::ModelAddCapacityExceeded { .. }
        ));
    }

    #[test]
    fn model_add_releases_lock_on_failure() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new().force_io();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 5, None);
        assert_eq!(r, ERRNO_EIO);
        assert!(!locks.is_held(b"foo.onnx"));
    }

    #[test]
    fn model_add_releases_lock_on_success() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 5, None);
        assert_eq!(r, 0);
        assert!(!locks.is_held(b"foo.onnx"));
    }

    #[test]
    fn model_add_with_signature_passes_through_to_backend() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let mut src = SliceSource::new(MODEL_BYTES);
        let sig = SIGNATURE_BYTES;
        let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 5, Some(sig));
        assert_eq!(r, 0);
        assert_eq!(backend.sig.get("foo.onnx").unwrap(), &sig.to_vec());
    }

    // ─── model_remove RBAC matrix ────────────────────────────────────────────

    #[test]
    fn model_remove_root_mode0_succeeds() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        backend
            .files
            .insert("custom.onnx".to_string(), MODEL_BYTES.to_vec());
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(&mut locks, &mut backend, &mut audit, Some(Role::Root), true);
        let r = handle_onnx_model_remove(&mut c, b"custom.onnx", 0);
        assert_eq!(r, 0);
        assert!(!backend.files.contains_key("custom.onnx"));
        assert!(matches!(
            audit.events[0],
            OverlayAuditEvent::ModelRemoved { ref name, .. } if name == "custom.onnx"
        ));
    }

    #[test]
    fn model_remove_root_mode1_writes_whiteout() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(&mut locks, &mut backend, &mut audit, Some(Role::Root), true);
        let r = handle_onnx_model_remove(&mut c, b"llama-3.onnx", 1);
        assert_eq!(r, 0);
        assert!(backend.whiteouts.contains_key("llama-3.onnx"));
        assert!(matches!(
            audit.events[0],
            OverlayAuditEvent::ModelHidden { .. }
        ));
    }

    #[test]
    fn model_remove_root_mode2_unhide_succeeds() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        backend.whiteouts.insert("foo.onnx".to_string(), ());
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Root),
            false,
        );
        let r = handle_onnx_model_remove(&mut c, b"foo.onnx", 2);
        assert_eq!(r, 0);
        assert!(!backend.whiteouts.contains_key("foo.onnx"));
        assert!(matches!(
            audit.events[0],
            OverlayAuditEvent::ModelUnhidden { ref name, role, .. }
                if name == "foo.onnx" && role == Role::Root.as_u8()
        ));
    }

    #[test]
    fn model_remove_operator_mode0_denied() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let r = handle_onnx_model_remove(&mut c, b"foo.onnx", 0);
        assert_eq!(r, ERRNO_EPERM);
        assert!(matches!(
            audit.events[0],
            OverlayAuditEvent::DenyModelRemove { mode: 0, .. }
        ));
    }

    #[test]
    fn model_remove_operator_mode1_denied() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let r = handle_onnx_model_remove(&mut c, b"foo.onnx", 1);
        assert_eq!(r, ERRNO_EPERM);
    }

    #[test]
    fn model_remove_operator_mode2_allowed_when_policy_on() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        backend.whiteouts.insert("foo.onnx".to_string(), ());
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            true,
        );
        let r = handle_onnx_model_remove(&mut c, b"foo.onnx", 2);
        assert_eq!(r, 0);
        assert!(!backend.whiteouts.contains_key("foo.onnx"));
    }

    #[test]
    fn model_remove_operator_mode2_denied_when_policy_off() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(
            &mut locks,
            &mut backend,
            &mut audit,
            Some(Role::Operator),
            false,
        );
        let r = handle_onnx_model_remove(&mut c, b"foo.onnx", 2);
        assert_eq!(r, ERRNO_EPERM);
        assert!(matches!(
            audit.events[0],
            OverlayAuditEvent::DenyModelRemove { mode: 2, .. }
        ));
    }

    #[test]
    fn model_remove_viewer_all_modes_denied() {
        for mode in 0..=2u8 {
            let mut locks = OverlayLockTable::new();
            let mut backend = MockBackend::new();
            let mut audit = CapturingOverlayAuditSink::new();
            let mut c = ctx(
                &mut locks,
                &mut backend,
                &mut audit,
                Some(Role::Viewer),
                true,
            );
            let r = handle_onnx_model_remove(&mut c, b"foo.onnx", mode);
            assert_eq!(r, ERRNO_EPERM, "viewer mode {mode}");
        }
    }

    #[test]
    fn model_remove_invalid_mode_returns_einval() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(&mut locks, &mut backend, &mut audit, Some(Role::Root), true);
        let r = handle_onnx_model_remove(&mut c, b"foo.onnx", 7);
        assert_eq!(r, ERRNO_EINVAL);
    }

    #[test]
    fn model_remove_root_invalid_name_returns_einval() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(&mut locks, &mut backend, &mut audit, Some(Role::Root), true);
        let r = handle_onnx_model_remove(&mut c, b"foo/bar", 0);
        assert_eq!(r, ERRNO_EINVAL);
    }

    // ─── Aborted-add cleanup ────────────────────────────────────────────────

    #[test]
    fn release_locks_on_session_drop_unlinks_orphan_tmp() {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockBackend::new();
        let holder = SessionId::from_raw(0x0100_0001);
        assert!(locks.try_acquire(b"a.onnx", holder).is_ok());
        assert!(locks.try_acquire(b"b.onnx", holder).is_ok());
        let n = release_locks_on_session_drop(&mut locks, &mut backend, holder);
        assert_eq!(n, 2);
        let mut got = backend.cleanup_calls.clone();
        got.sort();
        assert_eq!(got, vec!["a.onnx".to_string(), "b.onnx".to_string()]);
        assert!(!locks.is_held(b"a.onnx"));
        assert!(!locks.is_held(b"b.onnx"));
    }

    #[test]
    fn check_user_buffer_validates_pointers() {
        assert!(check_user_buffer(0, 0).is_ok());
        assert_eq!(check_user_buffer(0, 5), Err(ERRNO_EFAULT));
        assert!(check_user_buffer(0x1000, 5).is_ok());
        // Insanely large length is rejected.
        assert_eq!(
            check_user_buffer(0x1000, MODEL_NAME_MAX * 1024 * 1024 + 1),
            Err(ERRNO_EINVAL)
        );
    }

    #[test]
    fn slice_source_yields_until_eof() {
        let mut s = SliceSource::new(b"hello");
        let mut buf = [0u8; 3];
        assert_eq!(s.read(&mut buf), Ok(3));
        assert_eq!(&buf, b"hel");
        assert_eq!(s.read(&mut buf), Ok(2));
        assert_eq!(&buf[..2], b"lo");
        assert_eq!(s.read(&mut buf), Ok(0));
    }

    #[test]
    fn dispatch_unwired_returns_enosys() {
        assert_eq!(dispatch_model_add_unwired(), ERRNO_ENOSYS);
        assert_eq!(dispatch_model_remove_unwired(), ERRNO_ENOSYS);
    }

    // ─── Phase 5: signature audit variants (additive) ───────────────────────

    #[test]
    fn audit_variant_model_load_unsigned_round_trips() {
        let mut sink = CapturingOverlayAuditSink::new();
        sink.append(OverlayAuditEvent::ModelLoadUnsigned {
            who: 7,
            name: "foo.onnx".to_string(),
        });
        assert_eq!(sink.len(), 1);
        assert!(matches!(
            sink.events[0],
            OverlayAuditEvent::ModelLoadUnsigned { who, ref name }
                if who == 7 && name == "foo.onnx"
        ));
    }

    #[test]
    fn audit_variant_model_signature_invalid_round_trips() {
        let mut sink = CapturingOverlayAuditSink::new();
        sink.append(OverlayAuditEvent::ModelSignatureInvalid {
            who: 9,
            name: "bar.onnx".to_string(),
        });
        assert!(matches!(
            sink.events[0],
            OverlayAuditEvent::ModelSignatureInvalid { who, ref name }
                if who == 9 && name == "bar.onnx"
        ));
    }

    #[test]
    fn audit_variant_model_signature_verified_round_trips() {
        let mut sink = CapturingOverlayAuditSink::new();
        sink.append(OverlayAuditEvent::ModelSignatureVerified {
            who: 11,
            name: "baz.onnx".to_string(),
        });
        assert!(matches!(
            sink.events[0],
            OverlayAuditEvent::ModelSignatureVerified { who, ref name }
                if who == 11 && name == "baz.onnx"
        ));
    }

    #[test]
    fn audit_signature_variants_are_distinct() {
        // Defense-in-depth: the three new variants must not collapse
        // into one (an event-stream consumer matches on the variant
        // tag to dispatch to the right severity / sink).
        let unsigned = OverlayAuditEvent::ModelLoadUnsigned {
            who: 1,
            name: "x".to_string(),
        };
        let invalid = OverlayAuditEvent::ModelSignatureInvalid {
            who: 1,
            name: "x".to_string(),
        };
        let verified = OverlayAuditEvent::ModelSignatureVerified {
            who: 1,
            name: "x".to_string(),
        };
        assert_ne!(unsigned, invalid);
        assert_ne!(invalid, verified);
        assert_ne!(unsigned, verified);
    }

    #[test]
    fn backend_error_to_errno_mapping() {
        assert_eq!(
            BackendError::ReservedSuffix(".sha3".into()).errno(),
            ERRNO_EINVAL
        );
        assert_eq!(BackendError::Busy("x".into()).errno(), ERRNO_EBUSY);
        assert_eq!(BackendError::NoSpace.errno(), ERRNO_ENOSPC);
        assert_eq!(BackendError::ReadOnlyLower("x".into()).errno(), ERRNO_EROFS);
        assert_eq!(BackendError::SignatureRequired.errno(), ERRNO_EAUTH);
        assert_eq!(BackendError::SignatureInvalid.errno(), ERRNO_EAUTH);
        assert_eq!(BackendError::Io.errno(), ERRNO_EIO);
        assert_eq!(BackendError::PermissionDenied.errno(), ERRNO_EPERM);
    }
}
