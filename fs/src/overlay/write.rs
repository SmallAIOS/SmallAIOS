// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Overlay write path — `OverlayWriter` + per-name advisory locks.
//!
//! Spec: `openspec/changes/embedded-overlay-v1/specs/fs-overlay-write/spec.md`.
//!
//! The write path layers atomicity, integrity, and quota-cap policy
//! over a [`WritableUpperLayer`] backend:
//!
//! ## Stage-and-rename atomicity
//!
//! Every [`OverlayWriter::add`] runs:
//!
//! 1. Reject reserved-suffix names ([`super::reserved`]).
//! 2. Acquire the per-name advisory lock — second concurrent caller
//!    on the same name gets `BUSY`.
//! 3. Pre-flight the capacity cap using the operator-declared
//!    `expected_size` (when non-zero). If the upper would exceed
//!    `cap.upper_max_bytes`, reject with `NoSpace`.
//! 4. Stream bytes from `contents_fd` into `<name>.tmp`, computing
//!    SHA-3-256 on the fly. If we cross the cap mid-stream, unlink
//!    `<name>.tmp` and return `NoSpace` (audit `model_add_capacity_exceeded`).
//! 5. Write `<name>.sha3` with the hex-encoded digest.
//! 6. Optionally write `<name>.sig` if the operator supplied one.
//! 7. `fsync`.
//! 8. `rename` `<name>.tmp` → `<name>`.
//! 9. Release the lock.
//!
//! A crash before the rename leaves no partially-visible model. Boot
//! cleanup ([`OverlayWriter::cleanup_orphan_tmp`]) sweeps `*.tmp` from
//! the upper at boot.
//!
//! ## Per-name lock
//!
//! Lock state is in-memory `BTreeMap<String, ()>` (alloc, no `std`),
//! held by an `&mut` on the writer for the duration of the call. The
//! lock is released on success, on failure, and on the writing fd
//! closing without completing (the caller's responsibility — the
//! `OverlayWriter` releases on every return path).
//!
//! ## Capacity-cap enforcement
//!
//! Total upper-layer bytes are tracked by walking `iter_dir` at the
//! upper root and summing every regular file's size. The pre-flight
//! check uses `expected_size`; the in-stream check accumulates real
//! bytes copied. The "below floor" check is the validator's job (in
//! `mgmt::validate`); see the spec scenario "Cap below floor rejected
//! at config time".
//!
//! ## Reserved-suffix rejection
//!
//! Names ending in `.whiteout`, `.opaque`, `.sha3`, `.sig` (the
//! [`super::reserved`] list) are rejected at `add` with
//! [`OverlayWriteError::ReservedSuffix`] and a clear message naming
//! the offending suffix. The lock is NOT acquired in this case.
//!
//! ## Lower-only target
//!
//! [`OverlayWriter::truncate`] (and the higher-level VFS `truncate` /
//! `write` paths reaching it) returns `ReadOnlyLower` for any path
//! resolved to a lower-only entry, with the documented "use model_add
//! under a new name" hint surfaced in the error's `Display`.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use smallaios_security::crypto::sha3::{Sha3_256, Sha3_256Digest};

use super::integrity::{encode_sidecar_bytes, sha3_sidecar_path, sig_sidecar_path, IntegrityError};
use super::reserved::reject_if_reserved;
use super::upper_layer::{UpperEntryKind, UpperError, UpperLayer};

/// `<name>.tmp` staging suffix used by stage-and-rename.
pub const STAGING_SUFFIX: &str = ".tmp";

/// Result of a successful [`OverlayWriter::add`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddResult {
    /// Final upper-relative name (the staging name has been renamed
    /// to this). Provided so the syscall layer can echo it into the
    /// audit record.
    pub name: String,
    /// SHA-3-256 of the written content. The hex-encoded form lives in
    /// `<name>.sha3` on the upper.
    pub sha3: Sha3_256Digest,
    /// Bytes actually copied from `contents_fd` (i.e. the file's size).
    pub size: u64,
    /// `true` iff the caller provided a signature and we wrote
    /// `<name>.sig`.
    pub signature_written: bool,
}

/// Capability-cap and policy snapshot the writer enforces for one or
/// more `add` calls. Constructed by the syscall layer from
/// `mgmt::Config::overlay`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayCap {
    /// Hard cap on total bytes occupied by `/data/models-upper/`. The
    /// pre-flight uses `expected_size` for early rejection; the
    /// in-stream check trips when the running total crosses this
    /// boundary. Sourced from `mgmt.overlay.upper_max_bytes`. The
    /// floor (1 MiB at the validator; 64 MiB documented in the spec
    /// scenario "Cap below floor rejected") is enforced upstream.
    pub upper_max_bytes: u64,
    /// When `true`, `add` rejects an unsigned add and `verify_load`
    /// rejects an unsigned model. Sourced from
    /// `mgmt.overlay.require_signed`.
    pub require_signed: bool,
}

impl OverlayCap {
    /// Construct a [`OverlayCap`] from raw u64/bool inputs. Convenience
    /// for the syscall layer.
    pub const fn new(upper_max_bytes: u64, require_signed: bool) -> Self {
        Self {
            upper_max_bytes,
            require_signed,
        }
    }
}

/// Write-side errors. Distinct from [`super::OverlayError`] so the
/// caller can match against the precise error code without unpacking
/// a wrapped enum chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayWriteError {
    /// Operator-supplied name has a reserved overlay suffix
    /// (`.whiteout`, `.opaque`, `.sha3`, `.sig`). Maps to `-EINVAL`.
    ReservedSuffix(String),
    /// Another caller already holds the per-name advisory lock for
    /// this name. Maps to `-EBUSY` with `Retry-After: 1`.
    Busy(String),
    /// The cap rejected this write — either pre-flight (declared size
    /// too large) or mid-stream (running total crossed the cap).
    /// Maps to `-ENOSPC`. Audit record `model_add_capacity_exceeded`.
    NoSpace,
    /// `add` was called with a path that resolves to a lower-only file
    /// (or any other write that would require modifying the lower).
    /// Maps to `-EROFS`. The `Display` impl carries the documented
    /// hint "use model_add under a new name".
    ReadOnlyLower(String),
    /// Caller supplied an empty name.
    EmptyName,
    /// Required-signed policy is on but no signature was supplied.
    /// Maps to `-EAUTH`. Audit record `model_load_unsigned`.
    SignatureRequired,
    /// Underlying upper-layer I/O failed.
    Io(UpperError),
    /// Integrity-layer error (sidecar emit/decode).
    Integrity(IntegrityError),
    /// Source FD reported a read failure.
    SourceIo,
    /// Whiteout-write RBAC denial. Maps to `-EPERM`. The role enum
    /// lives in `auth`; the writer doesn't pull it in. The syscall
    /// layer translates into the audit record `DENY:model_unhide`.
    PermissionDenied,
}

impl fmt::Display for OverlayWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReservedSuffix(s) => write!(f, "overlay-write: name has reserved suffix: {}", s),
            Self::Busy(name) => {
                write!(
                    f,
                    "overlay-write: per-name lock held: {} (Retry-After: 1)",
                    name
                )
            }
            Self::NoSpace => write!(f, "overlay-write: capacity cap exceeded"),
            Self::ReadOnlyLower(name) => write!(
                f,
                "overlay-write: cannot modify lower-only file {} \
                 (use model_add under a new name)",
                name
            ),
            Self::EmptyName => write!(f, "overlay-write: empty name"),
            Self::SignatureRequired => write!(
                f,
                "overlay-write: require_signed=true but no signature supplied"
            ),
            Self::Io(e) => write!(f, "overlay-write: upper I/O error: {:?}", e),
            Self::Integrity(e) => write!(f, "overlay-write: integrity error: {}", e),
            Self::SourceIo => write!(f, "overlay-write: source fd read error"),
            Self::PermissionDenied => write!(f, "overlay-write: permission denied"),
        }
    }
}

impl From<UpperError> for OverlayWriteError {
    fn from(e: UpperError) -> Self {
        Self::Io(e)
    }
}

impl From<IntegrityError> for OverlayWriteError {
    fn from(e: IntegrityError) -> Self {
        Self::Integrity(e)
    }
}

/// `errno`-style integer codes emitted to the syscall ABI. The kernel
/// maps these into the documented spec values:
///
/// - `EINVAL = 22` — reserved suffix, empty name.
/// - `EBUSY  = 16` — per-name lock contended.
/// - `ENOSPC = 28` — capacity cap exceeded.
/// - `EROFS  = 30` — write to lower-only file.
/// - `EAUTH  = 13` — `EACCES` reused (POSIX convention; SmallAIOS
///   treats the auth-failure code as `13` in the syscall layer).
/// - `EIO    = 5`  — upper-layer I/O / integrity failure.
/// - `EPERM  = 1`  — RBAC denial.
impl OverlayWriteError {
    /// Map to the negative-errno value the syscall ABI returns.
    pub const fn errno(&self) -> i32 {
        match self {
            Self::ReservedSuffix(_) => -22,
            Self::Busy(_) => -16,
            Self::NoSpace => -28,
            Self::ReadOnlyLower(_) => -30,
            Self::EmptyName => -22,
            Self::SignatureRequired => -13,
            Self::Io(_) | Self::SourceIo => -5,
            Self::Integrity(IntegrityError::SignatureMissing)
            | Self::Integrity(IntegrityError::SignatureMalformed)
            | Self::Integrity(IntegrityError::SignatureInvalid) => -13,
            Self::Integrity(_) => -5,
            Self::PermissionDenied => -1,
        }
    }
}

/// Source-side I/O error returned by [`ReadableFd::read`].
///
/// Intentionally opaque (no inner data): the overlay writer treats
/// every source-side failure identically — abort the staging file,
/// release the lock, surface [`OverlayWriteError::SourceIo`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadableFdError;

impl fmt::Display for ReadableFdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "overlay-write: source fd I/O error")
    }
}

/// A read-side file descriptor abstracted over the kernel's fd table.
///
/// The overlay writer needs to stream bytes from an arbitrary source
/// — Zenoh subscriber, HTTP body, local file — without depending on
/// any concrete fd type. The syscall layer wraps each path in a
/// `dyn ReadableFd`.
pub trait ReadableFd {
    /// Read up to `buf.len()` bytes into `buf`. Returns `Ok(0)` at EOF;
    /// `Ok(n)` with `n > 0` for short reads; [`ReadableFdError`] on
    /// I/O error. The writer treats `Err(ReadableFdError)` as
    /// [`OverlayWriteError::SourceIo`].
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ReadableFdError>;
}

/// In-memory [`ReadableFd`] over a byte slice. Used by tests and by
/// the operator-supplied "small payload" path where the bytes are
/// already in user memory.
pub struct SliceReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> SliceReader<'a> {
    /// Construct a reader over `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
}

impl<'a> ReadableFd for SliceReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ReadableFdError> {
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

/// Test-only [`ReadableFd`] that returns [`ReadableFdError`] after a fixed
/// number of successful bytes — exercises the abort-mid-write path.
pub struct ErroringReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    fail_after: usize,
}

impl<'a> ErroringReader<'a> {
    /// Construct a reader that delivers `fail_after` bytes, then fails.
    pub fn new(bytes: &'a [u8], fail_after: usize) -> Self {
        Self {
            bytes,
            pos: 0,
            fail_after,
        }
    }
}

impl<'a> ReadableFd for ErroringReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ReadableFdError> {
        if self.pos >= self.fail_after {
            return Err(ReadableFdError);
        }
        let remaining = core::cmp::min(self.bytes.len(), self.fail_after) - self.pos;
        let n = core::cmp::min(remaining, buf.len());
        if n == 0 {
            return Err(ReadableFdError);
        }
        buf[..n].copy_from_slice(&self.bytes[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

// ─── Writable upper-layer trait ────────────────────────────────────────────────

/// Write extension to [`UpperLayer`]. Production backed by F2FS via
/// [`super::f2fs_upper::F2fsUpperLayer`]; tests use
/// [`MockWritableUpperLayer`].
///
/// `path` arguments are slash-relative to the upper-layer root, never
/// prefixed by `/data/models-upper/`. Implementations SHALL `fsync`
/// inside [`Self::fsync`] (NOT inside `create_or_replace`) so the
/// `OverlayWriter` can batch the durability barrier exactly once
/// before the rename.
pub trait WritableUpperLayer: UpperLayer {
    /// Open a fresh writable file at `path`, replacing any existing
    /// file at that name. Returns the in-memory write handle.
    fn create_or_replace(&mut self, path: &str) -> Result<(), UpperError>;

    /// Append `data` to the most-recently-created file at `path`.
    fn append(&mut self, path: &str, data: &[u8]) -> Result<(), UpperError>;

    /// Atomically rename `src` to `dst`, clobbering any `dst` that
    /// previously existed. The contract matches the F2FS rename
    /// (open readers of `dst` continue to see the pre-rename content).
    fn rename(&mut self, src: &str, dst: &str) -> Result<(), UpperError>;

    /// Remove the file at `path`. Idempotent: removing a non-existent
    /// path is `Ok(())`.
    fn unlink(&mut self, path: &str) -> Result<(), UpperError>;

    /// Force a checkpoint (`fsync`) covering every prior `append` /
    /// `rename` / `unlink` on this layer.
    fn fsync(&mut self) -> Result<(), UpperError>;

    /// Total bytes occupied by every regular file rooted at the upper.
    /// Used by [`OverlayWriter`] to compute the running cap usage.
    /// Default impl walks the upper via `iter_dir`; backends with a
    /// cheaper way (e.g. F2FS i_size accumulator) MAY override.
    fn total_bytes(&self) -> Result<u64, UpperError> {
        accumulate_bytes(self, "")
    }
}

/// Recursively sum the `size` of every regular file at or under
/// `path`. Sidecars (`.sha3`, `.sig`) and whiteouts/opaque markers
/// COUNT as regular files because the cap measures actual upper
/// occupancy; the spec's "total bytes occupied by /data/models-upper/"
/// is bytes-on-disk, not user-visible-bytes.
fn accumulate_bytes<U: UpperLayer + ?Sized>(upper: &U, path: &str) -> Result<u64, UpperError> {
    let entries = match upper.iter_dir(path) {
        Ok(e) => e,
        Err(UpperError::NotFound) | Err(UpperError::NotADirectory) => return Ok(0),
        Err(e) => return Err(e),
    };
    let mut total: u64 = 0;
    for entry in entries {
        let child_path = if path.is_empty() {
            entry.name.clone()
        } else {
            format!("{}/{}", path, entry.name)
        };
        match entry.kind {
            UpperEntryKind::RegularFile | UpperEntryKind::Whiteout => {
                total = total.saturating_add(entry.size);
            }
            UpperEntryKind::Directory | UpperEntryKind::Opaque => {
                total = total.saturating_add(accumulate_bytes(upper, &child_path)?);
            }
        }
    }
    Ok(total)
}

// ─── Mock writable upper layer (for tests) ─────────────────────────────────────

/// In-memory writable upper layer. Mirrors [`super::MockUpperLayer`]
/// but exposes the [`WritableUpperLayer`] trait. Used by Phase 2
/// conformance tests in `fs/tests/overlay_write_conformance.rs`.
#[derive(Debug, Default, Clone)]
pub struct MockWritableUpperLayer {
    files: BTreeMap<String, Vec<u8>>,
    /// Set of "directories" we know about — auto-populated from file
    /// paths but the mock doesn't enforce dir existence very strictly.
    dirs: BTreeSet<String>,
}

impl MockWritableUpperLayer {
    /// Construct an empty mock writable layer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a regular file with the given content (test helper).
    pub fn add_file(&mut self, path: &str, contents: &[u8]) {
        let key = normalize(path);
        self.ensure_parents(&key);
        self.files.insert(key, contents.to_vec());
    }

    /// Insert a whiteout marker `<name>.whiteout`.
    pub fn add_whiteout(&mut self, name: &str) {
        let key = format!("{}.whiteout", normalize(name));
        self.ensure_parents(&key);
        self.files.insert(key, Vec::new());
    }

    /// Read a file's bytes (test helper).
    pub fn raw_bytes(&self, path: &str) -> Option<&Vec<u8>> {
        self.files.get(&normalize(path))
    }

    /// `true` iff the path exists as a regular file.
    pub fn exists(&self, path: &str) -> bool {
        self.files.contains_key(&normalize(path))
    }

    /// Number of regular files in the mock.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    fn ensure_parents(&mut self, key: &str) {
        let mut walked = String::new();
        for segment in key.split('/').filter(|s| !s.is_empty()) {
            if !walked.is_empty() {
                walked.push('/');
            }
            let candidate = format!("{}{}", walked, segment);
            if candidate == key {
                break;
            }
            self.dirs.insert(candidate.clone());
            walked = candidate;
        }
    }
}

fn normalize(path: &str) -> String {
    path.trim_start_matches('/').trim_end_matches('/').into()
}

impl UpperLayer for MockWritableUpperLayer {
    fn lookup(&self, path: &str) -> Result<Option<super::upper_layer::UpperEntry>, UpperError> {
        let key = normalize(path);
        if key.is_empty() {
            return Ok(Some(super::upper_layer::UpperEntry {
                name: String::new(),
                kind: UpperEntryKind::Directory,
                size: 0,
                mode: 0o755,
            }));
        }
        let bare_name = key.rsplit('/').next().unwrap_or("").to_string();

        // Whiteout pseudo-entry: `foo` resolves to Whiteout if
        // `foo.whiteout` exists.
        let whiteout_key = format!("{}.whiteout", key);
        if self.files.contains_key(&whiteout_key) {
            return Ok(Some(super::upper_layer::UpperEntry {
                name: bare_name,
                kind: UpperEntryKind::Whiteout,
                size: 0,
                mode: 0o644,
            }));
        }
        if let Some(bytes) = self.files.get(&key) {
            return Ok(Some(super::upper_layer::UpperEntry {
                name: bare_name,
                kind: UpperEntryKind::RegularFile,
                size: bytes.len() as u64,
                mode: 0o644,
            }));
        }
        if self.dirs.contains(&key) {
            let opaque_key = format!("{}/.opaque", key);
            let kind = if self.files.contains_key(&opaque_key) {
                UpperEntryKind::Opaque
            } else {
                UpperEntryKind::Directory
            };
            return Ok(Some(super::upper_layer::UpperEntry {
                name: bare_name,
                kind,
                size: 0,
                mode: 0o755,
            }));
        }
        Ok(None)
    }

    fn read_file(&self, path: &str, offset: u64, buf: &mut [u8]) -> Result<usize, UpperError> {
        let key = normalize(path);
        let bytes = self.files.get(&key).ok_or(UpperError::NotFound)?;
        let len = bytes.len() as u64;
        if offset > len {
            return Err(UpperError::OutOfRange);
        }
        let start = offset as usize;
        let avail = bytes.len() - start;
        let n = core::cmp::min(buf.len(), avail);
        buf[..n].copy_from_slice(&bytes[start..start + n]);
        Ok(n)
    }

    fn iter_dir(&self, path: &str) -> Result<Vec<super::upper_layer::UpperEntry>, UpperError> {
        let key = normalize(path);
        if !key.is_empty() && !self.dirs.contains(&key) {
            // It might be a regular file path (Not a directory).
            if self.files.contains_key(&key) {
                return Err(UpperError::NotADirectory);
            }
            return Err(UpperError::NotFound);
        }
        let prefix = if key.is_empty() {
            String::new()
        } else {
            format!("{}/", key)
        };

        let mut out = Vec::new();
        // Files first.
        for (k, bytes) in &self.files {
            if !k.starts_with(&prefix) {
                continue;
            }
            let suffix = &k[prefix.len()..];
            if suffix.is_empty() || suffix.contains('/') {
                continue;
            }
            // Hide `.opaque` markers from listings.
            if suffix == ".opaque" {
                continue;
            }
            // Whiteout: surface bare name with kind Whiteout.
            if let Some(stripped) = suffix.strip_suffix(".whiteout") {
                out.push(super::upper_layer::UpperEntry {
                    name: stripped.into(),
                    kind: UpperEntryKind::Whiteout,
                    size: 0,
                    mode: 0o644,
                });
                continue;
            }
            out.push(super::upper_layer::UpperEntry {
                name: suffix.into(),
                kind: UpperEntryKind::RegularFile,
                size: bytes.len() as u64,
                mode: 0o644,
            });
        }
        // Directories.
        for d in &self.dirs {
            if !d.starts_with(&prefix) {
                continue;
            }
            let suffix = &d[prefix.len()..];
            if suffix.is_empty() || suffix.contains('/') {
                continue;
            }
            let opaque_key = format!("{}/.opaque", d);
            let kind = if self.files.contains_key(&opaque_key) {
                UpperEntryKind::Opaque
            } else {
                UpperEntryKind::Directory
            };
            out.push(super::upper_layer::UpperEntry {
                name: suffix.into(),
                kind,
                size: 0,
                mode: 0o755,
            });
        }
        Ok(out)
    }
}

impl WritableUpperLayer for MockWritableUpperLayer {
    fn create_or_replace(&mut self, path: &str) -> Result<(), UpperError> {
        let key = normalize(path);
        if key.is_empty() {
            return Err(UpperError::NotARegularFile);
        }
        self.ensure_parents(&key);
        self.files.insert(key, Vec::new());
        Ok(())
    }

    fn append(&mut self, path: &str, data: &[u8]) -> Result<(), UpperError> {
        let key = normalize(path);
        let entry = self.files.get_mut(&key).ok_or(UpperError::NotFound)?;
        entry.extend_from_slice(data);
        Ok(())
    }

    fn rename(&mut self, src: &str, dst: &str) -> Result<(), UpperError> {
        let src_key = normalize(src);
        let dst_key = normalize(dst);
        if src_key == dst_key {
            return Ok(());
        }
        let bytes = self.files.remove(&src_key).ok_or(UpperError::NotFound)?;
        self.ensure_parents(&dst_key);
        self.files.insert(dst_key, bytes);
        Ok(())
    }

    fn unlink(&mut self, path: &str) -> Result<(), UpperError> {
        let key = normalize(path);
        self.files.remove(&key);
        Ok(())
    }

    fn fsync(&mut self) -> Result<(), UpperError> {
        Ok(())
    }
}

// ─── OverlayWriter ─────────────────────────────────────────────────────────────

/// Per-name advisory lock map. Owned by the writer; held for the
/// duration of a single `add` call.
#[derive(Debug, Default)]
struct LockTable {
    held: BTreeSet<String>,
}

impl LockTable {
    fn new() -> Self {
        Self::default()
    }

    fn try_acquire(&mut self, name: &str) -> bool {
        if self.held.contains(name) {
            return false;
        }
        self.held.insert(name.to_string());
        true
    }

    fn release(&mut self, name: &str) {
        self.held.remove(name);
    }

    fn held(&self, name: &str) -> bool {
        self.held.contains(name)
    }
}

/// Stage-and-rename writer for the overlay's upper layer.
///
/// Generic over the writable backend. Production wires it to
/// [`super::f2fs_upper::F2fsUpperLayer`]; tests use
/// [`MockWritableUpperLayer`].
pub struct OverlayWriter<U: WritableUpperLayer> {
    upper: U,
    cap: OverlayCap,
    locks: LockTable,
}

impl<U: WritableUpperLayer> OverlayWriter<U> {
    /// Construct a writer with the given upper-layer backend and policy.
    pub fn new(upper: U, cap: OverlayCap) -> Self {
        Self {
            upper,
            cap,
            locks: LockTable::new(),
        }
    }

    /// Borrow the upper-layer for read inspection. Mostly used by the
    /// merge layer when wiring this writer's upper into a `lookup` call.
    pub fn upper(&self) -> &U {
        &self.upper
    }

    /// Mutable upper-layer accessor (used by tests + the syscall layer
    /// for whiteout / unlink that doesn't pass through `add`).
    pub fn upper_mut(&mut self) -> &mut U {
        &mut self.upper
    }

    /// Borrow the policy snapshot.
    pub fn cap(&self) -> &OverlayCap {
        &self.cap
    }

    /// Replace the policy snapshot. Used when mgmt config flips
    /// `require_signed` or `upper_max_bytes` at runtime.
    pub fn set_cap(&mut self, cap: OverlayCap) {
        self.cap = cap;
    }

    /// `true` iff the per-name lock for `name` is held.
    pub fn lock_held(&self, name: &str) -> bool {
        self.locks.held(name)
    }

    /// Stage-and-rename atomic add.
    ///
    /// On success the upper has `<name>` (final), `<name>.sha3`
    /// (fingerprint), and optionally `<name>.sig` (signature). The
    /// `<name>.tmp` staging file has been renamed and no longer exists.
    ///
    /// On any failure path the staging artifacts are unlinked and the
    /// per-name lock is released.
    pub fn add(
        &mut self,
        name: &str,
        contents_fd: &mut dyn ReadableFd,
        expected_size: u64,
        signature: Option<&[u8]>,
    ) -> Result<AddResult, OverlayWriteError> {
        // 1. Reject empty + reserved names.
        if name.is_empty() {
            return Err(OverlayWriteError::EmptyName);
        }
        if let Err(e) = reject_if_reserved(name) {
            if let super::OverlayError::ReservedSuffix(s) = e {
                return Err(OverlayWriteError::ReservedSuffix(s));
            }
            unreachable!("reject_if_reserved only emits ReservedSuffix");
        }

        // 2. Required-signed policy: a missing signature is a hard
        //    rejection BEFORE we acquire the lock or touch the upper.
        if self.cap.require_signed && signature.is_none() {
            return Err(OverlayWriteError::SignatureRequired);
        }

        // 3. Acquire the per-name lock.
        if !self.locks.try_acquire(name) {
            return Err(OverlayWriteError::Busy(name.to_string()));
        }

        // 4. Pre-flight cap check using `expected_size` (best effort —
        //    a zero declared size means the operator doesn't know yet,
        //    in which case the in-stream check is the real backstop).
        let cap_remaining = || -> Result<u64, OverlayWriteError> {
            let used = self.upper.total_bytes()?;
            Ok(self.cap.upper_max_bytes.saturating_sub(used))
        };
        if expected_size > 0 {
            let remaining = cap_remaining()?;
            if expected_size > remaining {
                self.locks.release(name);
                return Err(OverlayWriteError::NoSpace);
            }
        }

        // The actual streaming work — split into a closure so we can
        // unconditionally clean up state on the return path. Marked
        // `mut` because the closure mutably borrows `contents_fd` for
        // each chunked read.
        let mut streaming = |this: &mut OverlayWriter<U>| -> Result<AddResult, OverlayWriteError> {
            let staging = format!("{}{}", name, STAGING_SUFFIX);
            let sha3_path = sha3_sidecar_path(name);
            let sig_path = sig_sidecar_path(name);

            // Best-effort cleanup of any dangling staging file.
            this.upper.unlink(&staging)?;

            // Open the staging file.
            this.upper.create_or_replace(&staging)?;

            let initial_used = this.upper.total_bytes()?;
            let mut hasher = Sha3_256::new();
            let mut copied: u64 = 0;
            let mut chunk = [0u8; 8192];

            loop {
                let n = match contents_fd.read(&mut chunk) {
                    Ok(n) => n,
                    Err(ReadableFdError) => {
                        // Source aborted — clean up staging artifacts.
                        let _ = this.upper.unlink(&staging);
                        return Err(OverlayWriteError::SourceIo);
                    }
                };
                if n == 0 {
                    break;
                }
                let proposed = initial_used.saturating_add(copied + n as u64);
                if proposed > this.cap.upper_max_bytes {
                    // Cap tripped mid-stream.
                    let _ = this.upper.unlink(&staging);
                    return Err(OverlayWriteError::NoSpace);
                }
                hasher
                    .update(&chunk[..n])
                    .expect("sha3 update on fresh hasher cannot fail");
                this.upper.append(&staging, &chunk[..n])?;
                copied += n as u64;
            }

            let digest = hasher
                .finalize()
                .expect("sha3 finalize on fresh hasher cannot fail");

            // 5. Sidecar.
            let sidecar_bytes = encode_sidecar_bytes(&digest);
            this.upper.create_or_replace(&sha3_path)?;
            this.upper.append(&sha3_path, &sidecar_bytes)?;

            // 6. Optional signature sidecar.
            let mut signature_written = false;
            if let Some(sig) = signature {
                this.upper.create_or_replace(&sig_path)?;
                this.upper.append(&sig_path, sig)?;
                signature_written = true;
            } else {
                // Make sure no stale sig from a prior add lingers.
                this.upper.unlink(&sig_path)?;
            }

            // 7. fsync.
            this.upper.fsync()?;

            // 8. Atomic rename: staging -> final.
            this.upper.rename(&staging, name)?;
            this.upper.fsync()?;

            Ok(AddResult {
                name: name.to_string(),
                sha3: digest,
                size: copied,
                signature_written,
            })
        };

        let result = streaming(self);

        // 9. Always release the lock.
        self.locks.release(name);
        result
    }

    /// Remove the upper-layer entry at `name` and its `.sha3`/`.sig`
    /// sidecars. No-op for entries the upper doesn't have. This is the
    /// `mode=0` path of `model_remove`.
    pub fn remove_upper(&mut self, name: &str) -> Result<(), OverlayWriteError> {
        if name.is_empty() {
            return Err(OverlayWriteError::EmptyName);
        }
        if let Err(e) = reject_if_reserved(name) {
            if let super::OverlayError::ReservedSuffix(s) = e {
                return Err(OverlayWriteError::ReservedSuffix(s));
            }
            unreachable!();
        }
        self.upper.unlink(name)?;
        let sha3 = sha3_sidecar_path(name);
        self.upper.unlink(&sha3)?;
        let sig = sig_sidecar_path(name);
        self.upper.unlink(&sig)?;
        self.upper.fsync()?;
        Ok(())
    }

    /// Write a whiteout `<name>.whiteout` to hide the lower's `<name>`.
    /// This is the `mode=1` path of `model_remove`. RBAC enforcement
    /// (Root-only) is the syscall layer's job; the writer accepts the
    /// call unconditionally.
    pub fn write_whiteout(&mut self, name: &str) -> Result<(), OverlayWriteError> {
        if name.is_empty() {
            return Err(OverlayWriteError::EmptyName);
        }
        // The whiteout's filename ends in `.whiteout`, but `name` here
        // is the *bare* name being hidden. The bare name is rejected
        // for reserved suffixes so an operator can't backdoor a
        // sidecar by hiding it.
        if let Err(e) = reject_if_reserved(name) {
            if let super::OverlayError::ReservedSuffix(s) = e {
                return Err(OverlayWriteError::ReservedSuffix(s));
            }
            unreachable!();
        }
        let path = format!("{}.whiteout", name);
        self.upper.create_or_replace(&path)?;
        self.upper.fsync()?;
        Ok(())
    }

    /// Remove `<name>.whiteout` (un-hide the lower). This is the
    /// `mode=2` path of `model_remove`. RBAC enforcement (Root by
    /// default; Operator iff `allow_operator_unhide`) is the syscall
    /// layer's job — see [`whiteout_remove_allowed_for`].
    pub fn remove_whiteout(&mut self, name: &str) -> Result<(), OverlayWriteError> {
        if name.is_empty() {
            return Err(OverlayWriteError::EmptyName);
        }
        let path = format!("{}.whiteout", name);
        self.upper.unlink(&path)?;
        self.upper.fsync()?;
        Ok(())
    }

    /// Reject any attempt to truncate a path that resolves to a
    /// lower-only entry. This is the `-EROFS` path of the spec
    /// requirement "Write to lower-only file rejected".
    ///
    /// Implementation: caller passes the path's lower-vs-upper
    /// classification (the syscall layer already has this from the
    /// merge `lookup` it runs first). The writer just emits the right
    /// error variant + hint.
    pub fn truncate(&mut self, name: &str, lower_only: bool) -> Result<(), OverlayWriteError> {
        if lower_only {
            return Err(OverlayWriteError::ReadOnlyLower(name.to_string()));
        }
        Ok(())
    }

    /// Unlink any orphan `*.tmp` files at the upper root. Idempotent.
    /// Called once at boot before the overlay is exposed via VFS.
    pub fn cleanup_orphan_tmp(&mut self) -> Result<usize, OverlayWriteError> {
        let mut removed = 0usize;
        let entries = self.upper.iter_dir("")?;
        let names: Vec<String> = entries
            .into_iter()
            .filter(|e| {
                matches!(e.kind, UpperEntryKind::RegularFile) && e.name.ends_with(STAGING_SUFFIX)
            })
            .map(|e| e.name)
            .collect();
        for name in names {
            self.upper.unlink(&name)?;
            removed += 1;
        }
        if removed > 0 {
            self.upper.fsync()?;
        }
        Ok(removed)
    }
}

// ─── Whiteout-removal RBAC helper ─────────────────────────────────────────────

/// RBAC role enum opaque to the writer (which does NOT depend on
/// `auth`). The kernel's syscall layer translates `auth::Role` to this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallerRole {
    /// Kernel root account — full overlay write access.
    Root,
    /// Operator role — model lifecycle, but un-hide is policy-gated.
    Operator,
    /// Viewer role — read-only telemetry. Cannot write the overlay.
    Viewer,
}

/// `true` iff `role` is permitted to remove a whiteout when policy is
/// `allow_operator_unhide`. Spec: `fs-overlay-write` Requirement
/// "Whiteout removal RBAC".
///
/// - `Root` always allowed.
/// - `Operator` allowed iff `allow_operator_unhide = true`.
/// - `Viewer` never allowed.
pub const fn whiteout_remove_allowed_for(role: CallerRole, allow_operator_unhide: bool) -> bool {
    match role {
        CallerRole::Root => true,
        CallerRole::Operator => allow_operator_unhide,
        CallerRole::Viewer => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(max: u64, signed: bool) -> OverlayCap {
        OverlayCap::new(max, signed)
    }

    #[test]
    fn slice_reader_yields_all_then_zero() {
        let mut r = SliceReader::new(b"hello");
        let mut buf = [0u8; 3];
        assert_eq!(r.read(&mut buf), Ok(3));
        assert_eq!(&buf, b"hel");
        assert_eq!(r.read(&mut buf), Ok(2));
        assert_eq!(&buf[..2], b"lo");
        assert_eq!(r.read(&mut buf), Ok(0));
    }

    #[test]
    fn add_writes_file_and_sidecar() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, false));
        let mut src = SliceReader::new(b"hello");
        let r = w.add("foo.onnx", &mut src, 5, None).unwrap();
        assert_eq!(r.name, "foo.onnx");
        assert_eq!(r.size, 5);
        assert!(!r.signature_written);
        // Final file exists; staging gone.
        assert!(w.upper().exists("foo.onnx"));
        assert!(!w.upper().exists("foo.onnx.tmp"));
        // Sidecar exists with hex-encoded digest.
        let sidecar = w.upper().raw_bytes("foo.onnx.sha3").unwrap();
        assert_eq!(sidecar.len(), 64);
    }

    #[test]
    fn add_rejects_reserved_suffix() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, false));
        let mut src = SliceReader::new(b"x");
        let err = w
            .add("foo.whiteout", &mut src, 1, None)
            .expect_err("must reject");
        match err {
            OverlayWriteError::ReservedSuffix(s) => assert_eq!(s, ".whiteout"),
            e => panic!("expected ReservedSuffix, got {:?}", e),
        }
    }

    #[test]
    fn add_rejects_empty_name() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, false));
        let mut src = SliceReader::new(b"x");
        assert_eq!(
            w.add("", &mut src, 1, None),
            Err(OverlayWriteError::EmptyName)
        );
    }

    #[test]
    fn add_preflight_no_space_when_declared_size_too_large() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(100, false));
        let mut src = SliceReader::new(&[0u8; 200]);
        assert_eq!(
            w.add("big", &mut src, 200, None),
            Err(OverlayWriteError::NoSpace)
        );
    }

    #[test]
    fn add_in_stream_no_space_unlinks_staging() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(100, false));
        // Declared size 0 (caller doesn't know) → only the in-stream
        // check protects us. 200 bytes vs 100 cap → trips mid-stream.
        let mut src = SliceReader::new(&[0u8; 200]);
        assert_eq!(
            w.add("big", &mut src, 0, None),
            Err(OverlayWriteError::NoSpace)
        );
        // Staging file SHALL be unlinked.
        assert!(!w.upper().exists("big.tmp"));
        assert!(!w.upper().exists("big"));
    }

    #[test]
    fn add_releases_lock_on_failure() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(100, false));
        let mut src = SliceReader::new(&[0u8; 200]);
        let _ = w.add("big", &mut src, 200, None);
        assert!(!w.lock_held("big"));
    }

    #[test]
    fn lock_table_busy_round_trip() {
        let mut t = LockTable::new();
        assert!(t.try_acquire("foo"));
        assert!(!t.try_acquire("foo")); // contended
        assert!(t.try_acquire("bar")); // different name OK
        t.release("foo");
        assert!(t.try_acquire("foo"));
    }

    #[test]
    fn write_whiteout_creates_marker() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, false));
        w.write_whiteout("foo").unwrap();
        assert!(w.upper().exists("foo.whiteout"));
    }

    #[test]
    fn remove_whiteout_deletes_marker() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, false));
        w.upper_mut().add_whiteout("foo");
        assert!(w.upper().exists("foo.whiteout"));
        w.remove_whiteout("foo").unwrap();
        assert!(!w.upper().exists("foo.whiteout"));
    }

    #[test]
    fn truncate_lower_only_returns_rofs() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, false));
        let err = w.truncate("llama-3.onnx", true).expect_err("EROFS");
        match err {
            OverlayWriteError::ReadOnlyLower(ref n) => assert_eq!(n, "llama-3.onnx"),
            e => panic!("expected ReadOnlyLower, got {:?}", e),
        }
        // Display includes the documented hint.
        let s = alloc::format!("{}", err);
        assert!(s.contains("model_add under a new name"));
    }

    #[test]
    fn cleanup_orphan_tmp_removes_staging() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, false));
        w.upper_mut().add_file("foo.tmp", b"junk");
        w.upper_mut().add_file("real.onnx", b"good");
        let removed = w.cleanup_orphan_tmp().unwrap();
        assert_eq!(removed, 1);
        assert!(!w.upper().exists("foo.tmp"));
        assert!(w.upper().exists("real.onnx"));
    }

    #[test]
    fn require_signed_rejects_unsigned() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, true));
        let mut src = SliceReader::new(b"x");
        assert_eq!(
            w.add("foo", &mut src, 1, None),
            Err(OverlayWriteError::SignatureRequired)
        );
    }

    #[test]
    fn errno_mapping_matches_spec() {
        assert_eq!(
            OverlayWriteError::ReservedSuffix(".whiteout".into()).errno(),
            -22
        );
        assert_eq!(OverlayWriteError::Busy("x".into()).errno(), -16);
        assert_eq!(OverlayWriteError::NoSpace.errno(), -28);
        assert_eq!(OverlayWriteError::ReadOnlyLower("x".into()).errno(), -30);
        assert_eq!(OverlayWriteError::SignatureRequired.errno(), -13);
        assert_eq!(OverlayWriteError::PermissionDenied.errno(), -1);
    }

    #[test]
    fn whiteout_remove_allowed_matrix() {
        // Root: always allowed.
        assert!(whiteout_remove_allowed_for(CallerRole::Root, false));
        assert!(whiteout_remove_allowed_for(CallerRole::Root, true));
        // Operator: gated on policy.
        assert!(!whiteout_remove_allowed_for(CallerRole::Operator, false));
        assert!(whiteout_remove_allowed_for(CallerRole::Operator, true));
        // Viewer: never.
        assert!(!whiteout_remove_allowed_for(CallerRole::Viewer, false));
        assert!(!whiteout_remove_allowed_for(CallerRole::Viewer, true));
    }

    #[test]
    fn add_with_signature_writes_sig_sidecar() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, false));
        let mut src = SliceReader::new(b"hello");
        let sig = [0xABu8; 16];
        let r = w.add("foo", &mut src, 5, Some(&sig)).unwrap();
        assert!(r.signature_written);
        assert_eq!(w.upper().raw_bytes("foo.sig").unwrap(), &sig.to_vec());
    }

    #[test]
    fn remove_upper_clears_sidecars() {
        let mut w = OverlayWriter::new(MockWritableUpperLayer::new(), cap(1024, false));
        w.upper_mut().add_file("foo", b"x");
        w.upper_mut().add_file("foo.sha3", b"y");
        w.upper_mut().add_file("foo.sig", b"z");
        w.remove_upper("foo").unwrap();
        assert!(!w.upper().exists("foo"));
        assert!(!w.upper().exists("foo.sha3"));
        assert!(!w.upper().exists("foo.sig"));
    }
}
