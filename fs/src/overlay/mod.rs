// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! OverlayFS-style merged `/models/` view.
//!
//! Phase 1 of `embedded-overlay-v1` ships **read-only merge logic**.
//! The upper-layer write path, syscalls, and integrity verification
//! land in later phases.
//!
//! ## Layering
//!
//! At runtime `/models/` is a 2-layer merge:
//!
//! - **lower** — the active squashfs A/B slot (read-only, supplied by
//!   `embedded-filesystem-v1`'s [`squashfs::Squashfs`] reader).
//! - **upper** — `/data/models-upper/` on F2FS (read-write, supplied
//!   by `embedded-filesystem-v1` Phase 7 once the F2FS RW path lands).
//!
//! Phase 1 abstracts both layers behind traits ([`UpperLayer`] and
//! [`LowerLayer`]) so the merge logic compiles and tests against
//! synthetic in-memory fixtures while F2FS is still in flight.
//!
//! ## Merge semantics (read path)
//!
//! - **Upper-wins precedence.** A regular file in the upper shadows
//!   the same name in the lower.
//! - **Whiteout files.** A regular file `<name>.whiteout` in the upper
//!   hides the lower's `<name>`. The whiteout file's contents are
//!   never inspected — its mere presence is the marker.
//! - **Opaque dirs.** A regular file `<dir>/.opaque` in an upper
//!   directory hides the entire lower-side subtree at that path; only
//!   upper entries under `<dir>/` are listed.
//! - **Reserved suffixes.** Names ending in `.whiteout`, `.opaque`,
//!   `.sha3`, `.sig` are filtered from `readdir` output (they're
//!   internal markers) and rejected at the `model_add` boundary
//!   (Phase 3).
//!
//! See `openspec/changes/embedded-overlay-v1/specs/fs-overlay-mount/spec.md`.
//!
//! ## Module map
//!
//! | Module           | Role                                                          |
//! |------------------|---------------------------------------------------------------|
//! | [`upper_layer`]  | `UpperLayer` trait + `MockUpperLayer` for tests               |
//! | [`lower_layer`]  | `LowerLayer` trait + `MockLowerLayer` for tests               |
//! | [`whiteout`]     | `<name>.whiteout` regular-file detection                      |
//! | [`opaque`]       | `<dir>/.opaque` regular-file detection                        |
//! | [`reserved`]     | Reserved-suffix list + name rejection                         |
//! | [`lookup`]       | Upper-wins path lookup resolver                               |
//! | [`merge`]        | `readdir` merger (upper ∪ lower − whiteouts − opaque)         |
//! | [`write`]        | `OverlayWriter`: stage-and-rename, locks, cap (Phase 2)       |
//! | [`integrity`]    | SHA-3-256 sidecar + ML-DSA-65 signature sidecar (Phase 2)     |
//! | [`conflict`]     | Boot-time A/B-swap conflict detection (Phase 2)               |
//! | [`f2fs_upper`]   | Production `WritableUpperLayer` backed by F2FS (Phase 2)      |
//!
//! Phase 1 read-only merge is frozen; Phase 2 adds the write path
//! and integrity verification.

extern crate alloc;

use alloc::string::String;
use core::fmt;

pub mod boot_cleanup;
pub mod conflict;
pub mod conflict_scan;
#[cfg(feature = "fs-on-disk-mounts")]
pub mod f2fs_upper;
pub mod integrity;
pub mod kernel_adapter;
pub mod lookup;
pub mod lower_layer;
pub mod merge;
pub mod opaque;
pub mod reserved;
pub mod upper_layer;
pub mod whiteout;
pub mod write;

pub use boot_cleanup::{run_boot_cleanup, BootCleanupReport};
pub use conflict::{
    detect_conflicts, ConflictMemo, ConflictReport, ConflictWarning, ConflictWarningKind,
    LowerManifest, MapLowerManifest, OverlayConflict,
};
pub use conflict_scan::{
    run_boot_conflict_scan, CapturingConflictAuditSink, ConflictAuditSink, ConflictScanEvent,
};
#[cfg(feature = "fs-on-disk-mounts")]
pub use f2fs_upper::{F2fsUpperLayer, DEFAULT_UPPER_BASE};
pub use integrity::{
    decode_hex, encode_hex, encode_sidecar_bytes, hash_bytes, sha3_sidecar_path, sig_sidecar_path,
    signature_present, verify_fingerprint, verify_load, verify_signature, verify_signed,
    IntegrityError, LoadVerifyOutcome, SHA3_SIDECAR_SUFFIX, SIG_SIDECAR_SUFFIX,
};
pub use kernel_adapter::{KernelOverlayAdapter, OVERLAY_UPPER_BASE};
pub use lookup::{lookup, LookupResult};
pub use lower_layer::{LowerEntry, LowerEntryKind, LowerError, LowerLayer, MockLowerLayer};
pub use merge::{merge_readdir, MergedEntry, MergedEntryKind};
pub use opaque::{is_opaque_dir, OPAQUE_MARKER_NAME};
pub use reserved::{is_reserved_name, reject_if_reserved, RESERVED_SUFFIXES};
pub use upper_layer::{MockUpperLayer, UpperEntry, UpperEntryKind, UpperError, UpperLayer};
pub use whiteout::{is_whiteout, list_whiteouts_in, WHITEOUT_SUFFIX};
pub use write::{
    whiteout_remove_allowed_for, AddResult, CallerRole, ErroringReader, MockWritableUpperLayer,
    OverlayCap, OverlayWriteError, OverlayWriter, ReadableFd, ReadableFdError, SliceReader,
    WritableUpperLayer, STAGING_SUFFIX,
};

/// Errors produced by the overlay merge layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayError {
    /// Upper layer reported an error.
    Upper(UpperError),
    /// Lower layer reported an error.
    Lower(LowerError),
    /// Operator-supplied name conflicts with a reserved suffix
    /// (`.whiteout`, `.opaque`, `.sha3`, `.sig`).
    ReservedSuffix(String),
    /// A path was supplied that does not start with `/`.
    InvalidPath,
    /// A write was attempted on the read-only Phase 1 overlay
    /// (Phase 2 lifts this).
    ReadOnly,
}

impl fmt::Display for OverlayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upper(e) => write!(f, "overlay: upper layer error: {:?}", e),
            Self::Lower(e) => write!(f, "overlay: lower layer error: {:?}", e),
            Self::ReservedSuffix(s) => {
                write!(f, "overlay: name has reserved suffix: {}", s)
            }
            Self::InvalidPath => write!(f, "overlay: invalid path (must start with '/')"),
            Self::ReadOnly => write!(f, "overlay: writes not yet supported (Phase 1 read-only)"),
        }
    }
}

impl From<UpperError> for OverlayError {
    fn from(e: UpperError) -> Self {
        Self::Upper(e)
    }
}

impl From<LowerError> for OverlayError {
    fn from(e: LowerError) -> Self {
        Self::Lower(e)
    }
}

/// Strip a leading `/` and any trailing `/` from a path. Used by the
/// merge layer when normalizing operator-supplied paths into the
/// lookup-key form the upper/lower layers expect.
///
/// Returns `None` if the path is empty or doesn't start with `/`.
#[allow(dead_code)] // exercised by unit tests; reserved for the Phase 3 syscall layer
pub(crate) fn normalize_path(path: &str) -> Option<&str> {
    if !path.starts_with('/') {
        return None;
    }
    let trimmed = path.trim_start_matches('/').trim_end_matches('/');
    Some(trimmed)
}

/// Split a path into `(parent, name)` components. The parent is the
/// portion before the last `/`; the name is everything after. The
/// root path `""` returns `("", "")`.
#[allow(dead_code)] // exercised by unit tests; reserved for the Phase 3 syscall layer
pub(crate) fn split_parent_name(path: &str) -> (&str, &str) {
    match path.rfind('/') {
        Some(i) => (&path[..i], &path[i + 1..]),
        None => ("", path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_path_strips_leading_slash() {
        assert_eq!(normalize_path("/foo"), Some("foo"));
        assert_eq!(normalize_path("/foo/bar"), Some("foo/bar"));
        assert_eq!(normalize_path("/"), Some(""));
        assert_eq!(normalize_path("/foo/"), Some("foo"));
    }

    #[test]
    fn normalize_path_rejects_relative() {
        assert_eq!(normalize_path("foo"), None);
        assert_eq!(normalize_path(""), None);
    }

    #[test]
    fn split_parent_name_basic() {
        assert_eq!(split_parent_name("foo"), ("", "foo"));
        assert_eq!(split_parent_name("foo/bar"), ("foo", "bar"));
        assert_eq!(split_parent_name("a/b/c"), ("a/b", "c"));
        assert_eq!(split_parent_name(""), ("", ""));
    }

    #[test]
    fn overlay_error_displays() {
        use alloc::string::ToString;
        let e = OverlayError::ReservedSuffix(".whiteout".into());
        assert!(e.to_string().contains("reserved suffix"));
        let e = OverlayError::InvalidPath;
        assert!(e.to_string().contains("invalid path"));
        let e = OverlayError::ReadOnly;
        assert!(e.to_string().contains("read-only"));
    }
}
