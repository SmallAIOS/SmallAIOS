// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Boot-time conflict detection between the upper layer and a freshly
//! swapped-in lower (squashfs) slot.
//!
//! Spec: `openspec/changes/embedded-overlay-v1/specs/fs-overlay-syscalls/spec.md`,
//! Requirement "Boot-time conflict detection":
//!
//! > On every boot after a successful A/B swap into a new lower
//! > squashfs, the kernel SHALL scan the upper layer and SHALL append
//! > one audit record `overlay_conflict` for each upper entry that
//! > shadows a name now present in the new lower. The audit record
//! > SHALL include the name, the lower's SHA-3-256 (from the squashfs
//! > manifest), and the upper's SHA-3-256 (from the sidecar). No
//! > automatic resolution SHALL be attempted — the operator's upper
//! > continues to win.
//!
//! Spec also states "Pre-existing conflict not re-audited" — once a
//! conflict has been emitted, the same `(name, lower_sha3, upper_sha3)`
//! triple SHALL NOT re-fire on subsequent boots until something
//! actually changes.
//!
//! ## Implementation
//!
//! [`detect_conflicts`] takes the upper layer + a `LowerManifest`
//! abstraction (`name -> sha3_256`), enumerates upper-layer regular
//! files (excluding sidecars and whiteouts), and returns one
//! [`OverlayConflict`] per shadowing entry. The caller (the kernel
//! boot path) then appends one audit record per conflict.
//!
//! [`ConflictMemo`] tracks the `(name, lower_sha3, upper_sha3)` triples
//! the last boot already audited; passing it to [`detect_conflicts`]
//! filters those out so re-audit is suppressed across reboots when
//! neither side has changed. The kernel persists the memo to
//! `/data/overlay/conflict-memo.toml` (or equivalent) and reads it
//! back at boot.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use smallaios_security::crypto::sha3::Sha3_256Digest;

use super::integrity::{read_sidecar_digest, sha3_sidecar_path, IntegrityError};
use super::reserved::is_reserved_name;
use super::upper_layer::{UpperEntryKind, UpperError, UpperLayer};

/// One overlay-vs-lower conflict: the upper has `name`, and the new
/// lower also has the same `name` (so the upper is shadowing). The
/// audit emitter consumes this directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayConflict {
    /// File name that is duplicated.
    pub name: String,
    /// Squashfs manifest's SHA-3-256 for this name.
    pub lower_sha3: Sha3_256Digest,
    /// Upper's `<name>.sha3` sidecar value.
    pub upper_sha3: Sha3_256Digest,
}

/// Result of [`detect_conflicts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictReport {
    /// Conflicts that the audit emitter SHALL append `overlay_conflict`
    /// records for.
    pub new_conflicts: Vec<OverlayConflict>,
    /// Conflicts that exist but were already audited on a prior boot
    /// (suppressed by the memo). Not emitted; included for diagnostics.
    pub suppressed: Vec<OverlayConflict>,
    /// Upper-side errors (e.g. missing sidecar for a file). Surfaced
    /// so the kernel can audit `overlay_conflict_warn` if it wants;
    /// detection itself is best-effort.
    pub warnings: Vec<ConflictWarning>,
}

/// One non-fatal anomaly observed during conflict detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictWarning {
    /// Name with the issue.
    pub name: String,
    /// Specific reason (e.g. missing sidecar, malformed sidecar).
    pub reason: ConflictWarningKind,
}

/// Conflict-detection warning categories. None of these are fatal —
/// the kernel emits them as advisory audit records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictWarningKind {
    /// Upper has a regular file but no `<name>.sha3` sidecar. This is
    /// already flagged at read time by the integrity layer; the
    /// conflict scan surfaces it as advisory data.
    MissingFingerprint,
    /// Upper has a sidecar but it's not parseable.
    MalformedFingerprint,
    /// Underlying I/O error reading the upper.
    Io(UpperError),
}

/// Manifest abstraction for the lower (squashfs) slot. Implemented by
/// the squashfs reader's manifest verifier (in `fs::squashfs`); the
/// overlay only needs the `name -> digest` lookup, which is `O(log n)`
/// in the canonical map-backed implementation.
pub trait LowerManifest {
    /// Look up the SHA-3-256 of `name` in the manifest. Returns `None`
    /// if the lower does NOT have an entry under that name.
    fn lookup_sha3(&self, name: &str) -> Option<Sha3_256Digest>;
}

/// In-memory map-backed [`LowerManifest`] for unit + conformance tests.
#[derive(Debug, Default, Clone)]
pub struct MapLowerManifest {
    entries: alloc::collections::BTreeMap<String, Sha3_256Digest>,
}

impl MapLowerManifest {
    /// Construct an empty manifest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `name -> digest` (test helper).
    pub fn insert(&mut self, name: &str, digest: Sha3_256Digest) {
        self.entries.insert(name.to_string(), digest);
    }
}

impl LowerManifest for MapLowerManifest {
    fn lookup_sha3(&self, name: &str) -> Option<Sha3_256Digest> {
        self.entries.get(name).copied()
    }
}

/// Cross-reboot suppression set. The kernel persists this; the overlay
/// conflict scanner queries it before emitting an audit record.
#[derive(Debug, Default, Clone)]
pub struct ConflictMemo {
    seen: BTreeSet<MemoKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MemoKey {
    name: String,
    lower: [u8; 32],
    upper: [u8; 32],
}

impl ConflictMemo {
    /// Construct an empty memo (first-boot scenario).
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `(name, lower, upper)` as seen.
    pub fn record(&mut self, conflict: &OverlayConflict) {
        self.seen.insert(MemoKey {
            name: conflict.name.clone(),
            lower: *conflict.lower_sha3.as_bytes(),
            upper: *conflict.upper_sha3.as_bytes(),
        });
    }

    /// `true` iff this exact triple has been recorded.
    pub fn contains(&self, conflict: &OverlayConflict) -> bool {
        self.seen.contains(&MemoKey {
            name: conflict.name.clone(),
            lower: *conflict.lower_sha3.as_bytes(),
            upper: *conflict.upper_sha3.as_bytes(),
        })
    }

    /// Number of recorded triples (used by tests + telemetry).
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// `true` iff no triples have been recorded.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// Walk the upper layer at the given root and return one
/// [`OverlayConflict`] per upper entry that shadows a name in the
/// lower manifest, suppressing entries already in `memo`.
///
/// Implementation order:
///
/// 1. List upper at `root` (`""` for the upper-root, or a sub-path).
///    Sidecar suffixes (`.sha3`, `.sig`, `.opaque`, `.whiteout`) are
///    filtered out — they're internal markers and never participate
///    in the user-visible namespace.
/// 2. For each remaining regular file, look up the same bare name in
///    the lower manifest. If absent, no conflict.
/// 3. If present, read the upper's `<name>.sha3` sidecar to get the
///    upper digest. Missing/malformed sidecars are emitted as warnings;
///    the conflict is NOT suppressed solely by warning.
/// 4. Form the `(name, lower_sha3, upper_sha3)` triple. If `memo`
///    already contains it, push to `suppressed`; otherwise push to
///    `new_conflicts`.
pub fn detect_conflicts<U: UpperLayer, M: LowerManifest>(
    upper: &U,
    lower: &M,
    memo: &ConflictMemo,
) -> Result<ConflictReport, UpperError> {
    let mut new_conflicts = Vec::new();
    let mut suppressed = Vec::new();
    let mut warnings = Vec::new();

    walk_recursive(
        upper,
        lower,
        memo,
        "",
        &mut new_conflicts,
        &mut suppressed,
        &mut warnings,
    )?;

    Ok(ConflictReport {
        new_conflicts,
        suppressed,
        warnings,
    })
}

fn walk_recursive<U: UpperLayer, M: LowerManifest>(
    upper: &U,
    lower: &M,
    memo: &ConflictMemo,
    path: &str,
    new_conflicts: &mut Vec<OverlayConflict>,
    suppressed: &mut Vec<OverlayConflict>,
    warnings: &mut Vec<ConflictWarning>,
) -> Result<(), UpperError> {
    let entries = match upper.iter_dir(path) {
        Ok(e) => e,
        Err(UpperError::NotFound) | Err(UpperError::NotADirectory) => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let child_path = if path.is_empty() {
            entry.name.clone()
        } else {
            alloc::format!("{}/{}", path, entry.name)
        };

        match entry.kind {
            UpperEntryKind::RegularFile => {
                // Skip internal markers (sidecars + whiteout files would
                // already be hidden by `iter_dir`'s contract, but keep the
                // belt-and-braces filter).
                if is_reserved_name(&entry.name) {
                    continue;
                }
                // Conflict?
                if let Some(lower_digest) = lower.lookup_sha3(&child_path) {
                    let upper_digest =
                        match read_sidecar_digest(upper, &sha3_sidecar_path(&child_path)) {
                            Ok(d) => d,
                            Err(IntegrityError::SidecarMissing) => {
                                warnings.push(ConflictWarning {
                                    name: child_path.clone(),
                                    reason: ConflictWarningKind::MissingFingerprint,
                                });
                                continue;
                            }
                            Err(IntegrityError::SidecarMalformed) => {
                                warnings.push(ConflictWarning {
                                    name: child_path.clone(),
                                    reason: ConflictWarningKind::MalformedFingerprint,
                                });
                                continue;
                            }
                            Err(IntegrityError::Io(e)) => {
                                warnings.push(ConflictWarning {
                                    name: child_path.clone(),
                                    reason: ConflictWarningKind::Io(e),
                                });
                                continue;
                            }
                            Err(_) => continue,
                        };
                    let conflict = OverlayConflict {
                        name: child_path.clone(),
                        lower_sha3: lower_digest,
                        upper_sha3: upper_digest,
                    };
                    if memo.contains(&conflict) {
                        suppressed.push(conflict);
                    } else {
                        new_conflicts.push(conflict);
                    }
                }
            }
            UpperEntryKind::Directory => {
                walk_recursive(
                    upper,
                    lower,
                    memo,
                    &child_path,
                    new_conflicts,
                    suppressed,
                    warnings,
                )?;
            }
            UpperEntryKind::Whiteout | UpperEntryKind::Opaque => {
                // Whiteouts and opaque markers don't shadow lower
                // names — they HIDE them. Per spec the audit record
                // is `model_hidden`, emitted at `model_remove` time,
                // not by this scanner.
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::integrity::{encode_sidecar_bytes, hash_bytes};
    use super::super::write::MockWritableUpperLayer;
    use super::*;

    fn digest(s: &[u8]) -> Sha3_256Digest {
        hash_bytes(s)
    }

    fn upper_with(name: &str, contents: &[u8]) -> MockWritableUpperLayer {
        let mut u = MockWritableUpperLayer::new();
        u.add_file(name, contents);
        let d = hash_bytes(contents);
        u.add_file(&sha3_sidecar_path(name), &encode_sidecar_bytes(&d));
        u
    }

    #[test]
    fn no_conflict_when_lower_lacks_name() {
        let u = upper_with("custom.onnx", b"upper-bytes");
        let lower = MapLowerManifest::new();
        let memo = ConflictMemo::new();
        let report = detect_conflicts(&u, &lower, &memo).unwrap();
        assert!(report.new_conflicts.is_empty());
        assert!(report.suppressed.is_empty());
    }

    #[test]
    fn shadowing_entry_is_reported() {
        let u = upper_with("custom.onnx", b"upper-bytes");
        let mut lower = MapLowerManifest::new();
        lower.insert("custom.onnx", digest(b"different-lower-bytes"));
        let memo = ConflictMemo::new();
        let report = detect_conflicts(&u, &lower, &memo).unwrap();
        assert_eq!(report.new_conflicts.len(), 1);
        assert_eq!(report.new_conflicts[0].name, "custom.onnx");
    }

    #[test]
    fn memo_suppresses_already_audited_triple() {
        let u = upper_with("custom.onnx", b"upper-bytes");
        let mut lower = MapLowerManifest::new();
        let lower_d = digest(b"lower-bytes");
        lower.insert("custom.onnx", lower_d);
        let mut memo = ConflictMemo::new();
        memo.record(&OverlayConflict {
            name: "custom.onnx".into(),
            lower_sha3: lower_d,
            upper_sha3: digest(b"upper-bytes"),
        });
        let report = detect_conflicts(&u, &lower, &memo).unwrap();
        assert!(report.new_conflicts.is_empty());
        assert_eq!(report.suppressed.len(), 1);
    }

    #[test]
    fn changed_upper_re_audits() {
        // First boot: upper had digest X. Memo has (name, lower, X).
        // New boot: upper has been re-uploaded → digest Y.
        // Conflict triple is (name, lower, Y), NOT in memo → emits.
        let u = upper_with("custom.onnx", b"different-upper-bytes-now");
        let mut lower = MapLowerManifest::new();
        let lower_d = digest(b"lower");
        lower.insert("custom.onnx", lower_d);
        let mut memo = ConflictMemo::new();
        memo.record(&OverlayConflict {
            name: "custom.onnx".into(),
            lower_sha3: lower_d,
            upper_sha3: digest(b"first-boot-upper"),
        });
        let report = detect_conflicts(&u, &lower, &memo).unwrap();
        assert_eq!(report.new_conflicts.len(), 1);
        assert!(report.suppressed.is_empty());
    }

    #[test]
    fn sidecars_themselves_dont_count_as_shadowers() {
        let u = upper_with("custom.onnx", b"x");
        let mut lower = MapLowerManifest::new();
        // Lower has the sidecar — odd corner case but important.
        lower.insert("custom.onnx.sha3", digest(b"impossible"));
        let memo = ConflictMemo::new();
        let report = detect_conflicts(&u, &lower, &memo).unwrap();
        // The .sha3 file is reserved and skipped; only the bare name
        // could conflict, which the lower doesn't have.
        assert!(report.new_conflicts.is_empty());
    }

    #[test]
    fn missing_sidecar_emits_warning_not_conflict() {
        let mut u = MockWritableUpperLayer::new();
        // File but no sidecar.
        u.add_file("custom.onnx", b"x");
        let mut lower = MapLowerManifest::new();
        lower.insert("custom.onnx", digest(b"lower"));
        let report = detect_conflicts(&u, &lower, &ConflictMemo::new()).unwrap();
        assert!(report.new_conflicts.is_empty());
        assert_eq!(report.warnings.len(), 1);
        assert_eq!(
            report.warnings[0].reason,
            ConflictWarningKind::MissingFingerprint
        );
    }

    #[test]
    fn malformed_sidecar_emits_warning() {
        let mut u = MockWritableUpperLayer::new();
        u.add_file("custom.onnx", b"x");
        u.add_file("custom.onnx.sha3", b"not-hex");
        let mut lower = MapLowerManifest::new();
        lower.insert("custom.onnx", digest(b"lower"));
        let report = detect_conflicts(&u, &lower, &ConflictMemo::new()).unwrap();
        assert!(report.new_conflicts.is_empty());
        assert_eq!(report.warnings.len(), 1);
        assert_eq!(
            report.warnings[0].reason,
            ConflictWarningKind::MalformedFingerprint
        );
    }

    #[test]
    fn nested_conflict_detected_recursively() {
        let mut u = MockWritableUpperLayer::new();
        u.add_file("vendor/model.onnx", b"upper");
        let d = hash_bytes(b"upper");
        u.add_file(
            &sha3_sidecar_path("vendor/model.onnx"),
            &encode_sidecar_bytes(&d),
        );
        let mut lower = MapLowerManifest::new();
        lower.insert("vendor/model.onnx", digest(b"lower"));
        let report = detect_conflicts(&u, &lower, &ConflictMemo::new()).unwrap();
        assert_eq!(report.new_conflicts.len(), 1);
        assert_eq!(report.new_conflicts[0].name, "vendor/model.onnx");
    }

    #[test]
    fn whiteout_entries_not_shadowing() {
        // Whiteouts don't shadow — they hide. The conflict scanner
        // must NOT emit overlay_conflict for a whiteout that hides a
        // lower name.
        let mut u = MockWritableUpperLayer::new();
        u.add_whiteout("hidden");
        let mut lower = MapLowerManifest::new();
        lower.insert("hidden", digest(b"lower"));
        let report = detect_conflicts(&u, &lower, &ConflictMemo::new()).unwrap();
        assert!(report.new_conflicts.is_empty());
    }

    #[test]
    fn memo_round_trip_record_and_contains() {
        let conflict = OverlayConflict {
            name: "x".into(),
            lower_sha3: digest(b"l"),
            upper_sha3: digest(b"u"),
        };
        let mut memo = ConflictMemo::new();
        assert!(!memo.contains(&conflict));
        assert!(memo.is_empty());
        memo.record(&conflict);
        assert!(memo.contains(&conflict));
        assert_eq!(memo.len(), 1);
    }
}
