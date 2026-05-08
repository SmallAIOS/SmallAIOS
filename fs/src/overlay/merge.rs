// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `readdir` merge logic.
//!
//! Returns the union of upper + lower entries minus:
//!
//! - Names whitelisted by an upper-layer whiteout.
//! - Lower-side entries shadowed by an opaque-dir marker on the upper.
//! - Names ending in any reserved suffix (`.whiteout`, `.opaque`,
//!   `.sha3`, `.sig`) — those are overlay-internal markers.
//!
//! When the same name appears on both layers, the upper's entry wins.
//! No duplicate names are ever returned.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::lower_layer::{LowerEntryKind, LowerError, LowerLayer};
use super::reserved::is_reserved_name;
use super::upper_layer::{UpperEntryKind, UpperError, UpperLayer};
use super::OverlayError;

/// Kind of an entry yielded by the merged readdir.
///
/// Whiteouts and opaque markers are NEVER yielded — they're filtered
/// out of the listing entirely. The merge surface is "what would the
/// user see at this point in the namespace".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergedEntryKind {
    RegularFile,
    Directory,
    /// Symlink or other inode kind from the lower (squashfs may
    /// surface these). The merge layer doesn't try to filter — it
    /// yields whatever the lower had, with the same upper-wins
    /// shadowing rules.
    Other,
}

/// One entry in the merged readdir output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergedEntry {
    pub name: String,
    pub kind: MergedEntryKind,
    pub size: u64,
    /// `true` if the entry came from the upper layer; `false` if from
    /// the lower. Useful for diagnostics (operator-facing CLI may
    /// surface this), not used by the kernel's POSIX path.
    pub from_upper: bool,
}

/// Merge `readdir(dir)` across upper + lower.
///
/// `dir` is slash-relative to the merge root, like the lookup API.
/// Empty `dir` lists the merge root.
pub fn merge_readdir<U: UpperLayer, L: LowerLayer>(
    upper: &U,
    lower: &L,
    dir: &str,
) -> Result<Vec<MergedEntry>, OverlayError> {
    // Step 1 — does the upper opaqueify this directory? If yes, only
    // upper entries are visible.
    let opaque = match upper.lookup(dir) {
        Ok(Some(e)) => e.kind == UpperEntryKind::Opaque,
        Ok(None) => false,
        Err(e) => return Err(OverlayError::Upper(e)),
    };

    // Step 2 — collect upper entries (if any).
    let upper_entries = match upper.iter_dir(dir) {
        Ok(v) => v,
        Err(UpperError::NotFound) | Err(UpperError::NotADirectory) => Vec::new(),
        Err(e) => return Err(OverlayError::Upper(e)),
    };

    // Step 3 — collect lower entries (if any). Suppressed entirely
    // when the upper marks the directory opaque.
    let lower_entries = if opaque {
        Vec::new()
    } else {
        match lower.iter_dir(dir) {
            Ok(v) => v,
            Err(LowerError::NotFound) | Err(LowerError::NotADirectory) => Vec::new(),
            Err(e) => return Err(OverlayError::Lower(e)),
        }
    };

    // Step 4 — first pass: gather whiteout names from the upper. These
    // hide the lower's entries with the same bare name.
    let mut whiteouts: alloc::collections::BTreeSet<String> = alloc::collections::BTreeSet::new();
    for e in &upper_entries {
        if e.kind == UpperEntryKind::Whiteout {
            whiteouts.insert(e.name.clone());
        }
    }

    // Step 5 — index upper non-marker entries by name (for shadowing).
    let mut merged: BTreeMap<String, MergedEntry> = BTreeMap::new();
    for e in upper_entries {
        // Markers (Whiteout, Opaque) are NEVER user-facing.
        if matches!(e.kind, UpperEntryKind::Whiteout | UpperEntryKind::Opaque) {
            continue;
        }
        // Reserved-suffix safeguard: should be unreachable since the
        // upper layer normally surfaces markers via Whiteout/Opaque
        // kinds, but operator-supplied files could still appear with
        // a reserved suffix if write enforcement is bypassed (Phase 2
        // closes that gap; Phase 1 defends in depth here).
        if is_reserved_name(&e.name) {
            continue;
        }
        merged.insert(
            e.name.clone(),
            MergedEntry {
                name: e.name,
                kind: match e.kind {
                    UpperEntryKind::RegularFile => MergedEntryKind::RegularFile,
                    UpperEntryKind::Directory => MergedEntryKind::Directory,
                    // Whiteout/Opaque already filtered above.
                    UpperEntryKind::Whiteout | UpperEntryKind::Opaque => unreachable!(),
                },
                size: e.size,
                from_upper: true,
            },
        );
    }

    // Step 6 — overlay lower entries: skip those shadowed by upper or
    // by an active whiteout, and skip reserved-suffix names.
    for e in lower_entries {
        if whiteouts.contains(&e.name) {
            continue;
        }
        if merged.contains_key(&e.name) {
            // Upper wins — already present.
            continue;
        }
        if is_reserved_name(&e.name) {
            // Don't expose a lower-side `<...>.whiteout` or
            // `<...>.sha3` even if the lower somehow has one.
            continue;
        }
        merged.insert(
            e.name.clone(),
            MergedEntry {
                name: e.name,
                kind: match e.kind {
                    LowerEntryKind::RegularFile => MergedEntryKind::RegularFile,
                    LowerEntryKind::Directory => MergedEntryKind::Directory,
                    LowerEntryKind::Other => MergedEntryKind::Other,
                },
                size: e.size,
                from_upper: false,
            },
        );
    }

    // Stable, sorted output. Operator-facing readdir is small (tens to
    // low hundreds of entries), so the BTreeMap → Vec collect is cheap
    // and predictable.
    Ok(merged.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::super::lower_layer::MockLowerLayer;
    use super::super::upper_layer::MockUpperLayer;
    use super::*;

    fn names(entries: &[MergedEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.name.as_str()).collect()
    }

    #[test]
    fn merge_union_no_duplicates() {
        let mut u = MockUpperLayer::new();
        u.add_file("a", b"u");
        u.add_file("b", b"u");
        let mut l = MockLowerLayer::new();
        l.add_file("a", b"l");
        l.add_file("c", b"l");
        let entries = merge_readdir(&u, &l, "/").unwrap();
        let n = names(&entries);
        // a, b, c — no duplicate of a.
        assert_eq!(n, alloc::vec!["a", "b", "c"]);
        // a came from upper.
        let a = entries.iter().find(|e| e.name == "a").unwrap();
        assert!(a.from_upper);
        // c came from lower.
        let c = entries.iter().find(|e| e.name == "c").unwrap();
        assert!(!c.from_upper);
    }

    #[test]
    fn merge_filters_reserved_suffixes_from_upper() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo.onnx", b"x");
        u.add_whiteout("bar");
        u.add_opaque("baz");
        let l = MockLowerLayer::new();
        let entries = merge_readdir(&u, &l, "/").unwrap();
        let n = names(&entries);
        // Only foo.onnx is user-facing; `bar.whiteout` and `baz/.opaque`
        // markers are filtered. `baz` itself (the opaque dir) is NOT
        // listed because the merge code path skips upper entries
        // marked Opaque (they're internal markers).
        assert_eq!(n, alloc::vec!["foo.onnx"]);
    }

    #[test]
    fn merge_filters_reserved_suffixes_from_lower() {
        // Pathological: lower has a file with reserved suffix. Filter
        // it out of the listing — defense in depth.
        let u = MockUpperLayer::new();
        let mut l = MockLowerLayer::new();
        l.add_file("good.onnx", b"x");
        l.add_file("bad.whiteout", b"x");
        l.add_file("bad.sha3", b"x");
        l.add_file("bad.sig", b"x");
        let entries = merge_readdir(&u, &l, "/").unwrap();
        let n = names(&entries);
        assert_eq!(n, alloc::vec!["good.onnx"]);
    }

    #[test]
    fn merge_whiteout_hides_lower_entry() {
        let mut u = MockUpperLayer::new();
        u.add_whiteout("a");
        let mut l = MockLowerLayer::new();
        l.add_file("a", b"x");
        l.add_file("b", b"x");
        let entries = merge_readdir(&u, &l, "/").unwrap();
        let n = names(&entries);
        assert_eq!(n, alloc::vec!["b"]);
    }

    #[test]
    fn merge_opaque_dir_hides_entire_lower_subtree() {
        let mut u = MockUpperLayer::new();
        u.add_opaque("custom");
        u.add_file("custom/upper-only", b"u");
        let mut l = MockLowerLayer::new();
        l.add_file("custom/lower-only", b"l");
        l.add_file("custom/shared", b"l");
        let entries = merge_readdir(&u, &l, "custom").unwrap();
        let n = names(&entries);
        // Only the upper entry is visible; lower-side entries are
        // entirely suppressed.
        assert_eq!(n, alloc::vec!["upper-only"]);
    }

    #[test]
    fn merge_only_lower_when_upper_dir_missing() {
        let u = MockUpperLayer::new();
        let mut l = MockLowerLayer::new();
        l.add_file("foo", b"x");
        let entries = merge_readdir(&u, &l, "/").unwrap();
        let n = names(&entries);
        assert_eq!(n, alloc::vec!["foo"]);
    }

    #[test]
    fn merge_empty_when_both_empty() {
        let u = MockUpperLayer::new();
        let l = MockLowerLayer::new();
        let entries = merge_readdir(&u, &l, "/").unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn merge_yields_sorted_stable_output() {
        let mut u = MockUpperLayer::new();
        u.add_file("z", b"x");
        u.add_file("m", b"x");
        let mut l = MockLowerLayer::new();
        l.add_file("a", b"x");
        l.add_file("c", b"x");
        let entries = merge_readdir(&u, &l, "/").unwrap();
        let n = names(&entries);
        assert_eq!(n, alloc::vec!["a", "c", "m", "z"]);
    }

    #[test]
    fn merge_directory_entries_classified_correctly() {
        let mut u = MockUpperLayer::new();
        u.add_dir("upperdir");
        u.add_file("upperfile", b"x");
        let mut l = MockLowerLayer::new();
        l.add_dir("lowerdir");
        l.add_file("lowerfile", b"x");
        let entries = merge_readdir(&u, &l, "/").unwrap();
        for e in &entries {
            match e.name.as_str() {
                "upperdir" | "lowerdir" => assert_eq!(e.kind, MergedEntryKind::Directory),
                "upperfile" | "lowerfile" => assert_eq!(e.kind, MergedEntryKind::RegularFile),
                _ => {}
            }
        }
    }
}
