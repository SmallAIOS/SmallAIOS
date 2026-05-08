// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration conformance tests for `embedded-overlay-v1` Phase 1.
//!
//! These tests assert the **read-only merge logic** end-to-end against
//! synthetic in-memory `MockUpperLayer` + `MockLowerLayer` fixtures.
//! Every requirement-level scenario from
//! `openspec/changes/embedded-overlay-v1/specs/fs-overlay-mount/spec.md`
//! has at least one direct test here; the broader test suite covers
//! corner cases and reserved-suffix filtering.
//!
//! Phase 1 does NOT exercise:
//!
//! - F2FS RW backing store (Phase 2 of the change).
//! - The `model_add` / `model_remove` syscalls (Phase 3).
//! - SHA-3-256 fingerprint sidecar verification (Phase 5).
//! - Capacity-cap enforcement or audit emission (Phase 6).

#![cfg(feature = "fs-overlay-mounts")]

extern crate alloc;
use alloc::string::ToString;

use smallaios_fs::overlay::{
    is_opaque_dir, is_reserved_name, is_whiteout, list_whiteouts_in, lookup, merge_readdir,
    reject_if_reserved, LookupResult, LowerEntry, LowerLayer, MergedEntryKind, MockLowerLayer,
    MockUpperLayer, OverlayError, UpperEntryKind, UpperLayer, RESERVED_SUFFIXES,
};

// ─── Spec scenario tests ───────────────────────────────────────────────────────

/// `fs-overlay-mount` Requirement: Upper-wins lookup precedence
/// Scenario: Upper file shadows lower file
#[test]
fn spec_upper_file_shadows_lower_file() {
    let mut u = MockUpperLayer::new();
    u.add_file("llama-3.onnx", b"upper-content");
    let mut l = MockLowerLayer::new();
    l.add_file("llama-3.onnx", b"lower-content");

    let result = lookup(&u, &l, "llama-3.onnx").unwrap();
    match result {
        LookupResult::Upper(e) => {
            assert_eq!(e.name, "llama-3.onnx");
            assert_eq!(e.size, 13);
            // And reading bytes returns the upper content.
            let mut buf = [0u8; 13];
            let n = u.read_file("llama-3.onnx", 0, &mut buf).unwrap();
            assert_eq!(&buf[..n], b"upper-content");
        }
        other => panic!("expected Upper, got {:?}", other),
    }
}

/// Scenario: Lower visible when upper absent
#[test]
fn spec_lower_visible_when_upper_absent() {
    let u = MockUpperLayer::new();
    let mut l = MockLowerLayer::new();
    l.add_file("whisper.onnx", b"lower-bytes");

    let result = lookup(&u, &l, "whisper.onnx").unwrap();
    match result {
        LookupResult::Lower(e) => {
            assert_eq!(e.name, "whisper.onnx");
            let mut buf = [0u8; 11];
            let n = l.read_file("whisper.onnx", 0, &mut buf).unwrap();
            assert_eq!(&buf[..n], b"lower-bytes");
        }
        other => panic!("expected Lower, got {:?}", other),
    }
}

/// Scenario: Whiteout hides lower
#[test]
fn spec_whiteout_hides_lower_returns_whiteoutresult() {
    let mut u = MockUpperLayer::new();
    u.add_whiteout("llama-3.onnx");
    let mut l = MockLowerLayer::new();
    l.add_file("llama-3.onnx", b"lower-content");

    assert_eq!(
        lookup(&u, &l, "llama-3.onnx").unwrap(),
        LookupResult::WhiteoutHidesLower
    );
}

/// Scenario: Whiteout hides lower — readdir does NOT list the name
#[test]
fn spec_whiteout_filters_lower_from_readdir() {
    let mut u = MockUpperLayer::new();
    u.add_whiteout("llama-3.onnx");
    let mut l = MockLowerLayer::new();
    l.add_file("llama-3.onnx", b"lower");
    l.add_file("whisper.onnx", b"keep-me");

    let entries = merge_readdir(&u, &l, "/").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert!(!names.contains(&"llama-3.onnx"));
    assert!(names.contains(&"whisper.onnx"));
}

/// Scenario: Opaque dir hides whole subtree
#[test]
fn spec_opaque_dir_hides_whole_subtree() {
    let mut u = MockUpperLayer::new();
    u.add_opaque("custom");
    u.add_file("custom/upper.onnx", b"u");
    let mut l = MockLowerLayer::new();
    l.add_file("custom/lower-a.onnx", b"l");
    l.add_file("custom/lower-b.onnx", b"l");

    let entries = merge_readdir(&u, &l, "custom").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    // Only the upper-side entry under custom/ is visible.
    assert_eq!(names, alloc::vec!["upper.onnx"]);
    assert!(!names.contains(&"lower-a.onnx"));
    assert!(!names.contains(&"lower-b.onnx"));
}

/// Scenario: Empty whiteout file enough to hide
#[test]
fn spec_empty_whiteout_hides() {
    let mut u = MockUpperLayer::new();
    // Default add_whiteout creates an empty marker.
    u.add_whiteout("foo.onnx");
    let mut l = MockLowerLayer::new();
    l.add_file("foo.onnx", b"lower");

    assert!(is_whiteout(&u, "", "foo.onnx"));
    match lookup(&u, &l, "foo.onnx").unwrap() {
        LookupResult::WhiteoutHidesLower => {}
        other => panic!("expected WhiteoutHidesLower, got {:?}", other),
    }
}

/// Scenario: Non-empty whiteout still functional
#[test]
fn spec_nonempty_whiteout_still_hides() {
    let mut u = MockUpperLayer::new();
    u.add_whiteout_with_content("foo.onnx", b"# Operator: removed by Bob 2026-04-01");
    let mut l = MockLowerLayer::new();
    l.add_file("foo.onnx", b"lower");

    assert!(is_whiteout(&u, "", "foo.onnx"));
    match lookup(&u, &l, "foo.onnx").unwrap() {
        LookupResult::WhiteoutHidesLower => {}
        other => panic!("expected WhiteoutHidesLower, got {:?}", other),
    }
}

/// Scenario: Reserved suffix on add (the merge layer's reject_if_reserved
/// is the building block; the syscall path lands in Phase 3).
#[test]
fn spec_reserved_suffix_on_add_rejected() {
    for suffix in RESERVED_SUFFIXES {
        let candidate = alloc::format!("foo{}", suffix);
        let result = reject_if_reserved(&candidate);
        match result {
            Err(OverlayError::ReservedSuffix(s)) => {
                assert_eq!(&s, suffix, "suffix mismatch for {}", suffix);
            }
            other => panic!("expected ReservedSuffix({}), got {:?}", suffix, other),
        }
    }
}

// ─── Comprehensive merge behaviour ─────────────────────────────────────────────

#[test]
fn merge_root_union_upper_wins() {
    let mut u = MockUpperLayer::new();
    u.add_file("a", b"upper-a");
    u.add_file("shared", b"upper-shared");
    let mut l = MockLowerLayer::new();
    l.add_file("b", b"lower-b");
    l.add_file("shared", b"lower-shared");

    let entries = merge_readdir(&u, &l, "/").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["a", "b", "shared"]);

    let shared = entries.iter().find(|e| e.name == "shared").unwrap();
    assert!(shared.from_upper, "upper-wins on merged readdir");
    assert_eq!(shared.size, 12); // upper-shared = 12 bytes
}

#[test]
fn merge_no_duplicate_names() {
    let mut u = MockUpperLayer::new();
    u.add_file("dup", b"u");
    let mut l = MockLowerLayer::new();
    l.add_file("dup", b"l");
    let entries = merge_readdir(&u, &l, "/").unwrap();
    let count = entries.iter().filter(|e| e.name == "dup").count();
    assert_eq!(count, 1);
}

#[test]
fn merge_yields_directories_correctly() {
    let mut u = MockUpperLayer::new();
    u.add_dir("upper-dir");
    let mut l = MockLowerLayer::new();
    l.add_dir("lower-dir");
    let entries = merge_readdir(&u, &l, "/").unwrap();
    for e in &entries {
        if e.name.ends_with("-dir") {
            assert_eq!(e.kind, MergedEntryKind::Directory);
        }
    }
}

#[test]
fn merge_reserved_suffix_files_filtered_from_listing() {
    // Defense-in-depth: even if a `<name>.whiteout` somehow got into
    // the lower layer (shouldn't happen — squashfs is operator-built —
    // but any squashfs the kernel ever sees), the merge layer hides it.
    let u = MockUpperLayer::new();
    let mut l = MockLowerLayer::new();
    l.add_file("good.onnx", b"x");
    for suffix in RESERVED_SUFFIXES {
        let name = alloc::format!("evil{}", suffix);
        l.add_file(&name, b"x");
    }
    let entries = merge_readdir(&u, &l, "/").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["good.onnx"]);
}

#[test]
fn merge_marks_provenance() {
    let mut u = MockUpperLayer::new();
    u.add_file("only-upper", b"u");
    u.add_file("shared", b"u");
    let mut l = MockLowerLayer::new();
    l.add_file("only-lower", b"l");
    l.add_file("shared", b"l");

    let entries = merge_readdir(&u, &l, "/").unwrap();
    for e in &entries {
        match e.name.as_str() {
            "only-upper" | "shared" => assert!(e.from_upper),
            "only-lower" => assert!(!e.from_upper),
            _ => panic!("unexpected entry {}", e.name),
        }
    }
}

#[test]
fn merge_handles_deep_nested_paths() {
    let u = MockUpperLayer::new();
    let mut l = MockLowerLayer::new();
    l.add_file("a/b/c/d/e/f/leaf", b"deep");
    let entries = merge_readdir(&u, &l, "a/b/c/d/e/f").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["leaf"]);
}

#[test]
fn merge_dir_only_in_upper() {
    let mut u = MockUpperLayer::new();
    u.add_file("upper-only/file", b"u");
    let l = MockLowerLayer::new();
    let entries = merge_readdir(&u, &l, "upper-only").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["file"]);
}

#[test]
fn merge_dir_only_in_lower() {
    let u = MockUpperLayer::new();
    let mut l = MockLowerLayer::new();
    l.add_file("lower-only/file", b"l");
    let entries = merge_readdir(&u, &l, "lower-only").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["file"]);
}

#[test]
fn merge_completely_empty_returns_empty_vec() {
    let u = MockUpperLayer::new();
    let l = MockLowerLayer::new();
    let entries = merge_readdir(&u, &l, "/").unwrap();
    assert!(entries.is_empty());
}

#[test]
fn merge_skipping_lower_when_upper_dir_is_opaque() {
    let mut u = MockUpperLayer::new();
    u.add_opaque("models");
    u.add_file("models/foo", b"u");
    let mut l = MockLowerLayer::new();
    l.add_file("models/bar", b"l");
    let entries = merge_readdir(&u, &l, "models").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["foo"]);
    assert!(!names.contains(&"bar"));
}

#[test]
fn merge_multiple_whiteouts_in_one_dir() {
    let mut u = MockUpperLayer::new();
    u.add_whiteout("a");
    u.add_whiteout("b");
    u.add_whiteout("c");
    u.add_file("d", b"upper-d");
    let mut l = MockLowerLayer::new();
    l.add_file("a", b"l");
    l.add_file("b", b"l");
    l.add_file("c", b"l");
    l.add_file("e", b"l");

    let entries = merge_readdir(&u, &l, "/").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["d", "e"]);
}

// ─── Lookup precedence corner cases ────────────────────────────────────────────

#[test]
fn lookup_neither_layer_has_path() {
    let u = MockUpperLayer::new();
    let l = MockLowerLayer::new();
    assert_eq!(lookup(&u, &l, "missing").unwrap(), LookupResult::NotFound);
}

#[test]
fn lookup_root_returns_upper_root() {
    // Looking up the empty path yields the upper-root directory.
    let u = MockUpperLayer::new();
    let l = MockLowerLayer::new();
    match lookup(&u, &l, "").unwrap() {
        LookupResult::Upper(e) => assert_eq!(e.kind, UpperEntryKind::Directory),
        other => panic!("expected Upper(Directory), got {:?}", other),
    }
}

#[test]
fn lookup_upper_directory_wins_over_lower_file() {
    let mut u = MockUpperLayer::new();
    u.add_dir("collision");
    let mut l = MockLowerLayer::new();
    l.add_file("collision", b"lower");
    match lookup(&u, &l, "collision").unwrap() {
        LookupResult::Upper(e) => assert_eq!(e.kind, UpperEntryKind::Directory),
        other => panic!("expected Upper(Directory), got {:?}", other),
    }
}

#[test]
fn lookup_upper_file_wins_over_lower_directory() {
    let mut u = MockUpperLayer::new();
    u.add_file("collision", b"upper");
    let mut l = MockLowerLayer::new();
    l.add_dir("collision");
    match lookup(&u, &l, "collision").unwrap() {
        LookupResult::Upper(e) => assert_eq!(e.kind, UpperEntryKind::RegularFile),
        other => panic!("expected Upper(RegularFile), got {:?}", other),
    }
}

#[test]
fn lookup_inside_opaque_dir_with_no_upper_entry_returns_opaque() {
    let mut u = MockUpperLayer::new();
    u.add_opaque("custom");
    let mut l = MockLowerLayer::new();
    l.add_file("custom/leaf", b"lower");
    assert_eq!(
        lookup(&u, &l, "custom/leaf").unwrap(),
        LookupResult::OpaqueHidesLowerSubtree
    );
}

#[test]
fn lookup_inside_opaque_dir_with_upper_entry_returns_upper() {
    let mut u = MockUpperLayer::new();
    u.add_opaque("custom");
    u.add_file("custom/leaf", b"upper");
    let mut l = MockLowerLayer::new();
    l.add_file("custom/leaf", b"lower");
    match lookup(&u, &l, "custom/leaf").unwrap() {
        LookupResult::Upper(e) => assert_eq!(e.size, 5),
        other => panic!("expected Upper, got {:?}", other),
    }
}

#[test]
fn lookup_outside_opaque_dir_unaffected() {
    let mut u = MockUpperLayer::new();
    u.add_opaque("custom");
    let mut l = MockLowerLayer::new();
    l.add_file("other.onnx", b"lower");
    match lookup(&u, &l, "other.onnx").unwrap() {
        LookupResult::Lower(_) => {}
        other => panic!("expected Lower, got {:?}", other),
    }
}

#[test]
fn lookup_whiteout_only_in_specific_dir() {
    let mut u = MockUpperLayer::new();
    u.add_whiteout("a/foo");
    let mut l = MockLowerLayer::new();
    l.add_file("a/foo", b"lower");
    l.add_file("b/foo", b"lower");
    // a/foo whiteout-hidden.
    assert_eq!(
        lookup(&u, &l, "a/foo").unwrap(),
        LookupResult::WhiteoutHidesLower
    );
    // b/foo not affected by the whiteout in a/.
    match lookup(&u, &l, "b/foo").unwrap() {
        LookupResult::Lower(_) => {}
        other => panic!("expected Lower for b/foo, got {:?}", other),
    }
}

#[test]
fn lookup_path_under_models_routes_through_merge_layer() {
    // Phase 1's merge layer doesn't itself enforce a `/models/`
    // prefix — the kernel VFS hands paths that have already been
    // stripped of the prefix. This test pins that contract: the
    // merge layer treats paths slash-relative to the merge root.
    let mut u = MockUpperLayer::new();
    u.add_file("foo", b"u");
    let l = MockLowerLayer::new();
    match lookup(&u, &l, "foo").unwrap() {
        LookupResult::Upper(_) => {}
        _ => panic!("path under merge root resolved to Upper"),
    }
}

#[test]
fn lookup_path_outside_merge_returns_notfound_in_layered_world() {
    // When the upper has nothing for a path and the lower has nothing
    // for that path, the layered world says NotFound — the kernel's
    // VFS surfaces -ENOENT.
    let u = MockUpperLayer::new();
    let l = MockLowerLayer::new();
    assert_eq!(
        lookup(&u, &l, "definitely/not/here").unwrap(),
        LookupResult::NotFound
    );
}

// ─── Whiteout helpers ──────────────────────────────────────────────────────────

#[test]
fn is_whiteout_at_root() {
    let mut u = MockUpperLayer::new();
    u.add_whiteout("foo");
    assert!(is_whiteout(&u, "", "foo"));
}

#[test]
fn is_whiteout_in_nested_dir() {
    let mut u = MockUpperLayer::new();
    u.add_whiteout("a/b/c");
    assert!(is_whiteout(&u, "a/b", "c"));
    assert!(!is_whiteout(&u, "a", "c"));
}

#[test]
fn is_whiteout_returns_false_for_real_file() {
    let mut u = MockUpperLayer::new();
    u.add_file("foo", b"x");
    assert!(!is_whiteout(&u, "", "foo"));
}

#[test]
fn list_whiteouts_in_empty_layer() {
    let u = MockUpperLayer::new();
    let wh = list_whiteouts_in(&u, "/");
    assert!(wh.is_empty());
}

#[test]
fn list_whiteouts_in_dir_with_files_and_whiteouts() {
    let mut u = MockUpperLayer::new();
    u.add_whiteout("hidden-1");
    u.add_whiteout("hidden-2");
    u.add_file("real-file", b"x");
    let mut wh = list_whiteouts_in(&u, "/");
    wh.sort();
    assert_eq!(wh.len(), 2);
    assert!(wh.contains(&"hidden-1".to_string()));
    assert!(wh.contains(&"hidden-2".to_string()));
}

// ─── Opaque-dir helpers ────────────────────────────────────────────────────────

#[test]
fn is_opaque_dir_at_subdir() {
    let mut u = MockUpperLayer::new();
    u.add_opaque("custom");
    assert!(is_opaque_dir(&u, "custom"));
}

#[test]
fn is_opaque_dir_returns_false_for_normal_dir() {
    let mut u = MockUpperLayer::new();
    u.add_dir("custom");
    assert!(!is_opaque_dir(&u, "custom"));
}

#[test]
fn is_opaque_dir_returns_false_for_file() {
    let mut u = MockUpperLayer::new();
    u.add_file("custom", b"data");
    assert!(!is_opaque_dir(&u, "custom"));
}

// ─── Reserved-suffix conformance ───────────────────────────────────────────────

#[test]
fn reserved_suffix_constant_matches_spec() {
    // Spec Q9: exactly four suffixes.
    assert_eq!(RESERVED_SUFFIXES.len(), 4);
    assert!(RESERVED_SUFFIXES.contains(&".whiteout"));
    assert!(RESERVED_SUFFIXES.contains(&".opaque"));
    assert!(RESERVED_SUFFIXES.contains(&".sha3"));
    assert!(RESERVED_SUFFIXES.contains(&".sig"));
}

#[test]
fn reserved_suffix_rejection_for_each() {
    let ok_names = [
        "model.onnx",
        "model",
        ".hidden",
        ".whiteoutish", // doesn't end in `.whiteout`
        "x.WHITEOUT",   // case-sensitive
    ];
    for n in &ok_names {
        assert!(reject_if_reserved(n).is_ok(), "expected ok: {}", n);
    }
    let bad_names = ["foo.whiteout", "foo.opaque", "foo.sha3", "foo.sig"];
    for n in &bad_names {
        assert!(reject_if_reserved(n).is_err(), "expected reject: {}", n);
    }
}

#[test]
fn reserved_name_predicate() {
    assert!(is_reserved_name("foo.whiteout"));
    assert!(is_reserved_name("foo.opaque"));
    assert!(is_reserved_name("foo.sha3"));
    assert!(is_reserved_name("foo.sig"));
    assert!(!is_reserved_name("foo.onnx"));
}

#[test]
fn reserved_suffix_with_double_extension() {
    // `foo.bin.sig` ends in `.sig` → reserved.
    assert!(is_reserved_name("foo.bin.sig"));
    assert!(reject_if_reserved("foo.bin.sig").is_err());
}

// ─── Listing-layer reserved-suffix filtering ───────────────────────────────────

#[test]
fn listing_filters_lower_side_sha3_and_sig() {
    let u = MockUpperLayer::new();
    let mut l = MockLowerLayer::new();
    l.add_file("model.onnx", b"x");
    l.add_file("model.onnx.sha3", b"x");
    l.add_file("model.onnx.sig", b"x");
    let entries = merge_readdir(&u, &l, "/").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["model.onnx"]);
}

#[test]
fn listing_filters_upper_side_internal_markers() {
    let mut u = MockUpperLayer::new();
    u.add_file("real-model.onnx", b"x");
    u.add_whiteout("hidden");
    u.add_opaque("opaque-dir");
    let l = MockLowerLayer::new();
    let entries = merge_readdir(&u, &l, "/").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["real-model.onnx"]);
}

// ─── Mock fixture round-trips ──────────────────────────────────────────────────

#[test]
fn mock_upper_round_trip_file_contents() {
    let mut u = MockUpperLayer::new();
    u.add_file("a/b/c.bin", b"\x00\x01\x02\x03\x04");
    let mut buf = [0u8; 5];
    let n = u.read_file("a/b/c.bin", 0, &mut buf).unwrap();
    assert_eq!(n, 5);
    assert_eq!(&buf, b"\x00\x01\x02\x03\x04");
}

#[test]
fn mock_upper_offset_read() {
    let mut u = MockUpperLayer::new();
    u.add_file("file", b"0123456789");
    let mut buf = [0u8; 4];
    let n = u.read_file("file", 4, &mut buf).unwrap();
    assert_eq!(n, 4);
    assert_eq!(&buf, b"4567");
}

#[test]
fn mock_upper_partial_read() {
    let mut u = MockUpperLayer::new();
    u.add_file("file", b"abc");
    let mut buf = [0u8; 100];
    let n = u.read_file("file", 0, &mut buf).unwrap();
    assert_eq!(n, 3);
}

#[test]
fn mock_lower_round_trip_file_contents() {
    let mut l = MockLowerLayer::new();
    l.add_file("foo", b"hello world");
    let mut buf = [0u8; 11];
    let n = l.read_file("foo", 0, &mut buf).unwrap();
    assert_eq!(n, 11);
    assert_eq!(&buf, b"hello world");
}

// ─── Property-style sweeps ─────────────────────────────────────────────────────

#[test]
fn sweep_every_quadrant_of_upper_lower_state() {
    // Quadrants: { upper-has, upper-absent } × { lower-has, lower-absent }
    // × { upper-whiteout, no-whiteout } × { upper-opaque-parent, not }
    //
    // Here we sweep the four core quadrants without the marker
    // axes — those are tested separately. This is the matrix of
    // upper_present × lower_present with no markers:
    //
    //   upper=Y, lower=Y → Upper wins
    //   upper=Y, lower=N → Upper wins
    //   upper=N, lower=Y → Lower fall-through
    //   upper=N, lower=N → NotFound
    type Predicate = fn(LookupResult<LowerEntry>) -> bool;
    let cases: &[(bool, bool, Predicate)] = &[
        (true, true, |r| matches!(r, LookupResult::Upper(_))),
        (true, false, |r| matches!(r, LookupResult::Upper(_))),
        (false, true, |r| matches!(r, LookupResult::Lower(_))),
        (false, false, |r| matches!(r, LookupResult::NotFound)),
    ];

    for &(upper_has, lower_has, predicate) in cases {
        let mut u = MockUpperLayer::new();
        let mut l = MockLowerLayer::new();
        if upper_has {
            u.add_file("path", b"u");
        }
        if lower_has {
            l.add_file("path", b"l");
        }
        let r = lookup(&u, &l, "path").unwrap();
        assert!(
            predicate(r.clone()),
            "quadrant (upper={}, lower={}) gave {:?}",
            upper_has,
            lower_has,
            r
        );
    }
}

#[test]
fn sweep_with_whiteout_axis() {
    // Whiteout always wins regardless of lower state.
    for lower_has in [false, true] {
        let mut u = MockUpperLayer::new();
        u.add_whiteout("path");
        let mut l = MockLowerLayer::new();
        if lower_has {
            l.add_file("path", b"l");
        }
        assert_eq!(
            lookup(&u, &l, "path").unwrap(),
            LookupResult::WhiteoutHidesLower,
            "whiteout should win even when lower_has={}",
            lower_has
        );
    }
}

#[test]
fn sweep_with_opaque_axis_path_inside_opaque_dir() {
    for lower_has in [false, true] {
        let mut u = MockUpperLayer::new();
        u.add_opaque("dir");
        let mut l = MockLowerLayer::new();
        if lower_has {
            l.add_file("dir/leaf", b"l");
        }
        assert_eq!(
            lookup(&u, &l, "dir/leaf").unwrap(),
            LookupResult::OpaqueHidesLowerSubtree,
            "opaque should hide lower regardless of lower_has={}",
            lower_has
        );
    }
}

#[test]
fn sweep_merge_readdir_listing_completeness() {
    // For a fixed scenario we assert the listing is a precisely
    // computable function of the inputs.
    let mut u = MockUpperLayer::new();
    u.add_file("u-only", b"u");
    u.add_file("shared", b"u");
    u.add_whiteout("hidden");
    let mut l = MockLowerLayer::new();
    l.add_file("l-only", b"l");
    l.add_file("shared", b"l");
    l.add_file("hidden", b"l");

    let entries = merge_readdir(&u, &l, "/").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    // Expected: u-only, shared (upper), l-only — but NOT hidden.
    assert_eq!(names, alloc::vec!["l-only", "shared", "u-only"]);
}

// ─── Sanity: many entries, deep paths ──────────────────────────────────────────

#[test]
fn merge_handles_many_entries() {
    let mut u = MockUpperLayer::new();
    let mut l = MockLowerLayer::new();
    for i in 0..50 {
        u.add_file(&alloc::format!("u{:02}", i), b"u");
        l.add_file(&alloc::format!("l{:02}", i), b"l");
    }
    let entries = merge_readdir(&u, &l, "/").unwrap();
    assert_eq!(entries.len(), 100);
}

#[test]
fn merge_root_listing_is_sorted() {
    let mut u = MockUpperLayer::new();
    u.add_file("zoo", b"u");
    u.add_file("alpha", b"u");
    let mut l = MockLowerLayer::new();
    l.add_file("middle", b"l");
    let entries = merge_readdir(&u, &l, "/").unwrap();
    let names: alloc::vec::Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, alloc::vec!["alpha", "middle", "zoo"]);
}

#[test]
fn lookup_into_empty_overlay_returns_notfound() {
    let u = MockUpperLayer::new();
    let l = MockLowerLayer::new();
    let result = lookup(&u, &l, "anything").unwrap();
    assert_eq!(result, LookupResult::NotFound);
}

#[test]
fn merge_yields_independent_clones_no_shared_state() {
    // After collecting once, the upper and lower fixtures should be
    // unchanged. (Smoke test on the by-value collection path.)
    let mut u = MockUpperLayer::new();
    u.add_file("a", b"u");
    let mut l = MockLowerLayer::new();
    l.add_file("b", b"l");

    let _ = merge_readdir(&u, &l, "/").unwrap();
    let _ = merge_readdir(&u, &l, "/").unwrap();
    let entries = merge_readdir(&u, &l, "/").unwrap();
    assert_eq!(entries.len(), 2);
}
