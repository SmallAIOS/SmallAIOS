// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Kani harness for the overlay merge-lookup precedence invariant.
//!
//! Spec reference: `embedded-overlay-v1` design Q14 / `fs-overlay-mount`
//! Requirement "Upper-wins lookup precedence".
//!
//! ## Proof obligation
//!
//! For every (upper_state, lower_state, name) triple where `upper_state`
//! ∈ { Absent, RegularFile, Directory, Whiteout, Opaque } and
//! `lower_state` ∈ { Absent, RegularFile, Directory }, the merge
//! lookup function `lookup(upper, lower, name)` SHALL satisfy:
//!
//! ```text
//!   upper_state == Whiteout    →  result == WhiteoutHidesLower
//!   upper_state == RegularFile →  result == Upper(RegularFile)
//!   upper_state == Directory   →  result == Upper(Directory)
//!   upper_state == Opaque      →  result == Upper(Opaque)
//!   upper_state == Absent      ∧ lower_state == Absent       →  result == NotFound
//!   upper_state == Absent      ∧ lower_state == RegularFile  →  result == Lower(RegularFile)
//!   upper_state == Absent      ∧ lower_state == Directory    →  result == Lower(Directory)
//! ```
//!
//! No other state is reachable; the property is total over the
//! enumerated input space.
//!
//! ## Status
//!
//! This harness is a **sketch**. The repository does not currently
//! pin a Kani toolchain version (see `formal/kani/TODO.md`). When
//! Kani lands as a CI tool, this harness should be moved to a `kani`
//! crate, gated by `#[cfg(kani)]`, and run via `cargo kani --harness
//! verify_overlay_merge_precedence`.
//!
//! Until then, the property is exercised by the
//! `sweep_every_quadrant_of_upper_lower_state` and
//! `sweep_with_whiteout_axis` integration tests in
//! `fs/tests/overlay_phase1_conformance.rs`. Those tests cover the
//! same enumerated state space concretely; Kani would replace the
//! enumeration with `kani::any()` choices and prove the property
//! across all reachable bit patterns.

#![cfg(kani)]

use smallaios_fs::overlay::{
    lookup, LookupResult, MockLowerLayer, MockUpperLayer, UpperEntryKind,
};

#[derive(Clone, Copy, kani::Arbitrary)]
enum UpperState {
    Absent,
    RegularFile,
    Directory,
    Whiteout,
    Opaque,
}

#[derive(Clone, Copy, kani::Arbitrary)]
enum LowerState {
    Absent,
    RegularFile,
    Directory,
}

/// Build a `MockUpperLayer` with `name` in the requested state.
fn build_upper(state: UpperState, name: &str) -> MockUpperLayer {
    let mut u = MockUpperLayer::new();
    match state {
        UpperState::Absent => {}
        UpperState::RegularFile => u.add_file(name, b"u"),
        UpperState::Directory => u.add_dir(name),
        UpperState::Whiteout => u.add_whiteout(name),
        UpperState::Opaque => u.add_opaque(name),
    }
    u
}

/// Build a `MockLowerLayer` with `name` in the requested state.
fn build_lower(state: LowerState, name: &str) -> MockLowerLayer {
    let mut l = MockLowerLayer::new();
    match state {
        LowerState::Absent => {}
        LowerState::RegularFile => l.add_file(name, b"l"),
        LowerState::Directory => l.add_dir(name),
    }
    l
}

/// Property: lookup is a total function over the (upper, lower) state
/// space and respects upper-wins precedence.
#[kani::proof]
fn verify_overlay_merge_precedence() {
    let upper_state: UpperState = kani::any();
    let lower_state: LowerState = kani::any();
    let name = "p";

    let u = build_upper(upper_state, name);
    let l = build_lower(lower_state, name);
    let r = lookup(&u, &l, name).expect("lookup is total");

    match (upper_state, lower_state) {
        (UpperState::Whiteout, _) => {
            assert!(matches!(r, LookupResult::WhiteoutHidesLower));
        }
        (UpperState::RegularFile, _) => match r {
            LookupResult::Upper(e) => assert_eq!(e.kind, UpperEntryKind::RegularFile),
            _ => kani::cover!(false, "expected Upper(RegularFile)"),
        },
        (UpperState::Directory, _) => match r {
            LookupResult::Upper(e) => assert_eq!(e.kind, UpperEntryKind::Directory),
            _ => kani::cover!(false, "expected Upper(Directory)"),
        },
        (UpperState::Opaque, _) => match r {
            LookupResult::Upper(e) => assert_eq!(e.kind, UpperEntryKind::Opaque),
            _ => kani::cover!(false, "expected Upper(Opaque)"),
        },
        (UpperState::Absent, LowerState::Absent) => {
            assert!(matches!(r, LookupResult::NotFound));
        }
        (UpperState::Absent, LowerState::RegularFile)
        | (UpperState::Absent, LowerState::Directory) => {
            assert!(matches!(r, LookupResult::Lower(_)));
        }
    }
}
