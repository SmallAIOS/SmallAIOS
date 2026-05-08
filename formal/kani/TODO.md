# Kani harness TODO

This directory holds Kani-formatted proof sketches that the repository
does not yet wire into CI. When Kani is added to the toolchain pin and
a `cargo kani` task lands in `justfile`, the harnesses below should be
promoted from sketches to verified properties.

## Pending harnesses

### `overlay_merge.rs` — overlay-v1 Phase 1

**Property:** `lookup(upper, lower, path)` is a total function over
the enumerated `(UpperState, LowerState)` space and satisfies
upper-wins precedence per `embedded-overlay-v1` design Q14 /
`fs-overlay-mount` Requirement "Upper-wins lookup precedence".

**Proof obligation table:**

| `upper_state` | `lower_state` | Expected `lookup` result    |
|---------------|---------------|------------------------------|
| `Whiteout`    | any           | `WhiteoutHidesLower`         |
| `RegularFile` | any           | `Upper(RegularFile)`         |
| `Directory`   | any           | `Upper(Directory)`           |
| `Opaque`      | any           | `Upper(Opaque)`              |
| `Absent`      | `Absent`      | `NotFound`                   |
| `Absent`      | `RegularFile` | `Lower(RegularFile)`         |
| `Absent`      | `Directory`   | `Lower(Directory)`           |

**Today's coverage:** the
`sweep_every_quadrant_of_upper_lower_state` and
`sweep_with_whiteout_axis` integration tests in
`fs/tests/overlay_phase1_conformance.rs` walk the same enumeration
concretely. When the Kani harness ships, those tests stay (Kani is
a complement to, not a replacement for, the conformance suite).

## Wiring checklist (when Kani lands)

1. Add `kani` to `rust-toolchain.toml` extra components, or pin it
   in `.github/workflows/kani.yml` separately.
2. Add a `kani` crate at `formal/kani/Cargo.toml` listing all
   harnesses as `[[bin]]`s, with a `[features] kani` flag.
3. Add `just kani` recipe wrapping `cargo kani --workspace` with
   appropriate `--harness` filters.
4. Wire `cargo kani --harness verify_overlay_merge_precedence` into
   the formal-gate CI job.
