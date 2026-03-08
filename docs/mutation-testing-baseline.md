# Mutation Testing Baseline

Generated with `cargo-mutants` on 2026-03-07.

## Summary

| Module | Caught | Missed | Unviable | Total | Score |
|--------|--------|--------|----------|-------|-------|
| `kernel/src/mem/buddy.rs` | 15 | 7 | 0 | 22 | 68% |
| `security/src/crypto/constant_time.rs` | 53 | 12 | 6 | 71 | 82% |
| **Combined** | **68** | **19** | **6** | **93** | **78%** |

## Surviving Mutants

### `kernel/src/mem/buddy.rs` (7 missed)

| Line | Mutation | Analysis |
|------|----------|----------|
| 32:36 | replace `*` with `+` | Bitmap size calculation — not exercised by current tests |
| 32:36 | replace `*` with `/` | Bitmap size calculation — not exercised by current tests |
| 32:29 | replace `*` with `+` | Bitmap size calculation — not exercised by current tests |
| 32:29 | replace `*` with `/` | Bitmap size calculation — not exercised by current tests |
| 35:30 | replace `*` with `+` | Page count calculation — not exercised by current tests |
| 35:30 | replace `*` with `/` | Page count calculation — not exercised by current tests |
| 75:33 | replace `<<` with `>>` in `OrderBitmap::set_free` | Bit-shift direction — needs targeted test |

### `security/src/crypto/constant_time.rs` (12 missed)

| Line | Mutation | Analysis |
|------|----------|----------|
| 40:9 | replace `Display::fmt` with `Ok(Default::default())` | Error formatting — cosmetic, low risk |
| 126:44 | replace `>>` with `<<` in `ct_eq_byte` | Shift direction in constant-time equality — needs test for non-zero inputs |
| 202:16 | replace `\|` with `^` in `ct_select_byte` | Bitwise OR vs XOR — needs targeted edge-case test |
| 220:27 | replace `\|\|` with `&&` in `ct_select` | Boolean logic change — needs test where only one condition is true |
| 225:32 | replace `\|` with `^` in `ct_select` | Bitwise OR vs XOR — needs targeted test |
| 653-698 | replace Kani proof bodies with `()` (6 mutations) | Expected — Kani proofs are not exercised by `cargo test` |

## Follow-Up Test Tasks

1. **Buddy allocator**: Add tests that verify bitmap/page-count calculations with specific pool sizes to catch arithmetic operator mutations on lines 32 and 35.
2. **Buddy allocator**: Add test for `OrderBitmap::set_free` that verifies correct bit position is set (catches `<<` vs `>>` on line 75).
3. **Constant-time ops**: Add tests for `ct_eq_byte` with varied non-zero inputs to catch shift-direction mutations.
4. **Constant-time ops**: Add edge-case tests for `ct_select` / `ct_select_byte` where OR vs XOR would produce different results.
5. **Kani proof mutations**: Expected survivors — Kani proofs run under `cfg(kani)`, not `cargo test`. No action needed.
