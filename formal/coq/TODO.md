# Coq proof TODOs — embedded-filesystem-v1 Phase 4

The Coq directory `formal/coq/` is new in Phase 4. The
[`bsdiff_applier.v`](bsdiff_applier.v) file states the proof
obligations for the bsdiff applier in Phase 4 of
`embedded-filesystem-v1`. The Coq mechanisation is not yet integrated
into CI — the obligations below are tracked as TODOs and re-checked
against the Rust unit tests in `fs/src/delta.rs`.

## Open obligations

1. **Determinism** — applier is a pure function.
   Rust evidence: `apply_delta_round_trip_*` tests in
   `fs/src/delta.rs`.

2. **Length correctness** — produced output has exactly
   `header.new_size` bytes.
   Rust evidence: `apply_delta` returns
   `Err(DeltaError::NewSizeMismatch)` if the control block does not
   account for `new_size` exactly.

3. **Bounds safety** — every read of `reference`, `diff`, `extra` is
   in-bounds.
   Rust evidence: explicit `checked_add` + bounds checks in
   `apply_delta` produce `OffsetOutOfRange` / `DiffTruncated` /
   `ExtraTruncated`.

4. **Inverse with bsdiff** — for any `(reference, target)` there
   exists a patch that produces `target` from `reference`.
   Rust evidence: the trivial witness used in `build_signed_patch`
   (test helper) demonstrates the existential.

5. **ML-DSA-65 unforgeability** — inherited from FIPS 204.
   Rust evidence: `verify_patch_rejects_forged_signature`,
   `verify_patch_rejects_wrong_pubkey`.

6. **SHA-3-256 collision resistance** — inherited from FIPS 202.
   Rust evidence: `verify_patch_rejects_wrong_reference`.

7. **Post-apply soundness** — combination of 4 + 5 + 6.
   Rust evidence: `post-apply mount` step in `apply_delta_to_partition`
   re-runs the squashfs manifest verifier.

## Path to mechanisation

- Wire up Rust→Coq import (likely via [`hax`](https://hacspec.org/) or
  CompCert-style extraction) — tracked under
  `architecture-documentation-v1`.
- Replace each `Axiom` with a derived theorem.
- Add a `formal/coq` CI job that runs `coqc bsdiff_applier.v` on
  pull requests touching `fs/src/delta.rs`.

The Coq stub is checked into the repo so the obligation set is
visible to auditors today; full mechanisation is a future change.
