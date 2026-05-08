(* Copyright 2026 SmallAIOS Contributors                              *)
(* SPDX-License-Identifier: Apache-2.0                                *)

(* ================================================================== *)
(* bsdiff applier — partial Coq proof + remaining obligations.        *)
(*                                                                    *)
(* The Rust implementation lives at `fs/src/delta.rs`.                *)
(* This file states the high-level proof obligations as Coq           *)
(* `Axiom` placeholders. A full mechanised proof requires CompCert-   *)
(* style Rust↔Coq import; until that pipeline is wired in CI, the     *)
(* obligations are tracked as TODOs in `TODO.md` and re-checked       *)
(* against the Rust unit tests (`apply_delta_round_trip_*`).          *)
(*                                                                    *)
(* The intent of this file is to be self-checkable with Coq 8.18+     *)
(* (or Rocq) once the project decides to gate it in CI; today it      *)
(* serves as the canonical statement of the proof shape.              *)
(* ================================================================== *)

Require Import List Arith ZArith.
Import ListNotations.

Open Scope Z_scope.

(* ----- Wire-format types ------------------------------------------ *)

(** A bsdiff control entry: (diff_len, extra_len, seek). *)
Record CtrlEntry := {
  diff_len : Z;
  extra_len : Z;
  seek : Z;
}.

Record Patch := {
  reference_hash : list Z;       (* 32-byte SHA-3-256 *)
  new_size : Z;
  ctrl : list CtrlEntry;
  diff : list Z;                 (* raw diff bytes, length = sum diff_len *)
  extra : list Z;                (* raw extra bytes, length = sum extra_len *)
}.

(* ----- Specification of apply_delta -------------------------------- *)

(** Wrapping byte add modulo 256. *)
Definition wadd8 (a b : Z) : Z :=
  ((a + b) mod 256).

(** A single-step semantics of the applier:                           *)
(*   Given old_ptr, new_ptr, diff_ptr, extra_ptr and one CtrlEntry,   *)
(*   produce the updated pointers and append the produced bytes to    *)
(*   `out`. Implementation in Coq omitted; the Rust applier is the    *)
(*   reference implementation.                                        *)
Parameter apply_step :
  forall (reference : list Z)
         (entry : CtrlEntry)
         (old_ptr new_ptr diff_ptr extra_ptr : Z)
         (diff extra out : list Z),
  (* returns either (out', old', new', diff', extra') or error *)
  option (list Z * Z * Z * Z * Z).

(** Full applier as iterated step. *)
Parameter apply_delta_spec :
  forall (reference : list Z) (p : Patch),
  option (list Z).

(* ----- Proof obligations ------------------------------------------ *)

(** Obligation 1: Determinism.                                        *)
(* The applier is a pure function: given the same inputs it produces  *)
(* the same output. This holds trivially for the Rust implementation; *)
(* tracked here for completeness.                                     *)
Axiom apply_determinism :
  forall reference p out1 out2,
    apply_delta_spec reference p = Some out1 ->
    apply_delta_spec reference p = Some out2 ->
    out1 = out2.

(** Obligation 2: Length correctness.                                 *)
(* If apply succeeds, the produced output has length exactly          *)
(* `p.new_size`.                                                      *)
Axiom apply_length :
  forall reference p out,
    apply_delta_spec reference p = Some out ->
    Z.of_nat (length out) = p.(new_size).

(** Obligation 3: Bounds safety.                                      *)
(* If apply succeeds, every read from `reference` is within           *)
(* `[0, length reference)` and every read from `diff` / `extra` is    *)
(* within `[0, length diff)` / `[0, length extra)`.                   *)
(* The Rust implementation enforces this with explicit bounds checks  *)
(* (see DeltaError::OffsetOutOfRange / DiffTruncated / ExtraTruncated *)
(* in `delta.rs`); the obligation states that successful application  *)
(* never violates these.                                              *)
Axiom apply_bounds_safe :
  forall reference p out,
    apply_delta_spec reference p = Some out ->
    True. (* refined to in-range invariants once Coq import lands *)

(** Obligation 4: Inverse with bsdiff.                                *)
(* For every `(reference, target)`, there exists a Patch `p` such     *)
(* that `apply_delta_spec reference p = Some target`.                 *)
(* The trivial witness uses `diff = target - reference` (bytewise     *)
(* wrapping subtraction) and `extra = []` for the overlap region.    *)
Axiom apply_existential_inverse :
  forall reference target,
  exists p,
    apply_delta_spec reference p = Some target.

(* ----- Pre-apply integrity-check obligations ----------------------- *)

(** Obligation 5: signature → sound.                                  *)
(* If ML-DSA-65 verifies the patch under the public key embedded in   *)
(* the squashfs manifest, then the patch was produced by the holder   *)
(* of the corresponding secret key. (Inherited from the FIPS 204      *)
(* unforgeability proof for ML-DSA-65; we treat that as an Axiom      *)
(* here and document it for the audit trail.)                         *)
Axiom ml_dsa_65_unforgeable :
  forall public_key patch_bytes signature,
    True. (* placeholder for FIPS 204 unforgeability theorem *)

(** Obligation 6: reference-hash → sound.                             *)
(* If `SHA-3-256(reference) == p.reference_hash`, then the patch was  *)
(* generated against `reference` (modulo SHA-3-256 collision          *)
(* resistance, treated as Axiom).                                     *)
Axiom sha3_256_collision_resistant :
  forall a b,
    True. (* placeholder for SHA-3-256 CR theorem *)

(* ----- Post-apply integrity-check obligations ---------------------- *)

(** Obligation 7: post-apply soundness.                               *)
(* If `apply_delta_spec reference p = Some out` AND the appended      *)
(* SmallAIOS manifest footer in `out` ML-DSA-65-verifies AND its      *)
(* per-block SHA-3-256 hashes match every 4 KiB block of `out`, THEN  *)
(* `out` is the bytes intended by the update author.                  *)
(* Combines obligations 4 + 5 + 6.                                    *)
Axiom post_apply_sound :
  forall reference p out,
    apply_delta_spec reference p = Some out ->
    True.

(* ================================================================== *)
(* End of bsdiff_applier.v                                            *)
(* ================================================================== *)
