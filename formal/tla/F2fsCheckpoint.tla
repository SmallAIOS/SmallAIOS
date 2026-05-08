---------------------------- MODULE F2fsCheckpoint ----------------------------
(* Copyright 2026 SmallAIOS Contributors                                       *)
(* SPDX-License-Identifier: Apache-2.0                                         *)
(*                                                                             *)
(* F2FS checkpoint commit interleaving model — sketch.                         *)
(*                                                                             *)
(* Purpose: prove that for any interleaving of                                 *)
(*   - WriteCheckpointPack(pack, version, payload)                             *)
(*   - PowerLossBetweenWrites                                                  *)
(*   - Mount, which calls SelectCheckpoint(cp1, cp2)                           *)
(*                                                                             *)
(* the mount procedure NEVER returns a checkpoint pack with a CRC mismatch     *)
(* and ALWAYS returns the highest-version pack whose CRC validates.            *)
(*                                                                             *)
(* Phase 5 ships the *read* path of this contract: SelectCheckpoint() lives   *)
(* in `fs/src/f2fs/checkpoint.rs`. Phase 8's write path will extend this       *)
(* model with the dual checkpoint-pack writer + the 5-second idle commit      *)
(* timer, at which point this module's `WriteCheckpointPack` action gets a    *)
(* concrete refinement.                                                        *)
(*                                                                             *)
(* This file is a sketch (TLA+ syntax compiles via TLC's parser, but no       *)
(* full INVARIANT proof is checked yet — Phase 8 task 8.4 is the gating       *)
(* milestone). Promela-style pseudocode equivalents below the spec.            *)

EXTENDS Naturals, Sequences, TLC

CONSTANT MaxVersion        \* Maximum version counter we explore.
CONSTANT Slots             \* Set of pack slots; for F2FS this is {1, 2}.

VARIABLES
  packs,                   \* packs[s] = [version: Nat, crc_ok: Bool].
  power                    \* Power state ("on" | "off").

vars == <<packs, power>>

(* Initial state: both packs absent, power on. *)
Init ==
  /\ packs = [s \in Slots |-> [version |-> 0, crc_ok |-> FALSE]]
  /\ power = "on"

(* Action: write a fresh pack at slot `s` with monotonic version. The     *)
(* write is atomic in this model — Phase 8 will refine to a multi-step    *)
(* committee where the header and footer are written separately and a     *)
(* PowerLoss between them leaves the pack with crc_ok = FALSE.            *)
WriteCheckpointPack(s) ==
  /\ power = "on"
  /\ packs[s].version < MaxVersion
  /\ packs' = [packs EXCEPT ![s] = [version |-> packs[s].version + 1,
                                    crc_ok  |-> TRUE]]
  /\ UNCHANGED <<power>>

(* Action: simulate a power loss between two writes — leaves the in-      *)
(* progress pack with crc_ok = FALSE.                                     *)
PowerLossBetweenWrites(s) ==
  /\ power = "on"
  /\ packs[s].crc_ok = TRUE
  /\ packs' = [packs EXCEPT ![s] = [version |-> packs[s].version,
                                    crc_ok  |-> FALSE]]
  /\ power' = "off"

(* Power-on transition. *)
PowerOn ==
  /\ power = "off"
  /\ power' = "on"
  /\ UNCHANGED <<packs>>

(* Selection function: pick the slot with the highest version whose CRC   *)
(* validates. If no slot is valid, return -1 (mount fails). This refines  *)
(* `select_checkpoint()` in fs/src/f2fs/checkpoint.rs.                    *)
SelectCheckpoint ==
  LET valid == {s \in Slots : packs[s].crc_ok}
      vs    == {packs[s].version : s \in valid}
  IN  IF valid = {} THEN -1
                    ELSE CHOOSE s \in valid : packs[s].version = Max(vs)

(* Helper. *)
Max(S) == CHOOSE x \in S : \A y \in S : y <= x

(* Next step relation — non-deterministic interleaving. *)
Next ==
  \/ \E s \in Slots : WriteCheckpointPack(s)
  \/ \E s \in Slots : PowerLossBetweenWrites(s)
  \/ PowerOn

Spec == Init /\ [][Next]_vars

(* Invariants we want to check (Phase 8 task 8.4 enables full TLC run).  *)

(* Safety: SelectCheckpoint never picks a CRC-invalid pack. *)
InvSelectionCrcValid ==
  power = "on" =>
    LET sel == SelectCheckpoint
    IN  sel = -1 \/ packs[sel].crc_ok = TRUE

(* Liveness: if at least one pack has a valid CRC, mount succeeds. *)
InvMountAvailable ==
  (\E s \in Slots : packs[s].crc_ok) =>
    SelectCheckpoint /= -1

(* Safety: SelectCheckpoint always returns the highest-version valid     *)
(* pack (used by the read path in mount.rs).                             *)
InvHighestVersionWins ==
  LET sel == SelectCheckpoint
  IN  sel = -1
       \/ \A s \in Slots :
            packs[s].crc_ok => packs[s].version <= packs[sel].version

================================================================================
(*                                                                             *)
(* Promela-style pseudocode equivalent (when TLC is unavailable):              *)
(*                                                                             *)
(*   bit pack_crc_ok[2];                                                       *)
(*   byte pack_version[2];                                                     *)
(*   byte power = ON;                                                          *)
(*                                                                             *)
(*   inline write_pack(s) {                                                    *)
(*     atomic { pack_version[s] = pack_version[s] + 1;                         *)
(*              pack_crc_ok[s] = 1; }                                          *)
(*   }                                                                         *)
(*                                                                             *)
(*   inline power_loss(s) {                                                    *)
(*     atomic { pack_crc_ok[s] = 0; power = OFF; }                             *)
(*   }                                                                         *)
(*                                                                             *)
(*   inline select_checkpoint() {                                              *)
(*     if :: pack_crc_ok[0] && pack_crc_ok[1] ->                               *)
(*           if pack_version[0] >= pack_version[1] -> chosen = 0;              *)
(*                                                else -> chosen = 1; fi      *)
(*        :: pack_crc_ok[0] && !pack_crc_ok[1] -> chosen = 0;                  *)
(*        :: !pack_crc_ok[0] && pack_crc_ok[1] -> chosen = 1;                  *)
(*        :: else -> chosen = -1;                                              *)
(*     fi                                                                      *)
(*   }                                                                         *)
(*                                                                             *)
(*   /* Invariants */                                                          *)
(*   ltl crc_safe   { [] (chosen != -1 -> pack_crc_ok[chosen]) }               *)
(*   ltl available  { [] ((pack_crc_ok[0] || pack_crc_ok[1]) -> chosen != -1) }*)
(*                                                                             *)
(* TODO(phase 8): replace `WriteCheckpointPack` with a two-step write that     *)
(*                models the "header → payload → footer" sequence the kernel   *)
(*                actually performs, and prove that a power loss between       *)
(*                steps cannot leave both packs invalid simultaneously when    *)
(*                the previous pack was valid.                                 *)
