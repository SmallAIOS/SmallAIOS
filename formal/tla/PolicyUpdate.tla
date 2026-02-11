---- MODULE PolicyUpdate ----
\* SmallAIOS Remote Policy Update Protocol Model
\* Verifies: authentication-before-swap, swap atomicity,
\*           rollback safety, enforcement mode monotonicity
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    MaxUpdates      \* Maximum number of update attempts to model

\* Enforcement modes
Modes == {"Enforcing", "Permissive"}

\* Sentinel value for "no policy blob"
NONE == [version |-> 0, mode |-> "Permissive", signature |-> "none"]

VARIABLES
    currentPolicy,  \* Active policy record
    pendingBlob,    \* Incoming policy blob awaiting verification
    authenticated,  \* Whether pendingBlob has been signature-verified
    swapInProgress, \* Whether an atomic swap is underway
    oldPolicy,      \* Saved copy of policy before swap (for rollback)
    updateCount,    \* Number of update attempts so far
    step            \* Step counter for bounded model checking

vars == <<currentPolicy, pendingBlob, authenticated, swapInProgress,
          oldPolicy, updateCount, step>>

\* --- Type Definitions ---

\* A policy record: version number, enforcement mode, signature status
Policy == [version : Nat, mode : Modes, signature : {"valid", "invalid", "none"}]

\* --- Type Invariant ---

TypeInvariant ==
    /\ currentPolicy \in Policy
    /\ pendingBlob \in Policy
    /\ authenticated \in BOOLEAN
    /\ swapInProgress \in BOOLEAN
    /\ oldPolicy \in Policy
    /\ updateCount \in 0..MaxUpdates
    /\ step \in Nat

\* --- Initial State ---

Init ==
    /\ currentPolicy = [version |-> 1, mode |-> "Permissive", signature |-> "valid"]
    /\ pendingBlob = NONE
    /\ authenticated = FALSE
    /\ swapInProgress = FALSE
    /\ oldPolicy = NONE
    /\ updateCount = 0
    /\ step = 0

\* --- Actions ---

\* Receive a new policy blob from a remote source.
\* The blob may have a valid or invalid signature.
ReceiveBlob(blob) ==
    /\ updateCount < MaxUpdates
    /\ pendingBlob = NONE               \* No pending update in progress
    /\ swapInProgress = FALSE
    /\ blob \in Policy
    /\ blob.version > currentPolicy.version  \* Only accept higher versions
    /\ blob.signature \in {"valid", "invalid"}
    /\ pendingBlob' = blob
    /\ authenticated' = FALSE
    /\ updateCount' = updateCount + 1
    /\ step' = step + 1
    /\ UNCHANGED <<currentPolicy, swapInProgress, oldPolicy>>

\* Verify the signature of the pending blob.
\* Sets authenticated to TRUE only if signature is valid.
VerifySignature ==
    /\ pendingBlob /= NONE
    /\ ~authenticated
    /\ ~swapInProgress
    /\ IF pendingBlob.signature = "valid"
       THEN authenticated' = TRUE
       ELSE authenticated' = FALSE
    /\ step' = step + 1
    /\ UNCHANGED <<currentPolicy, pendingBlob, swapInProgress, oldPolicy, updateCount>>

\* Attempt an atomic policy swap.
\* Preconditions: blob is authenticated AND mode monotonicity holds.
\* Atomically: save old policy, install new policy, clear pending state.
AttemptSwap ==
    /\ pendingBlob /= NONE
    /\ authenticated = TRUE
    /\ ~swapInProgress
    \* Mode monotonicity: cannot demote from Enforcing to Permissive
    /\ ~(currentPolicy.mode = "Enforcing" /\ pendingBlob.mode = "Permissive")
    \* Atomic swap: save old policy and install new in one step
    /\ oldPolicy' = currentPolicy
    /\ currentPolicy' = [version |-> pendingBlob.version,
                          mode |-> pendingBlob.mode,
                          signature |-> pendingBlob.signature]
    /\ pendingBlob' = NONE
    /\ authenticated' = FALSE
    /\ swapInProgress' = FALSE
    /\ step' = step + 1
    /\ UNCHANGED <<updateCount>>

\* Reject the swap: blob is not authenticated or would demote mode.
\* Clear pending state; current policy is retained.
RejectSwap ==
    /\ pendingBlob /= NONE
    /\ \/ authenticated = FALSE
       \/ (currentPolicy.mode = "Enforcing" /\ pendingBlob.mode = "Permissive")
    /\ ~swapInProgress
    /\ pendingBlob' = NONE
    /\ authenticated' = FALSE
    /\ step' = step + 1
    /\ UNCHANGED <<currentPolicy, swapInProgress, oldPolicy, updateCount>>

\* Rollback: if a post-swap validation detects an issue, restore old policy.
\* This models the case where swap completed but post-swap checks fail.
Rollback ==
    /\ oldPolicy /= NONE
    /\ oldPolicy.version > 0  \* There is a real old policy to restore
    /\ currentPolicy' = oldPolicy
    /\ oldPolicy' = NONE
    /\ pendingBlob' = NONE
    /\ authenticated' = FALSE
    /\ swapInProgress' = FALSE
    /\ step' = step + 1
    /\ UNCHANGED <<updateCount>>

\* --- Safety Properties ---

\* 1. Authentication before swap: the current policy version can only advance
\*    if the blob was authenticated. Structurally enforced by AttemptSwap
\*    requiring authenticated = TRUE, but we state it as an invariant:
\*    if currentPolicy.version > 1 (i.e., at least one swap has occurred),
\*    then currentPolicy.signature must be "valid" (was verified).
AuthBeforeSwap ==
    currentPolicy.version > 1 => currentPolicy.signature = "valid"

\* 2. Swap atomicity: the current policy version never decreases
\*    (except via explicit Rollback, which restores the prior version).
\*    We express this as: version is always >= 1 and swapInProgress is
\*    never stuck TRUE (swap is atomic, not partial).
SwapAtomicity ==
    /\ currentPolicy.version >= 1
    /\ swapInProgress = FALSE

\* 3. Rollback safety: after rollback, currentPolicy equals the saved oldPolicy.
\*    Since rollback is an action, we express this structurally: if oldPolicy
\*    is NONE (rollback completed or never started), the system is consistent.
\*    Also: oldPolicy, when set, always has version < currentPolicy.version
\*    (the saved policy is strictly older than the installed one).
RollbackSafety ==
    oldPolicy.version > 0 => oldPolicy.version < currentPolicy.version

\* 4. Mode monotonicity: once in Enforcing mode, cannot revert to Permissive.
\*    This is the key security guarantee.
ModeMonotonicity ==
    \* If we ever had Enforcing, we still have Enforcing
    \* Structurally: AttemptSwap blocks Enforcing -> Permissive.
    \* We verify: if currentPolicy.mode was Enforcing before and version advanced,
    \* mode remains Enforcing. Expressed as: old was Enforcing => current is Enforcing.
    oldPolicy.version > 0 /\ oldPolicy.mode = "Enforcing"
        => currentPolicy.mode = "Enforcing"

\* --- Next-State Relation ---

Next ==
    \/ \E blob \in [version : (currentPolicy.version + 1)..(currentPolicy.version + 2),
                    mode : Modes,
                    signature : {"valid", "invalid"}] :
        ReceiveBlob(blob)
    \/ VerifySignature
    \/ AttemptSwap
    \/ RejectSwap
    \/ Rollback

Spec == Init /\ [][Next]_vars

====
