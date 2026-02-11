---- MODULE AuditHashChain ----
\* SmallAIOS Audit Log Hash Chain Integrity Model
\* Verifies: batch ordering, no gaps in sequence, no replay,
\*           hash chain continuity, genesis anchoring
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    MaxBatches,     \* Maximum number of batches to produce
    MaxEntries      \* Maximum entries per batch (256 in production)

VARIABLES
    chain,          \* Sequence of sealed batches
    nextSeq,        \* Next sequence number to assign
    prevHash,       \* Hash of the most recently sealed batch
    pendingEntries, \* Number of pending (unsealed) entries
    step            \* Step counter for bounded model checking

vars == <<chain, nextSeq, prevHash, pendingEntries, step>>

\* --- Constants ---

\* Genesis hash: symbolic constant representing 32 zero bytes
GenesisHash == 0

\* Hash function: abstract symbolic hash
\* In production this is SHA-3-256(previous_hash || serialized_entries)
\* Here we model it as a unique value derived from inputs.
\* Uses small coefficients to avoid integer overflow in TLC.
\* Injective for seq in 0..MaxBatches, entryCount in 1..MaxEntries:
\*   prev * (MaxEntries + 1) * (MaxBatches + 1) + seq * (MaxEntries + 1) + entryCount
\* This encoding is injective because each "digit" fits in its range.
Hash(prev, seq, entryCount) ==
    prev * ((MaxEntries + 1) * (MaxBatches + 1)) + seq * (MaxEntries + 1) + entryCount

\* A batch record
Batch == [seq : Nat, prev : Nat, hash : Nat, entryCount : 1..MaxEntries]

\* --- Type Invariant ---

TypeInvariant ==
    /\ nextSeq \in 0..MaxBatches
    /\ prevHash \in Nat
    /\ pendingEntries \in 0..MaxEntries
    /\ step \in Nat

\* --- Initial State ---

Init ==
    /\ chain = <<>>
    /\ nextSeq = 0
    /\ prevHash = GenesisHash
    /\ pendingEntries = 0
    /\ step = 0

\* --- Actions ---

\* An audit event arrives, incrementing pending count
AddEntry ==
    /\ pendingEntries < MaxEntries
    /\ pendingEntries' = pendingEntries + 1
    /\ step' = step + 1
    /\ UNCHANGED <<chain, nextSeq, prevHash>>

\* Seal a batch (count threshold or timeout)
SealBatch ==
    /\ pendingEntries > 0
    /\ Len(chain) < MaxBatches
    /\ LET newHash == Hash(prevHash, nextSeq, pendingEntries)
           newBatch == [seq |-> nextSeq, prev |-> prevHash,
                        hash |-> newHash, entryCount |-> pendingEntries]
       IN /\ chain' = Append(chain, newBatch)
          /\ nextSeq' = nextSeq + 1
          /\ prevHash' = newHash
          /\ pendingEntries' = 0
    /\ step' = step + 1

\* --- Safety Properties ---

\* 1. No gaps: sequence numbers are contiguous from 0
NoSequenceGaps ==
    \A i \in 1..Len(chain) :
        chain[i].seq = i - 1

\* 2. First batch anchored to genesis
GenesisAnchored ==
    Len(chain) > 0 => chain[1].prev = GenesisHash

\* 3. Hash chain continuity: each batch's prev equals the prior batch's hash
HashChainContinuity ==
    \A i \in 2..Len(chain) :
        chain[i].prev = chain[i-1].hash

\* 4. No replay: sequence numbers are strictly monotonically increasing
NoReplay ==
    \A i, j \in 1..Len(chain) :
        (i < j) => chain[i].seq < chain[j].seq

\* 5. No empty batches: every sealed batch has at least one entry
NoEmptyBatches ==
    \A i \in 1..Len(chain) :
        chain[i].entryCount > 0

\* 6. Hash uniqueness: all batch hashes are distinct
\* (follows from our abstract hash function being injective)
HashUniqueness ==
    \A i, j \in 1..Len(chain) :
        (i /= j) => chain[i].hash /= chain[j].hash

\* 7. Sequence number equals position (structural integrity)
SequenceMatchesPosition ==
    \A i \in 1..Len(chain) :
        chain[i].seq = i - 1

\* Combined invariant
SafetyInvariant ==
    /\ TypeInvariant
    /\ GenesisAnchored
    /\ NoSequenceGaps
    /\ HashChainContinuity
    /\ NoReplay
    /\ NoEmptyBatches
    /\ HashUniqueness
    /\ SequenceMatchesPosition

\* --- Termination ---

\* The model terminates when no more actions can be taken
\* (all batches sealed and entries full, or step bound reached)
Done ==
    /\ \/ step >= MaxBatches * (MaxEntries + 1)
       \/ (Len(chain) >= MaxBatches /\ pendingEntries >= MaxEntries)
    /\ UNCHANGED vars

\* --- Next State ---

Next ==
    \/ (/\ step < MaxBatches * (MaxEntries + 1)
        /\ \/ AddEntry
           \/ SealBatch)
    \/ Done

Spec == Init /\ [][Next]_vars

\* --- Liveness (with fairness) ---

FairSpec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* If entries keep arriving, batches eventually get sealed
EventualSeal ==
    (pendingEntries > 0) ~> (Len(chain) > 0)

====
