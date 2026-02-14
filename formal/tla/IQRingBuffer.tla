---- MODULE IQRingBuffer ----
\* IQ Sample Ring Buffer Overflow Protection Model
\* Verifies: reader never reads garbage, overflow counter correctness,
\*           indices stay in bounds, no partial block reads
\* Models lossy ring buffer for SDR IQ sample streaming to inference pipeline
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    BufSize         \* Number of block slots in the ring buffer (e.g., 4)

Slots == 0..(BufSize - 1)

\* Block validity states
INVALID == 0    \* Slot has never been written or was overwritten mid-read
VALID   == 1    \* Slot contains a complete, valid IQ sample block
WRITING == 2    \* Writer is actively filling this block

VARIABLES
    write_idx,      \* Writer (SDR RX callback) next write position
    read_idx,       \* Reader (inference pipeline) next read position
    block_state,    \* block_state[i] \in {INVALID, VALID, WRITING}
    block_seq,      \* block_seq[i] = sequence number of block written (0 = never)
    overflow_count, \* Number of times writer overwrote unread data
    total_written,  \* Total blocks successfully written
    total_read,     \* Total blocks successfully read
    reading,        \* TRUE if reader is currently reading a block
    step            \* Step counter for bounded checking

vars == <<write_idx, read_idx, block_state, block_seq,
          overflow_count, total_written, total_read, reading, step>>

MaxSteps == BufSize * 10

\* --- Type Invariant ---

TypeInvariant ==
    /\ write_idx \in Slots
    /\ read_idx \in Slots
    /\ \A i \in Slots : block_state[i] \in {INVALID, VALID, WRITING}
    /\ \A i \in Slots : block_seq[i] \in 0..MaxSteps
    /\ overflow_count \in 0..MaxSteps
    /\ total_written \in 0..MaxSteps
    /\ total_read \in 0..MaxSteps
    /\ reading \in BOOLEAN
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ write_idx = 0
    /\ read_idx = 0
    /\ block_state = [i \in Slots |-> INVALID]
    /\ block_seq = [i \in Slots |-> 0]
    /\ overflow_count = 0
    /\ total_written = 0
    /\ total_read = 0
    /\ reading = FALSE
    /\ step = 0

\* --- Helper ---

NextIdx(idx) == (idx + 1) % BufSize

\* Count valid blocks available for reading
ValidCount == Cardinality({i \in Slots : block_state[i] = VALID})

\* --- Actions ---

\* Writer begins writing an IQ sample block to current write slot
\* In lossy mode: if the slot is VALID (unread), this is an overflow
BeginWrite ==
    /\ step < MaxSteps
    /\ ~reading  \* Simplified: writer waits if reader is active on same slot
                 \* (in practice, different slots can be concurrent)
    /\ LET is_overflow == block_state[write_idx] = VALID
       IN /\ overflow_count' = IF is_overflow
                               THEN overflow_count + 1
                               ELSE overflow_count
          \* If overwriting unread data, advance read_idx past the overwritten slot
          /\ read_idx' = IF is_overflow /\ read_idx = write_idx
                         THEN NextIdx(read_idx)
                         ELSE read_idx
          /\ block_state' = [block_state EXCEPT ![write_idx] = WRITING]
    /\ step' = step + 1
    /\ UNCHANGED <<write_idx, block_seq, total_written, total_read, reading>>

\* Writer completes writing, block becomes valid
CompleteWrite ==
    /\ step < MaxSteps
    /\ block_state[write_idx] = WRITING
    /\ block_state' = [block_state EXCEPT ![write_idx] = VALID]
    /\ block_seq' = [block_seq EXCEPT ![write_idx] = total_written + 1]
    /\ write_idx' = NextIdx(write_idx)
    /\ total_written' = total_written + 1
    /\ step' = step + 1
    /\ UNCHANGED <<read_idx, overflow_count, total_read, reading>>

\* Reader begins reading a valid block
BeginRead ==
    /\ step < MaxSteps
    /\ ~reading
    /\ block_state[read_idx] = VALID
    /\ reading' = TRUE
    /\ step' = step + 1
    /\ UNCHANGED <<write_idx, read_idx, block_state, block_seq,
                   overflow_count, total_written, total_read>>

\* Reader completes reading, block becomes invalid (consumed)
CompleteRead ==
    /\ step < MaxSteps
    /\ reading = TRUE
    /\ block_state[read_idx] = VALID
    /\ block_state' = [block_state EXCEPT ![read_idx] = INVALID]
    /\ read_idx' = NextIdx(read_idx)
    /\ total_read' = total_read + 1
    /\ reading' = FALSE
    /\ step' = step + 1
    /\ UNCHANGED <<write_idx, block_seq, overflow_count, total_written>>

\* Writer aborts mid-write (error recovery): slot returns to INVALID
WriteAbort ==
    /\ step < MaxSteps
    /\ block_state[write_idx] = WRITING
    /\ block_state' = [block_state EXCEPT ![write_idx] = INVALID]
    /\ step' = step + 1
    /\ UNCHANGED <<write_idx, read_idx, block_seq, overflow_count,
                   total_written, total_read, reading>>

\* --- Next-State Relation ---

Next ==
    \/ BeginWrite
    \/ CompleteWrite
    \/ BeginRead
    \/ CompleteRead
    \/ WriteAbort

Fairness ==
    /\ WF_vars(CompleteWrite)
    /\ WF_vars(CompleteRead)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* Reader never reads garbage: can only read VALID blocks
ReaderNeverReadsGarbage ==
    reading => block_state[read_idx] \in {VALID}

\* Overflow counter increments correctly: overflow_count <= total_written
OverflowBounded ==
    overflow_count <= total_written

\* Buffer indices stay in bounds
IndicesInBounds ==
    /\ write_idx \in Slots
    /\ read_idx \in Slots

\* No partial block reads: reader only sees VALID (complete) blocks
\* A block in WRITING state is never read
NoPartialReads ==
    reading => block_state[read_idx] # WRITING

\* Read count never exceeds write count
ReadNeverExceedsWrite ==
    total_read <= total_written

\* All slots accounted for
SlotAccountingInvariant ==
    Cardinality({i \in Slots : block_state[i] = INVALID}) +
    Cardinality({i \in Slots : block_state[i] = VALID}) +
    Cardinality({i \in Slots : block_state[i] = WRITING}) = BufSize

\* At most one slot in WRITING state at any time
AtMostOneWriting ==
    Cardinality({i \in Slots : block_state[i] = WRITING}) <= 1

\* --- Liveness Properties ---

\* Data eventually flows through the buffer
EventualProgress ==
    (total_written > 0) ~> (total_read > 0)

====
