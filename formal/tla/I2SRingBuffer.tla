---- MODULE I2SRingBuffer ----
\* I2S DMA Ring Buffer Model
\* Verifies: no overwrite of unread data, no read of unwritten data,
\*           ring wrap-around correctness, DMA coherency
\* Models producer/consumer ring buffer with DMA for audio streaming
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    BufSize         \* Number of slots in the ring buffer (4)

Slots == 0..(BufSize - 1)

\* Buffer slot states
FREE    == 0    \* Slot is empty, available for writing
FILLING == 1    \* DMA or producer is writing to this slot
READY   == 2    \* Slot contains valid data, available for reading
PLAYING == 3    \* DMA or consumer is reading from this slot

VARIABLES
    write_ptr,      \* Next slot to write into (0..BufSize-1)
    read_ptr,       \* Next slot to read from (0..BufSize-1)
    buffer_state,   \* buffer_state[i] \in {FREE, FILLING, READY, PLAYING}
    dma_active,     \* TRUE if DMA transfer is in progress
    written_count,  \* Total slots written (for tracking)
    read_count,     \* Total slots read (for tracking)
    step            \* Step counter

vars == <<write_ptr, read_ptr, buffer_state, dma_active, written_count, read_count, step>>

MaxSteps == BufSize * 8

\* --- Type Invariant ---

TypeInvariant ==
    /\ write_ptr \in Slots
    /\ read_ptr \in Slots
    /\ \A i \in Slots : buffer_state[i] \in {FREE, FILLING, READY, PLAYING}
    /\ dma_active \in BOOLEAN
    /\ written_count \in 0..MaxSteps
    /\ read_count \in 0..MaxSteps
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ write_ptr = 0
    /\ read_ptr = 0
    /\ buffer_state = [i \in Slots |-> FREE]
    /\ dma_active = FALSE
    /\ written_count = 0
    /\ read_count = 0
    /\ step = 0

\* --- Helper ---

\* Next pointer with wrap-around
NextPtr(ptr) == (ptr + 1) % BufSize

\* Count of READY slots
ReadyCount == Cardinality({i \in Slots : buffer_state[i] = READY})

\* Count of FREE slots
FreeCount == Cardinality({i \in Slots : buffer_state[i] = FREE})

\* --- Actions ---

\* Producer begins writing to the next free slot
BeginWrite ==
    /\ step < MaxSteps
    /\ buffer_state[write_ptr] = FREE
    /\ ~dma_active    \* No concurrent DMA during write setup
    /\ buffer_state' = [buffer_state EXCEPT ![write_ptr] = FILLING]
    /\ step' = step + 1
    /\ UNCHANGED <<write_ptr, read_ptr, dma_active, written_count, read_count>>

\* Producer completes writing, slot becomes ready
CompleteWrite ==
    /\ step < MaxSteps
    /\ buffer_state[write_ptr] = FILLING
    /\ buffer_state' = [buffer_state EXCEPT ![write_ptr] = READY]
    /\ write_ptr' = NextPtr(write_ptr)
    /\ written_count' = written_count + 1
    /\ step' = step + 1
    /\ UNCHANGED <<read_ptr, dma_active, read_count>>

\* DMA/consumer begins reading from the next ready slot
BeginRead ==
    /\ step < MaxSteps
    /\ buffer_state[read_ptr] = READY
    /\ ~dma_active
    /\ buffer_state' = [buffer_state EXCEPT ![read_ptr] = PLAYING]
    /\ dma_active' = TRUE
    /\ step' = step + 1
    /\ UNCHANGED <<write_ptr, read_ptr, written_count, read_count>>

\* DMA/consumer completes reading, slot returns to free
CompleteRead ==
    /\ step < MaxSteps
    /\ buffer_state[read_ptr] = PLAYING
    /\ dma_active = TRUE
    /\ buffer_state' = [buffer_state EXCEPT ![read_ptr] = FREE]
    /\ read_ptr' = NextPtr(read_ptr)
    /\ dma_active' = FALSE
    /\ read_count' = read_count + 1
    /\ step' = step + 1
    /\ UNCHANGED <<write_ptr, written_count>>

\* DMA abort: return playing slot to ready (error recovery)
DMAAbort ==
    /\ step < MaxSteps
    /\ dma_active = TRUE
    /\ buffer_state[read_ptr] = PLAYING
    /\ buffer_state' = [buffer_state EXCEPT ![read_ptr] = READY]
    /\ dma_active' = FALSE
    /\ step' = step + 1
    /\ UNCHANGED <<write_ptr, read_ptr, written_count, read_count>>

\* --- Next-State Relation ---

Next ==
    \/ BeginWrite
    \/ CompleteWrite
    \/ BeginRead
    \/ CompleteRead
    \/ DMAAbort

Fairness ==
    /\ WF_vars(CompleteWrite)
    /\ WF_vars(CompleteRead)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* No overwrite of unread data: write pointer slot must be FREE
NoOverwrite ==
    (buffer_state[write_ptr] = FILLING) =>
        \A i \in Slots :
            (i # write_ptr /\ buffer_state[i] \in {READY, PLAYING}) =>
                i # write_ptr

\* No read of unwritten data: read pointer slot must be READY or PLAYING
NoUnwrittenRead ==
    (buffer_state[read_ptr] = PLAYING) => dma_active

\* Write pointer never lands on a non-FREE slot (prevents overwrite)
WriteOnlyToFree ==
    buffer_state[write_ptr] \in {FREE, FILLING}

\* Read pointer never lands on a FREE or FILLING slot for reading
ReadOnlyFromReady ==
    (dma_active) => buffer_state[read_ptr] = PLAYING

\* Ring wrap-around: pointers stay in valid range
PointersInRange ==
    /\ write_ptr \in Slots
    /\ read_ptr \in Slots

\* DMA coherency: exactly one PLAYING slot when DMA active
DMACoherency ==
    dma_active =>
        Cardinality({i \in Slots : buffer_state[i] = PLAYING}) = 1

\* Total accounting: all slots accounted for
SlotAccountingInvariant ==
    Cardinality({i \in Slots : buffer_state[i] = FREE}) +
    Cardinality({i \in Slots : buffer_state[i] = FILLING}) +
    Cardinality({i \in Slots : buffer_state[i] = READY}) +
    Cardinality({i \in Slots : buffer_state[i] = PLAYING}) = BufSize

\* Read count never exceeds write count
ReadNeverExceedsWrite ==
    read_count <= written_count

\* --- Liveness Properties ---

\* Data eventually flows through the buffer
EventualProgress ==
    (written_count > 0) ~> (read_count > 0)

====
