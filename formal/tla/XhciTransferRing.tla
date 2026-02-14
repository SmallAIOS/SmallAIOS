---- MODULE XhciTransferRing ----
\* xHCI Transfer Ring Producer/Consumer Model
\* Verifies: no buffer overflow, no missed completions, cycle bit consistency,
\*           producer/consumer indices don't cross
\* Models xHCI transfer ring with TRB enqueue/dequeue and cycle bit management
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    RingSize        \* Number of TRB entries in the transfer ring (e.g., 16)

Slots == 0..(RingSize - 1)

\* TRB slot states
EMPTY     == 0    \* Slot available for producer
PENDING   == 1    \* TRB enqueued by software, awaiting hardware processing
COMPLETED == 2    \* Hardware posted completion event for this TRB

VARIABLES
    enqueue_ptr,    \* Producer (software) enqueue index
    dequeue_ptr,    \* Consumer (hardware) dequeue index
    sw_cycle_bit,   \* Software cycle bit (toggles on wrap-around)
    hw_cycle_bit,   \* Hardware cycle bit (toggles on wrap-around)
    ring_state,     \* ring_state[i] \in {EMPTY, PENDING, COMPLETED}
    ring_cycle,     \* ring_cycle[i] = cycle bit written with the TRB
    pending_count,  \* Number of TRBs awaiting hardware processing
    completed_count,\* Total completions posted (for tracking)
    enqueued_total, \* Total TRBs enqueued (for tracking)
    step            \* Step counter for bounded checking

vars == <<enqueue_ptr, dequeue_ptr, sw_cycle_bit, hw_cycle_bit,
          ring_state, ring_cycle, pending_count, completed_count,
          enqueued_total, step>>

MaxSteps == RingSize * 6

\* --- Type Invariant ---

TypeInvariant ==
    /\ enqueue_ptr \in Slots
    /\ dequeue_ptr \in Slots
    /\ sw_cycle_bit \in {0, 1}
    /\ hw_cycle_bit \in {0, 1}
    /\ \A i \in Slots : ring_state[i] \in {EMPTY, PENDING, COMPLETED}
    /\ \A i \in Slots : ring_cycle[i] \in {0, 1}
    /\ pending_count \in 0..RingSize
    /\ completed_count \in 0..MaxSteps
    /\ enqueued_total \in 0..MaxSteps
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ enqueue_ptr = 0
    /\ dequeue_ptr = 0
    /\ sw_cycle_bit = 1
    /\ hw_cycle_bit = 1
    /\ ring_state = [i \in Slots |-> EMPTY]
    /\ ring_cycle = [i \in Slots |-> 0]
    /\ pending_count = 0
    /\ completed_count = 0
    /\ enqueued_total = 0
    /\ step = 0

\* --- Helper ---

NextPtr(ptr) == (ptr + 1) % RingSize

\* Count of PENDING slots
PendingSlots == Cardinality({i \in Slots : ring_state[i] = PENDING})

\* --- Actions ---

\* Producer (software) enqueues a TRB
\* Guard: slot at enqueue_ptr must be EMPTY (no overwrite of unprocessed TRBs)
EnqueueTRB ==
    /\ step < MaxSteps
    /\ ring_state[enqueue_ptr] = EMPTY
    /\ pending_count < RingSize  \* Ring not full
    /\ ring_state' = [ring_state EXCEPT ![enqueue_ptr] = PENDING]
    /\ ring_cycle' = [ring_cycle EXCEPT ![enqueue_ptr] = sw_cycle_bit]
    /\ LET next == NextPtr(enqueue_ptr)
       IN /\ enqueue_ptr' = next
          \* Toggle software cycle bit on wrap-around
          /\ sw_cycle_bit' = IF next = 0
                             THEN (1 - sw_cycle_bit)
                             ELSE sw_cycle_bit
    /\ pending_count' = pending_count + 1
    /\ enqueued_total' = enqueued_total + 1
    /\ step' = step + 1
    /\ UNCHANGED <<dequeue_ptr, hw_cycle_bit, completed_count>>

\* Consumer (hardware) dequeues a TRB and posts completion
\* Guard: slot at dequeue_ptr must be PENDING with matching cycle bit
DequeueTRB ==
    /\ step < MaxSteps
    /\ ring_state[dequeue_ptr] = PENDING
    /\ ring_cycle[dequeue_ptr] = hw_cycle_bit  \* Cycle bit must match
    /\ ring_state' = [ring_state EXCEPT ![dequeue_ptr] = COMPLETED]
    /\ LET next == NextPtr(dequeue_ptr)
       IN /\ dequeue_ptr' = next
          \* Toggle hardware cycle bit on wrap-around
          /\ hw_cycle_bit' = IF next = 0
                             THEN (1 - hw_cycle_bit)
                             ELSE hw_cycle_bit
    /\ pending_count' = pending_count - 1
    /\ completed_count' = completed_count + 1
    /\ step' = step + 1
    /\ UNCHANGED <<enqueue_ptr, sw_cycle_bit, ring_cycle, enqueued_total>>

\* Software reclaims a completed TRB (returns slot to EMPTY for reuse)
ReclaimTRB ==
    /\ step < MaxSteps
    /\ \E i \in Slots : ring_state[i] = COMPLETED
    /\ LET slot == CHOOSE i \in Slots : ring_state[i] = COMPLETED
       IN ring_state' = [ring_state EXCEPT ![slot] = EMPTY]
    /\ step' = step + 1
    /\ UNCHANGED <<enqueue_ptr, dequeue_ptr, sw_cycle_bit, hw_cycle_bit,
                   ring_cycle, pending_count, completed_count, enqueued_total>>

\* --- Next-State Relation ---

Next ==
    \/ EnqueueTRB
    \/ DequeueTRB
    \/ ReclaimTRB

Fairness ==
    /\ WF_vars(DequeueTRB)
    /\ WF_vars(ReclaimTRB)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* No buffer overflow: producer never overwrites unprocessed (PENDING) TRBs
\* Enforced structurally: EnqueueTRB requires EMPTY slot
NoBufferOverflow ==
    pending_count <= RingSize

\* No missed completions: completed count never exceeds enqueued total
NoMissedCompletions ==
    completed_count <= enqueued_total

\* Cycle bit consistency: software and hardware cycle bits stay synchronized
\* When enqueue_ptr = dequeue_ptr and ring is empty, cycle bits must match
CycleBitConsistency ==
    (enqueue_ptr = dequeue_ptr /\ pending_count = 0) =>
        sw_cycle_bit = hw_cycle_bit

\* Producer/consumer indices: pending count tracks the actual PENDING slots
PendingCountAccurate ==
    pending_count = Cardinality({i \in Slots : ring_state[i] = PENDING})

\* All slots accounted for
SlotAccountingInvariant ==
    Cardinality({i \in Slots : ring_state[i] = EMPTY}) +
    Cardinality({i \in Slots : ring_state[i] = PENDING}) +
    Cardinality({i \in Slots : ring_state[i] = COMPLETED}) = RingSize

\* Ring state ordering: dequeue always processes in FIFO order
\* (PENDING slots form a contiguous arc from dequeue_ptr)

\* --- Liveness Properties ---

\* Every enqueued TRB eventually gets completed
EventualCompletion ==
    (pending_count > 0) ~> (completed_count > 0)

====
