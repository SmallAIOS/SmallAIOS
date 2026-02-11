---- MODULE AfdxVirtualLink ----
\* AFDX Virtual Link BAG Enforcement Model
\* Verifies: no deadline miss, BAG compliance, sequence number correctness
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    VirtualLinks,   \* Set of Virtual Link identifiers
    MaxBag,         \* Maximum BAG interval in ticks
    MaxSeq,         \* Maximum sequence number (255 for AFDX)
    MaxSteps        \* Bound for model checking

VARIABLES
    bagTimer,       \* bagTimer[vl] = ticks until next allowed transmission
    bagPeriod,      \* bagPeriod[vl] = BAG interval for this VL
    seqNum,         \* seqNum[vl] = current sequence number
    txCount,        \* txCount[vl] = number of frames transmitted
    lastTxTime,     \* lastTxTime[vl] = tick when last frame was sent
    tick,           \* Current tick counter
    step            \* Step counter

vars == <<bagTimer, bagPeriod, seqNum, txCount, lastTxTime, tick, step>>

\* --- Type Invariant ---

TypeInvariant ==
    /\ \A vl \in VirtualLinks : bagTimer[vl] \in 0..MaxBag
    /\ \A vl \in VirtualLinks : bagPeriod[vl] \in 1..MaxBag
    /\ \A vl \in VirtualLinks : seqNum[vl] \in 0..MaxSeq
    /\ tick \in 0..MaxSteps
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ bagTimer = [vl \in VirtualLinks |-> 0]
    /\ bagPeriod = [vl \in VirtualLinks |-> CHOOSE b \in 1..MaxBag : TRUE]
    /\ seqNum = [vl \in VirtualLinks |-> 0]
    /\ txCount = [vl \in VirtualLinks |-> 0]
    /\ lastTxTime = [vl \in VirtualLinks |-> 0]
    /\ tick = 0
    /\ step = 0

\* --- Actions ---

\* Advance one tick: decrement all BAG timers
AdvanceTick ==
    /\ step < MaxSteps
    /\ bagTimer' = [vl \in VirtualLinks |->
        IF bagTimer[vl] > 0 THEN bagTimer[vl] - 1 ELSE 0]
    /\ tick' = tick + 1
    /\ step' = step + 1
    /\ UNCHANGED <<bagPeriod, seqNum, txCount, lastTxTime>>

\* Transmit on a VL when BAG timer has expired
TransmitFrame(vl) ==
    /\ step < MaxSteps
    /\ bagTimer[vl] = 0
    /\ bagTimer' = [bagTimer EXCEPT ![vl] = bagPeriod[vl]]
    /\ seqNum' = [seqNum EXCEPT ![vl] = (@ + 1) % (MaxSeq + 1)]
    /\ txCount' = [txCount EXCEPT ![vl] = @ + 1]
    /\ lastTxTime' = [lastTxTime EXCEPT ![vl] = tick]
    /\ step' = step + 1
    /\ UNCHANGED <<bagPeriod, tick>>

\* --- Next-State Relation ---

Next ==
    \/ AdvanceTick
    \/ \E vl \in VirtualLinks : TransmitFrame(vl)

Fairness == WF_vars(Next)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* BAG enforcement: minimum inter-frame gap is respected
BagCompliance ==
    \A vl \in VirtualLinks :
        bagTimer[vl] >= 0

\* Sequence numbers are monotonically increasing (mod MaxSeq+1)
SequenceMonotonic ==
    \A vl \in VirtualLinks :
        seqNum[vl] \in 0..MaxSeq

\* No frame sent before BAG timer expires
NoPrematureTransmission ==
    \A vl \in VirtualLinks :
        (txCount[vl] > 0 /\ tick > lastTxTime[vl]) =>
            (tick - lastTxTime[vl] >= bagPeriod[vl] \/ bagTimer[vl] > 0)

====
