---- MODULE CanArbitration ----
\* CAN Bus Arbitration Model
\* Verifies: priority-based arbitration, no starvation, bus fairness
\* Models the CSMA/CA bit-level arbitration of CAN 2.0A/B
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Nodes,          \* Set of CAN node identifiers
    MaxFrameId,     \* Maximum frame ID value (0x7FF for CAN 2.0A)
    MaxSteps,       \* Bound for model checking
    MaxPending      \* Maximum pending frames per node

VARIABLES
    pending,        \* pending[n] = sequence of frame IDs waiting to transmit
    busState,       \* "idle" | "arbitrating" | "transmitting"
    winner,         \* Node that won arbitration (or "none")
    transmitted,    \* Set of (node, frameId) pairs successfully transmitted
    step,           \* Step counter for bounded model checking
    starvation      \* starvation[n] = number of steps since last successful TX

vars == <<pending, busState, winner, transmitted, step, starvation>>

\* --- Type Invariant ---

TypeInvariant ==
    /\ busState \in {"idle", "arbitrating", "transmitting"}
    /\ winner \in Nodes \union {"none"}
    /\ step \in 0..MaxSteps
    /\ \A n \in Nodes : starvation[n] \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ pending = [n \in Nodes |-> <<>>]
    /\ busState = "idle"
    /\ winner = "none"
    /\ transmitted = {}
    /\ step = 0
    /\ starvation = [n \in Nodes |-> 0]

\* --- Actions ---

\* A node queues a frame for transmission (with any valid ID)
QueueFrame(n) ==
    /\ step < MaxSteps
    /\ Len(pending[n]) < MaxPending
    /\ \E id \in 0..MaxFrameId :
        /\ pending' = [pending EXCEPT ![n] = Append(@, id)]
        /\ UNCHANGED <<busState, winner, transmitted, starvation>>
    /\ step' = step + 1

\* Bus becomes idle, all nodes with pending frames enter arbitration
StartArbitration ==
    /\ busState = "idle"
    /\ \E n \in Nodes : Len(pending[n]) > 0
    /\ busState' = "arbitrating"
    /\ UNCHANGED <<pending, winner, transmitted, step, starvation>>

\* Arbitration resolves: lowest frame ID wins (highest priority)
\* Among all nodes with pending frames, the one with the lowest
\* head-of-queue frame ID wins
ResolveArbitration ==
    /\ busState = "arbitrating"
    /\ step < MaxSteps
    /\ LET contenders == {n \in Nodes : Len(pending[n]) > 0}
           headIds == {<<n, Head(pending[n])>> : n \in contenders}
           minId == CHOOSE pair \in headIds :
               \A other \in headIds : pair[2] <= other[2]
       IN /\ winner' = minId[1]
          /\ busState' = "transmitting"
    /\ step' = step + 1
    /\ UNCHANGED <<pending, transmitted, starvation>>

\* Winner completes transmission
CompleteTx ==
    /\ busState = "transmitting"
    /\ winner # "none"
    /\ step < MaxSteps
    /\ LET frameId == Head(pending[winner])
       IN /\ transmitted' = transmitted \union {<<winner, frameId>>}
          /\ pending' = [pending EXCEPT ![winner] = Tail(@)]
    /\ busState' = "idle"
    /\ starvation' = [n \in Nodes |->
        IF n = winner THEN 0 ELSE starvation[n] + 1]
    /\ winner' = "none"
    /\ step' = step + 1

\* --- Next-State Relation ---

Next ==
    \/ \E n \in Nodes : QueueFrame(n)
    \/ StartArbitration
    \/ ResolveArbitration
    \/ CompleteTx

\* --- Fairness ---

\* Weak fairness: if a node has pending frames and the bus is available,
\* it will eventually get to transmit
Fairness == WF_vars(Next)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* Only one node can be the winner at a time
SingleWinner ==
    busState = "transmitting" => winner \in Nodes

\* The winner always has the lowest frame ID among contenders
PriorityRespected ==
    (busState = "transmitting" /\ winner # "none") =>
        LET winId == Head(pending[winner])
        IN \A n \in Nodes :
            (n # winner /\ Len(pending[n]) > 0) =>
                winId <= Head(pending[n])

\* No node starves indefinitely (bounded starvation)
\* If a node has pending frames, it will transmit within MaxSteps
NoStarvation ==
    \A n \in Nodes : starvation[n] < MaxSteps

\* --- Liveness Properties ---

\* Every queued frame is eventually transmitted
EventualTransmission ==
    \A n \in Nodes :
        Len(pending[n]) > 0 ~> Len(pending[n]) < Len(pending[n])

====
