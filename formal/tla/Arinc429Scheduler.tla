---- MODULE Arinc429Scheduler ----
\* ARINC 429 Transmit Scheduler Model
\* Verifies: no label starvation, rate compliance, periodic scheduling
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Labels,         \* Set of ARINC 429 labels to schedule
    MaxRate,        \* Maximum transmission rate (ticks per label period)
    MaxSteps        \* Bound for model checking

VARIABLES
    counter,        \* counter[l] = ticks since last transmission of label l
    period,         \* period[l] = required transmission period for label l
    transmitted,    \* Set of labels transmitted in current tick
    tick,           \* Current tick counter
    lastTx,         \* lastTx[l] = tick when label l was last transmitted
    step            \* Step counter

vars == <<counter, period, transmitted, tick, lastTx, step>>

\* --- Type Invariant ---

TypeInvariant ==
    /\ \A l \in Labels : counter[l] \in 0..MaxRate
    /\ \A l \in Labels : period[l] \in 1..MaxRate
    /\ tick \in 0..MaxSteps
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ counter = [l \in Labels |-> 0]
    /\ period = [l \in Labels |-> CHOOSE r \in 1..MaxRate : TRUE]
    /\ transmitted = {}
    /\ tick = 0
    /\ lastTx = [l \in Labels |-> 0]
    /\ step = 0

\* --- Actions ---

\* Advance one tick: increment all counters
AdvanceTick ==
    /\ step < MaxSteps
    /\ counter' = [l \in Labels |-> counter[l] + 1]
    /\ tick' = tick + 1
    /\ transmitted' = {}
    /\ step' = step + 1
    /\ UNCHANGED <<period, lastTx>>

\* Transmit a label whose counter has reached its period
TransmitLabel(l) ==
    /\ step < MaxSteps
    /\ counter[l] >= period[l]
    /\ l \notin transmitted
    /\ counter' = [counter EXCEPT ![l] = 0]
    /\ transmitted' = transmitted \union {l}
    /\ lastTx' = [lastTx EXCEPT ![l] = tick]
    /\ step' = step + 1
    /\ UNCHANGED <<period, tick>>

\* --- Next-State Relation ---

Next ==
    \/ AdvanceTick
    \/ \E l \in Labels : TransmitLabel(l)

Fairness == WF_vars(Next)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* No label exceeds twice its period without transmission
NoPeriodViolation ==
    \A l \in Labels : counter[l] <= 2 * period[l]

\* At most one transmission per label per tick
NoDuplicateTransmission ==
    \A l \in Labels : l \in transmitted =>
        counter[l] = 0

\* --- Liveness Properties ---

\* Every label is eventually transmitted
NoLabelStarvation ==
    \A l \in Labels : counter[l] >= period[l] ~> l \in transmitted

====
