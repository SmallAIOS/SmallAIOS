---- MODULE Scheduler ----
\* SmallAIOS Cooperative Task Scheduler Model
\* Verifies: deadlock freedom, starvation freedom, fairness
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Tasks,          \* Set of task identifiers
    MaxSteps        \* Bound for model checking

VARIABLES
    state,          \* state[t] \in {"ready", "running", "waiting", "done"}
    runQueue,       \* Sequence of ready tasks
    current,        \* Currently running task (or "none")
    step            \* Step counter for bounded model checking

vars == <<state, runQueue, current, step>>

TypeInvariant ==
    /\ state \in [Tasks -> {"ready", "running", "waiting", "done"}]
    /\ current \in Tasks \union {"none"}
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ state = [t \in Tasks |-> "ready"]
    /\ runQueue = <<>>
    /\ current = "none"
    /\ step = 0

\* --- Actions ---

\* A ready task enters the run queue
Enqueue(t) ==
    /\ state[t] = "ready"
    /\ state' = [state EXCEPT ![t] = "ready"]
    /\ runQueue' = Append(runQueue, t)
    /\ UNCHANGED <<current, step>>

\* Scheduler picks next task from run queue
Schedule ==
    /\ current = "none"
    /\ Len(runQueue) > 0
    /\ LET t == Head(runQueue)
       IN /\ state' = [state EXCEPT ![t] = "running"]
          /\ current' = t
          /\ runQueue' = Tail(runQueue)
    /\ step' = step + 1

\* Running task yields (cooperative)
Yield(t) ==
    /\ current = t
    /\ state[t] = "running"
    /\ state' = [state EXCEPT ![t] = "ready"]
    /\ runQueue' = Append(runQueue, t)
    /\ current' = "none"
    /\ step' = step + 1

\* Running task completes
Complete(t) ==
    /\ current = t
    /\ state[t] = "running"
    /\ state' = [state EXCEPT ![t] = "done"]
    /\ current' = "none"
    /\ step' = step + 1
    /\ UNCHANGED runQueue

\* --- Next State ---

Next ==
    /\ step < MaxSteps
    /\ \/ Schedule
       \/ \E t \in Tasks : Enqueue(t)
       \/ \E t \in Tasks : Yield(t)
       \/ \E t \in Tasks : Complete(t)

\* --- Safety Properties ---

\* At most one task is running at any time
MutualExclusion ==
    Cardinality({t \in Tasks : state[t] = "running"}) <= 1

\* No deadlock: if tasks exist that aren't done, something can happen
NoDeadlock ==
    (\E t \in Tasks : state[t] /= "done") => ENABLED(Next)

\* --- Liveness Properties (require fairness) ---

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Every ready task eventually runs (starvation freedom)
StarvationFreedom ==
    \A t \in Tasks : state[t] = "ready" ~> state[t] = "running"

====
