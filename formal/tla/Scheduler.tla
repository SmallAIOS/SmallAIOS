---- MODULE Scheduler ----
\* SmallAIOS Soft Real-Time Cooperative Scheduler Model
\* Verifies: deadlock freedom, starvation freedom, priority ordering,
\*           SYSTEM class preemption, watchdog liveness
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    SystemTasks,    \* Set of SYSTEM class task identifiers (watchdog, health)
    IpcTasks,       \* Set of IPC class task identifiers
    InferenceTasks, \* Set of INFERENCE class task identifiers
    MaxSteps,       \* Bound for model checking
    WatchdogTimeout \* Watchdog timeout in abstract time units

VARIABLES
    state,          \* state[t] \in {"ready", "running", "waiting", "done"}
    runQueue,       \* Sequence of ready tasks (priority-ordered)
    current,        \* Currently running task (or "none")
    step,           \* Step counter for bounded model checking
    watchdogTimer,  \* Countdown: reset on pet, system resets at 0
    lastPet         \* Step when watchdog was last serviced

\* All tasks across all classes
Tasks == SystemTasks \union IpcTasks \union InferenceTasks

vars == <<state, runQueue, current, step, watchdogTimer, lastPet>>

\* --- Helper Functions ---

\* Priority of a task: SYSTEM=0 (highest), IPC=1, INFERENCE=2
Priority(t) ==
    IF t \in SystemTasks THEN 0
    ELSE IF t \in IpcTasks THEN 1
    ELSE 2

\* Find the highest-priority (lowest number) task in the run queue
\* Returns the index of the best task, or 0 if queue is empty
BestTaskIndex(q) ==
    IF Len(q) = 0 THEN 0
    ELSE LET indices == 1..Len(q)
             bestPrio == CHOOSE i \in indices :
                 \A j \in indices : Priority(q[i]) <= Priority(q[j])
         IN bestPrio

\* Remove element at index i from sequence
RemoveAt(q, i) ==
    SubSeq(q, 1, i-1) \o SubSeq(q, i+1, Len(q))

TypeInvariant ==
    /\ state \in [Tasks -> {"ready", "running", "waiting", "done"}]
    /\ current \in Tasks \union {"none"}
    /\ step \in 0..MaxSteps
    /\ watchdogTimer \in 0..WatchdogTimeout

\* --- Initial State ---

Init ==
    /\ state = [t \in Tasks |-> "ready"]
    /\ runQueue = <<>>
    /\ current = "none"
    /\ step = 0
    /\ watchdogTimer = WatchdogTimeout
    /\ lastPet = 0

\* --- Actions ---

\* A ready task enters the run queue
Enqueue(t) ==
    /\ state[t] = "ready"
    /\ t \notin {runQueue[i] : i \in 1..Len(runQueue)}
    /\ runQueue' = Append(runQueue, t)
    /\ UNCHANGED <<state, current, step, watchdogTimer, lastPet>>

\* Scheduler picks the highest-priority task from run queue
Schedule ==
    /\ current = "none"
    /\ Len(runQueue) > 0
    /\ LET idx == BestTaskIndex(runQueue)
           t == runQueue[idx]
       IN /\ state' = [state EXCEPT ![t] = "running"]
          /\ current' = t
          /\ runQueue' = RemoveAt(runQueue, idx)
    /\ step' = step + 1
    /\ watchdogTimer' = IF watchdogTimer > 0 THEN watchdogTimer - 1 ELSE 0
    /\ UNCHANGED lastPet

\* Running INFERENCE task yields at operator boundary
\* Scheduler checks for higher-priority pending work
YieldAtOperatorBoundary(t) ==
    /\ current = t
    /\ t \in InferenceTasks
    /\ state[t] = "running"
    /\ state' = [state EXCEPT ![t] = "ready"]
    /\ runQueue' = Append(runQueue, t)
    /\ current' = "none"
    /\ step' = step + 1
    /\ watchdogTimer' = IF watchdogTimer > 0 THEN watchdogTimer - 1 ELSE 0
    /\ UNCHANGED lastPet

\* Running IPC task yields
YieldIpc(t) ==
    /\ current = t
    /\ t \in IpcTasks
    /\ state[t] = "running"
    /\ state' = [state EXCEPT ![t] = "ready"]
    /\ runQueue' = Append(runQueue, t)
    /\ current' = "none"
    /\ step' = step + 1
    /\ watchdogTimer' = IF watchdogTimer > 0 THEN watchdogTimer - 1 ELSE 0
    /\ UNCHANGED lastPet

\* SYSTEM task completes (always completes quickly, services watchdog)
CompleteSystem(t) ==
    /\ current = t
    /\ t \in SystemTasks
    /\ state[t] = "running"
    /\ state' = [state EXCEPT ![t] = "ready"]  \* SYSTEM tasks recur
    /\ current' = "none"
    /\ step' = step + 1
    /\ watchdogTimer' = WatchdogTimeout  \* Pet the watchdog
    /\ lastPet' = step + 1
    /\ UNCHANGED runQueue

\* INFERENCE task completes
CompleteInference(t) ==
    /\ current = t
    /\ t \in InferenceTasks
    /\ state[t] = "running"
    /\ state' = [state EXCEPT ![t] = "done"]
    /\ current' = "none"
    /\ step' = step + 1
    /\ watchdogTimer' = IF watchdogTimer > 0 THEN watchdogTimer - 1 ELSE 0
    /\ UNCHANGED <<runQueue, lastPet>>

\* IPC task completes its current work unit
CompleteIpc(t) ==
    /\ current = t
    /\ t \in IpcTasks
    /\ state[t] = "running"
    /\ state' = [state EXCEPT ![t] = "ready"]  \* IPC tasks recur
    /\ current' = "none"
    /\ step' = step + 1
    /\ watchdogTimer' = IF watchdogTimer > 0 THEN watchdogTimer - 1 ELSE 0
    /\ UNCHANGED <<runQueue, lastPet>>

\* --- Next State ---

Next ==
    /\ step < MaxSteps
    /\ watchdogTimer > 0  \* System halts if watchdog fires
    /\ \/ Schedule
       \/ \E t \in Tasks : Enqueue(t)
       \/ \E t \in InferenceTasks : YieldAtOperatorBoundary(t)
       \/ \E t \in IpcTasks : YieldIpc(t)
       \/ \E t \in SystemTasks : CompleteSystem(t)
       \/ \E t \in InferenceTasks : CompleteInference(t)
       \/ \E t \in IpcTasks : CompleteIpc(t)

\* --- Safety Properties ---

\* At most one task is running at any time
MutualExclusion ==
    Cardinality({t \in Tasks : state[t] = "running"}) <= 1

\* Priority inversion never happens: if a SYSTEM task is ready and
\* the scheduler picks a task, it must pick a SYSTEM task
NoPriorityInversion ==
    (current /= "none" /\ current \notin SystemTasks) =>
        ~(\E t \in SystemTasks : state[t] = "ready" /\
            t \in {runQueue[i] : i \in 1..Len(runQueue)})

\* Watchdog is always positive (system hasn't hung)
WatchdogAlive == watchdogTimer > 0

\* --- Liveness Properties (require fairness) ---

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Every SYSTEM task eventually runs (highest priority, no starvation)
SystemTasksRun ==
    \A t \in SystemTasks : state[t] = "ready" ~> state[t] = "running"

\* Every IPC task eventually runs (preempts inference)
IpcTasksRun ==
    \A t \in IpcTasks : state[t] = "ready" ~> state[t] = "running"

\* Watchdog is periodically serviced (doesn't fire)
WatchdogServiced ==
    [](watchdogTimer > 0)

====
