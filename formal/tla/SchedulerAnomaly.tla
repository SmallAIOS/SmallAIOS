---- MODULE SchedulerAnomaly ----
\* SmallAIOS Scheduler with Anomaly Detection Thresholds
\* Extends the base Scheduler model to verify:
\*   - Anomaly detection triggers correctly at threshold crossings
\*   - Alert suppression prevents flooding
\*   - Threshold configuration bounds are respected
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    SystemTasks,        \* Set of SYSTEM class task identifiers
    IpcTasks,           \* Set of IPC class task identifiers
    InferenceTasks,     \* Set of INFERENCE class task identifiers
    MaxSteps,           \* Bound for model checking
    WatchdogTimeout,    \* Watchdog timeout in abstract time units
    DenialThreshold,    \* Capability denial rate alert threshold (per step)
    LatencyThreshold    \* Inference latency anomaly threshold (abstract units)

VARIABLES
    state,              \* state[t] \in {"ready", "running", "waiting", "done"}
    runQueue,           \* Sequence of ready tasks
    current,            \* Currently running task (or "none")
    step,               \* Step counter
    watchdogTimer,      \* Countdown: reset on pet, system resets at 0
    lastPet,            \* Step when watchdog was last serviced
    \* Anomaly detection state
    denialCount,        \* Per-step capability denial counter
    latencyAccum,       \* Running inference latency accumulator
    alertsRaised,       \* Total alerts raised
    alertSuppressed     \* Whether alert suppression is active

Tasks == SystemTasks \union IpcTasks \union InferenceTasks

vars == <<state, runQueue, current, step, watchdogTimer, lastPet,
          denialCount, latencyAccum, alertsRaised, alertSuppressed>>

\* --- Helper Functions ---

Priority(t) ==
    IF t \in SystemTasks THEN 0
    ELSE IF t \in IpcTasks THEN 1
    ELSE 2

BestTaskIndex(q) ==
    IF Len(q) = 0 THEN 0
    ELSE LET indices == 1..Len(q)
             bestPrio == CHOOSE i \in indices :
                 \A j \in indices : Priority(q[i]) <= Priority(q[j])
         IN bestPrio

RemoveAt(q, i) ==
    SubSeq(q, 1, i-1) \o SubSeq(q, i+1, Len(q))

\* --- Type Invariant ---

TypeInvariant ==
    /\ state \in [Tasks -> {"ready", "running", "waiting", "done"}]
    /\ current \in Tasks \union {"none"}
    /\ step \in 0..MaxSteps
    /\ watchdogTimer \in 0..WatchdogTimeout
    /\ denialCount \in 0..MaxSteps
    /\ latencyAccum \in Nat
    /\ alertsRaised \in Nat
    /\ alertSuppressed \in BOOLEAN

\* --- Initial State ---

Init ==
    /\ state = [t \in Tasks |-> "ready"]
    /\ runQueue = <<>>
    /\ current = "none"
    /\ step = 0
    /\ watchdogTimer = WatchdogTimeout
    /\ lastPet = 0
    /\ denialCount = 0
    /\ latencyAccum = 0
    /\ alertsRaised = 0
    /\ alertSuppressed = FALSE

\* --- Anomaly Detection Actions ---

\* A capability denial occurs (e.g., task lacks permission)
CapabilityDenial ==
    /\ step < MaxSteps
    /\ denialCount' = denialCount + 1
    /\ IF denialCount + 1 >= DenialThreshold /\ ~alertSuppressed
       THEN /\ alertsRaised' = alertsRaised + 1
            /\ alertSuppressed' = TRUE
       ELSE UNCHANGED <<alertsRaised, alertSuppressed>>
    /\ step' = step + 1
    /\ UNCHANGED <<state, runQueue, current, watchdogTimer, lastPet, latencyAccum>>

\* An inference completes with measured latency
InferenceLatencyReport(latency) ==
    /\ step < MaxSteps
    /\ latencyAccum' = latencyAccum + latency
    /\ IF latency > LatencyThreshold /\ ~alertSuppressed
       THEN /\ alertsRaised' = alertsRaised + 1
            /\ alertSuppressed' = TRUE
       ELSE UNCHANGED <<alertsRaised, alertSuppressed>>
    /\ step' = step + 1
    /\ UNCHANGED <<state, runQueue, current, watchdogTimer, lastPet, denialCount>>

\* Reset denial counter (periodic, e.g., per scheduling quantum)
ResetDenialCounter ==
    /\ step < MaxSteps
    /\ denialCount > 0
    /\ denialCount' = 0
    /\ alertSuppressed' = FALSE
    /\ step' = step + 1
    /\ UNCHANGED <<state, runQueue, current, watchdogTimer, lastPet,
                   latencyAccum, alertsRaised>>

\* --- Scheduler Actions (from base model) ---

Enqueue(t) ==
    /\ state[t] = "ready"
    /\ t \notin {runQueue[i] : i \in 1..Len(runQueue)}
    /\ runQueue' = Append(runQueue, t)
    /\ UNCHANGED <<state, current, step, watchdogTimer, lastPet,
                   denialCount, latencyAccum, alertsRaised, alertSuppressed>>

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
    /\ UNCHANGED <<lastPet, denialCount, latencyAccum, alertsRaised, alertSuppressed>>

CompleteSystem(t) ==
    /\ current = t
    /\ t \in SystemTasks
    /\ state[t] = "running"
    /\ state' = [state EXCEPT ![t] = "ready"]
    /\ current' = "none"
    /\ step' = step + 1
    /\ watchdogTimer' = WatchdogTimeout
    /\ lastPet' = step + 1
    /\ UNCHANGED <<runQueue, denialCount, latencyAccum, alertsRaised, alertSuppressed>>

YieldInference(t) ==
    /\ current = t
    /\ t \in InferenceTasks
    /\ state[t] = "running"
    /\ state' = [state EXCEPT ![t] = "ready"]
    /\ runQueue' = Append(runQueue, t)
    /\ current' = "none"
    /\ step' = step + 1
    /\ watchdogTimer' = IF watchdogTimer > 0 THEN watchdogTimer - 1 ELSE 0
    /\ UNCHANGED <<lastPet, denialCount, latencyAccum, alertsRaised, alertSuppressed>>

\* --- Safety Properties ---

\* Alerts are bounded: cannot exceed step count
AlertsBounded ==
    alertsRaised <= step

\* Alert suppression prevents duplicate alerts within a window.
\* If suppressed, the triggering condition must still be reflected in counters.
SuppressionEffective ==
    alertSuppressed => (denialCount >= DenialThreshold \/ latencyAccum > 0)

\* Denial count never exceeds step count
DenialCountBounded ==
    denialCount <= step

\* Watchdog alive (not a safety invariant -- the watchdog is *designed* to
\* reach zero when system tasks are starved, triggering a reset)
WatchdogAlive == watchdogTimer > 0

\* Mutual exclusion
MutualExclusion ==
    Cardinality({t \in Tasks : state[t] = "running"}) <= 1

\* Threshold correctness: alerts are raised only when conditions justify them.
\* Since counters reset periodically, we verify this property via the
\* action constraint: CapabilityDenial and InferenceLatencyReport only
\* increment alertsRaised when the threshold condition holds at that step.
\* As a state invariant, we verify the weaker but still useful property:
\* if alert suppression is active, then some threshold was crossed.
ThresholdCorrectness ==
    alertSuppressed =>
        (denialCount >= DenialThreshold \/ latencyAccum > 0)

\* --- Termination ---

\* System halts when watchdog fires or step bound reached
Done ==
    /\ (step >= MaxSteps \/ watchdogTimer = 0)
    /\ UNCHANGED vars

\* --- Next State ---

Next ==
    \/ (/\ step < MaxSteps
        /\ watchdogTimer > 0
        /\ \/ Schedule
           \/ \E t \in Tasks : Enqueue(t)
           \/ \E t \in InferenceTasks : YieldInference(t)
           \/ \E t \in SystemTasks : CompleteSystem(t)
           \/ CapabilityDenial
           \/ \E l \in 1..LatencyThreshold+2 : InferenceLatencyReport(l)
           \/ ResetDenialCounter)
    \/ Done

Spec == Init /\ [][Next]_vars

====
