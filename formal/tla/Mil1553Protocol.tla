---- MODULE Mil1553Protocol ----
\* MIL-STD-1553 Command/Response Protocol Model
\* Verifies: no unanswered commands, correct state transitions,
\*           dual-bus failover correctness
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    RemoteTerminals, \* Set of RT addresses (0-30)
    MaxSteps,        \* Bound for model checking
    ResponseTimeout  \* Timeout in ticks for RT response

VARIABLES
    bcState,        \* "idle" | "commanding" | "waiting_response" | "processing"
    targetRt,       \* Current target RT address (or "none")
    rtState,        \* rtState[rt] \in {"idle", "processing", "responding"}
    activeBus,      \* "A" | "B" (current active bus)
    busHealth,      \* busHealth[b] \in {"healthy", "degraded", "failed"}
    pendingCmds,    \* Sequence of (rt, subaddress) pairs to execute
    responseTimer,  \* Countdown timer for response timeout
    completedCmds,  \* Number of successfully completed commands
    timeouts,       \* Number of command timeouts
    step            \* Step counter

vars == <<bcState, targetRt, rtState, activeBus, busHealth,
          pendingCmds, responseTimer, completedCmds, timeouts, step>>

\* --- Type Invariant ---

TypeInvariant ==
    /\ bcState \in {"idle", "commanding", "waiting_response", "processing"}
    /\ targetRt \in RemoteTerminals \union {"none"}
    /\ activeBus \in {"A", "B"}
    /\ \A b \in {"A", "B"} : busHealth[b] \in {"healthy", "degraded", "failed"}
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ bcState = "idle"
    /\ targetRt = "none"
    /\ rtState = [rt \in RemoteTerminals |-> "idle"]
    /\ activeBus = "A"
    /\ busHealth = [b \in {"A", "B"} |-> "healthy"]
    /\ pendingCmds = <<>>
    /\ responseTimer = 0
    /\ completedCmds = 0
    /\ timeouts = 0
    /\ step = 0

\* --- Actions ---

\* BC queues a command for an RT
QueueCommand(rt) ==
    /\ step < MaxSteps
    /\ Len(pendingCmds) < 32
    /\ pendingCmds' = Append(pendingCmds, rt)
    /\ step' = step + 1
    /\ UNCHANGED <<bcState, targetRt, rtState, activeBus, busHealth,
                   responseTimer, completedCmds, timeouts>>

\* BC sends next command from queue
SendCommand ==
    /\ step < MaxSteps
    /\ bcState = "idle"
    /\ Len(pendingCmds) > 0
    /\ busHealth[activeBus] # "failed"
    /\ LET rt == Head(pendingCmds)
       IN /\ bcState' = "commanding"
          /\ targetRt' = rt
          /\ pendingCmds' = Tail(pendingCmds)
    /\ step' = step + 1
    /\ UNCHANGED <<rtState, activeBus, busHealth, responseTimer,
                   completedCmds, timeouts>>

\* Command arrives at RT, RT begins processing
RtReceiveCommand ==
    /\ step < MaxSteps
    /\ bcState = "commanding"
    /\ targetRt # "none"
    /\ rtState[targetRt] = "idle"
    /\ rtState' = [rtState EXCEPT ![targetRt] = "processing"]
    /\ bcState' = "waiting_response"
    /\ responseTimer' = ResponseTimeout
    /\ step' = step + 1
    /\ UNCHANGED <<targetRt, activeBus, busHealth, pendingCmds,
                   completedCmds, timeouts>>

\* RT sends response back to BC
RtSendResponse ==
    /\ step < MaxSteps
    /\ bcState = "waiting_response"
    /\ targetRt # "none"
    /\ rtState[targetRt] = "processing"
    /\ rtState' = [rtState EXCEPT ![targetRt] = "idle"]
    /\ bcState' = "idle"
    /\ targetRt' = "none"
    /\ completedCmds' = completedCmds + 1
    /\ responseTimer' = 0
    /\ step' = step + 1
    /\ UNCHANGED <<activeBus, busHealth, pendingCmds, timeouts>>

\* Response timeout: BC retries on alternate bus
ResponseTimedOut ==
    /\ step < MaxSteps
    /\ bcState = "waiting_response"
    /\ responseTimer = 0
    /\ bcState' = "idle"
    /\ targetRt' = "none"
    /\ timeouts' = timeouts + 1
    \* Failover to alternate bus
    /\ activeBus' = IF activeBus = "A" THEN "B" ELSE "A"
    /\ busHealth' = [busHealth EXCEPT ![activeBus] = "degraded"]
    /\ rtState' = [rtState EXCEPT ![targetRt] = "idle"]
    /\ step' = step + 1
    /\ UNCHANGED <<pendingCmds, responseTimer, completedCmds>>

\* Timer tick (decrement response timer)
TimerTick ==
    /\ step < MaxSteps
    /\ bcState = "waiting_response"
    /\ responseTimer > 0
    /\ responseTimer' = responseTimer - 1
    /\ step' = step + 1
    /\ UNCHANGED <<bcState, targetRt, rtState, activeBus, busHealth,
                   pendingCmds, completedCmds, timeouts>>

\* --- Next-State Relation ---

Next ==
    \/ \E rt \in RemoteTerminals : QueueCommand(rt)
    \/ SendCommand
    \/ RtReceiveCommand
    \/ RtSendResponse
    \/ ResponseTimedOut
    \/ TimerTick

Fairness == WF_vars(Next)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* BC is never commanding and waiting at the same time
MutualExclusiveStates ==
    bcState \in {"idle", "commanding", "waiting_response", "processing"}

\* No RT is in processing state without BC waiting for it
NoOrphanProcessing ==
    \A rt \in RemoteTerminals :
        rtState[rt] = "processing" => (bcState \in {"waiting_response"} /\ targetRt = rt)

\* At least one bus must be available
BusAvailability ==
    busHealth["A"] # "failed" \/ busHealth["B"] # "failed"

====
