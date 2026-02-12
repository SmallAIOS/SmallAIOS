---- MODULE I2CArbitration ----
\* I2C Multi-Master Bus Arbitration Model
\* Verifies: mutual exclusion on bus, no starvation, arbitration loss detection
\* Models wired-AND SDA line arbitration per I2C specification
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    NumMasters,     \* Number of masters on the bus (2-3)
    MaxRetries      \* Maximum retry attempts before starvation bound

VARIABLES
    bus_state,      \* "idle" | "arbitrating" | "busy"
    requesting,     \* Set of masters currently requesting the bus
    current_master, \* Master that owns the bus (0 = none)
    sda_line,       \* Wired-AND SDA line value (TRUE = high/released)
    retry_count,    \* retry_count[m] = number of consecutive losses
    arb_lost,       \* Set of masters that detected arbitration loss
    step            \* Step counter for bounded checking

Masters == 1..NumMasters

vars == <<bus_state, requesting, current_master, sda_line, retry_count, arb_lost, step>>

\* --- Type Invariant ---

TypeInvariant ==
    /\ bus_state \in {"idle", "arbitrating", "busy"}
    /\ requesting \subseteq Masters
    /\ current_master \in Masters \union {0}
    /\ sda_line \in BOOLEAN
    /\ \A m \in Masters : retry_count[m] \in 0..MaxRetries
    /\ arb_lost \subseteq Masters
    /\ step \in Nat

\* --- Initial State ---

Init ==
    /\ bus_state = "idle"
    /\ requesting = {}
    /\ current_master = 0
    /\ sda_line = TRUE
    /\ retry_count = [m \in Masters |-> 0]
    /\ arb_lost = {}
    /\ step = 0

\* --- Actions ---

\* A master requests the bus
RequestBus(m) ==
    /\ step < MaxRetries * NumMasters + 10
    /\ m \notin requesting
    /\ bus_state \in {"idle", "arbitrating"}
    /\ requesting' = requesting \union {m}
    /\ step' = step + 1
    /\ UNCHANGED <<bus_state, current_master, sda_line, retry_count, arb_lost>>

\* Bus transitions to arbitration when requests exist and bus is idle
StartArbitration ==
    /\ bus_state = "idle"
    /\ requesting # {}
    /\ bus_state' = "arbitrating"
    /\ arb_lost' = {}
    /\ UNCHANGED <<requesting, current_master, sda_line, retry_count, step>>

\* Arbitration resolves: lowest-numbered master wins (models address priority)
\* Losers detect loss via SDA line mismatch
ResolveArbitration ==
    /\ bus_state = "arbitrating"
    /\ requesting # {}
    /\ LET winner == CHOOSE m \in requesting : \A m2 \in requesting : m <= m2
           losers == requesting \ {winner}
       IN /\ current_master' = winner
          /\ bus_state' = "busy"
          /\ sda_line' = FALSE  \* Bus driven low by master
          /\ arb_lost' = losers
          /\ retry_count' = [m \in Masters |->
              IF m \in losers THEN IF retry_count[m] < MaxRetries
                                   THEN retry_count[m] + 1
                                   ELSE MaxRetries
              ELSE IF m = winner THEN 0
              ELSE retry_count[m]]
          /\ requesting' = requesting \ {winner}
    /\ step' = step + 1

\* Master completes its transaction and releases the bus
ReleaseBus ==
    /\ bus_state = "busy"
    /\ current_master # 0
    /\ bus_state' = "idle"
    /\ current_master' = 0
    /\ sda_line' = TRUE  \* Bus released (pulled high)
    /\ arb_lost' = {}
    /\ step' = step + 1
    /\ UNCHANGED <<requesting, retry_count>>

\* Arbitration loser retries (re-queues after losing)
RetryAfterLoss(m) ==
    /\ m \in arb_lost
    /\ retry_count[m] < MaxRetries
    /\ bus_state \in {"busy", "idle"}
    /\ requesting' = requesting \union {m}
    /\ arb_lost' = arb_lost \ {m}
    /\ step' = step + 1
    /\ UNCHANGED <<bus_state, current_master, sda_line, retry_count>>

\* --- Next-State Relation ---

Next ==
    \/ \E m \in Masters : RequestBus(m)
    \/ StartArbitration
    \/ ResolveArbitration
    \/ ReleaseBus
    \/ \E m \in Masters : RetryAfterLoss(m)

Fairness ==
    /\ WF_vars(StartArbitration)
    /\ WF_vars(ResolveArbitration)
    /\ WF_vars(ReleaseBus)
    /\ \A m \in Masters : WF_vars(RetryAfterLoss(m))

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* Mutual exclusion: at most one master owns the bus
MutualExclusion ==
    bus_state = "busy" => current_master \in Masters

\* Bus busy implies exactly one owner
SingleOwner ==
    (bus_state = "busy") => (current_master # 0)

\* Idle bus has no owner
IdleNoOwner ==
    (bus_state = "idle") => (current_master = 0)

\* Arbitration losers always detect their loss
\* When bus is busy, all non-owner masters with nonzero retry count were detected
LosersDetected ==
    (bus_state = "busy" /\ current_master # 0) =>
        \A m \in Masters :
            (m # current_master /\ retry_count[m] > 0) =>
                (m \in arb_lost \/ m \notin requesting)

\* No master exceeds retry limit (bounded starvation)
BoundedRetries ==
    \A m \in Masters : retry_count[m] <= MaxRetries

\* --- Liveness Properties ---

\* Every requesting master eventually gets the bus
EventualAccess ==
    \A m \in Masters :
        (m \in requesting) ~> (current_master = m)

====
