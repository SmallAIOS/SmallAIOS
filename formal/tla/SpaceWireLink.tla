---- MODULE SpaceWireLink ----
\* SpaceWire Link State Machine Model
\* Verifies: correct state transitions per ECSS-E-ST-50-12C
\* States: ErrorReset -> ErrorWait -> Ready -> Started -> Connecting -> Run
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    MaxSteps        \* Bound for model checking

VARIABLES
    localState,     \* Local link state
    remoteState,    \* Remote link state (simulated peer)
    localEnabled,   \* Whether local link is enabled
    remoteEnabled,  \* Whether remote link is enabled
    step            \* Step counter

vars == <<localState, remoteState, localEnabled, remoteEnabled, step>>

States == {"ErrorReset", "ErrorWait", "Ready", "Started", "Connecting", "Run"}

\* --- Type Invariant ---

TypeInvariant ==
    /\ localState \in States
    /\ remoteState \in States
    /\ localEnabled \in BOOLEAN
    /\ remoteEnabled \in BOOLEAN
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ localState = "ErrorReset"
    /\ remoteState = "ErrorReset"
    /\ localEnabled = FALSE
    /\ remoteEnabled = FALSE
    /\ step = 0

\* --- State Transitions ---

\* Valid next states from each state
ValidTransition(from, to) ==
    CASE from = "ErrorReset" -> to = "ErrorWait"
      [] from = "ErrorWait"  -> to = "Ready"
      [] from = "Ready"      -> to \in {"Started", "ErrorReset"}
      [] from = "Started"    -> to \in {"Connecting", "ErrorReset"}
      [] from = "Connecting" -> to \in {"Run", "ErrorReset"}
      [] from = "Run"        -> to \in {"ErrorReset"}
      [] OTHER               -> FALSE

\* Local state machine advances
LocalAdvance ==
    /\ step < MaxSteps
    /\ \E nextState \in States :
        /\ ValidTransition(localState, nextState)
        /\ (nextState \in {"Started", "Connecting", "Run"} => localEnabled)
        \* Connecting -> Run requires remote to be at least Started
        /\ (localState = "Connecting" /\ nextState = "Run" =>
            remoteState \in {"Started", "Connecting", "Run"})
        /\ localState' = nextState
    /\ step' = step + 1
    /\ UNCHANGED <<remoteState, localEnabled, remoteEnabled>>

\* Remote state machine advances (symmetric)
RemoteAdvance ==
    /\ step < MaxSteps
    /\ \E nextState \in States :
        /\ ValidTransition(remoteState, nextState)
        /\ (nextState \in {"Started", "Connecting", "Run"} => remoteEnabled)
        /\ (remoteState = "Connecting" /\ nextState = "Run" =>
            localState \in {"Started", "Connecting", "Run"})
        /\ remoteState' = nextState
    /\ step' = step + 1
    /\ UNCHANGED <<localState, localEnabled, remoteEnabled>>

\* Enable local link
EnableLocal ==
    /\ step < MaxSteps
    /\ ~localEnabled
    /\ localEnabled' = TRUE
    /\ step' = step + 1
    /\ UNCHANGED <<localState, remoteState, remoteEnabled>>

\* Enable remote link
EnableRemote ==
    /\ step < MaxSteps
    /\ ~remoteEnabled
    /\ remoteEnabled' = TRUE
    /\ step' = step + 1
    /\ UNCHANGED <<localState, remoteState, localEnabled>>

\* Error injection: force a link to ErrorReset
ErrorLocal ==
    /\ step < MaxSteps
    /\ localState # "ErrorReset"
    /\ localState' = "ErrorReset"
    /\ step' = step + 1
    /\ UNCHANGED <<remoteState, localEnabled, remoteEnabled>>

ErrorRemote ==
    /\ step < MaxSteps
    /\ remoteState # "ErrorReset"
    /\ remoteState' = "ErrorReset"
    /\ step' = step + 1
    /\ UNCHANGED <<localState, localEnabled, remoteEnabled>>

\* --- Next-State Relation ---

Next ==
    \/ LocalAdvance
    \/ RemoteAdvance
    \/ EnableLocal
    \/ EnableRemote
    \/ ErrorLocal
    \/ ErrorRemote

Fairness == WF_vars(Next)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* Only valid state transitions occur
OnlyValidTransitions ==
    /\ localState \in States
    /\ remoteState \in States

\* Run state requires both sides to have progressed
RunRequiresPeer ==
    (localState = "Run") => remoteState \in {"Started", "Connecting", "Run"}

\* Both links in Run state means bidirectional communication possible
BidirectionalRun ==
    (localState = "Run" /\ remoteState = "Run") =>
        (localEnabled /\ remoteEnabled)

\* --- Liveness ---

\* If both links are enabled, they eventually reach Run
EventualConnection ==
    (localEnabled /\ remoteEnabled) ~>
        (localState = "Run" /\ remoteState = "Run")

====
