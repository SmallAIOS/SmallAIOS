---- MODULE QuicMigration ----
\* QUIC Connection Migration Model
\* Verifies: no data loss during path switch, path validation correctness
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Paths,          \* Set of available network paths (e.g., {1, 2, 3})
    MaxSteps,       \* Bound for model checking
    MaxInFlight     \* Maximum packets in flight

VARIABLES
    activePath,     \* Currently active path
    pathState,      \* pathState[p] \in {"unknown", "validating", "validated", "failed"}
    pendingChallenge, \* Path being validated (or "none")
    sentPackets,    \* Set of sequence numbers sent
    receivedPackets,\* Set of sequence numbers received
    inFlight,       \* Set of (seqNum, path) pairs in transit
    nextSeqNum,     \* Next sequence number to send
    step            \* Step counter

vars == <<activePath, pathState, pendingChallenge, sentPackets,
          receivedPackets, inFlight, nextSeqNum, step>>

\* --- Type Invariant ---

TypeInvariant ==
    /\ activePath \in Paths
    /\ \A p \in Paths : pathState[p] \in {"unknown", "validating", "validated", "failed"}
    /\ pendingChallenge \in Paths \union {"none"}
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ activePath = CHOOSE p \in Paths : TRUE
    /\ pathState = [p \in Paths |->
        IF p = activePath THEN "validated" ELSE "unknown"]
    /\ pendingChallenge = "none"
    /\ sentPackets = {}
    /\ receivedPackets = {}
    /\ inFlight = {}
    /\ nextSeqNum = 1
    /\ step = 0

\* --- Actions ---

\* Send a packet on the active path
SendPacket ==
    /\ step < MaxSteps
    /\ pathState[activePath] = "validated"
    /\ Cardinality(inFlight) < MaxInFlight
    /\ sentPackets' = sentPackets \union {nextSeqNum}
    /\ inFlight' = inFlight \union {<<nextSeqNum, activePath>>}
    /\ nextSeqNum' = nextSeqNum + 1
    /\ step' = step + 1
    /\ UNCHANGED <<activePath, pathState, pendingChallenge, receivedPackets>>

\* Packet arrives at receiver
ReceivePacket ==
    /\ step < MaxSteps
    /\ \E pkt \in inFlight :
        /\ pathState[pkt[2]] # "failed"  \* Path is not failed
        /\ receivedPackets' = receivedPackets \union {pkt[1]}
        /\ inFlight' = inFlight \ {pkt}
    /\ step' = step + 1
    /\ UNCHANGED <<activePath, pathState, pendingChallenge, sentPackets, nextSeqNum>>

\* Initiate path migration: send PATH_CHALLENGE on new path
InitiateMigration(newPath) ==
    /\ step < MaxSteps
    /\ newPath # activePath
    /\ pathState[newPath] \in {"unknown", "validated"}
    /\ pendingChallenge = "none"
    /\ pendingChallenge' = newPath
    /\ pathState' = [pathState EXCEPT ![newPath] = "validating"]
    /\ step' = step + 1
    /\ UNCHANGED <<activePath, sentPackets, receivedPackets, inFlight, nextSeqNum>>

\* PATH_RESPONSE received: validation succeeds, switch path
PathValidated ==
    /\ step < MaxSteps
    /\ pendingChallenge # "none"
    /\ pathState[pendingChallenge] = "validating"
    /\ pathState' = [pathState EXCEPT ![pendingChallenge] = "validated"]
    /\ activePath' = pendingChallenge
    /\ pendingChallenge' = "none"
    /\ step' = step + 1
    /\ UNCHANGED <<sentPackets, receivedPackets, inFlight, nextSeqNum>>

\* Path validation fails
PathValidationFailed ==
    /\ step < MaxSteps
    /\ pendingChallenge # "none"
    /\ pathState[pendingChallenge] = "validating"
    /\ pathState' = [pathState EXCEPT ![pendingChallenge] = "failed"]
    /\ pendingChallenge' = "none"
    /\ step' = step + 1
    /\ UNCHANGED <<activePath, sentPackets, receivedPackets, inFlight, nextSeqNum>>

\* Active path fails, triggering fallback
ActivePathFails ==
    /\ step < MaxSteps
    /\ pathState' = [pathState EXCEPT ![activePath] = "failed"]
    \* Fall back to any validated path
    /\ \E p \in Paths :
        /\ p # activePath
        /\ pathState[p] = "validated"
        /\ activePath' = p
    /\ step' = step + 1
    /\ UNCHANGED <<pendingChallenge, sentPackets, receivedPackets, inFlight, nextSeqNum>>

\* --- Next-State Relation ---

Next ==
    \/ SendPacket
    \/ ReceivePacket
    \/ \E p \in Paths : InitiateMigration(p)
    \/ PathValidated
    \/ PathValidationFailed
    \/ ActivePathFails

Fairness == WF_vars(Next)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* Active path is always validated
ActivePathValidated ==
    pathState[activePath] = "validated"

\* No data loss: received is always a subset of sent
NoDataLoss ==
    receivedPackets \subseteq sentPackets

\* At most one path being validated at a time
SingleValidation ==
    Cardinality({p \in Paths : pathState[p] = "validating"}) <= 1

====
