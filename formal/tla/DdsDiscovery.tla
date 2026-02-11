---- MODULE DdsDiscovery ----
\* DDS SPDP/SEDP Discovery Model
\* Verifies: eventual consistency, no orphan endpoints, correct matching
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Participants,   \* Set of DDS participant identifiers
    Topics,         \* Set of topic names
    MaxSteps        \* Bound for model checking

VARIABLES
    \* SPDP state: each participant's view of discovered participants
    spdpDb,         \* spdpDb[p] = set of participant IDs known to p
    \* SEDP state: each participant's known writers and readers
    writers,        \* writers[p] = set of (participantId, topic) pairs for writers p knows about
    readers,        \* readers[p] = set of (participantId, topic) pairs for readers p knows about
    \* Local endpoints: what each participant has published/subscribed
    localWriters,   \* localWriters[p] = set of topics p is writing
    localReaders,   \* localReaders[p] = set of topics p is reading
    \* Matched endpoints
    matched,        \* matched[p] = set of (localTopic, remoteParticipant) matched pairs
    step            \* Step counter

vars == <<spdpDb, writers, readers, localWriters, localReaders, matched, step>>

\* --- Type Invariant ---

TypeInvariant ==
    /\ \A p \in Participants : spdpDb[p] \subseteq Participants
    /\ \A p \in Participants : localWriters[p] \subseteq Topics
    /\ \A p \in Participants : localReaders[p] \subseteq Topics
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ spdpDb = [p \in Participants |-> {p}]  \* Each knows itself
    /\ writers = [p \in Participants |-> {}]
    /\ readers = [p \in Participants |-> {}]
    /\ localWriters = [p \in Participants |-> {}]
    /\ localReaders = [p \in Participants |-> {}]
    /\ matched = [p \in Participants |-> {}]
    /\ step = 0

\* --- Actions ---

\* Participant creates a DataWriter for a topic
CreateWriter(p, topic) ==
    /\ step < MaxSteps
    /\ topic \notin localWriters[p]
    /\ localWriters' = [localWriters EXCEPT ![p] = @ \union {topic}]
    /\ step' = step + 1
    /\ UNCHANGED <<spdpDb, writers, readers, localReaders, matched>>

\* Participant creates a DataReader for a topic
CreateReader(p, topic) ==
    /\ step < MaxSteps
    /\ topic \notin localReaders[p]
    /\ localReaders' = [localReaders EXCEPT ![p] = @ \union {topic}]
    /\ step' = step + 1
    /\ UNCHANGED <<spdpDb, writers, readers, localWriters, matched>>

\* SPDP announcement: participant p announces itself to participant q
SpdpAnnounce(p, q) ==
    /\ step < MaxSteps
    /\ p # q
    /\ spdpDb' = [spdpDb EXCEPT ![q] = @ \union {p}]
    /\ step' = step + 1
    /\ UNCHANGED <<writers, readers, localWriters, localReaders, matched>>

\* SEDP: exchange writer endpoint info after SPDP discovery
SedpExchangeWriters(p, q) ==
    /\ step < MaxSteps
    /\ p # q
    /\ p \in spdpDb[q]  \* q has discovered p via SPDP
    /\ \E topic \in localWriters[p] :
        /\ <<p, topic>> \notin writers[q]
        /\ writers' = [writers EXCEPT ![q] = @ \union {<<p, topic>>}]
    /\ step' = step + 1
    /\ UNCHANGED <<spdpDb, readers, localWriters, localReaders, matched>>

\* SEDP: exchange reader endpoint info after SPDP discovery
SedpExchangeReaders(p, q) ==
    /\ step < MaxSteps
    /\ p # q
    /\ p \in spdpDb[q]
    /\ \E topic \in localReaders[p] :
        /\ <<p, topic>> \notin readers[q]
        /\ readers' = [readers EXCEPT ![q] = @ \union {<<p, topic>>}]
    /\ step' = step + 1
    /\ UNCHANGED <<spdpDb, writers, localWriters, localReaders, matched>>

\* Endpoint matching: if p has a writer for topic T and knows about
\* a remote reader for T, create a match
MatchEndpoints(p) ==
    /\ step < MaxSteps
    /\ \E topic \in localWriters[p] :
        \E pair \in readers[p] :
            /\ pair[2] = topic
            /\ pair[1] # p  \* Don't match with self
            /\ <<topic, pair[1]>> \notin matched[p]
            /\ matched' = [matched EXCEPT ![p] = @ \union {<<topic, pair[1]>>}]
    /\ step' = step + 1
    /\ UNCHANGED <<spdpDb, writers, readers, localWriters, localReaders>>

\* --- Next-State Relation ---

Next ==
    \/ \E p \in Participants, t \in Topics : CreateWriter(p, t)
    \/ \E p \in Participants, t \in Topics : CreateReader(p, t)
    \/ \E p, q \in Participants : SpdpAnnounce(p, q)
    \/ \E p, q \in Participants : SedpExchangeWriters(p, q)
    \/ \E p, q \in Participants : SedpExchangeReaders(p, q)
    \/ \E p \in Participants : MatchEndpoints(p)

Fairness == WF_vars(Next)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* Every participant always knows about itself
SelfAwareness ==
    \A p \in Participants : p \in spdpDb[p]

\* Matched endpoints correspond to real local/remote endpoints
MatchedEndpointsValid ==
    \A p \in Participants :
        \A m \in matched[p] :
            /\ m[1] \in localWriters[p]  \* Local writer exists
            /\ <<m[2], m[1]>> \in readers[p]  \* Remote reader is known

\* Writers database only contains real writers
WritersConsistent ==
    \A p \in Participants :
        \A w \in writers[p] :
            w[2] \in localWriters[w[1]]

\* Readers database only contains real readers
ReadersConsistent ==
    \A p \in Participants :
        \A r \in readers[p] :
            r[2] \in localReaders[r[1]]

\* --- Liveness (Eventual Consistency) ---

\* Eventually all participants discover each other
EventualSpdpConsistency ==
    \A p, q \in Participants : <>(q \in spdpDb[p])

\* Eventually all matching writers and readers are discovered and matched
EventualMatching ==
    \A p, q \in Participants :
        \A t \in Topics :
            (t \in localWriters[p] /\ t \in localReaders[q]) ~>
                <<t, q>> \in matched[p]

====
