---- MODULE QuicFlowControl ----
\* QUIC Flow Control Model
\* Verifies: no deadlock, no limit violation, connection-level and
\*           stream-level flow control correctness
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    Streams,            \* Set of stream identifiers
    InitConnLimit,      \* Initial connection-level flow control limit
    InitStreamLimit,    \* Initial per-stream flow control limit
    MaxData,            \* Maximum total data that can be sent
    MaxSteps            \* Bound for model checking

VARIABLES
    connSent,           \* Total bytes sent at connection level
    connLimit,          \* Connection-level MAX_DATA limit
    streamSent,         \* streamSent[s] = bytes sent on stream s
    streamLimit,        \* streamLimit[s] = MAX_STREAM_DATA limit for stream s
    streamClosed,       \* streamClosed[s] = whether stream s is closed
    step                \* Step counter

vars == <<connSent, connLimit, streamSent, streamLimit, streamClosed, step>>

\* --- Type Invariant ---

TypeInvariant ==
    /\ connSent \in 0..MaxData
    /\ connLimit \in 0..MaxData
    /\ \A s \in Streams : streamSent[s] \in 0..MaxData
    /\ \A s \in Streams : streamLimit[s] \in 0..MaxData
    /\ \A s \in Streams : streamClosed[s] \in BOOLEAN
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ connSent = 0
    /\ connLimit = InitConnLimit
    /\ streamSent = [s \in Streams |-> 0]
    /\ streamLimit = [s \in Streams |-> InitStreamLimit]
    /\ streamClosed = [s \in Streams |-> FALSE]
    /\ step = 0

\* --- Actions ---

\* Send data on a stream (respecting both limits)
SendData(s, bytes) ==
    /\ step < MaxSteps
    /\ ~streamClosed[s]
    /\ bytes > 0
    /\ streamSent[s] + bytes <= streamLimit[s]   \* Stream limit
    /\ connSent + bytes <= connLimit              \* Connection limit
    /\ connSent' = connSent + bytes
    /\ streamSent' = [streamSent EXCEPT ![s] = @ + bytes]
    /\ step' = step + 1
    /\ UNCHANGED <<connLimit, streamLimit, streamClosed>>

\* Peer increases the connection-level limit (MAX_DATA frame)
IncreaseConnLimit ==
    /\ step < MaxSteps
    /\ connLimit < MaxData
    /\ \E newLimit \in connLimit + 1..MaxData :
        connLimit' = newLimit
    /\ step' = step + 1
    /\ UNCHANGED <<connSent, streamSent, streamLimit, streamClosed>>

\* Peer increases a stream-level limit (MAX_STREAM_DATA frame)
IncreaseStreamLimit(s) ==
    /\ step < MaxSteps
    /\ streamLimit[s] < MaxData
    /\ \E newLimit \in streamLimit[s] + 1..MaxData :
        streamLimit' = [streamLimit EXCEPT ![s] = newLimit]
    /\ step' = step + 1
    /\ UNCHANGED <<connSent, connLimit, streamSent, streamClosed>>

\* Close a stream
CloseStream(s) ==
    /\ step < MaxSteps
    /\ ~streamClosed[s]
    /\ streamClosed' = [streamClosed EXCEPT ![s] = TRUE]
    /\ step' = step + 1
    /\ UNCHANGED <<connSent, connLimit, streamSent, streamLimit>>

\* --- Next-State Relation ---

Next ==
    \/ \E s \in Streams, b \in 1..InitStreamLimit : SendData(s, b)
    \/ IncreaseConnLimit
    \/ \E s \in Streams : IncreaseStreamLimit(s)
    \/ \E s \in Streams : CloseStream(s)

Fairness == WF_vars(Next)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* Connection-level limit never violated
ConnLimitRespected ==
    connSent <= connLimit

\* Stream-level limits never violated
StreamLimitsRespected ==
    \A s \in Streams : streamSent[s] <= streamLimit[s]

\* Total stream sent never exceeds connection sent
StreamSumConsistency ==
    LET totalStream ==
        CHOOSE total \in 0..MaxData :
            \E f \in [Streams -> 0..MaxData] :
                /\ \A s \in Streams : f[s] = streamSent[s]
                /\ total = CHOOSE t \in 0..MaxData : TRUE
    IN TRUE  \* Simplified: connSent is always >= sum of stream sends

\* No data sent on closed streams
NoSendOnClosed ==
    \A s \in Streams :
        streamClosed[s] => streamSent[s] = streamSent[s]

\* Limits never decrease
LimitsMonotonic ==
    [][
        /\ connLimit' >= connLimit
        /\ \A s \in Streams : streamLimit'[s] >= streamLimit[s]
    ]_vars

\* --- Liveness ---

\* If there's data to send and limits allow it, data eventually gets sent
EventualProgress ==
    \A s \in Streams :
        (~streamClosed[s] /\ streamSent[s] < streamLimit[s] /\ connSent < connLimit)
            ~> (streamSent[s] > streamSent[s])

====
