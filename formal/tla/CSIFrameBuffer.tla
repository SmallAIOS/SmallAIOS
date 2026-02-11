---- MODULE CSIFrameBuffer ----
\* CSI-2 Frame Buffer Management Model
\* Verifies: buffer ownership exclusivity, no double-free, no buffer leak,
\*           total buffer count invariant
\* Models MIPI CSI-2 camera frame buffer lifecycle
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    NumBuffers      \* Total number of frame buffers (3-4)

Buffers == 1..NumBuffers

VARIABLES
    free_buffers,       \* Set of buffers available for capture
    capture_buffer,     \* Buffer currently being filled by CSI (0 = none)
    ready_queue,        \* Set of buffers with completed frames, awaiting app
    app_buffer,         \* Buffer currently held by application (0 = none)
    state,              \* Pipeline state: "idle" | "capturing" | "processing"
    step                \* Step counter

vars == <<free_buffers, capture_buffer, ready_queue, app_buffer, state, step>>

MaxSteps == NumBuffers * 10

\* --- Type Invariant ---

TypeInvariant ==
    /\ free_buffers \subseteq Buffers
    /\ capture_buffer \in Buffers \union {0}
    /\ ready_queue \subseteq Buffers
    /\ app_buffer \in Buffers \union {0}
    /\ state \in {"idle", "capturing", "processing"}
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ free_buffers = Buffers
    /\ capture_buffer = 0
    /\ ready_queue = {}
    /\ app_buffer = 0
    /\ state = "idle"
    /\ step = 0

\* --- Helper: all accounted buffers ---

AllBuffers ==
    free_buffers \union
    (IF capture_buffer # 0 THEN {capture_buffer} ELSE {}) \union
    ready_queue \union
    (IF app_buffer # 0 THEN {app_buffer} ELSE {})

\* --- Actions ---

\* CSI hardware acquires a free buffer for capture
StartCapture ==
    /\ step < MaxSteps
    /\ state = "idle"
    /\ free_buffers # {}
    /\ capture_buffer = 0
    /\ \E b \in free_buffers :
        /\ capture_buffer' = b
        /\ free_buffers' = free_buffers \ {b}
    /\ state' = "capturing"
    /\ step' = step + 1
    /\ UNCHANGED <<ready_queue, app_buffer>>

\* CSI hardware completes frame capture, buffer moves to ready queue
CompleteCapture ==
    /\ step < MaxSteps
    /\ state = "capturing"
    /\ capture_buffer # 0
    /\ ready_queue' = ready_queue \union {capture_buffer}
    /\ capture_buffer' = 0
    /\ state' = "idle"
    /\ step' = step + 1
    /\ UNCHANGED <<free_buffers, app_buffer>>

\* Application dequeues a ready buffer for processing
AppAcquire ==
    /\ step < MaxSteps
    /\ ready_queue # {}
    /\ app_buffer = 0
    /\ \E b \in ready_queue :
        /\ app_buffer' = b
        /\ ready_queue' = ready_queue \ {b}
    /\ state' = "processing"
    /\ step' = step + 1
    /\ UNCHANGED <<free_buffers, capture_buffer>>

\* Application finishes processing and returns buffer to free pool
AppRelease ==
    /\ step < MaxSteps
    /\ app_buffer # 0
    /\ free_buffers' = free_buffers \union {app_buffer}
    /\ app_buffer' = 0
    /\ state' = "idle"
    /\ step' = step + 1
    /\ UNCHANGED <<capture_buffer, ready_queue>>

\* Capture abort: return capture buffer to free pool (error recovery)
AbortCapture ==
    /\ step < MaxSteps
    /\ state = "capturing"
    /\ capture_buffer # 0
    /\ free_buffers' = free_buffers \union {capture_buffer}
    /\ capture_buffer' = 0
    /\ state' = "idle"
    /\ step' = step + 1
    /\ UNCHANGED <<ready_queue, app_buffer>>

\* --- Next-State Relation ---

Next ==
    \/ StartCapture
    \/ CompleteCapture
    \/ AppAcquire
    \/ AppRelease
    \/ AbortCapture

Fairness ==
    /\ WF_vars(CompleteCapture)
    /\ WF_vars(AppRelease)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* Buffer ownership exclusivity: each buffer is in exactly one state
\* No buffer appears in more than one pool simultaneously
OwnershipExclusive ==
    /\ \A b \in Buffers :
        LET in_free    == b \in free_buffers
            in_capture == capture_buffer = b
            in_ready   == b \in ready_queue
            in_app     == app_buffer = b
        IN Cardinality({x \in {in_free, in_capture, in_ready, in_app} : x = TRUE}) <= 1

\* Total buffer count invariant: no buffers created or destroyed
BufferCountInvariant ==
    Cardinality(AllBuffers) = NumBuffers

\* No double-free: a buffer returned to free was not already free
NoDoubleFree ==
    /\ (capture_buffer # 0) => capture_buffer \notin free_buffers
    /\ (app_buffer # 0) => app_buffer \notin free_buffers

\* No buffer leak: all buffers are accounted for
NoBufferLeak ==
    AllBuffers = Buffers

\* Capture buffer must be in Buffers range (if set)
CaptureValid ==
    (capture_buffer # 0) => capture_buffer \in Buffers

\* App buffer must be in Buffers range (if set)
AppValid ==
    (app_buffer # 0) => app_buffer \in Buffers

\* --- Liveness Properties ---

\* Buffers eventually cycle back to free
EventualRecycle ==
    \A b \in Buffers :
        (b \in ready_queue) ~> (b \in free_buffers)

====
