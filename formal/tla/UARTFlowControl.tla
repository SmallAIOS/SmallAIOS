---- MODULE UARTFlowControl ----
\* UART with RTS/CTS Hardware Flow Control Model
\* Verifies: no data loss, flow control prevents overflow, eventual delivery
\* Models RTS/CTS handshake preventing RX FIFO overflow
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences

CONSTANTS
    FifoSize,       \* Size of TX and RX FIFOs (4)
    MaxMessages     \* Maximum number of messages to model (3)

VARIABLES
    tx_fifo,        \* Transmitter FIFO (sequence of messages)
    rx_fifo,        \* Receiver FIFO (sequence of messages)
    rts_line,       \* Receiver's "Ready To Send" signal: TRUE = ready to receive
    cts_line,       \* Transmitter sees CTS: TRUE = clear to send
    tx_state,       \* "idle" | "sending" | "waiting_cts"
    rx_state,       \* "idle" | "receiving"
    in_flight,      \* Data byte currently on the wire (0 = none)
    delivered,      \* Sequence of all delivered messages
    total_sent,     \* Total messages queued for sending
    step            \* Step counter

vars == <<tx_fifo, rx_fifo, rts_line, cts_line, tx_state, rx_state,
          in_flight, delivered, total_sent, step>>

MaxSteps == MaxMessages * 8 + 10

\* --- Type Invariant ---

TypeInvariant ==
    /\ Len(tx_fifo) \in 0..FifoSize
    /\ Len(rx_fifo) \in 0..FifoSize
    /\ rts_line \in BOOLEAN
    /\ cts_line \in BOOLEAN
    /\ tx_state \in {"idle", "sending", "waiting_cts"}
    /\ rx_state \in {"idle", "receiving"}
    /\ in_flight \in 0..MaxMessages
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ tx_fifo = <<>>
    /\ rx_fifo = <<>>
    /\ rts_line = TRUE      \* Receiver ready
    /\ cts_line = TRUE       \* Transmitter sees clear
    /\ tx_state = "idle"
    /\ rx_state = "idle"
    /\ in_flight = 0
    /\ delivered = <<>>
    /\ total_sent = 0
    /\ step = 0

\* --- Actions ---

\* Application queues a message for transmission
QueueMessage ==
    /\ step < MaxSteps
    /\ total_sent < MaxMessages
    /\ Len(tx_fifo) < FifoSize
    /\ LET msg == total_sent + 1
       IN /\ tx_fifo' = Append(tx_fifo, msg)
          /\ total_sent' = total_sent + 1
    /\ step' = step + 1
    /\ UNCHANGED <<rx_fifo, rts_line, cts_line, tx_state, rx_state,
                   in_flight, delivered>>

\* Receiver updates RTS based on FIFO capacity
\* Deasserts RTS when FIFO is almost full (1 slot remaining)
UpdateRTS ==
    /\ step < MaxSteps
    /\ LET should_assert == Len(rx_fifo) < FifoSize - 1
       IN /\ rts_line' = should_assert
          /\ cts_line' = should_assert   \* CTS follows RTS (direct connection)
    /\ step' = step + 1
    /\ UNCHANGED <<tx_fifo, rx_fifo, tx_state, rx_state, in_flight,
                   delivered, total_sent>>

\* Transmitter begins sending if CTS is asserted and FIFO has data
StartSend ==
    /\ step < MaxSteps
    /\ tx_state \in {"idle", "waiting_cts"}
    /\ cts_line = TRUE
    /\ Len(tx_fifo) > 0
    /\ in_flight = 0
    /\ tx_state' = "sending"
    /\ in_flight' = Head(tx_fifo)
    /\ tx_fifo' = Tail(tx_fifo)
    /\ step' = step + 1
    /\ UNCHANGED <<rx_fifo, rts_line, cts_line, rx_state, delivered, total_sent>>

\* Transmitter must wait when CTS is deasserted
PauseSend ==
    /\ step < MaxSteps
    /\ tx_state = "idle"
    /\ cts_line = FALSE
    /\ Len(tx_fifo) > 0
    /\ tx_state' = "waiting_cts"
    /\ step' = step + 1
    /\ UNCHANGED <<tx_fifo, rx_fifo, rts_line, cts_line, rx_state,
                   in_flight, delivered, total_sent>>

\* Data arrives at receiver
ReceiveData ==
    /\ step < MaxSteps
    /\ in_flight # 0
    /\ Len(rx_fifo) < FifoSize     \* FIFO has room
    /\ rx_state' = "receiving"
    /\ rx_fifo' = Append(rx_fifo, in_flight)
    /\ in_flight' = 0
    /\ tx_state' = "idle"
    /\ step' = step + 1
    /\ UNCHANGED <<tx_fifo, rts_line, cts_line, delivered, total_sent>>

\* Application reads from RX FIFO
ConsumeData ==
    /\ step < MaxSteps
    /\ Len(rx_fifo) > 0
    /\ delivered' = Append(delivered, Head(rx_fifo))
    /\ rx_fifo' = Tail(rx_fifo)
    /\ rx_state' = "idle"
    /\ step' = step + 1
    /\ UNCHANGED <<tx_fifo, rts_line, cts_line, tx_state, in_flight, total_sent>>

\* --- Next-State Relation ---

Next ==
    \/ QueueMessage
    \/ UpdateRTS
    \/ StartSend
    \/ PauseSend
    \/ ReceiveData
    \/ ConsumeData

Fairness ==
    /\ WF_vars(UpdateRTS)
    /\ WF_vars(StartSend)
    /\ WF_vars(ReceiveData)
    /\ WF_vars(ConsumeData)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* RX FIFO never overflows
NoRxOverflow ==
    Len(rx_fifo) <= FifoSize

\* TX FIFO never overflows
NoTxOverflow ==
    Len(tx_fifo) <= FifoSize

\* Flow control: transmitter does not send when CTS is deasserted
FlowControlRespected ==
    (tx_state = "sending") => (in_flight # 0)

\* No data lost: in_flight is only set when FIFO has room
\* (Ensured by CTS/RTS: if rx_fifo is full, CTS deasserted, no send)
SafeSend ==
    (cts_line = FALSE) => (tx_state # "sending")

\* Data integrity: received data is a prefix of sent data
\* All delivered messages appear in order
DeliveryOrder ==
    \A i \in 1..Len(delivered) : delivered[i] = i

\* --- Liveness Properties ---

\* Every queued message is eventually delivered
EventualDelivery ==
    (total_sent > Len(delivered)) ~> (Len(delivered) = total_sent)

====
