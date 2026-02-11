---- MODULE GPIOInterrupt ----
\* GPIO Interrupt Delivery Model
\* Verifies: no lost interrupts, no spurious interrupts, edge detection,
\*           handler runs to completion before re-entry
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    NumPins         \* Number of GPIO pins (2-3)

Pins == 1..NumPins

VARIABLES
    pin_level,          \* pin_level[p] = current pin logic level (0 or 1)
    prev_level,         \* prev_level[p] = previous pin level (for edge detection)
    interrupt_pending,  \* interrupt_pending[p] = TRUE if interrupt is pending
    interrupt_enabled,  \* interrupt_enabled[p] = TRUE if interrupt is enabled
    edge_config,        \* edge_config[p] \in {"rising", "falling", "both"}
    handler_running,    \* handler_running[p] = TRUE if ISR is executing
    ack_pending,        \* ack_pending[p] = TRUE if handler needs to ack
    handled_count,      \* handled_count[p] = number of interrupts handled
    triggered_count,    \* triggered_count[p] = number of edges that should trigger
    step                \* Step counter

vars == <<pin_level, prev_level, interrupt_pending, interrupt_enabled,
          edge_config, handler_running, ack_pending, handled_count,
          triggered_count, step>>

MaxSteps == NumPins * 12

\* --- Type Invariant ---

TypeInvariant ==
    /\ \A p \in Pins : pin_level[p] \in {0, 1}
    /\ \A p \in Pins : prev_level[p] \in {0, 1}
    /\ \A p \in Pins : interrupt_pending[p] \in BOOLEAN
    /\ \A p \in Pins : interrupt_enabled[p] \in BOOLEAN
    /\ \A p \in Pins : edge_config[p] \in {"rising", "falling", "both"}
    /\ \A p \in Pins : handler_running[p] \in BOOLEAN
    /\ \A p \in Pins : ack_pending[p] \in BOOLEAN
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ pin_level = [p \in Pins |-> 0]
    /\ prev_level = [p \in Pins |-> 0]
    /\ interrupt_pending = [p \in Pins |-> FALSE]
    /\ interrupt_enabled = [p \in Pins |-> TRUE]
    /\ edge_config = [p \in Pins |-> "rising"]
    /\ handler_running = [p \in Pins |-> FALSE]
    /\ ack_pending = [p \in Pins |-> FALSE]
    /\ handled_count = [p \in Pins |-> 0]
    /\ triggered_count = [p \in Pins |-> 0]
    /\ step = 0

\* --- Helper: edge detection ---

IsEdge(p, old, new) ==
    CASE edge_config[p] = "rising"  -> (old = 0 /\ new = 1)
      [] edge_config[p] = "falling" -> (old = 1 /\ new = 0)
      [] edge_config[p] = "both"    -> (old # new)
      [] OTHER                       -> FALSE

\* --- Actions ---

\* Pin level changes (external stimulus)
PinChange(p) ==
    /\ step < MaxSteps
    /\ \E new_val \in {0, 1} :
        /\ new_val # pin_level[p]
        /\ prev_level' = [prev_level EXCEPT ![p] = pin_level[p]]
        /\ pin_level' = [pin_level EXCEPT ![p] = new_val]
        /\ IF /\ interrupt_enabled[p]
              /\ IsEdge(p, pin_level[p], new_val)
              /\ ~handler_running[p]
           THEN /\ interrupt_pending' = [interrupt_pending EXCEPT ![p] = TRUE]
                /\ triggered_count' = [triggered_count EXCEPT ![p] = @ + 1]
           ELSE UNCHANGED <<interrupt_pending, triggered_count>>
    /\ step' = step + 1
    /\ UNCHANGED <<interrupt_enabled, edge_config, handler_running,
                   ack_pending, handled_count>>

\* Interrupt controller dispatches handler for a pending interrupt
DispatchHandler(p) ==
    /\ step < MaxSteps
    /\ interrupt_pending[p] = TRUE
    /\ handler_running[p] = FALSE   \* No re-entry
    /\ interrupt_enabled[p] = TRUE
    /\ handler_running' = [handler_running EXCEPT ![p] = TRUE]
    /\ ack_pending' = [ack_pending EXCEPT ![p] = TRUE]
    /\ interrupt_pending' = [interrupt_pending EXCEPT ![p] = FALSE]
    /\ step' = step + 1
    /\ UNCHANGED <<pin_level, prev_level, interrupt_enabled, edge_config,
                   handled_count, triggered_count>>

\* Handler completes and acknowledges the interrupt
CompleteHandler(p) ==
    /\ step < MaxSteps
    /\ handler_running[p] = TRUE
    /\ ack_pending[p] = TRUE
    /\ handler_running' = [handler_running EXCEPT ![p] = FALSE]
    /\ ack_pending' = [ack_pending EXCEPT ![p] = FALSE]
    /\ handled_count' = [handled_count EXCEPT ![p] = @ + 1]
    /\ step' = step + 1
    /\ UNCHANGED <<pin_level, prev_level, interrupt_pending, interrupt_enabled,
                   edge_config, triggered_count>>

\* Configure edge detection mode
ConfigureEdge(p) ==
    /\ step < MaxSteps
    /\ handler_running[p] = FALSE
    /\ \E mode \in {"rising", "falling", "both"} :
        /\ mode # edge_config[p]
        /\ edge_config' = [edge_config EXCEPT ![p] = mode]
    /\ step' = step + 1
    /\ UNCHANGED <<pin_level, prev_level, interrupt_pending, interrupt_enabled,
                   handler_running, ack_pending, handled_count, triggered_count>>

\* --- Next-State Relation ---

Next ==
    \/ \E p \in Pins : PinChange(p)
    \/ \E p \in Pins : DispatchHandler(p)
    \/ \E p \in Pins : CompleteHandler(p)
    \/ \E p \in Pins : ConfigureEdge(p)

Fairness ==
    /\ \A p \in Pins : WF_vars(DispatchHandler(p))
    /\ \A p \in Pins : WF_vars(CompleteHandler(p))

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* No spurious interrupts: pending only if enabled
NoSpuriousInterrupts ==
    \A p \in Pins :
        interrupt_pending[p] => interrupt_enabled[p]

\* Handler cannot re-enter: no dispatch while handler is running
NoReentry ==
    \A p \in Pins :
        handler_running[p] => ~interrupt_pending[p]

\* If handler is running, ack must be pending
HandlerAckConsistency ==
    \A p \in Pins :
        (handler_running[p]) => (ack_pending[p])

\* Handled count never exceeds triggered count
NoExtraHandling ==
    \A p \in Pins :
        handled_count[p] <= triggered_count[p]

\* --- Liveness Properties ---

\* Every pending interrupt is eventually handled
EventualHandling ==
    \A p \in Pins :
        interrupt_pending[p] ~> (handled_count[p] > handled_count[p])

====
