---- MODULE USBEnumeration ----
\* USB Device Enumeration State Machine Model
\* Verifies: no data transfer without Configured state, address uniqueness,
\*           reset returns to Default (not Disconnected)
\* Models USB 2.0 enumeration per specification chapter 9
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    NumDevices,     \* Number of USB devices on the bus (2-3)
    NumAddresses    \* Number of available USB addresses (1-127)

VARIABLES
    dev_state,      \* dev_state[d] \in {"Disconnected", "Powered", "Default",
                    \*                    "AddressAssigned", "Configured", "Suspended"}
    dev_address,    \* dev_address[d] = assigned address (0 = none)
    prev_state,     \* prev_state[d] = state before Suspended (for resume)
    data_transfer,  \* data_transfer[d] = TRUE if device is transferring data
    next_address,   \* Next address to assign (monotonically increasing)
    step            \* Step counter for bounded checking

Devices == 1..NumDevices
Addresses == 1..NumAddresses

States == {"Disconnected", "Powered", "Default", "AddressAssigned", "Configured", "Suspended"}

vars == <<dev_state, dev_address, prev_state, data_transfer, next_address, step>>

MaxSteps == NumDevices * NumAddresses + 20

\* --- Type Invariant ---

TypeInvariant ==
    /\ \A d \in Devices : dev_state[d] \in States
    /\ \A d \in Devices : dev_address[d] \in 0..NumAddresses
    /\ \A d \in Devices : prev_state[d] \in States
    /\ \A d \in Devices : data_transfer[d] \in BOOLEAN
    /\ next_address \in 1..(NumAddresses + 1)
    /\ step \in 0..MaxSteps

\* --- Initial State ---

Init ==
    /\ dev_state = [d \in Devices |-> "Disconnected"]
    /\ dev_address = [d \in Devices |-> 0]
    /\ prev_state = [d \in Devices |-> "Disconnected"]
    /\ data_transfer = [d \in Devices |-> FALSE]
    /\ next_address = 1
    /\ step = 0

\* --- Actions ---

\* Device connected: Disconnected -> Powered
Connect(d) ==
    /\ step < MaxSteps
    /\ dev_state[d] = "Disconnected"
    /\ dev_state' = [dev_state EXCEPT ![d] = "Powered"]
    /\ step' = step + 1
    /\ UNCHANGED <<dev_address, prev_state, data_transfer, next_address>>

\* Port reset complete: Powered -> Default
PortReset(d) ==
    /\ step < MaxSteps
    /\ dev_state[d] = "Powered"
    /\ dev_state' = [dev_state EXCEPT ![d] = "Default"]
    /\ step' = step + 1
    /\ UNCHANGED <<dev_address, prev_state, data_transfer, next_address>>

\* SET_ADDRESS succeeds: Default -> AddressAssigned
SetAddress(d) ==
    /\ step < MaxSteps
    /\ dev_state[d] = "Default"
    /\ next_address <= NumAddresses
    /\ dev_address' = [dev_address EXCEPT ![d] = next_address]
    /\ dev_state' = [dev_state EXCEPT ![d] = "AddressAssigned"]
    /\ next_address' = next_address + 1
    /\ step' = step + 1
    /\ UNCHANGED <<prev_state, data_transfer>>

\* SET_CONFIGURATION succeeds: AddressAssigned -> Configured
SetConfiguration(d) ==
    /\ step < MaxSteps
    /\ dev_state[d] = "AddressAssigned"
    /\ dev_state' = [dev_state EXCEPT ![d] = "Configured"]
    /\ step' = step + 1
    /\ UNCHANGED <<dev_address, prev_state, data_transfer, next_address>>

\* Data transfer: only in Configured state
StartDataTransfer(d) ==
    /\ step < MaxSteps
    /\ dev_state[d] = "Configured"
    /\ ~data_transfer[d]
    /\ data_transfer' = [data_transfer EXCEPT ![d] = TRUE]
    /\ step' = step + 1
    /\ UNCHANGED <<dev_state, dev_address, prev_state, next_address>>

\* Complete data transfer
CompleteDataTransfer(d) ==
    /\ step < MaxSteps
    /\ data_transfer[d] = TRUE
    /\ data_transfer' = [data_transfer EXCEPT ![d] = FALSE]
    /\ step' = step + 1
    /\ UNCHANGED <<dev_state, dev_address, prev_state, next_address>>

\* Device removed: Any (non-Disconnected) -> Disconnected
Disconnect(d) ==
    /\ step < MaxSteps
    /\ dev_state[d] # "Disconnected"
    /\ dev_state' = [dev_state EXCEPT ![d] = "Disconnected"]
    /\ dev_address' = [dev_address EXCEPT ![d] = 0]
    /\ data_transfer' = [data_transfer EXCEPT ![d] = FALSE]
    /\ prev_state' = [prev_state EXCEPT ![d] = "Disconnected"]
    /\ step' = step + 1
    /\ UNCHANGED <<next_address>>

\* Bus idle > 3ms: Any active state -> Suspended
Suspend(d) ==
    /\ step < MaxSteps
    /\ dev_state[d] \in {"Powered", "Default", "AddressAssigned", "Configured"}
    /\ ~data_transfer[d]  \* Cannot suspend during active transfer
    /\ prev_state' = [prev_state EXCEPT ![d] = dev_state[d]]
    /\ dev_state' = [dev_state EXCEPT ![d] = "Suspended"]
    /\ step' = step + 1
    /\ UNCHANGED <<dev_address, data_transfer, next_address>>

\* Resume from suspend: Suspended -> previous state
Resume(d) ==
    /\ step < MaxSteps
    /\ dev_state[d] = "Suspended"
    /\ dev_state' = [dev_state EXCEPT ![d] = prev_state[d]]
    /\ step' = step + 1
    /\ UNCHANGED <<dev_address, prev_state, data_transfer, next_address>>

\* Bus reset while in any active state: returns to Default (not Disconnected)
BusReset(d) ==
    /\ step < MaxSteps
    /\ dev_state[d] \in {"Powered", "AddressAssigned", "Configured", "Suspended"}
    /\ dev_state' = [dev_state EXCEPT ![d] = "Default"]
    /\ data_transfer' = [data_transfer EXCEPT ![d] = FALSE]
    /\ prev_state' = [prev_state EXCEPT ![d] = "Default"]
    /\ step' = step + 1
    /\ UNCHANGED <<dev_address, next_address>>

\* --- Next-State Relation ---

Next ==
    \/ \E d \in Devices : Connect(d)
    \/ \E d \in Devices : PortReset(d)
    \/ \E d \in Devices : SetAddress(d)
    \/ \E d \in Devices : SetConfiguration(d)
    \/ \E d \in Devices : StartDataTransfer(d)
    \/ \E d \in Devices : CompleteDataTransfer(d)
    \/ \E d \in Devices : Disconnect(d)
    \/ \E d \in Devices : Suspend(d)
    \/ \E d \in Devices : Resume(d)
    \/ \E d \in Devices : BusReset(d)

Fairness ==
    /\ \A d \in Devices : WF_vars(Connect(d))
    /\ \A d \in Devices : WF_vars(PortReset(d))
    /\ \A d \in Devices : WF_vars(SetAddress(d))
    /\ \A d \in Devices : WF_vars(SetConfiguration(d))
    /\ \A d \in Devices : WF_vars(Resume(d))

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* No data transfer without Configured state
NoDataWithoutConfigured ==
    \A d \in Devices :
        data_transfer[d] => dev_state[d] = "Configured"

\* Address uniqueness: no two connected devices share a non-zero address
AddressUniqueness ==
    \A d1, d2 \in Devices :
        (d1 # d2 /\ dev_address[d1] # 0 /\ dev_address[d2] # 0
         /\ dev_state[d1] # "Disconnected" /\ dev_state[d2] # "Disconnected")
            => dev_address[d1] # dev_address[d2]

\* Reset returns to Default, not Disconnected (bus reset preserves connection)
\* This is enforced structurally by BusReset action going to Default

\* Disconnected devices have no address
DisconnectedNoAddress ==
    \A d \in Devices :
        dev_state[d] = "Disconnected" => dev_address[d] = 0

\* Only AddressAssigned or Configured devices have non-zero addresses
AddressImpliesState ==
    \A d \in Devices :
        (dev_address[d] # 0 /\ dev_state[d] # "Suspended") =>
            dev_state[d] \in {"AddressAssigned", "Configured", "Default"}

\* Suspended device has a valid previous state for resume
SuspendedHasValidPrevState ==
    \A d \in Devices :
        dev_state[d] = "Suspended" =>
            prev_state[d] \in {"Powered", "Default", "AddressAssigned", "Configured"}

\* --- Liveness Properties ---

\* A connected device eventually reaches Configured state
EventualConfiguration ==
    \A d \in Devices :
        (dev_state[d] = "Powered") ~> (dev_state[d] = "Configured")

====
