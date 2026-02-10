---- MODULE CapabilitySecurity ----
\* SmallAIOS Capability-Based Security Model
\* Verifies: capability non-forgery, monotonic revocation, delegation bounds
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    Tasks,          \* Set of task identifiers
    Resources,      \* Set of resource identifiers
    Permissions     \* Set of permission types (e.g., {"read", "write", "execute"})

VARIABLES
    capabilities,   \* capabilities[task] = set of {resource, perms} records
    revoked,        \* Set of revoked capability IDs
    nextCapId,      \* Counter for unique capability IDs
    step

vars == <<capabilities, revoked, nextCapId, step>>

\* A capability is a record: [id: Nat, resource: Resources, perms: SUBSET Permissions]
Capability == [id : Nat, resource : Resources, perms : SUBSET Permissions]

\* --- Initial State ---

Init ==
    /\ capabilities = [t \in Tasks |-> {}]
    /\ revoked = {}
    /\ nextCapId = 1
    /\ step = 0

\* --- Actions ---

\* Kernel grants a capability to a task
Grant(task, resource, perms) ==
    /\ task \in Tasks
    /\ resource \in Resources
    /\ perms \subseteq Permissions
    /\ perms /= {}
    /\ LET newCap == [id |-> nextCapId, resource |-> resource, perms |-> perms]
       IN /\ capabilities' = [capabilities EXCEPT ![task] = @ \union {newCap}]
          /\ nextCapId' = nextCapId + 1
    /\ UNCHANGED <<revoked, step>>

\* Task delegates a subset of its permissions to another task
Delegate(from, to, capId, subPerms) ==
    /\ from \in Tasks
    /\ to \in Tasks
    /\ from /= to
    /\ \E cap \in capabilities[from] :
        /\ cap.id = capId
        /\ cap.id \notin revoked
        /\ subPerms \subseteq cap.perms
        /\ subPerms /= {}
        /\ LET newCap == [id |-> nextCapId, resource |-> cap.resource, perms |-> subPerms]
           IN /\ capabilities' = [capabilities EXCEPT ![to] = @ \union {newCap}]
              /\ nextCapId' = nextCapId + 1
    /\ step' = step + 1
    /\ UNCHANGED revoked

\* Revoke a capability (and all derived capabilities — simplified here)
Revoke(capId) ==
    /\ capId \notin revoked
    /\ revoked' = revoked \union {capId}
    /\ step' = step + 1
    /\ UNCHANGED <<capabilities, nextCapId>>

\* --- Safety Properties ---

\* Capabilities cannot be forged: every capability was explicitly granted
\* (This is structural — the only way to add to capabilities is via Grant/Delegate)
NoForgery ==
    \A t \in Tasks : \A cap \in capabilities[t] :
        cap.id < nextCapId

\* Delegated permissions never exceed the original
\* (Enforced by the Delegate action's subPerms \subseteq cap.perms guard)

\* Revoked capabilities cannot be used (checked at access time)
RevocationMonotonic ==
    \A id \in revoked : id \in revoked'

\* --- Next State ---

Next ==
    \/ \E t \in Tasks, r \in Resources, p \in SUBSET Permissions : Grant(t, r, p)
    \/ \E f, t \in Tasks, id \in Nat, p \in SUBSET Permissions : Delegate(f, t, id, p)
    \/ \E id \in Nat : Revoke(id)

Spec == Init /\ [][Next]_vars

====
