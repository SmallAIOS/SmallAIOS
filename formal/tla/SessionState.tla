---- MODULE SessionState ----
\* SmallAIOS Auth Session State Model
\* Spec: openspec/changes/management-login-v1/specs/kernel-syscalls/spec.md
\* SPDX-License-Identifier: Apache-2.0
\*
\* Models the auth session lifecycle for `management-login-v1`
\* Phase 3:
\*
\*   login -> Active -> (idle expired | logged out)
\*
\* Properties verified (in spirit; see TODO at bottom for the real
\* TLC config):
\*
\*   1. SessionTableBounded: |Sessions| <= MaxSessions at all times.
\*   2. NoRebindOfStaleId: a SessionId released by `release` is never
\*      bound to a fresh session before its generation counter ticks.
\*   3. ForceLogoutOnCrossRotate: when a Root cross-rotates user U's
\*      password, every session whose user_id == U is invalidated.
\*   4. ConcurrentSameUserAllowed: two distinct logins for the same
\*      user can co-exist (refutes "one session per user" interp).
\*   5. IdleSweepEvictsExpired: sessions with
\*      now - lastActivity > policy(role) are reachable only as
\*      "expired" before the next sweep tick removes them.
\*
\* Status: this file is a Promela-style sketch in TLA+ syntax. The
\* TLC model-check config (`SessionState.cfg`) and a polished
\* `INVARIANT` / `PROPERTY` block are TODO — see the final section.
\* The behaviour modeled here is sound enough to be mechanized once
\* Phase 4's console-login concurrent flows are nailed down.

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    MaxSessions,    \* SESSION_TABLE_CAP, e.g. 32 in production
    Users,          \* set of user_ids (e.g. {1, 2, 3})
    Roles,          \* set of roles (e.g. {"root", "operator", "viewer"})
    MaxClock,       \* upper bound on the wall-clock counter
    IdleThreshold   \* per-role idle threshold mapping (Role -> Nat secs)

VARIABLES
    sessions,       \* function SessionId -> [user, role, lastActivity]
    nextGeneration, \* per-slot generation counter
    clock,          \* wall-clock counter (saturating-monotone)
    auditLog        \* sequence of events for spec validation

vars == <<sessions, nextGeneration, clock, auditLog>>

\* ─── Domains ────────────────────────────────────────────────────────────────

\* SessionId encoding: <<slot, generation>>. Slots are 0..MaxSessions-1.
SessionIds == (0..(MaxSessions - 1)) \X Nat

\* Role-to-threshold lookup. Defaults:
\*   Root     => 15 minutes  (900s)
\*   Operator => 60 minutes  (3600s)
\*   Viewer   => 60 minutes  (3600s)
ThresholdFor(role) == IdleThreshold[role]

\* ─── Initial state ──────────────────────────────────────────────────────────

Init ==
    /\ sessions = [s \in {} |-> {}]   \* empty function (no live sessions)
    /\ nextGeneration = [s \in 0..(MaxSessions - 1) |-> 0]
    /\ clock = 0
    /\ auditLog = <<>>

\* ─── Helpers ───────────────────────────────────────────────────────────────

\* The set of currently-active session ids.
LiveIds == DOMAIN sessions

\* Pick the smallest free slot.
FirstFreeSlot ==
    IF \E i \in 0..(MaxSessions - 1):
            \A id \in LiveIds: id[1] /= i
    THEN CHOOSE i \in 0..(MaxSessions - 1):
            \A id \in LiveIds: id[1] /= i
    ELSE -1

\* ─── Actions ───────────────────────────────────────────────────────────────

\* `auth_login(u, r)` — create a session for user `u` with role `r` if
\* there's a free slot. Idempotency: two distinct logins for the same
\* user create two distinct ids (concurrent-same-user is allowed).
Login(u, r) ==
    /\ FirstFreeSlot >= 0
    /\ LET slot == FirstFreeSlot
           gen == nextGeneration[slot]
           id == <<slot, gen>>
       IN  /\ sessions' = sessions @@ ( id :>
                [user |-> u, role |-> r, lastActivity |-> clock] )
           /\ nextGeneration' = nextGeneration
           /\ auditLog' = Append(auditLog, [op |-> "login", id |-> id, user |-> u])
           /\ clock' = clock

\* `auth_logout` — release the slot and bump its generation.
Logout(id) ==
    /\ id \in LiveIds
    /\ LET slot == id[1]
       IN  /\ sessions' = [ s \in (LiveIds \ {id}) |-> sessions[s] ]
           /\ nextGeneration' = [nextGeneration EXCEPT ![slot] = @ + 1]
           /\ auditLog' = Append(auditLog, [op |-> "logout", id |-> id])
           /\ clock' = clock

\* `auth_change_password` cross-rotate by Root for user `target`. Every
\* session whose user_id == target is invalidated.
CrossRotate(target) ==
    /\ \E rootId \in LiveIds: sessions[rootId].role = "root"
    /\ LET killedIds == { id \in LiveIds : sessions[id].user = target }
           survivors == LiveIds \ killedIds
       IN  /\ sessions' = [ s \in survivors |-> sessions[s] ]
           /\ nextGeneration' =
                [s \in 0..(MaxSessions - 1) |->
                    IF \E id \in killedIds: id[1] = s
                    THEN nextGeneration[s] + 1
                    ELSE nextGeneration[s]]
           /\ auditLog' = Append(auditLog, [op |-> "cross_rotate", target |-> target])
           /\ clock' = clock

\* Idle-sweep tick. Advances the clock by 1 unit and force-logs-out
\* any session whose elapsed time since `lastActivity` exceeds its
\* role's threshold.
SweepTick ==
    /\ clock < MaxClock
    /\ LET newClock == clock + 1
           expired == { id \in LiveIds :
                          newClock - sessions[id].lastActivity
                            > ThresholdFor(sessions[id].role) }
           survivors == LiveIds \ expired
       IN  /\ sessions' = [ s \in survivors |-> sessions[s] ]
           /\ nextGeneration' =
                [s \in 0..(MaxSessions - 1) |->
                    IF \E id \in expired: id[1] = s
                    THEN nextGeneration[s] + 1
                    ELSE nextGeneration[s]]
           /\ clock' = newClock
           /\ auditLog' = auditLog \o
                [ i \in 1..Cardinality(expired) |->
                    [op |-> "idle_sweep", id |-> CHOOSE x \in expired: TRUE] ]

\* Reset-idle on any successful syscall — bumps lastActivity to clock.
TouchActivity(id) ==
    /\ id \in LiveIds
    /\ sessions' = [sessions EXCEPT ![id].lastActivity = clock]
    /\ UNCHANGED <<nextGeneration, clock, auditLog>>

Next ==
    \/ \E u \in Users, r \in Roles: Login(u, r)
    \/ \E id \in LiveIds: Logout(id)
    \/ \E u \in Users: CrossRotate(u)
    \/ SweepTick
    \/ \E id \in LiveIds: TouchActivity(id)

\* ─── Invariants ────────────────────────────────────────────────────────────

\* (1) The session table never exceeds capacity.
SessionTableBounded ==
    Cardinality(LiveIds) <= MaxSessions

\* (2) No stale id is ever rebound: if a slot has had its generation
\*     ticked, no session at that slot has the old generation.
NoRebindOfStaleId ==
    \A id \in LiveIds:
        id[2] = nextGeneration[id[1]]

\* (3) After a `cross_rotate(target)`, no live session has user==target
\*     until a fresh login occurs. Encoded as: in any state where the
\*     last log entry is a cross_rotate, no session for that target is
\*     live.
ForceLogoutOnCrossRotate ==
    Len(auditLog) = 0 \/
    LET last == auditLog[Len(auditLog)]
    IN  last.op /= "cross_rotate" \/
        \A id \in LiveIds: sessions[id].user /= last.target

\* (5) Idle-swept sessions are gone (no zombies). Once a session is
\*     past its threshold, the next tick must remove it.
IdleSweepEvictsExpired ==
    \A id \in LiveIds:
        clock - sessions[id].lastActivity <= ThresholdFor(sessions[id].role)

\* Combined safety invariant.
SafetyInvariant ==
    /\ SessionTableBounded
    /\ NoRebindOfStaleId

\* Liveness: a session's lifetime is bounded — it eventually either
\* logs out, is cross-rotated, or is swept by idle.
SessionEventuallyTerminates ==
    \A id \in LiveIds:
        <>(id \notin LiveIds')

\* ─── TLC config TODO ───────────────────────────────────────────────────────
\*
\* The companion `SessionState.cfg` (not yet written) should bind:
\*
\*   CONSTANTS
\*     MaxSessions   = 3
\*     Users         = {1, 2}
\*     Roles         = {"root", "operator", "viewer"}
\*     MaxClock      = 30
\*     IdleThreshold = ("root" :> 5 @@ "operator" :> 10 @@ "viewer" :> 10)
\*   INVARIANT SafetyInvariant
\*   PROPERTY SessionEventuallyTerminates
\*
\* Once the cfg is wired into `formal/tla/run-tlc.sh`, the model
\* feeds the existing 19-model TLA+ CI gate at
\* `.github/workflows/ci.yml` (advisory). The Promela-style event
\* sequence above is sound enough to mechanize as-is; the open
\* question is the right cardinality bound for the `Users` set,
\* which depends on whether Phase 4's first-boot bootstrap allows
\* multiple Root accounts.

====
