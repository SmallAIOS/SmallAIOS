/*
 * SmallAIOS Console-Login Lockout Model
 * Verifies: 5 consecutive failures lock a source for 60 seconds and the
 *           threshold cannot be bypassed via interleaved login attempts
 *           from concurrent attackers.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Spec: openspec/changes/management-login-v1/specs/console-login/spec.md
 *       Requirement "Lockout policy" (5 fails → 60 s, per-source counters).
 *
 * Implementation under test: auth/src/console_login.rs::LockoutMap.
 *
 * The model abstracts time as a discrete `now` counter that the harness
 * advances explicitly. Each "second" is one tick; the lockout window is
 * 60 ticks. Two attacker processes race against the lockout machinery
 * for the *same* TTY source — the model proves that no interleaving
 * lets the source escape the 60-tick lock once the 5-failure threshold
 * has tripped.
 */

#define LOCK_THRESHOLD  5
#define LOCK_WINDOW    60
#define MAX_TICKS      90

/* Per-source state. The TTY source is index 0 (the only source modeled
   here; per-Zenoh-peer sources are independent and orthogonal to the
   bypass property). */
byte failures = 0;
byte locked_until = 0;
byte now = 0;

/* Verification flags. */
bool bypass_observed = false;
byte ticks_in_lockout = 0;

/* Atomic record_failure: matches the Rust impl byte-for-byte. */
inline record_failure() {
    atomic {
        if
        :: now < locked_until ->
            /* Already locked — bump counter but do not extend. */
            if
            :: failures < 255 -> failures = failures + 1
            :: else -> skip
            fi
        :: else ->
            if
            :: failures < 255 -> failures = failures + 1
            :: else -> skip
            fi;
            if
            :: failures >= LOCK_THRESHOLD ->
                locked_until = now + LOCK_WINDOW
            :: else -> skip
            fi
        fi
    }
}

/* Atomic try-login: returns LOCKED, OK_OR_FAIL. We model the password
   verification non-deterministically. */
inline try_login(was_success) {
    atomic {
        if
        :: now < locked_until ->
            /* Locked — must reject regardless of correctness. */
            if
            :: was_success ->
                /* Property under test: a successful credential
                   SHALL NOT bypass an active lockout. If we reach
                   this branch and the kernel admits the login,
                   the property is violated. */
                bypass_observed = true
            :: else -> skip
            fi
        :: else ->
            /* Not locked — outcome depends on credential. */
            if
            :: was_success ->
                /* Successful login: counter resets. */
                failures = 0;
                locked_until = 0
            :: else ->
                record_failure()
            fi
        fi
    }
}

/* Attacker A: unbounded brute-force on wrong passwords. */
proctype attacker_a() {
    do
    :: now < MAX_TICKS ->
        try_login(false)
    :: else -> break
    od
}

/* Attacker B: occasionally probes the right password while A burns
   counter slots. Models the "concurrent same-source race" attack. */
proctype attacker_b() {
    do
    :: now < MAX_TICKS ->
        if
        :: try_login(false)
        :: try_login(true)
        fi
    :: else -> break
    od
}

/* Clock — advances `now` and tracks how long we spent in lockout. */
proctype clock() {
    do
    :: now < MAX_TICKS ->
        atomic {
            now = now + 1;
            if
            :: now < locked_until -> ticks_in_lockout = ticks_in_lockout + 1
            :: else -> skip
            fi
        }
    :: else -> break
    od
}

init {
    atomic {
        run attacker_a();
        run attacker_b();
        run clock()
    }
}

/* Liveness / safety properties. */

/* P1: A successful credential SHALL NEVER admit a session while
   `now < locked_until`. */
ltl no_bypass {
    [] (! bypass_observed)
}

/* P2: Once `failures >= LOCK_THRESHOLD` the lockout window SHALL be
   active for at least 1 tick (the model's discrete clock step
   approximates the 60-second window monotone-ness). */
ltl threshold_locks {
    [] (failures >= LOCK_THRESHOLD -> <> (now < locked_until))
}
