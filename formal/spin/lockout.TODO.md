# SPIN lockout model — verification TODO

The Promela model in `lockout.pml` proves that the
`management-login-v1` Phase 4 lockout policy (5 fails → 60 s) cannot be
bypassed by an interleaved login attempt from a concurrent attacker on
the same source.

## Run locally

```bash
spin -a lockout.pml
gcc -O2 -DSAFETY -o pan pan.c
./pan -m1000000 -a -f -N no_bypass
./pan -m1000000 -a -f -N threshold_locks
```

`spin` is not in the SmallAIOS dev container; the model is checked into
the repo so it can be verified ad-hoc on a developer workstation. CI
integration sits behind the existing `just spin-verify` recipe in
`justfile` and runs through the Promela models when SPIN is installed.

## Properties under test

- `no_bypass` — a successful credential SHALL NEVER admit a session
  while `now < locked_until`.
- `threshold_locks` — once `failures >= 5`, the lockout window is
  active.

Both properties are derived from the spec's "Lockout policy" requirement
and the inline Q20 wording.

## Mapping to Rust

| Promela inline   | Rust function                                                  |
|------------------|----------------------------------------------------------------|
| `record_failure` | `auth::console_login::LockoutMap::record_failure`              |
| `try_login`      | `auth::console_login::run_login_round` (the locked-out branch) |
| `clock`          | the kernel's tick-driven `Sweeper::tick(now)`                  |

A trace from `pan` would map directly onto a unit test using
`LockoutMap`'s public surface; the Phase 4 unit tests already cover the
deterministic threshold and reset behaviors.
