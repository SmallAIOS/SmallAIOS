## ADDED Requirements

### Requirement: Commit Arms A Three-Try Pending Boot

After a verified image is flashed to the inactive slot, the commit step SHALL set the boot pointer to `active = <new slot>, pending = Some(<new slot>), tries_remaining = 3` in a single boot-pointer update. This is the only point in the update pipeline that modifies the boot pointer.

#### Scenario: Post-commit pointer state

- **WHEN** an update targeting `slot_b` passes signature verification and the slot write completes
- **THEN** the boot pointer SHALL read `active = B, pending = Some(B), tries_remaining = 3`
- **AND** the previously active slot's image SHALL remain intact as the rollback target

### Requirement: 60-Second Confirm Window

A newly booted pending image SHALL be confirmed within 60 seconds of boot, by either (a) a call to the `system_update_confirm()` syscall — typically issued by the inference server after one successful inference round-trip — or (b) a successful response to a `smallaios/admin/system/healthy` ping over Zenoh. Confirmation SHALL clear `pending` and reset `tries_remaining`. The 60-second window is the v1 target across all platforms.

#### Scenario: Syscall confirmation marks the boot good

- **WHEN** the new image calls `system_update_confirm()` 20 seconds after boot
- **THEN** `pending` SHALL be cleared and `tries_remaining` SHALL be reset
- **AND** the watchdog SHALL NOT trigger a rollback for this boot

#### Scenario: Zenoh healthy ping is an equivalent confirmation

- **WHEN** a remote operator's `smallaios/admin/system/healthy` ping receives a successful response within the 60-second window
- **THEN** the effect SHALL be identical to `system_update_confirm()`: `pending` cleared, `tries_remaining` reset

### Requirement: Watchdog-Driven Rollback After Try Exhaustion

While an update is `pending` and unconfirmed, the platform watchdog SHALL fire at the end of the confirm window and reset the machine. On each subsequent boot attempt the boot loader SHALL decrement `tries_remaining`; when the counter is exhausted the boot loader SHALL revert to the prior slot. The net behavior SHALL be that a bad image auto-reverts after roughly three minutes (three 60-second windows) of nobody-home, with no operator intervention and no bricked box.

#### Scenario: Unconfirmed boot triggers the watchdog

- **WHEN** a pending image boots but neither `system_update_confirm()` nor a successful `smallaios/admin/system/healthy` response occurs within 60 seconds
- **THEN** the platform watchdog SHALL fire and reset the machine
- **AND** the boot loader SHALL decrement `tries_remaining` on the next attempt

#### Scenario: Three failed attempts auto-revert

- **WHEN** a pending image fails to confirm on three consecutive boot attempts
- **THEN** `tries_remaining` SHALL reach zero
- **AND** the boot loader SHALL boot the prior slot
- **AND** total elapsed time from first bad boot to a running prior image SHALL be on the order of three minutes

#### Scenario: Hang before user space still rolls back

- **WHEN** a pending image hangs so early that no syscall is ever issued
- **THEN** the watchdog SHALL still fire — confirmation requires an affirmative act, not the absence of an error
- **AND** rollback SHALL proceed exactly as for a booted-but-unhealthy image
