## ADDED Requirements

### Requirement: New `system_update_confirm` syscall

The kernel SHALL expose a new `system_update_confirm` syscall (`SYS_SYSTEM_UPDATE_CONFIRM = 0x58`, in the existing System category `0x50–0x5F`, immediately after `SYS_BOOT_SUCCESS = 0x57`):

```text
system_update_confirm() -> 0 | -errno
```

The syscall SHALL be called by the user-space process responsible for judging whether the new image actually came up correctly — typically the inference server after one successful inference round-trip. A successful call SHALL commit the pending update on the active `fs-ab-boot` boot config record: clear `tentative`, set `boot_success = 1`, and reset `tries_remaining`, marking the pending update's boot as good so the watchdog rollback path does not fire.

`system_update_confirm` SHALL NOT introduce a second boot-record model: it SHALL drive the same `fs-ab-boot` commit path as the existing `boot_success` syscall (`SYS_BOOT_SUCCESS = 0x57`), which is unchanged by this change; `system_update_confirm` exists as the update pipeline's dedicated confirmation entry point (per `update-watchdog-rollback`), additionally resetting `tries_remaining`. Like `boot_success`, it SHALL be idempotent: calling it when the active record already shows `boot_success = 1` SHALL return `0` without re-writing the record.

#### Scenario: Confirm clears tentative and resets the counter

- **WHEN** the active `fs-ab-boot` record reads `tentative = 1, boot_success = 0, tries_remaining = 2` and user space calls `system_update_confirm()`
- **THEN** the syscall SHALL return `0`
- **AND** the record SHALL transition to `tentative = 0, boot_success = 1` with `tries_remaining` reset

#### Scenario: Confirm within the window prevents rollback

- **WHEN** the inference server completes one successful inference round-trip and calls `system_update_confirm()` inside the 60-second confirm window
- **THEN** the watchdog SHALL NOT trigger a rollback for this boot
- **AND** the new slot SHALL remain the active slot on subsequent boots

#### Scenario: No duplicate commit when boot_success already ran

- **WHEN** `boot_success` has already committed the active record this boot and `system_update_confirm()` is then called
- **THEN** the syscall SHALL return `0` without re-writing the record
- **AND** exactly one boot-good commit SHALL have been written to the `fs-ab-boot` record

### Requirement: Documented syscall count after remote update

The architecture documentation SHALL list one new System-category syscall: `SYS_SYSTEM_UPDATE_CONFIRM = 0x58`. The post-`remote-update-v1` syscall count is the prior documented count plus one. `SYSCALL_TABLE_SIZE` SHALL remain `0xA0`.

#### Scenario: Architecture doc reflects the new syscall

- **WHEN** `docs/architecture.md` is read
- **THEN** it SHALL list `SYS_SYSTEM_UPDATE_CONFIRM = 0x58` in the System-category syscall table
- **AND** SHALL state the post-`remote-update-v1` syscall count as the prior documented count plus one
