## MODIFIED Requirements

### Requirement: A/B boot config layout

A dedicated 8 MiB GPT partition (#5, type GUID `A3F7C2E0-FACE-4FFF-BBBB-000000000000`) SHALL hold the A/B boot pointer. The partition SHALL contain two double-buffered records (call them slot-X and slot-Y, each 4 KiB) at fixed offsets 0x000 and 0x1000. Each record SHALL have:

```text
struct BootConfigRecord {
    magic:           [u8; 8]    = "SmAIOSBC",
    version:         u8         = 2,
    valid:           u8,        // 1 = valid, 0 = invalid
    active_slot:     u8,        // 0 = slot A, 1 = slot B
    tentative:       u8,        // 1 = updated, awaiting confirmation
    generation:      u64        // monotonically increasing
    boot_success:    u8         // 1 once the boot was confirmed good
    tries_remaining: u8         // v2 (remote-update-v1): boot attempts left while tentative
    manifest_hash:   [u8; 32]   // v2 (remote-update-v1): SHA3-256 hash of the active image's manifest
    pad:             [u8; ...]
    record_hash:     [u8; 32]   // SHA-3-256 of all preceding bytes
}
```

`tries_remaining` and `manifest_hash` are added by `remote-update-v1` (record `version = 2`) and SHALL be carved out of the previously reserved `pad` bytes, leaving all version-1 field offsets unchanged. This record is the only boot pointer in the system: the `remote-update-v1` update pipeline (`update-boot-slots`) SHALL read and write this same record and SHALL NOT introduce a second boot-pointer format.

The bootloader SHALL read both records, validate each by checking magic, version (1 or 2), and `record_hash`, and SHALL pick the valid record with the highest `generation`. A valid version-1 record SHALL be interpreted as `tries_remaining = 0` and `manifest_hash = [0; 32]`. If both records are invalid, the bootloader SHALL halt with a recovery message.

#### Scenario: Higher-generation record wins
- **WHEN** slot-X has generation 5 valid and slot-Y has generation 4 valid
- **THEN** the bootloader SHALL use slot-X (active_slot, tentative, etc.)

#### Scenario: Invalid record skipped
- **WHEN** slot-X has a wrong record_hash and slot-Y is valid at generation 4
- **THEN** the bootloader SHALL use slot-Y

#### Scenario: Both records invalid halts boot
- **WHEN** both records fail their record_hash
- **THEN** the bootloader SHALL halt with `Err: boot config corrupt, both slots`

#### Scenario: Version-1 record remains readable
- **WHEN** the bootloader reads a valid version-1 record written before `remote-update-v1`
- **THEN** the record SHALL validate and be usable
- **AND** SHALL be interpreted as `tries_remaining = 0, manifest_hash = [0; 32]`

### Requirement: Watchdog + boot_success rollback

After an A/B update, the new record SHALL be written with `tentative = 1, boot_success = 0, tries_remaining = 3` (per `update-watchdog-rollback`). A 60-s watchdog SHALL be armed at boot. The boot SHALL be confirmed good by any of: the existing `boot_success` syscall (called by the kernel after self-tests and the first successful login), the `system_update_confirm` syscall (`SYS_SYSTEM_UPDATE_CONFIRM = 0x58`, called by the health-judging user-space process per `kernel-syscalls`), or a successful `smallaios/admin/system/healthy` response over Zenoh. All three confirmations SHALL drive the same commit on the active record — clear `tentative`, set `boot_success = 1`, reset `tries_remaining` — via the same atomic-write-to-inactive-slot path, and SHALL disarm the watchdog.

On every boot attempt where the selected record reads `tentative = 1, boot_success = 0` (including the first boot after commit), the bootloader SHALL apply try-exhaustion rollback: if `tries_remaining > 0`, it SHALL decrement `tries_remaining` (via the same atomic write path) and hand off to the tentative slot; if `tries_remaining = 0`, it SHALL roll back to the previous slot (by choosing the second-highest valid generation), then SHALL invalidate the failed slot's record (set `valid = 0`). If the watchdog fires before a confirmation, the machine resets and the next boot attempt repeats this rule, so a bad image auto-reverts once the tries are exhausted, with no operator intervention.

The watchdog timeout SHALL be configurable via `mgmt/policy.toml` `fs.boot.watchdog_seconds` (default 60).

#### Scenario: Successful boot calls boot_success
- **WHEN** the new image self-tests pass and an operator logs in successfully
- **THEN** the kernel SHALL invoke `boot_success`
- **AND** the active record SHALL transition to `tentative = 0, boot_success = 1` with `tries_remaining` reset

#### Scenario: system_update_confirm commits through the same path
- **WHEN** the active record reads `tentative = 1, boot_success = 0, tries_remaining = 2` and user space calls `system_update_confirm()`
- **THEN** the active record SHALL transition to `tentative = 0, boot_success = 1` with `tries_remaining` reset
- **AND** the watchdog SHALL be disarmed

#### Scenario: Watchdog fires decrements the try counter and retries
- **WHEN** 60 s elapses without any confirmation and the record still shows `tries_remaining > 0`
- **THEN** the watchdog SHALL trigger a hardware reset
- **AND** on next boot the bootloader SHALL decrement `tries_remaining` and hand off to the tentative slot again

#### Scenario: Try exhaustion triggers rollback
- **WHEN** the bootloader observes `tentative = 1, boot_success = 0, tries_remaining = 0`
- **THEN** it SHALL select the previous slot (second-highest valid generation)
- **AND** the failed slot's record SHALL be marked `valid = 0`

#### Scenario: Watchdog timeout configurable
- **WHEN** `fs.boot.watchdog_seconds = 30` is set in `mgmt/policy.toml`
- **THEN** subsequent boots SHALL use a 30 s watchdog window
