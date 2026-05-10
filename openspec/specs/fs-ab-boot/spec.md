# fs-ab-boot Specification

## Purpose
TBD - created by archiving change embedded-filesystem-v1. Update Purpose after archive.
## Requirements
### Requirement: A/B boot config layout
A dedicated 8 MiB GPT partition (#5, type GUID `A3F7C2E0-FACE-4FFF-BBBB-000000000000`) SHALL hold the A/B boot pointer. The partition SHALL contain two double-buffered records (call them slot-X and slot-Y, each 4 KiB) at fixed offsets 0x000 and 0x1000. Each record SHALL have:

```text
struct BootConfigRecord {
    magic:        [u8; 8]    = "SmAIOSBC",
    version:      u8         = 1,
    valid:        u8,        // 1 = valid, 0 = invalid
    active_slot:  u8,        // 0 = slot A squashfs, 1 = slot B squashfs
    tentative:    u8,        // 1 = updated, awaiting boot_success
    generation:   u64        // monotonically increasing
    boot_success: u8         // 1 once kernel called boot_success
    pad:          [u8; ...]
    record_hash:  [u8; 32]   // SHA-3-256 of all preceding bytes
}
```

The bootloader SHALL read both records, validate each by checking magic, version, and `record_hash`, and SHALL pick the valid record with the highest `generation`. If both are invalid, the bootloader SHALL halt with a recovery message.

#### Scenario: Higher-generation record wins
- **WHEN** slot-X has generation 5 valid and slot-Y has generation 4 valid
- **THEN** the bootloader SHALL use slot-X (active_slot, tentative, etc.)

#### Scenario: Invalid record skipped
- **WHEN** slot-X has a wrong record_hash and slot-Y is valid at generation 4
- **THEN** the bootloader SHALL use slot-Y

#### Scenario: Both records invalid halts boot
- **WHEN** both records fail their record_hash
- **THEN** the bootloader SHALL halt with `Err: boot config corrupt, both slots`

### Requirement: Atomic update via the inactive slot
Updating the boot pointer SHALL: (1) compute the new record bytes (with `generation = max + 1` and the desired `active_slot`/`tentative` fields), (2) write the new record to the **inactive** slot (the one with the lower `generation`), (3) `flush()` to durable storage, (4) only then SHALL the bootloader's pick of "highest valid generation" select the new record. There SHALL be no in-place modification of the previously-active slot until the new slot is durably written.

#### Scenario: Power loss before flush leaves previous slot intact
- **WHEN** a write to the inactive slot is in progress and power is lost mid-write
- **THEN** the previously-active slot SHALL remain unchanged on next boot
- **AND** the bootloader SHALL select the previously-active slot via the highest-valid-generation rule

#### Scenario: Power loss after flush of inactive slot
- **WHEN** the inactive slot is fully written and flushed but power is lost before the OS resumes
- **THEN** the next boot SHALL pick the new slot (highest valid generation)

### Requirement: Atomicity invariant under arbitrary power loss
After any sequence of writes to the boot config followed by an arbitrary power loss, the bootloader SHALL find at least one slot that is valid and whose `generation` is one of the two most recent generations ever written. Generation counters SHALL be monotonically increasing; values SHALL NEVER be reused. This invariant SHALL be modeled in Kani.

#### Scenario: Kani harness covers all torn-write interleavings
- **WHEN** the Kani harness runs `boot_config_atomic_update_under_power_loss`
- **THEN** the proof SHALL hold for all valid bus-write granularities (per-byte, per-512-byte sector, per-4-KiB block)
- **AND** the proof SHALL show at least one valid slot exists in every reachable post-state

### Requirement: UEFI variable mirror
On UEFI-capable arches (x86-64, AArch64-with-UEFI), the kernel SHALL also write the active slot pointer to a UEFI variable `SmallAIOSBoot-A3F7C2E0-FACE-4FFF-BBBB-000000000000`. The UEFI bootloader MAY read this variable to pick the slot without parsing GPT first. On disagreement between the variable and the partition, the partition SHALL win — the variable is a hint, not the truth.

#### Scenario: UEFI variable matches partition after update
- **WHEN** an A/B update completes successfully
- **THEN** the UEFI variable SHALL be updated to match the new active slot
- **AND** subsequent reads of the variable SHALL return the new value

#### Scenario: Stale variable overruled by partition
- **WHEN** the partition says active_slot=B and the UEFI variable says A (e.g., NVRAM corruption)
- **THEN** the kernel SHALL boot from B and SHALL log `warn: UEFI boot var stale; rewriting`
- **AND** SHALL rewrite the variable to match

### Requirement: Watchdog + boot_success rollback
After an A/B update, the new record SHALL be written with `tentative = 1, boot_success = 0`. A 60-s watchdog SHALL be armed at boot. The kernel SHALL call a new `boot_success` syscall after self-tests and the first successful login. `boot_success` SHALL clear `tentative` and set `boot_success = 1` on the active record (via the same atomic-write-to-inactive-slot path).

If the watchdog fires before `boot_success` is called, the bootloader on the next boot SHALL observe `tentative=1, boot_success=0` and SHALL roll back to the previous slot (via choosing the second-highest valid generation), then SHALL invalidate the failed slot's record (set `valid=0`).

The watchdog timeout SHALL be configurable via `mgmt/policy.toml` `fs.boot.watchdog_seconds` (default 60).

#### Scenario: Successful boot calls boot_success
- **WHEN** the new image self-tests pass and an operator logs in successfully
- **THEN** the kernel SHALL invoke `boot_success`
- **AND** the active record SHALL transition to `tentative=0, boot_success=1`

#### Scenario: Watchdog fires triggers rollback
- **WHEN** 60 s elapses without `boot_success` being called
- **THEN** the watchdog SHALL trigger a hardware reset
- **AND** on next boot the bootloader SHALL observe the tentative state and select the previous slot
- **AND** the failed slot SHALL be marked `valid=0`

#### Scenario: Watchdog timeout configurable
- **WHEN** `fs.boot.watchdog_seconds = 30` is set in `mgmt/policy.toml`
- **THEN** subsequent boots SHALL use a 30 s watchdog window

