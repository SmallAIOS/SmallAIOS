# mgmt-config-model Specification

## Purpose
TBD - created by archiving change management-login-v1. Update Purpose after archive.
## Requirements
### Requirement: Typed Config source of truth
The `mgmt` crate SHALL define a single typed Rust struct `Config` that is the source of truth for every operator-tunable configuration field in SmallAIOS. Every reachable management surface (TOML loader, TTY console, Zenoh admin, future UDS, etc.) SHALL be a thin (de)serializer over this same struct. There SHALL NOT be any surface-specific configuration knob held outside `Config`.

#### Scenario: Single source of truth
- **WHEN** the same field is read via the TOML loader and the Zenoh admin surface
- **THEN** both reads SHALL return identical values from the same in-memory `Config` instance

### Requirement: Apply lifecycle
Every write to `Config` — regardless of originating surface — SHALL pass through the same lifecycle: (1) parse the input into the typed field, (2) validate per-field rules and cross-field constraints, (3) stage to a `<path>.tmp` file in the same directory as the target, (4) `fsync` then atomic `rename`, (5) notify subscribers via a broadcast channel, (6) append a `(who, when, surface, path, before, after)` record to the audit ring.

A failed validation at step 2 SHALL leave the target file and the in-memory `Config` unchanged. A crash between step 3 and step 4 SHALL leave the target file unchanged.

#### Scenario: Validation rejects out-of-range value
- **WHEN** a write attempts to set `metrics.cpu.interval_ms = 50` (below the 100 ms bound)
- **THEN** validation SHALL return `-EINVAL`
- **AND** the on-disk file SHALL be unchanged
- **AND** the in-memory `Config` SHALL be unchanged
- **AND** no audit record for the rejected write SHALL be appended

#### Scenario: Successful write through full lifecycle
- **WHEN** Root writes `metrics.cpu.interval_ms = 500`
- **THEN** the value SHALL be parsed into the field, validated, staged to `<file>.tmp`, fsync'd, renamed
- **AND** subscribers SHALL receive a notify event
- **AND** an audit record SHALL be appended with `before=1000, after=500, who=root, surface=zenoh|tty|toml`

### Requirement: Live versus boot-time field annotation
Every `Config` field SHALL declare a `#[reload("live"|"boot")]` attribute. Default SHALL be `live`. A `live` field's notify SHALL trigger affected subsystems to re-read and apply at their own pace. A `boot` field SHALL be persisted but flagged "pending reboot" until next boot; the in-memory value SHALL NOT change until reboot.

A build-time schema walker SHALL fail compilation if any field lacks the attribute.

#### Scenario: Live field applies immediately
- **WHEN** a field annotated `#[reload("live")]` is updated
- **THEN** subscribers SHALL receive notify
- **AND** subsystems that re-read SHALL see the new value

#### Scenario: Boot field defers to reboot
- **WHEN** a field annotated `#[reload("boot")]` is updated
- **THEN** the file SHALL be rewritten with the new value
- **AND** the in-memory `Config` SHALL still report the old value
- **AND** `Config::pending_reboot()` SHALL list the changed field

#### Scenario: Missing attribute fails build
- **WHEN** a developer adds a new `Config` field without `#[reload(...)]`
- **THEN** the schema walker SHALL fail compilation with a message naming the field

