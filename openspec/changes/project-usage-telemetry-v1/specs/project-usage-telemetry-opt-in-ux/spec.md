## ADDED Requirements

### Requirement: TTY First-Boot Opt-In Prompt Wording and Default

When the `enabled` field is unset (the `telemetry/usage.toml` file does not yet contain the field), the TTY first-boot setup SHALL prompt — after the root password is set:

```text
Help improve SmallAIOS by sharing anonymized usage
counters? See /docs/usage-telemetry.md for the full
schema. [y/N]:
```

The default SHALL be **N**. Answering `y` SHALL write `enabled = true` and `consent_recorded_at = <unix timestamp>` to the file. The final wording SHALL go through word-by-word review, but the prompt SHALL keep the committed shape: default N, a link to the schema document, and a two-character answer.

#### Scenario: Default answer leaves telemetry disabled

- **WHEN** the operator presses Enter (or answers `n`) at the first-boot opt-in prompt
- **THEN** usage telemetry SHALL remain disabled
- **AND** no `consent_recorded_at` timestamp SHALL be written
- **AND** the exporter SHALL NOT start

#### Scenario: Answering y records consent

- **WHEN** the operator answers `y` at the first-boot opt-in prompt
- **THEN** `enabled = true` SHALL be written to `telemetry/usage.toml`
- **AND** `consent_recorded_at` SHALL be written with the current unix timestamp in the same file

#### Scenario: Prompt references the schema document

- **WHEN** the first-boot opt-in prompt is displayed
- **THEN** it SHALL name `/docs/usage-telemetry.md` as the full-schema reference
- **AND** it SHALL display `[y/N]` with N as the default

### Requirement: Zenoh Opt-In Admin Verb

The Zenoh admin surface SHALL expose `smallaios/admin/telemetry/usage/opt_in` accepting a body `{accept: true, version: "<schema-version>"}`. A successful opt-in SHALL write `enabled = true` and `consent_recorded_at = <unix timestamp>` — the same consent record as the TTY flow.

#### Scenario: Zenoh opt-in records consent

- **WHEN** an authenticated admin publishes `{accept: true, version: "0"}` to `smallaios/admin/telemetry/usage/opt_in`
- **THEN** `enabled = true` and a positive `consent_recorded_at` unix timestamp SHALL be persisted to `telemetry/usage.toml`

### Requirement: Zenoh Opt-Out Admin Verb

The Zenoh admin surface SHALL expose `smallaios/admin/telemetry/usage/opt_out`. Opt-out SHALL flip `enabled = false` and write `consent_revoked_at = <unix timestamp>`. Any future re-enable SHALL require a new explicit opt-in.

#### Scenario: Zenoh opt-out disables and records revocation

- **WHEN** an authenticated admin invokes `smallaios/admin/telemetry/usage/opt_out` while telemetry is enabled
- **THEN** `enabled = false` SHALL be written to `telemetry/usage.toml`
- **AND** `consent_revoked_at` SHALL be written with the current unix timestamp
- **AND** the exporter SHALL stop emitting

#### Scenario: Re-enable requires a new explicit opt-in

- **WHEN** telemetry was opted out and the operator wants it back on
- **THEN** the flip back to `enabled = true` SHALL only occur through the TTY prompt or the `opt_in` Zenoh verb
- **AND** the new opt-in SHALL record a fresh `consent_recorded_at` timestamp

### Requirement: Consent Timestamp Persistence

The `consent_recorded_at` and `consent_revoked_at` timestamps SHALL be persisted in `telemetry/usage.toml` and SHALL survive reboots. The value `0` SHALL mean "never"; a positive value SHALL be the unix timestamp of the event.

#### Scenario: Consent survives reboot

- **WHEN** an operator opts in and the system subsequently reboots
- **THEN** the persisted `consent_recorded_at` value SHALL be unchanged
- **AND** the boot-time consent assertion SHALL pass without re-prompting

#### Scenario: Zero means never

- **WHEN** a system has never been opted in or out
- **THEN** `consent_recorded_at = 0` and `consent_revoked_at = 0` SHALL be the persisted values

### Requirement: Opt-Out Preserves the Install ID

Opting out SHALL NOT erase or re-roll the `install_id`. The preserved ID SHALL be used for longitudinal tracking (e.g. "did the bug-fix work") only if the operator explicitly opts back in later; while opted out, the ID SHALL NOT be transmitted anywhere.

#### Scenario: Install ID retained across opt-out and opt-in

- **WHEN** an operator opts out and later opts back in via the `opt_in` verb
- **THEN** the `install_id` field SHALL contain the same UUID as before the opt-out

#### Scenario: No transmission while opted out

- **WHEN** the system is opted out
- **THEN** the retained `install_id` SHALL NOT appear in any network traffic
- **AND** no usage-telemetry payload SHALL be emitted
