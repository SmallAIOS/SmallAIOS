## ADDED Requirements

### Requirement: Disabled-by-Default Code and Schema Defaults

The usage-telemetry `Config` type SHALL default `enabled = false` in code, and the `telemetry/usage.toml` schema SHALL treat a missing `enabled` field as `false`. While disabled — which is the permanent state until the project formally launches user telemetry collection — the usage-telemetry exporter SHALL generate zero network traffic and SHALL impose zero runtime cost.

#### Scenario: Code default is disabled

- **WHEN** a usage-telemetry `Config` value is constructed with its default field values
- **THEN** the `enabled` field SHALL be `false`

#### Scenario: Missing TOML field parses as disabled

- **WHEN** `telemetry/usage.toml` is loaded and the file does not contain the `enabled` field
- **THEN** the parsed configuration SHALL report `enabled = false`
- **AND** the exporter SHALL NOT be started

#### Scenario: Zero network traffic in the default state

- **WHEN** the system boots with a default (never-opted-in) usage-telemetry configuration
- **THEN** no connection SHALL be attempted to the `USAGE_TELEMETRY_ENDPOINT` URL
- **AND** no usage-telemetry payload SHALL be serialized

### Requirement: Boot-Time Consent Assertion

Before starting the usage-telemetry exporter at boot, the kernel SHALL assert that consent was recorded through the opt-in flow. If `enabled = true` and no positive `consent_recorded_at` timestamp exists in the same `telemetry/usage.toml` file, the kernel SHALL log a prominent warning and SHALL refuse to start the exporter. This is the third layer of the disabled-by-default defense, preventing a hand-edit from flipping the switch without going through the opt-in flow.

#### Scenario: Enabled without consent refuses to start the exporter

- **WHEN** `telemetry/usage.toml` contains `enabled = true` and `consent_recorded_at = 0` (or the field is absent)
- **THEN** the exporter SHALL NOT start
- **AND** a prominent warning SHALL be logged naming the missing consent record
- **AND** boot SHALL otherwise proceed normally

#### Scenario: Hand-edit cannot silently enable telemetry

- **WHEN** an operator hand-edits `telemetry/usage.toml` to set `enabled = true` without going through the TTY or Zenoh opt-in flow
- **THEN** the boot-time assertion SHALL keep the exporter stopped on the next boot
- **AND** re-enabling SHALL require a new explicit opt-in that records `consent_recorded_at`

#### Scenario: Valid consent record allows the exporter to start

- **WHEN** `telemetry/usage.toml` contains `enabled = true` and a `consent_recorded_at` timestamp greater than zero
- **THEN** the boot-time assertion SHALL pass
- **AND** the exporter SHALL be permitted to start

### Requirement: CI Enforcement of the Disabled-by-Default Invariant

CI SHALL carry a test asserting the disabled-by-default invariant cannot regress: a build whose usage-telemetry configuration defaults to `enabled = true` — in the code default or in the missing-TOML-field default — SHALL fail the test.

#### Scenario: Default-flip regression fails CI

- **WHEN** a change causes the `Config` code default or the missing-TOML-field default to yield `enabled = true`
- **THEN** the CI invariant test SHALL fail
- **AND** the change SHALL NOT be mergeable without a corresponding explicit policy change

### Requirement: Flipping the Default Is a Policy Event

Changing the shipped default to `enabled = true` SHALL be treated as a project-leadership policy event — involving privacy review, community communication, and release-notes language — and not as a code refactor. The privacy policy document `docs/usage-telemetry.md` SHALL exist and be reviewed before the project may flip the default. This change SHALL NOT flip the default.

#### Scenario: v1 ships in the disabled state

- **WHEN** the v1 implementation of this change is merged
- **THEN** the shipped default SHALL remain `enabled = false`
- **AND** the machinery SHALL be wired and testable without the exporter running by default

#### Scenario: Privacy policy gates the default flip

- **WHEN** the project prepares to flip the default to `enabled = true`
- **THEN** `docs/usage-telemetry.md` SHALL exist and SHALL have been reviewed
- **AND** the flip SHALL be accompanied by the CI invariant test being updated as part of the same reviewed policy change
