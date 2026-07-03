## ADDED Requirements

### Requirement: First-boot usage-telemetry opt-in prompt

The TTY first-boot setup sequence SHALL include the usage-telemetry opt-in prompt immediately after the initial root password is set, and only when the `enabled` field is unset in `telemetry/usage.toml` (per the `project-usage-telemetry-opt-in-ux` capability). The prompt default SHALL be N; answering `y` SHALL write `enabled = true` and `consent_recorded_at = <unix timestamp>`. The prompt SHALL never be shown silently as accepted, and an update SHALL never re-enable telemetry.

#### Scenario: Prompt follows root-password setup

- **WHEN** first boot completes the initial root password setup and `telemetry/usage.toml` does not contain the `enabled` field
- **THEN** the TTY SHALL display the usage-telemetry opt-in prompt referencing `/docs/usage-telemetry.md` with `[y/N]`
- **AND** the prompt SHALL appear after the root password confirmation, before any service that would export telemetry starts

#### Scenario: Declining leaves telemetry off

- **WHEN** the operator answers `n` (or presses Enter, accepting the default) at the first-boot prompt
- **THEN** usage telemetry SHALL remain disabled
- **AND** no consent timestamp SHALL be recorded

#### Scenario: Accepting records consent

- **WHEN** the operator answers `y` at the first-boot prompt
- **THEN** `enabled = true` and `consent_recorded_at = <unix timestamp>` SHALL be written to `telemetry/usage.toml`

#### Scenario: Consent write succeeds on freshly formatted /data/

- **WHEN** the operator answers `y` at the prompt on the first boot after `/data/` was freshly formatted
- **THEN** the write of `enabled = true` and `consent_recorded_at` to `/data/telemetry/usage.toml` SHALL succeed
- **AND** the target directory `/data/telemetry/` SHALL already exist with mode 0700, created by `mgmt-config-layout`'s first-boot directory-tree creation

#### Scenario: Prompt skipped when the field is already set

- **WHEN** first boot runs on a system whose `telemetry/usage.toml` already contains an `enabled` field
- **THEN** the opt-in prompt SHALL NOT be shown
