## ADDED Requirements

### Requirement: OTLP Export Status Line in the Monitor Header

When `[exporter] enabled = true` in `telemetry/otel.toml`, the live monitor's header SHALL include an optional one-line OTLP export status showing the time since the last push attempt and its result, in the form `OTLP last 4s ok` / `OTLP last 12s err`. The line SHALL be visible to any role and SHALL be strictly read-only. When the exporter is disabled the line SHALL NOT appear.

#### Scenario: Healthy exporter shown in header

- **WHEN** the exporter is enabled and the last push succeeded 4 seconds ago
- **THEN** the monitor header SHALL render `OTLP last 4s ok`

#### Scenario: Failing exporter shown in header

- **WHEN** the exporter is enabled and the last push attempt failed 12 seconds ago
- **THEN** the monitor header SHALL render `OTLP last 12s err`

#### Scenario: No status line when the exporter is disabled

- **WHEN** `[exporter] enabled = false`
- **THEN** the monitor header SHALL NOT contain an OTLP status line

#### Scenario: Viewer sees the status read-only

- **WHEN** a `Role::Viewer` session runs the monitor with the exporter enabled
- **THEN** the OTLP status line SHALL be visible
- **AND** no keybinding SHALL allow modifying exporter state from the monitor
