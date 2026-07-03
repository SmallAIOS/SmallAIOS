## ADDED Requirements

### Requirement: telemetry_export_attempt records

Every OTLP export attempt SHALL append an audit record with `action = "telemetry_export_attempt"` recording its outcome: success, failure (with the transport result code), or dropped-from-buffer (records evicted by ring-buffer overflow before they could be sent). These records are themselves exported over OTLP/Logs, so the audit trail of whether telemetry works is visible in the backend. Recursion SHALL be capped at one level: `telemetry_export_attempt` records about `telemetry_export_attempt` records SHALL NOT themselves produce audit entries.

#### Scenario: Successful push audited

- **WHEN** an OTLP push succeeds
- **THEN** an audit record `{ action: "telemetry_export_attempt", code: 0, ... }` SHALL be appended to the ring

#### Scenario: Failed push audited with result code

- **WHEN** an OTLP push fails at the transport
- **THEN** an audit record with `action = "telemetry_export_attempt"` and a non-zero `code` SHALL be appended

#### Scenario: Buffer eviction audited as dropped-from-buffer

- **WHEN** the exporter ring buffer evicts unsent records on overflow
- **THEN** an audit record SHALL be appended recording the dropped-from-buffer outcome

#### Scenario: Recursion capped at one level

- **WHEN** an export attempt whose payload consists solely of `telemetry_export_attempt` records completes or fails
- **THEN** no further `telemetry_export_attempt` audit entry SHALL be produced for that attempt
- **AND** a repeated failure loop SHALL NOT grow the audit ring recursively
