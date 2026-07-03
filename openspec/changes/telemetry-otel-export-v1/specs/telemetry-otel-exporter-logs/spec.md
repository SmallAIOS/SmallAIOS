## ADDED Requirements

### Requirement: Clean-Room `no_std` OTLP Logs Encoder

The `telemetry` crate SHALL extend the hand-written `#![no_std]` protobuf encoder to the OTLP logs signal, covering `opentelemetry.proto.collector.logs.v1.ExportLogsServiceRequest` and the `opentelemetry.proto.logs.v1` message set (`ResourceLogs`, `ScopeLogs`, `LogRecord`). Each encoded `LogRecord` SHALL carry body, severity, attributes, and timestamp. Trace-context fields SHALL be empty in v1 (traces are deferred to v2).

#### Scenario: Audit record round-trips through OTLP/Logs

- **WHEN** an audit-ring record is encoded as a `LogRecord` and the bytes are decoded with `protoc --decode`
- **THEN** the decoded body, severity, attributes, and timestamp SHALL match the source audit record's fields

#### Scenario: Trace context empty in v1

- **WHEN** any `LogRecord` is encoded by the v1 exporter
- **THEN** its trace-context fields (trace id, span id) SHALL be empty

### Requirement: Audit Ring Is the Sole v1 Log Source

The logs signal SHALL encode the same audit-ring records that the Zenoh `smallaios/metrics/log` keyspace publishes, and nothing else. Arbitrary application stdout SHALL NOT be shipped. This keeps exported log volume bounded and prevents free-text PII leakage.

#### Scenario: Audit record exported on the next push

- **WHEN** an audit record is appended to the audit ring while `[exporter.signals] logs = true`
- **THEN** the record SHALL appear as a `LogRecord` in the next OTLP logs push (subject to buffering)

#### Scenario: Application stdout is not exported

- **WHEN** a workload writes text to stdout that never enters the audit ring
- **THEN** no `LogRecord` SHALL be produced for that text

#### Scenario: Logs signal can be disabled independently

- **WHEN** `[exporter.signals]` sets `metrics = true` and `logs = false`
- **THEN** metric pushes SHALL continue
- **AND** no `ExportLogsServiceRequest` SHALL be sent
