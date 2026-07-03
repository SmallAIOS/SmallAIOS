## ADDED Requirements

### Requirement: /data/telemetry/ directory and keyfile permission entries

The `/data/` layout SHALL gain a `telemetry/` subsystem directory, created atomically alongside the rest of the tree per the existing first-boot directory-tree-creation requirement. The per-file permission table SHALL gain two entries: `/data/telemetry/otel.toml` mode 0644 (viewer-readable — operators may need to inspect endpoint config) and `/data/telemetry/otel.key` mode 0600, root-only. The loader SHALL refuse either file when its mode is laxer than declared.

#### Scenario: telemetry directory exists after first boot

- **WHEN** the system runs first-boot completion
- **THEN** `/data/telemetry/` SHALL exist
- **AND** `/data/telemetry/otel.toml` SHALL exist with mode 0644 and conservative defaults (`enabled = false`)

#### Scenario: Laxer keyfile mode rejected by the loader

- **WHEN** `/data/telemetry/otel.key` exists with mode 0640 (declared 0600)
- **THEN** the loader SHALL refuse to read the file
- **AND** SHALL treat it as corrupt per the per-file permission declaration rule

#### Scenario: Stricter otel.toml mode accepted

- **WHEN** `/data/telemetry/otel.toml` exists with mode 0600 (declared 0644)
- **THEN** the loader SHALL accept the file

### Requirement: Secret redaction for keyfile-related audit records

Audit records that mention the telemetry keyfile (key writes, key resets) SHALL never contain API-key material: the `before` and `after` fields SHALL carry a redaction placeholder instead of key bytes. Because audit records are themselves exported over OTLP/Logs, redaction SHALL happen at record-creation time so that no key material can leave the box via the export path.

#### Scenario: Keyfile write audited without key bytes

- **WHEN** Root writes a new API key to `/data/telemetry/otel.key`
- **THEN** the resulting audit record SHALL record the action against the keyfile path
- **AND** its `before` and `after` fields SHALL contain a redaction placeholder, not key material

#### Scenario: Exported audit record carries no key material

- **WHEN** a keyfile-related audit record is exported as an OTLP `LogRecord`
- **THEN** the exported body and attributes SHALL contain no API-key bytes
