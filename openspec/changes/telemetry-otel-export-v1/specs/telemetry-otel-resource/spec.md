## ADDED Requirements

### Requirement: Resource Attributes Present on Every Export

Every OTLP export request (metrics and logs signals) SHALL carry an OTel Resource block populated automatically by the `telemetry` crate's resource module. No per-export, per-metric, or per-log code SHALL be required to attach resource attributes. The Resource SHALL contain the standard attributes:

| Attribute | Source |
|-----------|--------|
| `service.name` | constant `smallaios` |
| `service.namespace` | constant `smallaios` |
| `service.version` | build-time version + commit (e.g. `0.2.3+abcd123`) |
| `service.instance.id` | per-host UUID persisted in `system.toml` |
| `host.id` | same UUID as `service.instance.id` in v1 |
| `host.name` | hostname from `system.toml` |
| `host.arch` | build-time arch triple component (`aarch64`, `x86_64`, `riscv64`) |
| `os.type` | constant `smallaios` |
| `os.version` | build-time version (e.g. `0.2.3`) |
| `deployment.environment` | operator-configured value from `[resource] deployment_environment` |

Attribute values SHALL be pinned to the OTel semantic-convention names of a specific OTel specification version (v1.32.0), and the upgrade path SHALL be documented.

#### Scenario: Metrics export carries the full resource block

- **WHEN** the exporter encodes an `ExportMetricsServiceRequest` on a host built as version `0.2.3+abcd123` for `aarch64` with hostname `orin-01` and `deployment_environment = "dev"`
- **THEN** the encoded `ResourceMetrics.resource` SHALL contain `service.name = "smallaios"`, `service.namespace = "smallaios"`, `service.version = "0.2.3+abcd123"`, `host.name = "orin-01"`, `host.arch = "aarch64"`, `os.type = "smallaios"`, `os.version = "0.2.3"`, and `deployment.environment = "dev"`
- **AND** `service.instance.id` and `host.id` SHALL both equal the persisted per-host UUID

#### Scenario: Logs export carries an identical resource block

- **WHEN** the exporter encodes an `ExportLogsServiceRequest` in the same push cycle as a metrics export
- **THEN** the `ResourceLogs.resource` attributes SHALL be identical to the `ResourceMetrics.resource` attributes

#### Scenario: Two test boxes are distinguishable in the backend

- **WHEN** two SmallAIOS units (a Jetson Orin and an x86 reference machine) export to the same OTLP backend
- **THEN** their exports SHALL carry distinct `service.instance.id` values
- **AND** distinct `host.name` and `host.arch` values
- **AND** no export from either box SHALL be attributable to the other

### Requirement: Per-Host UUID Generated Once and Persisted in /data/

The per-host UUID backing `service.instance.id` (and, in v1, `host.id`) SHALL be generated exactly once, on first boot, before the first export, and written atomically to `/data/system.toml`. The UUID SHALL never be regenerated except by an explicit `auth-create-user`-gated reset. Because the UUID lives in `/data/` rather than in the OS image, re-flashing the OS SHALL NOT change the host's identity.

#### Scenario: First boot generates and persists the UUID

- **WHEN** a freshly provisioned unit boots for the first time and `/data/system.toml` contains no instance UUID
- **THEN** the kernel SHALL generate a UUID and write it atomically to `/data/system.toml`
- **AND** the write SHALL complete before the first OTLP export is attempted

#### Scenario: Reboot preserves the UUID

- **WHEN** a unit that has already generated its UUID reboots
- **THEN** the same UUID SHALL be read from `/data/system.toml`
- **AND** no new UUID SHALL be generated

#### Scenario: OS re-flash preserves the UUID

- **WHEN** the OS image is re-flashed while the `/data/` partition is preserved
- **THEN** subsequent exports SHALL carry the same `service.instance.id` as before the re-flash

#### Scenario: UUID reset requires the auth-create-user gate

- **WHEN** a UUID reset is attempted without passing the explicit `auth-create-user`-gated reset path
- **THEN** the reset SHALL be refused
- **AND** the persisted UUID SHALL remain unchanged

### Requirement: Operator Labels Map Is the Only Free-Form Resource Metadata

The `[resource] labels` map in `telemetry/otel.toml` SHALL be the only source of free-form resource metadata (e.g. `rack = "lab-3"`, `experiment = "baseline"`). Labels SHALL be attached as resource attributes — never encoded into metric names — so user-supplied labels do not multiply active-series cardinality. Label values SHALL never be derived from inference content.

#### Scenario: Labels exported as resource attributes, not metric names

- **WHEN** `labels = { rack = "lab-3", experiment = "baseline" }` is configured and a metrics push occurs
- **THEN** the exported resource SHALL contain attributes `rack = "lab-3"` and `experiment = "baseline"`
- **AND** no exported metric name SHALL embed a label key or value

#### Scenario: No metadata derived from inference content

- **WHEN** the unit serves inference requests while exporting telemetry
- **THEN** no resource attribute or label value SHALL be derived from inference request or response content
