## ADDED Requirements

### Requirement: `telemetry/otel.toml` Schema

Exporter configuration SHALL live in `telemetry/otel.toml`, registered with the `mgmt` `Config` model and reusing the existing TOML / TTY / Zenoh surfaces (Config + ConfigSurface + atomic-rewrite convention from `management-login-v1`). The schema SHALL be:

- `[exporter]`: `enabled` (default `false` — explicit opt-in), `endpoint` (default empty; required when enabled), `auth_mode` (`"basic"` | `"bearer"`, default `"basic"`), `api_key_path` (default `/data/telemetry/otel.key`), `push_interval_seconds` (default `10`), `buffer_bytes` (default `1048576`)
- `[exporter.signals]`: `metrics` (default `true`), `logs` (default `true`), `traces` (default `false`; traces are v2)
- `[resource]`: `deployment_environment` (e.g. `dev`, `staging`, `prod`) and a free-form `labels` map

#### Scenario: Disabled by default with zero cost

- **WHEN** a fresh `telemetry/otel.toml` is generated
- **THEN** it SHALL contain `enabled = false`, `auth_mode = "basic"`, `push_interval_seconds = 10`, `buffer_bytes = 1048576`, `metrics = true`, `logs = true`, `traces = false`
- **AND** while `enabled = false` no exporter task SHALL run and no CPU SHALL be consumed by the export path

#### Scenario: Enabled without an endpoint rejected

- **WHEN** an operator writes `enabled = true` while `endpoint = ""`
- **THEN** the config validator SHALL reject the write
- **AND** the exporter SHALL NOT start

#### Scenario: Unknown auth_mode rejected

- **WHEN** an operator writes `auth_mode = "digest"`
- **THEN** the config validator SHALL reject the value as outside the `basic` | `bearer` set

### Requirement: API Key Lives in a Separate 0600 Keyfile

The API key SHALL never be stored in `telemetry/otel.toml`; the file SHALL reference the key only via `api_key_path`. The keyfile (default `/data/telemetry/otel.key`) SHALL be mode 0600, root-only, and the loader SHALL refuse a keyfile whose mode is laxer than declared (rule inherited from `mgmt-config-layout`). The setup script's first action on a new box SHALL be to prompt for the key and write it to the keyfile. Neither `otel.toml` nor the keyfile SHALL ever be checked into git.

#### Scenario: No key material in otel.toml

- **WHEN** `telemetry/otel.toml` is inspected on a configured box
- **THEN** it SHALL contain only `api_key_path`, never API-key bytes

#### Scenario: Laxer keyfile mode refused

- **WHEN** `/data/telemetry/otel.key` exists with mode 0644
- **THEN** the loader SHALL refuse to read it
- **AND** no export request SHALL be sent with a key obtained from the rejected file

### Requirement: Role Gate on Telemetry Configuration

Access to the exporter configuration SHALL be role-gated: `Role::Root` MAY write `telemetry/otel.toml` and the keyfile. `Role::Operator` MAY read `otel.toml` (not the keyfile) and SHALL NOT modify exporter state. `Role::Viewer` MAY read `otel.toml`, SHALL NOT modify anything, and SHALL NOT read the keyfile.

#### Scenario: Root writes the config

- **WHEN** a `Role::Root` session writes `telemetry/otel.toml`
- **THEN** the write SHALL succeed via the atomic-rewrite path

#### Scenario: Operator modification denied

- **WHEN** a `Role::Operator` session attempts to write `telemetry/otel.toml` or toggle the exporter
- **THEN** the request SHALL be denied
- **AND** exporter state SHALL remain unchanged

#### Scenario: Keyfile unreadable below Root

- **WHEN** a `Role::Operator` or `Role::Viewer` session attempts to read `/data/telemetry/otel.key`
- **THEN** the read SHALL be denied

### Requirement: Telemetry Management Surfaces

The TTY shell SHALL provide `telemetry status` (available to any role; shows the configured endpoint and last push success/failure) and `telemetry config` (root only). The Zenoh `smallaios/admin/telemetry/**` keyspace SHALL mirror the same operations under the same role gate.

#### Scenario: Viewer inspects export health

- **WHEN** an authenticated `Role::Viewer` session runs `telemetry status`
- **THEN** the output SHALL show the configured endpoint and the last push result
- **AND** no key material SHALL appear in the output

#### Scenario: Non-root telemetry config denied

- **WHEN** a `Role::Operator` session runs `telemetry config`
- **THEN** the command SHALL be refused

#### Scenario: Zenoh admin keyspace mirrors the gate

- **WHEN** a client acts on `smallaios/admin/telemetry/**` over Zenoh
- **THEN** the same role gate SHALL apply as on the TTY surface (any role for status, root for config changes)
