## ADDED Requirements

### Requirement: `/data/audit_export/immudb.toml` schema

The exporter SHALL read its configuration from `/data/audit_export/immudb.toml`. The file SHALL conform to the following schema (defaults shown):

```toml
[exporter]
enabled            = false
endpoint           = ""                         # required when enabled
fallback_endpoints = []                         # optional rotation list
auth_mode          = "bearer"                   # "bearer" (v1) or "mtls" (v2)
token_path         = "/data/audit_export/immudb.token"
state_path         = "/data/audit_export/last_state.bin"
database           = "smallaios_audit"
batch_size         = 100
batch_interval_ms  = 1000
buffer_bytes       = 4194304
backoff_initial_ms = 10000
backoff_cap_ms     = 300000

[tls]
require_pqc        = false
server_pubkey_fingerprint = ""                  # SHA-256 of immudb Ed25519 pubkey, hex

[record_filter]
include_actions = []
exclude_actions = [
  "audit_export_attempt",
  "audit_export_overflow",
  "audit_export_proof_failure",
  "audit_export_rollback_suspected",
  "audit_export_decode_failure",
  "audit_export_state_init",
  "immudb_state",
]
```

The schema SHALL be additive: future fields appended; existing fields never renamed or repurposed.

#### Scenario: Default config disables exporter
- **WHEN** a fresh `immudb.toml` is generated on first boot
- **THEN** `enabled = false` SHALL be present
- **AND** `endpoint = ""` SHALL be present
- **AND** no network traffic SHALL be issued

#### Scenario: Enabled without endpoint refuses to start
- **WHEN** `enabled = true` and `endpoint = ""` are written together
- **THEN** the validator SHALL reject the write with `-EINVAL`
- **AND** the previous value of `enabled` SHALL remain in effect

#### Scenario: Live reload of batch_size
- **WHEN** an operator writes `batch_size = 500` via the management surface
- **THEN** subsequent batches SHALL respect the new size without restart

### Requirement: Token keyfile at `/data/audit_export/immudb.token`

The bearer-mode authentication token SHALL be stored at `/data/audit_export/immudb.token` with mode 0600 and owned by kernel. The file SHALL contain exactly the token bytes (no leading or trailing whitespace, no newline, no PEM framing). The loader SHALL refuse to read the file if its mode is laxer than 0600.

The TOML file (`immudb.toml`) SHALL NOT contain the token itself; it SHALL only reference `token_path`. Audit records that mention the token path SHALL redact any token bytes appearing in `before` / `after` JSON via the existing `mgmt-config-layout` secret-redaction rule.

#### Scenario: Mode 0644 token rejected
- **WHEN** `/data/audit_export/immudb.token` exists with mode 0644
- **THEN** the loader SHALL refuse to read it
- **AND** the exporter SHALL refuse to start
- **AND** an audit record `audit_export_attempt code = -EACCES` SHALL be appended

#### Scenario: Token bytes redacted from audit
- **WHEN** an operator rotates the token by writing a new value to the path
- **THEN** the resulting `config_write` audit record SHALL show `before` and `after` as redacted placeholders
- **AND** SHALL NOT contain any byte of either the old or new token

### Requirement: Role gate for exporter configuration

`Role::Root` SHALL be the only role permitted to write `immudb.toml` or `immudb.token`. `Role::Operator` and `Role::Viewer` SHALL be permitted to read `immudb.toml` (so they can confirm endpoint configuration) but SHALL NOT be permitted to read `immudb.token`. The management shell SHALL expose `audit-export status` (any role; shows endpoint plus last result) and `audit-export config` (Root only; mutates `immudb.toml`).

#### Scenario: Operator config write denied
- **WHEN** `Role::Operator` issues `audit-export config endpoint=https://immudb.example.com:3322`
- **THEN** the syscall SHALL return `-EPERM`
- **AND** an audit `DENY:audit_export_config` record SHALL be appended

#### Scenario: Viewer reads endpoint
- **WHEN** `Role::Viewer` issues `audit-export status`
- **THEN** the output SHALL show the configured endpoint and the last batch result
- **AND** SHALL NOT include the token bytes or its path's contents

### Requirement: Atomic-rewrite of exporter state

The state file `/data/audit_export/last_state.bin` SHALL be updated via stage-and-rename: write to `last_state.bin.tmp`, `fsync`, then atomic `rename`. A crash mid-write SHALL leave either the previous valid state or no file at all; the loader SHALL handle the "no file" case as "cold start."

#### Scenario: Crash before rename leaves previous state intact
- **WHEN** the exporter writes `last_state.bin.tmp` and the kernel crashes before rename
- **THEN** on next boot `last_state.bin` SHALL contain the previous (pre-crash) state
- **AND** the orphan `last_state.bin.tmp` SHALL be removed during boot cleanup

#### Scenario: Cold start with no state file proceeds with proveSinceTx = 0
- **WHEN** the exporter starts and `last_state.bin` is absent
- **THEN** the first `VerifiableSet` SHALL use `proveSinceTx: 0`
- **AND** the response SHALL be persisted before any further batch is dispatched
