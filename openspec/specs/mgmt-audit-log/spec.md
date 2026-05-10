# mgmt-audit-log Specification

## Purpose
TBD - created by archiving change management-login-v1. Update Purpose after archive.
## Requirements
### Requirement: Audit record fields
Every audit record SHALL have the fields `{ ts, who, surface, action, before, after, code, prev_hash, hash }`. `ts` is a UNIX nanosecond timestamp. `who` is the user identity (`root`, an Operator name, a Viewer name, or `kernel` for system events). `surface` is one of `tty`, `zenoh`, `toml`, `kernel`. `action` is the verb (`auth_login`, `auth_logout`, `model_load`, `config_write`, `DENY:<syscall>`, `auto_logout`, `firstboot_complete`, etc.). `before` and `after` are JSON values for config writes (null for non-write actions). `code` is the result code (`0` on success, negative POSIX errno on failure).

#### Scenario: Successful login record
- **WHEN** Root logs in via TTY at time t
- **THEN** the audit ring SHALL contain `{ ts: t, who: "root", surface: "tty", action: "auth_login", code: 0, ... }`

#### Scenario: Config write captures before/after
- **WHEN** Root writes `metrics.cpu.interval_ms` from 1000 to 500
- **THEN** the audit record SHALL contain `before: 1000, after: 500`

### Requirement: In-memory ring with periodic flush
The audit ring SHALL be a fixed-size in-memory buffer of 16 MiB (4 KiB × 4096 records). A background flusher SHALL append new records to `/data/audit/log.jsonl` every 1 second or immediately on auth events (`auth_login`, `auth_logout`, `auto_logout`, `auth_create_user`, `auth_change_password`).

#### Scenario: Flush cadence under steady load
- **WHEN** records are appended at ~10/s
- **THEN** the on-disk file SHALL gain those records within 1100 ms of their creation

#### Scenario: Auth event flushes immediately
- **WHEN** an `auth_login` record is appended
- **THEN** the on-disk file SHALL contain that record before the next non-auth syscall completes

### Requirement: Hybrid rotation with hard failsafe
The audit log SHALL rotate based on the first of two thresholds: size (`audit.rotate_size_bytes`, default 64 MiB) or age (`audit.rotate_age_hours`, default 24). Rotated archives SHALL be gzipped to `log.<n>.jsonl.gz`. The most recent `audit.keep_archives` archives SHALL be retained (default 8); older archives SHALL be deleted on rotation.

Independently, the system SHALL enforce `audit.max_total_disk_bytes` (default 512 MiB) as a hard failsafe. When the total disk usage of the active log plus archives exceeds the failsafe, the oldest archive SHALL be evicted regardless of `keep_archives` to guarantee the writer never blocks on a full disk.

#### Scenario: Size threshold rotates
- **WHEN** `log.jsonl` exceeds 64 MiB
- **THEN** it SHALL be renamed and gzipped to `log.1.jsonl.gz` and a new empty `log.jsonl` SHALL be created
- **AND** existing archives SHALL be shifted (`log.1.jsonl.gz` → `log.2.jsonl.gz`, etc.)

#### Scenario: Failsafe evicts beyond keep_archives
- **WHEN** the total of `log.jsonl` plus archives exceeds 512 MiB and `keep_archives = 16` is configured
- **THEN** the oldest archive SHALL be deleted
- **AND** the writer SHALL never block on a "no space" condition

#### Scenario: Age threshold rotates with low traffic
- **WHEN** 24 hours pass since last rotation and the log is only 2 MiB
- **THEN** rotation SHALL occur on the next scheduled check

### Requirement: SHA-3-256 hash chain
Every record SHALL include `prev_hash` (the SHA-3-256 hash of the immediately preceding record's serialized form, or 32 zero bytes for the first record) and `hash` (the SHA-3-256 hash of this record's serialized form computed over all fields except `hash`). The latest chain head SHALL be exposed via `audit_read` and as a streamed metric on `smallaios/metrics/audit_fingerprint`.

#### Scenario: Chain head changes on append
- **WHEN** a new record is appended
- **THEN** the next `smallaios/metrics/audit_fingerprint` publication SHALL contain the new chain head

#### Scenario: External auditor verifies chain
- **WHEN** an external auditor reads the JSONL file and walks the chain
- **THEN** every record's `hash` SHALL be reproducible from its serialized fields
- **AND** every record's `prev_hash` SHALL match the previous record's `hash`

### Requirement: Optional ML-DSA-65 signed checkpoints
When `audit.signed_checkpoints.enabled = true` is set in `mgmt/policy.toml`, the kernel SHALL sign every Nth chain head with the kernel-held ML-DSA-65 key, where N is `audit.signed_checkpoints.interval` (default 1024). Signed checkpoints SHALL be appended as records with `action = "checkpoint"` and a `signature` field holding the signature bytes.

#### Scenario: Signed checkpoint emitted at interval
- **WHEN** signed checkpoints are enabled with interval 1024
- **AND** the 1024th, 2048th, ... record is appended
- **THEN** a checkpoint record SHALL be appended immediately after, with a valid ML-DSA-65 signature over the chain head

#### Scenario: Off-box verifier detects tampering
- **WHEN** an attacker rewrites a record between two checkpoints
- **THEN** an external verifier with the public key SHALL detect signature mismatch on the next checkpoint and report tampering

### Requirement: Denial audit with rate limit
Every `min_role` denial SHALL be appended to the audit ring with `action = "DENY:<syscall>"` and `code = -EPERM`. To prevent log flooding, denials from a single user SHALL be rate-limited to 10 per second; excess denials SHALL be coalesced into a single `action = "DENY_BURST"` record with a count.

#### Scenario: Single denial recorded
- **WHEN** an Operator calls `system_power(REBOOT)`
- **THEN** an audit record `{ who: "<op>", action: "DENY:system_power", code: -1 }` SHALL be appended

#### Scenario: Rapid denials coalesced
- **WHEN** an automated client triggers 200 denials in 1 second
- **THEN** the first 10 SHALL be recorded individually
- **AND** the remaining 190 SHALL be coalesced into one `DENY_BURST` record with `count: 190`

