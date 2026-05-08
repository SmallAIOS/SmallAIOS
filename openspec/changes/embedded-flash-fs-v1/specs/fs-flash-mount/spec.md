## ADDED Requirements

### Requirement: /flash/ VFS mount
The VFS SHALL gain a `/flash/` mount point when the `fs-flash` cargo feature is enabled AND a flash device is enumerated by the kernel's per-arch flash discovery. The mount SHALL happen during kernel boot before any service that depends on flash-resident files (e.g., the secure-config loader). When either condition is not met, the `/flash/` path SHALL not exist; opening it SHALL return `-ENOENT`.

#### Scenario: Feature on, device present, /flash/ available
- **WHEN** the kernel is built with `fs-flash` and a QSPI NOR is enumerated
- **THEN** `/flash/` SHALL be mounted as a littlefs filesystem
- **AND** subsequent file opens under `/flash/` SHALL succeed

#### Scenario: Feature on, no device, /flash/ not present
- **WHEN** the kernel is built with `fs-flash` but no flash device is enumerated
- **THEN** `/flash/` SHALL NOT be mounted
- **AND** `open("/flash/foo", ...)` SHALL return `-ENOENT`
- **AND** an info-level log SHALL note "no flash device, /flash/ not mounted"

#### Scenario: Feature off, /flash/ never present
- **WHEN** the kernel is built without `fs-flash`
- **THEN** the `/flash/` mount path SHALL NOT exist in the VFS
- **AND** even if a flash device is physically present, no mount SHALL be attempted

### Requirement: Coexistence with /data/ on F2FS
When both `/data/` (F2FS, per `embedded-filesystem-v1`) and `/flash/` (littlefs) are mounted, they SHALL operate independently. Files in one SHALL NOT shadow or mirror files in the other. There SHALL be no automatic synchronization between the two substrates.

By convention, the canonical locations SHALL be:
- `/data/auth/shadow` — single source of truth for user/role table (F2FS).
- `/data/audit/log.jsonl` — single source of truth for audit log (F2FS).
- `/data/mgmt/policy.toml` — single source of truth for runtime configuration (F2FS).
- `/flash/secrets/` — high-assurance content (boot keys, attestation state, ML-DSA-65 signing keys for the update pipeline). Independent from `/data/`.
- `/flash/secure-config/` — small power-fail-critical configuration that must survive even total `/data/` loss.

Targets without a block device (no `/data/`) SHALL store the canonical `auth/`, `audit/`, `mgmt/` content under `/flash/` directly. The convention is "F2FS where available, fall back to littlefs."

#### Scenario: Both mounted, distinct content
- **WHEN** the system has eMMC (F2FS `/data/`) and QSPI NOR (littlefs `/flash/`) both mounted
- **THEN** `/data/auth/shadow` SHALL exist on F2FS
- **AND** `/flash/secrets/update-key.pub` SHALL exist on littlefs
- **AND** the two SHALL be entirely independent

#### Scenario: Flash-only target falls back gracefully
- **WHEN** the system has only flash (no block device for /data/)
- **THEN** the canonical paths (`auth/`, `audit/`, `mgmt/`) SHALL be located under `/flash/` directly
- **AND** application code SHALL use the existing path lookup logic (no separate flash-only branch)

### Requirement: Boot-cleanup on mount
On mount of `/flash/`, the kernel SHALL run littlefs's natural orphan-recovery (sweep partial commits left over from a power loss). This is not a separate fsck — it is part of mount. Power-loss resilience proof: any sequence of program/erase calls followed by power loss SHALL produce a mountable FS with no data loss past the last successful `fsync`.

#### Scenario: Mount after power loss recovers cleanly
- **WHEN** the system loses power mid-write and reboots
- **THEN** mount of `/flash/` SHALL succeed
- **AND** all `fsync`-acknowledged data SHALL be intact
- **AND** any partial commits SHALL be rolled back

#### Scenario: Mount logs recovery activity
- **WHEN** mount detects and rolls back a partial commit
- **THEN** an audit record `flash_mount_recovery{ rolled_back_writes: <n> }` SHALL be appended
