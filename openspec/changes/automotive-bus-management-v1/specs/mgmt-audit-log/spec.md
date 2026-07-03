## MODIFIED Requirements

### Requirement: Audit record fields
Every audit record SHALL have the fields `{ ts, who, surface, action, before, after, code, prev_hash, hash }`. `ts` is a UNIX nanosecond timestamp. `who` is the user identity (`root`, an Operator name, a Viewer name, or `kernel` for system events). `surface` is one of `tty`, `zenoh`, `toml`, `kernel`, `uds`. `action` is the verb (`auth_login`, `auth_logout`, `model_load`, `config_write`, `DENY:<syscall>`, `auto_logout`, `firstboot_complete`, etc.). `before` and `after` are JSON values for config writes (null for non-write actions). `code` is the result code (`0` on success, negative POSIX errno on failure).

#### Scenario: Successful login record
- **WHEN** Root logs in via TTY at time t
- **THEN** the audit ring SHALL contain `{ ts: t, who: "root", surface: "tty", action: "auth_login", code: 0, ... }`

#### Scenario: Config write captures before/after
- **WHEN** Root writes `metrics.cpu.interval_ms` from 1000 to 500
- **THEN** the audit record SHALL contain `before: 1000, after: 500`

#### Scenario: UDS Security Access login records surface uds
- **WHEN** a tester completes a `0x27 Security Access` seed/key exchange that bridges to the `auth_login` syscall
- **THEN** the audit ring SHALL contain a record with `surface: "uds"` and `action: "auth_login"`

#### Scenario: Same write over Zenoh and UDS differs only in surface
- **WHEN** the same configuration write is performed once over Zenoh and once over UDS
- **THEN** both SHALL append audit records of identical shape
- **AND** the records SHALL differ only in the `surface` field (`zenoh` vs `uds`)
