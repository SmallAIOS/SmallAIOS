## ADDED Requirements

### Requirement: System power admin keyspace

The Zenoh admin surface SHALL add a `smallaios/admin/system/**` keyspace, gated on the auth, roles, and token machinery from `management-login-v1`:

- `smallaios/admin/system/reboot` — request/response; the request body SHALL carry the bearer token plus a fresh confirmation nonce previously returned by a GET on `smallaios/admin/system/reboot/nonce`.
- `smallaios/admin/system/shutdown` — same two-step token-plus-nonce pattern with its own nonce key.
- `smallaios/admin/system/status` — single-shot query, available to `Role::Viewer` and above.

The two-step nonce pattern exists to prevent a stale token from triggering a reboot during a network partition.

#### Scenario: Two-step Zenoh reboot succeeds for Root

- **WHEN** a Root client GETs `smallaios/admin/system/reboot/nonce`
- **AND** then sends its bearer token plus the returned nonce to `smallaios/admin/system/reboot`
- **THEN** the request SHALL be accepted and the reboot executed

#### Scenario: Reboot without prior nonce fetch rejected

- **WHEN** a Root client sends only its bearer token to `smallaios/admin/system/reboot` without a fresh nonce
- **THEN** the response SHALL be an error
- **AND** no reset SHALL occur

#### Scenario: Viewer queries system status

- **WHEN** a Viewer client sends a single-shot query with a valid token to `smallaios/admin/system/status`
- **THEN** the response SHALL report uptime, boot-slot, power-state, and watchdog state

#### Scenario: Viewer denied on the reboot key

- **WHEN** a Viewer client completes the nonce fetch and sends a reboot request to `smallaios/admin/system/reboot`
- **THEN** the response SHALL be a permission error
- **AND** the `system_power` syscall SHALL NOT be invoked
