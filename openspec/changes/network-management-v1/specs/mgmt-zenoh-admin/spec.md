## ADDED Requirements

### Requirement: Network configuration admin keyspace

The admin keyspace SHALL gain `smallaios/admin/network/<iface>` (including bond devices) for network configuration CRUD, handled in `container/src/mgmt_network.rs` and reusing the existing JSON request/response envelope and token authentication. Writes SHALL apply on commit and SHALL roll back on apply failure, following the same atomic-rewrite pattern as the shadow file.

#### Scenario: Committed write applies and persists

- **WHEN** an authenticated client writes a valid configuration to `smallaios/admin/network/eth0`
- **THEN** the response SHALL be `{ "ok": true, ... }`
- **AND** the configuration SHALL be applied to `eth0` live
- **AND** `/data/network/eth0.toml` SHALL be rewritten atomically with the new contents

#### Scenario: Failed apply rolls back

- **WHEN** an authenticated client commits a network configuration that fails to apply
- **THEN** the response SHALL be `{ "ok": false, "code": <negative-errno>, "reason": "<user-readable>" }`
- **AND** the interface SHALL be restored to its previous configuration
- **AND** `/data/network/<iface>.toml` SHALL retain its previous contents

#### Scenario: Read returns the active configuration

- **WHEN** an authenticated client queries `smallaios/admin/network/eth0`
- **THEN** the response payload SHALL contain the interface's current configuration
