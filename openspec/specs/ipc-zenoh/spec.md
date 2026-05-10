# ipc-zenoh Specification

## Purpose
TBD - created by archiving change management-login-v1. Update Purpose after archive.
## Requirements
### Requirement: Admin and metrics keyspace registration
The Zenoh subsystem SHALL register the keyspaces `smallaios/admin/**` (request/response queryables) and `smallaios/metrics/**` (publications) on session startup. The keyspaces SHALL be served by the `mgmt` crate's `Mgmt::serve(&Session)` entry point that `container/` invokes during boot. No new TLS or PSK configuration SHALL be required — both keyspaces SHALL reuse the existing PQC-backed Zenoh transport.

#### Scenario: Keyspaces visible after boot
- **WHEN** Zenoh has finished initialization
- **THEN** a peer SHALL see queryables under `smallaios/admin/*`
- **AND** the peer SHALL see publications under `smallaios/metrics/*`

### Requirement: Bearer-token authentication wrapper
Every request to `smallaios/admin/<verb>` (other than `login`) SHALL be wrapped in a kernel-side authentication layer that: (1) extracts the `token` field from the request body, (2) looks up the session in the kernel session table, (3) verifies the request's transport peer matches the session's recorded peer, (4) verifies the token has not idle-expired, (5) resets the idle clock, (6) dispatches to the verb handler with the resolved role.

A request that fails any of (1)–(4) SHALL be rejected with a structured error response (`{ ok: false, code, reason }`) before any verb-specific work runs.

#### Scenario: Missing token rejected
- **WHEN** an admin request omits the `token` field (or sets it to empty) on a non-login verb
- **THEN** the wrapper SHALL respond `{ ok: false, code: -EAUTH, reason: "Missing token" }`

#### Scenario: Expired token rejected
- **WHEN** the session has been idle past its per-role window
- **THEN** the wrapper SHALL respond `{ ok: false, code: -EAUTHEXPIRED, reason: "Session expired" }`
- **AND** SHALL NOT dispatch to the verb handler

#### Scenario: Peer mismatch rejected
- **WHEN** a request's transport peer does not match the recorded session peer
- **THEN** the wrapper SHALL respond `{ ok: false, code: -EPERM, reason: "Peer mismatch" }`
- **AND** an audit `DENY:peer_mismatch` record SHALL be appended

#### Scenario: Successful auth refreshes idle clock
- **WHEN** an admin request passes all checks
- **THEN** the wrapper SHALL reset the session's idle clock to zero
- **AND** SHALL invoke the verb handler with the resolved role from the session entry

