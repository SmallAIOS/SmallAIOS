# mgmt-zenoh-admin Specification

## Purpose
TBD - created by archiving change management-login-v1. Update Purpose after archive.
## Requirements
### Requirement: Admin keyspace contract
The kernel SHALL expose a Zenoh queryable namespace `smallaios/admin/<verb>` for management operations. v1 verbs SHALL include `login`, `logout`, `whoami`, `passwd`, `users/add`, `users/list`, `heartbeat`. Request and response payloads SHALL be JSON with the canonical shape:

```json
// request
{ "token": "<opaque-id-or-empty-on-login>", "args": { ... } }
// response (success)
{ "ok": true, "payload": { ... } }
// response (error)
{ "ok": false, "code": <negative-errno>, "reason": "<user-readable>" }
```

The `code` field SHALL be the same numeric POSIX-aligned errno used at the syscall boundary so a single shared `error_string()` table maps errors uniformly across surfaces.

#### Scenario: Successful login returns opaque token
- **WHEN** a client sends `{ "args": { "user": "root", "pass": "<correct>" } }` to `smallaios/admin/login`
- **THEN** the response SHALL be `{ "ok": true, "payload": { "token": "<16-byte-base64>", "role": "root", "expires_in": 900 } }`

#### Scenario: Login with wrong password
- **WHEN** a client sends a wrong password to `smallaios/admin/login`
- **THEN** the response SHALL be `{ "ok": false, "code": -1, "reason": "Authentication failed" }` (POSIX `EPERM`)
- **AND** the response time SHALL be indistinguishable from "user does not exist" to prevent enumeration

### Requirement: Bearer-token lifecycle
By default, the kernel SHALL issue an opaque random 16-byte token on `auth/login` and SHALL store the session in the kernel session table. The token SHALL be the only credential needed for subsequent admin requests during its lifetime. Tokens SHALL expire after the per-role idle window from `auth-roles`. Each authenticated request SHALL reset the idle clock (sliding TTL). Once a request has entered the kernel and started executing, its token SHALL be considered live for the duration of that call (no mid-call expiry).

The cargo features `mgmt-token-mldsa` (signed JWT-style with ML-DSA-65) and `mgmt-token-ed25519-legacy` (signed JWT-style with Ed25519) SHALL be available as opt-in alternatives. When `mgmt-token-ed25519-legacy` is disabled at compile time, no Ed25519 signing or verification code SHALL be linked into the production binary.

A `smallaios/admin/heartbeat` verb SHALL accept any valid token and SHALL reset the idle clock without performing any other action.

An optional two-tier (access + refresh) mode SHALL be available via `mgmt/policy.toml` for orchestration-class clients. The access token SHALL have a short configurable TTL; the refresh token SHALL have a longer configurable TTL and SHALL only be acceptable on `smallaios/admin/refresh`.

#### Scenario: Sliding TTL refreshes on use
- **WHEN** a Root client sends an admin request 14 minutes into a 15-minute window
- **THEN** the session idle timer SHALL reset to zero
- **AND** the session SHALL remain valid

#### Scenario: Heartbeat refreshes idle clock
- **WHEN** a UI sends `smallaios/admin/heartbeat` with a valid token
- **THEN** the response SHALL be `{ "ok": true, "payload": { "expires_in": <seconds> } }`
- **AND** the idle clock SHALL be reset

#### Scenario: Long-running call survives expiry mid-flight
- **WHEN** a client invokes a verb that takes 5 seconds and the token would idle-expire 1 second into the call
- **THEN** the call SHALL complete successfully
- **AND** the response SHALL be returned before the session is invalidated

#### Scenario: Ed25519 legacy excluded at compile time
- **WHEN** the binary is built without `mgmt-token-ed25519-legacy`
- **THEN** no symbol from any Ed25519 signing or verification routine SHALL appear in the binary

### Requirement: Peer-identity binding
On `auth/login` over Zenoh, the kernel SHALL record the peer's TLS-1.3 certificate fingerprint or PSK identity in the session entry. Subsequent admin requests SHALL be rejected with `-EPERM` if the request's transport peer does not match the session's recorded peer.

#### Scenario: Token replay from different peer rejected
- **WHEN** peer A receives a token via login
- **AND** peer B sends an admin request bearing peer A's token
- **THEN** the response SHALL be `{ "ok": false, "code": -1, "reason": "Peer mismatch" }`
- **AND** an audit `DENY:peer_mismatch` record SHALL be appended

### Requirement: Per-identity and total session caps
The kernel SHALL allow at most 4 concurrent admin sessions per remote PQC identity and at most 16 total Zenoh-backed sessions across all peers (out of the 32-slot kernel session table). Excess login requests SHALL fail with `-EAGAIN` and a `Retry-After` field. These caps SHALL be configurable in `mgmt/policy.toml` under `mgmt.zenoh.per_identity_max` and `mgmt.zenoh.total_max`.

#### Scenario: Fifth concurrent session from same identity rejected
- **WHEN** a single remote identity already holds 4 sessions
- **AND** that identity issues another `login` request
- **THEN** the response SHALL be `{ "ok": false, "code": -11, "reason": "Per-identity session cap reached" }` (POSIX `EAGAIN`)

#### Scenario: 17th total Zenoh session rejected
- **WHEN** 16 Zenoh sessions are currently active across multiple identities
- **AND** another identity issues `login`
- **THEN** the response SHALL fail with `EAGAIN`

