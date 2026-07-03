## ADDED Requirements

### Requirement: Update verbs join the admin keyspace

The admin namespace SHALL be extended with the `smallaios/admin/update/**` keyspace: `update/begin`, `update/chunk/<session>/<index>`, `update/commit/<session>`, and `update/abort/<session>`. These verbs SHALL sit under the existing `management-login-v1` admin tree and SHALL be subject to the same bearer-token authentication, session, and audit rules as the existing admin verbs.

#### Scenario: Update verbs require an authenticated session

- **WHEN** a client sends a request to `smallaios/admin/update/begin` without a valid session token
- **THEN** the request SHALL be rejected using the established admin error shape (`{ "ok": false, ... }`)
- **AND** no update session SHALL be created

#### Scenario: Authenticated begin succeeds

- **WHEN** an authenticated client sends image manifest metadata to `smallaios/admin/update/begin`
- **THEN** the response SHALL be `{ "ok": true, ... }` carrying an opaque session id and the chunk size

### Requirement: `smallaios/admin/system/healthy` boot-good endpoint

The admin namespace SHALL expose `smallaios/admin/system/healthy`, the Zenoh equivalent of `system_update_confirm()` for remote operators to mark a pending boot good. A successful response to a `healthy` ping while an update is pending SHALL clear the boot pointer's `pending` field and reset `tries_remaining`.

#### Scenario: Healthy ping confirms a pending update

- **WHEN** an update is pending and a remote operator's ping to `smallaios/admin/system/healthy` receives a successful response within the confirm window
- **THEN** `pending` SHALL be cleared and `tries_remaining` reset
- **AND** the effect SHALL be identical to a `system_update_confirm()` call
