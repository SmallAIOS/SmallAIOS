## ADDED Requirements

### Requirement: Three-role taxonomy
The kernel SHALL recognize exactly three roles in v1: `Role::Root` (full access), `Role::Operator` (model lifecycle and read-only access), and `Role::Viewer` (read-only telemetry). Roles are encoded as a `u8` enum in the syscall ABI: 0 = `Root`, 1 = `Operator`, 2 = `Viewer`. Any other value SHALL be rejected with `-EINVAL`. Future roles MAY append (3, 4, ...) without breaking existing callers.

#### Scenario: Valid role values accepted
- **WHEN** `auth_create_user` is called with `role` ∈ {0, 1, 2}
- **THEN** the syscall SHALL succeed (subject to other preconditions)

#### Scenario: Unknown role rejected
- **WHEN** `auth_create_user` is called with `role = 7`
- **THEN** the syscall SHALL return `-EINVAL` and SHALL NOT mutate the shadow file

### Requirement: Role-vs-syscall partition
Every syscall SHALL declare a `min_role: Role` capability guard. The kernel SHALL reject syscalls whose caller's role does not meet `min_role` with `-EPERM`. The v1 partition is:

| Operation | Root | Operator | Viewer |
|-----------|:----:|:--------:|:------:|
| `auth_login` / `auth_whoami` / `auth_logout` | ✓ | ✓ | ✓ |
| `auth_change_password` (own) | ✓ | ✓ | ✓ |
| `auth_change_password` (target other) | ✓ | — | — |
| `auth_create_user` | ✓ | — | — |
| `model_load` / `model_unload` | ✓ | ✓ | — |
| `metrics_read` / `audit_read` (own) | ✓ | ✓ | ✓ |
| `audit_read` (all) | ✓ | — | — |
| `system_power(STATUS)` | ✓ | ✓ | ✓ |
| `system_power(REBOOT \| SHUTDOWN)` | ✓ | — | — |
| `system_update_*` | ✓ | — | — |
| Config write — `auth/*`, `mgmt/*`, `update/*` | ✓ | — | — |
| Config write — `network/*`, `automotive/*` | ✓ | — | — |
| Config write — `system.toml` | ✓ | — | — |

#### Scenario: Operator denied REBOOT
- **WHEN** an Operator session calls `system_power(REBOOT)`
- **THEN** the syscall SHALL return `-EPERM` and SHALL NOT initiate reboot

#### Scenario: Viewer denied model_load
- **WHEN** a Viewer session calls `model_load`
- **THEN** the syscall SHALL return `-EPERM`

#### Scenario: Operator allowed model_load
- **WHEN** an Operator session calls `model_load` with a valid model path
- **THEN** the syscall SHALL load the model and return success

### Requirement: min_role capability guard
The capability check SHALL be performed before any side-effectful work begins. The denial SHALL be a non-fatal error returned over the transport that originated the request (syscall return register, Zenoh response, future UDS reply).

#### Scenario: Denial happens before side effects
- **WHEN** a Viewer calls `model_load`
- **THEN** the syscall SHALL return `-EPERM` and the model registry SHALL NOT be touched

### Requirement: No latent service accounts
Only the `root` user SHALL exist after first-boot setup. `Operator` and `Viewer` accounts SHALL be created explicitly via `auth_create_user`. The kernel SHALL NOT pre-provision any account other than `root`.

#### Scenario: Fresh first-boot has only root
- **WHEN** first-boot setup completes
- **THEN** `auth_whoami` for any user other than `root` SHALL fail with `-ENOENT`

### Requirement: Per-role idle auto-logout
Active sessions SHALL be auto-logged-out after a per-role idle window. Defaults: `Root` 15 minutes, `Operator` 60 minutes, `Viewer` 60 minutes. Any keypress on the originating surface SHALL reset the timer. Thresholds SHALL be configurable via `mgmt/policy.toml` keys `idle.root_minutes`, `idle.operator_minutes`, `idle.viewer_minutes`.

#### Scenario: Idle Root session expires after 15 minutes
- **WHEN** a Root session has no activity for 15 minutes
- **THEN** the session SHALL be invalidated and an audit record `auto_logout` SHALL be appended

#### Scenario: Keypress resets the idle timer
- **WHEN** a Viewer session running `console-monitor-v1` receives a keypress at minute 59 of a 60-minute window
- **THEN** the timer SHALL reset to zero and the session SHALL remain valid

### Requirement: Cross-target password change requires Root
`auth_change_password` MAY accept a `target_user` argument; non-null `target_user` SHALL require the caller to have `Role::Root`. A null `target_user` means the caller's own password and is allowed for every authenticated role.

#### Scenario: Operator cannot change another user's password
- **WHEN** an Operator calls `auth_change_password` with `target_user="alice"`
- **THEN** the syscall SHALL return `-EPERM`

#### Scenario: Root changes another user's password
- **WHEN** a Root session calls `auth_change_password` with `target_user="alice"` and a valid new password
- **THEN** alice's hash SHALL be updated and `must_change_password_on_login` SHALL be set on alice

### Requirement: must-change-password gating
When the caller's `flags` bit 0 (`must_change_password_on_login`) is set, the kernel SHALL only honor `auth_change_password`, `auth_whoami`, and `auth_logout`. Every other syscall SHALL return `-EAUTHEXPIRED`.

#### Scenario: Forced rotation blocks model_load
- **WHEN** a freshly-created user with `must_change_password_on_login` set calls `model_load`
- **THEN** the syscall SHALL return `-EAUTHEXPIRED`

#### Scenario: Forced rotation allows password change
- **WHEN** the same user calls `auth_change_password` with valid old/new passwords
- **THEN** the syscall SHALL succeed and the flag SHALL be cleared
