## ADDED Requirements

### Requirement: Three System Power Verbs

The management surface SHALL expose three power-control verbs. `system_reboot()` SHALL perform a graceful shutdown of the inference scheduler, flush the audit log, and then invoke the platform reset path. `system_shutdown()` SHALL follow the same sequence but halt at the lowest power state the platform supports without resuming (PSCI `SYSTEM_OFF` on AArch64, ACPI S5 on x86-64). `system_status()` SHALL be read-only and report uptime, boot-slot, power-state, and watchdog state. Inference-scheduler draining SHALL be bounded by a fixed 10-second ceiling; graceful draining beyond that ceiling is out of scope for v1.

#### Scenario: Reboot drains, flushes, then resets

- **WHEN** an authenticated Root session invokes `system_reboot()`
- **THEN** the inference scheduler SHALL be shut down gracefully
- **AND** the audit log SHALL be flushed to disk
- **AND** only after both steps complete SHALL the platform reset path be invoked

#### Scenario: Shutdown halts at the lowest power state

- **WHEN** an authenticated Root session invokes `system_shutdown()`
- **THEN** the platform SHALL enter the lowest power state it supports without resuming (PSCI `SYSTEM_OFF` on AArch64, ACPI S5 on x86-64)
- **AND** the machine SHALL NOT resume without an external power cycle

#### Scenario: Status is read-only and viewer-visible

- **WHEN** a `Role::Viewer` session invokes `system_status()`
- **THEN** the response SHALL contain uptime, boot-slot, power-state, and watchdog state
- **AND** no system state SHALL be mutated by the call

#### Scenario: Drain ceiling of 10 seconds

- **WHEN** `system_reboot()` is invoked while an inference job is still running 10 seconds after draining began
- **THEN** the reboot SHALL proceed once the 10-second ceiling elapses
- **AND** the final audit record SHALL still be flushed to disk before the platform call

### Requirement: Root-Only Power Actions

`system_reboot()` and `system_shutdown()` SHALL be invocable only from an authenticated `Role::Root` session; `system_status()` SHALL be available to `Role::Viewer` and above. The console and Zenoh surfaces SHALL each re-check the caller's role before invoking the `system_power` syscall, and the kernel SHALL enforce the same gate at the syscall boundary, so that `system_power(REBOOT | SHUTDOWN)` cannot be reached without an authenticated Root session.

#### Scenario: Operator reboot rejected at the surface

- **WHEN** an Operator session issues `reboot` via the console shell or `smallaios/admin/system/reboot` via Zenoh
- **THEN** the surface SHALL reject the request before the `system_power` syscall is invoked
- **AND** no reset SHALL occur

#### Scenario: Unauthenticated request cannot reach system_power

- **WHEN** a request carrying no valid authenticated session attempts any power verb on any surface
- **THEN** the request SHALL be rejected
- **AND** the `system_power` syscall SHALL NOT be invoked

#### Scenario: Viewer may read status

- **WHEN** a `Role::Viewer` session invokes `system_status()`
- **THEN** the call SHALL succeed and return the status payload

### Requirement: Confirmation-Nonce Protocol for Remote Power Actions

Remote (Zenoh) reboot and shutdown SHALL use a two-step confirmation protocol: the client SHALL first obtain a fresh confirmation nonce via a GET on the corresponding nonce key (e.g. `smallaios/admin/system/reboot/nonce`), then present that nonce together with its bearer token in the actual reboot or shutdown request. Nonces SHALL be single-use and SHALL expire, so that a stale token alone cannot trigger a reboot during a network partition.

#### Scenario: Two-step reboot with fresh nonce succeeds

- **WHEN** a Root client GETs `smallaios/admin/system/reboot/nonce` and then sends its bearer token plus the returned nonce to `smallaios/admin/system/reboot`
- **THEN** the reboot SHALL be accepted and executed

#### Scenario: Request without a nonce rejected

- **WHEN** a Root client sends a valid bearer token to `smallaios/admin/system/reboot` without a prior nonce fetch
- **THEN** the request SHALL be rejected with an error response
- **AND** no reset SHALL occur

#### Scenario: Expired nonce rejected

- **WHEN** a client presents a confirmation nonce after its expiry window has elapsed
- **THEN** the request SHALL be rejected with an error response
- **AND** no reset SHALL occur

#### Scenario: Nonce is single-use

- **WHEN** a confirmation nonce has already been consumed by one power request
- **AND** a second request presents the same nonce
- **THEN** the second request SHALL be rejected
- **AND** no reset SHALL occur

### Requirement: Console Confirmation Prompt

The post-login console shell (the command parser added by `management-login-v1`) SHALL provide `reboot` and `shutdown` commands. Both SHALL prompt `Confirm? [y/N]` before executing, with No as the default. No command-line flag SHALL bypass the prompt.

#### Scenario: Confirmed console reboot proceeds

- **WHEN** a Root console session enters `reboot` and answers `y` to the `Confirm? [y/N]` prompt
- **THEN** the reboot SHALL execute

#### Scenario: Default answer aborts

- **WHEN** a Root console session enters `shutdown` and answers with Enter or any input other than `y`
- **THEN** the command SHALL abort
- **AND** no power action SHALL occur

#### Scenario: No flag override exists

- **WHEN** `reboot` or `shutdown` is entered with any flag or argument intended to skip confirmation
- **THEN** the shell SHALL still present the `Confirm? [y/N]` prompt before executing

### Requirement: Power Audit Record Survives the Reset

Every successful `reboot` or `shutdown` SHALL write a final record to the in-kernel audit ring containing `who`, `when`, `transport`, and `nonce`. The record SHALL be persisted to disk before the platform power call returns, so it survives the reset. On the next boot, the first telemetry publish SHALL report whether the last shutdown was clean and, if so, which user initiated it via which transport.

#### Scenario: Audit record persisted before the platform call

- **WHEN** a Root user triggers `system_reboot()` via Zenoh
- **THEN** an audit record with `who`, `when`, `transport`, and `nonce` SHALL be written to disk
- **AND** the platform reset SHALL be invoked only after the record is durable

#### Scenario: Next boot reports the clean shutdown

- **WHEN** the system boots after a reboot initiated by user X via transport Y
- **THEN** the first telemetry publish SHALL include that the last shutdown was clean, initiated by user X via transport Y
