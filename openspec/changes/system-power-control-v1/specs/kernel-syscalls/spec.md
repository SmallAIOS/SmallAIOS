## ADDED Requirements

### Requirement: New `system_power` syscall

The kernel SHALL expose one new syscall (#46), `system_power`, wrapping the platform-specific power paths behind a single ABI entry:

```text
system_power(action: u8) -> 0 | -errno
```

where `action` is one of `{ REBOOT = 1, SHUTDOWN = 2, STATUS = 3 }`. `REBOOT` and `SHUTDOWN` SHALL be Root-only at the kernel boundary (the console and Zenoh layers re-check before invoking); `STATUS` SHALL be available to `Role::Viewer` per the `auth-roles` role-vs-syscall matrix. Any other `action` value SHALL be rejected with `-EINVAL` without side effects.

#### Scenario: Root reboot invokes the platform reset path

- **WHEN** a Root session calls `system_power(REBOOT)`
- **THEN** the kernel SHALL run the graceful sequence (scheduler shutdown, audit flush) and then invoke the platform reset path
- **AND** the syscall SHALL return `0` only on the shutdown/status paths that return at all

#### Scenario: Non-Root REBOOT or SHUTDOWN denied

- **WHEN** an Operator or Viewer session calls `system_power(REBOOT)` or `system_power(SHUTDOWN)`
- **THEN** the syscall SHALL return `-EPERM`
- **AND** no reset or power-off SHALL be initiated

#### Scenario: Viewer STATUS succeeds

- **WHEN** a Viewer session calls `system_power(STATUS)`
- **THEN** the syscall SHALL return `0` with the status payload (uptime, boot-slot, power-state, watchdog state)

#### Scenario: Unknown action rejected

- **WHEN** any session calls `system_power(4)`
- **THEN** the syscall SHALL return `-EINVAL`
- **AND** no power action SHALL be initiated
