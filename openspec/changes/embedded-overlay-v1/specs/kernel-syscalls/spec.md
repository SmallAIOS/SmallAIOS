## ADDED Requirements

### Requirement: model_add and model_remove syscalls
The kernel SHALL expose two new syscalls:

```text
model_add(name_ptr, name_len, contents_fd, expected_size) -> 0 | -errno
   number 49 — Operator+
model_remove(name_ptr, name_len, mode: u8) -> 0 | -errno
   number 50 — Root only
```

These syscalls implement the operator-facing API for the overlay layer specified in `fs-overlay-syscalls`. They SHALL participate in the existing `min_role` capability system from `auth-roles`.

#### Scenario: model_add ABI matches existing convention
- **WHEN** `model_add` is invoked
- **THEN** arguments SHALL pass via the existing kernel syscall convention
- **AND** the return value SHALL be 0 on success or a negative POSIX errno on failure

#### Scenario: model_remove ABI matches existing convention
- **WHEN** `model_remove` is invoked
- **THEN** arguments SHALL pass via the existing kernel syscall convention
- **AND** the return value SHALL be 0 on success or a negative POSIX errno on failure

## MODIFIED Requirements

### Requirement: Documented syscall count after overlay
The architecture documentation SHALL state the v1-with-overlay syscall count as the prior count plus two (i.e., the post-`embedded-filesystem-v1` count of "~52 syscalls" is updated to "~54 syscalls"). New syscall numbers are stable: `model_add = 49`, `model_remove = 50`.

#### Scenario: Architecture doc reflects new count
- **WHEN** `docs/architecture.md` is read
- **THEN** it SHALL state the syscall count of 54
- **AND** SHALL list `model_add = 49` and `model_remove = 50` in the syscall table
