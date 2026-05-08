## ADDED Requirements

### Requirement: model_add and model_remove syscalls
The kernel SHALL expose two new syscalls in the existing ONNX category (`0x30–0x3F`):

```text
model_add(name_ptr, name_len, contents_fd, expected_size) -> 0 | -errno
   ONNX_MODEL_ADD = 0x36 — Operator+
model_remove(name_ptr, name_len, mode: u8) -> 0 | -errno
   ONNX_MODEL_REMOVE = 0x37 — Root only
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
The architecture documentation SHALL list two new ONNX-category syscalls: `ONNX_MODEL_ADD = 0x36` and `ONNX_MODEL_REMOVE = 0x37`. The post-`embedded-overlay-v1` syscall count is the post-`embedded-filesystem-v1` count plus two.

#### Scenario: Architecture doc reflects new syscalls
- **WHEN** `docs/architecture.md` is read
- **THEN** it SHALL list `ONNX_MODEL_ADD = 0x36` and `ONNX_MODEL_REMOVE = 0x37` in the ONNX-category syscall table
