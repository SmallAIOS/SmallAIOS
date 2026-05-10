# kernel-syscalls Specification

## Purpose
TBD - created by archiving change management-login-v1. Update Purpose after archive.
## Requirements
### Requirement: Six new auth syscalls in a new Auth category
The kernel SHALL expose six new syscalls in a **new Auth category** (`0x90`–`0x95`), aligning with the existing SmallAIOS categorized scheme (Memory `0x00–0x0F`, Task `0x10–0x1F`, IPC `0x20–0x2F`, ONNX `0x30–0x3F`, Device `0x40–0x4F`, System `0x50–0x5F`, Capability `0x60–0x6F`, POSIX `0x70–0x8F`). `SYSCALL_TABLE_SIZE` is bumped from `0x90` to `0xA0` to make room (already pre-reserved by `wave0-scaffolding-stubs`); `0x96–0x9F` are reserved for future auth-related additions. Existing kernel syscall ABI convention applies (pointers + lengths in successive arg registers, return value in the standard return register):

```text
auth_login(user_ptr, user_len, pass_ptr, pass_len,
           factor2_ptr, factor2_len) -> session_id | -errno    // 0x90
auth_logout() -> 0 | -errno                                    // 0x91
auth_change_password(old_ptr, old_len, new_ptr, new_len,
                     target_user_ptr, target_user_len) -> 0 | -errno  // 0x92
auth_create_user(user_ptr, user_len, role: u8,
                 initial_pass_ptr, initial_pass_len) -> 0 | -errno    // 0x93
auth_whoami(out_ptr) -> 0 | -errno                             // 0x94
auth_totp_setup(user_ptr, user_len, secret_out_ptr) -> 0 | -errno     // 0x95 (RFC 6238 enrolment, opt-in per Q21)
```

`auth_login.factor2_*` SHALL be empty/zero when the user is not enrolled in TOTP. `auth_change_password.target_user_*` SHALL be null/zero for self; non-null SHALL require `Role::Root`. `auth_whoami` SHALL fill an `out_ptr`-pointed `{ role: u8, user_id: u32, login_unix_time: u64, idle_seconds: u32 }` struct. `auth_totp_setup` writes the new shared-secret bytes via `secret_out_ptr` and updates the shadow record's `totp_secret` field.

#### Scenario: Login returns session id on success
- **WHEN** `auth_login` is called with valid credentials
- **THEN** it SHALL return a positive `session_id`

#### Scenario: Login returns -EPERM on bad password
- **WHEN** `auth_login` is called with a known user and wrong password
- **THEN** it SHALL return `-EPERM`
- **AND** the response time SHALL be indistinguishable from "user does not exist"

#### Scenario: whoami fills the out struct
- **WHEN** an authenticated session calls `auth_whoami`
- **THEN** the out struct SHALL contain the caller's role, user_id, login time, and current idle seconds

### Requirement: Per-role idle-timeout sweeper
The kernel SHALL run a session-table sweeper that invalidates sessions whose idle window has elapsed (per `auth-roles`). A keypress on the originating surface (TTY or any authenticated Zenoh request) SHALL reset the idle clock. The sweeper SHALL append `auto_logout` audit records for every invalidated session.

#### Scenario: Idle session removed
- **WHEN** the sweeper runs and a Root session has been idle for >15 minutes
- **THEN** the session SHALL be removed from the table
- **AND** an `auto_logout` audit record SHALL be appended

### Requirement: Shadow file is reachable only through syscalls
The shadow file SHALL be readable and writable **only** through the six new auth syscalls. User space SHALL NOT be able to map or read `/data/auth/shadow` directly even as root.

#### Scenario: Direct read denied
- **WHEN** any user-space code attempts `open("/data/auth/shadow", O_RDONLY)`
- **THEN** the open SHALL fail with `-EACCES`

#### Scenario: Direct mmap denied
- **WHEN** any user-space code attempts to mmap the shadow path
- **THEN** the mmap SHALL fail with `-EACCES`

### Requirement: Documented syscall count
The architecture documentation SHALL state the post-`management-login-v1` syscall count as the prior count plus six. New syscall numbers are stable in the new Auth category: `AUTH_LOGIN=0x90`, `AUTH_LOGOUT=0x91`, `AUTH_CHANGE_PASSWORD=0x92`, `AUTH_CREATE_USER=0x93`, `AUTH_WHOAMI=0x94`, `AUTH_TOTP_SETUP=0x95`. `SYSCALL_TABLE_SIZE` SHALL be `0xA0` (already pre-reserved by `wave0-scaffolding-stubs`).

#### Scenario: Architecture doc reflects new count
- **WHEN** `docs/architecture.md` is read
- **THEN** it SHALL list the six new auth syscalls
- **AND** SHALL document the new Auth category at `0x90–0x9F`

### Requirement: New `boot_success` syscall
The kernel SHALL expose a new `boot_success` syscall (`SYS_BOOT_SUCCESS = 0x57`, in the existing System category). The syscall SHALL be callable only by `Role::Root`. Calling `boot_success` SHALL update the active boot config record (per `fs-ab-boot`) to clear `tentative` and set `boot_success = 1`. After successful return, the watchdog SHALL be disarmed.

```text
boot_success() -> 0 | -errno
```

The syscall SHALL be idempotent — calling it after the active record already shows `boot_success = 1` SHALL return `Ok(0)` without re-writing.

#### Scenario: boot_success commits tentative slot
- **WHEN** the active record has `tentative = 1, boot_success = 0`
- **AND** Root calls `boot_success`
- **THEN** the active record SHALL transition to `tentative = 0, boot_success = 1`
- **AND** the watchdog SHALL be disarmed
- **AND** an audit record `boot_success_committed` SHALL be appended

#### Scenario: Idempotent on already-committed
- **WHEN** Root calls `boot_success` on a record already at `boot_success = 1`
- **THEN** the syscall SHALL return `Ok(0)` without touching the record

#### Scenario: Non-Root denied
- **WHEN** an Operator or Viewer calls `boot_success`
- **THEN** the syscall SHALL return `-EPERM`
- **AND** an audit record `DENY:boot_success` SHALL be appended

### Requirement: Updated documented syscall count
The architecture documentation SHALL list one new System-category syscall: `SYS_BOOT_SUCCESS = 0x57`. The post-`embedded-filesystem-v1` syscall count is the post-`management-login-v1` count plus one.

#### Scenario: Architecture doc reflects new syscall
- **WHEN** `docs/architecture.md` is read
- **THEN** it SHALL list `SYS_BOOT_SUCCESS = 0x57` in the System-category syscall table

### Requirement: File syscalls operate on real backing
Existing file syscalls (`open`, `read`, `write`, `fsync`, `rename`, `stat`, `unlink`, `mkdir`, `rmdir`) SHALL operate against the new on-disk mounts (`/models/` squashfs, `/data/` F2FS) via `posix-vfs`. The syscall ABI SHALL NOT change. Behavior on the in-memory mounts (`/dev/`, `/proc/self/`) SHALL be unchanged.

#### Scenario: open(/data/x) creates a real file
- **WHEN** `open("/data/x", O_CREAT | O_WRONLY)` is called
- **THEN** the syscall SHALL return a valid fd
- **AND** the file SHALL appear in F2FS metadata
- **AND** SHALL persist across reboot if `fsync` is called

#### Scenario: read(/models/x) reads from squashfs
- **WHEN** `open("/models/some_model.onnx", O_RDONLY)` then `read(fd, ...)`
- **THEN** the read SHALL return the squashfs-decompressed bytes

#### Scenario: rename within /data/ is atomic
- **WHEN** `rename("/data/auth/shadow.tmp", "/data/auth/shadow")` is called
- **THEN** the syscall SHALL atomically replace the destination
- **AND** the previous content SHALL not be visible to any future open

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

### Requirement: Documented syscall count after overlay
The architecture documentation SHALL list two new ONNX-category syscalls: `ONNX_MODEL_ADD = 0x36` and `ONNX_MODEL_REMOVE = 0x37`. The post-`embedded-overlay-v1` syscall count is the post-`embedded-filesystem-v1` count plus two.

#### Scenario: Architecture doc reflects new syscalls
- **WHEN** `docs/architecture.md` is read
- **THEN** it SHALL list `ONNX_MODEL_ADD = 0x36` and `ONNX_MODEL_REMOVE = 0x37` in the ONNX-category syscall table

