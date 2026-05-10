## ADDED Requirements

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
