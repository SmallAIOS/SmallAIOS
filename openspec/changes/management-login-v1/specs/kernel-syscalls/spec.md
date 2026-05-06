## ADDED Requirements

### Requirement: Five new auth syscalls
The kernel SHALL expose five new syscalls (numbers 43–47, incrementing the v0 ~46 count) using the existing kernel syscall ABI convention (pointers + lengths in successive arg registers, return value in the standard return register):

```text
auth_login(user_ptr, user_len, pass_ptr, pass_len,
           factor2_ptr, factor2_len) -> session_id | -errno
auth_logout() -> 0 | -errno
auth_change_password(old_ptr, old_len, new_ptr, new_len,
                     target_user_ptr, target_user_len) -> 0 | -errno
auth_create_user(user_ptr, user_len, role: u8,
                 initial_pass_ptr, initial_pass_len) -> 0 | -errno
auth_whoami(out_ptr) -> 0 | -errno
```

`auth_login.factor2_*` SHALL be empty/zero when the user is not enrolled in TOTP. `auth_change_password.target_user_*` SHALL be null/zero for self; non-null SHALL require `Role::Root`. `auth_whoami` SHALL fill an `out_ptr`-pointed `{ role: u8, user_id: u32, login_unix_time: u64, idle_seconds: u32 }` struct.

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
The shadow file SHALL be readable and writable **only** through the five new auth syscalls. User space SHALL NOT be able to map or read `/data/auth/shadow` directly even as root.

#### Scenario: Direct read denied
- **WHEN** any user-space code attempts `open("/data/auth/shadow", O_RDONLY)`
- **THEN** the open SHALL fail with `-EACCES`

#### Scenario: Direct mmap denied
- **WHEN** any user-space code attempts to mmap the shadow path
- **THEN** the mmap SHALL fail with `-EACCES`

## MODIFIED Requirements

### Requirement: Documented syscall count
The architecture documentation SHALL state the v1 syscall count as the prior v0 count plus five (i.e., the previous "~46 syscalls" is updated to "~51 syscalls"). New syscall numbers are stable: `auth_login=43`, `auth_logout=44`, `auth_change_password=45`, `auth_create_user=46`, `auth_whoami=47`.

#### Scenario: Architecture doc reflects new count
- **WHEN** `docs/architecture.md` is read
- **THEN** it SHALL state the syscall count of 51 and SHALL list the five new syscalls in the syscall table
