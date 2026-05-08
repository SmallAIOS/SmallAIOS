## ADDED Requirements

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

## MODIFIED Requirements

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
