## ADDED Requirements

### Requirement: On-disk mount points
The VFS SHALL gain two on-disk mount points alongside the existing in-memory tree:

- `/models/` — backed by the active squashfs slot (read-only, per `fs-squashfs-readonly`).
- `/data/` — backed by the F2FS partition (read-write, per `fs-f2fs-readwrite`).

The existing in-memory mounts (`/dev/`, `/proc/self/`) SHALL remain unchanged. Mount points SHALL be fixed in kernel code (compile-time constants); they SHALL NOT be overrideable from a config file in v1.

#### Scenario: /models/ resolves to squashfs
- **WHEN** any path under `/models/` is opened
- **THEN** the open SHALL be routed to the squashfs reader for the active slot

#### Scenario: /data/ resolves to F2FS
- **WHEN** any path under `/data/` is opened
- **THEN** the open SHALL be routed to the F2FS implementation

### Requirement: BlockError to errno translation
The VFS SHALL convert internal `BlockError` enum values (per `fs-block-device`) into POSIX-aligned errno at the syscall boundary:

| BlockError | errno |
|------------|-------|
| MediaError | -EIO |
| NotPresent | -ENXIO |
| Timeout | -ETIMEDOUT |
| BadCrc | -EIO |
| Unaligned | -EINVAL |
| OutOfRange | -EINVAL |
| DeviceBusy | -EBUSY |

#### Scenario: Block read timeout returns -ETIMEDOUT
- **WHEN** a `read()` syscall on a `/models/` file triggers a `BlockError::Timeout`
- **THEN** the syscall SHALL return `-ETIMEDOUT`

#### Scenario: Block CRC failure returns -EIO
- **WHEN** a `read()` syscall on a `/data/` file triggers a `BlockError::BadCrc`
- **THEN** the syscall SHALL return `-EIO`

## MODIFIED Requirements

### Requirement: VFS write path returns real errors instead of EROFS
The VFS SHALL no longer return `-EROFS` unconditionally on writes. Writes to the in-memory mounts (`/dev/`, `/proc/self/`) SHALL continue to return `-EROFS`. Writes to `/models/` SHALL return `-EROFS` (squashfs is read-only). Writes to `/data/` SHALL be passed through to the F2FS implementation and SHALL return whatever real errno the F2FS layer produces (`-ENOSPC`, `-EIO`, `-EACCES`, etc.) — including `Ok(0..)` on success.

#### Scenario: Write to /models/ still returns -EROFS
- **WHEN** any write syscall targets a path under `/models/`
- **THEN** the syscall SHALL return `-EROFS`

#### Scenario: Write to /dev/null succeeds (existing behavior)
- **WHEN** a write targets `/dev/null`
- **THEN** the syscall SHALL return `Ok(write_len)` per existing in-memory behavior

#### Scenario: Write to /data/ returns real F2FS error
- **WHEN** a write to `/data/audit/log.jsonl` exhausts free space
- **THEN** the syscall SHALL return `-ENOSPC`
- **AND** SHALL NOT return `-EROFS`

#### Scenario: Successful write to /data/ persists
- **WHEN** a write to `/data/auth/shadow` completes
- **AND** `fsync` is called
- **THEN** the syscall SHALL return `Ok(write_len)`
- **AND** the bytes SHALL be durable across reboot
