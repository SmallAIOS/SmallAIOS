# posix-vfs Specification

## Purpose
TBD - created by archiving change embedded-filesystem-v1. Update Purpose after archive.
## Requirements
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

### Requirement: /models/ becomes a merged view when overlay feature is enabled
With the `fs-overlay-mounts` cargo feature enabled, the `/models/` mount point SHALL no longer be a direct squashfs mount. It SHALL instead be a merged view backed by the overlay implementation in `fs-overlay-mount`. The lower layer SHALL be the active squashfs slot per `embedded-filesystem-v1`'s A/B selection. The upper layer SHALL be `/data/models-upper/` on F2FS.

When the `fs-overlay-mounts` feature is disabled, the existing `embedded-filesystem-v1` behavior SHALL apply unchanged: `/models/` is a direct squashfs mount, all writes return `-EROFS`.

#### Scenario: Overlay-enabled merged view
- **WHEN** the kernel is built with `fs-overlay-mounts`
- **THEN** lookups under `/models/` SHALL apply upper-wins precedence
- **AND** writes to upper-only or new files SHALL succeed
- **AND** writes to lower-only files SHALL return `-EROFS` with the model_add hint

#### Scenario: Overlay-disabled compatibility
- **WHEN** the kernel is built without `fs-overlay-mounts`
- **THEN** `/models/` SHALL behave exactly as in `embedded-filesystem-v1`
- **AND** any write SHALL return `-EROFS`
- **AND** no upper-layer code SHALL execute

