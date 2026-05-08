## ADDED Requirements

### Requirement: /flash/ mount point
The VFS SHALL recognize `/flash/` as a top-level mount point alongside the existing `/dev/`, `/proc/self/`, `/models/`, and `/data/`. The implementation SHALL be conditionally compiled behind the `fs-flash` cargo feature (per `fs-flash-mount`); when the feature is off, `/flash/` SHALL be absent from the path-resolution tree and SHALL return `-ENOENT` on any open.

#### Scenario: /flash/ resolves to littlefs when available
- **WHEN** any path under `/flash/` is opened
- **AND** `fs-flash` is enabled and a flash device is mounted
- **THEN** the open SHALL be routed to the littlefs reader/writer

#### Scenario: /flash/ absent when feature off
- **WHEN** the kernel is built without `fs-flash`
- **THEN** `open("/flash/anything", ...)` SHALL return `-ENOENT`
- **AND** the path SHALL NOT appear in `readdir("/")`

### Requirement: fadvise SEQUENTIAL and RANDOM hints
The VFS `posix_fadvise` syscall SHALL accept `POSIX_FADV_SEQUENTIAL` and `POSIX_FADV_RANDOM` hints on file descriptors backed by `/flash/`. SEQUENTIAL SHALL enable write-batching in the littlefs writer (multiple appends within a window are coalesced into a single metadata-pair commit). RANDOM SHALL disable any read-ahead. Other POSIX `fadvise` constants SHALL be accepted as no-ops with `Ok(0)` to preserve POSIX semantics.

#### Scenario: SEQUENTIAL enables write-batching
- **WHEN** an audit-log writer issues `posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)` then 10 short appends within 100 ms
- **THEN** the underlying littlefs SHALL coalesce the appends into one metadata-pair commit
- **AND** the user-visible effect SHALL match Linux SEQUENTIAL semantics

#### Scenario: RANDOM disables read-ahead
- **WHEN** code issues `posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM)`
- **THEN** subsequent reads SHALL NOT prefetch beyond the requested offset+length

#### Scenario: Unsupported hints accepted as no-op
- **WHEN** code issues `posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED)`
- **THEN** the syscall SHALL return `Ok(0)`
- **AND** behavior SHALL be unaffected

## MODIFIED Requirements

### Requirement: BlockError to errno translation now also handles FlashError
The VFS errno translation table SHALL be extended to handle the `FlashError` variants from `fs-flash-device`:

| FlashError | errno |
|------------|-------|
| MediaError | -EIO |
| NotPresent | -ENXIO |
| ProgramOnDirty | -EROFS |
| EraseFailure | -EIO |
| BadBlock | -EIO |
| Timeout | -ETIMEDOUT |
| OutOfRange | -EINVAL |
| Unaligned | -EINVAL |

The existing `BlockError` translation table from `embedded-filesystem-v1` SHALL remain unchanged.

#### Scenario: ProgramOnDirty maps to EROFS
- **WHEN** a write under `/flash/` triggers `FlashError::ProgramOnDirty` (a serious driver bug)
- **THEN** the syscall SHALL return `-EROFS`
- **AND** an audit record `flash_program_dirty` SHALL be appended

#### Scenario: BadBlock maps to EIO
- **WHEN** a read under `/flash/` hits a block in the BBT
- **THEN** the syscall SHALL return `-EIO`
- **AND** the BBT SHALL ensure subsequent allocations skip the block
