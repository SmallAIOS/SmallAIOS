# fs-f2fs-readwrite Specification

## Purpose
TBD - created by archiving change embedded-filesystem-v1. Update Purpose after archive.
## Requirements
### Requirement: F2FS read path
The `fs` crate SHALL provide a clean-room `#![no_std]` Rust reader for the F2FS on-disk format matching the Linux 6.6 LTS feature set. The reader SHALL parse the superblock, the checkpoint journal, the SIT (Segment Information Table), the NAT (Node Address Table), the SSA (Segment Summary Area), and the data area per the F2FS specification. The reader SHALL successfully mount images produced by `mkfs.f2fs` from the matching `f2fs-tools` release and SHALL `cmp`-equal a byte-stream produced by Linux's F2FS driver reading the same image.

#### Scenario: mkfs.f2fs image mounts
- **WHEN** an image produced by `mkfs.f2fs` (from `f2fs-tools` matching Linux 6.6) is presented
- **THEN** mount SHALL succeed and the root directory listing SHALL match what Linux F2FS reports

#### Scenario: Linux-written file readable byte-for-byte
- **WHEN** a file is created by Linux F2FS, then read by SmallAIOS
- **THEN** the read bytes SHALL exactly match the bytes Linux wrote

#### Scenario: Unknown mandatory feature bit rejected
- **WHEN** the F2FS superblock declares an unknown mandatory feature bit
- **THEN** mount SHALL fail with `Err: F2FS mandatory feature <bit> unsupported`
- **AND** SHALL NOT proceed to a partial mount

### Requirement: F2FS write path
The `fs` crate SHALL provide a clean-room `#![no_std]` Rust writer for F2FS. The writer SHALL implement: file creation, directory creation, writes that extend a file, writes that overwrite existing blocks, file deletion, directory deletion, atomic rename, and `truncate`. Writes SHALL go through the SIT/NAT update machinery and SHALL produce on-disk state that Linux 6.6 F2FS can mount and read.

#### Scenario: Write then Linux reads
- **WHEN** SmallAIOS creates a file containing known bytes
- **AND** the F2FS partition is then mounted on Linux 6.6
- **THEN** Linux SHALL read back the same bytes

#### Scenario: Atomic rename preserves consistency
- **WHEN** SmallAIOS performs `rename(src, dst)` while another reader has `dst` open
- **THEN** the open reader's file handle SHALL continue to refer to the original `dst` content
- **AND** new opens of `dst` SHALL see `src`'s content

#### Scenario: Truncate to zero
- **WHEN** SmallAIOS truncates a 1 MiB file to 0 bytes
- **THEN** subsequent reads SHALL return zero bytes
- **AND** Linux re-mounting SHALL agree the file is 0 bytes

### Requirement: fsync and checkpoint commit
`fsync(fd)` SHALL force all dirty pages of `fd` and the corresponding NAT/SIT updates to disk via a checkpoint commit before returning. After successful `fsync`, the data SHALL be durable across an arbitrary power loss; Linux 6.6 reading the partition after the power cycle SHALL see the fsync'd state.

A 5-second background timer SHALL trigger a checkpoint commit if any dirty data is pending and `fsync` has not been called.

#### Scenario: fsync survives power loss
- **WHEN** SmallAIOS writes a file, calls `fsync`, then the system loses power
- **AND** the disk is read by Linux 6.6 after reboot
- **THEN** the fsync'd file content SHALL be intact

#### Scenario: Non-fsync data lost up to last checkpoint
- **WHEN** SmallAIOS writes data without calling `fsync` and power is lost before any checkpoint
- **THEN** the data MAY be absent on next mount
- **AND** the filesystem SHALL still be consistent

#### Scenario: 5-second timer commits idle dirty data
- **WHEN** a write happens at time t and no `fsync` is called for 5 seconds
- **THEN** at time t+5s a checkpoint commit SHALL flush the dirty data
- **AND** subsequent power loss after t+5s SHALL preserve the data

### Requirement: Garbage collection
The F2FS writer SHALL implement segment-level garbage collection. GC SHALL run opportunistically when free segments fall below a threshold (default: 5% of total segments) and SHALL relocate live blocks from the most fragmented victim segment to a fresh segment, then mark the old segment free. GC SHALL be cooperative — it SHALL yield between block relocations so it does not stall foreground writes.

#### Scenario: GC frees segments below threshold
- **WHEN** free segments drop below 5%
- **THEN** GC SHALL run and SHALL increase the free count
- **AND** existing file data SHALL remain intact (verified post-GC `cmp` against pre-GC content)

#### Scenario: GC yields to foreground writes
- **WHEN** GC is running and a foreground write arrives
- **THEN** GC SHALL release its work credit between block relocations
- **AND** the foreground write SHALL not be starved

### Requirement: VFS mount at `/data/`
The F2FS partition SHALL be mounted at `/data/` in the VFS. The mount SHALL happen during kernel boot before `auth_login` (since `/data/auth/shadow` lives on this mount).

#### Scenario: /data/ available before login
- **WHEN** the kernel reaches the login prompt
- **THEN** `/data/` SHALL already be mounted RW
- **AND** `auth_login` SHALL be able to read `/data/auth/shadow`

### Requirement: Format on first boot when physical presence asserted
When the F2FS partition is unformatted and a `PhysicalPresenceProvider` indicator is asserted, the kernel SHALL run an in-process equivalent of `mkfs.f2fs` to format the partition, then SHALL append an audit record describing the format event (which provider asserted presence, the partition size, the resulting filesystem UUID).

When the partition is unformatted and physical presence is **not** asserted, the kernel SHALL halt with a recovery hint pointing at the asserter mechanism for the current arch.

#### Scenario: Format with presence asserted
- **WHEN** the partition has no F2FS superblock and the GPIO presence pin is asserted
- **THEN** the kernel SHALL format the partition, mount it, and append a `data_format` audit record

#### Scenario: Halt without presence
- **WHEN** the partition has no F2FS superblock and no presence provider asserts
- **THEN** the kernel SHALL halt with `Err: /data/ unformatted; assert physical presence to format`

