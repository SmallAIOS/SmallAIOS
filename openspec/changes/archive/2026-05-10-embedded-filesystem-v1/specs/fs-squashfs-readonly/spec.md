## ADDED Requirements

### Requirement: Squashfs 4.0 read-only reader
The `fs` crate SHALL provide a clean-room `#![no_std]` Rust reader for the squashfs 4.0 on-disk format. The reader SHALL parse the superblock, the inode table, the directory table, the fragment table, the export table, and the ID/lookup tables per the squashfs 4.0 specification. The reader SHALL be read-only; any write attempt against a squashfs mount SHALL return `-EROFS`. The reader SHALL refuse to mount images of any other major version (1.x, 2.x, 3.x, or unknown future major versions ≥5).

#### Scenario: 4.0 image mounts
- **WHEN** a squashfs 4.0 image with no extended-feature bits is presented
- **THEN** the mount SHALL succeed and the directory tree SHALL be readable

#### Scenario: 3.x image rejected
- **WHEN** a squashfs 3.x image is presented
- **THEN** the mount SHALL fail with `Err: squashfs major version 3.x unsupported`

#### Scenario: Unknown future major rejected
- **WHEN** a squashfs image with major version 5 is presented
- **THEN** the mount SHALL fail with `Err: squashfs major version 5 unsupported`
- **AND** SHALL NOT silently fall through to a partial mount

#### Scenario: Write to squashfs returns EROFS
- **WHEN** any write syscall targets a path under `/models/`
- **THEN** the syscall SHALL return `-EROFS`

### Requirement: Compression algorithm support
The squashfs reader SHALL decompress blocks using zstd, xz, gzip, and lz4. Each decoder SHALL be a clean-room `#![no_std]` Rust implementation; the zstd decoder SHALL be reused from the existing `compute` crate. Each decoder SHALL be tested against externally-produced reference images (one per algorithm produced by `mksquashfs -comp <alg>`) and SHALL `cmp`-equal the original input bytes after round-trip.

#### Scenario: zstd decompression round-trip
- **WHEN** a squashfs image produced by `mksquashfs -comp zstd` is read
- **THEN** every file's contents SHALL match the original `cmp`-byte-for-byte

#### Scenario: xz decompression round-trip
- **WHEN** a squashfs image produced by `mksquashfs -comp xz` is read
- **THEN** every file's contents SHALL match the original

#### Scenario: gzip decompression round-trip
- **WHEN** a squashfs image produced by `mksquashfs -comp gzip` is read
- **THEN** every file's contents SHALL match the original

#### Scenario: lz4 decompression round-trip
- **WHEN** a squashfs image produced by `mksquashfs -comp lz4` is read
- **THEN** every file's contents SHALL match the original

#### Scenario: Unknown compression algorithm rejected
- **WHEN** a squashfs image declares an unknown compression algorithm in its superblock
- **THEN** mount SHALL fail with `Err: unsupported squashfs compression`

### Requirement: Manifest footer with ML-DSA-65 signature
A SmallAIOS-produced squashfs image SHALL have a sealed manifest footer appended after the squashfs's natural EOF. The footer SHALL contain: a magic number (`SmAIOSFS\0`), a version byte, a SHA-3-256 hash for every block in the image, the public-key fingerprint, an ML-DSA-65 signature over the hash array, and the footer length. The footer SHALL be 4-byte aligned to satisfy squashfs's trailing-padding tolerance so external `mount -t squashfs -o loop` continues to work.

#### Scenario: External mount works
- **WHEN** a SmallAIOS squashfs image (with footer) is mounted on stock Ubuntu via `mount -t squashfs -o loop`
- **THEN** the mount SHALL succeed and contents SHALL be readable

#### Scenario: SmallAIOS reads footer first
- **WHEN** SmallAIOS opens a squashfs blob
- **THEN** the implementation SHALL read the last `footer_length` bytes first to locate and validate the manifest before any block read is honored

#### Scenario: Missing footer rejected by SmallAIOS
- **WHEN** a squashfs image without the SmallAIOS footer is presented
- **THEN** SmallAIOS SHALL refuse to mount it on `/models/` and SHALL log `Err: squashfs missing SmallAIOS manifest footer`
- **AND** the same image SHALL still mount on stock Linux (manifest is SmallAIOS-only, not part of squashfs)

### Requirement: VFS mount at `/models/`
The active squashfs slot SHALL be mounted at `/models/` in the VFS. The mount SHALL happen during kernel boot, after the GPT parser has identified the active slot per `fs-ab-boot`, and before any service that depends on model files starts.

#### Scenario: /models/ mounted on boot
- **WHEN** the kernel finishes early init and the active slot has been determined
- **THEN** `/models/` SHALL be mounted from the active squashfs slot
- **AND** subsequent file opens under `/models/` SHALL succeed
