# fs-block-device Specification

## Purpose
TBD - created by archiving change embedded-filesystem-v1. Update Purpose after archive.
## Requirements
### Requirement: BlockDevice trait
The `fs` crate SHALL define a single `BlockDevice` trait used by every consumer (squashfs, F2FS, GPT parser, A/B boot, delta apply). The trait SHALL expose:

```rust
pub trait BlockDevice {
    fn read_block(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError>;
    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError>;
    fn block_size_bytes(&self) -> u32;
    fn block_count(&self) -> u64;
    fn flush(&mut self) -> Result<(), BlockError>;
}
```

`buf` length on read/write SHALL be a multiple of `block_size_bytes()`. Each implementation SHALL document its sector size and any device-specific alignment constraints.

#### Scenario: Read returns exactly the requested block
- **WHEN** `read_block(lba, buf)` is called with a `buf` of `block_size_bytes()`
- **THEN** the call SHALL fill `buf` with the contents of `lba` and return `Ok(())`

#### Scenario: Read with mis-sized buffer rejected
- **WHEN** `read_block(lba, buf)` is called with a `buf` length that is not a multiple of `block_size_bytes()`
- **THEN** the call SHALL return `Err(BlockError::Unaligned)` and SHALL NOT issue any I/O

#### Scenario: Read past end of device rejected
- **WHEN** `read_block(lba, buf)` is called with `lba >= block_count()`
- **THEN** the call SHALL return `Err(BlockError::OutOfRange)`

### Requirement: Typed BlockError enum
Block I/O errors SHALL be represented by a typed `BlockError` enum:

```rust
pub enum BlockError {
    MediaError,
    NotPresent,
    Timeout,
    BadCrc,
    Unaligned,
    OutOfRange,
    DeviceBusy,
}
```

The enum SHALL be converted to POSIX-aligned errno (`-EIO`, `-ENXIO`, `-ETIMEDOUT`, `-EINVAL`, `-ENOSPC`, `-EBUSY`) at the syscall boundary by `posix-vfs`.

#### Scenario: Timeout maps to ETIMEDOUT at syscall boundary
- **WHEN** a block read times out after exhausting retries
- **THEN** the internal API SHALL return `Err(BlockError::Timeout)`
- **AND** any syscall surfaced via `posix-vfs` SHALL return `-ETIMEDOUT`

#### Scenario: Bad CRC maps to EIO
- **WHEN** the device reports a CRC failure on read
- **THEN** the internal API SHALL return `Err(BlockError::BadCrc)`
- **AND** the syscall boundary SHALL return `-EIO`

### Requirement: Per-op timeout and retry policy
Block reads SHALL have a default 250 ms per-op timeout and SHALL retry up to 3 times with exponential backoff (500 ms, 1 s, 2 s) before returning `BlockError::Timeout`. Block writes SHALL have a default 1 s per-op timeout with the same retry schedule. Timeouts and retry counts SHALL be configurable via `mgmt/policy.toml` keys `fs.block.read_timeout_ms`, `fs.block.write_timeout_ms`, `fs.block.retry_count`, `fs.block.retry_backoff_ms`.

The fail-after-retries behavior SHALL NOT be configurable: an unattended appliance SHALL NEVER block indefinitely on a single bad sector.

#### Scenario: Three retries then fail
- **WHEN** a block read encounters transient failures and retry_count = 3
- **THEN** the implementation SHALL retry 3 times with the configured backoff
- **AND** the 4th failure SHALL return `Err(BlockError::Timeout)` to the caller

#### Scenario: Indefinite retry rejected at config write
- **WHEN** `mgmt/policy.toml` is written with `fs.block.retry_count = 0xFFFF_FFFF` or any "infinite" sentinel
- **THEN** the validator SHALL reject the value with `-EINVAL`

### Requirement: Per-arch BlockDevice implementations
v1 SHALL ship four `BlockDevice` implementations:

- `virtio-blk` — for QEMU x86-64 and aarch64 CI smoke tests.
- `nvme` — for x86-64 bare-metal servers and USB-NVMe carriers.
- `sdhci` (eMMC) — for the Jetson Orin Tegra234 BSP.
- `ahci` — for legacy x86-64 SATA hardware.

Each implementation SHALL pass the same generic conformance test suite (alignment, retry, error mapping) plus device-specific timing tests where applicable.

#### Scenario: virtio-blk passes generic conformance
- **WHEN** the conformance suite runs against a virtio-blk-backed device in QEMU
- **THEN** all 30+ generic tests SHALL pass

#### Scenario: NVMe passes generic conformance
- **WHEN** the conformance suite runs against an NVMe device
- **THEN** all 30+ generic tests SHALL pass

#### Scenario: SDHCI eMMC passes generic conformance on Jetson
- **WHEN** the conformance suite runs against a Jetson Orin's eMMC
- **THEN** all 30+ generic tests SHALL pass

### Requirement: Native 4 KiB block size with 512-byte fallback
The `fs` crate SHALL operate natively at 4 KiB block size. Devices reporting 4 KiB physical / 4 KiB logical sectors SHALL be used directly. Devices reporting 512 B logical / 4 KiB physical (Advanced Format) SHALL be accessed in 4 KiB chunks aligned to the 8-LBA boundary. Devices reporting 512 B logical / 512 B physical SHALL be accessed via a slow-path emulation layer that bundles 8 logical sectors per 4 KiB FS block.

#### Scenario: 4 KiB-native device used directly
- **WHEN** the device reports `block_size_bytes() == 4096`
- **THEN** FS reads SHALL issue 4 KiB I/O directly to the device

#### Scenario: 512/4 KiB Advanced Format device aligned
- **WHEN** the device reports 512 B logical / 4 KiB physical
- **THEN** FS reads SHALL issue 4 KiB-aligned 8-LBA I/O
- **AND** SHALL never split an FS block across two physical sectors

#### Scenario: 512-byte legacy device emulated
- **WHEN** the device reports 512 B logical / 512 B physical
- **THEN** the emulation layer SHALL bundle 8 consecutive logical reads into one FS block read
- **AND** SHALL log a one-time warning at mount about reduced performance

