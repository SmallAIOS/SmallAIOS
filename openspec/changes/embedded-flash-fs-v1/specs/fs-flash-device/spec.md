## ADDED Requirements

### Requirement: FlashDevice trait
The `fs` crate SHALL define a `FlashDevice` trait separate from `BlockDevice` (from `embedded-filesystem-v1`). Raw-flash devices have erase-block accounting and bad-block reporting that block devices abstract away. The trait SHALL expose:

```rust
pub trait FlashDevice {
    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), FlashError>;
    fn program(&mut self, offset: u64, buf: &[u8]) -> Result<(), FlashError>;
    fn erase(&mut self, block: u64) -> Result<(), FlashError>;
    fn block_size_bytes(&self) -> u32;
    fn page_size_bytes(&self) -> u32;
    fn block_count(&self) -> u64;
    fn is_bad(&self, block: u64) -> bool;
    fn mark_bad(&mut self, block: u64) -> Result<(), FlashError>;
}
```

`program` SHALL only flip bits 1→0 within a previously-erased page; flipping any 0→1 outside an erase SHALL fail with `FlashError::ProgramOnDirty`. `erase` SHALL clear an entire erase-block (size = `block_size_bytes()`); a partial erase is not permitted.

#### Scenario: Read returns programmed bytes
- **WHEN** `program(offset, b"hello")` then `read(offset, buf)` with `buf.len() == 5`
- **THEN** `buf` SHALL contain `b"hello"`

#### Scenario: Program-on-dirty rejected
- **WHEN** a page contains `0x00` and code attempts `program(offset, &[0xFF])`
- **THEN** the call SHALL return `Err(FlashError::ProgramOnDirty)`
- **AND** the page SHALL be unchanged

#### Scenario: Erase resets block to all-ones
- **WHEN** `erase(block)` is called on a programmed block
- **THEN** subsequent `read` of any offset within the block SHALL return `0xFF` bytes

### Requirement: Typed FlashError enum
Flash I/O errors SHALL be represented by a typed `FlashError` enum:

```rust
pub enum FlashError {
    MediaError,
    NotPresent,
    ProgramOnDirty,
    EraseFailure,
    BadBlock,
    Timeout,
    OutOfRange,
    Unaligned,
}
```

The enum SHALL be converted to POSIX-aligned errno (`-EIO`, `-ENXIO`, `-EROFS`, `-ENOMEDIUM`, `-EINVAL`, `-ETIMEDOUT`) at the syscall boundary by `posix-vfs`.

#### Scenario: BadBlock maps to EIO at syscall boundary
- **WHEN** a flash read encounters a known-bad block
- **THEN** the internal API SHALL return `Err(FlashError::BadBlock)`
- **AND** the syscall boundary SHALL return `-EIO`

#### Scenario: ProgramOnDirty maps to EROFS
- **WHEN** an internal write hits `ProgramOnDirty`
- **THEN** the syscall boundary SHALL return `-EROFS`
- **AND** an audit record `flash_program_dirty` SHALL be appended (rare, indicates serious driver bug)

### Requirement: Bad block management
Per-NAND devices SHALL maintain a Bad Block Table (BBT) tracking blocks marked bad either at manufacture (factory BBT) or during operation (runtime detected via failed program/erase). The BBT SHALL be duplicated at the start AND end of the flash, each in a dedicated reserved erase-block. On mount, both BBT copies SHALL be read; if either is corrupt, the surviving copy SHALL be used and the corrupt copy SHALL be rewritten.

The wear-leveling allocator SHALL skip bad blocks. `is_bad(block)` SHALL be consulted before any allocation.

#### Scenario: Bad block skipped by allocator
- **WHEN** the BBT has block 42 marked bad
- **AND** the allocator considers block 42 for new metadata-pair placement
- **THEN** block 42 SHALL be skipped
- **AND** the next non-bad block SHALL be used

#### Scenario: Single BBT survives loss of the other
- **WHEN** the start-of-flash BBT is corrupted but end-of-flash BBT is valid
- **THEN** mount SHALL succeed using end-of-flash BBT
- **AND** start-of-flash BBT SHALL be rewritten from the surviving copy
- **AND** an audit record `bbt_recovered` SHALL be appended

### Requirement: Per-medium drivers
v1 SHALL define driver scaffolding for two flash media types:

- **QSPI NOR** — `fs/src/flash/qspi.rs`. Default block_size_bytes = 4096, page_size_bytes = 256. Implementation per-arch via the architecture's QSPI controller.
- **ONFI NAND** — `fs/src/flash/onfi.rs`. Default block_size_bytes = 131072 (128 KiB), page_size_bytes = 4096. Implementation per-arch via the architecture's NAND controller. ECC handled by the controller; FlashDevice surface is post-ECC.
- **Mock** — `fs/src/flash/mock.rs`. In-memory simulator behind the separate `fs-flash-mock` cargo feature. Configurable bit-flip and bad-block-on-erase injection.

Per-arch QSPI/ONFI controller code SHALL ship as documented stubs in v1; full bringup happens when the first real MCU/FPGA target arrives.

#### Scenario: QSPI defaults
- **WHEN** a `QspiNorDevice` is initialized
- **THEN** `block_size_bytes()` SHALL return 4096 by default
- **AND** `page_size_bytes()` SHALL return 256

#### Scenario: ONFI defaults
- **WHEN** an `OnfiNandDevice` is initialized
- **THEN** `block_size_bytes()` SHALL return 131072 by default
- **AND** `page_size_bytes()` SHALL return 4096

#### Scenario: Mock device usable in CI
- **WHEN** the kernel is built with `fs-flash` and `fs-flash-mock` features
- **THEN** `MockFlashDevice::new(block_size, block_count)` SHALL produce a usable in-memory flash
- **AND** SHALL pass the same conformance test suite as real drivers
