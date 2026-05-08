## ADDED Requirements

### Requirement: GPT partition table parser
The `fs` crate SHALL parse the GUID Partition Table (GPT) per UEFI Specification 2.10 §5.3. Reads SHALL validate: protective-MBR signature, primary GPT header CRC32, partition entry array CRC32, header revision, and entry count bounds. The parser SHALL also read the secondary GPT (at the end of the device) and SHALL prefer it if the primary header is corrupt. v1 SHALL NOT support legacy MBR-only disks.

#### Scenario: Valid GPT parses successfully
- **WHEN** a device with a valid GPT is presented
- **THEN** the parser SHALL return the array of partition entries with their type GUIDs, start LBAs, end LBAs, and names

#### Scenario: Corrupt primary GPT falls through to secondary
- **WHEN** the primary header CRC fails but the secondary header CRC is valid
- **THEN** the parser SHALL read partition entries from the secondary location and SHALL log a one-time warning

#### Scenario: Both primary and secondary corrupt rejected
- **WHEN** both primary and secondary GPT headers fail their CRC checks
- **THEN** the parser SHALL return `Err(BlockError::BadCrc)`
- **AND** the kernel SHALL halt with a recovery hint

#### Scenario: MBR-only disk rejected
- **WHEN** a device with an MBR but no GPT signature is presented
- **THEN** the parser SHALL refuse to mount and SHALL log "GPT required, MBR-only disk rejected"

### Requirement: Protective MBR for tool compatibility
On a freshly-formatted SmallAIOS disk, the GPT writer SHALL write a protective MBR (single MBR partition entry of type `0xEE` covering the disk) per the GPT specification. This SHALL ensure legacy partition-table tools see one large unknown-type partition rather than free space they might overwrite.

#### Scenario: Protective MBR present after format
- **WHEN** SmallAIOS formats a disk with the v1 layout
- **THEN** the first 512 bytes SHALL contain a protective MBR with one partition of type `0xEE`

### Requirement: v1 partition layout enforcement
The v1 SmallAIOS GPT layout SHALL contain exactly five partitions in this order:

| Idx | Type GUID                              | Size      | Purpose                          |
|----:|----------------------------------------|-----------|----------------------------------|
|   1 | EFI System Partition (`C12A7328-...`)  | 256 MiB   | UEFI bootloader + kernel image   |
|   2 | SmallAIOS squashfs slot (`A3...A1`)    | 4 GiB     | `/models/` slot A                |
|   3 | SmallAIOS squashfs slot (`A3...A2`)    | 4 GiB     | `/models/` slot B                |
|   4 | Linux F2FS (`8DA63339-...`)            | remainder | `/data/`                         |
|   5 | SmallAIOS boot config (`A3...B0`)      | 8 MiB     | A/B boot pointer (`fs-ab-boot`)  |

The kernel SHALL refuse to mount if the layout does not match (correct partition types in the correct order).

#### Scenario: Correct layout mounts
- **WHEN** all five partitions exist with the expected types
- **THEN** the kernel SHALL identify each partition and pass it to the appropriate driver

#### Scenario: Missing F2FS partition rejected
- **WHEN** partition #4 is absent or has the wrong type GUID
- **THEN** the kernel SHALL halt with `Err: F2FS partition #4 missing`

#### Scenario: Missing boot config partition rejected
- **WHEN** partition #5 is absent
- **THEN** the kernel SHALL halt with `Err: boot config partition #5 missing`

### Requirement: SmallAIOS-specific partition type GUIDs
SmallAIOS-specific partitions (squashfs slots, boot config) SHALL use SmallAIOS-registered partition type GUIDs:

- Squashfs slot: `A3F7C2E0-FACE-4FFF-AAAA-000000000001` (slot A) and `A3F7C2E0-FACE-4FFF-AAAA-000000000002` (slot B)
- Boot config: `A3F7C2E0-FACE-4FFF-BBBB-000000000000`

These SHALL be documented in `docs/architecture.md` and SHALL NOT collide with any registered GUID in the `gdisk`/`parted` GUID database.

#### Scenario: GUIDs documented
- **WHEN** `docs/architecture.md` is read
- **THEN** it SHALL contain the SmallAIOS partition type GUID table

#### Scenario: Foreign GUID with same name rejected
- **WHEN** a partition named `SmallAIOS Models A` has a non-SmallAIOS type GUID
- **THEN** the kernel SHALL refuse to mount and SHALL log the type GUID mismatch
