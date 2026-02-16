## ADDED Requirements

### Requirement: ARM64 Image header in binary
The Tegra kernel binary SHALL include a 64-byte ARM64 Image header at the start of the `.text.boot` section, conforming to the Linux ARM64 boot protocol.

#### Scenario: Header magic present
- **WHEN** the built kernel binary is inspected
- **THEN** bytes at offset `0x38` SHALL contain the magic value `0x644D5241` ("ARM\x64" in little-endian)

#### Scenario: Branch instruction at offset 0
- **WHEN** U-Boot loads the Image and jumps to offset 0
- **THEN** the first instruction SHALL be an unconditional branch (`b`) that jumps over the 64-byte header to the actual `_start` code

#### Scenario: U-Boot booti compatibility
- **WHEN** U-Boot executes `booti <addr> - <dtb_addr>` with the SmallAIOS kernel
- **THEN** U-Boot SHALL recognize the Image header, validate the magic, and transfer control to the kernel entry point

### Requirement: Header fields
The ARM64 Image header SHALL populate the following fields: branch instruction (offset 0x00), text_offset (offset 0x08), image_size (offset 0x10), flags (offset 0x18), and magic (offset 0x38). All other fields SHALL be zero.

#### Scenario: Flags indicate little-endian kernel
- **WHEN** U-Boot reads the flags field at offset 0x18
- **THEN** bit 0 SHALL be 0 (little-endian) and bits 1-2 SHALL indicate 4K page granule

### Requirement: Header only on Tegra builds
The ARM64 Image header SHALL only be included when building with the `tegra-x1` feature. QEMU virt builds SHALL NOT include the header (QEMU loads raw ELF/binary without it).

#### Scenario: QEMU build has no Image header
- **WHEN** the kernel is built with default features (qemu-virt)
- **THEN** the `.text.boot` section SHALL begin directly with the `_start` function, with no 64-byte header prefix
