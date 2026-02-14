// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM64 Image header for U-Boot `booti` compatibility.
//!
//! The Linux ARM64 boot protocol defines a 64-byte header at the start
//! of the kernel Image. U-Boot's `booti` command validates this header
//! before jumping to the kernel entry point.
//!
//! Header layout (all little-endian):
//!
//! | Offset | Size | Field          | Description                      |
//! |--------|------|----------------|----------------------------------|
//! | 0x00   | 4    | code0          | Branch instruction over header   |
//! | 0x04   | 4    | code1          | Reserved (zero)                  |
//! | 0x08   | 8    | text_offset    | Image load offset from RAM base  |
//! | 0x10   | 8    | image_size     | Effective image size (0 = unknown)|
//! | 0x18   | 8    | flags          | Kernel flags                     |
//! | 0x20   | 8    | res2           | Reserved (zero)                  |
//! | 0x28   | 8    | res3           | Reserved (zero)                  |
//! | 0x30   | 8    | res4           | Reserved (zero)                  |
//! | 0x38   | 4    | magic          | 0x644D5241 ("ARM\x64" LE)       |
//! | 0x3C   | 4    | res5           | Reserved (zero for PE offset)    |
//!
//! Flags (offset 0x18):
//! - Bit 0: Endianness (0 = little-endian)
//! - Bits 1-2: Page size (1 = 4K, 2 = 16K, 3 = 64K)
//! - Bit 3: Placement (0 = dram_base + text_offset)

/// ARM64 Image header magic value: "ARM\x64" in little-endian.
pub const ARM64_IMAGE_MAGIC: u32 = 0x644D5241;

/// Flags: little-endian (bit 0 = 0) + 4K pages (bits 1-2 = 01) = 0x02.
pub const ARM64_IMAGE_FLAGS: u64 = 0x02;

/// Text offset from DRAM base (0x80000 = 512 KiB, standard for Tegra).
pub const TEXT_OFFSET: u64 = 0x80000;

/// The 64-byte ARM64 Image header.
///
/// Placed in `.text.image_header` which the Tegra linker script orders
/// before `.text.boot` so it appears at the very start of the binary.
///
/// The first instruction is `b _start_real` encoded as a relative branch.
/// Since this header is 64 bytes (16 instructions), we branch forward by
/// 16 instructions: `b . + 64` = opcode `0x14000010`.
#[cfg(feature = "tegra-x1")]
#[unsafe(link_section = ".text.image_header")]
#[used]
pub static ARM64_IMAGE_HEADER: [u32; 16] = [
    // code0: branch over header (b . + 64 = 16 instructions forward)
    0x14000010,
    // code1: reserved
    0,
    // text_offset (u64 LE): low word, high word
    TEXT_OFFSET as u32,
    (TEXT_OFFSET >> 32) as u32,
    // image_size (u64 LE): 0 = unknown (linker fills in)
    0,
    0,
    // flags (u64 LE): 0x02 = little-endian, 4K pages
    ARM64_IMAGE_FLAGS as u32,
    (ARM64_IMAGE_FLAGS >> 32) as u32,
    // res2 (u64)
    0,
    0,
    // res3 (u64)
    0,
    0,
    // res4 (u64)
    0,
    0,
    // magic: 0x644D5241 ("ARM\x64" in LE)
    ARM64_IMAGE_MAGIC,
    // res5: reserved (PE header offset, 0 for non-PE)
    0,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_value() {
        assert_eq!(ARM64_IMAGE_MAGIC, 0x644D5241);
        // Verify it matches "ARM\x64" in little-endian
        let bytes = ARM64_IMAGE_MAGIC.to_le_bytes();
        assert_eq!(bytes, [b'A', b'R', b'M', 0x64]);
    }

    #[test]
    fn test_flags_little_endian_4k() {
        // Bit 0 = 0 (little-endian), bits 1-2 = 01 (4K pages) = 0b010 = 2
        assert_eq!(ARM64_IMAGE_FLAGS, 0x02);
        // Bit 0 should be 0 (LE)
        assert_eq!(ARM64_IMAGE_FLAGS & 1, 0);
        // Bits 1-2 should be 01 (4K)
        assert_eq!((ARM64_IMAGE_FLAGS >> 1) & 3, 1);
    }

    #[test]
    fn test_text_offset() {
        // 512 KiB offset is standard for ARM64
        assert_eq!(TEXT_OFFSET, 0x80000);
        assert_eq!(TEXT_OFFSET, 512 * 1024);
    }

    #[test]
    fn test_branch_instruction() {
        // b . + 64: offset = 64/4 = 16 = 0x10
        // ARM64 B instruction: 0b000101 << 26 | imm26
        // 0x14 << 24 | 0x10 = 0x14000010
        let branch = 0x14000010u32;
        let opcode = branch >> 26;
        assert_eq!(opcode, 0b000101); // unconditional branch
        let offset = branch & 0x03FF_FFFF;
        assert_eq!(offset, 16); // 16 instructions = 64 bytes
    }
}
