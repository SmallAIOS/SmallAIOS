// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Format-level constants and helpers shared by the littlefs read and
//! write paths.
//!
//! All values here are **public format-spec constants** pinned to the
//! v2.10.0 SPEC.md commit, not secrets. CodeQL's
//! `rust/hard-coded-cryptographic-value` rule is suppressed via
//! standalone-line `lgtm[...]` directives.

// ─── Format constants (pinned to v2.10.0 SPEC.md) ───────────────────────────

/// littlefs v2 superblock magic (`b"littlefs"`).
// lgtm[rust/hard-coded-cryptographic-value] — littlefs format-spec magic byte sequence, not a secret
pub(super) const LFS2_SUPERBLOCK_MAGIC: [u8; 8] = *b"littlefs";

/// littlefs v2 major version.
pub(super) const LFS2_MAJOR_VERSION: u16 = 2;

/// CRC32C (Castagnoli) polynomial, reflected form.
// lgtm[rust/hard-coded-cryptographic-value] — public CRC polynomial, not a secret
pub(super) const CRC32C_POLY_REFLECTED: u32 = 0x82F63B78;

/// Maximum reasonable name length we accept from a tag (sanity bound).
pub(super) const NAME_MAX: usize = 1024;

/// Maximum tag count we'll process while parsing one metadata-pair half.
/// Bounds memory and CPU even for malformed inputs.
pub(super) const MAX_TAGS_PER_HALF: usize = 4096;

// ─── Tag type encoding ──────────────────────────────────────────────────────
//
// Per SPEC.md the 32-bit tag is encoded big-endian as:
//
// ```text
// [-- 1 --|-- 11 --|-- 10 --|-- 10 --]
//   valid    type     id      size
// ```
//
// Tags are stored XOR'd against the previous tag word; the running CRC
// covers everything (as on disk) and the CRC tag's payload is the
// expected post-update inverted CRC.

pub(super) const TAG_TYPE3_NAME: u16 = 0x0;
pub(super) const TAG_TYPE3_STRUCT: u16 = 0x2;
pub(super) const TAG_TYPE3_CRC: u16 = 0x5;
pub(super) const TAG_TYPE3_TAIL: u16 = 0x6;

// Sub-type-1 codes for `name` tags:
pub(super) const TAG_NAME_REG: u16 = 0x001; // (TYPE3_NAME << 8) | 0x01 — file
pub(super) const TAG_NAME_DIR: u16 = 0x002; // (TYPE3_NAME << 8) | 0x02 — directory
pub(super) const TAG_NAME_SUPERBLOCK: u16 = 0x0FF;

// Sub-type-1 codes for `struct` tags:
pub(super) const TAG_STRUCT_DIR: u16 = 0x200; // (TYPE3_STRUCT << 8) | 0x00 — dir entry pair
pub(super) const TAG_STRUCT_INLINE: u16 = 0x201; // inline-file struct
pub(super) const TAG_STRUCT_CTZ: u16 = 0x202; // ctz-list file struct

/// CRC tag type code (the writer uses this as its commit-marker).
pub(super) const TAG_CRC_NORMAL: u16 = 0x500;

// ─── CRC32C ──────────────────────────────────────────────────────────────────

/// Compute CRC32C (Castagnoli) over `data`, with running state `crc`.
///
/// Standard reflected formulation: input bits enter LSB first, the
/// polynomial is `0x82F63B78` reflected, initial value is `0xFFFFFFFF`,
/// final XOR is `0xFFFFFFFF`. Returns the post-update *un-finalized*
/// state so callers can chain.
pub(super) fn crc32c_update(mut crc: u32, data: &[u8]) -> u32 {
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let lsb = crc & 1;
            crc >>= 1;
            if lsb == 1 {
                crc ^= CRC32C_POLY_REFLECTED;
            }
        }
    }
    crc
}

/// Convenience: full CRC32C of `data` with the standard init/final XOR.
pub fn crc32c(data: &[u8]) -> u32 {
    !crc32c_update(0xFFFF_FFFF, data)
}

// ─── Endian helpers ─────────────────────────────────────────────────────────

#[inline]
pub(super) fn read_u32_le(buf: &[u8]) -> u32 {
    u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
}

#[inline]
pub(super) fn read_u32_be(buf: &[u8]) -> u32 {
    u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]])
}

// ─── Tag word pack/unpack ───────────────────────────────────────────────────

/// Pack a tag into its 32-bit on-disk word with `valid = 0`.
///
/// `valid=0` in bit 31; type in bits 30..20; id in bits 19..10; len in
/// bits 9..0.
#[inline]
pub(super) fn pack_tag_word(typ: u16, id: u16, len: u32) -> u32 {
    ((u32::from(typ) & 0x7FF) << 20) | ((u32::from(id) & 0x3FF) << 10) | (len & 0x3FF)
}
