// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GUID Partition Table (GPT) parser, protective-MBR writer, and v1
//! SmallAIOS layout enforcement.
//!
//! Implements the read path of UEFI Specification 2.10 §5.3:
//!
//! - Read the protective MBR at LBA 0 (only used to reject MBR-only
//!   disks; GPT itself does not depend on its contents).
//! - Read the primary GPT header at LBA 1 and validate signature,
//!   revision, header CRC32, and partition-array bounds.
//! - Read the partition entry array (typically LBA 2..=33) and
//!   validate its CRC32 against the header.
//! - On primary failure, fall through to the secondary GPT at the
//!   tail of the device (header at the last LBA, entry array at
//!   `header.partition_entry_lba`) and accept it if its CRCs pass,
//!   logging a one-time warning.
//! - On both-corrupt return [`GptError::BothHeadersCorrupt`].
//!
//! The write path is intentionally narrow — only the protective MBR
//! ([`protective_mbr::write_protective_mbr`]) is needed by the
//! fresh-format flow. GPT writes are out of scope for v1; SmallAIOS
//! disks are partitioned externally by `sgdisk`/`parted` and only
//! re-read by the kernel.
//!
//! GUIDs are stored on disk in mixed-endian per RFC 4122 §4.1.2:
//! the first three fields (`time_low`, `time_mid`, `time_hi_and_version`)
//! are little-endian, the remaining 8 bytes are big-endian.
//! [`Guid`] preserves the canonical text representation but stores
//! bytes in the on-disk byte order so equality comparison and CRC
//! reconstruction round-trip exactly.
//!
//! Owned by `embedded-filesystem-v1` Phase 2.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::block::{BlockDevice, BlockError};

pub mod layout;
pub mod protective_mbr;

pub use layout::{parse_layout, SmallaiosLayout};
pub use protective_mbr::write_protective_mbr;

/// "EFI PART" signature at the start of every GPT header.
///
// lgtm[rust/hard-coded-cryptographic-value] — UEFI 2.10 §5.3 fixed signature, public constant
pub const GPT_SIGNATURE: [u8; 8] = *b"EFI PART";

/// GPT header revision constant for v1.0 (UEFI 2.10 §5.3.2).
pub const GPT_REVISION_1_0: u32 = 0x0001_0000;

/// GPT header size in bytes per UEFI 2.10 §5.3.2.
pub const GPT_HEADER_SIZE: u32 = 92;

/// Maximum total size of the partition entry array, in bytes
/// (UEFI 2.10 §5.3.2: `num_partition_entries * partition_entry_size <= 16384`).
pub const PARTITION_ARRAY_MAX_BYTES: usize = 16_384;

/// On-disk partition entry size (UEFI 2.10 §5.3.3 — usually 128).
pub const PARTITION_ENTRY_SIZE: u32 = 128;

/// Maximum length, in bytes, of a partition name on disk: 36 UTF-16LE
/// code units.
pub const PARTITION_NAME_BYTES: usize = 72;

/// Maximum number of UTF-16LE code units in a partition name (per
/// UEFI 2.10 §5.3.3).
pub const PARTITION_NAME_CODE_UNITS: usize = 36;

/// MBR signature `0x55AA` at offset 510 of LBA 0.
///
// lgtm[rust/hard-coded-cryptographic-value] — IBM-PC MBR magic (legacy public constant)
pub const MBR_SIGNATURE_LE: [u8; 2] = [0x55, 0xAA];

/// MBR partition type byte `0xEE` reserved for "GPT protective".
pub const MBR_TYPE_GPT_PROTECTIVE: u8 = 0xEE;

/// Errors surfaced by the GPT parser and protective-MBR writer.
///
/// Carries [`BlockError`] for transport-level failures so the caller
/// can distinguish a media error from a parser error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptError {
    /// Primary header CRC failed; secondary was accepted instead.
    /// Surfaced only as a logged warning, never returned from
    /// [`parse_gpt`] when the secondary is valid.
    PrimaryHeaderCorrupt,
    /// Secondary header CRC failed; primary was used. Surfaced only
    /// as a logged warning.
    SecondaryHeaderCorrupt,
    /// Both primary and secondary GPT headers failed CRC validation.
    /// The disk is unrecoverable from SmallAIOS' perspective; manual
    /// intervention required.
    BothHeadersCorrupt,
    /// Header CRC32 mismatch on a header being read directly (e.g.,
    /// a single-header parse path).
    HeaderCrcMismatch,
    /// Partition entry array CRC32 mismatch.
    PartitionArrayCrcMismatch,
    /// Header `revision` is not [`GPT_REVISION_1_0`].
    UnsupportedRevision,
    /// `num_partition_entries * partition_entry_size > 16384`, OR
    /// `partition_entry_size` is not a power of two ≥ 128, OR
    /// `num_partition_entries == 0`.
    EntryCountOutOfRange,
    /// Disk has an MBR signature at LBA 0 but no GPT header at LBA 1.
    /// MBR-only disks are rejected per spec Q1 / `fs-partition-table`.
    MbrOnlyDiskRejected,
    /// Encountered a partition with a type GUID outside the registered
    /// SmallAIOS prefix or the EFI/Linux GUIDs the v1 layout expects.
    /// Used by [`layout::parse_layout`].
    UnknownPartitionType,
    /// The five-partition v1 layout did not match: wrong count, wrong
    /// order, or a type GUID is wrong for that slot.
    LayoutMismatch,
    /// Underlying block-device error.
    BlockError(BlockError),
}

impl From<BlockError> for GptError {
    fn from(e: BlockError) -> Self {
        Self::BlockError(e)
    }
}

impl fmt::Display for GptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrimaryHeaderCorrupt => f.write_str("primary GPT header CRC failed"),
            Self::SecondaryHeaderCorrupt => f.write_str("secondary GPT header CRC failed"),
            Self::BothHeadersCorrupt => f.write_str("both GPT headers corrupt; recovery required"),
            Self::HeaderCrcMismatch => f.write_str("GPT header CRC32 mismatch"),
            Self::PartitionArrayCrcMismatch => f.write_str("GPT partition array CRC32 mismatch"),
            Self::UnsupportedRevision => f.write_str("unsupported GPT revision"),
            Self::EntryCountOutOfRange => f.write_str("GPT entry count out of range"),
            Self::MbrOnlyDiskRejected => f.write_str("GPT required, MBR-only disk rejected"),
            Self::UnknownPartitionType => f.write_str("unknown partition type GUID"),
            Self::LayoutMismatch => f.write_str("v1 GPT layout mismatch"),
            Self::BlockError(e) => write!(f, "block error: {e}"),
        }
    }
}

/// 16-byte mixed-endian GUID (RFC 4122 §4.1.2).
///
/// Stored as the on-disk byte order so equality comparison and round-
/// trip reads are byte-identical. Use [`Self::from_str_canonical`] to
/// parse the textual form `XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX` and
/// [`Self::to_string_canonical`] to render it back.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Guid(pub [u8; 16]);

impl Guid {
    /// Construct from the 16 raw on-disk bytes.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Return the 16 raw on-disk bytes.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// All-zero GUID, used by the unused-entry test (UEFI 2.10 §5.3.3).
    pub const fn nil() -> Self {
        Self([0u8; 16])
    }

    /// True iff every byte is zero.
    pub fn is_nil(&self) -> bool {
        self.0 == [0u8; 16]
    }

    /// Parse from canonical text form `XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX`.
    ///
    /// Per RFC 4122 §4.1.2, the first three fields (`time_low`,
    /// `time_mid`, `time_hi`) are little-endian on disk, the remaining
    /// 8 bytes are big-endian. So the canonical text `01020304-0506-0708-090A-0B0C0D0E0F10`
    /// becomes the byte sequence `[04 03 02 01, 06 05, 08 07, 09 0A, 0B 0C 0D 0E 0F 10]`.
    ///
    /// Returns `None` if the input is malformed.
    pub fn from_str_canonical(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        if b.len() != 36 {
            return None;
        }
        // Position of the four hyphens.
        for &i in &[8usize, 13, 18, 23] {
            if b[i] != b'-' {
                return None;
            }
        }
        // Hex segments (counts of hex chars between hyphens).
        // 8-4-4-4-12 = 32 hex chars total.
        let mut nibbles = [0u8; 32];
        let mut idx = 0;
        for (i, &c) in b.iter().enumerate() {
            if i == 8 || i == 13 || i == 18 || i == 23 {
                continue;
            }
            let n = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return None,
            };
            if idx >= 32 {
                return None;
            }
            nibbles[idx] = n;
            idx += 1;
        }
        if idx != 32 {
            return None;
        }
        // Pack 32 nibbles → 16 bytes (big-endian as written in the text).
        let mut be = [0u8; 16];
        for i in 0..16 {
            be[i] = (nibbles[2 * i] << 4) | nibbles[2 * i + 1];
        }
        // Apply mixed-endian: reverse the first 4-byte field, then the
        // 2-byte field at offset 4, then the 2-byte field at offset 6.
        // The last 8 bytes (offsets 8..16) are stored as-written.
        let mut out = [0u8; 16];
        out[0] = be[3];
        out[1] = be[2];
        out[2] = be[1];
        out[3] = be[0];
        out[4] = be[5];
        out[5] = be[4];
        out[6] = be[7];
        out[7] = be[6];
        out[8..16].copy_from_slice(&be[8..16]);
        Some(Self(out))
    }

    /// Render in canonical text form.
    pub fn to_string_canonical(&self) -> alloc::string::String {
        use alloc::string::String;
        let bytes = &self.0;
        // Reverse the mixed-endian back to the canonical text byte order.
        let be = [
            bytes[3], bytes[2], bytes[1], bytes[0], bytes[5], bytes[4], bytes[7], bytes[6],
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ];
        let mut s = String::with_capacity(36);
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        let push = |s: &mut String, byte: u8| {
            s.push(HEX[(byte >> 4) as usize] as char);
            s.push(HEX[(byte & 0xF) as usize] as char);
        };
        for &byte in &be[0..4] {
            push(&mut s, byte);
        }
        s.push('-');
        push(&mut s, be[4]);
        push(&mut s, be[5]);
        s.push('-');
        push(&mut s, be[6]);
        push(&mut s, be[7]);
        s.push('-');
        push(&mut s, be[8]);
        push(&mut s, be[9]);
        s.push('-');
        for &byte in &be[10..16] {
            push(&mut s, byte);
        }
        s
    }
}

impl fmt::Debug for Guid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Guid")
            .field(&self.to_string_canonical())
            .finish()
    }
}

impl fmt::Display for Guid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_canonical())
    }
}

/// Fixed-capacity UTF-8 partition name (up to 72 UTF-8 bytes, decoded
/// from up to 36 UTF-16LE code units per UEFI 2.10 §5.3.3).
///
/// Implemented inline rather than via `heapless` so the `fs` crate
/// adds no new external dependencies (clean-room rule).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PartitionName {
    bytes: [u8; PARTITION_NAME_BYTES],
    len: u8,
}

impl PartitionName {
    /// Empty name.
    pub const fn empty() -> Self {
        Self {
            bytes: [0u8; PARTITION_NAME_BYTES],
            len: 0,
        }
    }

    /// Construct from a `&str`, returning `None` if it doesn't fit
    /// in [`PARTITION_NAME_BYTES`].
    pub fn try_from_str(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        if b.len() > PARTITION_NAME_BYTES {
            return None;
        }
        let mut out = [0u8; PARTITION_NAME_BYTES];
        out[..b.len()].copy_from_slice(b);
        Some(Self {
            bytes: out,
            len: b.len() as u8,
        })
    }

    /// Number of UTF-8 bytes in the name.
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// True if the name is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow as `&str`.
    pub fn as_str(&self) -> &str {
        // SAFETY: `bytes[..len]` is constructed only via UTF-8 validating
        // paths (`from_str`, `from_utf16le_bytes`).
        // `forbid(unsafe_code)` is set crate-wide; use the safe checked
        // conversion instead. The cost is one re-validation per access
        // which is fine for parser-output.
        core::str::from_utf8(&self.bytes[..self.len as usize]).unwrap_or("")
    }

    /// Decode 72 bytes of UTF-16LE (36 code units, NUL-terminated)
    /// into UTF-8 with replacement characters for invalid surrogate
    /// halves. Per UEFI 2.10 §5.3.3 the on-disk encoding is little-
    /// endian.
    pub fn from_utf16le_bytes(bytes: &[u8; PARTITION_NAME_BYTES]) -> Self {
        // Decode UTF-16LE up to first NUL.
        let mut units: [u16; PARTITION_NAME_CODE_UNITS] = [0; PARTITION_NAME_CODE_UNITS];
        for i in 0..PARTITION_NAME_CODE_UNITS {
            let lo = bytes[2 * i] as u16;
            let hi = bytes[2 * i + 1] as u16;
            units[i] = lo | (hi << 8);
        }
        // Find effective length (truncate at NUL).
        let mut nul_at = PARTITION_NAME_CODE_UNITS;
        for (i, &u) in units.iter().enumerate() {
            if u == 0 {
                nul_at = i;
                break;
            }
        }
        // Decode UTF-16 → UTF-8 with replacement on lone surrogates.
        let mut out = [0u8; PARTITION_NAME_BYTES];
        let mut out_len = 0usize;

        let mut i = 0usize;
        while i < nul_at {
            let u = units[i];
            let cp: u32 = if (0xD800..=0xDBFF).contains(&u) {
                // High surrogate; need a low surrogate next.
                if i + 1 < nul_at {
                    let v = units[i + 1];
                    if (0xDC00..=0xDFFF).contains(&v) {
                        i += 1;
                        0x10000 + (((u as u32) - 0xD800) << 10) + ((v as u32) - 0xDC00)
                    } else {
                        // Lone high surrogate — replace.
                        0xFFFD
                    }
                } else {
                    0xFFFD
                }
            } else if (0xDC00..=0xDFFF).contains(&u) {
                // Lone low surrogate.
                0xFFFD
            } else {
                u as u32
            };
            // Encode cp as UTF-8 into `out`.
            let needed = if cp < 0x80 {
                1
            } else if cp < 0x800 {
                2
            } else if cp < 0x10000 {
                3
            } else {
                4
            };
            if out_len + needed > PARTITION_NAME_BYTES {
                // Truncate — should never happen for 36 BMP code units
                // since 36 * 3 = 108 but max we accept is 72 bytes.
                // For 4-byte UTF-8 coming from surrogate pairs we
                // could exceed 72 bytes only if many CPs >= U+10000.
                // We stop here to preserve the buffer length invariant.
                break;
            }
            match needed {
                1 => out[out_len] = cp as u8,
                2 => {
                    out[out_len] = 0xC0 | ((cp >> 6) as u8);
                    out[out_len + 1] = 0x80 | ((cp & 0x3F) as u8);
                }
                3 => {
                    out[out_len] = 0xE0 | ((cp >> 12) as u8);
                    out[out_len + 1] = 0x80 | (((cp >> 6) & 0x3F) as u8);
                    out[out_len + 2] = 0x80 | ((cp & 0x3F) as u8);
                }
                4 => {
                    out[out_len] = 0xF0 | ((cp >> 18) as u8);
                    out[out_len + 1] = 0x80 | (((cp >> 12) & 0x3F) as u8);
                    out[out_len + 2] = 0x80 | (((cp >> 6) & 0x3F) as u8);
                    out[out_len + 3] = 0x80 | ((cp & 0x3F) as u8);
                }
                _ => unreachable!(),
            }
            out_len += needed;
            i += 1;
        }
        Self {
            bytes: out,
            len: out_len as u8,
        }
    }
}

impl fmt::Debug for PartitionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PartitionName")
            .field(&self.as_str())
            .finish()
    }
}

impl fmt::Display for PartitionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// GPT header (UEFI 2.10 §5.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GptHeader {
    pub signature: [u8; 8],
    pub revision: u32,
    pub header_size: u32,
    pub header_crc32: u32,
    pub current_lba: u64,
    pub backup_lba: u64,
    pub first_usable_lba: u64,
    pub last_usable_lba: u64,
    pub disk_guid: Guid,
    pub partition_entry_lba: u64,
    pub num_partition_entries: u32,
    pub partition_entry_size: u32,
    pub partition_array_crc32: u32,
}

impl GptHeader {
    /// Parse a header out of the first 92 bytes of a block.
    ///
    /// Performs structural validation (signature, revision, size) but
    /// does NOT verify the header CRC; that is done by [`parse_gpt`]
    /// against the surrounding block.
    pub fn parse(block: &[u8]) -> Result<Self, GptError> {
        if block.len() < GPT_HEADER_SIZE as usize {
            return Err(GptError::HeaderCrcMismatch);
        }
        let mut signature = [0u8; 8];
        signature.copy_from_slice(&block[0..8]);
        if signature != GPT_SIGNATURE {
            // Not a GPT header — caller decides whether to fall through
            // to the secondary or reject as MBR-only.
            return Err(GptError::HeaderCrcMismatch);
        }
        let revision = u32_le(&block[8..12]);
        if revision != GPT_REVISION_1_0 {
            return Err(GptError::UnsupportedRevision);
        }
        let header_size = u32_le(&block[12..16]);
        if header_size < GPT_HEADER_SIZE || header_size as usize > block.len() {
            return Err(GptError::HeaderCrcMismatch);
        }
        let header_crc32 = u32_le(&block[16..20]);
        // bytes [20..24] reserved (must be 0 per spec but parsers
        // accept any value).
        let current_lba = u64_le(&block[24..32]);
        let backup_lba = u64_le(&block[32..40]);
        let first_usable_lba = u64_le(&block[40..48]);
        let last_usable_lba = u64_le(&block[48..56]);
        let mut disk_guid_bytes = [0u8; 16];
        disk_guid_bytes.copy_from_slice(&block[56..72]);
        let disk_guid = Guid::from_bytes(disk_guid_bytes);
        let partition_entry_lba = u64_le(&block[72..80]);
        let num_partition_entries = u32_le(&block[80..84]);
        let partition_entry_size = u32_le(&block[84..88]);
        let partition_array_crc32 = u32_le(&block[88..92]);

        // Bounds check on entry array size.
        let entries_total = (num_partition_entries as u64)
            .checked_mul(partition_entry_size as u64)
            .ok_or(GptError::EntryCountOutOfRange)?;
        if entries_total == 0 || entries_total > PARTITION_ARRAY_MAX_BYTES as u64 {
            return Err(GptError::EntryCountOutOfRange);
        }
        // Per UEFI 2.10 §5.3.3, partition_entry_size must be a power
        // of two >= 128.
        if partition_entry_size < 128 || !partition_entry_size.is_power_of_two() {
            return Err(GptError::EntryCountOutOfRange);
        }

        Ok(Self {
            signature,
            revision,
            header_size,
            header_crc32,
            current_lba,
            backup_lba,
            first_usable_lba,
            last_usable_lba,
            disk_guid,
            partition_entry_lba,
            num_partition_entries,
            partition_entry_size,
            partition_array_crc32,
        })
    }

    /// Compute the header CRC32 over the canonical 92-byte window with
    /// the `header_crc32` field zero'd. Bytes outside the first
    /// `header_size` are excluded per UEFI 2.10 §5.3.2.
    pub fn compute_header_crc(block: &[u8], header_size: u32) -> u32 {
        let len = (header_size as usize).min(block.len());
        let mut buf = alloc::vec![0u8; len];
        buf.copy_from_slice(&block[..len]);
        // Zero-out the header_crc32 field (offset 16..20).
        for byte in buf.iter_mut().take(20).skip(16) {
            *byte = 0;
        }
        crc32(&buf)
    }
}

/// Parsed partition entry (UEFI 2.10 §5.3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionEntry {
    pub type_guid: Guid,
    pub unique_guid: Guid,
    pub starting_lba: u64,
    pub ending_lba: u64,
    pub attributes: u64,
    pub name: PartitionName,
}

impl PartitionEntry {
    /// Parse a single 128-byte partition entry. Trailing bytes (if
    /// `partition_entry_size > 128`) are ignored per UEFI 2.10 §5.3.3.
    pub fn parse(entry: &[u8]) -> Result<Self, GptError> {
        if entry.len() < 128 {
            return Err(GptError::EntryCountOutOfRange);
        }
        let mut type_guid_bytes = [0u8; 16];
        type_guid_bytes.copy_from_slice(&entry[0..16]);
        let mut unique_guid_bytes = [0u8; 16];
        unique_guid_bytes.copy_from_slice(&entry[16..32]);
        let starting_lba = u64_le(&entry[32..40]);
        let ending_lba = u64_le(&entry[40..48]);
        let attributes = u64_le(&entry[48..56]);
        let mut name_bytes = [0u8; PARTITION_NAME_BYTES];
        name_bytes.copy_from_slice(&entry[56..56 + PARTITION_NAME_BYTES]);
        Ok(Self {
            type_guid: Guid::from_bytes(type_guid_bytes),
            unique_guid: Guid::from_bytes(unique_guid_bytes),
            starting_lba,
            ending_lba,
            attributes,
            name: PartitionName::from_utf16le_bytes(&name_bytes),
        })
    }

    /// True iff this entry is unused (type GUID is the all-zero GUID
    /// per UEFI 2.10 §5.3.3).
    pub fn is_unused(&self) -> bool {
        self.type_guid.is_nil()
    }
}

/// Top-level GPT parse entry point.
///
/// Reads the protective MBR at LBA 0, the primary GPT header at LBA 1,
/// and the partition entry array. Validates header signature, revision,
/// header CRC32, partition array CRC32, and entry count bounds.
///
/// On primary corruption, falls through to the secondary GPT at the
/// disk's last LBA and uses it if its CRC passes (logging a one-time
/// warning). Returns [`GptError::BothHeadersCorrupt`] if both fail.
///
/// Hybrid-MBR layouts (some macOS dual-boot disks where the MBR
/// describes one GPT-protective entry plus other entries) are honored
/// as plain GPT — i.e., a non-protective MBR signature does not
/// invalidate a GPT that is otherwise well-formed.
pub fn parse_gpt(device: &dyn BlockDevice) -> Result<Vec<PartitionEntry>, GptError> {
    let bs = device.block_size_bytes() as usize;
    let block_count = device.block_count();
    if block_count < 4 {
        // Need at least LBA0 (MBR) + LBA1 (primary header) + 1 entry
        // block + secondary header. A 3-block disk is unsupportable.
        return Err(GptError::EntryCountOutOfRange);
    }

    // Step 1: read LBA 0, classify whether this looks like an MBR-only
    // disk or a (hybrid-)protective MBR + GPT.
    let mut lba0 = alloc::vec![0u8; bs];
    device.read_block(0, &mut lba0)?;
    // We do NOT reject hybrid-MBR layouts here. We only fail later if
    // BOTH GPT headers are corrupt AND LBA1 also lacks the GPT signature.
    let mbr_signature_present =
        bs >= 512 && lba0[510] == MBR_SIGNATURE_LE[0] && lba0[511] == MBR_SIGNATURE_LE[1];

    // Step 2: try primary GPT (LBA 1).
    let primary = read_and_validate_gpt(device, 1, false);

    // Step 3: try secondary GPT (last LBA).
    let secondary_lba = block_count - 1;
    let secondary = read_and_validate_gpt(device, secondary_lba, true);

    // Decide which to use.
    match (primary, secondary) {
        (Ok((header, entries)), _) => {
            // Sanity: header.current_lba should be 1.
            let _ = header;
            Ok(entries)
        }
        (Err(GptError::HeaderCrcMismatch), Err(GptError::HeaderCrcMismatch))
        | (Err(GptError::PartitionArrayCrcMismatch), Err(GptError::HeaderCrcMismatch))
        | (Err(GptError::HeaderCrcMismatch), Err(GptError::PartitionArrayCrcMismatch))
        | (Err(GptError::PartitionArrayCrcMismatch), Err(GptError::PartitionArrayCrcMismatch)) => {
            // Both corrupt — but if LBA0 had a valid MBR signature and
            // no GPT signature, classify as "MBR-only" instead.
            if mbr_signature_present && !lba0_looks_like_protective_mbr(&lba0) {
                Err(GptError::MbrOnlyDiskRejected)
            } else {
                Err(GptError::BothHeadersCorrupt)
            }
        }
        (Err(GptError::HeaderCrcMismatch), Ok((_, entries))) => {
            // Primary corrupt → fall through to secondary.
            warn_primary_fallthrough();
            Ok(entries)
        }
        (Err(GptError::PartitionArrayCrcMismatch), Ok((_, entries))) => {
            warn_primary_fallthrough();
            Ok(entries)
        }
        (Err(e), _) => Err(e),
    }
}

/// One-time warning latch for "primary GPT corrupt, using secondary".
static PRIMARY_FALLTHROUGH_WARNED: AtomicBool = AtomicBool::new(false);

fn warn_primary_fallthrough() {
    // Only warn on first occurrence per kernel boot. The current crate
    // has no logging infrastructure (the kernel log is in `kernel/`); we
    // record the state here so the boot-time mounter can read it via
    // [`primary_fallthrough_was_warned`] and emit the actual log line.
    PRIMARY_FALLTHROUGH_WARNED.store(true, Ordering::SeqCst);
}

/// Test introspection: whether [`parse_gpt`] has fallen through to the
/// secondary GPT at any point during this process's lifetime.
pub fn primary_fallthrough_was_warned() -> bool {
    PRIMARY_FALLTHROUGH_WARNED.load(Ordering::SeqCst)
}

/// Reset the one-time warn latch — only used by tests.
#[cfg(any(test, feature = "fs-on-disk-mounts"))]
pub fn reset_primary_fallthrough_latch() {
    PRIMARY_FALLTHROUGH_WARNED.store(false, Ordering::SeqCst);
}

/// Heuristic to detect a "pure" protective MBR (single partition entry
/// of type 0xEE covering most of the disk). If this returns false but
/// the MBR signature is present, the disk is classified as MBR-only.
fn lba0_looks_like_protective_mbr(lba0: &[u8]) -> bool {
    // The first MBR partition entry starts at offset 446. Each entry
    // is 16 bytes; the type byte is at offset +4.
    if lba0.len() < 512 {
        return false;
    }
    for slot in 0..4 {
        let off = 446 + slot * 16;
        if lba0[off + 4] == MBR_TYPE_GPT_PROTECTIVE {
            return true;
        }
    }
    false
}

/// Read and validate a GPT header at `header_lba` plus its partition
/// array. Returns the parsed header and the live (non-unused) entries.
///
/// `is_secondary` toggles whether `current_lba` is expected to equal
/// the device's last LBA (true) or LBA 1 (false).
fn read_and_validate_gpt(
    device: &dyn BlockDevice,
    header_lba: u64,
    is_secondary: bool,
) -> Result<(GptHeader, Vec<PartitionEntry>), GptError> {
    let bs = device.block_size_bytes() as usize;
    let mut header_block = alloc::vec![0u8; bs];
    device.read_block(header_lba, &mut header_block)?;

    let header = GptHeader::parse(&header_block)?;

    // Verify CRC32 over the first `header_size` bytes with the CRC
    // field zero'd.
    let computed = GptHeader::compute_header_crc(&header_block, header.header_size);
    if computed != header.header_crc32 {
        return Err(GptError::HeaderCrcMismatch);
    }

    // Sanity-check current_lba.
    if !is_secondary && header.current_lba != header_lba {
        return Err(GptError::HeaderCrcMismatch);
    }

    // Read the partition array.
    let total_bytes =
        (header.num_partition_entries as usize) * (header.partition_entry_size as usize);
    if total_bytes > PARTITION_ARRAY_MAX_BYTES {
        return Err(GptError::EntryCountOutOfRange);
    }
    // Round up to a multiple of `bs`.
    let blocks = total_bytes.div_ceil(bs);
    let mut array_buf = alloc::vec![0u8; blocks * bs];
    for i in 0..blocks {
        let block_lba = header
            .partition_entry_lba
            .checked_add(i as u64)
            .ok_or(GptError::EntryCountOutOfRange)?;
        let off = i * bs;
        device.read_block(block_lba, &mut array_buf[off..off + bs])?;
    }

    // Validate partition array CRC32 over exactly `total_bytes`.
    let array_crc = crc32(&array_buf[..total_bytes]);
    if array_crc != header.partition_array_crc32 {
        return Err(GptError::PartitionArrayCrcMismatch);
    }

    // Decode entries, skipping unused.
    let mut out = Vec::new();
    for i in 0..header.num_partition_entries as usize {
        let off = i * header.partition_entry_size as usize;
        let entry = PartitionEntry::parse(&array_buf[off..off + 128])?;
        if !entry.is_unused() {
            out.push(entry);
        }
    }

    Ok((header, out))
}

// ─── byte parsing helpers ────────────────────────────────────────────────────

#[inline]
fn u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
fn u64_le(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

// ─── CRC32 (IEEE 802.3 polynomial 0xEDB88320) ───────────────────────────────

/// CRC-32/ISO-HDLC, the polynomial used by GPT and UEFI.
///
/// Computed bit-reversed; same algorithm as `zlib`'s `crc32()` and
/// `crc32fast`'s default mode. Implemented inline so the `fs` crate
/// adds no new dependencies (clean-room rule).
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vectors() {
        // From ITU-T V.42 / IEEE 802.3 reference vectors.
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn guid_parse_and_render_roundtrip() {
        let s = "C12A7328-F81F-11D2-BA4B-00A0C93EC93B";
        let g = Guid::from_str_canonical(s).unwrap();
        // Mixed-endian on disk: time_low is little-endian.
        assert_eq!(g.0[0], 0x28);
        assert_eq!(g.0[1], 0x73);
        assert_eq!(g.0[2], 0x2A);
        assert_eq!(g.0[3], 0xC1);
        // time_mid little-endian.
        assert_eq!(g.0[4], 0x1F);
        assert_eq!(g.0[5], 0xF8);
        // time_hi little-endian.
        assert_eq!(g.0[6], 0xD2);
        assert_eq!(g.0[7], 0x11);
        // Last 8 bytes big-endian (as written).
        assert_eq!(g.0[8], 0xBA);
        assert_eq!(g.0[9], 0x4B);
        assert_eq!(&g.0[10..16], &[0x00, 0xA0, 0xC9, 0x3E, 0xC9, 0x3B]);
        // Round-trip the textual form.
        assert_eq!(g.to_string_canonical(), s);
    }

    #[test]
    fn guid_from_str_rejects_malformed() {
        assert!(Guid::from_str_canonical("").is_none());
        assert!(Guid::from_str_canonical("not-a-guid").is_none());
        // Wrong length.
        assert!(Guid::from_str_canonical("C12A7328-F81F-11D2-BA4B-00A0C93EC93").is_none());
        // Bad hex.
        assert!(Guid::from_str_canonical("C12A732G-F81F-11D2-BA4B-00A0C93EC93B").is_none());
        // Wrong hyphen positions.
        assert!(Guid::from_str_canonical("C12A73-28F81F-11D2-BA4B-00A0C93EC93B").is_none());
    }

    #[test]
    fn guid_lowercase_accepted() {
        let upper = Guid::from_str_canonical("C12A7328-F81F-11D2-BA4B-00A0C93EC93B").unwrap();
        let lower = Guid::from_str_canonical("c12a7328-f81f-11d2-ba4b-00a0c93ec93b").unwrap();
        assert_eq!(upper, lower);
    }

    #[test]
    fn guid_nil_works() {
        let g = Guid::nil();
        assert!(g.is_nil());
        assert_eq!(
            g.to_string_canonical(),
            "00000000-0000-0000-0000-000000000000"
        );
    }

    #[test]
    fn partition_name_ascii_decodes() {
        // "ESP" in UTF-16LE: 'E','S','P','\0'… padded.
        let mut bytes = [0u8; PARTITION_NAME_BYTES];
        bytes[0] = b'E';
        bytes[2] = b'S';
        bytes[4] = b'P';
        let name = PartitionName::from_utf16le_bytes(&bytes);
        assert_eq!(name.as_str(), "ESP");
    }

    #[test]
    fn partition_name_truncates_at_nul() {
        let mut bytes = [0u8; PARTITION_NAME_BYTES];
        bytes[0] = b'A';
        // bytes[2..4] = 0 → NUL terminator.
        bytes[4] = b'B';
        let name = PartitionName::from_utf16le_bytes(&bytes);
        assert_eq!(name.as_str(), "A");
    }

    #[test]
    fn partition_name_handles_lone_high_surrogate() {
        let mut bytes = [0u8; PARTITION_NAME_BYTES];
        // High surrogate 0xD800 followed by a non-surrogate.
        bytes[0] = 0x00;
        bytes[1] = 0xD8;
        bytes[2] = b'X';
        bytes[3] = 0;
        let name = PartitionName::from_utf16le_bytes(&bytes);
        // Expect U+FFFD then 'X'.
        assert!(name.as_str().starts_with('\u{FFFD}'));
        assert!(name.as_str().ends_with('X'));
    }

    #[test]
    fn partition_name_handles_supplementary_plane() {
        // U+1F4A9 PILE OF POO = surrogate pair (0xD83D, 0xDCA9).
        let mut bytes = [0u8; PARTITION_NAME_BYTES];
        bytes[0] = 0x3D;
        bytes[1] = 0xD8;
        bytes[2] = 0xA9;
        bytes[3] = 0xDC;
        let name = PartitionName::from_utf16le_bytes(&bytes);
        assert_eq!(name.as_str(), "\u{1F4A9}");
    }

    #[test]
    fn header_parse_rejects_bad_signature() {
        let block = [0u8; 92];
        assert!(matches!(
            GptHeader::parse(&block),
            Err(GptError::HeaderCrcMismatch)
        ));
    }

    #[test]
    fn header_parse_rejects_bad_revision() {
        let mut block = [0u8; 92];
        block[..8].copy_from_slice(&GPT_SIGNATURE);
        block[8..12].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        block[12..16].copy_from_slice(&92u32.to_le_bytes());
        assert!(matches!(
            GptHeader::parse(&block),
            Err(GptError::UnsupportedRevision)
        ));
    }

    #[test]
    fn header_parse_rejects_zero_entry_count() {
        let mut block = [0u8; 92];
        block[..8].copy_from_slice(&GPT_SIGNATURE);
        block[8..12].copy_from_slice(&GPT_REVISION_1_0.to_le_bytes());
        block[12..16].copy_from_slice(&92u32.to_le_bytes());
        // num_partition_entries=0 (offset 80..84)
        block[80..84].copy_from_slice(&0u32.to_le_bytes());
        block[84..88].copy_from_slice(&128u32.to_le_bytes());
        assert!(matches!(
            GptHeader::parse(&block),
            Err(GptError::EntryCountOutOfRange)
        ));
    }

    #[test]
    fn header_parse_rejects_oversized_entry_count() {
        let mut block = [0u8; 92];
        block[..8].copy_from_slice(&GPT_SIGNATURE);
        block[8..12].copy_from_slice(&GPT_REVISION_1_0.to_le_bytes());
        block[12..16].copy_from_slice(&92u32.to_le_bytes());
        // 200 entries * 128 bytes = 25600 > 16384.
        block[80..84].copy_from_slice(&200u32.to_le_bytes());
        block[84..88].copy_from_slice(&128u32.to_le_bytes());
        assert!(matches!(
            GptHeader::parse(&block),
            Err(GptError::EntryCountOutOfRange)
        ));
    }

    #[test]
    fn header_parse_rejects_non_pow2_entry_size() {
        let mut block = [0u8; 92];
        block[..8].copy_from_slice(&GPT_SIGNATURE);
        block[8..12].copy_from_slice(&GPT_REVISION_1_0.to_le_bytes());
        block[12..16].copy_from_slice(&92u32.to_le_bytes());
        block[80..84].copy_from_slice(&128u32.to_le_bytes());
        // 192 is not a power of two.
        block[84..88].copy_from_slice(&192u32.to_le_bytes());
        assert!(matches!(
            GptHeader::parse(&block),
            Err(GptError::EntryCountOutOfRange)
        ));
    }

    #[test]
    fn gpt_error_block_error_conversion() {
        let e: GptError = BlockError::Timeout.into();
        assert!(matches!(e, GptError::BlockError(BlockError::Timeout)));
    }

    #[test]
    fn gpt_error_display_includes_kind() {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        write!(s, "{}", GptError::BothHeadersCorrupt).unwrap();
        assert!(s.contains("both"));
    }
}
