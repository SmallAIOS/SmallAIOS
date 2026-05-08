// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Metadata-pair commit builder.
//!
//! This module turns a logical sequence of (name-tag, struct-tag) entries
//! plus an optional `tail` pointer into a byte stream that:
//!
//! * Starts with a 32-bit little-endian revision counter.
//! * Encodes each tag XOR'd against the previous tag word, big-endian
//!   on disk.
//! * Terminates with a CRC tag whose payload is the running CRC32C
//!   over everything previous (`!running_crc`).
//!
//! Phase 2 uses this to (a) materialise an in-memory metadata-pair half
//! before writing it to flash, and (b) commit a fresh half by erasing
//! the alternate block of the pair, programming the new bytes, and
//! incrementing the revision counter. The format matches the reader's
//! `parse_metadata_half` byte-for-byte.

use alloc::vec::Vec;

use super::format::{crc32c_update, pack_tag_word, TAG_CRC_NORMAL, TAG_TYPE3_CRC, TAG_TYPE3_TAIL};
use super::FlashCommitError;

/// One on-disk entry: the (name-tag + name-payload) followed by
/// (struct-tag + struct-payload) for a single id.
#[derive(Clone, Debug)]
pub(super) struct CommitEntry {
    pub typ_name: u16,
    pub id: u16,
    pub name: Vec<u8>,
    pub typ_struct: u16,
    pub struct_payload: Vec<u8>,
}

/// Build a metadata-pair half byte stream for `revision`.
///
/// The output starts with the 4-byte revision (little-endian) followed
/// by the XOR'd tag stream and a final CRC tag that commits everything.
///
/// `tail` (if `Some`) becomes a `tail`-typed tag at the end of the
/// entry list, before the CRC. Phase 1 reader populates
/// [`MetadataPairView::tail`](super::MetadataPairView) from the tag.
pub(super) fn build_metadata_half(
    revision: u32,
    entries: &[CommitEntry],
    tail: Option<[u32; 2]>,
) -> Result<Vec<u8>, FlashCommitError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&revision.to_le_bytes());

    let mut prev_tag_word: u32 = 0;
    let mut running_crc: u32 = 0xFFFF_FFFF;
    running_crc = crc32c_update(running_crc, &revision.to_le_bytes());

    for e in entries {
        // ── name tag ────────────────────────────────────────────────
        let name_len = e.name.len();
        if name_len > 0x3FF {
            return Err(FlashCommitError::TagTooLarge);
        }
        let name_tag_word = pack_tag_word(e.typ_name, e.id, name_len as u32);
        let on_disk = name_tag_word ^ prev_tag_word;
        buf.extend_from_slice(&on_disk.to_be_bytes());
        running_crc = crc32c_update(running_crc, &on_disk.to_be_bytes());
        buf.extend_from_slice(&e.name);
        running_crc = crc32c_update(running_crc, &e.name);
        prev_tag_word = name_tag_word;

        // ── struct tag ──────────────────────────────────────────────
        let payload_len = e.struct_payload.len();
        if payload_len > 0x3FF {
            return Err(FlashCommitError::TagTooLarge);
        }
        let struct_tag_word = pack_tag_word(e.typ_struct, e.id, payload_len as u32);
        let on_disk2 = struct_tag_word ^ prev_tag_word;
        buf.extend_from_slice(&on_disk2.to_be_bytes());
        running_crc = crc32c_update(running_crc, &on_disk2.to_be_bytes());
        buf.extend_from_slice(&e.struct_payload);
        running_crc = crc32c_update(running_crc, &e.struct_payload);
        prev_tag_word = struct_tag_word;
    }

    // ── tail tag (optional) ────────────────────────────────────────
    if let Some(t) = tail {
        let mut payload = Vec::with_capacity(8);
        payload.extend_from_slice(&t[0].to_le_bytes());
        payload.extend_from_slice(&t[1].to_le_bytes());
        let tail_typ = TAG_TYPE3_TAIL << 8;
        let tail_tag_word = pack_tag_word(tail_typ, 0, payload.len() as u32);
        let on_disk = tail_tag_word ^ prev_tag_word;
        buf.extend_from_slice(&on_disk.to_be_bytes());
        running_crc = crc32c_update(running_crc, &on_disk.to_be_bytes());
        buf.extend_from_slice(&payload);
        running_crc = crc32c_update(running_crc, &payload);
        prev_tag_word = tail_tag_word;
    }

    // ── final CRC tag ─────────────────────────────────────────────
    debug_assert_eq!((TAG_CRC_NORMAL >> 8), TAG_TYPE3_CRC);
    let crc_tag_word = pack_tag_word(TAG_CRC_NORMAL, 0, 4);
    let on_disk_crc = crc_tag_word ^ prev_tag_word;
    buf.extend_from_slice(&on_disk_crc.to_be_bytes());
    running_crc = crc32c_update(running_crc, &on_disk_crc.to_be_bytes());
    let final_crc = !running_crc;
    buf.extend_from_slice(&final_crc.to_le_bytes());

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash::littlefs::format::{TAG_NAME_REG, TAG_STRUCT_INLINE};

    #[test]
    fn builds_minimal_half_with_just_revision_and_crc() {
        let entries: [CommitEntry; 0] = [];
        let half = build_metadata_half(1, &entries, None).unwrap();
        // 4 (revision) + 4 (CRC tag word) + 4 (CRC payload) = 12 bytes.
        assert_eq!(half.len(), 12);
        // First 4 bytes are the LE revision.
        assert_eq!(&half[0..4], &1u32.to_le_bytes());
    }

    #[test]
    fn builds_one_entry_half() {
        let entries = [CommitEntry {
            typ_name: TAG_NAME_REG,
            id: 1,
            name: b"f".to_vec(),
            typ_struct: TAG_STRUCT_INLINE,
            struct_payload: b"x".to_vec(),
        }];
        let half = build_metadata_half(1, &entries, None).unwrap();
        // revision (4) + name-tag (4) + name (1) + struct-tag (4)
        // + struct-payload (1) + CRC-tag (4) + CRC payload (4) = 22.
        assert_eq!(half.len(), 22);
    }

    #[test]
    fn rejects_oversized_tag_payload() {
        let entries = [CommitEntry {
            typ_name: TAG_NAME_REG,
            id: 1,
            name: alloc::vec![0u8; 0x400], // == 0x3FF + 1
            typ_struct: TAG_STRUCT_INLINE,
            struct_payload: b"x".to_vec(),
        }];
        let res = build_metadata_half(1, &entries, None);
        assert!(matches!(res, Err(FlashCommitError::TagTooLarge)));
    }
}
