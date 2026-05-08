// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! v1 SmallAIOS GPT layout enforcement.
//!
//! Per the `fs-partition-table` spec, a SmallAIOS-formatted disk has
//! exactly five partitions in a fixed order:
//!
//! | Idx | Type GUID                              | Size      | Purpose                            |
//! |----:|----------------------------------------|-----------|------------------------------------|
//! |   1 | `C12A7328-F81F-11D2-BA4B-00A0C93EC93B` | 256 MiB   | UEFI ESP                           |
//! |   2 | `A3F7C2E0-FACE-4FFF-AAAA-000000000001` | 4 GiB     | squashfs `/models/` slot A         |
//! |   3 | `A3F7C2E0-FACE-4FFF-AAAA-000000000002` | 4 GiB     | squashfs `/models/` slot B         |
//! |   4 | `8DA63339-0007-60C0-C436-083AC8230908` | remainder | F2FS `/data/`                      |
//! |   5 | `A3F7C2E0-FACE-4FFF-BBBB-000000000000` | 8 MiB     | A/B boot config                    |
//!
//! [`parse_layout`] enforces both the count and the type-GUID-by-slot
//! ordering and rejects with [`GptError::LayoutMismatch`] on any
//! deviation.

use super::{GptError, Guid, PartitionEntry};

/// EFI System Partition type GUID (UEFI 2.10 §5.3.3 / canonical).
//
// lgtm[rust/hard-coded-cryptographic-value] — UEFI ESP type GUID per UEFI 2.10, public constant
pub const ESP_TYPE_GUID: Guid = Guid([
    0x28, 0x73, 0x2A, 0xC1, 0x1F, 0xF8, 0xD2, 0x11, 0xBA, 0x4B, 0x00, 0xA0, 0xC9, 0x3E, 0xC9, 0x3B,
]);

/// SmallAIOS squashfs slot A type GUID (mixed-endian on disk).
///
/// Canonical text: `A3F7C2E0-FACE-4FFF-AAAA-000000000001`.
//
// lgtm[rust/hard-coded-cryptographic-value] — SmallAIOS-registered type GUID, public constant per docs/architecture.md
pub const SQUASHFS_SLOT_A_TYPE_GUID: Guid = Guid([
    0xE0, 0xC2, 0xF7, 0xA3, 0xCE, 0xFA, 0xFF, 0x4F, 0xAA, 0xAA, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
]);

/// SmallAIOS squashfs slot B type GUID (mixed-endian on disk).
///
/// Canonical text: `A3F7C2E0-FACE-4FFF-AAAA-000000000002`.
//
// lgtm[rust/hard-coded-cryptographic-value] — SmallAIOS-registered type GUID, public constant per docs/architecture.md
pub const SQUASHFS_SLOT_B_TYPE_GUID: Guid = Guid([
    0xE0, 0xC2, 0xF7, 0xA3, 0xCE, 0xFA, 0xFF, 0x4F, 0xAA, 0xAA, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
]);

/// Linux F2FS partition type GUID (canonical text:
/// `8DA63339-0007-60C0-C436-083AC8230908`).
//
// lgtm[rust/hard-coded-cryptographic-value] — Linux F2FS type GUID per util-linux registry, public constant
pub const F2FS_TYPE_GUID: Guid = Guid([
    0x39, 0x33, 0xA6, 0x8D, 0x07, 0x00, 0xC0, 0x60, 0xC4, 0x36, 0x08, 0x3A, 0xC8, 0x23, 0x09, 0x08,
]);

/// SmallAIOS A/B boot config type GUID (mixed-endian on disk).
///
/// Canonical text: `A3F7C2E0-FACE-4FFF-BBBB-000000000000`.
//
// lgtm[rust/hard-coded-cryptographic-value] — SmallAIOS-registered type GUID, public constant per docs/architecture.md
pub const BOOT_CONFIG_TYPE_GUID: Guid = Guid([
    0xE0, 0xC2, 0xF7, 0xA3, 0xCE, 0xFA, 0xFF, 0x4F, 0xBB, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
]);

/// Parsed v1 SmallAIOS layout.
///
/// Each field is a clone of the corresponding [`PartitionEntry`] from
/// the GPT parse; consumers may inspect `starting_lba` / `ending_lba`
/// to set up squashfs / F2FS / boot-config readers.
#[derive(Debug, Clone)]
pub struct SmallaiosLayout {
    /// Partition #1 — UEFI ESP.
    pub esp: PartitionEntry,
    /// Partition #2 — squashfs `/models/` slot A.
    pub squashfs_a: PartitionEntry,
    /// Partition #3 — squashfs `/models/` slot B.
    pub squashfs_b: PartitionEntry,
    /// Partition #4 — F2FS `/data/`.
    pub data_f2fs: PartitionEntry,
    /// Partition #5 — A/B boot config.
    pub boot_config: PartitionEntry,
}

/// Validate that `entries` matches the v1 SmallAIOS layout in count,
/// order, and type GUIDs.
///
/// # Errors
///
/// - [`GptError::LayoutMismatch`] — count != 5, ordering wrong, OR
///   any slot's type GUID doesn't match the v1 expectation.
pub fn parse_layout(entries: &[PartitionEntry]) -> Result<SmallaiosLayout, GptError> {
    if entries.len() != 5 {
        return Err(GptError::LayoutMismatch);
    }
    let expected = [
        (&entries[0].type_guid, &ESP_TYPE_GUID),
        (&entries[1].type_guid, &SQUASHFS_SLOT_A_TYPE_GUID),
        (&entries[2].type_guid, &SQUASHFS_SLOT_B_TYPE_GUID),
        (&entries[3].type_guid, &F2FS_TYPE_GUID),
        (&entries[4].type_guid, &BOOT_CONFIG_TYPE_GUID),
    ];
    for (got, want) in expected {
        if got != want {
            return Err(GptError::LayoutMismatch);
        }
    }
    Ok(SmallaiosLayout {
        esp: entries[0].clone(),
        squashfs_a: entries[1].clone(),
        squashfs_b: entries[2].clone(),
        data_f2fs: entries[3].clone(),
        boot_config: entries[4].clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::PartitionName;

    fn entry(type_guid: Guid) -> PartitionEntry {
        PartitionEntry {
            type_guid,
            unique_guid: Guid::from_str_canonical("11111111-2222-3333-4444-555555555555").unwrap(),
            starting_lba: 100,
            ending_lba: 200,
            attributes: 0,
            name: PartitionName::empty(),
        }
    }

    #[test]
    fn esp_guid_matches_canonical_text() {
        let canonical = Guid::from_str_canonical("C12A7328-F81F-11D2-BA4B-00A0C93EC93B").unwrap();
        assert_eq!(canonical, ESP_TYPE_GUID);
    }

    #[test]
    fn squashfs_slot_a_guid_matches_canonical_text() {
        let canonical = Guid::from_str_canonical("A3F7C2E0-FACE-4FFF-AAAA-000000000001").unwrap();
        assert_eq!(canonical, SQUASHFS_SLOT_A_TYPE_GUID);
    }

    #[test]
    fn squashfs_slot_b_guid_matches_canonical_text() {
        let canonical = Guid::from_str_canonical("A3F7C2E0-FACE-4FFF-AAAA-000000000002").unwrap();
        assert_eq!(canonical, SQUASHFS_SLOT_B_TYPE_GUID);
    }

    #[test]
    fn f2fs_guid_matches_canonical_text() {
        let canonical = Guid::from_str_canonical("8DA63339-0007-60C0-C436-083AC8230908").unwrap();
        assert_eq!(canonical, F2FS_TYPE_GUID);
    }

    #[test]
    fn boot_config_guid_matches_canonical_text() {
        let canonical = Guid::from_str_canonical("A3F7C2E0-FACE-4FFF-BBBB-000000000000").unwrap();
        assert_eq!(canonical, BOOT_CONFIG_TYPE_GUID);
    }

    #[test]
    fn correct_layout_parses() {
        let entries = alloc::vec![
            entry(ESP_TYPE_GUID),
            entry(SQUASHFS_SLOT_A_TYPE_GUID),
            entry(SQUASHFS_SLOT_B_TYPE_GUID),
            entry(F2FS_TYPE_GUID),
            entry(BOOT_CONFIG_TYPE_GUID),
        ];
        let layout = parse_layout(&entries).unwrap();
        assert_eq!(layout.esp.type_guid, ESP_TYPE_GUID);
        assert_eq!(layout.squashfs_a.type_guid, SQUASHFS_SLOT_A_TYPE_GUID);
        assert_eq!(layout.squashfs_b.type_guid, SQUASHFS_SLOT_B_TYPE_GUID);
        assert_eq!(layout.data_f2fs.type_guid, F2FS_TYPE_GUID);
        assert_eq!(layout.boot_config.type_guid, BOOT_CONFIG_TYPE_GUID);
    }

    #[test]
    fn fewer_than_five_entries_rejected() {
        let entries = alloc::vec![
            entry(ESP_TYPE_GUID),
            entry(SQUASHFS_SLOT_A_TYPE_GUID),
            entry(SQUASHFS_SLOT_B_TYPE_GUID),
            entry(F2FS_TYPE_GUID),
        ];
        assert!(matches!(
            parse_layout(&entries),
            Err(GptError::LayoutMismatch)
        ));
    }

    #[test]
    fn more_than_five_entries_rejected() {
        let entries = alloc::vec![
            entry(ESP_TYPE_GUID),
            entry(SQUASHFS_SLOT_A_TYPE_GUID),
            entry(SQUASHFS_SLOT_B_TYPE_GUID),
            entry(F2FS_TYPE_GUID),
            entry(BOOT_CONFIG_TYPE_GUID),
            entry(ESP_TYPE_GUID),
        ];
        assert!(matches!(
            parse_layout(&entries),
            Err(GptError::LayoutMismatch)
        ));
    }

    #[test]
    fn wrong_order_rejected() {
        // Slot A and B swapped.
        let entries = alloc::vec![
            entry(ESP_TYPE_GUID),
            entry(SQUASHFS_SLOT_B_TYPE_GUID),
            entry(SQUASHFS_SLOT_A_TYPE_GUID),
            entry(F2FS_TYPE_GUID),
            entry(BOOT_CONFIG_TYPE_GUID),
        ];
        assert!(matches!(
            parse_layout(&entries),
            Err(GptError::LayoutMismatch)
        ));
    }

    #[test]
    fn missing_f2fs_rejected() {
        let entries = alloc::vec![
            entry(ESP_TYPE_GUID),
            entry(SQUASHFS_SLOT_A_TYPE_GUID),
            entry(SQUASHFS_SLOT_B_TYPE_GUID),
            entry(BOOT_CONFIG_TYPE_GUID), // wrong: should be F2FS
            entry(BOOT_CONFIG_TYPE_GUID),
        ];
        assert!(matches!(
            parse_layout(&entries),
            Err(GptError::LayoutMismatch)
        ));
    }

    #[test]
    fn missing_boot_config_rejected() {
        // Five entries but #5 has wrong type.
        let foreign = Guid::from_str_canonical("DEADBEEF-1234-5678-9ABC-DEF012345678").unwrap();
        let entries = alloc::vec![
            entry(ESP_TYPE_GUID),
            entry(SQUASHFS_SLOT_A_TYPE_GUID),
            entry(SQUASHFS_SLOT_B_TYPE_GUID),
            entry(F2FS_TYPE_GUID),
            entry(foreign),
        ];
        assert!(matches!(
            parse_layout(&entries),
            Err(GptError::LayoutMismatch)
        ));
    }

    #[test]
    fn foreign_guid_in_squashfs_slot_rejected() {
        let foreign = Guid::from_str_canonical("11112222-3333-4444-5555-666666666666").unwrap();
        let entries = alloc::vec![
            entry(ESP_TYPE_GUID),
            entry(foreign), // wrong type for squashfs slot A
            entry(SQUASHFS_SLOT_B_TYPE_GUID),
            entry(F2FS_TYPE_GUID),
            entry(BOOT_CONFIG_TYPE_GUID),
        ];
        assert!(matches!(
            parse_layout(&entries),
            Err(GptError::LayoutMismatch)
        ));
    }
}
