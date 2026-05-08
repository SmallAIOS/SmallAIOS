// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPT parser conformance suite.
//!
//! Implements every observable scenario from
//! `openspec/changes/embedded-filesystem-v1/specs/fs-partition-table/spec.md`
//! plus boundary cases on top of that. Fixtures are generated in
//! memory (no binary blobs in the repo) by `tests/common/mod.rs`.

#![cfg(feature = "fs-on-disk-mounts")]

mod common;

use smallaios_fs::block::mock::MockBlockDevice;
use smallaios_fs::block::{BlockDevice, BlockError};
use smallaios_fs::gpt::{
    crc32, layout::*, parse_gpt, parse_layout, write_protective_mbr, GptError, Guid,
    PartitionEntry, PartitionName, GPT_REVISION_1_0, GPT_SIGNATURE, MBR_SIGNATURE_LE,
    MBR_TYPE_GPT_PROTECTIVE, PARTITION_ARRAY_MAX_BYTES, PARTITION_ENTRY_SIZE,
};

use common::*;

// ─── 1. Valid GPT parses successfully ───────────────────────────────────────

#[test]
fn scenario_valid_gpt_parses_successfully() {
    let dev = build_valid_v1_disk(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).expect("valid GPT must parse");
    assert_eq!(entries.len(), 5, "v1 layout has exactly 5 partitions");
}

#[test]
fn valid_gpt_returns_type_guids_and_lba_ranges() {
    let dev = build_valid_v1_disk(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    for e in &entries {
        assert!(
            !e.type_guid.is_nil(),
            "live entries have non-zero type GUID"
        );
        assert!(e.starting_lba > 0);
        assert!(e.ending_lba >= e.starting_lba);
    }
}

#[test]
fn valid_gpt_decodes_partition_names_utf8() {
    let dev = build_valid_v1_disk(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    assert_eq!(entries[0].name.as_str(), "ESP");
    assert_eq!(entries[1].name.as_str(), "Models A");
    assert_eq!(entries[2].name.as_str(), "Models B");
    assert_eq!(entries[3].name.as_str(), "Data");
    assert_eq!(entries[4].name.as_str(), "Boot");
}

#[test]
fn valid_gpt_unique_guids_distinct() {
    let dev = build_valid_v1_disk(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    let mut seen = vec![];
    for e in &entries {
        assert!(
            !seen.contains(&e.unique_guid),
            "unique_guid collision among entries"
        );
        seen.push(e.unique_guid);
    }
}

// ─── 2. Corrupt primary GPT falls through to secondary ───────────────────────

#[test]
fn scenario_primary_corrupt_falls_through_to_secondary() {
    let mut dev = build_valid_v1_disk(DEFAULT_SECTORS);
    corrupt_primary_header(&mut dev);
    let entries = parse_gpt(&dev).unwrap();
    assert_eq!(entries.len(), 5);
}

#[test]
fn primary_array_corrupt_falls_through_to_secondary() {
    let mut dev = build_valid_v1_disk(DEFAULT_SECTORS);
    corrupt_primary_array(&mut dev);
    let entries = parse_gpt(&dev).unwrap();
    assert_eq!(entries.len(), 5);
}

#[test]
fn obliterating_primary_falls_through_to_secondary() {
    let mut dev = build_valid_v1_disk(DEFAULT_SECTORS);
    obliterate_primary_header(&mut dev);
    let entries = parse_gpt(&dev).unwrap();
    assert_eq!(entries.len(), 5);
}

#[test]
fn primary_corrupt_secondary_valid_warning_latch_set() {
    use smallaios_fs::gpt::{primary_fallthrough_was_warned, reset_primary_fallthrough_latch};
    reset_primary_fallthrough_latch();
    let mut dev = build_valid_v1_disk(DEFAULT_SECTORS);
    corrupt_primary_header(&mut dev);
    let _ = parse_gpt(&dev).unwrap();
    assert!(
        primary_fallthrough_was_warned(),
        "fallthrough latch must be set so kernel logs warning once"
    );
}

// ─── 3. Both primary and secondary corrupt rejected ─────────────────────────

#[test]
fn scenario_both_corrupt_rejected() {
    let mut dev = build_valid_v1_disk(DEFAULT_SECTORS);
    corrupt_primary_header(&mut dev);
    corrupt_secondary_header(&mut dev);
    assert!(matches!(parse_gpt(&dev), Err(GptError::BothHeadersCorrupt)));
}

#[test]
fn obliterating_both_headers_rejected() {
    let mut dev = build_valid_v1_disk(DEFAULT_SECTORS);
    obliterate_primary_header(&mut dev);
    obliterate_secondary_header(&mut dev);
    let res = parse_gpt(&dev);
    // Both headers obliterated → looks like protective MBR with no GPT.
    // Per parser policy we report BothHeadersCorrupt (a protective MBR
    // is NOT classified as MBR-only).
    assert!(matches!(res, Err(GptError::BothHeadersCorrupt)));
}

// ─── 4. MBR-only disk rejected ──────────────────────────────────────────────

#[test]
fn scenario_mbr_only_disk_rejected() {
    let dev = build_mbr_only_disk(DEFAULT_SECTORS);
    let res = parse_gpt(&dev);
    assert!(matches!(res, Err(GptError::MbrOnlyDiskRejected)));
}

#[test]
fn protective_mbr_without_gpt_rejected_as_both_corrupt() {
    // A disk with only a protective MBR but no GPT is malformed.
    // Since MBR has type 0xEE, we don't classify as MBR-only; we report
    // BothHeadersCorrupt (LBA 1 lacks an "EFI PART" signature).
    let dev = build_protective_mbr_only_disk(DEFAULT_SECTORS);
    let res = parse_gpt(&dev);
    assert!(matches!(res, Err(GptError::BothHeadersCorrupt)));
}

// ─── 5. Protective MBR present after format ─────────────────────────────────

#[test]
fn scenario_protective_mbr_present_after_format() {
    let mut dev = MockBlockDevice::new(512, 1024);
    write_protective_mbr(&mut dev).unwrap();
    let lba0 = dev.snapshot(0, 512);
    // Type 0xEE.
    assert_eq!(lba0[450], MBR_TYPE_GPT_PROTECTIVE);
    // 0x55AA signature.
    assert_eq!(lba0[510], MBR_SIGNATURE_LE[0]);
    assert_eq!(lba0[511], MBR_SIGNATURE_LE[1]);
}

#[test]
fn protective_mbr_covers_entire_disk() {
    let mut dev = MockBlockDevice::new(512, 1024);
    write_protective_mbr(&mut dev).unwrap();
    let lba0 = dev.snapshot(0, 512);
    // Starting LBA = 1.
    assert_eq!(&lba0[454..458], &1u32.to_le_bytes());
    // Size = 1023 (block_count - 1).
    assert_eq!(&lba0[458..462], &1023u32.to_le_bytes());
}

// ─── 6. Correct layout mounts ───────────────────────────────────────────────

#[test]
fn scenario_correct_layout_parses() {
    let dev = build_valid_v1_disk(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    let layout = parse_layout(&entries).expect("v1 layout valid");
    // Each slot has the right type GUID.
    assert_eq!(layout.esp.type_guid, ESP_TYPE_GUID);
    assert_eq!(layout.squashfs_a.type_guid, SQUASHFS_SLOT_A_TYPE_GUID);
    assert_eq!(layout.squashfs_b.type_guid, SQUASHFS_SLOT_B_TYPE_GUID);
    assert_eq!(layout.data_f2fs.type_guid, F2FS_TYPE_GUID);
    assert_eq!(layout.boot_config.type_guid, BOOT_CONFIG_TYPE_GUID);
}

// ─── 7. Missing F2FS partition rejected ─────────────────────────────────────

#[test]
fn scenario_missing_f2fs_partition_rejected() {
    let dev = build_disk_missing_f2fs(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    assert!(matches!(
        parse_layout(&entries),
        Err(GptError::LayoutMismatch)
    ));
}

// ─── 8. Missing boot config partition rejected ──────────────────────────────

#[test]
fn scenario_missing_boot_config_partition_rejected() {
    let dev = build_disk_missing_boot_config(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    assert!(matches!(
        parse_layout(&entries),
        Err(GptError::LayoutMismatch)
    ));
}

#[test]
fn fewer_than_5_partitions_rejected() {
    let dev = build_disk_with_n_partitions(DEFAULT_SECTORS, 4);
    let entries = parse_gpt(&dev).unwrap();
    assert_eq!(entries.len(), 4);
    assert!(matches!(
        parse_layout(&entries),
        Err(GptError::LayoutMismatch)
    ));
}

#[test]
fn three_partitions_rejected() {
    let dev = build_disk_with_n_partitions(DEFAULT_SECTORS, 3);
    let entries = parse_gpt(&dev).unwrap();
    assert!(matches!(
        parse_layout(&entries),
        Err(GptError::LayoutMismatch)
    ));
}

// ─── 9. SmallAIOS GUIDs round-trip with canonical text ──────────────────────

#[test]
fn esp_type_guid_has_canonical_text() {
    let canonical = Guid::from_str_canonical("C12A7328-F81F-11D2-BA4B-00A0C93EC93B").unwrap();
    assert_eq!(canonical, ESP_TYPE_GUID);
    assert_eq!(
        ESP_TYPE_GUID.to_string_canonical(),
        "C12A7328-F81F-11D2-BA4B-00A0C93EC93B"
    );
}

#[test]
fn squashfs_slot_a_type_guid_has_canonical_text() {
    let canonical = Guid::from_str_canonical("A3F7C2E0-FACE-4FFF-AAAA-000000000001").unwrap();
    assert_eq!(canonical, SQUASHFS_SLOT_A_TYPE_GUID);
}

#[test]
fn squashfs_slot_b_type_guid_has_canonical_text() {
    let canonical = Guid::from_str_canonical("A3F7C2E0-FACE-4FFF-AAAA-000000000002").unwrap();
    assert_eq!(canonical, SQUASHFS_SLOT_B_TYPE_GUID);
}

#[test]
fn f2fs_type_guid_has_canonical_text() {
    let canonical = Guid::from_str_canonical("8DA63339-0007-60C0-C436-083AC8230908").unwrap();
    assert_eq!(canonical, F2FS_TYPE_GUID);
}

#[test]
fn boot_config_type_guid_has_canonical_text() {
    let canonical = Guid::from_str_canonical("A3F7C2E0-FACE-4FFF-BBBB-000000000000").unwrap();
    assert_eq!(canonical, BOOT_CONFIG_TYPE_GUID);
}

// ─── 10. Foreign GUID with same name rejected ───────────────────────────────

#[test]
fn scenario_foreign_guid_with_smallaios_name_rejected() {
    // Build a disk where the slot-A position has a non-SmallAIOS type
    // GUID (so the type-by-slot check in parse_layout fails). The
    // partition name in the fixture is "Models A" — a foreign-typed
    // partition labelled with the SmallAIOS naming convention is
    // exactly what the spec says to reject.
    let dev = build_disk_foreign_type_with_smallaios_name(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    // The name still says "Models A" …
    assert_eq!(entries[1].name.as_str(), "Models A");
    // … but the type GUID is foreign, so layout enforcement rejects.
    assert!(matches!(
        parse_layout(&entries),
        Err(GptError::LayoutMismatch)
    ));
}

// ─── 11. Header-level structural validation ─────────────────────────────────

#[test]
fn corrupting_revision_field_rejects() {
    let mut dev = build_valid_v1_disk(DEFAULT_SECTORS);
    let mut block = vec![0u8; 512];
    dev.read_block(1, &mut block).unwrap();
    // Corrupt revision field to 0x00020000.
    block[8..12].copy_from_slice(&0x0002_0000u32.to_le_bytes());
    // Also recompute CRC otherwise we'll fail there first; but the
    // spec wants us to detect the *unsupported revision* path. We
    // recompute the header CRC after setting the bad revision to
    // exercise the right branch.
    let header_size = u32::from_le_bytes([block[12], block[13], block[14], block[15]]) as usize;
    block[16..20].copy_from_slice(&0u32.to_le_bytes());
    let crc = crc32(&block[..header_size]);
    block[16..20].copy_from_slice(&crc.to_le_bytes());
    dev.write_block(1, &block).unwrap();
    // Same on secondary so we don't fall through to a valid copy.
    let secondary_lba = dev.block_count() - 1;
    let mut sb = vec![0u8; 512];
    dev.read_block(secondary_lba, &mut sb).unwrap();
    sb[8..12].copy_from_slice(&0x0002_0000u32.to_le_bytes());
    sb[16..20].copy_from_slice(&0u32.to_le_bytes());
    let crc2 = crc32(&sb[..header_size]);
    sb[16..20].copy_from_slice(&crc2.to_le_bytes());
    dev.write_block(secondary_lba, &sb).unwrap();

    let res = parse_gpt(&dev);
    assert!(matches!(res, Err(GptError::UnsupportedRevision)));
}

// ─── 12. Block-error propagation ────────────────────────────────────────────

#[test]
fn block_error_during_parse_propagates() {
    let dev = MockBlockDevice::new(512, 2048);
    dev.set_permanent_failure(BlockError::MediaError);
    let res = parse_gpt(&dev);
    assert!(matches!(
        res,
        Err(GptError::BlockError(BlockError::MediaError))
    ));
}

#[test]
fn write_protective_mbr_propagates_block_error() {
    let mut dev = MockBlockDevice::new(512, 2048);
    dev.set_permanent_failure(BlockError::Timeout);
    let res = write_protective_mbr(&mut dev);
    assert!(matches!(
        res,
        Err(GptError::BlockError(BlockError::Timeout))
    ));
}

// ─── 13. Hybrid MBR is honored ──────────────────────────────────────────────

#[test]
fn hybrid_mbr_is_honored_as_plain_gpt() {
    // A hybrid MBR has 0xEE plus other entries. We accept it as plain GPT.
    let dev = build_hybrid_mbr_disk(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    assert_eq!(entries.len(), 5);
}

// ─── 14. Layout enforcement order strict ────────────────────────────────────

#[test]
fn layout_rejects_swapped_slot_a_and_b() {
    let dev = build_valid_v1_disk(DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    // Swap entries 1 and 2 (slot A and slot B).
    let mut shuffled = entries.clone();
    shuffled.swap(1, 2);
    assert!(matches!(
        parse_layout(&shuffled),
        Err(GptError::LayoutMismatch)
    ));
}

#[test]
fn layout_rejects_empty_slice() {
    let entries: Vec<PartitionEntry> = vec![];
    assert!(matches!(
        parse_layout(&entries),
        Err(GptError::LayoutMismatch)
    ));
}

// ─── 15. PartitionName edge cases ───────────────────────────────────────────

#[test]
fn partition_name_empty_decodes_to_empty_string() {
    let bytes = [0u8; 72];
    let name = PartitionName::from_utf16le_bytes(&bytes);
    assert_eq!(name.as_str(), "");
    assert!(name.is_empty());
}

// ─── 16. Constants are sane ─────────────────────────────────────────────────

#[test]
fn constants_have_expected_values() {
    assert_eq!(GPT_SIGNATURE, *b"EFI PART");
    assert_eq!(GPT_REVISION_1_0, 0x0001_0000);
    assert_eq!(PARTITION_ENTRY_SIZE, 128);
    assert_eq!(PARTITION_ARRAY_MAX_BYTES, 16384);
    assert_eq!(MBR_SIGNATURE_LE, [0x55, 0xAA]);
    assert_eq!(MBR_TYPE_GPT_PROTECTIVE, 0xEE);
}

// ─── 17. CRC32 known-vector regression ──────────────────────────────────────

#[test]
fn crc32_known_vectors_regression() {
    assert_eq!(crc32(b""), 0x0000_0000);
    assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    // Standard 32-byte zero block.
    assert_eq!(crc32(&[0u8; 32]), 0x190A_55AD);
}
