// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic GPT disk-image fixture helpers shared across the
//! integration test files. Lives at `tests/common/mod.rs` so each
//! test binary can `mod common;` it.

#![cfg(feature = "fs-on-disk-mounts")]
#![allow(dead_code)] // Each test binary uses a different subset.

use smallaios_fs::block::mock::MockBlockDevice;
use smallaios_fs::block::BlockDevice;
use smallaios_fs::gpt::{
    crc32, layout::*, write_protective_mbr, Guid, GPT_REVISION_1_0, GPT_SIGNATURE,
};

/// Default fixture disk size in 512 B sectors (1 MiB).
pub const DEFAULT_SECTORS: u64 = 2048;

/// Build a fully-valid v1 SmallAIOS GPT on a freshly-zeroed mock device.
pub fn build_valid_v1_disk(sectors: u64) -> MockBlockDevice {
    build_valid_v1_disk_with_block_size(sectors, 512)
}

/// Build a fully-valid v1 SmallAIOS GPT at the given LBA size.
///
/// Supports 512-byte and 4096-byte sectors.
pub fn build_valid_v1_disk_with_block_size(sectors: u64, block_size: u32) -> MockBlockDevice {
    assert!(sectors > 100);
    assert!(block_size == 512 || block_size == 4096);

    let bs = block_size as usize;
    let mut dev = MockBlockDevice::new(block_size, sectors);

    // Protective MBR.
    write_protective_mbr(&mut dev).unwrap();

    let num_entries: u32 = 128;
    let entry_size: u32 = 128;
    let array_bytes = (num_entries * entry_size) as usize; // 16384

    // Number of LBAs the entry array occupies.
    let array_blocks = array_bytes.div_ceil(bs);

    // 1 (MBR) + 1 (primary header) + array_blocks (primary array)
    let usable_first: u64 = 2 + array_blocks as u64;
    let usable_last: u64 = sectors - 2 - array_blocks as u64;

    let chunk = (usable_last - usable_first + 1) / 5;
    assert!(chunk >= 1, "test disk too small");

    let starts = [
        usable_first,
        usable_first + chunk,
        usable_first + 2 * chunk,
        usable_first + 3 * chunk,
        usable_first + 4 * chunk,
    ];
    let ends = [
        starts[1] - 1,
        starts[2] - 1,
        starts[3] - 1,
        starts[4] - 1,
        usable_last,
    ];
    let types = [
        ESP_TYPE_GUID,
        SQUASHFS_SLOT_A_TYPE_GUID,
        SQUASHFS_SLOT_B_TYPE_GUID,
        F2FS_TYPE_GUID,
        BOOT_CONFIG_TYPE_GUID,
    ];

    let mut array = vec![0u8; array_bytes];
    for (i, ((ty, &start), &end)) in types.iter().zip(starts.iter()).zip(ends.iter()).enumerate() {
        let off = i * entry_size as usize;
        write_partition_entry(
            &mut array[off..off + entry_size as usize],
            ty,
            start,
            end,
            i as u32,
        );
    }
    let array_crc = crc32(&array);
    let disk_guid = Guid::from_str_canonical("DEADBEEF-FACE-CAFE-1234-567890ABCDEF").unwrap();

    let primary_header = build_header(
        sectors,
        1,
        sectors - 1,
        usable_first,
        usable_last,
        &disk_guid,
        2,
        num_entries,
        entry_size,
        array_crc,
    );
    let primary_block = pad_block(&primary_header, bs);
    dev.preload(1, &primary_block);

    let mut padded_array = vec![0u8; array_blocks * bs];
    padded_array[..array_bytes].copy_from_slice(&array);
    dev.preload(2, &padded_array);

    // Secondary at tail.
    let secondary_array_lba = sectors - 1 - array_blocks as u64;
    let secondary_header = build_header(
        sectors,
        sectors - 1,
        1,
        usable_first,
        usable_last,
        &disk_guid,
        secondary_array_lba,
        num_entries,
        entry_size,
        array_crc,
    );
    let secondary_block = pad_block(&secondary_header, bs);
    dev.preload(sectors - 1, &secondary_block);
    dev.preload(secondary_array_lba, &padded_array);

    dev
}

/// Build a 92-byte GPT header with a correct CRC32.
#[allow(clippy::too_many_arguments)]
pub fn build_header(
    _sectors: u64,
    current_lba: u64,
    backup_lba: u64,
    first_usable_lba: u64,
    last_usable_lba: u64,
    disk_guid: &Guid,
    partition_entry_lba: u64,
    num_partition_entries: u32,
    partition_entry_size: u32,
    partition_array_crc32: u32,
) -> [u8; 92] {
    let mut header = [0u8; 92];
    header[0..8].copy_from_slice(&GPT_SIGNATURE);
    header[8..12].copy_from_slice(&GPT_REVISION_1_0.to_le_bytes());
    header[12..16].copy_from_slice(&92u32.to_le_bytes());
    header[16..20].copy_from_slice(&0u32.to_le_bytes());
    header[24..32].copy_from_slice(&current_lba.to_le_bytes());
    header[32..40].copy_from_slice(&backup_lba.to_le_bytes());
    header[40..48].copy_from_slice(&first_usable_lba.to_le_bytes());
    header[48..56].copy_from_slice(&last_usable_lba.to_le_bytes());
    header[56..72].copy_from_slice(disk_guid.as_bytes());
    header[72..80].copy_from_slice(&partition_entry_lba.to_le_bytes());
    header[80..84].copy_from_slice(&num_partition_entries.to_le_bytes());
    header[84..88].copy_from_slice(&partition_entry_size.to_le_bytes());
    header[88..92].copy_from_slice(&partition_array_crc32.to_le_bytes());
    let crc = crc32(&header);
    header[16..20].copy_from_slice(&crc.to_le_bytes());
    header
}

/// Build one 128-byte partition entry.
pub fn write_partition_entry(
    buf: &mut [u8],
    type_guid: &Guid,
    starting_lba: u64,
    ending_lba: u64,
    name_index: u32,
) {
    assert!(buf.len() >= 128);
    buf[0..16].copy_from_slice(type_guid.as_bytes());
    let mut unique = [0u8; 16];
    unique[0..4].copy_from_slice(&name_index.to_le_bytes());
    unique[4..8].copy_from_slice(&0xFEEDFACEu32.to_le_bytes());
    unique[8..16].copy_from_slice(&0xCAFE_BABE_DEAD_BEEFu64.to_le_bytes());
    buf[16..32].copy_from_slice(&unique);
    buf[32..40].copy_from_slice(&starting_lba.to_le_bytes());
    buf[40..48].copy_from_slice(&ending_lba.to_le_bytes());
    buf[48..56].copy_from_slice(&0u64.to_le_bytes());
    let label = match name_index {
        0 => "ESP",
        1 => "Models A",
        2 => "Models B",
        3 => "Data",
        4 => "Boot",
        _ => "Unknown",
    };
    let mut off = 56;
    for c in label.chars() {
        let mut units = [0u16; 2];
        let n = c.encode_utf16(&mut units).len();
        for &u in &units[..n] {
            buf[off] = (u & 0xFF) as u8;
            buf[off + 1] = (u >> 8) as u8;
            off += 2;
            if off >= 56 + 72 {
                break;
            }
        }
    }
}

/// Pad `data` to a block-sized buffer.
pub fn pad_block(data: &[u8], block_size: usize) -> Vec<u8> {
    let mut out = vec![0u8; block_size];
    out[..data.len()].copy_from_slice(data);
    out
}

// ─── Corruption helpers ─────────────────────────────────────────────────────

/// Flip bytes in the primary header so its CRC fails.
pub fn corrupt_primary_header(dev: &mut MockBlockDevice) {
    let mut block = vec![0u8; dev.block_size_bytes() as usize];
    dev.read_block(1, &mut block).unwrap();
    block[20] ^= 0xFF;
    block[24] ^= 0xFF;
    dev.write_block(1, &block).unwrap();
}

/// Flip bytes in the secondary header so its CRC fails.
pub fn corrupt_secondary_header(dev: &mut MockBlockDevice) {
    let secondary_lba = dev.block_count() - 1;
    let mut block = vec![0u8; dev.block_size_bytes() as usize];
    dev.read_block(secondary_lba, &mut block).unwrap();
    block[20] ^= 0xFF;
    block[24] ^= 0xFF;
    dev.write_block(secondary_lba, &block).unwrap();
}

/// Flip a byte in the primary partition array.
pub fn corrupt_primary_array(dev: &mut MockBlockDevice) {
    let mut block = vec![0u8; dev.block_size_bytes() as usize];
    dev.read_block(2, &mut block).unwrap();
    block[100] ^= 0xFF;
    dev.write_block(2, &block).unwrap();
}

/// Zero out the GPT signature at LBA 1 entirely.
pub fn obliterate_primary_header(dev: &mut MockBlockDevice) {
    let block = vec![0u8; dev.block_size_bytes() as usize];
    dev.write_block(1, &block).unwrap();
}

/// Zero out the secondary header at the disk tail.
pub fn obliterate_secondary_header(dev: &mut MockBlockDevice) {
    let secondary_lba = dev.block_count() - 1;
    let block = vec![0u8; dev.block_size_bytes() as usize];
    dev.write_block(secondary_lba, &block).unwrap();
}

/// Build an MBR-only disk: a 0x55AA signature at LBA 0 with no
/// protective entry of type 0xEE, and zeroed LBA 1..end (no GPT).
pub fn build_mbr_only_disk(sectors: u64) -> MockBlockDevice {
    let mut dev = MockBlockDevice::new(512, sectors);
    let mut block = vec![0u8; 512];
    // First MBR partition entry: type 0x83 (Linux), not 0xEE.
    let pt_off = 446;
    block[pt_off + 4] = 0x83;
    // Boot signature 0x55AA.
    block[510] = 0x55;
    block[511] = 0xAA;
    dev.write_block(0, &block).unwrap();
    // LBA 1..end stays zero — no GPT.
    dev
}

/// Build an MBR-only disk where the type is 0xEE (looks protective) but
/// no GPT exists. This is malformed; the parser should reject as
/// "both corrupt" since the secondary GPT is also absent.
pub fn build_protective_mbr_only_disk(sectors: u64) -> MockBlockDevice {
    let mut dev = MockBlockDevice::new(512, sectors);
    write_protective_mbr(&mut dev).unwrap();
    // Don't write any GPT structures.
    dev
}

/// Build a hybrid-MBR disk: protective MBR PLUS the GPT, with one
/// extra non-0xEE entry in the MBR slot 2. This emulates macOS dual-
/// boot. Per Q-spec, the parser should still honor the GPT.
pub fn build_hybrid_mbr_disk(sectors: u64) -> MockBlockDevice {
    let mut dev = build_valid_v1_disk(sectors);
    // Read LBA 0, add a "fake" entry in MBR slot 1 (offset 446 + 16 = 462).
    let mut block = vec![0u8; 512];
    dev.read_block(0, &mut block).unwrap();
    block[462 + 4] = 0x07; // NTFS, just for show
    block[462 + 8..462 + 12].copy_from_slice(&100u32.to_le_bytes()); // start LBA
    block[462 + 12..462 + 16].copy_from_slice(&50u32.to_le_bytes()); // size
    dev.write_block(0, &block).unwrap();
    dev
}

/// Build a disk where partition #4 is not F2FS.
pub fn build_disk_missing_f2fs(sectors: u64) -> MockBlockDevice {
    build_disk_with_swapped_type(sectors, 3, BOOT_CONFIG_TYPE_GUID)
}

/// Build a disk where partition #5 has the wrong type.
pub fn build_disk_missing_boot_config(sectors: u64) -> MockBlockDevice {
    build_disk_with_swapped_type(sectors, 4, F2FS_TYPE_GUID)
}

/// Build a disk where partition #2 has a foreign type GUID but the
/// partition name is "SmallAIOS Models A" (the spec scenario).
pub fn build_disk_foreign_type_with_smallaios_name(sectors: u64) -> MockBlockDevice {
    let foreign = Guid::from_str_canonical("11112222-3333-4444-5555-666666666666").unwrap();
    build_disk_with_swapped_type(sectors, 1, foreign)
}

/// Generic helper: rebuild the partition array with the given index's
/// type GUID replaced and rewrite the primary + secondary headers.
fn build_disk_with_swapped_type(
    sectors: u64,
    swap_index: usize,
    new_type: Guid,
) -> MockBlockDevice {
    let bs = 512usize;
    let mut dev = MockBlockDevice::new(bs as u32, sectors);
    write_protective_mbr(&mut dev).unwrap();

    let num_entries: u32 = 128;
    let entry_size: u32 = 128;
    let array_bytes = (num_entries * entry_size) as usize;
    let array_blocks = array_bytes.div_ceil(bs);

    let usable_first: u64 = 2 + array_blocks as u64;
    let usable_last: u64 = sectors - 2 - array_blocks as u64;
    let chunk = (usable_last - usable_first + 1) / 5;

    let starts = [
        usable_first,
        usable_first + chunk,
        usable_first + 2 * chunk,
        usable_first + 3 * chunk,
        usable_first + 4 * chunk,
    ];
    let ends = [
        starts[1] - 1,
        starts[2] - 1,
        starts[3] - 1,
        starts[4] - 1,
        usable_last,
    ];
    let mut types = [
        ESP_TYPE_GUID,
        SQUASHFS_SLOT_A_TYPE_GUID,
        SQUASHFS_SLOT_B_TYPE_GUID,
        F2FS_TYPE_GUID,
        BOOT_CONFIG_TYPE_GUID,
    ];
    types[swap_index] = new_type;

    let mut array = vec![0u8; array_bytes];
    for (i, ((ty, &start), &end)) in types.iter().zip(starts.iter()).zip(ends.iter()).enumerate() {
        let off = i * entry_size as usize;
        write_partition_entry(
            &mut array[off..off + entry_size as usize],
            ty,
            start,
            end,
            i as u32,
        );
    }
    let array_crc = crc32(&array);
    let disk_guid = Guid::from_str_canonical("DEADBEEF-FACE-CAFE-1234-567890ABCDEF").unwrap();

    let primary_header = build_header(
        sectors,
        1,
        sectors - 1,
        usable_first,
        usable_last,
        &disk_guid,
        2,
        num_entries,
        entry_size,
        array_crc,
    );
    dev.preload(1, &pad_block(&primary_header, bs));

    let mut padded_array = vec![0u8; array_blocks * bs];
    padded_array[..array_bytes].copy_from_slice(&array);
    dev.preload(2, &padded_array);

    let secondary_array_lba = sectors - 1 - array_blocks as u64;
    let secondary_header = build_header(
        sectors,
        sectors - 1,
        1,
        usable_first,
        usable_last,
        &disk_guid,
        secondary_array_lba,
        num_entries,
        entry_size,
        array_crc,
    );
    dev.preload(sectors - 1, &pad_block(&secondary_header, bs));
    dev.preload(secondary_array_lba, &padded_array);

    dev
}

/// Build a disk with too few partitions (e.g., only 4).
pub fn build_disk_with_n_partitions(sectors: u64, n: usize) -> MockBlockDevice {
    let bs = 512usize;
    let mut dev = MockBlockDevice::new(bs as u32, sectors);
    write_protective_mbr(&mut dev).unwrap();

    let num_entries: u32 = 128;
    let entry_size: u32 = 128;
    let array_bytes = (num_entries * entry_size) as usize;
    let array_blocks = array_bytes.div_ceil(bs);

    let usable_first: u64 = 2 + array_blocks as u64;
    let usable_last: u64 = sectors - 2 - array_blocks as u64;
    let chunk = (usable_last - usable_first + 1) / 5;

    let mut array = vec![0u8; array_bytes];
    let types = [
        ESP_TYPE_GUID,
        SQUASHFS_SLOT_A_TYPE_GUID,
        SQUASHFS_SLOT_B_TYPE_GUID,
        F2FS_TYPE_GUID,
        BOOT_CONFIG_TYPE_GUID,
    ];
    for (i, ty) in types.iter().enumerate().take(n.min(5)) {
        let off = i * entry_size as usize;
        let start = usable_first + i as u64 * chunk;
        let end = if i + 1 < 5 {
            start + chunk - 1
        } else {
            usable_last
        };
        write_partition_entry(
            &mut array[off..off + entry_size as usize],
            ty,
            start,
            end,
            i as u32,
        );
    }
    let array_crc = crc32(&array);
    let disk_guid = Guid::from_str_canonical("DEADBEEF-FACE-CAFE-1234-567890ABCDEF").unwrap();

    let primary_header = build_header(
        sectors,
        1,
        sectors - 1,
        usable_first,
        usable_last,
        &disk_guid,
        2,
        num_entries,
        entry_size,
        array_crc,
    );
    dev.preload(1, &pad_block(&primary_header, bs));

    let mut padded_array = vec![0u8; array_blocks * bs];
    padded_array[..array_bytes].copy_from_slice(&array);
    dev.preload(2, &padded_array);

    let secondary_array_lba = sectors - 1 - array_blocks as u64;
    let secondary_header = build_header(
        sectors,
        sectors - 1,
        1,
        usable_first,
        usable_last,
        &disk_guid,
        secondary_array_lba,
        num_entries,
        entry_size,
        array_crc,
    );
    dev.preload(sectors - 1, &pad_block(&secondary_header, bs));
    dev.preload(secondary_array_lba, &padded_array);

    dev
}
