// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Smoke tests for the synthetic-disk fixtures in `tests/common/mod.rs`.
//!
//! Larger-scope spec scenarios live in [`gpt_conformance.rs`].

#![cfg(feature = "fs-on-disk-mounts")]

mod common;

use smallaios_fs::block::BlockDevice;
use smallaios_fs::gpt::{layout::*, parse_gpt, parse_layout, GptError};

#[test]
fn synthetic_valid_disk_parses() {
    let dev = common::build_valid_v1_disk(common::DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).expect("valid disk should parse");
    assert_eq!(entries.len(), 5);
}

#[test]
fn synthetic_valid_disk_layout_matches_v1() {
    let dev = common::build_valid_v1_disk(common::DEFAULT_SECTORS);
    let entries = parse_gpt(&dev).unwrap();
    let layout = parse_layout(&entries).unwrap();
    assert_eq!(layout.esp.type_guid, ESP_TYPE_GUID);
    assert_eq!(layout.squashfs_a.type_guid, SQUASHFS_SLOT_A_TYPE_GUID);
    assert_eq!(layout.squashfs_b.type_guid, SQUASHFS_SLOT_B_TYPE_GUID);
    assert_eq!(layout.data_f2fs.type_guid, F2FS_TYPE_GUID);
    assert_eq!(layout.boot_config.type_guid, BOOT_CONFIG_TYPE_GUID);
}

#[test]
fn synthetic_disk_block_count_correct() {
    let dev = common::build_valid_v1_disk(common::DEFAULT_SECTORS);
    assert_eq!(dev.block_count(), common::DEFAULT_SECTORS);
    assert_eq!(dev.block_size_bytes(), 512);
}

#[test]
fn synthetic_disk_parses_at_4k_block_size() {
    let dev = common::build_valid_v1_disk_with_block_size(256, 4096);
    let entries = parse_gpt(&dev).unwrap();
    assert_eq!(entries.len(), 5);
    let layout = parse_layout(&entries).unwrap();
    assert_eq!(layout.boot_config.type_guid, BOOT_CONFIG_TYPE_GUID);
}

#[test]
fn corrupting_primary_header_makes_secondary_take_over() {
    let mut dev = common::build_valid_v1_disk(common::DEFAULT_SECTORS);
    common::corrupt_primary_header(&mut dev);
    let entries = parse_gpt(&dev).unwrap();
    assert_eq!(entries.len(), 5);
}

#[test]
fn corrupting_both_headers_rejects() {
    let mut dev = common::build_valid_v1_disk(common::DEFAULT_SECTORS);
    common::corrupt_primary_header(&mut dev);
    common::corrupt_secondary_header(&mut dev);
    let res = parse_gpt(&dev);
    assert!(matches!(res, Err(GptError::BothHeadersCorrupt)));
}
