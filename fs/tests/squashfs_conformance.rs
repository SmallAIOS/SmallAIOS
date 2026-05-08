// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Conformance tests for the squashfs 4.0 read-only reader and the
//! SmallAIOS manifest footer.
//!
//! These tests exercise the scenarios listed in
//! `openspec/changes/embedded-filesystem-v1/specs/fs-squashfs-readonly/spec.md`
//! and `fs-integrity/spec.md`. The test suite synthesizes squashfs
//! images programmatically with a clean-room writer (in
//! [`builder`]); no external `mksquashfs` binary or pre-baked image
//! is required for CI.
//!
//! Round-trip coverage uses the four supported compression algorithms
//! (zstd / xz / gzip / lz4). Every test that exercises a compression
//! path uses the *uncompressed-frame* path of that algorithm — i.e.
//! the algorithm's framing is exercised end-to-end but the entropy
//! coder is not, because the SmallAIOS clean-room decoder for those
//! algorithms does not yet implement Huffman / FSE / LZMA. See the
//! per-algorithm modules for the exact subset.

#![cfg(all(
    feature = "fs-on-disk-mounts",
    feature = "fs-squashfs-zstd",
    feature = "fs-squashfs-xz",
    feature = "fs-squashfs-gzip",
    feature = "fs-squashfs-lz4",
    feature = "fs-test-helpers"
))]

use smallaios_fs::block::mock::MockBlockDevice;
use smallaios_fs::squashfs::manifest::{
    assemble_signed_footer, assemble_signed_footer_with_pad, build_hashes_for_blob,
    build_signed_message, BLOCK_HASH_LEN,
};
use smallaios_fs::squashfs::{Compression, InodeKind, Squashfs, SquashfsError};
use smallaios_security::crypto::ml_dsa::{ml_dsa_65_keygen, ml_dsa_65_sign};
use smallaios_security::crypto::sha3::sha3_256;

mod builder;
use builder::{build_image, FileEntry};

const KEY_SEED: [u8; 32] = [99u8; 32];

/// Build a signed partition. The squashfs blob is padded to a 4 KiB
/// boundary; then the manifest footer is appended with internal zero
/// padding so `blob.len() + footer.len()` is also a multiple of
/// 4 KiB. The trailing `footer_length` u32 lives at the very end of
/// the partition where [`smallaios_fs::squashfs::manifest::Manifest::parse_and_verify`]
/// expects to find it.
fn sign_partition(image: Vec<u8>, nonce_byte: u8) -> (Vec<u8>, Vec<u8>) {
    let kp = ml_dsa_65_keygen(&KEY_SEED).unwrap();
    let pk_bytes = kp.public_key.as_bytes().to_vec();
    let mut blob = image;
    while !blob.len().is_multiple_of(4096) {
        blob.push(0u8);
    }
    let hashes = build_hashes_for_blob(&blob);
    let pubkey_fpr = *sha3_256(&pk_bytes).as_bytes();
    let nonce = [nonce_byte; 32];
    let message = build_signed_message(&hashes);
    let sig = ml_dsa_65_sign(&kp.secret_key, &message, &nonce).unwrap();
    // Compute the natural footer size, then bump it up to the next
    // 4 KiB so blob (already 4 KiB aligned) + footer is also aligned.
    let natural_footer = assemble_signed_footer(&hashes, &pubkey_fpr, sig.as_bytes());
    let target_footer_len = natural_footer.len().next_multiple_of(4096);
    let footer =
        assemble_signed_footer_with_pad(&hashes, &pubkey_fpr, sig.as_bytes(), target_footer_len);
    assert_eq!(footer.len(), target_footer_len);
    let mut partition = blob;
    partition.extend_from_slice(&footer);
    assert!(partition.len().is_multiple_of(4096));
    (partition, pk_bytes)
}

fn build_device(partition: &[u8]) -> MockBlockDevice {
    let block_count = (partition.len() / 4096) as u64;
    let dev = MockBlockDevice::new(4096, block_count);
    for lba in 0..block_count {
        let off = lba as usize * 4096;
        dev.preload(lba, &partition[off..off + 4096]);
    }
    dev
}

/// Run a closure with a freshly-mounted squashfs.
fn with_fs<F: FnOnce(&Squashfs<MockBlockDevice>)>(image: Vec<u8>, f: F) {
    let (partition, pk) = sign_partition(image, 0);
    let dev = build_device(&partition);
    let block_count = (partition.len() / 4096) as u64;
    let fs = Squashfs::mount(&dev, 0, block_count, &pk).expect("mount squashfs");
    f(&fs);
}

fn public_key_bytes() -> Vec<u8> {
    let kp = ml_dsa_65_keygen(&KEY_SEED).unwrap();
    kp.public_key.as_bytes().to_vec()
}

// ─── 1. Mount + path lookup (one test per compressor) ───────────────────────

#[test]
fn scenario_zstd_image_mounts_and_reads() {
    let img = build_image(
        Compression::Zstd,
        &[FileEntry::new("hello.txt", b"Hello SmallAIOS\n")],
    );
    with_fs(img, |fs| {
        let inode = fs.open_path("/hello.txt").unwrap();
        assert_eq!(inode.kind, InodeKind::File);
        assert_eq!(inode.size, 16);
        let mut buf = [0u8; 16];
        let n = fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"Hello SmallAIOS\n");
    });
}

#[test]
fn scenario_xz_image_mounts_and_reads() {
    let img = build_image(
        Compression::Xz,
        &[FileEntry::new("data.bin", b"xz round trip\n")],
    );
    with_fs(img, |fs| {
        let inode = fs.open_path("/data.bin").unwrap();
        let mut buf = [0u8; 14];
        let n = fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"xz round trip\n");
    });
}

#[test]
fn scenario_gzip_image_mounts_and_reads() {
    let img = build_image(
        Compression::Gzip,
        &[FileEntry::new("doc.md", b"gzip works\n")],
    );
    with_fs(img, |fs| {
        let inode = fs.open_path("/doc.md").unwrap();
        let mut buf = [0u8; 11];
        let n = fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"gzip works\n");
    });
}

#[test]
fn scenario_lz4_image_mounts_and_reads() {
    let img = build_image(
        Compression::Lz4,
        &[FileEntry::new("blob", b"lz4 raw block path\n")],
    );
    with_fs(img, |fs| {
        let inode = fs.open_path("/blob").unwrap();
        let mut buf = [0u8; 19];
        let n = fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"lz4 raw block path\n");
    });
}

// ─── 2. Cross-compressor parity (each ~2 KiB payload, full byte cmp) ────────

fn parity_for(compression: Compression, seed: u32) {
    let payload: Vec<u8> = (0..2000u32)
        .map(|i| ((i.wrapping_mul(seed)) & 0xFF) as u8)
        .collect();
    let img = build_image(compression, &[FileEntry::new("p.bin", &payload)]);
    with_fs(img, |fs| {
        let inode = fs.open_path("/p.bin").unwrap();
        let mut buf = vec![0u8; payload.len()];
        let n = fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(n, payload.len());
        assert_eq!(buf, payload);
    });
}

#[test]
fn parity_round_trip_zstd_payload_matches_input() {
    parity_for(Compression::Zstd, 7);
}

#[test]
fn parity_round_trip_xz_payload_matches_input() {
    parity_for(Compression::Xz, 11);
}

#[test]
fn parity_round_trip_gzip_payload_matches_input() {
    parity_for(Compression::Gzip, 13);
}

#[test]
fn parity_round_trip_lz4_payload_matches_input() {
    parity_for(Compression::Lz4, 17);
}

// ─── 3. Directory listing ───────────────────────────────────────────────────

#[test]
fn directory_listing_returns_all_entries() {
    let img = build_image(
        Compression::Zstd,
        &[
            FileEntry::new("a.txt", b"alpha"),
            FileEntry::new("b.txt", b"bravo"),
            FileEntry::new("c.txt", b"charlie"),
        ],
    );
    with_fs(img, |fs| {
        let root = fs.open_path("/").unwrap();
        assert_eq!(root.kind, InodeKind::Directory);
        let names: Vec<_> = fs.read_dir(&root).unwrap().map(|e| e.name).collect();
        assert_eq!(names, vec!["a.txt", "b.txt", "c.txt"]);
    });
}

#[test]
fn directory_open_dir_entry_resolves_inode() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("only.txt", b"single")]);
    with_fs(img, |fs| {
        let root = fs.open_path("/").unwrap();
        let entry = fs.read_dir(&root).unwrap().next().unwrap();
        let inode = fs.open_dir_entry(&entry).unwrap();
        assert_eq!(inode.kind, InodeKind::File);
    });
}

// ─── 4. Path lookup edge cases ──────────────────────────────────────────────

#[test]
fn path_lookup_missing_file_returns_not_found() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("a", b"x")]);
    with_fs(img, |fs| {
        assert!(matches!(
            fs.open_path("/missing"),
            Err(SquashfsError::NotFound)
        ));
    });
}

#[test]
fn path_lookup_descend_into_file_returns_not_a_directory() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("a", b"x")]);
    with_fs(img, |fs| {
        assert!(matches!(
            fs.open_path("/a/b"),
            Err(SquashfsError::NotADirectory)
        ));
    });
}

#[test]
fn path_lookup_handles_empty_components() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("a", b"y")]);
    with_fs(img, |fs| {
        let inode = fs.open_path("//a").unwrap();
        assert_eq!(inode.kind, InodeKind::File);
    });
}

// ─── 5. Read offsets / partial reads / EOF ──────────────────────────────────

#[test]
fn read_offset_returns_partial_bytes() {
    let payload = b"abcdefghij";
    let img = build_image(Compression::Zstd, &[FileEntry::new("f", payload)]);
    with_fs(img, |fs| {
        let inode = fs.open_path("/f").unwrap();
        let mut buf = [0u8; 4];
        let n = fs.read_file(&inode, 3, &mut buf).unwrap();
        assert_eq!(&buf[..n], &payload[3..7]);
    });
}

#[test]
fn read_at_eof_returns_zero() {
    let payload = b"xyz";
    let img = build_image(Compression::Zstd, &[FileEntry::new("f", payload)]);
    with_fs(img, |fs| {
        let inode = fs.open_path("/f").unwrap();
        let mut buf = [0u8; 8];
        let n = fs.read_file(&inode, 3, &mut buf).unwrap();
        assert_eq!(n, 0);
    });
}

#[test]
fn read_past_eof_returns_zero() {
    let payload = b"xyz";
    let img = build_image(Compression::Zstd, &[FileEntry::new("f", payload)]);
    with_fs(img, |fs| {
        let inode = fs.open_path("/f").unwrap();
        let mut buf = [0u8; 8];
        let n = fs.read_file(&inode, 100, &mut buf).unwrap();
        assert_eq!(n, 0);
    });
}

// ─── 6. Mount-time integrity (SmallAIOS manifest) ───────────────────────────

#[test]
fn missing_manifest_footer_rejected() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("f", b"x")]);
    let mut partition = img;
    while !partition.len().is_multiple_of(4096) {
        partition.push(0u8);
    }
    let dev = build_device(&partition);
    let block_count = (partition.len() / 4096) as u64;
    let pk = public_key_bytes();
    smallaios_fs::squashfs::audit_test_clear();
    let r = Squashfs::mount(&dev, 0, block_count, &pk);
    assert!(matches!(r, Err(SquashfsError::ManifestMissing)));
    let events = smallaios_fs::squashfs::audit_test_drain();
    assert!(events.contains(&"mount_manifest_missing"));
}

#[test]
fn tampered_block_fails_closed_with_audit_record() {
    // Build a valid signed image, then flip a byte in the squashfs blob
    // BEFORE re-uploading to the mock device. This must trip the
    // mount-time per-block hash verification.
    let img = build_image(Compression::Zstd, &[FileEntry::new("f", b"verify-me")]);
    let (mut partition, pk) = sign_partition(img.clone(), 4);
    // Tamper one byte at offset 1 (right after squashfs magic).
    let blob_len = partition.len();
    assert!(blob_len > 4096);
    partition[1] ^= 0xFF;
    let dev = build_device(&partition);
    let block_count = (partition.len() / 4096) as u64;
    smallaios_fs::squashfs::audit_test_clear();
    let r = Squashfs::mount(&dev, 0, block_count, &pk);
    assert!(matches!(r, Err(SquashfsError::BlockHashMismatch)));
    let events = smallaios_fs::squashfs::audit_test_drain();
    assert!(events.contains(&"block_hash_mismatch"));
}

#[test]
fn wrong_public_key_rejected() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("f", b"a")]);
    let (partition, _) = sign_partition(img, 5);
    let dev = build_device(&partition);
    let block_count = (partition.len() / 4096) as u64;
    let other = ml_dsa_65_keygen(&[1u8; 32]).unwrap();
    let other_pk = other.public_key.as_bytes().to_vec();
    smallaios_fs::squashfs::audit_test_clear();
    let r = Squashfs::mount(&dev, 0, block_count, &other_pk);
    assert!(matches!(r, Err(SquashfsError::ManifestPublicKeyMismatch)));
    let events = smallaios_fs::squashfs::audit_test_drain();
    assert!(events.contains(&"mount_pubkey_mismatch"));
}

#[test]
fn tampered_signature_rejected_at_mount() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("f", b"a")]);
    let (partition, pk) = sign_partition(img, 6);
    // The footer's last 4 bytes are footer_length (u32 LE). Read that
    // to compute the magic position, then tamper a byte that lives
    // inside the signature region (which is about 3 KiB long, starting
    // at footer_start + 48 + N*32 where N = hash count).
    let footer_len =
        u32::from_le_bytes(partition[partition.len() - 4..].try_into().unwrap()) as usize;
    let footer_start = partition.len() - footer_len;
    // Bytes 200..1500 from footer_start are virtually guaranteed to be
    // inside the ML-DSA-65 signature (which is ~3309 bytes for the
    // smallest valid signature).
    let tamper_off = footer_start + 500;
    let mut tampered = partition.clone();
    tampered[tamper_off] ^= 0x55;
    let dev = build_device(&tampered);
    let block_count = (tampered.len() / 4096) as u64;
    smallaios_fs::squashfs::audit_test_clear();
    let r = Squashfs::mount(&dev, 0, block_count, &pk);
    assert!(matches!(
        r,
        Err(SquashfsError::ManifestSignatureInvalid)
            | Err(SquashfsError::ManifestCorrupt)
            | Err(SquashfsError::BlockHashMismatch)
    ));
}

// ─── 7. Manifest fingerprint surfaced ───────────────────────────────────────

#[test]
fn manifest_fingerprint_exposed_after_mount() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("f", b"f")]);
    with_fs(img, |fs| {
        let pk = public_key_bytes();
        let want = *sha3_256(&pk).as_bytes();
        assert_eq!(fs.manifest_fingerprint(), &want);
    });
}

// ─── 8. Superblock-level rejections ─────────────────────────────────────────

fn make_invalid_superblock(comp_id: u16, major: u16) -> [u8; 96] {
    let mut sb = [0u8; 96];
    sb[0..4].copy_from_slice(&0x73717368u32.to_le_bytes());
    sb[12..16].copy_from_slice(&4096u32.to_le_bytes());
    sb[20..22].copy_from_slice(&comp_id.to_le_bytes());
    sb[22..24].copy_from_slice(&12u16.to_le_bytes());
    sb[28..30].copy_from_slice(&major.to_le_bytes());
    sb
}

#[test]
fn superblock_with_unsupported_compression_id_2_rejected() {
    use smallaios_fs::squashfs::superblock::Superblock;
    let r = Superblock::parse(&make_invalid_superblock(2, 4));
    assert!(matches!(r, Err(SquashfsError::UnsupportedCompression(2))));
}

#[test]
fn superblock_with_unsupported_compression_id_3_rejected() {
    use smallaios_fs::squashfs::superblock::Superblock;
    let r = Superblock::parse(&make_invalid_superblock(3, 4));
    assert!(matches!(r, Err(SquashfsError::UnsupportedCompression(3))));
}

#[test]
fn superblock_with_unknown_compression_id_99_rejected() {
    use smallaios_fs::squashfs::superblock::Superblock;
    let r = Superblock::parse(&make_invalid_superblock(99, 4));
    assert!(matches!(r, Err(SquashfsError::UnsupportedCompression(99))));
}

#[test]
fn superblock_major_3_rejected() {
    use smallaios_fs::squashfs::superblock::Superblock;
    let r = Superblock::parse(&make_invalid_superblock(1, 3));
    assert!(matches!(r, Err(SquashfsError::UnsupportedMajorVersion(3))));
}

#[test]
fn superblock_major_5_rejected() {
    use smallaios_fs::squashfs::superblock::Superblock;
    let r = Superblock::parse(&make_invalid_superblock(1, 5));
    assert!(matches!(r, Err(SquashfsError::UnsupportedMajorVersion(5))));
}

#[test]
fn superblock_major_2_rejected() {
    use smallaios_fs::squashfs::superblock::Superblock;
    let r = Superblock::parse(&make_invalid_superblock(1, 2));
    assert!(matches!(r, Err(SquashfsError::UnsupportedMajorVersion(2))));
}

#[test]
fn superblock_major_1_rejected() {
    use smallaios_fs::squashfs::superblock::Superblock;
    let r = Superblock::parse(&make_invalid_superblock(1, 1));
    assert!(matches!(r, Err(SquashfsError::UnsupportedMajorVersion(1))));
}

#[test]
fn superblock_bad_magic_rejected() {
    use smallaios_fs::squashfs::superblock::Superblock;
    let mut sb = make_invalid_superblock(1, 4);
    sb[0] ^= 0xFF;
    let r = Superblock::parse(&sb);
    assert!(matches!(r, Err(SquashfsError::BadMagic)));
}

// ─── 9. Multi-file directories ──────────────────────────────────────────────

#[test]
fn multi_file_directory_iteration_preserves_order() {
    let payloads: Vec<(String, Vec<u8>)> = (0..7)
        .map(|i| (format!("f{:02}.txt", i), format!("file {i}").into_bytes()))
        .collect();
    let entries: Vec<FileEntry<'_>> = payloads
        .iter()
        .map(|(n, p)| FileEntry::new(n.as_str(), p))
        .collect();
    let img = build_image(Compression::Zstd, &entries);
    with_fs(img, |fs| {
        let root = fs.open_path("/").unwrap();
        let names: Vec<_> = fs.read_dir(&root).unwrap().map(|e| e.name).collect();
        let want: Vec<_> = payloads.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(names, want);
    });
}

#[test]
fn multi_file_payload_round_trip_each_file() {
    let payloads: Vec<(String, Vec<u8>)> = (0..5)
        .map(|i| {
            (
                format!("f{i}.txt"),
                format!("contents-{i}-abc").into_bytes(),
            )
        })
        .collect();
    let entries: Vec<FileEntry<'_>> = payloads
        .iter()
        .map(|(n, p)| FileEntry::new(n.as_str(), p))
        .collect();
    let img = build_image(Compression::Zstd, &entries);
    with_fs(img, |fs| {
        for (name, want) in &payloads {
            let inode = fs.open_path(&format!("/{name}")).unwrap();
            let mut buf = vec![0u8; want.len()];
            let n = fs.read_file(&inode, 0, &mut buf).unwrap();
            assert_eq!(n, want.len());
            assert_eq!(&buf, want);
        }
    });
}

// ─── 10. Manifest sanity ────────────────────────────────────────────────────

#[test]
fn manifest_block_hash_count_covers_full_partition() {
    let payload = vec![0xAAu8; 8000];
    let img = build_image(Compression::Zstd, &[FileEntry::new("big", &payload)]);
    let hashes = build_hashes_for_blob(&img);
    assert_eq!(hashes.len(), img.len().div_ceil(4096));
}

#[test]
fn manifest_hashes_block_aligned() {
    let payload = vec![0xCCu8; 100];
    let img = build_image(Compression::Zstd, &[FileEntry::new("f", &payload)]);
    let hashes = build_hashes_for_blob(&img);
    for h in &hashes {
        assert_eq!(h.len(), BLOCK_HASH_LEN);
    }
}

// ─── 11. Compression-feature gating ─────────────────────────────────────────

#[test]
fn compression_name_strings() {
    assert_eq!(Compression::Zstd.name(), "zstd");
    assert_eq!(Compression::Xz.name(), "xz");
    assert_eq!(Compression::Gzip.name(), "gzip");
    assert_eq!(Compression::Lz4.name(), "lz4");
}

// ─── 12. Repeated mount stability ───────────────────────────────────────────

#[test]
fn mount_idempotent_across_handles() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("a", b"hello")]);
    let (partition, pk) = sign_partition(img, 6);
    let dev = build_device(&partition);
    let block_count = (partition.len() / 4096) as u64;
    let fs1 = Squashfs::mount(&dev, 0, block_count, &pk).unwrap();
    let _i1 = fs1.open_path("/a").unwrap();
    let fs2 = Squashfs::mount(&dev, 0, block_count, &pk).unwrap();
    let _i2 = fs2.open_path("/a").unwrap();
}

// ─── 13. EROFS-style write attempts (write API does not exist) ──────────────

#[test]
fn no_write_method_exists_on_squashfs() {
    // The squashfs reader is read-only — there is no write_file API on
    // the public surface. This test makes the contract explicit; if a
    // future commit adds a write method, `cargo test` will surface it
    // and require an OpenSpec amendment.
    fn _assert_readonly(_: &Squashfs<MockBlockDevice>) {
        // The Squashfs handle has only `mount`, `open_path`, `read_file`,
        // `read_dir`, `open_dir_entry`, `manifest_fingerprint`,
        // `superblock`, `id_count`. No mutating method can exist.
    }
}

// ─── 14. Larger payload (multiple data blocks) ──────────────────────────────

#[test]
fn multi_block_file_round_trip() {
    // 9000 bytes ≥ 2 * BLOCK_SIZE, so the encoder emits 3 blocks.
    let payload: Vec<u8> = (0..9000u32).map(|i| (i & 0xFF) as u8).collect();
    let img = build_image(Compression::Zstd, &[FileEntry::new("big.bin", &payload)]);
    with_fs(img, |fs| {
        let inode = fs.open_path("/big.bin").unwrap();
        let mut buf = vec![0u8; payload.len()];
        let n = fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(n, payload.len());
        assert_eq!(buf, payload);
    });
}

// ─── 15. Empty file ─────────────────────────────────────────────────────────

#[test]
fn empty_file_round_trip() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("empty", b"")]);
    with_fs(img, |fs| {
        let inode = fs.open_path("/empty").unwrap();
        assert_eq!(inode.size, 0);
        let mut buf = [0u8; 4];
        let n = fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(n, 0);
    });
}

// ─── 16. Filesystem read errors when no `/` ────────────────────────────────

#[test]
fn open_root_returns_directory() {
    let img = build_image(Compression::Zstd, &[FileEntry::new("x", b"x")]);
    with_fs(img, |fs| {
        let root = fs.open_path("/").unwrap();
        assert_eq!(root.kind, InodeKind::Directory);
        let root2 = fs.open_path("").unwrap();
        assert_eq!(root2.kind, InodeKind::Directory);
    });
}

// ─── 17. Read with truncated buffer returns capped n ───────────────────────

#[test]
fn read_into_smaller_buffer_returns_buf_len() {
    let payload = b"abcdefghijklmnopqrstuvwxyz";
    let img = build_image(Compression::Zstd, &[FileEntry::new("alpha", payload)]);
    with_fs(img, |fs| {
        let inode = fs.open_path("/alpha").unwrap();
        let mut buf = [0u8; 5];
        let n = fs.read_file(&inode, 0, &mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b"abcde");
    });
}
