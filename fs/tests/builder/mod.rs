// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Clean-room squashfs 4.0 image builder for the conformance tests.
//!
//! Produces a minimal-but-valid squashfs 4.0 image containing exactly
//! one root directory and one or more regular files, with the
//! caller-chosen compression algorithm. The builder uses each
//! compressor's "uncompressed" framing path so the round-trip
//! exercises every layer of the SmallAIOS clean-room reader without
//! requiring the heavy entropy-coder paths (Huffman / FSE / LZMA).
//!
//! Layout produced (offsets in the resulting `Vec<u8>`):
//!
//! 1. Superblock (96 bytes at offset 0).
//! 2. Data blocks (one or more 4-KiB-aligned encoded data chunks).
//! 3. Inode table (uncompressed metadata blocks).
//! 4. Directory table (uncompressed metadata block).
//! 5. Fragment table top-level (no fragments — count = 0).
//! 6. ID table top-level (single u32 entry = 0).
//! 7. Padding to 4-byte boundary.

extern crate alloc;
use alloc::vec::Vec;

use smallaios_fs::squashfs::superblock::Compression;

const BLOCK_SIZE: u32 = 4096;
const BLOCK_LOG: u16 = 12;

const ROOT_INODE_NUMBER: u32 = 1;
const FILE_INODE_KIND: u16 = 2; // basic file
const DIR_INODE_KIND: u16 = 1; // basic dir

/// One file to add to the root directory.
pub struct FileEntry<'a> {
    pub name: &'a str,
    pub data: &'a [u8],
}

impl<'a> FileEntry<'a> {
    pub fn new(name: &'a str, data: &'a [u8]) -> Self {
        Self { name, data }
    }
}

/// Build a complete squashfs 4.0 image (no manifest footer — caller
/// adds that).
pub fn build_image(compression: Compression, files: &[FileEntry<'_>]) -> Vec<u8> {
    let mut img = Vec::new();
    // Reserve 96 bytes for the superblock; we patch in fields after we
    // know the table offsets.
    img.extend_from_slice(&[0u8; 96]);

    // ── 1. Encode each file's payload as a single data block ──
    // Each file has either zero or one data block; if file_size <=
    // BLOCK_SIZE, the trailing partial block is encoded as one
    // "compressed-or-stored" block (we always store uncompressed:
    // raw zstd block / raw lz4 sequence / DEFLATE stored block /
    // xz with one uncompressed LZMA2 chunk).

    struct EncodedFile {
        blocks_start: u64,
        block_sizes: Vec<u32>,
        file_size: u32,
    }

    let mut encoded: Vec<EncodedFile> = Vec::with_capacity(files.len());
    for f in files {
        let file_size = f.data.len() as u32;
        let block_count = (file_size as usize).div_ceil(BLOCK_SIZE as usize);
        let blocks_start = img.len() as u64;
        let mut block_sizes = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let start = i * BLOCK_SIZE as usize;
            let end = core::cmp::min(start + BLOCK_SIZE as usize, f.data.len());
            let chunk = &f.data[start..end];
            let bytes = encode_block(compression, chunk);
            block_sizes.push(bytes.len() as u32);
            img.extend_from_slice(&bytes);
        }
        encoded.push(EncodedFile {
            blocks_start,
            block_sizes,
            file_size,
        });
    }

    // ── 2. Inode table ──
    //
    // Layout (all fields little-endian unless noted):
    //   inode 1 (root dir): kind=1, perm=0x01ED, uid=0, gid=0,
    //     mtime=0, inode_number=1, start_block=0, nlink=2, size=...,
    //     offset=0, parent_inode=1.
    //   inode 2..=N (files): kind=2, perm=0x01A4, uid=0, gid=0,
    //     mtime=0, inode_number=k, blocks_start, frag_idx=0xFFFFFFFF,
    //     frag_offset=0, file_size=..., block_sizes[].
    //
    // Track each inode's metadata-reference (start_block_offset_in_metadata,
    // offset_in_block) so the directory entries can point at them.

    let inode_table_start = img.len() as u64;
    let mut inode_payload: Vec<u8> = Vec::new();
    // Inode #1: root directory.
    let root_offset_in_block: u16 = 0;
    write_inode_header(
        &mut inode_payload,
        DIR_INODE_KIND,
        0o755,
        0,
        0,
        0,
        ROOT_INODE_NUMBER,
    );
    // Body: start_block=0, nlink=2+files, size=size_of_dir_payload+3,
    // offset=0, parent_inode=1.
    // We patch `dir_size` after building the directory table.
    let root_body_offset = inode_payload.len();
    inode_payload.extend_from_slice(&0u32.to_le_bytes()); // start_block
    inode_payload.extend_from_slice(&((2 + files.len()) as u32).to_le_bytes()); // nlink
    inode_payload.extend_from_slice(&0u16.to_le_bytes()); // size_raw, patched later
    inode_payload.extend_from_slice(&0u16.to_le_bytes()); // offset
    inode_payload.extend_from_slice(&ROOT_INODE_NUMBER.to_le_bytes()); // parent

    // Files.
    let mut file_offsets: Vec<u16> = Vec::with_capacity(files.len());
    for (i, e) in encoded.iter().enumerate() {
        file_offsets.push(inode_payload.len() as u16);
        let inode_number = ROOT_INODE_NUMBER + 1 + i as u32;
        write_inode_header(
            &mut inode_payload,
            FILE_INODE_KIND,
            0o644,
            0,
            0,
            0,
            inode_number,
        );
        inode_payload.extend_from_slice(&(e.blocks_start as u32).to_le_bytes()); // blocks_start
        inode_payload.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // frag_idx
        inode_payload.extend_from_slice(&0u32.to_le_bytes()); // frag_offset
        inode_payload.extend_from_slice(&e.file_size.to_le_bytes());
        for &bs in &e.block_sizes {
            inode_payload.extend_from_slice(&bs.to_le_bytes());
        }
    }
    // Wrap inode_payload in one uncompressed metadata block.
    write_uncompressed_metadata_block(&mut img, &inode_payload);

    // ── 3. Directory table ──
    //
    // Single header: count = files.len()-1, start_block=0,
    // inode_number_base=2 (root has number 1; files start at 2).
    //
    // Per-entry: offset_in_block, inode_number_delta (signed),
    // inode_kind=2, name_size = name.len()-1, name bytes.

    let directory_table_start = img.len() as u64;
    let mut dir_payload: Vec<u8> = Vec::new();
    if !files.is_empty() {
        let count_minus_1 = (files.len() as u32) - 1;
        dir_payload.extend_from_slice(&count_minus_1.to_le_bytes());
        dir_payload.extend_from_slice(&0u32.to_le_bytes()); // start_block (relative to inode_table_start)
        dir_payload.extend_from_slice(&(ROOT_INODE_NUMBER + 1).to_le_bytes()); // inode_number_base
        for (i, f) in files.iter().enumerate() {
            let off = file_offsets[i];
            dir_payload.extend_from_slice(&off.to_le_bytes());
            dir_payload.extend_from_slice(&(i as i16).to_le_bytes()); // delta = i (so file 0 → number=base, file 1 → base+1, …)
            dir_payload.extend_from_slice(&FILE_INODE_KIND.to_le_bytes());
            let name_bytes = f.name.as_bytes();
            assert!(!name_bytes.is_empty());
            dir_payload.extend_from_slice(&((name_bytes.len() - 1) as u16).to_le_bytes());
            dir_payload.extend_from_slice(name_bytes);
        }
    }
    write_uncompressed_metadata_block(&mut img, &dir_payload);

    // Patch root inode's `size_raw` (within the body at offset 8..10:
    // start_block(4) + nlink(4) → size_raw at +8). +3 per the squashfs
    // 4.0 spec quirk on Basic Directory size.
    {
        // The inode metadata block starts at inode_table_start; payload
        // begins at inode_table_start + 2.
        let payload_in_image = inode_table_start as usize + 2;
        let body_offset = payload_in_image + root_body_offset;
        let size_raw = (dir_payload.len() as u32 + 3) as u16;
        img[body_offset + 8..body_offset + 10].copy_from_slice(&size_raw.to_le_bytes());
    }

    // ── 4. Fragment table top-level (count = 0 → no top entries) ──
    let fragment_table_start = u64::MAX; // sentinel "no fragment table"

    // ── 5. ID table — one entry: u32(0) ──
    // Wrap in an uncompressed metadata block; top-level array = 1 u64
    // pointing at the metadata block offset.
    let id_meta_block_offset = img.len() as u64;
    write_uncompressed_metadata_block(&mut img, &0u32.to_le_bytes());
    let id_table_start = img.len() as u64;
    img.extend_from_slice(&id_meta_block_offset.to_le_bytes());

    // ── 6. Patch superblock fields ──
    let bytes_used = img.len() as u64;
    let mut sb = [0u8; 96];
    sb[0..4].copy_from_slice(&0x73717368u32.to_le_bytes()); // magic
    sb[4..8].copy_from_slice(&((files.len() as u32) + 1).to_le_bytes()); // inode_count
    sb[8..12].copy_from_slice(&0u32.to_le_bytes()); // mod_time
    sb[12..16].copy_from_slice(&BLOCK_SIZE.to_le_bytes()); // block_size
    sb[16..20].copy_from_slice(&0u32.to_le_bytes()); // fragment_count
    sb[20..22].copy_from_slice(&(compression as u16).to_le_bytes());
    sb[22..24].copy_from_slice(&BLOCK_LOG.to_le_bytes());
    sb[24..26].copy_from_slice(&0x0210u16.to_le_bytes()); // flags: NO_FRAGMENTS | NO_XATTR
    sb[26..28].copy_from_slice(&1u16.to_le_bytes()); // id_count
    sb[28..30].copy_from_slice(&4u16.to_le_bytes()); // version_major
    sb[30..32].copy_from_slice(&0u16.to_le_bytes()); // version_minor

    // Root inode lives at byte offset 0 of the inode metadata block, so
    // its metadata-reference high 48 bits are zero and the low 16 bits
    // are simply `root_offset_in_block`.
    let root_inode_ref = root_offset_in_block as u64;
    sb[32..40].copy_from_slice(&root_inode_ref.to_le_bytes());
    sb[40..48].copy_from_slice(&bytes_used.to_le_bytes());
    sb[48..56].copy_from_slice(&id_table_start.to_le_bytes());
    sb[56..64].copy_from_slice(&u64::MAX.to_le_bytes()); // xattr_table_start (none)
    sb[64..72].copy_from_slice(&inode_table_start.to_le_bytes());
    sb[72..80].copy_from_slice(&directory_table_start.to_le_bytes());
    sb[80..88].copy_from_slice(&fragment_table_start.to_le_bytes());
    sb[88..96].copy_from_slice(&u64::MAX.to_le_bytes()); // lookup_table_start (no exports)
    img[..96].copy_from_slice(&sb);

    img
}

#[allow(clippy::too_many_arguments)]
fn write_inode_header(
    out: &mut Vec<u8>,
    kind: u16,
    permissions: u16,
    uid_idx: u16,
    gid_idx: u16,
    mtime: u32,
    inode_number: u32,
) {
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&permissions.to_le_bytes());
    out.extend_from_slice(&uid_idx.to_le_bytes());
    out.extend_from_slice(&gid_idx.to_le_bytes());
    out.extend_from_slice(&mtime.to_le_bytes());
    out.extend_from_slice(&inode_number.to_le_bytes());
}

fn write_uncompressed_metadata_block(out: &mut Vec<u8>, payload: &[u8]) {
    let hdr = (payload.len() as u16) | 0x8000;
    out.extend_from_slice(&hdr.to_le_bytes());
    out.extend_from_slice(payload);
}

/// Encode a single data block (≤ BLOCK_SIZE bytes) using the chosen
/// compressor's "stored uncompressed" framing path. Returns the
/// on-disk bytes (with the high bit of the size set in the inode's
/// block-sizes table to mark uncompressed; the caller of
/// [`build_image`] handles that flag).
fn encode_block(compression: Compression, payload: &[u8]) -> Vec<u8> {
    match compression {
        Compression::Zstd => encode_zstd_raw_frame(payload),
        Compression::Xz => encode_xz_uncompressed_block(payload),
        Compression::Gzip => encode_zlib_stored_block(payload),
        Compression::Lz4 => encode_lz4_literal_block(payload),
    }
}

fn encode_zstd_raw_frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&0xFD2F_B528u32.to_le_bytes()); // magic
                                                          // FHD: single_segment=1, fcs_flag=2 → 4-byte FCS. Reserved bit must
                                                          // be zero. We'll use single_segment so no window descriptor.
    out.push(0b1010_0000);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    // Block header: last_block=1, block_type=0 (raw), block_size.
    // Block header bits: bit 0 = last_block, bits 1..3 = block_type (0
    // for Raw), bits 3..24 = block_size.
    let bh = 1u32 | ((payload.len() as u32) << 3);
    out.extend_from_slice(&bh.to_le_bytes()[..3]);
    out.extend_from_slice(payload);
    out
}

fn encode_xz_uncompressed_block(payload: &[u8]) -> Vec<u8> {
    // Use the same encoder shape as the unit test's helper.
    use smallaios_fs::squashfs::decompress;
    let _ = decompress::DecompressError::Truncated; // import-side sanity
                                                    //
                                                    // Inline a minimal valid xz stream (mirrors the unit-test helper).
    assert!(!payload.is_empty() && payload.len() <= 65536);
    const XZ_MAGIC: [u8; 6] = [0xFD, b'7', b'z', b'X', b'Z', 0x00];
    const XZ_FOOTER_MAGIC: [u8; 2] = [b'Y', b'Z'];
    let mut out = Vec::new();
    out.extend_from_slice(&XZ_MAGIC);
    let flags = [0u8, 0u8];
    out.extend_from_slice(&flags);
    out.extend_from_slice(&crc32_xz(&flags).to_le_bytes());

    let mut bh: Vec<u8> = Vec::new();
    bh.extend_from_slice(&[2u8, 0u8, 0x21u8, 0x01u8, 0u8, 0u8, 0u8, 0u8]);
    bh.extend_from_slice(&crc32_xz(&bh).to_le_bytes());
    let block_header_len = bh.len();
    out.extend_from_slice(&bh);

    let size_minus_1 = (payload.len() - 1) as u16;
    let chunk_start = out.len();
    out.push(0x01);
    out.push((size_minus_1 >> 8) as u8);
    out.push(size_minus_1 as u8);
    out.extend_from_slice(payload);
    out.push(0x00);
    let lzma2_len = out.len() - chunk_start;
    while out.len() % 4 != 0 {
        out.push(0u8);
    }

    let index_start = out.len();
    out.push(0x00);
    write_xz_multibyte(&mut out, 1);
    write_xz_multibyte(&mut out, (block_header_len + lzma2_len) as u64);
    write_xz_multibyte(&mut out, payload.len() as u64);
    while (out.len() - index_start) % 4 != 0 {
        out.push(0u8);
    }
    let index_end = out.len();
    let index_crc = crc32_xz(&out[index_start..index_end]).to_le_bytes();
    out.extend_from_slice(&index_crc);

    let index_size_bytes = (index_end + 4 - index_start) as u32;
    let backward_size = index_size_bytes / 4 - 1;
    let mut footer_body: Vec<u8> = Vec::new();
    footer_body.extend_from_slice(&backward_size.to_le_bytes());
    footer_body.extend_from_slice(&flags);
    let footer_crc = crc32_xz(&footer_body).to_le_bytes();
    out.extend_from_slice(&footer_crc);
    out.extend_from_slice(&footer_body);
    out.extend_from_slice(&XZ_FOOTER_MAGIC);
    out
}

fn write_xz_multibyte(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut b = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            return;
        }
    }
}

fn crc32_xz(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (POLY & mask);
        }
    }
    !crc
}

fn encode_zlib_stored_block(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // CMF = 0x78 (DEFLATE, 32 KiB window). FLG so that
    // (CMF*256 + FLG) % 31 == 0 and FDICT = 0. 0x78 0x9C is canonical.
    out.extend_from_slice(&[0x78, 0x9C]);
    // We can have multiple stored blocks if payload > 65535. For
    // typical test sizes (≤ 4 KiB), one block suffices.
    let mut written = 0;
    while written < payload.len() {
        let chunk_size = core::cmp::min(payload.len() - written, 65535);
        let bfinal = if written + chunk_size == payload.len() {
            1u8
        } else {
            0u8
        };
        // BFINAL + BTYPE=00 (lsb-first: bit 0 = bfinal, bits 1..2 = 0).
        out.push(bfinal);
        out.extend_from_slice(&(chunk_size as u16).to_le_bytes());
        out.extend_from_slice(&(!(chunk_size as u16)).to_le_bytes());
        out.extend_from_slice(&payload[written..written + chunk_size]);
        written += chunk_size;
    }
    out.extend_from_slice(&adler32(payload).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn encode_lz4_literal_block(payload: &[u8]) -> Vec<u8> {
    // Single all-literals sequence: token's high nibble = literal len,
    // extended via 0xFF runs if >= 15. No match section.
    let mut out = Vec::new();
    let lit_len = payload.len();
    if lit_len < 15 {
        out.push((lit_len as u8) << 4);
    } else {
        out.push(0xF0);
        let mut rem = lit_len - 15;
        while rem >= 0xFF {
            out.push(0xFF);
            rem -= 0xFF;
        }
        out.push(rem as u8);
    }
    out.extend_from_slice(payload);
    out
}
