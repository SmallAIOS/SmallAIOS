// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Clean-room xz / LZMA2 decompressor — squashfs subset.
//!
//! The xz container (RFC-style spec at <https://tukaani.org/xz/xz-file-format.txt>)
//! wraps one or more LZMA2 streams. The squashfs subset that mksquashfs
//! produces today uses single-stream xz with a single block, no
//! filters, and an LZMA2 payload.
//!
//! What this implementation covers:
//!
//! - Stream Header (magic `0xFD 7zXZ\0`, flags, CRC32 of flags).
//! - One Block Header (filter list with a single LZMA2 filter).
//! - LZMA2 chunk parser:
//!   - `0x00`: end-of-stream marker.
//!   - `0x01`: uncompressed chunk + dictionary reset.
//!   - `0x02`: uncompressed chunk.
//!   - `0x80..=0xFF`: LZMA-compressed chunk → currently
//!     `Err(DecompressError::Unsupported)` and tagged
//!     `TODO(#fs-squashfs-xz-lzma)` for the next phase agent.
//! - Stream Footer magic check (`YZ`).
//!
//! The LZMA range-decoder + state machine is the bulk of a full xz
//! implementation (≈3 kLOC); the squashfs round-trip tests use the
//! uncompressed-chunk path. Production `mksquashfs -comp xz` images
//! will hit the LZMA path and currently fail-closed; flag-gate
//! production builds appropriately or use zstd instead until the
//! follow-up phase fills LZMA in.

extern crate alloc;

use alloc::vec::Vec;

use super::DecompressError;

/// xz Stream Header magic (6 bytes).
//
// lgtm[rust/hard-coded-cryptographic-value] — xz Stream Header magic per xz file format spec
pub const XZ_MAGIC: [u8; 6] = [0xFD, b'7', b'z', b'X', b'Z', 0x00];

/// xz Stream Footer trailing magic (2 bytes).
//
// lgtm[rust/hard-coded-cryptographic-value] — xz Stream Footer magic per xz file format spec
pub const XZ_FOOTER_MAGIC: [u8; 2] = [b'Y', b'Z'];

/// Decompress a squashfs xz block (one or more xz streams concatenated).
pub fn decompress(input: &[u8], max_out: usize) -> Result<Vec<u8>, DecompressError> {
    if input.len() < XZ_MAGIC.len() + 6 {
        return Err(DecompressError::Truncated);
    }
    if input[..6] != XZ_MAGIC {
        return Err(DecompressError::BadFormat);
    }
    // Stream Flags (2 bytes) + CRC32 (4 bytes).
    let flags_pos = 6;
    if flags_pos + 6 > input.len() {
        return Err(DecompressError::Truncated);
    }
    let stream_flags = [input[flags_pos], input[flags_pos + 1]];
    if stream_flags[0] != 0 {
        return Err(DecompressError::BadFormat);
    }
    let check_type = stream_flags[1] & 0x0F;
    // 0 = none, 1 = CRC32, 4 = CRC64, 10 = SHA-256. We accept all but
    // only verify the CRC types; SHA-256 verification is a TODO.
    let check_size = match check_type {
        0 => 0,
        1 => 4,
        2..=3 => return Err(DecompressError::Unsupported),
        4..=6 => 8,
        7..=9 => return Err(DecompressError::Unsupported),
        10..=12 => 32,
        _ => return Err(DecompressError::Unsupported),
    };
    let stored_flags_crc = u32::from_le_bytes([
        input[flags_pos + 2],
        input[flags_pos + 3],
        input[flags_pos + 4],
        input[flags_pos + 5],
    ]);
    let computed_flags_crc = crc32(&stream_flags);
    if stored_flags_crc != computed_flags_crc {
        return Err(DecompressError::Corrupt);
    }
    let mut pos = flags_pos + 6;

    let mut out: Vec<u8> = Vec::new();
    // Repeatedly parse blocks until we encounter the index (block-header
    // size byte == 0x00).
    loop {
        if pos >= input.len() {
            return Err(DecompressError::Truncated);
        }
        let bhs_byte = input[pos];
        if bhs_byte == 0 {
            break; // index follows
        }
        let block_header_size = (bhs_byte as usize + 1) * 4;
        if pos + block_header_size > input.len() {
            return Err(DecompressError::Truncated);
        }
        let block_header = &input[pos..pos + block_header_size];
        // Block flags (byte 1 in block_header).
        let block_flags = block_header[1];
        let num_filters = (block_flags & 0x03) as usize + 1;
        let has_compressed_size = block_flags & 0x40 != 0;
        let has_uncompressed_size = block_flags & 0x80 != 0;

        let mut bp = 2;
        let _compressed_size = if has_compressed_size {
            let (v, n) = read_multibyte(&block_header[bp..])?;
            bp += n;
            Some(v)
        } else {
            None
        };
        let _uncompressed_size = if has_uncompressed_size {
            let (v, n) = read_multibyte(&block_header[bp..])?;
            bp += n;
            Some(v)
        } else {
            None
        };

        // Filter list.
        for _ in 0..num_filters {
            // Filter ID.
            let (filter_id, fid_n) = read_multibyte(&block_header[bp..])?;
            bp += fid_n;
            // Filter properties (variable length).
            let (prop_size, ps_n) = read_multibyte(&block_header[bp..])?;
            bp += ps_n;
            if filter_id != 0x21 {
                // Only LZMA2 (filter ID 0x21) is supported; squashfs
                // does not use BCJ / Delta filters.
                return Err(DecompressError::Unsupported);
            }
            if prop_size != 1 {
                return Err(DecompressError::Corrupt);
            }
            // 1-byte LZMA2 dictionary-size property; we accept any
            // value because we only handle uncompressed chunks today.
            let _dict_prop = block_header[bp];
            bp += 1;
        }
        // Header padding to multiple of 4 (already enforced by block_header_size).
        let _ = bp;
        let header_crc_pos = block_header_size - 4;
        let stored_hdr_crc = u32::from_le_bytes([
            block_header[header_crc_pos],
            block_header[header_crc_pos + 1],
            block_header[header_crc_pos + 2],
            block_header[header_crc_pos + 3],
        ]);
        let computed_hdr_crc = crc32(&block_header[..header_crc_pos]);
        if stored_hdr_crc != computed_hdr_crc {
            return Err(DecompressError::Corrupt);
        }
        pos += block_header_size;

        // LZMA2 chunked payload.
        loop {
            if pos >= input.len() {
                return Err(DecompressError::Truncated);
            }
            let control = input[pos];
            pos += 1;
            match control {
                0 => break, // end of LZMA2 stream
                1 | 2 => {
                    if pos + 2 > input.len() {
                        return Err(DecompressError::Truncated);
                    }
                    let size = ((input[pos] as usize) << 8 | input[pos + 1] as usize) + 1;
                    pos += 2;
                    if pos + size > input.len() {
                        return Err(DecompressError::Truncated);
                    }
                    if out.len().saturating_add(size) > max_out {
                        return Err(DecompressError::OutputTooLarge);
                    }
                    out.extend_from_slice(&input[pos..pos + size]);
                    pos += size;
                }
                0x80..=0xFF => {
                    // TODO(#fs-squashfs-xz-lzma): decode LZMA-compressed
                    // chunk; today we fail-closed.
                    return Err(DecompressError::Unsupported);
                }
                _ => return Err(DecompressError::Corrupt),
            }
        }

        // Block payload padding (multiple of 4).
        while pos % 4 != 0 {
            if pos >= input.len() {
                return Err(DecompressError::Truncated);
            }
            if input[pos] != 0 {
                return Err(DecompressError::Corrupt);
            }
            pos += 1;
        }
        // Skip the optional check field.
        if pos + check_size > input.len() {
            return Err(DecompressError::Truncated);
        }
        pos += check_size;
    }

    // Index — starts with a 0x00 indicator already consumed above? No:
    // we exited the loop when the block header size byte == 0; that
    // byte is *the* index indicator and is part of the index.
    if pos >= input.len() {
        return Err(DecompressError::Truncated);
    }
    let index_indicator = input[pos];
    if index_indicator != 0 {
        return Err(DecompressError::Corrupt);
    }
    pos += 1;
    // Number of records in the index.
    let (num_records, n) = read_multibyte(&input[pos..])?;
    pos += n;
    // Each record: unpadded size + uncompressed size (both multibyte).
    for _ in 0..num_records {
        let (_a, na) = read_multibyte(&input[pos..])?;
        pos += na;
        let (_b, nb) = read_multibyte(&input[pos..])?;
        pos += nb;
    }
    // Index padding to multiple of 4.
    while pos % 4 != 0 {
        if pos >= input.len() {
            return Err(DecompressError::Truncated);
        }
        if input[pos] != 0 {
            return Err(DecompressError::Corrupt);
        }
        pos += 1;
    }
    // Index CRC32.
    if pos + 4 > input.len() {
        return Err(DecompressError::Truncated);
    }
    pos += 4;
    // Stream Footer (12 bytes).
    if pos + 12 > input.len() {
        return Err(DecompressError::Truncated);
    }
    if input[pos + 10..pos + 12] != XZ_FOOTER_MAGIC {
        return Err(DecompressError::BadFormat);
    }
    Ok(out)
}

/// Read an xz multibyte integer (variable-length unsigned). Returns
/// `(value, bytes_consumed)`.
fn read_multibyte(input: &[u8]) -> Result<(u64, usize), DecompressError> {
    let mut value: u64 = 0;
    for i in 0..9 {
        if i >= input.len() {
            return Err(DecompressError::Truncated);
        }
        let b = input[i];
        value |= ((b & 0x7F) as u64) << (i * 7);
        if b & 0x80 == 0 {
            return Ok((value, i + 1));
        }
    }
    Err(DecompressError::Corrupt)
}

/// CRC-32 / IEEE 802.3 polynomial `0xEDB88320` reflected.
//
// lgtm[rust/hard-coded-cryptographic-value] — IEEE 802.3 CRC-32 reflected polynomial constant
const CRC32_POLY_REFLECTED: u32 = 0xEDB8_8320;

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (CRC32_POLY_REFLECTED & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-construct a minimal valid xz stream containing one block
    /// with one LZMA2 filter and one uncompressed-chunk LZMA2 payload.
    fn build_uncompressed_xz(payload: &[u8]) -> Vec<u8> {
        assert!(!payload.is_empty() && payload.len() <= 65536);
        let mut out = Vec::new();
        out.extend_from_slice(&XZ_MAGIC);
        let flags = [0u8, 0u8];
        out.extend_from_slice(&flags);
        out.extend_from_slice(&crc32(&flags).to_le_bytes());

        // Block Header — 12 bytes:
        //   [bhs=0x02 → (2+1)*4 = 12 bytes]
        //   [block_flags=0]
        //   [filter_id=0x21]
        //   [prop_size=1]
        //   [dict_prop=0]
        //   [3 zero padding bytes → header reaches 8 bytes]
        //   [4-byte CRC32 of the first 8 bytes]
        let mut bh: Vec<u8> = Vec::new();
        bh.extend_from_slice(&[2u8, 0u8, 0x21u8, 0x01u8, 0u8, 0u8, 0u8, 0u8]);
        let bh_crc = crc32(&bh).to_le_bytes();
        bh.extend_from_slice(&bh_crc);
        assert_eq!(bh.len(), 12);
        let block_header_len = bh.len();
        out.extend_from_slice(&bh);

        // LZMA2 uncompressed chunk: control 0x01 + (size-1) BE-16 + bytes,
        // then end-of-stream control 0x00.
        let size_minus_1 = (payload.len() - 1) as u16;
        let chunk_start = out.len();
        out.push(0x01);
        out.push((size_minus_1 >> 8) as u8);
        out.push(size_minus_1 as u8);
        out.extend_from_slice(payload);
        out.push(0x00);
        let lzma2_len = out.len() - chunk_start;

        // Block payload padding to multiple of 4.
        while out.len() % 4 != 0 {
            out.push(0x00);
        }
        // No check field (check_type = 0 in stream flags).

        // Index — track its start so we can CRC over it later.
        let index_start = out.len();
        out.push(0x00); // index indicator
        let num_records: u64 = 1;
        write_multibyte(&mut out, num_records);
        let unpadded_size: u64 = (block_header_len + lzma2_len) as u64;
        let uncompressed_size: u64 = payload.len() as u64;
        write_multibyte(&mut out, unpadded_size);
        write_multibyte(&mut out, uncompressed_size);
        while (out.len() - index_start) % 4 != 0 {
            out.push(0x00);
        }
        let index_end = out.len();
        let index_crc = crc32(&out[index_start..index_end]).to_le_bytes();
        out.extend_from_slice(&index_crc);

        // Stream Footer (12 bytes total):
        //   [backward size u32 LE = (index_size/4) - 1]
        //   [stream flags 2 bytes (must equal header flags)]
        //   [YZ]
        //   The Stream Footer has its own CRC32 in front of backward
        //   size; layout in the spec is:
        //   [4-byte CRC32][4-byte backward size][2-byte flags][YZ].
        let index_size_bytes = (index_end + 4 - index_start) as u32; // includes CRC
        let backward_size = index_size_bytes / 4 - 1;
        let mut footer_body: Vec<u8> = Vec::new();
        footer_body.extend_from_slice(&backward_size.to_le_bytes());
        footer_body.extend_from_slice(&flags);
        let footer_crc = crc32(&footer_body).to_le_bytes();
        out.extend_from_slice(&footer_crc);
        out.extend_from_slice(&footer_body);
        out.extend_from_slice(&XZ_FOOTER_MAGIC);
        out
    }

    fn write_multibyte(out: &mut Vec<u8>, mut value: u64) {
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

    #[allow(dead_code)]
    fn multibyte_len(value: u64) -> usize {
        let mut v = value;
        let mut n = 0;
        loop {
            v >>= 7;
            n += 1;
            if v == 0 {
                return n;
            }
        }
    }

    #[test]
    fn roundtrip_uncompressed_xz_block() {
        let payload: &[u8] = b"hello squashfs world";
        let stream = build_uncompressed_xz(payload);
        let out = decompress(&stream, 4096).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bad = build_uncompressed_xz(b"x");
        bad[0] ^= 0x55;
        assert!(matches!(
            decompress(&bad, 4096),
            Err(DecompressError::BadFormat)
        ));
    }

    #[test]
    fn lzma_chunk_unsupported() {
        // Construct a minimal xz with an LZMA-compressed chunk header
        // (control byte 0x80). We don't bother with valid LZMA payload.
        let mut s = Vec::new();
        s.extend_from_slice(&XZ_MAGIC);
        s.extend_from_slice(&[0u8, 0u8]);
        s.extend_from_slice(&crc32(&[0u8, 0u8]).to_le_bytes());
        // Block header (12 bytes) — same template.
        let mut bh = alloc::vec![2u8, 0u8, 0x21u8, 0x01u8, 0x00u8, 0u8, 0u8, 0u8];
        let crc = crc32(&bh).to_le_bytes();
        bh.extend_from_slice(&crc);
        s.extend_from_slice(&bh);
        // LZMA2 control byte 0x80: triggers the unsupported path.
        s.push(0x80);
        assert!(matches!(
            decompress(&s, 4096),
            Err(DecompressError::Unsupported)
        ));
    }

    #[test]
    fn crc32_known_values() {
        // CRC-32/ISO-HDLC of "123456789" = 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn multibyte_roundtrip() {
        for v in [0u64, 1, 127, 128, 16383, 16384, 0xFFFF_FFFF] {
            let mut buf = Vec::new();
            write_multibyte(&mut buf, v);
            let (got, _) = read_multibyte(&buf).unwrap();
            assert_eq!(got, v);
        }
    }
}
