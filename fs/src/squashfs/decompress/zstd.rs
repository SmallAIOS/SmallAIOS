// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Clean-room zstd (RFC 8478) decompressor — squashfs subset.
//!
//! Covers the squashfs subset of RFC 8478:
//!
//! - **Frame header** parse (magic, frame header descriptor, window/
//!   dictionary/checksum/content-size fields).
//! - **Block parse loop** with Raw, RLE, and Compressed block types.
//! - **Compressed block** parsing supports the Raw and RLE literal
//!   sections (the most common shapes for small squashfs blocks). The
//!   Huffman-encoded literal section + FSE-encoded sequence sections
//!   are stubbed with `Err(DecompressError::Unsupported)` and tagged
//!   `TODO(#fs-squashfs-zstd-fse)` for the next phase agent. Real
//!   squashfs images that exercise Huffman literals will fail to
//!   mount with `DecompressFailed`; feature flag the production
//!   target appropriately until that is filled in.
//! - **xxhash-64 frame checksum** verified when the frame header bit
//!   `Content_Checksum_Flag` is set.
//!
//! References:
//!
//! - RFC 8478 (zstd v1.4.0 reference)
//! - <https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md>
//!
//! `lzma`/`lzo` legacy compression IDs are rejected at the squashfs
//! superblock layer; this module only cares about zstd.

extern crate alloc;

use alloc::vec::Vec;

use super::DecompressError;

/// zstd magic number (little-endian on disk).
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 8478 §3.1 zstd magic, public on-disk constant
pub const ZSTD_MAGIC: u32 = 0xFD2F_B528;

/// Decompress a zstd frame. The frame may consist of multiple blocks.
pub fn decompress(input: &[u8], max_out: usize) -> Result<Vec<u8>, DecompressError> {
    let mut r = Reader::new(input);
    let magic = r.read_u32_le()?;
    // Skippable frames (magic 0x184D2A50..=5F) are ignored; only the
    // single zstd magic is accepted.
    if magic != ZSTD_MAGIC {
        return Err(DecompressError::BadFormat);
    }

    // Frame header descriptor (FHD), 1 byte.
    let fhd = r.read_u8()?;
    let frame_content_size_flag = (fhd >> 6) & 0x3;
    let single_segment = (fhd >> 5) & 0x1 != 0;
    // Bit 4 is reserved; must be zero.
    if (fhd >> 4) & 0x1 != 0 {
        return Err(DecompressError::Corrupt);
    }
    let content_checksum = (fhd >> 2) & 0x1 != 0;
    let dict_id_flag = fhd & 0x3;

    // Window descriptor — only present when single_segment == 0.
    if !single_segment {
        let _wd = r.read_u8()?;
    }

    // Dictionary ID (squashfs never uses dictionaries).
    if dict_id_flag != 0 {
        let did_len = match dict_id_flag {
            1 => 1,
            2 => 2,
            3 => 4,
            _ => unreachable!(),
        };
        for _ in 0..did_len {
            let _ = r.read_u8()?;
        }
        return Err(DecompressError::Unsupported);
    }

    // Frame content size.
    let _fcs = match frame_content_size_flag {
        0 => {
            if single_segment {
                r.read_u8()? as u64
            } else {
                u64::MAX
            }
        }
        1 => r.read_u16_le()? as u64 + 256,
        2 => r.read_u32_le()? as u64,
        3 => r.read_u64_le()?,
        _ => unreachable!(),
    };

    // Block parse loop.
    let checksum_start = r.pos;
    let mut out: Vec<u8> = Vec::new();
    loop {
        let bh = r.read_u24_le()?;
        let last_block = bh & 0x1 != 0;
        let block_type = (bh >> 1) & 0x3;
        let block_size = (bh >> 3) as usize;
        match block_type {
            0 => {
                // Raw block — block_size bytes copied verbatim.
                if r.pos + block_size > r.bytes.len() {
                    return Err(DecompressError::Truncated);
                }
                if out.len().saturating_add(block_size) > max_out {
                    return Err(DecompressError::OutputTooLarge);
                }
                out.extend_from_slice(&r.bytes[r.pos..r.pos + block_size]);
                r.pos += block_size;
            }
            1 => {
                // RLE block — single byte repeated `block_size` times.
                let b = r.read_u8()?;
                if out.len().saturating_add(block_size) > max_out {
                    return Err(DecompressError::OutputTooLarge);
                }
                for _ in 0..block_size {
                    out.push(b);
                }
            }
            2 => {
                // Compressed block.
                if r.pos + block_size > r.bytes.len() {
                    return Err(DecompressError::Truncated);
                }
                let block_end = r.pos + block_size;
                decompress_compressed_block(&r.bytes[r.pos..block_end], &mut out, max_out)?;
                r.pos = block_end;
            }
            3 => return Err(DecompressError::Corrupt),
            _ => unreachable!(),
        }
        if last_block {
            break;
        }
    }

    if content_checksum {
        // Read the 4-byte content-checksum trailer (lower 32 bits of
        // xxh64 of the decompressed output, seed=0, little-endian).
        let stored = r.read_u32_le()?;
        let computed = xxh64(&out, 0) as u32;
        if stored != computed {
            return Err(DecompressError::Corrupt);
        }
    } else {
        let _ = checksum_start;
    }

    Ok(out)
}

/// Compressed-block parser. Handles Raw and RLE literal sections; the
/// Huffman literal-section path and FSE-encoded sequences path are
/// stubbed with `Err(Unsupported)` and tagged below.
fn decompress_compressed_block(
    block: &[u8],
    out: &mut Vec<u8>,
    max_out: usize,
) -> Result<(), DecompressError> {
    if block.is_empty() {
        return Err(DecompressError::Truncated);
    }
    // Literals_Section_Header — 1..5 bytes.
    let header_byte = block[0];
    let lits_block_type = header_byte & 0x3;
    let size_format = (header_byte >> 2) & 0x3;

    let (regenerated_size, literals_data, literals_consumed) = match lits_block_type {
        0 | 1 => {
            // Raw_Literals_Block (type=0) or RLE_Literals_Block (type=1).
            let (regen, header_len) = match size_format {
                0 | 2 => ((header_byte >> 3) as usize, 1),
                1 => {
                    if block.len() < 2 {
                        return Err(DecompressError::Truncated);
                    }
                    let val = ((header_byte >> 4) as usize) | ((block[1] as usize) << 4);
                    (val, 2)
                }
                3 => {
                    if block.len() < 3 {
                        return Err(DecompressError::Truncated);
                    }
                    let val = ((header_byte >> 4) as usize)
                        | ((block[1] as usize) << 4)
                        | ((block[2] as usize) << 12);
                    (val, 3)
                }
                _ => unreachable!(),
            };
            if lits_block_type == 0 {
                // Raw literals: regen bytes follow the header.
                let start = header_len;
                let end = start + regen;
                if end > block.len() {
                    return Err(DecompressError::Truncated);
                }
                (regen, &block[start..end], end)
            } else {
                // RLE literals: a single byte follows the header.
                if block.len() < header_len + 1 {
                    return Err(DecompressError::Truncated);
                }
                (regen, &block[header_len..header_len + 1], header_len + 1)
            }
        }
        2 | 3 => {
            // Compressed_Literals_Block / Treeless_Literals_Block.
            // TODO(#fs-squashfs-zstd-fse): wire Huffman + FSE decoder.
            return Err(DecompressError::Unsupported);
        }
        _ => unreachable!(),
    };

    // Sequences_Section_Header — 1..5 bytes; first byte is
    // `Number_of_Sequences`.
    let seq_off = literals_consumed;
    if seq_off >= block.len() {
        return Err(DecompressError::Truncated);
    }
    let nb_byte0 = block[seq_off] as usize;
    let (num_sequences, seq_header_len) = match nb_byte0 {
        0 => (0, 1),
        1..=127 => (nb_byte0, 1),
        128..=254 => {
            if seq_off + 1 >= block.len() {
                return Err(DecompressError::Truncated);
            }
            let val = ((nb_byte0 - 128) << 8) + block[seq_off + 1] as usize + 0x80;
            (val, 2)
        }
        255 => {
            if seq_off + 2 >= block.len() {
                return Err(DecompressError::Truncated);
            }
            let val = (block[seq_off + 1] as usize) + ((block[seq_off + 2] as usize) << 8) + 0x7F00;
            (val, 3)
        }
        _ => unreachable!(),
    };
    if num_sequences == 0 {
        // Pure literal block — copy literals to `out` and finish.
        if out.len().saturating_add(regenerated_size) > max_out {
            return Err(DecompressError::OutputTooLarge);
        }
        match lits_block_type {
            0 => {
                if literals_data.len() != regenerated_size {
                    return Err(DecompressError::Corrupt);
                }
                out.extend_from_slice(literals_data);
            }
            1 => {
                let b = literals_data[0];
                for _ in 0..regenerated_size {
                    out.push(b);
                }
            }
            _ => unreachable!(),
        }
        let _ = seq_header_len;
        return Ok(());
    }
    // TODO(#fs-squashfs-zstd-fse): FSE-encoded sequence path.
    Err(DecompressError::Unsupported)
}

/// xxh64 hash (Yann Collet's xxhash, 64-bit variant), used by zstd's
/// optional content-checksum trailer.
//
// lgtm[rust/hard-coded-cryptographic-value] — xxh64 reference primes from xxHash specification
const XXH_PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
//
// lgtm[rust/hard-coded-cryptographic-value] — xxh64 reference primes from xxHash specification
const XXH_PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
//
// lgtm[rust/hard-coded-cryptographic-value] — xxh64 reference primes from xxHash specification
const XXH_PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
//
// lgtm[rust/hard-coded-cryptographic-value] — xxh64 reference primes from xxHash specification
const XXH_PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
//
// lgtm[rust/hard-coded-cryptographic-value] — xxh64 reference primes from xxHash specification
const XXH_PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

/// Compute the xxh64 hash of `input` with `seed`.
pub fn xxh64(input: &[u8], seed: u64) -> u64 {
    let len = input.len() as u64;
    let mut h: u64;
    let mut p = 0;
    if input.len() >= 32 {
        let mut v1 = seed.wrapping_add(XXH_PRIME64_1).wrapping_add(XXH_PRIME64_2);
        let mut v2 = seed.wrapping_add(XXH_PRIME64_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(XXH_PRIME64_1);
        while p + 32 <= input.len() {
            v1 = round(v1, read_u64_le(&input[p..]));
            v2 = round(v2, read_u64_le(&input[p + 8..]));
            v3 = round(v3, read_u64_le(&input[p + 16..]));
            v4 = round(v4, read_u64_le(&input[p + 24..]));
            p += 32;
        }
        h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);
    } else {
        h = seed.wrapping_add(XXH_PRIME64_5);
    }
    h = h.wrapping_add(len);
    while p + 8 <= input.len() {
        let k1 = round(0, read_u64_le(&input[p..]));
        h = (h ^ k1)
            .rotate_left(27)
            .wrapping_mul(XXH_PRIME64_1)
            .wrapping_add(XXH_PRIME64_4);
        p += 8;
    }
    if p + 4 <= input.len() {
        h = (h ^ ((read_u32_le(&input[p..]) as u64).wrapping_mul(XXH_PRIME64_1)))
            .rotate_left(23)
            .wrapping_mul(XXH_PRIME64_2)
            .wrapping_add(XXH_PRIME64_3);
        p += 4;
    }
    while p < input.len() {
        h = (h ^ ((input[p] as u64).wrapping_mul(XXH_PRIME64_5)))
            .rotate_left(11)
            .wrapping_mul(XXH_PRIME64_1);
        p += 1;
    }
    h ^= h >> 33;
    h = h.wrapping_mul(XXH_PRIME64_2);
    h ^= h >> 29;
    h = h.wrapping_mul(XXH_PRIME64_3);
    h ^= h >> 32;
    h
}

#[inline]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(XXH_PRIME64_2))
        .rotate_left(31)
        .wrapping_mul(XXH_PRIME64_1)
}

#[inline]
fn merge_round(acc: u64, val: u64) -> u64 {
    let val = round(0, val);
    (acc ^ val)
        .wrapping_mul(XXH_PRIME64_1)
        .wrapping_add(XXH_PRIME64_4)
}

#[inline]
fn read_u64_le(input: &[u8]) -> u64 {
    u64::from_le_bytes(input[..8].try_into().unwrap())
}

#[inline]
fn read_u32_le(input: &[u8]) -> u32 {
    u32::from_le_bytes(input[..4].try_into().unwrap())
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn read_u8(&mut self) -> Result<u8, DecompressError> {
        if self.pos >= self.bytes.len() {
            return Err(DecompressError::Truncated);
        }
        let v = self.bytes[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn read_u16_le(&mut self) -> Result<u16, DecompressError> {
        if self.pos + 2 > self.bytes.len() {
            return Err(DecompressError::Truncated);
        }
        let v = u16::from_le_bytes(self.bytes[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }
    fn read_u24_le(&mut self) -> Result<u32, DecompressError> {
        if self.pos + 3 > self.bytes.len() {
            return Err(DecompressError::Truncated);
        }
        let v = (self.bytes[self.pos] as u32)
            | ((self.bytes[self.pos + 1] as u32) << 8)
            | ((self.bytes[self.pos + 2] as u32) << 16);
        self.pos += 3;
        Ok(v)
    }
    fn read_u32_le(&mut self) -> Result<u32, DecompressError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(DecompressError::Truncated);
        }
        let v = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn read_u64_le(&mut self) -> Result<u64, DecompressError> {
        if self.pos + 8 > self.bytes.len() {
            return Err(DecompressError::Truncated);
        }
        let v = u64::from_le_bytes(self.bytes[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_raw_frame(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
        // FHD: single_segment=1 (bit 5), content_checksum=0,
        // dict_id_flag=0, frame_content_size_flag=0 (with
        // single_segment=1 → 1-byte FCS).
        out.push(0b0010_0000);
        // FCS: 1 byte = payload.len() (must be <= 255 for this helper).
        assert!(payload.len() < 256);
        out.push(payload.len() as u8);
        // Block header: last_block=1, block_type=0 (raw),
        // block_size=payload.len().
        // Block header bits: bit 0 = last_block, bits 1..3 = block_type=0
        // (Raw), bits 3..24 = block_size.
        let bh = 1u32 | ((payload.len() as u32) << 3);
        out.extend_from_slice(&bh.to_le_bytes()[..3]);
        out.extend_from_slice(payload);
        out
    }

    fn build_rle_frame(byte: u8, count: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
        // FHD: single_segment=0 → window descriptor present; FCS flag=2
        // → 4-byte FCS.
        out.push(0b1000_0000);
        // Window descriptor — exponent 10, mantissa 0 → window=2^10=1024
        // (any small value will do for tests).
        out.push(10 << 3);
        // FCS: 4 bytes.
        out.extend_from_slice(&count.to_le_bytes());
        // Block header: last_block=1, block_type=1 (RLE),
        // block_size=count.
        let bh = 1u32 | (1u32 << 1) | (count << 3);
        out.extend_from_slice(&bh.to_le_bytes()[..3]);
        out.push(byte);
        out
    }

    #[test]
    fn raw_block_round_trip() {
        let frame = build_raw_frame(b"hello squashfs");
        let out = decompress(&frame, 1024).unwrap();
        assert_eq!(out, b"hello squashfs");
    }

    #[test]
    fn rle_block_round_trip() {
        let frame = build_rle_frame(b'X', 16);
        let out = decompress(&frame, 1024).unwrap();
        assert_eq!(out, alloc::vec![b'X'; 16]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut frame = build_raw_frame(b"x");
        frame[0] ^= 0xFF;
        assert!(matches!(
            decompress(&frame, 64),
            Err(DecompressError::BadFormat)
        ));
    }

    #[test]
    fn rejects_dictionary_use() {
        let mut frame = Vec::new();
        frame.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
        // FHD: dict_id_flag=1 → 1 byte after window/FCS.
        frame.push(0b0010_0001);
        frame.push(0); // FCS (single_segment=1 → 1 byte)
        frame.push(0); // dict id
        let r = decompress(&frame, 64);
        assert!(matches!(r, Err(DecompressError::Unsupported)));
    }

    #[test]
    fn output_overflow_rejected() {
        let frame = build_rle_frame(b'A', 100);
        assert!(matches!(
            decompress(&frame, 16),
            Err(DecompressError::OutputTooLarge)
        ));
    }

    #[test]
    fn xxh64_known_vectors() {
        // Seed=0, empty → 0xEF46DB3751D8E999.
        assert_eq!(xxh64(b"", 0), 0xEF46_DB37_51D8_E999);
        // Seed=0, "abc" → 0x44BC2CF5AD770999 (xxhash reference).
        assert_eq!(xxh64(b"abc", 0), 0x44BC_2CF5_AD77_0999);
    }
}
