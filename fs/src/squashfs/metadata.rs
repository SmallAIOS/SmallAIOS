// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Squashfs 4.0 metadata-block reader.
//!
//! Metadata blocks are the unit of compression for the inode, directory,
//! fragment, export, and ID tables. Each block carries:
//!
//! ```text
//!   ┌────────┬──────────────────────────────┐
//!   │ u16 LE │  payload bytes (≤ 8 KiB)     │
//!   └────────┴──────────────────────────────┘
//!     │
//!     │ bit15 = 1 → payload stored uncompressed
//!     │ bit15 = 0 → payload is compressed
//!     │ low 15 bits = on-disk length of the payload
//! ```
//!
//! The decompressed payload is always between 1 and 8192 bytes; if a
//! producer would generate a metadata block that compresses worse than
//! storing it raw, it stores it raw with bit 15 set.
//!
//! Variable-length structures (inodes, directory entries) span metadata
//! blocks freely. The [`Cursor`] type below abstracts that: it pre-reads
//! one metadata block at a time, refilling from the underlying squashfs
//! image when callers walk past the current block.

extern crate alloc;

use alloc::vec::Vec;

use super::decompress::decompress;
use super::superblock::Superblock;
use super::{SquashfsError, METADATA_BLOCK_SIZE};

/// Header length of a metadata block (always 2 bytes).
pub const METADATA_HEADER_LEN: usize = 2;

/// Read one metadata block and return its decompressed payload plus the
/// total on-disk length consumed (header + payload).
///
/// `image[offset..]` MUST point at a metadata-block header.
pub fn read_metadata_block(
    image: &[u8],
    offset: usize,
    sb: &Superblock,
) -> Result<(Vec<u8>, usize), SquashfsError> {
    if offset.saturating_add(METADATA_HEADER_LEN) > image.len() {
        return Err(SquashfsError::OutOfRange);
    }
    let hdr = u16::from_le_bytes([image[offset], image[offset + 1]]);
    let stored_uncompressed = hdr & 0x8000 != 0;
    let on_disk_len = (hdr & 0x7FFF) as usize;
    if on_disk_len == 0 || on_disk_len > METADATA_BLOCK_SIZE {
        return Err(SquashfsError::CorruptMetadata);
    }
    let total = METADATA_HEADER_LEN
        .checked_add(on_disk_len)
        .ok_or(SquashfsError::SizeOverflow)?;
    if offset.saturating_add(total) > image.len() {
        return Err(SquashfsError::OutOfRange);
    }
    let payload = &image[offset + METADATA_HEADER_LEN..offset + total];
    let out = if stored_uncompressed {
        let mut v = Vec::with_capacity(payload.len());
        v.extend_from_slice(payload);
        v
    } else {
        decompress(sb.compression, payload, METADATA_BLOCK_SIZE)?
    };
    if out.is_empty() || out.len() > METADATA_BLOCK_SIZE {
        return Err(SquashfsError::CorruptMetadata);
    }
    Ok((out, total))
}

/// Cursor over a metadata table starting at `start_offset` in the
/// squashfs image. The cursor lazily decompresses one metadata block
/// at a time; readers walk forward by bytes and the cursor auto-
/// refills when a read crosses a block boundary.
///
/// Squashfs encodes "metadata reference" pointers as a 64-bit value
/// where the high 48 bits are the byte offset of the *first* metadata
/// block containing the structure (`block_offset`) and the low 16 bits
/// are the offset within that block's decompressed payload
/// (`offset_in_block`). [`Cursor::seek_ref`] decodes a metadata
/// reference and positions the cursor at the right byte.
pub struct Cursor<'a> {
    /// The full squashfs image bytes (post-footer-strip).
    image: &'a [u8],
    /// Reference to the parsed superblock.
    sb: &'a Superblock,
    /// Byte offset of the metadata block currently held in `current`.
    block_offset: usize,
    /// Decompressed payload of the metadata block at `block_offset`.
    current: Vec<u8>,
    /// Read position within `current`.
    pos: usize,
    /// Byte offset of the metadata block immediately following
    /// `block_offset` (i.e., what to refill from when `pos == current.len()`).
    next_offset: usize,
}

impl<'a> Cursor<'a> {
    /// Create a cursor positioned at the start of the metadata block at
    /// `start_offset` (an absolute squashfs-image byte offset).
    pub fn new(
        image: &'a [u8],
        sb: &'a Superblock,
        start_offset: usize,
    ) -> Result<Self, SquashfsError> {
        let (current, consumed) = read_metadata_block(image, start_offset, sb)?;
        Ok(Self {
            image,
            sb,
            block_offset: start_offset,
            current,
            pos: 0,
            next_offset: start_offset + consumed,
        })
    }

    /// Position the cursor at the metadata block at `block_offset` and
    /// `offset_in_block` bytes into its decompressed payload.
    pub fn seek_to(
        &mut self,
        block_offset: usize,
        offset_in_block: usize,
    ) -> Result<(), SquashfsError> {
        if block_offset != self.block_offset {
            let (current, consumed) = read_metadata_block(self.image, block_offset, self.sb)?;
            self.current = current;
            self.block_offset = block_offset;
            self.next_offset = block_offset + consumed;
        }
        if offset_in_block > self.current.len() {
            return Err(SquashfsError::OutOfRange);
        }
        self.pos = offset_in_block;
        Ok(())
    }

    /// Decode a 64-bit squashfs metadata reference (`high 48 = block
    /// offset relative to `base`, low 16 = offset in block`) and seek.
    pub fn seek_ref(&mut self, base: u64, reference: u64) -> Result<(), SquashfsError> {
        let block_rel = reference >> 16;
        let in_block = (reference & 0xFFFF) as usize;
        let abs = base
            .checked_add(block_rel)
            .ok_or(SquashfsError::SizeOverflow)?;
        let abs_usize = usize::try_from(abs).map_err(|_| SquashfsError::SizeOverflow)?;
        self.seek_to(abs_usize, in_block)
    }

    /// Read exactly `out.len()` bytes, refilling across metadata blocks
    /// as needed.
    pub fn read_exact(&mut self, out: &mut [u8]) -> Result<(), SquashfsError> {
        let mut written = 0;
        while written < out.len() {
            if self.pos >= self.current.len() {
                self.advance_to_next_block()?;
            }
            let take = core::cmp::min(out.len() - written, self.current.len() - self.pos);
            out[written..written + take].copy_from_slice(&self.current[self.pos..self.pos + take]);
            self.pos += take;
            written += take;
        }
        Ok(())
    }

    /// Read a `u16` little-endian.
    pub fn read_u16(&mut self) -> Result<u16, SquashfsError> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    /// Read a `u32` little-endian.
    pub fn read_u32(&mut self) -> Result<u32, SquashfsError> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Read a `u64` little-endian.
    pub fn read_u64(&mut self) -> Result<u64, SquashfsError> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Read a `i16` little-endian.
    pub fn read_i16(&mut self) -> Result<i16, SquashfsError> {
        Ok(self.read_u16()? as i16)
    }

    /// Read a `i32` little-endian.
    pub fn read_i32(&mut self) -> Result<i32, SquashfsError> {
        Ok(self.read_u32()? as i32)
    }

    /// Currently-held block offset.
    pub fn block_offset(&self) -> usize {
        self.block_offset
    }

    /// Current read position within the held block.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Squashfs metadata reference for the current cursor position
    /// (handy for storing pointers to inode entries).
    pub fn reference(&self, base: u64) -> u64 {
        ((self.block_offset as u64 - base) << 16) | (self.pos as u64 & 0xFFFF)
    }

    fn advance_to_next_block(&mut self) -> Result<(), SquashfsError> {
        let (current, consumed) = read_metadata_block(self.image, self.next_offset, self.sb)?;
        self.block_offset = self.next_offset;
        self.next_offset += consumed;
        self.current = current;
        self.pos = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::squashfs::superblock::{Compression, Superblock};

    fn dummy_superblock() -> Superblock {
        Superblock {
            inode_count: 1,
            mod_time: 0,
            block_size: 4096,
            fragment_count: 0,
            compression: Compression::Gzip, // unused for uncompressed metadata
            block_log: 12,
            flags: 0,
            id_count: 0,
            version_major: 4,
            version_minor: 0,
            root_inode_ref: 0,
            bytes_used: 0,
            id_table_start: 0,
            xattr_table_start: u64::MAX,
            inode_table_start: 0,
            directory_table_start: 0,
            fragment_table_start: u64::MAX,
            lookup_table_start: u64::MAX,
        }
    }

    fn write_uncompressed_block(image: &mut Vec<u8>, payload: &[u8]) -> usize {
        let off = image.len();
        let hdr = (payload.len() as u16) | 0x8000;
        image.extend_from_slice(&hdr.to_le_bytes());
        image.extend_from_slice(payload);
        off
    }

    #[test]
    fn reads_uncompressed_block() {
        let mut img = Vec::new();
        let off = write_uncompressed_block(&mut img, b"hello world");
        let sb = dummy_superblock();
        let (out, consumed) = read_metadata_block(&img, off, &sb).unwrap();
        assert_eq!(out, b"hello world");
        assert_eq!(consumed, 2 + b"hello world".len());
    }

    #[test]
    fn rejects_oversized_block_header() {
        let img = alloc::vec![0xFFu8, 0x7F, 0u8, 0u8];
        let sb = dummy_superblock();
        // 0x7FFF claims 32767 bytes, but the buffer only has 2 left.
        assert!(read_metadata_block(&img, 0, &sb).is_err());
    }

    #[test]
    fn rejects_zero_length_block() {
        let img = alloc::vec![0u8, 0u8];
        let sb = dummy_superblock();
        assert_eq!(
            read_metadata_block(&img, 0, &sb),
            Err(SquashfsError::CorruptMetadata)
        );
    }

    #[test]
    fn cursor_reads_across_blocks() {
        let mut img = Vec::new();
        write_uncompressed_block(&mut img, &[1, 2, 3, 4]);
        write_uncompressed_block(&mut img, &[5, 6, 7, 8]);
        let sb = dummy_superblock();
        let mut c = Cursor::new(&img, &sb, 0).unwrap();
        let mut out = [0u8; 8];
        c.read_exact(&mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn cursor_seek_ref_round_trip() {
        let mut img = Vec::new();
        let off0 = write_uncompressed_block(&mut img, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let off1 = write_uncompressed_block(&mut img, &[9, 10]);
        assert_eq!(off0, 0);
        let sb = dummy_superblock();
        let mut c = Cursor::new(&img, &sb, 0).unwrap();
        // The metadata-reference encoding uses the absolute byte offset
        // as the high 48 bits; choose `base = 0`.
        let r = ((off1 as u64) << 16) | 1;
        c.seek_ref(0, r).unwrap();
        let mut out = [0u8; 1];
        c.read_exact(&mut out).unwrap();
        assert_eq!(out, [10]);
    }
}
