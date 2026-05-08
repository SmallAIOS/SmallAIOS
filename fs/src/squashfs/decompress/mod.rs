// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Decompressors for the four squashfs 4.0 algorithms supported by
//! SmallAIOS: zstd, xz, gzip, lz4. Every implementation is clean-room
//! `#![no_std]` Rust; no external decompression crate is added to the
//! production dependency graph.
//!
//! Each module exposes a single `decompress(input: &[u8], max_out:
//! usize) -> Result<Vec<u8>, DecompressError>` function. The squashfs
//! metadata block size is fixed at 8 KiB; data blocks are bounded by
//! the superblock's `block_size` (≤ 1 MiB). Callers MUST pass an
//! `max_out` so a malicious image cannot drive the decoder to allocate
//! arbitrarily.
//!
//! Algorithms are gated behind the `fs-squashfs-{zstd,xz,gzip,lz4}`
//! features so a target with no need for a given compressor pays
//! zero binary size.

extern crate alloc;

use alloc::vec::Vec;

use super::superblock::Compression;
use super::SquashfsError;

#[cfg(feature = "fs-squashfs-gzip")]
pub mod gzip;
#[cfg(feature = "fs-squashfs-lz4")]
pub mod lz4;
#[cfg(feature = "fs-squashfs-xz")]
pub mod xz;
#[cfg(feature = "fs-squashfs-zstd")]
pub mod zstd;

/// Generic decompression entry point used by [`super::metadata`] and
/// the data-block reader.
///
/// Routes to the algorithm-specific decoder. Returns
/// `Err(SquashfsError::CompressionFeatureDisabled)` if the chosen
/// algorithm's feature is not built in.
pub fn decompress(
    compression: Compression,
    input: &[u8],
    max_out: usize,
) -> Result<Vec<u8>, SquashfsError> {
    match compression {
        Compression::Gzip => decompress_gzip(input, max_out),
        Compression::Xz => decompress_xz(input, max_out),
        Compression::Lz4 => decompress_lz4(input, max_out),
        Compression::Zstd => decompress_zstd(input, max_out),
    }
}

#[cfg(feature = "fs-squashfs-gzip")]
fn decompress_gzip(input: &[u8], max_out: usize) -> Result<Vec<u8>, SquashfsError> {
    gzip::decompress(input, max_out).map_err(|_| SquashfsError::DecompressFailed)
}
#[cfg(not(feature = "fs-squashfs-gzip"))]
fn decompress_gzip(_input: &[u8], _max_out: usize) -> Result<Vec<u8>, SquashfsError> {
    Err(SquashfsError::CompressionFeatureDisabled("gzip"))
}

#[cfg(feature = "fs-squashfs-xz")]
fn decompress_xz(input: &[u8], max_out: usize) -> Result<Vec<u8>, SquashfsError> {
    xz::decompress(input, max_out).map_err(|_| SquashfsError::DecompressFailed)
}
#[cfg(not(feature = "fs-squashfs-xz"))]
fn decompress_xz(_input: &[u8], _max_out: usize) -> Result<Vec<u8>, SquashfsError> {
    Err(SquashfsError::CompressionFeatureDisabled("xz"))
}

#[cfg(feature = "fs-squashfs-lz4")]
fn decompress_lz4(input: &[u8], max_out: usize) -> Result<Vec<u8>, SquashfsError> {
    lz4::decompress(input, max_out).map_err(|_| SquashfsError::DecompressFailed)
}
#[cfg(not(feature = "fs-squashfs-lz4"))]
fn decompress_lz4(_input: &[u8], _max_out: usize) -> Result<Vec<u8>, SquashfsError> {
    Err(SquashfsError::CompressionFeatureDisabled("lz4"))
}

#[cfg(feature = "fs-squashfs-zstd")]
fn decompress_zstd(input: &[u8], max_out: usize) -> Result<Vec<u8>, SquashfsError> {
    zstd::decompress(input, max_out).map_err(|_| SquashfsError::DecompressFailed)
}
#[cfg(not(feature = "fs-squashfs-zstd"))]
fn decompress_zstd(_input: &[u8], _max_out: usize) -> Result<Vec<u8>, SquashfsError> {
    Err(SquashfsError::CompressionFeatureDisabled("zstd"))
}

/// Common error type used by the four decoder modules. All variants
/// fold to [`SquashfsError::DecompressFailed`] at the squashfs layer
/// so the higher-level read loop is uniform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompressError {
    /// Input buffer truncated mid-frame.
    Truncated,
    /// Header magic / format byte invalid.
    BadFormat,
    /// Decoded output exceeded the `max_out` cap.
    OutputTooLarge,
    /// Decoder-internal invariant violation (codebook overrun, range
    /// state corruption, length code out of range, etc.).
    Corrupt,
    /// Algorithm-specific feature requested by the input is not
    /// supported (e.g. lzma2 dictionary size > 1 GiB, zstd frames with
    /// dictionaries, gzip with FNAME/FCOMMENT in a squashfs raw block).
    Unsupported,
}
