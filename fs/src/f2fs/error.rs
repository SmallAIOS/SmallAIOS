// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS reader error type.
//!
//! Mirrors the squashfs error enum's shape so callers in
//! `posix-vfs` can map both into the same syscall errno surface
//! (Phase 6).

use core::fmt;

use crate::block::BlockError;

/// Errors surfaced by the F2FS reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum F2fsError {
    /// The superblock magic at the canonical offset is not
    /// [`super::F2FS_SUPER_MAGIC`].
    BadMagic,
    /// `superblock.major_ver` is not the supported version.
    UnsupportedVersion(u16, u16),
    /// `superblock.feature` contains a mandatory bit this reader
    /// does not implement. Mount fails closed; no partial mount.
    UnsupportedFeature(u64),
    /// Primary superblock CRC failed AND the secondary copy was also
    /// invalid (or absent).
    SuperblockCrcMismatch,
    /// Both checkpoint packs (CP1, CP2) failed CRC verification.
    CheckpointCrcMismatch,
    /// A NAT block's CRC32C did not match its on-disk checksum.
    NatBlockCrcMismatch,
    /// A SIT block's CRC32C did not match its on-disk checksum.
    SitBlockCrcMismatch,
    /// An inode block's checksum did not match (when the
    /// `inode_checksum` feature is enabled in the superblock).
    InodeChecksumMismatch,
    /// An inode block's structure was unparseable (bad magic, extents
    /// out of range, indirection level deeper than supported, etc.).
    InodeCorrupt,
    /// A path component was not found during `open_path`.
    PathNotFound,
    /// A path component was traversed where the matched inode was not
    /// a directory.
    NotADirectory,
    /// A read overflowed the file's declared `i_size`.
    ReadPastEof,
    /// An on-disk size value exceeded host `usize`.
    SizeOverflow,
    /// A node-id lookup hit out-of-range bounds (NID > NAT capacity).
    NodeIdOutOfRange,
    /// An offset/block-address lookup hit a hole or out-of-range
    /// physical block.
    BadBlockAddress,
    /// Feature: double / triple indirect inode pointers are not yet
    /// supported in this phase. Files that need them surface this.
    DeepIndirectionUnsupported,
    /// Feature: inline extent map points beyond the inode block.
    InlineDataCorrupt,
    /// The reader cannot accept double-indirect or triple-indirect
    /// inode pointers in v1; large files that rely on them surface
    /// `DeepIndirectionUnsupported` (kept distinct so callers can
    /// retry with a narrower offset).
    UnsupportedInodeShape,
    /// Underlying block-device I/O error.
    IoError(BlockError),
}

impl From<BlockError> for F2fsError {
    fn from(e: BlockError) -> Self {
        Self::IoError(e)
    }
}

impl fmt::Display for F2fsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => f.write_str("F2FS bad superblock magic"),
            Self::UnsupportedVersion(maj, min) => {
                write!(f, "F2FS unsupported version {maj}.{min}")
            }
            Self::UnsupportedFeature(bit) => {
                write!(f, "F2FS mandatory feature 0x{bit:04x} unsupported")
            }
            Self::SuperblockCrcMismatch => f.write_str("F2FS superblock CRC mismatch"),
            Self::CheckpointCrcMismatch => f.write_str("F2FS checkpoint CRC mismatch"),
            Self::NatBlockCrcMismatch => f.write_str("F2FS NAT block CRC mismatch"),
            Self::SitBlockCrcMismatch => f.write_str("F2FS SIT block CRC mismatch"),
            Self::InodeChecksumMismatch => f.write_str("F2FS inode checksum mismatch"),
            Self::InodeCorrupt => f.write_str("F2FS inode corrupt"),
            Self::PathNotFound => f.write_str("F2FS path not found"),
            Self::NotADirectory => f.write_str("F2FS path component is not a directory"),
            Self::ReadPastEof => f.write_str("F2FS read past EOF"),
            Self::SizeOverflow => f.write_str("F2FS on-disk size exceeds host usize"),
            Self::NodeIdOutOfRange => f.write_str("F2FS node-id out of range"),
            Self::BadBlockAddress => f.write_str("F2FS bad block address"),
            Self::DeepIndirectionUnsupported => {
                f.write_str("F2FS double/triple-indirect not supported in v1 read path")
            }
            Self::InlineDataCorrupt => f.write_str("F2FS inline data corrupt"),
            Self::UnsupportedInodeShape => f.write_str("F2FS unsupported inode shape"),
            Self::IoError(e) => write!(f, "F2FS block I/O error: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;

    #[test]
    fn from_block_error_wraps() {
        let e: F2fsError = BlockError::Timeout.into();
        assert_eq!(e, F2fsError::IoError(BlockError::Timeout));
    }

    #[test]
    fn display_messages() {
        use core::fmt::Write;
        let cases: &[F2fsError] = &[
            F2fsError::BadMagic,
            F2fsError::UnsupportedVersion(2, 0),
            F2fsError::UnsupportedFeature(0x10),
            F2fsError::SuperblockCrcMismatch,
            F2fsError::CheckpointCrcMismatch,
            F2fsError::NatBlockCrcMismatch,
            F2fsError::SitBlockCrcMismatch,
            F2fsError::InodeChecksumMismatch,
            F2fsError::InodeCorrupt,
            F2fsError::PathNotFound,
            F2fsError::NotADirectory,
            F2fsError::ReadPastEof,
            F2fsError::SizeOverflow,
            F2fsError::NodeIdOutOfRange,
            F2fsError::BadBlockAddress,
            F2fsError::DeepIndirectionUnsupported,
            F2fsError::InlineDataCorrupt,
            F2fsError::UnsupportedInodeShape,
            F2fsError::IoError(BlockError::MediaError),
        ];
        for e in cases {
            let mut s = alloc::string::String::new();
            write!(s, "{e}").unwrap();
            assert!(!s.is_empty());
        }
    }
}
