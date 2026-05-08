// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Squashfs 4.0 read-only reader.
//!
//! Implements the on-disk format documented in
//! <https://dr-emann.github.io/squashfs/squashfs.html> (community
//! reference for squashfs 4.0; the upstream `mksquashfs(1)` man page
//! and the Linux 6.6 kernel `fs/squashfs/` source agree on every
//! field).
//!
//! Scope:
//!
//! - Read-only, single mount per device. No write path; the kernel
//!   returns `-EROFS` for any write to a path under the mounted
//!   squashfs (enforced higher in the VFS).
//! - Major version `4`. Versions `< 4` and `≥ 5` are rejected with
//!   [`SquashfsError::UnsupportedMajorVersion`].
//! - Compression IDs:
//!     - `1 = gzip` (DEFLATE / RFC 1951)
//!     - `4 = xz` (LZMA2 inside an xz container)
//!     - `5 = lz4` (block format)
//!     - `6 = zstd` (RFC 8478)
//!   Legacy IDs `2 = lzma` and `3 = lzo` are rejected.
//! - SmallAIOS manifest footer: a sealed footer appended after the
//!   squashfs's natural EOF carrying SHA-3-256-per-4 KiB hashes plus
//!   an ML-DSA-65 signature over the hash array. The reader verifies
//!   both the signature at mount time and every block's hash on read.
//!   See [`manifest`].
//!
//! ## Module map
//!
//! | Module        | Role                                                      |
//! |---------------|-----------------------------------------------------------|
//! | [`superblock`]| Squashfs 4.0 superblock parser                            |
//! | [`metadata`]  | Metadata-block reader (16-bit length prefix + decompress) |
//! | [`inodes`]    | Inode table reader (basic + extended types)               |
//! | [`dirs`]      | Directory table reader and iterator                       |
//! | [`fragments`] | Fragment-table reader                                     |
//! | [`exports`]   | Export-table reader (NFS handles)                         |
//! | [`ids`]       | UID/GID lookup-table reader                               |
//! | [`decompress`]| zstd/xz/gzip/lz4 decompressors                            |
//! | [`manifest`]  | SmallAIOS manifest footer parser + ML-DSA-65 verify       |
//! | [`mount`]     | Top-level [`mount::Squashfs`] read API                    |
//!
//! Owned by `embedded-filesystem-v1` Phase 3.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::block::BlockError;

pub mod decompress;
pub mod dirs;
pub mod exports;
pub mod fragments;
pub mod ids;
pub mod inodes;
pub mod manifest;
pub mod metadata;
pub mod mount;
pub mod superblock;

pub use mount::{DirEntry, DirIter, Inode, InodeKind, Squashfs};

pub use superblock::{Compression, Superblock};

/// Squashfs 4.0 magic at offset 0 of the superblock (`hsqs`, little-endian).
//
// lgtm[rust/hard-coded-cryptographic-value] — squashfs 4.0 superblock magic, public on-disk constant
pub const SQUASHFS_MAGIC: u32 = 0x73717368;

/// Maximum length, in bytes, of an uncompressed metadata block (squashfs 4.0).
pub const METADATA_BLOCK_SIZE: usize = 8192;

/// Squashfs 4.0 only — major version. Any other major version is rejected.
pub const SQUASHFS_MAJOR_VERSION: u16 = 4;

/// Errors surfaced by the squashfs reader and manifest verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SquashfsError {
    /// The superblock magic at offset 0 is not [`SQUASHFS_MAGIC`].
    BadMagic,
    /// `superblock.major != 4`.
    UnsupportedMajorVersion(u16),
    /// `superblock.compression_id` is not one of the supported values.
    UnsupportedCompression(u16),
    /// Compression-id is one we recognize at the type level but the
    /// corresponding cargo feature is not enabled.
    CompressionFeatureDisabled(&'static str),
    /// A metadata or data block failed to decompress.
    DecompressFailed,
    /// A metadata or data block had an obviously invalid header (length
    /// > 8 KiB, etc.).
    CorruptMetadata,
    /// An inode/directory cursor walked past the end of its table.
    OutOfRange,
    /// An on-disk integer exceeded `usize` on a 32-bit host.
    SizeOverflow,
    /// SmallAIOS manifest footer is missing, malformed, or has bad magic.
    ManifestMissing,
    /// SmallAIOS manifest footer was found but its declared length is
    /// inconsistent (out of range, footer size mismatch, etc.).
    ManifestCorrupt,
    /// SmallAIOS manifest footer's ML-DSA-65 signature failed verification.
    ManifestSignatureInvalid,
    /// SmallAIOS manifest footer has a public-key fingerprint that does
    /// not match the public key supplied at mount time.
    ManifestPublicKeyMismatch,
    /// A read returned bytes whose SHA-3-256 hash did not match the
    /// manifest entry. Mapped to [`BlockError::BadCrc`] at the block
    /// boundary.
    BlockHashMismatch,
    /// A path-resolution call descended through a component that wasn't
    /// a directory.
    NotADirectory,
    /// A path-resolution call could not find a component.
    NotFound,
    /// A read overflow at the file/inode level.
    ReadPastEof,
    /// Inode kind is not one of the eight defined in squashfs 4.0.
    UnknownInodeKind(u16),
    /// Underlying block-device error.
    Block(BlockError),
}

impl From<BlockError> for SquashfsError {
    fn from(e: BlockError) -> Self {
        Self::Block(e)
    }
}

impl fmt::Display for SquashfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => f.write_str("squashfs bad magic"),
            Self::UnsupportedMajorVersion(v) => {
                write!(f, "squashfs major version {v}.x unsupported")
            }
            Self::UnsupportedCompression(id) => {
                write!(f, "unsupported squashfs compression: id={id}")
            }
            Self::CompressionFeatureDisabled(name) => {
                write!(f, "squashfs compression {name} disabled by build feature")
            }
            Self::DecompressFailed => f.write_str("squashfs decompression failed"),
            Self::CorruptMetadata => f.write_str("squashfs metadata corrupt"),
            Self::OutOfRange => f.write_str("squashfs cursor out of range"),
            Self::SizeOverflow => f.write_str("squashfs on-disk size exceeds host usize"),
            Self::ManifestMissing => f.write_str("squashfs missing SmallAIOS manifest footer"),
            Self::ManifestCorrupt => f.write_str("squashfs manifest footer corrupt"),
            Self::ManifestSignatureInvalid => f.write_str("squashfs signature invalid"),
            Self::ManifestPublicKeyMismatch => {
                f.write_str("squashfs manifest public-key fingerprint mismatch")
            }
            Self::BlockHashMismatch => f.write_str("squashfs block hash mismatch"),
            Self::NotADirectory => f.write_str("squashfs path component is not a directory"),
            Self::NotFound => f.write_str("squashfs path component not found"),
            Self::ReadPastEof => f.write_str("squashfs read past EOF"),
            Self::UnknownInodeKind(k) => write!(f, "unknown squashfs inode kind {k}"),
            Self::Block(e) => write!(f, "block error: {e}"),
        }
    }
}

/// One-shot helper that runs `f` and converts any [`SquashfsError`]
/// that originates from per-block hash verification into the canonical
/// [`BlockError::BadCrc`] surfaced to filesystem callers.
///
/// Used by the integrity layer in [`mount`] so the block-I/O contract
/// stays exact: hash-mismatch is the same wire error as a hardware
/// CRC failure.
pub fn into_block_error(e: SquashfsError) -> BlockError {
    match e {
        SquashfsError::BlockHashMismatch => BlockError::BadCrc,
        SquashfsError::Block(b) => b,
        _ => BlockError::MediaError,
    }
}

/// Convenience: convert an [`SquashfsError`] to a short static message
/// suitable for the audit-ring stub.
pub(crate) fn audit_message(e: &SquashfsError) -> &'static str {
    match e {
        SquashfsError::ManifestMissing => "mount_manifest_missing",
        SquashfsError::ManifestCorrupt => "mount_manifest_corrupt",
        SquashfsError::ManifestSignatureInvalid => "mount_signature_invalid",
        SquashfsError::ManifestPublicKeyMismatch => "mount_pubkey_mismatch",
        SquashfsError::BlockHashMismatch => "block_hash_mismatch",
        _ => "squashfs_other_error",
    }
}

/// Append an audit record. This is a stub until `mgmt-audit-log`
/// lands in the crate; the real audit ring is filled by the
/// management-login change. Callers MUST still invoke it so the
/// real ring picks up automatically when wired.
pub(crate) fn audit_stub(event: &'static str, _detail: &str) {
    #[cfg(feature = "fs-test-helpers")]
    audit_test::record(event);
    let _ = event;
}

/// Drain the per-test audit-event log. Returns events fired since the
/// last call to [`audit_test_drain`] or [`audit_test_clear`].
///
/// Available only when the `fs-test-helpers` feature is enabled, which
/// pulls in `std` for `thread_local!`. Bare-metal kernel builds do
/// NOT enable this feature; production overhead is zero.
#[cfg(feature = "fs-test-helpers")]
pub fn audit_test_drain() -> alloc::vec::Vec<&'static str> {
    audit_test::drain()
}

/// Clear any pending audit events recorded by the test latch.
#[cfg(feature = "fs-test-helpers")]
pub fn audit_test_clear() {
    audit_test::clear()
}

#[cfg(feature = "fs-test-helpers")]
pub(crate) mod audit_test {
    extern crate std;
    use core::cell::RefCell;
    use std::thread_local;
    use std::vec::Vec;

    thread_local! {
        static EVENTS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    }

    pub fn record(event: &'static str) {
        EVENTS.with(|e| e.borrow_mut().push(event));
    }

    pub fn drain() -> Vec<&'static str> {
        EVENTS.with(|e| core::mem::take(&mut *e.borrow_mut()))
    }

    pub fn clear() {
        EVENTS.with(|e| e.borrow_mut().clear());
    }
}

/// Read raw bytes from a partition by absolute byte offset. The
/// helper is used by superblock + manifest parsing where the offsets
/// are not block-aligned.
///
/// Reads through the [`crate::block::SectorAdapter`]-style 4 KiB block
/// interface and scratches as little memory as possible.
#[allow(dead_code)]
pub(crate) fn read_partition_bytes<D: crate::block::BlockDevice + ?Sized>(
    device: &D,
    partition_lba: u64,
    partition_size_bytes: u64,
    offset: u64,
    out: &mut [u8],
) -> Result<(), SquashfsError> {
    let bs = device.block_size_bytes() as u64;
    if bs == 0 {
        return Err(SquashfsError::Block(BlockError::Unaligned));
    }
    let len = out.len() as u64;
    if offset.checked_add(len).ok_or(SquashfsError::SizeOverflow)? > partition_size_bytes {
        return Err(SquashfsError::ReadPastEof);
    }

    // Walk one underlying block at a time; copy a sub-slice into `out`.
    let mut written: usize = 0;
    let mut cursor = offset;
    let mut remaining = len;
    let mut scratch: Vec<u8> = alloc::vec![0u8; bs as usize];
    while remaining > 0 {
        let abs_byte = partition_lba.saturating_mul(bs) + cursor;
        let block = abs_byte / bs;
        let in_block_off = (abs_byte % bs) as usize;
        device.read_block(block, &mut scratch)?;
        let take = core::cmp::min(remaining as usize, bs as usize - in_block_off);
        out[written..written + take].copy_from_slice(&scratch[in_block_off..in_block_off + take]);
        written += take;
        cursor += take as u64;
        remaining -= take as u64;
    }
    Ok(())
}

/// Stable 32-byte fingerprint of an ML-DSA-65 public key, used by the
/// manifest footer.
///
/// Defined as `SHA-3-256(public_key_bytes)` so the kernel can verify
/// "this manifest was signed by exactly this key" before doing the
/// (much more expensive) ML-DSA-65 verification. The 32-byte length
/// matches the Sha3_256 digest in the existing `security` crate.
pub fn fingerprint_public_key(pk: &[u8]) -> [u8; 32] {
    use smallaios_security::crypto::sha3::sha3_256;
    *sha3_256(pk).as_bytes()
}

/// Small UTF-8 path utilities used by [`mount::Squashfs::open_path`].
///
/// Squashfs stores names as raw byte strings without an encoding flag.
/// We require valid UTF-8 for any name returned to user code so the
/// VFS layer can always render paths. Names that are not valid UTF-8
/// are returned as `String::from_utf8_lossy` (with replacement chars)
/// to avoid panicking; the caller can then reject them.
pub(crate) fn lossy_string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}
