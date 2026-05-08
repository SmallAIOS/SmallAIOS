// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Top-level squashfs mount API.
//!
//! `Squashfs::mount(device, partition_lba, size_lbas, public_key)`:
//!
//! 1. Reads the entire squashfs partition through the [`BlockDevice`]
//!    interface into a `Vec<u8>`. Each 4 KiB chunk is verified
//!    against the manifest's SHA-3-256 entry as it is read; on
//!    mismatch the mount fails immediately with
//!    [`SquashfsError::BlockHashMismatch`] and an audit-stub call is
//!    fired (`block_hash_mismatch`).
//! 2. Parses + signature-verifies the SmallAIOS manifest footer
//!    (per `fs-squashfs-readonly` and `fs-integrity`).
//! 3. Parses the squashfs superblock and required tables.
//! 4. Returns a [`Squashfs`] handle on success.
//!
//! After mount, the only data path that can read user bytes is the
//! `Squashfs::read_file` method; that path serves bytes from the
//! already-verified in-memory partition copy. Per the integrity spec,
//! every block returned has been validated against the manifest at
//! load time, so re-validation per syscall is not required (the
//! verification happens once, fail-closed, before mount succeeds).
//!
//! Note: the in-memory model is appropriate for typical squashfs
//! payloads (≤ 4 GiB) on hosts with sufficient RAM. The Phase 6
//! block-cache overhaul will replace it with a streaming + LRU layer.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::block::BlockDevice;

use super::decompress::decompress;
use super::dirs::{parse_directory, ParsedDirEntry};
use super::fragments::{read_fragment_table, FragmentEntry};
use super::ids::read_id_table;
use super::inodes::{
    BasicDirInode, BasicFileInode, BasicSymlinkInode, ExtDirInode, ExtFileInode, InodeHeader,
    INODE_KIND_BASIC_DIR, INODE_KIND_BASIC_FILE, INODE_KIND_BASIC_SYMLINK, INODE_KIND_EXT_DIR,
    INODE_KIND_EXT_FILE, INODE_KIND_EXT_SYMLINK,
};
use super::manifest::{Manifest, BLOCK_HASH_LEN, HASH_BLOCK_SIZE};
use super::metadata::Cursor;
use super::superblock::{Superblock, SUPERBLOCK_SIZE};
use super::SquashfsError;

/// Inode kinds surfaced to the user-facing read API.
///
/// Internally we track far more detail (Basic vs Extended, etc.); this
/// enum projects to what `posix-vfs` actually needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InodeKind {
    Directory,
    File,
    Symlink,
    Other,
}

impl InodeKind {
    fn from_raw(k: u16) -> Self {
        match k {
            INODE_KIND_BASIC_DIR | INODE_KIND_EXT_DIR => Self::Directory,
            INODE_KIND_BASIC_FILE | INODE_KIND_EXT_FILE => Self::File,
            INODE_KIND_BASIC_SYMLINK | INODE_KIND_EXT_SYMLINK => Self::Symlink,
            _ => Self::Other,
        }
    }
}

/// One squashfs inode resolved by [`Squashfs::open_path`].
#[derive(Debug, Clone)]
pub struct Inode {
    pub kind: InodeKind,
    /// Inode number (unique per squashfs image).
    pub inode_number: u32,
    /// File size in bytes (0 for non-files).
    pub size: u64,
    body: InodeBody,
}

#[derive(Debug, Clone)]
enum InodeBody {
    Dir(DirBody),
    File(FileBody),
    #[allow(dead_code)] // target string used by future getxattr / readlink path
    Symlink(String),
}

#[derive(Debug, Clone)]
struct DirBody {
    start_block: u32,
    size: u32,
    offset: u16,
}

#[derive(Debug, Clone)]
struct FileBody {
    blocks_start: u64,
    file_size: u64,
    block_sizes: Vec<u32>,
    frag_idx: u32,
    frag_offset: u32,
}

/// One directory entry produced by [`DirIter`].
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub kind: InodeKind,
    pub inode_number: u32,
    pub(crate) inode_ref: u64,
}

/// Iterator over a directory's entries.
///
/// The iterator is independent of the filesystem handle once
/// constructed. To open the inode behind an entry, call
/// [`Squashfs::open_dir_entry`].
pub struct DirIter {
    entries: alloc::collections::VecDeque<ParsedDirEntry>,
}

impl Iterator for DirIter {
    type Item = DirEntry;
    fn next(&mut self) -> Option<Self::Item> {
        let parsed = self.entries.pop_front()?;
        Some(DirEntry {
            name: parsed.name,
            kind: InodeKind::from_raw(parsed.kind),
            inode_number: parsed.inode_number,
            inode_ref: parsed.inode_ref,
        })
    }
}

/// In-memory squashfs reader.
///
/// Holds the entire partition in a `Vec<u8>` along with parsed
/// metadata structures (superblock, fragment table, ID table, manifest).
/// The original [`BlockDevice`] is consulted only at mount time and
/// is not retained by the [`Squashfs`] handle, so the handle has no
/// lifetime parameter.
pub struct Squashfs<D: BlockDevice> {
    /// The full partition bytes (squashfs blob followed by manifest
    /// footer), already block-hash-verified during mount.
    partition: Vec<u8>,
    /// Length of the squashfs blob (everything before the manifest
    /// footer).
    squashfs_len: usize,
    superblock: Superblock,
    manifest: Manifest,
    fragment_table: Vec<FragmentEntry>,
    id_table: Vec<u32>,
    _phantom: core::marker::PhantomData<D>,
}

impl<D: BlockDevice> Squashfs<D> {
    /// Mount a squashfs partition.
    ///
    /// `partition_lba` and `size_lbas` are in `device.block_size_bytes()`-
    /// sized units. The kernel SHALL pass `public_key` as the embedded
    /// SmallAIOS update-signing key.
    pub fn mount(
        device: &D,
        partition_lba: u64,
        size_lbas: u64,
        public_key: &[u8],
    ) -> Result<Self, SquashfsError> {
        let bs = device.block_size_bytes() as u64;
        if bs == 0 {
            return Err(SquashfsError::Block(crate::block::BlockError::Unaligned));
        }
        let total_bytes = size_lbas
            .checked_mul(bs)
            .ok_or(SquashfsError::SizeOverflow)?;
        let total_bytes_usize =
            usize::try_from(total_bytes).map_err(|_| SquashfsError::SizeOverflow)?;
        let mut partition = alloc::vec![0u8; total_bytes_usize];
        // Read the entire partition through the BlockDevice.
        let mut scratch = alloc::vec![0u8; bs as usize];
        for i in 0..size_lbas {
            let lba = partition_lba + i;
            device.read_block(lba, &mut scratch)?;
            let off = (i as usize) * bs as usize;
            partition[off..off + bs as usize].copy_from_slice(&scratch);
        }

        // Step 1 — locate + verify the manifest footer.
        let manifest = match Manifest::parse_and_verify(&partition, public_key) {
            Ok(m) => m,
            Err(e) => {
                super::audit_stub(super::audit_message(&e), "");
                return Err(e);
            }
        };
        let squashfs_len = manifest.squashfs_length(partition.len());

        // Step 2 — verify every 4 KiB block of the squashfs blob
        // against the manifest BEFORE any data is honored. Fail closed
        // on first mismatch.
        let n_hashes = manifest.hash_count();
        let blob = &partition[..squashfs_len];
        for i in 0..n_hashes {
            let start = i * HASH_BLOCK_SIZE as usize;
            let end = core::cmp::min(start + HASH_BLOCK_SIZE as usize, blob.len());
            let mut block_buf = alloc::vec![0u8; HASH_BLOCK_SIZE as usize];
            block_buf[..end - start].copy_from_slice(&blob[start..end]);
            if let Err(e) = manifest.verify_block(i as u64, &block_buf) {
                super::audit_stub(super::audit_message(&e), "");
                return Err(e);
            }
        }

        // Step 3 — parse the squashfs superblock.
        if squashfs_len < SUPERBLOCK_SIZE {
            return Err(SquashfsError::CorruptMetadata);
        }
        let superblock = Superblock::parse(&partition[..SUPERBLOCK_SIZE])?;

        // Step 4 — read fragment + ID tables (export table is optional).
        let fragment_table = read_fragment_table(&partition[..squashfs_len], &superblock)?;
        let id_table = read_id_table(&partition[..squashfs_len], &superblock)?;

        Ok(Self {
            partition,
            squashfs_len,
            superblock,
            manifest,
            fragment_table,
            id_table,
            _phantom: core::marker::PhantomData,
        })
    }

    /// 32-byte ML-DSA-65 public-key fingerprint embedded in the manifest.
    pub fn manifest_fingerprint(&self) -> &[u8; BLOCK_HASH_LEN] {
        self.manifest.fingerprint()
    }

    /// Parsed superblock (read-only view).
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Resolve a path like `/foo/bar/baz` against the root inode.
    ///
    /// Each path component must exist; symlinks are NOT followed
    /// transparently (POSIX is in Phase 6). Empty components are
    /// permitted (`//foo` is treated as `/foo`).
    pub fn open_path(&self, path: &str) -> Result<Inode, SquashfsError> {
        let mut current = self.open_inode_ref(self.superblock.root_inode_ref)?;
        for component in path.split('/').filter(|c| !c.is_empty()) {
            let body = match &current.body {
                InodeBody::Dir(b) => b.clone(),
                _ => return Err(SquashfsError::NotADirectory),
            };
            let entries = self.read_dir_body(&body)?;
            let mut found = None;
            for e in entries {
                if e.name == component {
                    found = Some(e);
                    break;
                }
            }
            let entry = found.ok_or(SquashfsError::NotFound)?;
            current = self.open_inode_ref(entry.inode_ref)?;
        }
        Ok(current)
    }

    /// Read up to `buf.len()` bytes from `inode` starting at `offset`.
    /// Returns the number of bytes actually written into `buf`.
    pub fn read_file(
        &self,
        inode: &Inode,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, SquashfsError> {
        let body = match &inode.body {
            InodeBody::File(b) => b,
            _ => return Err(SquashfsError::NotADirectory),
        };
        if offset >= body.file_size {
            return Ok(0);
        }
        let want = core::cmp::min(buf.len() as u64, body.file_size - offset) as usize;
        let mut written: usize = 0;
        let mut cursor_offset = offset;

        // Walk full data blocks.
        while written < want {
            let block_index = (cursor_offset / self.superblock.block_size as u64) as usize;
            let in_block_offset = (cursor_offset % self.superblock.block_size as u64) as usize;
            // Past the last full data block?
            if block_index >= body.block_sizes.len() {
                break;
            }
            // Compute starting absolute offset of this data block.
            let mut start: u64 = body.blocks_start;
            for prior in &body.block_sizes[..block_index] {
                start = start
                    .checked_add((*prior & 0x00FF_FFFF) as u64)
                    .ok_or(SquashfsError::SizeOverflow)?;
            }
            let raw_size = body.block_sizes[block_index];
            let on_disk_size = (raw_size & 0x00FF_FFFF) as usize;
            let uncompressed = raw_size & 0x0100_0000 != 0;
            let s_usize = usize::try_from(start).map_err(|_| SquashfsError::SizeOverflow)?;
            if s_usize + on_disk_size > self.squashfs_len {
                return Err(SquashfsError::OutOfRange);
            }
            let block_bytes: Vec<u8> = if on_disk_size == 0 {
                // Sparse: a stretch of zeros up to block_size.
                alloc::vec![0u8; self.superblock.block_size as usize]
            } else if uncompressed {
                self.partition[s_usize..s_usize + on_disk_size].to_vec()
            } else {
                decompress(
                    self.superblock.compression,
                    &self.partition[s_usize..s_usize + on_disk_size],
                    self.superblock.block_size as usize,
                )?
            };
            let block_take = core::cmp::min(want - written, block_bytes.len() - in_block_offset);
            buf[written..written + block_take]
                .copy_from_slice(&block_bytes[in_block_offset..in_block_offset + block_take]);
            written += block_take;
            cursor_offset += block_take as u64;
        }

        // Trailing fragment (if any).
        if written < want && body.frag_idx != 0xFFFF_FFFF {
            let frag = self
                .fragment_table
                .get(body.frag_idx as usize)
                .ok_or(SquashfsError::OutOfRange)?;
            let frag_start =
                usize::try_from(frag.start).map_err(|_| SquashfsError::SizeOverflow)?;
            let frag_size = frag.on_disk_size() as usize;
            if frag_start + frag_size > self.squashfs_len {
                return Err(SquashfsError::OutOfRange);
            }
            let frag_block: Vec<u8> = if frag.is_uncompressed() {
                self.partition[frag_start..frag_start + frag_size].to_vec()
            } else {
                decompress(
                    self.superblock.compression,
                    &self.partition[frag_start..frag_start + frag_size],
                    self.superblock.block_size as usize,
                )?
            };
            let frag_offset_in_file =
                body.block_sizes.len() as u64 * self.superblock.block_size as u64;
            // The bytes belonging to this file inside the fragment run
            // from `body.frag_offset` to `body.frag_offset +
            // (file_size - frag_offset_in_file)`.
            let cursor_in_frag = (cursor_offset - frag_offset_in_file) as usize;
            let avail = (body.file_size - frag_offset_in_file) as usize - cursor_in_frag;
            let take = core::cmp::min(want - written, avail);
            let off = body.frag_offset as usize + cursor_in_frag;
            buf[written..written + take].copy_from_slice(&frag_block[off..off + take]);
            written += take;
        }

        Ok(written)
    }

    /// Iterate the entries of a directory.
    pub fn read_dir(&self, inode: &Inode) -> Result<DirIter, SquashfsError> {
        let body = match &inode.body {
            InodeBody::Dir(b) => b.clone(),
            _ => return Err(SquashfsError::NotADirectory),
        };
        let entries = self.read_dir_body(&body)?;
        Ok(DirIter {
            entries: entries.into_iter().collect(),
        })
    }

    /// Open the inode pointed to by a [`DirEntry`] without re-walking.
    pub fn open_dir_entry(&self, entry: &DirEntry) -> Result<Inode, SquashfsError> {
        self.open_inode_ref(entry.inode_ref)
    }

    /// Number of UID/GID entries in the ID lookup table.
    pub fn id_count(&self) -> usize {
        self.id_table.len()
    }

    fn read_dir_body(&self, body: &DirBody) -> Result<Vec<ParsedDirEntry>, SquashfsError> {
        // Directory body lives in the directory table at
        // `directory_table_start + start_block`, then `offset` bytes
        // into the decompressed metadata block.
        let dir_table_base = self.superblock.directory_table_start;
        let start_abs = dir_table_base
            .checked_add(body.start_block as u64)
            .ok_or(SquashfsError::SizeOverflow)?;
        let start_abs_usize =
            usize::try_from(start_abs).map_err(|_| SquashfsError::SizeOverflow)?;
        let mut cursor = Cursor::new(
            &self.partition[..self.squashfs_len],
            &self.superblock,
            start_abs_usize,
        )?;
        // Position cursor at `body.offset` bytes within the first block.
        cursor.seek_to(start_abs_usize, body.offset as usize)?;
        parse_directory(&mut cursor, body.size)
    }

    fn open_inode_ref(&self, inode_ref: u64) -> Result<Inode, SquashfsError> {
        let mut cursor = Cursor::new(
            &self.partition[..self.squashfs_len],
            &self.superblock,
            usize::try_from(self.superblock.inode_table_start)
                .map_err(|_| SquashfsError::SizeOverflow)?,
        )?;
        cursor.seek_ref(self.superblock.inode_table_start, inode_ref)?;
        let header = InodeHeader::read(&mut cursor)?;
        let kind = InodeKind::from_raw(header.kind);
        let (size, body) = match header.kind {
            INODE_KIND_BASIC_DIR => {
                let d = BasicDirInode::read(&mut cursor)?;
                (
                    d.size as u64,
                    InodeBody::Dir(DirBody {
                        start_block: d.start_block,
                        size: d.size,
                        offset: d.offset,
                    }),
                )
            }
            INODE_KIND_EXT_DIR => {
                let d = ExtDirInode::read(&mut cursor)?;
                (
                    d.size as u64,
                    InodeBody::Dir(DirBody {
                        start_block: d.start_block,
                        size: d.size,
                        offset: d.offset,
                    }),
                )
            }
            INODE_KIND_BASIC_FILE => {
                let f = BasicFileInode::read(&mut cursor, &self.superblock)?;
                (
                    f.file_size as u64,
                    InodeBody::File(FileBody {
                        blocks_start: f.blocks_start as u64,
                        file_size: f.file_size as u64,
                        block_sizes: f.block_sizes,
                        frag_idx: f.frag_idx,
                        frag_offset: f.frag_offset,
                    }),
                )
            }
            INODE_KIND_EXT_FILE => {
                let f = ExtFileInode::read(&mut cursor, &self.superblock)?;
                (
                    f.file_size,
                    InodeBody::File(FileBody {
                        blocks_start: f.blocks_start,
                        file_size: f.file_size,
                        block_sizes: f.block_sizes,
                        frag_idx: f.frag_idx,
                        frag_offset: f.frag_offset,
                    }),
                )
            }
            INODE_KIND_BASIC_SYMLINK => {
                let s = BasicSymlinkInode::read(&mut cursor)?;
                (s.target.len() as u64, InodeBody::Symlink(s.target))
            }
            INODE_KIND_EXT_SYMLINK => {
                // Ext symlink is structurally the same as Basic for our
                // purposes (we only need `target`); skip xattr_idx field
                // by reading and discarding.
                let s = BasicSymlinkInode::read(&mut cursor)?;
                let _xattr_idx = cursor.read_u32().ok();
                (s.target.len() as u64, InodeBody::Symlink(s.target))
            }
            other => return Err(SquashfsError::UnknownInodeKind(other)),
        };
        Ok(Inode {
            kind,
            inode_number: header.inode_number,
            size,
            body,
        })
    }
}
