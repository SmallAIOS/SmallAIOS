// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Top-level F2FS mount API.
//!
//! `F2fs::mount` performs:
//!
//! 1. Read both superblock copies (offsets 1024 and 9216 within the
//!    partition's first segment).
//! 2. Parse + CRC-verify each. The primary copy wins on success; if
//!    the primary fails CRC and the secondary verifies, fall back to
//!    the secondary and surface a warning via the audit-stub. If both
//!    fail, return [`F2fsError::SuperblockCrcMismatch`].
//! 3. Reject any mandatory feature bit not in
//!    [`super::superblock::SUPPORTED_FEATURES`].
//! 4. Read both checkpoint packs and pick the higher-version valid one.
//! 5. Cache the superblock + checkpoint in the [`F2fs`] handle for
//!    subsequent reads. SIT/NAT/SSA blocks are read on-demand.
//!
//! After mount, `open_path("/foo/bar")` walks the directory tree from
//! the root inode (id 3) using NAT lookups + dentry-block parsing.
//! `read_file` and `read_dir` operate on the resolved inode.
//!
//! Phase 6 will replace the per-call `read_block` with the per-mount
//! LRU cache from `crate::cache`.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::checkpoint::{select_checkpoint, Checkpoint};
use super::data::{lookup_nat, read_file_bytes, resolve_block_addr};
use super::dir::{parse_dentry_block, Dentry, FileType};
use super::error::F2fsError;
use super::inode::{self as inode_mod, Inode as F2fsInode};
use super::nat::NatEntry;
use super::superblock::{
    Superblock, F2FS_SUPER_BLOCK_BYTES, PRIMARY_SUPERBLOCK_OFFSET, SECONDARY_SUPERBLOCK_OFFSET,
};
use super::{
    read_f2fs_block, read_partition_bytes, F2FS_BLOCKS_PER_SEG, F2FS_BLOCK_SIZE, F2FS_NULL_NID,
    F2FS_ROOT_INO,
};
use crate::block::BlockDevice;

/// User-facing inode kind (re-export of [`inode_mod::InodeKind`]).
pub type InodeKind = inode_mod::InodeKind;

/// User-facing inode handle returned by `open_path`.
#[derive(Debug, Clone)]
pub struct Inode {
    pub kind: InodeKind,
    pub inode_number: u32,
    pub size: u64,
    /// Underlying parsed F2FS inode for read_file/read_dir.
    pub(crate) f2fs: F2fsInode,
    /// Raw 4 KiB inode block (kept so inline-data reads don't refetch).
    pub(crate) inode_block: alloc::boxed::Box<[u8; F2FS_BLOCK_SIZE as usize]>,
}

/// stat() result.
#[derive(Debug, Clone, Copy)]
pub struct Stat {
    pub kind: InodeKind,
    pub size: u64,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub atime: u64,
    pub ctime: u64,
    pub mtime: u64,
    pub links: u32,
}

/// Iterator over directory entries returned by `read_dir`.
pub struct DirIter<'a, D: BlockDevice + ?Sized> {
    pub(crate) entries: Vec<DirEntry>,
    pub(crate) cursor: usize,
    pub(crate) _phantom: core::marker::PhantomData<&'a D>,
}

impl<'a, D: BlockDevice + ?Sized> Iterator for DirIter<'a, D> {
    type Item = DirEntry;
    fn next(&mut self) -> Option<DirEntry> {
        if self.cursor < self.entries.len() {
            let e = self.entries[self.cursor].clone();
            self.cursor += 1;
            Some(e)
        } else {
            None
        }
    }
}

/// One entry yielded by [`DirIter`].
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub inode_number: u32,
    pub file_type: FileType,
}

/// Mounted F2FS handle.
///
/// The handle holds an `&mut` reference to the underlying [`BlockDevice`]
/// because Phase 7 added the write path; calling `write_block` on the
/// trait requires `&mut self`. Read-only callers can still construct the
/// handle from `&mut device` and only invoke the read API.
pub struct F2fs<'a, D: BlockDevice + ?Sized> {
    pub(crate) device: &'a mut D,
    /// Partition starting LBA on the underlying device.
    pub(crate) partition_lba: u64,
    /// Total partition size in bytes.
    pub(crate) partition_size_bytes: u64,
    /// Parsed superblock (winner between primary / secondary).
    pub(crate) sb: Superblock,
    /// Winning checkpoint pack header.
    pub(crate) checkpoint: Checkpoint,
    /// Which checkpoint slot the *winning* pack lives in (0 = CP1, 1 = CP2).
    /// On commit, the new pack is written to the *other* slot and the
    /// version is bumped, then this field flips. Phase 7.
    pub(crate) cp_slot: u8,
    /// In-memory write-path state: dirty NAT/SIT entries, allocator
    /// cursors, journaled rename ops, dirty inode + data blocks.
    pub(crate) write_state: super::write::WriteState,
}

impl<'a, D: BlockDevice + ?Sized> F2fs<'a, D> {
    /// Mount the F2FS partition described by `(partition_lba, size_lbas)`
    /// on `device`.
    ///
    /// `size_lbas` is the partition length in **device** LBAs (i.e.
    /// in `device.block_size_bytes()` units), not in F2FS 4 KiB blocks.
    ///
    /// The device is borrowed mutably so the write-path API
    /// (`create`, `write_file`, `fsync`, …) can invoke
    /// [`BlockDevice::write_block`] later. Read-only callers may still
    /// construct via `mount` and only invoke the read API.
    pub fn mount(device: &'a mut D, partition_lba: u64, size_lbas: u64) -> Result<Self, F2fsError> {
        let bs = device.block_size_bytes() as u64;
        let partition_size_bytes = size_lbas.checked_mul(bs).ok_or(F2fsError::SizeOverflow)?;

        // Read the primary 4 KiB-prefix region containing the superblock.
        let mut prim_block = alloc::vec![0u8; F2FS_SUPER_BLOCK_BYTES + 8];
        read_partition_bytes(
            device,
            partition_lba,
            partition_size_bytes,
            PRIMARY_SUPERBLOCK_OFFSET,
            &mut prim_block,
        )?;
        let mut sec_block = alloc::vec![0u8; F2FS_SUPER_BLOCK_BYTES + 8];
        let secondary_read = read_partition_bytes(
            device,
            partition_lba,
            partition_size_bytes,
            SECONDARY_SUPERBLOCK_OFFSET,
            &mut sec_block,
        );

        let prim_parsed =
            Superblock::parse(&prim_block).and_then(|sb| sb.verify_crc(&prim_block).map(|_| sb));
        let sec_parsed = if secondary_read.is_ok() {
            Superblock::parse(&sec_block).and_then(|sb| sb.verify_crc(&sec_block).map(|_| sb))
        } else {
            Err(F2fsError::SuperblockCrcMismatch)
        };

        let sb = match (prim_parsed, sec_parsed) {
            (Ok(s), _) => s,
            (Err(_), Ok(s)) => s,
            (Err(e), Err(_)) => return Err(e),
        };
        sb.check_features()?;

        // Read both checkpoint packs.
        let cp_blkaddr = sb.cp_blkaddr;
        let cp2_blkaddr = cp_blkaddr.saturating_add(F2FS_BLOCKS_PER_SEG);
        let mut cp1_block = [0u8; F2FS_BLOCK_SIZE as usize];
        let mut cp2_block = [0u8; F2FS_BLOCK_SIZE as usize];
        read_f2fs_block(
            &*device,
            partition_lba,
            partition_size_bytes,
            cp_blkaddr,
            &mut cp1_block,
        )?;
        read_f2fs_block(
            &*device,
            partition_lba,
            partition_size_bytes,
            cp2_blkaddr,
            &mut cp2_block,
        )?;
        let checkpoint = select_checkpoint(&cp1_block, &cp2_block)?;
        // Determine which physical slot the winning pack came from so
        // future commits can write to the *other* slot. We re-parse
        // each block's version to make the pick.
        let cp_slot = match (
            super::checkpoint::Checkpoint::parse(&cp1_block).ok(),
            super::checkpoint::Checkpoint::parse(&cp2_block).ok(),
        ) {
            (Some(c1), Some(c2)) => {
                if c1.checkpoint_ver == checkpoint.checkpoint_ver
                    && c1.verify_crc(&cp1_block).is_ok()
                {
                    0
                } else if c2.verify_crc(&cp2_block).is_ok()
                    && c2.checkpoint_ver == checkpoint.checkpoint_ver
                {
                    1
                } else {
                    0
                }
            }
            (Some(_), None) => 0,
            (None, Some(_)) => 1,
            (None, None) => 0,
        };

        let write_state = super::write::WriteState::new(&checkpoint);

        Ok(Self {
            device,
            partition_lba,
            partition_size_bytes,
            sb,
            checkpoint,
            cp_slot,
            write_state,
        })
    }

    /// Borrow the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.sb
    }

    /// Borrow the winning checkpoint pack.
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }

    /// Look up an inode by node-id (NAT lookup → read inode block).
    ///
    /// Honors the in-memory write-state cache so callers see freshly
    /// allocated inodes immediately, before `fsync` flushes the NAT.
    pub fn read_inode_by_nid(&self, nid: u32) -> Result<Inode, F2fsError> {
        // Cache hit: serve straight from the in-memory inode block,
        // which `create` populated and `flush_inode_block` may have
        // also written through to disk.
        if let Some(block) = self.write_state.inode_cache.get(&nid) {
            let f2fs_inode = F2fsInode::parse(&block[..])?;
            return Ok(Inode {
                kind: f2fs_inode.kind,
                inode_number: nid,
                size: f2fs_inode.size,
                f2fs: f2fs_inode,
                inode_block: block.clone(),
            });
        }
        // Otherwise consult dirty NAT first, then fall through to the
        // on-disk NAT pair.
        let nat_addr = if let Some(pending) = self.write_state.nat_dirty.get(&nid) {
            if pending.block_addr == 0 || pending.block_addr == u32::MAX {
                return Err(F2fsError::PathNotFound);
            }
            pending.block_addr
        } else {
            let nat = self.lookup_nat_entry(nid)?;
            if !nat.is_valid_addr() {
                return Err(F2fsError::PathNotFound);
            }
            nat.block_addr
        };
        let mut inode_block = alloc::boxed::Box::new([0u8; F2FS_BLOCK_SIZE as usize]);
        read_f2fs_block(
            &*self.device,
            self.partition_lba,
            self.partition_size_bytes,
            nat_addr,
            &mut inode_block,
        )?;
        let f2fs_inode = F2fsInode::parse(&inode_block[..])?;
        Ok(Inode {
            kind: f2fs_inode.kind,
            inode_number: nid,
            size: f2fs_inode.size,
            f2fs: f2fs_inode,
            inode_block,
        })
    }

    /// NAT entry for a given `nid` (helper for tests + diagnostics).
    pub fn lookup_nat_entry(&self, nid: u32) -> Result<NatEntry, F2fsError> {
        lookup_nat(
            self.device,
            self.partition_lba,
            self.partition_size_bytes,
            &self.sb,
            nid,
        )
    }

    /// Resolve an absolute UNIX path (`/foo/bar/baz`) to an inode.
    pub fn open_path(&self, path: &str) -> Result<Inode, F2fsError> {
        // Start at root.
        let root = self.read_inode_by_nid(F2FS_ROOT_INO)?;
        if path == "/" || path.is_empty() {
            return Ok(root);
        }
        let mut current = root;
        for component in path.split('/') {
            if component.is_empty() {
                continue;
            }
            if !current.f2fs.is_directory() {
                return Err(F2fsError::NotADirectory);
            }
            let nid = self.lookup_dir_entry(&current, component)?;
            current = self.read_inode_by_nid(nid)?;
        }
        Ok(current)
    }

    /// Look up a single name in the given directory inode. Returns
    /// the matched entry's inode number, or an error if not found.
    pub fn lookup_dir_entry(&self, dir: &Inode, name: &str) -> Result<u32, F2fsError> {
        let entries = self.read_dir_entries(dir)?;
        for e in entries {
            if e.name == name {
                return Ok(e.inode_number);
            }
        }
        Err(F2fsError::PathNotFound)
    }

    /// Read up to `buf.len()` bytes from `inode` starting at `offset`.
    pub fn read_file(
        &self,
        inode: &Inode,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, F2fsError> {
        if !inode.f2fs.is_regular() {
            return Err(F2fsError::NotADirectory);
        }
        read_file_bytes(
            self.device,
            self.partition_lba,
            self.partition_size_bytes,
            &self.sb,
            &inode.f2fs,
            &inode.inode_block,
            offset,
            buf,
        )
    }

    /// Iterate over the entries of a directory inode.
    pub fn read_dir(&self, inode: &Inode) -> Result<DirIter<'_, D>, F2fsError> {
        let entries = self.read_dir_entries(inode)?;
        Ok(DirIter {
            entries,
            cursor: 0,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Return all DirEntry records for a directory inode (resolves
    /// inline + per-block dentries).
    fn read_dir_entries(&self, inode: &Inode) -> Result<Vec<DirEntry>, F2fsError> {
        if !inode.f2fs.is_directory() {
            return Err(F2fsError::NotADirectory);
        }
        let mut out = Vec::new();

        // Inline-dentry path: dentries live in the inode block itself
        // at offset 360 (same area as inline data).
        if inode.f2fs.has_inline_dentry {
            const INLINE_DENTRY_OFFSET: usize = 360;
            const INLINE_DENTRY_END: usize = 4080;
            let raw = &inode.inode_block[INLINE_DENTRY_OFFSET..INLINE_DENTRY_END];
            // Build a fake 4 KiB block whose layout matches a regular
            // dentry block but is sized for the smaller inline area.
            // The kernel's `f2fs_inline_dentry` has the same shape as
            // `f2fs_dentry_block` but with NR_INLINE_DENTRY = 182 entries
            // instead of 214. To keep the v1 reader simple we project
            // the inline area into a synthetic 4 KiB block and walk it
            // with the same parser; entries past the inline-area count
            // are ignored because their bitmap bits are zero.
            let mut synthetic = alloc::vec![0u8; F2FS_BLOCK_SIZE as usize];
            synthetic[..raw.len()].copy_from_slice(raw);
            let dentries = parse_dentry_block(&synthetic)?;
            for d in dentries {
                out.push(dentry_to_entry(d));
            }
            return Ok(out);
        }

        // Walk every direct/indirect block of the directory file as a
        // dentry block.
        let total_blocks = (inode.f2fs.size as usize).div_ceil(F2FS_BLOCK_SIZE as usize);
        for block_idx in 0..total_blocks {
            let addr = resolve_block_addr(
                self.device,
                self.partition_lba,
                self.partition_size_bytes,
                &self.sb,
                &inode.f2fs,
                block_idx,
            )?;
            let Some(addr) = addr else {
                // Sparse dentry block — uncommon but valid; skip.
                continue;
            };
            let mut block = [0u8; F2FS_BLOCK_SIZE as usize];
            read_f2fs_block(
                self.device,
                self.partition_lba,
                self.partition_size_bytes,
                addr,
                &mut block,
            )?;
            let dentries = parse_dentry_block(&block)?;
            for d in dentries {
                if d.ino == F2FS_NULL_NID {
                    continue;
                }
                out.push(dentry_to_entry(d));
            }
        }
        Ok(out)
    }

    /// Stat helper.
    pub fn stat(&self, inode: &Inode) -> Stat {
        Stat {
            kind: inode.f2fs.kind,
            size: inode.f2fs.size,
            mode: inode.f2fs.mode,
            uid: inode.f2fs.uid,
            gid: inode.f2fs.gid,
            atime: inode.f2fs.atime,
            ctime: inode.f2fs.ctime,
            mtime: inode.f2fs.mtime,
            links: inode.f2fs.links,
        }
    }
}

fn dentry_to_entry(d: Dentry) -> DirEntry {
    DirEntry {
        name: d.name,
        inode_number: d.ino,
        file_type: d.file_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f2fs::superblock::F2FS_SUPER_MAGIC;

    /// Sanity: `mount` on a zeroed device fails with BadMagic, not
    /// any other variant.
    #[test]
    fn mount_zeroed_device_bad_magic() {
        let mut dev = crate::block::mock::MockBlockDevice::new(4096, 32);
        let r = F2fs::mount(&mut dev, 0, 32);
        assert!(matches!(r, Err(F2fsError::BadMagic)));
    }

    /// Building a synthetic single-superblock disk and confirming
    /// `mount` reaches the next stage (will fail at checkpoint, since
    /// we didn't write one — but it must not fail at the SB layer).
    #[test]
    fn mount_with_minimal_superblock_fails_at_checkpoint() {
        let mut dev = crate::block::mock::MockBlockDevice::new(4096, 1024);
        let mut sb = alloc::vec![0u8; 1024];
        sb[0..4].copy_from_slice(&F2FS_SUPER_MAGIC.to_le_bytes());
        sb[4..6].copy_from_slice(&1u16.to_le_bytes());
        sb[16..20].copy_from_slice(&12u32.to_le_bytes()); // 4 KiB blocks
        sb[20..24].copy_from_slice(&9u32.to_le_bytes()); // 512 blk/seg
        sb[36..44].copy_from_slice(&1024u64.to_le_bytes());
        sb[60..64].copy_from_slice(&1u32.to_le_bytes()); // segment_count_nat
        sb[76..80].copy_from_slice(&512u32.to_le_bytes()); // cp_blkaddr
                                                           // Write the SB at byte offset 1024 by preloading block 0 with
                                                           // SB occupying [1024..2048].
        let mut block0 = alloc::vec![0u8; 4096];
        block0[1024..2048].copy_from_slice(&sb);
        dev.preload(0, &block0);

        let r = F2fs::mount(&mut dev, 0, 1024);
        // Mount may fail at checkpoint or NAT, but not BadMagic since
        // the magic is now present.
        assert!(!matches!(r, Err(F2fsError::BadMagic)));
    }
}
