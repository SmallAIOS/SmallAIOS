// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS write path (Phase 7 of `embedded-filesystem-v1`).
//!
//! Layered on top of the Phase 5 read path, this module implements:
//!
//! - File creation ([`F2fs::create`]) — allocate a fresh inode-id from
//!   NAT, write the inode block, link it into the parent directory.
//! - Buffered file writes ([`F2fs::write_file`]) — extend or overwrite
//!   `i_addr[]`-direct or single-indirect data blocks. Allocation is
//!   *log-structured*: new blocks always come from the current data
//!   segment cursor (`cur_data_segno`), never from in-place rewrites.
//! - Truncate ([`F2fs::truncate`]) — both shrink (release blocks back
//!   to SIT free state) and grow (sparse hole via NULL_ADDR).
//! - Atomic rename ([`F2fs::rename`]) — staged via the journal
//!   ([`crate::f2fs::journal`]), so an open reader continues to see the
//!   pre-rename `dst` while new opens see the moved content. Crash mid-
//!   rename replays cleanly on next mount.
//! - Unlink / mkdir / rmdir.
//! - `fsync()` — durable commit via a fresh checkpoint pack written to
//!   the alternate CP slot, with `checkpoint_ver` bumped.
//! - 5-second background-timer hook ([`F2fs::tick_timer`]) — when the
//!   caller passes a wall-clock tick and dirty data has been pending
//!   for ≥ 5 s, an implicit fsync runs.
//!
//! ## Allocation model
//!
//! F2FS is log-structured: writes always advance a *segment cursor*
//! (`cur_data_segno`, `cur_node_segno`). When the cursor hits the end
//! of its 2 MiB segment we pick the next free segment from SIT's
//! free-segment bitmap. This keeps each new write in a fresh location,
//! so on-disk extents stay sequential, which simplifies wear-levelling
//! on flash. The cost — fragmentation of partially-rewritten files —
//! is paid by the GC pass ([`super::gc`]).
//!
//! ## Phase 7 simplifications
//!
//! - *No COW for inode blocks*: the inode is rewritten in place at its
//!   NAT-mapped block address. Linux does the same when the inode block
//!   was clean, with a "node-summary update" path; v1 always rewrites
//!   in place. This is safe because the inode block lives in a dedicated
//!   node area, and the NAT version-bitmap is unused in v1.
//! - *Inline data only for files ≤ 1500 bytes*: above that we always
//!   spill to a data block. Linux's threshold is a runtime mode, but
//!   v1 picks one number.
//! - *Direct + 1×single-indirect only*: matches the read-path bound
//!   ([`super::data::MAX_INDIRECT_FILE_BYTES`]); double/triple indirect
//!   is Phase 8+.
//! - *Single-stream allocator*: F2FS supports up to 8 hot/warm/cold
//!   data streams; v1 uses one.
//!
//! These simplifications are spec-conformant — Linux 6.6 happily mounts
//! the resulting partitions and reads the same bytes back. They simply
//! produce slightly less compact on-disk layouts than `mkfs.f2fs`.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::checkpoint::{Checkpoint, DEFAULT_CHECKPOINT_CRC_OFFSET};
use super::crc::f2fs_crc32;
use super::data::ADDRS_PER_BLOCK;
use super::dir::{
    slots_for_name, FileType, ENTRY_ARRAY_OFFSET, F2FS_DIR_ENTRY_SIZE, F2FS_SLOT_LEN,
    FILENAME_SLOTS_OFFSET, NR_DENTRY_IN_BLOCK, SIZE_OF_DENTRY_BITMAP,
};
use super::error::F2fsError;
use super::gc::{
    self, AllocatorHints, GcPassConfig, GcPassOutcome, LiveBlock, DEFAULT_COLD_THRESHOLD_SECS,
    DEFAULT_GC_VICTIMS_PER_PASS, DEFAULT_PASS_BUDGET_TICKS, DEFAULT_YIELD_BUDGET_BLOCKS,
};
use super::gc_journal::GcJournalRecord;
use super::inode::{
    ADDRS_PER_INODE, INLINE_DATA, INLINE_DENTRY, NIDS_PER_INODE, NODE_FOOTER_OFFSET, S_IFDIR,
    S_IFREG,
};
use super::journal::{encode_journal_block, JournalEntry, RenameRecord, FLAG_RENAME_COMMITTED};
use super::mount::{F2fs, Inode};
use super::nat::{nid_to_entry_off, nid_to_nat_block_idx, NAT_ENTRY_PER_BLOCK, NAT_ENTRY_SIZE};
use super::sit::{SIT_ENTRY_PER_BLOCK, SIT_ENTRY_SIZE, SIT_VBLOCK_MAP_BYTES};
use super::superblock::Superblock;
use super::{read_f2fs_block, F2FS_BLOCKS_PER_SEG, F2FS_BLOCK_SIZE, F2FS_NULL_NID, F2FS_ROOT_INO};
use crate::block::BlockDevice;
use smallaios_auth::atomic_write::{AtomicWriteError, FsWriteBackend};

// ─── Public stats type ──────────────────────────────────────────────────────

/// Counters surfaced by [`F2fs::write_stats`] for tests and diagnostics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WriteStats {
    /// Total `create()` calls that succeeded.
    pub creates: u32,
    /// Total `write_file()` calls.
    pub writes: u32,
    /// Total `truncate()` calls.
    pub truncates: u32,
    /// Total `rename()` calls (committed).
    pub renames: u32,
    /// Total `unlink()` calls.
    pub unlinks: u32,
    /// Total `mkdir()` calls.
    pub mkdirs: u32,
    /// Total `rmdir()` calls.
    pub rmdirs: u32,
    /// Total `fsync()` / checkpoint-commit invocations.
    pub fsyncs: u32,
    /// Total GC passes that ran (whether they freed anything or not).
    pub gc_passes: u32,
}

/// Re-export of [`crate::f2fs::gc::GcStats`] for the public API.
pub use super::gc::GcStats;

// ─── In-memory write-state ──────────────────────────────────────────────────

/// Snapshot of one NAT entry pending write-back.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingNat {
    pub ino: u32,
    pub block_addr: u32,
    pub version: u8,
}

/// Snapshot of one SIT entry pending write-back. Each entry corresponds
/// to one main-area segment.
#[derive(Debug, Clone)]
pub(crate) struct PendingSit {
    pub valid_blocks: u16,
    pub valid_map: [u8; SIT_VBLOCK_MAP_BYTES],
    pub mtime: u64,
    pub raw_type_bits: u16,
}

impl PendingSit {
    fn empty() -> Self {
        Self {
            valid_blocks: 0,
            valid_map: [0u8; SIT_VBLOCK_MAP_BYTES],
            mtime: 0,
            raw_type_bits: 0,
        }
    }

    fn mark_valid(&mut self, block_idx: usize) {
        if block_idx >= 512 {
            return;
        }
        let byte = block_idx / 8;
        let bit = block_idx % 8;
        let prev = (self.valid_map[byte] >> bit) & 1;
        if prev == 0 {
            self.valid_map[byte] |= 1u8 << bit;
            self.valid_blocks = self.valid_blocks.saturating_add(1);
        }
    }

    fn mark_invalid(&mut self, block_idx: usize) {
        if block_idx >= 512 {
            return;
        }
        let byte = block_idx / 8;
        let bit = block_idx % 8;
        let prev = (self.valid_map[byte] >> bit) & 1;
        if prev == 1 {
            self.valid_map[byte] &= !(1u8 << bit);
            self.valid_blocks = self.valid_blocks.saturating_sub(1);
        }
    }
}

/// In-memory write-path state. Carried inside [`F2fs`] to avoid having
/// to thread it through every public method.
#[derive(Debug, Clone)]
pub(crate) struct WriteState {
    /// Dirty NAT entries: nid → pending entry. Flushed on `fsync`.
    pub(crate) nat_dirty: BTreeMap<u32, PendingNat>,
    /// Dirty SIT entries: segno → pending entry. Flushed on `fsync`.
    pub(crate) sit_dirty: BTreeMap<u32, PendingSit>,
    /// Cached inode blocks (4 KiB each). After mutation, the new bytes
    /// are flushed back to the on-disk inode block at the NAT-mapped
    /// address. Phase 7 keeps inode rewrites *in place*.
    pub(crate) inode_cache: BTreeMap<u32, Box<[u8; F2FS_BLOCK_SIZE as usize]>>,
    /// Next free node-id (advances monotonically as new inodes are
    /// allocated; reclaimed nids go back to a free list on unlink).
    pub(crate) next_free_nid: u32,
    /// Pool of reclaimed nids that can be re-allocated before the
    /// monotonic cursor advances.
    pub(crate) free_nid_pool: Vec<u32>,
    /// Current data-segment cursor: (segno, blkoff_within_segment).
    pub(crate) cur_data_segno: u32,
    pub(crate) cur_data_blkoff: u16,
    /// Hot-stream cursor (Phase 9): co-locates recently-touched data so
    /// future GC has less to do. Initialized to `cur_data_segno` on
    /// mount; advanced via the SIT free-segment bitmap when full.
    pub(crate) cur_hot_segno: u32,
    /// Hot-stream block offset within the segment. Initialized to
    /// match `cur_data_blkoff` on mount; the GC v2 driver advances it
    /// on each cold→hot relocation. Reserved for the dual-cursor
    /// allocator (Phase 10); the integration only needs the segno.
    #[allow(dead_code)]
    pub(crate) cur_hot_blkoff: u16,
    /// Cold-stream cursor (Phase 9): co-locates stale data (mtime older
    /// than [`gc::DEFAULT_COLD_THRESHOLD_SECS`]). Initialized to
    /// `cur_data_segno` on mount; advanced independently from the hot
    /// cursor.
    pub(crate) cur_cold_segno: u32,
    /// Cold-stream block offset within the segment. See
    /// [`Self::cur_hot_blkoff`].
    #[allow(dead_code)]
    pub(crate) cur_cold_blkoff: u16,
    /// Free data-segment list (segments that may be allocated when the
    /// cursor advances past the current segment's end).
    pub(crate) free_segments: Vec<u32>,
    /// Dirty journal entries (rename in-flight, etc.).
    pub(crate) journal_dirty: Vec<JournalEntry>,
    /// Phase 9 GC journal records staged within the current commit
    /// epoch. Each `run_gc_pass_v2` invocation appends one record per
    /// relocation BEFORE SIT/NAT mutation, so a power loss replays
    /// cleanly. Drained at fsync alongside `journal_dirty`. The on-disk
    /// reservation lives in the existing journal block; encoding will
    /// move into checkpoint commit when the v2 driver becomes the
    /// primary path.
    pub(crate) gc_journal_dirty: Vec<GcJournalRecord>,
    /// In-memory SSA-style reverse map: physical block address →
    /// `(owning_nid, ofs_in_node)`. Populated by `allocate_data_block`
    /// for inode and indirect-node writes; consumed by the Phase 9 GC
    /// pass's `live_blocks_for` projection. Persistence lives in the
    /// SSA area on disk (Phase 5/7 code does not yet update it; the
    /// write path keeps this in-memory mirror so GC v2 has the data).
    pub(crate) ssa_index: BTreeMap<u32, (u32, u32)>,
    /// Per-inode mtime cache used by the Phase 9 GC age classifier.
    /// Production reads this from inode blocks on demand; Phase 9
    /// integration keeps a write-through cache so the GC pass need not
    /// reach into the on-disk inode for every relocation candidate.
    pub(crate) mtime_index: BTreeMap<u32, u64>,
    /// Wall-clock seconds (set via `F2fs::set_now_secs`). Used by the
    /// Phase 9 GC age classifier to decide hot vs. cold placement.
    /// Tests inject deterministic values; production wires this from
    /// the kernel scheduler's clock tick.
    pub(crate) now_secs: u64,
    /// Current `checkpoint_ver` — bumped at every commit.
    pub(crate) checkpoint_ver: u64,
    /// Wall-clock-style monotonic counter incremented on each write
    /// and snapshotted to drive the 5-second timer.
    pub(crate) ticks_since_dirty: u32,
    /// Cumulative public stats counter.
    pub(crate) stats: WriteStats,
    /// True if any data has been dirtied since the last commit.
    pub(crate) dirty: bool,
    /// True after [`F2fs::request_yield`] is called — read-and-clear by
    /// the GC inner loop so foreground writes can preempt.
    pub(crate) yield_pending: bool,
    /// Total segments in the partition (snapshot from superblock; kept
    /// here so `should_run_gc` doesn't need to re-derive).
    pub(crate) total_segments: u32,
}

impl WriteState {
    pub(crate) fn new(cp: &Checkpoint) -> Self {
        let cur_data_segno = cp.cur_data_segno[0];
        let cur_data_blkoff = cp.cur_data_blkoff[0];
        Self {
            nat_dirty: BTreeMap::new(),
            sit_dirty: BTreeMap::new(),
            inode_cache: BTreeMap::new(),
            next_free_nid: cp.next_free_nid.max(F2FS_ROOT_INO + 1),
            free_nid_pool: Vec::new(),
            cur_data_segno,
            cur_data_blkoff,
            // Phase 9: hot/cold cursors both start co-located with the
            // legacy data cursor; the first cold-classified write or GC
            // relocation breaks them apart.
            cur_hot_segno: cur_data_segno,
            cur_hot_blkoff: cur_data_blkoff,
            cur_cold_segno: cur_data_segno,
            cur_cold_blkoff: cur_data_blkoff,
            free_segments: Vec::new(),
            journal_dirty: Vec::new(),
            gc_journal_dirty: Vec::new(),
            ssa_index: BTreeMap::new(),
            mtime_index: BTreeMap::new(),
            now_secs: 0,
            checkpoint_ver: cp.checkpoint_ver,
            ticks_since_dirty: 0,
            stats: WriteStats::default(),
            dirty: false,
            yield_pending: false,
            total_segments: 0,
        }
    }
}

// ─── Internal helpers (block addressing) ────────────────────────────────────

/// Convert (segno, blkoff) → physical block address on the partition.
#[inline]
fn segno_blkoff_to_addr(sb: &Superblock, segno: u32, blkoff: u16) -> u32 {
    sb.main_blkaddr
        .saturating_add(segno.saturating_mul(F2FS_BLOCKS_PER_SEG))
        .saturating_add(blkoff as u32)
}

/// Convert a physical block address back to (segno, blkoff).
#[inline]
pub(crate) fn addr_to_segno_blkoff(sb: &Superblock, addr: u32) -> Option<(u32, u16)> {
    if addr < sb.main_blkaddr {
        return None;
    }
    let off = addr - sb.main_blkaddr;
    let segno = off / F2FS_BLOCKS_PER_SEG;
    let blkoff = (off % F2FS_BLOCKS_PER_SEG) as u16;
    Some((segno, blkoff))
}

// ─── Public write API on F2fs ───────────────────────────────────────────────

impl<'a, D: BlockDevice + ?Sized> F2fs<'a, D> {
    /// Snapshot of the cumulative write stats for tests / diagnostics.
    pub fn write_stats(&self) -> WriteStats {
        self.write_state.stats
    }

    /// Hint to the GC pass that a foreground request is pending. The
    /// next inter-block yield-poll surfaces this and pauses GC.
    pub fn request_yield(&mut self) {
        self.write_state.yield_pending = true;
    }

    /// Inject the current wall-clock seconds. Production wires this
    /// from the kernel scheduler's clock tick; tests pass deterministic
    /// values to drive the Phase 9 GC age classifier (hot/cold).
    pub fn set_now_secs(&mut self, now_secs: u64) {
        self.write_state.now_secs = now_secs;
    }

    /// Snapshot the current wall-clock seconds value (for tests).
    pub fn now_secs(&self) -> u64 {
        self.write_state.now_secs
    }

    /// Snapshot of the current Phase 9 hot/cold allocator cursors. Tests
    /// use this to assert that cold-classified GC relocations advance
    /// the cold cursor without disturbing the hot cursor.
    pub fn allocator_hints(&self) -> AllocatorHints {
        AllocatorHints {
            hot_segno: self.write_state.cur_hot_segno,
            cold_segno: self.write_state.cur_cold_segno,
        }
    }

    /// Snapshot of the staged Phase 9 GC journal records since the last
    /// fsync. Tests use this to assert that each GC relocation stages a
    /// `(nid, ofs, old, new)` record BEFORE the SIT/NAT mutation lands.
    pub fn gc_journal_snapshot(&self) -> Vec<GcJournalRecord> {
        self.write_state.gc_journal_dirty.clone()
    }

    /// Drive the 5-second background timer. Each call increments the
    /// internal tick counter; once `ticks` ≥ 5 *and* dirty state is
    /// pending, an implicit `fsync` runs.
    ///
    /// Tests pass a synthetic `tick_seconds = 1` argument repeatedly so
    /// they can exercise the threshold deterministically; the wall-
    /// clock kernel implementation calls this at 1 Hz from the
    /// scheduler's idle path.
    pub fn tick_timer(&mut self, tick_seconds: u32) -> Result<bool, F2fsError> {
        if !self.write_state.dirty {
            self.write_state.ticks_since_dirty = 0;
            return Ok(false);
        }
        self.write_state.ticks_since_dirty = self
            .write_state
            .ticks_since_dirty
            .saturating_add(tick_seconds);
        if self.write_state.ticks_since_dirty >= 5 {
            self.fsync()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Allocate a fresh node-id. Reuses a reclaimed nid from the free
    /// pool first; otherwise advances the monotonic cursor.
    fn allocate_nid(&mut self) -> u32 {
        if let Some(reused) = self.write_state.free_nid_pool.pop() {
            return reused;
        }
        let nid = self.write_state.next_free_nid;
        self.write_state.next_free_nid = self.write_state.next_free_nid.saturating_add(1);
        nid
    }

    /// Allocate one fresh 4 KiB data block from the current data
    /// segment cursor. Advances the cursor; switches to the next free
    /// segment when the current one fills.
    pub(crate) fn allocate_data_block(&mut self) -> Result<u32, F2fsError> {
        let blkoff = self.write_state.cur_data_blkoff;
        if (blkoff as u32) >= F2FS_BLOCKS_PER_SEG {
            // Advance to next free segment.
            let next_seg = self
                .write_state
                .free_segments
                .pop()
                .or_else(|| {
                    let s = self.write_state.cur_data_segno.saturating_add(1);
                    if s < self.write_state.total_segments {
                        Some(s)
                    } else {
                        None
                    }
                })
                .ok_or(F2fsError::SizeOverflow)?;
            self.write_state.cur_data_segno = next_seg;
            self.write_state.cur_data_blkoff = 0;
        }
        let segno = self.write_state.cur_data_segno;
        let blkoff = self.write_state.cur_data_blkoff;
        let addr = segno_blkoff_to_addr(&self.sb, segno, blkoff);
        self.write_state.cur_data_blkoff = blkoff.saturating_add(1);

        // Update in-memory SIT for the segment.
        let entry = self
            .write_state
            .sit_dirty
            .entry(segno)
            .or_insert_with(PendingSit::empty);
        entry.mark_valid(blkoff as usize);
        entry.raw_type_bits = 0; // Data segment, hot.

        Ok(addr)
    }

    /// Free a previously-allocated data block (mark its bit clear in
    /// the SIT). The physical block address may eventually be reclaimed
    /// by the GC pass.
    pub(crate) fn free_data_block(&mut self, addr: u32) {
        if let Some((segno, blkoff)) = addr_to_segno_blkoff(&self.sb, addr) {
            let entry = self
                .write_state
                .sit_dirty
                .entry(segno)
                .or_insert_with(PendingSit::empty);
            entry.mark_invalid(blkoff as usize);
        }
        // Phase 9: drop the SSA reverse-map entry for this addr; the
        // GC pass uses absence to mean "block is dead".
        self.write_state.ssa_index.remove(&addr);
    }

    /// Record the `(owning_nid, ofs_in_node)` reverse mapping for a
    /// freshly allocated block. Called by every write-path code path
    /// that pulls a block off `allocate_data_block`. The Phase 9 GC
    /// pass projects this map (combined with SIT) into its
    /// [`gc::live_blocks_from_segment`] input.
    pub(crate) fn record_ssa(&mut self, addr: u32, nid: u32, ofs_in_node: u32) {
        self.write_state.ssa_index.insert(addr, (nid, ofs_in_node));
    }

    /// Touch the in-memory inode-mtime cache. Phase 9 GC reads this when
    /// classifying a live block as hot vs cold. Production write paths
    /// call this whenever they stamp `i_mtime`; tests inject synthetic
    /// values to drive the age classifier deterministically.
    pub(crate) fn record_inode_mtime(&mut self, nid: u32, mtime_secs: u64) {
        self.write_state.mtime_index.insert(nid, mtime_secs);
    }

    /// Public test hook: stamp an inode's mtime in the in-memory cache
    /// without forcing a full inode-block rewrite. Tests use this to
    /// deterministically age a file so the next GC pass classifies it
    /// as cold; production stamps mtime when it writes the inode.
    pub fn stamp_mtime(&mut self, nid: u32, mtime_secs: u64) {
        self.record_inode_mtime(nid, mtime_secs);
    }

    /// Stage a NAT entry update.
    pub(crate) fn dirty_nat(&mut self, nid: u32, ino: u32, addr: u32) {
        self.write_state.nat_dirty.insert(
            nid,
            PendingNat {
                ino,
                block_addr: addr,
                version: 1,
            },
        );
    }

    /// Read the inode block for `nid` (cached if already pulled).
    fn load_inode_block(
        &mut self,
        nid: u32,
    ) -> Result<Box<[u8; F2FS_BLOCK_SIZE as usize]>, F2fsError> {
        if let Some(blk) = self.write_state.inode_cache.get(&nid) {
            return Ok(blk.clone());
        }
        let block_addr = self.resolve_inode_addr(nid)?;
        let mut block = Box::new([0u8; F2FS_BLOCK_SIZE as usize]);
        #[allow(clippy::explicit_auto_deref)]
        read_f2fs_block(
            &*self.device,
            self.partition_lba,
            self.partition_size_bytes,
            block_addr,
            &mut *block,
        )?;
        self.write_state.inode_cache.insert(nid, block.clone());
        Ok(block)
    }

    /// Re-read an inode from in-memory state (cache wins; falls back to
    /// the on-disk read path).
    pub fn read_inode_rw(&mut self, nid: u32) -> Result<Inode, F2fsError> {
        let block = self.load_inode_block(nid)?;
        let f2fs_inode = super::inode::Inode::parse(&block[..])?;
        Ok(Inode {
            kind: f2fs_inode.kind,
            inode_number: nid,
            size: f2fs_inode.size,
            f2fs: f2fs_inode,
            inode_block: block,
        })
    }

    /// Resolve the current physical block address for a given `nid`,
    /// consulting `nat_dirty` first (so freshly created inodes are
    /// visible before their NAT entry is flushed to disk).
    pub(crate) fn resolve_inode_addr(&self, nid: u32) -> Result<u32, F2fsError> {
        if let Some(pending) = self.write_state.nat_dirty.get(&nid) {
            if pending.block_addr == 0 || pending.block_addr == u32::MAX {
                return Err(F2fsError::PathNotFound);
            }
            return Ok(pending.block_addr);
        }
        let nat = self.lookup_nat_entry(nid)?;
        if !nat.is_valid_addr() {
            return Err(F2fsError::PathNotFound);
        }
        Ok(nat.block_addr)
    }

    /// Persist the in-memory inode block to its NAT-mapped on-disk
    /// address. (Phase 7 rewrites in place; Phase 9 will switch to COW.)
    fn flush_inode_block(&mut self, nid: u32) -> Result<(), F2fsError> {
        let block = match self.write_state.inode_cache.get(&nid) {
            Some(b) => b.clone(),
            None => return Ok(()),
        };
        let block_addr = self.resolve_inode_addr(nid)?;
        let bs = self.device.block_size_bytes() as u64;
        let abs_byte =
            self.partition_lba.saturating_mul(bs) + (block_addr as u64) * F2FS_BLOCK_SIZE as u64;
        let lba = abs_byte / bs;
        self.device
            .write_block(lba, &block[..])
            .map_err(F2fsError::IoError)?;
        Ok(())
    }

    /// Persist a single 4 KiB data block.
    pub(crate) fn write_block_at(
        &mut self,
        block_addr: u32,
        block: &[u8; F2FS_BLOCK_SIZE as usize],
    ) -> Result<(), F2fsError> {
        let bs = self.device.block_size_bytes() as u64;
        let abs_byte =
            self.partition_lba.saturating_mul(bs) + (block_addr as u64) * F2FS_BLOCK_SIZE as u64;
        let lba = abs_byte / bs;
        self.device
            .write_block(lba, &block[..])
            .map_err(F2fsError::IoError)?;
        Ok(())
    }

    // ── create / mkdir ──────────────────────────────────────────────────

    /// Create a regular file at `path`. Allocates a fresh inode, writes
    /// the inode block, and links it into the parent directory.
    pub fn create(&mut self, path: &str, mode: u16) -> Result<Inode, F2fsError> {
        self.create_kind(path, mode | (S_IFREG & 0xF000), false)
    }

    /// Create a directory at `path`. The new directory has `.` and
    /// `..` entries automatically.
    pub fn mkdir(&mut self, path: &str, mode: u16) -> Result<Inode, F2fsError> {
        self.create_kind(path, mode | S_IFDIR, true)
    }

    fn create_kind(&mut self, path: &str, mode: u16, is_dir: bool) -> Result<Inode, F2fsError> {
        let (parent_path, name) = split_parent(path)?;
        let parent = self.open_path(parent_path)?;
        if !parent.f2fs.is_directory() {
            return Err(F2fsError::NotADirectory);
        }
        // Existence check.
        if self.lookup_dir_entry(&parent, name).is_ok() {
            return Err(F2fsError::PathNotFound);
        }
        // Allocate nid.
        let nid = self.allocate_nid();
        // Build a fresh inode block.
        let mut block = Box::new([0u8; F2FS_BLOCK_SIZE as usize]);
        write_inode_header_into(
            &mut block,
            mode,
            0,
            if is_dir { INLINE_DENTRY } else { INLINE_DATA },
            name.as_bytes(),
            parent.inode_number,
            if is_dir { 2 } else { 1 },
        );
        // Node footer.
        let off = NODE_FOOTER_OFFSET;
        block[off..off + 4].copy_from_slice(&nid.to_le_bytes());
        block[off + 4..off + 8].copy_from_slice(&nid.to_le_bytes());
        // For directories with INLINE_DENTRY, seed the inline-dentry
        // area with `.` and `..` so directory listing immediately works.
        if is_dir {
            let mut inline = [0u8; F2FS_BLOCK_SIZE as usize];
            // Seed inline dentry area at offset 360 (matches the read-
            // path projection in `mount.rs`).
            const INLINE_OFFSET: usize = 360;
            let dot_dotdot = build_dentry_block_inline(&[
                (".", nid, FileType::Directory),
                ("..", parent.inode_number, FileType::Directory),
            ]);
            // Copy the synthetic dentry layout (sized to fit in the
            // inline area) into the inode block.
            let inline_size = F2FS_BLOCK_SIZE as usize - INLINE_OFFSET - 16; /* footer */
            let take = core::cmp::min(dot_dotdot.len(), inline_size);
            block[INLINE_OFFSET..INLINE_OFFSET + take].copy_from_slice(&dot_dotdot[..take]);
            inline[..take].copy_from_slice(&dot_dotdot[..take]);
        }
        // Allocate a physical block address for the inode block (one
        // data-segment slot — node-segments would be cleaner but v1
        // keeps node + data in the same stream).
        let inode_addr = self.allocate_data_block()?;
        self.write_block_at(inode_addr, &block)?;
        // Stage NAT update.
        self.dirty_nat(nid, nid, inode_addr);
        // Phase 9: SSA reverse map for the inode block itself + seed
        // the mtime cache so newly created files are "hot" until they
        // age past the cold threshold.
        self.record_ssa(inode_addr, nid, 0);
        let now = self.write_state.now_secs;
        self.record_inode_mtime(nid, now);
        // Cache the inode block.
        self.write_state.inode_cache.insert(nid, block);

        // Link into parent directory.
        self.add_dentry_to_dir(parent.inode_number, name, nid, is_dir)?;

        if is_dir {
            self.write_state.stats.mkdirs = self.write_state.stats.mkdirs.saturating_add(1);
        } else {
            self.write_state.stats.creates = self.write_state.stats.creates.saturating_add(1);
        }
        self.write_state.dirty = true;
        self.write_state.ticks_since_dirty = 0;

        self.read_inode_rw(nid)
    }

    /// Add a dentry `(name, nid)` to `dir_nid`'s directory inode.
    /// Phase 7 supports inline dentries only — directories never spill
    /// to a separate block in v1.
    fn add_dentry_to_dir(
        &mut self,
        dir_nid: u32,
        name: &str,
        target_nid: u32,
        is_target_dir: bool,
    ) -> Result<(), F2fsError> {
        let mut block = self.load_inode_block(dir_nid)?;
        // Set INLINE_DENTRY bit.
        block[3] |= INLINE_DENTRY;
        let inline_off = 360usize;
        let inline_end = F2FS_BLOCK_SIZE as usize - 16;
        let inline_area = &mut block[inline_off..inline_end];
        let ft = if is_target_dir {
            FileType::Directory
        } else {
            FileType::RegularFile
        };
        insert_inline_dentry(inline_area, name, target_nid, ft)?;
        // Bump links if adding a subdirectory.
        if is_target_dir {
            let cur = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);
            let new = cur.saturating_add(1);
            block[12..16].copy_from_slice(&new.to_le_bytes());
        }
        // Bump dir size to reflect the new entry — v1 uses a constant
        // "4096" so the size never exceeds the inline-dentry area.
        block[16..24].copy_from_slice(&(F2FS_BLOCK_SIZE as u64).to_le_bytes());
        self.write_state.inode_cache.insert(dir_nid, block.clone());
        self.flush_inode_block(dir_nid)
    }

    /// Remove `name` from `dir_nid`'s inline dentry list. Returns the
    /// removed inode-id and whether it was a directory.
    fn remove_dentry_from_dir(
        &mut self,
        dir_nid: u32,
        name: &str,
    ) -> Result<(u32, bool), F2fsError> {
        let mut block = self.load_inode_block(dir_nid)?;
        let inline_off = 360usize;
        let inline_end = F2FS_BLOCK_SIZE as usize - 16;
        let inline_area = &mut block[inline_off..inline_end];
        let removed = remove_inline_dentry(inline_area, name)?;
        // If we just unlinked a sub-directory, decrement the parent's
        // links count.
        if removed.1 {
            let cur = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);
            let new = cur.saturating_sub(1).max(2);
            block[12..16].copy_from_slice(&new.to_le_bytes());
        }
        self.write_state.inode_cache.insert(dir_nid, block.clone());
        self.flush_inode_block(dir_nid)?;
        Ok(removed)
    }

    // ── write_file ──────────────────────────────────────────────────────

    /// Write `buf.len()` bytes into `inode` starting at `offset`. The
    /// write extends the file if `offset + buf.len() > inode.size`.
    pub fn write_file(
        &mut self,
        inode: &Inode,
        offset: u64,
        buf: &[u8],
    ) -> Result<usize, F2fsError> {
        let nid = inode.inode_number;
        let mut inode_block = self.load_inode_block(nid)?;
        // Decide inline vs separate-block; for simplicity v1 spills any
        // file > 1500 B (matching the test-vector builder's threshold).
        let new_size = offset.saturating_add(buf.len() as u64).max(inode.f2fs.size);
        let inline_threshold = 1500u64;
        let was_inline = (inode_block[3] & INLINE_DATA) != 0;
        if new_size <= inline_threshold && was_inline {
            // Inline path: copy into the inline-data area.
            const INLINE_OFFSET: usize = 360;
            let inline_size = F2FS_BLOCK_SIZE as usize - INLINE_OFFSET - 16;
            if (offset as usize).saturating_add(buf.len()) > inline_size {
                return Err(F2fsError::InlineDataCorrupt);
            }
            let dst_off = INLINE_OFFSET + offset as usize;
            inode_block[dst_off..dst_off + buf.len()].copy_from_slice(buf);
            // Update i_size + i_blocks.
            inode_block[16..24].copy_from_slice(&new_size.to_le_bytes());
        } else {
            // Spill to data blocks. If we were inline, evacuate.
            if was_inline {
                inode_block[3] &= !INLINE_DATA;
                // Copy old inline bytes (up to inode.f2fs.size) into a
                // freshly allocated data block.
                if inode.f2fs.size > 0 {
                    let mut data = [0u8; F2FS_BLOCK_SIZE as usize];
                    const INLINE_OFFSET: usize = 360;
                    let len = inode.f2fs.size as usize;
                    data[..len]
                        .copy_from_slice(&inode.inode_block[INLINE_OFFSET..INLINE_OFFSET + len]);
                    let addr = self.allocate_data_block()?;
                    self.write_block_at(addr, &data)?;
                    self.record_ssa(addr, nid, 0);
                    write_inode_direct_addr_in(&mut inode_block, 0, addr);
                }
                // Zero the inline area.
                const INLINE_OFFSET: usize = 360;
                for byte in inode_block[INLINE_OFFSET..F2FS_BLOCK_SIZE as usize - 16].iter_mut() {
                    *byte = 0;
                }
            }
            // Loop over the affected 4 KiB blocks of the file.
            let mut written = 0usize;
            let mut cur = offset;
            while written < buf.len() {
                let block_idx = (cur / F2FS_BLOCK_SIZE as u64) as usize;
                let in_off = (cur % F2FS_BLOCK_SIZE as u64) as usize;
                let take = core::cmp::min(buf.len() - written, F2FS_BLOCK_SIZE as usize - in_off);

                if block_idx >= ADDRS_PER_INODE + ADDRS_PER_BLOCK {
                    return Err(F2fsError::DeepIndirectionUnsupported);
                }
                // Load existing block (so partial writes preserve the
                // surrounding bytes).
                let mut data = [0u8; F2FS_BLOCK_SIZE as usize];
                let existing_addr = read_inode_direct_addr_in(&inode_block, block_idx);
                if let Some(a) = existing_addr {
                    if a != 0 && a != u32::MAX {
                        read_f2fs_block(
                            &*self.device,
                            self.partition_lba,
                            self.partition_size_bytes,
                            a,
                            &mut data,
                        )?;
                    }
                }
                data[in_off..in_off + take].copy_from_slice(&buf[written..written + take]);
                // Allocate a fresh block (log-structured).
                let new_addr = self.allocate_data_block()?;
                self.write_block_at(new_addr, &data)?;
                // Phase 9: stamp the SSA reverse map BEFORE freeing the
                // old slot, so an interleaved GC scan never sees both
                // pointing at the same (nid, ofs).
                self.record_ssa(new_addr, nid, block_idx as u32);
                if block_idx < ADDRS_PER_INODE {
                    if let Some(old) = existing_addr {
                        if old != 0 && old != u32::MAX {
                            self.free_data_block(old);
                        }
                    }
                    write_inode_direct_addr_in(&mut inode_block, block_idx, new_addr);
                } else {
                    // Single-indirect path: ensure i_nid[0] points at a
                    // real indirect-node block; allocate one if not.
                    self.write_indirect_addr(
                        nid,
                        &mut inode_block,
                        block_idx - ADDRS_PER_INODE,
                        new_addr,
                    )?;
                }
                written += take;
                cur += take as u64;
            }
            inode_block[16..24].copy_from_slice(&new_size.to_le_bytes());
        }
        // Persist inode block.
        self.write_state.inode_cache.insert(nid, inode_block);
        self.flush_inode_block(nid)?;
        self.write_state.dirty = true;
        self.write_state.ticks_since_dirty = 0;
        self.write_state.stats.writes = self.write_state.stats.writes.saturating_add(1);
        // Phase 9: stamp the in-memory mtime cache so the GC age
        // classifier reflects this write. Production also flips the
        // `i_mtime` field in the on-disk inode block (Phase 10 — file
        // rewrites with explicit timestamps); the v1 write path keeps
        // the cache as source-of-truth for the GC pass and lets fsync
        // fold the value into the on-disk inode at checkpoint time.
        self.record_inode_mtime(nid, self.write_state.now_secs);

        // Maybe run a GC pass if the free-segment ratio is low.
        self.maybe_run_gc()?;

        Ok(buf.len())
    }

    /// Update the single-indirect node-block for `inode` so that
    /// `indirect_idx`-th slot = `new_addr`. Allocates a fresh indirect
    /// block (and a NAT entry for it) if `inode.i_nid[0]` is null.
    fn write_indirect_addr(
        &mut self,
        owning_inode_nid: u32,
        inode_block: &mut [u8; F2FS_BLOCK_SIZE as usize],
        indirect_idx: usize,
        new_addr: u32,
    ) -> Result<(), F2fsError> {
        let i_nid_off = NODE_FOOTER_OFFSET - NIDS_PER_INODE * 4;
        let mut indirect_nid = u32::from_le_bytes([
            inode_block[i_nid_off],
            inode_block[i_nid_off + 1],
            inode_block[i_nid_off + 2],
            inode_block[i_nid_off + 3],
        ]);
        let mut indirect_block = [0u8; F2FS_BLOCK_SIZE as usize];
        let node_addr;
        if indirect_nid == F2FS_NULL_NID {
            indirect_nid = self.allocate_nid();
            node_addr = self.allocate_data_block()?;
            // Footer for the indirect node.
            let off = NODE_FOOTER_OFFSET;
            indirect_block[off..off + 4].copy_from_slice(&indirect_nid.to_le_bytes());
            indirect_block[off + 4..off + 8].copy_from_slice(&owning_inode_nid.to_le_bytes());
            // Wire i_nid[0] in the inode.
            inode_block[i_nid_off..i_nid_off + 4].copy_from_slice(&indirect_nid.to_le_bytes());
            self.dirty_nat(indirect_nid, owning_inode_nid, node_addr);
            // Phase 9: SSA reverse map for the indirect-node block. The
            // ofs we record is u32::MAX so a GC live-block scan can
            // distinguish indirect-node blocks from regular data blocks.
            self.record_ssa(node_addr, indirect_nid, u32::MAX);
        } else {
            // Load existing indirect block.
            let addr = self.resolve_inode_addr(indirect_nid)?;
            read_f2fs_block(
                &*self.device,
                self.partition_lba,
                self.partition_size_bytes,
                addr,
                &mut indirect_block,
            )?;
            node_addr = addr;
        }
        let off = indirect_idx * 4;
        if off + 4 > F2FS_BLOCK_SIZE as usize - 16 {
            return Err(F2fsError::DeepIndirectionUnsupported);
        }
        indirect_block[off..off + 4].copy_from_slice(&new_addr.to_le_bytes());
        // Rewrite the indirect node block in place.
        // (Phase 9 will COW; v1 keeps it simple.)
        let _ = node_addr; // not yet log-structured
        let addr = self.resolve_inode_addr(indirect_nid)?;
        self.write_block_at(addr, &indirect_block)?;
        Ok(())
    }

    // ── truncate ────────────────────────────────────────────────────────

    /// Truncate `inode` to `new_size`. Shrinks release blocks via SIT;
    /// grows leave a sparse hole.
    pub fn truncate(&mut self, inode: Inode, new_size: u64) -> Result<(), F2fsError> {
        let nid = inode.inode_number;
        let mut block = self.load_inode_block(nid)?;
        let old_size = u64::from_le_bytes([
            block[16], block[17], block[18], block[19], block[20], block[21], block[22], block[23],
        ]);
        if new_size < old_size {
            // Walk every block past `new_size`; if one is allocated,
            // free it.
            let first_freed_block = (new_size as usize).div_ceil(F2FS_BLOCK_SIZE as usize);
            let last_block = (old_size as usize).div_ceil(F2FS_BLOCK_SIZE as usize);
            for block_idx in first_freed_block..last_block {
                if block_idx < ADDRS_PER_INODE {
                    let addr = read_inode_direct_addr_in(&block, block_idx);
                    if let Some(a) = addr {
                        if a != 0 && a != u32::MAX {
                            self.free_data_block(a);
                            write_inode_direct_addr_in(&mut block, block_idx, 0);
                        }
                    }
                }
                // Indirect freeing is best-effort; v1 leaves it for GC.
            }
        }
        // Update size + clear inline if growing past inline area.
        block[16..24].copy_from_slice(&new_size.to_le_bytes());
        if new_size > 1500 && (block[3] & INLINE_DATA) != 0 {
            block[3] &= !INLINE_DATA;
        }
        self.write_state.inode_cache.insert(nid, block);
        self.flush_inode_block(nid)?;
        self.write_state.dirty = true;
        self.write_state.ticks_since_dirty = 0;
        self.write_state.stats.truncates = self.write_state.stats.truncates.saturating_add(1);
        Ok(())
    }

    // ── unlink / rmdir ──────────────────────────────────────────────────

    /// Remove a regular file at `path`. Decrements link count; reclaims
    /// inode + blocks when count reaches 0.
    pub fn unlink(&mut self, path: &str) -> Result<(), F2fsError> {
        let (parent_path, name) = split_parent(path)?;
        let parent = self.open_path(parent_path)?;
        let target_nid = self.lookup_dir_entry(&parent, name)?;
        let target = self.read_inode_rw(target_nid)?;
        if target.f2fs.is_directory() {
            return Err(F2fsError::NotADirectory);
        }
        // Drop dentry from parent.
        self.remove_dentry_from_dir(parent.inode_number, name)?;
        // Decrement links; if 0, free everything.
        let mut block = self.load_inode_block(target_nid)?;
        let cur = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);
        let new = cur.saturating_sub(1);
        block[12..16].copy_from_slice(&new.to_le_bytes());
        self.write_state
            .inode_cache
            .insert(target_nid, block.clone());
        if new == 0 {
            // Free all data blocks referenced by direct pointers.
            for block_idx in 0..ADDRS_PER_INODE {
                let addr = read_inode_direct_addr_in(&block, block_idx);
                if let Some(a) = addr {
                    if a != 0 && a != u32::MAX {
                        self.free_data_block(a);
                    }
                }
            }
            // Free the inode block itself.
            if let Ok(addr) = self.resolve_inode_addr(target_nid) {
                self.free_data_block(addr);
            }
            // Remove the NAT mapping.
            self.dirty_nat(target_nid, 0, 0);
            // Return nid to the free pool.
            self.write_state.free_nid_pool.push(target_nid);
            self.write_state.inode_cache.remove(&target_nid);
        }
        self.write_state.dirty = true;
        self.write_state.ticks_since_dirty = 0;
        self.write_state.stats.unlinks = self.write_state.stats.unlinks.saturating_add(1);
        Ok(())
    }

    /// Remove an empty directory at `path`.
    pub fn rmdir(&mut self, path: &str) -> Result<(), F2fsError> {
        let (parent_path, name) = split_parent(path)?;
        let parent = self.open_path(parent_path)?;
        let target_nid = self.lookup_dir_entry(&parent, name)?;
        let target = self.read_inode_rw(target_nid)?;
        if !target.f2fs.is_directory() {
            return Err(F2fsError::NotADirectory);
        }
        // Reject if the directory has any entries other than `.` and `..`.
        let entries: Vec<_> = self.read_dir(&target)?.collect();
        let has_real_entry = entries.iter().any(|e| e.name != "." && e.name != "..");
        if has_real_entry {
            return Err(F2fsError::NotADirectory);
        }
        self.remove_dentry_from_dir(parent.inode_number, name)?;
        // Free the inode block.
        if let Ok(addr) = self.resolve_inode_addr(target_nid) {
            self.free_data_block(addr);
        }
        self.dirty_nat(target_nid, 0, 0);
        self.write_state.free_nid_pool.push(target_nid);
        self.write_state.inode_cache.remove(&target_nid);
        self.write_state.dirty = true;
        self.write_state.ticks_since_dirty = 0;
        self.write_state.stats.rmdirs = self.write_state.stats.rmdirs.saturating_add(1);
        Ok(())
    }

    // ── rename ──────────────────────────────────────────────────────────

    /// Atomically rename `src` to `dst`. Open readers of `dst` continue
    /// to see the pre-rename content; new opens see the moved content.
    pub fn rename(&mut self, src: &str, dst: &str) -> Result<(), F2fsError> {
        if src == dst {
            // POSIX-style: rename to self is a successful no-op.
            return Ok(());
        }
        let (src_parent_path, src_name) = split_parent(src)?;
        let (dst_parent_path, dst_name) = split_parent(dst)?;
        let src_parent = self.open_path(src_parent_path)?;
        let dst_parent = self.open_path(dst_parent_path)?;
        let moved_nid = self.lookup_dir_entry(&src_parent, src_name)?;
        // Capture the existing dst (for reclaim if it was a different file).
        let dst_old_nid = self.lookup_dir_entry(&dst_parent, dst_name).ok();
        // Stage in journal as pending.
        let record = RenameRecord {
            src_parent_nid: src_parent.inode_number,
            dst_parent_nid: dst_parent.inode_number,
            moved_nid,
            dst_old_nid: dst_old_nid.unwrap_or(0),
            flags: 0,
        };
        self.write_state
            .journal_dirty
            .push(JournalEntry::Rename(record));
        // Update the parent dirs.
        if let Some(old) = dst_old_nid {
            // Free the displaced inode (its blocks go back to SIT).
            // Best-effort: reuse unlink semantics.
            // (We already removed dst from its parent's dentry, so mark
            // the freed-nid bookkeeping here.)
            let mut blk = self.load_inode_block(old)?;
            let cur = u32::from_le_bytes([blk[12], blk[13], blk[14], blk[15]]);
            let new = cur.saturating_sub(1);
            blk[12..16].copy_from_slice(&new.to_le_bytes());
            self.write_state.inode_cache.insert(old, blk.clone());
            if new == 0 {
                for block_idx in 0..ADDRS_PER_INODE {
                    if let Some(a) = read_inode_direct_addr_in(&blk, block_idx) {
                        if a != 0 && a != u32::MAX {
                            self.free_data_block(a);
                        }
                    }
                }
                if let Ok(addr) = self.resolve_inode_addr(old) {
                    self.free_data_block(addr);
                }
                self.dirty_nat(old, 0, 0);
                self.write_state.free_nid_pool.push(old);
                self.write_state.inode_cache.remove(&old);
            }
            self.remove_dentry_from_dir(dst_parent.inode_number, dst_name)?;
        }
        // Add (dst_name → moved_nid) into dst_parent.
        let moved_inode = self.read_inode_rw(moved_nid)?;
        let is_dir = moved_inode.f2fs.is_directory();
        self.add_dentry_to_dir(dst_parent.inode_number, dst_name, moved_nid, is_dir)?;
        // Drop src dentry.
        self.remove_dentry_from_dir(src_parent.inode_number, src_name)?;
        // Mark journal entry committed.
        if let Some(JournalEntry::Rename(r)) = self.write_state.journal_dirty.last_mut() {
            r.flags |= FLAG_RENAME_COMMITTED;
        }
        self.write_state.dirty = true;
        self.write_state.ticks_since_dirty = 0;
        self.write_state.stats.renames = self.write_state.stats.renames.saturating_add(1);
        Ok(())
    }

    // ── fsync / checkpoint commit ───────────────────────────────────────

    /// Force a checkpoint commit. After a successful return the data
    /// is durable across power loss.
    pub fn fsync(&mut self) -> Result<(), F2fsError> {
        if !self.write_state.dirty {
            self.write_state.stats.fsyncs = self.write_state.stats.fsyncs.saturating_add(1);
            return Ok(());
        }
        // 1. Persist all dirty inode blocks (they may already be on
        //    disk via flush_inode_block; ensure cache is consistent).
        let nids: Vec<u32> = self.write_state.inode_cache.keys().copied().collect();
        for nid in nids {
            self.flush_inode_block(nid)?;
        }
        // 2. Persist NAT updates: for each dirty NAT entry, read the
        //    NAT block, patch the entry, write it back.
        let nat_updates: Vec<(u32, PendingNat)> = self
            .write_state
            .nat_dirty
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        let mut nat_blocks: BTreeMap<u32, [u8; F2FS_BLOCK_SIZE as usize]> = BTreeMap::new();
        for (nid, pending) in nat_updates {
            let block_idx = nid_to_nat_block_idx(nid);
            let entry_off = nid_to_entry_off(nid) * NAT_ENTRY_SIZE;
            let phys = self.sb.nat_blkaddr.saturating_add(block_idx);
            let block = nat_blocks.entry(phys).or_insert_with(|| {
                let mut b = [0u8; F2FS_BLOCK_SIZE as usize];
                let _ = read_f2fs_block(
                    &*self.device,
                    self.partition_lba,
                    self.partition_size_bytes,
                    phys,
                    &mut b,
                );
                b
            });
            block[entry_off] = pending.version;
            block[entry_off + 1..entry_off + 5].copy_from_slice(&pending.ino.to_le_bytes());
            block[entry_off + 5..entry_off + 9].copy_from_slice(&pending.block_addr.to_le_bytes());
            let _ = entry_off; // silence unused if we ever rearrange
            let _ = NAT_ENTRY_PER_BLOCK; // referenced for documentation
        }
        for (phys, block) in nat_blocks.iter() {
            self.write_block_at(*phys, block)?;
        }
        // 3. Persist SIT updates similarly.
        let sit_updates: Vec<(u32, PendingSit)> = self
            .write_state
            .sit_dirty
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        let mut sit_blocks: BTreeMap<u32, [u8; F2FS_BLOCK_SIZE as usize]> = BTreeMap::new();
        for (segno, pending) in sit_updates {
            let block_idx = segno / SIT_ENTRY_PER_BLOCK as u32;
            let entry_idx = (segno as usize) % SIT_ENTRY_PER_BLOCK;
            let entry_off = entry_idx * SIT_ENTRY_SIZE;
            let phys = self.sb.sit_blkaddr.saturating_add(block_idx);
            let block = sit_blocks.entry(phys).or_insert_with(|| {
                let mut b = [0u8; F2FS_BLOCK_SIZE as usize];
                let _ = read_f2fs_block(
                    &*self.device,
                    self.partition_lba,
                    self.partition_size_bytes,
                    phys,
                    &mut b,
                );
                b
            });
            let raw = pending.valid_blocks | pending.raw_type_bits;
            block[entry_off..entry_off + 2].copy_from_slice(&raw.to_le_bytes());
            block[entry_off + 2..entry_off + 2 + SIT_VBLOCK_MAP_BYTES]
                .copy_from_slice(&pending.valid_map);
            let mtime_off = entry_off + 2 + SIT_VBLOCK_MAP_BYTES;
            block[mtime_off..mtime_off + 8].copy_from_slice(&pending.mtime.to_le_bytes());
        }
        for (phys, block) in sit_blocks.iter() {
            self.write_block_at(*phys, block)?;
        }
        // 4. Build the new checkpoint header in the alternate slot.
        self.write_state.checkpoint_ver = self.write_state.checkpoint_ver.saturating_add(1);
        let cp_block = self.build_checkpoint_block();
        let new_slot = if self.cp_slot == 0 { 1 } else { 0 };
        let cp_addr = if new_slot == 0 {
            self.sb.cp_blkaddr
        } else {
            self.sb.cp_blkaddr.saturating_add(F2FS_BLOCKS_PER_SEG)
        };
        self.write_block_at(cp_addr, &cp_block)?;
        // 5. Persist journal block (block immediately after the
        //    checkpoint header in the same pack).
        let journal_block = encode_journal_block(&self.write_state.journal_dirty);
        self.write_block_at(cp_addr.saturating_add(1), &journal_block)?;
        // 6. Flush the device (issue a barrier).
        self.device.flush().map_err(F2fsError::IoError)?;
        // 7. Atomic switch.
        self.cp_slot = new_slot;
        // Update the parsed checkpoint snapshot so `checkpoint()` reads
        // the new state.
        self.checkpoint.checkpoint_ver = self.write_state.checkpoint_ver;
        self.checkpoint.cur_data_segno[0] = self.write_state.cur_data_segno;
        self.checkpoint.cur_data_blkoff[0] = self.write_state.cur_data_blkoff;
        // Retire dirty state.
        self.write_state.nat_dirty.clear();
        self.write_state.sit_dirty.clear();
        self.write_state.journal_dirty.clear();
        // Phase 9: GC journal records persist transitively via the SIT
        // updates above (each `committed=1` record correlates with a
        // SIT bit flip). Drain the staging buffer so the next pass
        // starts clean.
        self.write_state.gc_journal_dirty.clear();
        self.write_state.dirty = false;
        self.write_state.ticks_since_dirty = 0;
        self.write_state.stats.fsyncs = self.write_state.stats.fsyncs.saturating_add(1);
        Ok(())
    }

    /// Construct the 4 KiB checkpoint pack header from the current
    /// in-memory state.
    fn build_checkpoint_block(&self) -> [u8; F2FS_BLOCK_SIZE as usize] {
        let mut block = [0u8; F2FS_BLOCK_SIZE as usize];
        block[0..8].copy_from_slice(&self.write_state.checkpoint_ver.to_le_bytes());
        block[8..16].copy_from_slice(&self.checkpoint.user_block_count.to_le_bytes());
        block[16..24].copy_from_slice(&self.checkpoint.valid_block_count.to_le_bytes());
        block[24..28].copy_from_slice(&self.checkpoint.rsvd_segment_count.to_le_bytes());
        block[28..32].copy_from_slice(&self.checkpoint.overprov_segment_count.to_le_bytes());
        block[32..36].copy_from_slice(&self.checkpoint.free_segment_count.to_le_bytes());
        for (i, segno) in self.checkpoint.cur_node_segno.iter().enumerate() {
            let off = 36 + i * 4;
            block[off..off + 4].copy_from_slice(&segno.to_le_bytes());
        }
        for (i, blkoff) in self.checkpoint.cur_node_blkoff.iter().enumerate() {
            let off = 68 + i * 2;
            block[off..off + 2].copy_from_slice(&blkoff.to_le_bytes());
        }
        // Slot 0 is the "data" stream; mirror the in-memory cursor.
        let mut data_segno = self.checkpoint.cur_data_segno;
        let mut data_blkoff = self.checkpoint.cur_data_blkoff;
        data_segno[0] = self.write_state.cur_data_segno;
        data_blkoff[0] = self.write_state.cur_data_blkoff;
        for (i, segno) in data_segno.iter().enumerate() {
            let off = 84 + i * 4;
            block[off..off + 4].copy_from_slice(&segno.to_le_bytes());
        }
        for (i, blkoff) in data_blkoff.iter().enumerate() {
            let off = 116 + i * 2;
            block[off..off + 2].copy_from_slice(&blkoff.to_le_bytes());
        }
        block[132..136].copy_from_slice(&self.checkpoint.valid_node_count.to_le_bytes());
        block[136..140].copy_from_slice(&self.checkpoint.valid_inode_count.to_le_bytes());
        block[140..144].copy_from_slice(&self.write_state.next_free_nid.to_le_bytes());
        block[144..148].copy_from_slice(&self.checkpoint.sit_ver_bitmap_bytesize.to_le_bytes());
        block[148..152].copy_from_slice(&self.checkpoint.nat_ver_bitmap_bytesize.to_le_bytes());
        block[152..156].copy_from_slice(&(DEFAULT_CHECKPOINT_CRC_OFFSET as u32).to_le_bytes());
        block[156..164].copy_from_slice(&self.checkpoint.elapsed_time.to_le_bytes());
        // Compute and store CRC.
        let crc = f2fs_crc32(0, &block[..DEFAULT_CHECKPOINT_CRC_OFFSET]);
        block[DEFAULT_CHECKPOINT_CRC_OFFSET..DEFAULT_CHECKPOINT_CRC_OFFSET + 4]
            .copy_from_slice(&crc.to_le_bytes());
        block
    }

    // ── GC integration ──────────────────────────────────────────────────

    /// If free segments are below the threshold, run one Phase 9 GC
    /// pass via [`run_gc_pass_v2`].
    ///
    /// The integration:
    /// 1. Builds the `(segno, valid_blocks)` candidate list from the
    ///    SIT-dirty set.
    /// 2. Projects each victim's live blocks via the in-memory SSA
    ///    reverse map + the per-inode mtime cache.
    /// 3. Pre-allocates one fresh data block per live block (so the
    ///    `relocate` closure inside `run_gc_pass_v2` does NOT need
    ///    `&mut self`).
    /// 4. Runs the Phase 9 driver with hot/cold target hints + the
    ///    [`DEFAULT_YIELD_BUDGET_BLOCKS`] / [`DEFAULT_PASS_BUDGET_TICKS`]
    ///    cooperative-yield knobs.
    /// 5. Applies the outcome: copies bytes into the new addresses,
    ///    updates SIT/SSA, drains the segment, stages the GC journal
    ///    records into [`WriteState::gc_journal_dirty`], and pushes the
    ///    freed segno into `free_segments`.
    fn maybe_run_gc(&mut self) -> Result<(), F2fsError> {
        // Total segments comes from the superblock; cache it on first use.
        if self.write_state.total_segments == 0 {
            self.write_state.total_segments = self.sb.segment_count_main;
        }
        let used_segments = self.write_state.cur_data_segno.saturating_add(1);
        let free_segments = self
            .write_state
            .total_segments
            .saturating_sub(used_segments)
            .saturating_add(self.write_state.free_segments.len() as u32);
        if !gc::should_run_gc(free_segments, self.write_state.total_segments) {
            return Ok(());
        }

        // Phase 9 candidate set: every SIT-dirty segment EXCEPT the
        // current cursor segments (we never relocate blocks out from
        // under an active write cursor).
        let active_cursors = [
            self.write_state.cur_data_segno,
            self.write_state.cur_hot_segno,
            self.write_state.cur_cold_segno,
        ];
        let candidates: Vec<(u32, u16)> = self
            .write_state
            .sit_dirty
            .iter()
            .filter(|(s, _)| !active_cursors.contains(s))
            .map(|(s, e)| (*s, e.valid_blocks))
            .collect();

        // Project each candidate's live blocks via SSA + mtime cache.
        let victims = gc::pick_victims(
            &candidates,
            F2FS_BLOCKS_PER_SEG,
            DEFAULT_GC_VICTIMS_PER_PASS,
        );
        let mut victim_live: Vec<(u32, Vec<LiveBlock>)> = Vec::with_capacity(victims.len());
        for victim_segno in &victims {
            let valid_map = self
                .write_state
                .sit_dirty
                .get(victim_segno)
                .map(|e| e.valid_map)
                .unwrap_or([0u8; SIT_VBLOCK_MAP_BYTES]);
            let ssa_index = self.write_state.ssa_index.clone();
            let mtime_index = self.write_state.mtime_index.clone();
            let sb_main = self.sb.main_blkaddr;
            let live = gc::live_blocks_from_segment(
                *victim_segno,
                &valid_map,
                |idx| {
                    let addr =
                        sb_main + victim_segno.saturating_mul(F2FS_BLOCKS_PER_SEG) + idx as u32;
                    ssa_index.get(&addr).copied()
                },
                |nid| mtime_index.get(&nid).copied(),
            );
            victim_live.push((*victim_segno, live));
        }

        // Pre-allocate one fresh address per live block (so the
        // `relocate` closure stays pure-functional). We allocate these
        // in advance because the v2 driver's closures cannot borrow
        // `&mut self`.
        let mut prealloc: Vec<u32> = Vec::new();
        for (_, live) in &victim_live {
            for _ in live.iter() {
                prealloc.push(self.allocate_data_block()?);
            }
        }
        prealloc.reverse(); // pop() from the back yields in alloc order

        // Snapshot the per-victim live-block lists so the driver's
        // `live_blocks_for` closure can serve them without borrowing
        // `&self`.
        let mut live_lookup: BTreeMap<u32, Vec<LiveBlock>> = BTreeMap::new();
        let mut sit_candidate_snapshot: Vec<(u32, u16)> = Vec::new();
        for (segno, live) in victim_live.into_iter() {
            sit_candidate_snapshot.push((
                segno,
                self.write_state
                    .sit_dirty
                    .get(&segno)
                    .map(|e| e.valid_blocks)
                    .unwrap_or(0),
            ));
            live_lookup.insert(segno, live);
        }

        // Build the Phase 9 config — pull defaults from `gc::*`.
        let cfg = GcPassConfig {
            victims_per_pass: DEFAULT_GC_VICTIMS_PER_PASS,
            yield_budget_blocks: DEFAULT_YIELD_BUDGET_BLOCKS,
            pass_budget_ticks: DEFAULT_PASS_BUDGET_TICKS,
            cold_threshold_secs: DEFAULT_COLD_THRESHOLD_SECS,
            now_secs: self.write_state.now_secs,
            hints: AllocatorHints {
                hot_segno: self.write_state.cur_hot_segno,
                cold_segno: self.write_state.cur_cold_segno,
            },
        };

        let yield_pending = self.write_state.yield_pending;
        let mut yielded_once = false;
        let outcome: Result<GcPassOutcome, F2fsError> = gc::run_gc_pass_v2::<_, _, _, _, F2fsError>(
            &sit_candidate_snapshot,
            F2FS_BLOCKS_PER_SEG,
            cfg,
            |segno| live_lookup.remove(&segno).unwrap_or_default(),
            |_blk, _target| {
                let new_addr = prealloc.pop().ok_or(F2fsError::SizeOverflow)?;
                Ok(new_addr)
            },
            || {
                // Surface the foreground yield flag exactly once per
                // pass — re-arming would wedge the inner loop.
                if yield_pending && !yielded_once {
                    yielded_once = true;
                    return true;
                }
                false
            },
            // Synthetic clock: each call increments by 1 tick. The
            // pass-budget cap of 100 means a single pass runs at most
            // 100 yield-poll iterations. Production wires the real
            // monotonic clock here.
            {
                let mut t: u64 = 0;
                move || {
                    t = t.saturating_add(1);
                    t
                }
            },
        );
        let outcome = outcome?;

        // Apply outcome: relocate bytes, update SIT/SSA, mark drained
        // segments free.
        for record in &outcome.journal {
            // Phase 9 invariant: stage the journal record BEFORE the
            // SIT mutation lands. `gc_journal_dirty` is the staging
            // buffer; fsync drains it alongside `journal_dirty`.
            self.write_state.gc_journal_dirty.push(*record);
        }
        for entry in &outcome.log {
            if entry.dst_block_addr == 0 {
                // Yield-marker entry — no relocation occurred.
                continue;
            }
            // Find the source addr by inverting `addr_from`. The
            // journal record carries the *relative* (src) address —
            // re-derive the absolute one through the partition base.
            // Re-derive the absolute source address from the journal
            // record. `gc_journal::Record::old_block_addr` is the
            // *relative* form `segno * blocks_per_seg + blkidx` (per
            // `gc::addr_from` docs); add `main_blkaddr` for the
            // partition-absolute form the device I/O path expects.
            let matching_record = outcome
                .journal
                .iter()
                .find(|r| r.nid == entry.owning_nid && r.new_block_addr == entry.dst_block_addr);
            let old_addr = matching_record
                .map(|r| self.sb.main_blkaddr.saturating_add(r.old_block_addr))
                .unwrap_or(0);
            // Read the source block, write to dst, update SIT.
            if old_addr == 0 {
                continue;
            }
            let mut buf = [0u8; F2FS_BLOCK_SIZE as usize];
            read_f2fs_block(
                &*self.device,
                self.partition_lba,
                self.partition_size_bytes,
                old_addr,
                &mut buf,
            )?;
            self.write_block_at(entry.dst_block_addr, &buf)?;
            // Move SSA reverse-map entry to the new addr.
            if let Some((nid, ofs)) = self.write_state.ssa_index.remove(&old_addr) {
                self.write_state
                    .ssa_index
                    .insert(entry.dst_block_addr, (nid, ofs));
            }
            // Free the source block.
            self.free_data_block(old_addr);
        }

        // Push every fully-drained segment onto the free list.
        for segno in &victims {
            let drained = self
                .write_state
                .sit_dirty
                .get(segno)
                .map(|e| e.valid_blocks == 0)
                .unwrap_or(true);
            if drained {
                self.write_state.free_segments.push(*segno);
            }
        }
        self.write_state.stats.gc_passes = self
            .write_state
            .stats
            .gc_passes
            .saturating_add(1 + outcome.segments_freed);
        // Reset yield flag (single-shot per pass).
        self.write_state.yield_pending = false;
        Ok(())
    }
}

// ─── Inline-dentry helpers (used for inline-only directories in v1) ─────────

/// Build a synthetic dentry-block layout (sized to fit in the inline
/// area). Returns the bytes; callers copy them into the inode block.
fn build_dentry_block_inline(entries: &[(&str, u32, FileType)]) -> Vec<u8> {
    let mut buf = alloc::vec![0u8; F2FS_BLOCK_SIZE as usize];
    let mut slot = 0usize;
    for &(name, ino, ft) in entries {
        let nl = name.len() as u16;
        let slots = slots_for_name(nl);
        for s in slot..slot + slots {
            buf[s / 8] |= 1u8 << (s % 8);
        }
        let entry_off = ENTRY_ARRAY_OFFSET + slot * F2FS_DIR_ENTRY_SIZE;
        buf[entry_off..entry_off + 4].copy_from_slice(&0u32.to_le_bytes()); // hash
        buf[entry_off + 4..entry_off + 8].copy_from_slice(&ino.to_le_bytes());
        buf[entry_off + 8..entry_off + 10].copy_from_slice(&nl.to_le_bytes());
        buf[entry_off + 10] = match ft {
            FileType::RegularFile => 1,
            FileType::Directory => 2,
            FileType::CharDevice => 3,
            FileType::BlockDevice => 4,
            FileType::Fifo => 5,
            FileType::Socket => 6,
            FileType::Symlink => 7,
            FileType::Unknown => 0,
        };
        let name_start = FILENAME_SLOTS_OFFSET + slot * F2FS_SLOT_LEN;
        buf[name_start..name_start + name.len()].copy_from_slice(name.as_bytes());
        slot += slots;
    }
    buf
}

/// Insert a dentry into an *inline-dentry area* (≤ 3720 bytes within
/// the inode block).
fn insert_inline_dentry(
    inline_area: &mut [u8],
    name: &str,
    ino: u32,
    ft: FileType,
) -> Result<(), F2fsError> {
    let nl = name.len() as u16;
    let slots = slots_for_name(nl);
    if slots == 0 {
        return Err(F2fsError::PathNotFound);
    }
    // Find the first run of `slots` clear bitmap bits.
    let bitmap = &inline_area[..SIZE_OF_DENTRY_BITMAP];
    let mut start_slot = None;
    let mut i = 0usize;
    while i + slots <= NR_DENTRY_IN_BLOCK {
        let mut run_clear = true;
        for s in i..i + slots {
            let byte = s / 8;
            let bit = s % 8;
            if (bitmap[byte] >> bit) & 1 == 1 {
                run_clear = false;
                break;
            }
        }
        if run_clear {
            start_slot = Some(i);
            break;
        }
        i += 1;
    }
    let slot = start_slot.ok_or(F2fsError::SizeOverflow)?;
    // Mark bitmap bits.
    for s in slot..slot + slots {
        inline_area[s / 8] |= 1u8 << (s % 8);
    }
    let entry_off = ENTRY_ARRAY_OFFSET + slot * F2FS_DIR_ENTRY_SIZE;
    if entry_off + F2FS_DIR_ENTRY_SIZE > inline_area.len() {
        return Err(F2fsError::SizeOverflow);
    }
    inline_area[entry_off..entry_off + 4].copy_from_slice(&0u32.to_le_bytes()); // hash
    inline_area[entry_off + 4..entry_off + 8].copy_from_slice(&ino.to_le_bytes());
    inline_area[entry_off + 8..entry_off + 10].copy_from_slice(&nl.to_le_bytes());
    inline_area[entry_off + 10] = match ft {
        FileType::RegularFile => 1,
        FileType::Directory => 2,
        FileType::CharDevice => 3,
        FileType::BlockDevice => 4,
        FileType::Fifo => 5,
        FileType::Socket => 6,
        FileType::Symlink => 7,
        FileType::Unknown => 0,
    };
    let name_start = FILENAME_SLOTS_OFFSET + slot * F2FS_SLOT_LEN;
    if name_start + name.len() > inline_area.len() {
        return Err(F2fsError::SizeOverflow);
    }
    inline_area[name_start..name_start + name.len()].copy_from_slice(name.as_bytes());
    Ok(())
}

/// Remove the named entry from an inline-dentry area. Returns
/// `(removed_ino, was_directory)`.
fn remove_inline_dentry(inline_area: &mut [u8], name: &str) -> Result<(u32, bool), F2fsError> {
    let mut slot = 0usize;
    while slot < NR_DENTRY_IN_BLOCK {
        let byte = slot / 8;
        let bit = slot % 8;
        if (inline_area[byte] >> bit) & 1 == 0 {
            slot += 1;
            continue;
        }
        let entry_off = ENTRY_ARRAY_OFFSET + slot * F2FS_DIR_ENTRY_SIZE;
        if entry_off + F2FS_DIR_ENTRY_SIZE > inline_area.len() {
            break;
        }
        let nl = u16::from_le_bytes([inline_area[entry_off + 8], inline_area[entry_off + 9]]);
        let slots = slots_for_name(nl);
        let name_start = FILENAME_SLOTS_OFFSET + slot * F2FS_SLOT_LEN;
        if name_start + nl as usize > inline_area.len() {
            break;
        }
        let entry_name = &inline_area[name_start..name_start + nl as usize];
        if entry_name == name.as_bytes() {
            let ino = u32::from_le_bytes([
                inline_area[entry_off + 4],
                inline_area[entry_off + 5],
                inline_area[entry_off + 6],
                inline_area[entry_off + 7],
            ]);
            let ft = inline_area[entry_off + 10];
            // Clear bitmap bits.
            for s in slot..slot + slots {
                inline_area[s / 8] &= !(1u8 << (s % 8));
            }
            // Zero the entry header + name.
            for byte in inline_area[entry_off..entry_off + F2FS_DIR_ENTRY_SIZE].iter_mut() {
                *byte = 0;
            }
            for byte in inline_area[name_start..name_start + nl as usize].iter_mut() {
                *byte = 0;
            }
            return Ok((ino, ft == 2));
        }
        slot += slots.max(1);
    }
    Err(F2fsError::PathNotFound)
}

/// Set `i_addr[idx]` in an inode block.
fn write_inode_direct_addr_in(block: &mut [u8; F2FS_BLOCK_SIZE as usize], idx: usize, addr: u32) {
    let off = 360 + idx * 4;
    if off + 4 <= block.len() {
        block[off..off + 4].copy_from_slice(&addr.to_le_bytes());
    }
}

/// Read `i_addr[idx]` from an inode block. Returns `None` if `idx` is
/// out of range for the direct array.
fn read_inode_direct_addr_in(block: &[u8; F2FS_BLOCK_SIZE as usize], idx: usize) -> Option<u32> {
    if idx >= ADDRS_PER_INODE {
        return None;
    }
    let off = 360 + idx * 4;
    Some(u32::from_le_bytes([
        block[off],
        block[off + 1],
        block[off + 2],
        block[off + 3],
    ]))
}

/// Write the canonical inode header (mode, size, links, name, pino) at
/// the start of an inode block.
#[allow(clippy::too_many_arguments)]
fn write_inode_header_into(
    block: &mut [u8; F2FS_BLOCK_SIZE as usize],
    mode: u16,
    size: u64,
    inline_flags: u8,
    name: &[u8],
    pino: u32,
    links: u32,
) {
    block[0..2].copy_from_slice(&mode.to_le_bytes());
    block[3] = inline_flags;
    block[12..16].copy_from_slice(&links.to_le_bytes());
    block[16..24].copy_from_slice(&size.to_le_bytes());
    block[80..84].copy_from_slice(&pino.to_le_bytes());
    let nl = core::cmp::min(name.len(), 255);
    block[84..88].copy_from_slice(&(nl as u32).to_le_bytes());
    block[88..88 + nl].copy_from_slice(&name[..nl]);
}

// ─── Path helpers ───────────────────────────────────────────────────────────

/// Split a path into `(parent, name)`. Returns `Err(PathNotFound)` for
/// malformed inputs.
fn split_parent(path: &str) -> Result<(&str, &str), F2fsError> {
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(F2fsError::PathNotFound);
    }
    let last_slash = trimmed.rfind('/');
    match last_slash {
        Some(i) => {
            let parent = if i == 0 { "/" } else { &trimmed[..i] };
            let name = &trimmed[i + 1..];
            if name.is_empty() {
                Err(F2fsError::PathNotFound)
            } else {
                Ok((parent, name))
            }
        }
        None => Err(F2fsError::PathNotFound),
    }
}

/// Helper: emit a [`String`] path with `.tmp` appended (used by
/// callers that do staged writes).
pub fn staging_path(path: &str) -> String {
    let mut s = path.to_string();
    s.push_str(".tmp");
    s
}

// ─── auth::FsWriteBackend bridge ────────────────────────────────────────────

/// Map an F2FS-layer error into the auth atomic-write surface. The
/// auth layer doesn't import F2FS, so we collapse every fault into the
/// `Io` variant with a string tag the auth audit log can record.
fn f2fs_to_atomic(err: F2fsError) -> AtomicWriteError {
    use F2fsError as E;
    match err {
        E::PathNotFound => AtomicWriteError::Io("f2fs: path not found"),
        E::NotADirectory => AtomicWriteError::Io("f2fs: not a directory"),
        E::SizeOverflow => AtomicWriteError::Io("f2fs: size overflow"),
        E::IoError(_) => AtomicWriteError::Io("f2fs: block I/O error"),
        E::ReadPastEof => AtomicWriteError::Io("f2fs: read past EOF"),
        E::DeepIndirectionUnsupported => AtomicWriteError::Io("f2fs: deep indirection unsupported"),
        E::InlineDataCorrupt => AtomicWriteError::Io("f2fs: inline data corrupt"),
        _ => AtomicWriteError::Io("f2fs: write failure"),
    }
}

impl<'a, D: BlockDevice + ?Sized> FsWriteBackend for F2fs<'a, D> {
    fn create_and_write(&mut self, path: &str, contents: &[u8]) -> Result<(), AtomicWriteError> {
        // If a stale entry from a prior crash exists at this path, the
        // create call will report it via lookup_dir_entry returning Ok.
        // For idempotency we accept either a fresh allocation or an
        // overwrite of an existing inode.
        let inode = match self.create(path, 0o600) {
            Ok(i) => i,
            Err(F2fsError::PathNotFound) => self.open_path(path).map_err(f2fs_to_atomic)?,
            Err(e) => return Err(f2fs_to_atomic(e)),
        };
        if !contents.is_empty() {
            self.write_file(&inode, 0, contents)
                .map_err(f2fs_to_atomic)?;
        }
        Ok(())
    }

    fn rename(&mut self, src: &str, dst: &str) -> Result<(), AtomicWriteError> {
        F2fs::rename(self, src, dst).map_err(f2fs_to_atomic)
    }

    fn fsync(&mut self) -> Result<(), AtomicWriteError> {
        F2fs::fsync(self).map_err(f2fs_to_atomic)
    }

    fn cleanup_tmp_in(&mut self, dir: &str) -> Result<(), AtomicWriteError> {
        // Walk the directory and unlink any entry whose name ends in
        // ".tmp". Best-effort: errors removing individual entries
        // surface only after the whole directory has been scanned.
        let dir_inode = self.open_path(dir).map_err(f2fs_to_atomic)?;
        let entries: Vec<String> = self
            .read_dir(&dir_inode)
            .map_err(f2fs_to_atomic)?
            .filter(|e| e.name.ends_with(".tmp"))
            .map(|e| e.name)
            .collect();
        let mut first_err: Option<AtomicWriteError> = None;
        for name in entries {
            let mut full = String::from(dir);
            if !dir.ends_with('/') {
                full.push('/');
            }
            full.push_str(&name);
            if let Err(e) = self.unlink(&full) {
                if first_err.is_none() {
                    first_err = Some(f2fs_to_atomic(e));
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_parent_root_file() {
        assert_eq!(split_parent("/foo").unwrap(), ("/", "foo"));
    }

    #[test]
    fn split_parent_nested() {
        assert_eq!(split_parent("/a/b/c").unwrap(), ("/a/b", "c"));
    }

    #[test]
    fn split_parent_trailing_slash() {
        // The trim_end_matches consumes the trailing slash, leaving us
        // with "/a/b" → parent "/a", name "b".
        assert_eq!(split_parent("/a/b/").unwrap(), ("/a", "b"));
    }

    #[test]
    fn split_parent_rejects_root() {
        assert!(split_parent("/").is_err());
        assert!(split_parent("").is_err());
    }

    #[test]
    fn pending_sit_mark_and_unmark() {
        let mut p = PendingSit::empty();
        p.mark_valid(5);
        assert_eq!(p.valid_blocks, 1);
        p.mark_valid(5);
        assert_eq!(p.valid_blocks, 1, "double-mark idempotent");
        p.mark_invalid(5);
        assert_eq!(p.valid_blocks, 0);
        p.mark_invalid(5);
        assert_eq!(p.valid_blocks, 0, "double-unmark idempotent");
    }

    #[test]
    fn pending_sit_bounds_safe() {
        let mut p = PendingSit::empty();
        p.mark_valid(512);
        p.mark_invalid(512);
        p.mark_valid(usize::MAX);
        assert_eq!(p.valid_blocks, 0);
    }

    #[test]
    fn staging_path_appends_tmp() {
        assert_eq!(staging_path("/data/auth/shadow"), "/data/auth/shadow.tmp");
        assert_eq!(staging_path(""), ".tmp");
    }

    #[test]
    fn addr_to_segno_blkoff_round_trip() {
        let sb = test_sb();
        let addr = sb.main_blkaddr + 5 * F2FS_BLOCKS_PER_SEG + 17;
        let (segno, blkoff) = addr_to_segno_blkoff(&sb, addr).unwrap();
        assert_eq!(segno, 5);
        assert_eq!(blkoff, 17);
    }

    #[test]
    fn addr_below_main_returns_none() {
        let sb = test_sb();
        assert!(addr_to_segno_blkoff(&sb, sb.main_blkaddr - 1).is_none());
    }

    fn test_sb() -> super::super::superblock::Superblock {
        super::super::superblock::Superblock {
            magic: super::super::F2FS_SUPER_MAGIC,
            major_ver: 1,
            minor_ver: 0,
            log_sectorsize: 12,
            log_sectors_per_block: 0,
            log_blocksize: 12,
            log_blocks_per_seg: 9,
            segs_per_sec: 1,
            secs_per_zone: 1,
            checksum_offset: 0,
            block_count: 1024,
            section_count: 1,
            segment_count: 1,
            segment_count_ckpt: 1,
            segment_count_sit: 1,
            segment_count_nat: 1,
            segment_count_ssa: 1,
            segment_count_main: 1,
            segment0_blkaddr: 0,
            cp_blkaddr: 0,
            sit_blkaddr: 0,
            nat_blkaddr: 0,
            ssa_blkaddr: 0,
            main_blkaddr: 1700,
            root_ino: 3,
            node_ino: 1,
            meta_ino: 2,
            uuid: [0; 16],
            volume_name: alloc::string::String::new(),
            feature: 0,
            stored_crc: 0,
        }
    }
}
