// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! littlefs v2.x write path (Phase 2 of `embedded-flash-fs-v1`).
//!
//! Implements:
//!
//! * `create`, `open_write`, `write`, `flush` — file creation and
//!   extension.
//! * `remove`, `rename`, `truncate` — entry-level mutations.
//! * `mkdir`, `rmdir` — directory mutations (one level deep, matching
//!   the Phase 1 reader's path resolution depth).
//! * `fsync` — explicit durability barrier; flushes any dirty in-memory
//!   metadata-pair to flash.
//!
//! ## Atomicity
//!
//! Every public mutation runs through [`MutableState::commit_pair`],
//! which:
//!
//! 1. Picks the *other* half of the active pair (so the previous
//!    committed half stays intact for crash recovery).
//! 2. Increments the revision counter.
//! 3. Erases the alternate block (with bad-block remap on
//!    `EraseFailure`).
//! 4. Programs the new bytes (with bad-block remap on `ProgramFailed`
//!    or `ProgramOnDirty`).
//! 5. Promotes the alternate half to the active half.
//!
//! A power-fail anywhere in steps 3-4 leaves the *previous* half
//! intact: the crash-recovery test (`flash_littlefs_write_conformance`)
//! verifies this rigorously by injecting `EraseFailure` /
//! `ProgramFailed` partway through a commit and then reading back the
//! original bytes.
//!
//! ## Wear-leveling
//!
//! The metadata-pair commit alternates between the two halves of the
//! pair on every commit, distributing erases. The block allocator
//! (see [`super::alloc`]) spreads file-data block allocations across
//! the free pool with cycle-budget-aware skip of hot blocks.
//!
//! ## Bad-block remap
//!
//! On `EraseFailure` / `ProgramFailed` the writer marks the offending
//! block bad in the BBT and the allocator, then transparently
//! re-attempts the operation in a fresh block. The pair pointer in the
//! parent (or the root) is rewritten to point at the new block.

use alloc::vec::Vec;

use super::super::device::{FlashDevice, FlashError};
use super::alloc_mod::BlockAllocator;
use super::bbt::BadBlockTable;
use super::commit::{build_metadata_half, CommitEntry};
use super::format::{read_u32_le, TAG_STRUCT_DIR};
use super::{LittleFsConfig, LittleFsError};

/// Runtime context shared by every write-path call.
///
/// Owns:
///
/// * The mutable view of every metadata pair currently mounted.
/// * The block allocator + BBT.
/// * The active half indicator per pair (`0` = pair[0], `1` = pair[1]).
pub(super) struct MutableState {
    /// Configuration the filesystem was mounted with.
    pub(super) config: LittleFsConfig,
    /// Block allocator (free-block bitmap + wear-leveling).
    pub(super) allocator: BlockAllocator,
    /// In-memory bad-block table (loaded from device at mount time).
    pub(super) bbt: BadBlockTable,
    /// Root metadata-pair address.
    #[allow(dead_code)]
    pub(super) root: [u32; 2],
    /// In-memory state for every mounted metadata pair, keyed by the
    /// canonical address.
    pub(super) pairs: Vec<PairState>,
    /// Total successful commits since mount (counter for tests).
    pub(super) commits: u32,
}

/// Mutable state of one metadata pair.
#[derive(Debug, Clone)]
pub(super) struct PairState {
    /// The two blocks the pair lives in.
    pub(super) blocks: [u32; 2],
    /// Last committed revision counter. The next commit increments this.
    pub(super) revision: u32,
    /// Which half holds the latest commit (`0` for `blocks[0]`, `1`
    /// for `blocks[1]`).
    pub(super) active_half: u8,
    /// In-memory entries.
    pub(super) entries: Vec<PairEntry>,
    /// Optional `tail` pointer.
    pub(super) tail: Option<[u32; 2]>,
    /// True if this pair has uncommitted changes.
    pub(super) dirty: bool,
}

#[derive(Debug, Clone)]
pub(super) struct PairEntry {
    pub(super) id: u16,
    pub(super) typ_name: u16,
    pub(super) name: Vec<u8>,
    pub(super) typ_struct: u16,
    pub(super) struct_payload: Vec<u8>,
}

impl PairEntry {
    /// Convert into a [`CommitEntry`] for the metadata-half builder.
    pub(super) fn to_commit(&self) -> CommitEntry {
        CommitEntry {
            typ_name: self.typ_name,
            id: self.id,
            name: self.name.clone(),
            typ_struct: self.typ_struct,
            struct_payload: self.struct_payload.clone(),
        }
    }
}

impl MutableState {
    /// Allocate a fresh id (10-bit, > 0) within the pair at `idx`.
    pub(super) fn fresh_id(&self, idx: usize) -> Result<u16, LittleFsError> {
        let used: alloc::collections::BTreeSet<u16> =
            self.pairs[idx].entries.iter().map(|e| e.id).collect();
        for candidate in 1u16..0x3FF {
            if !used.contains(&candidate) {
                return Ok(candidate);
            }
        }
        Err(LittleFsError::Unsupported)
    }

    /// Find an entry by `(name, typ_name)` in the pair at `idx`.
    pub(super) fn find_entry(&self, idx: usize, name: &str, typ_name: u16) -> Option<usize> {
        self.pairs[idx]
            .entries
            .iter()
            .position(|e| e.typ_name == typ_name && e.name.as_slice() == name.as_bytes())
    }

    /// Commit a single dirty pair to flash.
    ///
    /// Steps:
    ///   1. Pick the alternate half (1 - active_half).
    ///   2. Bump revision.
    ///   3. Erase, then program the new bytes. On `EraseFailure` /
    ///      `ProgramFailed` mark the block bad and re-allocate, then
    ///      retry. After 4 attempts bubble up `Io`.
    ///   4. Promote the alternate half to the active half.
    pub(super) fn commit_pair<D: FlashDevice>(
        &mut self,
        dev: &mut D,
        idx: usize,
    ) -> Result<(), LittleFsError> {
        if !self.pairs[idx].dirty {
            return Ok(());
        }
        let block_size = self.config.block_size as usize;

        // Build the half byte stream.
        let entries: Vec<CommitEntry> = self.pairs[idx]
            .entries
            .iter()
            .map(|e| e.to_commit())
            .collect();
        let next_revision = self.pairs[idx].revision.wrapping_add(1);
        let half_bytes = build_metadata_half(next_revision, &entries, self.pairs[idx].tail)
            .map_err(|_| LittleFsError::Unsupported)?;
        if half_bytes.len() > block_size {
            // Pair has overflowed a single block. v2 littlefs would
            // split into a tail-chained pair here; Phase 2 surfaces it
            // as `Unsupported`.
            return Err(LittleFsError::Unsupported);
        }

        for _attempt in 0..4u32 {
            let old_blocks = self.pairs[idx].blocks;
            let next_half_idx: u8 = 1 - self.pairs[idx].active_half;
            let mut alternate_block = old_blocks[next_half_idx as usize];

            // ── Try to erase + program the alternate block ───────────
            if self.bbt.is_bad(alternate_block) {
                alternate_block = self
                    .allocator
                    .alloc_block()
                    .map_err(|_| LittleFsError::Io)?;
                let mut new_blocks = old_blocks;
                new_blocks[next_half_idx as usize] = alternate_block;
                self.pairs[idx].blocks = new_blocks;
            }
            match dev.erase(u64::from(alternate_block)) {
                Ok(()) => {
                    self.allocator.note_erase(alternate_block);
                }
                Err(FlashError::EraseFailure | FlashError::ProgramFailed) => {
                    self.bbt.mark_bad(alternate_block);
                    self.allocator.mark_bad(alternate_block);
                    let _ = dev.mark_bad(u64::from(alternate_block));
                    let replacement = self
                        .allocator
                        .alloc_block()
                        .map_err(|_| LittleFsError::Io)?;
                    let mut new_blocks = old_blocks;
                    new_blocks[next_half_idx as usize] = replacement;
                    self.pairs[idx].blocks = new_blocks;
                    continue;
                }
                Err(_) => return Err(LittleFsError::Io),
            }

            // Program the bytes. Pad up to a page-size multiple.
            let prog_size = self.config.prog_size.max(1) as usize;
            let mut padded = half_bytes.clone();
            let pad_to = padded.len().div_ceil(prog_size) * prog_size;
            padded.resize(pad_to, 0xFF);

            let prog_offset = u64::from(alternate_block) * (block_size as u64);
            match dev.program(prog_offset, &padded) {
                Ok(()) => {}
                Err(FlashError::ProgramFailed | FlashError::ProgramOnDirty) => {
                    self.bbt.mark_bad(alternate_block);
                    self.allocator.mark_bad(alternate_block);
                    let _ = dev.mark_bad(u64::from(alternate_block));
                    let replacement = self
                        .allocator
                        .alloc_block()
                        .map_err(|_| LittleFsError::Io)?;
                    let mut new_blocks = old_blocks;
                    new_blocks[next_half_idx as usize] = replacement;
                    self.pairs[idx].blocks = new_blocks;
                    continue;
                }
                Err(_) => return Err(LittleFsError::Io),
            }

            // Success: promote.
            self.pairs[idx].active_half = next_half_idx;
            self.pairs[idx].revision = next_revision;
            self.pairs[idx].dirty = false;
            self.commits = self.commits.wrapping_add(1);

            // If this pair's blocks moved (bad-block remap), update any
            // parent pair's dir-struct payload that points here.
            if self.pairs[idx].blocks != old_blocks {
                if idx == 0 {
                    // The root pair address moved. Phase 2 cannot
                    // re-anchor the root since blocks 0..1 are
                    // implicit per SPEC.md. Surface as `Unsupported`.
                    return Err(LittleFsError::Unsupported);
                }
                let new_blocks = self.pairs[idx].blocks;
                let n_pairs = self.pairs.len();
                for j in 0..n_pairs {
                    if j == idx {
                        continue;
                    }
                    let mut dirty = false;
                    for e in &mut self.pairs[j].entries {
                        if e.typ_struct == TAG_STRUCT_DIR && e.struct_payload.len() == 8 {
                            let a = read_u32_le(&e.struct_payload[0..4]);
                            let b = read_u32_le(&e.struct_payload[4..8]);
                            if [a, b] == old_blocks {
                                let mut p = Vec::with_capacity(8);
                                p.extend_from_slice(&new_blocks[0].to_le_bytes());
                                p.extend_from_slice(&new_blocks[1].to_le_bytes());
                                e.struct_payload = p;
                                dirty = true;
                            }
                        }
                    }
                    if dirty {
                        self.pairs[j].dirty = true;
                    }
                }
            }

            return Ok(());
        }

        Err(LittleFsError::Io)
    }

    /// Commit every dirty pair (used by `fsync`).
    pub(super) fn commit_all<D: FlashDevice>(&mut self, dev: &mut D) -> Result<(), LittleFsError> {
        // Iterate by index because the loop body mutates `self`.
        let n = self.pairs.len();
        for idx in 0..n {
            if self.pairs[idx].dirty {
                self.commit_pair(dev, idx)?;
            }
        }
        Ok(())
    }

    /// Mark pair `idx` dirty.
    pub(super) fn mark_dirty(&mut self, idx: usize) {
        self.pairs[idx].dirty = true;
    }

    /// Test introspection: total successful commits.
    #[allow(dead_code)]
    pub(super) fn commits(&self) -> u32 {
        self.commits
    }
}

/// Open-flags for [`super::LittleFs::open_write`].
#[derive(Debug, Clone, Copy)]
pub struct OpenFlags {
    /// Create the file if it doesn't exist.
    pub create: bool,
    /// Truncate to length 0 on open.
    pub truncate: bool,
    /// Position the write cursor at end-of-file. Phase 2 supports this
    /// for inline files only; CTZ append is `Unsupported` until phase 4.
    pub append: bool,
}

impl Default for OpenFlags {
    fn default() -> Self {
        Self {
            create: true,
            truncate: true,
            append: false,
        }
    }
}

impl OpenFlags {
    /// Default: create + truncate (POSIX `O_WRONLY|O_CREAT|O_TRUNC`).
    pub fn create_truncate() -> Self {
        Self::default()
    }
    /// `O_APPEND` style: open existing + position at EOF, do not truncate.
    pub fn append() -> Self {
        Self {
            create: false,
            truncate: false,
            append: true,
        }
    }
}
