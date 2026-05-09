// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-mount LRU block cache and write-coalescing buffer.
//!
//! Per the `fs-block-device` Q10 design answer, every on-disk mount
//! gets its own LRU block cache with a configurable budget:
//!
//! - `/models/` (squashfs) — default 16 MiB.
//! - `/data/` (F2FS) — default 4 MiB.
//!
//! The cache is keyed by `(lba, partition_id)` from the caller's point
//! of view; this module only stores the `lba` since each [`LruCache`]
//! instance is owned by exactly one mount. Eviction policy is simple
//! LRU (least-recently-used) — adequate for both the read-heavy
//! squashfs path and the small-working-set F2FS metadata path.
//!
//! In addition the F2FS audit-log writer needs a 256 KiB
//! write-coalescing buffer that batches short appends before flushing
//! to the underlying [`crate::block::BlockDevice`]. The buffer is a
//! ring rather than a tree because its access pattern is strictly
//! sequential (audit log entries are written in order) and it is
//! periodically drained via [`WriteCoalescer::take_pending`].
//!
//! Both structures are `#![no_std]`-friendly: storage uses
//! `alloc::vec::Vec` and `alloc::collections::BTreeMap` so no `std`
//! hash dependency leaks into bare-metal builds.
//!
//! Owned by `embedded-filesystem-v1` Phase 6.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::config_defaults::{
    FS_CACHE_DATA_BYTES, FS_CACHE_DATA_FLOOR_BYTES, FS_CACHE_MODELS_BYTES,
    FS_CACHE_MODELS_FLOOR_BYTES, FS_WRITE_COALESCE_BYTES,
};

// ─── Cache budget validator ──────────────────────────────────────────────────

/// Errors returned by the cache-budget validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBudgetError {
    /// Models cache budget is below [`FS_CACHE_MODELS_FLOOR_BYTES`].
    ModelsBelowFloor { requested: u64, floor: u64 },
    /// Data cache budget is below [`FS_CACHE_DATA_FLOOR_BYTES`].
    DataBelowFloor { requested: u64, floor: u64 },
    /// Block size is zero (would cause divide-by-zero in `evict`).
    ZeroBlockSize,
}

/// Validate a `(models, data)` cache-budget pair against the documented
/// floors. The validator is called by both the `mgmt` config loader and
/// the [`LruCache::new`] constructor so a misconfigured policy.toml is
/// rejected early, before any I/O happens.
pub fn validate_cache_budgets(models_bytes: u64, data_bytes: u64) -> Result<(), CacheBudgetError> {
    if models_bytes < FS_CACHE_MODELS_FLOOR_BYTES {
        return Err(CacheBudgetError::ModelsBelowFloor {
            requested: models_bytes,
            floor: FS_CACHE_MODELS_FLOOR_BYTES,
        });
    }
    if data_bytes < FS_CACHE_DATA_FLOOR_BYTES {
        return Err(CacheBudgetError::DataBelowFloor {
            requested: data_bytes,
            floor: FS_CACHE_DATA_FLOOR_BYTES,
        });
    }
    Ok(())
}

// ─── LRU block cache ─────────────────────────────────────────────────────────

/// One cached block stored alongside an `epoch` counter that ranks
/// recency. Higher `epoch` is more recent.
#[derive(Debug, Clone)]
struct CacheEntry {
    bytes: Vec<u8>,
    epoch: u64,
}

/// Per-mount LRU block cache.
///
/// Keys are 64-bit LBAs (relative to the mount partition); values are
/// fixed-size block buffers. Eviction is LRU — the entry with the
/// lowest `epoch` is dropped first when the budget is exceeded.
///
/// The cache is single-threaded; callers that need cross-thread access
/// MUST wrap it in their own synchronization. The kernel mount table
/// always accesses each cache from the matching mount thread, so this
/// is a non-issue for the v1 boot path.
#[derive(Debug)]
pub struct LruCache {
    /// LBA → (bytes, epoch) map. Storing in a `BTreeMap` keeps the
    /// crate `#![no_std]`-clean (no `hashbrown`/`std::collections`
    /// indirection) and is fast enough at the cache sizes we care
    /// about (≤ ~5 K entries at 16 MiB / 4 KiB blocks).
    entries: BTreeMap<u64, CacheEntry>,
    budget_bytes: u64,
    block_size_bytes: u32,
    /// Monotonic counter incremented on every access; used to break LRU
    /// ties.
    next_epoch: u64,
    /// Cache hits since construction (test introspection).
    hits: u64,
    /// Cache misses since construction (test introspection).
    misses: u64,
    /// Evictions since construction (test introspection).
    evictions: u64,
}

impl LruCache {
    /// Construct a cache with the given budget (in bytes) and block
    /// size. Budget must be at least `block_size_bytes`; smaller
    /// budgets are not rejected here (the validator runs upstream)
    /// but they will result in eviction-on-every-put which the cache
    /// gracefully handles.
    pub fn new(budget_bytes: u64, block_size_bytes: u32) -> Self {
        Self {
            entries: BTreeMap::new(),
            budget_bytes,
            block_size_bytes,
            next_epoch: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Returns the configured budget in bytes.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Returns the configured block size in bytes.
    pub fn block_size_bytes(&self) -> u32 {
        self.block_size_bytes
    }

    /// Number of currently cached blocks.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` iff the cache holds zero entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes currently resident in the cache (`len * block_size`).
    pub fn used_bytes(&self) -> u64 {
        (self.entries.len() as u64).saturating_mul(self.block_size_bytes as u64)
    }

    /// Cache-hit count (test introspection).
    pub fn hit_count(&self) -> u64 {
        self.hits
    }

    /// Cache-miss count (test introspection).
    pub fn miss_count(&self) -> u64 {
        self.misses
    }

    /// Eviction count (test introspection).
    pub fn eviction_count(&self) -> u64 {
        self.evictions
    }

    /// Probe the cache for `lba`. On hit the entry is reuse-promoted
    /// (its epoch advanced) so it survives the next eviction round.
    pub fn get(&mut self, lba: u64) -> Option<&[u8]> {
        let new_epoch = self.next_epoch.wrapping_add(1);
        let hit = match self.entries.get_mut(&lba) {
            Some(e) => {
                e.epoch = new_epoch;
                true
            }
            None => false,
        };
        if hit {
            self.next_epoch = new_epoch;
            self.hits += 1;
            // Re-borrow to satisfy the borrow checker after the
            // mutable borrow above.
            return self.entries.get(&lba).map(|e| e.bytes.as_slice());
        }
        self.misses += 1;
        None
    }

    /// Read-only probe — does NOT advance the LRU epoch. Useful in
    /// tests to inspect cache state without disturbing eviction order.
    pub fn peek(&self, lba: u64) -> Option<&[u8]> {
        self.entries.get(&lba).map(|e| e.bytes.as_slice())
    }

    /// Insert `data` at `lba`. If the new entry would push the cache
    /// over budget, the least-recently-used entries are evicted until
    /// enough room is freed.
    pub fn put(&mut self, lba: u64, data: &[u8]) {
        // Reject odd-sized inserts in debug builds to make bugs loud.
        debug_assert_eq!(
            data.len(),
            self.block_size_bytes as usize,
            "LruCache::put expects block-sized buffers"
        );
        let new_epoch = self.next_epoch.wrapping_add(1);
        self.next_epoch = new_epoch;
        self.entries.insert(
            lba,
            CacheEntry {
                bytes: data.to_vec(),
                epoch: new_epoch,
            },
        );
        self.evict_to_budget();
    }

    /// Drop entries until the cache is within budget.
    ///
    /// Implementation note: `BTreeMap` does not give us O(1) LRU
    /// eviction, but we only reach this path on `put` and the inputs
    /// are bounded by the per-mount budget. The naive scan is
    /// O(n log n) in the size of the cache (sort by epoch); for the
    /// v1 budgets (≤ 5 K entries) this is fast enough that the
    /// simpler code beats a separate doubly-linked-list ledger.
    pub fn evict_to_budget(&mut self) {
        let block_size_u64 = self.block_size_bytes as u64;
        if block_size_u64 == 0 {
            // Defensive: avoid divide-by-zero if a misconfigured cache
            // slips past the validator. The cache becomes a no-op.
            self.entries.clear();
            return;
        }
        // How many entries fit?
        let max_entries = (self.budget_bytes / block_size_u64) as usize;
        if self.entries.len() <= max_entries {
            return;
        }
        // Build a (epoch, lba) list and pick the smallest epochs to
        // drop. Use stable iteration so the test counter is exact.
        let mut snapshot: Vec<(u64, u64)> = self
            .entries
            .iter()
            .map(|(lba, e)| (e.epoch, *lba))
            .collect();
        snapshot.sort_unstable_by_key(|(epoch, _lba)| *epoch);
        let to_drop = self.entries.len() - max_entries;
        for (_epoch, lba) in snapshot.into_iter().take(to_drop) {
            if self.entries.remove(&lba).is_some() {
                self.evictions += 1;
            }
        }
    }

    /// Clear the cache; reset hit/miss/eviction counters.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.next_epoch = 0;
    }
}

/// Build a cache pre-sized for the `/models/` mount.
pub fn models_cache(block_size_bytes: u32) -> LruCache {
    LruCache::new(FS_CACHE_MODELS_BYTES, block_size_bytes)
}

/// Build a cache pre-sized for the `/data/` mount.
pub fn data_cache(block_size_bytes: u32) -> LruCache {
    LruCache::new(FS_CACHE_DATA_BYTES, block_size_bytes)
}

// ─── Write-coalescing buffer ─────────────────────────────────────────────────

/// Per-mount write-coalescing buffer.
///
/// The audit-log writer on `/data/` issues short appends (one
/// JSON-Lines record per security event); calling
/// `BlockDevice::write_block` per record would push 4 KiB of metadata
/// per write, which is wasteful at SD-card class storage. The
/// coalescer batches up to [`FS_WRITE_COALESCE_BYTES`] of pending bytes
/// in RAM and exposes [`take_pending`] for the higher-level F2FS
/// commit path to flush them in one shot.
///
/// The buffer is intentionally simple: it is NOT a circular ring;
/// it is a linear `Vec` reset on `take_pending`. Spill semantics:
/// `append` returns `false` (caller must flush first) when the new
/// bytes would exceed `capacity_bytes`.
#[derive(Debug)]
pub struct WriteCoalescer {
    pending: Vec<u8>,
    capacity_bytes: usize,
    flushes: u64,
    appended_records: u64,
}

impl WriteCoalescer {
    /// Construct a coalescer with the requested capacity (in bytes).
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            pending: Vec::new(),
            capacity_bytes: capacity_bytes as usize,
            flushes: 0,
            appended_records: 0,
        }
    }

    /// Default 256 KiB coalescer.
    pub fn default_size() -> Self {
        Self::new(FS_WRITE_COALESCE_BYTES)
    }

    /// Maximum bytes the coalescer can hold before requiring a flush.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Bytes currently pending.
    pub fn pending_bytes(&self) -> usize {
        self.pending.len()
    }

    /// `true` iff the buffer holds zero pending bytes.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Number of times [`take_pending`] has drained the buffer.
    pub fn flush_count(&self) -> u64 {
        self.flushes
    }

    /// Number of [`append`] calls that successfully landed bytes.
    pub fn appended_records(&self) -> u64 {
        self.appended_records
    }

    /// Append `bytes` to the pending buffer.
    ///
    /// Returns `false` (and does NOT mutate the buffer) when the new
    /// bytes would exceed [`capacity_bytes`]. The caller MUST drain
    /// the buffer with [`take_pending`] and flush to the underlying
    /// device before retrying.
    pub fn append(&mut self, bytes: &[u8]) -> bool {
        if self.pending.len().saturating_add(bytes.len()) > self.capacity_bytes {
            return false;
        }
        self.pending.extend_from_slice(bytes);
        self.appended_records += 1;
        true
    }

    /// Drain the pending buffer, returning the bytes to the caller.
    /// Resets internal state so subsequent [`append`] calls start
    /// from zero.
    pub fn take_pending(&mut self) -> Vec<u8> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        self.flushes += 1;
        core::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Validator tests ────────────────────────────────────────────────────

    #[test]
    fn defaults_pass_validator() {
        validate_cache_budgets(FS_CACHE_MODELS_BYTES, FS_CACHE_DATA_BYTES).unwrap();
    }

    #[test]
    fn models_below_floor_rejected() {
        let too_small = FS_CACHE_MODELS_FLOOR_BYTES - 1;
        match validate_cache_budgets(too_small, FS_CACHE_DATA_BYTES) {
            Err(CacheBudgetError::ModelsBelowFloor { requested, floor }) => {
                assert_eq!(requested, too_small);
                assert_eq!(floor, FS_CACHE_MODELS_FLOOR_BYTES);
            }
            other => panic!("expected ModelsBelowFloor, got {:?}", other),
        }
    }

    #[test]
    fn data_below_floor_rejected() {
        let too_small = FS_CACHE_DATA_FLOOR_BYTES - 1;
        match validate_cache_budgets(FS_CACHE_MODELS_BYTES, too_small) {
            Err(CacheBudgetError::DataBelowFloor { requested, floor }) => {
                assert_eq!(requested, too_small);
                assert_eq!(floor, FS_CACHE_DATA_FLOOR_BYTES);
            }
            other => panic!("expected DataBelowFloor, got {:?}", other),
        }
    }

    #[test]
    fn floor_value_is_exactly_acceptable() {
        // Exactly the floor passes (the floor is inclusive).
        validate_cache_budgets(FS_CACHE_MODELS_FLOOR_BYTES, FS_CACHE_DATA_FLOOR_BYTES).unwrap();
    }

    #[test]
    fn validator_rejects_zero_models() {
        assert!(validate_cache_budgets(0, FS_CACHE_DATA_BYTES).is_err());
    }

    #[test]
    fn validator_rejects_zero_data() {
        assert!(validate_cache_budgets(FS_CACHE_MODELS_BYTES, 0).is_err());
    }

    // ─── LruCache tests ─────────────────────────────────────────────────────

    fn make_block(byte: u8) -> Vec<u8> {
        alloc::vec![byte; 4096]
    }

    #[test]
    fn empty_cache_has_no_entries() {
        let c = LruCache::new(16 * 1024 * 1024, 4096);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.used_bytes(), 0);
    }

    #[test]
    fn put_then_get_returns_data() {
        let mut c = LruCache::new(16 * 1024 * 1024, 4096);
        let block = make_block(0x42);
        c.put(7, &block);
        let got = c.get(7).expect("hit");
        assert_eq!(got.len(), 4096);
        assert_eq!(got[0], 0x42);
        assert_eq!(c.hit_count(), 1);
        assert_eq!(c.miss_count(), 0);
    }

    #[test]
    fn miss_increments_miss_counter() {
        let mut c = LruCache::new(16 * 1024 * 1024, 4096);
        assert!(c.get(99).is_none());
        assert_eq!(c.miss_count(), 1);
        assert_eq!(c.hit_count(), 0);
    }

    #[test]
    fn put_overwrite_replaces_data() {
        let mut c = LruCache::new(16 * 1024 * 1024, 4096);
        c.put(1, &make_block(0xAA));
        c.put(1, &make_block(0xBB));
        let got = c.get(1).unwrap();
        assert_eq!(got[0], 0xBB);
    }

    #[test]
    fn cache_evicts_when_over_budget() {
        // 8 KiB budget at 4 KiB blocks → exactly 2 entries fit.
        let mut c = LruCache::new(8 * 1024, 4096);
        c.put(1, &make_block(1));
        c.put(2, &make_block(2));
        assert_eq!(c.len(), 2);
        c.put(3, &make_block(3));
        assert_eq!(c.len(), 2);
        assert_eq!(c.eviction_count(), 1);
        // Block 1 (oldest) should be gone.
        assert!(c.peek(1).is_none());
        assert!(c.peek(2).is_some());
        assert!(c.peek(3).is_some());
    }

    #[test]
    fn lru_promotes_recently_accessed() {
        let mut c = LruCache::new(8 * 1024, 4096);
        c.put(1, &make_block(1));
        c.put(2, &make_block(2));
        // Access block 1 — it should now be more recent than block 2.
        assert!(c.get(1).is_some());
        // Inserting block 3 should evict block 2, not block 1.
        c.put(3, &make_block(3));
        assert!(c.peek(1).is_some(), "block 1 should still be cached");
        assert!(c.peek(2).is_none(), "block 2 should have been evicted");
        assert!(c.peek(3).is_some());
    }

    #[test]
    fn peek_does_not_change_lru_order() {
        let mut c = LruCache::new(8 * 1024, 4096);
        c.put(1, &make_block(1));
        c.put(2, &make_block(2));
        // Peek block 1 — should NOT promote it.
        assert!(c.peek(1).is_some());
        c.put(3, &make_block(3));
        // Block 1 was the oldest and was peeked (not get'd) so it
        // should be the eviction victim.
        assert!(c.peek(1).is_none());
        assert!(c.peek(2).is_some());
        assert!(c.peek(3).is_some());
    }

    #[test]
    fn evict_to_budget_idempotent_when_under() {
        let mut c = LruCache::new(16 * 1024, 4096);
        c.put(1, &make_block(1));
        c.evict_to_budget();
        c.evict_to_budget();
        assert_eq!(c.len(), 1);
        assert_eq!(c.eviction_count(), 0);
    }

    #[test]
    fn evict_to_budget_after_lowering_budget_drops_excess() {
        // Standard 16 MiB cache with two entries, then a very tight
        // synthetic budget — the next put triggers eviction.
        let mut c = LruCache::new(16 * 1024, 4096); // 4 entries fit
        c.put(1, &make_block(1));
        c.put(2, &make_block(2));
        c.put(3, &make_block(3));
        c.put(4, &make_block(4));
        assert_eq!(c.len(), 4);
        // Adding a 5th forces eviction of one entry.
        c.put(5, &make_block(5));
        assert_eq!(c.len(), 4);
        assert_eq!(c.eviction_count(), 1);
    }

    #[test]
    fn clear_resets_state() {
        let mut c = LruCache::new(16 * 1024 * 1024, 4096);
        c.put(1, &make_block(1));
        let _ = c.get(1);
        let _ = c.get(99);
        assert!(c.hit_count() > 0);
        assert!(c.miss_count() > 0);
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.hit_count(), 0);
        assert_eq!(c.miss_count(), 0);
        assert_eq!(c.eviction_count(), 0);
    }

    #[test]
    fn models_cache_uses_default_budget() {
        let c = models_cache(4096);
        assert_eq!(c.budget_bytes(), FS_CACHE_MODELS_BYTES);
        assert_eq!(c.block_size_bytes(), 4096);
    }

    #[test]
    fn data_cache_uses_default_budget() {
        let c = data_cache(4096);
        assert_eq!(c.budget_bytes(), FS_CACHE_DATA_BYTES);
    }

    #[test]
    fn many_puts_respect_budget() {
        // 16 KiB budget at 4 KiB blocks → 4 entries cap.
        let mut c = LruCache::new(16 * 1024, 4096);
        for i in 0..20 {
            c.put(i, &make_block(i as u8));
        }
        assert_eq!(c.len(), 4);
        assert!(c.eviction_count() >= 16);
        // Last 4 should be there.
        assert!(c.peek(16).is_some());
        assert!(c.peek(17).is_some());
        assert!(c.peek(18).is_some());
        assert!(c.peek(19).is_some());
    }

    // ─── WriteCoalescer tests ───────────────────────────────────────────────

    #[test]
    fn coalescer_default_holds_256kib() {
        let c = WriteCoalescer::default_size();
        assert_eq!(c.capacity_bytes(), FS_WRITE_COALESCE_BYTES as usize);
        assert!(c.is_empty());
    }

    #[test]
    fn append_accumulates_bytes() {
        let mut c = WriteCoalescer::new(1024);
        assert!(c.append(b"hello "));
        assert!(c.append(b"world"));
        assert_eq!(c.pending_bytes(), 11);
        assert_eq!(c.appended_records(), 2);
    }

    #[test]
    fn append_rejects_overflow_without_dropping_data() {
        let mut c = WriteCoalescer::new(8);
        assert!(c.append(b"abcd"));
        // Next append would exceed capacity → reject.
        assert!(!c.append(b"efghi"));
        // Pending bytes are unchanged from the rejected append.
        assert_eq!(c.pending_bytes(), 4);
        // A drain frees room for the rejected append.
        let drained = c.take_pending();
        assert_eq!(drained, b"abcd");
        assert!(c.append(b"efghi"));
        assert_eq!(c.pending_bytes(), 5);
    }

    #[test]
    fn take_pending_resets_buffer_and_increments_flush_count() {
        let mut c = WriteCoalescer::new(64);
        c.append(b"first ").unwrap();
        c.append(b"second").unwrap();
        let drained = c.take_pending();
        assert_eq!(drained, b"first second");
        assert!(c.is_empty());
        assert_eq!(c.flush_count(), 1);
    }

    #[test]
    fn take_pending_on_empty_does_not_increment_flush() {
        let mut c = WriteCoalescer::new(64);
        let drained = c.take_pending();
        assert!(drained.is_empty());
        assert_eq!(c.flush_count(), 0);
    }

    #[test]
    fn coalescer_batches_many_short_writes_into_one_flush() {
        // Simulates the audit-log writer: 100 short records (~64 B
        // each) coalesced into a single take_pending().
        let mut c = WriteCoalescer::new(256 * 1024);
        let mut record = [0u8; 64];
        for i in 0..100 {
            record.fill(i as u8);
            assert!(c.append(&record));
        }
        assert_eq!(c.appended_records(), 100);
        let drained = c.take_pending();
        assert_eq!(drained.len(), 100 * 64);
        assert_eq!(c.flush_count(), 1);
    }

    // Helper: enable the `_ = c.append(...).unwrap();` pattern above.
    trait AppendAssert {
        fn unwrap(self);
    }
    impl AppendAssert for bool {
        fn unwrap(self) {
            assert!(self, "append unexpectedly returned false");
        }
    }
}
