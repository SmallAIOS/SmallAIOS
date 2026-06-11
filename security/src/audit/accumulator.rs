// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Audit batch accumulator.
//!
//! Collects audit entries and seals them into integrity-chained batches
//! when either the entry count reaches 256 or a time threshold elapses.
//! Sealed batches are ready for signing and export.

use alloc::vec::Vec;

use super::entry::AuditEntry;
use super::integrity::{AuditBatch, IntegrityChain};

/// Maximum entries per batch before forced seal.
pub const BATCH_MAX_ENTRIES: usize = 256;

/// Default batch timeout in nanoseconds (1 second).
pub const BATCH_TIMEOUT_NS: u64 = 1_000_000_000;

/// Accumulates audit entries and produces sealed batches.
pub struct BatchAccumulator {
    /// Pending entries not yet sealed into a batch.
    pending: Vec<AuditEntry>,
    /// Timestamp (ns) when the current batch was opened.
    batch_opened_ns: u64,
    /// Batch timeout in nanoseconds.
    timeout_ns: u64,
    /// Integrity chain for hash-linking batches.
    chain: IntegrityChain,
    /// Total batches sealed.
    batches_sealed: u64,
}

impl BatchAccumulator {
    /// Create a new accumulator with default timeout (1 second).
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            batch_opened_ns: 0,
            timeout_ns: BATCH_TIMEOUT_NS,
            chain: IntegrityChain::new(),
            batches_sealed: 0,
        }
    }

    /// Create an accumulator with a custom timeout.
    pub fn with_timeout_ns(timeout_ns: u64) -> Self {
        Self {
            pending: Vec::new(),
            batch_opened_ns: 0,
            timeout_ns,
            chain: IntegrityChain::new(),
            batches_sealed: 0,
        }
    }

    /// Add an entry to the accumulator.
    ///
    /// If the batch reaches `BATCH_MAX_ENTRIES`, a sealed batch is returned.
    /// The caller must provide the current time in nanoseconds so the
    /// accumulator can track when the batch was opened.
    pub fn push(&mut self, entry: AuditEntry, now_ns: u64) -> Option<AuditBatch> {
        if self.pending.is_empty() {
            self.batch_opened_ns = now_ns;
        }
        self.pending.push(entry);

        if self.pending.len() >= BATCH_MAX_ENTRIES {
            Some(self.seal())
        } else {
            None
        }
    }

    /// Check if the timeout has elapsed and seal the batch if so.
    ///
    /// Returns None if no entries have accumulated or if the timeout
    /// has not elapsed. Empty batches are suppressed per spec.
    pub fn tick(&mut self, now_ns: u64) -> Option<AuditBatch> {
        if self.pending.is_empty() {
            // Empty batch suppression: reset timer, produce nothing
            self.batch_opened_ns = now_ns;
            return None;
        }

        if now_ns.saturating_sub(self.batch_opened_ns) >= self.timeout_ns {
            Some(self.seal())
        } else {
            None
        }
    }

    /// Force-seal the current batch regardless of count or time.
    ///
    /// Returns None if no entries are pending.
    pub fn flush(&mut self) -> Option<AuditBatch> {
        if self.pending.is_empty() {
            return None;
        }
        Some(self.seal())
    }

    /// Number of pending (unsealed) entries.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Total batches sealed so far.
    pub fn batches_sealed(&self) -> u64 {
        self.batches_sealed
    }

    /// The configured timeout in nanoseconds.
    pub fn timeout_ns(&self) -> u64 {
        self.timeout_ns
    }

    fn seal(&mut self) -> AuditBatch {
        let entries = core::mem::take(&mut self.pending);
        let batch = self.chain.seal_batch(entries);
        self.batches_sealed += 1;
        batch
    }
}

impl Default for BatchAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::entry::{AuditResult, Operation};
    use crate::audit::integrity::verify_batch_hash;
    use crate::audit::taxonomy::AuditEventType;

    fn make_entry(ts: u64) -> AuditEntry {
        AuditEntry {
            timestamp_ns: ts,
            event_type: AuditEventType::capability_grant(),
            task_id: 1,
            resource_type: 0,
            resource_instance: 0,
            operation: Operation::new("test"),
            result: AuditResult::Success,
            capability_id: 0,
        }
    }

    #[test]
    fn push_under_threshold_no_batch() {
        let mut acc = BatchAccumulator::new();
        for i in 0..255 {
            let result = acc.push(make_entry(i), i);
            assert!(result.is_none());
        }
        assert_eq!(acc.pending_count(), 255);
    }

    #[test]
    fn push_at_threshold_seals_batch() {
        let mut acc = BatchAccumulator::new();
        for i in 0..255 {
            assert!(acc.push(make_entry(i), i).is_none());
        }
        let batch = acc.push(make_entry(255), 255);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.entries.len(), 256);
        assert!(verify_batch_hash(&batch));
        assert_eq!(acc.pending_count(), 0);
        assert_eq!(acc.batches_sealed(), 1);
    }

    #[test]
    fn tick_before_timeout_no_seal() {
        let mut acc = BatchAccumulator::new();
        acc.push(make_entry(0), 0);
        // 500ms elapsed, timeout is 1s
        let result = acc.tick(500_000_000);
        assert!(result.is_none());
        assert_eq!(acc.pending_count(), 1);
    }

    #[test]
    fn tick_at_timeout_seals() {
        let mut acc = BatchAccumulator::new();
        acc.push(make_entry(0), 0);
        acc.push(make_entry(100), 100);
        // 1 second elapsed
        let batch = acc.tick(1_000_000_000);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.entries.len(), 2);
        assert!(verify_batch_hash(&batch));
        assert_eq!(acc.pending_count(), 0);
    }

    #[test]
    fn tick_empty_suppressed() {
        let mut acc = BatchAccumulator::new();
        // No entries, timeout elapsed: should not produce a batch
        let result = acc.tick(2_000_000_000);
        assert!(result.is_none());
        assert_eq!(acc.batches_sealed(), 0);
    }

    #[test]
    fn flush_pending() {
        let mut acc = BatchAccumulator::new();
        acc.push(make_entry(0), 0);
        acc.push(make_entry(1), 1);
        let batch = acc.flush();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().entries.len(), 2);
        assert_eq!(acc.pending_count(), 0);
    }

    #[test]
    fn flush_empty_returns_none() {
        let mut acc = BatchAccumulator::new();
        assert!(acc.flush().is_none());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "seals two full 256-entry batches (~2.5 min under Miri's interpreter); UB coverage comes from the single-batch accumulator tests / native runs"
    )]
    fn multiple_batches_chain_correctly() {
        let mut acc = BatchAccumulator::new();
        let mut batches = alloc::vec::Vec::new();

        // Fill and seal 3 batches via count threshold
        for batch_num in 0..3 {
            for i in 0..256 {
                let ts = batch_num * 256 + i;
                if let Some(batch) = acc.push(make_entry(ts as u64), ts as u64) {
                    batches.push(batch);
                }
            }
        }

        assert_eq!(batches.len(), 3);
        // Verify chain linkage
        assert_eq!(
            batches[0].previous_hash,
            crate::audit::integrity::GENESIS_HASH
        );
        assert_eq!(batches[1].previous_hash, batches[0].batch_hash);
        assert_eq!(batches[2].previous_hash, batches[1].batch_hash);

        // Verify full chain
        assert_eq!(crate::audit::integrity::verify_chain(&batches), None);
    }

    #[test]
    fn custom_timeout() {
        let mut acc = BatchAccumulator::with_timeout_ns(100_000);
        assert_eq!(acc.timeout_ns(), 100_000);
        acc.push(make_entry(0), 0);
        // Not yet at 100us
        assert!(acc.tick(50_000).is_none());
        // At 100us
        assert!(acc.tick(100_000).is_some());
    }

    #[test]
    fn mixed_seal_by_count_and_timeout() {
        let mut acc = BatchAccumulator::new();
        let mut batches = alloc::vec::Vec::new();

        // Seal by count
        for i in 0..256 {
            if let Some(batch) = acc.push(make_entry(i as u64), i as u64) {
                batches.push(batch);
            }
        }
        assert_eq!(batches.len(), 1);

        // Add a few entries, then seal by timeout
        acc.push(make_entry(300), 300);
        acc.push(make_entry(400), 400);
        if let Some(batch) = acc.tick(1_000_000_300) {
            batches.push(batch);
        }
        assert_eq!(batches.len(), 2);

        // Chain should still be valid
        assert_eq!(crate::audit::integrity::verify_chain(&batches), None);
    }
}
