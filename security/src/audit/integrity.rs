// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Audit log integrity chain using SHA-3-256.
//!
//! Each batch of audit entries is linked to the previous batch via a hash
//! chain: `batch_hash = SHA-3-256(previous_batch_hash || entry_1 || ... || entry_N)`.
//!
//! The first batch uses a genesis hash of 32 zero bytes. Any modification
//! to a batch invalidates all subsequent batch hashes, providing
//! tamper evidence per NIST AU-9.

use super::entry::AuditEntry;
use crate::crypto::sha3::{Sha3_256, SHA3_256_DIGEST_LEN};

/// Genesis hash: 32 zero bytes used as the "previous hash" for the first batch.
pub const GENESIS_HASH: [u8; SHA3_256_DIGEST_LEN] = [0u8; SHA3_256_DIGEST_LEN];

/// A sealed batch of audit entries with integrity hash.
#[derive(Debug, Clone)]
pub struct AuditBatch {
    /// SHA-3-256 hash of the previous batch (genesis = all zeros).
    pub previous_hash: [u8; SHA3_256_DIGEST_LEN],
    /// The entries in this batch.
    pub entries: alloc::vec::Vec<AuditEntry>,
    /// SHA-3-256(previous_hash || serialized_entries).
    pub batch_hash: [u8; SHA3_256_DIGEST_LEN],
    /// Monotonically increasing batch sequence number.
    pub sequence: u64,
}

/// Maintains the hash chain state for sequential batch sealing.
pub struct IntegrityChain {
    /// Hash of the most recently sealed batch.
    previous_hash: [u8; SHA3_256_DIGEST_LEN],
    /// Next batch sequence number.
    next_sequence: u64,
}

impl IntegrityChain {
    /// Create a new integrity chain starting from the genesis hash.
    pub fn new() -> Self {
        Self {
            previous_hash: GENESIS_HASH,
            next_sequence: 0,
        }
    }

    /// The hash that will be used as `previous_hash` for the next batch.
    pub fn current_hash(&self) -> &[u8; SHA3_256_DIGEST_LEN] {
        &self.previous_hash
    }

    /// Next batch sequence number.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Seal a batch of entries, computing the hash chain.
    ///
    /// Returns the sealed batch with its computed hash. Updates internal
    /// state so the next batch will chain from this one.
    pub fn seal_batch(&mut self, entries: alloc::vec::Vec<AuditEntry>) -> AuditBatch {
        let batch_hash = compute_batch_hash(&self.previous_hash, &entries);
        let sequence = self.next_sequence;

        let batch = AuditBatch {
            previous_hash: self.previous_hash,
            entries,
            batch_hash,
            sequence,
        };

        self.previous_hash = batch_hash;
        self.next_sequence += 1;

        batch
    }
}

impl Default for IntegrityChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute SHA-3-256(previous_hash || serialized_entries).
pub fn compute_batch_hash(
    previous_hash: &[u8; SHA3_256_DIGEST_LEN],
    entries: &[AuditEntry],
) -> [u8; SHA3_256_DIGEST_LEN] {
    let mut hasher = Sha3_256::new();
    hasher
        .update(previous_hash)
        .expect("fresh hasher update cannot fail");

    let mut entry_buf = [0u8; AuditEntry::SERIALIZED_SIZE];
    for entry in entries {
        entry.serialize(&mut entry_buf);
        hasher
            .update(&entry_buf)
            .expect("hasher update cannot fail");
    }

    let digest = hasher.finalize().expect("hasher finalize cannot fail");
    *digest.as_bytes()
}

/// Verify a batch's hash matches the claimed previous hash and entries.
pub fn verify_batch_hash(batch: &AuditBatch) -> bool {
    let recomputed = compute_batch_hash(&batch.previous_hash, &batch.entries);
    recomputed == batch.batch_hash
}

/// Verify hash chain continuity across a sequence of batches.
///
/// Checks that:
/// 1. The first batch's previous_hash equals the genesis hash.
/// 2. Each batch's hash is correctly computed.
/// 3. Each batch's previous_hash equals the preceding batch's batch_hash.
/// 4. Sequence numbers are monotonically increasing from 0.
///
/// Returns the index of the first invalid batch, or None if all are valid.
pub fn verify_chain(batches: &[AuditBatch]) -> Option<usize> {
    if batches.is_empty() {
        return None;
    }

    // First batch must reference genesis
    if batches[0].previous_hash != GENESIS_HASH {
        return Some(0);
    }
    if batches[0].sequence != 0 {
        return Some(0);
    }
    if !verify_batch_hash(&batches[0]) {
        return Some(0);
    }

    for i in 1..batches.len() {
        // Chain linkage
        if batches[i].previous_hash != batches[i - 1].batch_hash {
            return Some(i);
        }
        // Sequence continuity
        if batches[i].sequence != batches[i - 1].sequence + 1 {
            return Some(i);
        }
        // Hash correctness
        if !verify_batch_hash(&batches[i]) {
            return Some(i);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::entry::{AuditResult, Operation};
    use crate::audit::taxonomy::AuditEventType;
    use crate::crypto::sha3::sha3_256;
    use alloc::vec;
    use alloc::vec::Vec;

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
    fn chain_starts_from_genesis() {
        let chain = IntegrityChain::new();
        assert_eq!(*chain.current_hash(), GENESIS_HASH);
        assert_eq!(chain.next_sequence(), 0);
    }

    #[test]
    fn seal_single_batch() {
        let mut chain = IntegrityChain::new();
        let entries = vec![make_entry(100), make_entry(200)];
        let batch = chain.seal_batch(entries);

        assert_eq!(batch.previous_hash, GENESIS_HASH);
        assert_eq!(batch.sequence, 0);
        assert_eq!(batch.entries.len(), 2);
        assert!(verify_batch_hash(&batch));
        assert_eq!(*chain.current_hash(), batch.batch_hash);
        assert_eq!(chain.next_sequence(), 1);
    }

    #[test]
    fn seal_chained_batches() {
        let mut chain = IntegrityChain::new();

        let batch0 = chain.seal_batch(vec![make_entry(100)]);
        let batch1 = chain.seal_batch(vec![make_entry(200)]);
        let batch2 = chain.seal_batch(vec![make_entry(300)]);

        // Chain linkage
        assert_eq!(batch1.previous_hash, batch0.batch_hash);
        assert_eq!(batch2.previous_hash, batch1.batch_hash);

        // Sequence
        assert_eq!(batch0.sequence, 0);
        assert_eq!(batch1.sequence, 1);
        assert_eq!(batch2.sequence, 2);

        // All hashes valid
        assert!(verify_batch_hash(&batch0));
        assert!(verify_batch_hash(&batch1));
        assert!(verify_batch_hash(&batch2));
    }

    #[test]
    fn verify_chain_valid() {
        let mut chain = IntegrityChain::new();
        let batches: Vec<AuditBatch> = (0..5)
            .map(|i| chain.seal_batch(vec![make_entry(i * 100)]))
            .collect();

        assert_eq!(verify_chain(&batches), None);
    }

    #[test]
    fn verify_chain_empty() {
        assert_eq!(verify_chain(&[]), None);
    }

    #[test]
    fn verify_chain_tampered_entry() {
        let mut chain = IntegrityChain::new();
        let mut batches: Vec<AuditBatch> = (0..3)
            .map(|i| chain.seal_batch(vec![make_entry(i * 100)]))
            .collect();

        // Tamper with an entry in batch 1
        batches[1].entries[0].timestamp_ns = 999999;

        // Batch 1's hash should now be invalid
        assert_eq!(verify_chain(&batches), Some(1));
    }

    #[test]
    fn verify_chain_broken_link() {
        let mut chain = IntegrityChain::new();
        let mut batches: Vec<AuditBatch> = (0..3)
            .map(|i| chain.seal_batch(vec![make_entry(i * 100)]))
            .collect();

        // Break the chain link: modify batch 2's previous_hash
        batches[2].previous_hash = GENESIS_HASH;

        assert_eq!(verify_chain(&batches), Some(2));
    }

    #[test]
    fn verify_chain_bad_first_batch() {
        let mut chain = IntegrityChain::new();
        let mut batches: Vec<AuditBatch> = (0..2)
            .map(|i| chain.seal_batch(vec![make_entry(i * 100)]))
            .collect();

        // First batch must have genesis hash
        batches[0].previous_hash = [0xFF; SHA3_256_DIGEST_LEN];

        assert_eq!(verify_chain(&batches), Some(0));
    }

    #[test]
    fn verify_chain_sequence_gap() {
        let mut chain = IntegrityChain::new();
        let mut batches: Vec<AuditBatch> = (0..3)
            .map(|i| chain.seal_batch(vec![make_entry(i * 100)]))
            .collect();

        // Skip sequence number
        batches[2].sequence = 5;

        assert_eq!(verify_chain(&batches), Some(2));
    }

    #[test]
    fn batch_hash_deterministic() {
        let entries = vec![make_entry(100), make_entry(200)];
        let hash1 = compute_batch_hash(&GENESIS_HASH, &entries);
        let hash2 = compute_batch_hash(&GENESIS_HASH, &entries);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn batch_hash_different_entries() {
        let hash1 = compute_batch_hash(&GENESIS_HASH, &[make_entry(100)]);
        let hash2 = compute_batch_hash(&GENESIS_HASH, &[make_entry(200)]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn batch_hash_different_previous() {
        let entries = vec![make_entry(100)];
        let hash1 = compute_batch_hash(&GENESIS_HASH, &entries);
        let other_prev = [0xFF; SHA3_256_DIGEST_LEN];
        let hash2 = compute_batch_hash(&other_prev, &entries);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn batch_hash_empty_entries() {
        // Empty batch is valid (though spec says suppress empty batches)
        let hash = compute_batch_hash(&GENESIS_HASH, &[]);
        // Should be SHA-3-256 of just the genesis hash
        let expected = sha3_256(&GENESIS_HASH);
        assert_eq!(hash, *expected.as_bytes());
    }

    #[test]
    fn seal_empty_batch() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![]);
        assert!(verify_batch_hash(&batch));
        assert_eq!(batch.entries.len(), 0);
    }

    #[test]
    fn large_batch() {
        let mut chain = IntegrityChain::new();
        let entries: Vec<AuditEntry> = (0..256).map(|i| make_entry(i as u64)).collect();
        let batch = chain.seal_batch(entries);
        assert!(verify_batch_hash(&batch));
        assert_eq!(batch.entries.len(), 256);
    }
}
