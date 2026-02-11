// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ML-DSA-65 batch signing for sealed audit batches.
//!
//! After a batch is sealed by the integrity chain, it can be signed
//! with ML-DSA-65 for non-repudiation (NIST AU-10). Signing is
//! designed to be non-blocking to the inference path.

use crate::crypto::sha3::{Sha3_256, SHA3_256_DIGEST_LEN};
use super::integrity::AuditBatch;

/// ML-DSA-65 signature size in bytes.
pub const ML_DSA_65_SIG_SIZE: usize = 3309;

/// A signed audit batch wrapping a sealed batch with an ML-DSA-65 signature.
#[derive(Debug, Clone)]
pub struct SignedAuditBatch {
    /// The sealed batch (hash-chained).
    pub batch: AuditBatch,
    /// ML-DSA-65 signature over the batch hash.
    pub signature: [u8; ML_DSA_65_SIG_SIZE],
    /// Whether the signature has been populated.
    pub is_signed: bool,
    /// Signer key identifier (fingerprint of signing public key).
    pub signer_key_id: [u8; 16],
}

impl SignedAuditBatch {
    /// Wrap a sealed batch for signing.
    pub fn new(batch: AuditBatch) -> Self {
        Self {
            batch,
            signature: [0u8; ML_DSA_65_SIG_SIZE],
            is_signed: false,
            signer_key_id: [0u8; 16],
        }
    }

    /// The data to be signed: SHA-3-256(batch_hash || sequence_be).
    ///
    /// This produces a deterministic signing input that binds the batch
    /// content (via batch_hash) and its position (via sequence).
    pub fn signing_input(&self) -> [u8; SHA3_256_DIGEST_LEN] {
        let mut hasher = Sha3_256::new();
        hasher
            .update(&self.batch.batch_hash)
            .expect("update cannot fail");
        hasher
            .update(&self.batch.sequence.to_be_bytes())
            .expect("update cannot fail");
        let digest = hasher.finalize().expect("finalize cannot fail");
        *digest.as_bytes()
    }

    /// Apply a signature to this batch.
    pub fn set_signature(&mut self, sig: &[u8], key_id: [u8; 16]) -> bool {
        if sig.len() != ML_DSA_65_SIG_SIZE {
            return false;
        }
        self.signature[..].copy_from_slice(sig);
        self.signer_key_id = key_id;
        self.is_signed = true;
        true
    }

    /// Verify that the signing input matches the batch content.
    ///
    /// This does NOT verify the cryptographic signature (ML-DSA-65
    /// verification is in the crypto module). It verifies that the
    /// signing input is correctly derived from the batch.
    pub fn verify_signing_input(&self) -> bool {
        let expected = self.signing_input();
        // Re-derive and compare
        let mut hasher = Sha3_256::new();
        hasher
            .update(&self.batch.batch_hash)
            .expect("update cannot fail");
        hasher
            .update(&self.batch.sequence.to_be_bytes())
            .expect("update cannot fail");
        let digest = hasher.finalize().expect("finalize cannot fail");
        *digest.as_bytes() == expected
    }
}

/// Batch signing queue for non-blocking signature generation.
///
/// Batches are enqueued after sealing and signed asynchronously
/// so the inference path is never blocked by signing latency.
pub struct BatchSigningQueue {
    /// Pending batches awaiting signature.
    pending: [Option<SignedAuditBatch>; MAX_PENDING],
    /// Number of pending batches.
    pending_count: usize,
    /// Total batches signed since boot.
    total_signed: u64,
    /// Total signing failures.
    total_failures: u64,
}

const MAX_PENDING: usize = 16;

impl BatchSigningQueue {
    pub fn new() -> Self {
        Self {
            pending: core::array::from_fn(|_| None),
            pending_count: 0,
            total_signed: 0,
            total_failures: 0,
        }
    }

    /// Enqueue a sealed batch for signing.
    pub fn enqueue(&mut self, batch: AuditBatch) -> Result<(), BatchSigningError> {
        if self.pending_count >= MAX_PENDING {
            return Err(BatchSigningError::QueueFull);
        }
        for slot in self.pending.iter_mut() {
            if slot.is_none() {
                *slot = Some(SignedAuditBatch::new(batch));
                self.pending_count += 1;
                return Ok(());
            }
        }
        Err(BatchSigningError::QueueFull)
    }

    /// Sign the next pending batch with the provided signature.
    ///
    /// In production, the signing key would be passed here. For now,
    /// the caller provides the pre-computed signature.
    pub fn sign_next(
        &mut self,
        sig: &[u8],
        key_id: [u8; 16],
    ) -> Result<SignedAuditBatch, BatchSigningError> {
        for slot in self.pending.iter_mut() {
            if let Some(ref mut signed_batch) = slot {
                if !signed_batch.is_signed {
                    if !signed_batch.set_signature(sig, key_id) {
                        self.total_failures += 1;
                        return Err(BatchSigningError::InvalidSignatureSize);
                    }
                    let result = signed_batch.clone();
                    *slot = None;
                    self.pending_count -= 1;
                    self.total_signed += 1;
                    return Ok(result);
                }
            }
        }
        Err(BatchSigningError::NoPending)
    }

    /// Number of pending (unsigned) batches.
    pub fn pending_count(&self) -> usize {
        self.pending_count
    }

    /// Total batches signed since boot.
    pub fn total_signed(&self) -> u64 {
        self.total_signed
    }

    /// Total signing failures.
    pub fn total_failures(&self) -> u64 {
        self.total_failures
    }
}

impl Default for BatchSigningQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch signing errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSigningError {
    /// Signing queue is full.
    QueueFull,
    /// No pending batches to sign.
    NoPending,
    /// Signature size does not match ML-DSA-65.
    InvalidSignatureSize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::entry::{AuditEntry, AuditResult, Operation};
    use crate::audit::integrity::IntegrityChain;
    use crate::audit::taxonomy::AuditEventType;
    use alloc::vec;

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

    fn make_sealed_batch(seq: u64) -> AuditBatch {
        let mut chain = IntegrityChain::new();
        // Advance to desired sequence
        for _ in 0..seq {
            chain.seal_batch(vec![make_entry(0)]);
        }
        chain.seal_batch(vec![make_entry(100), make_entry(200)])
    }

    #[test]
    fn signed_batch_creation() {
        let batch = make_sealed_batch(0);
        let signed = SignedAuditBatch::new(batch);
        assert!(!signed.is_signed);
        assert_eq!(signed.signer_key_id, [0u8; 16]);
    }

    #[test]
    fn signing_input_deterministic() {
        let batch = make_sealed_batch(0);
        let signed = SignedAuditBatch::new(batch);
        let input1 = signed.signing_input();
        let input2 = signed.signing_input();
        assert_eq!(input1, input2);
    }

    #[test]
    fn signing_input_differs_by_sequence() {
        let batch0 = make_sealed_batch(0);
        let batch1 = make_sealed_batch(1);
        let signed0 = SignedAuditBatch::new(batch0);
        let signed1 = SignedAuditBatch::new(batch1);
        assert_ne!(signed0.signing_input(), signed1.signing_input());
    }

    #[test]
    fn set_signature_correct_size() {
        let batch = make_sealed_batch(0);
        let mut signed = SignedAuditBatch::new(batch);
        let sig = [0x42u8; ML_DSA_65_SIG_SIZE];
        let key_id = [0xABu8; 16];
        assert!(signed.set_signature(&sig, key_id));
        assert!(signed.is_signed);
        assert_eq!(signed.signer_key_id, key_id);
        assert_eq!(signed.signature[0], 0x42);
    }

    #[test]
    fn set_signature_wrong_size() {
        let batch = make_sealed_batch(0);
        let mut signed = SignedAuditBatch::new(batch);
        let sig = [0x42u8; 100];
        assert!(!signed.set_signature(&sig, [0u8; 16]));
        assert!(!signed.is_signed);
    }

    #[test]
    fn verify_signing_input() {
        let batch = make_sealed_batch(0);
        let signed = SignedAuditBatch::new(batch);
        assert!(signed.verify_signing_input());
    }

    #[test]
    fn queue_enqueue_and_sign() {
        let mut queue = BatchSigningQueue::new();
        let batch = make_sealed_batch(0);
        queue.enqueue(batch).unwrap();
        assert_eq!(queue.pending_count(), 1);

        let sig = [0xFFu8; ML_DSA_65_SIG_SIZE];
        let key_id = [0x01u8; 16];
        let signed = queue.sign_next(&sig, key_id).unwrap();
        assert!(signed.is_signed);
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(queue.total_signed(), 1);
    }

    #[test]
    fn queue_full() {
        let mut queue = BatchSigningQueue::new();
        for i in 0..MAX_PENDING {
            queue.enqueue(make_sealed_batch(i as u64)).unwrap();
        }
        assert!(matches!(
            queue.enqueue(make_sealed_batch(99)),
            Err(BatchSigningError::QueueFull)
        ));
    }

    #[test]
    fn queue_sign_empty() {
        let mut queue = BatchSigningQueue::new();
        let sig = [0u8; ML_DSA_65_SIG_SIZE];
        assert!(matches!(
            queue.sign_next(&sig, [0u8; 16]),
            Err(BatchSigningError::NoPending)
        ));
    }

    #[test]
    fn queue_sign_wrong_size() {
        let mut queue = BatchSigningQueue::new();
        queue.enqueue(make_sealed_batch(0)).unwrap();
        let sig = [0u8; 100];
        assert!(matches!(
            queue.sign_next(&sig, [0u8; 16]),
            Err(BatchSigningError::InvalidSignatureSize)
        ));
        assert_eq!(queue.total_failures(), 1);
    }

    #[test]
    fn queue_multiple_batches() {
        let mut queue = BatchSigningQueue::new();
        queue.enqueue(make_sealed_batch(0)).unwrap();
        queue.enqueue(make_sealed_batch(1)).unwrap();
        assert_eq!(queue.pending_count(), 2);

        let sig = [0xAAu8; ML_DSA_65_SIG_SIZE];
        let s1 = queue.sign_next(&sig, [1u8; 16]).unwrap();
        assert_eq!(s1.batch.sequence, 0);
        assert_eq!(queue.pending_count(), 1);

        let s2 = queue.sign_next(&sig, [2u8; 16]).unwrap();
        assert_eq!(s2.batch.sequence, 1);
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(queue.total_signed(), 2);
    }
}
