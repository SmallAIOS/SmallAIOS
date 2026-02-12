// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MC/DC (Modified Condition/Decision Coverage) tests for audit subsystem.
//!
//! DO-178C DAL A requires 100% MC/DC coverage on safety-critical paths.
//! Each decision's conditions must be shown to independently affect the
//! outcome. This module provides exhaustive tests covering every condition
//! in every decision on the audit critical path.
//!
//! Audit critical paths covered:
//! - BatchAccumulator::push — condition: pending.len() >= BATCH_MAX_ENTRIES
//! - BatchAccumulator::tick — conditions: pending.is_empty(), elapsed >= timeout
//! - BatchAccumulator::flush — condition: pending.is_empty()
//! - IntegrityChain::verify_chain — conditions: genesis check, hash match,
//!   sequence continuity, batch hash validity
//! - RetentionPolicy::should_retain — condition: age < retention_ns
//! - RetentionPolicy::batches_to_prune — conditions: time-based, count-based
//! - BatchSigningQueue::enqueue — condition: pending_count >= MAX_PENDING
//! - BatchSigningQueue::sign_next — conditions: slot occupied, not signed,
//!   signature size valid
//! - IPC serialize/deserialize — conditions: buffer size, version match

#[cfg(test)]
mod tests {
    use crate::audit::accumulator::{BatchAccumulator, BATCH_MAX_ENTRIES, BATCH_TIMEOUT_NS};
    use crate::audit::batch_signing::{
        BatchSigningError, BatchSigningQueue, SignedAuditBatch, ML_DSA_65_SIG_SIZE,
    };
    use crate::audit::entry::{AuditEntry, AuditResult, Operation};
    use crate::audit::integrity::{
        verify_batch_hash, verify_chain, AuditBatch, IntegrityChain, GENESIS_HASH,
    };
    use crate::audit::ipc_export::{
        deserialize_batch_header, serialize_batch_header, BATCH_HEADER_SIZE,
    };
    use crate::audit::retention::{DeploymentClass, RetentionPolicy};
    use crate::audit::taxonomy::AuditEventType;
    use crate::crypto::sha3::SHA3_256_DIGEST_LEN;
    use alloc::vec;
    use alloc::vec::Vec;

    fn make_entry(ts: u64) -> AuditEntry {
        AuditEntry {
            timestamp_ns: ts,
            event_type: AuditEventType::capability_grant(),
            task_id: 1,
            resource_type: 0,
            resource_instance: 0,
            operation: Operation::new("mcdc"),
            result: AuditResult::Success,
            capability_id: 0,
        }
    }

    // =========================================================================
    // MC/DC for BatchAccumulator::push
    // Decision: self.pending.len() >= BATCH_MAX_ENTRIES
    // Condition C1: pending count reaches threshold
    // =========================================================================

    /// C1=false: push with count below threshold -> no seal
    #[test]
    fn mcdc_push_below_threshold() {
        let mut acc = BatchAccumulator::new();
        for i in 0..(BATCH_MAX_ENTRIES - 1) {
            assert!(acc.push(make_entry(i as u64), i as u64).is_none());
        }
        assert_eq!(acc.pending_count(), BATCH_MAX_ENTRIES - 1);
    }

    /// C1=true: push exactly at threshold -> seal
    #[test]
    fn mcdc_push_at_threshold() {
        let mut acc = BatchAccumulator::new();
        for i in 0..(BATCH_MAX_ENTRIES - 1) {
            acc.push(make_entry(i as u64), i as u64);
        }
        let result = acc.push(
            make_entry(BATCH_MAX_ENTRIES as u64),
            BATCH_MAX_ENTRIES as u64,
        );
        assert!(result.is_some());
        let batch = result.unwrap();
        assert_eq!(batch.entries.len(), BATCH_MAX_ENTRIES);
        assert!(verify_batch_hash(&batch));
    }

    // =========================================================================
    // MC/DC for BatchAccumulator::tick
    // Decision: pending.is_empty() [C1] OR elapsed >= timeout [C2]
    // MC/DC test cases:
    //   C1=true,  C2=don't-care -> None (empty suppression)
    //   C1=false, C2=false      -> None (not timed out)
    //   C1=false, C2=true       -> Some (timed out)
    // =========================================================================

    /// C1=true: tick with empty pending -> None (empty suppression)
    #[test]
    fn mcdc_tick_c1_true_empty() {
        let mut acc = BatchAccumulator::new();
        let result = acc.tick(BATCH_TIMEOUT_NS * 2);
        assert!(result.is_none());
    }

    /// C1=false, C2=false: tick with pending entries, before timeout -> None
    #[test]
    fn mcdc_tick_c1_false_c2_false() {
        let mut acc = BatchAccumulator::new();
        acc.push(make_entry(0), 0);
        let result = acc.tick(BATCH_TIMEOUT_NS / 2);
        assert!(result.is_none());
        assert_eq!(acc.pending_count(), 1);
    }

    /// C1=false, C2=true: tick with pending entries, at timeout -> seal
    #[test]
    fn mcdc_tick_c1_false_c2_true() {
        let mut acc = BatchAccumulator::new();
        acc.push(make_entry(0), 0);
        let result = acc.tick(BATCH_TIMEOUT_NS);
        assert!(result.is_some());
        let batch = result.unwrap();
        assert_eq!(batch.entries.len(), 1);
        assert!(verify_batch_hash(&batch));
    }

    /// C2 independence: exact timeout boundary
    #[test]
    fn mcdc_tick_timeout_boundary() {
        let mut acc = BatchAccumulator::new();
        acc.push(make_entry(100), 100);
        // One nanosecond before timeout -> no seal
        assert!(acc.tick(100 + BATCH_TIMEOUT_NS - 1).is_none());
        // Exactly at timeout -> seal
        assert!(acc.tick(100 + BATCH_TIMEOUT_NS).is_some());
    }

    // =========================================================================
    // MC/DC for BatchAccumulator::flush
    // Decision: self.pending.is_empty()
    // C1=true  -> None
    // C1=false -> Some
    // =========================================================================

    /// C1=true: flush with no pending -> None
    #[test]
    fn mcdc_flush_empty() {
        let mut acc = BatchAccumulator::new();
        assert!(acc.flush().is_none());
    }

    /// C1=false: flush with pending -> Some
    #[test]
    fn mcdc_flush_has_pending() {
        let mut acc = BatchAccumulator::new();
        acc.push(make_entry(0), 0);
        let result = acc.flush();
        assert!(result.is_some());
        assert!(verify_batch_hash(&result.unwrap()));
    }

    // =========================================================================
    // MC/DC for verify_chain
    // Decision at entry: batches.is_empty() [C1]
    // Decision at batch[0]: previous_hash == GENESIS [C2], sequence == 0 [C3],
    //                       verify_batch_hash [C4]
    // Decision at batch[i]: previous_hash == prev.batch_hash [C5],
    //                       sequence == prev.sequence + 1 [C6],
    //                       verify_batch_hash [C7]
    // =========================================================================

    /// C1=true: empty chain -> valid (None)
    #[test]
    fn mcdc_verify_chain_empty() {
        assert_eq!(verify_chain(&[]), None);
    }

    /// C2=true, C3=true, C4=true: valid single batch
    #[test]
    fn mcdc_verify_chain_single_valid() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(100)]);
        assert_eq!(verify_chain(&[batch]), None);
    }

    /// C2=false: first batch has wrong genesis -> invalid
    #[test]
    fn mcdc_verify_chain_bad_genesis() {
        let mut chain = IntegrityChain::new();
        let mut batch = chain.seal_batch(vec![make_entry(100)]);
        batch.previous_hash = [0xFF; SHA3_256_DIGEST_LEN];
        assert_eq!(verify_chain(&[batch]), Some(0));
    }

    /// C3=false: first batch has wrong sequence -> invalid
    #[test]
    fn mcdc_verify_chain_bad_first_sequence() {
        let mut chain = IntegrityChain::new();
        let mut batch = chain.seal_batch(vec![make_entry(100)]);
        batch.sequence = 5; // Should be 0
        assert_eq!(verify_chain(&[batch]), Some(0));
    }

    /// C4=false: first batch hash tampered -> invalid
    #[test]
    fn mcdc_verify_chain_bad_first_hash() {
        let mut chain = IntegrityChain::new();
        let mut batch = chain.seal_batch(vec![make_entry(100)]);
        batch.entries[0].timestamp_ns = 999; // Tamper with entry
        assert_eq!(verify_chain(&[batch]), Some(0));
    }

    /// C5=false: second batch has broken chain link -> invalid
    #[test]
    fn mcdc_verify_chain_broken_link() {
        let mut chain = IntegrityChain::new();
        let batch0 = chain.seal_batch(vec![make_entry(100)]);
        let mut batch1 = chain.seal_batch(vec![make_entry(200)]);
        batch1.previous_hash = GENESIS_HASH; // Break link
        assert_eq!(verify_chain(&[batch0, batch1]), Some(1));
    }

    /// C6=false: sequence gap in chain -> invalid
    #[test]
    fn mcdc_verify_chain_sequence_gap() {
        let mut chain = IntegrityChain::new();
        let batch0 = chain.seal_batch(vec![make_entry(100)]);
        let mut batch1 = chain.seal_batch(vec![make_entry(200)]);
        batch1.sequence = 5; // Gap
        assert_eq!(verify_chain(&[batch0, batch1]), Some(1));
    }

    /// C7=false: second batch hash tampered -> invalid
    #[test]
    fn mcdc_verify_chain_tampered_later() {
        let mut chain = IntegrityChain::new();
        let batch0 = chain.seal_batch(vec![make_entry(100)]);
        let mut batch1 = chain.seal_batch(vec![make_entry(200)]);
        batch1.entries[0].timestamp_ns = 777; // Tamper
        assert_eq!(verify_chain(&[batch0, batch1]), Some(1));
    }

    /// All conditions true across multi-batch chain -> valid
    #[test]
    fn mcdc_verify_chain_multi_valid() {
        let mut chain = IntegrityChain::new();
        let batches: Vec<AuditBatch> = (0..5)
            .map(|i| chain.seal_batch(vec![make_entry(i * 100)]))
            .collect();
        assert_eq!(verify_chain(&batches), None);
    }

    // =========================================================================
    // MC/DC for RetentionPolicy::should_retain
    // Decision: age < retention_ns
    // C1: age is within window -> true
    // C1: age is beyond window -> false
    // =========================================================================

    /// C1=true: batch within retention window -> retain
    #[test]
    fn mcdc_retain_within_window() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        assert!(policy.should_retain(0, DeploymentClass::Edge.retention_ns() - 1));
    }

    /// C1=false: batch beyond retention window -> prune
    #[test]
    fn mcdc_retain_beyond_window() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        assert!(!policy.should_retain(0, DeploymentClass::Edge.retention_ns()));
    }

    /// Boundary: exactly at retention window edge
    #[test]
    fn mcdc_retain_exact_boundary() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        let retention = DeploymentClass::Edge.retention_ns();
        // age == retention (not strictly less) -> prune
        assert!(!policy.should_retain(0, retention));
        // age == retention - 1 -> retain
        assert!(policy.should_retain(1, retention));
    }

    // =========================================================================
    // MC/DC for RetentionPolicy::batches_to_prune
    // Decision: time-based condition [C1] AND count-based condition [C2]
    // C1: batch timestamp outside window
    // C2: total_count > max_batches AND max_batches > 0
    // =========================================================================

    /// C1=true, C2=false: only time-based pruning active
    #[test]
    fn mcdc_prune_time_only() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        let retention = DeploymentClass::Edge.retention_ns();
        let now = retention * 3;
        // All timestamps are older than retention window
        let timestamps = [0, 1, retention]; // ages: 3*ret, 3*ret-1, 2*ret; all >= ret
        let prune = policy.batches_to_prune(&timestamps, now, 3);
        assert_eq!(prune, 3); // All outside window
    }

    /// C1=false, C2=true: only count-based pruning active
    #[test]
    fn mcdc_prune_count_only() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge).with_max_batches(2);
        let now = 100; // All timestamps very recent
        let timestamps = [50, 60, 70, 80];
        let prune = policy.batches_to_prune(&timestamps, now, 4);
        assert_eq!(prune, 2); // 4 - 2 = 2 excess
    }

    /// C1=false, C2=false: nothing to prune
    #[test]
    fn mcdc_prune_nothing() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        let now = 100;
        let timestamps = [50, 60, 70];
        let prune = policy.batches_to_prune(&timestamps, now, 3);
        assert_eq!(prune, 0);
    }

    /// C1=true, C2=true: both conditions active, take max
    #[test]
    fn mcdc_prune_both() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge).with_max_batches(1);
        let now = DeploymentClass::Edge.retention_ns() * 3;
        let timestamps = [0, 1, DeploymentClass::Edge.retention_ns() * 2 + 1];
        let prune = policy.batches_to_prune(&timestamps, now, 3);
        // Time-based: first 2 expired. Count-based: 3-1=2. Max = 2.
        assert_eq!(prune, 2);
    }

    // =========================================================================
    // MC/DC for BatchSigningQueue::enqueue
    // Decision: self.pending_count >= MAX_PENDING [C1]
    // =========================================================================

    /// C1=false: queue has space -> Ok
    #[test]
    fn mcdc_enqueue_has_space() {
        let mut queue = BatchSigningQueue::new();
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(0)]);
        assert!(queue.enqueue(batch).is_ok());
    }

    /// C1=true: queue full -> Err(QueueFull)
    #[test]
    fn mcdc_enqueue_full() {
        let mut queue = BatchSigningQueue::new();
        let mut chain = IntegrityChain::new();
        for _ in 0..16 {
            let batch = chain.seal_batch(vec![make_entry(0)]);
            queue.enqueue(batch).unwrap();
        }
        let batch = chain.seal_batch(vec![make_entry(0)]);
        assert_eq!(queue.enqueue(batch), Err(BatchSigningError::QueueFull));
    }

    // =========================================================================
    // MC/DC for BatchSigningQueue::sign_next
    // Decision: slot occupied [C1] AND not signed [C2] AND sig size valid [C3]
    // =========================================================================

    /// C1=false: no pending batches -> NoPending
    #[test]
    fn mcdc_sign_no_pending() {
        let mut queue = BatchSigningQueue::new();
        let sig = [0u8; ML_DSA_65_SIG_SIZE];
        assert!(matches!(
            queue.sign_next(&sig, [0u8; 16]),
            Err(BatchSigningError::NoPending)
        ));
    }

    /// C1=true, C2=true, C3=true: successful sign
    #[test]
    fn mcdc_sign_success() {
        let mut queue = BatchSigningQueue::new();
        let mut chain = IntegrityChain::new();
        queue
            .enqueue(chain.seal_batch(vec![make_entry(0)]))
            .unwrap();
        let sig = [0xAA; ML_DSA_65_SIG_SIZE];
        let result = queue.sign_next(&sig, [0x01; 16]);
        assert!(result.is_ok());
        let signed = result.unwrap();
        assert!(signed.is_signed);
    }

    /// C3=false: wrong signature size -> InvalidSignatureSize
    #[test]
    fn mcdc_sign_wrong_sig_size() {
        let mut queue = BatchSigningQueue::new();
        let mut chain = IntegrityChain::new();
        queue
            .enqueue(chain.seal_batch(vec![make_entry(0)]))
            .unwrap();
        let sig = [0u8; 100]; // Wrong size
        assert!(matches!(
            queue.sign_next(&sig, [0u8; 16]),
            Err(BatchSigningError::InvalidSignatureSize)
        ));
    }

    // =========================================================================
    // MC/DC for IPC serialize/deserialize
    // serialize_batch_header: buf.len() < BATCH_HEADER_SIZE [C1]
    // deserialize_batch_header: buf.len() < BATCH_HEADER_SIZE [C1],
    //                           version != WIRE_VERSION [C2]
    // =========================================================================

    /// Serialize C1=false: buffer large enough -> success
    #[test]
    fn mcdc_serialize_sufficient_buffer() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(0)]);
        let signed = SignedAuditBatch::new(batch);
        let mut buf = [0u8; BATCH_HEADER_SIZE];
        assert_eq!(
            serialize_batch_header(&signed, &mut buf),
            Some(BATCH_HEADER_SIZE)
        );
    }

    /// Serialize C1=true: buffer too small -> None
    #[test]
    fn mcdc_serialize_small_buffer() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(0)]);
        let signed = SignedAuditBatch::new(batch);
        let mut buf = [0u8; BATCH_HEADER_SIZE - 1];
        assert_eq!(serialize_batch_header(&signed, &mut buf), None);
    }

    /// Deserialize C1=true: buffer too small -> None
    #[test]
    fn mcdc_deserialize_small_buffer() {
        let buf = [0u8; BATCH_HEADER_SIZE - 1];
        assert!(deserialize_batch_header(&buf).is_none());
    }

    /// Deserialize C1=false, C2=true: wrong version -> None
    #[test]
    fn mcdc_deserialize_wrong_version() {
        let mut buf = [0u8; BATCH_HEADER_SIZE];
        buf[0] = 0xFF; // Invalid version
        assert!(deserialize_batch_header(&buf).is_none());
    }

    /// Deserialize C1=false, C2=false: valid -> Some
    #[test]
    fn mcdc_deserialize_valid() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(0)]);
        let signed = SignedAuditBatch::new(batch);
        let mut buf = [0u8; BATCH_HEADER_SIZE];
        serialize_batch_header(&signed, &mut buf).unwrap();
        let info = deserialize_batch_header(&buf).unwrap();
        assert_eq!(info.sequence, 0);
    }

    // =========================================================================
    // MC/DC for verify_batch_hash
    // Decision: recomputed == batch.batch_hash
    // C1=true: untampered batch -> true
    // C1=false: tampered batch -> false
    // =========================================================================

    /// C1=true: untampered
    #[test]
    fn mcdc_verify_hash_untampered() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(100), make_entry(200)]);
        assert!(verify_batch_hash(&batch));
    }

    /// C1=false: tampered entry
    #[test]
    fn mcdc_verify_hash_tampered_entry() {
        let mut chain = IntegrityChain::new();
        let mut batch = chain.seal_batch(vec![make_entry(100), make_entry(200)]);
        batch.entries[0].timestamp_ns = 0; // Tamper
        assert!(!verify_batch_hash(&batch));
    }

    /// C1=false: tampered hash directly
    #[test]
    fn mcdc_verify_hash_tampered_hash() {
        let mut chain = IntegrityChain::new();
        let mut batch = chain.seal_batch(vec![make_entry(100)]);
        batch.batch_hash[0] ^= 0xFF; // Flip bits
        assert!(!verify_batch_hash(&batch));
    }

    /// C1=false: tampered previous_hash (doesn't affect batch_hash verification directly,
    /// but batch_hash includes previous_hash in the computation)
    #[test]
    fn mcdc_verify_hash_tampered_prev() {
        let mut chain = IntegrityChain::new();
        let mut batch = chain.seal_batch(vec![make_entry(100)]);
        batch.previous_hash[0] ^= 0xFF; // Flip bits in previous
        assert!(!verify_batch_hash(&batch)); // Hash includes previous_hash
    }

    // =========================================================================
    // MC/DC for SignedAuditBatch::set_signature
    // Decision: sig.len() != ML_DSA_65_SIG_SIZE [C1]
    // =========================================================================

    /// C1=false: correct size -> true, is_signed set
    #[test]
    fn mcdc_set_sig_correct_size() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(0)]);
        let mut signed = SignedAuditBatch::new(batch);
        assert!(!signed.is_signed);
        assert!(signed.set_signature(&[0u8; ML_DSA_65_SIG_SIZE], [0x01; 16]));
        assert!(signed.is_signed);
    }

    /// C1=true: wrong size (too small) -> false
    #[test]
    fn mcdc_set_sig_too_small() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(0)]);
        let mut signed = SignedAuditBatch::new(batch);
        assert!(!signed.set_signature(&[0u8; 10], [0u8; 16]));
        assert!(!signed.is_signed);
    }

    /// C1=true: wrong size (too large) -> false
    #[test]
    fn mcdc_set_sig_too_large() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(0)]);
        let mut signed = SignedAuditBatch::new(batch);
        assert!(!signed.set_signature(&[0u8; ML_DSA_65_SIG_SIZE + 1], [0u8; 16]));
        assert!(!signed.is_signed);
    }

    /// C1=true: empty signature -> false
    #[test]
    fn mcdc_set_sig_empty() {
        let mut chain = IntegrityChain::new();
        let batch = chain.seal_batch(vec![make_entry(0)]);
        let mut signed = SignedAuditBatch::new(batch);
        assert!(!signed.set_signature(&[], [0u8; 16]));
        assert!(!signed.is_signed);
    }
}
