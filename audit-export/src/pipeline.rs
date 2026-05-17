// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Audit-export pipeline state machine.
//!
//! Drives the producer → batcher → ring buffer → exporter
//! handoff entirely as a `#![no_std]` synchronous state machine
//! parameterized over wall-clock time. The integration layer
//! in `container/` is responsible for:
//!
//! - Calling [`Pipeline::push`] whenever a new audit record is
//!   appended to the local audit ring (the fan-out tap).
//! - Calling [`Pipeline::tick`] on a periodic scheduler so
//!   batch-cut deadlines and backoff timers can advance.
//! - Driving network I/O when [`Pipeline::tick`] returns a
//!   [`PipelineAction::SendBatch`], and reporting the result
//!   back via [`Pipeline::on_send_success`] /
//!   [`Pipeline::on_send_failure`].
//!
//! Everything that is testable without a clock or a network
//! lives here.

use crate::immudb::client::RetryClass;
use crate::immudb::schema::KeyValue;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

/// Configuration knobs sourced from `immudb.toml`.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub batch_size: u32,
    pub batch_interval_ms: u64,
    pub buffer_bytes: usize,
    pub backoff_initial_ms: u64,
    pub backoff_cap_ms: u64,
    pub overflow_log_interval_ms: u64,
    pub include_actions: Vec<String>,
    pub exclude_actions: Vec<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            batch_interval_ms: 1_000,
            buffer_bytes: 4 * 1024 * 1024,
            backoff_initial_ms: 10_000,
            backoff_cap_ms: 300_000,
            overflow_log_interval_ms: 60_000,
            include_actions: Vec::new(),
            exclude_actions: alloc::vec![
                String::from("audit_export_attempt"),
                String::from("audit_export_overflow"),
                String::from("audit_export_proof_failure"),
                String::from("audit_export_rollback_suspected"),
                String::from("audit_export_decode_failure"),
                String::from("audit_export_state_init"),
                String::from("immudb_state"),
            ],
        }
    }
}

/// One audit record on its way to the wire. The `action`
/// is consulted by the include/exclude filter; `key` and
/// `value` are the immudb K/V pair the record encodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingRecord {
    pub action: String,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

impl PendingRecord {
    /// Bytes the record contributes to the buffer accounting.
    pub fn byte_size(&self) -> usize {
        self.action.len() + self.key.len() + self.value.len()
    }
}

/// Why the pipeline refused to accept or process a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    Halted(HaltReason),
}

/// What `tick` wants the caller to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineAction {
    /// No I/O needed; check back at or after `next_wake_at_ms`.
    Idle { next_wake_at_ms: Option<u64> },
    /// Cut and ship this batch via the immudb client. Caller
    /// MUST call [`Pipeline::on_send_success`] or
    /// [`Pipeline::on_send_failure`] when complete; records
    /// stay buffered until then.
    SendBatch {
        records: Vec<PendingRecord>,
        prove_since_tx: u64,
    },
    /// Pipeline is now halted; the caller should emit any
    /// pending audit records reflecting the halt reason and
    /// stop attempting new sends until the operator clears
    /// the failure.
    Halted(HaltReason),
}

/// The audit-pipeline-self records we emit back into the
/// local audit ring (never shipped off-box by default).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelfAudit {
    /// `audit_export_overflow` with the count of dropped records.
    Overflow { dropped: u32 },
    /// `audit_export_attempt` with `code = 0`.
    AttemptOk,
    /// `audit_export_attempt` with a non-zero code (retried).
    AttemptRetry { code: i32 },
    /// `audit_export_proof_failure` or similar halt event.
    Halted(HaltReason),
}

/// Reasons the pipeline may halt and refuse further sends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HaltReason {
    /// gRPC status 7 (PERMISSION_DENIED) or 16 (UNAUTHENTICATED).
    Unauthenticated,
    /// Verifier rejected a proof / signature.
    ProofFailure,
    /// Server's `txId` regressed below our locally-stored value.
    RollbackSuspected,
    /// Protobuf decoder rejected a server response.
    DecodeFailure,
}

/// Whether to keep the batch buffered after a failed attempt
/// or drop it (e.g. on `HALT`-class errors we keep it so the
/// operator can inspect it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SendOutcome {
    /// Status OK; transition to the next batch.
    Success { new_tx_id: u64 },
    /// Retryable failure; keep batch, apply backoff.
    Retry { code: i32 },
    /// Halt-class failure; pipeline stops shipping.
    Halt(HaltReason),
}

/// The pipeline state machine.
pub struct Pipeline {
    cfg: PipelineConfig,

    buffer: VecDeque<PendingRecord>,
    buffer_bytes: usize,

    in_flight: Option<Vec<PendingRecord>>,

    last_tx_id: u64,

    batch_started_at: Option<u64>,

    backoff_current_ms: u64,
    next_attempt_at: u64,

    halt: Option<HaltReason>,

    dropped_since_last_overflow_log: u32,
    last_overflow_log_at: Option<u64>,
}

impl Pipeline {
    pub fn new(cfg: PipelineConfig, last_known_tx_id: u64) -> Self {
        Self {
            cfg,
            buffer: VecDeque::new(),
            buffer_bytes: 0,
            in_flight: None,
            last_tx_id: last_known_tx_id,
            batch_started_at: None,
            backoff_current_ms: 0,
            next_attempt_at: 0,
            halt: None,
            dropped_since_last_overflow_log: 0,
            last_overflow_log_at: None,
        }
    }

    /// True iff the pipeline has stopped shipping for the
    /// given reason and is awaiting operator ack.
    pub fn halt_reason(&self) -> Option<&HaltReason> {
        self.halt.as_ref()
    }

    /// Records currently buffered (not in-flight).
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Bytes currently buffered (not in-flight).
    pub fn buffered_bytes(&self) -> usize {
        self.buffer_bytes
    }

    /// Attempt to enqueue a record. Returns `Ok(true)` if the
    /// record was accepted, `Ok(false)` if the include/exclude
    /// filter dropped it (NOT an error), and `Err` if the
    /// pipeline is halted. The producer side never blocks:
    /// when the buffer is full, oldest records are evicted to
    /// make room; the new record is always queued.
    pub fn push(&mut self, now_ms: u64, record: PendingRecord) -> Result<bool, PipelineError> {
        if let Some(reason) = &self.halt {
            return Err(PipelineError::Halted(reason.clone()));
        }
        if !self.passes_filter(&record) {
            return Ok(false);
        }
        let need = record.byte_size();
        while self.buffer_bytes + need > self.cfg.buffer_bytes {
            if let Some(victim) = self.buffer.pop_front() {
                self.buffer_bytes -= victim.byte_size();
                self.dropped_since_last_overflow_log =
                    self.dropped_since_last_overflow_log.saturating_add(1);
            } else {
                // The new record alone is larger than the buffer; drop it
                // and count it.
                self.dropped_since_last_overflow_log =
                    self.dropped_since_last_overflow_log.saturating_add(1);
                return Ok(false);
            }
        }
        self.buffer_bytes += need;
        self.buffer.push_back(record);
        if self.batch_started_at.is_none() {
            self.batch_started_at = Some(now_ms);
        }
        Ok(true)
    }

    fn passes_filter(&self, record: &PendingRecord) -> bool {
        if !self.cfg.include_actions.is_empty()
            && !self.cfg.include_actions.iter().any(|a| a == &record.action)
        {
            return false;
        }
        if self.cfg.exclude_actions.iter().any(|a| a == &record.action) {
            return false;
        }
        true
    }

    /// Drive the state machine one step. Returns the next
    /// action the caller should perform.
    pub fn tick(&mut self, now_ms: u64) -> PipelineAction {
        if let Some(reason) = &self.halt {
            return PipelineAction::Halted(reason.clone());
        }
        if self.in_flight.is_some() {
            return PipelineAction::Idle {
                next_wake_at_ms: None,
            };
        }
        if now_ms < self.next_attempt_at {
            return PipelineAction::Idle {
                next_wake_at_ms: Some(self.next_attempt_at),
            };
        }
        let size_ready = self.buffer.len() >= self.cfg.batch_size as usize;
        let time_ready = matches!(
            self.batch_started_at,
            Some(t) if now_ms.saturating_sub(t) >= self.cfg.batch_interval_ms
        ) && !self.buffer.is_empty();
        if size_ready || time_ready {
            let take = self.cfg.batch_size as usize;
            let n = take.min(self.buffer.len());
            let mut batch = Vec::with_capacity(n);
            for _ in 0..n {
                let r = self.buffer.pop_front().unwrap();
                self.buffer_bytes -= r.byte_size();
                batch.push(r);
            }
            self.in_flight = Some(batch.clone());
            self.batch_started_at = if self.buffer.is_empty() {
                None
            } else {
                Some(now_ms)
            };
            return PipelineAction::SendBatch {
                records: batch,
                prove_since_tx: self.last_tx_id,
            };
        }
        let next_wake = self
            .batch_started_at
            .map(|t| t.saturating_add(self.cfg.batch_interval_ms));
        PipelineAction::Idle {
            next_wake_at_ms: next_wake,
        }
    }

    /// Caller reports the outcome of a `SendBatch` action.
    pub fn on_send(&mut self, now_ms: u64, outcome: SendOutcome) -> Option<SelfAudit> {
        let batch = self.in_flight.take();
        match outcome {
            SendOutcome::Success { new_tx_id } => {
                self.backoff_current_ms = 0;
                self.next_attempt_at = 0;
                self.last_tx_id = new_tx_id;
                Some(SelfAudit::AttemptOk)
            }
            SendOutcome::Retry { code } => {
                // Re-enqueue the batch at the front of the buffer so
                // ordering is preserved.
                if let Some(b) = batch {
                    for r in b.into_iter().rev() {
                        self.buffer_bytes += r.byte_size();
                        self.buffer.push_front(r);
                    }
                    if self.batch_started_at.is_none() {
                        self.batch_started_at = Some(now_ms);
                    }
                }
                self.backoff_current_ms = if self.backoff_current_ms == 0 {
                    self.cfg.backoff_initial_ms
                } else {
                    (self.backoff_current_ms.saturating_mul(2)).min(self.cfg.backoff_cap_ms)
                };
                self.next_attempt_at = now_ms.saturating_add(self.backoff_current_ms);
                Some(SelfAudit::AttemptRetry { code })
            }
            SendOutcome::Halt(reason) => {
                if let Some(b) = batch {
                    for r in b.into_iter().rev() {
                        self.buffer_bytes += r.byte_size();
                        self.buffer.push_front(r);
                    }
                }
                self.halt = Some(reason.clone());
                Some(SelfAudit::Halted(reason))
            }
        }
    }

    /// Map a [`RetryClass`] to a [`SendOutcome`] using the
    /// gRPC status code returned by the immudb client. Helper
    /// for callers that already have the raw status code.
    pub fn outcome_from_status(class: RetryClass, code: i32) -> SendOutcome {
        match class {
            RetryClass::Ok => SendOutcome::Success { new_tx_id: 0 },
            RetryClass::Retry | RetryClass::HardFail => SendOutcome::Retry { code },
            RetryClass::HaltWithAudit => SendOutcome::Halt(HaltReason::Unauthenticated),
        }
    }

    /// If the overflow-log throttle is due, emit a record
    /// summarizing dropped-since-last-emission and reset the
    /// counter. Returns `None` if nothing is owed.
    pub fn maybe_emit_overflow(&mut self, now_ms: u64) -> Option<SelfAudit> {
        if self.dropped_since_last_overflow_log == 0 {
            return None;
        }
        let due = self
            .last_overflow_log_at
            .is_none_or(|last| now_ms.saturating_sub(last) >= self.cfg.overflow_log_interval_ms);
        if !due {
            return None;
        }
        let dropped = self.dropped_since_last_overflow_log;
        self.dropped_since_last_overflow_log = 0;
        self.last_overflow_log_at = Some(now_ms);
        Some(SelfAudit::Overflow { dropped })
    }
}

/// Convert a [`PendingRecord`] into the [`KeyValue`] the client
/// will encode on the wire. Caller is responsible for any
/// per-record dual-digest (D2) wrapping.
pub fn pending_to_kv(record: PendingRecord) -> KeyValue {
    KeyValue {
        key: record.key,
        value: record.value,
        metadata: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn cfg() -> PipelineConfig {
        PipelineConfig {
            batch_size: 3,
            batch_interval_ms: 1_000,
            buffer_bytes: 1_024,
            backoff_initial_ms: 10_000,
            backoff_cap_ms: 300_000,
            overflow_log_interval_ms: 60_000,
            include_actions: Vec::new(),
            exclude_actions: Vec::new(),
        }
    }

    fn rec(action: &str, value: &[u8]) -> PendingRecord {
        PendingRecord {
            action: action.into(),
            key: action.as_bytes().to_vec(),
            value: value.to_vec(),
        }
    }

    #[test]
    fn push_accepts_within_capacity() {
        let mut p = Pipeline::new(cfg(), 0);
        for i in 0..10 {
            assert!(p.push(0, rec("login", &[i; 8])).unwrap());
        }
        assert_eq!(p.buffered(), 10);
    }

    #[test]
    fn exclude_filter_drops_record() {
        let mut c = cfg();
        c.exclude_actions = vec!["audit_export_attempt".into()];
        let mut p = Pipeline::new(c, 0);
        assert!(!p.push(0, rec("audit_export_attempt", &[1])).unwrap());
        assert!(p.push(0, rec("login", &[1])).unwrap());
        assert_eq!(p.buffered(), 1);
    }

    #[test]
    fn include_filter_admits_only_listed() {
        let mut c = cfg();
        c.include_actions = vec!["login".into()];
        let mut p = Pipeline::new(c, 0);
        assert!(p.push(0, rec("login", &[1])).unwrap());
        assert!(!p.push(0, rec("logout", &[1])).unwrap());
        assert_eq!(p.buffered(), 1);
    }

    #[test]
    fn drop_oldest_when_buffer_full() {
        let mut c = cfg();
        c.buffer_bytes = 100;
        let mut p = Pipeline::new(c, 0);
        for i in 0..20u8 {
            // Each record costs roughly action.len()+key.len()+value.len()
            // ≈ 5+5+8 = 18 bytes. 20 records ~= 360 bytes, far over the cap.
            p.push(0, rec("login", &[i; 8])).unwrap();
        }
        assert!(p.buffered_bytes() <= 100);
        // At least one record dropped → overflow due.
        let overflow = p.maybe_emit_overflow(1_000);
        match overflow {
            Some(SelfAudit::Overflow { dropped }) => assert!(dropped >= 1),
            other => panic!("expected overflow, got {other:?}"),
        }
    }

    #[test]
    fn producer_never_blocks_under_pressure() {
        let mut c = cfg();
        c.buffer_bytes = 200;
        let mut p = Pipeline::new(c, 0);
        // 10_000 records at ~18 B each = ~180 KiB on a 200 B buffer.
        // The push must never fail (drop-oldest); buffer stays bounded.
        for i in 0..10_000u32 {
            let r = PendingRecord {
                action: "login".into(),
                key: alloc::vec![0u8; 5],
                value: i.to_le_bytes().to_vec(),
            };
            p.push(0, r).unwrap();
        }
        assert!(p.buffered_bytes() <= 200);
    }

    #[test]
    fn size_threshold_cuts_batch() {
        let mut p = Pipeline::new(cfg(), 0);
        p.push(0, rec("a", &[0])).unwrap();
        p.push(0, rec("b", &[0])).unwrap();
        p.push(0, rec("c", &[0])).unwrap();
        // batch_size = 3 ⇒ ready now.
        match p.tick(0) {
            PipelineAction::SendBatch { records, .. } => assert_eq!(records.len(), 3),
            other => panic!("expected SendBatch, got {other:?}"),
        }
    }

    #[test]
    fn time_threshold_cuts_partial_batch() {
        let mut p = Pipeline::new(cfg(), 0);
        p.push(0, rec("a", &[0])).unwrap();
        // Before deadline: idle.
        match p.tick(500) {
            PipelineAction::Idle { .. } => {}
            other => panic!("expected Idle, got {other:?}"),
        }
        // After deadline: ship the partial batch.
        match p.tick(1_001) {
            PipelineAction::SendBatch { records, .. } => assert_eq!(records.len(), 1),
            other => panic!("expected SendBatch, got {other:?}"),
        }
    }

    #[test]
    fn empty_buffer_after_time_threshold_stays_idle() {
        let mut p = Pipeline::new(cfg(), 0);
        match p.tick(5_000) {
            PipelineAction::Idle { .. } => {}
            other => panic!("expected Idle, got {other:?}"),
        }
    }

    #[test]
    fn success_resets_backoff_and_advances_tx_id() {
        let mut p = Pipeline::new(cfg(), 100);
        for _ in 0..3 {
            p.push(0, rec("a", &[0])).unwrap();
        }
        let _action = p.tick(0);
        let audit = p
            .on_send(0, SendOutcome::Success { new_tx_id: 123 })
            .unwrap();
        assert_eq!(audit, SelfAudit::AttemptOk);
        assert_eq!(p.last_tx_id, 123);
    }

    #[test]
    fn retry_re_enqueues_and_schedules_backoff() {
        let mut p = Pipeline::new(cfg(), 0);
        for _ in 0..3 {
            p.push(0, rec("a", &[0])).unwrap();
        }
        let _ = p.tick(0);
        let audit = p.on_send(100, SendOutcome::Retry { code: 14 }).unwrap();
        assert!(matches!(audit, SelfAudit::AttemptRetry { code: 14 }));
        // Batch is back in the buffer; next tick before backoff
        // elapses must NOT cut a new batch.
        match p.tick(100) {
            PipelineAction::Idle { next_wake_at_ms } => {
                assert_eq!(next_wake_at_ms, Some(100 + 10_000));
            }
            other => panic!("expected Idle, got {other:?}"),
        }
        assert_eq!(p.buffered(), 3);
    }

    #[test]
    fn backoff_doubles_then_caps() {
        let mut c = cfg();
        c.backoff_initial_ms = 10;
        c.backoff_cap_ms = 40;
        let mut p = Pipeline::new(c, 0);
        for _ in 0..3 {
            p.push(0, rec("a", &[0])).unwrap();
        }
        let _ = p.tick(0);
        p.on_send(0, SendOutcome::Retry { code: 14 });
        assert_eq!(p.backoff_current_ms, 10);
        // Advance past first backoff, fail again.
        let _ = p.tick(100);
        p.on_send(100, SendOutcome::Retry { code: 14 });
        assert_eq!(p.backoff_current_ms, 20);
        let _ = p.tick(200);
        p.on_send(200, SendOutcome::Retry { code: 14 });
        assert_eq!(p.backoff_current_ms, 40);
        // Capped: stays at 40 even after another failure.
        let _ = p.tick(300);
        p.on_send(300, SendOutcome::Retry { code: 14 });
        assert_eq!(p.backoff_current_ms, 40);
    }

    #[test]
    fn first_success_resets_backoff() {
        let mut p = Pipeline::new(cfg(), 0);
        for _ in 0..3 {
            p.push(0, rec("a", &[0])).unwrap();
        }
        let _ = p.tick(0);
        p.on_send(0, SendOutcome::Retry { code: 14 });
        assert_eq!(p.backoff_current_ms, 10_000);
        let _ = p.tick(11_000);
        p.on_send(11_000, SendOutcome::Success { new_tx_id: 1 });
        assert_eq!(p.backoff_current_ms, 0);
    }

    #[test]
    fn halt_stops_further_sends() {
        let mut p = Pipeline::new(cfg(), 0);
        for _ in 0..3 {
            p.push(0, rec("a", &[0])).unwrap();
        }
        let _ = p.tick(0);
        let audit = p
            .on_send(0, SendOutcome::Halt(HaltReason::ProofFailure))
            .unwrap();
        assert!(matches!(audit, SelfAudit::Halted(HaltReason::ProofFailure)));
        assert!(p.halt_reason().is_some());
        match p.tick(10_000_000) {
            PipelineAction::Halted(HaltReason::ProofFailure) => {}
            other => panic!("expected Halted, got {other:?}"),
        }
        assert!(matches!(
            p.push(0, rec("a", &[0])).unwrap_err(),
            PipelineError::Halted(HaltReason::ProofFailure)
        ));
    }

    #[test]
    fn overflow_log_throttled_to_once_per_interval() {
        let mut c = cfg();
        c.buffer_bytes = 30; // tiny to force drops
        c.overflow_log_interval_ms = 60_000;
        let mut p = Pipeline::new(c, 0);
        for i in 0..50u8 {
            p.push(0, rec("a", &[i; 8])).unwrap();
        }
        // First call emits.
        let first = p.maybe_emit_overflow(0);
        assert!(matches!(first, Some(SelfAudit::Overflow { .. })));
        // Subsequent call within the interval emits nothing.
        for i in 0..50u8 {
            p.push(0, rec("a", &[i; 8])).unwrap();
        }
        let none = p.maybe_emit_overflow(30_000);
        assert!(none.is_none());
        // After the interval, another emission is due.
        for i in 0..50u8 {
            p.push(0, rec("a", &[i; 8])).unwrap();
        }
        let second = p.maybe_emit_overflow(60_000);
        assert!(matches!(second, Some(SelfAudit::Overflow { .. })));
    }

    #[test]
    fn no_overflow_when_zero_dropped() {
        let mut p = Pipeline::new(cfg(), 0);
        p.push(0, rec("a", &[0])).unwrap();
        assert!(p.maybe_emit_overflow(1_000_000).is_none());
    }

    #[test]
    fn in_flight_batch_blocks_new_send() {
        let mut p = Pipeline::new(cfg(), 0);
        for _ in 0..6 {
            p.push(0, rec("a", &[0])).unwrap();
        }
        let _ = p.tick(0);
        // While the first batch is in-flight, tick stays Idle even
        // though more records are queued.
        match p.tick(0) {
            PipelineAction::Idle { .. } => {}
            other => panic!("expected Idle, got {other:?}"),
        }
        // Acknowledge the first batch.
        p.on_send(0, SendOutcome::Success { new_tx_id: 1 });
        // Now the next batch is available.
        match p.tick(0) {
            PipelineAction::SendBatch { records, .. } => assert_eq!(records.len(), 3),
            other => panic!("expected SendBatch, got {other:?}"),
        }
    }

    #[test]
    fn ordering_preserved_across_retry() {
        let mut p = Pipeline::new(cfg(), 0);
        for i in 0..3u8 {
            p.push(0, rec("a", &[i])).unwrap();
        }
        let _ = p.tick(0);
        p.on_send(0, SendOutcome::Retry { code: 14 });
        // After retry, buffer contains the records in original order.
        assert_eq!(p.buffer[0].value, vec![0]);
        assert_eq!(p.buffer[1].value, vec![1]);
        assert_eq!(p.buffer[2].value, vec![2]);
    }
}
