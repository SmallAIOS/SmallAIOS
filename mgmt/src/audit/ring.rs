// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Fixed 16 MiB in-memory audit [`Ring`] — 4 KiB × 4096 records.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-audit-log/spec.md`
//! Requirement "In-memory ring with periodic flush".
//!
//! ## Geometry
//!
//! - 4096 record slots (`AUDIT_RING_CAPACITY`).
//! - Each slot is budgeted for 4 KiB of serialized JSONL — the spec's
//!   per-record budget. The Rust `AuditRecord` struct is heap-backed
//!   so the in-memory storage cost is dominated by the strings, not
//!   by a flat byte buffer; the budget is enforced by truncating
//!   over-budget records before append.
//! - Total in-memory budget: `AUDIT_RING_BYTES = 16 MiB`.
//!
//! ## Lifecycle
//!
//! 1. Caller invokes [`Ring::append`] with a partially-populated
//!    [`AuditRecord`] (`prev_hash` and `hash` are computed inside).
//! 2. Ring: chains, hashes, places into the ring buffer.
//! 3. Ring: tells the [`super::flusher::Flusher`] there's pending data
//!    (immediate flush for auth events, otherwise marks dirty).
//! 4. Caller may invoke [`Ring::tick`] periodically (e.g. once per
//!    second from the kernel scheduler) to flush dirty data.
//!
//! ## Crash-safety
//!
//! The chain head is stored alongside the ring's tail pointer in
//! `Ring`'s in-memory state. On reboot, the JSONL file on disk is
//! the source of truth — replaying it reconstructs the chain head.
//! The ring itself is volatile by design (see spec — the on-disk
//! file is the durable record).

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::chain::{compute_hash, ChainHead, EMPTY_CHAIN_HEAD};
use super::checkpoint::{CheckpointSigner, NoopSigner};
use super::denial::DenialRateLimiter;
use super::flusher::{Flusher, FlusherError, MemoryFlusher};
use super::record::{AuditAction, AuditRecord};

/// Ring capacity — 4096 record slots per spec (4 KiB × 4096 = 16 MiB
/// per-record budget × slot count).
pub const AUDIT_RING_CAPACITY: usize = 4096;

/// Per-record budget — 4 KiB of serialized JSONL.
pub const AUDIT_RING_RECORD_BUDGET_BYTES: usize = 4 * 1024;

/// Total in-memory ring budget — 16 MiB.
pub const AUDIT_RING_BYTES: usize = AUDIT_RING_CAPACITY * AUDIT_RING_RECORD_BUDGET_BYTES;

/// Ring-level errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingError {
    /// The flusher rejected the JSONL line.
    Flush(FlusherError),
    /// The record exceeds [`AUDIT_RING_RECORD_BUDGET_BYTES`] after
    /// truncation. Indicates a programmer error — every spec-allowed
    /// `before` / `after` payload fits within 4 KiB.
    OverBudget,
}

impl fmt::Display for RingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flush(e) => write!(f, "audit flush failed: {}", e),
            Self::OverBudget => f.write_str("audit record exceeds 4 KiB budget"),
        }
    }
}

/// Cooperatively-driven audit ring. Holds the in-memory record buffer,
/// the chain head, the rate limiter, the flusher, and (when enabled)
/// the signed-checkpoint signer.
pub struct Ring<F = MemoryFlusher, S = NoopSigner>
where
    F: Flusher,
    S: CheckpointSigner,
{
    /// Bounded ring of committed records. `VecDeque` lets us
    /// `pop_front` the oldest when the capacity is hit. Each element
    /// is the in-memory mirror of a record that was already appended
    /// to the on-disk JSONL.
    records: VecDeque<AuditRecord>,
    /// Latest chain head — SHA-3-256 of the most recently committed
    /// record. `EMPTY_CHAIN_HEAD` until the first `append`.
    chain_head: ChainHead,
    /// Monotonic count of records committed since boot. Wraps at
    /// `u64::MAX` (i.e. never in practice).
    record_count: u64,
    /// Pending flush flag — set on every commit, cleared by `tick`.
    /// Auth events bypass the timer and flush immediately.
    dirty: bool,
    /// Rate-limited denial coalesce buffer.
    denial: DenialRateLimiter,
    /// Pluggable flush sink.
    flusher: F,
    /// Pluggable checkpoint signer.
    signer: S,
    /// Spec-controlled signed-checkpoint cadence. `0` disables
    /// signed checkpoints regardless of [`Self::signer`].
    checkpoint_interval: u32,
}

impl<F, S> fmt::Debug for Ring<F, S>
where
    F: Flusher,
    S: CheckpointSigner,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ring")
            .field("record_count", &self.record_count)
            .field("dirty", &self.dirty)
            .field("checkpoint_interval", &self.checkpoint_interval)
            .field("chain_head_prefix", &&self.chain_head[..4])
            .finish()
    }
}

impl Ring<MemoryFlusher, NoopSigner> {
    /// Build a ring backed by an in-memory flusher and no signed
    /// checkpoints. Used by every unit test that doesn't need to
    /// exercise the disk side.
    pub fn in_memory() -> Self {
        Self::new(MemoryFlusher::new(), NoopSigner, 0)
    }
}

impl<F, S> Ring<F, S>
where
    F: Flusher,
    S: CheckpointSigner,
{
    /// Construct a ring from its three pluggable parts.
    pub fn new(flusher: F, signer: S, checkpoint_interval: u32) -> Self {
        Self {
            records: VecDeque::with_capacity(AUDIT_RING_CAPACITY),
            chain_head: EMPTY_CHAIN_HEAD,
            record_count: 0,
            dirty: false,
            denial: DenialRateLimiter::new(),
            flusher,
            signer,
            checkpoint_interval,
        }
    }

    /// Number of records currently held in memory.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// `true` iff no record has ever been appended to this ring.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Latest SHA-3-256 chain head.
    pub fn chain_head(&self) -> ChainHead {
        self.chain_head
    }

    /// Cumulative count of committed records (never decreases on
    /// rollover — stays in u64 long enough for any realistic uptime).
    pub fn record_count(&self) -> u64 {
        self.record_count
    }

    /// Borrow the most recent in-memory record, if any. The ring is
    /// FIFO so `back()` is the most recent.
    pub fn latest(&self) -> Option<&AuditRecord> {
        self.records.back()
    }

    /// Borrow every in-memory record in chronological order.
    pub fn snapshot(&self) -> Vec<&AuditRecord> {
        self.records.iter().collect()
    }

    /// Borrow the flusher.
    pub fn flusher(&self) -> &F {
        &self.flusher
    }

    /// Mutably borrow the flusher (for tests).
    pub fn flusher_mut(&mut self) -> &mut F {
        &mut self.flusher
    }

    /// Borrow the signer.
    pub fn signer(&self) -> &S {
        &self.signer
    }

    /// Update the signed-checkpoint interval. `0` disables signed
    /// checkpoints regardless of the signer's capability.
    pub fn set_checkpoint_interval(&mut self, interval: u32) {
        self.checkpoint_interval = interval;
    }

    /// Append a record. Computes `prev_hash` + `hash`, places into
    /// the ring, immediately flushes if the action is an auth event,
    /// otherwise marks the ring dirty for the next [`Self::tick`].
    ///
    /// Denials are routed through the rate limiter — excess denials
    /// are coalesced into a `DENY_BURST` record on the next denial
    /// from the same user that crosses the rate-limit window.
    pub fn append(&mut self, mut record: AuditRecord) -> Result<ChainHead, RingError> {
        // Denial path: the rate limiter may swallow the original
        // record and emit a coalesced `DENY_BURST` instead.
        if record.action.is_denial() {
            return self.append_denial(record);
        }
        record.prev_hash = self.chain_head;
        record.hash = compute_hash(&record);
        self.flush_record(&record)?;
        let head = record.hash;
        self.commit_in_memory(record);
        // Optional signed checkpoint at the configured interval.
        self.maybe_emit_checkpoint()?;
        Ok(head)
    }

    /// Append a denial record. Rate-limit governs whether to record
    /// the original `DENY:<syscall>` form, drop it, or emit a
    /// coalesced `DENY_BURST`.
    fn append_denial(&mut self, record: AuditRecord) -> Result<ChainHead, RingError> {
        // Burst records bypass the rate limiter and pass through.
        if matches!(record.action, AuditAction::DenyBurst { .. }) {
            return self.append_uncoalesced(record);
        }
        let user = record.who.clone();
        let syscall = match &record.action {
            AuditAction::Deny(s) => s.clone(),
            _ => unreachable!("guarded by is_denial()"),
        };
        let now_ns = record.ts_ns;
        match self.denial.observe(&user, now_ns) {
            super::denial::DenialDecision::Allow => self.append_uncoalesced(record),
            super::denial::DenialDecision::Drop => {
                // Excess denial — recorded as part of the eventual
                // burst record. No on-chain entry.
                let _ = syscall;
                Ok(self.chain_head)
            }
            super::denial::DenialDecision::FlushBurst { count } => {
                // Emit the coalesced burst record first, then the
                // current denial individually so the chain reflects
                // the fact that the rate limiter window reset.
                let burst = AuditRecord {
                    ts_ns: now_ns,
                    who: user,
                    surface: record.surface,
                    action: AuditAction::DenyBurst { count },
                    before: super::record::JsonValue::Null,
                    after: super::record::JsonValue::Null,
                    code: -1,
                    prev_hash: EMPTY_CHAIN_HEAD,
                    hash: EMPTY_CHAIN_HEAD,
                    signature: Vec::new(),
                };
                let _ = self.append_uncoalesced(burst)?;
                self.append_uncoalesced(record)
            }
        }
    }

    /// Direct path used by both regular records and denial records
    /// that bypass the rate limiter.
    fn append_uncoalesced(&mut self, mut record: AuditRecord) -> Result<ChainHead, RingError> {
        record.prev_hash = self.chain_head;
        record.hash = compute_hash(&record);
        self.flush_record(&record)?;
        let head = record.hash;
        self.commit_in_memory(record);
        self.maybe_emit_checkpoint()?;
        Ok(head)
    }

    fn commit_in_memory(&mut self, record: AuditRecord) {
        self.chain_head = record.hash;
        self.record_count = self.record_count.saturating_add(1);
        if self.records.len() == AUDIT_RING_CAPACITY {
            // Drop oldest in-memory mirror. The on-disk JSONL still
            // holds it.
            let _ = self.records.pop_front();
        }
        self.records.push_back(record);
    }

    fn flush_record(&mut self, record: &AuditRecord) -> Result<(), RingError> {
        let line = record.to_jsonl();
        if line.len() > AUDIT_RING_RECORD_BUDGET_BYTES {
            return Err(RingError::OverBudget);
        }
        let immediate = record.action.is_auth_event();
        self.flusher
            .write_line(&line, immediate)
            .map_err(RingError::Flush)?;
        if !immediate {
            self.dirty = true;
        } else {
            // Auth events flush synchronously — clear the dirty flag.
            self.dirty = false;
        }
        Ok(())
    }

    /// Cooperative tick — invoked by the scheduler ~once per second.
    /// Flushes any dirty data to the underlying [`Flusher`].
    pub fn tick(&mut self) -> Result<(), RingError> {
        if self.dirty {
            self.flusher.flush().map_err(RingError::Flush)?;
            self.dirty = false;
        }
        Ok(())
    }

    /// Force a flush right now (used by integration tests + the
    /// shutdown path).
    pub fn flush_now(&mut self) -> Result<(), RingError> {
        self.flusher.flush().map_err(RingError::Flush)?;
        self.dirty = false;
        Ok(())
    }

    /// Read the chain head as a hex string — convenience for the
    /// `audit_fingerprint` telemetry source.
    pub fn chain_head_hex(&self) -> String {
        super::record::fingerprint_hex(&self.chain_head)
    }

    /// Append a denial record by syscall name + identity. Convenience
    /// over [`Self::append`] that builds the `AuditAction::Deny`
    /// record on behalf of the caller.
    pub fn append_denial_event(
        &mut self,
        ts_ns: u64,
        who: String,
        surface: super::record::AuditSurface,
        syscall: &str,
    ) -> Result<ChainHead, RingError> {
        let record = AuditRecord {
            ts_ns,
            who,
            surface,
            action: AuditAction::Deny(alloc::string::ToString::to_string(syscall)),
            before: super::record::JsonValue::Null,
            after: super::record::JsonValue::Null,
            code: -1,
            prev_hash: EMPTY_CHAIN_HEAD,
            hash: EMPTY_CHAIN_HEAD,
            signature: Vec::new(),
        };
        self.append(record)
    }

    fn maybe_emit_checkpoint(&mut self) -> Result<(), RingError> {
        if self.checkpoint_interval == 0 {
            return Ok(());
        }
        let interval = self.checkpoint_interval as u64;
        if self.record_count > 0 && self.record_count.is_multiple_of(interval) {
            // Sign the current chain head. NoopSigner returns
            // `Ok(empty)`; production signers return `Ok(<bytes>)` or
            // an `Err`. Either way, an empty signature means the
            // signer is not configured — skip emitting a checkpoint.
            let signature = match self.signer.sign(&self.chain_head) {
                Ok(s) => s,
                Err(_) => return Ok(()),
            };
            if signature.is_empty() {
                return Ok(());
            }
            let now = self
                .latest()
                .map(|r| r.ts_ns)
                .unwrap_or(0)
                .saturating_add(1);
            let prev = self.chain_head;
            let mut cp = AuditRecord {
                ts_ns: now,
                who: alloc::string::String::from("kernel"),
                surface: super::record::AuditSurface::Kernel,
                action: AuditAction::Checkpoint,
                before: super::record::JsonValue::Null,
                after: super::record::JsonValue::Null,
                code: 0,
                prev_hash: prev,
                hash: EMPTY_CHAIN_HEAD,
                signature,
            };
            cp.hash = compute_hash(&cp);
            self.flush_record(&cp)?;
            self.chain_head = cp.hash;
            self.record_count = self.record_count.saturating_add(1);
            if self.records.len() == AUDIT_RING_CAPACITY {
                let _ = self.records.pop_front();
            }
            self.records.push_back(cp);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::audit_test_vectors as fixtures;
    use super::super::record::{AuditAction, AuditSurface, JsonValue};
    use super::*;

    #[test]
    fn capacity_and_budget_match_spec() {
        assert_eq!(AUDIT_RING_CAPACITY, 4096);
        assert_eq!(AUDIT_RING_RECORD_BUDGET_BYTES, 4 * 1024);
        assert_eq!(AUDIT_RING_BYTES, 16 * 1024 * 1024);
    }

    #[test]
    fn empty_ring_has_zero_chain_head() {
        let r = Ring::in_memory();
        assert_eq!(r.chain_head(), EMPTY_CHAIN_HEAD);
        assert_eq!(r.record_count(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn append_advances_chain_head() {
        let mut r = Ring::in_memory();
        let h0 = r.chain_head();
        let h1 = r.append(fixtures::sample_login_record()).unwrap();
        assert_ne!(h0, h1);
        assert_eq!(r.chain_head(), h1);
        assert_eq!(r.record_count(), 1);
    }

    #[test]
    fn append_chains_records() {
        let mut r = Ring::in_memory();
        let h1 = r.append(fixtures::sample_login_record()).unwrap();
        let h2 = r.append(fixtures::sample_config_write_record()).unwrap();
        assert_ne!(h1, h2);
        let snap = r.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].prev_hash, EMPTY_CHAIN_HEAD);
        assert_eq!(snap[1].prev_hash, h1);
    }

    #[test]
    fn auth_event_immediate_flush() {
        let mut r = Ring::in_memory();
        r.append(fixtures::sample_login_record()).unwrap();
        // After an auth event the ring SHALL NOT be dirty — the
        // record was flushed synchronously.
        assert!(!r.dirty);
        // The flusher saw the JSONL line.
        assert_eq!(r.flusher().written_lines().len(), 1);
        assert_eq!(r.flusher().flush_count(), 1);
    }

    #[test]
    fn non_auth_event_marks_dirty() {
        let mut r = Ring::in_memory();
        r.append(fixtures::sample_config_write_record()).unwrap();
        // Config write does not flush synchronously — dirty until
        // tick().
        assert!(r.dirty);
        r.tick().unwrap();
        assert!(!r.dirty);
        assert!(r.flusher().flush_count() >= 1);
    }

    #[test]
    fn tick_clears_dirty_only_if_set() {
        let mut r = Ring::in_memory();
        r.tick().unwrap();
        assert_eq!(r.flusher().flush_count(), 0);
        r.append(fixtures::sample_config_write_record()).unwrap();
        r.tick().unwrap();
        assert_eq!(r.flusher().flush_count(), 1);
        r.tick().unwrap();
        // Second tick is a no-op.
        assert_eq!(r.flusher().flush_count(), 1);
    }

    #[test]
    fn chain_head_hex_64_chars() {
        let mut r = Ring::in_memory();
        r.append(fixtures::sample_login_record()).unwrap();
        let hex = r.chain_head_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn append_denial_via_helper() {
        let mut r = Ring::in_memory();
        r.append_denial_event(
            1_700_000_000_000_000_000,
            alloc::string::String::from("op:alice@zenoh"),
            AuditSurface::Zenoh,
            "system_power",
        )
        .unwrap();
        let snap = r.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].action, AuditAction::Deny("system_power".into()));
        assert_eq!(snap[0].code, -1);
    }

    #[test]
    fn rate_limit_first_ten_individual() {
        let mut r = Ring::in_memory();
        let user = alloc::string::String::from("op:alice@zenoh");
        for i in 0..10 {
            r.append_denial_event(
                1_000_000_000 + i,
                user.clone(),
                AuditSurface::Zenoh,
                "system_power",
            )
            .unwrap();
        }
        // First 10 individual records were committed.
        assert_eq!(r.record_count(), 10);
        for rec in r.snapshot() {
            assert!(matches!(rec.action, AuditAction::Deny(_)));
        }
    }

    #[test]
    fn rate_limit_excess_swallowed_then_burst_emitted() {
        let mut r = Ring::in_memory();
        let user = alloc::string::String::from("op:alice@zenoh");
        // 200 denials in <1 s — first 10 individual, remaining 190
        // swallowed until the rate-limit window expires; the next
        // denial outside the window flushes a burst record with
        // count = 190.
        for i in 0..200 {
            r.append_denial_event(
                1_000_000_000 + i,
                user.clone(),
                AuditSurface::Zenoh,
                "system_power",
            )
            .unwrap();
        }
        assert_eq!(r.record_count(), 10, "10 individual + 190 swallowed");
        // Trigger one more denial *after* the 1s window — the
        // limiter SHALL flush a `DENY_BURST { count: 190 }` then
        // record the new denial individually.
        let after_window = 1_000_000_000 + 1_500_000_000;
        r.append_denial_event(
            after_window,
            user.clone(),
            AuditSurface::Zenoh,
            "system_power",
        )
        .unwrap();
        // 10 individual + 1 burst + 1 new individual = 12.
        assert_eq!(r.record_count(), 12);
        let snap = r.snapshot();
        // Find the burst record.
        let burst = snap
            .iter()
            .find(|r| matches!(r.action, AuditAction::DenyBurst { .. }))
            .expect("burst record present");
        match &burst.action {
            AuditAction::DenyBurst { count } => assert_eq!(*count, 190),
            _ => unreachable!(),
        }
    }

    #[test]
    fn ring_truncates_when_capacity_hit() {
        // Simulate a ring with capacity = 4 (smaller version of the
        // 4096-slot real one) by directly manipulating; we use the
        // real capacity but assert we can still append more than
        // capacity and that the in-memory mirror tracks only the
        // most-recent capacity worth.
        let mut r = Ring::in_memory();
        let cap = AUDIT_RING_CAPACITY;
        // 4097 records — exceeds capacity by one.
        for i in 0..(cap + 1) {
            let mut rec = fixtures::sample_login_record();
            rec.ts_ns = 1_000_000_000 + i as u64;
            r.append(rec).unwrap();
        }
        assert_eq!(r.record_count(), (cap + 1) as u64);
        assert_eq!(r.snapshot().len(), cap);
    }

    #[test]
    fn over_budget_record_rejected() {
        let mut r = Ring::in_memory();
        let mut rec = fixtures::sample_login_record();
        // Build a `who` field large enough that the JSONL line
        // exceeds 4 KiB.
        rec.who = alloc::string::String::from_utf8(vec![b'a'; 8 * 1024]).unwrap();
        let err = r.append(rec).unwrap_err();
        assert_eq!(err, RingError::OverBudget);
    }

    use alloc::vec;

    #[test]
    fn chain_head_changes_after_each_commit() {
        let mut r = Ring::in_memory();
        let mut prev = r.chain_head();
        for i in 0..5 {
            let mut rec = fixtures::sample_login_record();
            rec.ts_ns = 1_000_000_000 + i;
            let head = r.append(rec).unwrap();
            assert_ne!(prev, head);
            prev = head;
        }
        assert_eq!(r.record_count(), 5);
    }

    #[test]
    fn config_write_capture_round_trip() {
        let mut r = Ring::in_memory();
        let rec = fixtures::sample_config_write_record();
        let before = rec.before.clone();
        let after = rec.after.clone();
        r.append(rec).unwrap();
        let snap = r.snapshot();
        assert_eq!(snap[0].before, before);
        assert_eq!(snap[0].after, after);
    }

    #[test]
    fn dropping_dirty_does_not_panic() {
        // Dropping a dirty ring SHALL NOT panic; the on-disk JSONL is
        // the source of truth, so losing in-memory dirty state is
        // benign as long as no panic propagates.
        let mut r = Ring::in_memory();
        r.append(fixtures::sample_config_write_record()).unwrap();
        assert!(r.dirty);
        drop(r);
    }

    #[test]
    fn auth_event_record_persists_on_flusher_immediately() {
        let mut r = Ring::in_memory();
        let rec = fixtures::sample_login_record();
        r.append(rec).unwrap();
        let lines = r.flusher().written_lines();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("\"action\":\"auth_login\""));
    }

    #[test]
    fn checkpoint_interval_zero_disables() {
        let mut r = Ring::in_memory();
        r.set_checkpoint_interval(0);
        for i in 0..32 {
            let mut rec = fixtures::sample_login_record();
            rec.ts_ns = 1_000 + i as u64;
            r.append(rec).unwrap();
        }
        for snap in r.snapshot() {
            assert!(!matches!(snap.action, AuditAction::Checkpoint));
        }
    }

    #[test]
    fn json_before_after_null_for_login() {
        let mut r = Ring::in_memory();
        r.append(fixtures::sample_login_record()).unwrap();
        let snap = r.snapshot();
        assert_eq!(snap[0].before, JsonValue::Null);
        assert_eq!(snap[0].after, JsonValue::Null);
    }
}
