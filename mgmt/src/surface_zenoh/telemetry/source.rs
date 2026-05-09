// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Source of metric samples consumed by the telemetry publisher.
//!
//! `MetricsSource` is the seam between the in-kernel counters
//! (kernel-side phase, not Phase 8) and the Zenoh telemetry surface.
//! Phase 8 ships:
//!
//! - the trait,
//! - a [`StubMetricsSource`] used by tests + integration smoke,
//! - a [`PendingLogs`] FIFO and [`PendingAudit`] latest-fingerprint
//!   slot the trait exposes for event-driven keys.
//!
//! Production wiring (a `KernelMetricsSource`) lands in a future
//! `kernel-stats` phase.

use alloc::collections::VecDeque;
use alloc::string::ToString;
use alloc::vec::Vec;

use super::schema::{AuditFingerprint, CpuStat, InferenceStat, LogLevel, LogRecord, MemStat};

/// Source of metric samples. Each periodic key has a dedicated getter
/// returning the **current** sample. Event-driven keys (`log`,
/// `audit_fingerprint`) read from FIFO / slot accessors that the
/// publisher drains on each pass.
///
/// All methods are infallible — sources are expected to fall back to
/// safe defaults (zeros, empty slices) when the underlying counter is
/// not yet initialised.
pub trait MetricsSource {
    /// Per-core CPU utilisation snapshot.
    fn cpu(&self) -> CpuStat;

    /// Memory snapshot.
    fn mem(&self) -> MemStat;

    /// Per-model inference snapshot.
    fn inference(&self) -> Vec<InferenceStat>;

    /// Drain pending log records. The implementation SHALL return at
    /// most `max` records and remove the returned ones from any
    /// internal queue. Order SHALL be FIFO.
    fn drain_logs(&mut self, max: usize) -> Vec<LogRecord>;

    /// Latest audit fingerprint. Returns `None` if the chain is empty
    /// (no records have been committed). The publisher uses this
    /// alongside [`Self::audit_dirty`] to decide whether to publish.
    fn audit_fingerprint(&self) -> Option<AuditFingerprint>;

    /// Has the audit fingerprint changed since the last call?
    /// Implementations SHALL clear the dirty flag when this returns
    /// `true`. Used by the publisher's event-driven sweep so a flat
    /// chain is not republished every tick.
    fn audit_dirty(&mut self) -> bool;
}

// ─── Stub source ─────────────────────────────────────────────────────────────

/// Synthetic metric source — drives unit tests and the integration
/// smoke binary. Mutating helpers let tests inject specific samples
/// then poke the publisher.
#[derive(Debug, Default)]
pub struct StubMetricsSource {
    /// Current CPU stat. Tests overwrite this directly; the sample
    /// returned from [`MetricsSource::cpu`] is a clone.
    pub cpu: CpuStat,
    /// Current mem stat.
    pub mem: MemStat,
    /// Per-model inference stats.
    pub inference: Vec<InferenceStat>,
    /// Pending log records, FIFO. [`MetricsSource::drain_logs`] pops
    /// from the front.
    pub logs: VecDeque<LogRecord>,
    /// Latest audit fingerprint. `None` means the chain is empty.
    pub audit: Option<AuditFingerprint>,
    /// Set by [`Self::set_audit`] / [`Self::push_audit`]; cleared by
    /// [`MetricsSource::audit_dirty`] when the publisher reads it.
    audit_dirty_flag: bool,
}

impl StubMetricsSource {
    /// Construct a fresh stub with empty samples.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the current CPU sample.
    pub fn set_cpu(&mut self, cpu: CpuStat) {
        self.cpu = cpu;
    }

    /// Replace the current mem sample.
    pub fn set_mem(&mut self, mem: MemStat) {
        self.mem = mem;
    }

    /// Replace the per-model inference list.
    pub fn set_inference(&mut self, stats: Vec<InferenceStat>) {
        self.inference = stats;
    }

    /// Push a log record onto the FIFO.
    pub fn push_log(&mut self, level: LogLevel, target: &str, msg: &str) {
        self.logs.push_back(LogRecord {
            ts_ns: 0,
            level,
            target: target.to_string(),
            msg: msg.to_string(),
            fields: Vec::new(),
        });
    }

    /// Push a structured log record with explicit `ts_ns` + fields.
    pub fn push_log_record(&mut self, record: LogRecord) {
        self.logs.push_back(record);
    }

    /// Set the latest audit fingerprint and mark the dirty flag.
    pub fn set_audit(&mut self, fp: AuditFingerprint) {
        self.audit = Some(fp);
        self.audit_dirty_flag = true;
    }

    /// Update only the chain head + record count, reusing the last
    /// timestamp. Convenience for tests that simulate appending audit
    /// records.
    pub fn push_audit(&mut self, hex_fingerprint: &str, record_count: u64) {
        let ts = self.audit.as_ref().map(|a| a.ts_ns).unwrap_or(0);
        self.set_audit(AuditFingerprint {
            ts_ns: ts,
            hex_fingerprint: hex_fingerprint.to_string(),
            record_count,
        });
    }

    /// Number of pending log records.
    pub fn log_pending(&self) -> usize {
        self.logs.len()
    }
}

impl MetricsSource for StubMetricsSource {
    fn cpu(&self) -> CpuStat {
        self.cpu.clone()
    }

    fn mem(&self) -> MemStat {
        self.mem
    }

    fn inference(&self) -> Vec<InferenceStat> {
        self.inference.clone()
    }

    fn drain_logs(&mut self, max: usize) -> Vec<LogRecord> {
        let take = max.min(self.logs.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(r) = self.logs.pop_front() {
                out.push(r);
            }
        }
        out
    }

    fn audit_fingerprint(&self) -> Option<AuditFingerprint> {
        self.audit.clone()
    }

    fn audit_dirty(&mut self) -> bool {
        let v = self.audit_dirty_flag;
        self.audit_dirty_flag = false;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::super::schema::PerCpuUtil;
    use super::*;

    #[test]
    fn stub_returns_cloned_cpu_sample() {
        let mut s = StubMetricsSource::new();
        s.set_cpu(CpuStat {
            ts_ns: 1,
            per_core: alloc::vec![PerCpuUtil {
                id: 0,
                util_pct: 42,
            }],
        });
        let snap = s.cpu();
        assert_eq!(snap.peak_util_pct(), 42);
    }

    #[test]
    fn drain_logs_respects_fifo_order() {
        let mut s = StubMetricsSource::new();
        s.push_log(LogLevel::Info, "t", "first");
        s.push_log(LogLevel::Warn, "t", "second");
        s.push_log(LogLevel::Error, "t", "third");
        let drained = s.drain_logs(10);
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].msg, "first");
        assert_eq!(drained[1].msg, "second");
        assert_eq!(drained[2].msg, "third");
        assert_eq!(s.log_pending(), 0);
    }

    #[test]
    fn drain_logs_caps_at_max() {
        let mut s = StubMetricsSource::new();
        for i in 0..10 {
            s.push_log(LogLevel::Info, "t", &alloc::format!("m{}", i));
        }
        let drained = s.drain_logs(3);
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].msg, "m0");
        assert_eq!(s.log_pending(), 7);
    }

    #[test]
    fn drain_logs_zero_max_returns_empty() {
        let mut s = StubMetricsSource::new();
        s.push_log(LogLevel::Info, "t", "x");
        let drained = s.drain_logs(0);
        assert!(drained.is_empty());
        assert_eq!(s.log_pending(), 1);
    }

    #[test]
    fn audit_dirty_set_and_cleared_on_read() {
        let mut s = StubMetricsSource::new();
        assert!(!s.audit_dirty());
        s.push_audit("abc", 1);
        assert!(s.audit_dirty());
        // Second call clears.
        assert!(!s.audit_dirty());
    }

    #[test]
    fn audit_fingerprint_starts_empty() {
        let s = StubMetricsSource::new();
        assert!(s.audit_fingerprint().is_none());
    }

    #[test]
    fn audit_push_increments_record_count() {
        let mut s = StubMetricsSource::new();
        s.push_audit("aa", 1);
        s.push_audit("bb", 2);
        let fp = s.audit_fingerprint().unwrap();
        assert_eq!(fp.hex_fingerprint, "bb");
        assert_eq!(fp.record_count, 2);
    }

    #[test]
    fn inference_stats_default_empty() {
        let s = StubMetricsSource::new();
        assert!(s.inference().is_empty());
    }

    #[test]
    fn mem_default_zero() {
        let s = StubMetricsSource::new();
        let m = s.mem();
        assert_eq!(m.heap_used_bytes, 0);
        assert_eq!(m.heap_cap_bytes, 0);
    }
}
