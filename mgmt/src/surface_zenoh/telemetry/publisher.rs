// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cooperative telemetry publisher.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-zenoh-telemetry/spec.md`.
//!
//! [`TelemetryPublisher`] is transport-agnostic — it emits
//! [`PublishRecord`] values to a [`PublishSink`] callback the caller
//! supplies. Production wires this to a Zenoh publisher; tests wire
//! it to a `Vec`-backed sink so cadence + content can be asserted.
//!
//! ## Cooperative driver
//!
//! The publisher does not spawn its own thread or own a timer. The
//! Core 0 cooperative scheduler calls [`TelemetryPublisher::tick`] with
//! the current monotonic-now-millis. The publisher walks per-key
//! deadlines: any key whose `next_due_ms <= now_ms` is sampled and
//! published, then has its deadline rolled forward by the resolved
//! interval. This keeps the publisher allocation-light and re-entrant.
//!
//! ## Per-key deadlines
//!
//! Each [`MetricKey`] carries:
//!
//! - a [`super::CadenceState`] (clamped, optional adaptive),
//! - a `next_due_ms` deadline,
//! - an `enabled` flag (toggled by [`TelemetryPublisher::stop`] and
//!   re-enabled by [`TelemetryPublisher::start`]).

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::cadence::{AdaptiveCadence, CadenceState, EffectiveCadence};
use super::schema::{
    inference_stats_to_json, AuditFingerprint, CpuStat, InferenceStat, LogRecord, MemStat,
};
use super::source::MetricsSource;
use super::{config_interval_ms, validate_interval_ms};
use crate::config::MetricsConfig;
use crate::surface_zenoh::json::{encode, JsonValue};

// ─── Keys ────────────────────────────────────────────────────────────────────

/// One enum per Phase 8 telemetry key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetricKey {
    /// `smallaios/metrics/cpu` — periodic.
    Cpu,
    /// `smallaios/metrics/mem` — periodic.
    Mem,
    /// `smallaios/metrics/inference` — periodic.
    Inference,
    /// `smallaios/metrics/log` — event-driven (drained on every tick
    /// when records are pending).
    Log,
    /// `smallaios/metrics/audit_fingerprint` — event-driven (republished
    /// when the chain head changes).
    AuditFingerprint,
}

impl MetricKey {
    /// Full Zenoh keyspace.
    pub const fn keyspace(self) -> &'static str {
        match self {
            Self::Cpu => "smallaios/metrics/cpu",
            Self::Mem => "smallaios/metrics/mem",
            Self::Inference => "smallaios/metrics/inference",
            Self::Log => "smallaios/metrics/log",
            Self::AuditFingerprint => "smallaios/metrics/audit_fingerprint",
        }
    }

    /// `true` iff the key publishes on a periodic cadence (CPU, mem,
    /// inference). Event-driven keys (`Log`, `AuditFingerprint`)
    /// return `false`.
    pub const fn is_periodic(self) -> bool {
        matches!(self, Self::Cpu | Self::Mem | Self::Inference)
    }

    /// All known keys, in declaration order.
    pub const fn all() -> &'static [MetricKey] {
        &METRIC_KEYS
    }
}

/// Stable iteration order for every [`MetricKey`]. Re-exported so
/// production glue (the future Zenoh wiring) can register publishers
/// in a deterministic sequence.
pub const METRIC_KEYS: [MetricKey; 5] = [
    MetricKey::Cpu,
    MetricKey::Mem,
    MetricKey::Inference,
    MetricKey::Log,
    MetricKey::AuditFingerprint,
];

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors returnable by the publisher's user-facing entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelemetryError {
    /// Configured `interval_ms` was below [`super::MIN_INTERVAL_MS`].
    IntervalBelowFloor,
    /// Configured `interval_ms` was above [`super::MAX_INTERVAL_MS`].
    IntervalAboveCeiling,
    /// `start` called twice without an intervening `stop`.
    AlreadyRunning,
    /// `publish_now` called against a stopped publisher.
    NotRunning,
    /// Sink callback returned an error.
    SinkError,
}

// ─── Sink ────────────────────────────────────────────────────────────────────

/// One published record handed to the sink. The `body` field is the
/// final UTF-8 JSON bytes ready for the Zenoh wire — the sink does
/// not need to re-encode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishRecord {
    /// Full Zenoh keyspace (`smallaios/metrics/<key>`).
    pub keyspace: &'static str,
    /// Encoded JSON payload.
    pub body: String,
}

/// Sink trait — the publisher routes every record to a `&mut Sink`.
/// Implementations SHALL return `Err(TelemetryError::SinkError)` only
/// on transport failure; bookkeeping bugs are the sink's concern.
pub trait PublishSink {
    /// Hand off `record` for transport. The publisher does not retain
    /// `record` — the sink may consume it directly.
    fn publish(&mut self, record: PublishRecord) -> Result<(), TelemetryError>;
}

/// `Vec`-backed sink for tests and the integration smoke. Records are
/// pushed in publication order.
#[derive(Debug, Default)]
pub struct CapturingSink {
    /// Captured records, in publication order.
    pub records: Vec<PublishRecord>,
}

impl CapturingSink {
    /// Construct an empty capturing sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of captured records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// `true` iff no records have been captured.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Records published to `keyspace`, in publication order.
    pub fn for_key(&self, keyspace: &str) -> Vec<&PublishRecord> {
        self.records
            .iter()
            .filter(|r| r.keyspace == keyspace)
            .collect()
    }

    /// Drop all captured records.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

impl PublishSink for CapturingSink {
    fn publish(&mut self, record: PublishRecord) -> Result<(), TelemetryError> {
        self.records.push(record);
        Ok(())
    }
}

// ─── Per-key state ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct KeyState {
    cadence: CadenceState,
    next_due_ms: u64,
    enabled: bool,
}

impl KeyState {
    fn new(cadence: CadenceState, start_ms: u64) -> Self {
        Self {
            cadence,
            next_due_ms: start_ms.saturating_add(cadence.base_interval_ms as u64),
            enabled: true,
        }
    }
}

// ─── Publisher ───────────────────────────────────────────────────────────────

/// Cooperative telemetry publisher.
pub struct TelemetryPublisher {
    keys: BTreeMap<MetricKey, KeyState>,
    started_at_ms: u64,
    running: bool,
    /// Soft cap on log records drained per tick. Prevents a noisy
    /// burst from monopolising the cooperative slot.
    log_drain_max_per_tick: usize,
}

impl TelemetryPublisher {
    /// Default soft cap on log drain per tick.
    pub const DEFAULT_LOG_DRAIN_PER_TICK: usize = 64;

    /// Build a stopped publisher. Call [`Self::start`] to engage.
    pub fn new() -> Self {
        Self {
            keys: BTreeMap::new(),
            started_at_ms: 0,
            running: false,
            log_drain_max_per_tick: Self::DEFAULT_LOG_DRAIN_PER_TICK,
        }
    }

    /// Replace the soft per-tick log drain cap. Bounded to `[1, 1024]`
    /// — out-of-range values clamp to the nearest endpoint.
    pub fn set_log_drain_max_per_tick(&mut self, max: usize) {
        self.log_drain_max_per_tick = max.clamp(1, 1024);
    }

    /// `true` iff [`Self::start`] has been called and [`Self::stop`]
    /// has not been called since.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Resolve the per-key cadence from `config` and engage the
    /// publisher. Subsequent calls return [`TelemetryError::AlreadyRunning`].
    pub fn start(&mut self, config: &MetricsConfig, now_ms: u64) -> Result<(), TelemetryError> {
        if self.running {
            return Err(TelemetryError::AlreadyRunning);
        }
        self.start_at(config, now_ms, None)
    }

    /// Variant of [`Self::start`] that lets the caller attach an
    /// adaptive override to specific keys. Used by the production
    /// glue when `metrics.cpu.adaptive` is set in `mgmt/policy.toml`.
    pub fn start_with_adaptive(
        &mut self,
        config: &MetricsConfig,
        now_ms: u64,
        cpu_adaptive: Option<AdaptiveCadence>,
    ) -> Result<(), TelemetryError> {
        if self.running {
            return Err(TelemetryError::AlreadyRunning);
        }
        self.start_at(config, now_ms, cpu_adaptive)
    }

    fn start_at(
        &mut self,
        config: &MetricsConfig,
        now_ms: u64,
        cpu_adaptive: Option<AdaptiveCadence>,
    ) -> Result<(), TelemetryError> {
        let mut keys = BTreeMap::new();
        for &k in METRIC_KEYS.iter() {
            let interval = config_interval_ms(k, config);
            let interval = validate_interval_ms(interval)?;
            let cadence = if k == MetricKey::Cpu {
                if let Some(a) = cpu_adaptive {
                    CadenceState::adaptive(interval, a)
                } else {
                    CadenceState::fixed(interval)
                }
            } else {
                CadenceState::fixed(interval)
            };
            keys.insert(k, KeyState::new(cadence, now_ms));
        }
        self.keys = keys;
        self.started_at_ms = now_ms;
        self.running = true;
        Ok(())
    }

    /// Disable the publisher. After `stop`, [`Self::tick`] becomes a
    /// no-op. The internal state is preserved so a subsequent
    /// [`Self::start`] can resume — but note that `start` rebuilds
    /// the deadlines from the supplied `now_ms`.
    pub fn stop(&mut self) {
        self.running = false;
        for ks in self.keys.values_mut() {
            ks.enabled = false;
        }
    }

    /// Disable a single key without stopping the whole publisher.
    pub fn disable_key(&mut self, key: MetricKey) {
        if let Some(ks) = self.keys.get_mut(&key) {
            ks.enabled = false;
        }
    }

    /// Re-enable a single key. The next [`Self::tick`] call will
    /// reset the key's deadline.
    pub fn enable_key(&mut self, key: MetricKey, now_ms: u64) {
        if let Some(ks) = self.keys.get_mut(&key) {
            ks.enabled = true;
            ks.next_due_ms = now_ms.saturating_add(ks.cadence.base_interval_ms as u64);
        }
    }

    /// Replace the cadence for `key`. The new cadence takes effect on
    /// the next deadline. Returns `Err(TelemetryError::NotRunning)`
    /// if the publisher has not been started.
    pub fn set_cadence(
        &mut self,
        key: MetricKey,
        cadence: CadenceState,
    ) -> Result<(), TelemetryError> {
        if !self.running {
            return Err(TelemetryError::NotRunning);
        }
        validate_interval_ms(cadence.base_interval_ms)?;
        if let Some(ks) = self.keys.get_mut(&key) {
            ks.cadence = cadence;
        }
        Ok(())
    }

    /// Diagnostic — read the configured cadence for `key`.
    pub fn cadence(&self, key: MetricKey) -> Option<CadenceState> {
        self.keys.get(&key).map(|k| k.cadence)
    }

    /// Diagnostic — read the next deadline for `key` (in `now_ms`
    /// units).
    pub fn next_due_ms(&self, key: MetricKey) -> Option<u64> {
        self.keys.get(&key).map(|k| k.next_due_ms)
    }

    /// Cooperative tick. Walks every key, publishes any that are due,
    /// rolls deadlines forward, and updates adaptive state. Returns
    /// the number of records published.
    pub fn tick<S, K>(
        &mut self,
        now_ms: u64,
        source: &mut S,
        sink: &mut K,
    ) -> Result<usize, TelemetryError>
    where
        S: MetricsSource,
        K: PublishSink,
    {
        if !self.running {
            return Ok(0);
        }
        let mut published = 0usize;
        // Iterate a stable order — METRIC_KEYS — to keep test
        // assertions reproducible.
        for &key in METRIC_KEYS.iter() {
            let due = matches!(
                self.keys.get(&key),
                Some(ks) if ks.enabled && ks.next_due_ms <= now_ms
            );
            if !due && !key.is_event_driven() {
                continue;
            }
            // Event-driven keys publish only when there's actually
            // something to send, regardless of deadline.
            if key.is_event_driven() {
                published += self.publish_event_key(key, now_ms, source, sink)?;
                self.advance_deadline(key, now_ms);
            } else if due {
                published += self.publish_periodic_key(key, now_ms, source, sink)?;
                self.advance_deadline(key, now_ms);
            }
        }
        Ok(published)
    }

    /// Force-publish every periodic key now, ignoring deadlines.
    /// Returns the count.
    pub fn publish_now<S, K>(
        &mut self,
        now_ms: u64,
        source: &mut S,
        sink: &mut K,
    ) -> Result<usize, TelemetryError>
    where
        S: MetricsSource,
        K: PublishSink,
    {
        if !self.running {
            return Err(TelemetryError::NotRunning);
        }
        let mut published = 0usize;
        for &key in METRIC_KEYS.iter() {
            if !key.is_periodic() {
                continue;
            }
            let enabled = self.keys.get(&key).map(|k| k.enabled).unwrap_or(false);
            if !enabled {
                continue;
            }
            published += self.publish_periodic_key(key, now_ms, source, sink)?;
            self.advance_deadline(key, now_ms);
        }
        Ok(published)
    }

    fn publish_periodic_key<S, K>(
        &mut self,
        key: MetricKey,
        now_ms: u64,
        source: &mut S,
        sink: &mut K,
    ) -> Result<usize, TelemetryError>
    where
        S: MetricsSource,
        K: PublishSink,
    {
        match key {
            MetricKey::Cpu => self.publish_cpu(now_ms, source, sink),
            MetricKey::Mem => self.publish_mem(source, sink),
            MetricKey::Inference => self.publish_inference(source, sink),
            // Periodic dispatch never reaches event-driven keys.
            MetricKey::Log | MetricKey::AuditFingerprint => Ok(0),
        }
    }

    fn publish_event_key<S, K>(
        &mut self,
        key: MetricKey,
        _now_ms: u64,
        source: &mut S,
        sink: &mut K,
    ) -> Result<usize, TelemetryError>
    where
        S: MetricsSource,
        K: PublishSink,
    {
        match key {
            MetricKey::Log => self.publish_logs(source, sink),
            MetricKey::AuditFingerprint => self.publish_audit_fingerprint(source, sink),
            MetricKey::Cpu | MetricKey::Mem | MetricKey::Inference => Ok(0),
        }
    }

    fn publish_cpu<S, K>(
        &mut self,
        _now_ms: u64,
        source: &mut S,
        sink: &mut K,
    ) -> Result<usize, TelemetryError>
    where
        S: MetricsSource,
        K: PublishSink,
    {
        let stat: CpuStat = source.cpu();
        let metric = stat.peak_util_pct();
        // Cadence reacts AFTER we capture this sample so the next
        // deadline reflects the latest reading.
        if let Some(ks) = self.keys.get_mut(&MetricKey::Cpu) {
            let _ = ks.cadence.update(metric);
        }
        let body = encode(&stat.to_json());
        sink.publish(PublishRecord {
            keyspace: MetricKey::Cpu.keyspace(),
            body,
        })?;
        Ok(1)
    }

    fn publish_mem<S, K>(&mut self, source: &mut S, sink: &mut K) -> Result<usize, TelemetryError>
    where
        S: MetricsSource,
        K: PublishSink,
    {
        let stat: MemStat = source.mem();
        let body = encode(&stat.to_json());
        sink.publish(PublishRecord {
            keyspace: MetricKey::Mem.keyspace(),
            body,
        })?;
        Ok(1)
    }

    fn publish_inference<S, K>(
        &mut self,
        source: &mut S,
        sink: &mut K,
    ) -> Result<usize, TelemetryError>
    where
        S: MetricsSource,
        K: PublishSink,
    {
        let per_model: Vec<InferenceStat> = source.inference();
        let stat = source.mem();
        let body = encode(&inference_stats_to_json(stat.ts_ns, &per_model));
        sink.publish(PublishRecord {
            keyspace: MetricKey::Inference.keyspace(),
            body,
        })?;
        Ok(1)
    }

    fn publish_logs<S, K>(&mut self, source: &mut S, sink: &mut K) -> Result<usize, TelemetryError>
    where
        S: MetricsSource,
        K: PublishSink,
    {
        let drained: Vec<LogRecord> = source.drain_logs(self.log_drain_max_per_tick);
        let mut count = 0usize;
        for record in drained {
            let body = encode(&record.to_json());
            sink.publish(PublishRecord {
                keyspace: MetricKey::Log.keyspace(),
                body,
            })?;
            count += 1;
        }
        Ok(count)
    }

    fn publish_audit_fingerprint<S, K>(
        &mut self,
        source: &mut S,
        sink: &mut K,
    ) -> Result<usize, TelemetryError>
    where
        S: MetricsSource,
        K: PublishSink,
    {
        if !source.audit_dirty() {
            return Ok(0);
        }
        let Some(fp): Option<AuditFingerprint> = source.audit_fingerprint() else {
            return Ok(0);
        };
        let body = encode(&fp.to_json());
        sink.publish(PublishRecord {
            keyspace: MetricKey::AuditFingerprint.keyspace(),
            body,
        })?;
        Ok(1)
    }

    fn advance_deadline(&mut self, key: MetricKey, now_ms: u64) {
        if let Some(ks) = self.keys.get_mut(&key) {
            // Use the current effective interval (post-update for
            // adaptive keys).
            let eff: EffectiveCadence = ks.cadence.current();
            let next = now_ms.saturating_add(eff.interval_ms as u64);
            // Catch up if we drifted: never schedule the deadline in
            // the past or the publisher would burn CPU until it
            // recovered.
            ks.next_due_ms = next.max(now_ms.saturating_add(eff.interval_ms as u64));
        }
    }
}

impl Default for TelemetryPublisher {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Internal helpers on MetricKey ───────────────────────────────────────────

impl MetricKey {
    /// `true` iff the key publishes only when there is something to
    /// publish (logs, audit fingerprint).
    pub const fn is_event_driven(self) -> bool {
        matches!(self, Self::Log | Self::AuditFingerprint)
    }

    /// Decode a wire body assuming the key's documented schema. Used
    /// by tests + by the production verifier to assert the wire
    /// schema is stable.
    pub fn decode(self, body: &[u8]) -> Result<JsonValue, TelemetryError> {
        crate::surface_zenoh::json::decode(body).map_err(|_| TelemetryError::SinkError)
    }
}

#[cfg(test)]
mod tests {
    use super::super::schema::{LogLevel, PerCpuUtil};
    use super::super::source::StubMetricsSource;
    use super::super::DEFAULT_INTERVAL_MS;
    use super::*;
    use crate::config::MetricsConfig;
    use alloc::string::ToString;

    fn quiet_config() -> MetricsConfig {
        // 1 Hz everywhere so the 1 Hz cadence sanity test is
        // unambiguous.
        let mut c = MetricsConfig::v1_defaults();
        c.cpu_interval_ms = 1000;
        c.memory_interval_ms = 1000;
        c.network_interval_ms = 1000;
        c.inference_interval_ms = 1000;
        c
    }

    #[test]
    fn metric_key_keyspaces_match_spec() {
        assert_eq!(MetricKey::Cpu.keyspace(), "smallaios/metrics/cpu");
        assert_eq!(MetricKey::Mem.keyspace(), "smallaios/metrics/mem");
        assert_eq!(
            MetricKey::Inference.keyspace(),
            "smallaios/metrics/inference"
        );
        assert_eq!(MetricKey::Log.keyspace(), "smallaios/metrics/log");
        assert_eq!(
            MetricKey::AuditFingerprint.keyspace(),
            "smallaios/metrics/audit_fingerprint"
        );
    }

    #[test]
    fn metric_key_periodic_split() {
        assert!(MetricKey::Cpu.is_periodic());
        assert!(MetricKey::Mem.is_periodic());
        assert!(MetricKey::Inference.is_periodic());
        assert!(!MetricKey::Log.is_periodic());
        assert!(!MetricKey::AuditFingerprint.is_periodic());
    }

    #[test]
    fn metric_key_event_driven_split() {
        assert!(!MetricKey::Cpu.is_event_driven());
        assert!(!MetricKey::Mem.is_event_driven());
        assert!(!MetricKey::Inference.is_event_driven());
        assert!(MetricKey::Log.is_event_driven());
        assert!(MetricKey::AuditFingerprint.is_event_driven());
    }

    #[test]
    fn all_returns_five_keys() {
        assert_eq!(MetricKey::all().len(), 5);
        assert_eq!(METRIC_KEYS.len(), 5);
    }

    #[test]
    fn new_publisher_is_stopped() {
        let p = TelemetryPublisher::new();
        assert!(!p.is_running());
    }

    #[test]
    fn start_engages_publisher_and_initialises_keys() {
        let mut p = TelemetryPublisher::new();
        p.start(&quiet_config(), 0).unwrap();
        assert!(p.is_running());
        // All 5 keys present.
        for &k in METRIC_KEYS.iter() {
            assert!(p.cadence(k).is_some());
        }
    }

    #[test]
    fn start_twice_returns_already_running() {
        let mut p = TelemetryPublisher::new();
        p.start(&quiet_config(), 0).unwrap();
        let err = p.start(&quiet_config(), 0).unwrap_err();
        assert_eq!(err, TelemetryError::AlreadyRunning);
    }

    #[test]
    fn start_below_floor_rejected() {
        let mut c = MetricsConfig::v1_defaults();
        c.cpu_interval_ms = 50;
        let mut p = TelemetryPublisher::new();
        let err = p.start(&c, 0).unwrap_err();
        assert_eq!(err, TelemetryError::IntervalBelowFloor);
        assert!(!p.is_running());
    }

    #[test]
    fn start_above_ceiling_rejected() {
        let mut c = MetricsConfig::v1_defaults();
        c.memory_interval_ms = 60_001;
        let mut p = TelemetryPublisher::new();
        let err = p.start(&c, 0).unwrap_err();
        assert_eq!(err, TelemetryError::IntervalAboveCeiling);
    }

    #[test]
    fn stop_disables_publisher() {
        let mut p = TelemetryPublisher::new();
        p.start(&quiet_config(), 0).unwrap();
        p.stop();
        assert!(!p.is_running());
    }

    #[test]
    fn stop_then_start_resumes() {
        let mut p = TelemetryPublisher::new();
        p.start(&quiet_config(), 0).unwrap();
        p.stop();
        // Resume.
        p.start(&quiet_config(), 5_000).unwrap();
        assert!(p.is_running());
        // Deadlines reset relative to the new start time.
        assert_eq!(p.next_due_ms(MetricKey::Cpu), Some(6_000));
    }

    #[test]
    fn publish_now_against_stopped_returns_not_running() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        let err = p.publish_now(0, &mut src, &mut sink).unwrap_err();
        assert_eq!(err, TelemetryError::NotRunning);
    }

    #[test]
    fn publish_now_emits_three_periodic_records() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        let n = p.publish_now(0, &mut src, &mut sink).unwrap();
        assert_eq!(n, 3);
        assert_eq!(sink.records.len(), 3);
    }

    #[test]
    fn publish_now_records_each_key_exactly_once() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.publish_now(0, &mut src, &mut sink).unwrap();
        assert_eq!(sink.for_key("smallaios/metrics/cpu").len(), 1);
        assert_eq!(sink.for_key("smallaios/metrics/mem").len(), 1);
        assert_eq!(sink.for_key("smallaios/metrics/inference").len(), 1);
    }

    #[test]
    fn tick_does_nothing_when_stopped() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        let n = p.tick(0, &mut src, &mut sink).unwrap();
        assert_eq!(n, 0);
        assert!(sink.is_empty());
    }

    #[test]
    fn tick_below_deadline_publishes_nothing_periodic() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        // 999 ms < 1000 ms deadline.
        let n = p.tick(999, &mut src, &mut sink).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn tick_at_deadline_publishes_periodic_keys() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        let n = p.tick(1000, &mut src, &mut sink).unwrap();
        assert_eq!(n, 3); // cpu + mem + inference
    }

    #[test]
    fn tick_advances_deadlines_to_next_window() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.tick(1000, &mut src, &mut sink).unwrap();
        assert_eq!(p.next_due_ms(MetricKey::Cpu), Some(2000));
    }

    #[test]
    fn five_second_window_yields_five_cpu_publications_at_1_hz() {
        // Spec scenario "Default 1 Hz cadence": exactly once per
        // 1000 ms.
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        for ms in (0..=5000).step_by(100) {
            p.tick(ms, &mut src, &mut sink).unwrap();
        }
        let cpu = sink.for_key("smallaios/metrics/cpu").len();
        // Deadlines: 1000, 2000, 3000, 4000, 5000 → 5 publications.
        assert_eq!(cpu, 5);
    }

    #[test]
    fn ten_hz_inference_override_yields_50_publications_in_5s() {
        // Spec scenario "Operator tunes inference cadence":
        // metrics.inference.interval_ms = 100 → 10 Hz.
        let mut c = quiet_config();
        c.inference_interval_ms = 100;
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&c, 0).unwrap();
        for ms in (0..=5000).step_by(50) {
            p.tick(ms, &mut src, &mut sink).unwrap();
        }
        let inference = sink.for_key("smallaios/metrics/inference").len();
        assert_eq!(inference, 50);
    }

    #[test]
    fn cpu_publication_round_trips_per_core_array() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        src.set_cpu(CpuStat {
            ts_ns: 1,
            per_core: alloc::vec![
                PerCpuUtil { id: 0, util_pct: 1 },
                PerCpuUtil { id: 1, util_pct: 2 },
                PerCpuUtil { id: 2, util_pct: 3 },
                PerCpuUtil { id: 3, util_pct: 4 },
            ],
        });
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.publish_now(0, &mut src, &mut sink).unwrap();
        let recs = sink.for_key("smallaios/metrics/cpu");
        assert_eq!(recs.len(), 1);
        let parsed = MetricKey::Cpu.decode(recs[0].body.as_bytes()).unwrap();
        let arr = match parsed.get("per_core").unwrap() {
            JsonValue::Array(a) => a,
            _ => panic!("array expected"),
        };
        assert_eq!(arr.len(), 4);
    }

    #[test]
    fn mem_publication_round_trips_all_fields() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        src.set_mem(MemStat {
            ts_ns: 999,
            heap_used_bytes: 4096,
            heap_cap_bytes: 8192,
            page_alloc_count: 11,
            page_free_count: 9,
        });
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.publish_now(0, &mut src, &mut sink).unwrap();
        let recs = sink.for_key("smallaios/metrics/mem");
        let parsed = MetricKey::Mem.decode(recs[0].body.as_bytes()).unwrap();
        assert_eq!(
            parsed.get("heap_used_bytes").unwrap().as_int().unwrap(),
            4096
        );
    }

    #[test]
    fn inference_publication_round_trips_per_model_array() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        src.set_inference(alloc::vec![
            InferenceStat {
                name: "a".to_string(),
                qps: 10,
                p50_us: 100,
                p99_us: 200,
                error_count: 0,
            },
            InferenceStat {
                name: "b".to_string(),
                qps: 5,
                p50_us: 50,
                p99_us: 75,
                error_count: 1,
            },
        ]);
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.publish_now(0, &mut src, &mut sink).unwrap();
        let recs = sink.for_key("smallaios/metrics/inference");
        let parsed = MetricKey::Inference
            .decode(recs[0].body.as_bytes())
            .unwrap();
        let arr = match parsed.get("per_model").unwrap() {
            JsonValue::Array(a) => a,
            _ => panic!(),
        };
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn log_records_drained_on_tick() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        src.push_log(LogLevel::Info, "t", "first");
        src.push_log(LogLevel::Warn, "t", "second");
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.tick(0, &mut src, &mut sink).unwrap();
        let logs = sink.for_key("smallaios/metrics/log");
        assert_eq!(logs.len(), 2);
        assert_eq!(src.log_pending(), 0);
    }

    #[test]
    fn log_drain_respects_per_tick_cap() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        for i in 0..200 {
            src.push_log(LogLevel::Info, "t", &alloc::format!("m{}", i));
        }
        let mut sink = CapturingSink::new();
        p.set_log_drain_max_per_tick(10);
        p.start(&quiet_config(), 0).unwrap();
        p.tick(0, &mut src, &mut sink).unwrap();
        assert_eq!(sink.for_key("smallaios/metrics/log").len(), 10);
        assert_eq!(src.log_pending(), 190);
    }

    #[test]
    fn log_drain_cap_clamps_to_one() {
        let mut p = TelemetryPublisher::new();
        p.set_log_drain_max_per_tick(0);
        // log_drain_max_per_tick is private; observe via behaviour.
        let mut src = StubMetricsSource::new();
        for _ in 0..3 {
            src.push_log(LogLevel::Info, "t", "x");
        }
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.tick(0, &mut src, &mut sink).unwrap();
        assert_eq!(sink.for_key("smallaios/metrics/log").len(), 1);
    }

    #[test]
    fn log_drain_cap_clamps_to_max_1024() {
        let mut p = TelemetryPublisher::new();
        p.set_log_drain_max_per_tick(usize::MAX);
        let mut src = StubMetricsSource::new();
        for _ in 0..2000 {
            src.push_log(LogLevel::Info, "t", "x");
        }
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.tick(0, &mut src, &mut sink).unwrap();
        assert_eq!(sink.for_key("smallaios/metrics/log").len(), 1024);
    }

    #[test]
    fn audit_fingerprint_silent_when_chain_empty() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.tick(0, &mut src, &mut sink).unwrap();
        assert!(sink
            .for_key("smallaios/metrics/audit_fingerprint")
            .is_empty());
    }

    #[test]
    fn audit_fingerprint_publishes_on_first_record() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        src.push_audit("aabb", 1);
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.tick(0, &mut src, &mut sink).unwrap();
        let fps = sink.for_key("smallaios/metrics/audit_fingerprint");
        assert_eq!(fps.len(), 1);
    }

    #[test]
    fn audit_fingerprint_skipped_when_clean() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        src.push_audit("aabb", 1);
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.tick(0, &mut src, &mut sink).unwrap();
        sink.clear();
        // Second tick — no new audit records.
        p.tick(0, &mut src, &mut sink).unwrap();
        assert!(sink
            .for_key("smallaios/metrics/audit_fingerprint")
            .is_empty());
    }

    #[test]
    fn audit_fingerprint_publishes_again_after_new_record() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        src.push_audit("aa", 1);
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.tick(0, &mut src, &mut sink).unwrap();
        // Second record → fingerprint changes.
        src.push_audit("bb", 2);
        sink.clear();
        p.tick(0, &mut src, &mut sink).unwrap();
        let fps = sink.for_key("smallaios/metrics/audit_fingerprint");
        assert_eq!(fps.len(), 1);
        let parsed = MetricKey::AuditFingerprint
            .decode(fps[0].body.as_bytes())
            .unwrap();
        assert_eq!(
            parsed.get("hex_fingerprint").unwrap().as_str().unwrap(),
            "bb"
        );
        assert_eq!(parsed.get("record_count").unwrap().as_int().unwrap(), 2);
    }

    #[test]
    fn adaptive_cpu_engages_on_threshold_cross() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start_with_adaptive(&quiet_config(), 0, Some(AdaptiveCadence::new(80, 10, 1)))
            .unwrap();
        // Below threshold → slow tier.
        src.set_cpu(CpuStat {
            ts_ns: 0,
            per_core: alloc::vec![PerCpuUtil {
                id: 0,
                util_pct: 50,
            }],
        });
        p.tick(1000, &mut src, &mut sink).unwrap();
        // After tick the cadence should still be slow.
        let cur = p.cadence(MetricKey::Cpu).unwrap().current();
        assert!(!cur.fast_tier);
        assert_eq!(cur.interval_ms, 1000);

        // Above threshold → fast tier.
        src.set_cpu(CpuStat {
            ts_ns: 0,
            per_core: alloc::vec![PerCpuUtil {
                id: 0,
                util_pct: 95,
            }],
        });
        p.tick(2000, &mut src, &mut sink).unwrap();
        let cur = p.cadence(MetricKey::Cpu).unwrap().current();
        assert!(cur.fast_tier);
        assert_eq!(cur.interval_ms, 100);
    }

    #[test]
    fn adaptive_cpu_reverts_to_slow_below_threshold() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start_with_adaptive(&quiet_config(), 0, Some(AdaptiveCadence::new(80, 10, 1)))
            .unwrap();
        src.set_cpu(CpuStat {
            ts_ns: 0,
            per_core: alloc::vec![PerCpuUtil {
                id: 0,
                util_pct: 95,
            }],
        });
        p.tick(1000, &mut src, &mut sink).unwrap();
        assert!(p.cadence(MetricKey::Cpu).unwrap().current().fast_tier);
        // Drop below.
        src.set_cpu(CpuStat {
            ts_ns: 0,
            per_core: alloc::vec![PerCpuUtil {
                id: 0,
                util_pct: 30,
            }],
        });
        p.tick(2000, &mut src, &mut sink).unwrap();
        assert!(!p.cadence(MetricKey::Cpu).unwrap().current().fast_tier);
    }

    #[test]
    fn adaptive_fast_tier_publishes_at_10hz() {
        let mut c = quiet_config();
        c.cpu_interval_ms = 1000;
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        // Engage fast tier from the start by setting the metric.
        src.set_cpu(CpuStat {
            ts_ns: 0,
            per_core: alloc::vec![PerCpuUtil {
                id: 0,
                util_pct: 99,
            }],
        });
        let mut sink = CapturingSink::new();
        p.start_with_adaptive(&c, 0, Some(AdaptiveCadence::new(80, 10, 1)))
            .unwrap();
        // Force the first publication to lock in the fast tier.
        p.publish_now(0, &mut src, &mut sink).unwrap();
        sink.clear();
        for ms in 100..=1100 {
            if ms % 10 == 0 {
                p.tick(ms, &mut src, &mut sink).unwrap();
            }
        }
        // Approximately 10 publications in the 1000ms window. The
        // exact value can be 10 or 11 depending on alignment of the
        // first deadline; assert a reasonable range.
        let cpu = sink.for_key("smallaios/metrics/cpu").len();
        assert!(
            (9..=11).contains(&cpu),
            "expected ~10 cpu publications, got {}",
            cpu
        );
    }

    #[test]
    fn set_cadence_takes_effect_after_next_tick() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.set_cadence(MetricKey::Cpu, CadenceState::fixed(500))
            .unwrap();
        // Advance to first publication then reset.
        p.tick(1000, &mut src, &mut sink).unwrap();
        sink.clear();
        // Now at 500ms cadence: advance by 500ms, expect 1 cpu pub.
        p.tick(1500, &mut src, &mut sink).unwrap();
        assert_eq!(sink.for_key("smallaios/metrics/cpu").len(), 1);
    }

    #[test]
    fn set_cadence_below_floor_rejected() {
        let mut p = TelemetryPublisher::new();
        p.start(&quiet_config(), 0).unwrap();
        let err = p
            .set_cadence(MetricKey::Cpu, CadenceState::fixed(50))
            .unwrap_err();
        assert_eq!(err, TelemetryError::IntervalBelowFloor);
    }

    #[test]
    fn set_cadence_when_stopped_returns_not_running() {
        let mut p = TelemetryPublisher::new();
        let err = p
            .set_cadence(MetricKey::Cpu, CadenceState::fixed(1000))
            .unwrap_err();
        assert_eq!(err, TelemetryError::NotRunning);
    }

    #[test]
    fn disable_key_suppresses_periodic_publication() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.disable_key(MetricKey::Cpu);
        for ms in (0..=5000).step_by(100) {
            p.tick(ms, &mut src, &mut sink).unwrap();
        }
        assert!(sink.for_key("smallaios/metrics/cpu").is_empty());
        // Other periodic keys still publish.
        assert_eq!(sink.for_key("smallaios/metrics/mem").len(), 5);
    }

    #[test]
    fn enable_key_resets_deadline_relative_to_now() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.disable_key(MetricKey::Cpu);
        p.tick(2000, &mut src, &mut sink).unwrap();
        sink.clear();
        p.enable_key(MetricKey::Cpu, 2000);
        assert_eq!(p.next_due_ms(MetricKey::Cpu), Some(3000));
    }

    #[test]
    fn periodic_publish_tracks_drift_after_late_tick() {
        // If the cooperative scheduler is late, the publisher
        // SHALL not schedule the next deadline in the past.
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        // Fire a tick 5s late. The publisher emits one record (it
        // does not catch up multiple intervals) and reschedules
        // based on now_ms + interval.
        p.tick(5000, &mut src, &mut sink).unwrap();
        assert_eq!(p.next_due_ms(MetricKey::Cpu), Some(6000));
    }

    #[test]
    fn record_keyspace_is_static_str() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.publish_now(0, &mut src, &mut sink).unwrap();
        // All keyspaces match METRIC_KEYS entries (compile-time
        // static strings).
        for r in &sink.records {
            let mut found = false;
            for &k in METRIC_KEYS.iter() {
                if r.keyspace == k.keyspace() {
                    found = true;
                    break;
                }
            }
            assert!(found, "unknown keyspace {}", r.keyspace);
        }
    }

    #[test]
    fn publisher_does_not_publish_log_when_no_records() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        let mut sink = CapturingSink::new();
        p.start(&quiet_config(), 0).unwrap();
        p.tick(1000, &mut src, &mut sink).unwrap();
        assert!(sink.for_key("smallaios/metrics/log").is_empty());
    }

    #[test]
    fn default_publisher_has_drain_cap_64() {
        assert_eq!(TelemetryPublisher::DEFAULT_LOG_DRAIN_PER_TICK, 64);
    }

    #[test]
    fn custom_drain_cap_round_trips() {
        let mut p = TelemetryPublisher::new();
        let mut src = StubMetricsSource::new();
        for _ in 0..100 {
            src.push_log(LogLevel::Info, "t", "x");
        }
        let mut sink = CapturingSink::new();
        p.set_log_drain_max_per_tick(50);
        p.start(&quiet_config(), 0).unwrap();
        p.tick(0, &mut src, &mut sink).unwrap();
        assert_eq!(sink.for_key("smallaios/metrics/log").len(), 50);
    }

    #[test]
    fn cadence_returns_none_for_unknown_key_after_stop() {
        let mut p = TelemetryPublisher::new();
        // Before start, no keys initialised.
        assert!(p.cadence(MetricKey::Cpu).is_none());
        p.start(&quiet_config(), 0).unwrap();
        assert!(p.cadence(MetricKey::Cpu).is_some());
    }

    #[test]
    fn next_due_ms_for_unknown_key_is_none_before_start() {
        let p = TelemetryPublisher::new();
        assert!(p.next_due_ms(MetricKey::Mem).is_none());
    }

    #[test]
    fn default_interval_constant_matches_spec() {
        assert_eq!(DEFAULT_INTERVAL_MS, 1000);
    }
}
