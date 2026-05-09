// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Zenoh telemetry keyspace — periodic + event-driven publishers under
//! `smallaios/metrics/**`.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-zenoh-telemetry/spec.md`
//! Requirements:
//!
//! - "Telemetry keyspace schema"
//! - "Per-key configurable cadence with adaptive option"
//!
//! ## Keyspace
//!
//! Phase 8 v1 keys (every payload is a JSON object):
//!
//! | Key                                  | Cadence | Shape                                                                                |
//! |--------------------------------------|---------|--------------------------------------------------------------------------------------|
//! | `smallaios/metrics/cpu`              | 1 Hz    | `{ ts, per_core: [{ id, util_pct }] }`                                               |
//! | `smallaios/metrics/mem`              | 1 Hz    | `{ ts, heap_used_bytes, heap_cap_bytes, page_alloc_count, page_free_count }`         |
//! | `smallaios/metrics/inference`        | 1 Hz    | `{ ts, per_model: [{ name, qps, p50_us, p99_us, error_count }] }`                    |
//! | `smallaios/metrics/log`              | event   | `{ ts, level, target, msg, fields }`                                                 |
//! | `smallaios/metrics/audit_fingerprint`| event   | `{ ts, hex_fingerprint, record_count }`                                              |
//!
//! ## Module map
//!
//! - [`source`] — `MetricsSource` trait and the synthetic
//!   `StubMetricsSource` used by tests + integration smoke.
//! - [`schema`] — strongly-typed `*Stat` records and their
//!   `to_json` projections through the existing
//!   [`super::json::JsonValue`] codec.
//! - [`cadence`] — per-key cadence resolver: clamps interval bounds
//!   (100 ms..=60 s) and applies an optional adaptive override.
//! - [`publisher`] — the cooperative driver: holds per-key deadlines,
//!   pulls from a [`MetricsSource`], routes to a [`PublishSink`].
//!
//! ## Transport-agnostic by design
//!
//! Mirroring [`super::AdminDispatcher`], `TelemetryPublisher` does not
//! own a Zenoh session — it accepts a [`PublishSink`] callback so the
//! same driver runs against the in-flight `ipc-zenoh` transport in
//! production and a synthetic `Vec`-backed sink in tests. The
//! production glue (a future [`smallaios-ipc`] phase) supplies the
//! Zenoh-backed implementation.

pub mod cadence;
pub mod publisher;
pub mod schema;
pub mod source;

#[cfg(test)]
mod telemetry_test_vectors;

pub use cadence::{CadenceState, EffectiveCadence};
pub use publisher::{
    MetricKey, PublishRecord, PublishSink, TelemetryError, TelemetryPublisher, METRIC_KEYS,
};
pub use schema::{
    AuditFingerprint, CpuStat, InferenceStat, LogLevel, LogRecord, MemStat, PerCpuUtil,
};
pub use source::{MetricsSource, StubMetricsSource};

use crate::config::MetricsConfig;

/// Default cadence for every metric key, per the spec — 1 Hz.
pub const DEFAULT_INTERVAL_MS: u32 = 1000;

/// Lower bound on per-key interval. Mirrors
/// [`crate::validate::METRICS_MIN_INTERVAL_MS`]; re-exported here so
/// callers don't have to reach across modules.
pub const MIN_INTERVAL_MS: u32 = crate::validate::METRICS_MIN_INTERVAL_MS;

/// Upper bound on per-key interval (60 s).
pub const MAX_INTERVAL_MS: u32 = crate::validate::METRICS_MAX_INTERVAL_MS;

/// Resolve the configured interval (in ms) for `key` from a
/// [`MetricsConfig`]. Falls back to [`DEFAULT_INTERVAL_MS`] when the
/// `Config` does not have a dedicated field for `key` (e.g. the
/// event-driven keys).
pub fn config_interval_ms(key: MetricKey, config: &MetricsConfig) -> u32 {
    match key {
        MetricKey::Cpu => config.cpu_interval_ms,
        MetricKey::Mem => config.memory_interval_ms,
        MetricKey::Inference => config.inference_interval_ms,
        // Event-driven keys ignore periodic cadence — but the
        // publisher still uses the value as a sanity floor when the
        // caller asks for periodic publication on those keys (used
        // only by tests).
        MetricKey::Log | MetricKey::AuditFingerprint => DEFAULT_INTERVAL_MS,
    }
}

/// Clamp `ms` to the [`MIN_INTERVAL_MS`]..=[`MAX_INTERVAL_MS`] window.
/// Out-of-range inputs return `Err`. Callers that want best-effort
/// clamping should use [`clamp_interval`].
pub fn validate_interval_ms(ms: u32) -> Result<u32, TelemetryError> {
    if ms < MIN_INTERVAL_MS {
        return Err(TelemetryError::IntervalBelowFloor);
    }
    if ms > MAX_INTERVAL_MS {
        return Err(TelemetryError::IntervalAboveCeiling);
    }
    Ok(ms)
}

/// Clamp `ms` to the legal window without erroring. Used internally by
/// the publisher when a runtime override (e.g. adaptive `fast_hz`)
/// translates into an interval below the floor.
pub fn clamp_interval(ms: u32) -> u32 {
    ms.clamp(MIN_INTERVAL_MS, MAX_INTERVAL_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_interval_resolves_cpu_mem_inference() {
        let mut c = MetricsConfig::v1_defaults();
        c.cpu_interval_ms = 500;
        c.memory_interval_ms = 750;
        c.inference_interval_ms = 250;
        assert_eq!(config_interval_ms(MetricKey::Cpu, &c), 500);
        assert_eq!(config_interval_ms(MetricKey::Mem, &c), 750);
        assert_eq!(config_interval_ms(MetricKey::Inference, &c), 250);
    }

    #[test]
    fn config_interval_event_driven_falls_back_to_default() {
        let c = MetricsConfig::v1_defaults();
        assert_eq!(config_interval_ms(MetricKey::Log, &c), DEFAULT_INTERVAL_MS);
        assert_eq!(
            config_interval_ms(MetricKey::AuditFingerprint, &c),
            DEFAULT_INTERVAL_MS
        );
    }

    #[test]
    fn validate_interval_ms_accepts_legal_values() {
        assert_eq!(validate_interval_ms(MIN_INTERVAL_MS).unwrap(), 100);
        assert_eq!(validate_interval_ms(1000).unwrap(), 1000);
        assert_eq!(validate_interval_ms(MAX_INTERVAL_MS).unwrap(), 60_000);
    }

    #[test]
    fn validate_interval_ms_rejects_below_floor() {
        assert_eq!(
            validate_interval_ms(99),
            Err(TelemetryError::IntervalBelowFloor)
        );
        assert_eq!(
            validate_interval_ms(0),
            Err(TelemetryError::IntervalBelowFloor)
        );
    }

    #[test]
    fn validate_interval_ms_rejects_above_ceiling() {
        assert_eq!(
            validate_interval_ms(60_001),
            Err(TelemetryError::IntervalAboveCeiling)
        );
        assert_eq!(
            validate_interval_ms(u32::MAX),
            Err(TelemetryError::IntervalAboveCeiling)
        );
    }

    #[test]
    fn clamp_interval_clips_below_and_above() {
        assert_eq!(clamp_interval(0), MIN_INTERVAL_MS);
        assert_eq!(clamp_interval(50), MIN_INTERVAL_MS);
        assert_eq!(clamp_interval(1000), 1000);
        assert_eq!(clamp_interval(120_000), MAX_INTERVAL_MS);
    }
}
