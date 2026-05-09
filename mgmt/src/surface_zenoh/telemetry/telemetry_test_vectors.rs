// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Test fixtures for the Zenoh telemetry surface — sample audit
// fingerprints, canonical CPU/mem/inference snapshots, sample log
// records. Lives in a `*_test_vectors.rs` file so CodeQL's byte-array
// / hex-constant scanners (`paths-ignored` rule in
// `.github/codeql/codeql-config.yml`) skip the file. Production code
// never references these constants; only `#[cfg(test)]` consumers in
// this module tree do.

#![cfg(test)]

use alloc::string::ToString;
use alloc::vec;

use super::schema::{
    AuditFingerprint, CpuStat, InferenceStat, LogLevel, LogRecord, MemStat, PerCpuUtil,
};

/// Worked-example all-zeroes fingerprint used to exercise the empty
/// chain path. Not a real fingerprint for any production audit chain.
pub const SAMPLE_FINGERPRINT_ZERO_HEX: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Worked-example "deadbeef" prefix fingerprint used to assert the
/// publisher's hex pass-through. The last 56 chars are arbitrary
/// padding.
pub const SAMPLE_FINGERPRINT_DEADBEEF_HEX: &str =
    "deadbeefcafebabefeedfacedeadbeefcafebabefeedfacedeadbeefcafebabe";

/// Construct a 4-core CPU sample with monotonically rising util_pct.
/// Used by the schema round-trip test to assert per-core ordering is
/// preserved by the JSON codec.
pub fn sample_cpu_stat_4core(ts_ns: u64) -> CpuStat {
    CpuStat {
        ts_ns,
        per_core: vec![
            PerCpuUtil {
                id: 0,
                util_pct: 10,
            },
            PerCpuUtil {
                id: 1,
                util_pct: 25,
            },
            PerCpuUtil {
                id: 2,
                util_pct: 50,
            },
            PerCpuUtil {
                id: 3,
                util_pct: 75,
            },
        ],
    }
}

/// Construct a memory sample with arbitrary-but-stable counters.
pub fn sample_mem_stat() -> MemStat {
    MemStat {
        ts_ns: 1_700_000_000_000_000_000,
        heap_used_bytes: 4 * 1024 * 1024,
        heap_cap_bytes: 16 * 1024 * 1024,
        page_alloc_count: 1024,
        page_free_count: 512,
    }
}

/// Construct two inference-stat fixtures.
pub fn sample_inference_stats() -> alloc::vec::Vec<InferenceStat> {
    vec![
        InferenceStat {
            name: "resnet50".to_string(),
            qps: 250,
            p50_us: 1500,
            p99_us: 9500,
            error_count: 0,
        },
        InferenceStat {
            name: "yolov8n".to_string(),
            qps: 60,
            p50_us: 5000,
            p99_us: 32000,
            error_count: 3,
        },
    ]
}

/// Construct a log record with structured fields.
pub fn sample_log_record() -> LogRecord {
    LogRecord {
        ts_ns: 42_000_000,
        level: LogLevel::Warn,
        target: "kernel::sched".to_string(),
        msg: "queue contention".to_string(),
        fields: vec![
            ("queue".to_string(), "core1".to_string()),
            ("depth".to_string(), "17".to_string()),
        ],
    }
}

/// Construct an audit fingerprint fixture using the deadbeef sample.
pub fn sample_audit_fingerprint() -> AuditFingerprint {
    AuditFingerprint {
        ts_ns: 1_700_000_000_000_000_000,
        hex_fingerprint: SAMPLE_FINGERPRINT_DEADBEEF_HEX.to_string(),
        record_count: 1024,
    }
}

#[cfg(test)]
mod tests {
    use super::super::publisher::{CapturingSink, MetricKey, PublishSink, TelemetryPublisher};
    use super::super::source::StubMetricsSource;
    use super::*;
    use crate::config::MetricsConfig;
    use crate::surface_zenoh::json::{decode, JsonValue};

    #[test]
    fn cpu_fixture_has_four_cores() {
        let s = sample_cpu_stat_4core(1);
        assert_eq!(s.per_core.len(), 4);
        assert_eq!(s.peak_util_pct(), 75);
    }

    #[test]
    fn mem_fixture_capacity_exceeds_used() {
        let m = sample_mem_stat();
        assert!(m.heap_cap_bytes > m.heap_used_bytes);
    }

    #[test]
    fn inference_fixture_has_two_models() {
        let s = sample_inference_stats();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].name, "resnet50");
        assert_eq!(s[1].name, "yolov8n");
    }

    #[test]
    fn log_fixture_has_two_fields() {
        let r = sample_log_record();
        assert_eq!(r.fields.len(), 2);
    }

    #[test]
    fn fingerprint_zero_constant_is_64_chars() {
        assert_eq!(SAMPLE_FINGERPRINT_ZERO_HEX.len(), 64);
    }

    #[test]
    fn fingerprint_deadbeef_constant_is_64_chars() {
        assert_eq!(SAMPLE_FINGERPRINT_DEADBEEF_HEX.len(), 64);
    }

    #[test]
    fn full_round_trip_of_cpu_fixture_through_publisher() {
        // End-to-end: stub source → publisher → capturing sink → JSON
        // decode → assert per_core length matches fixture.
        let mut src = StubMetricsSource::new();
        src.set_cpu(sample_cpu_stat_4core(1_700_000_000_000_000_000));
        let mut sink = CapturingSink::new();
        let mut p = TelemetryPublisher::new();
        let cfg = MetricsConfig::v1_defaults();
        p.start(&cfg, 0).unwrap();
        p.publish_now(0, &mut src, &mut sink).unwrap();
        let recs = sink.for_key("smallaios/metrics/cpu");
        assert_eq!(recs.len(), 1);
        let parsed = decode(recs[0].body.as_bytes()).unwrap();
        let arr = match parsed.get("per_core").unwrap() {
            JsonValue::Array(a) => a,
            _ => panic!(),
        };
        assert_eq!(arr.len(), 4);
    }

    #[test]
    fn audit_fingerprint_fixture_round_trips() {
        let fp = sample_audit_fingerprint();
        let v = fp.to_json();
        let s = crate::surface_zenoh::json::encode(&v);
        let back = decode(s.as_bytes()).unwrap();
        let obj = back.as_object().unwrap();
        assert_eq!(
            obj.get("hex_fingerprint").unwrap().as_str().unwrap(),
            SAMPLE_FINGERPRINT_DEADBEEF_HEX
        );
        assert_eq!(obj.get("record_count").unwrap().as_int().unwrap(), 1024);
    }

    #[test]
    fn audit_fingerprint_zero_fixture_round_trips() {
        let fp = AuditFingerprint {
            ts_ns: 0,
            hex_fingerprint: SAMPLE_FINGERPRINT_ZERO_HEX.to_string(),
            record_count: 0,
        };
        let v = fp.to_json();
        let s = crate::surface_zenoh::json::encode(&v);
        let back = decode(s.as_bytes()).unwrap();
        let obj = back.as_object().unwrap();
        assert_eq!(
            obj.get("hex_fingerprint").unwrap().as_str().unwrap(),
            SAMPLE_FINGERPRINT_ZERO_HEX
        );
    }

    #[test]
    fn capturing_sink_round_trips_records() {
        let mut sink = CapturingSink::new();
        let r = super::super::publisher::PublishRecord {
            keyspace: MetricKey::Cpu.keyspace(),
            body: "{}".to_string(),
        };
        sink.publish(r.clone()).unwrap();
        assert_eq!(sink.len(), 1);
        assert_eq!(sink.records[0], r);
    }
}
