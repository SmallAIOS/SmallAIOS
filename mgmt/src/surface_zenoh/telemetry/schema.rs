// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Strongly-typed metric records and their JSON projections.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-zenoh-telemetry/spec.md`
//! Requirement "Telemetry keyspace schema".
//!
//! Every record has a `to_json` method that yields a
//! [`super::super::json::JsonValue::Object`] in the shape the spec
//! mandates. The schema is **additive** — new optional fields may be
//! appended; existing fields SHALL NOT be renamed or repurposed.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::super::json::JsonValue;

// ─── CPU ─────────────────────────────────────────────────────────────────────

/// Per-core CPU utilisation sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerCpuUtil {
    /// Logical core id (0-based).
    pub id: u32,
    /// Utilisation in `0..=100` percent.
    pub util_pct: u32,
}

impl PerCpuUtil {
    /// Project to a flat JSON object.
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("id".to_string(), JsonValue::from(self.id));
        m.insert("util_pct".to_string(), JsonValue::from(self.util_pct));
        JsonValue::Object(m)
    }
}

/// `smallaios/metrics/cpu` payload — `{ ts, per_core: [...] }`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CpuStat {
    /// Wall-clock timestamp in nanoseconds since UNIX epoch (or
    /// monotonic-since-boot when no RTC is available).
    pub ts_ns: u64,
    /// Per-core utilisation samples in core-id order.
    pub per_core: Vec<PerCpuUtil>,
}

impl CpuStat {
    /// Project to the on-wire JSON shape.
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("ts".to_string(), JsonValue::from(self.ts_ns));
        let arr: Vec<JsonValue> = self.per_core.iter().map(PerCpuUtil::to_json).collect();
        m.insert("per_core".to_string(), JsonValue::Array(arr));
        JsonValue::Object(m)
    }

    /// Highest `util_pct` observed in this sample. Returns `0` when
    /// `per_core` is empty.
    pub fn peak_util_pct(&self) -> u32 {
        self.per_core.iter().map(|c| c.util_pct).max().unwrap_or(0)
    }
}

// ─── Memory ──────────────────────────────────────────────────────────────────

/// `smallaios/metrics/mem` payload —
/// `{ ts, heap_used_bytes, heap_cap_bytes, page_alloc_count, page_free_count }`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemStat {
    /// Timestamp (ns since epoch).
    pub ts_ns: u64,
    /// Bytes currently in use by the kernel heap.
    pub heap_used_bytes: u64,
    /// Total heap capacity in bytes.
    pub heap_cap_bytes: u64,
    /// Cumulative page allocations since boot.
    pub page_alloc_count: u64,
    /// Cumulative page frees since boot.
    pub page_free_count: u64,
}

impl MemStat {
    /// Project to JSON.
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("ts".to_string(), JsonValue::from(self.ts_ns));
        m.insert(
            "heap_used_bytes".to_string(),
            JsonValue::from(self.heap_used_bytes),
        );
        m.insert(
            "heap_cap_bytes".to_string(),
            JsonValue::from(self.heap_cap_bytes),
        );
        m.insert(
            "page_alloc_count".to_string(),
            JsonValue::from(self.page_alloc_count),
        );
        m.insert(
            "page_free_count".to_string(),
            JsonValue::from(self.page_free_count),
        );
        JsonValue::Object(m)
    }
}

// ─── Inference ───────────────────────────────────────────────────────────────

/// Per-model inference statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceStat {
    /// Model name (matches the `/models/<name>.onnx` basename).
    pub name: String,
    /// Queries-per-second over the most recent sample window.
    pub qps: u32,
    /// p50 inference latency in microseconds.
    pub p50_us: u32,
    /// p99 inference latency in microseconds.
    pub p99_us: u32,
    /// Cumulative error count.
    pub error_count: u64,
}

impl InferenceStat {
    /// Project to JSON.
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("name".to_string(), JsonValue::String(self.name.clone()));
        m.insert("qps".to_string(), JsonValue::from(self.qps));
        m.insert("p50_us".to_string(), JsonValue::from(self.p50_us));
        m.insert("p99_us".to_string(), JsonValue::from(self.p99_us));
        m.insert("error_count".to_string(), JsonValue::from(self.error_count));
        JsonValue::Object(m)
    }
}

/// Build the wrapper `{ ts, per_model: [...] }` payload for
/// `smallaios/metrics/inference`.
pub fn inference_stats_to_json(ts_ns: u64, per_model: &[InferenceStat]) -> JsonValue {
    let mut m = BTreeMap::new();
    m.insert("ts".to_string(), JsonValue::from(ts_ns));
    let arr: Vec<JsonValue> = per_model.iter().map(InferenceStat::to_json).collect();
    m.insert("per_model".to_string(), JsonValue::Array(arr));
    JsonValue::Object(m)
}

// ─── Logs ────────────────────────────────────────────────────────────────────

/// Severity of a `smallaios/metrics/log` record. Mirrors the
/// five-level `tracing` ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Verbose internal traces.
    Trace,
    /// Routine debug output.
    Debug,
    /// Informational messages.
    Info,
    /// Warning conditions.
    Warn,
    /// Error conditions.
    Error,
}

impl LogLevel {
    /// Static lowercase name used on the wire.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

/// A single structured log record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogRecord {
    /// Wall-clock timestamp (ns since epoch).
    pub ts_ns: u64,
    /// Severity.
    pub level: LogLevel,
    /// `tracing`-style target (module path).
    pub target: String,
    /// Human-readable message.
    pub msg: String,
    /// Structured key-value pairs. Values are coerced to JSON strings
    /// to keep the wire schema flat — callers SHALL pre-format
    /// numbers if they want them numeric.
    pub fields: Vec<(String, String)>,
}

impl LogRecord {
    /// Project to JSON.
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("ts".to_string(), JsonValue::from(self.ts_ns));
        m.insert(
            "level".to_string(),
            JsonValue::String(self.level.as_str().to_string()),
        );
        m.insert("target".to_string(), JsonValue::String(self.target.clone()));
        m.insert("msg".to_string(), JsonValue::String(self.msg.clone()));
        let mut fields = BTreeMap::new();
        for (k, v) in &self.fields {
            fields.insert(k.clone(), JsonValue::String(v.clone()));
        }
        m.insert("fields".to_string(), JsonValue::Object(fields));
        JsonValue::Object(m)
    }
}

// ─── Audit fingerprint ───────────────────────────────────────────────────────

/// `smallaios/metrics/audit_fingerprint` payload —
/// `{ ts, hex_fingerprint, record_count }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditFingerprint {
    /// Timestamp (ns since epoch).
    pub ts_ns: u64,
    /// Lowercase-hex SHA-3-256 chain head (64 chars). The all-zeroes
    /// fingerprint represents an empty chain.
    pub hex_fingerprint: String,
    /// Cumulative count of audit records committed to the chain.
    pub record_count: u64,
}

impl AuditFingerprint {
    /// Project to JSON.
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("ts".to_string(), JsonValue::from(self.ts_ns));
        m.insert(
            "hex_fingerprint".to_string(),
            JsonValue::String(self.hex_fingerprint.clone()),
        );
        m.insert(
            "record_count".to_string(),
            JsonValue::from(self.record_count),
        );
        JsonValue::Object(m)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::json::{decode, encode};
    use super::*;

    fn round_trip(v: &JsonValue) -> JsonValue {
        let s = encode(v);
        decode(s.as_bytes()).unwrap()
    }

    #[test]
    fn cpu_stat_round_trip_preserves_per_core() {
        let s = CpuStat {
            ts_ns: 12_345_678,
            per_core: alloc::vec![
                PerCpuUtil {
                    id: 0,
                    util_pct: 12,
                },
                PerCpuUtil {
                    id: 1,
                    util_pct: 88,
                },
            ],
        };
        let v = s.to_json();
        let back = round_trip(&v);
        let obj = back.as_object().unwrap();
        assert_eq!(obj.get("ts").unwrap().as_int().unwrap(), 12_345_678);
        let per_core = match obj.get("per_core").unwrap() {
            JsonValue::Array(a) => a,
            _ => panic!("per_core must be array"),
        };
        assert_eq!(per_core.len(), 2);
        let core1 = per_core[1].as_object().unwrap();
        assert_eq!(core1.get("id").unwrap().as_int().unwrap(), 1);
        assert_eq!(core1.get("util_pct").unwrap().as_int().unwrap(), 88);
    }

    #[test]
    fn cpu_stat_peak_util_pct_picks_max() {
        let s = CpuStat {
            ts_ns: 0,
            per_core: alloc::vec![
                PerCpuUtil { id: 0, util_pct: 5 },
                PerCpuUtil {
                    id: 1,
                    util_pct: 90,
                },
                PerCpuUtil {
                    id: 2,
                    util_pct: 50,
                },
            ],
        };
        assert_eq!(s.peak_util_pct(), 90);
    }

    #[test]
    fn cpu_stat_peak_util_pct_empty_returns_zero() {
        let s = CpuStat {
            ts_ns: 0,
            per_core: alloc::vec![],
        };
        assert_eq!(s.peak_util_pct(), 0);
    }

    #[test]
    fn cpu_stat_emits_four_cores_when_supplied() {
        let s = CpuStat {
            ts_ns: 1,
            per_core: (0..4)
                .map(|i| PerCpuUtil {
                    id: i,
                    util_pct: 10 * i,
                })
                .collect(),
        };
        let v = s.to_json();
        let arr = match v.get("per_core").unwrap() {
            JsonValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        assert_eq!(arr.len(), 4);
    }

    #[test]
    fn mem_stat_round_trip_preserves_all_fields() {
        let s = MemStat {
            ts_ns: 999,
            heap_used_bytes: 1024,
            heap_cap_bytes: 8192,
            page_alloc_count: 10,
            page_free_count: 5,
        };
        let v = s.to_json();
        let back = round_trip(&v);
        let obj = back.as_object().unwrap();
        assert_eq!(obj.get("ts").unwrap().as_int().unwrap(), 999);
        assert_eq!(obj.get("heap_used_bytes").unwrap().as_int().unwrap(), 1024);
        assert_eq!(obj.get("heap_cap_bytes").unwrap().as_int().unwrap(), 8192);
        assert_eq!(obj.get("page_alloc_count").unwrap().as_int().unwrap(), 10);
        assert_eq!(obj.get("page_free_count").unwrap().as_int().unwrap(), 5);
    }

    #[test]
    fn inference_stat_round_trip_preserves_name_and_metrics() {
        let s = InferenceStat {
            name: "resnet50".to_string(),
            qps: 200,
            p50_us: 1500,
            p99_us: 9000,
            error_count: 2,
        };
        let v = s.to_json();
        let back = round_trip(&v);
        let obj = back.as_object().unwrap();
        assert_eq!(obj.get("name").unwrap().as_str().unwrap(), "resnet50");
        assert_eq!(obj.get("qps").unwrap().as_int().unwrap(), 200);
        assert_eq!(obj.get("p50_us").unwrap().as_int().unwrap(), 1500);
        assert_eq!(obj.get("p99_us").unwrap().as_int().unwrap(), 9000);
        assert_eq!(obj.get("error_count").unwrap().as_int().unwrap(), 2);
    }

    #[test]
    fn inference_wrapper_emits_per_model_array() {
        let stats = alloc::vec![
            InferenceStat {
                name: "a".to_string(),
                qps: 1,
                p50_us: 1,
                p99_us: 1,
                error_count: 0,
            },
            InferenceStat {
                name: "b".to_string(),
                qps: 2,
                p50_us: 2,
                p99_us: 2,
                error_count: 0,
            },
        ];
        let v = inference_stats_to_json(123, &stats);
        let obj = v.as_object().unwrap();
        assert_eq!(obj.get("ts").unwrap().as_int().unwrap(), 123);
        let arr = match obj.get("per_model").unwrap() {
            JsonValue::Array(a) => a,
            _ => panic!("array expected"),
        };
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn log_level_renders_lowercase() {
        assert_eq!(LogLevel::Trace.as_str(), "trace");
        assert_eq!(LogLevel::Debug.as_str(), "debug");
        assert_eq!(LogLevel::Info.as_str(), "info");
        assert_eq!(LogLevel::Warn.as_str(), "warn");
        assert_eq!(LogLevel::Error.as_str(), "error");
    }

    #[test]
    fn log_record_round_trip_preserves_fields() {
        let r = LogRecord {
            ts_ns: 42,
            level: LogLevel::Warn,
            target: "kernel::sched".to_string(),
            msg: "queue contention".to_string(),
            fields: alloc::vec![
                ("queue".to_string(), "core1".to_string()),
                ("depth".to_string(), "17".to_string()),
            ],
        };
        let back = round_trip(&r.to_json());
        let obj = back.as_object().unwrap();
        assert_eq!(obj.get("level").unwrap().as_str().unwrap(), "warn");
        assert_eq!(
            obj.get("target").unwrap().as_str().unwrap(),
            "kernel::sched"
        );
        let f = obj.get("fields").unwrap().as_object().unwrap();
        assert_eq!(f.get("queue").unwrap().as_str().unwrap(), "core1");
        assert_eq!(f.get("depth").unwrap().as_str().unwrap(), "17");
    }

    #[test]
    fn audit_fingerprint_round_trip_preserves_hex() {
        let f = AuditFingerprint {
            ts_ns: 100,
            hex_fingerprint: "deadbeef".to_string(),
            record_count: 7,
        };
        let back = round_trip(&f.to_json());
        let obj = back.as_object().unwrap();
        assert_eq!(
            obj.get("hex_fingerprint").unwrap().as_str().unwrap(),
            "deadbeef"
        );
        assert_eq!(obj.get("record_count").unwrap().as_int().unwrap(), 7);
    }
}
