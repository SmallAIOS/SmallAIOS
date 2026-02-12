// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Zenoh IPC metrics export on key expression `smallaios/v1/metrics`.
//!
//! Serializes monitoring metrics into a compact binary format for
//! efficient transport over the Zenoh IPC bus. External monitoring
//! systems can subscribe to receive real-time security metrics.

use super::prometheus::MetricsSnapshot;

/// Zenoh key expression for security metrics.
pub const METRICS_KEY_EXPR: &str = "smallaios/v1/metrics";

/// Binary message format version.
const MESSAGE_VERSION: u8 = 1;

/// Size of the serialized metrics message header.
const HEADER_SIZE: usize = 9; // version(1) + timestamp(8)

/// Number of u64 metric fields serialized.
const NUM_METRIC_FIELDS: usize = 19;

/// Total message size: header + metrics.
pub const MESSAGE_SIZE: usize = HEADER_SIZE + NUM_METRIC_FIELDS * 8;

/// A serialized metrics message for Zenoh IPC transport.
#[derive(Debug)]
pub struct MetricsMessage {
    /// Raw serialized bytes.
    pub data: [u8; MESSAGE_SIZE],
    /// Timestamp when the snapshot was taken (nanoseconds since epoch).
    pub timestamp_ns: u64,
}

impl MetricsMessage {
    /// Serialize a metrics snapshot into a binary message.
    ///
    /// Format (all big-endian):
    ///
    /// | Offset    | Field                      |
    /// |-----------|----------------------------|
    /// | 0         | version (u8)               |
    /// | 1..9      | timestamp_ns (u64 BE)      |
    /// | 9..17     | cap_denial_total_alerts     |
    /// | 17..25    | mem_buddy_failures          |
    /// | 25..33    | mem_buddy_alerts            |
    /// | 33..41    | mem_slab_failures           |
    /// | 41..49    | mem_slab_alerts             |
    /// | 49..57    | mem_tensor_failures         |
    /// | 57..65    | mem_tensor_alerts           |
    /// | 65..73    | latency_p50_ns             |
    /// | 73..81    | latency_p99_ns             |
    /// | 81..89    | latency_p999_ns            |
    /// | 89..97    | latency_mean_ns            |
    /// | 97..105   | latency_stddev_ns          |
    /// | 105..113  | latency_total_observations |
    /// | 113..121  | latency_anomalies          |
    /// | 121..129  | watchdog_remaining_ns      |
    /// | 129..137  | watchdog_timeout_ns        |
    /// | 137..145  | watchdog_warnings          |
    /// | 145..153  | net_conn_total             |
    /// | 153..161  | net_conn_alerts            |
    pub fn serialize(snapshot: &MetricsSnapshot, timestamp_ns: u64) -> Self {
        let mut data = [0u8; MESSAGE_SIZE];
        let mut offset = 0;

        // Header
        data[offset] = MESSAGE_VERSION;
        offset += 1;
        data[offset..offset + 8].copy_from_slice(&timestamp_ns.to_be_bytes());
        offset += 8;

        // Metric fields in order
        let fields: [u64; NUM_METRIC_FIELDS] = [
            snapshot.cap_denial_total_alerts,
            snapshot.mem_buddy_failures,
            snapshot.mem_buddy_alerts,
            snapshot.mem_slab_failures,
            snapshot.mem_slab_alerts,
            snapshot.mem_tensor_failures,
            snapshot.mem_tensor_alerts,
            snapshot.latency_p50_ns,
            snapshot.latency_p99_ns,
            snapshot.latency_p999_ns,
            snapshot.latency_mean_ns,
            snapshot.latency_stddev_ns,
            snapshot.latency_total_observations,
            snapshot.latency_anomalies,
            snapshot.watchdog_remaining_ns,
            snapshot.watchdog_timeout_ns,
            snapshot.watchdog_warnings,
            snapshot.net_conn_total,
            snapshot.net_conn_alerts,
        ];

        for field in &fields {
            data[offset..offset + 8].copy_from_slice(&field.to_be_bytes());
            offset += 8;
        }

        Self { data, timestamp_ns }
    }

    /// Deserialize a binary message back into a snapshot.
    /// Returns None if the message is malformed.
    pub fn deserialize(data: &[u8]) -> Option<(MetricsSnapshot, u64)> {
        if data.len() < MESSAGE_SIZE {
            return None;
        }

        if data[0] != MESSAGE_VERSION {
            return None;
        }

        let timestamp_ns = u64::from_be_bytes(data[1..9].try_into().ok()?);
        let mut offset = HEADER_SIZE;

        let mut read_u64 = || -> Option<u64> {
            if offset + 8 > data.len() {
                return None;
            }
            let val = u64::from_be_bytes(data[offset..offset + 8].try_into().ok()?);
            offset += 8;
            Some(val)
        };

        let snapshot = MetricsSnapshot {
            cap_denial_total_alerts: read_u64()?,
            mem_buddy_failures: read_u64()?,
            mem_buddy_alerts: read_u64()?,
            mem_slab_failures: read_u64()?,
            mem_slab_alerts: read_u64()?,
            mem_tensor_failures: read_u64()?,
            mem_tensor_alerts: read_u64()?,
            latency_p50_ns: read_u64()?,
            latency_p99_ns: read_u64()?,
            latency_p999_ns: read_u64()?,
            latency_mean_ns: read_u64()?,
            latency_stddev_ns: read_u64()?,
            latency_total_observations: read_u64()?,
            latency_anomalies: read_u64()?,
            watchdog_remaining_ns: read_u64()?,
            watchdog_timeout_ns: read_u64()?,
            watchdog_warnings: read_u64()?,
            net_conn_total: read_u64()?,
            net_conn_alerts: read_u64()?,
        };

        Some((snapshot, timestamp_ns))
    }

    /// Get the raw message bytes for transmission.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Message format version.
    pub fn version(&self) -> u8 {
        self.data[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            cap_denial_total_alerts: 5,
            mem_buddy_failures: 10,
            mem_buddy_alerts: 1,
            mem_slab_failures: 20,
            mem_slab_alerts: 2,
            mem_tensor_failures: 30,
            mem_tensor_alerts: 3,
            latency_p50_ns: 1_000_000,
            latency_p99_ns: 5_000_000,
            latency_p999_ns: 10_000_000,
            latency_mean_ns: 1_500_000,
            latency_stddev_ns: 500_000,
            latency_total_observations: 50_000,
            latency_anomalies: 12,
            watchdog_remaining_ns: 750_000_000,
            watchdog_timeout_ns: 1_000_000_000,
            watchdog_warnings: 7,
            net_conn_total: 8888,
            net_conn_alerts: 4,
        }
    }

    #[test]
    fn serialize_roundtrip() {
        let snap = sample_snapshot();
        let msg = MetricsMessage::serialize(&snap, 42_000_000_000);

        let (decoded, ts) = MetricsMessage::deserialize(&msg.data).unwrap();
        assert_eq!(ts, 42_000_000_000);
        assert_eq!(decoded.cap_denial_total_alerts, 5);
        assert_eq!(decoded.mem_buddy_failures, 10);
        assert_eq!(decoded.mem_buddy_alerts, 1);
        assert_eq!(decoded.mem_slab_failures, 20);
        assert_eq!(decoded.mem_slab_alerts, 2);
        assert_eq!(decoded.mem_tensor_failures, 30);
        assert_eq!(decoded.mem_tensor_alerts, 3);
        assert_eq!(decoded.latency_p50_ns, 1_000_000);
        assert_eq!(decoded.latency_p99_ns, 5_000_000);
        assert_eq!(decoded.latency_p999_ns, 10_000_000);
        assert_eq!(decoded.latency_mean_ns, 1_500_000);
        assert_eq!(decoded.latency_stddev_ns, 500_000);
        assert_eq!(decoded.latency_total_observations, 50_000);
        assert_eq!(decoded.latency_anomalies, 12);
        assert_eq!(decoded.watchdog_remaining_ns, 750_000_000);
        assert_eq!(decoded.watchdog_timeout_ns, 1_000_000_000);
        assert_eq!(decoded.watchdog_warnings, 7);
        assert_eq!(decoded.net_conn_total, 8888);
        assert_eq!(decoded.net_conn_alerts, 4);
    }

    #[test]
    fn serialize_empty_snapshot() {
        let snap = MetricsSnapshot::empty();
        let msg = MetricsMessage::serialize(&snap, 0);
        let (decoded, ts) = MetricsMessage::deserialize(&msg.data).unwrap();
        assert_eq!(ts, 0);
        assert_eq!(decoded.cap_denial_total_alerts, 0);
        assert_eq!(decoded.latency_p50_ns, 0);
        assert_eq!(decoded.net_conn_total, 0);
    }

    #[test]
    fn message_version() {
        let snap = MetricsSnapshot::empty();
        let msg = MetricsMessage::serialize(&snap, 0);
        assert_eq!(msg.version(), MESSAGE_VERSION);
    }

    #[test]
    fn message_size_correct() {
        let snap = MetricsSnapshot::empty();
        let msg = MetricsMessage::serialize(&snap, 0);
        assert_eq!(msg.as_bytes().len(), MESSAGE_SIZE);
    }

    #[test]
    fn deserialize_too_short() {
        let short_data = [0u8; 10];
        assert!(MetricsMessage::deserialize(&short_data).is_none());
    }

    #[test]
    fn deserialize_wrong_version() {
        let mut data = [0u8; MESSAGE_SIZE];
        data[0] = 255; // invalid version
        assert!(MetricsMessage::deserialize(&data).is_none());
    }

    #[test]
    fn timestamp_preserved() {
        let snap = MetricsSnapshot::empty();
        let ts = 123_456_789_012_345;
        let msg = MetricsMessage::serialize(&snap, ts);
        assert_eq!(msg.timestamp_ns, ts);
        let (_, decoded_ts) = MetricsMessage::deserialize(&msg.data).unwrap();
        assert_eq!(decoded_ts, ts);
    }

    #[test]
    fn key_expression() {
        assert_eq!(METRICS_KEY_EXPR, "smallaios/v1/metrics");
    }

    #[test]
    fn large_values_roundtrip() {
        let mut snap = MetricsSnapshot::empty();
        snap.latency_total_observations = u64::MAX;
        snap.net_conn_total = u64::MAX;

        let msg = MetricsMessage::serialize(&snap, u64::MAX);
        let (decoded, ts) = MetricsMessage::deserialize(&msg.data).unwrap();
        assert_eq!(ts, u64::MAX);
        assert_eq!(decoded.latency_total_observations, u64::MAX);
        assert_eq!(decoded.net_conn_total, u64::MAX);
    }
}
