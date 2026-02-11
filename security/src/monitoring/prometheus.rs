// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Prometheus / OpenMetrics text exposition format for security metrics.
//!
//! Formats monitoring counters and gauges as text for `GET /metrics` endpoint.
//! Follows the OpenMetrics specification (text/openmetrics-text).
//!
//! Metrics exported:
//! - smallaios_capability_denials_total (counter, per-task)
//! - smallaios_memory_failures_total (counter, per-region)
//! - smallaios_inference_latency_ns (gauge: p50, p99, p999, mean, stddev)
//! - smallaios_inference_anomalies_total (counter)
//! - smallaios_watchdog_remaining_ns (gauge)
//! - smallaios_watchdog_warnings_total (counter)
//! - smallaios_network_connections_total (counter)
//! - smallaios_alerts_total (counter)

use alloc::string::String;
use core::fmt::Write;

/// Maximum output buffer size for metrics exposition.
const MAX_METRICS_BUFFER: usize = 4096;

/// Snapshot of all monitoring metrics for exposition.
#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    // Capability denial metrics
    pub cap_denial_total_alerts: u64,

    // Memory failure metrics (per region)
    pub mem_buddy_failures: u64,
    pub mem_buddy_alerts: u64,
    pub mem_slab_failures: u64,
    pub mem_slab_alerts: u64,
    pub mem_tensor_failures: u64,
    pub mem_tensor_alerts: u64,

    // Inference latency metrics
    pub latency_p50_ns: u64,
    pub latency_p99_ns: u64,
    pub latency_p999_ns: u64,
    pub latency_mean_ns: u64,
    pub latency_stddev_ns: u64,
    pub latency_total_observations: u64,
    pub latency_anomalies: u64,

    // Watchdog metrics
    pub watchdog_remaining_ns: u64,
    pub watchdog_timeout_ns: u64,
    pub watchdog_warnings: u64,

    // Network metrics
    pub net_conn_total: u64,
    pub net_conn_alerts: u64,
}

impl MetricsSnapshot {
    /// Create a zeroed snapshot.
    pub fn empty() -> Self {
        Self {
            cap_denial_total_alerts: 0,
            mem_buddy_failures: 0,
            mem_buddy_alerts: 0,
            mem_slab_failures: 0,
            mem_slab_alerts: 0,
            mem_tensor_failures: 0,
            mem_tensor_alerts: 0,
            latency_p50_ns: 0,
            latency_p99_ns: 0,
            latency_p999_ns: 0,
            latency_mean_ns: 0,
            latency_stddev_ns: 0,
            latency_total_observations: 0,
            latency_anomalies: 0,
            watchdog_remaining_ns: 0,
            watchdog_timeout_ns: 0,
            watchdog_warnings: 0,
            net_conn_total: 0,
            net_conn_alerts: 0,
        }
    }
}

/// Format a metrics snapshot as OpenMetrics text exposition.
///
/// Returns the formatted text, truncated to `MAX_METRICS_BUFFER` bytes.
/// Each metric includes TYPE and HELP annotations per OpenMetrics spec.
pub fn format_metrics(snapshot: &MetricsSnapshot) -> String {
    let mut buf = String::with_capacity(MAX_METRICS_BUFFER);

    // Capability denial alerts
    write_help(&mut buf, "smallaios_capability_denial_alerts_total", "Total capability denial threshold alerts");
    write_type(&mut buf, "smallaios_capability_denial_alerts_total", "counter");
    write_counter(&mut buf, "smallaios_capability_denial_alerts_total", None, snapshot.cap_denial_total_alerts);

    // Memory failure counters (per region)
    write_help(&mut buf, "smallaios_memory_failures_total", "Total memory allocation failures by region");
    write_type(&mut buf, "smallaios_memory_failures_total", "counter");
    write_counter(&mut buf, "smallaios_memory_failures_total", Some("region=\"buddy\""), snapshot.mem_buddy_failures);
    write_counter(&mut buf, "smallaios_memory_failures_total", Some("region=\"slab\""), snapshot.mem_slab_failures);
    write_counter(&mut buf, "smallaios_memory_failures_total", Some("region=\"tensor\""), snapshot.mem_tensor_failures);

    // Memory failure alerts
    write_help(&mut buf, "smallaios_memory_failure_alerts_total", "Total memory failure threshold alerts by region");
    write_type(&mut buf, "smallaios_memory_failure_alerts_total", "counter");
    write_counter(&mut buf, "smallaios_memory_failure_alerts_total", Some("region=\"buddy\""), snapshot.mem_buddy_alerts);
    write_counter(&mut buf, "smallaios_memory_failure_alerts_total", Some("region=\"slab\""), snapshot.mem_slab_alerts);
    write_counter(&mut buf, "smallaios_memory_failure_alerts_total", Some("region=\"tensor\""), snapshot.mem_tensor_alerts);

    // Inference latency gauges
    write_help(&mut buf, "smallaios_inference_latency_ns", "Inference latency percentiles in nanoseconds");
    write_type(&mut buf, "smallaios_inference_latency_ns", "gauge");
    write_gauge(&mut buf, "smallaios_inference_latency_ns", Some("quantile=\"0.5\""), snapshot.latency_p50_ns);
    write_gauge(&mut buf, "smallaios_inference_latency_ns", Some("quantile=\"0.99\""), snapshot.latency_p99_ns);
    write_gauge(&mut buf, "smallaios_inference_latency_ns", Some("quantile=\"0.999\""), snapshot.latency_p999_ns);

    write_help(&mut buf, "smallaios_inference_latency_mean_ns", "Mean inference latency in nanoseconds");
    write_type(&mut buf, "smallaios_inference_latency_mean_ns", "gauge");
    write_gauge(&mut buf, "smallaios_inference_latency_mean_ns", None, snapshot.latency_mean_ns);

    write_help(&mut buf, "smallaios_inference_latency_stddev_ns", "Standard deviation of inference latency");
    write_type(&mut buf, "smallaios_inference_latency_stddev_ns", "gauge");
    write_gauge(&mut buf, "smallaios_inference_latency_stddev_ns", None, snapshot.latency_stddev_ns);

    // Inference observation count and anomalies
    write_help(&mut buf, "smallaios_inference_observations_total", "Total inference latency observations");
    write_type(&mut buf, "smallaios_inference_observations_total", "counter");
    write_counter(&mut buf, "smallaios_inference_observations_total", None, snapshot.latency_total_observations);

    write_help(&mut buf, "smallaios_inference_anomalies_total", "Total inference latency anomalies detected");
    write_type(&mut buf, "smallaios_inference_anomalies_total", "counter");
    write_counter(&mut buf, "smallaios_inference_anomalies_total", None, snapshot.latency_anomalies);

    // Watchdog metrics
    write_help(&mut buf, "smallaios_watchdog_remaining_ns", "Watchdog time remaining in nanoseconds");
    write_type(&mut buf, "smallaios_watchdog_remaining_ns", "gauge");
    write_gauge(&mut buf, "smallaios_watchdog_remaining_ns", None, snapshot.watchdog_remaining_ns);

    write_help(&mut buf, "smallaios_watchdog_timeout_ns", "Watchdog timeout period in nanoseconds");
    write_type(&mut buf, "smallaios_watchdog_timeout_ns", "gauge");
    write_gauge(&mut buf, "smallaios_watchdog_timeout_ns", None, snapshot.watchdog_timeout_ns);

    write_help(&mut buf, "smallaios_watchdog_warnings_total", "Total watchdog time-pressure warnings");
    write_type(&mut buf, "smallaios_watchdog_warnings_total", "counter");
    write_counter(&mut buf, "smallaios_watchdog_warnings_total", None, snapshot.watchdog_warnings);

    // Network connection metrics
    write_help(&mut buf, "smallaios_network_connections_total", "Total network connections tracked");
    write_type(&mut buf, "smallaios_network_connections_total", "counter");
    write_counter(&mut buf, "smallaios_network_connections_total", None, snapshot.net_conn_total);

    write_help(&mut buf, "smallaios_network_connection_alerts_total", "Total SYN flood detection alerts");
    write_type(&mut buf, "smallaios_network_connection_alerts_total", "counter");
    write_counter(&mut buf, "smallaios_network_connection_alerts_total", None, snapshot.net_conn_alerts);

    // OpenMetrics EOF marker
    let _ = writeln!(buf, "# EOF");

    buf
}

fn write_help(buf: &mut String, name: &str, help: &str) {
    let _ = writeln!(buf, "# HELP {} {}", name, help);
}

fn write_type(buf: &mut String, name: &str, metric_type: &str) {
    let _ = writeln!(buf, "# TYPE {} {}", name, metric_type);
}

fn write_counter(buf: &mut String, name: &str, labels: Option<&str>, value: u64) {
    match labels {
        Some(l) => {
            let _ = writeln!(buf, "{}{{{}}} {}", name, l, value);
        }
        None => {
            let _ = writeln!(buf, "{} {}", name, value);
        }
    }
}

fn write_gauge(buf: &mut String, name: &str, labels: Option<&str>, value: u64) {
    // Gauges use the same format as counters in text exposition
    write_counter(buf, name, labels, value);
}

/// Parse a single metric value from OpenMetrics text.
/// Returns the numeric value for the first line matching the given metric name and labels.
pub fn parse_metric_value(text: &str, metric_name: &str, labels: Option<&str>) -> Option<u64> {
    let target = match labels {
        Some(l) => alloc::format!("{}{{{}}} ", metric_name, l),
        None => alloc::format!("{} ", metric_name),
    };

    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        if line.starts_with(&target) {
            let value_str = line[target.len()..].trim();
            return value_str.parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_formats() {
        let snap = MetricsSnapshot::empty();
        let text = format_metrics(&snap);
        assert!(!text.is_empty());
        assert!(text.contains("# EOF"));
    }

    #[test]
    fn metrics_contain_all_expected_names() {
        let snap = MetricsSnapshot::empty();
        let text = format_metrics(&snap);

        assert!(text.contains("smallaios_capability_denial_alerts_total"));
        assert!(text.contains("smallaios_memory_failures_total"));
        assert!(text.contains("smallaios_memory_failure_alerts_total"));
        assert!(text.contains("smallaios_inference_latency_ns"));
        assert!(text.contains("smallaios_inference_latency_mean_ns"));
        assert!(text.contains("smallaios_inference_latency_stddev_ns"));
        assert!(text.contains("smallaios_inference_observations_total"));
        assert!(text.contains("smallaios_inference_anomalies_total"));
        assert!(text.contains("smallaios_watchdog_remaining_ns"));
        assert!(text.contains("smallaios_watchdog_timeout_ns"));
        assert!(text.contains("smallaios_watchdog_warnings_total"));
        assert!(text.contains("smallaios_network_connections_total"));
        assert!(text.contains("smallaios_network_connection_alerts_total"));
    }

    #[test]
    fn metrics_contain_help_and_type() {
        let snap = MetricsSnapshot::empty();
        let text = format_metrics(&snap);

        // Every metric should have HELP and TYPE
        assert!(text.contains("# HELP smallaios_capability_denial_alerts_total"));
        assert!(text.contains("# TYPE smallaios_capability_denial_alerts_total counter"));
        assert!(text.contains("# TYPE smallaios_inference_latency_ns gauge"));
    }

    #[test]
    fn metrics_with_values() {
        let mut snap = MetricsSnapshot::empty();
        snap.cap_denial_total_alerts = 42;
        snap.mem_buddy_failures = 100;
        snap.latency_p50_ns = 5_000_000;
        snap.watchdog_remaining_ns = 500_000_000;
        snap.net_conn_total = 9999;

        let text = format_metrics(&snap);

        assert_eq!(
            parse_metric_value(&text, "smallaios_capability_denial_alerts_total", None),
            Some(42)
        );
        assert_eq!(
            parse_metric_value(&text, "smallaios_memory_failures_total", Some("region=\"buddy\"")),
            Some(100)
        );
        assert_eq!(
            parse_metric_value(&text, "smallaios_inference_latency_ns", Some("quantile=\"0.5\"")),
            Some(5_000_000)
        );
        assert_eq!(
            parse_metric_value(&text, "smallaios_watchdog_remaining_ns", None),
            Some(500_000_000)
        );
        assert_eq!(
            parse_metric_value(&text, "smallaios_network_connections_total", None),
            Some(9999)
        );
    }

    #[test]
    fn metrics_labels_per_region() {
        let mut snap = MetricsSnapshot::empty();
        snap.mem_buddy_failures = 10;
        snap.mem_slab_failures = 20;
        snap.mem_tensor_failures = 30;

        let text = format_metrics(&snap);

        assert_eq!(
            parse_metric_value(&text, "smallaios_memory_failures_total", Some("region=\"buddy\"")),
            Some(10)
        );
        assert_eq!(
            parse_metric_value(&text, "smallaios_memory_failures_total", Some("region=\"slab\"")),
            Some(20)
        );
        assert_eq!(
            parse_metric_value(&text, "smallaios_memory_failures_total", Some("region=\"tensor\"")),
            Some(30)
        );
    }

    #[test]
    fn metrics_latency_quantiles() {
        let mut snap = MetricsSnapshot::empty();
        snap.latency_p50_ns = 1000;
        snap.latency_p99_ns = 5000;
        snap.latency_p999_ns = 10000;

        let text = format_metrics(&snap);

        assert_eq!(
            parse_metric_value(&text, "smallaios_inference_latency_ns", Some("quantile=\"0.5\"")),
            Some(1000)
        );
        assert_eq!(
            parse_metric_value(&text, "smallaios_inference_latency_ns", Some("quantile=\"0.99\"")),
            Some(5000)
        );
        assert_eq!(
            parse_metric_value(&text, "smallaios_inference_latency_ns", Some("quantile=\"0.999\"")),
            Some(10000)
        );
    }

    #[test]
    fn parse_metric_not_found() {
        let text = "# HELP foo bar\n# TYPE foo counter\nfoo 42\n";
        assert_eq!(parse_metric_value(text, "missing", None), None);
    }

    #[test]
    fn parse_metric_skips_comments() {
        let text = "# foo 999\nfoo 42\n";
        assert_eq!(parse_metric_value(text, "foo", None), Some(42));
    }

    #[test]
    fn eof_marker_present() {
        let snap = MetricsSnapshot::empty();
        let text = format_metrics(&snap);
        assert!(text.trim_end().ends_with("# EOF"));
    }
}
