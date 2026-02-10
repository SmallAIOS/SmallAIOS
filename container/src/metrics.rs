// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Prometheus-compatible metrics exporter.
//!
//! Provides a simple in-memory metrics registry that can export metrics in the
//! Prometheus text exposition format. Supports counter, gauge, histogram, and
//! summary metric types with arbitrary string labels.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use crate::ContainerError;

/// Maximum number of metrics that can be registered in a single registry.
const MAX_METRICS: usize = 256;

/// The kind of metric being tracked.
#[derive(Clone, Debug, PartialEq)]
pub enum MetricType {
    /// A monotonically increasing counter (e.g., total requests).
    Counter,
    /// A value that can go up or down (e.g., current memory usage).
    Gauge,
    /// Observations bucketed into configurable ranges.
    Histogram,
    /// Client-side calculated quantiles over a sliding time window.
    Summary,
}

impl MetricType {
    /// Returns the Prometheus type string for this metric type.
    fn as_prometheus_str(&self) -> &str {
        match self {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Summary => "summary",
        }
    }
}

/// A single metric entry in the registry.
#[derive(Clone, Debug)]
pub struct MetricEntry {
    /// Metric name (e.g., "smallaios_inference_total").
    pub name: String,
    /// Human-readable help text describing what this metric measures.
    pub help: String,
    /// The kind of metric (counter, gauge, etc.).
    pub metric_type: MetricType,
    /// Current numeric value.
    pub value: f64,
    /// Key-value label pairs attached to this metric.
    pub labels: Vec<(String, String)>,
}

/// In-memory registry holding all metrics for a container instance.
///
/// Metrics are registered by name (duplicates are rejected) and can be
/// exported in Prometheus text exposition format for scraping.
#[derive(Debug)]
pub struct MetricsRegistry {
    /// All registered metric entries.
    metrics: Vec<MetricEntry>,
    /// Monotonic ID counter for internal bookkeeping.
    next_id: u64,
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsRegistry {
    /// Create a new, empty metrics registry.
    pub fn new() -> Self {
        MetricsRegistry {
            metrics: Vec::new(),
            next_id: 0,
        }
    }

    /// Register a new metric with the given name, help text, and type.
    ///
    /// Returns the index of the newly registered metric within the registry.
    ///
    /// # Errors
    ///
    /// Returns [`ContainerError::TableFull`] if the registry already contains
    /// [`MAX_METRICS`] entries.
    ///
    /// Returns [`ContainerError::InvalidConfig`] if a metric with the same
    /// name is already registered.
    pub fn register(
        &mut self,
        name: &str,
        help: &str,
        metric_type: MetricType,
    ) -> Result<usize, ContainerError> {
        if self.metrics.len() >= MAX_METRICS {
            return Err(ContainerError::TableFull);
        }
        if self.metrics.iter().any(|m| m.name == name) {
            return Err(ContainerError::InvalidConfig(String::from(
                "duplicate metric name",
            )));
        }
        let index = self.metrics.len();
        self.metrics.push(MetricEntry {
            name: String::from(name),
            help: String::from(help),
            metric_type,
            value: 0.0,
            labels: Vec::new(),
        });
        self.next_id += 1;
        Ok(index)
    }

    /// Increment a counter metric by the given value.
    ///
    /// # Errors
    ///
    /// Returns [`ContainerError::NotFound`] if no metric with the given name
    /// exists in the registry.
    pub fn increment(&mut self, name: &str, value: f64) -> Result<(), ContainerError> {
        let entry = self
            .metrics
            .iter_mut()
            .find(|m| m.name == name)
            .ok_or_else(|| ContainerError::NotFound(String::from(name)))?;
        entry.value += value;
        Ok(())
    }

    /// Set a gauge metric to the given value.
    ///
    /// # Errors
    ///
    /// Returns [`ContainerError::NotFound`] if no metric with the given name
    /// exists in the registry.
    pub fn set(&mut self, name: &str, value: f64) -> Result<(), ContainerError> {
        let entry = self
            .metrics
            .iter_mut()
            .find(|m| m.name == name)
            .ok_or_else(|| ContainerError::NotFound(String::from(name)))?;
        entry.value = value;
        Ok(())
    }

    /// Get the current value of a metric by name.
    ///
    /// Returns `None` if no metric with the given name exists.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.value)
    }

    /// Returns the number of metrics currently registered.
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }

    /// Register the standard set of SmallAIOS default metrics.
    ///
    /// This includes counters for inference totals and errors, and gauges for
    /// duration, loaded models, memory usage, and uptime.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`register`](Self::register).
    pub fn register_defaults(&mut self) -> Result<(), ContainerError> {
        self.register(
            "smallaios_inference_total",
            "Total inference requests",
            MetricType::Counter,
        )?;
        self.register(
            "smallaios_inference_errors_total",
            "Total inference errors",
            MetricType::Counter,
        )?;
        self.register(
            "smallaios_inference_duration_seconds",
            "Last inference duration",
            MetricType::Gauge,
        )?;
        self.register(
            "smallaios_models_loaded",
            "Number of loaded models",
            MetricType::Gauge,
        )?;
        self.register(
            "smallaios_memory_used_bytes",
            "Memory usage in bytes",
            MetricType::Gauge,
        )?;
        self.register(
            "smallaios_uptime_seconds",
            "Uptime in seconds",
            MetricType::Gauge,
        )?;
        Ok(())
    }

    /// Export all registered metrics in Prometheus text exposition format.
    ///
    /// Each metric produces three lines:
    /// ```text
    /// # HELP <name> <help>
    /// # TYPE <name> <type>
    /// <name> <value>
    /// ```
    ///
    /// Floating-point values that are whole numbers are formatted without a
    /// decimal point (e.g., `42` instead of `42.0`).
    pub fn to_prometheus_format(&self) -> String {
        let mut output = String::new();
        for entry in &self.metrics {
            let _ = writeln!(output, "# HELP {} {}", entry.name, entry.help);
            let _ = writeln!(
                output,
                "# TYPE {} {}",
                entry.name,
                entry.metric_type.as_prometheus_str()
            );
            // Format value: use integer representation if it is a whole number.
            if entry.value == (entry.value as i64) as f64
                && !entry.value.is_nan()
                && !entry.value.is_infinite()
            {
                let _ = writeln!(output, "{} {}", entry.name, entry.value as i64);
            } else {
                let _ = writeln!(output, "{} {}", entry.name, entry.value);
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registry_is_empty() {
        let reg = MetricsRegistry::new();
        assert_eq!(reg.metric_count(), 0);
    }

    #[test]
    fn register_metric() {
        let mut reg = MetricsRegistry::new();
        let idx = reg
            .register("test_counter", "A test counter", MetricType::Counter)
            .unwrap();
        assert_eq!(idx, 0);
        assert_eq!(reg.metric_count(), 1);
    }

    #[test]
    fn register_duplicate_name_returns_error() {
        let mut reg = MetricsRegistry::new();
        reg.register("dup", "first", MetricType::Counter).unwrap();
        let result = reg.register("dup", "second", MetricType::Gauge);
        assert!(result.is_err());
        match result {
            Err(ContainerError::InvalidConfig(_)) => {}
            other => panic!("expected InvalidConfig, got {:?}", other),
        }
    }

    #[test]
    fn register_defaults_creates_six_metrics() {
        let mut reg = MetricsRegistry::new();
        reg.register_defaults().unwrap();
        assert_eq!(reg.metric_count(), 6);
    }

    #[test]
    fn increment_counter() {
        let mut reg = MetricsRegistry::new();
        reg.register("req_total", "total requests", MetricType::Counter)
            .unwrap();
        reg.increment("req_total", 1.0).unwrap();
        reg.increment("req_total", 4.0).unwrap();
        assert_eq!(reg.get("req_total"), Some(5.0));
    }

    #[test]
    fn set_gauge() {
        let mut reg = MetricsRegistry::new();
        reg.register("mem_bytes", "memory usage", MetricType::Gauge)
            .unwrap();
        reg.set("mem_bytes", 1024.0).unwrap();
        assert_eq!(reg.get("mem_bytes"), Some(1024.0));
        reg.set("mem_bytes", 2048.0).unwrap();
        assert_eq!(reg.get("mem_bytes"), Some(2048.0));
    }

    #[test]
    fn get_value() {
        let mut reg = MetricsRegistry::new();
        reg.register("val", "a value", MetricType::Gauge).unwrap();
        assert_eq!(reg.get("val"), Some(0.0));
        reg.set("val", 42.5).unwrap();
        assert_eq!(reg.get("val"), Some(42.5));
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let reg = MetricsRegistry::new();
        assert_eq!(reg.get("does_not_exist"), None);
    }

    #[test]
    fn metric_count() {
        let mut reg = MetricsRegistry::new();
        assert_eq!(reg.metric_count(), 0);
        reg.register("a", "a", MetricType::Counter).unwrap();
        assert_eq!(reg.metric_count(), 1);
        reg.register("b", "b", MetricType::Gauge).unwrap();
        assert_eq!(reg.metric_count(), 2);
        reg.register("c", "c", MetricType::Histogram).unwrap();
        assert_eq!(reg.metric_count(), 3);
    }

    #[test]
    fn prometheus_format_counter() {
        let mut reg = MetricsRegistry::new();
        reg.register(
            "http_requests_total",
            "Total HTTP requests",
            MetricType::Counter,
        )
        .unwrap();
        reg.increment("http_requests_total", 42.0).unwrap();
        let output = reg.to_prometheus_format();
        assert!(output.contains("# HELP http_requests_total Total HTTP requests"));
        assert!(output.contains("# TYPE http_requests_total counter"));
        assert!(output.contains("http_requests_total 42"));
    }

    #[test]
    fn prometheus_format_gauge() {
        let mut reg = MetricsRegistry::new();
        reg.register("temperature", "Current temperature", MetricType::Gauge)
            .unwrap();
        reg.set("temperature", 36.6).unwrap();
        let output = reg.to_prometheus_format();
        assert!(output.contains("# HELP temperature Current temperature"));
        assert!(output.contains("# TYPE temperature gauge"));
        assert!(output.contains("temperature 36.6"));
    }

    #[test]
    fn prometheus_format_multiple_metrics() {
        let mut reg = MetricsRegistry::new();
        reg.register("counter_a", "Counter A", MetricType::Counter)
            .unwrap();
        reg.register("gauge_b", "Gauge B", MetricType::Gauge)
            .unwrap();
        reg.increment("counter_a", 10.0).unwrap();
        reg.set("gauge_b", 99.0).unwrap();
        let output = reg.to_prometheus_format();
        assert!(output.contains("# HELP counter_a Counter A"));
        assert!(output.contains("counter_a 10"));
        assert!(output.contains("# HELP gauge_b Gauge B"));
        assert!(output.contains("gauge_b 99"));
    }

    #[test]
    fn table_full_error() {
        let mut reg = MetricsRegistry::new();
        for i in 0..MAX_METRICS {
            let name = alloc::format!("metric_{}", i);
            reg.register(&name, "help", MetricType::Counter).unwrap();
        }
        let result = reg.register("overflow", "too many", MetricType::Counter);
        assert_eq!(result, Err(ContainerError::TableFull));
    }

    #[test]
    fn increment_nonexistent_returns_error() {
        let mut reg = MetricsRegistry::new();
        let result = reg.increment("missing", 1.0);
        match result {
            Err(ContainerError::NotFound(name)) => assert_eq!(name, "missing"),
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn set_nonexistent_returns_error() {
        let mut reg = MetricsRegistry::new();
        let result = reg.set("missing", 1.0);
        match result {
            Err(ContainerError::NotFound(name)) => assert_eq!(name, "missing"),
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn default_metric_names_present() {
        let mut reg = MetricsRegistry::new();
        reg.register_defaults().unwrap();

        let expected_names = [
            "smallaios_inference_total",
            "smallaios_inference_errors_total",
            "smallaios_inference_duration_seconds",
            "smallaios_models_loaded",
            "smallaios_memory_used_bytes",
            "smallaios_uptime_seconds",
        ];

        for name in &expected_names {
            assert!(
                reg.get(name).is_some(),
                "default metric '{}' not found",
                name,
            );
        }
    }

    #[test]
    fn metric_type_clone_and_eq() {
        let a = MetricType::Counter;
        let b = a.clone();
        assert_eq!(a, b);
        assert_ne!(MetricType::Counter, MetricType::Gauge);
        assert_ne!(MetricType::Histogram, MetricType::Summary);
    }

    #[test]
    fn prometheus_format_whole_number_no_decimal() {
        let mut reg = MetricsRegistry::new();
        reg.register("whole", "whole number", MetricType::Counter)
            .unwrap();
        reg.increment("whole", 100.0).unwrap();
        let output = reg.to_prometheus_format();
        // Should contain "whole 100" not "whole 100.0"
        assert!(output.contains("whole 100\n"));
    }
}
