// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS Benchmark Infrastructure
//!
//! Measurement harnesses for evaluating SmallAIOS performance against
//! Linux bare metal, Docker, and Kubernetes baselines. Covers:
//!
//! - **Cold start**: Power-on to first inference result
//! - **Warm inference**: Latency distribution (p50/p99/p999) over N runs
//! - **Throughput**: Maximum inferences per second at batch sizes 1, 4, 16, 64
//! - **Jitter**: Standard deviation and max deviation from mean latency
//! - **Memory footprint**: Peak resident set size during inference
//!
//! All timing uses monotonic microsecond timestamps. Results are collected
//! into [`BenchmarkReport`] structs that can be serialized to Markdown
//! tables or CSV.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use core::fmt;

/// Benchmark configuration for a single measurement run.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Human-readable name of the benchmark (e.g., "mobilenetv2-warm-b1").
    pub name: String,
    /// Model identifier (path or URL).
    pub model: String,
    /// Batch size for inference.
    pub batch_size: u32,
    /// Number of warmup iterations (discarded from stats).
    pub warmup_iters: u32,
    /// Number of measured iterations.
    pub measure_iters: u32,
    /// Target platform description.
    pub platform: String,
}

impl BenchConfig {
    /// Create a new benchmark configuration.
    pub fn new(name: &str, model: &str, batch_size: u32, measure_iters: u32) -> Self {
        Self {
            name: String::from(name),
            model: String::from(model),
            batch_size,
            warmup_iters: 10,
            measure_iters,
            platform: String::from("unknown"),
        }
    }

    /// Set the warmup iteration count.
    pub fn with_warmup(mut self, warmup: u32) -> Self {
        self.warmup_iters = warmup;
        self
    }

    /// Set the platform description.
    pub fn with_platform(mut self, platform: &str) -> Self {
        self.platform = String::from(platform);
        self
    }
}

/// Latency statistics computed from a series of measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct LatencyStats {
    /// Number of samples.
    pub count: usize,
    /// Minimum latency in microseconds.
    pub min_us: u64,
    /// Maximum latency in microseconds.
    pub max_us: u64,
    /// Mean latency in microseconds.
    pub mean_us: u64,
    /// Median (p50) latency in microseconds.
    pub p50_us: u64,
    /// 99th percentile latency in microseconds.
    pub p99_us: u64,
    /// 99.9th percentile latency in microseconds.
    pub p999_us: u64,
    /// Standard deviation in microseconds.
    pub stddev_us: u64,
    /// Sum of all latencies in microseconds.
    pub total_us: u64,
}

impl LatencyStats {
    /// Compute latency statistics from a slice of measurements (microseconds).
    ///
    /// Returns `None` if the input is empty.
    pub fn from_samples(samples: &[u64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut sorted = Vec::from(samples);
        sorted.sort_unstable();

        let count = sorted.len();
        let total_us: u64 = sorted.iter().sum();
        let mean_us = total_us / count as u64;

        let p50_us = percentile(&sorted, 50);
        let p99_us = percentile(&sorted, 99);
        let p999_us = percentile(&sorted, 999);

        // Compute standard deviation
        let variance: u64 = if count > 1 {
            let sum_sq_diff: u128 = sorted.iter().map(|&s| {
                let diff = s.abs_diff(mean_us);
                (diff as u128) * (diff as u128)
            }).sum();
            (sum_sq_diff / (count as u128 - 1)) as u64
        } else {
            0
        };
        let stddev_us = isqrt(variance);

        Some(LatencyStats {
            count,
            min_us: sorted[0],
            max_us: sorted[count - 1],
            mean_us,
            p50_us,
            p99_us,
            p999_us,
            stddev_us,
            total_us,
        })
    }

    /// Compute jitter: maximum deviation from the mean.
    pub fn jitter_max_us(&self) -> u64 {
        let dev_min = self.mean_us.abs_diff(self.min_us);
        let dev_max = self.max_us.abs_diff(self.mean_us);
        dev_min.max(dev_max)
    }
}

/// Compute percentile from a sorted slice.
///
/// `pct` is in tenths of a percent (e.g., 999 = 99.9th percentile).
fn percentile(sorted: &[u64], pct: u32) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    // pct is 0-1000 for 0.0%-100.0%
    // For standard percentiles: 50 = 50th, 99 = 99th, 999 = 99.9th
    let pct_f = if pct > 100 {
        // It's in tenths: 999 = 99.9%
        pct as u64
    } else {
        // It's a standard percentile: 50 = 50%, 99 = 99%
        pct as u64 * 10
    };
    let idx = (sorted.len() as u64 * pct_f / 1000) as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

/// Integer square root (Newton's method).
fn isqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Cold start measurement result.
#[derive(Debug, Clone)]
pub struct ColdStartResult {
    /// Configuration used.
    pub config: BenchConfig,
    /// Time from power-on/process-start to first inference result (microseconds).
    pub boot_to_inference_us: u64,
    /// Breakdown: boot time (microseconds).
    pub boot_us: u64,
    /// Breakdown: model load time (microseconds).
    pub model_load_us: u64,
    /// Breakdown: first inference time (microseconds).
    pub first_inference_us: u64,
}

/// Warm inference measurement result.
#[derive(Debug, Clone)]
pub struct WarmInferenceResult {
    /// Configuration used.
    pub config: BenchConfig,
    /// Latency statistics over all measured iterations.
    pub latency: LatencyStats,
}

/// Throughput measurement result.
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    /// Configuration used.
    pub config: BenchConfig,
    /// Inferences per second.
    pub inferences_per_sec: f64,
    /// Total time for all iterations (microseconds).
    pub total_us: u64,
}

/// Memory footprint measurement result.
#[derive(Debug, Clone)]
pub struct MemoryResult {
    /// Configuration used.
    pub config: BenchConfig,
    /// Peak resident set size in bytes.
    pub peak_rss_bytes: u64,
    /// Kernel image size in bytes.
    pub kernel_size_bytes: u64,
    /// Model size in bytes.
    pub model_size_bytes: u64,
}

/// A complete benchmark report containing all measurement types.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Report title.
    pub title: String,
    /// Timestamp (ISO 8601 string).
    pub timestamp: String,
    /// Cold start results.
    pub cold_start: Vec<ColdStartResult>,
    /// Warm inference results.
    pub warm_inference: Vec<WarmInferenceResult>,
    /// Throughput results.
    pub throughput: Vec<ThroughputResult>,
    /// Memory results.
    pub memory: Vec<MemoryResult>,
}

impl BenchmarkReport {
    /// Create an empty report.
    pub fn new(title: &str) -> Self {
        Self {
            title: String::from(title),
            timestamp: String::from(""),
            cold_start: Vec::new(),
            warm_inference: Vec::new(),
            throughput: Vec::new(),
            memory: Vec::new(),
        }
    }

    /// Set the timestamp.
    pub fn with_timestamp(mut self, ts: &str) -> Self {
        self.timestamp = String::from(ts);
        self
    }

    /// Generate a Markdown-formatted report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!("# {}\n\n", self.title));
        if !self.timestamp.is_empty() {
            md.push_str(&format!("Generated: {}\n\n", self.timestamp));
        }

        // Cold start table
        if !self.cold_start.is_empty() {
            md.push_str("## Cold Start (Power-on to First Inference)\n\n");
            md.push_str("| Platform | Model | Boot (us) | Model Load (us) | First Inference (us) | Total (us) |\n");
            md.push_str("|----------|-------|-----------|-----------------|---------------------|------------|\n");
            for r in &self.cold_start {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} |\n",
                    r.config.platform,
                    r.config.model,
                    r.boot_us,
                    r.model_load_us,
                    r.first_inference_us,
                    r.boot_to_inference_us,
                ));
            }
            md.push('\n');
        }

        // Warm inference table
        if !self.warm_inference.is_empty() {
            md.push_str("## Warm Inference Latency\n\n");
            md.push_str("| Platform | Model | Batch | N | p50 (us) | p99 (us) | p999 (us) | Mean (us) | Stddev (us) | Jitter Max (us) |\n");
            md.push_str("|----------|-------|-------|---|----------|----------|-----------|-----------|-------------|------------------|\n");
            for r in &self.warm_inference {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    r.config.platform,
                    r.config.model,
                    r.config.batch_size,
                    r.latency.count,
                    r.latency.p50_us,
                    r.latency.p99_us,
                    r.latency.p999_us,
                    r.latency.mean_us,
                    r.latency.stddev_us,
                    r.latency.jitter_max_us(),
                ));
            }
            md.push('\n');
        }

        // Throughput table
        if !self.throughput.is_empty() {
            md.push_str("## Throughput\n\n");
            md.push_str("| Platform | Model | Batch | Inf/sec | Total (us) |\n");
            md.push_str("|----------|-------|-------|---------|------------|\n");
            for r in &self.throughput {
                md.push_str(&format!(
                    "| {} | {} | {} | {:.1} | {} |\n",
                    r.config.platform,
                    r.config.model,
                    r.config.batch_size,
                    r.inferences_per_sec,
                    r.total_us,
                ));
            }
            md.push('\n');
        }

        // Memory table
        if !self.memory.is_empty() {
            md.push_str("## Memory Footprint\n\n");
            md.push_str("| Platform | Model | Peak RSS (KB) | Kernel (KB) | Model (KB) |\n");
            md.push_str("|----------|-------|---------------|-------------|------------|\n");
            for r in &self.memory {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    r.config.platform,
                    r.config.model,
                    r.peak_rss_bytes / 1024,
                    r.kernel_size_bytes / 1024,
                    r.model_size_bytes / 1024,
                ));
            }
            md.push('\n');
        }

        md
    }

    /// Generate a CSV-formatted report (one section at a time).
    pub fn warm_inference_to_csv(&self) -> String {
        let mut csv = String::from("platform,model,batch_size,n,min_us,p50_us,p99_us,p999_us,max_us,mean_us,stddev_us,jitter_max_us\n");
        for r in &self.warm_inference {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{}\n",
                r.config.platform,
                r.config.model,
                r.config.batch_size,
                r.latency.count,
                r.latency.min_us,
                r.latency.p50_us,
                r.latency.p99_us,
                r.latency.p999_us,
                r.latency.max_us,
                r.latency.mean_us,
                r.latency.stddev_us,
                r.latency.jitter_max_us(),
            ));
        }
        csv
    }

    /// Generate CSV for throughput results.
    pub fn throughput_to_csv(&self) -> String {
        let mut csv = String::from("platform,model,batch_size,inferences_per_sec,total_us\n");
        for r in &self.throughput {
            csv.push_str(&format!(
                "{},{},{},{:.1},{}\n",
                r.config.platform,
                r.config.model,
                r.config.batch_size,
                r.inferences_per_sec,
                r.total_us,
            ));
        }
        csv
    }
}

impl fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n={} min={} p50={} p99={} p999={} max={} mean={} stddev={} (us)",
            self.count,
            self.min_us,
            self.p50_us,
            self.p99_us,
            self.p999_us,
            self.max_us,
            self.mean_us,
            self.stddev_us,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // LatencyStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_single_sample() {
        let stats = LatencyStats::from_samples(&[1000]).unwrap();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.min_us, 1000);
        assert_eq!(stats.max_us, 1000);
        assert_eq!(stats.mean_us, 1000);
        assert_eq!(stats.p50_us, 1000);
        assert_eq!(stats.stddev_us, 0);
    }

    #[test]
    fn test_stats_empty_samples() {
        assert!(LatencyStats::from_samples(&[]).is_none());
    }

    #[test]
    fn test_stats_uniform_samples() {
        let samples = [100u64; 100];
        let stats = LatencyStats::from_samples(&samples).unwrap();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.min_us, 100);
        assert_eq!(stats.max_us, 100);
        assert_eq!(stats.mean_us, 100);
        assert_eq!(stats.stddev_us, 0);
    }

    #[test]
    fn test_stats_ascending_samples() {
        let samples: Vec<u64> = (1..=1000).collect();
        let stats = LatencyStats::from_samples(&samples).unwrap();
        assert_eq!(stats.count, 1000);
        assert_eq!(stats.min_us, 1);
        assert_eq!(stats.max_us, 1000);
        // Sum = 1000*1001/2 = 500500, mean = 500500/1000 = 500
        assert!(stats.mean_us >= 500 && stats.mean_us <= 501);
        // p50 at index 500 = 501
        assert!(stats.p50_us >= 500 && stats.p50_us <= 501);
    }

    #[test]
    fn test_stats_p99() {
        let samples: Vec<u64> = (1..=1000).collect();
        let stats = LatencyStats::from_samples(&samples).unwrap();
        // p99 = 99th percentile ~ 990
        assert!(stats.p99_us >= 980 && stats.p99_us <= 1000);
    }

    #[test]
    fn test_stats_p999() {
        let samples: Vec<u64> = (1..=1000).collect();
        let stats = LatencyStats::from_samples(&samples).unwrap();
        // p999 = 99.9th percentile ~ 999
        assert!(stats.p999_us >= 990);
    }

    #[test]
    fn test_stats_stddev() {
        // Known variance: samples 1,2,3,4,5 -> mean=3, var=2.5, stddev~1.58
        let samples = [1u64, 2, 3, 4, 5];
        let stats = LatencyStats::from_samples(&samples).unwrap();
        assert_eq!(stats.mean_us, 3);
        // Variance = 10/4 = 2, isqrt(2) = 1
        assert!(stats.stddev_us >= 1 && stats.stddev_us <= 2);
    }

    #[test]
    fn test_stats_total() {
        let samples = [100u64, 200, 300];
        let stats = LatencyStats::from_samples(&samples).unwrap();
        assert_eq!(stats.total_us, 600);
    }

    #[test]
    fn test_jitter_max() {
        let samples = [100u64, 200, 300, 400, 500];
        let stats = LatencyStats::from_samples(&samples).unwrap();
        // mean = 300, max deviation = max(300-100, 500-300) = 200
        assert_eq!(stats.jitter_max_us(), 200);
    }

    #[test]
    fn test_jitter_zero() {
        let samples = [500u64; 50];
        let stats = LatencyStats::from_samples(&samples).unwrap();
        assert_eq!(stats.jitter_max_us(), 0);
    }

    // -----------------------------------------------------------------------
    // isqrt tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_isqrt_zero() {
        assert_eq!(isqrt(0), 0);
    }

    #[test]
    fn test_isqrt_one() {
        assert_eq!(isqrt(1), 1);
    }

    #[test]
    fn test_isqrt_perfect_squares() {
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(10000), 100);
    }

    #[test]
    fn test_isqrt_non_perfect() {
        assert_eq!(isqrt(2), 1);
        assert_eq!(isqrt(8), 2);
        assert_eq!(isqrt(99), 9);
    }

    // -----------------------------------------------------------------------
    // percentile tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50), 0);
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42], 50), 42);
        assert_eq!(percentile(&[42], 99), 42);
    }

    #[test]
    fn test_percentile_ten_elements() {
        let sorted: Vec<u64> = (1..=10).collect();
        // 10 elements, p50 = index (10*500/1000) = 5 -> sorted[5] = 6
        assert_eq!(percentile(&sorted, 50), 6);
    }

    // -----------------------------------------------------------------------
    // BenchConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bench_config_new() {
        let cfg = BenchConfig::new("test-bench", "model.onnx", 1, 1000);
        assert_eq!(cfg.name, "test-bench");
        assert_eq!(cfg.model, "model.onnx");
        assert_eq!(cfg.batch_size, 1);
        assert_eq!(cfg.measure_iters, 1000);
        assert_eq!(cfg.warmup_iters, 10); // default
    }

    #[test]
    fn test_bench_config_with_warmup() {
        let cfg = BenchConfig::new("test", "m.onnx", 1, 100)
            .with_warmup(50);
        assert_eq!(cfg.warmup_iters, 50);
    }

    #[test]
    fn test_bench_config_with_platform() {
        let cfg = BenchConfig::new("test", "m.onnx", 1, 100)
            .with_platform("DGX Spark");
        assert_eq!(cfg.platform, "DGX Spark");
    }

    // -----------------------------------------------------------------------
    // BenchmarkReport tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_report_new() {
        let report = BenchmarkReport::new("Test Report");
        assert_eq!(report.title, "Test Report");
        assert!(report.cold_start.is_empty());
        assert!(report.warm_inference.is_empty());
        assert!(report.throughput.is_empty());
        assert!(report.memory.is_empty());
    }

    #[test]
    fn test_report_with_timestamp() {
        let report = BenchmarkReport::new("Test")
            .with_timestamp("2026-02-10T12:00:00Z");
        assert_eq!(report.timestamp, "2026-02-10T12:00:00Z");
    }

    #[test]
    fn test_report_to_markdown_empty() {
        let report = BenchmarkReport::new("Empty Report");
        let md = report.to_markdown();
        assert!(md.contains("# Empty Report"));
    }

    #[test]
    fn test_report_to_markdown_with_cold_start() {
        let mut report = BenchmarkReport::new("Test");
        report.cold_start.push(ColdStartResult {
            config: BenchConfig::new("test", "model.onnx", 1, 1)
                .with_platform("SmallAIOS"),
            boot_to_inference_us: 50000,
            boot_us: 20000,
            model_load_us: 25000,
            first_inference_us: 5000,
        });
        let md = report.to_markdown();
        assert!(md.contains("Cold Start"));
        assert!(md.contains("SmallAIOS"));
        assert!(md.contains("50000"));
    }

    #[test]
    fn test_report_to_markdown_with_warm_inference() {
        let mut report = BenchmarkReport::new("Test");
        let latency = LatencyStats::from_samples(&[100, 110, 120, 130, 140]).unwrap();
        report.warm_inference.push(WarmInferenceResult {
            config: BenchConfig::new("test", "model.onnx", 1, 5)
                .with_platform("SmallAIOS"),
            latency,
        });
        let md = report.to_markdown();
        assert!(md.contains("Warm Inference"));
        assert!(md.contains("p50"));
        assert!(md.contains("p99"));
    }

    #[test]
    fn test_report_to_markdown_with_throughput() {
        let mut report = BenchmarkReport::new("Test");
        report.throughput.push(ThroughputResult {
            config: BenchConfig::new("test", "model.onnx", 4, 100)
                .with_platform("SmallAIOS"),
            inferences_per_sec: 5000.0,
            total_us: 80000,
        });
        let md = report.to_markdown();
        assert!(md.contains("Throughput"));
        assert!(md.contains("5000.0"));
    }

    #[test]
    fn test_report_to_markdown_with_memory() {
        let mut report = BenchmarkReport::new("Test");
        report.memory.push(MemoryResult {
            config: BenchConfig::new("test", "model.onnx", 1, 1)
                .with_platform("SmallAIOS"),
            peak_rss_bytes: 8 * 1024 * 1024,
            kernel_size_bytes: 4 * 1024 * 1024,
            model_size_bytes: 2 * 1024 * 1024,
        });
        let md = report.to_markdown();
        assert!(md.contains("Memory Footprint"));
        assert!(md.contains("Peak RSS"));
    }

    // -----------------------------------------------------------------------
    // CSV export tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_warm_inference_csv() {
        let mut report = BenchmarkReport::new("Test");
        let latency = LatencyStats::from_samples(&[100, 200, 300]).unwrap();
        report.warm_inference.push(WarmInferenceResult {
            config: BenchConfig::new("test", "model.onnx", 1, 3)
                .with_platform("SmallAIOS"),
            latency,
        });
        let csv = report.warm_inference_to_csv();
        assert!(csv.starts_with("platform,model,batch_size"));
        assert!(csv.contains("SmallAIOS,model.onnx,1,3"));
    }

    #[test]
    fn test_throughput_csv() {
        let mut report = BenchmarkReport::new("Test");
        report.throughput.push(ThroughputResult {
            config: BenchConfig::new("test", "model.onnx", 4, 100)
                .with_platform("SmallAIOS"),
            inferences_per_sec: 1234.5,
            total_us: 500000,
        });
        let csv = report.throughput_to_csv();
        assert!(csv.starts_with("platform,model,batch_size"));
        assert!(csv.contains("SmallAIOS,model.onnx,4,1234.5,500000"));
    }

    #[test]
    fn test_empty_csv() {
        let report = BenchmarkReport::new("Test");
        let csv = report.warm_inference_to_csv();
        // Should have just the header
        assert_eq!(csv.lines().count(), 1);
    }

    // -----------------------------------------------------------------------
    // LatencyStats Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_latency_stats_display() {
        let stats = LatencyStats::from_samples(&[100, 200, 300]).unwrap();
        let s = format!("{}", stats);
        assert!(s.contains("n=3"));
        assert!(s.contains("min=100"));
        assert!(s.contains("max=300"));
    }

    // -----------------------------------------------------------------------
    // Large sample set
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_large_sample() {
        let mut samples = Vec::with_capacity(10000);
        for i in 0..10000u64 {
            samples.push(1000 + (i % 100)); // 1000-1099
        }
        let stats = LatencyStats::from_samples(&samples).unwrap();
        assert_eq!(stats.count, 10000);
        assert_eq!(stats.min_us, 1000);
        assert_eq!(stats.max_us, 1099);
        assert!(stats.mean_us >= 1040 && stats.mean_us <= 1060);
    }
}
