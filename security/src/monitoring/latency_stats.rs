// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Rolling inference latency statistics with anomaly detection.
//!
//! Maintains a circular buffer of latency observations and computes:
//! - p50, p99, p999 percentiles
//! - Mean and standard deviation
//! - 3-sigma anomaly detection (configurable)
//!
//! No alerts during warmup period (< min_samples observations).

/// Window size for rolling statistics.
const WINDOW_SIZE: usize = 1024;

/// Tracks inference latency observations and detects anomalies.
pub struct LatencyTracker {
    /// Circular buffer of latency observations (nanoseconds).
    samples: [u64; WINDOW_SIZE],
    /// Write position in the circular buffer.
    write_pos: usize,
    /// Total observations recorded.
    total_count: u64,
    /// Minimum samples before alerting (warmup period).
    min_samples: u32,
    /// Sigma multiplier for anomaly detection.
    sigma: u32,
    /// Cached sorted buffer for percentile computation.
    sorted: [u64; WINDOW_SIZE],
    /// Whether sorted cache is valid.
    sorted_valid: bool,
    /// Running sum for incremental mean computation.
    running_sum: u128,
    /// Running sum of squares for incremental variance.
    running_sum_sq: u128,
    /// Total anomalies detected.
    anomaly_count: u64,
}

impl LatencyTracker {
    /// Create a new latency tracker.
    ///
    /// - `min_samples`: minimum observations before alerting (default 100)
    /// - `sigma`: standard deviation multiplier for anomaly threshold (default 3)
    pub fn new(min_samples: u32, sigma: u32) -> Self {
        Self {
            samples: [0; WINDOW_SIZE],
            write_pos: 0,
            total_count: 0,
            min_samples,
            sigma,
            sorted: [0; WINDOW_SIZE],
            sorted_valid: false,
            running_sum: 0,
            running_sum_sq: 0,
            anomaly_count: 0,
        }
    }

    /// Record a latency observation in nanoseconds.
    ///
    /// Returns `true` if the observation is an anomaly (exceeds mean + sigma * stddev)
    /// and the warmup period has passed.
    pub fn record(&mut self, latency_ns: u64) -> bool {
        let n = self.count();
        let idx = self.write_pos % WINDOW_SIZE;

        // If buffer is full, subtract the evicted sample from running stats
        if self.total_count >= WINDOW_SIZE as u64 {
            let old = self.samples[idx] as u128;
            self.running_sum -= old;
            self.running_sum_sq -= old * old;
        }

        self.samples[idx] = latency_ns;
        self.write_pos += 1;
        self.total_count += 1;
        self.sorted_valid = false;

        // Update running stats
        let val = latency_ns as u128;
        self.running_sum += val;
        self.running_sum_sq += val * val;

        // Check for anomaly (only after warmup)
        if n >= self.min_samples as usize {
            let mean = self.mean_ns();
            let std = self.stddev_ns();
            let threshold = mean + (self.sigma as u64) * std;
            if latency_ns > threshold {
                self.anomaly_count += 1;
                return true;
            }
        }

        false
    }

    /// Number of samples currently in the window.
    pub fn count(&self) -> usize {
        (self.total_count as usize).min(WINDOW_SIZE)
    }

    /// Total observations ever recorded.
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Whether the warmup period is complete.
    pub fn is_warmed_up(&self) -> bool {
        self.count() >= self.min_samples as usize
    }

    /// Total anomalies detected.
    pub fn anomaly_count(&self) -> u64 {
        self.anomaly_count
    }

    /// Mean latency in nanoseconds.
    pub fn mean_ns(&self) -> u64 {
        let n = self.count();
        if n == 0 {
            return 0;
        }
        (self.running_sum / n as u128) as u64
    }

    /// Standard deviation of latency in nanoseconds.
    ///
    /// Uses the population standard deviation formula:
    /// sqrt(E[X^2] - (E[X])^2)
    pub fn stddev_ns(&self) -> u64 {
        let n = self.count();
        if n < 2 {
            return 0;
        }
        let n128 = n as u128;
        let mean_sq = self.running_sum_sq / n128;
        let sq_mean = (self.running_sum / n128) * (self.running_sum / n128);
        let variance = mean_sq.saturating_sub(sq_mean);
        isqrt_u128(variance) as u64
    }

    /// p50 (median) latency in nanoseconds.
    pub fn p50_ns(&mut self) -> u64 {
        self.percentile(500)
    }

    /// p99 latency in nanoseconds.
    pub fn p99_ns(&mut self) -> u64 {
        self.percentile(990)
    }

    /// p999 latency in nanoseconds (99.9th percentile).
    pub fn p999_ns(&mut self) -> u64 {
        self.percentile(999)
    }

    /// Compute a percentile (0-1000 scale, where 500 = p50, 999 = p99.9).
    fn percentile(&mut self, pct_tenths: u32) -> u64 {
        let n = self.count();
        if n == 0 {
            return 0;
        }

        self.ensure_sorted();

        // Nearest-rank method
        let rank = ((pct_tenths as u64) * (n as u64)).div_ceil(1000);
        let rank = rank.min(n as u64).max(1) - 1;
        self.sorted[rank as usize]
    }

    fn ensure_sorted(&mut self) {
        if self.sorted_valid {
            return;
        }
        let n = self.count();
        let src_len = if self.total_count <= WINDOW_SIZE as u64 {
            n
        } else {
            WINDOW_SIZE
        };
        self.sorted[..n].copy_from_slice(&self.samples[..src_len]);
        // Insertion sort is fine for 1024 elements in a no_std kernel
        insertion_sort(&mut self.sorted[..n]);
        self.sorted_valid = true;
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new(100, 3)
    }
}

/// Integer square root of a u128 value using Newton's method.
fn isqrt_u128(n: u128) -> u128 {
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

/// Insertion sort for small arrays (avoids alloc dependency).
fn insertion_sort(arr: &mut [u64]) {
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i;
        while j > 0 && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tracker() {
        let mut tracker = LatencyTracker::new(100, 3);
        assert_eq!(tracker.count(), 0);
        assert_eq!(tracker.mean_ns(), 0);
        assert_eq!(tracker.stddev_ns(), 0);
        assert_eq!(tracker.p50_ns(), 0);
        assert!(!tracker.is_warmed_up());
    }

    #[test]
    fn single_sample() {
        let mut tracker = LatencyTracker::new(100, 3);
        assert!(!tracker.record(1000));
        assert_eq!(tracker.count(), 1);
        assert_eq!(tracker.mean_ns(), 1000);
        assert!(!tracker.is_warmed_up());
    }

    #[test]
    fn warmup_period() {
        let mut tracker = LatencyTracker::new(10, 3);
        for i in 0..9 {
            assert!(!tracker.record(1000 + i));
        }
        assert!(!tracker.is_warmed_up());
        assert!(!tracker.record(1000)); // 10th sample
        assert!(tracker.is_warmed_up());
    }

    #[test]
    fn no_anomaly_during_warmup() {
        let mut tracker = LatencyTracker::new(100, 3);
        // Even a huge outlier shouldn't alert during warmup
        for _ in 0..99 {
            tracker.record(1000);
        }
        assert!(!tracker.record(1_000_000)); // should not alert
        assert_eq!(tracker.anomaly_count(), 0);
    }

    #[test]
    fn anomaly_detection_after_warmup() {
        let mut tracker = LatencyTracker::new(10, 3);
        // Build baseline of consistent values
        for _ in 0..100 {
            tracker.record(1000);
        }
        assert!(tracker.is_warmed_up());
        // Inject an outlier far above mean + 3*sigma
        let is_anomaly = tracker.record(1_000_000);
        assert!(is_anomaly);
        assert_eq!(tracker.anomaly_count(), 1);
    }

    #[test]
    fn no_anomaly_for_normal_variation() {
        let mut tracker = LatencyTracker::new(10, 3);
        // Samples with some natural variation
        for i in 0..200 {
            let latency = 1000 + (i % 100);
            tracker.record(latency);
        }
        // Value within normal range should not be an anomaly
        let is_anomaly = tracker.record(1050);
        assert!(!is_anomaly);
    }

    #[test]
    fn mean_computation() {
        let mut tracker = LatencyTracker::new(100, 3);
        tracker.record(100);
        tracker.record(200);
        tracker.record(300);
        assert_eq!(tracker.mean_ns(), 200);
    }

    #[test]
    fn percentile_median() {
        let mut tracker = LatencyTracker::new(100, 3);
        for i in 1..=100 {
            tracker.record(i * 10);
        }
        let p50 = tracker.p50_ns();
        // Median of 10,20,...,1000 should be around 500-510
        assert!(p50 >= 490 && p50 <= 510, "p50={}", p50);
    }

    #[test]
    fn percentile_p99() {
        let mut tracker = LatencyTracker::new(100, 3);
        for i in 1..=1000 {
            tracker.record(i);
        }
        let p99 = tracker.p99_ns();
        // p99 of 1..=1000 should be around 990
        assert!(p99 >= 985 && p99 <= 1000, "p99={}", p99);
    }

    #[test]
    fn percentile_p999() {
        let mut tracker = LatencyTracker::new(100, 3);
        for i in 1..=1000 {
            tracker.record(i);
        }
        let p999 = tracker.p999_ns();
        // p999 of 1..=1000 should be very close to 1000
        assert!(p999 >= 998 && p999 <= 1000, "p999={}", p999);
    }

    #[test]
    fn window_eviction() {
        let mut tracker = LatencyTracker::new(10, 3);
        // Fill the window
        for _ in 0..WINDOW_SIZE {
            tracker.record(1000);
        }
        assert_eq!(tracker.count(), WINDOW_SIZE);
        // Add more: should stay at WINDOW_SIZE
        for _ in 0..100 {
            tracker.record(2000);
        }
        assert_eq!(tracker.count(), WINDOW_SIZE);
        // Mean should shift toward 2000
        let mean = tracker.mean_ns();
        assert!(mean > 1000, "mean should shift: {}", mean);
    }

    #[test]
    fn stddev_constant_values() {
        let mut tracker = LatencyTracker::new(100, 3);
        for _ in 0..50 {
            tracker.record(1000);
        }
        assert_eq!(tracker.stddev_ns(), 0);
    }

    #[test]
    fn stddev_varied_values() {
        let mut tracker = LatencyTracker::new(100, 3);
        for _ in 0..50 {
            tracker.record(100);
        }
        for _ in 0..50 {
            tracker.record(200);
        }
        let std = tracker.stddev_ns();
        assert!(std > 0, "stddev should be non-zero: {}", std);
        assert!(std <= 100, "stddev should be <= 100: {}", std);
    }

    #[test]
    fn isqrt_values() {
        assert_eq!(isqrt_u128(0), 0);
        assert_eq!(isqrt_u128(1), 1);
        assert_eq!(isqrt_u128(4), 2);
        assert_eq!(isqrt_u128(9), 3);
        assert_eq!(isqrt_u128(10), 3);
        assert_eq!(isqrt_u128(100), 10);
        assert_eq!(isqrt_u128(1_000_000), 1000);
    }

    #[test]
    fn insertion_sort_basic() {
        let mut arr = [5, 3, 1, 4, 2];
        insertion_sort(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn insertion_sort_already_sorted() {
        let mut arr = [1, 2, 3, 4, 5];
        insertion_sort(&mut arr);
        assert_eq!(arr, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn insertion_sort_empty() {
        let mut arr: [u64; 0] = [];
        insertion_sort(&mut arr);
    }

    #[test]
    fn insertion_sort_single() {
        let mut arr = [42];
        insertion_sort(&mut arr);
        assert_eq!(arr, [42]);
    }
}
