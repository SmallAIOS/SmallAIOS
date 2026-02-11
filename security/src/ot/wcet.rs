// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! WCET (Worst-Case Execution Time) instrumentation and measurement.
//!
//! Provides timestamp-based instrumentation for critical kernel paths:
//! - Syscall dispatch
//! - Capability check
//! - Memory allocation (buddy + slab)
//! - Task scheduling
//! - Interrupt handling
//!
//! Collects N>=10,000 samples per path and reports p99.9, min, max,
//! mean, and standard deviation per NIST requirements.

use alloc::vec::Vec;

/// Critical kernel paths that require WCET analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CriticalPath {
    /// Syscall dispatch: entry through capability validation to handler return.
    SyscallDispatch = 0,
    /// Capability check: token validation against capability table.
    CapabilityCheck = 1,
    /// Buddy allocator: page-level memory allocation.
    BuddyAlloc = 2,
    /// Slab allocator: small object allocation.
    SlabAlloc = 3,
    /// Task scheduling: scheduling decision to context switch.
    TaskSchedule = 4,
    /// Interrupt handling: hardware assertion to handler entry.
    InterruptHandle = 5,
}

impl CriticalPath {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::SyscallDispatch),
            1 => Some(Self::CapabilityCheck),
            2 => Some(Self::BuddyAlloc),
            3 => Some(Self::SlabAlloc),
            4 => Some(Self::TaskSchedule),
            5 => Some(Self::InterruptHandle),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SyscallDispatch => "syscall-dispatch",
            Self::CapabilityCheck => "capability-check",
            Self::BuddyAlloc => "buddy-alloc",
            Self::SlabAlloc => "slab-alloc",
            Self::TaskSchedule => "task-schedule",
            Self::InterruptHandle => "interrupt-handle",
        }
    }
}

/// Number of critical paths.
const NUM_PATHS: usize = 6;

/// Maximum samples per path for measurement.
const MAX_SAMPLES: usize = 16384;

/// Minimum samples required for valid statistics.
const MIN_SAMPLES: usize = 10_000;

/// WCET measurement statistics for a single critical path.
#[derive(Debug, Clone, Copy)]
pub struct WcetStats {
    /// Minimum observed execution time (nanoseconds).
    pub min_ns: u64,
    /// Maximum observed execution time (nanoseconds).
    pub max_ns: u64,
    /// Mean execution time (nanoseconds).
    pub mean_ns: u64,
    /// Standard deviation (nanoseconds).
    pub stddev_ns: u64,
    /// 99.9th percentile (nanoseconds).
    pub p999_ns: u64,
    /// Number of samples collected.
    pub sample_count: u32,
    /// Static WCET bound (nanoseconds), set by analysis.
    pub static_bound_ns: u64,
}

impl WcetStats {
    const fn empty() -> Self {
        Self {
            min_ns: u64::MAX,
            max_ns: 0,
            mean_ns: 0,
            stddev_ns: 0,
            p999_ns: 0,
            sample_count: 0,
            static_bound_ns: 0,
        }
    }

    /// Whether enough samples have been collected.
    pub fn has_sufficient_samples(&self) -> bool {
        self.sample_count as usize >= MIN_SAMPLES
    }

    /// Whether p99.9 exceeds 80% of the static bound (requires investigation).
    pub fn exceeds_investigation_threshold(&self) -> bool {
        if self.static_bound_ns == 0 {
            return false;
        }
        // p999 > 80% of static bound
        self.p999_ns > (self.static_bound_ns * 4) / 5
    }

    /// Whether p99.9 exceeds the static bound (CI failure).
    pub fn exceeds_static_bound(&self) -> bool {
        if self.static_bound_ns == 0 {
            return false;
        }
        self.p999_ns > self.static_bound_ns
    }
}

/// Per-path sample collector.
struct PathCollector {
    /// Circular buffer of samples (nanoseconds), heap-allocated.
    samples: Vec<u64>,
    /// Current write position.
    write_pos: usize,
    /// Total samples collected (may exceed MAX_SAMPLES).
    total_collected: u64,
    /// Running sum for mean calculation.
    running_sum: u128,
    /// Running sum of squares for stddev.
    running_sum_sq: u128,
    /// Minimum observed.
    min_ns: u64,
    /// Maximum observed.
    max_ns: u64,
    /// Static WCET bound.
    static_bound_ns: u64,
}

impl PathCollector {
    fn new() -> Self {
        let samples = alloc::vec![0u64; MAX_SAMPLES];
        Self {
            samples,
            write_pos: 0,
            total_collected: 0,
            running_sum: 0,
            running_sum_sq: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            static_bound_ns: 0,
        }
    }

    fn record(&mut self, duration_ns: u64) {
        self.samples[self.write_pos] = duration_ns;
        self.write_pos = (self.write_pos + 1) % MAX_SAMPLES;
        self.total_collected += 1;
        self.running_sum += duration_ns as u128;
        self.running_sum_sq += (duration_ns as u128) * (duration_ns as u128);

        if duration_ns < self.min_ns {
            self.min_ns = duration_ns;
        }
        if duration_ns > self.max_ns {
            self.max_ns = duration_ns;
        }
    }

    fn active_count(&self) -> usize {
        (self.total_collected as usize).min(MAX_SAMPLES)
    }

    fn compute_stats(&self) -> WcetStats {
        let n = self.active_count();
        if n == 0 {
            return WcetStats::empty();
        }

        let mean = (self.running_sum / self.total_collected as u128) as u64;

        // Stddev from running sums
        let mean128 = self.running_sum / self.total_collected as u128;
        let var = self.running_sum_sq / self.total_collected as u128 - mean128 * mean128;
        let stddev = isqrt_u128(var) as u64;

        // p99.9 requires sorting the active samples
        let p999 = self.compute_percentile(999);

        WcetStats {
            min_ns: self.min_ns,
            max_ns: self.max_ns,
            mean_ns: mean,
            stddev_ns: stddev,
            p999_ns: p999,
            sample_count: self.total_collected.min(u32::MAX as u64) as u32,
            static_bound_ns: self.static_bound_ns,
        }
    }

    fn compute_percentile(&self, permille: u32) -> u64 {
        let n = self.active_count();
        if n == 0 {
            return 0;
        }

        // Copy active samples and sort (heap-allocated to avoid stack overflow)
        let mut sorted: Vec<u64> = self.samples[..n].to_vec();
        insertion_sort(&mut sorted);

        let idx = ((n as u64) * (permille as u64)).div_ceil(1000) as usize;
        let idx = idx.min(n).saturating_sub(1);
        sorted[idx]
    }
}

/// Integer square root for u128 (Newton's method).
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

/// In-place insertion sort for u64 slices.
fn insertion_sort(data: &mut [u64]) {
    for i in 1..data.len() {
        let key = data[i];
        let mut j = i;
        while j > 0 && data[j - 1] > key {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
    }
}

/// WCET measurement framework for all critical kernel paths.
///
/// Records execution time samples via `record_entry`/`record_exit`
/// and computes statistics on demand.
pub struct WcetFramework {
    collectors: Vec<PathCollector>,
    /// Pending entry timestamps per path (for entry/exit pairing).
    pending_entry: [u64; NUM_PATHS],
}

impl WcetFramework {
    pub fn new() -> Self {
        let mut collectors = Vec::with_capacity(NUM_PATHS);
        for _ in 0..NUM_PATHS {
            collectors.push(PathCollector::new());
        }
        Self {
            collectors,
            pending_entry: [0; NUM_PATHS],
        }
    }

    /// Record entry to a critical path (call at path start).
    pub fn record_entry(&mut self, path: CriticalPath, timestamp_ns: u64) {
        self.pending_entry[path as usize] = timestamp_ns;
    }

    /// Record exit from a critical path (call at path end).
    ///
    /// Computes duration and adds sample.
    pub fn record_exit(&mut self, path: CriticalPath, timestamp_ns: u64) {
        let idx = path as usize;
        let entry = self.pending_entry[idx];
        if entry > 0 && timestamp_ns >= entry {
            let duration = timestamp_ns - entry;
            self.collectors[idx].record(duration);
        }
        self.pending_entry[idx] = 0;
    }

    /// Record a pre-computed duration directly.
    pub fn record_duration(&mut self, path: CriticalPath, duration_ns: u64) {
        self.collectors[path as usize].record(duration_ns);
    }

    /// Set the static WCET bound for a path.
    pub fn set_static_bound(&mut self, path: CriticalPath, bound_ns: u64) {
        self.collectors[path as usize].static_bound_ns = bound_ns;
    }

    /// Get statistics for a specific path.
    pub fn stats(&self, path: CriticalPath) -> WcetStats {
        self.collectors[path as usize].compute_stats()
    }

    /// Total samples collected across all paths.
    pub fn total_samples(&self) -> u64 {
        self.collectors.iter().map(|c| c.total_collected).sum()
    }

    /// Check if all paths have sufficient samples.
    pub fn all_paths_sufficient(&self) -> bool {
        self.collectors.iter().all(|c| c.active_count() >= MIN_SAMPLES)
    }

    /// Check if any path exceeds its static bound.
    pub fn any_exceeds_bound(&self) -> bool {
        (0..NUM_PATHS).any(|i| {
            let stats = self.collectors[i].compute_stats();
            stats.exceeds_static_bound()
        })
    }
}

impl Default for WcetFramework {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_path_roundtrip() {
        for i in 0..=5u8 {
            let p = CriticalPath::from_u8(i).unwrap();
            assert_eq!(p as u8, i);
        }
        assert!(CriticalPath::from_u8(6).is_none());
    }

    #[test]
    fn critical_path_as_str() {
        assert_eq!(CriticalPath::SyscallDispatch.as_str(), "syscall-dispatch");
        assert_eq!(CriticalPath::CapabilityCheck.as_str(), "capability-check");
        assert_eq!(CriticalPath::BuddyAlloc.as_str(), "buddy-alloc");
        assert_eq!(CriticalPath::SlabAlloc.as_str(), "slab-alloc");
        assert_eq!(CriticalPath::TaskSchedule.as_str(), "task-schedule");
        assert_eq!(CriticalPath::InterruptHandle.as_str(), "interrupt-handle");
    }

    #[test]
    fn empty_stats() {
        let fw = WcetFramework::new();
        let s = fw.stats(CriticalPath::SyscallDispatch);
        assert_eq!(s.sample_count, 0);
        assert!(!s.has_sufficient_samples());
    }

    #[test]
    fn record_entry_exit() {
        let mut fw = WcetFramework::new();
        fw.record_entry(CriticalPath::CapabilityCheck, 1000);
        fw.record_exit(CriticalPath::CapabilityCheck, 1500);

        let s = fw.stats(CriticalPath::CapabilityCheck);
        assert_eq!(s.sample_count, 1);
        assert_eq!(s.min_ns, 500);
        assert_eq!(s.max_ns, 500);
        assert_eq!(s.mean_ns, 500);
    }

    #[test]
    fn record_duration_directly() {
        let mut fw = WcetFramework::new();
        fw.record_duration(CriticalPath::BuddyAlloc, 100);
        fw.record_duration(CriticalPath::BuddyAlloc, 200);
        fw.record_duration(CriticalPath::BuddyAlloc, 300);

        let s = fw.stats(CriticalPath::BuddyAlloc);
        assert_eq!(s.sample_count, 3);
        assert_eq!(s.min_ns, 100);
        assert_eq!(s.max_ns, 300);
        assert_eq!(s.mean_ns, 200);
    }

    #[test]
    fn static_bound_check() {
        let mut fw = WcetFramework::new();
        fw.set_static_bound(CriticalPath::SyscallDispatch, 1000);

        // Record samples below bound
        for i in 0..100 {
            fw.record_duration(CriticalPath::SyscallDispatch, 500 + i);
        }
        let s = fw.stats(CriticalPath::SyscallDispatch);
        assert!(!s.exceeds_static_bound());

        // p999 at 500+99=599, which is < 800 (80% of 1000)
        assert!(!s.exceeds_investigation_threshold());
    }

    #[test]
    fn exceeds_bound_detected() {
        let mut fw = WcetFramework::new();
        fw.set_static_bound(CriticalPath::TaskSchedule, 100);

        // Record samples exceeding bound
        for _ in 0..100 {
            fw.record_duration(CriticalPath::TaskSchedule, 200);
        }
        let s = fw.stats(CriticalPath::TaskSchedule);
        assert!(s.exceeds_static_bound());
        assert!(s.exceeds_investigation_threshold());
    }

    #[test]
    fn investigation_threshold() {
        let mut fw = WcetFramework::new();
        fw.set_static_bound(CriticalPath::SlabAlloc, 1000);

        // Record at exactly 81% of bound -> should trigger investigation
        for _ in 0..100 {
            fw.record_duration(CriticalPath::SlabAlloc, 810);
        }
        let s = fw.stats(CriticalPath::SlabAlloc);
        assert!(s.exceeds_investigation_threshold());
        assert!(!s.exceeds_static_bound());
    }

    #[test]
    fn total_samples_across_paths() {
        let mut fw = WcetFramework::new();
        fw.record_duration(CriticalPath::SyscallDispatch, 100);
        fw.record_duration(CriticalPath::CapabilityCheck, 200);
        fw.record_duration(CriticalPath::BuddyAlloc, 300);
        assert_eq!(fw.total_samples(), 3);
    }

    #[test]
    fn entry_without_exit_ignored() {
        let mut fw = WcetFramework::new();
        fw.record_entry(CriticalPath::SyscallDispatch, 1000);
        // No exit recorded
        let s = fw.stats(CriticalPath::SyscallDispatch);
        assert_eq!(s.sample_count, 0);
    }

    #[test]
    fn exit_before_entry_ignored() {
        let mut fw = WcetFramework::new();
        // Exit without prior entry (pending_entry is 0)
        fw.record_exit(CriticalPath::SyscallDispatch, 500);
        let s = fw.stats(CriticalPath::SyscallDispatch);
        assert_eq!(s.sample_count, 0);
    }

    #[test]
    fn isqrt_values() {
        assert_eq!(isqrt_u128(0), 0);
        assert_eq!(isqrt_u128(1), 1);
        assert_eq!(isqrt_u128(4), 2);
        assert_eq!(isqrt_u128(9), 3);
        assert_eq!(isqrt_u128(100), 10);
        assert_eq!(isqrt_u128(1000000), 1000);
    }

    #[test]
    fn insertion_sort_basic() {
        let mut data = [5u64, 3, 1, 4, 2];
        insertion_sort(&mut data);
        assert_eq!(data, [1, 2, 3, 4, 5]);
    }
}
