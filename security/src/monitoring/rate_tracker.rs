// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Windowed rate tracker for per-second event counting.
//!
//! Tracks events in 1-second windows and detects when the rate exceeds
//! a configurable threshold. Used for capability denial rate (6.1),
//! memory allocation failure rate (6.2), and network connection rate (6.5).

/// Number of per-task slots for capability denial tracking.
const MAX_TRACKED_TASKS: usize = 64;

/// A windowed rate counter for a single metric.
///
/// Counts events within discrete 1-second windows. When a new window
/// starts, the previous window's count becomes queryable and the
/// current window resets.
#[derive(Debug)]
pub struct RateTracker {
    /// Event count in the current window.
    current_count: u32,
    /// Event count from the previous (completed) window.
    previous_count: u32,
    /// Start time of the current window in nanoseconds.
    window_start_ns: u64,
    /// Window duration in nanoseconds (default 1 second).
    window_ns: u64,
    /// Alert threshold: events per window.
    threshold: u32,
    /// Total events ever recorded.
    total_events: u64,
    /// Total alerts ever triggered.
    total_alerts: u64,
}

impl RateTracker {
    /// Create a rate tracker with a 1-second window.
    pub fn new(threshold: u32) -> Self {
        Self {
            current_count: 0,
            previous_count: 0,
            window_start_ns: 0,
            window_ns: 1_000_000_000,
            threshold,
            total_events: 0,
            total_alerts: 0,
        }
    }

    /// Record an event. Returns `true` if the threshold was just crossed
    /// in the current window (fires once per window crossing).
    pub fn record(&mut self, now_ns: u64) -> bool {
        self.advance_window(now_ns);
        self.current_count += 1;
        self.total_events += 1;

        // Fire alert exactly when crossing the threshold
        if self.current_count == self.threshold {
            self.total_alerts += 1;
            return true;
        }
        false
    }

    /// Get the rate (events/window) from the most recently completed window.
    pub fn last_rate(&self) -> u32 {
        self.previous_count
    }

    /// Get the count in the current (incomplete) window.
    pub fn current_count(&self) -> u32 {
        self.current_count
    }

    /// Whether the current window has exceeded the threshold.
    pub fn is_over_threshold(&self) -> bool {
        self.current_count >= self.threshold
    }

    /// Total events recorded across all windows.
    pub fn total_events(&self) -> u64 {
        self.total_events
    }

    /// Total alerts triggered.
    pub fn total_alerts(&self) -> u64 {
        self.total_alerts
    }

    /// The configured threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Update the threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
    }

    /// Advance the window if enough time has passed.
    fn advance_window(&mut self, now_ns: u64) {
        if now_ns.saturating_sub(self.window_start_ns) >= self.window_ns {
            self.previous_count = self.current_count;
            self.current_count = 0;
            self.window_start_ns = now_ns;
        }
    }
}

/// Per-task capability denial rate tracker.
///
/// Maintains separate rate counters for up to `MAX_TRACKED_TASKS` tasks.
/// When a new task exceeds the slot limit, the oldest slot is evicted.
pub struct CapabilityDenialTracker {
    slots: [TaskSlot; MAX_TRACKED_TASKS],
    threshold: u32,
    total_alerts: u64,
}

#[derive(Debug, Clone, Copy)]
struct TaskSlot {
    task_id: u64,
    count: u32,
    window_start_ns: u64,
    active: bool,
}

impl TaskSlot {
    const fn empty() -> Self {
        Self {
            task_id: 0,
            count: 0,
            window_start_ns: 0,
            active: false,
        }
    }
}

impl CapabilityDenialTracker {
    /// Create a new tracker with the given per-task threshold.
    pub fn new(threshold: u32) -> Self {
        Self {
            slots: [TaskSlot::empty(); MAX_TRACKED_TASKS],
            threshold,
            total_alerts: 0,
        }
    }

    /// Record a capability denial for a task.
    /// Returns `true` if the threshold was crossed for this task.
    pub fn record_denial(&mut self, task_id: u64, now_ns: u64) -> bool {
        let window_ns = 1_000_000_000u64;

        // Find existing slot or allocate new one
        let slot = if let Some(idx) = self.find_slot(task_id) {
            &mut self.slots[idx]
        } else {
            let idx = self.allocate_slot(task_id, now_ns);
            &mut self.slots[idx]
        };

        // Advance window if needed
        if now_ns.saturating_sub(slot.window_start_ns) >= window_ns {
            slot.count = 0;
            slot.window_start_ns = now_ns;
        }

        slot.count += 1;

        if slot.count == self.threshold {
            self.total_alerts += 1;
            return true;
        }
        false
    }

    /// Get the current denial count for a task in the active window.
    pub fn task_count(&self, task_id: u64) -> u32 {
        self.find_slot(task_id)
            .map(|idx| self.slots[idx].count)
            .unwrap_or(0)
    }

    /// The configured threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Total alerts triggered across all tasks.
    pub fn total_alerts(&self) -> u64 {
        self.total_alerts
    }

    fn find_slot(&self, task_id: u64) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| s.active && s.task_id == task_id)
    }

    fn allocate_slot(&mut self, task_id: u64, now_ns: u64) -> usize {
        // Find first inactive slot
        if let Some(idx) = self.slots.iter().position(|s| !s.active) {
            self.slots[idx] = TaskSlot {
                task_id,
                count: 0,
                window_start_ns: now_ns,
                active: true,
            };
            return idx;
        }
        // All slots full: evict the one with oldest window_start
        let oldest = self
            .slots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.window_start_ns)
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.slots[oldest] = TaskSlot {
            task_id,
            count: 0,
            window_start_ns: now_ns,
            active: true,
        };
        oldest
    }
}

impl Default for CapabilityDenialTracker {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Per-region memory allocation failure tracker.
pub struct MemoryFailureTracker {
    buddy: RateTracker,
    slab: RateTracker,
    tensor: RateTracker,
}

impl MemoryFailureTracker {
    /// Create with the given threshold per region.
    pub fn new(threshold: u32) -> Self {
        Self {
            buddy: RateTracker::new(threshold),
            slab: RateTracker::new(threshold),
            tensor: RateTracker::new(threshold),
        }
    }

    /// Record a failure in the specified region.
    /// Returns `true` if the threshold was crossed for that region.
    pub fn record_failure(&mut self, region: super::alerts::AllocatorRegion, now_ns: u64) -> bool {
        match region {
            super::alerts::AllocatorRegion::Buddy => self.buddy.record(now_ns),
            super::alerts::AllocatorRegion::Slab => self.slab.record(now_ns),
            super::alerts::AllocatorRegion::Tensor => self.tensor.record(now_ns),
        }
    }

    /// Get the tracker for a specific region.
    pub fn region_tracker(&self, region: super::alerts::AllocatorRegion) -> &RateTracker {
        match region {
            super::alerts::AllocatorRegion::Buddy => &self.buddy,
            super::alerts::AllocatorRegion::Slab => &self.slab,
            super::alerts::AllocatorRegion::Tensor => &self.tensor,
        }
    }
}

impl Default for MemoryFailureTracker {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitoring::alerts::AllocatorRegion;

    // --- RateTracker tests ---

    #[test]
    fn rate_tracker_no_alert_below_threshold() {
        let mut rt = RateTracker::new(10);
        for i in 0..9 {
            assert!(!rt.record(i));
        }
        assert_eq!(rt.current_count(), 9);
        assert!(!rt.is_over_threshold());
    }

    #[test]
    fn rate_tracker_alert_at_threshold() {
        let mut rt = RateTracker::new(5);
        for i in 0..4 {
            assert!(!rt.record(i));
        }
        // 5th event crosses threshold
        assert!(rt.record(4));
        assert!(rt.is_over_threshold());
        assert_eq!(rt.total_alerts(), 1);
    }

    #[test]
    fn rate_tracker_alert_fires_once_per_window() {
        let mut rt = RateTracker::new(3);
        assert!(!rt.record(0));
        assert!(!rt.record(1));
        assert!(rt.record(2)); // threshold crossed
        assert!(!rt.record(3)); // still in same window, no re-alert
        assert!(!rt.record(4));
        assert_eq!(rt.total_alerts(), 1);
    }

    #[test]
    fn rate_tracker_window_advance() {
        let mut rt = RateTracker::new(5);
        rt.record(0);
        rt.record(100);
        rt.record(200);
        assert_eq!(rt.current_count(), 3);

        // New window (1 second later)
        rt.record(1_000_000_000);
        assert_eq!(rt.current_count(), 1);
        assert_eq!(rt.last_rate(), 3);
    }

    #[test]
    fn rate_tracker_threshold_update() {
        let mut rt = RateTracker::new(10);
        rt.set_threshold(5);
        assert_eq!(rt.threshold(), 5);
    }

    #[test]
    fn rate_tracker_total_events() {
        let mut rt = RateTracker::new(100);
        for i in 0..50 {
            rt.record(i);
        }
        assert_eq!(rt.total_events(), 50);
    }

    // --- CapabilityDenialTracker tests ---

    #[test]
    fn cap_denial_no_alert_below_threshold() {
        let mut tracker = CapabilityDenialTracker::new(10);
        for i in 0..9 {
            assert!(!tracker.record_denial(1, i));
        }
        assert_eq!(tracker.task_count(1), 9);
    }

    #[test]
    fn cap_denial_alert_at_threshold() {
        let mut tracker = CapabilityDenialTracker::new(10);
        for i in 0..9 {
            assert!(!tracker.record_denial(42, i));
        }
        assert!(tracker.record_denial(42, 9));
        assert_eq!(tracker.total_alerts(), 1);
    }

    #[test]
    fn cap_denial_per_task_isolation() {
        let mut tracker = CapabilityDenialTracker::new(3);
        // Task 1: 2 denials
        tracker.record_denial(1, 0);
        tracker.record_denial(1, 0);
        // Task 2: 2 denials
        tracker.record_denial(2, 0);
        tracker.record_denial(2, 0);
        // Neither should have alerted
        assert_eq!(tracker.total_alerts(), 0);
        assert_eq!(tracker.task_count(1), 2);
        assert_eq!(tracker.task_count(2), 2);
        // Task 1 crosses
        assert!(tracker.record_denial(1, 0));
        assert_eq!(tracker.total_alerts(), 1);
    }

    #[test]
    fn cap_denial_window_reset() {
        let mut tracker = CapabilityDenialTracker::new(5);
        for i in 0..4 {
            tracker.record_denial(1, i);
        }
        // New window
        tracker.record_denial(1, 2_000_000_000);
        assert_eq!(tracker.task_count(1), 1);
    }

    #[test]
    fn cap_denial_unknown_task_zero() {
        let tracker = CapabilityDenialTracker::new(10);
        assert_eq!(tracker.task_count(999), 0);
    }

    // --- MemoryFailureTracker tests ---

    #[test]
    fn mem_failure_per_region() {
        let mut tracker = MemoryFailureTracker::new(3);
        tracker.record_failure(AllocatorRegion::Buddy, 0);
        tracker.record_failure(AllocatorRegion::Slab, 0);
        tracker.record_failure(AllocatorRegion::Tensor, 0);

        assert_eq!(
            tracker
                .region_tracker(AllocatorRegion::Buddy)
                .current_count(),
            1
        );
        assert_eq!(
            tracker
                .region_tracker(AllocatorRegion::Slab)
                .current_count(),
            1
        );
        assert_eq!(
            tracker
                .region_tracker(AllocatorRegion::Tensor)
                .current_count(),
            1
        );
    }

    #[test]
    fn mem_failure_alert_per_region() {
        let mut tracker = MemoryFailureTracker::new(2);
        assert!(!tracker.record_failure(AllocatorRegion::Buddy, 0));
        assert!(tracker.record_failure(AllocatorRegion::Buddy, 1));
        // Slab hasn't alerted
        assert!(!tracker.record_failure(AllocatorRegion::Slab, 2));
    }
}
