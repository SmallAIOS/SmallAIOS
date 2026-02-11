// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Watchdog time-remaining tracker with threshold alerting.
//!
//! Monitors the watchdog timer's remaining time and generates an alert
//! when it drops below a configurable percentage (default 50%) of the
//! timeout period. Clears the alert condition when the watchdog is refreshed.

/// Tracks watchdog time remaining and alerts on time pressure.
#[derive(Debug)]
pub struct WatchdogMonitor {
    /// Configured watchdog timeout period in nanoseconds.
    timeout_ns: u64,
    /// Warning threshold as percentage (0-100).
    warn_pct: u32,
    /// Whether the warning condition is currently active.
    warning_active: bool,
    /// Last known remaining time.
    last_remaining_ns: u64,
    /// Timestamp of last pet (refresh).
    last_pet_ns: u64,
    /// Total warnings generated.
    total_warnings: u64,
}

impl WatchdogMonitor {
    /// Create a new watchdog monitor.
    ///
    /// - `timeout_ns`: the watchdog timeout period in nanoseconds
    /// - `warn_pct`: percentage of timeout below which to warn (e.g., 50)
    pub fn new(timeout_ns: u64, warn_pct: u32) -> Self {
        Self {
            timeout_ns,
            warn_pct: warn_pct.min(100),
            warning_active: false,
            last_remaining_ns: timeout_ns,
            last_pet_ns: 0,
            total_warnings: 0,
        }
    }

    /// Update the watchdog state with the current remaining time.
    ///
    /// Returns `true` if a new warning was generated (transitions from
    /// OK to warning state). Does not re-alert if already in warning state.
    pub fn update(&mut self, remaining_ns: u64) -> bool {
        self.last_remaining_ns = remaining_ns;
        let threshold = self.threshold_ns();

        if remaining_ns < threshold {
            if !self.warning_active {
                self.warning_active = true;
                self.total_warnings += 1;
                return true;
            }
        } else {
            // Clear warning condition
            self.warning_active = false;
        }
        false
    }

    /// Record a watchdog pet (refresh). Resets remaining time to full timeout.
    pub fn pet(&mut self, now_ns: u64) {
        self.last_pet_ns = now_ns;
        self.last_remaining_ns = self.timeout_ns;
        self.warning_active = false;
    }

    /// Whether the warning condition is currently active.
    pub fn is_warning(&self) -> bool {
        self.warning_active
    }

    /// Last known remaining time in nanoseconds.
    pub fn remaining_ns(&self) -> u64 {
        self.last_remaining_ns
    }

    /// The timeout period in nanoseconds.
    pub fn timeout_ns(&self) -> u64 {
        self.timeout_ns
    }

    /// The warning threshold in nanoseconds.
    pub fn threshold_ns(&self) -> u64 {
        (self.timeout_ns / 100) * self.warn_pct as u64
    }

    /// Warning percentage.
    pub fn warn_pct(&self) -> u32 {
        self.warn_pct
    }

    /// Total warnings generated.
    pub fn total_warnings(&self) -> u64 {
        self.total_warnings
    }

    /// Timestamp of last pet.
    pub fn last_pet_ns(&self) -> u64 {
        self.last_pet_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_monitor_no_warning() {
        let monitor = WatchdogMonitor::new(1_000_000_000, 50);
        assert!(!monitor.is_warning());
        assert_eq!(monitor.remaining_ns(), 1_000_000_000);
        assert_eq!(monitor.timeout_ns(), 1_000_000_000);
        assert_eq!(monitor.threshold_ns(), 500_000_000);
        assert_eq!(monitor.total_warnings(), 0);
    }

    #[test]
    fn warning_on_low_remaining() {
        let mut monitor = WatchdogMonitor::new(1_000_000_000, 50);
        // Remaining at 60% - no warning
        assert!(!monitor.update(600_000_000));
        assert!(!monitor.is_warning());

        // Remaining drops below 50% - warning
        assert!(monitor.update(400_000_000));
        assert!(monitor.is_warning());
        assert_eq!(monitor.total_warnings(), 1);
    }

    #[test]
    fn no_repeated_warning() {
        let mut monitor = WatchdogMonitor::new(1_000_000_000, 50);
        // First drop below threshold
        assert!(monitor.update(300_000_000));
        // Subsequent updates below threshold: no re-alert
        assert!(!monitor.update(200_000_000));
        assert!(!monitor.update(100_000_000));
        assert_eq!(monitor.total_warnings(), 1);
    }

    #[test]
    fn clear_on_pet() {
        let mut monitor = WatchdogMonitor::new(1_000_000_000, 50);
        monitor.update(300_000_000);
        assert!(monitor.is_warning());

        // Pet clears the warning
        monitor.pet(500_000_000);
        assert!(!monitor.is_warning());
        assert_eq!(monitor.remaining_ns(), 1_000_000_000);
        assert_eq!(monitor.last_pet_ns(), 500_000_000);
    }

    #[test]
    fn re_alert_after_clear() {
        let mut monitor = WatchdogMonitor::new(1_000_000_000, 50);
        // First warning
        assert!(monitor.update(300_000_000));
        assert_eq!(monitor.total_warnings(), 1);

        // Pet clears
        monitor.pet(1_000_000);
        assert!(!monitor.is_warning());

        // Warning again
        assert!(monitor.update(400_000_000));
        assert_eq!(monitor.total_warnings(), 2);
    }

    #[test]
    fn clear_on_remaining_above_threshold() {
        let mut monitor = WatchdogMonitor::new(1_000_000_000, 50);
        // Below threshold
        monitor.update(300_000_000);
        assert!(monitor.is_warning());

        // Above threshold (e.g., after pet without calling pet())
        monitor.update(600_000_000);
        assert!(!monitor.is_warning());
    }

    #[test]
    fn threshold_at_boundary() {
        let mut monitor = WatchdogMonitor::new(1_000_000_000, 50);
        // Exactly at threshold: not below, no warning
        assert!(!monitor.update(500_000_000));
        assert!(!monitor.is_warning());

        // One ns below threshold
        assert!(monitor.update(499_999_999));
        assert!(monitor.is_warning());
    }

    #[test]
    fn custom_warn_percentage() {
        let monitor = WatchdogMonitor::new(1_000_000_000, 75);
        assert_eq!(monitor.warn_pct(), 75);
        assert_eq!(monitor.threshold_ns(), 750_000_000);
    }

    #[test]
    fn warn_pct_clamped_to_100() {
        let monitor = WatchdogMonitor::new(1_000_000_000, 150);
        assert_eq!(monitor.warn_pct(), 100);
    }

    #[test]
    fn zero_timeout() {
        let mut monitor = WatchdogMonitor::new(0, 50);
        assert_eq!(monitor.threshold_ns(), 0);
        // Any remaining time >= 0 means not below 0 threshold
        assert!(!monitor.update(0));
    }
}
