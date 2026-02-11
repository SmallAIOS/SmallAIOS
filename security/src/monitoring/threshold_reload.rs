// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Runtime threshold hot-reload for monitoring configuration.
//!
//! Allows updating monitoring thresholds without restarting the system.
//! Changes are validated before being applied. A history of threshold
//! changes is maintained for audit trail purposes.

use super::alerts::MonitoringConfig;

/// Maximum number of threshold change records to retain.
const MAX_HISTORY: usize = 32;

/// Result of a threshold update operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdUpdateResult {
    /// All thresholds updated successfully.
    Applied,
    /// Some thresholds were invalid and replaced with defaults.
    AppliedWithDefaults,
    /// No changes detected (same as current config).
    NoChange,
}

/// Record of a threshold change for audit purposes.
#[derive(Debug, Clone, Copy)]
pub struct ThresholdChangeRecord {
    /// Timestamp of the change (nanoseconds since epoch).
    pub timestamp_ns: u64,
    /// Configuration before the change.
    pub old_config: MonitoringConfig,
    /// Configuration after the change.
    pub new_config: MonitoringConfig,
    /// Result of the update.
    pub result: ThresholdUpdateResult,
}

/// Hot-reloadable threshold manager.
///
/// Wraps a `MonitoringConfig` and provides safe update operations
/// with validation and audit history.
pub struct ThresholdManager {
    /// Current active configuration.
    current: MonitoringConfig,
    /// History of changes (circular buffer).
    history: [Option<ThresholdChangeRecord>; MAX_HISTORY],
    /// Write position in the history buffer.
    history_pos: usize,
    /// Total number of updates applied.
    total_updates: u64,
}

impl ThresholdManager {
    /// Create a new manager with default thresholds.
    pub fn new() -> Self {
        Self {
            current: MonitoringConfig::default_config(),
            history: [None; MAX_HISTORY],
            history_pos: 0,
            total_updates: 0,
        }
    }

    /// Create a new manager with a specific initial configuration.
    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            current: config.validated(),
            history: [None; MAX_HISTORY],
            history_pos: 0,
            total_updates: 0,
        }
    }

    /// Get the current active configuration.
    pub fn current(&self) -> &MonitoringConfig {
        &self.current
    }

    /// Update thresholds at runtime.
    ///
    /// The new configuration is validated: zero values are replaced
    /// with defaults. Returns the result of the update operation.
    pub fn update(&mut self, new_config: MonitoringConfig, now_ns: u64) -> ThresholdUpdateResult {
        let validated = new_config.validated();
        let old = self.current;

        // Check if anything actually changed
        if configs_equal(&old, &validated) {
            return ThresholdUpdateResult::NoChange;
        }

        // Determine if defaults were substituted
        let had_defaults = !configs_equal(&new_config, &validated);

        let result = if had_defaults {
            ThresholdUpdateResult::AppliedWithDefaults
        } else {
            ThresholdUpdateResult::Applied
        };

        // Apply the change
        self.current = validated;

        // Record the change
        let record = ThresholdChangeRecord {
            timestamp_ns: now_ns,
            old_config: old,
            new_config: validated,
            result,
        };
        self.history[self.history_pos % MAX_HISTORY] = Some(record);
        self.history_pos += 1;
        self.total_updates += 1;

        result
    }

    /// Update a single threshold field.
    pub fn update_cap_denial_threshold(&mut self, value: u32, now_ns: u64) -> ThresholdUpdateResult {
        let mut new_config = self.current;
        new_config.cap_denial_threshold = value;
        self.update(new_config, now_ns)
    }

    /// Update memory failure threshold.
    pub fn update_mem_failure_threshold(&mut self, value: u32, now_ns: u64) -> ThresholdUpdateResult {
        let mut new_config = self.current;
        new_config.mem_failure_threshold = value;
        self.update(new_config, now_ns)
    }

    /// Update latency sigma multiplier.
    pub fn update_latency_sigma(&mut self, value: u32, now_ns: u64) -> ThresholdUpdateResult {
        let mut new_config = self.current;
        new_config.latency_sigma = value;
        self.update(new_config, now_ns)
    }

    /// Update network connection threshold.
    pub fn update_net_conn_threshold(&mut self, value: u32, now_ns: u64) -> ThresholdUpdateResult {
        let mut new_config = self.current;
        new_config.net_conn_threshold = value;
        self.update(new_config, now_ns)
    }

    /// Update watchdog warning percentage.
    pub fn update_watchdog_warn_pct(&mut self, value: u32, now_ns: u64) -> ThresholdUpdateResult {
        let mut new_config = self.current;
        new_config.watchdog_warn_pct = value;
        self.update(new_config, now_ns)
    }

    /// Total number of successful threshold updates.
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Get the most recent change record.
    pub fn last_change(&self) -> Option<&ThresholdChangeRecord> {
        if self.history_pos == 0 {
            return None;
        }
        self.history[(self.history_pos - 1) % MAX_HISTORY].as_ref()
    }

    /// Number of change records available.
    pub fn history_count(&self) -> usize {
        self.history_pos.min(MAX_HISTORY)
    }

    /// Iterate over change history (oldest to newest).
    pub fn history_iter(&self) -> impl Iterator<Item = &ThresholdChangeRecord> {
        let count = self.history_count();
        let start = if self.history_pos > MAX_HISTORY {
            self.history_pos % MAX_HISTORY
        } else {
            0
        };

        (0..count).filter_map(move |i| {
            let idx = (start + i) % MAX_HISTORY;
            self.history[idx].as_ref()
        })
    }
}

impl Default for ThresholdManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Compare two configs for equality (field-by-field since we can't derive PartialEq on MonitoringConfig).
fn configs_equal(a: &MonitoringConfig, b: &MonitoringConfig) -> bool {
    a.cap_denial_threshold == b.cap_denial_threshold
        && a.mem_failure_threshold == b.mem_failure_threshold
        && a.latency_sigma == b.latency_sigma
        && a.latency_min_samples == b.latency_min_samples
        && a.watchdog_warn_pct == b.watchdog_warn_pct
        && a.net_conn_threshold == b.net_conn_threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_manager() {
        let mgr = ThresholdManager::new();
        let cfg = mgr.current();
        assert_eq!(cfg.cap_denial_threshold, 10);
        assert_eq!(cfg.net_conn_threshold, 100);
        assert_eq!(mgr.total_updates(), 0);
        assert_eq!(mgr.history_count(), 0);
        assert!(mgr.last_change().is_none());
    }

    #[test]
    fn update_full_config() {
        let mut mgr = ThresholdManager::new();
        let new_cfg = MonitoringConfig {
            cap_denial_threshold: 20,
            mem_failure_threshold: 15,
            latency_sigma: 4,
            latency_min_samples: 200,
            watchdog_warn_pct: 60,
            net_conn_threshold: 150,
        };

        let result = mgr.update(new_cfg, 1_000_000);
        assert_eq!(result, ThresholdUpdateResult::Applied);
        assert_eq!(mgr.current().cap_denial_threshold, 20);
        assert_eq!(mgr.current().net_conn_threshold, 150);
        assert_eq!(mgr.total_updates(), 1);
    }

    #[test]
    fn update_with_defaults_substituted() {
        let mut mgr = ThresholdManager::new();
        let new_cfg = MonitoringConfig {
            cap_denial_threshold: 20,
            mem_failure_threshold: 0, // invalid, will be replaced
            latency_sigma: 4,
            latency_min_samples: 200,
            watchdog_warn_pct: 60,
            net_conn_threshold: 150,
        };

        let result = mgr.update(new_cfg, 1_000_000);
        assert_eq!(result, ThresholdUpdateResult::AppliedWithDefaults);
        assert_eq!(mgr.current().cap_denial_threshold, 20);
        assert_eq!(mgr.current().mem_failure_threshold, 10); // default
    }

    #[test]
    fn update_no_change() {
        let mut mgr = ThresholdManager::new();
        let same_cfg = MonitoringConfig::default_config();

        let result = mgr.update(same_cfg, 1_000_000);
        assert_eq!(result, ThresholdUpdateResult::NoChange);
        assert_eq!(mgr.total_updates(), 0);
    }

    #[test]
    fn update_single_field() {
        let mut mgr = ThresholdManager::new();

        let result = mgr.update_cap_denial_threshold(25, 1_000);
        assert_eq!(result, ThresholdUpdateResult::Applied);
        assert_eq!(mgr.current().cap_denial_threshold, 25);
        // Other fields unchanged
        assert_eq!(mgr.current().net_conn_threshold, 100);
    }

    #[test]
    fn update_single_field_zero_gets_default() {
        let mut mgr = ThresholdManager::new();

        // Setting to 0 triggers validation which replaces with default
        // But default is already 10, so no actual change
        let result = mgr.update_cap_denial_threshold(0, 1_000);
        assert_eq!(result, ThresholdUpdateResult::NoChange);
    }

    #[test]
    fn update_mem_failure_threshold() {
        let mut mgr = ThresholdManager::new();
        let result = mgr.update_mem_failure_threshold(50, 1_000);
        assert_eq!(result, ThresholdUpdateResult::Applied);
        assert_eq!(mgr.current().mem_failure_threshold, 50);
    }

    #[test]
    fn update_latency_sigma() {
        let mut mgr = ThresholdManager::new();
        let result = mgr.update_latency_sigma(5, 1_000);
        assert_eq!(result, ThresholdUpdateResult::Applied);
        assert_eq!(mgr.current().latency_sigma, 5);
    }

    #[test]
    fn update_net_conn_threshold() {
        let mut mgr = ThresholdManager::new();
        let result = mgr.update_net_conn_threshold(200, 1_000);
        assert_eq!(result, ThresholdUpdateResult::Applied);
        assert_eq!(mgr.current().net_conn_threshold, 200);
    }

    #[test]
    fn update_watchdog_warn_pct() {
        let mut mgr = ThresholdManager::new();
        let result = mgr.update_watchdog_warn_pct(75, 1_000);
        assert_eq!(result, ThresholdUpdateResult::Applied);
        assert_eq!(mgr.current().watchdog_warn_pct, 75);
    }

    #[test]
    fn change_history_records() {
        let mut mgr = ThresholdManager::new();
        mgr.update_cap_denial_threshold(20, 1_000);
        mgr.update_net_conn_threshold(200, 2_000);

        assert_eq!(mgr.history_count(), 2);
        assert_eq!(mgr.total_updates(), 2);

        let last = mgr.last_change().unwrap();
        assert_eq!(last.timestamp_ns, 2_000);
        assert_eq!(last.new_config.net_conn_threshold, 200);
        assert_eq!(last.result, ThresholdUpdateResult::Applied);
    }

    #[test]
    fn change_history_circular() {
        let mut mgr = ThresholdManager::new();

        // Fill beyond history limit
        for i in 1..=(MAX_HISTORY as u32 + 5) {
            mgr.update_cap_denial_threshold(10 + i, i as u64 * 1000);
        }

        assert_eq!(mgr.history_count(), MAX_HISTORY);
        assert_eq!(mgr.total_updates(), MAX_HISTORY as u64 + 5);

        // Most recent should be preserved
        let last = mgr.last_change().unwrap();
        assert_eq!(
            last.new_config.cap_denial_threshold,
            10 + MAX_HISTORY as u32 + 5
        );
    }

    #[test]
    fn history_iterator() {
        let mut mgr = ThresholdManager::new();
        mgr.update_cap_denial_threshold(20, 1_000);
        mgr.update_cap_denial_threshold(30, 2_000);
        mgr.update_cap_denial_threshold(40, 3_000);

        let changes: alloc::vec::Vec<_> = mgr.history_iter().collect();
        assert_eq!(changes.len(), 3);
        assert_eq!(changes[0].timestamp_ns, 1_000);
        assert_eq!(changes[1].timestamp_ns, 2_000);
        assert_eq!(changes[2].timestamp_ns, 3_000);
    }

    #[test]
    fn with_initial_config() {
        let cfg = MonitoringConfig {
            cap_denial_threshold: 50,
            mem_failure_threshold: 25,
            latency_sigma: 5,
            latency_min_samples: 500,
            watchdog_warn_pct: 80,
            net_conn_threshold: 300,
        };
        let mgr = ThresholdManager::with_config(cfg);
        assert_eq!(mgr.current().cap_denial_threshold, 50);
        assert_eq!(mgr.current().net_conn_threshold, 300);
    }

    #[test]
    fn change_record_preserves_old_and_new() {
        let mut mgr = ThresholdManager::new();
        mgr.update_cap_denial_threshold(25, 1_000);

        let record = mgr.last_change().unwrap();
        assert_eq!(record.old_config.cap_denial_threshold, 10); // was default
        assert_eq!(record.new_config.cap_denial_threshold, 25); // new value
    }
}
