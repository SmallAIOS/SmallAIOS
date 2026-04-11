// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Configurable audit log retention policy.
//!
//! Retention periods per deployment class:
//! - Edge: 7 days
//! - Datacenter: 90 days
//! - Safety-critical: 365 days (1 year)

/// Nanoseconds per day.
const NS_PER_DAY: u64 = 86_400_000_000_000;

/// Deployment classes with different retention requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DeploymentClass {
    /// Edge device (RPi, Jetson).
    Edge = 0,
    /// Datacenter / cloud.
    Datacenter = 1,
    /// Safety-critical (automotive, avionics, industrial).
    SafetyCritical = 2,
}

impl DeploymentClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Edge => "edge",
            Self::Datacenter => "datacenter",
            Self::SafetyCritical => "safety-critical",
        }
    }

    /// Default retention period in days for this deployment class.
    pub fn retention_days(&self) -> u32 {
        match self {
            Self::Edge => 7,
            Self::Datacenter => 90,
            Self::SafetyCritical => 365,
        }
    }

    /// Default retention period in nanoseconds.
    pub fn retention_ns(&self) -> u64 {
        self.retention_days() as u64 * NS_PER_DAY
    }
}

/// Retention policy configuration.
#[derive(Debug, Clone, Copy)]
pub struct RetentionPolicy {
    /// Deployment class.
    pub deployment: DeploymentClass,
    /// Retention period in nanoseconds.
    pub retention_ns: u64,
    /// Maximum number of batches to retain (0 = unlimited).
    pub max_batches: u64,
}

impl RetentionPolicy {
    /// Create a policy with default retention for the deployment class.
    pub fn for_deployment(deployment: DeploymentClass) -> Self {
        Self {
            deployment,
            retention_ns: deployment.retention_ns(),
            max_batches: 0,
        }
    }

    /// Override the retention period (in days).
    pub fn with_retention_days(mut self, days: u32) -> Self {
        self.retention_ns = days as u64 * NS_PER_DAY;
        self
    }

    /// Set a maximum batch count.
    pub fn with_max_batches(mut self, max: u64) -> Self {
        self.max_batches = max;
        self
    }

    /// Check whether a batch should be retained given its timestamp and the current time.
    pub fn should_retain(&self, batch_timestamp_ns: u64, now_ns: u64) -> bool {
        let age = now_ns.saturating_sub(batch_timestamp_ns);
        age < self.retention_ns
    }

    /// Determine how many batches to prune from a list of batch timestamps.
    ///
    /// Assumes timestamps are in chronological order (oldest first).
    /// Returns the number of batches from the front that should be pruned.
    pub fn batches_to_prune(
        &self,
        batch_timestamps: &[u64],
        now_ns: u64,
        total_count: u64,
    ) -> usize {
        let mut prune_count = 0;

        // Time-based pruning
        for &ts in batch_timestamps {
            if !self.should_retain(ts, now_ns) {
                prune_count += 1;
            } else {
                break; // Timestamps are ordered, so remaining are newer
            }
        }

        // Count-based pruning
        if self.max_batches > 0 && total_count > self.max_batches {
            let excess = (total_count - self.max_batches) as usize;
            if excess > prune_count {
                prune_count = excess;
            }
        }

        prune_count.min(batch_timestamps.len())
    }
}

/// Retention statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct RetentionStats {
    /// Total batches pruned since boot.
    pub total_pruned: u64,
    /// Total bytes reclaimed.
    pub bytes_reclaimed: u64,
    /// Last prune timestamp.
    pub last_prune_ns: u64,
}

impl RetentionStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a prune event.
    pub fn record_prune(&mut self, batches_pruned: u64, bytes: u64, now_ns: u64) {
        self.total_pruned += batches_pruned;
        self.bytes_reclaimed += bytes;
        self.last_prune_ns = now_ns;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deployment_retention_days() {
        assert_eq!(DeploymentClass::Edge.retention_days(), 7);
        assert_eq!(DeploymentClass::Datacenter.retention_days(), 90);
        assert_eq!(DeploymentClass::SafetyCritical.retention_days(), 365);
    }

    #[test]
    fn deployment_retention_ns() {
        assert_eq!(DeploymentClass::Edge.retention_ns(), 7 * NS_PER_DAY);
        assert_eq!(DeploymentClass::Datacenter.retention_ns(), 90 * NS_PER_DAY);
        assert_eq!(
            DeploymentClass::SafetyCritical.retention_ns(),
            365 * NS_PER_DAY
        );
    }

    #[test]
    fn deployment_strings() {
        assert_eq!(DeploymentClass::Edge.as_str(), "edge");
        assert_eq!(DeploymentClass::Datacenter.as_str(), "datacenter");
        assert_eq!(DeploymentClass::SafetyCritical.as_str(), "safety-critical");
    }

    #[test]
    fn policy_for_edge() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        assert_eq!(policy.retention_ns, 7 * NS_PER_DAY);
        assert_eq!(policy.max_batches, 0);
    }

    #[test]
    fn policy_custom_retention() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge).with_retention_days(14);
        assert_eq!(policy.retention_ns, 14 * NS_PER_DAY);
    }

    #[test]
    fn should_retain_within_window() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        let now = 10 * NS_PER_DAY;
        let batch_ts = 5 * NS_PER_DAY;
        assert!(policy.should_retain(batch_ts, now)); // 5 days old < 7 day limit
    }

    #[test]
    fn should_not_retain_past_window() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        let now = 10 * NS_PER_DAY;
        let batch_ts = NS_PER_DAY;
        assert!(!policy.should_retain(batch_ts, now)); // 9 days old > 7 day limit
    }

    #[test]
    fn prune_time_based() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        let now = 20 * NS_PER_DAY;
        // Batches at day 1, 5, 10, 15, 18
        let timestamps = [
            NS_PER_DAY,
            5 * NS_PER_DAY,
            10 * NS_PER_DAY,
            15 * NS_PER_DAY,
            18 * NS_PER_DAY,
        ];
        let prune = policy.batches_to_prune(&timestamps, now, 5);
        // Day 1: 19 days old -> prune
        // Day 5: 15 days old -> prune
        // Day 10: 10 days old -> prune
        // Day 15: 5 days old -> retain
        // Day 18: 2 days old -> retain
        assert_eq!(prune, 3);
    }

    #[test]
    fn prune_count_based() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge).with_max_batches(3);
        let now = 5 * NS_PER_DAY;
        // All within retention window, but too many
        let timestamps = [NS_PER_DAY, 2 * NS_PER_DAY, 3 * NS_PER_DAY, 4 * NS_PER_DAY];
        let prune = policy.batches_to_prune(&timestamps, now, 4);
        assert_eq!(prune, 1); // 4 - 3 = 1 excess
    }

    #[test]
    fn prune_empty() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::Edge);
        let prune = policy.batches_to_prune(&[], 0, 0);
        assert_eq!(prune, 0);
    }

    #[test]
    fn retention_stats() {
        let mut stats = RetentionStats::new();
        stats.record_prune(5, 4096, 1000);
        assert_eq!(stats.total_pruned, 5);
        assert_eq!(stats.bytes_reclaimed, 4096);
        assert_eq!(stats.last_prune_ns, 1000);

        stats.record_prune(3, 2048, 2000);
        assert_eq!(stats.total_pruned, 8);
        assert_eq!(stats.bytes_reclaimed, 6144);
    }

    #[test]
    fn safety_critical_long_retention() {
        let policy = RetentionPolicy::for_deployment(DeploymentClass::SafetyCritical);
        let now = 200 * NS_PER_DAY;
        let batch_ts = 100 * NS_PER_DAY;
        assert!(policy.should_retain(batch_ts, now)); // 100 days < 365 days
    }
}
