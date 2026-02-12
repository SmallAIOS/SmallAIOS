// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Alert types, severity levels, and configurable threshold system.

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AlertSeverity {
    /// Low severity: approaching threshold, informational.
    Low = 0,
    /// Medium severity: anomaly detected, operator attention needed.
    Medium = 1,
    /// High severity: security-relevant event (capability bypass, DoS).
    High = 2,
    /// Critical severity: system compromise or imminent failure.
    Critical = 3,
}

/// Type of security alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertType {
    /// Excessive capability denials for a single task.
    CapabilityDenialRate {
        task_id: u64,
        count: u32,
        window_secs: u32,
    },
    /// Memory allocation failure spike.
    MemoryAllocationFailure {
        region: AllocatorRegion,
        count: u32,
        window_secs: u32,
    },
    /// Inference latency outlier detected.
    InferenceLatencyAnomaly {
        task_id: u64,
        observed_ns: u64,
        p50_ns: u64,
        stddev_ns: u64,
    },
    /// Watchdog time remaining below threshold.
    WatchdogTimePressure { remaining_ns: u64, timeout_ns: u64 },
    /// SYN flood detection: excessive new connections.
    NetworkConnectionRate { rate_per_sec: u32, window_secs: u32 },
}

/// Memory allocator regions for failure tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AllocatorRegion {
    /// Buddy allocator (page-level).
    Buddy = 0,
    /// Slab allocator (small objects).
    Slab = 1,
    /// Tensor buffer allocator.
    Tensor = 2,
}

impl AllocatorRegion {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Buddy),
            1 => Some(Self::Slab),
            2 => Some(Self::Tensor),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Buddy => "buddy",
            Self::Slab => "slab",
            Self::Tensor => "tensor",
        }
    }
}

/// A generated security alert.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Nanosecond timestamp when the alert was generated.
    pub timestamp_ns: u64,
    /// Severity level.
    pub severity: AlertSeverity,
    /// Type-specific alert details.
    pub alert_type: AlertType,
}

/// Configuration for monitoring thresholds.
///
/// All thresholds are configurable at boot time. Invalid (zero or negative
/// equivalent) values are rejected and fall back to defaults.
#[derive(Debug, Clone, Copy)]
pub struct MonitoringConfig {
    /// Max capability denials per second per task before alert.
    pub cap_denial_threshold: u32,
    /// Max memory allocation failures per second before alert.
    pub mem_failure_threshold: u32,
    /// Sigma multiplier for inference latency anomaly detection.
    pub latency_sigma: u32,
    /// Minimum inference samples before alerting (warmup period).
    pub latency_min_samples: u32,
    /// Watchdog warning percentage (alert when remaining < this % of timeout).
    pub watchdog_warn_pct: u32,
    /// Max new network connections per second before SYN flood alert.
    pub net_conn_threshold: u32,
}

impl MonitoringConfig {
    /// Default configuration per spec.
    pub const fn default_config() -> Self {
        Self {
            cap_denial_threshold: 10,
            mem_failure_threshold: 10,
            latency_sigma: 3,
            latency_min_samples: 100,
            watchdog_warn_pct: 50,
            net_conn_threshold: 100,
        }
    }

    /// Validate and return config, replacing invalid values with defaults.
    pub fn validated(self) -> Self {
        let defaults = Self::default_config();
        Self {
            cap_denial_threshold: if self.cap_denial_threshold == 0 {
                defaults.cap_denial_threshold
            } else {
                self.cap_denial_threshold
            },
            mem_failure_threshold: if self.mem_failure_threshold == 0 {
                defaults.mem_failure_threshold
            } else {
                self.mem_failure_threshold
            },
            latency_sigma: if self.latency_sigma == 0 {
                defaults.latency_sigma
            } else {
                self.latency_sigma
            },
            latency_min_samples: if self.latency_min_samples == 0 {
                defaults.latency_min_samples
            } else {
                self.latency_min_samples
            },
            watchdog_warn_pct: if self.watchdog_warn_pct == 0 || self.watchdog_warn_pct > 100 {
                defaults.watchdog_warn_pct
            } else {
                self.watchdog_warn_pct
            },
            net_conn_threshold: if self.net_conn_threshold == 0 {
                defaults.net_conn_threshold
            } else {
                self.net_conn_threshold
            },
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = MonitoringConfig::default_config();
        assert_eq!(cfg.cap_denial_threshold, 10);
        assert_eq!(cfg.mem_failure_threshold, 10);
        assert_eq!(cfg.latency_sigma, 3);
        assert_eq!(cfg.latency_min_samples, 100);
        assert_eq!(cfg.watchdog_warn_pct, 50);
        assert_eq!(cfg.net_conn_threshold, 100);
    }

    #[test]
    fn validated_keeps_valid_values() {
        let cfg = MonitoringConfig {
            cap_denial_threshold: 20,
            mem_failure_threshold: 5,
            latency_sigma: 2,
            latency_min_samples: 50,
            watchdog_warn_pct: 75,
            net_conn_threshold: 200,
        };
        let v = cfg.validated();
        assert_eq!(v.cap_denial_threshold, 20);
        assert_eq!(v.mem_failure_threshold, 5);
        assert_eq!(v.latency_sigma, 2);
        assert_eq!(v.latency_min_samples, 50);
        assert_eq!(v.watchdog_warn_pct, 75);
        assert_eq!(v.net_conn_threshold, 200);
    }

    #[test]
    fn validated_replaces_zero_with_defaults() {
        let cfg = MonitoringConfig {
            cap_denial_threshold: 0,
            mem_failure_threshold: 0,
            latency_sigma: 0,
            latency_min_samples: 0,
            watchdog_warn_pct: 0,
            net_conn_threshold: 0,
        };
        let v = cfg.validated();
        let d = MonitoringConfig::default_config();
        assert_eq!(v.cap_denial_threshold, d.cap_denial_threshold);
        assert_eq!(v.mem_failure_threshold, d.mem_failure_threshold);
        assert_eq!(v.latency_sigma, d.latency_sigma);
        assert_eq!(v.latency_min_samples, d.latency_min_samples);
        assert_eq!(v.watchdog_warn_pct, d.watchdog_warn_pct);
        assert_eq!(v.net_conn_threshold, d.net_conn_threshold);
    }

    #[test]
    fn validated_rejects_invalid_watchdog_pct() {
        let cfg = MonitoringConfig {
            watchdog_warn_pct: 101,
            ..MonitoringConfig::default_config()
        };
        let v = cfg.validated();
        assert_eq!(v.watchdog_warn_pct, 50);
    }

    #[test]
    fn allocator_region_roundtrip() {
        assert_eq!(AllocatorRegion::from_u8(0), Some(AllocatorRegion::Buddy));
        assert_eq!(AllocatorRegion::from_u8(1), Some(AllocatorRegion::Slab));
        assert_eq!(AllocatorRegion::from_u8(2), Some(AllocatorRegion::Tensor));
        assert_eq!(AllocatorRegion::from_u8(3), None);
    }

    #[test]
    fn allocator_region_as_str() {
        assert_eq!(AllocatorRegion::Buddy.as_str(), "buddy");
        assert_eq!(AllocatorRegion::Slab.as_str(), "slab");
        assert_eq!(AllocatorRegion::Tensor.as_str(), "tensor");
    }

    #[test]
    fn alert_construction() {
        let alert = Alert {
            timestamp_ns: 1_000_000,
            severity: AlertSeverity::High,
            alert_type: AlertType::CapabilityDenialRate {
                task_id: 42,
                count: 15,
                window_secs: 1,
            },
        };
        assert_eq!(alert.severity, AlertSeverity::High);
    }
}
