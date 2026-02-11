// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Continuous security monitoring subsystem.
//!
//! Provides statistical anomaly detection using configurable thresholds:
//! - Capability denial rate tracking (per-task, per-second)
//! - Memory allocation failure rate tracking (per-region)
//! - Inference latency anomaly detection (rolling percentiles, 3-sigma)
//! - Watchdog time-remaining tracking
//! - Network connection rate tracking (SYN flood detection)
//!
//! All detection uses arithmetic (moving average, standard deviation).
//! No machine learning models are used per design decision #4.

pub mod alerts;
pub mod ipc_export;
pub mod latency_stats;
pub mod prometheus;
pub mod rate_tracker;
pub mod threshold_reload;
pub mod watchdog_monitor;

pub use alerts::{Alert, AlertSeverity, MonitoringConfig};
pub use latency_stats::LatencyTracker;
pub use rate_tracker::RateTracker;
pub use watchdog_monitor::WatchdogMonitor;
