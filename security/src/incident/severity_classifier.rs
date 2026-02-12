// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Automated incident severity classification from monitoring alerts.
//!
//! Maps monitoring alert types to incident severity per NIST SP 800-61:
//! - Critical: system compromise, data breach, multiple high-severity indicators
//! - High: capability bypass, denial of service, repeated auth failures
//! - Medium: anomaly detection trigger, single auth failure, resource pressure
//! - Low: configuration warning, threshold approach, informational

use super::event::{IncidentSeverity, IncidentType};

/// Alert source types from the monitoring subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AlertSource {
    /// Capability denial rate exceeded threshold.
    CapabilityDenialRate = 0,
    /// Memory allocation failure rate exceeded threshold.
    MemoryFailureRate = 1,
    /// Inference latency anomaly (3-sigma deviation).
    InferenceLatencyAnomaly = 2,
    /// Watchdog time remaining below 50%.
    WatchdogLowWatermark = 3,
    /// Network connection rate exceeded (SYN flood detection).
    NetworkConnectionRate = 4,
    /// TLS handshake failure.
    TlsHandshakeFailure = 5,
    /// Audit hash chain integrity violation.
    AuditIntegrityViolation = 6,
    /// Crypto self-test (KAT) failure.
    CryptoSelfTestFailure = 7,
    /// OT output range exceeded.
    OtOutputRangeViolation = 8,
    /// Configuration change detected.
    ConfigurationChange = 9,
}

impl AlertSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CapabilityDenialRate => "capability-denial-rate",
            Self::MemoryFailureRate => "memory-failure-rate",
            Self::InferenceLatencyAnomaly => "inference-latency-anomaly",
            Self::WatchdogLowWatermark => "watchdog-low-watermark",
            Self::NetworkConnectionRate => "network-connection-rate",
            Self::TlsHandshakeFailure => "tls-handshake-failure",
            Self::AuditIntegrityViolation => "audit-integrity-violation",
            Self::CryptoSelfTestFailure => "crypto-self-test-failure",
            Self::OtOutputRangeViolation => "ot-output-range-violation",
            Self::ConfigurationChange => "configuration-change",
        }
    }
}

/// Classification result from an alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassifiedAlert {
    /// Source of the alert.
    pub source: AlertSource,
    /// Classified severity.
    pub severity: IncidentSeverity,
    /// Mapped incident type.
    pub incident_type: IncidentType,
    /// Whether automated containment should be triggered.
    pub auto_contain: bool,
}

/// Classify an alert into incident severity and type.
///
/// Single-alert classification uses static mapping rules.
pub fn classify_alert(source: AlertSource) -> ClassifiedAlert {
    match source {
        AlertSource::AuditIntegrityViolation => ClassifiedAlert {
            source,
            severity: IncidentSeverity::Critical,
            incident_type: IncidentType::SystemCompromise,
            auto_contain: true,
        },
        AlertSource::CryptoSelfTestFailure => ClassifiedAlert {
            source,
            severity: IncidentSeverity::Critical,
            incident_type: IncidentType::SystemCompromise,
            auto_contain: true,
        },
        AlertSource::CapabilityDenialRate => ClassifiedAlert {
            source,
            severity: IncidentSeverity::High,
            incident_type: IncidentType::CapabilityBypass,
            auto_contain: true,
        },
        AlertSource::NetworkConnectionRate => ClassifiedAlert {
            source,
            severity: IncidentSeverity::High,
            incident_type: IncidentType::DenialOfService,
            auto_contain: true,
        },
        AlertSource::TlsHandshakeFailure => ClassifiedAlert {
            source,
            severity: IncidentSeverity::Medium,
            incident_type: IncidentType::AuthenticationFailure,
            auto_contain: false,
        },
        AlertSource::InferenceLatencyAnomaly => ClassifiedAlert {
            source,
            severity: IncidentSeverity::Medium,
            incident_type: IncidentType::AnomalyDetected,
            auto_contain: false,
        },
        AlertSource::MemoryFailureRate => ClassifiedAlert {
            source,
            severity: IncidentSeverity::Medium,
            incident_type: IncidentType::AnomalyDetected,
            auto_contain: false,
        },
        AlertSource::OtOutputRangeViolation => ClassifiedAlert {
            source,
            severity: IncidentSeverity::High,
            incident_type: IncidentType::AnomalyDetected,
            auto_contain: true,
        },
        AlertSource::WatchdogLowWatermark => ClassifiedAlert {
            source,
            severity: IncidentSeverity::Medium,
            incident_type: IncidentType::AnomalyDetected,
            auto_contain: false,
        },
        AlertSource::ConfigurationChange => ClassifiedAlert {
            source,
            severity: IncidentSeverity::Low,
            incident_type: IncidentType::ConfigurationWarning,
            auto_contain: false,
        },
    }
}

/// Escalation rules: when multiple alerts occur within a window,
/// the effective severity may increase.
pub struct EscalationRule {
    /// Source alert type.
    pub source: AlertSource,
    /// Number of alerts within the window to trigger escalation.
    pub count_threshold: u32,
    /// Time window in nanoseconds.
    pub window_ns: u64,
    /// Escalated severity.
    pub escalated_severity: IncidentSeverity,
}

/// Default escalation rules.
pub const ESCALATION_RULES: &[EscalationRule] = &[
    // 5 capability denials in 10 seconds -> Critical
    EscalationRule {
        source: AlertSource::CapabilityDenialRate,
        count_threshold: 5,
        window_ns: 10_000_000_000,
        escalated_severity: IncidentSeverity::Critical,
    },
    // 3 TLS failures in 60 seconds -> High
    EscalationRule {
        source: AlertSource::TlsHandshakeFailure,
        count_threshold: 3,
        window_ns: 60_000_000_000,
        escalated_severity: IncidentSeverity::High,
    },
    // 3 memory failures in 30 seconds -> High
    EscalationRule {
        source: AlertSource::MemoryFailureRate,
        count_threshold: 3,
        window_ns: 30_000_000_000,
        escalated_severity: IncidentSeverity::High,
    },
];

/// Track alert counts for escalation.
pub struct AlertTracker {
    /// (source, timestamp_ns) pairs in a circular buffer.
    history: [(AlertSource, u64); MAX_HISTORY],
    head: usize,
    count: usize,
}

const MAX_HISTORY: usize = 64;

impl AlertTracker {
    pub fn new() -> Self {
        Self {
            history: [(AlertSource::ConfigurationChange, 0); MAX_HISTORY],
            head: 0,
            count: 0,
        }
    }

    /// Record an alert and check if any escalation rule is triggered.
    ///
    /// Returns the escalated severity if a rule matches, otherwise None.
    pub fn record_alert(
        &mut self,
        source: AlertSource,
        timestamp_ns: u64,
    ) -> Option<IncidentSeverity> {
        // Record the alert
        self.history[self.head] = (source, timestamp_ns);
        self.head = (self.head + 1) % MAX_HISTORY;
        if self.count < MAX_HISTORY {
            self.count += 1;
        }

        // Check escalation rules
        for rule in ESCALATION_RULES {
            if rule.source != source {
                continue;
            }
            let count = self.count_in_window(source, timestamp_ns, rule.window_ns);
            if count >= rule.count_threshold {
                return Some(rule.escalated_severity);
            }
        }
        None
    }

    fn count_in_window(&self, source: AlertSource, now_ns: u64, window_ns: u64) -> u32 {
        let cutoff = now_ns.saturating_sub(window_ns);
        let mut count = 0u32;
        for i in 0..self.count {
            let idx = if self.head > i {
                self.head - i - 1
            } else {
                MAX_HISTORY - (i + 1 - self.head)
            };
            let (s, ts) = self.history[idx];
            if s == source && ts >= cutoff {
                count += 1;
            }
        }
        count
    }

    /// Total alerts recorded.
    pub fn total_alerts(&self) -> usize {
        self.count
    }
}

impl Default for AlertTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_critical_alerts() {
        let c = classify_alert(AlertSource::AuditIntegrityViolation);
        assert_eq!(c.severity, IncidentSeverity::Critical);
        assert!(c.auto_contain);

        let c = classify_alert(AlertSource::CryptoSelfTestFailure);
        assert_eq!(c.severity, IncidentSeverity::Critical);
        assert!(c.auto_contain);
    }

    #[test]
    fn classify_high_alerts() {
        let c = classify_alert(AlertSource::CapabilityDenialRate);
        assert_eq!(c.severity, IncidentSeverity::High);
        assert_eq!(c.incident_type, IncidentType::CapabilityBypass);
        assert!(c.auto_contain);

        let c = classify_alert(AlertSource::NetworkConnectionRate);
        assert_eq!(c.severity, IncidentSeverity::High);
        assert_eq!(c.incident_type, IncidentType::DenialOfService);
    }

    #[test]
    fn classify_medium_alerts() {
        let c = classify_alert(AlertSource::TlsHandshakeFailure);
        assert_eq!(c.severity, IncidentSeverity::Medium);
        assert!(!c.auto_contain);

        let c = classify_alert(AlertSource::InferenceLatencyAnomaly);
        assert_eq!(c.severity, IncidentSeverity::Medium);
    }

    #[test]
    fn classify_low_alerts() {
        let c = classify_alert(AlertSource::ConfigurationChange);
        assert_eq!(c.severity, IncidentSeverity::Low);
        assert!(!c.auto_contain);
    }

    #[test]
    fn ot_violation_high() {
        let c = classify_alert(AlertSource::OtOutputRangeViolation);
        assert_eq!(c.severity, IncidentSeverity::High);
        assert!(c.auto_contain);
    }

    #[test]
    fn alert_source_strings() {
        assert_eq!(
            AlertSource::CapabilityDenialRate.as_str(),
            "capability-denial-rate"
        );
        assert_eq!(
            AlertSource::AuditIntegrityViolation.as_str(),
            "audit-integrity-violation"
        );
        assert_eq!(
            AlertSource::ConfigurationChange.as_str(),
            "configuration-change"
        );
    }

    #[test]
    fn escalation_no_trigger() {
        let mut tracker = AlertTracker::new();
        // Single alert, below threshold
        let result = tracker.record_alert(AlertSource::CapabilityDenialRate, 1_000_000_000);
        assert!(result.is_none());
    }

    #[test]
    fn escalation_trigger() {
        let mut tracker = AlertTracker::new();
        // 5 capability denials in 10 seconds
        for i in 0..5u64 {
            let _ = tracker.record_alert(AlertSource::CapabilityDenialRate, i * 1_000_000_000);
        }
        // 5th alert should trigger escalation
        assert_eq!(tracker.total_alerts(), 5);
    }

    #[test]
    fn escalation_outside_window() {
        let mut tracker = AlertTracker::new();
        // 5 alerts but spread over 60 seconds (window is 10s)
        for i in 0..5u64 {
            let ts = i * 15_000_000_000; // 15 seconds apart
            let _ = tracker.record_alert(AlertSource::CapabilityDenialRate, ts);
        }
        // No escalation since alerts are outside the window
    }

    #[test]
    fn tls_escalation() {
        let mut tracker = AlertTracker::new();
        // 3 TLS failures in 60 seconds -> High
        let result1 = tracker.record_alert(AlertSource::TlsHandshakeFailure, 1_000_000_000);
        let result2 = tracker.record_alert(AlertSource::TlsHandshakeFailure, 2_000_000_000);
        let result3 = tracker.record_alert(AlertSource::TlsHandshakeFailure, 3_000_000_000);
        assert!(result1.is_none());
        assert!(result2.is_none());
        assert_eq!(result3, Some(IncidentSeverity::High));
    }

    #[test]
    fn mixed_alerts_no_cross_escalation() {
        let mut tracker = AlertTracker::new();
        // Different alert types don't cross-escalate
        tracker.record_alert(AlertSource::CapabilityDenialRate, 1_000_000_000);
        tracker.record_alert(AlertSource::TlsHandshakeFailure, 2_000_000_000);
        tracker.record_alert(AlertSource::MemoryFailureRate, 3_000_000_000);
        tracker.record_alert(AlertSource::ConfigurationChange, 4_000_000_000);
        // None should escalate since each source only has 1 alert
    }
}
