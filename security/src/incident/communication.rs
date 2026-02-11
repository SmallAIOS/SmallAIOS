// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Incident communication procedures.
//!
//! Defines the notification matrix per severity level, escalation
//! timelines, and alert routing configuration.

use super::event::IncidentSeverity;

/// Communication channel types for incident notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NotificationChannel {
    /// Zenoh IPC topic (`smallaios/v1/incidents`).
    ZenohIpc = 0,
    /// Prometheus alertmanager webhook.
    AlertmanagerWebhook = 1,
    /// Syslog (RFC 5424) export.
    Syslog = 2,
    /// Audit log entry (always enabled).
    AuditLog = 3,
}

impl NotificationChannel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ZenohIpc => "zenoh-ipc",
            Self::AlertmanagerWebhook => "alertmanager-webhook",
            Self::Syslog => "syslog",
            Self::AuditLog => "audit-log",
        }
    }
}

/// Notification routing rule: which channels are used for each severity.
#[derive(Debug, Clone, Copy)]
pub struct NotificationRule {
    /// Minimum severity to activate this channel.
    pub min_severity: IncidentSeverity,
    /// Channel to notify.
    pub channel: NotificationChannel,
    /// Whether this notification is mandatory (cannot be disabled).
    pub mandatory: bool,
}

/// Default notification matrix per severity.
///
/// | Severity | Zenoh IPC | Alertmanager | Syslog | Audit Log |
/// |----------|-----------|-------------|--------|-----------|
/// | Critical | Yes | Yes | Yes | Yes (mandatory) |
/// | High | Yes | Yes | Yes | Yes (mandatory) |
/// | Medium | Yes | No | Yes | Yes (mandatory) |
/// | Low | No | No | No | Yes (mandatory) |
pub const NOTIFICATION_MATRIX: &[NotificationRule] = &[
    // Audit log is always mandatory for all severities
    NotificationRule {
        min_severity: IncidentSeverity::Low,
        channel: NotificationChannel::AuditLog,
        mandatory: true,
    },
    // Syslog for Medium and above
    NotificationRule {
        min_severity: IncidentSeverity::Medium,
        channel: NotificationChannel::Syslog,
        mandatory: false,
    },
    // Zenoh IPC for Medium and above
    NotificationRule {
        min_severity: IncidentSeverity::Medium,
        channel: NotificationChannel::ZenohIpc,
        mandatory: false,
    },
    // Alertmanager for High and above
    NotificationRule {
        min_severity: IncidentSeverity::High,
        channel: NotificationChannel::AlertmanagerWebhook,
        mandatory: false,
    },
];

/// Get the list of channels to notify for a given severity.
pub fn channels_for_severity(severity: IncidentSeverity) -> ChannelIter {
    ChannelIter {
        severity,
        pos: 0,
    }
}

/// Iterator over notification channels for a severity level.
pub struct ChannelIter {
    severity: IncidentSeverity,
    pos: usize,
}

impl Iterator for ChannelIter {
    type Item = &'static NotificationRule;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < NOTIFICATION_MATRIX.len() {
            let rule = &NOTIFICATION_MATRIX[self.pos];
            self.pos += 1;
            if self.severity >= rule.min_severity {
                return Some(rule);
            }
        }
        None
    }
}

/// Escalation timeline configuration.
#[derive(Debug, Clone, Copy)]
pub struct EscalationTimeline {
    /// Severity level.
    pub severity: IncidentSeverity,
    /// Time to first notification (nanoseconds).
    pub initial_notification_ns: u64,
    /// Re-notification interval if unacknowledged (nanoseconds).
    pub re_notify_interval_ns: u64,
    /// Auto-escalation if unacknowledged after this period (nanoseconds, 0 = never).
    pub auto_escalate_ns: u64,
}

/// Nanoseconds per minute.
const NS_PER_MINUTE: u64 = 60_000_000_000;

/// Default escalation timelines.
pub const ESCALATION_TIMELINES: &[EscalationTimeline] = &[
    EscalationTimeline {
        severity: IncidentSeverity::Critical,
        initial_notification_ns: 0,                  // Immediate
        re_notify_interval_ns: 5 * NS_PER_MINUTE,   // Every 5 minutes
        auto_escalate_ns: 0,                         // Already highest
    },
    EscalationTimeline {
        severity: IncidentSeverity::High,
        initial_notification_ns: 0,                  // Immediate
        re_notify_interval_ns: 15 * NS_PER_MINUTE,  // Every 15 minutes
        auto_escalate_ns: 60 * NS_PER_MINUTE,       // Escalate to Critical after 1 hour
    },
    EscalationTimeline {
        severity: IncidentSeverity::Medium,
        initial_notification_ns: NS_PER_MINUTE,      // 1 minute delay
        re_notify_interval_ns: 60 * NS_PER_MINUTE,  // Hourly
        auto_escalate_ns: 4 * 60 * NS_PER_MINUTE,   // Escalate to High after 4 hours
    },
    EscalationTimeline {
        severity: IncidentSeverity::Low,
        initial_notification_ns: 5 * NS_PER_MINUTE,  // 5 minute delay
        re_notify_interval_ns: 0,                    // No re-notification
        auto_escalate_ns: 24 * 60 * NS_PER_MINUTE,  // Escalate to Medium after 24 hours
    },
];

/// Look up the escalation timeline for a severity.
pub fn timeline_for_severity(severity: IncidentSeverity) -> Option<&'static EscalationTimeline> {
    ESCALATION_TIMELINES.iter().find(|t| t.severity == severity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_uses_all_channels() {
        let channels: alloc::vec::Vec<_> = channels_for_severity(IncidentSeverity::Critical).collect();
        assert_eq!(channels.len(), 4);
        let channel_types: alloc::vec::Vec<_> = channels.iter().map(|r| r.channel).collect();
        assert!(channel_types.contains(&NotificationChannel::AuditLog));
        assert!(channel_types.contains(&NotificationChannel::ZenohIpc));
        assert!(channel_types.contains(&NotificationChannel::Syslog));
        assert!(channel_types.contains(&NotificationChannel::AlertmanagerWebhook));
    }

    #[test]
    fn high_uses_most_channels() {
        let channels: alloc::vec::Vec<_> = channels_for_severity(IncidentSeverity::High).collect();
        assert_eq!(channels.len(), 4);
    }

    #[test]
    fn medium_skips_alertmanager() {
        let channels: alloc::vec::Vec<_> = channels_for_severity(IncidentSeverity::Medium).collect();
        assert_eq!(channels.len(), 3);
        let channel_types: alloc::vec::Vec<_> = channels.iter().map(|r| r.channel).collect();
        assert!(!channel_types.contains(&NotificationChannel::AlertmanagerWebhook));
    }

    #[test]
    fn low_only_audit_log() {
        let channels: alloc::vec::Vec<_> = channels_for_severity(IncidentSeverity::Low).collect();
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].channel, NotificationChannel::AuditLog);
        assert!(channels[0].mandatory);
    }

    #[test]
    fn audit_log_always_mandatory() {
        for severity_val in 0..=3u8 {
            let severity = IncidentSeverity::from_u8(severity_val).unwrap();
            let channels: alloc::vec::Vec<_> = channels_for_severity(severity).collect();
            let audit = channels.iter().find(|r| r.channel == NotificationChannel::AuditLog);
            assert!(audit.is_some(), "AuditLog missing for {:?}", severity);
            assert!(audit.unwrap().mandatory);
        }
    }

    #[test]
    fn channel_strings() {
        assert_eq!(NotificationChannel::ZenohIpc.as_str(), "zenoh-ipc");
        assert_eq!(NotificationChannel::AlertmanagerWebhook.as_str(), "alertmanager-webhook");
        assert_eq!(NotificationChannel::Syslog.as_str(), "syslog");
        assert_eq!(NotificationChannel::AuditLog.as_str(), "audit-log");
    }

    #[test]
    fn escalation_timelines_exist() {
        assert_eq!(ESCALATION_TIMELINES.len(), 4);
    }

    #[test]
    fn critical_immediate_notification() {
        let t = timeline_for_severity(IncidentSeverity::Critical).unwrap();
        assert_eq!(t.initial_notification_ns, 0);
        assert_eq!(t.auto_escalate_ns, 0); // Already highest
    }

    #[test]
    fn high_escalates_to_critical() {
        let t = timeline_for_severity(IncidentSeverity::High).unwrap();
        assert_eq!(t.auto_escalate_ns, 60 * NS_PER_MINUTE); // 1 hour
    }

    #[test]
    fn medium_escalates_to_high() {
        let t = timeline_for_severity(IncidentSeverity::Medium).unwrap();
        assert_eq!(t.auto_escalate_ns, 4 * 60 * NS_PER_MINUTE); // 4 hours
    }

    #[test]
    fn low_escalates_to_medium() {
        let t = timeline_for_severity(IncidentSeverity::Low).unwrap();
        assert_eq!(t.auto_escalate_ns, 24 * 60 * NS_PER_MINUTE); // 24 hours
    }
}
