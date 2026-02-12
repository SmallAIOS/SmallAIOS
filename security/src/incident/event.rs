// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Incident event types for structured alerting and Zenoh publishing.
//!
//! Events are published on `smallaios/v1/incidents` with severity,
//! description, timestamp, and affected resource identifiers.

use alloc::vec::Vec;
use core::fmt;

/// Incident severity levels per NIST SP 800-61 classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum IncidentSeverity {
    /// Configuration warning, threshold approach.
    Low = 0,
    /// Anomaly detection trigger, failed authentication.
    Medium = 1,
    /// Capability bypass, denial of service.
    High = 2,
    /// System compromise, data breach.
    Critical = 3,
}

impl IncidentSeverity {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Low),
            1 => Some(Self::Medium),
            2 => Some(Self::High),
            3 => Some(Self::Critical),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// Type of incident event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncidentType {
    /// Monitoring threshold exceeded (capability denials, memory failures, etc.).
    AnomalyDetected,
    /// Authentication failure (TLS handshake, etc.).
    AuthenticationFailure,
    /// Capability bypass attempt.
    CapabilityBypass,
    /// Denial of service detected.
    DenialOfService,
    /// Containment action executed.
    ContainmentAction,
    /// Severity escalation.
    SeverityEscalation,
    /// System compromise indicator.
    SystemCompromise,
    /// Configuration issue.
    ConfigurationWarning,
}

impl IncidentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AnomalyDetected => "anomaly-detected",
            Self::AuthenticationFailure => "auth-failure",
            Self::CapabilityBypass => "capability-bypass",
            Self::DenialOfService => "denial-of-service",
            Self::ContainmentAction => "containment-action",
            Self::SeverityEscalation => "severity-escalation",
            Self::SystemCompromise => "system-compromise",
            Self::ConfigurationWarning => "config-warning",
        }
    }
}

/// A structured incident event for Zenoh publishing.
#[derive(Debug, Clone)]
pub struct IncidentEvent {
    /// Unique incident identifier (monotonically increasing).
    pub incident_id: u64,
    /// Nanosecond-precision timestamp.
    pub timestamp_ns: u64,
    /// Severity level.
    pub severity: IncidentSeverity,
    /// Previous severity (for escalation events).
    pub previous_severity: Option<IncidentSeverity>,
    /// Type of incident.
    pub incident_type: IncidentType,
    /// Affected task ID (0 = system-wide).
    pub task_id: u64,
    /// Affected resource identifiers.
    pub affected_resources: Vec<u64>,
    /// Human-readable description (fixed-size inline).
    pub description: IncidentDescription,
}

/// Fixed-size inline description for incident events.
#[derive(Clone)]
pub struct IncidentDescription {
    buf: [u8; 128],
    len: u8,
}

impl IncidentDescription {
    pub fn new(s: &str) -> Self {
        let bytes = s.as_bytes();
        let len = bytes.len().min(128);
        let mut buf = [0u8; 128];
        buf[..len].copy_from_slice(&bytes[..len]);
        Self {
            buf,
            len: len as u8,
        }
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len as usize]).unwrap_or("")
    }
}

impl fmt::Debug for IncidentDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.as_str())
    }
}

/// Generator for incident events with auto-incrementing IDs.
pub struct IncidentEventGenerator {
    next_id: u64,
}

impl IncidentEventGenerator {
    pub fn new() -> Self {
        Self { next_id: 1 }
    }

    /// Create a new incident event.
    pub fn create_event(
        &mut self,
        timestamp_ns: u64,
        severity: IncidentSeverity,
        incident_type: IncidentType,
        task_id: u64,
        description: &str,
    ) -> IncidentEvent {
        let id = self.next_id;
        self.next_id += 1;
        IncidentEvent {
            incident_id: id,
            timestamp_ns,
            severity,
            previous_severity: None,
            incident_type,
            task_id,
            affected_resources: Vec::new(),
            description: IncidentDescription::new(description),
        }
    }

    /// Create a severity escalation event.
    pub fn create_escalation(
        &mut self,
        timestamp_ns: u64,
        original_id: u64,
        previous_severity: IncidentSeverity,
        new_severity: IncidentSeverity,
        task_id: u64,
        reason: &str,
    ) -> IncidentEvent {
        let id = self.next_id;
        self.next_id += 1;
        IncidentEvent {
            incident_id: id,
            timestamp_ns,
            severity: new_severity,
            previous_severity: Some(previous_severity),
            incident_type: IncidentType::SeverityEscalation,
            task_id,
            affected_resources: alloc::vec![original_id],
            description: IncidentDescription::new(reason),
        }
    }

    /// Number of events generated.
    pub fn events_generated(&self) -> u64 {
        self.next_id - 1
    }
}

impl Default for IncidentEventGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Serialized size of an incident event header (for Zenoh transport).
///
/// Layout:
/// - [0..8]   incident_id (u64 BE)
/// - [8..16]  timestamp_ns (u64 BE)
/// - [16]     severity (u8)
/// - [17]     previous_severity (u8, 0xFF = none)
/// - [18]     incident_type (u8)
/// - [19..27] task_id (u64 BE)
/// - [27]     num_affected_resources (u8)
/// - [28..29] description_len (u16 BE)
pub const EVENT_HEADER_SIZE: usize = 30;

impl IncidentEvent {
    /// Serialize the event header to bytes.
    pub fn serialize_header(&self, buf: &mut [u8; EVENT_HEADER_SIZE]) {
        buf[0..8].copy_from_slice(&self.incident_id.to_be_bytes());
        buf[8..16].copy_from_slice(&self.timestamp_ns.to_be_bytes());
        buf[16] = self.severity as u8;
        buf[17] = self.previous_severity.map(|s| s as u8).unwrap_or(0xFF);
        buf[18] = self.incident_type as u8;
        buf[19..27].copy_from_slice(&self.task_id.to_be_bytes());
        buf[27] = self.affected_resources.len().min(255) as u8;
        let desc_len = self.description.len as u16;
        buf[28..30].copy_from_slice(&desc_len.to_be_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_ordering() {
        assert!(IncidentSeverity::Low < IncidentSeverity::Medium);
        assert!(IncidentSeverity::Medium < IncidentSeverity::High);
        assert!(IncidentSeverity::High < IncidentSeverity::Critical);
    }

    #[test]
    fn severity_roundtrip() {
        for i in 0..=3u8 {
            let s = IncidentSeverity::from_u8(i).unwrap();
            assert_eq!(s as u8, i);
        }
        assert!(IncidentSeverity::from_u8(4).is_none());
    }

    #[test]
    fn severity_as_str() {
        assert_eq!(IncidentSeverity::Critical.as_str(), "critical");
        assert_eq!(IncidentSeverity::High.as_str(), "high");
        assert_eq!(IncidentSeverity::Medium.as_str(), "medium");
        assert_eq!(IncidentSeverity::Low.as_str(), "low");
    }

    #[test]
    fn description_basic() {
        let desc = IncidentDescription::new("test incident");
        assert_eq!(desc.as_str(), "test incident");
    }

    #[test]
    fn description_truncation() {
        let long = "x".repeat(200);
        let desc = IncidentDescription::new(&long);
        assert_eq!(desc.as_str().len(), 128);
    }

    #[test]
    fn event_generator_ids() {
        let mut gen = IncidentEventGenerator::new();
        let e1 = gen.create_event(
            0,
            IncidentSeverity::Low,
            IncidentType::AnomalyDetected,
            1,
            "test",
        );
        let e2 = gen.create_event(
            1,
            IncidentSeverity::High,
            IncidentType::DenialOfService,
            2,
            "dos",
        );
        assert_eq!(e1.incident_id, 1);
        assert_eq!(e2.incident_id, 2);
        assert_eq!(gen.events_generated(), 2);
    }

    #[test]
    fn event_escalation() {
        let mut gen = IncidentEventGenerator::new();
        let e1 = gen.create_event(
            0,
            IncidentSeverity::Medium,
            IncidentType::AnomalyDetected,
            1,
            "anomaly",
        );
        let esc = gen.create_escalation(
            100,
            e1.incident_id,
            IncidentSeverity::Medium,
            IncidentSeverity::High,
            1,
            "impact greater than expected",
        );
        assert_eq!(esc.severity, IncidentSeverity::High);
        assert_eq!(esc.previous_severity, Some(IncidentSeverity::Medium));
        assert_eq!(esc.incident_type, IncidentType::SeverityEscalation);
        assert_eq!(esc.affected_resources, alloc::vec![1]);
    }

    #[test]
    fn serialize_header() {
        let mut gen = IncidentEventGenerator::new();
        let event = gen.create_event(
            1_000_000_000,
            IncidentSeverity::High,
            IncidentType::CapabilityBypass,
            42,
            "bypass detected",
        );
        let mut buf = [0u8; EVENT_HEADER_SIZE];
        event.serialize_header(&mut buf);

        // Verify incident_id = 1
        assert_eq!(u64::from_be_bytes(buf[0..8].try_into().unwrap()), 1);
        // Verify timestamp
        assert_eq!(
            u64::from_be_bytes(buf[8..16].try_into().unwrap()),
            1_000_000_000
        );
        // Verify severity = High(2)
        assert_eq!(buf[16], 2);
        // Verify no previous severity
        assert_eq!(buf[17], 0xFF);
        // Verify task_id = 42
        assert_eq!(u64::from_be_bytes(buf[19..27].try_into().unwrap()), 42);
    }

    #[test]
    fn incident_type_as_str() {
        assert_eq!(IncidentType::AnomalyDetected.as_str(), "anomaly-detected");
        assert_eq!(
            IncidentType::ContainmentAction.as_str(),
            "containment-action"
        );
        assert_eq!(IncidentType::SystemCompromise.as_str(), "system-compromise");
    }
}
