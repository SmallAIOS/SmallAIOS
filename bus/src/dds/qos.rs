// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DDS QoS (Quality of Service) policies.
//!
//! Defines reliability, durability, history, deadline, liveliness, and
//! ownership QoS policies with compatibility checking.

/// Reliability QoS kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReliabilityKind {
    /// Best-effort: samples may be lost.
    BestEffort,
    /// Reliable: guaranteed delivery with retransmission.
    Reliable,
}

/// Durability QoS kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityKind {
    /// Volatile: no persistence of samples.
    Volatile,
    /// Transient-local: retain samples for late joiners.
    TransientLocal,
}

/// History QoS kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryKind {
    /// Keep last N samples.
    KeepLast,
    /// Keep all samples (bounded by resource limits).
    KeepAll,
}

/// Liveliness QoS kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LivelinessKind {
    /// Automatic: DDS asserts liveliness on behalf of the writer.
    Automatic,
    /// Manual by participant: application must assert periodically.
    ManualByParticipant,
    /// Manual by topic: application must assert per-topic.
    ManualByTopic,
}

/// Ownership QoS kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipKind {
    /// Shared: multiple writers can publish concurrently.
    Shared,
    /// Exclusive: only highest-strength writer delivers.
    Exclusive,
}

/// History QoS policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryQos {
    pub kind: HistoryKind,
    pub depth: u32,
}

impl Default for HistoryQos {
    fn default() -> Self {
        Self {
            kind: HistoryKind::KeepLast,
            depth: 1,
        }
    }
}

/// Deadline QoS policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeadlineQos {
    /// Deadline period in microseconds (0 = infinite).
    pub period_us: u64,
}

/// Aggregate QoS policy for a DDS endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QosPolicy {
    pub reliability: ReliabilityKind,
    pub durability: DurabilityKind,
    pub history: HistoryQos,
    pub deadline: DeadlineQos,
    pub liveliness: LivelinessKind,
    pub ownership: OwnershipKind,
    pub ownership_strength: u32,
}

impl Default for QosPolicy {
    fn default() -> Self {
        Self {
            reliability: ReliabilityKind::BestEffort,
            durability: DurabilityKind::Volatile,
            history: HistoryQos::default(),
            deadline: DeadlineQos::default(),
            liveliness: LivelinessKind::Automatic,
            ownership: OwnershipKind::Shared,
            ownership_strength: 0,
        }
    }
}

impl QosPolicy {
    /// Check whether a writer's offered QoS is compatible with a reader's requested QoS.
    ///
    /// Returns `true` if compatible (matching is allowed).
    pub fn is_compatible(offered: &QosPolicy, requested: &QosPolicy) -> bool {
        // Reliability: writer best-effort incompatible with reader reliable
        if offered.reliability == ReliabilityKind::BestEffort
            && requested.reliability == ReliabilityKind::Reliable
        {
            return false;
        }

        // Durability: writer volatile incompatible with reader transient-local
        if offered.durability == DurabilityKind::Volatile
            && requested.durability == DurabilityKind::TransientLocal
        {
            return false;
        }

        // Ownership: must agree
        if offered.ownership != requested.ownership {
            return false;
        }

        // Deadline: writer's offered period must be <= reader's requested period
        // (0 means infinite, which is the most relaxed)
        if offered.deadline.period_us > 0
            && requested.deadline.period_us > 0
            && offered.deadline.period_us > requested.deadline.period_us
        {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_qos() {
        let qos = QosPolicy::default();
        assert_eq!(qos.reliability, ReliabilityKind::BestEffort);
        assert_eq!(qos.durability, DurabilityKind::Volatile);
        assert_eq!(qos.history.kind, HistoryKind::KeepLast);
        assert_eq!(qos.history.depth, 1);
        assert_eq!(qos.deadline.period_us, 0);
        assert_eq!(qos.liveliness, LivelinessKind::Automatic);
        assert_eq!(qos.ownership, OwnershipKind::Shared);
    }

    #[test]
    fn test_compatible_default_both() {
        let offered = QosPolicy::default();
        let requested = QosPolicy::default();
        assert!(QosPolicy::is_compatible(&offered, &requested));
    }

    #[test]
    fn test_incompatible_reliability() {
        let offered = QosPolicy {
            reliability: ReliabilityKind::BestEffort,
            ..Default::default()
        };
        let requested = QosPolicy {
            reliability: ReliabilityKind::Reliable,
            ..Default::default()
        };
        assert!(!QosPolicy::is_compatible(&offered, &requested));
    }

    #[test]
    fn test_compatible_reliability_upgrade() {
        // Writer reliable, reader best-effort — compatible
        let offered = QosPolicy {
            reliability: ReliabilityKind::Reliable,
            ..Default::default()
        };
        let requested = QosPolicy {
            reliability: ReliabilityKind::BestEffort,
            ..Default::default()
        };
        assert!(QosPolicy::is_compatible(&offered, &requested));
    }

    #[test]
    fn test_incompatible_durability() {
        let offered = QosPolicy {
            durability: DurabilityKind::Volatile,
            ..Default::default()
        };
        let requested = QosPolicy {
            durability: DurabilityKind::TransientLocal,
            ..Default::default()
        };
        assert!(!QosPolicy::is_compatible(&offered, &requested));
    }

    #[test]
    fn test_compatible_durability_upgrade() {
        let offered = QosPolicy {
            durability: DurabilityKind::TransientLocal,
            ..Default::default()
        };
        let requested = QosPolicy {
            durability: DurabilityKind::Volatile,
            ..Default::default()
        };
        assert!(QosPolicy::is_compatible(&offered, &requested));
    }

    #[test]
    fn test_incompatible_ownership() {
        let offered = QosPolicy {
            ownership: OwnershipKind::Shared,
            ..Default::default()
        };
        let requested = QosPolicy {
            ownership: OwnershipKind::Exclusive,
            ..Default::default()
        };
        assert!(!QosPolicy::is_compatible(&offered, &requested));
    }

    #[test]
    fn test_compatible_deadline() {
        let offered = QosPolicy {
            deadline: DeadlineQos { period_us: 50_000 },
            ..Default::default()
        };
        let requested = QosPolicy {
            deadline: DeadlineQos { period_us: 100_000 },
            ..Default::default()
        };
        assert!(QosPolicy::is_compatible(&offered, &requested));
    }

    #[test]
    fn test_incompatible_deadline() {
        let offered = QosPolicy {
            deadline: DeadlineQos { period_us: 200_000 },
            ..Default::default()
        };
        let requested = QosPolicy {
            deadline: DeadlineQos { period_us: 100_000 },
            ..Default::default()
        };
        assert!(!QosPolicy::is_compatible(&offered, &requested));
    }

    #[test]
    fn test_deadline_zero_always_compatible() {
        let offered = QosPolicy {
            deadline: DeadlineQos { period_us: 0 },
            ..Default::default()
        };
        let requested = QosPolicy {
            deadline: DeadlineQos { period_us: 100_000 },
            ..Default::default()
        };
        assert!(QosPolicy::is_compatible(&offered, &requested));
    }

    #[test]
    fn test_history_keep_all() {
        let qos = QosPolicy {
            history: HistoryQos {
                kind: HistoryKind::KeepAll,
                depth: 0,
            },
            ..Default::default()
        };
        assert_eq!(qos.history.kind, HistoryKind::KeepAll);
    }
}
