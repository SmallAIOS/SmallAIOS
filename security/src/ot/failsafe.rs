// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Fail-safe state definitions and enforcement.
//!
//! Defines fail-safe behavior for all failure modes:
//! - Inference timeout: return error, no partial results, release resources
//! - Memory exhaustion: reject allocation, preserve existing state
//! - Watchdog timeout: controlled reset with audit flush
//! - Capability violation: deny and log, no crash

/// Fail-safe state type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FailSafeState {
    /// Normal operation.
    Normal = 0,
    /// Inference timeout: operation terminated, resources released.
    InferenceTimeout = 1,
    /// Memory exhaustion: new allocations rejected, existing preserved.
    MemoryExhaustion = 2,
    /// Watchdog timeout: controlled reset in progress.
    WatchdogTimeout = 3,
    /// Capability violation: operation denied, task continues.
    CapabilityViolation = 4,
}

impl FailSafeState {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Normal),
            1 => Some(Self::InferenceTimeout),
            2 => Some(Self::MemoryExhaustion),
            3 => Some(Self::WatchdogTimeout),
            4 => Some(Self::CapabilityViolation),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::InferenceTimeout => "inference-timeout",
            Self::MemoryExhaustion => "memory-exhaustion",
            Self::WatchdogTimeout => "watchdog-timeout",
            Self::CapabilityViolation => "capability-violation",
        }
    }

    /// Whether this state requires immediate recovery action.
    pub fn requires_recovery(&self) -> bool {
        !matches!(self, Self::Normal)
    }

    /// Maximum time to reach safe state (nanoseconds).
    /// 0 = immediate (no transition delay).
    pub fn max_transition_ns(&self) -> u64 {
        match self {
            Self::Normal => 0,
            Self::InferenceTimeout => 0, // bounded by cleanup overhead, platform-specific
            Self::MemoryExhaustion => 0, // immediate rejection
            Self::WatchdogTimeout => 100_000_000, // 100ms default
            Self::CapabilityViolation => 0, // immediate denial
        }
    }
}

/// Recovery action for a fail-safe state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// No action needed (normal state).
    None,
    /// Caller retries or reports failure upstream.
    CallerRetry,
    /// Free unused resources or wait for others to release.
    FreeResources,
    /// System reboots into known-good configuration.
    SystemReboot,
    /// Request missing capability through authorized grant.
    RequestCapability,
}

impl FailSafeState {
    /// Recommended recovery action for this state.
    pub fn recovery_action(&self) -> RecoveryAction {
        match self {
            Self::Normal => RecoveryAction::None,
            Self::InferenceTimeout => RecoveryAction::CallerRetry,
            Self::MemoryExhaustion => RecoveryAction::FreeResources,
            Self::WatchdogTimeout => RecoveryAction::SystemReboot,
            Self::CapabilityViolation => RecoveryAction::RequestCapability,
        }
    }
}

/// A fail-safe event record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FailSafeEvent {
    /// Timestamp when the event occurred (nanoseconds).
    pub timestamp_ns: u64,
    /// The fail-safe state entered.
    pub state: FailSafeState,
    /// Task ID that triggered the event (0 = system-wide).
    pub task_id: u64,
    /// Additional context (interpretation depends on state).
    pub context: u64,
}

/// Tracks fail-safe state transitions and history.
pub struct FailSafeTracker {
    /// Current system fail-safe state.
    current_state: FailSafeState,
    /// Recent fail-safe events.
    events: [FailSafeEvent; MAX_EVENTS],
    /// Number of events recorded.
    event_count: u32,
    /// Write position in circular buffer.
    write_pos: usize,
    /// Total events since boot.
    total_events: u64,
}

const MAX_EVENTS: usize = 64;

impl FailSafeTracker {
    pub fn new() -> Self {
        Self {
            current_state: FailSafeState::Normal,
            events: [FailSafeEvent {
                timestamp_ns: 0,
                state: FailSafeState::Normal,
                task_id: 0,
                context: 0,
            }; MAX_EVENTS],
            event_count: 0,
            write_pos: 0,
            total_events: 0,
        }
    }

    /// Record a fail-safe state transition.
    pub fn record_event(&mut self, event: FailSafeEvent) {
        self.current_state = event.state;
        self.events[self.write_pos] = event;
        self.write_pos = (self.write_pos + 1) % MAX_EVENTS;
        if self.event_count < MAX_EVENTS as u32 {
            self.event_count += 1;
        }
        self.total_events += 1;
    }

    /// Record an inference timeout.
    pub fn record_inference_timeout(&mut self, timestamp_ns: u64, task_id: u64, elapsed_ns: u64) {
        self.record_event(FailSafeEvent {
            timestamp_ns,
            state: FailSafeState::InferenceTimeout,
            task_id,
            context: elapsed_ns,
        });
    }

    /// Record a memory exhaustion event.
    pub fn record_memory_exhaustion(
        &mut self,
        timestamp_ns: u64,
        task_id: u64,
        requested_bytes: u64,
    ) {
        self.record_event(FailSafeEvent {
            timestamp_ns,
            state: FailSafeState::MemoryExhaustion,
            task_id,
            context: requested_bytes,
        });
    }

    /// Record a watchdog timeout.
    pub fn record_watchdog_timeout(&mut self, timestamp_ns: u64) {
        self.record_event(FailSafeEvent {
            timestamp_ns,
            state: FailSafeState::WatchdogTimeout,
            task_id: 0,
            context: 0,
        });
    }

    /// Record a capability violation.
    pub fn record_capability_violation(&mut self, timestamp_ns: u64, task_id: u64, cap_id: u64) {
        self.record_event(FailSafeEvent {
            timestamp_ns,
            state: FailSafeState::CapabilityViolation,
            task_id,
            context: cap_id,
        });
    }

    /// Return to normal state.
    pub fn clear(&mut self) {
        self.current_state = FailSafeState::Normal;
    }

    /// Current fail-safe state.
    pub fn current_state(&self) -> FailSafeState {
        self.current_state
    }

    /// Number of events recorded in history.
    pub fn event_count(&self) -> u32 {
        self.event_count
    }

    /// Total events since boot.
    pub fn total_events(&self) -> u64 {
        self.total_events
    }

    /// Get the most recent event, if any.
    pub fn last_event(&self) -> Option<&FailSafeEvent> {
        if self.event_count == 0 {
            return None;
        }
        let idx = if self.write_pos == 0 {
            MAX_EVENTS - 1
        } else {
            self.write_pos - 1
        };
        Some(&self.events[idx])
    }

    /// Count events of a specific state type.
    pub fn count_by_state(&self, state: FailSafeState) -> u32 {
        let n = self.event_count as usize;
        self.events[..n].iter().filter(|e| e.state == state).count() as u32
    }
}

impl Default for FailSafeTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_roundtrip() {
        for i in 0..=4u8 {
            let s = FailSafeState::from_u8(i).unwrap();
            assert_eq!(s as u8, i);
        }
        assert!(FailSafeState::from_u8(5).is_none());
    }

    #[test]
    fn state_as_str() {
        assert_eq!(FailSafeState::Normal.as_str(), "normal");
        assert_eq!(
            FailSafeState::InferenceTimeout.as_str(),
            "inference-timeout"
        );
        assert_eq!(
            FailSafeState::MemoryExhaustion.as_str(),
            "memory-exhaustion"
        );
        assert_eq!(FailSafeState::WatchdogTimeout.as_str(), "watchdog-timeout");
        assert_eq!(
            FailSafeState::CapabilityViolation.as_str(),
            "capability-violation"
        );
    }

    #[test]
    fn requires_recovery() {
        assert!(!FailSafeState::Normal.requires_recovery());
        assert!(FailSafeState::InferenceTimeout.requires_recovery());
        assert!(FailSafeState::MemoryExhaustion.requires_recovery());
        assert!(FailSafeState::WatchdogTimeout.requires_recovery());
        assert!(FailSafeState::CapabilityViolation.requires_recovery());
    }

    #[test]
    fn max_transition_times() {
        assert_eq!(FailSafeState::Normal.max_transition_ns(), 0);
        assert_eq!(FailSafeState::MemoryExhaustion.max_transition_ns(), 0);
        assert_eq!(FailSafeState::CapabilityViolation.max_transition_ns(), 0);
        assert_eq!(
            FailSafeState::WatchdogTimeout.max_transition_ns(),
            100_000_000
        );
    }

    #[test]
    fn recovery_actions() {
        assert_eq!(
            FailSafeState::Normal.recovery_action(),
            RecoveryAction::None
        );
        assert_eq!(
            FailSafeState::InferenceTimeout.recovery_action(),
            RecoveryAction::CallerRetry
        );
        assert_eq!(
            FailSafeState::MemoryExhaustion.recovery_action(),
            RecoveryAction::FreeResources
        );
        assert_eq!(
            FailSafeState::WatchdogTimeout.recovery_action(),
            RecoveryAction::SystemReboot
        );
        assert_eq!(
            FailSafeState::CapabilityViolation.recovery_action(),
            RecoveryAction::RequestCapability
        );
    }

    #[test]
    fn tracker_new() {
        let tracker = FailSafeTracker::new();
        assert_eq!(tracker.current_state(), FailSafeState::Normal);
        assert_eq!(tracker.event_count(), 0);
        assert!(tracker.last_event().is_none());
    }

    #[test]
    fn record_inference_timeout() {
        let mut tracker = FailSafeTracker::new();
        tracker.record_inference_timeout(1000, 42, 5_000_000);
        assert_eq!(tracker.current_state(), FailSafeState::InferenceTimeout);
        assert_eq!(tracker.event_count(), 1);
        let last = tracker.last_event().unwrap();
        assert_eq!(last.task_id, 42);
        assert_eq!(last.context, 5_000_000);
    }

    #[test]
    fn record_memory_exhaustion() {
        let mut tracker = FailSafeTracker::new();
        tracker.record_memory_exhaustion(2000, 10, 65536);
        assert_eq!(tracker.current_state(), FailSafeState::MemoryExhaustion);
        let last = tracker.last_event().unwrap();
        assert_eq!(last.context, 65536);
    }

    #[test]
    fn record_watchdog_timeout() {
        let mut tracker = FailSafeTracker::new();
        tracker.record_watchdog_timeout(3000);
        assert_eq!(tracker.current_state(), FailSafeState::WatchdogTimeout);
    }

    #[test]
    fn record_capability_violation() {
        let mut tracker = FailSafeTracker::new();
        tracker.record_capability_violation(4000, 7, 99);
        assert_eq!(tracker.current_state(), FailSafeState::CapabilityViolation);
        let last = tracker.last_event().unwrap();
        assert_eq!(last.task_id, 7);
        assert_eq!(last.context, 99);
    }

    #[test]
    fn clear_restores_normal() {
        let mut tracker = FailSafeTracker::new();
        tracker.record_watchdog_timeout(1000);
        assert_eq!(tracker.current_state(), FailSafeState::WatchdogTimeout);
        tracker.clear();
        assert_eq!(tracker.current_state(), FailSafeState::Normal);
    }

    #[test]
    fn count_by_state() {
        let mut tracker = FailSafeTracker::new();
        tracker.record_inference_timeout(1000, 1, 0);
        tracker.record_inference_timeout(2000, 2, 0);
        tracker.record_memory_exhaustion(3000, 3, 0);
        assert_eq!(tracker.count_by_state(FailSafeState::InferenceTimeout), 2);
        assert_eq!(tracker.count_by_state(FailSafeState::MemoryExhaustion), 1);
        assert_eq!(tracker.count_by_state(FailSafeState::WatchdogTimeout), 0);
    }

    #[test]
    fn multiple_events_total() {
        let mut tracker = FailSafeTracker::new();
        for i in 0..10 {
            tracker.record_capability_violation(i * 100, i, 0);
        }
        assert_eq!(tracker.total_events(), 10);
        assert_eq!(tracker.event_count(), 10);
    }
}
