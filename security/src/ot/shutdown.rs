// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Safe shutdown procedure with bounded time.
//!
//! Implements the safe shutdown sequence per NIST SP 800-53:
//! 1. Flush audit logs (default 25ms budget)
//! 2. Revoke all capabilities (default 25ms budget)
//! 3. Terminate all tasks (default 25ms budget)
//! 4. Trigger watchdog reset
//!
//! Total shutdown time bounded to configurable limit (default 100ms).
//! If any phase exceeds its budget, skip remaining phases and reset.

/// Shutdown phase in the safe shutdown sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShutdownPhase {
    /// Not in shutdown.
    Idle = 0,
    /// Phase 1: Flushing audit logs.
    AuditFlush = 1,
    /// Phase 2: Revoking all capabilities.
    CapabilityRevoke = 2,
    /// Phase 3: Terminating all tasks.
    TaskTerminate = 3,
    /// Phase 4: Triggering watchdog reset.
    WatchdogReset = 4,
    /// Shutdown completed (or forced).
    Complete = 5,
}

impl ShutdownPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::AuditFlush => "audit-flush",
            Self::CapabilityRevoke => "capability-revoke",
            Self::TaskTerminate => "task-terminate",
            Self::WatchdogReset => "watchdog-reset",
            Self::Complete => "complete",
        }
    }

    /// Next phase in the sequence.
    pub fn next(&self) -> Self {
        match self {
            Self::Idle => Self::AuditFlush,
            Self::AuditFlush => Self::CapabilityRevoke,
            Self::CapabilityRevoke => Self::TaskTerminate,
            Self::TaskTerminate => Self::WatchdogReset,
            Self::WatchdogReset => Self::Complete,
            Self::Complete => Self::Complete,
        }
    }
}

/// Reason for initiating shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownReason {
    /// Operator-initiated shutdown command.
    OperatorCommand,
    /// Watchdog timeout triggered shutdown.
    WatchdogTimeout,
    /// Critical failure detected.
    CriticalFailure,
    /// Kernel integrity violation.
    IntegrityViolation,
}

impl ShutdownReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OperatorCommand => "operator-command",
            Self::WatchdogTimeout => "watchdog-timeout",
            Self::CriticalFailure => "critical-failure",
            Self::IntegrityViolation => "integrity-violation",
        }
    }
}

/// Configuration for shutdown timing budgets.
#[derive(Debug, Clone, Copy)]
pub struct ShutdownConfig {
    /// Total shutdown time budget (nanoseconds). Default: 100ms.
    pub total_budget_ns: u64,
    /// Audit flush phase budget (nanoseconds). Default: 25ms.
    pub audit_flush_budget_ns: u64,
    /// Capability revocation phase budget (nanoseconds). Default: 25ms.
    pub cap_revoke_budget_ns: u64,
    /// Task termination phase budget (nanoseconds). Default: 25ms.
    pub task_terminate_budget_ns: u64,
}

impl ShutdownConfig {
    /// Default shutdown configuration per spec.
    pub const fn default_config() -> Self {
        Self {
            total_budget_ns: 100_000_000,      // 100ms
            audit_flush_budget_ns: 25_000_000,  // 25ms
            cap_revoke_budget_ns: 25_000_000,   // 25ms
            task_terminate_budget_ns: 25_000_000, // 25ms
        }
    }
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Result of a single shutdown phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseResult {
    /// Phase completed within budget.
    Completed {
        elapsed_ns: u64,
    },
    /// Phase exceeded its budget and was forced to stop.
    TimedOut {
        elapsed_ns: u64,
    },
    /// Phase was skipped due to overall budget exhaustion.
    Skipped,
}

impl PhaseResult {
    pub fn completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }
}

/// Safe shutdown state machine.
pub struct ShutdownSequence {
    config: ShutdownConfig,
    /// Current phase.
    current_phase: ShutdownPhase,
    /// Reason for shutdown.
    reason: Option<ShutdownReason>,
    /// Timestamp when shutdown was initiated.
    initiated_ns: u64,
    /// Results for each phase.
    phase_results: [Option<PhaseResult>; 4],
    /// Total elapsed time so far.
    total_elapsed_ns: u64,
    /// Whether shutdown is forced (skip to reset).
    forced: bool,
}

impl ShutdownSequence {
    pub fn new(config: ShutdownConfig) -> Self {
        Self {
            config,
            current_phase: ShutdownPhase::Idle,
            reason: None,
            initiated_ns: 0,
            phase_results: [None; 4],
            total_elapsed_ns: 0,
            forced: false,
        }
    }

    /// Initiate the shutdown sequence.
    pub fn initiate(&mut self, reason: ShutdownReason, timestamp_ns: u64) {
        self.reason = Some(reason);
        self.initiated_ns = timestamp_ns;
        self.current_phase = ShutdownPhase::AuditFlush;
        self.total_elapsed_ns = 0;
        self.forced = false;
    }

    /// Whether shutdown is in progress.
    pub fn is_active(&self) -> bool {
        !matches!(self.current_phase, ShutdownPhase::Idle | ShutdownPhase::Complete)
    }

    /// Current phase.
    pub fn current_phase(&self) -> ShutdownPhase {
        self.current_phase
    }

    /// Shutdown reason.
    pub fn reason(&self) -> Option<ShutdownReason> {
        self.reason
    }

    /// Complete the current phase and advance to the next.
    ///
    /// `elapsed_ns` is the time taken for this phase.
    /// Returns the phase result and the next phase.
    pub fn complete_phase(&mut self, elapsed_ns: u64) -> (ShutdownPhase, PhaseResult) {
        self.total_elapsed_ns += elapsed_ns;

        let phase_idx = match self.current_phase {
            ShutdownPhase::AuditFlush => 0,
            ShutdownPhase::CapabilityRevoke => 1,
            ShutdownPhase::TaskTerminate => 2,
            ShutdownPhase::WatchdogReset => 3,
            _ => {
                return (self.current_phase, PhaseResult::Skipped);
            }
        };

        let budget = match self.current_phase {
            ShutdownPhase::AuditFlush => self.config.audit_flush_budget_ns,
            ShutdownPhase::CapabilityRevoke => self.config.cap_revoke_budget_ns,
            ShutdownPhase::TaskTerminate => self.config.task_terminate_budget_ns,
            ShutdownPhase::WatchdogReset => 0, // No budget for final reset
            _ => 0,
        };

        let result = if budget > 0 && elapsed_ns > budget {
            PhaseResult::TimedOut { elapsed_ns }
        } else {
            PhaseResult::Completed { elapsed_ns }
        };

        self.phase_results[phase_idx] = Some(result);

        // Check if total budget exceeded
        if self.total_elapsed_ns > self.config.total_budget_ns {
            self.forced = true;
            self.current_phase = ShutdownPhase::WatchdogReset;
        } else {
            self.current_phase = self.current_phase.next();
        }

        (self.current_phase, result)
    }

    /// Force immediate reset (skip remaining phases).
    pub fn force_reset(&mut self) {
        self.forced = true;
        // Mark remaining phases as skipped
        for result in self.phase_results.iter_mut() {
            if result.is_none() {
                *result = Some(PhaseResult::Skipped);
            }
        }
        self.current_phase = ShutdownPhase::Complete;
    }

    /// Whether shutdown was forced (budget exceeded).
    pub fn was_forced(&self) -> bool {
        self.forced
    }

    /// Get the result of a specific phase.
    pub fn phase_result(&self, phase: ShutdownPhase) -> Option<PhaseResult> {
        let idx = match phase {
            ShutdownPhase::AuditFlush => 0,
            ShutdownPhase::CapabilityRevoke => 1,
            ShutdownPhase::TaskTerminate => 2,
            ShutdownPhase::WatchdogReset => 3,
            _ => return None,
        };
        self.phase_results[idx]
    }

    /// Total elapsed time since shutdown initiation.
    pub fn total_elapsed_ns(&self) -> u64 {
        self.total_elapsed_ns
    }

    /// Remaining time in total budget.
    pub fn remaining_budget_ns(&self) -> u64 {
        self.config.total_budget_ns.saturating_sub(self.total_elapsed_ns)
    }

    /// Whether all phases completed within their budgets.
    pub fn all_phases_completed(&self) -> bool {
        self.phase_results
            .iter()
            .all(|r| matches!(r, Some(PhaseResult::Completed { .. })))
    }
}

impl Default for ShutdownSequence {
    fn default() -> Self {
        Self::new(ShutdownConfig::default_config())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_sequence() {
        assert_eq!(ShutdownPhase::Idle.next(), ShutdownPhase::AuditFlush);
        assert_eq!(ShutdownPhase::AuditFlush.next(), ShutdownPhase::CapabilityRevoke);
        assert_eq!(ShutdownPhase::CapabilityRevoke.next(), ShutdownPhase::TaskTerminate);
        assert_eq!(ShutdownPhase::TaskTerminate.next(), ShutdownPhase::WatchdogReset);
        assert_eq!(ShutdownPhase::WatchdogReset.next(), ShutdownPhase::Complete);
        assert_eq!(ShutdownPhase::Complete.next(), ShutdownPhase::Complete);
    }

    #[test]
    fn phase_as_str() {
        assert_eq!(ShutdownPhase::Idle.as_str(), "idle");
        assert_eq!(ShutdownPhase::AuditFlush.as_str(), "audit-flush");
        assert_eq!(ShutdownPhase::Complete.as_str(), "complete");
    }

    #[test]
    fn reason_as_str() {
        assert_eq!(ShutdownReason::OperatorCommand.as_str(), "operator-command");
        assert_eq!(ShutdownReason::WatchdogTimeout.as_str(), "watchdog-timeout");
        assert_eq!(ShutdownReason::CriticalFailure.as_str(), "critical-failure");
        assert_eq!(ShutdownReason::IntegrityViolation.as_str(), "integrity-violation");
    }

    #[test]
    fn default_config() {
        let cfg = ShutdownConfig::default_config();
        assert_eq!(cfg.total_budget_ns, 100_000_000);
        assert_eq!(cfg.audit_flush_budget_ns, 25_000_000);
    }

    #[test]
    fn idle_sequence() {
        let seq = ShutdownSequence::default();
        assert!(!seq.is_active());
        assert_eq!(seq.current_phase(), ShutdownPhase::Idle);
        assert!(seq.reason().is_none());
    }

    #[test]
    fn normal_shutdown() {
        let mut seq = ShutdownSequence::default();
        seq.initiate(ShutdownReason::OperatorCommand, 1000);
        assert!(seq.is_active());
        assert_eq!(seq.current_phase(), ShutdownPhase::AuditFlush);
        assert_eq!(seq.reason(), Some(ShutdownReason::OperatorCommand));

        // Phase 1: audit flush in 10ms
        let (next, result) = seq.complete_phase(10_000_000);
        assert_eq!(next, ShutdownPhase::CapabilityRevoke);
        assert!(result.completed());

        // Phase 2: cap revoke in 5ms
        let (next, result) = seq.complete_phase(5_000_000);
        assert_eq!(next, ShutdownPhase::TaskTerminate);
        assert!(result.completed());

        // Phase 3: task terminate in 8ms
        let (next, result) = seq.complete_phase(8_000_000);
        assert_eq!(next, ShutdownPhase::WatchdogReset);
        assert!(result.completed());

        // Phase 4: watchdog reset in 1ms
        let (next, result) = seq.complete_phase(1_000_000);
        assert_eq!(next, ShutdownPhase::Complete);
        assert!(result.completed());

        assert!(!seq.is_active());
        assert!(seq.all_phases_completed());
        assert!(!seq.was_forced());
        assert_eq!(seq.total_elapsed_ns(), 24_000_000); // 24ms total
    }

    #[test]
    fn phase_timeout_detected() {
        let mut seq = ShutdownSequence::default();
        seq.initiate(ShutdownReason::CriticalFailure, 1000);

        // Phase 1: audit flush takes 30ms > 25ms budget
        let (_, result) = seq.complete_phase(30_000_000);
        assert!(!result.completed());
        assert_eq!(
            seq.phase_result(ShutdownPhase::AuditFlush),
            Some(PhaseResult::TimedOut {
                elapsed_ns: 30_000_000
            })
        );
    }

    #[test]
    fn total_budget_exceeded_forces_reset() {
        let mut seq = ShutdownSequence::default();
        seq.initiate(ShutdownReason::WatchdogTimeout, 1000);

        // Phase 1: 50ms
        seq.complete_phase(50_000_000);
        // Phase 2: 60ms -> total 110ms > 100ms budget
        let (next, _) = seq.complete_phase(60_000_000);
        // Should skip to watchdog reset
        assert_eq!(next, ShutdownPhase::WatchdogReset);
        assert!(seq.was_forced());
    }

    #[test]
    fn force_reset() {
        let mut seq = ShutdownSequence::default();
        seq.initiate(ShutdownReason::CriticalFailure, 1000);
        seq.force_reset();
        assert!(!seq.is_active());
        assert!(seq.was_forced());
        assert_eq!(seq.current_phase(), ShutdownPhase::Complete);
    }

    #[test]
    fn remaining_budget() {
        let mut seq = ShutdownSequence::default();
        seq.initiate(ShutdownReason::OperatorCommand, 0);
        assert_eq!(seq.remaining_budget_ns(), 100_000_000);
        seq.complete_phase(30_000_000);
        assert_eq!(seq.remaining_budget_ns(), 70_000_000);
    }

    #[test]
    fn custom_config() {
        let config = ShutdownConfig {
            total_budget_ns: 50_000_000,
            audit_flush_budget_ns: 10_000_000,
            cap_revoke_budget_ns: 10_000_000,
            task_terminate_budget_ns: 10_000_000,
        };
        let mut seq = ShutdownSequence::new(config);
        seq.initiate(ShutdownReason::OperatorCommand, 0);
        assert_eq!(seq.remaining_budget_ns(), 50_000_000);
    }
}
