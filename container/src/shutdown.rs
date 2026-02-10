// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Graceful shutdown management.
//!
//! Provides an ordered multi-phase shutdown sequence for container instances.
//! Shutdown proceeds through well-defined phases (drain, close, save, cleanup,
//! terminate) and supports prioritized shutdown hooks that execute callbacks
//! in a deterministic order.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::ContainerError;

/// Maximum number of shutdown hooks that can be registered.
const MAX_HOOKS: usize = 32;

/// Default timeout (in milliseconds) for the request-drain phase.
const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 30_000;

/// The current phase of the shutdown sequence.
///
/// Phases proceed in strict order from [`Running`](ShutdownPhase::Running)
/// through to [`Terminated`](ShutdownPhase::Terminated).
#[derive(Clone, Debug, PartialEq)]
pub enum ShutdownPhase {
    /// Normal operation; no shutdown in progress.
    Running,
    /// Draining in-flight requests (no new requests accepted).
    DrainRequests,
    /// Closing network connections and IPC channels.
    CloseConnections,
    /// Persisting state (checkpoints, logs).
    SaveState,
    /// Releasing resources and running cleanup hooks.
    Cleanup,
    /// Shutdown complete; the container is stopped.
    Terminated,
}

impl ShutdownPhase {
    /// Returns the next phase in the shutdown sequence, or `None` if already
    /// terminated.
    fn next(&self) -> Option<ShutdownPhase> {
        match self {
            ShutdownPhase::Running => Some(ShutdownPhase::DrainRequests),
            ShutdownPhase::DrainRequests => Some(ShutdownPhase::CloseConnections),
            ShutdownPhase::CloseConnections => Some(ShutdownPhase::SaveState),
            ShutdownPhase::SaveState => Some(ShutdownPhase::Cleanup),
            ShutdownPhase::Cleanup => Some(ShutdownPhase::Terminated),
            ShutdownPhase::Terminated => None,
        }
    }
}

/// The reason why a shutdown was initiated.
#[derive(Clone, Debug, PartialEq)]
pub enum ShutdownReason {
    /// SIGTERM received from the orchestrator.
    Sigterm,
    /// SIGINT received (e.g., Ctrl-C in development).
    Sigint,
    /// Health check failure triggered automatic shutdown.
    HealthCheckFailed,
    /// Operator or API-initiated manual stop.
    ManualStop,
    /// An error condition forced the shutdown.
    Error(String),
}

/// A registered shutdown hook that runs during the cleanup phase.
#[derive(Clone, Debug)]
pub struct ShutdownHook {
    /// Human-readable name for this hook.
    pub name: String,
    /// Priority value; higher priorities execute first.
    pub priority: u8,
    /// Whether this hook has already been executed.
    pub executed: bool,
}

/// Manages the lifecycle of a graceful shutdown sequence.
///
/// The manager tracks the current phase, the reason for shutdown, in-flight
/// request counts, and an ordered set of cleanup hooks.
#[derive(Debug)]
pub struct ShutdownManager {
    /// Current shutdown phase.
    phase: ShutdownPhase,
    /// Why the shutdown was initiated (set when [`initiate`](Self::initiate) is called).
    reason: Option<ShutdownReason>,
    /// Registered shutdown hooks, executed during the cleanup phase.
    hooks: Vec<ShutdownHook>,
    /// How long (in milliseconds) to wait for in-flight requests to drain.
    drain_timeout_ms: u64,
    /// Number of currently in-flight requests.
    active_requests: u32,
    /// Whether [`initiate`](Self::initiate) has been called.
    shutdown_initiated: bool,
}

impl Default for ShutdownManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ShutdownManager {
    /// Create a new shutdown manager in the [`Running`](ShutdownPhase::Running) state.
    pub fn new() -> Self {
        ShutdownManager {
            phase: ShutdownPhase::Running,
            reason: None,
            hooks: Vec::new(),
            drain_timeout_ms: DEFAULT_DRAIN_TIMEOUT_MS,
            active_requests: 0,
            shutdown_initiated: false,
        }
    }

    /// Register a shutdown hook with the given name and priority.
    ///
    /// Hooks with higher priority values execute first during the cleanup
    /// phase. Hooks are not executed until [`execute_hooks`](Self::execute_hooks)
    /// is called.
    ///
    /// # Errors
    ///
    /// Returns [`ContainerError::TableFull`] if [`MAX_HOOKS`] hooks are
    /// already registered.
    pub fn register_hook(&mut self, name: &str, priority: u8) -> Result<(), ContainerError> {
        if self.hooks.len() >= MAX_HOOKS {
            return Err(ContainerError::TableFull);
        }
        self.hooks.push(ShutdownHook {
            name: String::from(name),
            priority,
            executed: false,
        });
        Ok(())
    }

    /// Initiate the shutdown sequence with the given reason.
    ///
    /// Transitions the phase from [`Running`](ShutdownPhase::Running) to
    /// [`DrainRequests`](ShutdownPhase::DrainRequests) and records the reason.
    ///
    /// # Errors
    ///
    /// Returns [`ContainerError::AlreadyInitiated`] if shutdown has already
    /// been initiated.
    pub fn initiate(&mut self, reason: ShutdownReason) -> Result<(), ContainerError> {
        if self.shutdown_initiated {
            return Err(ContainerError::AlreadyInitiated);
        }
        self.shutdown_initiated = true;
        self.reason = Some(reason);
        self.phase = ShutdownPhase::DrainRequests;
        Ok(())
    }

    /// Advance to the next shutdown phase.
    ///
    /// # Errors
    ///
    /// Returns [`ContainerError::ShutdownFailed`] if the manager is already
    /// in the [`Terminated`](ShutdownPhase::Terminated) phase and cannot
    /// advance further.
    pub fn advance(&mut self) -> Result<ShutdownPhase, ContainerError> {
        match self.phase.next() {
            Some(next_phase) => {
                self.phase = next_phase.clone();
                Ok(next_phase)
            }
            None => Err(ContainerError::ShutdownFailed(String::from(
                "already terminated",
            ))),
        }
    }

    /// Returns `true` if a shutdown has been initiated.
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown_initiated
    }

    /// Returns a reference to the current shutdown phase.
    pub fn phase(&self) -> &ShutdownPhase {
        &self.phase
    }

    /// Returns the shutdown reason, if one has been set.
    pub fn reason(&self) -> Option<&ShutdownReason> {
        self.reason.as_ref()
    }

    /// Update the count of in-flight active requests.
    pub fn set_active_requests(&mut self, count: u32) {
        self.active_requests = count;
    }

    /// Returns the current count of in-flight active requests.
    pub fn active_requests(&self) -> u32 {
        self.active_requests
    }

    /// Returns `true` if the container can safely terminate.
    ///
    /// Termination is safe when there are no active requests and the shutdown
    /// has progressed to at least the [`CloseConnections`](ShutdownPhase::CloseConnections)
    /// phase.
    pub fn can_terminate(&self) -> bool {
        if self.active_requests != 0 {
            return false;
        }
        matches!(
            self.phase,
            ShutdownPhase::CloseConnections
                | ShutdownPhase::SaveState
                | ShutdownPhase::Cleanup
                | ShutdownPhase::Terminated
        )
    }

    /// Execute all registered shutdown hooks in priority order (highest first).
    ///
    /// Returns the names of the hooks that were executed. Hooks that have
    /// already been executed are skipped.
    pub fn execute_hooks(&mut self) -> Vec<&str> {
        // Sort by priority descending (highest first).
        self.hooks.sort_by_key(|b| core::cmp::Reverse(b.priority));

        let mut executed_names = Vec::new();
        for hook in self.hooks.iter_mut() {
            if !hook.executed {
                hook.executed = true;
                executed_names.push(hook.name.as_str());
            }
        }
        executed_names
    }

    /// Returns the number of registered shutdown hooks.
    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    /// Run a complete shutdown sequence from initiation through termination.
    ///
    /// This is a convenience method that calls [`initiate`](Self::initiate)
    /// followed by [`advance`](Self::advance) until the
    /// [`Terminated`](ShutdownPhase::Terminated) phase is reached.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`initiate`](Self::initiate) or
    /// [`advance`](Self::advance).
    pub fn run_full_shutdown(
        &mut self,
        reason: ShutdownReason,
    ) -> Result<ShutdownPhase, ContainerError> {
        self.initiate(reason)?;
        // Advance through: DrainRequests -> CloseConnections -> SaveState -> Cleanup -> Terminated
        loop {
            let phase = self.advance()?;
            if phase == ShutdownPhase::Terminated {
                return Ok(phase);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn new_manager_is_running() {
        let mgr = ShutdownManager::new();
        assert_eq!(*mgr.phase(), ShutdownPhase::Running);
        assert!(!mgr.is_shutting_down());
        assert_eq!(mgr.reason(), None);
    }

    #[test]
    fn register_hook() {
        let mut mgr = ShutdownManager::new();
        mgr.register_hook("cleanup_temp", 10).unwrap();
        assert_eq!(mgr.hook_count(), 1);
    }

    #[test]
    fn initiate_shutdown_transitions_to_drain() {
        let mut mgr = ShutdownManager::new();
        mgr.initiate(ShutdownReason::Sigterm).unwrap();
        assert_eq!(*mgr.phase(), ShutdownPhase::DrainRequests);
        assert!(mgr.is_shutting_down());
    }

    #[test]
    fn initiate_twice_returns_error() {
        let mut mgr = ShutdownManager::new();
        mgr.initiate(ShutdownReason::Sigterm).unwrap();
        let result = mgr.initiate(ShutdownReason::Sigint);
        assert_eq!(result, Err(ContainerError::AlreadyInitiated));
    }

    #[test]
    fn advance_through_all_phases() {
        let mut mgr = ShutdownManager::new();
        mgr.initiate(ShutdownReason::ManualStop).unwrap();

        assert_eq!(mgr.advance().unwrap(), ShutdownPhase::CloseConnections);
        assert_eq!(mgr.advance().unwrap(), ShutdownPhase::SaveState);
        assert_eq!(mgr.advance().unwrap(), ShutdownPhase::Cleanup);
        assert_eq!(mgr.advance().unwrap(), ShutdownPhase::Terminated);

        // Cannot advance past Terminated.
        assert!(mgr.advance().is_err());
    }

    #[test]
    fn phase_order_correct() {
        let phases = [
            ShutdownPhase::Running,
            ShutdownPhase::DrainRequests,
            ShutdownPhase::CloseConnections,
            ShutdownPhase::SaveState,
            ShutdownPhase::Cleanup,
            ShutdownPhase::Terminated,
        ];

        for i in 0..phases.len() - 1 {
            assert_eq!(phases[i].next(), Some(phases[i + 1].clone()));
        }
        assert_eq!(ShutdownPhase::Terminated.next(), None);
    }

    #[test]
    fn reason_recorded() {
        let mut mgr = ShutdownManager::new();
        mgr.initiate(ShutdownReason::HealthCheckFailed).unwrap();
        assert_eq!(mgr.reason(), Some(&ShutdownReason::HealthCheckFailed));
    }

    #[test]
    fn active_requests_tracking() {
        let mut mgr = ShutdownManager::new();
        assert_eq!(mgr.active_requests(), 0);
        mgr.set_active_requests(5);
        assert_eq!(mgr.active_requests(), 5);
        mgr.set_active_requests(0);
        assert_eq!(mgr.active_requests(), 0);
    }

    #[test]
    fn can_terminate_logic() {
        let mut mgr = ShutdownManager::new();

        // Running phase with no active requests: cannot terminate.
        assert!(!mgr.can_terminate());

        mgr.initiate(ShutdownReason::Sigterm).unwrap();
        // DrainRequests with 0 active requests: cannot terminate (too early).
        assert!(!mgr.can_terminate());

        mgr.advance().unwrap(); // -> CloseConnections
                                // CloseConnections, 0 active: can terminate.
        assert!(mgr.can_terminate());

        mgr.set_active_requests(3);
        // CloseConnections, 3 active: cannot terminate.
        assert!(!mgr.can_terminate());

        mgr.set_active_requests(0);
        assert!(mgr.can_terminate());

        mgr.advance().unwrap(); // -> SaveState
        assert!(mgr.can_terminate());

        mgr.advance().unwrap(); // -> Cleanup
        assert!(mgr.can_terminate());

        mgr.advance().unwrap(); // -> Terminated
        assert!(mgr.can_terminate());
    }

    #[test]
    fn execute_hooks_in_priority_order() {
        let mut mgr = ShutdownManager::new();
        mgr.register_hook("low_priority", 1).unwrap();
        mgr.register_hook("high_priority", 100).unwrap();
        mgr.register_hook("medium_priority", 50).unwrap();

        let executed = mgr.execute_hooks();
        assert_eq!(
            executed,
            vec!["high_priority", "medium_priority", "low_priority"]
        );

        // Executing again should return empty (all already executed).
        let executed_again = mgr.execute_hooks();
        assert!(executed_again.is_empty());
    }

    #[test]
    fn hook_count() {
        let mut mgr = ShutdownManager::new();
        assert_eq!(mgr.hook_count(), 0);
        mgr.register_hook("a", 1).unwrap();
        mgr.register_hook("b", 2).unwrap();
        assert_eq!(mgr.hook_count(), 2);
    }

    #[test]
    fn full_shutdown_sequence() {
        let mut mgr = ShutdownManager::new();
        let result = mgr.run_full_shutdown(ShutdownReason::Sigterm);
        assert_eq!(result, Ok(ShutdownPhase::Terminated));
        assert_eq!(*mgr.phase(), ShutdownPhase::Terminated);
        assert!(mgr.is_shutting_down());
    }

    #[test]
    fn sigterm_and_sigint_reasons() {
        let sigterm = ShutdownReason::Sigterm;
        let sigint = ShutdownReason::Sigint;
        assert_ne!(sigterm, sigint);

        let sigterm_clone = sigterm.clone();
        assert_eq!(sigterm, sigterm_clone);

        let error_reason = ShutdownReason::Error(String::from("disk full"));
        assert_eq!(
            error_reason,
            ShutdownReason::Error(String::from("disk full"))
        );
        assert_ne!(error_reason, ShutdownReason::Error(String::from("oom")),);
    }

    #[test]
    fn table_full_error() {
        let mut mgr = ShutdownManager::new();
        for i in 0..MAX_HOOKS {
            let name = alloc::format!("hook_{}", i);
            mgr.register_hook(&name, i as u8).unwrap();
        }
        let result = mgr.register_hook("overflow", 99);
        assert_eq!(result, Err(ContainerError::TableFull));
    }

    #[test]
    fn drain_timeout_default() {
        let mgr = ShutdownManager::new();
        assert_eq!(mgr.drain_timeout_ms, DEFAULT_DRAIN_TIMEOUT_MS);
    }

    #[test]
    fn shutdown_phase_debug_and_clone() {
        let phase = ShutdownPhase::SaveState;
        let cloned = phase.clone();
        assert_eq!(phase, cloned);
        // Debug formatting should not panic.
        let _ = alloc::format!("{:?}", phase);
    }
}
