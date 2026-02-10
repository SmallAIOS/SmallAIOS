// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Boot sequence orchestration.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::config::ContainerConfig;
use crate::ContainerError;

// ---------------------------------------------------------------------------
// BootPhase
// ---------------------------------------------------------------------------

/// Ordered phases of the container boot sequence.
///
/// Each phase corresponds to a major subsystem that must be initialised before
/// the container can accept traffic.
#[derive(Clone, Debug, PartialEq)]
pub enum BootPhase {
    /// Configuration has been parsed and validated.
    ConfigLoaded,
    /// Physical / virtual memory subsystem is ready.
    MemoryReady,
    /// Task scheduler is running.
    SchedulerReady,
    /// Capability system and crypto primitives are initialised.
    SecurityReady,
    /// Network stack is bound and listening.
    NetworkReady,
    /// Zenoh-style IPC channels are open.
    IpcReady,
    /// ONNX inference runtime is initialised.
    RuntimeReady,
    /// Pre-loaded models are resident in memory.
    ModelsLoaded,
    /// All subsystems are healthy — container is serving.
    Ready,
}

impl BootPhase {
    /// Return the phase that follows `self`, or `None` if already at `Ready`.
    fn next(&self) -> Option<BootPhase> {
        match self {
            BootPhase::ConfigLoaded => Some(BootPhase::MemoryReady),
            BootPhase::MemoryReady => Some(BootPhase::SchedulerReady),
            BootPhase::SchedulerReady => Some(BootPhase::SecurityReady),
            BootPhase::SecurityReady => Some(BootPhase::NetworkReady),
            BootPhase::NetworkReady => Some(BootPhase::IpcReady),
            BootPhase::IpcReady => Some(BootPhase::RuntimeReady),
            BootPhase::RuntimeReady => Some(BootPhase::ModelsLoaded),
            BootPhase::ModelsLoaded => Some(BootPhase::Ready),
            BootPhase::Ready => None,
        }
    }

    /// Human-readable completion message for this phase.
    fn message(&self) -> &'static str {
        match self {
            BootPhase::ConfigLoaded => "Configuration loaded and validated",
            BootPhase::MemoryReady => "Memory subsystem initialized",
            BootPhase::SchedulerReady => "Scheduler started",
            BootPhase::SecurityReady => "Security subsystem initialized",
            BootPhase::NetworkReady => "Network stack ready",
            BootPhase::IpcReady => "IPC channels established",
            BootPhase::RuntimeReady => "ONNX runtime initialized",
            BootPhase::ModelsLoaded => "Models loaded into memory",
            BootPhase::Ready => "Container ready to serve",
        }
    }
}

// ---------------------------------------------------------------------------
// BootStatus
// ---------------------------------------------------------------------------

/// Snapshot produced after each successful phase transition.
#[derive(Clone, Debug, PartialEq)]
pub struct BootStatus {
    /// The phase that was just completed.
    pub phase: BootPhase,
    /// Human-readable description of what happened.
    pub message: String,
    /// Elapsed milliseconds since the boot sequence started.
    pub elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// BootSequence
// ---------------------------------------------------------------------------

/// Drives the ordered bring-up of every subsystem.
pub struct BootSequence {
    /// The container configuration driving this boot.
    config: ContainerConfig,
    /// Phases that have been successfully completed.
    phases_completed: Vec<BootPhase>,
    /// The phase the sequence is currently at.
    current_phase: BootPhase,
    /// Monotonic timestamp (ms) of sequence creation.
    start_time: u64,
    /// Accumulated error messages (non-fatal warnings).
    errors: Vec<String>,
}

impl BootSequence {
    /// Create a new boot sequence starting at [`BootPhase::ConfigLoaded`].
    pub fn new(config: ContainerConfig) -> Self {
        BootSequence {
            config,
            phases_completed: Vec::new(),
            current_phase: BootPhase::ConfigLoaded,
            start_time: 0,
            errors: Vec::new(),
        }
    }

    /// Advance to the next phase.
    ///
    /// Returns the [`BootStatus`] of the newly entered phase, or
    /// [`ContainerError::AlreadyReady`] if the sequence is already complete.
    pub fn advance(&mut self) -> Result<BootStatus, ContainerError> {
        let next = self
            .current_phase
            .next()
            .ok_or(ContainerError::AlreadyReady)?;

        // Record the current phase as completed before moving on.
        self.phases_completed.push(self.current_phase.clone());

        let message = String::from(next.message());
        self.current_phase = next.clone();

        // Simulate a tiny elapsed-time increment (real boot would read a
        // monotonic clock).
        let elapsed_ms = (self.phases_completed.len() as u64) * 10;

        Ok(BootStatus {
            phase: next,
            message,
            elapsed_ms,
        })
    }

    /// The current phase of the boot sequence.
    pub fn phase(&self) -> &BootPhase {
        &self.current_phase
    }

    /// Returns `true` when every phase has completed and the container is
    /// ready to serve traffic.
    pub fn is_ready(&self) -> bool {
        self.current_phase == BootPhase::Ready
    }

    /// Phases that have been successfully completed so far.
    pub fn completed_phases(&self) -> &[BootPhase] {
        &self.phases_completed
    }

    /// Drive the boot sequence from its current position all the way through
    /// to [`BootPhase::Ready`], collecting every status along the way.
    pub fn run_all(&mut self) -> Result<Vec<BootStatus>, ContainerError> {
        let mut statuses = Vec::new();
        while self.current_phase != BootPhase::Ready {
            statuses.push(self.advance()?);
        }
        Ok(statuses)
    }

    /// Record a non-fatal error message.
    pub fn add_error(&mut self, msg: String) {
        self.errors.push(msg);
    }

    /// Whether any errors have been recorded.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Number of recorded errors.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ContainerConfig;

    fn default_seq() -> BootSequence {
        BootSequence::new(ContainerConfig::default())
    }

    // -- Initial state -----------------------------------------------------

    #[test]
    fn test_new_starts_at_config_loaded() {
        let seq = default_seq();
        assert_eq!(*seq.phase(), BootPhase::ConfigLoaded);
    }

    #[test]
    fn test_new_not_ready() {
        let seq = default_seq();
        assert!(!seq.is_ready());
    }

    #[test]
    fn test_new_no_completed_phases() {
        let seq = default_seq();
        assert!(seq.completed_phases().is_empty());
    }

    // -- Advance -----------------------------------------------------------

    #[test]
    fn test_advance_moves_to_memory_ready() {
        let mut seq = default_seq();
        let status = seq.advance().unwrap();
        assert_eq!(status.phase, BootPhase::MemoryReady);
        assert_eq!(*seq.phase(), BootPhase::MemoryReady);
    }

    #[test]
    fn test_advance_through_all_phases() {
        let mut seq = default_seq();
        let expected = [
            BootPhase::MemoryReady,
            BootPhase::SchedulerReady,
            BootPhase::SecurityReady,
            BootPhase::NetworkReady,
            BootPhase::IpcReady,
            BootPhase::RuntimeReady,
            BootPhase::ModelsLoaded,
            BootPhase::Ready,
        ];
        for expected_phase in &expected {
            let status = seq.advance().unwrap();
            assert_eq!(status.phase, *expected_phase);
        }
        assert!(seq.is_ready());
    }

    #[test]
    fn test_phase_order_correct() {
        let mut seq = default_seq();
        // Advance all the way and collect completed phases.
        seq.run_all().unwrap();
        let completed = seq.completed_phases();
        let expected = [
            BootPhase::ConfigLoaded,
            BootPhase::MemoryReady,
            BootPhase::SchedulerReady,
            BootPhase::SecurityReady,
            BootPhase::NetworkReady,
            BootPhase::IpcReady,
            BootPhase::RuntimeReady,
            BootPhase::ModelsLoaded,
        ];
        assert_eq!(completed, &expected);
    }

    #[test]
    fn test_is_ready_after_all_phases() {
        let mut seq = default_seq();
        seq.run_all().unwrap();
        assert!(seq.is_ready());
    }

    // -- run_all -----------------------------------------------------------

    #[test]
    fn test_run_all_completes_all_phases() {
        let mut seq = default_seq();
        let statuses = seq.run_all().unwrap();
        assert_eq!(statuses.len(), 8); // 8 transitions to reach Ready
        assert_eq!(statuses.last().unwrap().phase, BootPhase::Ready);
    }

    // -- Advance past Ready ------------------------------------------------

    #[test]
    fn test_advance_past_ready_returns_error() {
        let mut seq = default_seq();
        seq.run_all().unwrap();
        assert_eq!(seq.advance(), Err(ContainerError::AlreadyReady));
    }

    // -- Error recording ---------------------------------------------------

    #[test]
    fn test_error_recording() {
        let mut seq = default_seq();
        assert!(!seq.has_errors());
        seq.add_error(String::from("something went wrong"));
        assert!(seq.has_errors());
    }

    #[test]
    fn test_has_errors_and_error_count() {
        let mut seq = default_seq();
        assert_eq!(seq.error_count(), 0);
        seq.add_error(String::from("err1"));
        seq.add_error(String::from("err2"));
        assert_eq!(seq.error_count(), 2);
        assert!(seq.has_errors());
    }

    // -- Completed phases tracking -----------------------------------------

    #[test]
    fn test_completed_phases_tracking() {
        let mut seq = default_seq();
        seq.advance().unwrap(); // → MemoryReady
        seq.advance().unwrap(); // → SchedulerReady
        assert_eq!(
            seq.completed_phases(),
            &[BootPhase::ConfigLoaded, BootPhase::MemoryReady]
        );
    }

    // -- Boot status messages ----------------------------------------------

    #[test]
    fn test_boot_status_messages() {
        let mut seq = default_seq();
        let s1 = seq.advance().unwrap();
        assert_eq!(s1.message, "Memory subsystem initialized");
        let s2 = seq.advance().unwrap();
        assert_eq!(s2.message, "Scheduler started");
    }

    // -- Phase enum variants -----------------------------------------------

    #[test]
    fn test_phase_enum_variants_count() {
        // There are 9 variants; advancing from ConfigLoaded takes 8 steps.
        let mut seq = default_seq();
        let statuses = seq.run_all().unwrap();
        assert_eq!(statuses.len(), 8);
    }

    // -- Elapsed time in status --------------------------------------------

    #[test]
    fn test_boot_status_elapsed_increases() {
        let mut seq = default_seq();
        let s1 = seq.advance().unwrap();
        let s2 = seq.advance().unwrap();
        assert!(s2.elapsed_ms > s1.elapsed_ms);
    }

    // -- Phase message for every variant -----------------------------------

    #[test]
    fn test_every_phase_has_message() {
        let phases = [
            BootPhase::ConfigLoaded,
            BootPhase::MemoryReady,
            BootPhase::SchedulerReady,
            BootPhase::SecurityReady,
            BootPhase::NetworkReady,
            BootPhase::IpcReady,
            BootPhase::RuntimeReady,
            BootPhase::ModelsLoaded,
            BootPhase::Ready,
        ];
        for p in &phases {
            assert!(!p.message().is_empty());
        }
    }
}
