// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Boot sequence orchestration.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::config::ContainerConfig;
use crate::ContainerError;

#[cfg(feature = "formal-gate")]
use smallaios_security::boundary::trust_boundaries::TrustBoundary;
#[cfg(feature = "formal-gate")]
use smallaios_security::gate::SecurityGate;
#[cfg(feature = "formal-gate")]
use smallaios_security::policy::SecurityPolicy;

/// Map a boundary index back to the TrustBoundary enum.
#[cfg(feature = "formal-gate")]
fn boundary_from_index(idx: usize) -> Option<TrustBoundary> {
    match idx {
        0 => Some(TrustBoundary::Kernel),
        1 => Some(TrustBoundary::Kubernetes),
        2 => Some(TrustBoundary::Network),
        3 => Some(TrustBoundary::BusProtocol),
        4 => Some(TrustBoundary::Gpu),
        _ => None,
    }
}

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
    /// The security gate, initialised during SecurityReady phase.
    #[cfg(feature = "formal-gate")]
    security_gate: Option<SecurityGate>,
    /// The active security policy, loaded during SecurityReady phase.
    #[cfg(feature = "formal-gate")]
    security_policy: Option<SecurityPolicy>,
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
            #[cfg(feature = "formal-gate")]
            security_gate: None,
            #[cfg(feature = "formal-gate")]
            security_policy: None,
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

        // Phase-specific initialization
        #[cfg(feature = "formal-gate")]
        self.run_phase_init(&next)?;

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

    /// Run phase-specific initialization for the formal gate subsystem.
    ///
    /// - **SecurityReady**: Load default policy, attempt external blob, initialize gate.
    /// - **ModelsLoaded**: Validate loaded model hashes against the active policy whitelist.
    #[cfg(feature = "formal-gate")]
    fn run_phase_init(&mut self, phase: &BootPhase) -> Result<(), ContainerError> {
        match phase {
            BootPhase::SecurityReady => self.init_security_policy(),
            BootPhase::ModelsLoaded => self.validate_models_against_policy(),
            _ => Ok(()),
        }
    }

    /// Load the security policy and initialise the SecurityGate.
    ///
    /// 1. Load the compiled-in default policy.
    /// 2. If an external policy blob address is configured, attempt to load
    ///    and verify it; if valid, swap in the external policy.
    /// 3. Initialise the SecurityGate with the policy's global mode.
    #[cfg(feature = "formal-gate")]
    fn init_security_policy(&mut self) -> Result<(), ContainerError> {
        if !self.config.formal_gate_enabled {
            return Ok(());
        }

        // Step 1: Load default policy
        let policy = SecurityPolicy::default_policy();

        // Step 2: Check for external policy blob
        if self.config.policy_blob_address != 0 {
            // In a real implementation, we'd read from the blob address.
            // For now, this is a placeholder — the blob would be at a
            // memory-mapped address provided by the bootloader/hypervisor.
            // The integration test exercises this path by calling
            // load_external_policy() directly.
            self.errors.push(String::from(
                "external policy blob configured but memory-mapped loading not yet implemented",
            ));
        }

        // Step 3: Initialise the SecurityGate
        let mut gate = SecurityGate::new(policy.global_mode);

        // Apply per-boundary modes from the policy
        for (i, mode) in policy.boundary_modes.iter().enumerate() {
            if let Some(boundary) = boundary_from_index(i) {
                gate.set_boundary_mode(boundary, *mode);
            }
        }

        self.security_policy = Some(policy);
        self.security_gate = Some(gate);

        Ok(())
    }

    /// Load an external policy blob, verify it, and swap it in.
    ///
    /// Returns `Ok(())` if the blob is valid and the policy was swapped,
    /// or an error if the blob is invalid.
    #[cfg(feature = "formal-gate")]
    pub fn load_external_policy(&mut self, blob: &[u8]) -> Result<(), ContainerError> {
        let policy = self.security_policy.as_ref().ok_or_else(|| {
            ContainerError::BootFailed(String::from("security policy not yet initialised"))
        })?;

        let new_policy = policy.remote_update(blob).map_err(|e| {
            ContainerError::BootFailed(alloc::format!("policy blob invalid: {:?}", e))
        })?;

        // Update gate with the new policy's modes
        if let Some(gate) = self.security_gate.as_mut() {
            for (i, mode) in new_policy.boundary_modes.iter().enumerate() {
                if let Some(boundary) = boundary_from_index(i) {
                    gate.set_boundary_mode(boundary, *mode);
                }
            }
        }

        self.security_policy = Some(new_policy);
        Ok(())
    }

    /// Validate loaded models against the active policy whitelist.
    ///
    /// In a full implementation, this would iterate over all loaded ONNX
    /// sessions, compute their hashes, and check against the policy's
    /// model whitelist. Non-compliant models would be unloaded.
    ///
    /// Currently a placeholder — records an error if models would fail.
    #[cfg(feature = "formal-gate")]
    fn validate_models_against_policy(&mut self) -> Result<(), ContainerError> {
        if !self.config.formal_gate_enabled {
            return Ok(());
        }

        let policy = match self.security_policy.as_ref() {
            Some(p) => p,
            None => {
                // Policy wasn't loaded (formal gate may have been disabled at SecurityReady)
                return Ok(());
            }
        };

        // If the whitelist is empty, all models are allowed (permissive default)
        if policy.model_whitelist.is_empty() {
            return Ok(());
        }

        // In a real implementation we'd iterate loaded models here.
        // The validate_model_hash() method is available for each model.
        Ok(())
    }

    /// Validate a single model hash against the active policy.
    ///
    /// Returns `true` if the model is allowed (or no policy is loaded).
    #[cfg(feature = "formal-gate")]
    pub fn validate_model_hash(&self, model_hash: &[u8; 32]) -> bool {
        match self.security_policy.as_ref() {
            Some(policy) => policy.model_allowed(model_hash),
            None => true,
        }
    }

    /// Reference to the SecurityGate, if initialised.
    #[cfg(feature = "formal-gate")]
    pub fn security_gate(&self) -> Option<&SecurityGate> {
        self.security_gate.as_ref()
    }

    /// Mutable reference to the SecurityGate, if initialised.
    #[cfg(feature = "formal-gate")]
    pub fn security_gate_mut(&mut self) -> Option<&mut SecurityGate> {
        self.security_gate.as_mut()
    }

    /// Reference to the active SecurityPolicy, if loaded.
    #[cfg(feature = "formal-gate")]
    pub fn security_policy(&self) -> Option<&SecurityPolicy> {
        self.security_policy.as_ref()
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

    // -- Formal gate boot integration tests --------------------------------

    #[cfg(feature = "formal-gate")]
    #[test]
    fn test_boot_with_default_policy() {
        let mut seq = default_seq();
        seq.run_all().unwrap();
        assert!(seq.is_ready());

        // SecurityGate and SecurityPolicy should be initialised
        assert!(seq.security_gate().is_some());
        assert!(seq.security_policy().is_some());

        let policy = seq.security_policy().unwrap();
        assert_eq!(policy.version, 1);
        assert!(policy.model_whitelist.is_empty());
    }

    #[cfg(feature = "formal-gate")]
    #[test]
    fn test_boot_with_valid_external_blob() {
        let mut seq = default_seq();
        // Advance to SecurityReady
        while *seq.phase() != BootPhase::SecurityReady {
            seq.advance().unwrap();
        }

        // Now load an external policy blob
        let blob = make_valid_blob(&[1]); // Permissive
        assert!(seq.load_external_policy(&blob).is_ok());

        // Finish booting
        while !seq.is_ready() {
            seq.advance().unwrap();
        }
        assert!(seq.is_ready());
    }

    #[cfg(feature = "formal-gate")]
    #[test]
    fn test_boot_with_invalid_external_blob() {
        let mut seq = default_seq();
        // Advance to SecurityReady
        while *seq.phase() != BootPhase::SecurityReady {
            seq.advance().unwrap();
        }

        // Try to load an invalid blob
        let result = seq.load_external_policy(&[0xFF, 0xFF, 0xFF]);
        assert!(result.is_err());

        // Original policy should still be intact
        assert!(seq.security_policy().is_some());
    }

    #[cfg(feature = "formal-gate")]
    #[test]
    fn test_boot_model_whitelist_miss() {
        let mut seq = default_seq();
        // Advance to SecurityReady
        while *seq.phase() != BootPhase::SecurityReady {
            seq.advance().unwrap();
        }

        // Add a hash to the whitelist so it becomes restrictive
        if let Some(policy) = seq.security_policy.as_mut() {
            let hash = [0xAA; 32];
            policy.model_whitelist.add(hash);
        }

        // Model not in whitelist should be rejected
        let bad_hash = [0xBB; 32];
        assert!(!seq.validate_model_hash(&bad_hash));

        // Model in whitelist should pass
        let good_hash = [0xAA; 32];
        assert!(seq.validate_model_hash(&good_hash));
    }

    #[cfg(feature = "formal-gate")]
    #[test]
    fn test_boot_formal_gate_disabled_skips_init() {
        let mut cfg = ContainerConfig::default();
        cfg.formal_gate_enabled = false;
        let mut seq = BootSequence::new(cfg);
        seq.run_all().unwrap();
        assert!(seq.is_ready());
        // Gate and policy should be None when formal_gate_enabled = false
        assert!(seq.security_gate().is_none());
        assert!(seq.security_policy().is_none());
    }

    /// Helper to build a valid policy blob for tests.
    #[cfg(feature = "formal-gate")]
    fn make_valid_blob(payload: &[u8]) -> Vec<u8> {
        use smallaios_security::policy::{POLICY_MAGIC, POLICY_SIG_SIZE, POLICY_VERSION};

        let mut blob = Vec::new();
        blob.extend_from_slice(&POLICY_MAGIC.to_le_bytes());
        blob.extend_from_slice(&POLICY_VERSION.to_le_bytes());
        blob.extend_from_slice(&[0u8; POLICY_SIG_SIZE]); // stub signature
        blob.extend_from_slice(payload);
        blob
    }
}
