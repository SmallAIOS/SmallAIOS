// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Formal verification status tracking.
//!
//! Tracks the status of TLA+, SPIN, Lean 4, fuzz testing, and static
//! analysis verification activities for DO-178C compliance.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

/// Maximum number of verification targets.
pub const MAX_TARGETS: usize = 256;

/// Formal verification tool used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationTool {
    /// TLA+ model checker for concurrency properties.
    TlaPlus,
    /// SPIN model checker for protocol verification.
    Spin,
    /// Lean 4 theorem prover for type-level properties.
    Lean4,
    /// Fuzz testing for input space exploration.
    Fuzz,
    /// Static analysis tools (clippy, miri, etc.).
    StaticAnalysis,
}

/// Status of a verification target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationStatus {
    /// Verification has not been started.
    NotStarted,
    /// Verification is currently in progress.
    InProgress,
    /// Verification passed: all properties hold.
    Passed,
    /// Verification failed with a description of the failure.
    Failed(String),
    /// Verification was skipped (e.g., not applicable at current DAL).
    Skipped,
}

/// A single verification target (a property or subsystem being verified).
#[derive(Debug, Clone)]
pub struct VerificationTarget {
    /// Name of the verification target.
    pub name: String,
    /// Tool used for verification.
    pub tool: VerificationTool,
    /// Current verification status.
    pub status: VerificationStatus,
    /// Number of properties checked.
    pub properties_checked: u32,
    /// Number of properties that passed.
    pub properties_passed: u32,
    /// Duration of last verification run in milliseconds.
    pub last_run_ms: u64,
}

/// Summary of all verification targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationSummary {
    /// Total number of registered targets.
    pub total: usize,
    /// Number of targets that passed.
    pub passed: usize,
    /// Number of targets that failed.
    pub failed: usize,
    /// Number of targets currently in progress.
    pub in_progress: usize,
    /// Number of targets not yet started.
    pub not_started: usize,
}

/// Errors that can occur in verification operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationError {
    /// A target with this name already exists.
    DuplicateTarget,
    /// The referenced target was not found.
    NotFound,
    /// The target registry is full.
    TableFull,
}

/// Registry of formal verification targets and their status.
///
/// Tracks all verification activities (TLA+, SPIN, Lean 4, fuzzing,
/// static analysis) and provides summary reporting.
pub struct VerificationRegistry {
    /// All registered verification targets.
    targets: Vec<VerificationTarget>,
}

impl VerificationRegistry {
    /// Creates a new empty verification registry.
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
        }
    }

    /// Adds a new verification target.
    ///
    /// The target starts with `NotStarted` status and zero properties.
    pub fn add_target(
        &mut self,
        name: &str,
        tool: VerificationTool,
    ) -> Result<(), VerificationError> {
        if self.targets.len() >= MAX_TARGETS {
            return Err(VerificationError::TableFull);
        }
        if self.targets.iter().any(|t| t.name == name) {
            return Err(VerificationError::DuplicateTarget);
        }
        self.targets.push(VerificationTarget {
            name: String::from(name),
            tool,
            status: VerificationStatus::NotStarted,
            properties_checked: 0,
            properties_passed: 0,
            last_run_ms: 0,
        });
        Ok(())
    }

    /// Updates the status and property counts of a verification target.
    pub fn update_status(
        &mut self,
        name: &str,
        status: VerificationStatus,
        props_checked: u32,
        props_passed: u32,
    ) -> Result<(), VerificationError> {
        let target = self
            .targets
            .iter_mut()
            .find(|t| t.name == name)
            .ok_or(VerificationError::NotFound)?;
        target.status = status;
        target.properties_checked = props_checked;
        target.properties_passed = props_passed;
        Ok(())
    }

    /// Retrieves a verification target by name.
    pub fn get_target(&self, name: &str) -> Option<&VerificationTarget> {
        self.targets.iter().find(|t| t.name == name)
    }

    /// Returns the total number of registered targets.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Checks whether all registered targets have passed verification.
    pub fn all_passed(&self) -> bool {
        !self.targets.is_empty()
            && self
                .targets
                .iter()
                .all(|t| t.status == VerificationStatus::Passed)
    }

    /// Returns the number of targets with Passed status.
    pub fn passed_count(&self) -> usize {
        self.targets
            .iter()
            .filter(|t| t.status == VerificationStatus::Passed)
            .count()
    }

    /// Returns all targets with Failed status.
    pub fn failed_targets(&self) -> Vec<&VerificationTarget> {
        self.targets
            .iter()
            .filter(|t| matches!(t.status, VerificationStatus::Failed(_)))
            .collect()
    }

    /// Registers the default set of verification targets.
    ///
    /// Adds 6 standard verification targets covering the major kernel
    /// subsystems and their appropriate verification tools.
    pub fn register_defaults(&mut self) -> Result<(), VerificationError> {
        self.add_target("buddy_allocator", VerificationTool::TlaPlus)?;
        self.add_target("scheduler_fairness", VerificationTool::TlaPlus)?;
        self.add_target("tcp_state_machine", VerificationTool::Spin)?;
        self.add_target("pubsub_routing", VerificationTool::Spin)?;
        self.add_target("capability_nonforgery", VerificationTool::Lean4)?;
        self.add_target("syscall_fuzz", VerificationTool::Fuzz)?;
        Ok(())
    }

    /// Produces a summary of all verification target statuses.
    pub fn summary(&self) -> VerificationSummary {
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut in_progress = 0usize;
        let mut not_started = 0usize;

        for target in &self.targets {
            match &target.status {
                VerificationStatus::Passed => passed += 1,
                VerificationStatus::Failed(_) => failed += 1,
                VerificationStatus::InProgress => in_progress += 1,
                VerificationStatus::NotStarted => not_started += 1,
                VerificationStatus::Skipped => {}
            }
        }

        VerificationSummary {
            total: self.targets.len(),
            passed,
            failed,
            in_progress,
            not_started,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_target() {
        let mut reg = VerificationRegistry::new();
        let result = reg.add_target("test_model", VerificationTool::TlaPlus);
        assert!(result.is_ok());
        assert_eq!(reg.target_count(), 1);
    }

    #[test]
    fn test_update_status() {
        let mut reg = VerificationRegistry::new();
        reg.add_target("model", VerificationTool::TlaPlus).unwrap();
        reg.update_status("model", VerificationStatus::Passed, 10, 10)
            .unwrap();
        let target = reg.get_target("model").unwrap();
        assert_eq!(target.status, VerificationStatus::Passed);
        assert_eq!(target.properties_checked, 10);
        assert_eq!(target.properties_passed, 10);
    }

    #[test]
    fn test_get_target() {
        let mut reg = VerificationRegistry::new();
        reg.add_target("buddy", VerificationTool::TlaPlus).unwrap();
        let target = reg.get_target("buddy");
        assert!(target.is_some());
        let target = target.unwrap();
        assert_eq!(target.name, "buddy");
        assert_eq!(target.tool, VerificationTool::TlaPlus);
        assert_eq!(target.status, VerificationStatus::NotStarted);
    }

    #[test]
    fn test_get_target_not_found() {
        let reg = VerificationRegistry::new();
        assert!(reg.get_target("nonexistent").is_none());
    }

    #[test]
    fn test_all_passed() {
        let mut reg = VerificationRegistry::new();
        reg.add_target("a", VerificationTool::TlaPlus).unwrap();
        reg.add_target("b", VerificationTool::Spin).unwrap();
        assert!(!reg.all_passed());

        reg.update_status("a", VerificationStatus::Passed, 5, 5)
            .unwrap();
        assert!(!reg.all_passed());

        reg.update_status("b", VerificationStatus::Passed, 3, 3)
            .unwrap();
        assert!(reg.all_passed());
    }

    #[test]
    fn test_all_passed_empty_is_false() {
        let reg = VerificationRegistry::new();
        assert!(!reg.all_passed());
    }

    #[test]
    fn test_failed_targets() {
        let mut reg = VerificationRegistry::new();
        reg.add_target("a", VerificationTool::TlaPlus).unwrap();
        reg.add_target("b", VerificationTool::Spin).unwrap();
        reg.update_status("a", VerificationStatus::Passed, 5, 5)
            .unwrap();
        reg.update_status(
            "b",
            VerificationStatus::Failed(String::from("deadlock found")),
            3,
            2,
        )
        .unwrap();

        let failed = reg.failed_targets();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].name, "b");
    }

    #[test]
    fn test_register_defaults_creates_6() {
        let mut reg = VerificationRegistry::new();
        reg.register_defaults().unwrap();
        assert_eq!(reg.target_count(), 6);

        // Verify specific targets exist.
        assert!(reg.get_target("buddy_allocator").is_some());
        assert!(reg.get_target("scheduler_fairness").is_some());
        assert!(reg.get_target("tcp_state_machine").is_some());
        assert!(reg.get_target("pubsub_routing").is_some());
        assert!(reg.get_target("capability_nonforgery").is_some());
        assert!(reg.get_target("syscall_fuzz").is_some());
    }

    #[test]
    fn test_passed_count() {
        let mut reg = VerificationRegistry::new();
        reg.add_target("a", VerificationTool::TlaPlus).unwrap();
        reg.add_target("b", VerificationTool::Spin).unwrap();
        reg.add_target("c", VerificationTool::Lean4).unwrap();
        assert_eq!(reg.passed_count(), 0);

        reg.update_status("a", VerificationStatus::Passed, 5, 5)
            .unwrap();
        reg.update_status("b", VerificationStatus::Passed, 3, 3)
            .unwrap();
        assert_eq!(reg.passed_count(), 2);
    }

    #[test]
    fn test_summary() {
        let mut reg = VerificationRegistry::new();
        reg.add_target("a", VerificationTool::TlaPlus).unwrap();
        reg.add_target("b", VerificationTool::Spin).unwrap();
        reg.add_target("c", VerificationTool::Lean4).unwrap();
        reg.add_target("d", VerificationTool::Fuzz).unwrap();

        reg.update_status("a", VerificationStatus::Passed, 5, 5)
            .unwrap();
        reg.update_status(
            "b",
            VerificationStatus::Failed(String::from("error")),
            3,
            1,
        )
        .unwrap();
        reg.update_status("c", VerificationStatus::InProgress, 0, 0)
            .unwrap();
        // "d" stays NotStarted.

        let summary = reg.summary();
        assert_eq!(summary.total, 4);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.in_progress, 1);
        assert_eq!(summary.not_started, 1);
    }

    #[test]
    fn test_duplicate_error() {
        let mut reg = VerificationRegistry::new();
        reg.add_target("model", VerificationTool::TlaPlus).unwrap();
        let result = reg.add_target("model", VerificationTool::Spin);
        assert_eq!(result, Err(VerificationError::DuplicateTarget));
    }

    #[test]
    fn test_not_found_error() {
        let mut reg = VerificationRegistry::new();
        let result = reg.update_status("nonexistent", VerificationStatus::Passed, 0, 0);
        assert_eq!(result, Err(VerificationError::NotFound));
    }

    #[test]
    fn test_table_full() {
        let mut reg = VerificationRegistry::new();
        for i in 0..MAX_TARGETS {
            reg.add_target(
                &alloc::format!("target-{}", i),
                VerificationTool::TlaPlus,
            )
            .unwrap();
        }
        let result = reg.add_target("overflow", VerificationTool::Spin);
        assert_eq!(result, Err(VerificationError::TableFull));
    }

    #[test]
    fn test_tool_enum_variants() {
        let mut reg = VerificationRegistry::new();
        reg.add_target("t1", VerificationTool::TlaPlus).unwrap();
        reg.add_target("t2", VerificationTool::Spin).unwrap();
        reg.add_target("t3", VerificationTool::Lean4).unwrap();
        reg.add_target("t4", VerificationTool::Fuzz).unwrap();
        reg.add_target("t5", VerificationTool::StaticAnalysis)
            .unwrap();

        assert_eq!(reg.get_target("t1").unwrap().tool, VerificationTool::TlaPlus);
        assert_eq!(reg.get_target("t2").unwrap().tool, VerificationTool::Spin);
        assert_eq!(reg.get_target("t3").unwrap().tool, VerificationTool::Lean4);
        assert_eq!(reg.get_target("t4").unwrap().tool, VerificationTool::Fuzz);
        assert_eq!(
            reg.get_target("t5").unwrap().tool,
            VerificationTool::StaticAnalysis
        );
    }
}
