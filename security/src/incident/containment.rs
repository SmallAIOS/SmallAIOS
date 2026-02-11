// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Automated containment actions for incident response.
//!
//! Supports:
//! - Capability revocation for compromised tasks
//! - Task termination requests
//! - Network connection reset
//! - Inference request rejection
//!
//! Actions are recorded as audit events and published as incident events.

use alloc::vec::Vec;

/// Types of automated containment actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainmentAction {
    /// Revoke all capabilities held by a task.
    RevokeCapabilities { task_id: u64 },
    /// Terminate a task and release its resources.
    TerminateTask { task_id: u64 },
    /// Reset all network connections for a task or source.
    ResetNetworkConnections { task_id: u64 },
    /// Reject inference requests from a task.
    RejectInference { task_id: u64 },
}

impl ContainmentAction {
    /// The task ID affected by this action.
    pub fn task_id(&self) -> u64 {
        match self {
            Self::RevokeCapabilities { task_id }
            | Self::TerminateTask { task_id }
            | Self::ResetNetworkConnections { task_id }
            | Self::RejectInference { task_id } => *task_id,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RevokeCapabilities { .. } => "revoke-capabilities",
            Self::TerminateTask { .. } => "terminate-task",
            Self::ResetNetworkConnections { .. } => "reset-network",
            Self::RejectInference { .. } => "reject-inference",
        }
    }
}

/// Result of executing a containment action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainmentResult {
    /// Action executed successfully.
    Success {
        action: ContainmentAction,
        details: ContainmentDetails,
    },
    /// Action failed.
    Failed {
        action: ContainmentAction,
        reason: ContainmentError,
    },
}

/// Details about a successful containment action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainmentDetails {
    /// Number of capabilities revoked.
    CapabilitiesRevoked(u32),
    /// Task was terminated.
    TaskTerminated,
    /// Number of connections reset.
    ConnectionsReset(u32),
    /// Inference rejection activated.
    InferenceRejected,
}

/// Errors from containment actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainmentError {
    /// Task not found.
    TaskNotFound,
    /// Task already contained.
    AlreadyContained,
    /// Insufficient privileges to perform the action.
    InsufficientPrivileges,
}

/// Tracks which tasks are currently under containment.
///
/// Tasks under containment have their capabilities revoked and
/// new inference requests rejected.
pub struct ContainmentRegistry {
    /// Task IDs currently under containment.
    contained_tasks: [ContainedTask; MAX_CONTAINED],
    /// Total containment actions executed.
    total_actions: u64,
}

const MAX_CONTAINED: usize = 64;

#[derive(Debug, Clone, Copy)]
struct ContainedTask {
    task_id: u64,
    timestamp_ns: u64,
    active: bool,
}

impl ContainedTask {
    const fn empty() -> Self {
        Self {
            task_id: 0,
            timestamp_ns: 0,
            active: false,
        }
    }
}

impl ContainmentRegistry {
    pub fn new() -> Self {
        Self {
            contained_tasks: [ContainedTask::empty(); MAX_CONTAINED],
            total_actions: 0,
        }
    }

    /// Place a task under containment.
    ///
    /// Returns the list of containment actions that should be executed.
    pub fn contain_task(&mut self, task_id: u64, timestamp_ns: u64) -> Result<Vec<ContainmentAction>, ContainmentError> {
        if self.is_contained(task_id) {
            return Err(ContainmentError::AlreadyContained);
        }

        // Find a free slot
        let slot = self
            .contained_tasks
            .iter_mut()
            .find(|s| !s.active)
            .ok_or(ContainmentError::InsufficientPrivileges)?;

        *slot = ContainedTask {
            task_id,
            timestamp_ns,
            active: true,
        };
        self.total_actions += 1;

        // Return the standard containment action set
        Ok(alloc::vec![
            ContainmentAction::RevokeCapabilities { task_id },
            ContainmentAction::RejectInference { task_id },
            ContainmentAction::ResetNetworkConnections { task_id },
        ])
    }

    /// Release a task from containment (re-authorize).
    pub fn release_task(&mut self, task_id: u64) -> bool {
        for slot in self.contained_tasks.iter_mut() {
            if slot.active && slot.task_id == task_id {
                slot.active = false;
                return true;
            }
        }
        false
    }

    /// Check whether a task is currently under containment.
    pub fn is_contained(&self, task_id: u64) -> bool {
        self.contained_tasks
            .iter()
            .any(|s| s.active && s.task_id == task_id)
    }

    /// Number of currently contained tasks.
    pub fn contained_count(&self) -> usize {
        self.contained_tasks.iter().filter(|s| s.active).count()
    }

    /// Total containment actions executed since boot.
    pub fn total_actions(&self) -> u64 {
        self.total_actions
    }

    /// Get the containment timestamp for a task, if contained.
    pub fn containment_timestamp(&self, task_id: u64) -> Option<u64> {
        self.contained_tasks
            .iter()
            .find(|s| s.active && s.task_id == task_id)
            .map(|s| s.timestamp_ns)
    }

    /// List all currently contained task IDs.
    pub fn contained_task_ids(&self, buf: &mut [u64]) -> usize {
        let mut count = 0;
        for slot in &self.contained_tasks {
            if slot.active && count < buf.len() {
                buf[count] = slot.task_id;
                count += 1;
            }
        }
        count
    }
}

impl Default for ContainmentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_task_id() {
        let action = ContainmentAction::RevokeCapabilities { task_id: 42 };
        assert_eq!(action.task_id(), 42);
    }

    #[test]
    fn action_as_str() {
        assert_eq!(
            ContainmentAction::RevokeCapabilities { task_id: 0 }.as_str(),
            "revoke-capabilities"
        );
        assert_eq!(
            ContainmentAction::TerminateTask { task_id: 0 }.as_str(),
            "terminate-task"
        );
        assert_eq!(
            ContainmentAction::ResetNetworkConnections { task_id: 0 }.as_str(),
            "reset-network"
        );
        assert_eq!(
            ContainmentAction::RejectInference { task_id: 0 }.as_str(),
            "reject-inference"
        );
    }

    #[test]
    fn contain_task_success() {
        let mut reg = ContainmentRegistry::new();
        let actions = reg.contain_task(42, 1000).unwrap();
        assert_eq!(actions.len(), 3);
        assert!(reg.is_contained(42));
        assert_eq!(reg.contained_count(), 1);
        assert_eq!(reg.total_actions(), 1);
    }

    #[test]
    fn contain_task_already_contained() {
        let mut reg = ContainmentRegistry::new();
        reg.contain_task(42, 1000).unwrap();
        let result = reg.contain_task(42, 2000);
        assert_eq!(result, Err(ContainmentError::AlreadyContained));
    }

    #[test]
    fn release_task() {
        let mut reg = ContainmentRegistry::new();
        reg.contain_task(42, 1000).unwrap();
        assert!(reg.is_contained(42));
        assert!(reg.release_task(42));
        assert!(!reg.is_contained(42));
        assert_eq!(reg.contained_count(), 0);
    }

    #[test]
    fn release_nonexistent() {
        let mut reg = ContainmentRegistry::new();
        assert!(!reg.release_task(99));
    }

    #[test]
    fn multiple_tasks() {
        let mut reg = ContainmentRegistry::new();
        reg.contain_task(1, 100).unwrap();
        reg.contain_task(2, 200).unwrap();
        reg.contain_task(3, 300).unwrap();
        assert_eq!(reg.contained_count(), 3);
        assert!(reg.is_contained(1));
        assert!(reg.is_contained(2));
        assert!(reg.is_contained(3));
        assert!(!reg.is_contained(4));
    }

    #[test]
    fn list_contained_ids() {
        let mut reg = ContainmentRegistry::new();
        reg.contain_task(10, 100).unwrap();
        reg.contain_task(20, 200).unwrap();
        let mut buf = [0u64; 10];
        let count = reg.contained_task_ids(&mut buf);
        assert_eq!(count, 2);
        assert!(buf[..count].contains(&10));
        assert!(buf[..count].contains(&20));
    }

    #[test]
    fn contain_after_release_reuses_slot() {
        let mut reg = ContainmentRegistry::new();
        reg.contain_task(1, 100).unwrap();
        reg.release_task(1);
        // Should be able to contain a new task (or same task) in the freed slot
        reg.contain_task(1, 200).unwrap();
        assert!(reg.is_contained(1));
    }

    #[test]
    fn containment_result_success() {
        let result = ContainmentResult::Success {
            action: ContainmentAction::RevokeCapabilities { task_id: 1 },
            details: ContainmentDetails::CapabilitiesRevoked(5),
        };
        if let ContainmentResult::Success { details, .. } = result {
            assert_eq!(details, ContainmentDetails::CapabilitiesRevoked(5));
        }
    }
}
