// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Post-incident review data model.
//!
//! Provides structured templates for root cause analysis,
//! corrective action tracking, and lessons learned per NIST SP 800-61.

use super::event::IncidentSeverity;
use alloc::string::String;
use alloc::vec::Vec;

/// Maximum corrective actions per review.
const MAX_CORRECTIVE_ACTIONS: usize = 16;

/// Maximum lessons learned per review.
const MAX_LESSONS: usize = 8;

/// Root cause category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RootCauseCategory {
    /// Software defect (bug in kernel, driver, etc.).
    SoftwareDefect = 0,
    /// Configuration error (misconfigured threshold, policy).
    ConfigurationError = 1,
    /// Hardware failure (memory, network interface, etc.).
    HardwareFailure = 2,
    /// External attack (capability bypass attempt, DoS).
    ExternalAttack = 3,
    /// Operational error (incorrect procedure followed).
    OperationalError = 4,
    /// Design weakness (inadequate boundary enforcement).
    DesignWeakness = 5,
    /// Unknown / under investigation.
    Unknown = 6,
}

impl RootCauseCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SoftwareDefect => "software-defect",
            Self::ConfigurationError => "configuration-error",
            Self::HardwareFailure => "hardware-failure",
            Self::ExternalAttack => "external-attack",
            Self::OperationalError => "operational-error",
            Self::DesignWeakness => "design-weakness",
            Self::Unknown => "unknown",
        }
    }
}

/// Corrective action status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionStatus {
    /// Action identified but not started.
    Planned = 0,
    /// Action in progress.
    InProgress = 1,
    /// Action completed and verified.
    Completed = 2,
    /// Action cancelled (superseded or no longer needed).
    Cancelled = 3,
}

impl ActionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::InProgress => "in-progress",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled)
    }
}

/// A corrective action from a post-incident review.
#[derive(Debug, Clone)]
pub struct CorrectiveAction {
    /// Action identifier (sequential within the review).
    pub id: u16,
    /// Description of the corrective action.
    pub description: String,
    /// Current status.
    pub status: ActionStatus,
    /// Responsible party / owner.
    pub owner: String,
    /// Target completion timestamp (nanoseconds, 0 = no deadline).
    pub deadline_ns: u64,
}

/// A lesson learned entry.
#[derive(Debug, Clone)]
pub struct LessonLearned {
    /// What was observed.
    pub observation: String,
    /// What should change.
    pub recommendation: String,
    /// Whether this has been integrated into standard procedures.
    pub integrated: bool,
}

/// Post-incident review record.
#[derive(Debug, Clone)]
pub struct PostIncidentReview {
    /// Associated incident ID.
    pub incident_id: u64,
    /// Incident severity at time of resolution.
    pub severity: IncidentSeverity,
    /// Incident detection timestamp (nanoseconds).
    pub detected_ns: u64,
    /// Incident resolution timestamp (nanoseconds).
    pub resolved_ns: u64,
    /// Root cause category.
    pub root_cause: RootCauseCategory,
    /// Detailed root cause description.
    pub root_cause_description: String,
    /// Corrective actions.
    pub corrective_actions: Vec<CorrectiveAction>,
    /// Lessons learned.
    pub lessons_learned: Vec<LessonLearned>,
    /// Review completion timestamp (0 = not completed).
    pub review_completed_ns: u64,
}

impl PostIncidentReview {
    /// Create a new review for an incident.
    pub fn new(
        incident_id: u64,
        severity: IncidentSeverity,
        detected_ns: u64,
        resolved_ns: u64,
    ) -> Self {
        Self {
            incident_id,
            severity,
            detected_ns,
            resolved_ns,
            root_cause: RootCauseCategory::Unknown,
            root_cause_description: String::new(),
            corrective_actions: Vec::new(),
            lessons_learned: Vec::new(),
            review_completed_ns: 0,
        }
    }

    /// Set the root cause.
    pub fn set_root_cause(&mut self, category: RootCauseCategory, description: String) {
        self.root_cause = category;
        self.root_cause_description = description;
    }

    /// Add a corrective action.
    pub fn add_corrective_action(
        &mut self,
        description: String,
        owner: String,
        deadline_ns: u64,
    ) -> Option<u16> {
        if self.corrective_actions.len() >= MAX_CORRECTIVE_ACTIONS {
            return None;
        }
        let id = self.corrective_actions.len() as u16;
        self.corrective_actions.push(CorrectiveAction {
            id,
            description,
            status: ActionStatus::Planned,
            owner,
            deadline_ns,
        });
        Some(id)
    }

    /// Update the status of a corrective action.
    pub fn update_action_status(&mut self, action_id: u16, status: ActionStatus) -> bool {
        if let Some(action) = self
            .corrective_actions
            .iter_mut()
            .find(|a| a.id == action_id)
        {
            action.status = status;
            true
        } else {
            false
        }
    }

    /// Add a lesson learned.
    pub fn add_lesson(&mut self, observation: String, recommendation: String) -> bool {
        if self.lessons_learned.len() >= MAX_LESSONS {
            return false;
        }
        self.lessons_learned.push(LessonLearned {
            observation,
            recommendation,
            integrated: false,
        });
        true
    }

    /// Mark the review as completed.
    pub fn complete(&mut self, now_ns: u64) {
        self.review_completed_ns = now_ns;
    }

    /// Whether the review is completed.
    pub fn is_completed(&self) -> bool {
        self.review_completed_ns > 0
    }

    /// Time to resolution (nanoseconds).
    pub fn time_to_resolution_ns(&self) -> u64 {
        self.resolved_ns.saturating_sub(self.detected_ns)
    }

    /// Count of open (non-terminal) corrective actions.
    pub fn open_actions_count(&self) -> usize {
        self.corrective_actions
            .iter()
            .filter(|a| !a.status.is_terminal())
            .count()
    }

    /// Count of completed corrective actions.
    pub fn completed_actions_count(&self) -> usize {
        self.corrective_actions
            .iter()
            .filter(|a| a.status == ActionStatus::Completed)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_review() {
        let review = PostIncidentReview::new(1, IncidentSeverity::High, 1_000_000, 2_000_000);
        assert_eq!(review.incident_id, 1);
        assert_eq!(review.severity, IncidentSeverity::High);
        assert_eq!(review.root_cause, RootCauseCategory::Unknown);
        assert!(!review.is_completed());
    }

    #[test]
    fn set_root_cause() {
        let mut review = PostIncidentReview::new(1, IncidentSeverity::High, 0, 100);
        review.set_root_cause(
            RootCauseCategory::SoftwareDefect,
            String::from("Buffer overflow in parser"),
        );
        assert_eq!(review.root_cause, RootCauseCategory::SoftwareDefect);
        assert_eq!(review.root_cause_description, "Buffer overflow in parser");
    }

    #[test]
    fn add_corrective_actions() {
        let mut review = PostIncidentReview::new(1, IncidentSeverity::High, 0, 100);
        let id = review.add_corrective_action(
            String::from("Fix parser bounds check"),
            String::from("dev-team"),
            500_000,
        );
        assert_eq!(id, Some(0));
        assert_eq!(review.corrective_actions.len(), 1);
        assert_eq!(review.corrective_actions[0].status, ActionStatus::Planned);
    }

    #[test]
    fn update_action_status() {
        let mut review = PostIncidentReview::new(1, IncidentSeverity::High, 0, 100);
        review.add_corrective_action(String::from("Fix bug"), String::from("dev"), 0);
        assert!(review.update_action_status(0, ActionStatus::InProgress));
        assert_eq!(
            review.corrective_actions[0].status,
            ActionStatus::InProgress
        );

        assert!(review.update_action_status(0, ActionStatus::Completed));
        assert_eq!(review.corrective_actions[0].status, ActionStatus::Completed);
    }

    #[test]
    fn update_nonexistent_action() {
        let mut review = PostIncidentReview::new(1, IncidentSeverity::High, 0, 100);
        assert!(!review.update_action_status(99, ActionStatus::Completed));
    }

    #[test]
    fn corrective_action_limit() {
        let mut review = PostIncidentReview::new(1, IncidentSeverity::High, 0, 100);
        for i in 0..MAX_CORRECTIVE_ACTIONS {
            assert!(review
                .add_corrective_action(alloc::format!("action-{}", i), String::from("owner"), 0,)
                .is_some());
        }
        assert!(review
            .add_corrective_action(String::from("overflow"), String::from("owner"), 0)
            .is_none());
    }

    #[test]
    fn add_lessons() {
        let mut review = PostIncidentReview::new(1, IncidentSeverity::High, 0, 100);
        assert!(review.add_lesson(
            String::from("Parsing was not bounds-checked"),
            String::from("Add bounds checking to all parsers"),
        ));
        assert_eq!(review.lessons_learned.len(), 1);
        assert!(!review.lessons_learned[0].integrated);
    }

    #[test]
    fn lesson_limit() {
        let mut review = PostIncidentReview::new(1, IncidentSeverity::High, 0, 100);
        for _ in 0..MAX_LESSONS {
            assert!(review.add_lesson(String::from("obs"), String::from("rec")));
        }
        assert!(!review.add_lesson(String::from("overflow"), String::from("rec")));
    }

    #[test]
    fn complete_review() {
        let mut review = PostIncidentReview::new(1, IncidentSeverity::High, 0, 100);
        assert!(!review.is_completed());
        review.complete(200);
        assert!(review.is_completed());
        assert_eq!(review.review_completed_ns, 200);
    }

    #[test]
    fn time_to_resolution() {
        let review = PostIncidentReview::new(1, IncidentSeverity::High, 1_000, 5_000);
        assert_eq!(review.time_to_resolution_ns(), 4_000);
    }

    #[test]
    fn open_and_completed_counts() {
        let mut review = PostIncidentReview::new(1, IncidentSeverity::High, 0, 100);
        review.add_corrective_action(String::from("a"), String::from("o"), 0);
        review.add_corrective_action(String::from("b"), String::from("o"), 0);
        review.add_corrective_action(String::from("c"), String::from("o"), 0);

        assert_eq!(review.open_actions_count(), 3);
        assert_eq!(review.completed_actions_count(), 0);

        review.update_action_status(0, ActionStatus::Completed);
        review.update_action_status(1, ActionStatus::Cancelled);

        assert_eq!(review.open_actions_count(), 1);
        assert_eq!(review.completed_actions_count(), 1);
    }

    #[test]
    fn root_cause_strings() {
        assert_eq!(
            RootCauseCategory::SoftwareDefect.as_str(),
            "software-defect"
        );
        assert_eq!(
            RootCauseCategory::ExternalAttack.as_str(),
            "external-attack"
        );
        assert_eq!(RootCauseCategory::Unknown.as_str(), "unknown");
    }

    #[test]
    fn action_status_strings() {
        assert_eq!(ActionStatus::Planned.as_str(), "planned");
        assert_eq!(ActionStatus::InProgress.as_str(), "in-progress");
        assert_eq!(ActionStatus::Completed.as_str(), "completed");
        assert_eq!(ActionStatus::Cancelled.as_str(), "cancelled");
    }

    #[test]
    fn terminal_status() {
        assert!(!ActionStatus::Planned.is_terminal());
        assert!(!ActionStatus::InProgress.is_terminal());
        assert!(ActionStatus::Completed.is_terminal());
        assert!(ActionStatus::Cancelled.is_terminal());
    }
}
