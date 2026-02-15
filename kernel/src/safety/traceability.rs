// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Requirements traceability matrix for DO-178C compliance.
//!
//! Tracks requirements -> specifications -> implementations -> tests
//! to satisfy DO-178C objectives for bidirectional traceability.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

/// Maximum number of artifacts in the traceability matrix.
pub const MAX_ARTIFACTS: usize = 2048;

/// Types of artifacts tracked in the traceability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactType {
    /// High-level or derived requirement.
    Requirement,
    /// Design specification or architecture document.
    Specification,
    /// Source code implementation artifact.
    Implementation,
    /// Test case or test procedure.
    Test,
    /// Verification result or analysis.
    Verification,
}

/// DO-178C Design Assurance Levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DalLevel {
    /// Level A — Catastrophic failure condition.
    A,
    /// Level B — Hazardous failure condition.
    B,
    /// Level C — Major failure condition.
    C,
    /// Level D — Minor failure condition.
    D,
    /// Level E — No safety effect.
    E,
}

/// Status of traceability for an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceStatus {
    /// Fully traced: all required links present.
    Traced,
    /// Partially traced: some links present but not all.
    Partial,
    /// Not traced: no links present.
    Untraced,
    /// Orphan: artifact (typically a test) with no requirement link.
    Orphan,
}

/// A directional link between two artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceLink {
    /// Source artifact identifier.
    pub from_id: String,
    /// Source artifact type.
    pub from_type: ArtifactType,
    /// Destination artifact identifier.
    pub to_id: String,
    /// Destination artifact type.
    pub to_type: ArtifactType,
}

/// A tracked artifact in the traceability matrix.
#[derive(Debug, Clone)]
pub struct Artifact {
    /// Unique identifier for the artifact.
    pub id: String,
    /// Type of this artifact.
    pub artifact_type: ArtifactType,
    /// Human-readable description.
    pub description: String,
    /// Design assurance level.
    pub dal_level: DalLevel,
    /// Traceability links from/to this artifact.
    pub links: Vec<TraceLink>,
}

/// Summary of traceability coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceCoverageReport {
    /// Total number of requirements.
    pub total_reqs: usize,
    /// Number of fully traced requirements.
    pub traced: usize,
    /// Number of partially traced requirements.
    pub partial: usize,
    /// Number of untraced requirements.
    pub untraced: usize,
    /// Number of orphan tests.
    pub orphan_tests: usize,
    /// Coverage percentage (0-100).
    pub coverage_pct: u8,
}

/// Errors that can occur in traceability operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    /// An artifact with this ID already exists.
    DuplicateId,
    /// The referenced artifact was not found.
    NotFound,
    /// The traceability matrix is full.
    TableFull,
}

/// Requirements traceability matrix.
///
/// Maintains a collection of artifacts and their bidirectional trace links
/// to satisfy DO-178C traceability objectives.
pub struct TraceMatrix {
    /// All tracked artifacts.
    artifacts: Vec<Artifact>,
    /// Next auto-increment ID counter.
    next_id: u64,
}

impl Default for TraceMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceMatrix {
    /// Creates a new empty traceability matrix.
    pub fn new() -> Self {
        Self {
            artifacts: Vec::new(),
            next_id: 1,
        }
    }

    /// Adds an artifact to the matrix.
    ///
    /// Returns `TraceError::DuplicateId` if an artifact with the same ID exists.
    /// Returns `TraceError::TableFull` if the matrix is at capacity.
    pub fn add_artifact(
        &mut self,
        id: &str,
        artifact_type: ArtifactType,
        description: &str,
        dal: DalLevel,
    ) -> Result<(), TraceError> {
        if self.artifacts.len() >= MAX_ARTIFACTS {
            return Err(TraceError::TableFull);
        }
        if self.artifacts.iter().any(|a| a.id == id) {
            return Err(TraceError::DuplicateId);
        }
        self.artifacts.push(Artifact {
            id: String::from(id),
            artifact_type,
            description: String::from(description),
            dal_level: dal,
            links: Vec::new(),
        });
        self.next_id += 1;
        Ok(())
    }

    /// Adds a bidirectional trace link between two artifacts.
    ///
    /// Both `from_id` and `to_id` must exist in the matrix.
    pub fn add_link(&mut self, from_id: &str, to_id: &str) -> Result<(), TraceError> {
        // Validate both artifacts exist and collect their types.
        let from_type = self
            .artifacts
            .iter()
            .find(|a| a.id == from_id)
            .map(|a| a.artifact_type)
            .ok_or(TraceError::NotFound)?;
        let to_type = self
            .artifacts
            .iter()
            .find(|a| a.id == to_id)
            .map(|a| a.artifact_type)
            .ok_or(TraceError::NotFound)?;

        // Add forward link (from -> to).
        let forward = TraceLink {
            from_id: String::from(from_id),
            from_type,
            to_id: String::from(to_id),
            to_type,
        };
        if let Some(artifact) = self.artifacts.iter_mut().find(|a| a.id == from_id) {
            if !artifact
                .links
                .iter()
                .any(|l| l.to_id == to_id && l.from_id == from_id)
            {
                artifact.links.push(forward);
            }
        }

        // Add reverse link (to -> from).
        let reverse = TraceLink {
            from_id: String::from(to_id),
            from_type: to_type,
            to_id: String::from(from_id),
            to_type: from_type,
        };
        if let Some(artifact) = self.artifacts.iter_mut().find(|a| a.id == to_id) {
            if !artifact
                .links
                .iter()
                .any(|l| l.to_id == from_id && l.from_id == to_id)
            {
                artifact.links.push(reverse);
            }
        }

        Ok(())
    }

    /// Retrieves an artifact by its ID.
    pub fn get_artifact(&self, id: &str) -> Option<&Artifact> {
        self.artifacts.iter().find(|a| a.id == id)
    }

    /// Returns the total number of artifacts in the matrix.
    pub fn artifact_count(&self) -> usize {
        self.artifacts.len()
    }

    /// Finds all artifacts of a given type.
    pub fn find_by_type(&self, t: ArtifactType) -> Vec<&Artifact> {
        self.artifacts
            .iter()
            .filter(|a| a.artifact_type == t)
            .collect()
    }

    /// Determines the traceability status of an artifact.
    ///
    /// For requirements: Traced if linked to Spec+Impl+Test, Partial if some, Untraced if none.
    /// For tests: Traced if linked to a Requirement, Orphan if no links.
    /// For others: Traced if any links, Untraced if none.
    pub fn trace_status(&self, id: &str) -> TraceStatus {
        let artifact = match self.get_artifact(id) {
            Some(a) => a,
            None => return TraceStatus::Untraced,
        };

        match artifact.artifact_type {
            ArtifactType::Requirement => self.requirement_status(artifact),
            ArtifactType::Test => self.test_status(artifact),
            _ => self.generic_status(artifact),
        }
    }

    /// Determine trace status for a Requirement artifact.
    ///
    /// Fully traced when linked to Specification, Implementation, and Test.
    /// Partial when some links exist. Untraced when no links at all.
    fn requirement_status(&self, artifact: &Artifact) -> TraceStatus {
        if artifact.links.is_empty() {
            return TraceStatus::Untraced;
        }

        let has_all = self.has_linked_type(artifact, ArtifactType::Specification)
            && self.has_linked_type(artifact, ArtifactType::Implementation)
            && self.has_linked_type(artifact, ArtifactType::Test);

        if has_all {
            TraceStatus::Traced
        } else {
            TraceStatus::Partial
        }
    }

    /// Determine trace status for a Test artifact.
    ///
    /// Traced when linked to a Requirement, Orphan otherwise.
    fn test_status(&self, artifact: &Artifact) -> TraceStatus {
        if self.has_linked_type(artifact, ArtifactType::Requirement) {
            TraceStatus::Traced
        } else {
            TraceStatus::Orphan
        }
    }

    /// Determine trace status for Specification, Implementation, or Verification artifacts.
    ///
    /// Traced when any links exist, Untraced otherwise.
    fn generic_status(&self, artifact: &Artifact) -> TraceStatus {
        if artifact.links.is_empty() {
            TraceStatus::Untraced
        } else {
            TraceStatus::Traced
        }
    }

    /// Check whether an artifact has at least one link to the given artifact type.
    fn has_linked_type(&self, artifact: &Artifact, target: ArtifactType) -> bool {
        artifact
            .links
            .iter()
            .any(|l| self.linked_type(l) == Some(target))
    }

    /// Generates a coverage report for all requirements in the matrix.
    pub fn coverage_report(&self) -> TraceCoverageReport {
        let requirements: Vec<&Artifact> = self.find_by_type(ArtifactType::Requirement);
        let total_reqs = requirements.len();

        let mut traced = 0usize;
        let mut partial = 0usize;
        let mut untraced = 0usize;

        for req in &requirements {
            match self.trace_status(&req.id) {
                TraceStatus::Traced => traced += 1,
                TraceStatus::Partial => partial += 1,
                TraceStatus::Untraced => untraced += 1,
                _ => {}
            }
        }

        let tests: Vec<&Artifact> = self.find_by_type(ArtifactType::Test);
        let orphan_tests = tests
            .iter()
            .filter(|t| self.trace_status(&t.id) == TraceStatus::Orphan)
            .count();

        let coverage_pct = (traced * 100).checked_div(total_reqs).unwrap_or(0) as u8;

        TraceCoverageReport {
            total_reqs,
            traced,
            partial,
            untraced,
            orphan_tests,
            coverage_pct,
        }
    }

    /// Helper: get the artifact type of the linked-to artifact in a trace link.
    fn linked_type(&self, link: &TraceLink) -> Option<ArtifactType> {
        self.get_artifact(&link.to_id).map(|a| a.artifact_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_matrix() -> TraceMatrix {
        TraceMatrix::new()
    }

    #[test]
    fn test_add_artifact() {
        let mut m = make_matrix();
        let result = m.add_artifact(
            "REQ-001",
            ArtifactType::Requirement,
            "Memory safety",
            DalLevel::A,
        );
        assert!(result.is_ok());
        assert_eq!(m.artifact_count(), 1);
    }

    #[test]
    fn test_add_duplicate_returns_error() {
        let mut m = make_matrix();
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req 1", DalLevel::A)
            .unwrap();
        let result = m.add_artifact(
            "REQ-001",
            ArtifactType::Requirement,
            "Req 1 dup",
            DalLevel::A,
        );
        assert_eq!(result, Err(TraceError::DuplicateId));
    }

    #[test]
    fn test_add_link_between_artifacts() {
        let mut m = make_matrix();
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req", DalLevel::A)
            .unwrap();
        m.add_artifact("SPEC-001", ArtifactType::Specification, "Spec", DalLevel::A)
            .unwrap();
        let result = m.add_link("REQ-001", "SPEC-001");
        assert!(result.is_ok());

        let req = m.get_artifact("REQ-001").unwrap();
        assert_eq!(req.links.len(), 1);
        assert_eq!(req.links[0].to_id, "SPEC-001");
    }

    #[test]
    fn test_link_to_nonexistent_returns_error() {
        let mut m = make_matrix();
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req", DalLevel::A)
            .unwrap();
        let result = m.add_link("REQ-001", "NONEXISTENT");
        assert_eq!(result, Err(TraceError::NotFound));
    }

    #[test]
    fn test_get_artifact_by_id() {
        let mut m = make_matrix();
        m.add_artifact(
            "REQ-001",
            ArtifactType::Requirement,
            "Memory safety",
            DalLevel::A,
        )
        .unwrap();
        let a = m.get_artifact("REQ-001");
        assert!(a.is_some());
        let a = a.unwrap();
        assert_eq!(a.id, "REQ-001");
        assert_eq!(a.artifact_type, ArtifactType::Requirement);
        assert_eq!(a.description, "Memory safety");
        assert_eq!(a.dal_level, DalLevel::A);
    }

    #[test]
    fn test_get_artifact_not_found() {
        let m = make_matrix();
        assert!(m.get_artifact("NOPE").is_none());
    }

    #[test]
    fn test_find_by_type() {
        let mut m = make_matrix();
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req 1", DalLevel::A)
            .unwrap();
        m.add_artifact("REQ-002", ArtifactType::Requirement, "Req 2", DalLevel::B)
            .unwrap();
        m.add_artifact("TEST-001", ArtifactType::Test, "Test 1", DalLevel::A)
            .unwrap();
        let reqs = m.find_by_type(ArtifactType::Requirement);
        assert_eq!(reqs.len(), 2);
        let tests = m.find_by_type(ArtifactType::Test);
        assert_eq!(tests.len(), 1);
    }

    #[test]
    fn test_trace_status_fully_traced_requirement() {
        let mut m = make_matrix();
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req", DalLevel::A)
            .unwrap();
        m.add_artifact("SPEC-001", ArtifactType::Specification, "Spec", DalLevel::A)
            .unwrap();
        m.add_artifact(
            "IMPL-001",
            ArtifactType::Implementation,
            "Impl",
            DalLevel::A,
        )
        .unwrap();
        m.add_artifact("TEST-001", ArtifactType::Test, "Test", DalLevel::A)
            .unwrap();
        m.add_link("REQ-001", "SPEC-001").unwrap();
        m.add_link("REQ-001", "IMPL-001").unwrap();
        m.add_link("REQ-001", "TEST-001").unwrap();
        assert_eq!(m.trace_status("REQ-001"), TraceStatus::Traced);
    }

    #[test]
    fn test_trace_status_partial_requirement() {
        let mut m = make_matrix();
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req", DalLevel::A)
            .unwrap();
        m.add_artifact("SPEC-001", ArtifactType::Specification, "Spec", DalLevel::A)
            .unwrap();
        m.add_link("REQ-001", "SPEC-001").unwrap();
        // Only linked to spec, not impl or test.
        assert_eq!(m.trace_status("REQ-001"), TraceStatus::Partial);
    }

    #[test]
    fn test_trace_status_untraced_requirement() {
        let mut m = make_matrix();
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req", DalLevel::A)
            .unwrap();
        assert_eq!(m.trace_status("REQ-001"), TraceStatus::Untraced);
    }

    #[test]
    fn test_trace_status_orphan_test() {
        let mut m = make_matrix();
        m.add_artifact("TEST-001", ArtifactType::Test, "Orphan test", DalLevel::A)
            .unwrap();
        assert_eq!(m.trace_status("TEST-001"), TraceStatus::Orphan);
    }

    #[test]
    fn test_trace_status_traced_test() {
        let mut m = make_matrix();
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req", DalLevel::A)
            .unwrap();
        m.add_artifact("TEST-001", ArtifactType::Test, "Test", DalLevel::A)
            .unwrap();
        m.add_link("TEST-001", "REQ-001").unwrap();
        assert_eq!(m.trace_status("TEST-001"), TraceStatus::Traced);
    }

    #[test]
    fn test_coverage_report() {
        let mut m = make_matrix();
        // Fully traced requirement.
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req 1", DalLevel::A)
            .unwrap();
        m.add_artifact(
            "SPEC-001",
            ArtifactType::Specification,
            "Spec 1",
            DalLevel::A,
        )
        .unwrap();
        m.add_artifact(
            "IMPL-001",
            ArtifactType::Implementation,
            "Impl 1",
            DalLevel::A,
        )
        .unwrap();
        m.add_artifact("TEST-001", ArtifactType::Test, "Test 1", DalLevel::A)
            .unwrap();
        m.add_link("REQ-001", "SPEC-001").unwrap();
        m.add_link("REQ-001", "IMPL-001").unwrap();
        m.add_link("REQ-001", "TEST-001").unwrap();

        // Untraced requirement.
        m.add_artifact("REQ-002", ArtifactType::Requirement, "Req 2", DalLevel::B)
            .unwrap();

        // Orphan test.
        m.add_artifact("TEST-002", ArtifactType::Test, "Orphan", DalLevel::A)
            .unwrap();

        let report = m.coverage_report();
        assert_eq!(report.total_reqs, 2);
        assert_eq!(report.traced, 1);
        assert_eq!(report.untraced, 1);
        assert_eq!(report.partial, 0);
        assert_eq!(report.orphan_tests, 1);
        assert_eq!(report.coverage_pct, 50);
    }

    #[test]
    fn test_artifact_count() {
        let mut m = make_matrix();
        assert_eq!(m.artifact_count(), 0);
        m.add_artifact("A-1", ArtifactType::Requirement, "r", DalLevel::A)
            .unwrap();
        m.add_artifact("A-2", ArtifactType::Test, "t", DalLevel::A)
            .unwrap();
        assert_eq!(m.artifact_count(), 2);
    }

    #[test]
    fn test_table_full_error() {
        let mut m = make_matrix();
        for i in 0..MAX_ARTIFACTS {
            m.add_artifact(
                &alloc::format!("ART-{}", i),
                ArtifactType::Requirement,
                "r",
                DalLevel::A,
            )
            .unwrap();
        }
        let result = m.add_artifact("OVERFLOW", ArtifactType::Requirement, "r", DalLevel::A);
        assert_eq!(result, Err(TraceError::TableFull));
    }

    #[test]
    fn test_dal_levels() {
        let mut m = make_matrix();
        m.add_artifact("A", ArtifactType::Requirement, "a", DalLevel::A)
            .unwrap();
        m.add_artifact("B", ArtifactType::Requirement, "b", DalLevel::B)
            .unwrap();
        m.add_artifact("C", ArtifactType::Requirement, "c", DalLevel::C)
            .unwrap();
        m.add_artifact("D", ArtifactType::Requirement, "d", DalLevel::D)
            .unwrap();
        m.add_artifact("E", ArtifactType::Requirement, "e", DalLevel::E)
            .unwrap();

        assert_eq!(m.get_artifact("A").unwrap().dal_level, DalLevel::A);
        assert_eq!(m.get_artifact("B").unwrap().dal_level, DalLevel::B);
        assert_eq!(m.get_artifact("C").unwrap().dal_level, DalLevel::C);
        assert_eq!(m.get_artifact("D").unwrap().dal_level, DalLevel::D);
        assert_eq!(m.get_artifact("E").unwrap().dal_level, DalLevel::E);
    }

    #[test]
    fn test_multiple_artifact_types() {
        let mut m = make_matrix();
        m.add_artifact("REQ-1", ArtifactType::Requirement, "r", DalLevel::A)
            .unwrap();
        m.add_artifact("SPEC-1", ArtifactType::Specification, "s", DalLevel::A)
            .unwrap();
        m.add_artifact("IMPL-1", ArtifactType::Implementation, "i", DalLevel::A)
            .unwrap();
        m.add_artifact("TEST-1", ArtifactType::Test, "t", DalLevel::A)
            .unwrap();
        m.add_artifact("VER-1", ArtifactType::Verification, "v", DalLevel::A)
            .unwrap();

        assert_eq!(m.find_by_type(ArtifactType::Requirement).len(), 1);
        assert_eq!(m.find_by_type(ArtifactType::Specification).len(), 1);
        assert_eq!(m.find_by_type(ArtifactType::Implementation).len(), 1);
        assert_eq!(m.find_by_type(ArtifactType::Test).len(), 1);
        assert_eq!(m.find_by_type(ArtifactType::Verification).len(), 1);
    }

    #[test]
    fn test_bidirectional_linking() {
        let mut m = make_matrix();
        m.add_artifact("REQ-001", ArtifactType::Requirement, "Req", DalLevel::A)
            .unwrap();
        m.add_artifact("TEST-001", ArtifactType::Test, "Test", DalLevel::A)
            .unwrap();
        m.add_link("REQ-001", "TEST-001").unwrap();

        // Forward link: REQ -> TEST.
        let req = m.get_artifact("REQ-001").unwrap();
        assert_eq!(req.links.len(), 1);
        assert_eq!(req.links[0].to_id, "TEST-001");

        // Reverse link: TEST -> REQ.
        let test = m.get_artifact("TEST-001").unwrap();
        assert_eq!(test.links.len(), 1);
        assert_eq!(test.links[0].to_id, "REQ-001");
    }

    #[test]
    fn test_empty_coverage_report() {
        let m = make_matrix();
        let report = m.coverage_report();
        assert_eq!(report.total_reqs, 0);
        assert_eq!(report.traced, 0);
        assert_eq!(report.coverage_pct, 0);
    }

    #[test]
    fn test_link_nonexistent_from() {
        let mut m = make_matrix();
        m.add_artifact("A", ArtifactType::Requirement, "a", DalLevel::A)
            .unwrap();
        let result = m.add_link("NONEXISTENT", "A");
        assert_eq!(result, Err(TraceError::NotFound));
    }
}
