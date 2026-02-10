// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MC/DC coverage tracking for DO-178C DAL A compliance.
//!
//! Provides runtime instrumentation points and coverage metric collection
//! for statement, branch, and Modified Condition/Decision Coverage (MC/DC)
//! as required by DO-178C Design Assurance Level A.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

/// Maximum number of coverage entries that can be tracked.
pub const MAX_ENTRIES: usize = 8192;

/// Type of coverage being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageType {
    /// Statement coverage: was this line executed?
    Statement,
    /// Branch coverage: were both true/false paths taken?
    Branch,
    /// MC/DC: did each condition independently affect the decision?
    MCDC,
    /// Function coverage: was this function called?
    Function,
    /// Condition coverage: was each boolean sub-expression evaluated both ways?
    Condition,
}

/// A single coverage instrumentation point.
#[derive(Debug, Clone)]
pub struct CoverageEntry {
    /// Source file path.
    pub file: String,
    /// Line number.
    pub line: u32,
    /// Type of coverage tracked at this point.
    pub coverage_type: CoverageType,
    /// Number of times this point has been hit.
    pub hit_count: u64,
    /// Total number of conditions in this decision (for MC/DC).
    pub total_conditions: u32,
    /// Number of conditions independently shown to affect the outcome.
    pub conditions_covered: u32,
}

/// Coverage percentage summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoveragePercent {
    /// Number of covered entries.
    pub covered: usize,
    /// Total number of entries.
    pub total: usize,
    /// Coverage as a percentage (0-100).
    pub percent: u8,
}

/// Errors that can occur in coverage operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverageError {
    /// The provided index is out of bounds.
    IndexOutOfBounds,
    /// The coverage table is full.
    TableFull,
}

/// Runtime coverage metric tracker.
///
/// Collects statement, branch, and MC/DC coverage data for DO-178C
/// DAL A compliance verification.
pub struct CoverageTracker {
    /// All registered coverage entries.
    entries: Vec<CoverageEntry>,
    /// Next auto-increment ID counter.
    next_id: u64,
}

impl Default for CoverageTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CoverageTracker {
    /// Creates a new empty coverage tracker.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 0,
        }
    }

    /// Registers a new coverage instrumentation point.
    ///
    /// Returns the index of the registered entry for later recording.
    /// Returns `CoverageError::TableFull` if the table is at capacity.
    pub fn register(
        &mut self,
        file: &str,
        line: u32,
        cov_type: CoverageType,
        total_conditions: u32,
    ) -> Result<usize, CoverageError> {
        if self.entries.len() >= MAX_ENTRIES {
            return Err(CoverageError::TableFull);
        }
        let index = self.entries.len();
        self.entries.push(CoverageEntry {
            file: String::from(file),
            line,
            coverage_type: cov_type,
            hit_count: 0,
            total_conditions,
            conditions_covered: 0,
        });
        self.next_id += 1;
        Ok(index)
    }

    /// Records a hit at the given coverage point index.
    ///
    /// Increments the hit count by 1.
    pub fn record_hit(&mut self, index: usize) -> Result<(), CoverageError> {
        let entry = self
            .entries
            .get_mut(index)
            .ok_or(CoverageError::IndexOutOfBounds)?;
        entry.hit_count += 1;
        Ok(())
    }

    /// Records condition coverage at the given MC/DC point.
    ///
    /// Updates `conditions_covered` to the maximum of the current value
    /// and the new value, reflecting cumulative condition independence.
    pub fn record_condition(
        &mut self,
        index: usize,
        conditions_covered: u32,
    ) -> Result<(), CoverageError> {
        let entry = self
            .entries
            .get_mut(index)
            .ok_or(CoverageError::IndexOutOfBounds)?;
        if conditions_covered > entry.conditions_covered {
            entry.conditions_covered = conditions_covered;
        }
        Ok(())
    }

    /// Returns the total number of registered entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Computes statement coverage percentage.
    ///
    /// Returns the percentage of Statement entries with hit_count > 0.
    pub fn statement_coverage(&self) -> CoveragePercent {
        self.coverage_for_type(CoverageType::Statement, |e| e.hit_count > 0)
    }

    /// Computes branch coverage percentage.
    ///
    /// Returns the percentage of Branch entries with hit_count > 0.
    pub fn branch_coverage(&self) -> CoveragePercent {
        self.coverage_for_type(CoverageType::Branch, |e| e.hit_count > 0)
    }

    /// Computes MC/DC coverage percentage.
    ///
    /// Returns the percentage of MC/DC entries where all conditions
    /// have been independently shown to affect the decision outcome.
    pub fn mcdc_coverage(&self) -> CoveragePercent {
        self.coverage_for_type(CoverageType::MCDC, |e| {
            e.total_conditions > 0 && e.conditions_covered == e.total_conditions
        })
    }

    /// Computes overall coverage across all entry types.
    ///
    /// An entry is considered covered if hit_count > 0.
    pub fn overall_coverage(&self) -> CoveragePercent {
        let total = self.entries.len();
        if total == 0 {
            return CoveragePercent {
                covered: 0,
                total: 0,
                percent: 0,
            };
        }
        let covered = self.entries.iter().filter(|e| e.hit_count > 0).count();
        let percent = ((covered * 100) / total) as u8;
        CoveragePercent {
            covered,
            total,
            percent,
        }
    }

    /// Returns all entries that have not been hit (hit_count == 0).
    pub fn uncovered_entries(&self) -> Vec<&CoverageEntry> {
        self.entries.iter().filter(|e| e.hit_count == 0).collect()
    }

    /// Helper to compute coverage for a specific type with a predicate.
    fn coverage_for_type<F>(&self, cov_type: CoverageType, is_covered: F) -> CoveragePercent
    where
        F: Fn(&CoverageEntry) -> bool,
    {
        let type_entries: Vec<&CoverageEntry> = self
            .entries
            .iter()
            .filter(|e| e.coverage_type == cov_type)
            .collect();
        let total = type_entries.len();
        if total == 0 {
            return CoveragePercent {
                covered: 0,
                total: 0,
                percent: 0,
            };
        }
        let covered = type_entries.iter().filter(|e| is_covered(e)).count();
        let percent = ((covered * 100) / total) as u8;
        CoveragePercent {
            covered,
            total,
            percent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_entry() {
        let mut tracker = CoverageTracker::new();
        let idx = tracker.register("main.rs", 10, CoverageType::Statement, 0);
        assert!(idx.is_ok());
        assert_eq!(idx.unwrap(), 0);
        assert_eq!(tracker.entry_count(), 1);
    }

    #[test]
    fn test_record_hit() {
        let mut tracker = CoverageTracker::new();
        let idx = tracker
            .register("main.rs", 10, CoverageType::Statement, 0)
            .unwrap();
        tracker.record_hit(idx).unwrap();
        tracker.record_hit(idx).unwrap();
        assert_eq!(tracker.entries[idx].hit_count, 2);
    }

    #[test]
    fn test_record_condition() {
        let mut tracker = CoverageTracker::new();
        let idx = tracker
            .register("main.rs", 10, CoverageType::MCDC, 3)
            .unwrap();
        tracker.record_condition(idx, 2).unwrap();
        assert_eq!(tracker.entries[idx].conditions_covered, 2);
        // Max should be retained.
        tracker.record_condition(idx, 1).unwrap();
        assert_eq!(tracker.entries[idx].conditions_covered, 2);
        // Increase should update.
        tracker.record_condition(idx, 3).unwrap();
        assert_eq!(tracker.entries[idx].conditions_covered, 3);
    }

    #[test]
    fn test_statement_coverage_100() {
        let mut tracker = CoverageTracker::new();
        let i0 = tracker
            .register("a.rs", 1, CoverageType::Statement, 0)
            .unwrap();
        let i1 = tracker
            .register("a.rs", 2, CoverageType::Statement, 0)
            .unwrap();
        tracker.record_hit(i0).unwrap();
        tracker.record_hit(i1).unwrap();
        let cov = tracker.statement_coverage();
        assert_eq!(cov.covered, 2);
        assert_eq!(cov.total, 2);
        assert_eq!(cov.percent, 100);
    }

    #[test]
    fn test_statement_coverage_partial() {
        let mut tracker = CoverageTracker::new();
        let i0 = tracker
            .register("a.rs", 1, CoverageType::Statement, 0)
            .unwrap();
        let _i1 = tracker
            .register("a.rs", 2, CoverageType::Statement, 0)
            .unwrap();
        tracker.record_hit(i0).unwrap();
        let cov = tracker.statement_coverage();
        assert_eq!(cov.covered, 1);
        assert_eq!(cov.total, 2);
        assert_eq!(cov.percent, 50);
    }

    #[test]
    fn test_branch_coverage() {
        let mut tracker = CoverageTracker::new();
        let i0 = tracker
            .register("a.rs", 1, CoverageType::Branch, 0)
            .unwrap();
        let _i1 = tracker
            .register("a.rs", 2, CoverageType::Branch, 0)
            .unwrap();
        let _i2 = tracker
            .register("a.rs", 3, CoverageType::Branch, 0)
            .unwrap();
        tracker.record_hit(i0).unwrap();
        let cov = tracker.branch_coverage();
        assert_eq!(cov.covered, 1);
        assert_eq!(cov.total, 3);
        assert_eq!(cov.percent, 33);
    }

    #[test]
    fn test_mcdc_coverage_all_conditions_met() {
        let mut tracker = CoverageTracker::new();
        let idx = tracker.register("a.rs", 1, CoverageType::MCDC, 3).unwrap();
        tracker.record_hit(idx).unwrap();
        tracker.record_condition(idx, 3).unwrap();
        let cov = tracker.mcdc_coverage();
        assert_eq!(cov.covered, 1);
        assert_eq!(cov.total, 1);
        assert_eq!(cov.percent, 100);
    }

    #[test]
    fn test_mcdc_coverage_partial() {
        let mut tracker = CoverageTracker::new();
        let i0 = tracker.register("a.rs", 1, CoverageType::MCDC, 3).unwrap();
        let i1 = tracker.register("a.rs", 2, CoverageType::MCDC, 4).unwrap();
        tracker.record_hit(i0).unwrap();
        tracker.record_condition(i0, 3).unwrap(); // Fully covered.
        tracker.record_hit(i1).unwrap();
        tracker.record_condition(i1, 2).unwrap(); // Partially covered.
        let cov = tracker.mcdc_coverage();
        assert_eq!(cov.covered, 1);
        assert_eq!(cov.total, 2);
        assert_eq!(cov.percent, 50);
    }

    #[test]
    fn test_overall_coverage() {
        let mut tracker = CoverageTracker::new();
        let i0 = tracker
            .register("a.rs", 1, CoverageType::Statement, 0)
            .unwrap();
        let _i1 = tracker
            .register("a.rs", 2, CoverageType::Branch, 0)
            .unwrap();
        let i2 = tracker
            .register("a.rs", 3, CoverageType::Function, 0)
            .unwrap();
        tracker.record_hit(i0).unwrap();
        tracker.record_hit(i2).unwrap();
        let cov = tracker.overall_coverage();
        assert_eq!(cov.covered, 2);
        assert_eq!(cov.total, 3);
        assert_eq!(cov.percent, 66);
    }

    #[test]
    fn test_uncovered_entries() {
        let mut tracker = CoverageTracker::new();
        let i0 = tracker
            .register("a.rs", 1, CoverageType::Statement, 0)
            .unwrap();
        let _i1 = tracker
            .register("a.rs", 2, CoverageType::Statement, 0)
            .unwrap();
        tracker.record_hit(i0).unwrap();
        let uncovered = tracker.uncovered_entries();
        assert_eq!(uncovered.len(), 1);
        assert_eq!(uncovered[0].line, 2);
    }

    #[test]
    fn test_empty_tracker_zero_percent() {
        let tracker = CoverageTracker::new();
        assert_eq!(tracker.entry_count(), 0);
        let cov = tracker.statement_coverage();
        assert_eq!(cov.percent, 0);
        assert_eq!(cov.total, 0);
        let cov = tracker.overall_coverage();
        assert_eq!(cov.percent, 0);
    }

    #[test]
    fn test_table_full_error() {
        let mut tracker = CoverageTracker::new();
        for i in 0..MAX_ENTRIES {
            tracker
                .register("fill.rs", i as u32, CoverageType::Statement, 0)
                .unwrap();
        }
        let result = tracker.register("overflow.rs", 0, CoverageType::Statement, 0);
        assert_eq!(result, Err(CoverageError::TableFull));
    }

    #[test]
    fn test_out_of_bounds_error() {
        let mut tracker = CoverageTracker::new();
        let result = tracker.record_hit(999);
        assert_eq!(result, Err(CoverageError::IndexOutOfBounds));
        let result = tracker.record_condition(999, 1);
        assert_eq!(result, Err(CoverageError::IndexOutOfBounds));
    }

    #[test]
    fn test_coverage_percent_calculation() {
        let mut tracker = CoverageTracker::new();
        // Register 3 statements, hit 1: 33%.
        for i in 0..3 {
            tracker
                .register("a.rs", i, CoverageType::Statement, 0)
                .unwrap();
        }
        tracker.record_hit(0).unwrap();
        let cov = tracker.statement_coverage();
        assert_eq!(cov.percent, 33);

        // Register a 4th, hit it too: 2/4 = 50%.
        let idx = tracker
            .register("a.rs", 3, CoverageType::Statement, 0)
            .unwrap();
        tracker.record_hit(idx).unwrap();
        let cov = tracker.statement_coverage();
        assert_eq!(cov.percent, 50);
    }
}
