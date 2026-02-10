// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MISRA-Rust coding standard rules and enforcement.
//!
//! Provides runtime tracking of coding standard rules inspired by MISRA C/C++
//! adapted for Rust, including violation recording and compliance checking
//! for DO-178C certification.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

/// Maximum number of rules that can be registered.
pub const MAX_RULES: usize = 512;

/// Category of a coding standard rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleCategory {
    /// Mandatory: must be followed without exception.
    Mandatory,
    /// Required: must be followed unless a documented deviation exists.
    Required,
    /// Advisory: should be followed, deviation does not affect compliance.
    Advisory,
}

/// Scope to which a rule applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleScope {
    /// Module-level rules (imports, visibility, documentation).
    Module,
    /// Function-level rules (length, parameters, exit points).
    Function,
    /// Expression-level rules (complexity, readability).
    Expression,
    /// Type-level rules (conversions, safety).
    Type,
    /// Unsafe code rules (safety comments, encapsulation).
    Unsafe,
}

/// A single coding standard rule.
#[derive(Debug, Clone)]
pub struct MisraRule {
    /// Unique rule identifier (e.g., "MR-1.1").
    pub id: String,
    /// Rule category.
    pub category: RuleCategory,
    /// Scope of applicability.
    pub scope: RuleScope,
    /// Human-readable description.
    pub description: String,
    /// Whether the rule is currently enabled.
    pub enabled: bool,
}

/// A recorded violation of a coding standard rule.
#[derive(Debug, Clone)]
pub struct Violation {
    /// ID of the violated rule.
    pub rule_id: String,
    /// File where the violation occurred.
    pub file: String,
    /// Line number of the violation.
    pub line: u32,
    /// Description of the specific violation.
    pub message: String,
}

/// Errors that can occur in coding standard operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodingStandardError {
    /// A rule with this ID already exists.
    DuplicateRule,
    /// The referenced rule was not found.
    RuleNotFound,
    /// The rule table is full.
    TableFull,
}

/// Coding standard rule set and violation tracker.
///
/// Manages MISRA-Rust-inspired rules and records violations against them.
/// Compliance is determined by the absence of Required/Mandatory violations.
pub struct CodingStandard {
    /// Registered rules.
    rules: Vec<MisraRule>,
    /// Recorded violations.
    violations: Vec<Violation>,
    /// Maximum number of violations to retain.
    max_violations: usize,
}

impl Default for CodingStandard {
    fn default() -> Self {
        Self::new()
    }
}

impl CodingStandard {
    /// Creates a new empty coding standard with default limits.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            violations: Vec::new(),
            max_violations: 10000,
        }
    }

    /// Adds a new rule to the coding standard.
    ///
    /// Returns `CodingStandardError::DuplicateRule` if a rule with the same ID exists.
    /// Returns `CodingStandardError::TableFull` if the rule table is at capacity.
    pub fn add_rule(
        &mut self,
        id: &str,
        cat: RuleCategory,
        scope: RuleScope,
        desc: &str,
    ) -> Result<(), CodingStandardError> {
        if self.rules.len() >= MAX_RULES {
            return Err(CodingStandardError::TableFull);
        }
        if self.rules.iter().any(|r| r.id == id) {
            return Err(CodingStandardError::DuplicateRule);
        }
        self.rules.push(MisraRule {
            id: String::from(id),
            category: cat,
            scope,
            description: String::from(desc),
            enabled: true,
        });
        Ok(())
    }

    /// Registers the default set of MISRA-Rust-inspired rules.
    ///
    /// Registers 10 rules covering modules, functions, unsafe code, types,
    /// and expressions.
    pub fn register_defaults(&mut self) -> Result<(), CodingStandardError> {
        self.add_rule(
            "MR-1.1",
            RuleCategory::Required,
            RuleScope::Module,
            "All code shall be reachable",
        )?;
        self.add_rule(
            "MR-1.2",
            RuleCategory::Required,
            RuleScope::Function,
            "Functions shall have a single exit point",
        )?;
        self.add_rule(
            "MR-2.1",
            RuleCategory::Required,
            RuleScope::Unsafe,
            "All unsafe blocks shall have a SAFETY comment",
        )?;
        self.add_rule(
            "MR-2.2",
            RuleCategory::Required,
            RuleScope::Unsafe,
            "Unsafe code shall be minimized and encapsulated",
        )?;
        self.add_rule(
            "MR-3.1",
            RuleCategory::Mandatory,
            RuleScope::Type,
            "No implicit type conversions",
        )?;
        self.add_rule(
            "MR-3.2",
            RuleCategory::Advisory,
            RuleScope::Expression,
            "Avoid complex expressions (cognitive complexity \u{2264} 25)",
        )?;
        self.add_rule(
            "MR-4.1",
            RuleCategory::Required,
            RuleScope::Function,
            "Functions shall not exceed 100 lines",
        )?;
        self.add_rule(
            "MR-4.2",
            RuleCategory::Advisory,
            RuleScope::Function,
            "Functions shall not exceed 7 parameters",
        )?;
        self.add_rule(
            "MR-5.1",
            RuleCategory::Required,
            RuleScope::Module,
            "All public items shall be documented",
        )?;
        self.add_rule(
            "MR-5.2",
            RuleCategory::Required,
            RuleScope::Module,
            "No wildcard imports",
        )?;
        Ok(())
    }

    /// Records a violation of a rule.
    ///
    /// If the violation buffer is full, the oldest violation is dropped.
    pub fn add_violation(&mut self, rule_id: &str, file: &str, line: u32, msg: &str) {
        if self.violations.len() >= self.max_violations {
            self.violations.remove(0);
        }
        self.violations.push(Violation {
            rule_id: String::from(rule_id),
            file: String::from(file),
            line,
            message: String::from(msg),
        });
    }

    /// Returns the total number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Returns the total number of recorded violations.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Returns all violations for a specific rule ID.
    pub fn violations_for_rule(&self, rule_id: &str) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.rule_id == rule_id)
            .collect()
    }

    /// Checks whether the codebase is compliant.
    ///
    /// Compliance means no violations exist for Required or Mandatory rules.
    /// Advisory violations do not affect compliance.
    pub fn is_compliant(&self) -> bool {
        for violation in &self.violations {
            if let Some(rule) = self.rules.iter().find(|r| r.id == violation.rule_id) {
                match rule.category {
                    RuleCategory::Required | RuleCategory::Mandatory => return false,
                    RuleCategory::Advisory => {}
                }
            }
        }
        true
    }

    /// Returns the count of Required-category rules.
    pub fn required_rule_count(&self) -> usize {
        self.rules
            .iter()
            .filter(|r| r.category == RuleCategory::Required)
            .count()
    }

    /// Enables a rule by ID.
    pub fn enable_rule(&mut self, id: &str) -> Result<(), CodingStandardError> {
        let rule = self
            .rules
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or(CodingStandardError::RuleNotFound)?;
        rule.enabled = true;
        Ok(())
    }

    /// Disables a rule by ID.
    pub fn disable_rule(&mut self, id: &str) -> Result<(), CodingStandardError> {
        let rule = self
            .rules
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or(CodingStandardError::RuleNotFound)?;
        rule.enabled = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_rule() {
        let mut cs = CodingStandard::new();
        let result = cs.add_rule(
            "R-1",
            RuleCategory::Required,
            RuleScope::Module,
            "Test rule",
        );
        assert!(result.is_ok());
        assert_eq!(cs.rule_count(), 1);
    }

    #[test]
    fn test_register_defaults_creates_10_rules() {
        let mut cs = CodingStandard::new();
        cs.register_defaults().unwrap();
        assert_eq!(cs.rule_count(), 10);
    }

    #[test]
    fn test_add_violation() {
        let mut cs = CodingStandard::new();
        cs.add_rule("R-1", RuleCategory::Required, RuleScope::Module, "rule")
            .unwrap();
        cs.add_violation("R-1", "main.rs", 42, "unreachable code");
        assert_eq!(cs.violation_count(), 1);
    }

    #[test]
    fn test_violations_for_rule() {
        let mut cs = CodingStandard::new();
        cs.add_rule("R-1", RuleCategory::Required, RuleScope::Module, "rule")
            .unwrap();
        cs.add_rule("R-2", RuleCategory::Advisory, RuleScope::Function, "rule2")
            .unwrap();
        cs.add_violation("R-1", "a.rs", 1, "v1");
        cs.add_violation("R-1", "b.rs", 2, "v2");
        cs.add_violation("R-2", "c.rs", 3, "v3");

        let r1_violations = cs.violations_for_rule("R-1");
        assert_eq!(r1_violations.len(), 2);
        let r2_violations = cs.violations_for_rule("R-2");
        assert_eq!(r2_violations.len(), 1);
    }

    #[test]
    fn test_is_compliant_with_no_violations() {
        let mut cs = CodingStandard::new();
        cs.register_defaults().unwrap();
        assert!(cs.is_compliant());
    }

    #[test]
    fn test_is_compliant_false_with_required_violation() {
        let mut cs = CodingStandard::new();
        cs.register_defaults().unwrap();
        cs.add_violation("MR-1.1", "main.rs", 10, "dead code found");
        assert!(!cs.is_compliant());
    }

    #[test]
    fn test_advisory_violations_dont_affect_compliance() {
        let mut cs = CodingStandard::new();
        cs.register_defaults().unwrap();
        // MR-3.2 is Advisory.
        cs.add_violation("MR-3.2", "main.rs", 10, "complex expression");
        assert!(cs.is_compliant());
    }

    #[test]
    fn test_rule_count() {
        let mut cs = CodingStandard::new();
        assert_eq!(cs.rule_count(), 0);
        cs.add_rule("R-1", RuleCategory::Required, RuleScope::Module, "r")
            .unwrap();
        assert_eq!(cs.rule_count(), 1);
    }

    #[test]
    fn test_violation_count() {
        let mut cs = CodingStandard::new();
        assert_eq!(cs.violation_count(), 0);
        cs.add_violation("R-1", "a.rs", 1, "v1");
        cs.add_violation("R-1", "b.rs", 2, "v2");
        assert_eq!(cs.violation_count(), 2);
    }

    #[test]
    fn test_enable_disable_rules() {
        let mut cs = CodingStandard::new();
        cs.add_rule("R-1", RuleCategory::Required, RuleScope::Module, "r")
            .unwrap();

        // Rule starts enabled.
        assert!(cs.rules.iter().find(|r| r.id == "R-1").unwrap().enabled);

        cs.disable_rule("R-1").unwrap();
        assert!(!cs.rules.iter().find(|r| r.id == "R-1").unwrap().enabled);

        cs.enable_rule("R-1").unwrap();
        assert!(cs.rules.iter().find(|r| r.id == "R-1").unwrap().enabled);
    }

    #[test]
    fn test_required_rule_count() {
        let mut cs = CodingStandard::new();
        cs.register_defaults().unwrap();
        // Default rules: MR-1.1, MR-1.2, MR-2.1, MR-2.2, MR-4.1, MR-5.1, MR-5.2 = 7 Required
        assert_eq!(cs.required_rule_count(), 7);
    }

    #[test]
    fn test_duplicate_rule_error() {
        let mut cs = CodingStandard::new();
        cs.add_rule("R-1", RuleCategory::Required, RuleScope::Module, "r")
            .unwrap();
        let result = cs.add_rule("R-1", RuleCategory::Advisory, RuleScope::Function, "dup");
        assert_eq!(result, Err(CodingStandardError::DuplicateRule));
    }

    #[test]
    fn test_rule_not_found_error() {
        let mut cs = CodingStandard::new();
        let result = cs.enable_rule("NONEXISTENT");
        assert_eq!(result, Err(CodingStandardError::RuleNotFound));
        let result = cs.disable_rule("NONEXISTENT");
        assert_eq!(result, Err(CodingStandardError::RuleNotFound));
    }

    #[test]
    fn test_default_rule_ids_present() {
        let mut cs = CodingStandard::new();
        cs.register_defaults().unwrap();

        let expected_ids = [
            "MR-1.1", "MR-1.2", "MR-2.1", "MR-2.2", "MR-3.1", "MR-3.2", "MR-4.1", "MR-4.2",
            "MR-5.1", "MR-5.2",
        ];
        for id in &expected_ids {
            assert!(
                cs.rules.iter().any(|r| r.id == *id),
                "Expected rule {} not found",
                id
            );
        }
    }

    #[test]
    fn test_table_full_error() {
        let mut cs = CodingStandard::new();
        for i in 0..MAX_RULES {
            cs.add_rule(
                &alloc::format!("R-{}", i),
                RuleCategory::Advisory,
                RuleScope::Module,
                "r",
            )
            .unwrap();
        }
        let result = cs.add_rule("OVERFLOW", RuleCategory::Advisory, RuleScope::Module, "r");
        assert_eq!(result, Err(CodingStandardError::TableFull));
    }

    #[test]
    fn test_mandatory_violation_affects_compliance() {
        let mut cs = CodingStandard::new();
        cs.register_defaults().unwrap();
        // MR-3.1 is Mandatory.
        cs.add_violation("MR-3.1", "types.rs", 5, "implicit conversion");
        assert!(!cs.is_compliant());
    }
}
