// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429 label-based receive filtering.
//!
//! Filters incoming words based on their 8-bit label and optional SDI
//! matching. Only words matching a registered label+SDI combination are
//! delivered to the application; all others are silently discarded.

use alloc::vec::Vec;

use super::word::{Label, Sdi};
use crate::BusError;

/// Maximum number of registered label filters.
pub const MAX_LABEL_FILTERS: usize = 256;

/// A label filter entry matching label and optional SDI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelFilterEntry {
    /// The label to match.
    pub label: Label,
    /// Optional SDI filter. If `None`, any SDI matches.
    pub sdi: Option<Sdi>,
}

impl LabelFilterEntry {
    /// Create a filter that matches any SDI for the given label.
    pub fn any_sdi(label: Label) -> Self {
        LabelFilterEntry { label, sdi: None }
    }

    /// Create a filter that matches a specific label+SDI combination.
    pub fn with_sdi(label: Label, sdi: Sdi) -> Self {
        LabelFilterEntry {
            label,
            sdi: Some(sdi),
        }
    }

    /// Check if a label+SDI pair matches this filter entry.
    pub fn matches(&self, label: Label, sdi: Sdi) -> bool {
        if self.label != label {
            return false;
        }
        match self.sdi {
            None => true,
            Some(expected_sdi) => expected_sdi == sdi,
        }
    }
}

/// Label-based receive filter for ARINC 429 words.
///
/// Maintains a list of accepted label+SDI combinations. Words with labels
/// not in the accept list are discarded.
#[derive(Debug, Clone)]
pub struct LabelFilter {
    entries: Vec<LabelFilterEntry>,
}

impl LabelFilter {
    /// Create a new empty filter (rejects all words).
    pub fn new() -> Self {
        LabelFilter {
            entries: Vec::new(),
        }
    }

    /// Register interest in a specific label (any SDI).
    pub fn register(&mut self, label: Label) -> Result<(), BusError> {
        self.add_entry(LabelFilterEntry::any_sdi(label))
    }

    /// Register interest in a specific label+SDI combination.
    pub fn register_with_sdi(&mut self, label: Label, sdi: Sdi) -> Result<(), BusError> {
        self.add_entry(LabelFilterEntry::with_sdi(label, sdi))
    }

    /// Add a filter entry.
    fn add_entry(&mut self, entry: LabelFilterEntry) -> Result<(), BusError> {
        if self.entries.len() >= MAX_LABEL_FILTERS {
            return Err(BusError::FilterTableFull);
        }
        // Avoid duplicates
        if !self.entries.contains(&entry) {
            self.entries.push(entry);
        }
        Ok(())
    }

    /// Unregister a label (removes all entries for this label regardless of SDI).
    pub fn unregister(&mut self, label: Label) {
        self.entries.retain(|e| e.label != label);
    }

    /// Unregister a specific label+SDI combination.
    pub fn unregister_with_sdi(&mut self, label: Label, sdi: Sdi) {
        self.entries
            .retain(|e| !(e.label == label && e.sdi == Some(sdi)));
    }

    /// Clear all registered labels.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the number of registered filter entries.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Check if a word with the given label and SDI should be accepted.
    pub fn accepts(&self, label: Label, sdi: Sdi) -> bool {
        self.entries.iter().any(|e| e.matches(label, sdi))
    }

    /// Check if a raw 32-bit word should be accepted.
    ///
    /// Extracts the label and SDI from the raw word and checks against
    /// registered filters. Does not verify parity.
    pub fn accepts_raw(&self, raw_word: u32) -> bool {
        let label = Label::from_wire((raw_word & 0xFF) as u8);
        let sdi_raw = ((raw_word >> 8) & 0x03) as u8;
        // SDI::from_raw cannot fail for values 0-3
        if let Ok(sdi) = Sdi::from_raw(sdi_raw) {
            self.accepts(label, sdi)
        } else {
            false
        }
    }
}

impl Default for LabelFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_filter_rejects_all() {
        let filter = LabelFilter::new();
        assert!(!filter.accepts(Label::new(0o310), Sdi::Zero));
        assert!(!filter.accepts(Label::new(0), Sdi::Zero));
    }

    #[test]
    fn test_register_label_any_sdi() {
        let mut filter = LabelFilter::new();
        filter.register(Label::new(0o310)).unwrap();

        assert!(filter.accepts(Label::new(0o310), Sdi::Zero));
        assert!(filter.accepts(Label::new(0o310), Sdi::One));
        assert!(filter.accepts(Label::new(0o310), Sdi::Two));
        assert!(filter.accepts(Label::new(0o310), Sdi::Three));
        assert!(!filter.accepts(Label::new(0o311), Sdi::Zero));
    }

    #[test]
    fn test_register_label_specific_sdi() {
        let mut filter = LabelFilter::new();
        filter
            .register_with_sdi(Label::new(0o310), Sdi::One)
            .unwrap();

        assert!(!filter.accepts(Label::new(0o310), Sdi::Zero));
        assert!(filter.accepts(Label::new(0o310), Sdi::One));
        assert!(!filter.accepts(Label::new(0o310), Sdi::Two));
    }

    #[test]
    fn test_multiple_labels() {
        let mut filter = LabelFilter::new();
        filter.register(Label::new(0o310)).unwrap();
        filter.register(Label::new(0o311)).unwrap();
        filter.register(Label::new(0o312)).unwrap();

        assert!(filter.accepts(Label::new(0o310), Sdi::Zero));
        assert!(filter.accepts(Label::new(0o311), Sdi::Zero));
        assert!(filter.accepts(Label::new(0o312), Sdi::Zero));
        assert!(!filter.accepts(Label::new(0o313), Sdi::Zero));
    }

    #[test]
    fn test_unregister_label() {
        let mut filter = LabelFilter::new();
        filter.register(Label::new(0o310)).unwrap();
        filter.register(Label::new(0o311)).unwrap();

        filter.unregister(Label::new(0o310));
        assert!(!filter.accepts(Label::new(0o310), Sdi::Zero));
        assert!(filter.accepts(Label::new(0o311), Sdi::Zero));
    }

    #[test]
    fn test_unregister_with_sdi() {
        let mut filter = LabelFilter::new();
        filter
            .register_with_sdi(Label::new(0o310), Sdi::Zero)
            .unwrap();
        filter
            .register_with_sdi(Label::new(0o310), Sdi::One)
            .unwrap();

        filter.unregister_with_sdi(Label::new(0o310), Sdi::Zero);
        assert!(!filter.accepts(Label::new(0o310), Sdi::Zero));
        assert!(filter.accepts(Label::new(0o310), Sdi::One));
    }

    #[test]
    fn test_clear() {
        let mut filter = LabelFilter::new();
        filter.register(Label::new(0o310)).unwrap();
        filter.register(Label::new(0o311)).unwrap();

        filter.clear();
        assert_eq!(filter.count(), 0);
        assert!(!filter.accepts(Label::new(0o310), Sdi::Zero));
    }

    #[test]
    fn test_no_duplicates() {
        let mut filter = LabelFilter::new();
        filter.register(Label::new(0o310)).unwrap();
        filter.register(Label::new(0o310)).unwrap(); // duplicate
        assert_eq!(filter.count(), 1);
    }

    #[test]
    fn test_filter_table_full() {
        let mut filter = LabelFilter::new();
        for i in 0..MAX_LABEL_FILTERS {
            filter.register(Label::new(i as u8)).unwrap();
        }
        // Should fail on next add (but labels wrap at 256 so this may add duplicates)
        // Use register_with_sdi to ensure uniqueness
        let mut filter2 = LabelFilter::new();
        for i in 0..MAX_LABEL_FILTERS {
            filter2
                .register_with_sdi(
                    Label::new((i % 256) as u8),
                    Sdi::from_raw((i / 256) as u8 % 4).unwrap(),
                )
                .unwrap();
        }
        assert_eq!(
            filter2.register(Label::new(0)),
            Err(BusError::FilterTableFull)
        );
    }

    #[test]
    fn test_accepts_raw() {
        let mut filter = LabelFilter::new();
        filter.register(Label::new(0o310)).unwrap();

        // Build a raw word with label 0o310
        use super::super::word::{Arinc429Word, Ssm};
        let word = Arinc429Word {
            label: Label::new(0o310),
            sdi: Sdi::Zero,
            data: 0,
            ssm: Ssm::NormalOp,
        };
        let raw = word.encode();
        assert!(filter.accepts_raw(raw));

        // Different label should not match
        let word2 = Arinc429Word {
            label: Label::new(0o200),
            sdi: Sdi::Zero,
            data: 0,
            ssm: Ssm::NormalOp,
        };
        let raw2 = word2.encode();
        assert!(!filter.accepts_raw(raw2));
    }

    #[test]
    fn test_default() {
        let filter = LabelFilter::default();
        assert_eq!(filter.count(), 0);
    }
}
