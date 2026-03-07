// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 429 fixed-rate transmit scheduler.
//!
//! Schedules periodic transmission of ARINC 429 words at per-label
//! configurable rates. The scheduler determines which words are due for
//! transmission based on elapsed time, maintaining the minimum inter-word
//! gap required by the ARINC 429 specification.
//!
//! # Inter-word gap
//!
//! ARINC 429 requires a minimum 4-bit-time gap between consecutive words:
//! - High speed (100 kbps): 4 * 10us = 40us minimum gap
//! - Low speed (12.5 kbps): 4 * 80us = 320us minimum gap

use alloc::vec::Vec;

use super::word::Label;
use crate::BusError;

/// Maximum number of scheduled labels.
pub const MAX_SCHEDULED_LABELS: usize = 128;

/// A scheduled label entry with its transmission period and current data.
#[derive(Debug, Clone)]
struct ScheduleEntry {
    /// The label being scheduled.
    label: Label,
    /// Transmission period in microseconds.
    period_us: u64,
    /// Timestamp of last transmission (microseconds, system monotonic).
    last_tx_us: u64,
    /// The current 32-bit word to transmit (pre-encoded with parity).
    current_word: u32,
    /// Whether this entry is ready for its first transmission.
    first_tx_pending: bool,
}

/// Fixed-rate transmit scheduler for ARINC 429 words.
///
/// Each label is independently scheduled at a configurable rate. The
/// scheduler uses monotonic timestamps to determine which words are
/// due for transmission.
#[derive(Debug, Clone)]
pub struct TxScheduler {
    entries: Vec<ScheduleEntry>,
    /// Minimum inter-word gap in microseconds.
    min_gap_us: u64,
    /// Timestamp of the last word transmitted (any label).
    last_any_tx_us: u64,
    /// Whether any word has ever been transmitted.
    has_transmitted: bool,
}

impl TxScheduler {
    /// Minimum inter-word gap for high speed (100 kbps): 40 microseconds.
    pub const HIGH_SPEED_GAP_US: u64 = 40;

    /// Minimum inter-word gap for low speed (12.5 kbps): 320 microseconds.
    pub const LOW_SPEED_GAP_US: u64 = 320;

    /// Create a new scheduler with the given minimum inter-word gap.
    pub fn new(min_gap_us: u64) -> Self {
        TxScheduler {
            entries: Vec::new(),
            min_gap_us,
            last_any_tx_us: 0,
            has_transmitted: false,
        }
    }

    /// Create a scheduler configured for high-speed (100 kbps) operation.
    pub fn high_speed() -> Self {
        Self::new(Self::HIGH_SPEED_GAP_US)
    }

    /// Create a scheduler configured for low-speed (12.5 kbps) operation.
    pub fn low_speed() -> Self {
        Self::new(Self::LOW_SPEED_GAP_US)
    }

    /// Schedule a label for periodic transmission at the given rate in Hz.
    ///
    /// `initial_word` is the first 32-bit word to transmit (pre-encoded).
    /// Returns [`BusError::SchedulerFull`] if the schedule table is full.
    /// Returns [`BusError::OutOfRange`] if `rate_hz` is zero.
    pub fn schedule(
        &mut self,
        label: Label,
        rate_hz: u32,
        initial_word: u32,
    ) -> Result<(), BusError> {
        if rate_hz == 0 {
            return Err(BusError::OutOfRange);
        }
        if self.entries.len() >= MAX_SCHEDULED_LABELS {
            return Err(BusError::SchedulerFull);
        }

        // Remove existing entry for this label if any
        self.entries.retain(|e| e.label != label);

        let period_us = 1_000_000u64 / rate_hz as u64;

        self.entries.push(ScheduleEntry {
            label,
            period_us,
            last_tx_us: 0,
            current_word: initial_word,
            first_tx_pending: true,
        });

        Ok(())
    }

    /// Remove a label from the schedule.
    pub fn unschedule(&mut self, label: Label) {
        self.entries.retain(|e| e.label != label);
    }

    /// Update the data word for a scheduled label.
    ///
    /// The new word will be used starting with the next scheduled transmission.
    /// Returns `false` if the label is not scheduled.
    pub fn update_word(&mut self, label: Label, new_word: u32) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.label == label) {
            entry.current_word = new_word;
            true
        } else {
            false
        }
    }

    /// Returns the number of scheduled labels.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Clear all scheduled labels.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Poll for the next word that is due for transmission.
    ///
    /// `now_us` is the current monotonic time in microseconds.
    /// Returns `Some(word)` if a word is due and the inter-word gap has
    /// elapsed, or `None` if no word is due.
    ///
    /// When a word is returned, its `last_tx_us` is updated to `now_us`.
    pub fn poll(&mut self, now_us: u64) -> Option<u32> {
        if !self.gap_elapsed(now_us) {
            return None;
        }

        let idx = self.find_most_overdue(now_us)?;
        Some(self.commit_transmission(idx, now_us))
    }

    /// Returns `true` when the minimum inter-word gap has elapsed (or no
    /// word has been transmitted yet).
    fn gap_elapsed(&self, now_us: u64) -> bool {
        !self.has_transmitted || now_us >= self.last_any_tx_us + self.min_gap_us
    }

    /// Scan entries and return the index of the most overdue entry, if any.
    /// Entries pending their first transmission are always preferred.
    fn find_most_overdue(&self, now_us: u64) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_overdue: u64 = 0;
        let mut best_is_first: bool = false;

        for (i, entry) in self.entries.iter().enumerate() {
            if entry.first_tx_pending {
                self.consider_first_tx(i, &mut best_idx, &mut best_overdue, &mut best_is_first);
            } else {
                self.consider_overdue(
                    i,
                    entry,
                    now_us,
                    best_is_first,
                    &mut best_idx,
                    &mut best_overdue,
                );
            }
        }

        best_idx
    }

    /// Consider a first-transmission-pending entry as candidate.
    fn consider_first_tx(
        &self,
        i: usize,
        best_idx: &mut Option<usize>,
        best_overdue: &mut u64,
        best_is_first: &mut bool,
    ) {
        if !*best_is_first {
            *best_idx = Some(i);
            *best_overdue = u64::MAX;
            *best_is_first = true;
        }
    }

    /// Consider an overdue entry as candidate (only if no first-tx entry found).
    fn consider_overdue(
        &self,
        i: usize,
        entry: &ScheduleEntry,
        now_us: u64,
        best_is_first: bool,
        best_idx: &mut Option<usize>,
        best_overdue: &mut u64,
    ) {
        if best_is_first {
            return;
        }
        let next_due = entry.last_tx_us + entry.period_us;
        if now_us >= next_due {
            let overdue = now_us - next_due;
            if best_idx.is_none() || overdue > *best_overdue {
                *best_idx = Some(i);
                *best_overdue = overdue;
            }
        }
    }

    /// Mark entry `idx` as transmitted at `now_us` and return its word.
    fn commit_transmission(&mut self, idx: usize, now_us: u64) -> u32 {
        let entry = &mut self.entries[idx];
        entry.last_tx_us = now_us;
        entry.first_tx_pending = false;
        self.last_any_tx_us = now_us;
        self.has_transmitted = true;
        entry.current_word
    }

    /// Returns the minimum inter-word gap in microseconds.
    pub fn min_gap_us(&self) -> u64 {
        self.min_gap_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_word(label_val: u8) -> u32 {
        // Simple test word: just the label value as a marker
        label_val as u32 | 0x8000_0000 // set parity bit for identification
    }

    #[test]
    fn test_new_scheduler() {
        let sched = TxScheduler::high_speed();
        assert_eq!(sched.count(), 0);
        assert_eq!(sched.min_gap_us(), TxScheduler::HIGH_SPEED_GAP_US);
    }

    #[test]
    fn test_low_speed_scheduler() {
        let sched = TxScheduler::low_speed();
        assert_eq!(sched.min_gap_us(), TxScheduler::LOW_SPEED_GAP_US);
    }

    #[test]
    fn test_schedule_label() {
        let mut sched = TxScheduler::high_speed();
        sched
            .schedule(Label::new(0o310), 50, make_word(0o310))
            .unwrap();
        assert_eq!(sched.count(), 1);
    }

    #[test]
    fn test_schedule_zero_rate() {
        let mut sched = TxScheduler::high_speed();
        assert_eq!(
            sched.schedule(Label::new(0o310), 0, 0),
            Err(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_schedule_replaces_existing() {
        let mut sched = TxScheduler::high_speed();
        sched.schedule(Label::new(0o310), 50, make_word(1)).unwrap();
        sched
            .schedule(Label::new(0o310), 100, make_word(2))
            .unwrap();
        assert_eq!(sched.count(), 1);
    }

    #[test]
    fn test_unschedule() {
        let mut sched = TxScheduler::high_speed();
        sched
            .schedule(Label::new(0o310), 50, make_word(0o310))
            .unwrap();
        sched.unschedule(Label::new(0o310));
        assert_eq!(sched.count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut sched = TxScheduler::high_speed();
        sched.schedule(Label::new(0o310), 50, make_word(1)).unwrap();
        sched.schedule(Label::new(0o311), 25, make_word(2)).unwrap();
        sched.clear();
        assert_eq!(sched.count(), 0);
    }

    // -----------------------------------------------------------------------
    // Poll tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_poll_immediate_first_word() {
        let mut sched = TxScheduler::high_speed();
        sched
            .schedule(Label::new(0o310), 50, make_word(0o310))
            .unwrap();

        // First poll at t=0: word is due (last_tx_us=0, period has passed)
        let word = sched.poll(0);
        assert_eq!(word, Some(make_word(0o310)));
    }

    #[test]
    fn test_poll_respects_period() {
        let mut sched = TxScheduler::high_speed();
        // 50 Hz = 20000us period
        sched
            .schedule(Label::new(0o310), 50, make_word(0o310))
            .unwrap();

        // First transmission at t=0
        sched.poll(0);

        // t=10000us: not yet due
        assert_eq!(sched.poll(10_000), None);

        // t=20000us: due again
        let word = sched.poll(20_000);
        assert_eq!(word, Some(make_word(0o310)));
    }

    #[test]
    fn test_poll_respects_inter_word_gap() {
        let mut sched = TxScheduler::high_speed();
        sched.schedule(Label::new(0o310), 50, make_word(1)).unwrap();
        sched.schedule(Label::new(0o311), 50, make_word(2)).unwrap();

        // Both due at t=0
        let w1 = sched.poll(0);
        assert!(w1.is_some());

        // Second word at t=10us: too soon (gap=40us)
        assert_eq!(sched.poll(10), None);

        // Second word at t=40us: gap met
        let w2 = sched.poll(40);
        assert!(w2.is_some());
    }

    #[test]
    fn test_poll_most_overdue_first() {
        let mut sched = TxScheduler::new(0); // no gap for simplicity
                                             // Label 1 at 100 Hz (10000us period)
        sched.schedule(Label::new(1), 100, make_word(1)).unwrap();
        // Label 2 at 50 Hz (20000us period)
        sched.schedule(Label::new(2), 50, make_word(2)).unwrap();

        // Transmit both at t=0
        sched.poll(0);
        sched.poll(0);

        // At t=10000us, label 1 is due. Label 2 not yet.
        let word = sched.poll(10_000);
        assert_eq!(word, Some(make_word(1)));

        // At t=20000us, both are due. Both have same overdue time.
        // Label 2 might be returned (or label 1 - depends on iteration)
        let word = sched.poll(20_000);
        assert!(word.is_some());
    }

    #[test]
    fn test_update_word() {
        let mut sched = TxScheduler::new(0);
        sched.schedule(Label::new(0o310), 50, make_word(1)).unwrap();

        // Update the word
        assert!(sched.update_word(Label::new(0o310), make_word(2)));

        // Next poll should return the updated word
        let word = sched.poll(0);
        assert_eq!(word, Some(make_word(2)));
    }

    #[test]
    fn test_update_word_not_scheduled() {
        let mut sched = TxScheduler::new(0);
        assert!(!sched.update_word(Label::new(0o310), make_word(1)));
    }

    #[test]
    fn test_update_does_not_disrupt_timing() {
        let mut sched = TxScheduler::new(0);
        // 50 Hz = 20000us period
        sched.schedule(Label::new(0o310), 50, make_word(1)).unwrap();

        // Transmit at t=0
        sched.poll(0);

        // Update at t=5000us
        sched.update_word(Label::new(0o310), make_word(2));

        // At t=10000us: not yet due (period=20000us)
        assert_eq!(sched.poll(10_000), None);

        // At t=20000us: due, returns updated word
        assert_eq!(sched.poll(20_000), Some(make_word(2)));
    }

    #[test]
    fn test_empty_scheduler_poll() {
        let mut sched = TxScheduler::high_speed();
        assert_eq!(sched.poll(0), None);
        assert_eq!(sched.poll(1_000_000), None);
    }

    #[test]
    fn test_scheduler_full() {
        let mut sched = TxScheduler::high_speed();
        for i in 0..MAX_SCHEDULED_LABELS {
            sched.schedule(Label::new(i as u8), 50, 0).unwrap();
        }
        assert_eq!(
            sched.schedule(Label::new(255), 50, 0),
            Err(BusError::SchedulerFull)
        );
    }

    // -----------------------------------------------------------------------
    // Multiple rates interleaving
    // -----------------------------------------------------------------------

    #[test]
    fn test_different_rates_interleave() {
        let mut sched = TxScheduler::new(0);
        // Label 1 at 50 Hz (20ms period = 20000us)
        sched.schedule(Label::new(1), 50, make_word(1)).unwrap();
        // Label 2 at 25 Hz (40ms period = 40000us)
        sched.schedule(Label::new(2), 25, make_word(2)).unwrap();

        // t=0: both due
        sched.poll(0);
        sched.poll(0);

        // t=20000: label 1 due, label 2 not
        let w = sched.poll(20_000);
        assert_eq!(w, Some(make_word(1)));
        assert_eq!(sched.poll(20_000), None); // label 2 not due yet

        // t=40000: both due again
        let w = sched.poll(40_000);
        assert!(w.is_some());
        let w = sched.poll(40_000);
        assert!(w.is_some());
    }
}
