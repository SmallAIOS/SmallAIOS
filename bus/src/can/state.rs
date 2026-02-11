// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CAN bus error state machine per ISO 11898.
//!
//! Implements the three-state error management model:
//! - **Error Active**: Normal operation. TEC and REC both <= 127.
//! - **Error Passive**: Degraded operation. TEC or REC > 127.
//! - **Bus Off**: No bus activity. TEC > 255.
//!
//! Transitions follow ISO 11898 rules:
//! - Error Active -> Error Passive: TEC > 127 or REC > 127
//! - Error Passive -> Bus Off: TEC > 255
//! - Bus Off -> Error Active: 128 occurrences of 11 recessive bits observed
//! - Successful TX/RX decrements the relevant counter by 1

/// CAN bus error state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusState {
    /// Normal operation. Active error flags are sent on errors.
    ErrorActive,
    /// Degraded operation. Passive error flags are sent on errors.
    ErrorPassive,
    /// No bus activity. Controller is disconnected from the bus.
    BusOff,
}

/// Event that may trigger a state transition or counter update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusEvent {
    /// A frame was successfully transmitted.
    TxSuccess,
    /// A frame was successfully received.
    RxSuccess,
    /// A transmit error occurred (increments TEC by the given amount).
    TxError(u16),
    /// A receive error occurred (increments REC by the given amount).
    RxError(u16),
    /// An 11-bit recessive sequence was observed during Bus Off recovery.
    RecessiveSequence,
}

/// Notification emitted when a state transition occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateChange {
    /// Transitioned to Error Active (from Error Passive or Bus Off recovery).
    EnteredErrorActive,
    /// Transitioned to Error Passive.
    EnteredErrorPassive,
    /// Transitioned to Bus Off.
    EnteredBusOff,
}

/// CAN bus error state machine.
///
/// Tracks the Transmit Error Counter (TEC), Receive Error Counter (REC),
/// current bus state, and Bus Off recovery progress.
#[derive(Debug, Clone)]
pub struct BusStateMachine {
    /// Transmit Error Counter.
    tec: u16,
    /// Receive Error Counter.
    rec: u16,
    /// Current bus state.
    state: BusState,
    /// Number of 11-recessive-bit sequences observed during Bus Off recovery.
    recovery_count: u8,
}

impl BusStateMachine {
    /// Threshold for transition from Error Active to Error Passive.
    const PASSIVE_THRESHOLD: u16 = 127;
    /// Threshold for transition from Error Passive to Bus Off.
    const BUS_OFF_THRESHOLD: u16 = 255;
    /// Number of 11-recessive-bit sequences needed for Bus Off recovery.
    const RECOVERY_SEQUENCES: u8 = 128;

    /// Create a new state machine in the Error Active state.
    pub fn new() -> Self {
        BusStateMachine {
            tec: 0,
            rec: 0,
            state: BusState::ErrorActive,
            recovery_count: 0,
        }
    }

    /// Returns the current bus state.
    pub fn state(&self) -> BusState {
        self.state
    }

    /// Returns the current Transmit Error Counter.
    pub fn tec(&self) -> u16 {
        self.tec
    }

    /// Returns the current Receive Error Counter.
    pub fn rec(&self) -> u16 {
        self.rec
    }

    /// Returns true if the controller can transmit (not in Bus Off state).
    pub fn can_transmit(&self) -> bool {
        self.state != BusState::BusOff
    }

    /// Process a bus event and return any state change notification.
    pub fn process_event(&mut self, event: BusEvent) -> Option<StateChange> {
        match event {
            BusEvent::TxSuccess => {
                if self.state == BusState::BusOff {
                    return None;
                }
                self.tec = self.tec.saturating_sub(1);
                self.update_state()
            }
            BusEvent::RxSuccess => {
                if self.state == BusState::BusOff {
                    return None;
                }
                self.rec = self.rec.saturating_sub(1);
                self.update_state()
            }
            BusEvent::TxError(increment) => {
                if self.state == BusState::BusOff {
                    return None;
                }
                self.tec = self.tec.saturating_add(increment);
                self.update_state()
            }
            BusEvent::RxError(increment) => {
                if self.state == BusState::BusOff {
                    return None;
                }
                self.rec = self.rec.saturating_add(increment);
                self.update_state()
            }
            BusEvent::RecessiveSequence => {
                if self.state != BusState::BusOff {
                    return None;
                }
                self.recovery_count = self.recovery_count.saturating_add(1);
                if self.recovery_count >= Self::RECOVERY_SEQUENCES {
                    self.tec = 0;
                    self.rec = 0;
                    self.recovery_count = 0;
                    self.state = BusState::ErrorActive;
                    Some(StateChange::EnteredErrorActive)
                } else {
                    None
                }
            }
        }
    }

    /// Re-evaluate the bus state based on current counters.
    /// Returns a state change notification if a transition occurred.
    fn update_state(&mut self) -> Option<StateChange> {
        let old_state = self.state;

        if self.tec > Self::BUS_OFF_THRESHOLD {
            self.state = BusState::BusOff;
            self.recovery_count = 0;
        } else if self.tec > Self::PASSIVE_THRESHOLD || self.rec > Self::PASSIVE_THRESHOLD {
            self.state = BusState::ErrorPassive;
        } else {
            self.state = BusState::ErrorActive;
        }

        if self.state != old_state {
            match self.state {
                BusState::ErrorActive => Some(StateChange::EnteredErrorActive),
                BusState::ErrorPassive => Some(StateChange::EnteredErrorPassive),
                BusState::BusOff => Some(StateChange::EnteredBusOff),
            }
        } else {
            None
        }
    }
}

impl Default for BusStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let sm = BusStateMachine::new();
        assert_eq!(sm.state(), BusState::ErrorActive);
        assert_eq!(sm.tec(), 0);
        assert_eq!(sm.rec(), 0);
        assert!(sm.can_transmit());
    }

    #[test]
    fn test_default_is_error_active() {
        let sm = BusStateMachine::default();
        assert_eq!(sm.state(), BusState::ErrorActive);
    }

    // -----------------------------------------------------------------------
    // Error Active -> Error Passive transitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_tec_exceeds_127_transitions_to_passive() {
        let mut sm = BusStateMachine::new();
        let change = sm.process_event(BusEvent::TxError(128));
        assert_eq!(sm.state(), BusState::ErrorPassive);
        assert_eq!(change, Some(StateChange::EnteredErrorPassive));
        assert!(sm.can_transmit());
    }

    #[test]
    fn test_rec_exceeds_127_transitions_to_passive() {
        let mut sm = BusStateMachine::new();
        let change = sm.process_event(BusEvent::RxError(128));
        assert_eq!(sm.state(), BusState::ErrorPassive);
        assert_eq!(change, Some(StateChange::EnteredErrorPassive));
    }

    #[test]
    fn test_tec_at_127_stays_active() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(127));
        assert_eq!(sm.state(), BusState::ErrorActive);
        assert_eq!(sm.tec(), 127);
    }

    #[test]
    fn test_rec_at_127_stays_active() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::RxError(127));
        assert_eq!(sm.state(), BusState::ErrorActive);
    }

    // -----------------------------------------------------------------------
    // Error Passive -> Bus Off transition
    // -----------------------------------------------------------------------

    #[test]
    fn test_tec_exceeds_255_transitions_to_bus_off() {
        let mut sm = BusStateMachine::new();
        let change = sm.process_event(BusEvent::TxError(256));
        assert_eq!(sm.state(), BusState::BusOff);
        assert_eq!(change, Some(StateChange::EnteredBusOff));
        assert!(!sm.can_transmit());
    }

    #[test]
    fn test_tec_at_255_stays_passive() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(255));
        // TEC=255 is still Error Passive (threshold is 255, not >=255)
        assert_eq!(sm.state(), BusState::ErrorPassive);
    }

    #[test]
    fn test_rec_high_does_not_cause_bus_off() {
        // Only TEC > 255 causes Bus Off, not REC
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::RxError(500));
        assert_eq!(sm.state(), BusState::ErrorPassive);
        assert!(sm.can_transmit());
    }

    // -----------------------------------------------------------------------
    // Bus Off recovery
    // -----------------------------------------------------------------------

    #[test]
    fn test_bus_off_recovery_128_sequences() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(256));
        assert_eq!(sm.state(), BusState::BusOff);

        // 127 sequences: still Bus Off
        for _ in 0..127 {
            let change = sm.process_event(BusEvent::RecessiveSequence);
            assert_eq!(change, None);
            assert_eq!(sm.state(), BusState::BusOff);
        }

        // 128th sequence: recover to Error Active
        let change = sm.process_event(BusEvent::RecessiveSequence);
        assert_eq!(change, Some(StateChange::EnteredErrorActive));
        assert_eq!(sm.state(), BusState::ErrorActive);
        assert_eq!(sm.tec(), 0);
        assert_eq!(sm.rec(), 0);
        assert!(sm.can_transmit());
    }

    #[test]
    fn test_recessive_sequence_ignored_when_not_bus_off() {
        let mut sm = BusStateMachine::new();
        let change = sm.process_event(BusEvent::RecessiveSequence);
        assert_eq!(change, None);
        assert_eq!(sm.state(), BusState::ErrorActive);
    }

    // -----------------------------------------------------------------------
    // Counter decrement on success
    // -----------------------------------------------------------------------

    #[test]
    fn test_tx_success_decrements_tec() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(10));
        assert_eq!(sm.tec(), 10);
        sm.process_event(BusEvent::TxSuccess);
        assert_eq!(sm.tec(), 9);
    }

    #[test]
    fn test_rx_success_decrements_rec() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::RxError(5));
        assert_eq!(sm.rec(), 5);
        sm.process_event(BusEvent::RxSuccess);
        assert_eq!(sm.rec(), 4);
    }

    #[test]
    fn test_tx_success_at_zero_stays_zero() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxSuccess);
        assert_eq!(sm.tec(), 0);
    }

    #[test]
    fn test_rx_success_at_zero_stays_zero() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::RxSuccess);
        assert_eq!(sm.rec(), 0);
    }

    // -----------------------------------------------------------------------
    // Recovery from Error Passive to Error Active
    // -----------------------------------------------------------------------

    #[test]
    fn test_passive_to_active_on_counter_decrease() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(128));
        assert_eq!(sm.state(), BusState::ErrorPassive);

        // Successful TX decrements TEC: 128 -> 127 -> Error Active
        let change = sm.process_event(BusEvent::TxSuccess);
        assert_eq!(sm.tec(), 127);
        assert_eq!(sm.state(), BusState::ErrorActive);
        assert_eq!(change, Some(StateChange::EnteredErrorActive));
    }

    // -----------------------------------------------------------------------
    // Bus Off ignores non-recovery events
    // -----------------------------------------------------------------------

    #[test]
    fn test_bus_off_ignores_tx_success() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(256));
        let change = sm.process_event(BusEvent::TxSuccess);
        assert_eq!(change, None);
        assert_eq!(sm.state(), BusState::BusOff);
    }

    #[test]
    fn test_bus_off_ignores_rx_success() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(256));
        let change = sm.process_event(BusEvent::RxSuccess);
        assert_eq!(change, None);
    }

    #[test]
    fn test_bus_off_ignores_tx_error() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(256));
        let change = sm.process_event(BusEvent::TxError(8));
        assert_eq!(change, None);
    }

    #[test]
    fn test_bus_off_ignores_rx_error() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(256));
        let change = sm.process_event(BusEvent::RxError(8));
        assert_eq!(change, None);
    }

    // -----------------------------------------------------------------------
    // Incremental error accumulation
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradual_tec_increase() {
        let mut sm = BusStateMachine::new();
        // Each TX error adds 8 (typical CAN error increment)
        for _ in 0..16 {
            sm.process_event(BusEvent::TxError(8));
        }
        // 16 * 8 = 128 -> Error Passive
        assert_eq!(sm.tec(), 128);
        assert_eq!(sm.state(), BusState::ErrorPassive);

        for _ in 0..16 {
            sm.process_event(BusEvent::TxError(8));
        }
        // 32 * 8 = 256 -> Bus Off
        assert_eq!(sm.tec(), 256);
        assert_eq!(sm.state(), BusState::BusOff);
    }

    // -----------------------------------------------------------------------
    // Counter saturation
    // -----------------------------------------------------------------------

    #[test]
    fn test_tec_saturates() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::TxError(u16::MAX));
        assert_eq!(sm.state(), BusState::BusOff);
        // Counter should saturate, not overflow
        assert_eq!(sm.tec(), u16::MAX);
    }

    #[test]
    fn test_rec_saturates() {
        let mut sm = BusStateMachine::new();
        sm.process_event(BusEvent::RxError(u16::MAX));
        assert_eq!(sm.state(), BusState::ErrorPassive);
        assert_eq!(sm.rec(), u16::MAX);
    }
}
