// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SpaceWire link interface state machine.
//!
//! Implements the 6-state link FSM per ECSS-E-ST-50-12C:
//! ErrorReset -> ErrorWait -> Ready -> Started -> Connecting -> Run.
//! Data transfer is only permitted in the Run state.

/// Disconnect timeout in nanoseconds (850 ns).
pub const DISCONNECT_TIMEOUT_NS: u64 = 850;

/// Reset timeout in nanoseconds (6.4 us).
pub const RESET_TIMEOUT_NS: u64 = 6_400;

/// Link interface state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkState {
    /// Link is being reset.
    ErrorReset,
    /// Waiting after reset before attempting reconnection.
    ErrorWait,
    /// Ready for connection (waiting for link start or autostart).
    Ready,
    /// Link start has been commanded; sending NULLs.
    Started,
    /// Both ends sending NULLs; awaiting FCT exchange.
    Connecting,
    /// Link is operational; data transfer permitted.
    Run,
}

/// Events that drive the link state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkEvent {
    /// Reset timeout has elapsed.
    ResetTimeout,
    /// Error wait timeout has elapsed.
    ErrorWaitTimeout,
    /// Link start commanded by application.
    LinkStart,
    /// Autostart enabled and NULL received.
    AutostartNull,
    /// NULL character received from remote end.
    NullReceived,
    /// FCT (Flow Control Token) received from remote end.
    FctReceived,
    /// Disconnect timeout elapsed (no character received within 850 ns).
    DisconnectTimeout,
    /// Link error detected (parity error, escape error, etc.).
    LinkError,
    /// Link disable commanded by application.
    LinkDisable,
}

/// Link state machine with transition tracking.
#[derive(Debug)]
pub struct LinkStateMachine {
    /// Current link state.
    state: LinkState,
    /// Whether autostart is enabled.
    autostart: bool,
    /// Whether link start has been commanded.
    link_enabled: bool,
    /// Count of error resets (for diagnostics).
    error_reset_count: u64,
    /// Count of successful Run entries.
    run_count: u64,
    /// Count of disconnect events.
    disconnect_count: u64,
}

impl LinkStateMachine {
    /// Create a new link state machine in ErrorReset state.
    pub fn new(autostart: bool) -> Self {
        Self {
            state: LinkState::ErrorReset,
            autostart,
            link_enabled: false,
            error_reset_count: 0,
            run_count: 0,
            disconnect_count: 0,
        }
    }

    /// Returns the current link state.
    pub fn state(&self) -> LinkState {
        self.state
    }

    /// Returns true if data transfer is permitted (link is in Run state).
    pub fn can_transfer(&self) -> bool {
        self.state == LinkState::Run
    }

    /// Enable the link (equivalent to link start command).
    pub fn enable(&mut self) {
        self.link_enabled = true;
    }

    /// Disable the link.
    pub fn disable(&mut self) {
        self.link_enabled = false;
    }

    /// Process a link event and return the new state.
    pub fn process_event(&mut self, event: LinkEvent) -> LinkState {
        let new_state = match (self.state, event) {
            // ErrorReset -> ErrorWait on reset timeout
            (LinkState::ErrorReset, LinkEvent::ResetTimeout) => LinkState::ErrorWait,

            // ErrorWait -> Ready on error wait timeout
            (LinkState::ErrorWait, LinkEvent::ErrorWaitTimeout) => LinkState::Ready,

            // Ready -> Started on link start or autostart
            (LinkState::Ready, LinkEvent::LinkStart) if self.link_enabled => LinkState::Started,
            (LinkState::Ready, LinkEvent::AutostartNull) if self.autostart => LinkState::Started,

            // Started -> Connecting on NULL received
            (LinkState::Started, LinkEvent::NullReceived) => LinkState::Connecting,

            // Connecting -> Run on FCT received
            (LinkState::Connecting, LinkEvent::FctReceived) => {
                self.run_count += 1;
                LinkState::Run
            }

            // Any state -> ErrorReset on disconnect, error, or disable
            (_, LinkEvent::DisconnectTimeout) if self.state == LinkState::Run => {
                self.disconnect_count += 1;
                self.error_reset_count += 1;
                LinkState::ErrorReset
            }
            (_, LinkEvent::LinkError) => {
                self.error_reset_count += 1;
                LinkState::ErrorReset
            }
            (_, LinkEvent::LinkDisable) => LinkState::ErrorReset,

            // No transition for unhandled events
            _ => self.state,
        };
        self.state = new_state;
        new_state
    }

    /// Returns the error reset count.
    pub fn error_reset_count(&self) -> u64 {
        self.error_reset_count
    }

    /// Returns the successful Run entry count.
    pub fn run_count(&self) -> u64 {
        self.run_count
    }

    /// Returns the disconnect event count.
    pub fn disconnect_count(&self) -> u64 {
        self.disconnect_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let fsm = LinkStateMachine::new(false);
        assert_eq!(fsm.state(), LinkState::ErrorReset);
        assert!(!fsm.can_transfer());
    }

    #[test]
    fn test_normal_initialization_sequence() {
        let mut fsm = LinkStateMachine::new(false);
        fsm.enable();

        assert_eq!(
            fsm.process_event(LinkEvent::ResetTimeout),
            LinkState::ErrorWait
        );
        assert_eq!(
            fsm.process_event(LinkEvent::ErrorWaitTimeout),
            LinkState::Ready
        );
        assert_eq!(fsm.process_event(LinkEvent::LinkStart), LinkState::Started);
        assert_eq!(
            fsm.process_event(LinkEvent::NullReceived),
            LinkState::Connecting
        );
        assert_eq!(fsm.process_event(LinkEvent::FctReceived), LinkState::Run);
        assert!(fsm.can_transfer());
        assert_eq!(fsm.run_count(), 1);
    }

    #[test]
    fn test_autostart() {
        let mut fsm = LinkStateMachine::new(true);

        assert_eq!(
            fsm.process_event(LinkEvent::ResetTimeout),
            LinkState::ErrorWait
        );
        assert_eq!(
            fsm.process_event(LinkEvent::ErrorWaitTimeout),
            LinkState::Ready
        );
        assert_eq!(
            fsm.process_event(LinkEvent::AutostartNull),
            LinkState::Started
        );
        assert_eq!(
            fsm.process_event(LinkEvent::NullReceived),
            LinkState::Connecting
        );
        assert_eq!(fsm.process_event(LinkEvent::FctReceived), LinkState::Run);
    }

    #[test]
    fn test_autostart_disabled() {
        let mut fsm = LinkStateMachine::new(false);
        fsm.process_event(LinkEvent::ResetTimeout);
        fsm.process_event(LinkEvent::ErrorWaitTimeout);
        // AutostartNull should not transition when autostart is disabled
        assert_eq!(
            fsm.process_event(LinkEvent::AutostartNull),
            LinkState::Ready
        );
    }

    #[test]
    fn test_link_start_without_enable() {
        let mut fsm = LinkStateMachine::new(false);
        fsm.process_event(LinkEvent::ResetTimeout);
        fsm.process_event(LinkEvent::ErrorWaitTimeout);
        // LinkStart without enable should not transition
        assert_eq!(fsm.process_event(LinkEvent::LinkStart), LinkState::Ready);
    }

    #[test]
    fn test_disconnect_in_run() {
        let mut fsm = LinkStateMachine::new(false);
        fsm.enable();
        fsm.process_event(LinkEvent::ResetTimeout);
        fsm.process_event(LinkEvent::ErrorWaitTimeout);
        fsm.process_event(LinkEvent::LinkStart);
        fsm.process_event(LinkEvent::NullReceived);
        fsm.process_event(LinkEvent::FctReceived);
        assert_eq!(fsm.state(), LinkState::Run);

        assert_eq!(
            fsm.process_event(LinkEvent::DisconnectTimeout),
            LinkState::ErrorReset
        );
        assert_eq!(fsm.disconnect_count(), 1);
        assert_eq!(fsm.error_reset_count(), 1);
        assert!(!fsm.can_transfer());
    }

    #[test]
    fn test_link_error_from_any_state() {
        let mut fsm = LinkStateMachine::new(false);
        fsm.process_event(LinkEvent::ResetTimeout);
        fsm.process_event(LinkEvent::ErrorWaitTimeout);
        assert_eq!(fsm.state(), LinkState::Ready);

        assert_eq!(
            fsm.process_event(LinkEvent::LinkError),
            LinkState::ErrorReset
        );
        assert_eq!(fsm.error_reset_count(), 1);
    }

    #[test]
    fn test_link_disable() {
        let mut fsm = LinkStateMachine::new(false);
        fsm.enable();
        fsm.process_event(LinkEvent::ResetTimeout);
        fsm.process_event(LinkEvent::ErrorWaitTimeout);
        fsm.process_event(LinkEvent::LinkStart);
        assert_eq!(fsm.state(), LinkState::Started);

        assert_eq!(
            fsm.process_event(LinkEvent::LinkDisable),
            LinkState::ErrorReset
        );
    }

    #[test]
    fn test_recovery_after_disconnect() {
        let mut fsm = LinkStateMachine::new(false);
        fsm.enable();

        // First connection
        fsm.process_event(LinkEvent::ResetTimeout);
        fsm.process_event(LinkEvent::ErrorWaitTimeout);
        fsm.process_event(LinkEvent::LinkStart);
        fsm.process_event(LinkEvent::NullReceived);
        fsm.process_event(LinkEvent::FctReceived);
        assert_eq!(fsm.run_count(), 1);

        // Disconnect
        fsm.process_event(LinkEvent::DisconnectTimeout);

        // Recovery
        fsm.process_event(LinkEvent::ResetTimeout);
        fsm.process_event(LinkEvent::ErrorWaitTimeout);
        fsm.process_event(LinkEvent::LinkStart);
        fsm.process_event(LinkEvent::NullReceived);
        fsm.process_event(LinkEvent::FctReceived);
        assert_eq!(fsm.run_count(), 2);
        assert!(fsm.can_transfer());
    }

    #[test]
    fn test_ignored_events() {
        let mut fsm = LinkStateMachine::new(false);
        // FCT in ErrorReset should be ignored
        assert_eq!(
            fsm.process_event(LinkEvent::FctReceived),
            LinkState::ErrorReset
        );
        // NULL in ErrorReset should be ignored
        assert_eq!(
            fsm.process_event(LinkEvent::NullReceived),
            LinkState::ErrorReset
        );
    }
}
