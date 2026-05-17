// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HTTP/2 flow control window arithmetic (RFC 7540 § 5.2 and § 6.9).
//!
//! Each flow-control window is a signed 32-bit counter:
//!
//! - Sending a DATA frame of N bytes consumes N from the window.
//! - Receiving a WINDOW_UPDATE of M bytes adds M back (capped at
//!   `i32::MAX`).
//!
//! The window MUST NOT go negative on the sender side; this layer
//! enforces that with [`Http2Error::FlowControl`]. The receiver
//! side maintains the same arithmetic and emits WINDOW_UPDATE
//! when the available window drops below half of the initial size.

use super::{Http2Error, Result, DEFAULT_INITIAL_WINDOW_SIZE};

/// Flow-control window state for either the connection or a single
/// stream. The two are tracked independently per RFC 7540 § 5.2.1.
#[derive(Debug, Clone, Copy)]
pub struct Window {
    available: i64,
    initial: u32,
}

impl Window {
    /// Construct a window initialized to the negotiated initial size.
    pub fn new(initial: u32) -> Self {
        Self {
            available: initial as i64,
            initial,
        }
    }

    /// Default-initialised window (`SETTINGS_INITIAL_WINDOW_SIZE` = 65535).
    pub fn default_initial() -> Self {
        Self::new(DEFAULT_INITIAL_WINDOW_SIZE)
    }

    /// Currently-available send credit.
    pub fn available(&self) -> i64 {
        self.available
    }

    /// Try to consume `n` bytes of send credit. Fails on underflow.
    pub fn consume(&mut self, n: u32) -> Result<()> {
        let after = self.available - n as i64;
        if after < 0 {
            return Err(Http2Error::FlowControl);
        }
        self.available = after;
        Ok(())
    }

    /// Add `n` bytes of credit (i.e. process an inbound WINDOW_UPDATE).
    /// Saturates at `i32::MAX` per RFC 7540 § 6.9.1.
    pub fn replenish(&mut self, n: u32) -> Result<()> {
        let after = self.available.saturating_add(n as i64);
        if after > i32::MAX as i64 {
            return Err(Http2Error::FlowControl);
        }
        self.available = after;
        Ok(())
    }

    /// Returns true when the receiver should emit a WINDOW_UPDATE
    /// to refill credit (default policy: when below half the
    /// initial window).
    pub fn should_replenish(&self) -> bool {
        self.available < (self.initial as i64) / 2
    }

    /// Compute the increment the receiver should send to refill the
    /// window back to its initial size, returning 0 if no
    /// replenishment is needed.
    pub fn refill_increment(&self) -> u32 {
        if !self.should_replenish() {
            return 0;
        }
        let target = self.initial as i64;
        let delta = target - self.available;
        delta.clamp(0, i32::MAX as i64) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_window_is_65535() {
        let w = Window::default_initial();
        assert_eq!(w.available(), 65_535);
    }

    #[test]
    fn consume_within_credit() {
        let mut w = Window::new(100);
        w.consume(60).unwrap();
        assert_eq!(w.available(), 40);
    }

    #[test]
    fn consume_underflow_rejected() {
        let mut w = Window::new(10);
        assert_eq!(w.consume(11).unwrap_err(), Http2Error::FlowControl);
    }

    #[test]
    fn replenish_adds_credit() {
        let mut w = Window::new(0);
        w.replenish(50).unwrap();
        assert_eq!(w.available(), 50);
    }

    #[test]
    fn replenish_overflow_rejected() {
        let mut w = Window {
            available: i32::MAX as i64,
            initial: DEFAULT_INITIAL_WINDOW_SIZE,
        };
        assert_eq!(w.replenish(1).unwrap_err(), Http2Error::FlowControl);
    }

    #[test]
    fn should_replenish_below_half() {
        let mut w = Window::new(100);
        w.consume(60).unwrap();
        assert!(w.should_replenish());
        assert_eq!(w.refill_increment(), 60);
    }

    #[test]
    fn no_replenish_above_half() {
        let mut w = Window::new(100);
        w.consume(40).unwrap();
        assert!(!w.should_replenish());
        assert_eq!(w.refill_increment(), 0);
    }
}
